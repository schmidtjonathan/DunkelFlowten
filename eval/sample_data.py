import geoutils.utils.general_utils as gut
import geoutils.utils.spatial_utils as sput
import torch
import numpy as np
import xarray as xr
from pathlib import Path
import time
from importlib import reload
import sda_atmos.eval.util as eval_util
from sda_atmos.flow.solver.ode_solver import ODESolver
from sda_atmos.eval.posterior import PosteriorSequenceModel
from sda_atmos.eval.posterior import FlowModel
import sda_atmos.eval.conditioning as cond
import geoutils.utils.time_utils as tu
from sda_atmos.training.dataset import ERADataset
import sda_atmos.eval.posterior_sequence_model as psm
import sda_atmos.eval.sampling as sampling


reload(eval_util)
reload(tu)
reload(psm)
reload(sampling)


def edm_time_discretization(nfes: int, rho=7):
    step_indices = torch.arange(nfes, dtype=torch.float64)
    sigma_min = 0.002
    sigma_max = 80.0
    sigma_vec = (
        sigma_max ** (1 / rho)
        + step_indices / (nfes - 1) * (sigma_min **
                                       (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    sigma_vec = torch.cat([sigma_vec, torch.zeros_like(sigma_vec[:1])])
    time_vec = (sigma_vec / (1 + sigma_vec)).squeeze()
    t_samples = 1.0 - torch.clip(time_vec, min=0.0, max=1.0)
    return t_samples


def load_samples(p):
    gt_ds = xr.open_dataset(p / "gt.nc")
    obs_ds = xr.open_dataset(p / "obs.nc")
    samples_ds = xr.open_dataset(p / "samples.nc")
    return samples_ds, gt_ds, obs_ds


def sample_prior(
    *,
    L,  # length of time series,
    dataset_test_gt,
    model,
    args_dict,
    denoising_steps=100,
    N_samples=1,
    stride_order=0,
    batch_size=None,
    out_root="./eval/out",
    device=None
):
    if device is None:
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    print(f'Set device to {device}')
    base_device = device if batch_size is None else "cpu"
    print(f'Base device: {base_device}')

    out_root = Path(out_root)

    # LOAD GT
    gt_ds = dataset_test_gt.xr_dataset

    start_index = np.random.randint(gt_ds.sizes["time"] - L)
    gt_ds = gt_ds.isel(
        time=slice(start_index, start_index + L),
    )

    dirname = f"prior_L{L or dataset_test_gt.xr_dataset.sizes['time']}_N{N_samples}"

    out_dir = out_root / dirname
    out_dir = eval_util.create_unique_directory(out_dir)

    flowmodel = FlowModel(model)

    print(f'Use time embedding for months and hours')
    hours = torch.LongTensor(gt_ds.time.dt.hour.values) // 6
    months = torch.LongTensor(gt_ds.time.dt.month.values) - 1

    print(f"Output directory: {out_dir}")

    posterior_model = PosteriorSequenceModel(
        model=flowmodel,
        markov_order=args_dict["markov_order"],
        A=None,
        y=None,
        month_cond=months.to(device=base_device),
        hour_cond=hours.to(device=base_device),
        stride_order=stride_order,
        batch_size=batch_size,
        disable_guidance=True,
        device=device,
    )

    # Sample from posterior
    prior_samples_list = []
    for nsmpl in range(N_samples):
        smpl_start = time.time()
        print(f"Drawing sample [{nsmpl + 1}/{N_samples}]", end=" ... ")
        x_0 = torch.randn(
            (L, len(gt_ds.data_vars), gt_ds.sizes["lat"], gt_ds.sizes["lon"]),
            device=base_device,
        )
        solver = ODESolver(velocity_model=posterior_model)
        ode_opts = args_dict["ode_options"]
        ode_opts["method"] = args_dict["ode_method"]

        generated_samples = solver.sample(
            time_grid=edm_time_discretization(denoising_steps).to(
                dtype=torch.float32, device=base_device
            ),
            x_init=x_0,
            method="midpoint",
            atol=args_dict["ode_options"].get("atol", None),
            rtol=args_dict["ode_options"].get("rtol", None),
            step_size=args_dict["ode_options"].get("step_size", None),
            return_intermediates=False,
        ).cpu()

        # x = x_0
        # for t in edm_time_discretization(256).to(dtype=torch.float32, device=device):
        #     x = x + posterior_model(x, t)

        if torch.any(torch.isnan(generated_samples)):
            raise RuntimeError("NaN in sample")

        smpling_time = time.time() - smpl_start
        print(f"took {smpling_time:.2f} seconds.")

        prior_samples_list.append(generated_samples.detach().cpu())
    posterior_samples = torch.stack(prior_samples_list)

    # Renormalize samples to original (data) scale
    renormed_samples = dataset_test_gt.denormalize_tensor(
        x_LCHW=torch.clamp((posterior_samples + 1) / 2, min=0.0, max=1.0)
    )
    renormed_samples = renormed_samples.to(
        dtype=torch.float32).detach().numpy()

    samples_ds = xr.concat(
        [
            eval_util.np_to_ds(s, gt_ds, dataset_test_gt.feature_names)
            for s in renormed_samples
        ],
        dim="sample_id",
        create_index_for_new_dim=True,
    )

    return samples_ds, out_dir


def get_sample_dates(ds, start_date, end_date, L=None, include_enddate=True):
    if start_date is None and end_date is None:
        start_index = np.random.randint(ds.sizes["time"] - L)
        ds = ds.isel(
            time=slice(start_index, start_index + L),
        )
    else:
        if L is not None:
            print(
                f"Warning: L is set to {L}, but start_date and end_date are provided. Ignoring L.")
        ds = tu.get_time_range_data(ds=ds,
                                    start_date=start_date,
                                    end_date=end_date,
                                    include_enddate=include_enddate)
        L = ds.sizes["time"]
        t0 = ds.time.values[0]
        start_index = ds.indexes["time"].get_loc(t0)

    return ds, start_index, L


def normalize_observation_data(dataset_test_obs, obs_ds):
    # Observation data is often already coarse grained
    obs_ds_normalized = dataset_test_obs.normalize_ds(obs_ds)
    obs_tensor_normalized = torch.from_numpy(
        dataset_test_obs._stack_to_np(
            obs_ds_normalized,
        )
    ).to(torch.float32)

    y = obs_tensor_normalized
    return y


def sample_conditioned_on(
    model,
    dataset_test_obs: ERADataset,
    dataset_test_gt: ERADataset = None,
    *,
    args_dict,
    TIME_DS_FACTOR,
    SPACE_DS_FACTOR,
    start_date,
    end_date,
    coords=None,
    denoising_steps=100,
    N_samples=1,
    gamma=1e-2,
    std=1e-2,
    batch_size=None,
    exact_grad=False,
    device=None,
    dry_run=False,
    log_sr=False,
    clip_radiation=True,
):
    if start_date is None or end_date is None:
        raise ValueError(
            "start_date and end_date must be provided")

    if device is None:
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    gut.myprint(f'Set device to {device}')
    base_device = device if batch_size is None else "cpu"
    gut.myprint(f'Base device: {base_device}')

    include_enddate = False
    # LOAD GT
    if dataset_test_gt is None:
        gut.myprint(
            'GT dataset is not available, using the full observation dataset as conditioning dataset')
        coords = {}
        dataset_test_gt = dataset_test_obs

        lats = dataset_test_obs.xr_dataset.lat.values
        lons = dataset_test_obs.xr_dataset.lon.values
        gs_lon, gs_lat, gs = sput.get_grid_step(dataset_test_obs.xr_dataset)
        gs_lat_fine = gs_lat / SPACE_DS_FACTOR
        gs_lon_fine = gs_lon / SPACE_DS_FACTOR
        lats_fine = np.arange(lats[0],
                              lats[-1]+SPACE_DS_FACTOR*gs_lat_fine, gs_lat_fine)
        lons_fine = np.arange(lons[0],
                              lons[-1]+SPACE_DS_FACTOR*gs_lon_fine, gs_lon_fine)
        coords['lat'] = lats_fine
        coords['lon'] = lons_fine

        dataset_test_gt = dataset_test_obs
        gt_ds = dataset_test_gt.xr_dataset

        gt_ds, start_index, L = get_sample_dates(
            ds=gt_ds,
            start_date=start_date,
            end_date=end_date,
            include_enddate=include_enddate)
        sd, ed = tu.get_time_range(ds=gt_ds,
                                   freq='D')

        td = int(24/TIME_DS_FACTOR)
        times = tu.get_dates_in_range(start_date=sd,
                                      end_date=ed,
                                      freq='h',
                                      time_delta=td,
                                      additional_tps=TIME_DS_FACTOR-1)
        # check if gt data contains leap years or not
        times_data = gt_ds.time
        if len(times_data)*TIME_DS_FACTOR != len(times):
            gut.myprint(
                f"Data does not contain leap years, removing them from output time series as well.")
            times = tu.remove_leap_days(times)
            if len(times) != len(times_data)*TIME_DS_FACTOR:
                raise ValueError(
                    "Still mismatch in the length of times after removing leap days.")

        coords['time'] = times

        gt_shape = torch.Size([
            len(times),
            len(dataset_test_obs.feature_names),
            len(lats_fine),
            len(lons_fine)]
        )
        hours = times.dt.hour.values
        months = times.dt.month.values
    else:
        gut.myprint('GT dataset is available')
        gt_ds = dataset_test_gt.xr_dataset
        gt_ds, start_index, L = get_sample_dates(
            ds=gt_ds,
            start_date=start_date,
            end_date=end_date,
            include_enddate=include_enddate)  # Do end at last our
        coords = gt_ds.coords

        gt_ds_normalized = dataset_test_gt.normalize_ds(gt_ds)
        gt_tensor_normalized = torch.from_numpy(
            dataset_test_gt._stack_to_np(gt_ds_normalized)
        ).to(torch.float32)
        gt_shape = gt_tensor_normalized.shape

        hours = gt_ds.time.dt.hour.values
        months = gt_ds.time.dt.month.values

    sd, ed = tu.tps2str(tu.get_time_range(gt_ds), h=False)
    gut.myprint(f"Sampling time range: {sd} to {ed}")

    # Define observation model
    obs_ds = dataset_test_obs.xr_dataset
    # if log_sr:
    #     sv = "surface_solar_radiation_downwards"
    #     obs_ds[sv] = log_distribution(obs_ds[sv])
    # Select the time range for observation dataset
    freq_res_obs = int(tu.get_frequency_resolution_hours(
        dataset_test_obs.xr_dataset))
    if freq_res_obs == int(24 / TIME_DS_FACTOR):
        print(
            f"Observation dataset has a frequency resolution of {freq_res_obs}h, coarsening to {int(freq_res_obs * TIME_DS_FACTOR)}h"
        )
        obs_ds_sel = obs_ds.isel(time=slice(start_index, start_index + L))
        obs_ds = obs_ds_sel.coarsen(time=TIME_DS_FACTOR).mean()
    else:
        print('Input is already coarse grained!')
        obs_ds = tu.get_time_range_data(
            ds=obs_ds,
            start_date=start_date,
            end_date=end_date,
            include_enddate=include_enddate)

    # Normalize observation
    y = normalize_observation_data(dataset_test_obs, obs_ds)

    gut.myprint(f"Observation shape: {y.shape}")
    gut.myprint(f"Target (GT) shape: {gt_shape}")

    # conditioning operator
    def A(X):
        ret = cond.temporal_window_averaging(
            cond.spatial_window_averaging(X, SPACE_DS_FACTOR), TIME_DS_FACTOR
        )
        return ret

    flowmodel = psm.FlowModel(model)

    print(f'Use time embedding for months and hours')
    hours = torch.LongTensor(hours) // 6
    months = torch.LongTensor(months) - 1
    if not dry_run:
        posterior_model = psm.PosteriorSequenceModel(
            model=flowmodel,
            markov_order=args_dict["markov_order"],
            A=A,
            y=y.to(device=base_device),
            month_cond=months.to(device=base_device),
            hour_cond=hours.to(device=base_device),
            std=std,  # hyperparameter
            gamma=gamma,  # hyperparameter
            batch_size=batch_size,
            clip_target=False,
            exact_grad=exact_grad,
        )

        # Sample from posterior
        posterior_samples_list = []
        for nsmpl in range(N_samples):
            smpl_start = time.time()
            gut.myprint(
                f"Drawing sample [{nsmpl + 1}/{N_samples}]", end=" ... ")
            x_0 = torch.randn(
                gt_shape,
                device=base_device,
            )
            generated_samples = sampling.sampler(
                posterior_model,
                x_0,
                num_steps=denoising_steps,
                midpoint=False,
                # extra={},
            )
            generated_samples = generated_samples.cpu()

            if torch.any(torch.isnan(generated_samples)):
                gut.myprint(generated_samples)
                raise RuntimeError("NaN in sample")

            smpling_time = time.time() - smpl_start
            smpling_time = smpling_time / 60.0
            gut.myprint(f"took {smpling_time:.2f} minuites.")
            gut.myprint(f"Posterior model NFE: {posterior_model.get_nfe()}")
            posterior_model.reset_nfe_counter()

            posterior_samples_list.append(generated_samples.detach().cpu())
        posterior_samples = (
            torch.stack(posterior_samples_list).to(
                dtype=torch.float32).detach().numpy()
        )

        samples_ds = xr.concat(
            [
                eval_util.np_to_ds(np_arr=s,
                                   coords=coords,
                                   data_vars=dataset_test_gt.feature_names)
                for s in posterior_samples
            ],
            dim="sample_id",
            create_index_for_new_dim=True,
        )
        samples_ds = dataset_test_gt.denormalize_ds(samples_ds)
        sv = "surface_solar_radiation_downwards"
        if clip_radiation:
            samples_ds[sv] = clip_da(samples_ds[sv], min=0.0)
        if log_sr:
            sv = "surface_solar_radiation_downwards"
            print(f"post-processing {sv}")
            print("10**(v) - 1")
            # Undoes:  see process_training_data.py
            # > fine_ds[sv] = np.log10(fine_ds[sv] + 1)
            # > coarse_ds[sv] = np.log10(coarse_ds[sv] + 1)
            samples_ds[sv] = unlog_distribution(samples_ds[sv])
            gt_ds[sv] = unlog_distribution(gt_ds[sv])
            obs_ds[sv] = unlog_distribution(obs_ds[sv])

        torch.cuda.empty_cache()
    else:
        samples_ds = y

    return samples_ds, gt_ds, obs_ds


def unlog_distribution(da):
    gut.myprint("10**(v) - 1 to surface solar radiation downwards")
    da = 10 ** (da) - 1
    print("Clamp (lower) to 0.0 to surface solar radiation downwards")
    da = da.clip(min=0.0)
    return da


def log_distribution(da):
    print("Clamp (lower) to 0.0 to surface solar radiation downwards")
    da = da.clip(min=0.0)
    print("log10(v + 1) to surface solar radiation downwards")
    da = np.log10(da + 1)
    return da


def clip_da(da, min=None, max=None):
    if min is not None:
        da = da.clip(min=min)
    if max is not None:
        da = da.clip(max=max)
    return da