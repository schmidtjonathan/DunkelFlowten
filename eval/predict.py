import json
import time
from pathlib import Path
from typing import Optional

import fire
import numpy as np
import torch
import xarray as xr

import eval.posterior_sequence_model as psm
import eval.sampling as sampling
import eval.util as eval_util
from models.model_configs import instantiate_model
from training.dataset import ERADataset

CKPT_ROOT_DIR = "/media/jschmidt/data/Projects/sda-atmos/slurm/experiments/"


def to_netcdf_f32(ds, p):
    ds.to_netcdf(p, encoding={var: {"dtype": "float32"} for var in ds.data_vars})


def spatial_window_averaging(X, win_size):
    # print("spat", X.shape)
    X = X.unfold(2, win_size, win_size).unfold(3, win_size, win_size)
    # print("spat", X.shape)
    X = X.mean(dim=(4, 5))
    # print("spat", X.shape)
    return X


def nearest_neighor_ds(X, factor):
    # return torch.nn.functional.interpolate(
    #     X,
    #     scale_factor=(1 / factor, 1 / factor),
    #     mode="nearest",
    #     # align_corners=True,
    #     # antialias=True,
    # )
    return X[..., ::factor, ::factor]


def temporal_window_averaging(X, win_size):
    X = X.unfold(0, win_size, win_size)
    # print("temp", X.shape)
    X = X.mean(dim=-1)
    # print("temp", X.shape)
    return X


def sample_conditioned_on(
    model,
    dataset_test_gt: ERADataset,
    dataset_test_obs: ERADataset,
    *,
    args_dict,
    with_age,
    L,
    num_cond_steps,
    denoising_steps,
    N_samples,
    gamma,
    std,
    batch_size,
    exact_grad,
    out_root,
):
    TIME_FINE_RES = 6
    TIME_COARSE_RES = 24
    TIME_DS_FACTOR = int(TIME_COARSE_RES / TIME_FINE_RES)
    SPACE_FINE_RES = 0.5
    SPACE_COARSE_RES = 2
    SPACE_DS_FACTOR = int(SPACE_COARSE_RES / SPACE_FINE_RES)
    print(f"Time downscaling factor: {TIME_DS_FACTOR}")
    print(f"Space downscaling factor: {SPACE_DS_FACTOR}")

    base_device = "cuda" if batch_size is None else "cpu"
    print(f"Base device: {base_device}")

    gt_ds = dataset_test_gt.xr_dataset
    start_index = np.random.randint(gt_ds.sizes["time"] - L)
    gt_ds = gt_ds.isel(
        time=slice(start_index, start_index + L),
    )

    out_root = Path(out_root)
    dirname = f"t0-{start_index}_L{L or dataset_test_gt.xr_dataset.sizes['time']}_N{N_samples}"
    out_dir = out_root / dirname
    out_dir = eval_util.create_unique_directory(out_dir)
    print(f"Output directory: {out_dir}")

    obs_ds = dataset_test_obs.xr_dataset
    obs_ds = (
        obs_ds.isel(time=slice(start_index, start_index + L))
        .coarsen(time=TIME_DS_FACTOR)
        .mean()
    )
    obs_ds_normalized = dataset_test_obs.normalize_ds(obs_ds)
    obs_tensor_normalized = torch.from_numpy(
        dataset_test_obs._stack_to_np(
            obs_ds_normalized,
        )
    ).to(torch.float32)

    def A(X):
        ret = temporal_window_averaging(
            spatial_window_averaging(X, SPACE_DS_FACTOR),
            TIME_DS_FACTOR,
            # nearest_neighor_ds(ret, SPACE_DS_FACTOR),
            # TIME_DS_FACTOR,
        )
        return ret

    C = len(dataset_test_gt.feature_names)
    H, W = dataset_test_gt.spatial_res
    gt_shape = (L, C, H, W)

    print(f"Ground-truth shape: {gt_shape}")

    y = obs_tensor_normalized
    print(f"Observation shape: {y.shape}")

    flow_model = psm.FlowModel(model)
    hours = torch.LongTensor(gt_ds.time.dt.hour.values) // 6
    months = torch.LongTensor(gt_ds.time.dt.month.values) - 1

    posterior_model = psm.PosteriorSequenceModel(
        model=flow_model,
        markov_order=args_dict["markov_order"],
        A=A,
        y=y.to(device=base_device),
        month_cond=months.to(device=base_device),
        hour_cond=hours.to(device=base_device),
        std=std,
        gamma=gamma,
        batch_size=batch_size,
        clip_target=False,
        exact_grad=exact_grad,
    )

    # Sample from posterior
    posterior_samples_list = []
    for nsmpl in range(N_samples):
        smpl_start = time.time()
        print(f"Drawing sample [{nsmpl + 1}/{N_samples}]", end=" ... ")
        x_0 = torch.randn(
            gt_shape,
            device=base_device,
        )

        # generated_samples, jacs = sampler(
        generated_samples = sampling.sampler(
            posterior_model,
            x_0,
            num_steps=denoising_steps,
            midpoint=False,
            extra={},
        )
        generated_samples = generated_samples.cpu()

        if torch.any(torch.isnan(generated_samples)):
            raise RuntimeError("NaN in sample")

        smpling_time = time.time() - smpl_start
        print(f"took {smpling_time:.2f} seconds; {posterior_model.get_nfe()} NFE")
        posterior_model.reset_nfe_counter()

        posterior_samples_list.append(generated_samples.detach().cpu())
    posterior_samples = (
        torch.stack(posterior_samples_list).to(dtype=torch.float32).detach().numpy()
    )

    samples_ds = xr.concat(
        [
            eval_util.np_to_ds(s, gt_ds, dataset_test_gt.feature_names)
            for s in posterior_samples
        ],
        dim="sample_id",
        create_index_for_new_dim=True,
    )
    samples_ds = dataset_test_gt.denormalize_ds(samples_ds)

    sv = "surface_solar_radiation_downwards"
    print(f"post-processing {sv}")
    print("10**(v) - 1")
    # Undoes:
    # > fine_ds[sv] = np.log10(fine_ds[sv] + 1)
    # > coarse_ds[sv] = np.log10(coarse_ds[sv] + 1)
    samples_ds[sv] = 10 ** samples_ds[sv] - 1
    gt_ds[sv] = 10 ** gt_ds[sv] - 1

    return samples_ds, gt_ds, obs_ds, out_dir


def draw_samples(
    checkpoint_dir: str,
    dataset_fine: str = "/mnt/data/climate/ERA/processed_025_2/0_25/eval/eval_data.nc",
    dataset_coarse: str = "/mnt/data/climate/ERA/processed_025_2/2/eval/eval_data_coarse.nc",
    traj_len: Optional[int] = None,
    num_cond_steps: int = 5,
    denoising_steps: int = 100,
    n_samples: int = 3,
    gamma: float = 1e-2,
    std: float = 1e-2,
    batch_size: int = 8,
    seed: int = 43,
    exact_grad: bool = True,
):
    # ==================================================================================
    eval_util.set_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    # ==================================================================================
    out_dir_root = Path("./eval/out")
    out_dir_root.mkdir(exist_ok=True, parents=True)
    # ==================================================================================
    checkpoint_dir_path = Path(CKPT_ROOT_DIR) / checkpoint_dir
    assert checkpoint_dir_path.exists() and checkpoint_dir_path.is_dir()
    checkpoint_file_path = checkpoint_dir_path / "checkpoint.pth"
    assert checkpoint_file_path.exists() and checkpoint_file_path.is_file()
    args_filepath = checkpoint_dir_path / "args.json"
    with open(args_filepath, "r") as f:
        args_dict = json.load(f)
        # print(json.dumps(args_dict, indent=4))

    for k, v in args_dict.items():
        print(f"{k}: {v}")
    # ==================================================================================
    dataset_test_gt = ERADataset(
        data_path=dataset_fine,
        cached=True,
        norm_mode=args_dict.get("norm_mode"),
        order=args_dict["markov_order"],
    )

    dataset_test_obs = ERADataset(
        data_path=dataset_coarse,
        cached=True,
        norm_mode=args_dict.get("norm_mode"),
        order=args_dict["markov_order"],
    )
    print("GT dataset: ", dataset_test_gt.xr_dataset.nbytes / 1e9)
    print("Obs dataset: ", dataset_test_obs.xr_dataset.nbytes / 1e9)
    # ==================================================================================
    model, _ = instantiate_model(
        num_features=len(dataset_test_gt.feature_names),
        markov_oder=args_dict["markov_order"],
        use_ema=args_dict["use_ema"],
    )
    checkpoint = torch.load(
        checkpoint_file_path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    model.train(False)

    device = "cuda"
    model = model.to(device=device)  # .to(dtype=torch.float16)
    # ==================================================================================
    smpl_cfg = dict(
        args_dict=args_dict,
        L=traj_len,
        num_cond_steps=num_cond_steps,
        denoising_steps=denoising_steps,
        N_samples=n_samples,
        gamma=gamma,
        std=std,
        batch_size=batch_size,
        exact_grad=exact_grad,
        out_root=str(out_dir_root),
    )
    # ==================================================================================
    # Draw samples ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    samples_ds, gt_ds, obs_ds, cur_out_dir = sample_conditioned_on(
        model,
        dataset_test_gt=dataset_test_gt,
        dataset_test_obs=dataset_test_obs,
        **smpl_cfg,
    )
    to_netcdf_f32(gt_ds, cur_out_dir / "gt.nc")
    to_netcdf_f32(obs_ds, cur_out_dir / "obs.nc")
    to_netcdf_f32(samples_ds, cur_out_dir / "samples_raw.nc")
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ==================================================================================
    smpl_cfg["checkpoint_dir"] = str(checkpoint_dir_path)
    with open(cur_out_dir / "sampling_cfg.json", "w") as scf:
        json.dump(smpl_cfg, scf, indent=2)
    # ==================================================================================

    print("Done.")
    print(f"Saved all results to {cur_out_dir}")


if __name__ == "__main__":
    fire.Fire(
        {
            "sample": draw_samples,
        }
    )
