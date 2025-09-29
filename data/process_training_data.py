# %%
import numpy as np
import os
from pathlib import Path
import geoutils.utils.file_utils as fut
import geoutils.utils.time_utils as tu
import geoutils.utils.general_utils as gut
import xarray as xr
import geoutils.preprocessing.open_nc_file as onf
from importlib import reload

reload(fut)


def to_netcdf_f32(ds, p):
    ds.to_netcdf(p, encoding={var: {"dtype": "float32"}
                 for var in ds.data_vars})


def open_datafiles(fine_data_path, coarse_data_path):
    fut.assert_file_exists(fine_data_path)
    fut.assert_file_exists(coarse_data_path)
    fine_ds = onf.open_nc_file(fine_data_path)
    coarse_ds = onf.open_nc_file(coarse_data_path)

    return fine_ds, coarse_ds


def generate_output_path(fine_data_path, coarse_data_path,
                         stats_dir=None,
                         use_log=False):
    fine_train_path = Path(fine_data_path).parent / 'train'
    coarse_train_path = Path(coarse_data_path).parent / 'train'
    fine_eval_path = Path(fine_data_path).parent / 'eval'
    coarse_eval_path = Path(coarse_data_path).parent / 'eval'
    # set the statistics directory to the fine data directory on which the statistics are computed
    stats_dir_name = 'stats' if not use_log else 'stats_log'
    if stats_dir is None:
        stats_dir = Path(fine_data_path).parent / stats_dir_name
    stats_dir = Path(stats_dir)
    for path in [fine_train_path, coarse_train_path, fine_eval_path, coarse_eval_path, stats_dir]:
        fut.create_folder(path)

    path_dict = {
        'fine_train_path': fine_train_path,
        'coarse_train_path': coarse_train_path,
        'fine_eval_path': fine_eval_path,
        'coarse_eval_path': coarse_eval_path,
        'stats_dir': stats_dir
    }

    return path_dict


def compute_statistics(fine_ds, fine_train_ds, stats_dir):
    print(f"Computing statistics on fine dataset (saving to {stats_dir})")
    mean_ds = fine_train_ds.mean()
    std_ds = fine_train_ds.std()
    min_ds = fine_train_ds.min()
    max_ds = fine_train_ds.max()
    quantiles_ds = fine_train_ds.quantile(
        [0.01, 0.025, 0.05, 0.95, 0.975, 0.99])

    zscore_shift = mean_ds
    zscore_scale = std_ds
    minmax_shift = min_ds
    minmax_scale = max_ds - min_ds

    print(f"Z-Score: shift: {zscore_shift} scale: {zscore_scale}")
    zscore_test = (fine_ds - zscore_shift) / zscore_scale
    print(f"zscore test: mean {zscore_test.mean()} std {zscore_test.std()}")

    print(f"Min-Max: shift: {minmax_shift} scale: {minmax_scale}")
    minmax_test = (fine_ds - minmax_shift) / minmax_scale
    print(f"minmax test: min {minmax_test.min()} max {minmax_test.max()}")

    print("Computing mixed normalization")
    mixed_shift = xr.zeros_like(min_ds)
    mixed_shift["10m_u_component_of_wind"] = quantiles_ds.sel(quantile=0.01)[
        "10m_u_component_of_wind"
    ]
    mixed_shift["10m_v_component_of_wind"] = quantiles_ds.sel(quantile=0.01)[
        "10m_v_component_of_wind"
    ]
    mixed_shift["2m_temperature"] = quantiles_ds.sel(quantile=0.01)[
        "2m_temperature"]
    mixed_shift["surface_solar_radiation_downwards"] = min_ds[
        "surface_solar_radiation_downwards"
    ]
    mixed_shift = mixed_shift.drop_vars("quantile", errors="ignore")
    print("NORM SHIFT")
    print(mixed_shift)

    mixed_scale = xr.zeros_like(min_ds)
    mixed_scale["10m_u_component_of_wind"] = (
        quantiles_ds.sel(quantile=0.99)["10m_u_component_of_wind"]
        - quantiles_ds.sel(quantile=0.01)["10m_u_component_of_wind"]
    )
    mixed_scale["10m_v_component_of_wind"] = (
        quantiles_ds.sel(quantile=0.99)["10m_v_component_of_wind"]
        - quantiles_ds.sel(quantile=0.01)["10m_v_component_of_wind"]
    )
    mixed_scale["2m_temperature"] = (
        quantiles_ds.sel(quantile=0.99)["2m_temperature"]
        - quantiles_ds.sel(quantile=0.01)["2m_temperature"]
    )
    mixed_scale["surface_solar_radiation_downwards"] = (
        max_ds["surface_solar_radiation_downwards"]
        - min_ds["surface_solar_radiation_downwards"]
    )
    mixed_scale = mixed_scale.drop_vars("quantile", errors="ignore")
    print("NORM SCALE")
    print(mixed_scale)

    # Save statistics for zscore normalization based on fine resolution
    gut.myprint(
        f"Saving statistics to {stats_dir} (zscore, minmax, mixed normalization)")
    fut.save_ds(mean_ds, filepath=stats_dir / "mean.nc")
    fut.save_ds(std_ds, filepath=stats_dir / "std.nc")
    fut.save_ds(zscore_shift, filepath=stats_dir / "zscore_shift.nc")
    fut.save_ds(zscore_scale, filepath=stats_dir / "zscore_scale.nc")

    # Save statistics for min-max normalization
    fut.save_ds(min_ds, filepath=stats_dir / "min.nc")
    fut.save_ds(max_ds, filepath=stats_dir / "max.nc")
    fut.save_ds(minmax_shift, filepath=stats_dir / "minmax_shift.nc")
    fut.save_ds(minmax_scale, filepath=stats_dir / "minmax_scale.nc")

    # Save statistics for quantiles normalization
    fut.save_ds(quantiles_ds, filepath=stats_dir / "quantiles.nc")
    fut.save_ds(mixed_shift, filepath=stats_dir / "norm_shift.nc")
    fut.save_ds(mixed_scale, filepath=stats_dir / "norm_scale.nc")


def log_distribution(da):
    print("Clamp (lower) to 0.0")
    da = da.clip(min=0.0)
    print("log10(v + 1)")
    da = np.log10(da + 1)
    return da


def preprocess_data(ds,
                    preprocess_variables=[]):
    """
    Preprocess the dataset by applying log distribution to the specified variables.
    """
    for var in preprocess_variables:
        if var in ds.data_vars:
            if var == 'surface_solar_radiation_downwards':
                gut.myprint(
                    "Applying log distribution to surface solar radiation downwards")
                ds[var] = log_distribution(ds[var])
        else:
            gut.myprint(f"Variable {var} not found in dataset.")
    return ds


def process_gt_data(
    fine_ds, coarse_ds, path_dict,
    preprocess_variables=[],
    eval_start="2023-01-01",
):

    use_log = True if 'surface_solar_radiation_downwards' in preprocess_variables else False

    # Pre-processing
    fine_ds = preprocess_data(fine_ds,
                              preprocess_variables=preprocess_variables)
    coarse_ds = preprocess_data(coarse_ds,
                                preprocess_variables=preprocess_variables)

    # Split into train and test
    fine_train_ds = tu.get_data_sd_ed(da=fine_ds, end_date=eval_start)
    fine_eval_ds = tu.get_data_sd_ed(da=fine_ds, start_date=eval_start)

    coarse_train_ds = tu.get_data_sd_ed(da=coarse_ds, end_date=eval_start)
    coarse_eval_ds = tu.get_data_sd_ed(da=coarse_ds, start_date=eval_start)

    # Compute statistics
    stats_dir = path_dict['stats_dir']

    compute_statistics(fine_ds, fine_train_ds, stats_dir)

    # Save data
    print("Saving data")
    fine_traindir_path = path_dict['fine_train_path']
    coarse_traindir_path = path_dict['coarse_train_path']
    fine_evaldir_path = path_dict['fine_eval_path']
    coarse_evaldir_path = path_dict['coarse_eval_path']
    print(
        f"Saving fine/coarse training data to {fine_traindir_path}/{coarse_traindir_path}"
    )
    print(
        f"Saving fine/coarse eval data to {fine_evaldir_path}/{coarse_evaldir_path}")


    train_fine_name = "train_fine" if not use_log else "train_fine_log"
    train_coarse_name = "train_coarse" if not use_log else "train_coarse_log"
    eval_fine_name = "eval_fine" if not use_log else "eval_fine_log"
    eval_coarse_name = "eval_coarse" if not use_log else "eval_coarse_log"

    fut.save_ds(fine_train_ds, fine_traindir_path / f"{train_fine_name}.nc")
    fut.save_ds(coarse_train_ds, coarse_traindir_path / f"{train_coarse_name}.nc")

    fut.save_ds(fine_eval_ds, fine_evaldir_path / f"{eval_fine_name}.nc")
    fut.save_ds(coarse_eval_ds, coarse_evaldir_path / f"{eval_coarse_name}.nc")

    return fine_train_ds, fine_eval_ds, coarse_train_ds, coarse_eval_ds


def process_cmip_data(coarse_ds, datapath, gcm, ssp,
                      use_log=False, eval_start=None):
    # Split into train and test if eval_start is not None
    preprocess_variables = ['surface_solar_radiation_downwards'] if use_log else []
    datapath = Path(datapath)
    coarse_ds = preprocess_data(
        coarse_ds,
        preprocess_variables=preprocess_variables
    )
    if eval_start is not None:
        coarse_train_ds = tu.get_data_sd_ed(da=coarse_ds, end_date=eval_start)
        coarse_eval_ds = tu.get_data_sd_ed(da=coarse_ds, start_date=eval_start)
    else:
        coarse_train_ds = None
        coarse_eval_ds = coarse_ds  # the whole dataset as evaluation set
    # Save data
    cmip_str = f"{gcm}_{ssp}"
    eval_name = "eval_log" if use_log else "eval"
    savepath = datapath / f"{cmip_str}_{eval_name}.nc"
    print("Saving data")
    fut.save_ds(coarse_eval_ds, savepath)
    if coarse_train_ds is not None:
        train_name = "train_log" if use_log else "train"
        fut.save_ds(coarse_train_ds, datapath / f"{cmip_str}_{train_name}.nc")
        return coarse_train_ds, coarse_eval_ds, savepath
    else:
        return coarse_eval_ds, savepath


if __name__ == "__main__":

    if os.getenv("HOME") == '/home/ludwig/fstrnad80':
        data_dir = "/mnt/qb/work/ludwig/fstrnad80/data/processed_data/Europe/"
        output_dir = "/mnt/lustre/work/ludwig/shared_datasets/weatherbench2/Europe/"
    else:
        data_dir = "/home/strnad/data/climate_data/Europe"
        output_dir = "/home/strnad/data/climate_data/Europe"

    country_name = 'Germany'
    hourly_res = 6
    fine_res = 0.25
    coarse_res = 1.0
    used_variables = [
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
        '2m_temperature',
        'surface_solar_radiation_downwards',
    ]
    names = gut.list2str(used_variables, sep='_')
    fine_data_path = f'{output_dir}/training/{country_name}/{fine_res}/training_{names}_{hourly_res}h.nc'
    coarse_data_path = f'{output_dir}/training/{country_name}/{coarse_res}/training_{names}_{hourly_res}h.nc'

    use_log_distr = False
    if use_log_distr:
        print("Using log distribution for surface solar radiation downwards")
        preprocess_variables = ['surface_solar_radiation_downwards']
    else:
        print("Not using log distribution for surface solar radiation downwards")
        preprocess_variables = []

    fine_ds, coarse_ds = open_datafiles(fine_data_path=fine_data_path,
                                        coarse_data_path=coarse_data_path)

    path_dict = generate_output_path(
        fine_data_path=fine_data_path,
        coarse_data_path=coarse_data_path,
        use_log=use_log_distr,
        # stats_dir='./stats/'
    )

    fine_train, fine_eval, coarse_train, coarse_eval = process_gt_data(
        fine_ds=fine_ds, coarse_ds=coarse_ds, path_dict=path_dict,
        preprocess_variables=preprocess_variables)

# %%
