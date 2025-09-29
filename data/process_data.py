# %%
from pathlib import Path

import fire
import numpy as np
import xarray as xr


def to_netcdf_f32(ds, p):
    ds.to_netcdf(p, encoding={var: {"dtype": "float32"}
                 for var in ds.data_vars})


def process_data(
    fine_data_path: str, coarse_data_path: str, outdir_path: str, eval_start="2023"
):
    print("Processing data")
    # SET UP PATHS AND DIRECTORIES
    fine_data_path = Path(fine_data_path)
    coarse_data_path = Path(coarse_data_path)
    outdir_path = Path(outdir_path)
    outdir_path.mkdir(parents=True, exist_ok=True)

    fine_dirname = fine_data_path.parent.name
    coarse_dirname = coarse_data_path.parent.name
    fine_outdir_path = outdir_path / fine_dirname
    coarse_outdir_path = outdir_path / coarse_dirname
    fine_outdir_path.mkdir(parents=True, exist_ok=False)
    coarse_outdir_path.mkdir(parents=True, exist_ok=False)

    fine_traindir_path = fine_outdir_path / "train"
    fine_evaldir_path = fine_outdir_path / "eval"
    coarse_traindir_path = coarse_outdir_path / "train"
    coarse_evaldir_path = coarse_outdir_path / "eval"
    fine_traindir_path.mkdir(parents=True, exist_ok=False)
    fine_evaldir_path.mkdir(parents=True, exist_ok=False)
    coarse_traindir_path.mkdir(parents=True, exist_ok=False)
    coarse_evaldir_path.mkdir(parents=True, exist_ok=False)

    # Load data
    print(f"Loading fine data from {fine_data_path}")
    fine_ds = xr.open_dataset(fine_data_path)
    print(fine_ds)

    print(f"Loading coarse data from {coarse_data_path}")
    coarse_ds = xr.open_dataset(coarse_data_path)
    print(coarse_ds)

    # Pre-processing
    sv = "surface_solar_radiation_downwards"
    print(f"pre-processing {sv}")
    print("Clamp (lower) to 0.0")
    fine_ds[sv] = fine_ds[sv].clip(min=0.0)
    coarse_ds[sv] = coarse_ds[sv].clip(min=0.0)

    print("log10(v + 1)")
    fine_ds[sv] = np.log10(fine_ds[sv] + 1)
    coarse_ds[sv] = np.log10(coarse_ds[sv] + 1)

    # Split into train and test
    train_end = str(int(eval_start) - 1)
    print(
        f"Split into train (before and including {train_end}) and eval (after and including {eval_start})"
    )
    fine_train_ds = fine_ds.sel(time=slice(None, str(train_end)))
    fine_eval_ds = fine_ds.sel(time=slice(str(eval_start), None))

    coarse_train_ds = coarse_ds.sel(time=slice(None, str(train_end)))
    coarse_eval_ds = coarse_ds.sel(time=slice(str(eval_start), None))

    # Compute statistics
    stats_dir = fine_outdir_path / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    print(f"Computing statistics on fine dataset (saving to {stats_dir})")
    mean_ds = fine_train_ds.mean()
    std_ds = fine_train_ds.std()
    min_ds = fine_train_ds.min()
    max_ds = fine_train_ds.max()
    quantiles_ds = fine_train_ds.quantile([0.01, 0.025, 0.05, 0.95, 0.975, 0.99])

    to_netcdf_f32(mean_ds, stats_dir / "mean.nc")
    to_netcdf_f32(std_ds, stats_dir / "std.nc")
    to_netcdf_f32(min_ds, stats_dir / "min.nc")
    to_netcdf_f32(max_ds, stats_dir / "max.nc")
    to_netcdf_f32(quantiles_ds, stats_dir / "quantiles.nc")

    # Save data
    print("Saving data")
    print(
        f"Saving fine/coarse training data to {fine_traindir_path}/{coarse_traindir_path}"
    )
    print(
        f"Saving fine/coarse eval data to {fine_evaldir_path}/{coarse_evaldir_path}")

    to_netcdf_f32(fine_train_ds, fine_traindir_path / "train_data.nc")
    to_netcdf_f32(coarse_train_ds, coarse_traindir_path / "train_data_coarse.nc")

    to_netcdf_f32(fine_eval_ds, fine_evaldir_path / "eval_data.nc")
    to_netcdf_f32(coarse_eval_ds, coarse_evaldir_path / "eval_data_coarse.nc")

    print("Computing mixed normalization")
    mixed_shift = xr.zeros_like(min_ds)
    mixed_shift["10m_u_component_of_wind"] = quantiles_ds.sel(quantile=0.01)[
        "10m_u_component_of_wind"
    ]
    mixed_shift["10m_v_component_of_wind"] = quantiles_ds.sel(quantile=0.01)[
        "10m_v_component_of_wind"
    ]
    mixed_shift["2m_temperature"] = quantiles_ds.sel(quantile=0.01)["2m_temperature"]
    mixed_shift["surface_solar_radiation_downwards"] = min_ds[
        "surface_solar_radiation_downwards"
    ]
    mixed_shift = mixed_shift.drop_vars("quantile", errors="ignore")
    print("NORM SHIFT")
    print(mixed_shift)
    to_netcdf_f32(mixed_shift, stats_dir / "norm_shift.nc")

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
    to_netcdf_f32(mixed_scale, stats_dir / "norm_scale.nc")


if __name__ == "__main__":
    fire.Fire(process_data)
