# %%
import os
import torch
import sda_atmos.data.process_training_data as ptd
import pandas as pd
import numpy as np
import xarray as xr
import geoutils.preprocessing.open_nc_file as of
import geoutils.utils.general_utils as gut
import geoutils.utils.file_utils as fut
import sda_atmos.eval.sample_data as sda
import sda_atmos.training.dataset as tds
import sda_atmos.eval.util as eval_util
import argparse
from pathlib import Path
import json
from importlib import reload

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
    device = torch.device("cuda")
else:
    print("No GPU available. Training will run on CPU.")
    device = torch.device("cpu")
    # raise ValueError("No GPU available. Cancelled run on CPU.")

TIME_FINE_RES = 6
TIME_COARSE_RES = 24
TIME_DS_FACTOR = int(TIME_COARSE_RES / TIME_FINE_RES)

SPACE_FINE_RES = 0.25
SPACE_COARSE_RES = 1.0
SPACE_DS_FACTOR = int(SPACE_COARSE_RES / SPACE_FINE_RES)

if os.getenv("HOME") == '/home/ludwig/fstrnad80':
    data_dir = "/mnt/lustre/work/ludwig/shared_datasets/weatherbench2/Europe/"
    cmip6_dir = f"/mnt/lustre/work/ludwig/shared_datasets/CMIP6/"
    era5_dir = data_dir
    eval_dir = f"{cmip6_dir}/downscaling/{SPACE_COARSE_RES}"

else:
    data_dir = "/home/strnad/data/"
    cmip6_dir = f"{data_dir}/CMIP6/"
    era5_dir = f'{data_dir}/climate_data/Europe'
    eval_dir = f"{cmip6_dir}/downscaling/{SPACE_COARSE_RES}"
OUT_DIR = Path(eval_dir)
OUT_DIR.mkdir(exist_ok=True, parents=True)

reload(fut)
use_log = False
if use_log:
    import sda_atmos.models.model_configs_log as mcf
    ckp_path = "./checkpoints/model_64_64_corr/checkpoint.pth"
    print("Using log-normalized model configuration.")
else:
    import sda_atmos.models.model_configs as mcf
    ckp_path = "./checkpoints/model_64_64/checkpoint.pth"
    print("Using standard model configuration.")

checkpoint_path = Path(
    ckp_path
)
args_filepath = checkpoint_path.parent / "args.json"
with open(args_filepath, "r") as f:
    args_dict = json.load(f)

country_name = 'Germany'
hourly_res = 6
fine_res = 0.25
coarse_res = 1.0
data_path = f'{era5_dir}/training/{country_name}/{fine_res}/'
stats_dir_name = "stats_log" if use_log else "stats"
stats_dir = f'{data_path}/{stats_dir_name}/'

variables = ['tas', 'uas', 'vas', 'rsds']
NUM_FEATURES = len(variables)
print(f"Time downscaling factor: {TIME_DS_FACTOR}")
print(f"Space downscaling factor: {SPACE_DS_FACTOR}")

N_samples = 3

trs_ssp = [
    ("2020-01-01", "2030-01-01"),
    ("2030-01-01", "2040-01-01"),
    ("2040-01-01", "2050-01-01"),
    ("2050-01-01", "2060-01-01"),
    ("2060-01-01", "2070-01-01"),
    ("2070-01-01", "2080-01-01"),
    ("2080-01-01", "2090-01-01"),
    ("2090-01-01", "2100-01-01"),
]

trs_historical = [
    ('1980-01-01', '1990-01-01'),
    ('1990-01-01', '2000-01-01'),
    ('2000-01-01', '2010-01-01'),
    ('2010-01-01', '2015-01-01'),
]
trs_historical = [('1980-01-01', '2015-01-01')]  # full range
trs_ssp = [('2020-01-01', '2100-01-01')]  # full range


# %%
# Load the model and the checkpoint
# device = torch.device("cpu")
def main():
    reload(mcf)
    model, _ = mcf.instantiate_model(
        num_features=NUM_FEATURES,  # =4
        markov_oder=args_dict["markov_order"],
        use_ema=args_dict["use_ema"],
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.train(False)
    model = model.to(device=device)

    # %%
    reload(of)
    reload(gut)
    reload(fut)
    reload(ptd)
    reload(sda)
    reload(fut)
    parser = argparse.ArgumentParser(description="Run downscaling a climate model simulation.")

    # Add an optional argument for the model name
    parser.add_argument(
        "--model",
        type=str,
        help="Name of the Earth System Model to use (e.g., CMCC-CM2-SR5)"
    )
    # Parse the arguments
    args = parser.parse_args()

    avail_gcms = [
        'MPI-ESM1-2-HR',
        'GFDL-ESM4',
        'MIROC6',
        'IPSL-CM6A-LR',
        'CanESM5'
    ]
    # If a model is specified, use that; otherwise, use the full list
    if args.model:
        gcms = [args.model]
    else:
        gcms = avail_gcms

    gut.myprint(f"Using GCMs: {gcms}")

    ssps = [
        'historical',
        'ssp245',
        'ssp585'
    ]
    for gcm in gcms:
        for ssp in ssps:
            trs = trs_historical if ssp == 'historical' else trs_ssp

            gcm_str = f"{gcm}_{ssp}"
            cmip_out_dir = f'{eval_dir}/{gcm_str}/'

            files_arr = []
            cmip6_folder = f"{cmip6_dir}/{country_name}/{coarse_res}_bc/"

            for variable in variables:
                files = fut.get_files_in_folder(cmip6_folder,
                                                include_string=f'{gcm}_{ssp}_{variable}')
                if len(files) > 0:
                    filename = files[0]  # always take 1st run in case of multiple
                    files_arr.append(filename)

            ds_cmip_raw = of.open_nc_file(files_arr, compat='override',
                                        check_vars=True
                                        )
            ds_cmip = gut.translate_cmip2era5(ds_cmip_raw)

            # prepare the data for CMIP
            _, cmip_eval_path = ptd.process_cmip_data(
                coarse_ds=ds_cmip,
                datapath=cmip6_folder,
                gcm=gcm,
                ssp=ssp)

            dataset_cmip_obs = tds.ERADataset(
                data_path=cmip_eval_path,
                cached=True,
                stats_dir=stats_dir,
                norm_mode=args_dict.get("data_norm_mode"),
                order=args_dict["markov_order"],
            )

            for start_date, end_date in trs:
                gut.myprint(f'\n GCM: {gcm_str}\n SSP: {ssp} \n N_samples: {N_samples}')
                gut.myprint(f"Start date: {start_date}, End date: {end_date}")

                OUT_DIR = Path(cmip_out_dir)
                OUT_DIR.mkdir(exist_ok=True, parents=True)

                explicit_outdir = False
                dirname = f"{start_date}_{end_date}_N{N_samples}"
                out_dir = OUT_DIR / dirname

                out_dir = eval_util.create_unique_directory(out_dir,
                                                            unique=explicit_outdir)

                print(f"Output directory: {out_dir}")
                tr_str = f"{gcm_str}_{start_date}_{end_date}"
                savepath_samples = out_dir / \
                    f"samples_{tr_str}_{fine_res}_log_{use_log}.nc"
                savepath_obs = out_dir / f"obs_{tr_str}_{coarse_res}_log_{use_log}.nc"
                if not fut.exist_file(savepath_samples):
                    samples, gt_ds, obs_ds = sda.sample_conditioned_on(
                        start_date=start_date,
                        end_date=end_date,
                        dataset_test_obs=dataset_cmip_obs,
                        TIME_DS_FACTOR=TIME_DS_FACTOR,
                        SPACE_DS_FACTOR=SPACE_DS_FACTOR,
                        model=model,
                        args_dict=args_dict,
                        # denoising_steps=256,
                        N_samples=N_samples,
                        batch_size=256,
                        std=0.01,
                        gamma=0.01,
                        # ---
                        log_sr=use_log,
                        # dry_run=True,
                    )
                    # write samples to netcdf
                    fut.save_ds(ds=samples, filepath=savepath_samples)
                    fut.save_ds(ds=obs_ds, filepath=savepath_obs)
                else:
                    gut.myprint(f"Samples already exist: {savepath_samples}")
# %%
if __name__ == "__main__":
    main()