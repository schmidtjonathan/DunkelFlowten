# %%
import sda_atmos.eval.util as eval_util
import geoutils.utils.general_utils as gut
import geoutils.utils.time_utils as tu
import geoutils.plotting.plots as gplt
import geoutils.utils.file_utils as fut
from importlib import reload
import sda_atmos.training.dataset as tds
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import torch
import json
import xarray as xr
from pathlib import Path
import os
import sda_atmos.eval.conditioning as cond
import sda_atmos.eval.sample_data as sda
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
    device = torch.device("cuda")
else:
    print("No GPU available. Training will run on CPU.")
    device = torch.device("cpu")

if os.getenv("HOME") == '/home/ludwig/fstrnad80':
    era5_dir = "/mnt/lustre/work/ludwig/shared_datasets/weatherbench2/Europe/"
    eval_dir = f"/mnt/lustre/home/ludwig/fstrnad80/data/dunkelflauten/downscaling/eval_with_gt"

else:
    era5_dir = "/home/strnad/data/climate_data/Europe"
    eval_dir = f"/home/strnad/data/dunkelflauten/downscaling/eval_with_gt"


# %%
reload(tds)
country_name = 'Germany'
hourly_res = 6
fine_res = 0.25
coarse_res = 1.0
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
    # print(json.dumps(args_dict, indent=4))

data_path = f'{era5_dir}/training/{country_name}/{fine_res}/'
stats_dir_name = "stats_log" if use_log else "stats"
stats_dir = f'{data_path}/{stats_dir_name}/'
eval_fine_name = "eval_fine_log.nc" if use_log else "eval_fine.nc"
eval_coarse_name = "eval_coarse_log.nc" if use_log else "eval_coarse.nc"
dataset_test_gt = tds.ERADataset(
    data_path=f"{data_path}/eval/{eval_fine_name}",
    cached=True,
    stats_dir=stats_dir,
    norm_mode=args_dict.get("data_norm_mode"),
    order=args_dict["markov_order"],
)

dataset_test_obs = tds.ERADataset(
    data_path=f'{era5_dir}/training/{country_name}/{coarse_res}/eval/{eval_coarse_name}',
    cached=True,
    stats_dir=stats_dir,
    norm_mode=args_dict.get("data_norm_mode"),
    order=args_dict["markov_order"],
)

# %%
# Load the model and the checkpoint
# device = torch.device("cpu")
reload(mcf)
model, _ = mcf.instantiate_model(
    num_features=len(dataset_test_gt.feature_names),
    markov_oder=args_dict["markov_order"],
    use_ema=args_dict["use_ema"],
)
checkpoint = torch.load(
    checkpoint_path, map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["model"])
model.train(False)
model = model.to(device=device)

# %%
reload(eval_util)
reload(fut)
reload(sda)

TIME_FINE_RES = 6
TIME_COARSE_RES = 24
TIME_DS_FACTOR = int(TIME_COARSE_RES / TIME_FINE_RES)

SPACE_FINE_RES = 0.25
SPACE_COARSE_RES = 1
SPACE_DS_FACTOR = int(SPACE_COARSE_RES / SPACE_FINE_RES)

print(f"Time downscaling factor: {TIME_DS_FACTOR}")
print(f"Space downscaling factor: {SPACE_DS_FACTOR}")

# Conditioning
N_samples = 10
start_date = "2023-01-01"
end_date = "2025-01-01"

OUT_DIR = Path(eval_dir)
OUT_DIR.mkdir(exist_ok=True, parents=True)

explicit_outdir = False
dirname = f"{start_date}_{end_date}_N{N_samples}"
out_dir = OUT_DIR / dirname

out_dir = eval_util.create_unique_directory(out_dir,
                                            unique=explicit_outdir)

print(f"Output directory: {out_dir}")
tr_str = f"era5_{start_date}_{end_date}"
savepath_samples = out_dir / \
    f"samples_{tr_str}_{fine_res}_log_{use_log}.nc"
savepath_gt = out_dir / f"gt_{tr_str}_{fine_res}_log_{use_log}.nc"
savepath_obs = out_dir / f"obs_{tr_str}_{coarse_res}_log_{use_log}.nc"

if not fut.exist_file(savepath_samples):
    samples, gt_ds, obs_ds = sda.sample_conditioned_on(
        # ---
        # L=TIME_DS_FACTOR * num_days,
        start_date=start_date,
        end_date=end_date,
        dataset_test_gt=dataset_test_gt,
        dataset_test_obs=dataset_test_obs,
        TIME_DS_FACTOR=TIME_DS_FACTOR,
        SPACE_DS_FACTOR=SPACE_DS_FACTOR,
        model=model,
        args_dict=args_dict,
        # denoising_steps=256,
        N_samples=N_samples,
        batch_size=128,
        std=0.01,
        gamma=0.01,
        log_sr=use_log,
    )

    fut.save_ds(ds=samples, filepath=savepath_samples)
    fut.save_ds(ds=gt_ds, filepath=savepath_gt)
    fut.save_ds(ds=obs_ds, filepath=savepath_obs)
else:
    gut.myprint(f"Samples already exist at {savepath_samples}, loading them.")

# %%
ds_dict = {
    'samples': samples,
    'gt': gt_ds,
    'obs': obs_ds,
}

variable_dict = {
    '2m_temperature': dict(
        cmap='RdBu_r',
        vmin=-10, vmax=35,
        label='Temperature [°C]',
        levels=20,
        vname='Surface Air Temperature',
        offset=-271.15),
    '10m_u_component_of_wind': dict(
        cmap='plasma',
        vmin=-15, vmax=15,
        label='Wind speed [m/s]',
        offset=0,
        levels=20,
        vname='10m U Component of Wind',),
    '10m_v_component_of_wind': dict(
        cmap='viridis',
        vmin=-13, vmax=13,
        label='Wind speed [m/s]',
        offset=0,
        levels=20,
        vname='10m V Component of Wind',),
    'surface_solar_radiation_downwards': dict(
        cmap='inferno',
        vmin=0, vmax=1.5e6,
        label=r'Solar radiation [W/m$^2$]',
        offset=0,
        levels=20,
        yrange=(0, .25e-5),
        vname='Surface Solar Radiation Downwards',),
}

# %%
reload(gplt)
variables = gut.get_vars(samples)
nrows = 2
sd_str, ed_str = tu.get_time_range(
    samples, asstr=True, m=False, d=False, h=False)

tr_distr = f'{sd_str}_{ed_str}'
timemean = 'day'
# timemean = None
im = gplt.create_multi_plot(nrows=nrows,
                            ncols=len(variables) // nrows,
                            figsize=(15, 10),
                            title=f'Distribution for ERA5 ({sd_str}-{ed_str})',
                            y_title=1.,
                            hspace=0.5)

colors = ['red', 'blue', 'tab:blue', 'tab:green']
sv = 'surface_solar_radiation_downwards'
for idx, variable in enumerate(variables):
    for i, (ds_type, ds) in enumerate(ds_dict.items()):
        ds = tu.compute_timemean(
            ds, timemean=timemean)  # if variable == 'surface_solar_radiation_downwards' else ds
        plot_data = []
        offset = variable_dict[variable]['offset']
        if ds_type == 'samples':
            for sample_id in ds.sample_id.values:
                plot_data.append(ds.sel(sample_id=sample_id)
                                 [variable].values.flatten() + offset)
        else:
            plot_data.append(ds[variable].values.flatten() + offset)

        this_im = gplt.plot_hist(plot_data,
                                 ax=im['ax'][idx],
                                 title=variable_dict[variable]['vname'],
                                 color=colors[i],
                                 label=ds_type,
                                 nbins=100,
                                 lw=1 if ds_type == 'samples' else 2,
                                 alpha=0.8 if ds_type == 'samples' else 1,
                                 set_yaxis=False,
                                 ylim=variable_dict[variable]['yrange'] if variable == 'surface_solar_radiation_downwards' else None,
                                 ylabel='Density',
                                 density=True,
                                 xlabel=variable_dict[variable]['label'],
                                 xlim=(variable_dict[variable]['vmin'],
                                        variable_dict[variable]['vmax']),
                                 )
        if ds_type == 'samples':
            gplt.fill_between(ax=this_im['ax'],
                              x=this_im['bc'],
                              y=this_im['hc'],
                              y2=0,
                              color=colors[i],
                              alpha=0.15,
                              )
