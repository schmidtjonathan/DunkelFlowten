# %%
import sda_atmos.eval.conditioning as cond
import os
import torch
import eval.plotting as plotting
import sda_atmos.data.process_training_data as ptd
import geoutils.utils.statistic_utils as sut
import pandas as pd
import numpy as np
import xarray as xr
import geoutils.geodata.wind_dataset as wds
import geoutils.preprocessing.open_nc_file as of
import geoutils.plotting.plots as gplt
import geoutils.utils.time_utils as tu
import geoutils.geodata.base_dataset as bds
import geoutils.utils.spatial_utils as sput
import geoutils.utils.general_utils as gut
import geoutils.utils.file_utils as fut
import geoutils.plotting.plots as gplt
import geoutils.preprocessing.open_nc_file as of
import sda_atmos.eval.sample_data as sda
import sda_atmos.eval.util as sda_utils
import sda_atmos.training.dataset as tds
from pathlib import Path
import json
from importlib import reload

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
    device = torch.device("cuda")
else:
    print("No GPU available. Training will run on CPU.")
    device = torch.device("cpu")

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


print(f"Time downscaling factor: {TIME_DS_FACTOR}")
print(f"Space downscaling factor: {SPACE_DS_FACTOR}")

# %%
reload(of)
reload(gut)
reload(fut)
variables = ['tas', 'uas', 'vas', 'rsds']

gcm = 'MPI-ESM1-2-HR'
# gcm = 'GFDL-ESM4'

scenario = 'historical'
scenario = 'ssp585'
time = 'day'
country_name = 'Germany'
gs = 1.0

gcm_str = f"{gcm}_{scenario}"

files_arr = []
cmip6_folder = f"{cmip6_dir}/{country_name}/{gs}/"
# cmip6_folder = f"{cmip6_dir}/{country_name}/{gs}_bc/"
for variable in variables:
    files = fut.get_files_in_folder(cmip6_folder,
                                    include_string=f'{gcm_str}_{variable}')
    if len(files) > 0:
        filename = files[0]  # always take 1st run in case of multiple
        files_arr.append(filename)

ds_cmip_raw = of.open_nc_file(files_arr, compat='override',
                              check_vars=True
                              )
ds_cmip = gut.translate_cmip2era5(ds_cmip_raw)

# %%
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
# %%
reload(tds)
hourly_res = 6
fine_res = 0.25
coarse_res = 1.0
data_path = f'{era5_dir}/training/{country_name}/{fine_res}/'
stats_dir_name = "stats_log" if use_log else "stats"
stats_dir = f'{data_path}/{stats_dir_name}/'

# prepare the data for SDA
reload(ptd)
eval_cmip_ds, cmip_eval_path = ptd.process_cmip_data(coarse_ds=ds_cmip,
                                                     datapath=cmip6_folder,
                                                     gcm=gcm,
                                                     ssp=scenario)

dataset_cmip_obs = tds.ERADataset(
    data_path=cmip_eval_path,
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
    num_features=len(dataset_cmip_obs.feature_names),
    markov_oder=args_dict["markov_order"],
    use_ema=args_dict["use_ema"],
)
checkpoint = torch.load(
    checkpoint_path, map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["model"])
model.train(False)
model = model.to(device=device)

# %% Conditioning
reload(sda)
N_samples = 1
start_date = "2020-01-01"
end_date = "2100-01-01"
out_dir = f'{eval_dir}/{gcm_str}/'
tr_str = f"{gcm_str}_{start_date}_{end_date}"
samples, gt_ds, obs_ds, cur_out_dir = sda.sample_conditioned_on(
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
    log_sr=use_log,
    # ---
)

# write samples to netcdf
reload(fut)
savepath_samples = cur_out_dir / f"samples_{tr_str}_{fine_res}_log_{use_log}.nc"
fut.save_ds(ds=samples, filepath=savepath_samples)
savepath_obs = cur_out_dir / f"obs_{tr_str}_{coarse_res}_log_{use_log}.nc"
fut.save_ds(ds=obs_ds, filepath=savepath_obs)
samples
# %%
ds_dict = {
    # 'CMIP6': ds_obs.load(),
    'CMIP6 bc': obs_ds.load(),
    'samples': samples.load(),
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
        yrange=(0, .23e-5),
        vname='Surface Solar Radiation Downwards',),
}

# %%
variables = gut.get_vars(samples)
nrows = 2
sd_str, ed_str = tu.get_time_range(
    samples, asstr=True, m=False, d=False, h=False)

time_range = tu.get_time_range(samples)
ssp = scenario
im = gplt.create_multi_plot(nrows=nrows,
                            ncols=len(variables) // nrows,
                            figsize=(15, 10),
                            title=f'Distribution for {gcm} {ssp} ({sd_str}-{ed_str})',
                            y_title=1.,
                            hspace=0.5)

colors = ['red', 'blue', 'tab:blue', 'tab:green']

for idx, variable in enumerate(variables):
    for i, (ds_type, ds) in enumerate(ds_dict.items()):
        ds = tu.get_time_range_data(ds, time_range=time_range)
        ds = tu.compute_timemean(
            ds, timemean='day')  # if variable == 'surface_solar_radiation_downwards' else ds
        plot_data = []
        offset = variable_dict[variable]['offset']
        if ds_type == 'Samples':
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
                                #  ylim=variable_dict[variable]['yrange'] if variable == 'surface_solar_radiation_downwards' else None,
                                 ylabel='Density',
                                 density=True,
                                 xlabel=variable_dict[variable]['label'],
                                #  xlim=(variable_dict[variable]['vmin'],
                                #         variable_dict[variable]['vmax']),
                                 )
        if ds_type == 'samples':
            gplt.fill_between(ax=this_im['ax'],
                              x=this_im['bc'],
                              y=this_im['hc'],
                              y2=0,
                              color=colors[i],
                              alpha=0.15,
                              )


