export ATMOS_SLURM_DIR=/mnt/lustre/work/hennig/jschmidt14/Documents/projects/sda-atmos/slurm

# echo "Copying data to scratch ${SCRATCH}"
# cp -r /mnt/lustre/work/hennig/jschmidt14/Documents/data/ice/train_data/ $SCRATCH/train_data/
# ls -lisah $SCRATCH/train_data/
# ls -lisah $SCRATCH/train_data/stats
# echo "Done. Starting training..."

/mnt/lustre/work/hennig/jschmidt14/Documents/projects/sda-ice/venv-ice/bin/python submit_train.py \
	--dataset /mnt/lustre/work/hennig/jschmidt14/Documents/data/ERA/train-2022_10mWind_2mTemp_Solar_6h_025deg/train_data.nc \
	--output_dir /mnt/lustre/work/hennig/jschmidt14/Documents/projects/sda-atmos/runs \
	--cache_dataset \
	--batch_size=128 \
	--accum_iter=1 \
	--epochs=3000 \
	--eval_frequency 250 \
	--eval_start 500 \
	--ode_method heun2 \
	--data_norm_mode mixed \
	--ode_options '{"nfe": 50}' \
	--use_ema \
	--lr 0.0001 \
	--optimizer_weight_decay 0.001 \
	--decay_lr \
	--edm_schedule \
	--skewed_timesteps \
	--markov_order 4 \
	--timeout=4320 \
	--num_workers 8 \
	--partition 'a100-galvani' \
	--ngpus 4 \
	--use_wandb
