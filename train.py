# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. and affiliates.

import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import wandb
from models.model_configs import instantiate_model
from train_arg_parser import get_args_parser
from training import distributed_mode
from training.dataset import ERADataset

# from training.eval_loop import eval_model
from training.grad_scaler import NativeScalerWithGradNormCount as NativeScaler
from training.load_and_save import load_model, save_model
from training.train_loop import train_one_epoch

logger = logging.getLogger(__name__)


def wandb_set_startup_timeout(seconds: int):
    assert isinstance(seconds, int)
    os.environ["WANDB__SERVICE_WAIT"] = f"{seconds}"


@torch.no_grad()
def print_module_summary(module, inputs, max_nesting=3, skip_redundant=True):
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    assert isinstance(inputs, (tuple, list))

    # Register hooks.
    entries = []
    nesting = [0]

    def pre_hook(_mod, _inputs):
        nesting[0] += 1

    def post_hook(mod, _inputs, outputs):
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, torch.Tensor)]
            entries.append(dict(mod=mod, outputs=outputs))

    hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    # Run module.
    outputs = module(*inputs)
    for hook in hooks:
        hook.remove()

    # Identify unique outputs, parameters, and buffers.
    tensors_seen = set()
    for e in entries:
        e["unique_params"] = [
            t for t in e["mod"].parameters() if id(t) not in tensors_seen
        ]
        e["unique_buffers"] = [
            t for t in e["mod"].buffers() if id(t) not in tensors_seen
        ]
        e["unique_outputs"] = [t for t in e["outputs"] if id(t) not in tensors_seen]
        tensors_seen |= {
            id(t)
            for t in e["unique_params"] + e["unique_buffers"] + e["unique_outputs"]
        }

    # Filter out redundant entries.
    if skip_redundant:
        entries = [
            e
            for e in entries
            if len(e["unique_params"])
            or len(e["unique_buffers"])
            or len(e["unique_outputs"])
        ]

    # Construct table.
    rows = [
        [type(module).__name__, "Parameters", "Buffers", "Output shape", "Datatype"]
    ]
    rows += [["-----"] * len(rows[0])]
    rows += [
        [f"Input {lbl}", "-", "-", str(list(t.shape)), str(t.dtype).split(".")[-1]]
        for lbl, t in zip(["data", "noise"], inputs[:2])
    ]
    param_total = 0
    buffer_total = 0
    submodule_names = {mod: name for name, mod in module.named_modules()}
    for e in entries:
        name = "<top-level>" if e["mod"] is module else submodule_names[e["mod"]]
        param_size = sum(t.numel() for t in e["unique_params"])
        buffer_size = sum(t.numel() for t in e["unique_buffers"])
        output_shapes = [str(list(t.shape)) for t in e["outputs"]]
        output_dtypes = [str(t.dtype).split(".")[-1] for t in e["outputs"]]
        rows += [
            [
                name + (":0" if len(e["outputs"]) >= 2 else ""),
                str(param_size) if param_size else "-",
                str(buffer_size) if buffer_size else "-",
                (output_shapes + ["-"])[0],
                (output_dtypes + ["-"])[0],
            ]
        ]
        for idx in range(1, len(e["outputs"])):
            rows += [
                [name + f":{idx}", "-", "-", output_shapes[idx], output_dtypes[idx]]
            ]
        param_total += param_size
        buffer_total += buffer_size
    rows += [["---"] * len(rows[0])]
    rows += [["Total", str(param_total), str(buffer_total), "-", "-"]]

    # Print table.
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    print()
    for row in rows:
        print(
            "  ".join(
                cell + " " * (width - len(cell)) for cell, width in zip(row, widths)
            )
        )
    print()


def calc_mem(m: torch.nn.Module) -> float:
    """
    Calculate the total memory requirement of a PyTorch module in megabytes (MB).

    Args:
        m (torch.nn.Module): The PyTorch model.

    Returns:
        float: The total memory requirement in MB.
    """
    total_bytes = sum(p.numel() * p.element_size() for p in m.parameters())
    total_bytes += sum(b.numel() * b.element_size() for b in m.buffers())

    return total_bytes / (1024**2)  # Convert bytes to MB


def main(args):
    wandb_set_startup_timeout(600)

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    distributed_mode.init_distributed_mode(args)

    logger.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    logger.info("{}".format(args).replace(", ", ",\n"))
    wandb_logger = None

    if distributed_mode.is_main_process():
        args_filepath = Path(args.output_dir) / "args.json"
        logger.info(f"Saving args to {args_filepath}")
        with open(args_filepath, "w") as f:
            json.dump(vars(args), f, indent=2)

        if args.use_wandb:
            wandb_logger = wandb.init(
                project="sda-atmos",
                id=args.job_id,
                config=vars(args),
                resume="allow",
            )

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + distributed_mode.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    logger.info(f"Initializing Dataset: {args.dataset}")
    dataset_train = ERADataset(
        data_path=args.dataset,
        cached=args.cache_dataset,
        norm_mode=args.data_norm_mode,
        order=args.markov_order,
    )

    logger.info(dataset_train)

    logger.info("Intializing DataLoader")
    num_tasks = distributed_mode.get_world_size()
    global_rank = distributed_mode.get_rank()
    logger.info(f"num_tasks: {num_tasks}, global_rank: {global_rank}")
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    logger.info(str(sampler_train))

    # define the model
    logger.info("Initializing Model")
    model, model_cfg_dict = instantiate_model(
        num_features=len(dataset_train.ORDERED_FEATURE_NAMES),
        markov_oder=args.markov_order,
        use_ema=args.use_ema,
    )

    model.to(device)

    if distributed_mode.is_main_process():
        mcfg_filepath = Path(args.output_dir) / "model_config.json"
        logger.info(f"Saving model config to {mcfg_filepath}")
        with open(mcfg_filepath, "w") as f:
            json.dump(model_cfg_dict, f, indent=2)

        ref_data, ref_months, ref_hours = dataset_train[0]
        print(f"Ref data shape: {ref_data.shape}")
        print(f"months: {ref_months.shape}, hours: {ref_hours.shape}")
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            print_module_summary(
                model,
                [
                    torch.zeros(
                        [1, *ref_data.shape],
                        device=device,
                    ),
                    torch.ones([1], device=device),
                    {
                        "month": ref_months.unsqueeze(0).to(device),
                        "hour": ref_hours.unsqueeze(0).to(device),
                    },
                ],
                # max_nesting=6,
            )
            print(f"Total memory: {calc_mem(model):.2f} MB")

    model_without_ddp = model
    logger.info(str(model_without_ddp))

    eff_batch_size = (
        args.batch_size * args.accum_iter * distributed_mode.get_world_size()
    )

    logger.info(f"Learning rate: {args.lr:.2e}")

    logger.info(f"Accumulate grad iterations: {args.accum_iter}")
    logger.info(f"Effective batch size: {eff_batch_size}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],  # find_unused_parameters=True
        )
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(
        model_without_ddp.parameters(),
        lr=args.lr,
        betas=args.optimizer_betas,
        weight_decay=args.optimizer_weight_decay,
    )
    if args.decay_lr:
        lr_schedule = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=args.epochs,
            start_factor=1.0,
            end_factor=1e-8 / args.lr,
        )
    else:
        lr_schedule = torch.optim.lr_scheduler.ConstantLR(
            optimizer, total_iters=args.epochs, factor=1.0
        )

    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Learning-Rate Schedule: {lr_schedule}")

    loss_scaler = NativeScaler()

    load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        lr_schedule=lr_schedule,
    )

    logger.info(f"Start from {args.start_epoch} to {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if not args.eval_only:
            train_stats = train_one_epoch(
                model=model,
                data_loader=data_loader_train,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                device=device,
                epoch=epoch,
                loss_scaler=loss_scaler,
                args=args,
            )
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
            }
        else:
            log_stats = {
                "epoch": epoch,
            }

        if args.output_dir and (
            (
                epoch >= args.eval_start
                and args.eval_frequency > 0
                and (epoch + 1) % args.eval_frequency == 0
            )
            or args.eval_only
            or args.test_run
        ):
            if not args.eval_only:
                logger.info(f"+++ Saving model at epoch {epoch}")
                save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    lr_schedule=lr_schedule,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                )
            # if args.distributed:
            #     data_loader_train.sampler.set_epoch(0)

            # if distributed_mode.is_main_process():
            #     fid_samples = args.fid_samples - (num_tasks - 1) * (
            #         args.fid_samples // num_tasks
            #     )
            # else:
            #     fid_samples = args.fid_samples // num_tasks
            # eval_stats = eval_model(
            #     model,
            #     data_loader_train,
            #     device,
            #     epoch=epoch,
            #     fid_samples=fid_samples,
            #     args=args,
            # )
            # log_stats.update({f"eval_{k}": v for k, v in eval_stats.items()})

        if args.output_dir and distributed_mode.is_main_process():
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

        if distributed_mode.is_main_process():
            if wandb_logger is not None:
                wandb_logger.log({f"train_{k}": v for k, v in train_stats.items()})

        if args.test_run or args.eval_only:
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
