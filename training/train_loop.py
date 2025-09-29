# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import gc
import logging
import math
from typing import Iterable

import torch
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.aggregation import MeanMetric

from flow.path.affine import CondOTProbPath
from models.ema import EMA
from training.grad_scaler import NativeScalerWithGradNormCount

logger = logging.getLogger(__name__)

PRINT_FREQUENCY = 50


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    time = 1 / (1 + sigma)
    time = torch.clip(time, min=0.0001, max=1.0)
    return time


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    lr_schedule: torch.torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    loss_scaler: NativeScalerWithGradNormCount,
    args: argparse.Namespace,
):
    gc.collect()
    model.train(True)
    batch_loss = MeanMetric().to(device, non_blocking=True)
    epoch_loss = MeanMetric().to(device, non_blocking=True)

    accum_iter = args.accum_iter

    path = CondOTProbPath()

    for data_iter_step, (samples, months, hours) in enumerate(data_loader):
        if data_iter_step % accum_iter == 0:
            optimizer.zero_grad()
            batch_loss.reset()
            if data_iter_step > 0 and args.test_run:
                break

        samples = samples.to(device, non_blocking=True)
        months = months.to(device, non_blocking=True)
        hours = hours.to(device, non_blocking=True)

        conditioning = {"month": months, "hour": hours}

        noise = torch.randn_like(samples).to(device)
        if args.skewed_timesteps:
            t = skewed_timestep_sample(samples.shape[0], device=device)
        else:
            t = torch.torch.rand(samples.shape[0]).to(device)
        path_sample = path.sample(t=t, x_0=noise, x_1=samples)
        x_t = path_sample.x_t
        u_t = path_sample.dx_t

        with torch.amp.autocast(device_type="cuda"):
            loss = torch.pow(model(x_t, t, extra=conditioning) - u_t, 2).mean()

        loss_value = loss.item()
        batch_loss.update(loss)
        epoch_loss.update(loss)

        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss /= accum_iter

        # Loss scaler applies the optimizer when update_grad is set to true.
        # Otherwise just updates the internal gradient scales
        apply_update = (data_iter_step + 1) % accum_iter == 0
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=apply_update,
        )
        if apply_update and isinstance(model, EMA):
            model.update_ema()
        elif (
            apply_update
            and isinstance(model, DistributedDataParallel)
            and isinstance(model.module, EMA)
        ):
            model.module.update_ema()

        lr = optimizer.param_groups[0]["lr"]
        if data_iter_step % PRINT_FREQUENCY == 0:
            logger.info(
                f"Epoch {epoch} [{data_iter_step}/{len(data_loader)}]: loss = {batch_loss.compute()}, lr = {lr}"
            )

    lr_schedule.step()
    return {"loss": float(epoch_loss.compute().detach().cpu()), "lr": lr}
