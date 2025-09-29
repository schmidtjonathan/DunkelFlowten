# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from models.ema import EMA
from models.unet import UNetModel

MODEL_CONFIGS = {
    "atmos-sda": {
        "model_channels": 128,
        "num_res_blocks": 3,
        # "attention_resolutions": [2, 4, 8],
        "attention_resolutions": [],
        "dropout": 0.1,
        "channel_mult": [1, 1, 2, 3, 4],
        # "num_classes": 1000,
        "use_checkpoint": False,
        # "num_heads": 4,
        # "num_head_channels": 64,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
}


def instantiate_model(
    num_features: int, markov_oder: int, use_ema: bool, architecture: str = "atmos-sda"
) -> UNetModel:
    assert architecture in MODEL_CONFIGS, (
        f"Model architecture {architecture} is missing its config."
    )

    window_size = 2 * markov_oder + 1
    cfg = MODEL_CONFIGS[architecture]
    cfg["in_channels"] = num_features * window_size
    cfg["out_channels"] = num_features * window_size
    model = UNetModel(**cfg)

    if use_ema:
        return EMA(model=model), cfg
    else:
        return model, cfg


def instantiate_model_from_cfg(
    cfg: dict,
    use_ema: bool,
):
    model = UNetModel(**cfg)

    if use_ema:
        return EMA(model=model), cfg
    else:
        return model, cfg
