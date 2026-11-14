# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small-decode MoE engine (SMALL_DECODE_MOE_ENGINE): host policy and engine for the TP2 small-decode cell.

The engine (``small_decode_engine_kernel.py``, the TP4-geometry m3_moe kernel) accepts
i_tp=1536 and sh_i=1536, so it runs this target's geometry; the workspaces and
configs keep this target's sizes and compilation keys.
"""

from pathlib import Path

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.m31_moe_sfa_image import (
    fc1_scale_image,
)


def engine_path() -> str:
    return str(Path(__file__).resolve().parent / "small_decode_engine_kernel.py")


def configure(config: dict, disabled, *, enabled: bool = True) -> None:
    """Apply the established default-on keys; caller admits the small-row engine."""
    if not enabled:
        return
    if config["sfa_cp"] and not disabled("MOE_SFA_BULK"):
        config["sfa_bulk"] = 1
    if not disabled("MOE_K128"):
        config["k128"] = 1


def prepare_layer(config: dict, layer) -> None:
    """Materialize the shared image eagerly, before graph capture."""
    if config.get("sfa_bulk"):
        layer.sfa1_img = fc1_scale_image(layer.w3_w1_weight_scale)


def scale_words(config: dict, layer):
    if config.get("sfa_bulk"):
        return layer.sfa1_img
    return layer.w3_w1_weight_scale.view(torch.uint8).reshape(-1).view(torch.int32)
