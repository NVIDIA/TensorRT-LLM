# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Beam family routing: which ModelingV2 target serves a ``BeamForCausalLM`` deployment."""

from __future__ import annotations

from typing import Any, Optional, Tuple

from ..._router_index import NoTarget


def _quant_algo(pretrained_config: Any) -> Optional[str]:
    quant = getattr(pretrained_config, "quantization_config", None)
    if isinstance(quant, dict):
        return quant.get("quant_algo")
    return getattr(quant, "quant_algo", None)


def route(
    pretrained_config: Any,
    mapping: Any,
    compute_capability: Tuple[int, int],
    spec_config: Optional[Any],
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits.

    The one target is the 52-layer NVFP4-expert checkpoint on four sm_103 ranks: attention
    TP4, MoE TP4/EP1, PP1, no attention data parallelism, no speculative decoding.
    """
    shape = (
        pretrained_config.num_hidden_layers,
        pretrained_config.hidden_size,
        getattr(pretrained_config, "num_experts", None),
    )
    layout = (
        mapping.world_size,
        mapping.tp_size,
        mapping.pp_size,
        mapping.moe_tp_size,
        mapping.moe_ep_size,
        bool(mapping.enable_attention_dp),
    )
    quant_algo = _quant_algo(pretrained_config)
    if (
        compute_capability == (10, 3)
        and shape == (52, 5120, 256)
        and layout == (4, 4, 1, 4, 1, False)
        and quant_algo == "NVFP4"
        and spec_config is None
    ):
        from .full_model_nvfp4__sm_103__tp4.modeling import ModelingV2BeamFullModelNvfp4Sm103Tp4

        return ModelingV2BeamFullModelNvfp4Sm103Tp4

    raise NoTarget(
        f"compute capability {compute_capability}, config shape {shape}, quantization "
        f"{quant_algo}, layout (world, tp, pp, moe_tp, moe_ep, attention_dp) {layout}, "
        f"speculative decoding {type(spec_config).__name__ if spec_config is not None else 'off'}; "
        "the one Beam target is the 52-layer NVFP4 checkpoint on sm103, attention TP4, MoE "
        "TP4/EP1, PP1, without attention DP or speculative decoding"
    )
