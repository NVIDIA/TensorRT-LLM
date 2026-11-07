# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize decoder GLOBAL cache and recover missing Encoder working state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tensorrt_llm._torch.modules.mhc.hyper_connection import HCState, mHC

from .....pyexecutor.ced_replay import EncoderReplay
from .metadata import CSA2TrtllmMetadata

if TYPE_CHECKING:
    from .....pyexecutor.llm_request import LlmRequest


def prepare_ced_global_kv(
    layer, hc_state: HCState, metadata: CSA2TrtllmMetadata
) -> torch.Tensor | None:
    """Write GLOBAL; a context-only source also returns its output before normalization."""
    attention = layer.self_attn
    if layer.engram is not None or hc_state.is_deferred or hc_state.pre_mix is None:
        raise ValueError("CED requires resolved encoder states and an Engram-free boundary")
    if attention.layer.kv_source != layer.layer_idx or attention.layer.compress_ratio != 1:
        raise ValueError("CED precomputation requires an unpooled GLOBAL owner")
    collapsed = None
    if getattr(metadata, "csa2_remote_tail_mode", None) == "source":
        # A pruned source retains only the boundary norm and GLOBAL projections.
        residual = hc_state.residual
        pre_mix = hc_state.pre_mix.reshape(*residual.shape[:-1], 1).float()
        collapsed = mHC.collapse(residual, pre_mix)
        layer_input = layer.input_layernorm(collapsed)
    else:
        layer_input = layer._decoder_global_input(hc_state)
    attention.prepare_global_cache(layer_input, metadata)
    return collapsed


def complete_encoder_replay(requests: list[LlmRequest]) -> None:
    """Consume recovery plans only after the model forward succeeds."""
    for req in requests:
        replay = req.py_ced_replay
        if isinstance(replay, EncoderReplay):
            replay.consumed = True
