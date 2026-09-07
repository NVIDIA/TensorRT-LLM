# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Startup-time model-runner dispatch and transitional helper exports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

from .common import (
    apply_position_id_offset,
    get_all_rank_num_tokens,
    get_padding_params,
    get_position_id_offset,
    get_top_level_model,
    prepare_multimodal_indices,
    set_spec_metadata_all_rank_num_tokens,
    ship_multimodal_indices,
)
from .mm_encoder import MultimodalEncoderRunner
from .pooling import PoolingRunner

if TYPE_CHECKING:
    from .interface import ModelRunner

__all__ = [
    "apply_position_id_offset",
    "get_all_rank_num_tokens",
    "get_padding_params",
    "get_position_id_offset",
    "get_top_level_model",
    "prepare_multimodal_indices",
    "resolve_runner_type",
    "set_spec_metadata_all_rank_num_tokens",
    "ship_multimodal_indices",
]


def resolve_runner_type(
    model: nn.Module,
    llm_args: TorchLlmArgs,
) -> type[ModelRunner] | None:
    """Resolve the startup runner class without constructing its dependencies."""
    if llm_args.encode_only and not llm_args.mm_encoder_only:
        return None  # Encoder-only execution uses the existing encoder path.

    if llm_args.mm_encoder_only:
        return MultimodalEncoderRunner

    if not model.model_config.is_generation:
        return PoolingRunner

    if model.model_config.is_encoder_decoder:
        return None  # Encoder-decoder execution uses the existing engine path.

    return None  # Decoder execution uses the existing engine path.
