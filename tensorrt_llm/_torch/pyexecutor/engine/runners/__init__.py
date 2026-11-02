# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Startup-time model-runner dispatch and exports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

from .decoder import DecoderRunner
from .encoder import EncoderRunner
from .encoder_decoder import EncoderDecoderRunner
from .mm_encoder import MultimodalEncoderRunner
from .pooling import PoolingRunner

if TYPE_CHECKING:
    from .interface import ModelRunner

__all__ = [
    "DecoderRunner",
    "EncoderDecoderRunner",
    "EncoderRunner",
    "resolve_runner_type",
]


def resolve_runner_type(
    model: nn.Module,
    llm_args: TorchLlmArgs,
) -> type[ModelRunner]:
    """Resolve the startup runner class without constructing its dependencies."""
    if llm_args.encode_only and not llm_args.mm_encoder_only:
        return EncoderRunner

    if llm_args.mm_encoder_only:
        return MultimodalEncoderRunner

    if not model.model_config.is_generation:
        return PoolingRunner

    if model.model_config.is_encoder_decoder:
        return EncoderDecoderRunner

    return DecoderRunner
