# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Startup-time model-runner dispatch and exports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from .decoder import DecoderRunner
from .encoder import EncoderRunner
from .encoder_decoder import EncoderDecoderRunner

if TYPE_CHECKING:
    from .interface import ModelRunner

__all__ = [
    "DecoderRunner",
    "EncoderDecoderRunner",
    "EncoderRunner",
    "resolve_runner_type",
]


def resolve_runner_type(model: nn.Module) -> type[ModelRunner]:
    """Resolve the startup runner class without constructing its dependencies."""
    if model.model_config.is_encoder_decoder:
        return EncoderDecoderRunner

    return DecoderRunner
