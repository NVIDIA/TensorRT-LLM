# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RocketKV module-layer integration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm.logger import logger

from ..hooks import AttentionSparseHooks, register_attention_sparse_hooks

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.attention import Attention
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig


def initialize_sparse_attn(
    self,
    *,
    config,
    mapping,
    mapping_o,
    rms_norm_eps: float,
    quant_config,
    q_scaling: float,
    bias: bool,
    dtype: torch.dtype,
    reduce_output: bool,
    aux_stream: Optional[torch.cuda.Stream],
) -> None:
    """Configure the Attention module for RocketKV."""
    del config, mapping, mapping_o, rms_norm_eps, quant_config, q_scaling
    del bias, dtype, reduce_output, aux_stream

    logger.warning_once(
        "disable rope_fusion for RocketKV.",
        key="disable_rope_fusion_for_rocketkv",
    )
    self.rope_fusion = False


class RocketKVHooks(AttentionSparseHooks):
    """Typed RocketKV adapter for the shared Attention module."""

    def initialize(
        self,
        attention: Attention,
        *,
        config: ModelConfig,
        mapping: Mapping,
        mapping_o: Mapping,
        rms_norm_eps: float,
        quant_config: QuantConfig,
        q_scaling: float,
        bias: bool,
        dtype: torch.dtype,
        reduce_output: bool,
        aux_stream: Optional[torch.cuda.Stream],
    ) -> None:
        initialize_sparse_attn(
            attention,
            config=config,
            mapping=mapping,
            mapping_o=mapping_o,
            rms_norm_eps=rms_norm_eps,
            quant_config=quant_config,
            q_scaling=q_scaling,
            bias=bias,
            dtype=dtype,
            reduce_output=reduce_output,
            aux_stream=aux_stream,
        )


register_attention_sparse_hooks("rocket", RocketKVHooks)
