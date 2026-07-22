# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RocketKV module-layer integration."""

from typing import Optional

import torch

from tensorrt_llm.logger import logger


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
