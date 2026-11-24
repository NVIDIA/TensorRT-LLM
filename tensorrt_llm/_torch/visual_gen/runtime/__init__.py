# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process configuration, device capabilities and types used by VisualGen models."""

from enum import Enum

from tensorrt_llm._torch.cute_dsl_utils import (
    IS_CUTLASS_DSL_AVAILABLE,
    IS_CUTLASS_DSL_RUBIN_AVAILABLE,
)
from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._torch.utils import get_model_extra_attrs
from tensorrt_llm._utils import get_sm_version, is_sm_100f, mpi_disabled
from tensorrt_llm.bindings import ipc_nvls_supported
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.visual_gen.args import AttentionConfig, QuantAttentionConfig
from tensorrt_llm.visual_gen.params import VisualGenParams
from tensorrt_llm.visual_gen.sparse_attention import (
    SkipSoftmaxAttentionConfig,
    SolAttentionConfig,
    VideoSparseAttentionConfig,
)


class AttentionTensorLayout(str, Enum):
    NHD = "NHD"
    HND = "HND"


__all__ = [
    "IS_CUTLASS_DSL_AVAILABLE",
    "IS_CUTLASS_DSL_RUBIN_AVAILABLE",
    "IS_FLASHINFER_AVAILABLE",
    "AttentionConfig",
    "AttentionTensorLayout",
    "Mapping",
    "QuantAlgo",
    "QuantAttentionConfig",
    "QuantConfig",
    "SkipSoftmaxAttentionConfig",
    "SolAttentionConfig",
    "VideoSparseAttentionConfig",
    "VisualGenParams",
    "get_model_extra_attrs",
    "get_sm_version",
    "ipc_nvls_supported",
    "is_sm_100f",
    "logger",
    "mpi_disabled",
]
