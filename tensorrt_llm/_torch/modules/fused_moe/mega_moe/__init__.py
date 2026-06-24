# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MegaMoE first-class MoE backends.

Two backends share the ``MoESchedulerKind.FUSED_COMM`` contract:

* :class:`MegaMoEDeepGemm` — DeepGEMM ``fp8_fp4_mega_moe`` fused kernel for
  W4A8_MXFP4_MXFP8 weights. ``W4A8MXFP4MXFP8MegaMoEDeepGemmMethod`` owns the
  DG-native weight tensors, scale conversion, and DeepGEMM weight transform.
* :class:`MegaMoECuteDsl` — CuteDSL ``Sm100MegaMoEKernel`` fused dispatch +
  FC1 + activation + FC2 + combine kernel for NVFP4 weights. The kernel and
  helper sources are ported into
  ``tensorrt_llm/_torch/cute_dsl_kernels/mega_moe_nvfp4``;
  ``NVFP4MegaMoECuteDslMethod`` owns the NVFP4 weight tensors, MegaMoE-format
  derived buffers, and per-expert scale tensors consumed by the kernel ABI.
"""

from ..quantization import NVFP4MegaMoECuteDslMethod, W4A8MXFP4MXFP8MegaMoEDeepGemmMethod
from .mega_moe_cute_dsl import (
    MegaMoECuteDsl,
    MegaMoeCuteDslUnavailable,
    MegaMoECuteDslWeightView,
    is_megamoe_cute_dsl_runtime_available,
)
from .mega_moe_deepgemm import MegaMoEDeepGemm

__all__ = [
    "MegaMoECuteDsl",
    "MegaMoECuteDslWeightView",
    "MegaMoeCuteDslUnavailable",
    "MegaMoEDeepGemm",
    "NVFP4MegaMoECuteDslMethod",
    "W4A8MXFP4MXFP8MegaMoEDeepGemmMethod",
    "is_megamoe_cute_dsl_runtime_available",
]
