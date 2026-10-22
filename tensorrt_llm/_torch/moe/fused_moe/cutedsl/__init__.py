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
"""The CuteDSL MoE implementations that share code, laid out like their ids.

An id is ``trtllm.cutedsl.<kernel>.<quant>``: one subpackage or module per
kernel, one module per quant where a kernel serves several. Bottom to top:

* :mod:`.identity` -- the provider / technique / kernel strings.
* :mod:`.kernel_inputs` -- what the kernels of more than one leaf take, as free
  functions: the NVFP4 weight view, input quantization, the finalize scales.
* :mod:`.runners` -- the outer autotune runners.
* :mod:`.grouped_gemm` -- FC1 and FC2 as two grouped-GEMM launches: an abstract
  root and one leaf per quant.
* :mod:`.fused_fc12` -- FC1 and FC2 in one persistent kernel.
* :mod:`.dense_gemm` -- every expert packed into one dense matrix.

Inheritance stays inside a kernel; kernels share only the modules above them.
``trtllm.cutedsl.mega_moe.nvfp4`` lives in :mod:`..mega_moe` with the other
fused-communication kernel.
"""

from .dense_gemm import TrtllmCutedslDenseGemmNvfp4Impl
from .fused_fc12 import TrtllmCutedslFusedFc12Nvfp4Impl
from .grouped_gemm import (
    CuteDslFusedMoEBase,
    TrtllmCutedslGroupedGemmBf16Impl,
    TrtllmCutedslGroupedGemmNvfp4Impl,
    cutedsl_grouped_gemm_leaf,
    find_cutedsl_grouped_gemm_leaf,
)

__all__ = [
    "CuteDslFusedMoEBase",
    "TrtllmCutedslDenseGemmNvfp4Impl",
    "TrtllmCutedslFusedFc12Nvfp4Impl",
    "TrtllmCutedslGroupedGemmBf16Impl",
    "TrtllmCutedslGroupedGemmNvfp4Impl",
    "cutedsl_grouped_gemm_leaf",
    "find_cutedsl_grouped_gemm_leaf",
]
