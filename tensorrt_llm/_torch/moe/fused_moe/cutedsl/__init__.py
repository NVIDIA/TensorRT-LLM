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
"""CuteDSL MoE implementations, laid out like their ids ``trtllm.cutedsl.<kernel>.<quant>``.

See "CuteDSL leaves" in MOE_DEVELOPER_GUIDE.md for the layout.
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
