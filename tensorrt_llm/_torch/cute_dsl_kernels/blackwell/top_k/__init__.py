# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CuTE DSL Top-K kernels for Blackwell architecture."""

from .filtered_top_k_decode_varlen import (
    FilteredTopKKernelVarlenDecode,
    cute_dsl_topk_multi_cta_wrapper,
    cute_dsl_topk_wrapper,
    run_topk_decode,
)
from .filtered_top_k_varlen_util import (
    FilteredTopKKernelVarlen,
    compare_top_k_results,
    create_random_logits,
    run_reference_top_k,
)

__all__ = [
    "FilteredTopKKernelVarlenDecode",
    "FilteredTopKKernelVarlen",
    "cute_dsl_topk_wrapper",
    "cute_dsl_topk_multi_cta_wrapper",
    "run_topk_decode",
    "create_random_logits",
    "run_reference_top_k",
    "compare_top_k_results",
]
