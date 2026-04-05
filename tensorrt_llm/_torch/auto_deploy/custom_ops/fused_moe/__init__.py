# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Fused MoE operations.

This module provides various Mixture-of-Experts implementations:
- torch_moe: PyTorch reference implementations (torch_moe, torch_moe_dense_mlp, etc.)
- triton_moe: Triton-based fused MoE (vLLM-inspired)
- triton_moe_dense_mlp: Triton-accelerated dense MoE with fused GLU activation
- trtllm_moe: TRT-LLM optimized MoE
- mxfp4_moe: MXFP4 quantized MoE
- triton_routing: Triton-based routing kernels
- load_moe_align: MoE alignment utilities
"""

__all__ = [
    "torch_moe",
    "triton_moe",
    "triton_moe_dense_mlp",
    "trtllm_moe",
    "mxfp4_moe",
    "triton_routing",
    "load_moe_align",
]
