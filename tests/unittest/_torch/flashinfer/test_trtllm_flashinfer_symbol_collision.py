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
"""
Unit tests verifying no symbol collision between TensorRT-LLM and FlashInfer.

FlashInfer copies several TensorRT-LLM CUTLASS MOE kernel source files
(under nv_internal/tensorrt_llm/) and JIT-compiles them. Without the
inline-namespace fix (TRTLLM_ABI_NAMESPACE _v1), the resulting .so exports
symbols with identical mangled names as libth_common.so (loaded with
RTLD_GLOBAL), causing heap corruption when the dynamic linker resolves
to the wrong implementation.

This test triggers the FlashInfer CUTLASS fused-MOE JIT build (with
use_fast_build=True to minimize compilation time), then calls into the
compiled module to verify no symbol collision occurs.
"""

import pytest
import torch

import tensorrt_llm._torch.custom_ops.torch_custom_ops as trt_ops  # noqa: F401


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flashinfer_cutlass_fused_moe_jit_no_collision():
    """
    JIT-compiling and calling FlashInfer CUTLASS fused-MOE must not crash.

    get_cutlass_fused_moe_module JIT-compiles TensorRT-LLM CUTLASS MOE
    kernels that share the tensorrt_llm:: namespace with libth_common.so.
    use_fast_build=True reduces template instantiations for speed.
    The actual collision manifests when module.init() is called (inside
    cutlass_fused_moe), so we must invoke the operation with small tensors.
    """
    from flashinfer.fused_moe.core import get_cutlass_fused_moe_module

    sm = torch.cuda.get_device_capability()
    backend = str(sm[0] * 10 + sm[1])
    fused_moe_ns = get_cutlass_fused_moe_module(backend, use_fast_build=True)

    device = "cuda"
    dtype = torch.bfloat16
    M, H, N, E, top_k = 4, 64, 128, 4, 2

    x = torch.randn(M, H, device=device, dtype=dtype)
    w1_w3 = torch.randn(E, 2 * N, H, device=device, dtype=dtype)
    w2 = torch.randn(E, H, N, device=device, dtype=dtype)

    logits = torch.randn(M, E, device=device, dtype=torch.float32)
    weights, experts = torch.topk(torch.softmax(logits, -1), top_k)
    weights = weights / weights.sum(-1, keepdim=True)

    fused_moe_ns.cutlass_fused_moe(
        output=torch.empty(M, H, device=device, dtype=dtype),
        input=x,
        token_selected_experts=experts.to(torch.int32),
        token_final_scales=weights,
        fc1_expert_weights=w1_w3,
        fc1_expert_biases=None,
        fc2_expert_weights=w2,
        fc2_expert_biases=None,
        output_dtype=dtype,
        quant_scales=[],
    )
