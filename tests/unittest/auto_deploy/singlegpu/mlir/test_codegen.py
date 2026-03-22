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

"""Tests for Triton kernel generation from MLIR fused ops."""

import pytest
import torch

xdsl = pytest.importorskip("xdsl")


@pytest.mark.parametrize("hidden_size", [128, 1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_triton_fused_add_rmsnorm_correctness(hidden_size, dtype):
    """Generated Triton kernel matches reference PyTorch implementation."""
    from tensorrt_llm._torch.auto_deploy.mlir.codegen.templates.fused_add_rmsnorm import (
        fused_add_rmsnorm,
    )

    bsz, seq_len = 2, 8
    x = torch.randn(bsz, seq_len, hidden_size, device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.ones(hidden_size, device="cuda", dtype=dtype)
    eps = 1e-5

    # Reference: manual add + rmsnorm
    ref_add = x + residual
    ref_add_f32 = ref_add.float()
    variance = ref_add_f32.pow(2).mean(-1, keepdim=True)
    ref_norm = (ref_add_f32 / torch.sqrt(variance + eps) * weight.float()).to(dtype)

    # Generated kernel
    norm_out, add_out = fused_add_rmsnorm(x, residual, weight, eps)

    torch.testing.assert_close(add_out, ref_add, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(norm_out, ref_norm, atol=1e-2, rtol=1e-2)


def test_triton_codegen_preexisting_mode():
    """TritonCodegen in 'preexisting' mode returns FlashInfer impl."""
    from tensorrt_llm._torch.auto_deploy.custom_ops.normalization.flashinfer_fused_add_rms_norm import (
        flashinfer_fused_add_rms_norm,
    )
    from tensorrt_llm._torch.auto_deploy.mlir.codegen.triton_emitter import TritonCodegen

    codegen = TritonCodegen(mode="preexisting")
    impl = codegen.get_fused_add_rmsnorm_impl()
    assert impl is flashinfer_fused_add_rms_norm


def test_triton_codegen_generate_mode():
    """TritonCodegen in 'generate' mode returns a callable."""
    from tensorrt_llm._torch.auto_deploy.mlir.codegen.triton_emitter import TritonCodegen

    codegen = TritonCodegen(mode="generate")
    impl = codegen.get_fused_add_rmsnorm_impl()
    assert callable(impl)

    # Verify it works
    x = torch.randn(2, 8, 128, device="cuda", dtype=torch.bfloat16)
    res = torch.randn_like(x)
    w = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    norm_out, add_out = impl(x, res, w, 1e-5)
    assert norm_out.shape == x.shape
    assert add_out.shape == x.shape


@pytest.mark.parametrize("hidden_size", [128, 1024])
def test_generated_vs_flashinfer(hidden_size):
    """Generated Triton kernel produces same results as FlashInfer."""
    from tensorrt_llm._torch.auto_deploy.custom_ops.normalization.flashinfer_fused_add_rms_norm import (
        flashinfer_fused_add_rms_norm,
    )
    from tensorrt_llm._torch.auto_deploy.mlir.codegen.templates.fused_add_rmsnorm import (
        fused_add_rmsnorm as triton_impl,
    )

    bsz, seq_len = 4, 16
    x = torch.randn(bsz, seq_len, hidden_size, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
    eps = 1e-5

    # FlashInfer result
    fi_norm, fi_add = flashinfer_fused_add_rms_norm(x.clone(), residual.clone(), weight, eps)

    # Generated Triton result
    tr_norm, tr_add = triton_impl(x.clone(), residual.clone(), weight, eps)

    torch.testing.assert_close(tr_add, fi_add, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(tr_norm, fi_norm, atol=1e-2, rtol=1e-2)
