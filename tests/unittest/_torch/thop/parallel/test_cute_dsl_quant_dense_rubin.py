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
"""Dispatch tests for the SM107 CuTe DSL quantized dense GEMM/BMM custom ops.

The ops are registered only when the CuTe DSL package provides the SM107
helpers, and every one of them raises unless get_sm_version() == 107. These
tests run on every architecture and check both halves of that contract; the
SM107 correctness tests live in test_fp8_block_scale_gemm.py and
test_fp8_linear.py.
"""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from utils.util import getSMVersion

import tensorrt_llm._torch.custom_ops  # noqa: F401
from tensorrt_llm._torch.cute_dsl_utils import (
    IS_CUTLASS_DSL_AVAILABLE,
    IS_CUTLASS_DSL_RUBIN_AVAILABLE,
)

skip_no_cute_dsl = pytest.mark.skipif(
    not IS_CUTLASS_DSL_AVAILABLE, reason="cutlass-dsl is not available"
)
skip_no_sm107_cute_dsl = pytest.mark.skipif(
    not IS_CUTLASS_DSL_RUBIN_AVAILABLE, reason="CuTe DSL package without SM107 support"
)
skip_on_sm107 = pytest.mark.skipif(getSMVersion() == 107, reason="checks the non-SM107 path")

SM107_QUANT_OPS = (
    "cute_dsl_fp8_gemm_rubin",
    "cute_dsl_fp8_bmm_rubin",
    "cute_dsl_fp8_bmm_quantize_rubin_out",
    "cute_dsl_fp8_per_tensor_gemm_rubin",
    "cute_dsl_mxfp8_gemm_rubin",
    "cute_dsl_nvfp4_gemm_inplace_rubin",
    "cute_dsl_nvfp4_gemm_locality_domain_inplace_rubin",
    "cute_dsl_dsv4_qb_gemm_fused_rmsnorm_rope_quant",
)

SM107_QUANT_RUNNERS = (
    "CuteDSLFp8RubinGemmRunner",
    "CuteDSLFp8RubinBmmRunner",
    "CuteDSLFp8RubinPerTensorGemmRunner",
    "CuteDSLMXFP8RubinLinear",
    "CuteDSLNVFP4RubinLinear",
)


def _fp8(*shape):
    return torch.empty(*shape, dtype=torch.float8_e4m3fn, device="cuda")


def _f32(*shape):
    return torch.empty(*shape, dtype=torch.float32, device="cuda")


def _u8(*shape):
    return torch.empty(*shape, dtype=torch.uint8, device="cuda")


def _op_calls():
    """One representative call per op with shape-consistent operands."""
    m, n, k, b = 8, 256, 512, 2
    return {
        "cute_dsl_fp8_gemm_rubin": lambda op: op(
            _fp8(m, k), _fp8(n, k), _f32(k // 128, m), _f32(n // 128, k // 128)
        ),
        "cute_dsl_fp8_bmm_rubin": lambda op: op(
            _fp8(b, m, k),
            _fp8(b, n, k),
            _f32(b, k // 128, m),
            _f32(b, n // 128, k // 128),
            torch.empty(b, m, n, dtype=torch.bfloat16, device="cuda"),
        ),
        "cute_dsl_fp8_bmm_quantize_rubin_out": lambda op: op(
            _fp8(b, m, k),
            _fp8(b, n, k),
            _f32(b, k // 128, m),
            _f32(b, n // 128, k // 128),
            _fp8(b, m, n),
            _u8(128 * ((b * n // 32 + 3) // 4 * 4)),
        ),
        "cute_dsl_fp8_per_tensor_gemm_rubin": lambda op: op(
            _fp8(m, k), _fp8(n, k), _f32(1), _f32(1)
        ),
        "cute_dsl_mxfp8_gemm_rubin": lambda op: op(
            _fp8(m, k), _fp8(n, k), _u8(128 * (k // 32)), _u8(n * (k // 32))
        ),
        "cute_dsl_nvfp4_gemm_inplace_rubin": lambda op: op(
            _u8(m, k // 2),
            _u8(n, k // 2),
            _u8(128 * (k // 16)),
            _u8(n * (k // 16)),
            _f32(1),
            torch.bfloat16,
            False,
            True,
            torch.empty(m, n, dtype=torch.bfloat16, device="cuda"),
            0,
        ),
        "cute_dsl_nvfp4_gemm_locality_domain_inplace_rubin": lambda op: op(
            _u8(m, k // 2),
            _u8(n, k // 2),
            _u8(n, k // 2),
            _u8(128 * (k // 16)),
            _u8(n * (k // 16)),
            _u8(n * (k // 16)),
            _f32(1),
            torch.bfloat16,
            False,
            True,
            torch.empty(m, 2 * n, dtype=torch.bfloat16, device="cuda"),
        ),
        "cute_dsl_dsv4_qb_gemm_fused_rmsnorm_rope_quant": lambda op: op(
            _fp8(m, k),
            _fp8(512, k),
            _u8(128 * (k // 32)),
            _u8(512 * (k // 32)),
            _f32(m, 128),
            torch.tensor([0, m], dtype=torch.int32, device="cuda"),
            torch.tensor([m], dtype=torch.int32, device="cuda"),
            torch.arange(m, dtype=torch.int32, device="cuda"),
            _f32(1),
            1e-6,
        ),
    }


@skip_no_cute_dsl
def test_sm107_quant_ops_registered_only_with_sm107_cute_dsl():
    for name in SM107_QUANT_OPS:
        assert hasattr(torch.ops.trtllm, name) == IS_CUTLASS_DSL_RUBIN_AVAILABLE, name


@skip_no_sm107_cute_dsl
@skip_on_sm107
@pytest.mark.parametrize("op_name", SM107_QUANT_OPS)
def test_sm107_quant_ops_reject_other_archs(op_name):
    op = getattr(torch.ops.trtllm, op_name)
    with pytest.raises(ValueError, match="SM ?107"):
        _op_calls()[op_name](op)


@skip_no_sm107_cute_dsl
@skip_on_sm107
@pytest.mark.parametrize("runner_name", SM107_QUANT_RUNNERS)
def test_sm107_quant_runners_offer_no_tactics_off_sm107(runner_name):
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops

    runner_class = getattr(cute_dsl_custom_ops, runner_name)
    if runner_name.endswith("Linear"):
        runner = runner_class(output_dtype=torch.bfloat16)
        inputs = [_fp8(8, 512), _fp8(256, 512), _u8(2048), _u8(4096), _f32(1)]
    elif runner_name == "CuteDSLFp8RubinBmmRunner":
        runner = runner_class()
        inputs = [
            _fp8(2, 8, 512),
            _fp8(2, 256, 512),
            _f32(2, 4, 8),
            _f32(2, 2, 4),
            torch.empty(2, 8, 256, dtype=torch.bfloat16, device="cuda"),
        ]
    else:
        runner = runner_class()
        inputs = [_fp8(8, 512), _fp8(256, 512), _f32(4, 8), _f32(2, 4)]
    assert runner.get_valid_tactics(inputs, None) == []


@skip_no_sm107_cute_dsl
def test_sm107_quant_ops_fake_registration():
    m, n, k, b = 8, 256, 512, 2
    with FakeTensorMode():
        out = torch.ops.trtllm.cute_dsl_fp8_gemm_rubin(
            _fp8(m, k), _fp8(n, k), _f32(k // 128, m), _f32(n // 128, k // 128)
        )
        assert out.shape == (m, n) and out.dtype == torch.bfloat16
        out = torch.ops.trtllm.cute_dsl_fp8_per_tensor_gemm_rubin(
            _fp8(m, k), _fp8(n, k), _f32(1), _f32(1), output_dtype=torch.float16
        )
        assert out.shape == (m, n) and out.dtype == torch.float16
        out = torch.ops.trtllm.cute_dsl_mxfp8_gemm_rubin(
            _fp8(m, k), _fp8(n, k), _u8(128 * (k // 32)), _u8(n * (k // 32))
        )
        assert out.shape == (m, n) and out.dtype == torch.bfloat16
        out = torch.ops.trtllm.cute_dsl_dsv4_qb_gemm_fused_rmsnorm_rope_quant(
            _fp8(m, k),
            _fp8(512, k),
            _u8(128 * (k // 32)),
            _u8(512 * (k // 32)),
            _f32(m, 128),
            torch.tensor([0, m], dtype=torch.int32, device="cuda"),
            torch.tensor([m], dtype=torch.int32, device="cuda"),
            torch.arange(m, dtype=torch.int32, device="cuda"),
            _f32(1),
            1e-6,
        )
        assert out.shape == (m, 512) and out.dtype == torch.float8_e4m3fn
        bmm_out = torch.empty(b, m, n, dtype=torch.bfloat16, device="cuda")
        torch.ops.trtllm.cute_dsl_fp8_bmm_rubin(
            _fp8(b, m, k), _fp8(b, n, k), _f32(b, k // 128, m), _f32(b, n // 128, k // 128), bmm_out
        )
        with pytest.raises(AssertionError):
            torch.ops.trtllm.cute_dsl_fp8_bmm_rubin(
                _fp8(b, m, k),
                _fp8(b, n, k),
                _f32(b, k // 128, m),
                _f32(b, n // 128, k // 128),
                torch.empty(b, m, n, dtype=torch.float32, device="cuda"),
            )


# --------------------------------------------------------------------------
# The existing Blackwell call sites keep their routing.
# --------------------------------------------------------------------------


@skip_no_cute_dsl
@skip_on_sm107
def test_fp8_bmm_helper_routes_blackwell():
    from tensorrt_llm._torch.attention.mla import _is_cute_dsl_fp8_bmm_available

    assert _is_cute_dsl_fp8_bmm_available() == (getSMVersion() in (100, 103))
    assert _is_cute_dsl_fp8_bmm_available(107) == IS_CUTLASS_DSL_RUBIN_AVAILABLE
    assert not _is_cute_dsl_fp8_bmm_available(90)


@skip_on_sm107
def test_fp8_block_scales_sm107_predicates_off_sm107():
    from types import SimpleNamespace

    from tensorrt_llm._torch.modules.linear import (
        _fp8_block_scales_uses_cute_dsl_sm107,
        _fp8_per_tensor_uses_cute_dsl_sm107,
    )

    module = SimpleNamespace(use_cute_dsl_blockscaling_mm=True, disable_deep_gemm=True)
    assert not _fp8_block_scales_uses_cute_dsl_sm107(module)
    assert not _fp8_per_tensor_uses_cute_dsl_sm107()
