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
"""Tests for the SM107 CuTe DSL BF16 dense GEMM/BMM custom ops.

The dispatch tests run on every architecture with CuTe DSL installed and check
that ``trtllm::cute_dsl_bf16_{gemm,bmm}_rubin`` are unreachable off SM107. The
correctness tests need SM107 and a CuTe DSL package with the SM107 helpers.
"""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from utils.util import getSMVersion

import tensorrt_llm._torch.custom_ops  # noqa: F401
from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.cute_dsl_utils import (
    IS_CUTLASS_DSL_AVAILABLE,
    IS_CUTLASS_DSL_RUBIN_AVAILABLE,
)

skip_no_cute_dsl = pytest.mark.skipif(
    not IS_CUTLASS_DSL_AVAILABLE, reason="cutlass-dsl is not available"
)
skip_not_sm107 = pytest.mark.skipif(
    getSMVersion() != 107 or not IS_CUTLASS_DSL_RUBIN_AVAILABLE,
    reason="requires SM107 and a CuTe DSL package with SM107 support",
)
skip_on_sm107 = pytest.mark.skipif(getSMVersion() == 107, reason="checks the non-SM107 path")


def _runner_class(name):
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops

    return getattr(cute_dsl_custom_ops, name)


def _fresh_runner(name, **kwargs):
    runner_class = _runner_class(name)
    AutoTuner.get().clear_cache()
    runner_class.kernel_cache.clear()
    if hasattr(runner_class, "split_k_gemm_cache"):
        runner_class.split_k_gemm_cache.clear()
    return runner_class(**kwargs)


def _select_tactic(tactics, kernel_variant, split_k_slices=None):
    candidates = [t for t in tactics if t[0] == kernel_variant and t[1] is False]
    if split_k_slices is not None:
        candidates = [t for t in candidates if len(t) == 6 and t[-1] == split_k_slices]
    assert candidates, f"no {kernel_variant} tactic (split_k={split_k_slices})"
    return candidates[0]


def _gemm_operands(m, n, k, c_dtype=torch.bfloat16):
    act = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    output = torch.empty(m, n, dtype=c_dtype, device="cuda")
    return act, weight, output


def _bmm_operands(b, m, n, k):
    act = torch.randn(b, m, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(b, n, k, dtype=torch.bfloat16, device="cuda")
    output = torch.empty(b, m, n, dtype=torch.bfloat16, device="cuda")
    return act, weight, output


# --------------------------------------------------------------------------
# Dispatch: the SM107 ops must be a no-op everywhere else.
# --------------------------------------------------------------------------


@skip_no_cute_dsl
@skip_on_sm107
def test_sm107_bf16_gemm_rejects_other_archs():
    act, weight, output = _gemm_operands(64, 128, 256)
    with pytest.raises(ValueError, match="SM107"):
        torch.ops.trtllm.cute_dsl_bf16_gemm_rubin(act, weight, output)


@skip_no_cute_dsl
@skip_on_sm107
def test_sm107_bf16_bmm_rejects_other_archs():
    act, weight, output = _bmm_operands(2, 64, 128, 256)
    with pytest.raises(ValueError, match="SM107"):
        torch.ops.trtllm.cute_dsl_bf16_bmm_rubin(act, weight, output)


@skip_no_cute_dsl
@skip_on_sm107
def test_sm107_bf16_runners_offer_no_tactics_off_sm107():
    gemm_runner = _fresh_runner("CuteDSLBf16RubinGemmRunner")
    assert gemm_runner.get_valid_tactics(list(_gemm_operands(64, 128, 256)), None) == []
    bmm_runner = _fresh_runner("CuteDSLBf16RubinBmmRunner")
    assert bmm_runner.get_valid_tactics(list(_bmm_operands(2, 64, 128, 256)), None) == []


@skip_no_cute_dsl
def test_sm107_bf16_ops_fake_registration():
    with FakeTensorMode():
        act = torch.empty(8, 32, dtype=torch.bfloat16, device="cuda")
        weight = torch.empty(16, 32, dtype=torch.bfloat16, device="cuda")
        for c_dtype in (torch.bfloat16, torch.float32):
            output = torch.empty(8, 16, dtype=c_dtype, device="cuda")
            torch.ops.trtllm.cute_dsl_bf16_gemm_rubin(act, weight, output)
        with pytest.raises(AssertionError):
            torch.ops.trtllm.cute_dsl_bf16_gemm_rubin(
                act, weight, torch.empty(8, 17, dtype=torch.bfloat16, device="cuda")
            )

        act = torch.empty(2, 8, 32, dtype=torch.bfloat16, device="cuda")
        weight = torch.empty(2, 16, 32, dtype=torch.bfloat16, device="cuda")
        output = torch.empty(2, 8, 16, dtype=torch.bfloat16, device="cuda")
        torch.ops.trtllm.cute_dsl_bf16_bmm_rubin(act, weight, output)
        with pytest.raises(AssertionError):
            torch.ops.trtllm.cute_dsl_bf16_bmm_rubin(
                act, weight, torch.empty(2, 8, 16, dtype=torch.float32, device="cuda")
            )


# --------------------------------------------------------------------------
# SM107 correctness.
# --------------------------------------------------------------------------


@skip_not_sm107
@pytest.mark.parametrize("c_dtype", [torch.bfloat16, torch.float32])
def test_cute_dsl_bf16_gemm_rubin_op(c_dtype):
    torch.manual_seed(0)
    AutoTuner.get().clear_cache()
    act, weight, output = _gemm_operands(256, 1024, 2048, c_dtype)
    torch.ops.trtllm.cute_dsl_bf16_gemm_rubin(act, weight, output)
    torch.cuda.synchronize()
    expected = act.float() @ weight.t().float()
    torch.testing.assert_close(output.float(), expected, rtol=1e-2, atol=1.0)


@skip_not_sm107
@pytest.mark.parametrize("kernel_variant", ["base", "preferred_cluster"])
def test_cute_dsl_bf16_gemm_rubin_tactics(kernel_variant):
    torch.manual_seed(1)
    runner = _fresh_runner("CuteDSLBf16RubinGemmRunner", output_dtype=torch.bfloat16)
    act, weight, output = _gemm_operands(1024, 2048, 1024)
    tactics = runner.get_valid_tactics([act, weight, output], None)
    tactic = _select_tactic(tactics, kernel_variant)
    runner([act, weight, output], tactic=tactic)
    torch.cuda.synchronize()
    expected = act.float() @ weight.t().float()
    torch.testing.assert_close(output.float(), expected, rtol=1e-2, atol=1.0)


@skip_not_sm107
@pytest.mark.parametrize("split_k_slices", [2, 4, 8])
@pytest.mark.parametrize("c_dtype", [torch.bfloat16, torch.float32])
def test_cute_dsl_bf16_split_k_gemm_rubin(split_k_slices, c_dtype):
    """Split-K matches the dense reference for large-K, small-N shapes."""
    torch.manual_seed(2026)
    runner = _fresh_runner("CuteDSLBf16RubinGemmRunner", output_dtype=c_dtype)

    # Large K and small N so get_valid_tactics offers split>1 candidates.
    act, weight, output = _gemm_operands(64, 256, 7168, c_dtype)
    tactics = runner.get_valid_tactics([act, weight, output], None)
    tactic = _select_tactic(tactics, "base", split_k_slices=split_k_slices)

    # Direct split-K rounds each partial to the output dtype before the
    # atomic TMA ADD, so BF16 output needs tolerance for both rounding and
    # arrival-order changes.
    rtol, atol = (2e-2, 2.5) if c_dtype == torch.bfloat16 else (1e-2, 1.0)

    expected = act.float() @ weight.t().float()
    runner([act, weight, output], tactic=tactic)
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), expected, rtol=rtol, atol=atol)

    # A second launch must not accumulate on the previous output; poisoning C
    # also catches a missing zero inside CUDA-graph replay and normal dispatch.
    output.fill_(float("nan"))
    runner([act, weight, output], tactic=tactic)
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), expected, rtol=rtol, atol=atol)

    if c_dtype == torch.float32 and split_k_slices in (2, 4):
        split1_output = torch.empty_like(output)
        split1_tactic = _select_tactic(tactics, "base", split_k_slices=1)
        runner([act, weight, split1_output], tactic=split1_tactic)
        torch.cuda.synchronize()
        # Both write FP32; direct split-K only changes the reduction order.
        torch.testing.assert_close(output, split1_output, rtol=1e-3, atol=1e-2)


@skip_not_sm107
def test_cute_dsl_bf16_bmm_rubin_op():
    torch.manual_seed(3)
    AutoTuner.get().clear_cache()
    act, weight, output = _bmm_operands(4, 256, 512, 1024)
    torch.ops.trtllm.cute_dsl_bf16_bmm_rubin(act, weight, output)
    torch.cuda.synchronize()
    expected = torch.bmm(act.float(), weight.transpose(1, 2).float())
    torch.testing.assert_close(output.float(), expected, rtol=1e-2, atol=1.0)


@skip_not_sm107
@pytest.mark.parametrize("kernel_variant", ["base", "preferred_cluster"])
def test_cute_dsl_bf16_bmm_rubin_tactics(kernel_variant):
    torch.manual_seed(4)
    runner = _fresh_runner("CuteDSLBf16RubinBmmRunner")
    act, weight, output = _bmm_operands(2, 1024, 1024, 512)
    tactics = runner.get_valid_tactics([act, weight, output], None)
    tactic = _select_tactic(tactics, kernel_variant)
    runner([act, weight, output], tactic=tactic)
    torch.cuda.synchronize()
    expected = torch.bmm(act.float(), weight.transpose(1, 2).float())
    torch.testing.assert_close(output.float(), expected, rtol=1e-2, atol=1.0)


@skip_not_sm107
def test_cute_dsl_bf16_bmm_rubin_strided_views():
    """Non-contiguous A/B views with K innermost are consumed without a copy;
    a K-non-innermost view is rejected instead of computing the wrong product."""
    torch.manual_seed(5)
    runner = _fresh_runner("CuteDSLBf16RubinBmmRunner")
    b, m, n, k = 4, 128, 256, 512
    # [M, B, K] storage viewed as [B, M, K] and a broadcast batch for B.
    act_storage = torch.randn(m, b, k, dtype=torch.bfloat16, device="cuda")
    act = act_storage.transpose(0, 1)
    weight = torch.randn(1, n, k, dtype=torch.bfloat16, device="cuda").expand(b, n, k)
    output = torch.empty(b, m, n, dtype=torch.bfloat16, device="cuda")
    tactics = runner.get_valid_tactics([act, weight, output], None)
    runner([act, weight, output], tactic=_select_tactic(tactics, "base"))
    torch.cuda.synchronize()
    expected = torch.bmm(act.float(), weight.transpose(1, 2).float())
    torch.testing.assert_close(output.float(), expected, rtol=1e-2, atol=1.0)

    bad_weight = torch.randn(b, k, n, dtype=torch.bfloat16, device="cuda").transpose(1, 2)
    with pytest.raises(ValueError, match="K innermost"):
        runner([act, bad_weight, output], tactic=_select_tactic(tactics, "base"))
