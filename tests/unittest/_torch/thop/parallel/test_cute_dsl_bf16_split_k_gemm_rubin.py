# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Split-K correctness tests for the Rubin (SM107) BF16 dense GEMM runner.

Exercises the ``CuteDSLBf16RubinGemmRunner`` split-K path end to end: the GEMM
writes FP32 partials into a workspace and a reduction kernel sums them into the
output.  The reference is ``act @ weight.T``.
"""

import pytest
import torch
from _torch.thop.parallel._cute_dsl_bf16_rubin_test_utils import (
    RUBIN_CUTE_DSL_MARKS,
    make_bf16_gemm_runner,
    reset_bf16_gemm_state,
    run_ugpu_composite,
    select_bf16_tactic,
    skip_if_no_ugpu,
)

pytestmark = RUBIN_CUTE_DSL_MARKS


@pytest.mark.parametrize("split_k_slices", [2, 4, 8])
@pytest.mark.parametrize("c_dtype", [torch.bfloat16, torch.float32])
def test_cute_dsl_bf16_split_k_gemm_rubin(split_k_slices, c_dtype):
    """Split-K GEMM matches the dense reference for large-K, small-N shapes."""
    torch.manual_seed(2026)
    runner = make_bf16_gemm_runner()

    # Large K and small N so get_valid_tactics offers split>1 candidates.
    m, n, k = 256, 256, 8192
    act = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    output = torch.empty(m, n, dtype=c_dtype, device="cuda")

    tactics = runner.get_valid_tactics([act, weight, output], None)
    tactic = select_bf16_tactic(tactics, "base", split_k_slices=split_k_slices)

    expected = act.float() @ weight.t().float()
    runner([act, weight, output], tactic=tactic)
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), expected, rtol=1e-2, atol=1.0)

    if c_dtype == torch.float32 and split_k_slices in (2, 4):
        split1_output = torch.empty_like(output)
        split1_tactic = select_bf16_tactic(tactics, "base", split_k_slices=1)
        runner([act, weight, split1_output], tactic=split1_tactic)
        torch.cuda.synchronize()
        # Both accumulate in FP32; only the K reduction order differs.
        torch.testing.assert_close(output, split1_output, rtol=1e-3, atol=1e-2)


def test_cute_dsl_bf16_split_k_ugpu_rubin():
    """Split-K runs one tactic concurrently across both real uGPU partitions."""
    skip_if_no_ugpu()

    torch.manual_seed(99)
    reset_bf16_gemm_state()

    split_k_slices = 4
    m, n, k = 256, 256, 8192
    act = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    weight_0 = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    weight_1 = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    expected_0 = act.float() @ weight_0.t().float()
    expected_1 = act.float() @ weight_1.t().float()

    wide_output = torch.empty(m, n * 2, dtype=torch.bfloat16, device="cuda")
    run_ugpu_composite(
        "cute_dsl_bf16_gemm_ugpu_inplace_rubin",
        (act, weight_0, weight_1, wide_output),
        (expected_0, expected_1),
        partition_dim=1,
        kernel_variant="base",
        split_k_slices=split_k_slices,
        capture_graph=False,
    )
