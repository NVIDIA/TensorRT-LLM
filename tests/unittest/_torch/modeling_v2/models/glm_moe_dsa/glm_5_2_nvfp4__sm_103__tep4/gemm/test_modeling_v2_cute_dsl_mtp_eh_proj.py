# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_mtp_eh_proj entry."""

import zlib

import pytest
import torch
from torch import nn

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel needs an sm_103 GPU", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm import (  # noqa: E402, E501
    cute_dsl_mtp_eh_proj as entry,
)

__extra_import_path__ = ["../../../.."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
    validating,
)

op = entry.cute_dsl_mtp_eh_proj
N, K = 6144, entry.LOCAL_INPUT_SIZE


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """2 ulp per element against the row scale, 0.5 ulp relative RMS.

    Kernel and reference take bit-identical bf16 operands, accumulate in fp32
    and round to bf16 once; they differ only in summation order over K=3072.
    Over random operands that order costs about sqrt(K) fp32 ulps of the
    partial sums, around 1e-6 of the row's largest output, which is three
    orders under one bf16 ulp. What is left is the rare output whose fp32 value
    sits that close to a bf16 rounding midpoint: a one-ulp flip, up to 2 units
    of `ULP[bf16]` against the row scale. A wrong tile, a missing K block or a
    transposed weight misses by O(1).
    """
    assert_within_ulp(out, ref, element_ulp=2.0, rms_ulp=0.5)


@pytest.fixture(scope="module")
def projection():
    """One rank's weight slice and its runner: sixteen GEMM compiles, done once."""
    g = torch.Generator(device="cuda").manual_seed(0)
    weight = nn.Parameter(
        (torch.randn(N, K, generator=g, device="cuda") / K**0.5).to(torch.bfloat16),
        requires_grad=False,
    )
    return weight, entry.build(weight)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, projection, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    weight, runner = projection
    rows = cell.spec["rows"]
    assert rows in entry.SUPPORTED_M_VALUES
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    # The compact [M, 3072] buffer cute_dsl_mtp_norm_shard writes.
    x = torch.randn(rows, K, generator=g, device="cuda").to(torch.bfloat16)
    out = op(x, weight, runner)
    assert out.shape == (rows, N) and out.dtype == torch.bfloat16 and out.is_contiguous()
    _compare(out, op.reference(x, weight, runner))


def test_guard_refuses_a_runner_built_on_another_slice(projection) -> None:
    """The runner multiplies by its own weight and ignores the one passed."""
    weight, runner = projection
    other = nn.Parameter(weight.detach().clone(), requires_grad=False)
    x = torch.randn(1, K, device="cuda").to(torch.bfloat16)
    with pytest.raises(AssertionError, match="different weight"):
        with validating(op):
            op(x, other, runner)


def test_guard_refuses_a_runner_with_a_collective(projection) -> None:
    """With a collective the runner returns the reduced sum, not this rank's partial."""
    weight, runner = projection
    x = torch.randn(1, K, device="cuda").to(torch.bfloat16)
    # Set on the shared runner rather than building a second one, which would
    # cost another sixteen compiles; restored before any other test sees it.
    runner.all_reduce = lambda t, **kwargs: t
    try:
        with pytest.raises(AssertionError, match="without a collective"):
            with validating(op):
                op(x, weight, runner)
    finally:
        runner.all_reduce = None
