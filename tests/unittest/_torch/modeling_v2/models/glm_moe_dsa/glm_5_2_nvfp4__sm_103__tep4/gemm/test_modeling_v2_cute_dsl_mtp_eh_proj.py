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
    """One rank's weight slice and its runner: seventeen GEMM compiles, done once."""
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
    assert rows in entry.SUPPORTED_M_VALUES + entry.MTP_ROWS
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    # The compact [M, 3072] buffer cute_dsl_mtp_norm_shard writes.
    x = torch.randn(rows, K, generator=g, device="cuda").to(torch.bfloat16)
    out = op(x, weight, runner)
    assert out.shape == (rows, N) and out.dtype == torch.bfloat16 and out.is_contiguous()
    _compare(out, op.reference(x, weight, runner))


@pytest.mark.parametrize("rows", entry.MTP_ROWS)
def test_mtp_rows_replay_bitwise_in_a_cuda_graph(rows, projection) -> None:
    """MTP5's draft step 0 runs this inside the decode CUDA graph: a replay writes the
    eager launch's bytes, and a replay after a new input is copied into the captured
    buffer writes the eager launch's product on it. (The weight is the runner's own, fixed
    at load.)"""
    weight, runner = projection
    g = torch.Generator(device="cuda").manual_seed(rows)
    x = torch.randn(rows, K, generator=g, device="cuda").to(torch.bfloat16)
    eager = op(x, weight, runner).clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(x, weight, runner)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager), "the replay differs from the eager launch"
    x.copy_(torch.randn(rows, K, generator=g, device="cuda").to(torch.bfloat16))
    again = op(x, weight, runner).clone()
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, again), "the replay on a new input differs from the eager launch"
    assert not torch.equal(again, eager), "the new input did not change the product"
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


#: MTP3 / MTP5 draft-step-0 row counts without an instance: each bucket's edges, the
#: 24-request batches (96, 144) and partial batches at widths 4 / 6.
WIDE_ROWS = (65, 68, 96, 100, 126, 127, 129, 144, 150, 191)


def test_wide_rows_on_the_next_instance_are_its_rows_bitwise(projection) -> None:
    """`at_rows` runs a wide decode row count without an instance on the next one up
    (`WIDE_ROWS`: 128 / 192) at its exact rows: each row bitwise what that instance writes
    for it with the input padded by other (random) rows, within the band; a captured
    launch replays bitwise on a new input. The instances' own counts and 64 rows refuse."""
    weight, runner = projection
    g = torch.Generator(device="cuda").manual_seed(17)
    for rows in WIDE_ROWS:
        r = entry.at_rows(runner, rows)
        assert r is not None and r.weight is weight and r.all_reduce is None
        bucket = min(m for m in entry.WIDE_ROWS if m > rows)
        x = torch.randn(rows, K, generator=g, device="cuda").to(torch.bfloat16)
        with validating(op):
            out = op(x, weight, r)
        padded = torch.randn(bucket, K, generator=g, device="cuda").to(torch.bfloat16)
        padded[:rows] = x
        full = op(padded, weight, runner)
        torch.cuda.synchronize()
        assert out.shape == (rows, N) and torch.equal(out, full[:rows]), (
            f"{rows} rows: not the {bucket}-row instance's"
        )
        _compare(out, op.reference(x, weight, r))
    rows = 96
    r = entry.at_rows(runner, rows)
    x = torch.randn(rows, K, generator=g, device="cuda").to(torch.bfloat16)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(x, weight, r)
    x.copy_(torch.randn(rows, K, generator=g, device="cuda").to(torch.bfloat16))
    again = op(x, weight, r).clone()
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, again), "the replay on a new input differs from the eager launch"
    for rows in (64, 128, 192, 193):
        assert entry.at_rows(runner, rows) is None, rows


#: MTP5 at 1-3 requests (6 / 12 / 18 rows; MTP3 at 3 is 12): captured decode row counts without
#: an instance, which the target runs on the next instance up to 64 (`_cell_up`).
DECODE_ROWS = (6, 12, 18)


def test_decode_rows_on_the_next_instance_are_its_rows_bitwise(projection) -> None:
    """`at_rows` at the MTP decode row counts up to 64 without an instance: the next
    `SUPPORTED_M_VALUES` instance up at the exact rows, each row bitwise that instance's
    with the input padded by other (random) rows, within the band; a captured launch replays
    bitwise on a new input. A count with its own instance (16, the control) refuses."""
    weight, runner = projection
    g = torch.Generator(device="cuda").manual_seed(23)
    for rows in DECODE_ROWS:
        if rows in entry.SUPPORTED_M_VALUES:
            continue
        r = entry.at_rows(runner, rows)
        bucket = min(m for m in entry.SUPPORTED_M_VALUES if rows < m <= 64)
        assert r is not None and r.bucket == bucket and r.rows == rows
        x = torch.randn(rows, K, generator=g, device="cuda").to(torch.bfloat16)
        with validating(op):
            out = op(x, weight, r)
        padded = torch.randn(bucket, K, generator=g, device="cuda").to(torch.bfloat16)
        padded[:rows] = x
        full = op(padded, weight, runner)
        torch.cuda.synchronize()
        assert out.shape == (rows, N) and torch.equal(out, full[:rows]), (
            f"{rows} rows: not the {bucket}-row instance's"
        )
        _compare(out, op.reference(x, weight, r))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = op(x, weight, r)
        x.copy_(torch.randn(rows, K, generator=g, device="cuda").to(torch.bfloat16))
        again = op(x, weight, r).clone()
        captured.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, again), f"{rows} rows: the replay differs"
    assert 16 in entry.SUPPORTED_M_VALUES and entry.at_rows(runner, 16) is None
