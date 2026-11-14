# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the glm5_next fp8 tp4 mhc_gemm_sqrsum entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.mhc.mhc_gemm_sqrsum import (  # noqa: E501
    TILE_N_OPTIONS,
    gemm_sqrsum_tile_n,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.mhc.mhc_gemm_sqrsum import (  # noqa: E501
    mhc_gemm_sqrsum as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = [".", "../../../.."]
from _glm53_mhc_inputs import generator, mixer, streams  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 100, reason="the glm5_next fp8 tp4 entries are certified on sm_100"
)

#: The band, as a fraction of the summed magnitudes; see `_compare`.
BAND = 2.0**-18


def _compare(out, ref, x: torch.Tensor, fn: torch.Tensor) -> None:
    """Every output within ``2^-18`` of the sum of its terms' magnitudes.

    Both sides add the same terms -- a BF16 by FP32 product is exact in FP32 -- in different
    orders: the kernel in a 64-term FMA chain per thread, then a warp butterfly and a sum over 8
    warps; the reference in the FP32 GEMM's and the reduction's own orders. The rounding error of a
    sum is a few FP32 ulp (2^-24) of the sum of its terms' magnitudes in a typical draw, and grows
    with the chain length only in the worst case; 2^-18 is 64 such ulp, which covers both orders'
    tails over every output of a cell. For the sum of squares that magnitude is the result itself.
    A lost 1024-element step moves a mixer output by about 32 of its typical terms, some 2^-8 of
    the magnitude sum of 16384 terms: about a thousand times the band.
    """
    mixes, sqrsum = out
    ref_mixes, ref_sqrsum = ref
    with true_fp32_matmul():
        magnitude = x.float().abs() @ fn.float().abs().t()
    excess = ((mixes - ref_mixes).abs() / (BAND * magnitude)).max().item()
    assert excess <= 1.0, f"mixes off by {excess:.2f}x the band"
    excess = ((sqrsum - ref_sqrsum).abs() / (BAND * ref_sqrsum)).max().item()
    assert excess <= 1.0, f"sqrsum off by {excess:.2f}x the band"


def _inputs(rows: int, seed: int):
    gen = generator(seed)
    residual = streams(rows, gen)
    fn, _, _ = mixer(gen)
    return residual.view(rows, -1), fn


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    """Every tile width, against the reference; the tile width moves no bit."""
    certifying(op)
    rows = cell.spec["rows"]
    x, fn = _inputs(rows, seed=zlib.crc32(cell.why.encode()) % 2**31)
    ref = op.reference(x, fn, gemm_sqrsum_tile_n(rows))
    first = None
    for tile_n in TILE_N_OPTIONS:
        out = op(x, fn, tile_n)
        assert out[0].shape == (rows, 24) and out[1].shape == (rows,)
        _compare(out, ref, x, fn)
        if first is None:
            first = out
        else:
            assert torch.equal(out[0], first[0]) and torch.equal(out[1], first[1]), (
                f"tile_n={tile_n} changed the sums"
            )


def test_cuda_graph_capture_and_replay() -> None:
    """The decode step's first pre-mapping is captured: replay on new streams."""
    x, fn = _inputs(4, seed=31)
    tile_n = gemm_sqrsum_tile_n(4)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(x, fn, tile_n)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(x, fn, tile_n)
    x.copy_(_inputs(4, seed=32)[0])
    graph.replay()
    torch.cuda.synchronize()
    _compare(out, op.reference(x, fn, tile_n), x, fn)


def test_guard_refuses_what_the_kernel_misreads() -> None:
    """The kernel reads both operands with a row stride of K, whatever their strides."""
    x, fn = _inputs(4, seed=7)
    wide = torch.zeros(4, 2 * x.shape[1], dtype=x.dtype, device="cuda")
    cases = {
        "fn transposed in memory": (x, fn.t().contiguous().t()),
        "x a column slice": (wide[:, : x.shape[1]], fn),
        "fn of another width": (x, fn[:, : x.shape[1] // 2].contiguous()),
    }
    for name, args in cases.items():
        with pytest.raises(AssertionError):
            with validating(op):
                op(*args, 1)
            pytest.fail(f"is_valid accepted {name}")
