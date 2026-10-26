# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the cuda_tile_rms_norm_fuse_residual entry.

Inputs are drawn over what Step-5-Preview's own tensors measure, so neither a
near-unit scale nor a well-conditioned row hides a wrong normalization:

- residual rows: RMS log-uniform over [0.375, 165], the per-row RMS range of
  the fp32 residual stream entering layers 3..91 in the reference oracle, with
  four outlier channels at 50x the row scale (the oracle's max |x| is 9375);
- x, the widened attention/MLP output added to it: 0.15x the row scale (the
  oracle's layer-87 -> 88 delta has row RMS <= 23.9 against a residual of 146);
- weight: fp32, uniform over [-1.00133, 0.324887], the range of the
  checkpoint's 184 input/post-attention norm weights of layers 0..91. The op
  applies it as (1 + w).
"""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.cuda_tile_rms_norm_fuse_residual import (  # noqa: E501
    PATHS,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.cuda_tile_rms_norm_fuse_residual import (  # noqa: E501
    cuda_tile_rms_norm_fuse_residual as op,
)
from tensorrt_llm._torch.cuda_tile_utils import IS_CUDA_TILE_AVAILABLE

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "cuda_tile_rms_norm_fuse_residual requires a CUDA device"
assert IS_CUDA_TILE_AVAILABLE and hasattr(torch.ops.trtllm, "cuda_tile_rms_norm_fuse_residual_"), (
    "the CUDA Tile toolchain is absent, so the op is not registered"
)

DEVICE = torch.device("cuda")
HIDDEN = 4096
EPS = 1e-5
#: One fp32 ulp at [1, 2).
U32 = 2.0**-23
#: Per-element band on the normalized output, relative to the fp64 reference.
BAND_ULP = 5.0
ROW_RMS = (0.375, 165.0)
WEIGHT_RANGE = (-1.00133, 0.324887)


def _gen(seed: int) -> torch.Generator:
    return torch.Generator(device=DEVICE).manual_seed(seed)


def _draw(rows: int, n: int, gen: torch.Generator, row_rms=None):
    if row_rms is None:
        lo, hi = math.log(ROW_RMS[0]), math.log(ROW_RMS[1])
        scale = (torch.rand(rows, 1, generator=gen, device=DEVICE) * (hi - lo) + lo).exp()
    else:
        scale = torch.full((rows, 1), row_rms, device=DEVICE)
    residual = torch.randn(rows, n, generator=gen, device=DEVICE) * scale
    outliers = torch.rand(n, generator=gen, device=DEVICE).argsort()[:4]
    residual[:, outliers] *= 50.0
    x = torch.randn(rows, n, generator=gen, device=DEVICE) * scale * 0.15
    w_lo, w_hi = WEIGHT_RANGE
    weight = torch.rand(n, generator=gen, device=DEVICE) * (w_hi - w_lo) + w_lo
    return x.contiguous(), residual.contiguous(), weight


def _build(spec, seed):
    x, residual, weight = _draw(spec["rows"], spec["hidden"], _gen(seed))
    dtype = spec["dtype"]
    return (
        x.to(dtype),
        residual.to(dtype),
        weight.to(dtype),
        spec["eps"],
        spec["static_persistent"],
        spec["gather"],
        spec["use_gemma"],
    )


def _excess(out: torch.Tensor, ref: torch.Tensor) -> float:
    """Worst per-element |out - ref| / |ref|, in fp32 ulps (exact zeros must match)."""
    diff = (out.double() - ref.double()).abs()
    rel = torch.where(diff == 0, torch.zeros_like(diff), diff / ref.double().abs())
    return rel.max().item() / U32


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Per element, |out - ref| <= 5 fp32 ulp of |ref|.

    The kernel forms fl(fl(h * rsqrt(fl(S / N + eps))) * fl(1 + w)), S the fp32
    sum of 4096 squares. The chain is multiplicative, with no cancellation
    (1 + w is exact or rounded once from two exact operands), so the error is
    relative per element: three roundings of <= 0.5 ulp (1.5), the hardware
    rsqrt (about 1), the mean and eps add (0.5), and half the relative error
    of S, a per-lane fp32 accumulation and a tree reduction over 4096 terms
    whose random-walk error stays well under 4 ulp (2). The fp64 reference is
    rounded once. Measured on GB300: <= 3.13 ulp over 1..8192 rows on all three
    paths and on the oracle's real residual stream; torch's own fp32
    composition lands at <= 3.63 ulp on the same rows. Tighter than torch's
    default fp32 rtol (1.3e-6, about 10.9 ulp), so `test_band_rejects_wrong_variants`
    is a discrimination check, not a licence for a wide band.
    """
    assert out.dtype == ref.dtype and out.shape == ref.shape
    worst = _excess(out, ref)
    assert worst <= BAND_ULP, f"normalized output off by {worst:.2f} fp32 ulp > {BAND_ULP}"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    x, residual = args[0], args[1]
    ref_x, ref_residual = op.reference(*args)
    weight_before = args[2].clone()
    out_x, out_residual = op(*args)
    assert out_x is x and out_residual is residual, "the op writes in place"
    assert torch.equal(args[2], weight_before), "weight must not be written"
    assert torch.equal(residual, ref_residual), "the residual is fp32 + fp32 rounded once"
    _compare(x, ref_x)


@pytest.mark.parametrize("path", list(PATHS), ids=list(PATHS.values()))
@pytest.mark.parametrize("rows", [1, 37])
def test_rows_past_the_view_are_untouched(path, rows):
    """A chunk's rows are a view of a larger stream buffer; the op writes only its rows."""
    static_persistent, gather = path
    x, residual, weight = _draw(rows + 8, HIDDEN, _gen(rows))
    x0, r0 = x.clone(), residual.clone()
    op(x[:rows], residual[:rows], weight, EPS, static_persistent, gather, True)
    assert torch.equal(x[rows:], x0[rows:]) and torch.equal(residual[rows:], r0[rows:])
    ref_x, ref_residual = op.reference(
        x0[:rows], r0[:rows], weight, EPS, static_persistent, gather, True
    )
    assert torch.equal(residual[:rows], ref_residual)
    _compare(x[:rows], ref_x)


@pytest.mark.parametrize("path", list(PATHS), ids=list(PATHS.values()))
def test_cuda_graph_capture_and_replay(path):
    """The decode graph captures both norms of every layer: replay on refilled inputs equals eager."""
    static_persistent, gather = path
    x, residual, weight = _draw(128, HIDDEN, _gen(11))
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(x.clone(), residual.clone(), weight, EPS, static_persistent, gather, True)  # compile
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(x, residual, weight, EPS, static_persistent, gather, True)
    x1, r1, _ = _draw(128, HIDDEN, _gen(12))
    x.copy_(x1)
    residual.copy_(r1)
    graph.replay()
    torch.cuda.synchronize()
    xe, re = x1.clone(), r1.clone()
    op(xe, re, weight, EPS, static_persistent, gather, True)
    assert torch.equal(x, xe) and torch.equal(residual, re)


def test_band_rejects_wrong_variants():
    """Each plausible mistake lands far outside the 5-ulp band.

    Rows at RMS 0.375 (the stream's smallest) make eps visible: 1e-6 instead of
    1e-5 moves the output by 9e-6 / (2 * 0.375^2) relative, about 270 ulp.
    """
    gen = _gen(5)
    x, residual, weight = _draw(64, HIDDEN, gen, row_rms=0.375)
    x0, r0 = x.clone(), residual.clone()
    op(x, residual, weight, EPS, True, True, True)
    h = x0.double() + r0.double()
    ms = h.pow(2).mean(-1, keepdim=True)

    variants = {
        "weight without the 1 +": h * torch.rsqrt(ms + EPS) * weight.double(),
        "eps 1e-6": h * torch.rsqrt(ms + 1e-6) * (weight.double() + 1),
        "variance over N - 1": h
        * torch.rsqrt(ms * HIDDEN / (HIDDEN - 1) + EPS)
        * (weight.double() + 1),
        "normalize x, not the sum": x0.double()
        * torch.rsqrt(x0.double().pow(2).mean(-1, keepdim=True) + EPS)
        * (weight.double() + 1),
    }
    _compare(x, op.reference(x0, r0, weight, EPS, True, True, True)[0])
    for name, wrong in variants.items():
        excess = _excess(x, wrong.float())
        assert excess > 20 * BAND_ULP, f"{name}: only {excess:.1f} ulp from the op"


# ---------------------------------------------------------------- guards
def test_guard_refuses_aliased_buffers():
    """Unguarded, x passed as its own residual returns norm(2x) and loses the residual."""
    x, _, weight = _draw(16, HIDDEN, _gen(3))
    x0 = x.clone()
    op(x, x, weight, EPS, True, True, True)
    _compare(x, op.reference(2 * x0, torch.zeros_like(x0), weight, EPS, True, True, True)[0])
    with pytest.raises(AssertionError, match="distinct buffers"):
        with validating(op):
            op(x, x, weight, EPS, True, True, True)


@pytest.mark.parametrize("path", list(PATHS), ids=list(PATHS.values()))
def test_guard_refuses_a_short_weight(path):
    """Unguarded, a weight 16 short of the row is answered, with the missing tail as padding."""
    static_persistent, gather = path
    x, residual, weight = _draw(16, HIDDEN, _gen(3))
    buf = torch.full((HIDDEN,), 7.0, device=DEVICE)  # the tail the op must not read as weight
    buf[: HIDDEN - 16] = weight[: HIDDEN - 16]
    short = buf[: HIDDEN - 16]
    x0, r0 = x.clone(), residual.clone()
    op(x, residual, short, EPS, static_persistent, gather, True)
    assert torch.equal(buf[HIDDEN - 16 :], torch.full((16,), 7.0, device=DEVICE))
    full = torch.cat([short, buf[HIDDEN - 16 :]])
    excess = _excess(x, op.reference(x0, r0, full, EPS, static_persistent, gather, True)[0])
    assert excess > 20 * BAND_ULP, f"short weight answered within {excess:.1f} ulp"
    with pytest.raises(AssertionError, match="missing tail"):
        with validating(op):
            op(x0, r0, short, EPS, static_persistent, gather, True)


def test_guard_refuses_a_residual_with_fewer_rows():
    """Unguarded (persistent path), the rows the residual lacks are normalized as if it were 0.

    The residual is a view of a buffer the test owns, so the rows the op
    reads past its end are in-bounds memory and any stray write would show.
    """
    x, residual, weight = _draw(16, HIDDEN, _gen(3))
    x0, r0 = x.clone(), residual.clone()
    op(x, residual[:8], weight, EPS, True, True, True)
    assert torch.equal(residual[8:], r0[8:]), "rows past the view were written"
    _compare(
        x[8:], op.reference(x0[8:], torch.zeros_like(x0[8:]), weight, EPS, True, True, True)[0]
    )
    assert (
        _excess(x[8:], op.reference(x0[8:], r0[8:], weight, EPS, True, True, True)[0])
        > 20 * BAND_ULP
    )
    with pytest.raises(AssertionError, match="missing rows"):
        with validating(op):
            op(x0, r0[:8], weight, EPS, True, True, True)


def test_guard_refuses_a_ragged_row_on_the_tiled_path():
    """Unguarded, the tiled path at N 4000 (not a multiple of its 512 tile) answers wrongly;
    the persistent and gather paths answer the same inputs within the band.

    x and residual are views of a buffer one row longer, so a stray write past
    the last row would show.
    """
    n = 4000
    x, residual, weight = _draw(17, n, _gen(9))
    x0, r0 = x.clone(), residual.clone()
    ref_x, ref_residual = op.reference(x0[:16], r0[:16], weight, EPS, False, False, True)
    for static_persistent, gather in ((True, True), (False, True)):
        xs, rs = x0[:16].clone(), r0[:16].clone()
        op(xs, rs, weight, EPS, static_persistent, gather, True)
        _compare(xs, ref_x)
    op(x[:16], residual[:16], weight, EPS, False, False, True)
    assert torch.equal(x[16:], x0[16:]) and torch.equal(residual[16:], r0[16:])
    assert torch.equal(residual[:16], ref_residual)
    bad = ~torch.isfinite(x[:16]).all(-1) | (
        ((x[:16].double() - ref_x.double()).abs() > BAND_ULP * U32 * ref_x.double().abs()).any(-1)
    )
    assert bad.any(), "the tiled path answered a ragged row within the band"
    with pytest.raises(AssertionError, match="tiled path"):
        with validating(op):
            op(x0[:16].clone(), r0[:16].clone(), weight, EPS, False, False, True)


def test_the_guard_is_not_on_the_hot_path():
    """Outside `validating`, the wrapper goes straight to the kernel."""
    x, residual, weight = _draw(16, HIDDEN, _gen(3))
    op(x, residual[:8], weight, EPS, True, True, True)
