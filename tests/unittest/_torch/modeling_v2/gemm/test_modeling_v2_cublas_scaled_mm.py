# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the cublas_scaled_mm entry (FP8 per-tensor GEMM).

The entry owns the contract. This file turns a cell into operands and holds
the claims no cell carries: that the band rejects a wrong scale, that the
scales are read at graph replay, the guards against the layouts the op
accepts and misreads, and the loud rejections the note relies on.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_scaled_mm import (  # noqa: E501
    cublas_scaled_mm as op,
)

__extra_import_path__ = [".."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    ULP,
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "cublas_scaled_mm requires a CUDA device"

DEV = "cuda"
E4M3 = torch.float8_e4m3fn


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """One bf16 rounding at the row's scale, and almost never that.

    The operands are e4m3, so every product is exact in fp32 and kernel and
    reference differ only in the order fp32 sums them -- relative 1e-7, far
    below the output's rounding. What remains is the final rounding to bf16:
    an exact sum within that 1e-7 of a rounding midpoint can round either way,
    and one bf16 ulp of an element is at most 2^-7 of its row's largest
    magnitude, i.e. 2 units of the 2^-8 the gate counts in.

    So: 2 ulp element-wise against the row scale; 0.25 ulp of relative RMS.
    Measured across the cells: the largest element difference is exactly that
    one ulp, on 0 to 0.04% of elements; the largest relative RMS is 5.3e-5,
    0.014 ulp. Against a float64 product both the op and the reference sit
    within 0.5002 ulp -- the output's own rounding -- so the op is as accurate
    as bf16 allows. A 1% scale error misses the RMS gate tenfold (see below).
    """
    assert_within_ulp(out, ref, element_ulp=2.0, rms_ulp=0.25, ulp=ULP[torch.bfloat16])


def _build(spec, seed: int):
    """Materialize one cell as the checkpoint presents it.

    e4m3 codes from a gaussian at a quarter of the e4m3 range (a few saturate),
    the weight built [N, K] and passed as `.t()`, `input_scale` as fp32 [1] and
    `weight_scale` as fp32 0-d -- the two shapes the checkpoint stores.
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    m, k, n = spec["m"], spec["k"], spec["n"]
    mat_a = (torch.randn(m, k, generator=g, device=DEV) * 112.0).clamp(-448, 448).to(E4M3)
    weight = (torch.randn(n, k, generator=g, device=DEV) * 112.0).clamp(-448, 448).to(E4M3)
    scale_a = torch.tensor([spec["scale_a"]], dtype=torch.float32, device=DEV)
    scale_b = torch.tensor(spec["scale_b"], dtype=torch.float32, device=DEV)
    return mat_a, weight.t(), scale_a, scale_b


# ── the certified cells ───────────────────────────────────────────────────


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    mat_a, mat_b, scale_a, scale_b = _build(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    out_dtype = cell.spec["out_dtype"]
    out = op(mat_a, mat_b, scale_a, scale_b, None, out_dtype)
    again = op(mat_a, mat_b, scale_a, scale_b, None, out_dtype)
    assert out.shape == (mat_a.shape[0], mat_b.shape[1]) and out.dtype is out_dtype
    assert torch.equal(out, again), "two identical calls returned different bits"
    _compare(out, op.reference(mat_a, mat_b, scale_a, scale_b, None, out_dtype))


# ── what a cell cannot carry ──────────────────────────────────────────────


def _small(m=16, k=4096, n=8192, seed=11):
    spec = dict(m=m, k=k, n=n, scale_a=0.05, scale_b=0.002)
    return _build(spec, seed)


def test_the_band_rejects_a_one_percent_scale_error():
    """The gate is not loose: a 1% error in either scale fails it.

    A 1% scale error is a 0.01 relative error everywhere, i.e. 2.56 ulp of
    RMS against a gate of 0.25 -- and 2.56 ulp at the row maximum against an
    element gate of 2.
    """
    mat_a, mat_b, scale_a, scale_b = _small()
    out = op(mat_a, mat_b, scale_a * 1.01, scale_b, None, torch.bfloat16)
    with pytest.raises(AssertionError):
        _compare(out, op.reference(mat_a, mat_b, scale_a, scale_b, None, torch.bfloat16))


def test_a_graph_replay_reads_the_scales_and_operands_it_is_given():
    """The scales are device pointers, not values captured with the graph."""
    mat_a, mat_b, scale_a, scale_b = _small(m=7)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(mat_a, mat_b, scale_a, scale_b, None, torch.bfloat16)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(mat_a, mat_b, scale_a, scale_b, None, torch.bfloat16)
    g = torch.Generator(device=DEV).manual_seed(12)
    for sa, sb in ((0.3, 0.001), (1.8214285373687744, 0.0038888114504516125)):
        mat_a.copy_((torch.randn(mat_a.shape, generator=g, device=DEV) * 112).clamp(-448, 448))
        scale_a.fill_(sa)
        scale_b.fill_(sb)
        graph.replay()
        torch.cuda.synchronize()
        _compare(captured, op.reference(mat_a, mat_b, scale_a, scale_b, None, torch.bfloat16))


def test_guard_refuses_a_k_split_weight_view():
    """out_proj's per-rank weight as a column slice of the checkpoint tensor.

    The op takes the view's `.t()` -- it is column-major with a 16-aligned
    stride -- and multiplies the wrong weights. The materialized slice is what
    a loader must store.
    """
    mat_a, _, scale_a, scale_b = _small()
    full = (torch.randn(8192, 4 * 4096, device=DEV) * 112).clamp(-448, 448).to(E4M3)
    view = full[:, 4096:8192]
    ref = op.reference(mat_a, view.t(), scale_a, scale_b, None, torch.bfloat16)
    silent = op(mat_a, view.t(), scale_a, scale_b, None, torch.bfloat16)
    assert (silent != ref).float().mean().item() > 0.9, "the op stopped misreading the view"
    _compare(op(mat_a, view.contiguous().t(), scale_a, scale_b, None, torch.bfloat16), ref)
    with pytest.raises(AssertionError, match="column-major"):
        with validating(op):
            op(mat_a, view.t(), scale_a, scale_b, None, torch.bfloat16)


def test_guard_refuses_a_row_strided_mat_a():
    mat_a, mat_b, scale_a, scale_b = _small()
    wide = torch.zeros(mat_a.shape[0], 2 * mat_a.shape[1], dtype=torch.uint8, device=DEV)
    wide[:, : mat_a.shape[1]] = mat_a.view(torch.uint8)
    view = wide.view(E4M3)[:, : mat_a.shape[1]]
    ref = op.reference(mat_a, mat_b, scale_a, scale_b, None, torch.bfloat16)
    silent = op(view, mat_b, scale_a, scale_b, None, torch.bfloat16)
    assert (silent != ref).float().mean().item() > 0.5, "the op stopped misreading the view"
    with pytest.raises(AssertionError, match="row-major"):
        with validating(op):
            op(view, mat_b, scale_a, scale_b, None, torch.bfloat16)


def test_guard_refuses_a_scale_vector():
    """A per-row `scale_a` is accepted; only its first element is applied."""
    mat_a, mat_b, _, scale_b = _small()
    vec = torch.linspace(0.01, 0.1, mat_a.shape[0], device=DEV)
    silent = op(mat_a, mat_b, vec, scale_b, None, torch.bfloat16)
    first = op(mat_a, mat_b, vec[:1].contiguous(), scale_b, None, torch.bfloat16)
    assert torch.equal(silent, first), "the op started honouring a per-row scale"
    with pytest.raises(AssertionError, match="exactly one value"):
        with validating(op):
            op(mat_a, mat_b, vec, scale_b, None, torch.bfloat16)


def test_guard_refuses_a_bf16_scale():
    """Accepted by the op; its two bytes and two more are read as an fp32."""
    mat_a, mat_b, scale_a, scale_b = _small()
    with pytest.raises(AssertionError, match="float32"):
        with validating(op):
            op(mat_a, mat_b, scale_a.to(torch.bfloat16), scale_b, None, torch.bfloat16)


@pytest.mark.parametrize("case", ("fp8 output", "K not a multiple of 16", "no rows"))
def test_loud_rejections(case):
    """The preconditions the note calls loud, raised by the op itself."""
    mat_a, mat_b, scale_a, scale_b = _small()
    out_dtype = torch.bfloat16
    if case == "fp8 output":
        out_dtype = None
    elif case == "K not a multiple of 16":
        mat_a = mat_a[:, :4088].contiguous()
        mat_b = mat_b.t()[:, :4088].contiguous().t()
    else:
        mat_a = mat_a[:0]
    with pytest.raises(RuntimeError):
        op(mat_a, mat_b, scale_a, scale_b, None, out_dtype)
        torch.cuda.synchronize()
