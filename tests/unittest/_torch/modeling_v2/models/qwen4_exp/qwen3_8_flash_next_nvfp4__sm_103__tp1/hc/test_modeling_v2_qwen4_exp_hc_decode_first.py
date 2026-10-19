# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp_hc_decode_first entry.

Cells against the float64 reference: `hc_out` within one ulp (as
`qwen4_exp_hc_inject_norm`), `part` within `2^-16` of the reference (relative
to its largest element, plus the same relative term), `ss` to fp32 accuracy.
The MoE-combine cells widen `hc_out`'s band by the combine's own rounding and
check `part` / `ss` against the projection of the `hc_out` the op stored (see
`_compare_moe`). Beyond the cells: wrong formulas land outside, bitwise
agreement with the combine + this op composition, the weight-orientation
guard, determinism and graph capture.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_hc_decode_first import (  # noqa: E501
    COMBINE_CONFIG,
    FOLD_CONFIG,
    launch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_hc_decode_first import (  # noqa: E501
    qwen4_exp_hc_decode_first as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.moe.fp4_block_scale_moe_renorm_permuted import (  # noqa: E501
    finalize,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.moe.fused_sigmoid_gate_mul_add import (  # noqa: E501
    fused_sigmoid_gate_mul_add,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "qwen4_exp_hc_decode_first requires a CUDA device"

DEV = "cuda"
HC_COUNT, HIDDEN, RANK = 4, 2560, 320
# This checkpoint's layer-0 hyper-connection element statistics.
DOWN_RMS, INJECT_RMS, GAMMA_RMS, GAMMA_ABSMAX = 0.0296, 0.0107, 0.477, 6.625
PART_EPS = 2.0**-16


def _band(got, want, eps) -> float:
    """Worst element's distance as a fraction of `rtol = eps`, `atol = eps * max|want|`."""
    w = want.double()
    return ((got.double() - w).abs() / (eps * w.abs().max().item() + eps * w.abs())).max().item()


def _ulp(got, want) -> float:
    eps, tiny = torch.finfo(want.dtype).eps, torch.finfo(want.dtype).tiny
    w = want.double()
    return ((got.double() - w).abs() / (eps * w.abs() + tiny)).max().item()


def _compare(out, ref) -> None:
    """`hc_out` one ulp: an fp32 accumulate rounded once. `part`: a 2560-term
    fp32 dot per stream, so ~2^-20 relative to its scale is the expected error;
    the gate is 2^-16. `ss`: a 2560-term fp32 sum of squares."""
    (hc, part, ss), (ref_hc, ref_part, ref_ss) = out, ref
    assert hc.dtype == ref_hc.dtype and hc.shape == ref_hc.shape
    assert part.dtype == torch.float32 and part.shape == ref_part.shape and part.stride(2) == 1
    assert ss.dtype == torch.float32 and ss.shape == ref_ss.shape
    if not hc.numel():
        return
    assert _ulp(hc, ref_hc) <= 1.0, f"hc_out {_ulp(hc, ref_hc):.4g} ulp"
    assert _band(part, ref_part, PART_EPS) <= 1.0, f"part {_band(part, ref_part, PART_EPS):.4g}x"
    assert torch.allclose(ss, ref_ss, rtol=1e-5, atol=0), "ss"


def _combine(block_out, shared, gate_logits) -> torch.Tensor:
    """The MoE block output in float64, rounded once to `block_out`'s dtype."""
    s = torch.sigmoid(gate_logits.double().reshape(-1, 1))
    return (block_out.double() + s * shared.double()).to(block_out.dtype)


def _routed(block_out, topk_weights, topk_rows) -> torch.Tensor:
    """The routed row: `block_out` itself, or with the fold the float64
    top-k weighted sum of the permuted rows rounded once to their dtype."""
    if topk_rows is None:
        return block_out
    valid = (topk_rows >= 0).unsqueeze(-1)
    rows = block_out.double()[topk_rows.clamp(min=0).long()]
    return (
        torch.where(valid, rows * topk_weights.double().unsqueeze(-1), 0.0)
        .sum(1)
        .to(block_out.dtype)
    )


def _hc_band_ratio(
    hc, ref_hc, block_out, inject, shared, gate, topk_weights=None, topk_rows=None
) -> float:
    """`hc_out` against the reference: one ulp of `hc_out` plus one ulp of the
    combine times `|inject|` (the op's fp32 combine and the reference's float64
    one can round to neighbouring values at a midpoint); with the fold, one ulp
    of the routed row times `|inject|` more (the op's fp32 FMA finalize and the
    reference's exact sum can round apart the same way)."""
    feps, tiny = torch.finfo(ref_hc.dtype).eps, torch.finfo(ref_hc.dtype).tiny
    routed = _routed(block_out, topk_weights, topk_rows)
    b = _combine(routed, shared, gate).double().abs()
    if topk_rows is not None:
        b = b + routed.double().abs()
    band = feps * ref_hc.double().abs() + (
        feps * b.unsqueeze(-2) * inject.double().abs().unsqueeze(-1)
    ).flatten(-2)
    return ((hc.double() - ref_hc.double()).abs() / (band + tiny)).max().item()


def _compare_moe(args, out, ref) -> None:
    """MoE-combine cells: `hc_out` in `_hc_band_ratio`'s band of the float64
    result; `part` and `ss` as `_compare` against `reference(hc_out, None,
    None, ...)` -- the projection of the residual the op stored."""
    hyper_input, block_out, inject, gamma, first, c, n, shared, gate = args[:9]
    topk_weights, topk_rows = args[9:] if len(args) > 9 else (None, None)
    (hc, part, ss), (ref_hc, _, _) = out, ref
    assert hc.dtype == ref_hc.dtype and hc.shape == ref_hc.shape
    if not hc.numel():
        return
    ratio = _hc_band_ratio(hc, ref_hc, block_out, inject, shared, gate, topk_weights, topk_rows)
    assert ratio <= 1.0, f"hc_out {ratio:.4g}x the band"
    _compare((hc, part, ss), op.reference(hc, None, None, gamma, first, c, n))


def _build(spec, seed):
    rows = spec["rows"]
    hc_count = spec.get("hc_count", HC_COUNT)
    hidden = spec.get("hidden", HIDDEN)
    dtype = spec.get("dtype", torch.bfloat16)
    width = hc_count * hidden
    n_cols = spec.get("n_cols", RANK + hc_count)
    g = torch.Generator(device=DEV).manual_seed(seed)

    def normal(*shape, scale):
        return torch.randn(*shape, generator=g, device=DEV) * scale

    hyper_input = normal(rows, width, scale=3.0).to(dtype)
    if spec.get("layout") == "row_slice":
        buf = normal(2 * rows, width, scale=1.0).to(dtype)
        buf[::2] = hyper_input
        hyper_input = buf[::2]
    gamma = normal(width, scale=GAMMA_RMS)
    gamma[0], gamma[width // 2] = GAMMA_ABSMAX, -GAMMA_ABSMAX
    gamma = gamma.to(dtype)
    n_rows = spec.get("n_rows", -(-n_cols // 128) * 128)
    weight = torch.zeros(n_rows, width, device=DEV)
    weight[:n_cols] = normal(n_cols, width, scale=DOWN_RMS)
    weight[n_cols - hc_count : n_cols] *= INJECT_RMS / DOWN_RMS
    first = weight.to(dtype).t()
    block_out = inject = shared = gate_logits = None
    if spec.get("inject", True):
        block_out = normal(rows, hidden, scale=2.0).to(dtype)
        inject = (2.0 * torch.sigmoid(normal(rows, hc_count, scale=1.0))).to(dtype)
    if spec.get("moe", False):
        shared = normal(rows, hidden, scale=2.0).to(dtype)
        gate_logits = normal(rows, 1, scale=2.0).to(dtype)
        if spec.get("gate") == "flat":
            gate_logits = gate_logits.reshape(rows)
    top_k = spec.get("fold", 0)
    if not top_k:
        return hyper_input, block_out, inject, gamma, first, hc_count, n_cols, shared, gate_logits
    # The runner's unfinalized output: padded permuted rows (garbage where no
    # slot maps), a distinct row per slot, softmax-like bf16 weights.
    n_perm = rows * top_k + 37
    block_out = normal(n_perm, hidden, scale=2.0).to(dtype)
    order = torch.randperm(n_perm, generator=g, device=DEV)[: rows * top_k]
    topk_rows = order.reshape(rows, top_k).to(torch.int32)
    if spec.get("drop"):
        topk_rows[::2, -3:] = -1
    topk_weights = torch.softmax(normal(rows, top_k, scale=1.0), -1).to(dtype)
    return (
        hyper_input,
        block_out,
        inject,
        gamma,
        first,
        hc_count,
        n_cols,
        shared,
        gate_logits,
        topk_weights,
        topk_rows,
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % (2**31))
    before = [a.clone() if isinstance(a, torch.Tensor) else a for a in args]
    out = op(*args)
    for a, b in zip(args, before):
        assert not isinstance(a, torch.Tensor) or torch.equal(a, b), "every input is read-only"
    if args[1] is None:
        assert out[0] is args[0], "norm-only returns hyper_input itself"
    if args[7] is None:
        _compare(out, op.reference(*args))
    else:
        _compare_moe(args, out, op.reference(*args))


def test_measured_distance_from_float64() -> None:
    for rows in (1, 16):
        args = _build(dict(rows=rows), seed=rows)
        (hc, part, ss), (rh, rp, rs) = op(*args), op.reference(*args)
        print(
            f"\n  rows {rows}: hc {_ulp(hc, rh):.3g} ulp, part {_band(part, rp, PART_EPS):.4f}x, "
            f"ss rel {((ss - rs).abs() / rs).max().item():.3g}"
        )


def test_deterministic() -> None:
    args = _build(dict(rows=4), seed=3)
    for a, b in zip(op(*args), op(*args)):
        assert torch.equal(a, b)


def test_wrong_variants_land_outside_the_band() -> None:
    """Each mis-reading keeps the shapes; each lands > 5x outside. The streams
    carry different scales so a stream mix-up is visible."""
    args = list(_build(dict(rows=8), seed=5))
    scales = torch.tensor([0.5, 1.0, 2.0, 4.0], device=DEV).repeat_interleave(HIDDEN)
    args[0] = (args[0].float() * scales).to(torch.bfloat16)
    hyper_input, block_out, inject, gamma, first, c, n, _, _ = args
    want_hc, want_part, _ = op.reference(*args)
    pre = hyper_input.double().unflatten(-1, (c, HIDDEN))
    hc = want_hc.double().unflatten(-1, (c, HIDDEN))
    g = (1.0 + gamma.double()).unflatten(-1, (c, HIDDEN))
    w = first[:, :n].double().unflatten(0, (c, HIDDEN))

    def proj(y):
        return torch.einsum("tsi,sin->stn", y, w)

    variants = {
        "w instead of 1 + w": proj(hc * (g - 1.0)),
        "pre-inject stream": proj(pre * g),
        "no (1 + w)": proj(hc),
        "streams summed into every partial": proj(hc * g).sum(0, keepdim=True).expand(c, -1, -1),
    }
    for name, value in variants.items():
        ratio = _band(value.float(), want_part, PART_EPS)
        print(f"\n  part {name:<36} {ratio:10.4g}x")
        assert ratio > 5.0, name
    stream_minor = pre.flatten(-2) + (
        block_out.double().unsqueeze(-1) * inject.double().unsqueeze(-2)
    ).flatten(-2)
    ratio = _ulp(stream_minor.to(torch.bfloat16), want_hc)
    print(f"  hc_out stream-minor inject {ratio:10.4g} ulp")
    assert ratio > 5.0


def test_guard_refuses_a_row_major_weight() -> None:
    """A contiguous `[width, N]` weight is read with rows and columns swapped
    and returns plausible numbers unguarded."""
    args = list(_build(dict(rows=2), seed=6))
    want = op.reference(*args)[1]
    args[4] = args[4].contiguous()
    assert _band(op(*args)[1], want, PART_EPS) > 5.0, "a row-major weight should answer wrongly"
    with validating(op), pytest.raises(AssertionError):
        op(*args)


def test_cuda_graph_capture_and_replay() -> None:
    args = list(_build(dict(rows=1), seed=8))
    static_in = args[0]
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            op(*args)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = op(*args)
    fresh = _build(dict(rows=1), seed=18)[0]
    static_in.copy_(fresh)
    graph.replay()
    torch.cuda.synchronize()
    for a, b in zip(graph_out, op(fresh, *args[1:])):
        assert torch.equal(a, b)


@pytest.mark.parametrize("rows,n_cols", [(1, RANK + HC_COUNT), (1, RANK), (16, RANK + HC_COUNT)])
def test_moe_combine_matches_the_composition_it_replaces(rows, n_cols) -> None:
    """Against `fused_sigmoid_gate_mul_add` (in place on the routed output, as
    the forward called it) and then this op without the combine, at the tile
    the combine path launches with: bitwise. (The default tile sums `part` in
    another fp32 order; the cells bound both against the reference.)"""
    args = _build(dict(rows=rows, moe=True, n_cols=n_cols), seed=200 + rows)
    hyper_input, block_out, inject, gamma, first, c, n, shared, gate = args
    fused = op(*args)
    combined = fused_sigmoid_gate_mul_add(block_out.clone(), gate, shared)
    hc, part, ss = launch(hyper_input, combined, inject, gamma, first, c, n, config=COMBINE_CONFIG)
    for name, a, b in zip(("hc_out", "part", "ss"), fused, (hc, part[:, :, :n], ss)):
        assert torch.equal(a, b), name
    default = op(hyper_input, combined, inject, gamma, first, c, n)
    assert torch.equal(fused[0], default[0]), "hc_out does not depend on the tile"
    print(
        f"\n  rows {rows} cols {n_cols}: vs the default tile, part bitwise "
        f"{(fused[1] == default[1]).float().mean().item():.4f} "
        f"({_band(fused[1], default[1], PART_EPS):.3g}x the band apart), ss bitwise "
        f"{(fused[2] == default[2]).float().mean().item():.4f}"
    )


def test_moe_wrong_variants_land_outside_the_band() -> None:
    """Each mis-reading of the combine lands > 5x outside `_hc_band_ratio`."""
    args = _build(dict(rows=8, moe=True), seed=11)
    hyper_input, block_out, inject, gamma, first, c, n, shared, gate = args
    want_hc = op.reference(*args)[0]
    r, sh, s = block_out.double(), shared.double(), torch.sigmoid(gate.double())
    variants = {
        "no sigmoid": r + gate.double() * sh,
        "gate on the routed half": s * r + sh,
        "gate dropped": r + sh,
        "shared dropped": r,
    }
    x = hyper_input.double().unflatten(-1, (c, HIDDEN))
    for name, blk in variants.items():
        wrong = (x + blk.unsqueeze(-2) * inject.double().unsqueeze(-1)).flatten(-2)
        ratio = _hc_band_ratio(wrong.to(torch.bfloat16), want_hc, block_out, inject, shared, gate)
        print(f"\n  hc_out {name:<26} {ratio:10.4g}x the band")
        assert ratio > 5.0, name


def test_guard_refuses_a_partial_combine() -> None:
    args = list(_build(dict(rows=2, moe=True), seed=12))
    hyper_input, block_out, inject, gamma, first, c, n, shared, gate = args
    with validating(op), pytest.raises(AssertionError):
        op(hyper_input, block_out, inject, gamma, first, c, n, shared, None)
    with validating(op), pytest.raises(AssertionError):
        op(hyper_input, None, None, gamma, first, c, n, shared, gate)


def test_moe_cuda_graph_capture_and_replay() -> None:
    args = list(_build(dict(rows=1, moe=True), seed=13))
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            op(*args)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = op(*args)
    fresh = _build(dict(rows=1, moe=True), seed=23)
    for i in (0, 1, 7, 8):
        args[i].copy_(fresh[i])
    graph.replay()
    torch.cuda.synchronize()
    for a, b in zip(graph_out, op(*args)):
        assert torch.equal(a, b)


@pytest.mark.parametrize(
    "rows,n_cols,drop",
    [
        (1, RANK + HC_COUNT, False),
        (1, RANK, False),
        (4, RANK + HC_COUNT, True),
        (16, RANK + HC_COUNT, False),
    ],
)
def test_fold_matches_finalize_then_combine(rows, n_cols, drop) -> None:
    """The fold against the runner's finalize (emulated exactly: fp32 FMAs in
    slot order, one rounding; the MoE entry's test pins that emulation to the
    runner bitwise) followed by this op's combine path at the same tile:
    `hc_out` and `ss` bitwise; `part` bitwise from 2 rows, and at 1 row (where
    the two Triton specializations reduce the dot in a different fp32 order)
    within 2^-22 of it relative to its scale. And `hc_out` bitwise the combine
    path at its own tile."""
    args = _build(dict(rows=rows, moe=True, fold=10, drop=drop, n_cols=n_cols), seed=300 + rows)
    hyper_input, perm, inject, gamma, first, c, n, shared, gate, topk_w, topk_rows = args
    fused = op(*args)
    routed = finalize(perm, topk_w, topk_rows)
    hc, part, ss = launch(
        hyper_input, routed, inject, gamma, first, c, n, shared, gate, config=FOLD_CONFIG
    )
    part = part[:, :, :n]
    assert torch.equal(fused[0], hc), "hc_out"
    assert torch.equal(fused[2], ss), "ss"
    if rows > 1:
        assert torch.equal(fused[1], part), "part"
    else:
        assert _band(fused[1], part, 2.0**-22) <= 1.0, "part"
    print(
        f"\n  rows {rows} cols {n_cols}: part vs finalize + combine bitwise "
        f"{(fused[1] == part).float().mean().item():.4f} ({_band(fused[1], part, PART_EPS):.3g}x the band apart)"
    )
    combined = op(hyper_input, routed, inject, gamma, first, c, n, shared, gate)
    assert torch.equal(fused[0], combined[0]), "hc_out does not depend on the tile"


def test_fold_wrong_variants_land_outside_the_band() -> None:
    """Each mis-reading of the finalize lands > 5x outside `_hc_band_ratio`."""
    args = _build(dict(rows=8, moe=True, fold=10), seed=17)
    hyper_input, perm, inject, gamma, first, c, n, shared, gate, topk_w, topk_rows = args
    want_hc = op.reference(*args)[0]
    rows = perm.double()[topk_rows.long()]
    w = topk_w.double().unsqueeze(-1)
    s = torch.sigmoid(gate.double()).reshape(-1, 1)
    variants = {
        "unweighted sum": rows.sum(1),
        "first slot only": rows[:, 0] * w[:, 0],
        "weights by the wrong slot": (rows * w.flip(1)).sum(1),
        "token-major rows (no index map)": (perm.double()[: 8 * 10].reshape(8, 10, -1) * w).sum(1),
    }
    x = hyper_input.double().unflatten(-1, (c, HIDDEN))
    for name, routed in variants.items():
        blk = routed.to(torch.bfloat16).double() + s * shared.double()
        wrong = (x + blk.unsqueeze(-2) * inject.double().unsqueeze(-1)).flatten(-2)
        ratio = _hc_band_ratio(
            wrong.to(torch.bfloat16), want_hc, perm, inject, shared, gate, topk_w, topk_rows
        )
        print(f"\n  hc_out {name:<34} {ratio:10.4g}x the band")
        assert ratio > 5.0, name


def test_guard_refuses_a_partial_fold() -> None:
    args = list(_build(dict(rows=2, moe=True, fold=10), seed=19))
    hyper_input, perm, inject, gamma, first, c, n, shared, gate, topk_w, topk_rows = args
    with validating(op), pytest.raises(AssertionError, match="needs the combine"):
        op(hyper_input, perm, inject, gamma, first, c, n, None, None, topk_w, topk_rows)
    with validating(op), pytest.raises(AssertionError, match="come together"):
        op(hyper_input, perm, inject, gamma, first, c, n, shared, gate, None, topk_rows)
    with validating(op), pytest.raises(AssertionError, match="int32"):
        op(hyper_input, perm, inject, gamma, first, c, n, shared, gate, topk_w, topk_rows.long())


def test_fold_cuda_graph_capture_and_replay() -> None:
    args = list(_build(dict(rows=1, moe=True, fold=10), seed=29))
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            op(*args)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = op(*args)
    fresh = _build(dict(rows=1, moe=True, fold=10), seed=31)
    for i in (0, 1, 2, 7, 8, 9, 10):
        args[i].copy_(fresh[i])
    graph.replay()
    torch.cuda.synchronize()
    for a, b in zip(graph_out, op(*args)):
        assert torch.equal(a, b)
