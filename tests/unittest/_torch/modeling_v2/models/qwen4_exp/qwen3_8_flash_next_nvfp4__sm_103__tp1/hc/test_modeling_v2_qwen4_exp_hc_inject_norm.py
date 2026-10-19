# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp_hc_inject_norm entry.

Cells are gated at one ulp of the activation dtype against the float64
reference; the MoE-combine cells widen the residual's band by the combine's own
rounding (see `_compare_moe`). Beyond them: agreement with the compositions it
replaces, the separation from plausible wrong formulas, the norm-only aliasing
claim, and graph capture.
"""

import sys
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_hc_inject_norm import (  # noqa: E501
    qwen4_exp_hc_inject_norm as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_post_inject import (  # noqa: E501
    qwen4_exp_post_inject,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.moe.fp4_block_scale_moe_renorm_permuted import (  # noqa: E501
    finalize,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.moe.fused_sigmoid_gate_mul_add import (  # noqa: E501
    fused_sigmoid_gate_mul_add,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.norm.flashinfer_gemma_rmsnorm import (  # noqa: E501
    flashinfer_gemma_rmsnorm,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "qwen4_exp_hc_inject_norm requires a CUDA device"

#: The op's module (its kernels and dispatch guards).
M = sys.modules[type(op).__module__]

DEV = "cuda"
HC_COUNT = 4
HIDDEN = 2560
EPS = 1e-6
# This checkpoint's grouped-norm weight statistics (layer-0 hyper-connection norm).
GAMMA_RMS = 0.477
GAMMA_ABSMAX = 6.625


def _ulp_ratio(out: torch.Tensor, ref: torch.Tensor) -> float:
    """Worst element's distance from `ref` in units of `eps(dtype) * |ref|`
    (one ulp of the reference value, or a little more). 1 means one ulp."""
    eps = torch.finfo(ref.dtype).eps
    ref64 = ref.double()
    tiny = torch.finfo(ref.dtype).tiny
    return ((out.double() - ref64).abs() / (eps * ref64.abs() + tiny)).max().item()


def _compare(out, ref) -> None:
    """Both outputs within one ulp of the float64 result rounded once.

    The op rounds each output once from an fp32 value whose own error is
    ~1e-6 relative (a 16-bit product is exact in fp32; the RMS is a 2560-term
    fp32 sum), so it can differ from the once-rounded float64 value only where
    the two straddle a rounding midpoint: by exactly one ulp."""
    (hc, normed), (ref_hc, ref_normed) = out, ref
    for got, want in ((hc, ref_hc), (normed, ref_normed)):
        assert got.dtype == want.dtype and got.shape == want.shape
        if got.numel():
            ratio = _ulp_ratio(got, want)
            assert ratio <= 1.0, f"{ratio:.4g} ulp"


def _combine(block_out, shared, gate_logits) -> torch.Tensor:
    """The MoE block output in float64, rounded once to `block_out`'s dtype."""
    s = torch.sigmoid(gate_logits.double().reshape(-1, 1))
    return (block_out.double() + s * shared.double()).to(block_out.dtype)


def _routed(block_out, topk_weights, topk_rows):
    """The routed rows: `block_out` itself, or with the fold the float64 finalize of
    the permuted rows rounded once."""
    if topk_rows is None:
        return block_out
    valid = (topk_rows >= 0).unsqueeze(-1)
    picked = block_out.double()[topk_rows.clamp(min=0).long()]
    w = topk_weights.double().unsqueeze(-1)
    return torch.where(valid, picked * w, 0.0).sum(1).to(block_out.dtype)


def _compare_moe(args, out, ref) -> None:
    """MoE-combine cells.

    `hc_out`: the op's combine is an fp32 value rounded once, the reference's a
    float64 value rounded once; where the two straddle a rounding midpoint they
    differ by one ulp of the block output, and the accumulate carries that
    scaled by `inject`. So the band is one ulp of `hc_out` plus one ulp of the
    combine times `|inject|`, element by element -- still well under one ulp
    of `hc_out` wherever the block term is small against the residual. With the
    finalize fold the routed row has its own rounding (fp32 FMAs against the
    float64 sum), one more ulp of the routed row times `|inject|`.

    `normed`: one ulp of the float64 norm of the residual the op stored,
    `reference(hc_out, None, None, ...)` -- the dataflow the op states."""
    hyper_input, block_out, inject, gamma, eps, c, shared, gate = args[:8]
    topk_weights, topk_rows = args[8:] if len(args) > 8 else (None, None)
    (hc, normed), (ref_hc, _) = out, ref
    assert hc.dtype == ref_hc.dtype and hc.shape == ref_hc.shape
    if not hc.numel():
        return
    feps, tiny = torch.finfo(ref_hc.dtype).eps, torch.finfo(ref_hc.dtype).tiny
    routed = _routed(block_out, topk_weights, topk_rows)
    b = _combine(routed, shared, gate).double().abs()
    if topk_rows is not None:
        b = b + routed.double().abs()
    band = feps * ref_hc.double().abs() + (
        feps * b.unsqueeze(-2) * inject.double().abs().unsqueeze(-1)
    ).flatten(-2)
    ratio = ((hc.double() - ref_hc.double()).abs() / (band + tiny)).max().item()
    assert ratio <= 1.0, f"hc_out {ratio:.4g}x the band"
    own = op.reference(hc, None, None, gamma, eps, c)[1]
    assert _ulp_ratio(normed, own) <= 1.0, f"normed {_ulp_ratio(normed, own):.4g} ulp"


def _build(spec, seed):
    rows = spec["rows"]
    hc_count = spec.get("hc_count", HC_COUNT)
    hidden = spec.get("hidden", HIDDEN)
    dtype = spec.get("dtype", torch.bfloat16)
    width = hc_count * hidden
    g = torch.Generator(device=DEV).manual_seed(seed)
    hyper_input = (torch.randn(rows, width, generator=g, device=DEV) * 3.0).to(dtype)
    if spec.get("layout") == "row_slice":
        buf = torch.randn(2 * rows, width, generator=g, device=DEV).to(dtype)
        buf[::2] = hyper_input
        hyper_input = buf[::2]
    gamma = torch.randn(width, generator=g, device=DEV) * GAMMA_RMS
    gamma[0], gamma[width // 2] = GAMMA_ABSMAX, -GAMMA_ABSMAX
    gamma = gamma.to(dtype)
    block_out = inject = shared = gate_logits = None
    if spec.get("inject", True):
        block_out = (torch.randn(rows, hidden, generator=g, device=DEV) * 2.0).to(dtype)
        inject = (2.0 * torch.sigmoid(torch.randn(rows, hc_count, generator=g, device=DEV))).to(
            dtype
        )
    if spec.get("moe", False):
        shared = (torch.randn(rows, hidden, generator=g, device=DEV) * 2.0).to(dtype)
        gate_logits = (torch.randn(rows, 1, generator=g, device=DEV) * 2.0).to(dtype)
        if spec.get("gate") == "flat":
            gate_logits = gate_logits.reshape(rows)
    top_k = spec.get("fold", 0)
    if not top_k:
        return hyper_input, block_out, inject, gamma, EPS, hc_count, shared, gate_logits
    # The runner's unfinalized output: permuted FC2 rows (more rows than slots, as
    # its padded buffer has), each slot's row and its routing weight.
    n_perm = rows * top_k + 37
    block_out = (torch.randn(n_perm, hidden, generator=g, device=DEV) * 2.0).to(dtype)
    order = torch.randperm(n_perm, generator=g, device=DEV)[: rows * top_k]
    topk_rows = order.reshape(rows, top_k).to(torch.int32)
    if spec.get("drop"):
        topk_rows[:, -3:] = -1
    topk_weights = torch.softmax(torch.randn(rows, top_k, generator=g, device=DEV), -1).to(dtype)
    return (
        hyper_input,
        block_out,
        inject,
        gamma,
        EPS,
        hc_count,
        shared,
        gate_logits,
        topk_weights,
        topk_rows,
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    before = [a.clone() if isinstance(a, torch.Tensor) else a for a in args]
    hc, normed = op(*args)
    for a, b in zip(args, before):
        assert not isinstance(a, torch.Tensor) or torch.equal(a, b), "every input is read-only"
    if args[1] is None:
        assert hc is args[0], "norm-only returns hyper_input itself"
    else:
        assert hc.is_contiguous()
        assert args[0].numel() == 0 or hc.data_ptr() != args[0].data_ptr(), "hc_out is fresh"
    assert normed.is_contiguous()
    if args[6] is None:
        _compare((hc, normed), op.reference(*args))
    else:
        _compare_moe(args, (hc, normed), op.reference(*args))


def _old_composition(hyper_input, block_out, inject, gamma, eps, hc_count):
    hc = qwen4_exp_post_inject(hyper_input, block_out, inject)
    hidden = hc.shape[1] // hc_count
    parts = torch.split(hc, [hidden] * hc_count, 1)
    weights = torch.split(gamma, hidden, 0)
    normed = torch.cat([flashinfer_gemma_rmsnorm(p, w, eps) for w, p in zip(weights, parts)], 1)
    return hc, normed


@pytest.mark.parametrize("rows", [1, 128, 8192])
def test_matches_the_composition_it_replaces(rows) -> None:
    """Against post_inject + four gemma norms + cat: the residual within one
    ulp (fma contraction only), the normed stream within one ulp."""
    args = _build(dict(rows=rows), seed=rows)[:6]
    hc, normed = op(*args)
    old_hc, old_normed = _old_composition(*args)
    assert _ulp_ratio(hc, old_hc) <= 1.0
    assert _ulp_ratio(normed, old_normed) <= 1.0
    print(f"\n  rows {rows}: hc bitwise-equal fraction {(hc == old_hc).float().mean().item():.6f}")


def test_wrong_variants_land_outside_the_band() -> None:
    """Each mis-reading keeps the shapes; each lands > 5x outside one ulp. The
    streams carry different scales, as the trained residual's do, so a norm
    over the wrong group is visible."""
    hyper_input, block_out, inject, gamma, eps, c, _, _ = _build(dict(rows=64), seed=5)
    scales = torch.tensor([0.5, 1.0, 2.0, 4.0], device=DEV).repeat_interleave(HIDDEN)
    hyper_input = (hyper_input.float() * scales).to(torch.bfloat16)
    want_hc, want_normed = op.reference(hyper_input, block_out, inject, gamma, eps, c)
    x = hyper_input.double().unflatten(-1, (c, HIDDEN))
    b, g = block_out.double(), inject.double()
    wrong_hc = {
        "stream-minor inject": (x.flatten(-2) + (b.unsqueeze(-1) * g.unsqueeze(-2)).flatten(-2)),
        "no gate": (x + b.unsqueeze(-2)).flatten(-2),
    }
    for name, value in wrong_hc.items():
        ratio = _ulp_ratio(value.to(torch.bfloat16), want_hc)
        print(f"\n  hc {name:<24} {ratio:10.4g} ulp")
        assert ratio > 5.0, name
    hc = want_hc.double().unflatten(-1, (c, HIDDEN))
    full = hc.flatten(-2)
    wrong_normed = {
        "RMS over the full width": full
        * torch.rsqrt(full.pow(2).mean(-1, keepdim=True) + eps)
        * (1.0 + gamma.double()),
        "w instead of 1 + w": (hc * torch.rsqrt(hc.pow(2).mean(-1, keepdim=True) + eps)).flatten(-2)
        * gamma.double(),
        "norm of the pre-inject stream": (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))
        .flatten(-2)
        .mul(1.0 + gamma.double()),
    }
    for name, value in wrong_normed.items():
        ratio = _ulp_ratio(value.to(torch.bfloat16), want_normed)
        print(f"  normed {name:<30} {ratio:10.4g} ulp")
        assert ratio > 5.0, name


def test_guard_refuses_block_out_without_inject() -> None:
    hyper_input, block_out, inject, gamma, eps, c, _, _ = _build(dict(rows=4), seed=6)
    with validating(op), pytest.raises(AssertionError):
        op(hyper_input, block_out, None, gamma, eps, c)
    with validating(op), pytest.raises(AssertionError):
        op(hyper_input, block_out, inject[0], gamma, eps, c)


def test_cuda_graph_capture_and_replay() -> None:
    args = list(_build(dict(rows=8), seed=8))
    static_in = args[0]
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            op(*args)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_hc, graph_normed = op(*args)
    fresh = _build(dict(rows=8), seed=18)[0]
    static_in.copy_(fresh)
    graph.replay()
    torch.cuda.synchronize()
    eager_hc, eager_normed = op(fresh, *args[1:])
    assert torch.equal(graph_hc, eager_hc)
    assert torch.equal(graph_normed, eager_normed)


@pytest.mark.parametrize("rows", [2, 128, 2048])
def test_fold_matches_the_runner_finalize(rows) -> None:
    """With the fold, bitwise this op on the runner's finalized rows: `finalize`
    (the permuted MoE entry's emulation of the runner's finalize, bitwise its
    output) and then the combine without the fold."""
    args = _build(dict(rows=rows, moe=True, fold=10), seed=300 + rows)
    hyper_input, rows_, inject, gamma, eps, c, shared, gate, topk_weights, topk_rows = args
    hc, normed = op(*args)
    routed = finalize(rows_, topk_weights, topk_rows)
    old_hc, old_normed = op(hyper_input, routed, inject, gamma, eps, c, shared, gate)
    assert torch.equal(hc, old_hc)
    assert torch.equal(normed, old_normed)


@pytest.mark.parametrize("rows", [1, 128, 8192])
def test_moe_combine_matches_the_composition_it_replaces(rows) -> None:
    """Against `fused_sigmoid_gate_mul_add` (in place on the routed output, as
    the forward called it) and then this op without the combine: bitwise."""
    args = _build(dict(rows=rows, moe=True), seed=100 + rows)
    hyper_input, block_out, inject, gamma, eps, c, shared, gate = args
    hc, normed = op(*args)
    combined = fused_sigmoid_gate_mul_add(block_out.clone(), gate, shared)
    old_hc, old_normed = op(hyper_input, combined, inject, gamma, eps, c)
    print(
        f"\n  rows {rows}: hc bitwise {(hc == old_hc).float().mean().item():.6f}, "
        f"normed bitwise {(normed == old_normed).float().mean().item():.6f}"
    )
    assert torch.equal(hc, old_hc)
    assert torch.equal(normed, old_normed)


def test_moe_wrong_variants_land_outside_the_band() -> None:
    """Each mis-reading of the combine keeps the shapes; each lands > 5x outside
    the `_compare_moe` residual band."""
    args = _build(dict(rows=64, moe=True), seed=7)
    hyper_input, block_out, inject, gamma, eps, c, shared, gate = args
    want_hc = op.reference(*args)[0]
    feps = torch.finfo(torch.bfloat16).eps
    b = _combine(block_out, shared, gate).double().abs()
    band = feps * want_hc.double().abs() + (
        feps * b.unsqueeze(-2) * inject.double().abs().unsqueeze(-1)
    ).flatten(-2)
    r, sh, s = block_out.double(), shared.double(), torch.sigmoid(gate.double())
    variants = {
        "no sigmoid": r + gate.double() * sh,
        "gate on the routed half": s * r + sh,
        "gate dropped": r + sh,
        "shared dropped": r,
        "1 - sigmoid": r + (1.0 - s) * sh,
    }
    x = hyper_input.double().unflatten(-1, (c, HIDDEN))
    for name, blk in variants.items():
        wrong = (x + blk.unsqueeze(-2) * inject.double().unsqueeze(-1)).flatten(-2)
        ratio = ((wrong.to(torch.bfloat16).double() - want_hc.double()).abs() / band).max().item()
        print(f"\n  hc {name:<26} {ratio:10.4g}x the band")
        assert ratio > 5.0, name


def test_guard_refuses_a_partial_combine() -> None:
    hyper_input, block_out, inject, gamma, eps, c, shared, gate = _build(
        dict(rows=4, moe=True), seed=9
    )
    fold = _build(dict(rows=4, moe=True, fold=10), seed=9)
    with validating(op), pytest.raises(AssertionError):
        op(*fold[:6], None, None, *fold[8:])
    with validating(op), pytest.raises(AssertionError):
        op(*fold[:9], None)
    with validating(op), pytest.raises(AssertionError):
        op(*fold[:9], fold[9].long())
    with validating(op), pytest.raises(AssertionError):
        op(hyper_input, block_out, inject, gamma, eps, c, shared, None)
    with validating(op), pytest.raises(AssertionError):
        op(hyper_input, None, None, gamma, eps, c, shared, gate)
    with validating(op), pytest.raises(AssertionError):
        op(hyper_input, block_out, inject, gamma, eps, c, shared, gate[:1].expand(4, 1))


def test_moe_cuda_graph_capture_and_replay() -> None:
    args = list(_build(dict(rows=8, moe=True), seed=10))
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            op(*args)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_hc, graph_normed = op(*args)
    fresh = _build(dict(rows=8, moe=True), seed=20)
    for i in (0, 1, 6, 7):
        args[i].copy_(fresh[i])
    graph.replay()
    torch.cuda.synchronize()
    eager_hc, eager_normed = op(*args)
    assert torch.equal(graph_hc, eager_hc)
    assert torch.equal(graph_normed, eager_normed)


@pytest.mark.parametrize(
    "spec",
    [
        dict(rows=1),
        dict(rows=2, drop=True),
        dict(rows=8),
        dict(rows=33, drop=True),
        dict(rows=128),
        dict(rows=17, gate="flat", layout="row_slice"),
        dict(rows=300, drop=True),
        dict(rows=511),
    ],
    ids=lambda s: "-".join(f"{k}{v}" for k, v in s.items()),
)
def test_fold_cute_is_the_triton_kernel_bitwise(spec, monkeypatch) -> None:
    """Below `_FEW_WARPS_MIN_ROWS` rows the fold runs the CuTe DSL kernel; its
    `hc_out` and `normed` are bitwise the Triton kernel's (run by refusing the
    CuTe guard)."""
    args = _build(dict(spec, moe=True, fold=10), seed=500 + spec["rows"])
    calls = []
    cute_fold = M._fold_cute

    def recording(*a):
        calls.append(a[0].shape[0])
        cute_fold(*a)

    monkeypatch.setattr(M, "_fold_cute", recording)
    hc, normed = op(*args)
    assert calls == [spec["rows"]], "the decode MoE site runs the CuTe kernel"
    monkeypatch.setattr(M, "_fold_cute_ok", lambda *a: False)
    want_hc, want_normed = op(*args)
    assert torch.equal(hc, want_hc)
    assert torch.equal(normed, want_normed)


def test_fold_from_the_few_warps_rows_keeps_triton(monkeypatch) -> None:
    """From `_FEW_WARPS_MIN_ROWS` rows (the Triton kernel's two-warp form, another
    reduction order) the fold stays on the Triton kernel."""
    args = _build(dict(rows=M._FEW_WARPS_MIN_ROWS, moe=True, fold=10), seed=77)
    monkeypatch.setattr(M, "_fold_cute", lambda *a: pytest.fail("CuTe kernel at 512 rows"))
    op(*args)
