# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp_hc_inject_norm entry.

Cells are gated at one ulp of the activation dtype against the float64
reference; the MoE-combine cells widen the residual's band by the combine's own
rounding (see `_compare_moe`). Beyond them: the CuTe fold kernel bitwise the
Triton kernel at the last row count before the switch, and the slot-map guard.
"""

import sys
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_hc_inject_norm import (  # noqa: E501
    qwen4_exp_hc_inject_norm as op,
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
    rows, hc_count, hidden, dtype = spec["rows"], HC_COUNT, HIDDEN, torch.bfloat16
    width = hc_count * hidden
    g = torch.Generator(device=DEV).manual_seed(seed)
    hyper_input = (torch.randn(rows, width, generator=g, device=DEV) * 3.0).to(dtype)
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


def test_guard_refuses_an_int64_slot_map() -> None:
    """The fold's slot map is the runner's int32 index map; `is_valid` refuses an int64 one."""
    fold = _build(dict(rows=4, moe=True, fold=10), seed=9)
    with validating(op), pytest.raises(AssertionError, match="int32"):
        op(*fold[:9], fold[9].long())


def test_fold_cute_is_the_triton_kernel_bitwise(monkeypatch) -> None:
    """Below `_FEW_WARPS_MIN_ROWS` rows the fold runs the CuTe DSL kernel; its
    `hc_out` and `normed` are bitwise the Triton kernel's (run by refusing the
    CuTe guard) at the last row count before the switch."""
    rows = M._FEW_WARPS_MIN_ROWS - 1
    args = _build(dict(rows=rows, moe=True, fold=10), seed=500 + rows)
    calls = []
    cute_fold = M._fold_cute

    def recording(*a):
        calls.append(a[0].shape[0])
        cute_fold(*a)

    monkeypatch.setattr(M, "_fold_cute", recording)
    hc, normed = op(*args)
    assert calls == [rows], "the decode MoE site runs the CuTe kernel"
    monkeypatch.setattr(M, "_fold_cute_ok", lambda *a: False)
    want_hc, want_normed = op(*args)
    assert torch.equal(hc, want_hc)
    assert torch.equal(normed, want_normed)
