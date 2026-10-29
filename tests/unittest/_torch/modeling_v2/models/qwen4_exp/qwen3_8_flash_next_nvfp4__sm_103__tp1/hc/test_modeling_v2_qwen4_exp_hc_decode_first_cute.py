# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp_hc_decode_first_cute entry.

Cells against the entry's float64 reference: `hc_out` within one ulp (plus the
combine's and the finalize's own rounding at the MoE sites), `part` within `2^-16` of
the reference relative to its scale, `ss` to fp32 accuracy. Beyond the cells: a
stale-read check under CUDA-graph replay with PDL, where a kernel that read its
activations before `griddepcontrol.wait` would see the previous replay's; the one-row
guard.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_hc_decode_first_cute import (  # noqa: E501
    qwen4_exp_hc_decode_first_cute as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "qwen4_exp_hc_decode_first_cute requires a CUDA device"

DEV = "cuda"
HC_COUNT, HIDDEN, RANK = 4, 2560, 320
DOWN_RMS, INJECT_RMS, GAMMA_RMS, GAMMA_ABSMAX = 0.0296, 0.0107, 0.477, 6.625
PART_EPS = 2.0**-16


def _band(got, want, eps) -> float:
    w = want.double()
    return ((got.double() - w).abs() / (eps * w.abs().max().item() + eps * w.abs())).max().item()


def _ulp(got, want) -> float:
    eps, tiny = torch.finfo(want.dtype).eps, torch.finfo(want.dtype).tiny
    w = want.double()
    return ((got.double() - w).abs() / (eps * w.abs() + tiny)).max().item()


def _compare(out, ref) -> None:
    (hc, part, ss), (ref_hc, ref_part, ref_ss) = out, ref
    assert hc.dtype == ref_hc.dtype and hc.shape == ref_hc.shape
    assert part.dtype == torch.float32 and part.shape == ref_part.shape and part.stride(2) == 1
    assert ss.dtype == torch.float32 and ss.shape == ref_ss.shape
    assert _ulp(hc, ref_hc) <= 1.0, f"hc_out {_ulp(hc, ref_hc):.4g} ulp"
    assert _band(part, ref_part, PART_EPS) <= 1.0, f"part {_band(part, ref_part, PART_EPS):.4g}x"
    assert torch.allclose(ss, ref_ss, rtol=1e-5, atol=0), "ss"


def _routed(block_out, topk_weights, topk_rows):
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
    """The `hc_out` band at the MoE sites: one ulp of `hc_out`, plus one
    ulp of the combine (and with the fold, of the routed row) times `|inject|`."""
    feps, tiny = torch.finfo(ref_hc.dtype).eps, torch.finfo(ref_hc.dtype).tiny
    routed = _routed(block_out, topk_weights, topk_rows)
    s = torch.sigmoid(gate.double().reshape(-1, 1))
    b = (routed.double() + s * shared.double()).to(routed.dtype).double().abs()
    if topk_rows is not None:
        b = b + routed.double().abs()
    band = feps * ref_hc.double().abs() + (
        feps * b.unsqueeze(-2) * inject.double().abs().unsqueeze(-1)
    ).flatten(-2)
    return ((hc.double() - ref_hc.double()).abs() / (band + tiny)).max().item()


def _compare_moe(args, out, ref) -> None:
    hyper_input, block_out, inject, gamma, first, c, n, shared, gate = args[:9]
    topk_weights, topk_rows = args[9:] if len(args) > 9 else (None, None)
    (hc, part, ss), (ref_hc, _, _) = out, ref
    ratio = _hc_band_ratio(hc, ref_hc, block_out, inject, shared, gate, topk_weights, topk_rows)
    assert ratio <= 1.0, f"hc_out {ratio:.4g}x the band"
    _compare((hc, part, ss), op.reference(hc, None, None, gamma, first, c, n))


def _build(spec, seed):
    """The test operands at one row."""
    rows, hc_count, hidden, dtype = 1, HC_COUNT, HIDDEN, torch.bfloat16
    width = hc_count * hidden
    n_cols = spec.get("n_cols", RANK + hc_count)
    g = torch.Generator(device=DEV).manual_seed(seed)

    def normal(*shape, scale):
        return torch.randn(*shape, generator=g, device=DEV) * scale

    hyper_input = normal(rows, width, scale=3.0).to(dtype)
    gamma = normal(width, scale=GAMMA_RMS)
    gamma[0], gamma[width // 2] = GAMMA_ABSMAX, -GAMMA_ABSMAX
    gamma = gamma.to(dtype)
    n_rows = -(-n_cols // 128) * 128
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
    top_k = spec.get("fold", 0)
    if not top_k:
        return hyper_input, block_out, inject, gamma, first, hc_count, n_cols, shared, gate_logits
    n_perm = rows * top_k + 37
    block_out = normal(n_perm, hidden, scale=2.0).to(dtype)
    order = torch.randperm(n_perm, generator=g, device=DEV)[: rows * top_k]
    topk_rows = order.reshape(rows, top_k).to(torch.int32)
    if spec.get("drop"):
        topk_rows[:, -3:] = -1
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


def test_reads_after_the_wait_under_graph_replay() -> None:
    """A producer kernel rewrites every activation inside the graph right before this op.
    With PDL the op may start before the producer finishes; reading any activation before
    `griddepcontrol.wait` would return the previous replay's value."""
    args = list(_build(dict(moe=True, fold=10), seed=8))
    sources = [a.clone() for a in (args[0], args[1], args[2], args[7], args[8], args[9])]
    targets = (args[0], args[1], args[2], args[7], args[8], args[9])

    def step():
        for dst, src in zip(targets, sources):
            dst.copy_(src * 1.0)  # an elementwise producer kernel per operand
        return op(*args)

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = step()
    for trial in range(3):
        fresh = _build(dict(moe=True, fold=10), seed=100 + trial)
        for src, new in zip(sources, (fresh[0], fresh[1], fresh[2], fresh[7], fresh[8], fresh[9])):
            src.copy_(new)
        graph.replay()
        torch.cuda.synchronize()
        want = op(*[*args[:10], args[10]])
        for a, b in zip(graph_out, want):
            assert torch.equal(a, b), f"replay {trial}"


def test_guards() -> None:
    args = list(_build(dict(), seed=6))
    two = args[0].expand(2, -1).contiguous()
    with validating(op), pytest.raises(AssertionError, match="one row"):
        op(two, args[1].expand(2, -1).contiguous(), args[2].expand(2, -1).contiguous(), *args[3:])
