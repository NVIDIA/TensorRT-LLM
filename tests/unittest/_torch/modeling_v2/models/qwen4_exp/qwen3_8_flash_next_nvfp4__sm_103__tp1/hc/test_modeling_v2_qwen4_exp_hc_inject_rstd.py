# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp_hc_inject_rstd entry.

Cells: `hc_out` within one ulp of the float64 reference (the MoE-combine cells
widened by the combine's own rounding, as `qwen4_exp_hc_inject_norm`'s test),
rstd within 2^-16 relative of the float64 statistic of the stored residual.
Beyond them: `hc_out` and the normed stream rebuilt from rstd bitwise
`qwen4_exp_hc_inject_norm`'s at the cells' sizes, the operand guard, graph
capture.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.hc.qwen4_exp_hc_inject_norm import (  # noqa: E501
    qwen4_exp_hc_inject_norm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_hc_inject_rstd import (  # noqa: E501
    qwen4_exp_hc_inject_rstd as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "qwen4_exp_hc_inject_rstd requires a CUDA device"

DEV = "cuda"
HC_COUNT = 4
HIDDEN = 2560
EPS = 1e-6
# This checkpoint's grouped-norm weight statistics (layer-0 hyper-connection norm).
GAMMA_RMS = 0.477
GAMMA_ABSMAX = 6.625
#: rstd band, relative: an fp32 2560-term sum of squares and an fp32 rsqrt.
RSTD_RTOL = 2.0**-16


def _ulp_ratio(out: torch.Tensor, ref: torch.Tensor) -> float:
    """Worst element's distance from `ref` in units of `eps(dtype) * |ref|`."""
    eps = torch.finfo(ref.dtype).eps
    tiny = torch.finfo(ref.dtype).tiny
    return ((out.double() - ref.double()).abs() / (eps * ref.double().abs() + tiny)).max().item()


def _combine(block_out, shared, gate_logits) -> torch.Tensor:
    """The MoE block output in float64, rounded once to `block_out`'s dtype."""
    s = torch.sigmoid(gate_logits.double().reshape(-1, 1))
    return (block_out.double() + s * shared.double()).to(block_out.dtype)


def _compare(args, out, ref) -> None:
    """`hc_out`: one ulp of the once-rounded float64 accumulate; with the MoE combine,
    plus one ulp of the combine times `|inject|` (the combine is an fp32 value
    rounded once, the reference's a float64 one). rstd: `RSTD_RTOL` of the float64
    statistic of the residual the op stored -- the dataflow the op states.
    """
    hyper_input, block_out, inject, eps, c, shared, gate = args
    (hc, rstd), (ref_hc, ref_rstd) = out, ref
    assert hc.dtype == ref_hc.dtype and hc.shape == ref_hc.shape
    assert rstd.dtype == torch.float32 and rstd.shape == ref_rstd.shape and rstd.is_contiguous()
    if not hc.numel():
        return
    if shared is None:
        assert _ulp_ratio(hc, ref_hc) <= 1.0, f"hc_out {_ulp_ratio(hc, ref_hc):.4g} ulp"
    else:
        feps, tiny = torch.finfo(ref_hc.dtype).eps, torch.finfo(ref_hc.dtype).tiny
        b = _combine(block_out, shared, gate).double().abs()
        band = feps * ref_hc.double().abs() + (
            feps * b.unsqueeze(-2) * inject.double().abs().unsqueeze(-1)
        ).flatten(-2)
        ratio = ((hc.double() - ref_hc.double()).abs() / (band + tiny)).max().item()
        assert ratio <= 1.0, f"hc_out {ratio:.4g}x the band"
    own = op.reference(hc, None, None, eps, c)[1].double()
    rel = ((rstd.double() - own).abs() / own).max().item()
    assert rel <= RSTD_RTOL, f"rstd {rel:.3g} relative"


def _gamma(seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    width = HC_COUNT * HIDDEN
    gamma = torch.randn(width, generator=g, device=DEV) * GAMMA_RMS
    gamma[0], gamma[width // 2] = GAMMA_ABSMAX, -GAMMA_ABSMAX
    return gamma.to(torch.bfloat16)


def _build(spec, seed):
    rows, dtype = spec["rows"], torch.bfloat16
    width = HC_COUNT * HIDDEN
    g = torch.Generator(device=DEV).manual_seed(seed)
    hyper_input = (torch.randn(rows, width, generator=g, device=DEV) * 3.0).to(dtype)
    block_out = inject = shared = gate_logits = None
    if spec.get("inject", True):
        block_out = (torch.randn(rows, HIDDEN, generator=g, device=DEV) * 2.0).to(dtype)
        inject = (2.0 * torch.sigmoid(torch.randn(rows, HC_COUNT, generator=g, device=DEV))).to(
            dtype
        )
    if spec.get("moe", False):
        shared = (torch.randn(rows, HIDDEN, generator=g, device=DEV) * 2.0).to(dtype)
        gate_logits = (torch.randn(rows, 1, generator=g, device=DEV) * 2.0).to(dtype)
    return hyper_input, block_out, inject, EPS, HC_COUNT, shared, gate_logits


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % (2**31)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args = _build(cell.spec, seed=_seed(cell))
    before = [a.clone() if isinstance(a, torch.Tensor) else a for a in args]
    hc, rstd = op(*args)
    for a, b in zip(args, before):
        assert not isinstance(a, torch.Tensor) or torch.equal(a, b), "every input is read-only"
    if args[1] is None:
        assert hc is args[0], "statistic-only returns hyper_input itself"
    else:
        assert hc.is_contiguous() and hc.data_ptr() != args[0].data_ptr(), "hc_out is fresh"
    _compare(args, (hc, rstd), op.reference(*args))


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_normed_rebuilt_from_rstd_is_inject_norm_bitwise(cell) -> None:
    """The consumers' operand `bf16((hc_out * rstd) * (1 + w))` (fp32 products) is
    bitwise `qwen4_exp_hc_inject_norm`'s `normed`, and `hc_out` its `hc_out`.
    """
    hyper_input, block_out, inject, eps, c, shared, gate = _build(cell.spec, seed=_seed(cell))
    gamma = _gamma(_seed(cell) + 1)
    hc, rstd = op(hyper_input, block_out, inject, eps, c, shared, gate)
    want_hc, want_normed = qwen4_exp_hc_inject_norm(
        hyper_input, block_out, inject, gamma, eps, c, shared, gate
    )
    assert torch.equal(hc, want_hc), f"{(hc != want_hc).sum().item()} hc_out elements differ"
    x = hc.float().unflatten(-1, (c, -1)) * rstd.unsqueeze(-1)
    normed = (x.flatten(-2) * (1.0 + gamma.float())).to(hc.dtype)
    diff = (normed != want_normed).sum().item()
    assert diff == 0, f"{diff} normed elements differ"


def test_guard_refuses_block_out_without_inject() -> None:
    hyper_input, block_out, _, eps, c, _, _ = _build(dict(rows=4), seed=4)
    with validating(op), pytest.raises(AssertionError, match="together"):
        op(hyper_input, block_out, None, eps, c)


def test_cuda_graph_capture_and_replay() -> None:
    args = list(_build(dict(rows=8, moe=True), seed=5))
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(*args)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_hc, g_rstd = op(*args)
    fresh = _build(dict(rows=8, moe=True), seed=6)
    for i in (0, 1, 2, 5, 6):
        args[i].copy_(fresh[i])
    graph.replay()
    torch.cuda.synchronize()
    e_hc, e_rstd = op(*fresh)
    assert torch.equal(g_hc, e_hc) and torch.equal(g_rstd, e_rstd)
