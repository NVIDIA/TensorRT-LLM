# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's fp8_mqa_logits entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention.fp8_mqa_logits import (  # noqa: E501
    fp8_mqa_logits as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "fp8_mqa_logits requires a CUDA device"

E4M3_MAX = 448.0

# Gate, in units of the row's largest |logit|. See `_compare`.
ELEMENT_TOL = 2.0**-12
RMS_TOL = 2.0**-14


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """2**-12 of the row's largest |logit| per element, 2**-14 relative RMS.

    fp8 x fp8 products are exact in fp32, so kernel and reference differ
    only in the order of two fp32 sums: 128 terms per head dot product,
    then 32 heads. That costs a few fp32 ulps of the row scale, far inside
    the band. A wrong formula -- a missing key scale, the relu after the
    head sum, a head weight from another row -- moves logits by O(1) of
    the row scale.

    The -inf pattern outside the windows is exact: the top-k that follows
    treats those columns as absent. NaN in `ref` means unspecified, -inf
    must be -inf, and a finite entry must sit within the band.
    """
    assert out.dtype == ref.dtype == torch.float32, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    must_inf = torch.isneginf(ref)
    assert bool(torch.isneginf(out[must_inf]).all()), "an out-of-window entry is not -inf"
    live = torch.isfinite(ref)
    assert bool(torch.isfinite(out[live]).all()), "an in-window entry is not finite"
    o = torch.where(live, out, torch.zeros_like(out)).double()
    r = torch.where(live, ref, torch.zeros_like(ref)).double()
    row_scale = r.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    err = (o - r).abs() / row_scale
    assert float(err.max()) <= ELEMENT_TOL, (
        f"max error {float(err.max()):.3e} of the row scale exceeds {ELEMENT_TOL:.3e}"
    )
    rms = float(((o - r) ** 2).sum().sqrt() / (r**2).sum().sqrt().clamp_min(1e-30))
    assert rms <= RMS_TOL, f"relative RMS {rms:.3e} exceeds {RMS_TOL:.3e}"


def _quantize_rows(x: torch.Tensor):
    """Per-row e4m3 with an fp32 scale, as the indexer quantizes its keys."""
    scale = (x.float().abs().amax(dim=-1) / E4M3_MAX).clamp_min(1e-12)
    return (x.float() / scale.unsqueeze(-1)).to(torch.float8_e4m3fn), scale.contiguous()


def _build(spec, seed):
    """`(q, (k, k_scale), weights, ks, ke)` for one cell.

    The weights carry the indexer's own scale, `D**-0.5 * H**-0.5`, so the
    logits sit at the magnitudes the target produces.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    heads, dim = spec["heads"], spec["head_dim"]
    starts, ends, col = [], [], 0
    for cached, new in spec["seqs"]:
        for t in range(new):
            starts.append(col)
            ends.append(col + cached + t + 1)
        col += cached + new
    rows = len(starts)
    q = torch.randn(rows, heads, dim, generator=g, device="cuda").to(torch.float8_e4m3fn)
    k, k_scale = _quantize_rows(torch.randn(col, dim, generator=g, device="cuda"))
    w = torch.randn(rows, heads, generator=g, device="cuda") * (dim**-0.5 * heads**-0.5)
    ks = torch.tensor(starts, dtype=torch.int32, device="cuda")
    ke = torch.tensor(ends, dtype=torch.int32, device="cuda")
    return q, (k, k_scale), w.contiguous(), ks, ke


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    with validating(op):
        out = op(*args)
    q, (k, _), *_ = args
    assert tuple(out.shape) == (q.shape[0], k.shape[0])
    # A row-padded view: the prefill top-k reads it at stride(0) as it stands.
    assert out.stride(1) == 1 and out.stride(0) >= out.shape[1]
    _compare(out, op.reference(*args))


def test_the_gate_rejects_wrong_formulas() -> None:
    """The band is only worth something if a plausible mistake misses it."""
    args = _build(dict(heads=32, head_dim=128, seqs=[(256, 64)]), seed=4)
    out = op(*args)
    q, (k, k_scale), w, ks, ke = args
    # relu after the head sum instead of per head
    dots = torch.einsum("qhd,kd->qhk", q.float().double(), k.float().double())
    after = (dots * w.double().unsqueeze(-1)).sum(1).relu() * k_scale.double().unsqueeze(0)
    cols = torch.arange(k.shape[0], device="cuda").unsqueeze(0)
    after = after.masked_fill((cols < ks.unsqueeze(1)) | (cols >= ke.unsqueeze(1)), float("-inf"))
    wrong = {
        "no key scale": op.reference(q, (k, torch.ones_like(k_scale)), w, ks, ke),
        "relu after the head sum": after.float(),
        "weights of the neighbouring row": op.reference(q, (k, k_scale), w.roll(1, 0), ks, ke),
    }
    for tag, ref in wrong.items():
        try:
            _compare(out, ref)
        except AssertionError:
            continue
        raise AssertionError(f"the gate accepted a wrong formula: {tag}")


def _ok():
    return _build(dict(heads=32, head_dim=128, seqs=[(64, 8)]), seed=6)


def test_guard_refuses_row_padded_weights() -> None:
    """The op checks stride(1) only and reads a packed [T, H] block."""
    q, kv, w, ks, ke = _ok()
    padded = torch.zeros(w.shape[0], 64, device="cuda")[:, : w.shape[1]]
    padded.copy_(w)
    with pytest.raises(AssertionError, match="weights rows must be packed"):
        with validating(op):
            op(q, kv, padded, ks, ke)


def test_guard_refuses_half_weights() -> None:
    q, kv, w, ks, ke = _ok()
    with pytest.raises(AssertionError, match="weights must be fp32"):
        with validating(op):
            op(q, kv, w.half(), ks, ke)


def test_guard_refuses_a_window_past_the_keys() -> None:
    q, kv, w, ks, ke = _ok()
    with pytest.raises(AssertionError, match="ends past the"):
        with validating(op):
            op(q, kv, w, ks, ke + 1)
