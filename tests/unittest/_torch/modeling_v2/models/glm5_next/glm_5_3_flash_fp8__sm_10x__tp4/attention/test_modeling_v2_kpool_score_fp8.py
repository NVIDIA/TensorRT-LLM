# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the glm5_next fp8 tp4 entry kpool_score_fp8."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.attention.kpool_score_fp8 import (  # noqa: E501
    kpool_score_fp8,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 100, reason="the glm5_next fp8 tp4 entries are certified on sm_100"
)

DEVICE = "cuda"
TPB, KPOOL, HD, HEADS, PAGES, SLOTS = 32, 4, 128, 32, 4100, 1024
Q_SCALE, W_SCALE = HD**-0.5, HEADS**-0.5


def _on_e4m3_grid(shape, g: torch.Generator) -> torch.Tensor:
    """BF16 rows of 128 that E4M3 holds exactly at the quantizer's per-row scale: each row is
    E4M3 values times ``2**e`` (``e`` drawn per row) with an absolute maximum of exactly ``448 *
    2**e``, so ``amax / 448`` is a power of two and every value is a code times it."""
    x = torch.randn(*shape, generator=g, device=DEVICE)
    x = (x / x.abs().amax(-1, keepdim=True) * 448).to(torch.float8_e4m3fn).float()
    top = x.abs().argmax(-1, keepdim=True)
    x.scatter_(-1, top, 448 * torch.gather(x, -1, top).sign())
    e = torch.randint(-12, -1, (*shape[:-1], 1), generator=g, device=DEVICE)
    return (x * torch.exp2(e.float())).to(torch.bfloat16)


def _args(spec: dict, seed: int, exact: bool = True) -> tuple[tuple, dict]:
    """Packed context rows of ``(cached, new)`` requests, restricted to ``rows``, with the
    host bound ``max_kv_len`` the target passes: the largest visible length among those rows.
    With ``exact`` the queries and every pool key are on the E4M3 grid (``_on_e4m3_grid``)."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    visible, request_ids = [], []
    for i, (cached, new) in enumerate(spec["context"]):
        visible.append(torch.arange(cached + 1, cached + new + 1))
        request_ids.append(torch.full((new,), i, dtype=torch.int32))
    lo, hi = spec["rows"]
    visible = torch.cat(visible)[lo:hi].to(DEVICE)
    request_ids = torch.cat(request_ids)[lo:hi].to(DEVICE)
    pages, slots = spec.get("pages", PAGES), spec.get("slots", SLOTS)
    max_kv_len = int(visible.max())
    tables = torch.zeros(len(spec["context"]), pages, dtype=torch.int64, device=DEVICE)
    for t, (cached, new) in enumerate(spec["context"]):
        used = -(-(cached + new) // TPB)
        tables[t, :used] = torch.randperm(slots, generator=g, device=DEVICE)[:used]
    storage = torch.randn(slots, 2, TPB, 3 * HD + 64, generator=g, device=DEVICE)
    pool = storage.to(torch.bfloat16)[:, 1, :, : 3 * HD]
    n = visible.shape[0]
    if exact:
        pool[..., 2 * HD :] = _on_e4m3_grid((slots, TPB, HD), g)
        q = _on_e4m3_grid((n, HEADS, HD), g)
    else:
        q = torch.randn(n, HEADS, HD, generator=g, device=DEVICE).to(torch.bfloat16)
    projection = torch.randn(n, 2 * HD + HEADS, generator=g, device=DEVICE).to(torch.bfloat16)
    weights = projection[:, 2 * HD :]
    kwargs = dict(
        head_dim=HD,
        kpool=KPOOL,
        q_scale=Q_SCALE,
        w_scale=W_SCALE,
        request_ids=request_ids,
        max_kv_len=max_kv_len,
    )
    return (q, weights, pool, tables, visible, TPB), kwargs


def _visible(args, pools):
    col = torch.arange(pools, device=DEVICE)
    return col[None, :] < (args[4].long() // KPOOL)[:, None]


def _check_layout(out, args, kwargs):
    """One table: DeepGEMM's row-padded logits as a strided view; several: the aligned copy."""
    n, tables = args[0].shape[0], args[3].shape[0]
    pools = kwargs["max_kv_len"] // KPOOL
    assert out.dtype == torch.float32 and out.stride(1) == 1 and out.shape[0] == n
    if tables == 1:
        assert out.shape[1] == pools and out.stride(0) >= pools and out.stride(0) % 4 == 0
    else:
        assert out.shape[1] == max(1024, -(-pools // 64) * 64) and out.is_contiguous()


@pytest.mark.parametrize(
    "cell", kpool_score_fp8.CELLS, ids=[c.why[:44] for c in kpool_score_fp8.CELLS]
)
def test_kpool_score_fp8_matches_reference_on_the_e4m3_grid(cell, certifying):  # noqa: F811
    """Operands on the E4M3 grid quantize without loss (the query and key scales come out as
    powers of two, up to one rounding of ``1/448``), so the FP8 scores equal the exact ones up
    to FP32 accumulation: within ``2**-13 * bound``, ``bound`` the reference on absolute
    values, as for ``kpool_score`` (exact products of 4-bit significands, 128 per head, 32
    heads). This pins the gather, the table ranges and every scale, not the rounding."""
    certifying(kpool_score_fp8)
    args, kwargs = _args(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    out = kpool_score_fp8(*args, **kwargs)
    ref = kpool_score_fp8.reference(*args, **kwargs)
    _check_layout(out, args, kwargs)
    q, weights, pool, tables, kv_lens, tpb = args
    bound = kpool_score_fp8.reference(
        q.abs(), weights.abs(), pool.abs(), tables, kv_lens, tpb, **kwargs
    )
    pools = ref.shape[1]
    visible = _visible(args, pools)
    err = (out[:, :pools] - ref).abs()[visible]
    worst = float((err / bound[visible].clamp_min(1e-30)).max())
    assert bool((err <= 2**-13 * bound[visible]).all()), f"worst error {worst:.3e} of the bound"


def test_kpool_score_fp8_rounding_stays_in_band():
    """General BF16 operands: the E4M3 rounding of the queries and keys moves each score.

    With per-row scale ``s = amax / 448`` an operand ``x`` rounds by at most ``h(x) =
    max(2**-4 |x|, 2**-10 s)`` (half an E4M3 step, the subnormal step below ``2**-6 s``),
    modelled as uniform: variance ``h**2 / 3``. A score's error is then a sum over heads and
    dims of independent terms of variance ``(w'_h)**2 (h(q)**2 k**2 + q**2 h(k)**2) / 3``
    (``w'_h = w_h * w_scale * q_scale``; the relu only shrinks it). Every visible score must
    sit within 8 of those standard deviations plus the accumulation band of the exact-grid
    test; across the ~1.4M visible scores of this cell a Gaussian tail past 8 sigma is below 1e-8.
    """
    cell = kpool_score_fp8.CELLS[0]
    args, kwargs = _args(cell.spec, zlib.crc32(cell.why.encode()) % 2**31, exact=False)
    out = kpool_score_fp8(*args, **kwargs)
    ref = kpool_score_fp8.reference(*args, **kwargs)
    q, weights, pool, tables, kv_lens, tpb = args
    pools = ref.shape[1]
    first = torch.arange(pools, device=DEVICE) * KPOOL
    keys = pool[:, :, 2 * HD :][tables[:, first // TPB], first % TPB].float()  # [T, P, HD]

    def half_step(x):
        scale = x.abs().amax(-1, keepdim=True) / 448
        return torch.maximum(2**-4 * x.abs(), 2**-10 * scale)

    qf = q.float()
    hq, hk = half_step(qf), half_step(keys)
    head_w = (weights.float() * W_SCALE * Q_SCALE) ** 2  # [N, H]
    table = kwargs["request_ids"].long()
    var = torch.empty_like(ref)
    for r0 in range(0, q.shape[0], 64):
        rows = slice(r0, min(r0 + 64, q.shape[0]))
        k2, hk2 = keys[table[rows]] ** 2, hk[table[rows]] ** 2  # [c, P, HD]
        per_head = torch.einsum("rhd,rpd->rhp", hq[rows] ** 2, k2) + torch.einsum(
            "rhd,rpd->rhp", qf[rows] ** 2, hk2
        )
        var[rows] = (per_head * head_w[rows, :, None]).sum(dim=1) / 3
    bound = kpool_score_fp8.reference(
        q.abs(), weights.abs(), pool.abs(), tables, kv_lens, tpb, **kwargs
    )
    visible = _visible(args, pools)
    err = (out[:, :pools] - ref).abs()
    band = 8 * var.sqrt() + 2**-13 * bound
    worst = float((err / band)[visible].max())
    assert bool((err <= band)[visible].all()), f"worst error {worst:.3f} of the band"


def test_guard_refuses_lengths_past_max_kv_len():
    """Row r's key range is ``request_ids[r] * pools`` plus its pool count: a visible length
    past ``max_kv_len`` runs into the next table's keys."""
    spec = dict(context=((0, 300), (0, 300)), rows=(0, 600), pages=16, slots=64)
    args, kwargs = _args(spec, seed=0)
    kwargs["max_kv_len"] = 200
    with pytest.raises(AssertionError, match="max_kv_len"):
        with validating(kpool_score_fp8):
            kpool_score_fp8(*args, **kwargs)
