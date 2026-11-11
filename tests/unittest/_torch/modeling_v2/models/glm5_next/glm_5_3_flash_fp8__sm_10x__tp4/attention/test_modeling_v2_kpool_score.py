# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the glm5_next fp8 tp4 entry kpool_score."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.attention.kpool_score import (  # noqa: E501
    FP32_MIN,
    kpool_score,
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


def _schedule(spec: dict) -> tuple[torch.Tensor, int]:
    """``(visible int64 [N], tables)`` of a cell.

    Request ``b``'s KV length after the step is ``kv - b % 4``; with ``T > 1`` its rows are
    positions ``kv_b - T .. kv_b - 1`` (visible ``position + 1``) of generation table ``b``,
    which follows one context request's table.
    """
    requests, tokens = spec["requests"], spec["tokens"]
    kv = spec["kv"] - torch.arange(requests, device=DEVICE) % 4
    if tokens == 1:
        return kv, requests
    visible = (kv[:, None] - tokens + 1 + torch.arange(tokens, device=DEVICE)).reshape(-1)
    return visible, 1 + requests


def _args(spec: dict, seed: int) -> tuple[tuple, dict]:
    """The op's arguments the way the target passes them: per-row copies of the generation
    tables (``GenerationRows.tables``, int64). The pool is a coalesced view: each slot's rows sit
    in the second of two subpages, with padded rows."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    visible, num_tables = _schedule(spec)
    pages, slots = spec.get("pages", PAGES), spec.get("slots", SLOTS)
    used = -(-int(visible.max()) // TPB)
    tables = torch.zeros(num_tables, pages, dtype=torch.int64, device=DEVICE)
    for t in range(num_tables):
        tables[t, :used] = torch.randperm(slots, generator=g, device=DEVICE)[:used]
    storage = torch.randn(slots, 2, TPB, 3 * HD + 64, generator=g, device=DEVICE)
    pool = storage.to(torch.bfloat16)[:, 1, :, : 3 * HD]
    n = visible.shape[0]
    q = torch.randn(n, HEADS, HD, generator=g, device=DEVICE).to(torch.bfloat16)
    # The head weights are the tail of the indexer's [k | gate | weights] GEMM output.
    projection = torch.randn(n, 2 * HD + HEADS, generator=g, device=DEVICE).to(torch.bfloat16)
    weights = projection[:, 2 * HD :]
    kwargs = dict(
        num_pools_max=-(-pages * TPB // KPOOL),
        head_dim=HD,
        kpool=KPOOL,
        q_scale=Q_SCALE,
        w_scale=W_SCALE,
    )
    if spec["tokens"] > 1:
        tables = tables[1:].repeat_interleave(spec["tokens"], dim=0)
    return (q, weights, pool, tables, visible, TPB), kwargs


def _check(out: torch.Tensor, args: tuple, kwargs: dict) -> None:
    """Every visible score within ``2**-13 * bound`` of the reference, where ``bound = q_scale
    * sum_h |w_h w_scale| sum_d |q_hd k_d|`` (the reference on absolute values). Both sides
    sum exact BF16 products in FP32, 128 per head and 32 heads, so each is within ``~161 *
    2**-24 * bound`` of the exact score; ``2**-13`` leaves eight times that for the
    tensor-core accumulation order. Every invisible column is the FP32 minimum."""
    ref = kpool_score.reference(*args, **kwargs)
    q, weights, pool, tables, kv_lens, tpb = args
    bound = kpool_score.reference(
        q.abs(), weights.abs(), pool.abs(), tables, kv_lens, tpb, **kwargs
    )
    assert out.shape == ref.shape and out.dtype == torch.float32
    col = torch.arange(out.shape[1], device=DEVICE)
    visible = col[None, :] < (kv_lens.long() // KPOOL)[:, None]
    err = (out - ref).abs()[visible]
    worst = float((err / bound[visible].clamp_min(1e-30)).max()) if err.numel() else 0.0
    assert bool((err <= 2**-13 * bound[visible]).all()), f"worst error {worst:.3e} of the bound"
    assert bool((out[~visible] == FP32_MIN).all())


@pytest.mark.parametrize("cell", kpool_score.CELLS, ids=[c.why[:44] for c in kpool_score.CELLS])
def test_kpool_score_matches_reference(cell, certifying):  # noqa: F811
    """Visible pool scores within the accumulation band of the FP32 reference, every invisible
    pool at the FP32 minimum; see ``_check`` for the band."""
    certifying(kpool_score)
    args, kwargs = _args(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    out = kpool_score(*args, **kwargs)
    _check(out, args, kwargs)


def test_guard_refuses_fp32_queries():
    """The dot reads the queries as BF16: an FP32 query would be rounded without a sign."""
    args, kwargs = _args(dict(requests=1, tokens=4, kv=600, pages=32, slots=64), seed=0)
    q = args[0].float()
    with pytest.raises(AssertionError, match="BF16"):
        with validating(kpool_score):
            kpool_score(q, *args[1:], **kwargs)
