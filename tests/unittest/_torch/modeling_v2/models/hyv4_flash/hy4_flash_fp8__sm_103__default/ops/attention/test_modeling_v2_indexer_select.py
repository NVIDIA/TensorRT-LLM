# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the hyv4_flash compressed-indexer selection entry `indexer_select`."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.attention.indexer_select import (  # noqa: E501
    _reference_logits,
    indexer_select,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../../.."]
from _validating import certifying  # noqa: E402,F401

assert torch.cuda.is_available(), "indexer_select requires a CUDA device"

pytestmark = pytest.mark.skipif(
    get_sm_version() != 103, reason="the hyv4_flash target's entries are certified on sm_103"
)

DEVICE = "cuda"
H, D = 16, 128
RATIO, INDEX_TOPK = 4, 2048
SELECT_K = INDEX_TOPK // RATIO - 1
#: Checkpoint scales of the query, the normed group keys and the head weights.
QUERY_STD = 0.02163 * 1024**0.5
KEY_STD = 1.055
HEAD_STD = 0.05004 * 3072**0.5
#: Well above the fp32 reference's distance to float64; kernel and reference sum the same exact
#: products in different orders, so a top-k boundary wider than 2 * LOGIT_ATOL is decided identically.
LOGIT_ATOL = 1e-5


def _args(spec: dict, seed: int) -> tuple:
    """`(query, group_keys, head_weights, row_batch, positions)`: one decode row per request at `kv - 1`
    (`inactive` rows at -1), or with `mode="prefill"` the last `rows` positions of one request."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    rows, kv = spec["rows"], spec["kv"]
    groups = kv // RATIO
    if spec.get("mode") == "prefill":
        batch = 1
        positions = torch.arange(kv - rows, kv, dtype=torch.int32, device=DEVICE)
        row_batch = torch.zeros(rows, dtype=torch.int32, device=DEVICE)
    else:
        batch = rows
        positions = torch.full((rows,), kv - 1, dtype=torch.int32, device=DEVICE)
        row_batch = torch.arange(rows, dtype=torch.int32, device=DEVICE)
        inactive = spec.get("inactive", 0)
        if inactive:
            positions[-inactive:] = -1
    query = (torch.randn(rows, H, D, generator=g, device=DEVICE) * QUERY_STD).to(torch.bfloat16)
    group_keys = (torch.randn(batch, groups, D, generator=g, device=DEVICE) * KEY_STD).to(
        torch.bfloat16
    )
    head_weights = torch.randn(rows, H, generator=g, device=DEVICE) * HEAD_STD
    return query, group_keys, head_weights, row_batch, positions


def _check(out: tuple, ref: tuple, logits: torch.Tensor, positions: torch.Tensor) -> None:
    """`lengths`/`nonempty`/padding/own-group tail exactly; selected logits as a multiset within the tie
    band; selected group ids exactly on rows whose k-th vs (k+1)-th gap exceeds the band."""
    packed, lengths, nonempty = out
    rpacked, rlengths, rnonempty = ref
    rows = packed.shape[0]
    col = torch.arange(INDEX_TOPK, device=DEVICE)
    assert lengths.dtype == torch.int32 and torch.equal(lengths, rlengths)
    assert torch.equal(nonempty, rnonempty)
    assert bool((packed[col[None, :] >= lengths[:, None].long()] == -1).all())
    active = positions >= 0
    qg = torch.div(positions.long(), RATIO, rounding_mode="floor").clamp(min=0)
    tail = torch.where(active, positions.long() - qg * RATIO, torch.full_like(qg, -1))
    n_sel = (lengths.long() - tail - 1) // RATIO
    assert bool((n_sel >= 0).all()) and bool((n_sel <= SELECT_K).all())
    got = packed[:, : SELECT_K * RATIO].reshape(rows, SELECT_K, RATIO).long()
    want = rpacked[:, : SELECT_K * RATIO].reshape(rows, SELECT_K, RATIO).long()
    sel = torch.arange(SELECT_K, device=DEVICE)[None, :] < n_sel[:, None]
    got_g = torch.where(sel, got[:, :, 0] // RATIO, torch.zeros_like(got[:, :, 0]))
    want_g = torch.where(sel, want[:, :, 0] // RATIO, torch.zeros_like(want[:, :, 0]))
    run = torch.arange(RATIO, device=DEVICE)
    assert bool((got[sel] == got_g[sel][:, None] * RATIO + run[None, :]).all()), "partial group run"
    neg = torch.full(got_g.shape, float("-inf"), dtype=logits.dtype, device=DEVICE)
    a = torch.where(sel, torch.gather(logits, 1, got_g), neg).sort(dim=1).values
    b = torch.where(sel, torch.gather(logits, 1, want_g), neg).sort(dim=1).values
    assert torch.equal(a.isinf(), b.isinf())
    finite = ~a.isinf()
    worst = float((a - b).abs()[finite].max()) if bool(finite.any()) else 0.0
    assert worst <= 2 * LOGIT_ATOL, f"selected logits differ by {worst:.3e}"
    best = logits.topk(min(SELECT_K + 1, logits.shape[1]), dim=1).values
    at = n_sel.clamp(min=1, max=SELECT_K) - 1
    kth = best.gather(1, at[:, None]).squeeze(1)
    nxt = best.gather(1, (at + 1).clamp(max=best.shape[1] - 1)[:, None]).squeeze(1)
    untied = (n_sel > 0) & (kth - nxt > 2 * LOGIT_ATOL)
    assert torch.equal(got_g[untied], want_g[untied]), "untied rows select different groups"
    tail_cols = (col[None, :] >= (n_sel * RATIO)[:, None]) & (
        col[None, :] < lengths[:, None].long()
    )
    expect = (qg[:, None] * RATIO + col[None, :] - (n_sel * RATIO)[:, None]).to(torch.int32)
    assert torch.equal(packed[tail_cols], expect[tail_cols]), "own-group tail"
    empty = ~nonempty
    if bool(empty.any()):
        assert bool((packed[empty, 0] == 0).all()) and bool((lengths[empty] == 1).all())


@pytest.mark.parametrize(
    "cell", indexer_select.CELLS, ids=[c.why[:44] for c in indexer_select.CELLS]
)
def test_indexer_select_matches_reference(cell, certifying):  # noqa: F811
    certifying(indexer_select)
    args = _args(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    query, group_keys, head_weights, row_batch, positions = args
    out = indexer_select(query, group_keys, head_weights, row_batch, positions)
    ref = indexer_select.reference(query, group_keys, head_weights, row_batch, positions)
    assert out[0].shape == (cell.spec["rows"], INDEX_TOPK)
    logits = _reference_logits(
        query.float(),
        group_keys.float(),
        head_weights.float(),
        row_batch,
        positions,
        ratio=RATIO,
        scale=(H * D) ** -0.5,
    )
    _check(out, ref, logits, positions)
