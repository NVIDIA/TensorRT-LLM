# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 FP8 dense prefill on FA4 (msa_prefill._run_dense_fa4).

Runs run_msa_prefill_gqa's dense branch over a shuffled paged FP8 pool for a batch of context rows, with FA4 and
with its kill switch (fmha_sm100), and checks FA4 engages, is deterministic, and stays within fmha_sm100's error
against an fp64 reference.
"""

import math
from unittest.mock import create_autospec

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha import msa_prefill
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import (
    MiniMaxM3KVCacheManagerV2,
    MiniMaxM3MsaSparseAttention,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
    msa_package_available,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_backend import (
    MiniMaxM3MsaSparseAttentionMetadata,
)

HEAD_DIM, PAGE, NUM_HEADS, NUM_KV_HEADS = 128, 128, 32, 2


def _fa4_ready() -> bool:
    # FA4 dense prefill engages on SM107 only (other SM10x keep fmha_sm100; see the _arch CPU test).
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        return False
    return msa_package_available() and msa_prefill._dense_fa4_forward() is not None


pytestmark = pytest.mark.skipif(
    not _fa4_ready(), reason="requires SM107, fmha_sm100 and FA4 with the FP8 rescale change"
)


def _case(seed: int, rows: list, outlier: bool):
    """A batch whose rows (qo_len, kv_len) own shuffled pages of one [slots, 2, kv_heads, page, d] FP8 pool."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages_per_row = [math.ceil(kv / PAGE) for _, kv in rows]
    slots = sum(pages_per_row) + 11
    pool = torch.randn(slots, 2, NUM_KV_HEADS, PAGE, HEAD_DIM, device="cuda", generator=g) * 0.5
    num_tokens = sum(qo for qo, _ in rows)
    q = torch.randn(num_tokens, NUM_HEADS * HEAD_DIM, device="cuda", generator=g) * 0.5
    if outlier:
        q.view(num_tokens, NUM_HEADS, HEAD_DIM)[:, :, 3] *= 12.0
        pool[:, 0, :, :, 3] *= 12.0
        pool[::5, 1] *= 8.0
    pool = pool.clamp(-448, 448).to(torch.float8_e4m3fn)
    q = q.to(torch.bfloat16)
    perm = torch.randperm(slots, device="cuda", generator=g).to(torch.int32)
    table = torch.full((len(rows), max(pages_per_row) + 3), 7, dtype=torch.int32, device="cuda")
    flat, start = [], 0
    for i, n in enumerate(pages_per_row):
        table[i, :n] = perm[start : start + n]
        flat.append(perm[start : start + n])
        start += n
    qo_lens = torch.tensor([qo for qo, _ in rows], dtype=torch.int32)
    kv_lens = torch.tensor([kv for _, kv in rows], dtype=torch.int32)
    kv_cache_manager = create_autospec(MiniMaxM3KVCacheManagerV2, instance=True)
    kv_cache_manager.get_buffers.return_value = pool
    kv_cache_manager.is_nvfp4_layer.return_value = False
    metadata = create_autospec(MiniMaxM3MsaSparseAttentionMetadata, instance=True)
    metadata.kv_cache_manager = kv_cache_manager
    metadata.msa_qo_lens_cpu = qo_lens
    metadata.msa_kv_lens_cpu = kv_lens
    metadata.msa_qo_offset_cpu = kv_lens - qo_lens
    metadata.msa_block_table = table
    metadata.msa_kv_indices = torch.cat(flat)
    return q, pool, metadata, flat


def _reference(q, pool, rows, flat):
    outs, tok = [], 0
    for (qo, kv), pages in zip(rows, flat):
        k = pool[pages.long(), 0].permute(1, 0, 2, 3).reshape(NUM_KV_HEADS, -1, HEAD_DIM)[:, :kv]
        v = pool[pages.long(), 1].permute(1, 0, 2, 3).reshape(NUM_KV_HEADS, -1, HEAD_DIM)[:, :kv]
        group = NUM_HEADS // NUM_KV_HEADS
        k = k.double().repeat_interleave(group, 0)
        v = v.double().repeat_interleave(group, 0)
        qd = (
            q[tok : tok + qo]
            .view(qo, NUM_HEADS, HEAD_DIM)
            .to(torch.float8_e4m3fn)
            .double()
            .permute(1, 0, 2)
        )
        s = torch.einsum("htd,hkd->htk", qd, k) * HEAD_DIM**-0.5
        mask = (
            torch.arange(kv, device="cuda")[None, :]
            > (kv - qo + torch.arange(qo, device="cuda"))[:, None]
        )
        p = torch.softmax(s.masked_fill(mask[None], float("-inf")), -1)
        outs.append(torch.einsum("htk,hkd->htd", p, v).permute(1, 0, 2))
        tok += qo
    return torch.cat(outs)


def _run(q, metadata, row_first, num_rows):
    attn = create_autospec(MiniMaxM3MsaSparseAttention, instance=True)
    attn.head_dim, attn.num_heads, attn.layer_idx, attn.q_scaling = HEAD_DIM, NUM_HEADS, 0, 1.0
    out = torch.full_like(q, float("nan"))
    msa_prefill.run_msa_prefill_gqa(
        attn,
        q,
        metadata,
        out,
        kv_block_indexes=None,
        plan=None,
        row_first=row_first,
        num_rows=num_rows,
    )
    torch.cuda.synchronize()
    return out.view(-1, NUM_HEADS, HEAD_DIM)


def _max_err(x, ref):
    return float((x.double() - ref).abs().max())


# A long chunked prefill, a fresh prompt, a short tail whose packed queries fit one 128-row tile, and a chunk
# ending mid-page.
ROWS = [(300, 2_000), (257, 257), (5, 900), (64, 640)]


@pytest.mark.parametrize("outlier", [False, True], ids=["randn", "outlier"])
def test_dense_prefill_fa4_matches_fmha_sm100_band(
    monkeypatch: pytest.MonkeyPatch, outlier: bool
) -> None:
    q_all, pool, metadata, flat = _case(1234 + outlier, ROWS, outlier)
    q, row_first, served = q_all, 0, ROWS
    ref = _reference(q, pool, served, flat)

    calls = []
    real = msa_prefill._dense_fa4_forward()

    def spy(*args, **kwargs):
        calls.append(kwargs.get("out") is not None)
        return real(*args, **kwargs)

    monkeypatch.setattr(msa_prefill, "_dense_fa4_forward", lambda: spy)
    fa4 = _run(q, metadata, row_first, len(served))
    fa4_again = _run(q, metadata, row_first, len(served))
    assert any(calls), "FA4 did not serve the dense prefill"
    assert torch.equal(fa4.view(torch.int16), fa4_again.view(torch.int16)), (
        "FA4 dense prefill is not deterministic"
    )

    monkeypatch.setattr(msa_prefill, "dense_prefill_fa4_enabled", lambda *_: False)
    stock = _run(q, metadata, row_first, len(served))

    assert torch.isfinite(fa4).all()
    fa4_err, stock_err = _max_err(fa4, ref), _max_err(stock, ref)
    # Rule 1b: FA4 rounds differently from fmha_sm100, so compare against fp64, not bitwise; a single BF16
    # output-rounding step is the allowance over fmha_sm100's own worst error.
    assert fa4_err <= stock_err * 1.05 + 2.0**-9 * float(ref.abs().max()), (fa4_err, stock_err)


def test_dense_prefill_fa4_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(msa_prefill, "_DENSE_PREFILL_FA4", False)
    msa_prefill._dense_fa4_forward.cache_clear()
    try:
        assert not msa_prefill.dense_prefill_fa4_enabled(True, HEAD_DIM, PAGE)
    finally:
        monkeypatch.undo()
        msa_prefill._dense_fa4_forward.cache_clear()
    monkeypatch.setattr(msa_prefill, "get_sm_version", lambda: 107)
    assert msa_prefill.dense_prefill_fa4_enabled(True, HEAD_DIM, PAGE)
    assert not msa_prefill.dense_prefill_fa4_enabled(False, HEAD_DIM, PAGE)
    assert not msa_prefill.dense_prefill_fa4_enabled(True, HEAD_DIM, 64)
