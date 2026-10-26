# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 fmha_sm100 prefill over a partial last page whose V tail past kv_len holds FP8 NaN bytes.

A recycled page keeps what an earlier request wrote past this request's kv_len. fmha_sm100 masks only the scores,
so P @ V multiplies those V rows by zero, and an E4M3 NaN byte (0x7F) turns the tile NaN. run_msa_prefill_gqa
zeroes the V tail first (TRTLLM_MSA_PREFILL_V_TAIL_ZERO), so the output equals the one over a zeroed tail.
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

HEAD_DIM, PAGE = 128, 128
FP8_NAN_BYTE = 0x7F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 10
    or not msa_package_available(),
    reason="requires SM10x and fmha_sm100",
)


def _case(rows: list, num_kv_heads: int, tail_byte: int):
    """A batch whose rows (qo_len, kv_len) own pages of an FP8 pool filled with tail_byte past each kv_len."""
    g = torch.Generator(device="cuda").manual_seed(912)
    pages_per_row = [math.ceil(kv / PAGE) for _, kv in rows]
    slots = sum(pages_per_row) + 3
    pool = torch.full(
        (slots, 2, num_kv_heads, PAGE, HEAD_DIM), tail_byte, dtype=torch.uint8, device="cuda"
    ).view(torch.float8_e4m3fn)
    table = torch.zeros(len(rows), max(pages_per_row), dtype=torch.int32, device="cuda")
    flat, start = [], 0
    for i, ((_, kv), n) in enumerate(zip(rows, pages_per_row)):
        pages = torch.arange(start, start + n, dtype=torch.int32, device="cuda")
        table[i, :n] = pages
        flat.append(pages)
        start += n
        kv_tokens = torch.randn(2, num_kv_heads, kv, HEAD_DIM, device="cuda", generator=g) * 0.5
        padded = torch.zeros(2, num_kv_heads, n * PAGE, HEAD_DIM, device="cuda")
        padded[:, :, :kv] = kv_tokens
        live = padded.view(2, num_kv_heads, n, PAGE, HEAD_DIM).permute(2, 0, 1, 3, 4)
        # Only the first kv_len tokens are written; the rest of the last page keeps tail_byte.
        for p in range(n):
            count = min(PAGE, kv - p * PAGE)
            pool[pages[p], :, :, :count] = live[p, :, :, :count].to(torch.float8_e4m3fn)
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
    return pool, metadata


def _run(q, metadata, num_heads):
    attn = create_autospec(MiniMaxM3MsaSparseAttention, instance=True)
    attn.head_dim, attn.num_heads, attn.layer_idx, attn.q_scaling = HEAD_DIM, num_heads, 0, 1.0
    out = torch.zeros_like(q)
    msa_prefill.run_msa_prefill_gqa(
        attn,
        q,
        metadata,
        out,
        kv_block_indexes=None,
        plan=None,
        row_first=0,
        num_rows=int(metadata.msa_qo_lens_cpu.shape[0]),
    )
    torch.cuda.synchronize()
    return out


# Rows ending mid-page: short prompts, a short chunk over a long prefix, and a two-request batch.
ROWS = [[(5, 5)], [(38, 38)], [(129, 129)], [(170, 170)], [(5, 900)], [(38, 38), (86, 86)]]


@pytest.mark.parametrize("rows", ROWS, ids=lambda rows: "_".join(f"{q}x{kv}" for q, kv in rows))
@pytest.mark.parametrize("num_heads,num_kv_heads", [(32, 2), (16, 1)], ids=["32x2", "16x1"])
def test_fmha_sm100_prefill_ignores_nan_past_kv_len(
    monkeypatch: pytest.MonkeyPatch, rows: list, num_heads: int, num_kv_heads: int
) -> None:
    monkeypatch.setattr(msa_prefill, "dense_prefill_fa4_enabled", lambda *_: False)
    num_tokens = sum(qo for qo, _ in rows)
    g = torch.Generator(device="cuda").manual_seed(7)
    q = (torch.randn(num_tokens, num_heads * HEAD_DIM, device="cuda", generator=g) * 0.5).to(
        torch.bfloat16
    )

    pool, metadata = _case(rows, num_kv_heads, FP8_NAN_BYTE)
    out = _run(q, metadata, num_heads)
    _, zero_tail_metadata = _case(rows, num_kv_heads, 0)
    expected = _run(q, zero_tail_metadata, num_heads)

    assert torch.isfinite(out).all()
    assert torch.equal(out, expected)
    # The V tail of each partial last page is zeroed; K keeps its stale bytes (the scores mask them).
    for i, (_, kv) in enumerate(rows):
        if kv % PAGE:
            page = int(metadata.msa_block_table[i, kv // PAGE])
            tail = pool.view(torch.uint8)[page, :, :, kv % PAGE :]
            assert (tail[1] == 0).all()
            assert (tail[0] == FP8_NAN_BYTE).all()
