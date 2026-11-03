# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The MLA DSpark drafter's context split over a plain TP group (Kimi K3.1 generation servers, one GPU).

* ``ctx_split_mapping``: a helix mapping (tp 1, cp = tp, cp rank = tp rank) only on plain TP.
* ``ctx_split_local_page_table``: a fresh row-major table of this rank's pages (local page ``k`` = global page
  ``k * cp + rank``), at least two columns wide.
* ``ctx_split_block_fixup`` (one launch) builds exactly the local table and the helix block fixup of
  ``ctx_split_local_page_table`` + ``_build_mla_block_fixup`` on this batch's table rows.
* Every rank's block decode over its local pages of the replicated context, merged with the decode's (max, sum)
  softmax statistics as the helix exchange combines them, equals the unsplit decode over the whole context, for
  BF16 and FP8 pools, contexts that end on and off page boundaries, and ranks that hold no key of a request.
  The exchange itself is the helix all-to-all of the multi-GPU tests.
* ``TLLM_DSPARK_DRAFT_ATTN_WINDOW`` under the split: every rank's windowed local fixup together attends exactly
  the window's positions, each once, and the merged windowed split decodes equal the windowed unsplit decode.
"""

import pytest
import torch
from test_dspark_ull2_drafter import (
    BLOCK,
    LATENT,
    PAGE,
    _build_drafter,
    _tiny_mla_weights,
    needs_sm100f,
)

from tensorrt_llm._torch.models import modeling_dspark
from tensorrt_llm._torch.pyexecutor import input_commit
from tensorrt_llm._torch.speculative.dflash_attention import HelixDraftShard, to_draft_fp8
from tensorrt_llm._torch.speculative.dspark_ull2 import (
    ctx_split_block_fixup,
    ctx_split_local_page_table,
    ctx_split_mapping,
    ctx_split_min_batch,
)
from tensorrt_llm.mapping import CpType, Mapping


def test_ctx_split_mapping_only_on_plain_tp(monkeypatch):
    tp8 = Mapping(world_size=8, rank=5, tp_size=8, moe_ep_size=8)
    monkeypatch.setattr(input_commit, "_K31_DRAFTS", True)
    monkeypatch.setattr(input_commit, "_K31_ATTENTION_DP", False)
    monkeypatch.setattr(input_commit, "_K31_GENERATION", False)  # aggregated / context server
    assert ctx_split_mapping(tp8) is None
    monkeypatch.setattr(input_commit, "_K31_GENERATION", True)
    split = ctx_split_mapping(tp8)
    assert split.has_cp_helix() and split.cp_size == 8 and split.tp_size == 1
    assert split.cp_rank == tp8.tp_rank == 5
    assert split.cp_config["use_nccl_for_alltoall"] is False
    monkeypatch.setenv("TRTLLM_DSPARK_CTX_SPLIT_A2A", "nccl")
    assert ctx_split_mapping(tp8).cp_config["use_nccl_for_alltoall"] is True
    assert ctx_split_mapping(Mapping(world_size=1, rank=0, tp_size=1)) is None
    adp = Mapping(world_size=8, rank=1, tp_size=8, moe_ep_size=8, enable_attention_dp=True)
    assert ctx_split_mapping(adp) is None
    pp = Mapping(world_size=8, rank=1, tp_size=4, pp_size=2)
    assert ctx_split_mapping(pp) is None
    helix = Mapping(world_size=4, rank=1, tp_size=1, cp_size=4, cp_config={"cp_type": CpType.HELIX})
    assert ctx_split_mapping(helix) is None
    monkeypatch.delenv("TRTLLM_DSPARK_CTX_SPLIT_MIN_BATCH", raising=False)
    assert ctx_split_min_batch() == 8
    monkeypatch.setenv("TRTLLM_DSPARK_CTX_SPLIT_MIN_BATCH", "1")
    assert ctx_split_min_batch() == 1


@pytest.mark.parametrize("pages,cp_size", [(1, 2), (5, 2), (7, 8), (16, 8), (33, 16)])
def test_local_page_table_layout(pages, cp_size):
    table = torch.randperm(4 * pages).view(4, pages).to(torch.int32)
    for rank in range(cp_size):
        local = ctx_split_local_page_table(table, cp_size, rank)
        assert local.dtype == torch.int32 and local.is_contiguous()
        assert local.shape == (4, max(2, -(-pages // cp_size)))
        assert local.stride() == (local.shape[1], 1)
        owned = list(range(rank, pages, cp_size))
        for k, g in enumerate(owned):
            assert torch.equal(local[:, k], table[:, g])
        assert not local[:, len(owned) :].any()


def _full_case(ctx_lens, fp8, seed):
    """Replicated pools holding every position of every request (as each TP rank keeps them) and its tables."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    batch = len(ctx_lens)
    max_pages = -(-(max(ctx_lens) + BLOCK) // PAGE) + 1
    pool = torch.randn(batch * max_pages + 3, 1, 1, PAGE, LATENT, device="cuda", generator=gen) * 2
    pool = to_draft_fp8(pool.bfloat16()) if fp8 else pool.bfloat16()
    perm = torch.randperm(batch * max_pages + 3, device="cuda", generator=gen)[: batch * max_pages]
    tables = perm.view(batch, max_pages).to(torch.int32).contiguous()
    return pool, tables


@needs_sm100f
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16_pool", "fp8_pool"])
@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_split_decode_merges_to_the_unsplit_decode(fp8, cp_size):
    drafter = _build_drafter(_tiny_mla_weights())
    heads = drafter._num_heads
    # On and off page boundaries, a request shorter than one page (every other rank holds none of it) and one
    # spanning several cp spans.
    ctx_lens = [
        PAGE * cp_size + 3,
        37,
        PAGE - BLOCK - 2,
        5 * PAGE * cp_size // 2 + 30,
        PAGE * 2 - BLOCK,
    ]
    pool, tables = _full_case(ctx_lens, fp8, seed=cp_size + 10 * fp8)
    batch = len(ctx_lens)
    num_ctx = torch.tensor(ctx_lens, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(3)
    query = (
        torch.randn(batch, BLOCK, heads, LATENT, device="cuda", generator=gen) * 0.5
    ).bfloat16()
    with torch.no_grad():
        full = modeling_dspark._build_mla_block_fixup(num_ctx, tables.long(), BLOCK, PAGE)
        ref = drafter._cute_dsl_block_decode(query, pool, full, BLOCK).float()
        outs, maxes, sums, empties = [], [], [], []
        for rank in range(cp_size):
            local = ctx_split_local_page_table(tables, cp_size, rank)
            fixup = modeling_dspark._build_mla_block_fixup(
                num_ctx, local.long(), BLOCK, PAGE, HelixDraftShard(cp_size, rank, PAGE)
            )
            out, stats = drafter._cute_dsl_block_decode(query, pool, fixup, BLOCK)
            outs.append(out.float().reshape(batch * BLOCK, heads, -1))
            maxes.append(stats[..., 0])
            sums.append(stats[..., 1])
            empties.append((fixup.kv_bounds_i32 == 0).view(-1, 1).expand(-1, heads))
    torch.cuda.synchronize()
    # The helix combine: rank weight sum_r * exp(max_r - max), ranks without a key of the token left out.
    mx = torch.stack(maxes).masked_fill(torch.stack(empties), float("-inf"))
    peak = mx.amax(0, keepdim=True)
    w = torch.stack(sums) * torch.exp(mx - peak)
    w = w.masked_fill(torch.stack(empties), 0.0)
    merged = (w.unsqueeze(-1) * torch.stack(outs).nan_to_num(0.0)).sum(0) / w.sum(0).unsqueeze(-1)
    ref = ref.reshape(batch * BLOCK, heads, -1)
    rel = (merged - ref).norm() / ref.norm()
    assert rel < (0.03 if fp8 else 0.005), rel
    # Some rank held no key of the short request: the case the zero-KV mask covers.
    assert torch.stack(empties).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("block,page", [(5, 32), (6, 64), (7, 32)])
@pytest.mark.parametrize("cp_size", [2, 8, 16])
def test_fused_fixup_matches_the_torch_fixup(block, page, cp_size):
    gen = torch.Generator(device="cuda").manual_seed(block * 100 + cp_size)
    rows_total, width = 12, 3 * cp_size + 5
    table = torch.randint(
        0, 1 << 20, (rows_total, width), generator=gen, device="cuda", dtype=torch.int32
    )
    rows = torch.randperm(rows_total, generator=gen, device="cuda")[:7]
    cap = width * page - block
    ctx_len = torch.tensor(
        [0, 1, page - 1, page * cp_size, page * cp_size + 3, cap - 1, cap // 2], device="cuda"
    )
    for rank in (0, 1, cp_size - 1):
        got = ctx_split_block_fixup(ctx_len, table, rows, block, page, cp_size, rank)
        local = ctx_split_local_page_table(table.index_select(0, rows), cp_size, rank)
        ref = modeling_dspark._build_mla_block_fixup(
            ctx_len, local.long(), block, page, HelixDraftShard(cp_size, rank, page)
        )
        g_local, g_pages, g_offsets, g_owned, g_seq, g_bounds = got
        assert torch.equal(g_local, local)
        assert torch.equal(g_pages, ref.pages) and g_pages.dtype == ref.pages.dtype
        assert torch.equal(g_offsets, ref.offsets)
        assert torch.equal(g_owned, ref.owned) and g_owned.shape == ref.owned.shape
        assert torch.equal(g_seq, ref.seq_lens_i32) and g_seq.dtype == ref.seq_lens_i32.dtype
        assert torch.equal(g_bounds, ref.kv_bounds_i32)
        assert torch.equal(ref.page_tables_i32, g_local)


def _attended(fixup, b, page):
    """(physical page, offset) of every key position request ``b`` attends under ``fixup``."""
    n = int(fixup.seq_lens_i32[b])
    table = fixup.page_tables_i32[b].tolist()
    return [(table[p // page], p % page) for p in range(n)]


@pytest.mark.parametrize("cp_size", [2, 4, 8, 32])
@pytest.mark.parametrize("window", [1, PAGE - 3, 3 * PAGE + 5, 40 * PAGE, 10**6])
def test_split_window_attends_each_window_position_once(cp_size, window):
    """The context split's windowed local fixups: their union is the window (global pages from
    max(0, ctx - window) // page on, through the block), no position twice, the key bounds follow the
    lengths, and a request whose context is not longer than the window keeps its rank-local rows."""
    ctx_lens = [
        0,
        5,
        PAGE - BLOCK,
        9 * PAGE,
        PAGE * cp_size + 3,
        7 * PAGE * cp_size + 30,
        100 * PAGE + 17,
    ]
    width = -(-(max(ctx_lens) + BLOCK) // PAGE) + 1
    table = torch.arange(len(ctx_lens) * width, dtype=torch.int32).view(len(ctx_lens), width)
    num_ctx = torch.tensor(ctx_lens)
    full = modeling_dspark._build_mla_block_fixup(num_ctx, table.long(), BLOCK, PAGE)
    ref = modeling_dspark._window_mla_block_fixup(full, window, BLOCK, PAGE)
    ranks = []
    for rank in range(cp_size):
        local = ctx_split_local_page_table(table, cp_size, rank)
        fixup = modeling_dspark._build_mla_block_fixup(
            num_ctx, local.long(), BLOCK, PAGE, HelixDraftShard(cp_size, rank, PAGE)
        )
        win = modeling_dspark._window_mla_block_fixup(
            fixup, window, BLOCK, PAGE, ctx_len=num_ctx, cp_size=cp_size, cp_rank=rank
        )
        assert torch.equal(win.kv_bounds_i32, win.seq_lens_i32.repeat_interleave(BLOCK))
        assert win.pages is fixup.pages and win.owned is fixup.owned
        for b, ctx in enumerate(ctx_lens):
            if ctx <= window:
                assert torch.equal(win.page_tables_i32[b], fixup.page_tables_i32[b])
                assert win.seq_lens_i32[b] == fixup.seq_lens_i32[b]
        ranks.append(win)
    for b, ctx in enumerate(ctx_lens):
        first = max(0, ctx - window) // PAGE
        want = [(int(table[b, q // PAGE]), q % PAGE) for q in range(first * PAGE, ctx + BLOCK)]
        assert _attended(ref, b, PAGE) == want
        got = [pos for win in ranks for pos in _attended(win, b, PAGE)]
        assert len(got) == len(want) and sorted(got) == sorted(want)


@needs_sm100f
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16_pool", "fp8_pool"])
@pytest.mark.parametrize("cp_size", [2, 4, 8, 32])
@pytest.mark.parametrize("window", [3 * PAGE + 5, 9 * PAGE])
def test_windowed_split_decode_merges_to_the_windowed_unsplit_decode(fp8, cp_size, window):
    """The helix combine of the windowed split decodes equals the windowed unsplit decode; a context
    not longer than the window (37 tokens) keeps every rank's rows, and contexts on a page boundary
    (12 pages) and past a cp span are clipped."""
    drafter = _build_drafter(_tiny_mla_weights())
    heads = drafter._num_heads
    ctx_lens = [PAGE * cp_size + 3, 37, 5 * PAGE * cp_size // 2 + 30, 12 * PAGE, window + PAGE]
    pool, tables = _full_case(ctx_lens, fp8, seed=cp_size + 10 * fp8 + window)
    batch = len(ctx_lens)
    num_ctx = torch.tensor(ctx_lens, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(5)
    query = (
        torch.randn(batch, BLOCK, heads, LATENT, device="cuda", generator=gen) * 0.5
    ).bfloat16()
    with torch.no_grad():
        full = modeling_dspark._build_mla_block_fixup(num_ctx, tables.long(), BLOCK, PAGE)
        full = modeling_dspark._window_mla_block_fixup(full, window, BLOCK, PAGE)
        ref = drafter._cute_dsl_block_decode(query, pool, full, BLOCK).float()
        outs, maxes, sums, empties = [], [], [], []
        for rank in range(cp_size):
            local, pages, offsets, owned, seq_lens, kv_bounds = ctx_split_block_fixup(
                num_ctx, tables, torch.arange(batch, device="cuda"), BLOCK, PAGE, cp_size, rank
            )
            fixup = modeling_dspark._MLABlockFixup(
                pages=pages,
                offsets=offsets,
                valid=modeling_dspark._block_upper_triangle(BLOCK, "cuda"),
                page_tables_i32=local,
                seq_lens_i32=seq_lens,
                kv_bounds_i32=kv_bounds,
                owned=owned,
            )
            clipped = modeling_dspark._window_mla_block_fixup(
                fixup, window, BLOCK, PAGE, ctx_len=num_ctx, cp_size=cp_size, cp_rank=rank
            )
            assert torch.equal(clipped.page_tables_i32[1], fixup.page_tables_i32[1])
            assert torch.equal(clipped.kv_bounds_i32, clipped.seq_lens_i32.repeat_interleave(BLOCK))
            fixup = clipped
            out, stats = drafter._cute_dsl_block_decode(query, pool, fixup, BLOCK)
            outs.append(out.float().reshape(batch * BLOCK, heads, -1))
            maxes.append(stats[..., 0])
            sums.append(stats[..., 1])
            empties.append((fixup.kv_bounds_i32 == 0).view(-1, 1).expand(-1, heads))
    torch.cuda.synchronize()
    mx = torch.stack(maxes).masked_fill(torch.stack(empties), float("-inf"))
    peak = mx.amax(0, keepdim=True)
    w = torch.stack(sums) * torch.exp(mx - peak)
    w = w.masked_fill(torch.stack(empties), 0.0)
    merged = (w.unsqueeze(-1) * torch.stack(outs).nan_to_num(0.0)).sum(0) / w.sum(0).unsqueeze(-1)
    ref = ref.reshape(batch * BLOCK, heads, -1)
    rel = (merged - ref).norm() / ref.norm()
    assert rel < (0.03 if fp8 else 0.005), rel
