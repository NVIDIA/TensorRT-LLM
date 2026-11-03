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
"""The MLA DSpark drafter's draft3 glue pieces (one GPU).

* ``draft3_side_launch``: a side-stream launch lands before the helix drafter forward's first
  draft-pool write.
* ``TLLM_DSPARK_DRAFT_ATTN_WINDOW``: the FP8 MLA decode of the windowed fixup decodes exactly what a
  pool holding only each request's last pages decodes.
"""

import math

import pytest
import torch
from test_dspark_ull2_drafter import (
    KV_LORA,
    LATENT,
    PAGE,
    _build_drafter,
    _helix_case,
    _tiny_mla_weights,
    needs_sm100f,
)

from tensorrt_llm._torch.models import modeling_dspark
from tensorrt_llm.mapping import CpType, Mapping

# ---------------------------------------------------------------------------
# Helix drafter forward.
# ---------------------------------------------------------------------------


def _helix_drafter(monkeypatch, cp_size, cp_rank):
    monkeypatch.setenv("TLLM_SPEC_TAIL_FUSION", "1")
    monkeypatch.setenv("TLLM_DSPARK_TAIL_V2", "1")
    monkeypatch.setenv("TRTLLM_K3_ULL2_DRAFT_STORE", "1")
    drafter = _build_drafter(_tiny_mla_weights())
    drafter._helix_mapping = Mapping(
        world_size=cp_size,
        rank=cp_rank,
        cp_size=cp_size,
        tp_size=1,
        cp_config={"cp_type": CpType.HELIX},
    )
    monkeypatch.setattr(
        modeling_dspark,
        "_helix_post_process",
        lambda partial_o, stats, mapping, num_heads, value_dim, zero_kv_mask=None, **_: (
            partial_o.masked_fill(zero_kv_mask.view(-1, 1), 0.0)
        ),
    )
    return drafter


@needs_sm100f
def test_side_launch_is_joined_before_the_first_pool_write(monkeypatch):
    """A side-stream launch (the worker's context store) lands before the forward's first pool
    write: the pools hold junk until a slow side launch restores them; the forward must attend
    the restored context and keep its own block rows."""
    monkeypatch.setenv("TRTLLM_K3_DRAFT3_GLUE", "1")
    drafter = _helix_drafter(monkeypatch, 2, 1)
    ctx_lens = [PAGE * 2 + 3, 37]
    pools, kwargs = _helix_case(drafter, 2, 1, True, ctx_lens, seed=9)
    with torch.no_grad():
        ref = drafter.dflash_forward(ctx_k_cache=pools[0], ctx_kv_cache=pools, **kwargs)
    torch.cuda.synchronize()
    ref_pools = [p.clone() for p in pools]

    pools, kwargs = _helix_case(drafter, 2, 1, True, ctx_lens, seed=9)
    saved = [p.clone() for p in pools]
    for p in pools:
        p.view(torch.uint8).fill_(0x7E)

    def slow_restore():
        torch.cuda._sleep(5_000_000)
        for p, s in zip(pools, saved):
            p.copy_(s)

    drafter.draft3_side_launch(slow_restore, keep=tuple(saved))
    with torch.no_grad():
        out = drafter.dflash_forward(ctx_k_cache=pools[0], ctx_kv_cache=pools, **kwargs)
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
    for p, r in zip(pools, ref_pools):
        assert torch.equal(p.view(torch.uint8), r.view(torch.uint8))
    assert not drafter._draft3_side_keep


# ---------------------------------------------------------------------------
# FP8 MLA decode: attention window.
# ---------------------------------------------------------------------------

HEADS, SEQ_Q, DPAGE = 64, 8, 128
# Context lengths of three requests: 87, 6 and 24 KV tiles of 128.
CONTEXT = (11000, 700, 3000)


def _runner_cls():
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    return CuteDSLNVMlaDecodeBlackwellRunner


def _decode_operands(seed):
    from tensorrt_llm._torch.speculative.dflash_attention import HelixDraftShard

    g = torch.Generator().manual_seed(seed)
    batch = len(CONTEXT)
    lengths = HelixDraftShard(2, 1, DPAGE).local_lengths(torch.tensor(CONTEXT) + SEQ_Q)
    blocks = max(1, math.ceil(int(lengths.max()) / DPAGE))
    num_pages = batch * blocks
    page_rows = torch.randperm(num_pages, generator=g).view(batch, blocks)
    q = torch.randn(batch, SEQ_Q, HEADS, LATENT, generator=g) * 0.1
    cache = torch.randn(num_pages, DPAGE, LATENT, generator=g) * 0.1
    q, cache = q.cuda().to(torch.float8_e4m3fn), cache.cuda().to(torch.float8_e4m3fn)
    cache_seqs = lengths.cuda().to(torch.int32)
    return dict(
        q_latent=q[..., :KV_LORA].permute(2, 3, 1, 0),
        q_rope=q[..., KV_LORA:].permute(2, 3, 1, 0),
        c_latent=cache[..., :KV_LORA].permute(1, 2, 0),
        c_rope=cache[..., KV_LORA:].permute(1, 2, 0),
        page_table=page_rows.cuda().to(torch.int32).t(),
        cache_seqs=cache_seqs,
        kv_bounds=cache_seqs.repeat_interleave(SEQ_Q),
        scales=(
            torch.tensor([1.0 / math.sqrt(192)], device="cuda"),
            torch.tensor([1.0], device="cuda"),
        ),
    )


def _decode(ops, plan):
    import cutlass

    batch = ops["cache_seqs"].shape[0]
    out = torch.full((batch, SEQ_Q, HEADS, KV_LORA), float("nan"), device="cuda").bfloat16()
    stats = torch.full((batch * SEQ_Q, HEADS, 2), float("nan"), device="cuda")
    workspace = torch.zeros(
        _runner_cls().get_max_padded_workspace_size(HEADS, SEQ_Q, KV_LORA, batch, cutlass.Float32),
        dtype=torch.uint8,
        device="cuda",
    )
    torch.ops.trtllm.cute_dsl_mla_decode_fp8_blackwell(
        ops["q_latent"],
        ops["q_rope"],
        ops["c_latent"],
        ops["c_rope"],
        ops["page_table"],
        ops["cache_seqs"],
        out.permute(2, 3, 1, 0),
        workspace,
        HEADS,
        SEQ_Q,
        DPAGE,
        *ops["scales"],
        batch,
        stats,
        ops["kv_bounds"],
        plan,
    )
    return out, stats


def _decode_with(ops, page_table, cache_seqs, kv_bounds, c_latent=None, c_rope=None):
    view = dict(ops)
    view.update(page_table=page_table, cache_seqs=cache_seqs, kv_bounds=kv_bounds)
    if c_latent is not None:
        view.update(c_latent=c_latent, c_rope=c_rope)
    return _decode(view, None)


@needs_sm100f
@pytest.mark.parametrize("window", [1000, 2048, 100000])
def test_attention_window_slices_the_page_table(window):
    """TLLM_DSPARK_DRAFT_ATTN_WINDOW: the windowed fixup decodes exactly what a pool holding only
    each request's last pages (from max(0, ctx - window) // page on) decodes, and a window past
    every context leaves the decode unchanged."""
    ops = _decode_operands(5)
    table = ops["page_table"].t().contiguous()  # [B, blocks]
    fixup = modeling_dspark._MLABlockFixup(
        pages=None,
        offsets=None,
        valid=None,
        page_tables_i32=table,
        seq_lens_i32=ops["cache_seqs"],
        kv_bounds_i32=ops["kv_bounds"],
    )
    windowed = modeling_dspark._window_mla_block_fixup(fixup, window, SEQ_Q, DPAGE)
    got = _decode_with(
        ops, windowed.page_tables_i32.t(), windowed.seq_lens_i32, windowed.kv_bounds_i32
    )
    # Reference: a compact pool with each request's kept pages only.
    seqs = ops["cache_seqs"].long().tolist()
    starts = [max(0, s - SEQ_Q - window) // DPAGE for s in seqs]
    # [pages, page, 576] rows, sliced like the original pool.
    rows576 = torch.cat([ops["c_latent"].permute(2, 0, 1), ops["c_rope"].permute(2, 0, 1)], dim=-1)
    kept, rows = [], []
    for b, (s, st) in enumerate(zip(seqs, starts)):
        n = math.ceil(s / DPAGE)
        ids = table[b, st:n].long()
        rows.append(list(range(len(kept), len(kept) + len(ids))))
        kept.extend(ids.tolist())
    width = max(len(r) for r in rows)
    ref_table = torch.tensor([r + [r[-1]] * (width - len(r)) for r in rows], dtype=torch.int32)
    ids = torch.tensor(kept, device="cuda")
    lens = torch.tensor([s - st * DPAGE for s, st in zip(seqs, starts)], dtype=torch.int32)
    lens = lens.cuda()
    ref = _decode_with(
        ops,
        ref_table.cuda().t(),
        lens,
        lens.repeat_interleave(SEQ_Q),
        rows576[ids][..., :KV_LORA].permute(1, 2, 0),
        rows576[ids][..., KV_LORA:].permute(1, 2, 0),
    )
    torch.cuda.synchronize()
    assert torch.equal(windowed.seq_lens_i32, lens)
    assert torch.equal(got[0].view(torch.int16), ref[0].view(torch.int16))
    if all(st == 0 for st in starts):
        full = _decode(ops, None)
        assert torch.equal(got[0].view(torch.int16), full[0].view(torch.int16))
