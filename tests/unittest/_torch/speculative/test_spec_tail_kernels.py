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
"""The spec-decode step kernels reproduce the eager chains they replace."""

import copy
from types import SimpleNamespace
from typing import Optional

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch.models.modeling_dspark import _build_mla_block_fixup
from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm._torch.speculative.dflash_attention import HelixDraftShard
from tensorrt_llm._torch.speculative.spec_tail_kernels import (
    decode_draft_block_table,
    strict_accept,
)
from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels")


def _eager_strict_accept(target, draft, num_ctx, batch):
    k = draft.shape[-1]
    num_gens = batch - num_ctx
    accepted = torch.zeros((batch, k + 1), dtype=torch.int32, device="cuda")
    num_accepted = torch.ones(batch, dtype=torch.int32, device="cuda")
    accepted[:num_ctx, 0] = target[:num_ctx]
    gen_target = target[num_ctx:].reshape(num_gens, k + 1)
    num_accepted[num_ctx:] += torch.cumprod((draft == gen_target[:, :k]).int(), dim=-1).sum(1)
    accepted[num_ctx:] = gen_target
    return accepted, num_accepted


@pytest.mark.parametrize("num_ctx,num_gens", [(0, 1), (3, 0), (2, 5), (1, 130), (200, 7)])
@pytest.mark.parametrize("draft_len", [1, 3, 7])
def test_strict_accept_matches_eager(num_ctx, num_gens, draft_len):
    gen = torch.Generator(device="cuda").manual_seed(num_gens * 10 + draft_len)
    batch = num_ctx + num_gens
    target = torch.randint(
        0, 4, (num_ctx + num_gens * (draft_len + 1),), device="cuda", generator=gen
    )
    # A strided view, as the draft tokens of a wider buffer.
    wide = torch.randint(0, 4, (num_gens, draft_len + 2), device="cuda", generator=gen)
    draft = wide.to(torch.int32)[:, 1 : draft_len + 1]
    accepted, num_accepted = strict_accept(target, draft, num_ctx, batch)
    ref_accepted, ref_num = _eager_strict_accept(target, draft, num_ctx, batch)
    assert torch.equal(num_accepted, ref_num)
    assert torch.equal(accepted[:num_ctx, 0], ref_accepted[:num_ctx, 0])
    assert torch.equal(accepted[num_ctx:], ref_accepted[num_ctx:])


def test_strict_accept_one_variant_for_all_batches():
    """Serving changes the request counts every iteration: one compiled kernel serves them all."""
    import triton

    from tensorrt_llm._torch.speculative import spec_tail_kernels

    caches = getattr(spec_tail_kernels._strict_accept_kernel, "device_caches", None)
    if caches is None:
        pytest.skip("this Triton has no JITFunction.device_caches")
    device = triton.runtime.driver.active.get_current_device()
    draft_len = 5

    def run(num_ctx, num_gens, offset, row_stride, col_stride):
        target = torch.randint(0, 4, (num_ctx + num_gens * (draft_len + 1),), device="cuda")
        # The draft tokens as a view into a wider buffer: any start offset (pointer alignment),
        # row stride and column stride. Rows may overlap; the kernel only reads the draft.
        buffer = torch.randint(
            0,
            4,
            (offset + num_gens * row_stride + draft_len * col_stride,),
            device="cuda",
            dtype=torch.int32,
        )
        draft = buffer.as_strided((num_gens, draft_len), (row_stride, col_stride), offset)
        accepted, num_accepted = strict_accept(target, draft, num_ctx, num_ctx + num_gens)
        ref_accepted, ref_num = _eager_strict_accept(target, draft, num_ctx, num_ctx + num_gens)
        assert torch.equal(num_accepted, ref_num)
        assert torch.equal(accepted[num_ctx:], ref_accepted[num_ctx:])

    run(3, 0, 0, draft_len + 2, 1)
    variants = len(caches[device][0])
    for row_stride in (draft_len + 2, 16):
        for col_stride in (1, 2):
            for num_ctx, num_gens in [
                (1, 0),
                (16, 0),
                (17, 0),
                (0, 1),
                (0, 16),
                (2, 5),
                (1, 130),
                (200, 7),
            ]:
                for offset in (0, 1):
                    run(num_ctx, num_gens, offset, row_stride, col_stride)
    assert len(caches[device][0]) == variants


@pytest.mark.parametrize("block_offset", [0, 3])
def test_decode_draft_block_table_matches_eager(block_offset: int) -> None:
    """The decoded table is the clamped, divided layer-relative offsets
    (``src - block_offset``, with ``src`` left untouched); the count is each
    row's extent, one past its last backed entry, so a sliding window that
    released the low entries leaves the count covering the positions the row
    still addresses."""
    num_pools, rows, n_blocks, divisor = 2, 9, 3000, 6
    gen = torch.Generator(device="cuda").manual_seed(1)
    src = torch.randint(
        -(1 << 20), 1 << 20, (num_pools, rows + 3, 2, n_blocks), device="cuda", generator=gen
    )
    # About a quarter of the entries are the zero offset and a quarter the -1
    # of an unallocated block; besides, one row is all negative and one all zero.
    kind = torch.randint(0, 4, src.shape, device="cuda", generator=gen)
    src = torch.where(kind == 0, 0, torch.where(kind == 1, -1, src)).to(torch.int32)
    view = src[1, :rows, 0]
    view[0] = -torch.randint(1, 1 << 20, (n_blocks,), device="cuda", generator=gen).to(torch.int32)
    view[1] = 0
    view[2] = block_offset
    # A windowed row: entries 0..99 released, 100..139 backed, the rest unfunded.
    view[3] = -1
    view[3, 100:140] = (
        torch.arange(100, 140, device="cuda", dtype=torch.int32) * divisor + block_offset
    )
    assert bool((view < 0).float().mean() > 0.3) and bool((view == 0).float().mean() > 0.2)
    table = torch.full((rows + 3, n_blocks), -7, dtype=torch.int32, device="cuda")
    counts = torch.full((rows + 3,), -7, dtype=torch.long, device="cuda")
    original = src.clone()
    decode_draft_block_table(view, divisor, table, counts, block_offset=block_offset)
    assert torch.equal(src, original)
    encoded = view.to(torch.int64) - block_offset
    ordinals = torch.arange(1, n_blocks + 1, device="cuda")
    extent = torch.where(encoded >= 0, ordinals, 0).max(dim=1).values
    assert torch.equal(counts[:rows], extent)
    assert counts[3].item() == 140
    assert torch.equal(table[3, 100:140], torch.arange(100, 140, device="cuda", dtype=torch.int32))
    expected = encoded.clamp(min=0).div(divisor, rounding_mode="floor").to(torch.int32)
    assert torch.equal(table[:rows], expected)
    assert counts[0].item() == 0 and bool((table[0] == 0).all())
    assert counts[1].item() == (n_blocks if block_offset == 0 else 0)
    assert bool((table[1] == 0).all())
    assert counts[2].item() == n_blocks and bool((table[2] == 0).all())
    assert bool((table[rows:] == -7).all()) and bool((counts[rows:] == -7).all())


class _LatentDrafter(nn.Module):
    """The surface of a single-latent (MLA) drafter the worker's preparation reads."""

    _kv_factor = 1
    lilicorr = None

    def __init__(self, vocab, hidden, features, layers, head_dim, mask_token_id):
        super().__init__()
        gen = torch.Generator(device="cuda").manual_seed(5)
        bf16 = dict(device="cuda", dtype=torch.bfloat16)
        self.mask_token_id = mask_token_id
        self.fc = nn.Linear(features, hidden, bias=False, **bf16)
        self.fc.weight.data = torch.randn(hidden, features, generator=gen, **bf16) * 0.05
        self.hidden_norm = nn.RMSNorm(hidden, eps=1e-6, **bf16)
        embed = nn.Embedding(vocab, hidden, **bf16)
        embed.weight.data = torch.randn(vocab, hidden, generator=gen, **bf16)
        self.draft_model_full = SimpleNamespace(model=SimpleNamespace(embed_tokens=embed))
        self.latent_weight = torch.randn(layers * head_dim, hidden, generator=gen, **bf16) * 0.1
        self.layers, self.head_dim = layers, head_dim

    def project_target_hidden(self, hidden_states):
        return self.hidden_norm(self.fc(hidden_states.to(self.fc.weight.dtype)))

    def precompute_context_kv(self, projected, positions):
        latent = F.linear(projected, self.latent_weight).view(-1, self.layers, self.head_dim)
        latent = latent + (positions.view(-1, 1, 1) % 7).to(latent.dtype)
        latent[0, 0, 0] = -latent[0, 0, 0].abs() - 1  # a negative entry for the signed zero
        return latent.unsqueeze(2).contiguous(), None


def _paged_worker(draft_len, page_size, max_blocks, num_slots, layers, head_dim, max_ctx, gen):
    worker = DFlashWorker(DFlashDecodingConfig(max_draft_len=draft_len), Mapping())
    pages = num_slots * max_blocks
    # Nonzero contents, so that a write (even of a masked, zeroed latent) and a
    # missing or stray one are both visible.
    storage = (
        torch.rand((pages, layers, 1, 1, page_size, head_dim), device="cuda", generator=gen) + 0.5
    ).to(torch.bfloat16)
    pools = [storage[:, layer] for layer in range(layers)]
    worker._ctx_paged = True
    worker._ctx_kv_buf = pools
    worker._ctx_kv_ptrs = torch.tensor([p.data_ptr() for p in pools], device="cuda")
    worker._ctx_page_size = page_size
    worker._compute_block_size = draft_len
    worker._max_ctx = max_ctx
    worker._dummy_slot = num_slots - 1
    worker._ctx_len = torch.zeros(num_slots, dtype=torch.long, device="cuda")
    worker._batch_to_slot_buffer = SimpleNamespace(
        tensor=torch.zeros(num_slots - 1, dtype=torch.long, device="cuda")
    )
    perm = torch.randperm(pages, device="cuda", generator=gen).to(torch.int32)
    worker._ctx_block_tables = perm[: num_slots * max_blocks].view(num_slots, max_blocks).clone()
    worker._ctx_block_counts = torch.zeros(num_slots, dtype=torch.long, device="cuda")
    worker._ctx_base_pages = torch.zeros(num_slots, dtype=torch.long, device="cuda")
    worker._ctx_block_tables_read = torch.zeros_like(worker._ctx_block_tables)
    worker._resolved_mask_token_id = 3
    return worker, storage


@pytest.mark.parametrize("helix_rank", [None, 0, 1], ids=["no_helix", "helix_r0", "helix_r1"])
@pytest.mark.parametrize("side_stream", [False, True])
@pytest.mark.parametrize("based", [False, True], ids=["no_base", "based"])
@pytest.mark.parametrize("draft_len", [3, 7])
def test_paged_latent_prepare_matches_eager(
    draft_len: int, side_stream: bool, helix_rank: Optional[int], based: bool
) -> None:
    """The fused generation-step preparation matches the eager one, including
    the read view rebased past each slot's base pages: the advertised context,
    the rebased block-table rows and the block placement.

    ``side_stream``: the fused path stores the context latents on the
    worker's side stream and hands the drafter the event to wait on.
    ``helix_rank``: the pages are sharded over a helix CP group of two (a
    unified cache shared with a helix target); each rank stores and places only
    its own positions."""
    if based and helix_rank is not None:
        pytest.skip("a helix-sharded context is a unified cache, whose slots start at base 0")
    vocab, hidden, features, layers, head_dim = 100, 64, 128, 3, 96
    page_size, max_blocks, num_slots, max_ctx = 16, 6, 10, 70
    num_ctx_reqs, num_gens = 1, 6
    batch = num_ctx_reqs + num_gens
    width = draft_len + 1
    gen = torch.Generator(device="cuda").manual_seed(draft_len)
    worker, storage = _paged_worker(
        draft_len, page_size, max_blocks, num_slots, layers, head_dim, max_ctx, gen
    )
    drafter = _LatentDrafter(vocab, hidden, features, layers, head_dim, mask_token_id=3)
    cp_size = 1 if helix_rank is None else 2
    shard = None if helix_rank is None else HelixDraftShard(cp_size, helix_rank, page_size)
    worker._ctx_helix = shard

    # Rows: an ordinary request, one at the max_ctx ceiling, one whose context
    # outruns its pages, one with no pages, and two padding rows on the dummy slot.
    slots = torch.tensor([0, 2, 4, 1, 3, worker._dummy_slot, worker._dummy_slot], device="cuda")
    worker._batch_to_slot_buffer.tensor[:batch] = slots
    worker._ctx_len[:5] = torch.tensor([10, 3, 66, 5, 40], device="cuda")
    worker._ctx_block_counts[:batch] = torch.tensor([0, 3, 6, 2, 0, 1, 1], device="cuda")
    if based:
        # Slot 2 (row 2) advertises past its base, so its read row shifts by
        # it; slot 4 (row 3) advertises less than its base, reads nothing and
        # shifts only to the page holding its context. The dummy slot keeps 0.
        worker._ctx_base_pages[:5] = torch.tensor([0, 1, 1, 1, 2], device="cuda")
    num_accepted = torch.tensor([1, 2, width, 3, 1, 2, 1], dtype=torch.int32, device="cuda")
    accepted = torch.randint(
        0, vocab, (batch, width), dtype=torch.int32, device="cuda", generator=gen
    )
    num_ctx_tokens = 5
    captured = torch.randn(
        num_ctx_tokens + num_gens * width + 4, features, device="cuda", generator=gen
    )
    spec_metadata = SimpleNamespace(
        runtime_draft_len=draft_len,
        hidden_size=0,
        captured_hidden_states=captured,
        get_hidden_states=lambda n: captured[:n],
    )

    def attn_metadata():
        return SimpleNamespace(
            num_contexts=num_ctx_reqs,
            num_seqs=batch,
            num_ctx_tokens=num_ctx_tokens,
            _seq_lens_cuda=torch.full((batch,), 99, dtype=torch.int32, device="cuda"),
            _seq_lens=torch.full((batch,), 99, dtype=torch.int32),
        )

    def prepare(fused):
        w = copy.copy(worker)
        w._ctx_len = worker._ctx_len.clone()
        w._ctx_block_tables_read = worker._ctx_block_tables_read.clone()
        pool_storage = storage.clone()
        w._ctx_kv_buf = [pool_storage[:, layer] for layer in range(layers)]
        w._ctx_kv_ptrs = torch.tensor([p.data_ptr() for p in w._ctx_kv_buf], device="cuda")
        w._fused_gen_prepare_applies = lambda *_: fused
        w._ctx_store_stream = torch.cuda.Stream() if fused and side_stream else None
        md = attn_metadata()
        inputs = w.prepare_1st_drafter_inputs(
            input_ids=None,
            position_ids=None,
            hidden_states=captured,
            accepted_tokens=accepted,
            num_accepted_tokens=num_accepted,
            attn_metadata=md,
            spec_metadata=spec_metadata,
            draft_model=drafter,
            total_target_tokens=captured.shape[0],
        )
        stored = None if inputs["block_placement"] is None else inputs["block_placement"].ctx_stored
        assert (stored is not None) == (fused and side_stream)
        if stored is not None:
            torch.cuda.current_stream().wait_event(stored)
        return inputs, w._ctx_len, md, pool_storage

    eager, eager_len, eager_md, eager_pool = prepare(False)
    fused, fused_len, fused_md, fused_pool = prepare(True)

    for key in ("noise_embedding", "query_positions", "num_ctx_per_req", "first_prev_tokens"):
        assert torch.equal(fused[key], eager[key]), key
    assert torch.equal(fused["ctx_cache_batch_idx"], eager["ctx_cache_batch_idx"])
    assert torch.equal(fused_len, eager_len)
    assert torch.equal(fused_md._seq_lens_cuda, eager_md._seq_lens_cuda)
    assert torch.equal(fused_md._seq_lens, eager_md._seq_lens)
    # Tokens whose column is clamped into a short allocation share a
    # destination, and which of them lands last is unordered in both paths;
    # every other pool entry must match bit for bit.
    writers = {}
    old_len = worker._ctx_len.tolist()
    counts = worker._ctx_block_counts.tolist()
    for g in range(num_gens):
        row = num_ctx_reqs + g
        capacity = counts[row] * page_size * cp_size
        for j in range(width):
            col = max(min(old_len[slots[row].item()] + j, capacity - 1), 0)
            if (col // page_size) % cp_size != (helix_rank or 0):
                continue  # another rank's position: left untouched
            col = (col // page_size // cp_size) * page_size + col % page_size
            page = worker._ctx_block_tables[row, col // page_size].item()
            writers.setdefault((page, col % page_size), []).append((g, j))
    shared = torch.zeros(storage.shape[0], page_size, dtype=torch.bool)
    written = torch.zeros(storage.shape[0], page_size, dtype=torch.bool)
    for (page, offset), tokens in writers.items():
        shared[page, offset] = len(tokens) > 1
        written[page, offset] = True
    entry_shape = (storage.shape[0], 1, 1, 1, page_size, 1)
    keep = ~shared.view(entry_shape).cuda()
    untouched = ~written.view(entry_shape).cuda()
    fused_bits, eager_bits = fused_pool.view(torch.int16), eager_pool.view(torch.int16)
    assert torch.equal(fused_bits * keep, eager_bits * keep)
    # Every entry outside this step's destinations keeps its contents, and
    # every destination is overwritten (a stored or a masked, zeroed latent).
    init_bits = storage.view(torch.int16)
    for bits in (fused_bits, eager_bits):
        assert torch.equal(bits * untouched, init_bits * untouched)
        changed = (bits != init_bits).any(dim=-1).flatten(1, 3).any(dim=1)
        assert bool(changed[written.cuda()].all())
    assert bool(untouched.any()) and (bool(shared.any()) or helix_rank == 1)

    rows = eager["ctx_cache_batch_idx"]
    read_rows = eager["ctx_page_table"].index_select(0, rows)
    assert torch.equal(fused["ctx_page_table"].index_select(0, rows), read_rows)
    if based:
        assert not torch.equal(read_rows, worker._ctx_block_tables.index_select(0, rows))
    fixup = _build_mla_block_fixup(
        eager["num_ctx_per_req"],
        read_rows.long(),
        draft_len,
        page_size,
        shard,
    )
    placement = fused["block_placement"]
    assert eager["block_placement"] is None
    assert torch.equal(placement.pages, fixup.pages)
    assert torch.equal(placement.offsets, fixup.offsets)
    assert torch.equal(placement.seq_lens, fixup.seq_lens_i32)
    assert torch.equal(placement.kv_bounds, fixup.kv_bounds_i32)
    if shard is None:
        assert placement.owned is None
    else:
        assert torch.equal(placement.owned, fixup.owned)


@pytest.mark.parametrize("helix_rank", [None, 1], ids=["no_helix", "helix_r1"])
def test_context_side_stream_matches_main_stream(helix_rank):
    """The step's context latents computed on the worker's side stream (beside
    the target-token selection) give the fused preparation the same outputs,
    context lengths and pool contents as computing them on the main stream."""
    draft_len, vocab, hidden, features, layers, head_dim = 7, 100, 64, 128, 3, 96
    page_size, max_blocks, num_slots, max_ctx = 16, 6, 6, 70
    num_gens, width = 4, draft_len + 1
    gen = torch.Generator(device="cuda").manual_seed(11)
    worker, storage = _paged_worker(
        draft_len, page_size, max_blocks, num_slots, layers, head_dim, max_ctx, gen
    )
    worker._ctx_dtype = torch.bfloat16
    if helix_rank is not None:
        worker._ctx_helix = HelixDraftShard(2, helix_rank, page_size)
    drafter = _LatentDrafter(vocab, hidden, features, layers, head_dim, mask_token_id=3)
    drafter.context_kv_only = False
    drafter.projects_context_beside = lambda rows: True
    drafter.project_target_hidden_beside = drafter.project_target_hidden

    slots = torch.tensor([3, 0, 2, worker._dummy_slot], device="cuda")
    worker._batch_to_slot_buffer.tensor[:num_gens] = slots
    worker._ctx_len[:4] = torch.tensor([10, 3, 33, 50], device="cuda")
    worker._ctx_block_counts[:num_gens] = torch.tensor([4, 2, 6, 1], device="cuda")
    num_accepted = torch.tensor([1, width, 3, 1], dtype=torch.int32, device="cuda")
    accepted = torch.randint(
        0, vocab, (num_gens, width), dtype=torch.int32, device="cuda", generator=gen
    )
    captured = torch.randn(num_gens * width + 3, features, device="cuda", generator=gen)
    spec_metadata = SimpleNamespace(
        runtime_draft_len=draft_len,
        hidden_size=0,
        captured_hidden_states=captured,
        get_hidden_states=lambda n: captured[:n],
    )

    def prepare(side):
        w = copy.copy(worker)
        w._ctx_len = worker._ctx_len.clone()
        pool_storage = storage.clone()
        w._ctx_kv_buf = [pool_storage[:, layer] for layer in range(layers)]
        w._ctx_kv_ptrs = torch.tensor([p.data_ptr() for p in w._ctx_kv_buf], device="cuda")
        w._fused_gen_prepare_applies = lambda *_: True
        md = SimpleNamespace(
            num_contexts=0,
            num_seqs=num_gens,
            num_ctx_tokens=0,
            _seq_lens_cuda=torch.full((num_gens,), 99, dtype=torch.int32, device="cuda"),
            _seq_lens=torch.full((num_gens,), 99, dtype=torch.int32),
        )
        w._ctx_side = (
            w._launch_context_side(drafter, spec_metadata, md, captured.shape[0]) if side else None
        )
        assert (w._ctx_side is not None) == side
        inputs = w.prepare_1st_drafter_inputs(
            input_ids=None,
            position_ids=None,
            hidden_states=captured,
            accepted_tokens=accepted,
            num_accepted_tokens=num_accepted,
            attn_metadata=md,
            spec_metadata=spec_metadata,
            draft_model=drafter,
            total_target_tokens=captured.shape[0],
        )
        assert w._ctx_side is None
        torch.cuda.synchronize()
        return inputs, w._ctx_len, md, pool_storage

    main, main_len, main_md, main_pool = prepare(False)
    side, side_len, side_md, side_pool = prepare(True)
    for key in ("noise_embedding", "query_positions", "num_ctx_per_req", "first_prev_tokens"):
        assert torch.equal(side[key], main[key]), key
    for field in ("pages", "offsets", "seq_lens", "kv_bounds"):
        assert torch.equal(
            getattr(side["block_placement"], field), getattr(main["block_placement"], field)
        ), field
    assert torch.equal(side_len, main_len)
    assert torch.equal(side_md._seq_lens_cuda, main_md._seq_lens_cuda)
    assert torch.equal(side_pool.view(torch.int16), main_pool.view(torch.int16))
    assert not torch.equal(main_pool.view(torch.int16), storage.view(torch.int16))


def test_store_context_latents_saturates_into_an_fp8_pool():
    from tensorrt_llm._torch.speculative.dflash_attention import to_draft_fp8
    from tensorrt_llm._torch.speculative.spec_tail_kernels import store_context_latents

    fp8 = torch.float8_e4m3fn
    num_tokens, num_layers, num_pages, page, latent_dim = 9, 3, 4, 32, 576
    gen = torch.Generator(device="cuda").manual_seed(5)
    latent = (torch.randn(num_tokens, num_layers, latent_dim, device="cuda", generator=gen) * 3).to(
        torch.bfloat16
    )
    # Saturating, in range, subnormal and NaN values.
    latent[0, :, :5] = torch.tensor([600.0, -1000.0, 447.0, 1e-3, float("nan")], device="cuda").to(
        torch.bfloat16
    )
    flat = torch.randperm(num_pages * page, device="cuda", generator=gen)[:num_tokens]
    pages, offsets = flat // page, flat % page
    # -1: another helix rank's position, left untouched.
    mask = torch.tensor([1, 1, 0, 1, -1, 1, 1, 0, 1], dtype=torch.int8, device="cuda")
    stored = mask >= 0

    pools = [
        torch.full((num_pages, 1, 1, page, latent_dim), 7.0, device="cuda").to(fp8)
        for _ in range(num_layers)
    ]
    expected = [p.view(torch.uint8).clone() for p in pools]
    for layer in range(num_layers):
        rows = to_draft_fp8(latent[:, layer] * (mask > 0).to(torch.bfloat16)[:, None])
        expected[layer][:, 0, 0][pages[stored], offsets[stored]] = rows[stored].view(torch.uint8)
    ptrs = torch.tensor([p.data_ptr() for p in pools], dtype=torch.int64, device="cuda")
    store_context_latents(latent, pools, ptrs, pages, offsets, mask)
    for layer in range(num_layers):
        assert torch.equal(pools[layer].view(torch.uint8), expected[layer])
    assert pools[0][pages[0], 0, 0, offsets[0], :3].float().tolist() == [448.0, -448.0, 448.0]
    assert int(pools[0].float().isnan().sum()) == 1


@pytest.mark.parametrize("rows", [7, 14, 28])
def test_q_kv_a_projection_is_one_gemm_matching_the_two(rows):
    """The K3 MLA drafter's [q_a | kv_a] projection: one GEMM (split-K where pinned) whose
    column ranges match the separate projections up to the GEMM's rounding, with the modules'
    weights re-pointed at the stacked weight."""
    from tensorrt_llm._torch.models.modeling_dspark import (
        MLADSparkForCausalLM,
        _q_kv_a_splitk_tactic,
        _stack_q_kv_a_weight,
    )
    from tensorrt_llm._torch.modules.linear import Linear

    gen = torch.Generator(device="cuda").manual_seed(rows)
    hidden, q_lora, kv = 7168, 1536, 576
    attn = SimpleNamespace(
        q_a_proj=Linear(hidden, q_lora, bias=False, dtype=torch.bfloat16).cuda(),
        kv_a_proj_with_mqa=Linear(hidden, kv, bias=False, dtype=torch.bfloat16).cuda(),
    )
    for linear in (attn.q_a_proj, attn.kv_a_proj_with_mqa):
        linear.weight.data.copy_(
            torch.randn(linear.weight.shape, device="cuda", generator=gen).to(torch.bfloat16) * 0.02
        )
    q_ref_w = attn.q_a_proj.weight.detach().clone()
    weight = _stack_q_kv_a_weight(attn)
    assert weight.shape == (q_lora + kv, hidden)
    assert attn.q_a_proj.weight.data_ptr() == weight.data_ptr()
    assert torch.equal(attn.q_a_proj.weight, q_ref_w)

    drafter = object.__new__(MLADSparkForCausalLM)
    drafter._q_kv_a_weights = [weight]
    x = torch.randn(rows, hidden, device="cuda", generator=gen).to(torch.bfloat16)
    out = drafter._q_kv_a_projection(0, x)
    ref = F.linear(x.float(), weight.float())
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
    if _q_kv_a_splitk_tactic(rows, q_lora + kv, hidden) is None:
        pytest.skip("no pinned split-K tactic on this architecture")
