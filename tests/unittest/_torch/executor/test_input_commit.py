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
"""Bit-exactness of the batched input commit and the other llprep upload kernels.

Every check compares against the operations the kernels replace: ``Tensor.copy_`` for
staged host->device copies, the overlap-scheduler gather sequence of
``PyTorchModelEngine._prepare_tp_inputs`` for the fixup, the C++
``copy_batch_block_offsets_to_device`` for the block offsets (one or two tables per
launch), and ``index_copy_`` for the spec sampler's store scatter.
"""

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.pyexecutor.input_commit import (
    InputCommit,
    active_input_commit,
    copy_batch_block_offsets_from_host,
    flush_input_commit,
    input_commit_scope,
    scatter_spec_store,
    stage_or_copy,
    stage_scatter,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _cpp_block_offsets(host, dst, copy_idx, scales, kv_offset):
    from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
        copy_batch_block_offsets_to_device,
    )

    copy_batch_block_offsets_to_device(
        host, dst, copy_idx, scales, kv_offset, torch.cuda.current_stream().cuda_stream
    )


@pytest.mark.parametrize(
    "num_pools,num_blocks,num_seqs",
    [(1, 4, 1), (2, 976, 2), (1, 15584, 2), (3, 2052, 7), (2, 1024, 64)],
)
def test_block_offsets_match_cpp_kernel(num_pools: int, num_blocks: int, num_seqs: int) -> None:
    gen = torch.Generator().manual_seed(num_pools * 1000 + num_blocks + num_seqs)
    src_max_seqs = num_seqs + 5
    dst_pools = num_pools + 1  # destination may have more pools than the source table
    dst_max_seqs = num_seqs + 3
    host = torch.randint(
        0, 1 << 20, (num_pools, src_max_seqs, 2, num_blocks), generator=gen, dtype=torch.int32
    )
    host[torch.rand(host.shape, generator=gen) < 0.2] = -1  # BAD_PAGE_INDEX
    host = host.pin_memory()
    copy_idx = torch.randperm(src_max_seqs, generator=gen)[:num_seqs].to(torch.int32).pin_memory()
    scales = torch.randint(1, 97, (num_pools,), generator=gen, dtype=torch.int32).pin_memory()
    kv_offset = torch.randint(
        0, 1 << 16, (num_pools,), generator=gen, dtype=torch.int32
    ).pin_memory()

    init = torch.randint(
        -5, 5, (dst_pools, dst_max_seqs, 2, num_blocks), generator=gen, dtype=torch.int32
    ).cuda()
    ref = init.clone()
    got = init.clone()
    # The C++ op requires equal pool counts, so its reference writes the source pools of
    # ``ref`` (a contiguous leading slice, as MiniMax-M3's copy_batch_block_offsets passes);
    # the extra destination pool keeps ``init`` and the Triton copy must leave it untouched.
    _cpp_block_offsets(host, ref[:num_pools], copy_idx, scales, kv_offset)
    assert copy_batch_block_offsets_from_host(host, got, copy_idx, scales, kv_offset)
    torch.cuda.synchronize()
    assert torch.equal(got, ref)


def test_block_offsets_wraps_like_cpp_and_falls_back_on_bad_layout() -> None:
    host = torch.full((1, 2, 2, 8), (1 << 30) + 3, dtype=torch.int32).pin_memory()
    copy_idx = torch.tensor([1, 0], dtype=torch.int32).pin_memory()
    scales = torch.tensor([7], dtype=torch.int32).pin_memory()  # int32 product wraps
    kv_offset = torch.tensor([(1 << 31) - 5], dtype=torch.int32).pin_memory()
    ref = torch.zeros((1, 2, 2, 8), dtype=torch.int32, device="cuda")
    got = torch.zeros_like(ref)
    _cpp_block_offsets(host, ref, copy_idx, scales, kv_offset)
    assert copy_batch_block_offsets_from_host(host, got, copy_idx, scales, kv_offset)
    torch.cuda.synchronize()
    assert torch.equal(got, ref)
    # Pageable host table or mismatched block width: not handled, caller uses the C++ op.
    assert not copy_batch_block_offsets_from_host(host.clone(), got, copy_idx, scales, kv_offset)
    assert not copy_batch_block_offsets_from_host(host, got[..., :4], copy_idx, scales, kv_offset)
    # Large copies are bandwidth bound; left to the C++ kernel.
    big = torch.zeros((1, 2, 2, 600_000), dtype=torch.int32).pin_memory()
    big_dst = torch.zeros((1, 2, 2, 600_000), dtype=torch.int32, device="cuda")
    assert not copy_batch_block_offsets_from_host(big, big_dst, copy_idx, scales, kv_offset)


def _random_values(dtype: torch.dtype, n: int, rng: random.Random):
    if dtype == torch.bool:
        return [rng.random() < 0.5 for _ in range(n)]
    if dtype == torch.float32:
        return [rng.uniform(-3.0, 3.0) for _ in range(n)]
    if dtype == torch.int64:
        return [rng.randint(-(1 << 40), 1 << 40) for _ in range(n)]
    return [rng.randint(-(1 << 30), 1 << 30) for _ in range(n)]


@pytest.mark.parametrize("capacity", [1 << 20, 64])
def test_staged_copies_match_direct_copies(capacity: int) -> None:
    rng = random.Random(capacity)
    commit = InputCommit(capacity=capacity)
    dtypes = [torch.int32, torch.int64, torch.bool, torch.float32, torch.int32]
    sizes = [1, 3, 14, 112, 129, 700, 2049]
    for step in range(6):
        dsts, refs = [], []
        commit.begin()
        for i, size in enumerate(sizes):
            dtype = dtypes[(i + step) % len(dtypes)]
            buf = torch.full((size + 9,), 7, dtype=dtype, device="cuda")
            ref = buf.clone()
            values = _random_values(dtype, size, rng)
            off = (i + step) % 5
            if i % 2:
                src = torch.tensor(values, dtype=dtype)
                commit.stage(buf[off : off + size], src)
            else:
                commit.stage(buf[off : off + size], values)
            ref[off : off + size].copy_(torch.tensor(values, dtype=dtype))
            dsts.append(buf)
            refs.append(ref)
        commit.commit()
        torch.cuda.synchronize()
        for buf, ref in zip(dsts, refs):
            assert torch.equal(buf, ref)


@pytest.mark.parametrize("capacity", [1 << 20, 64])
def test_restaged_destinations_keep_the_last_value(capacity: int) -> None:
    """Bytes staged more than once in one commit end with the value staged last, as the
    same direct copies leave them: exact repeats (a slot reset then restored), partial and
    nested overlaps, next to untouched entries. The kernel writes entries in parallel, so
    without the merge the repeated writes race."""
    rng = random.Random(capacity)
    commit = InputCommit(capacity=capacity)
    for step in range(4):
        lens = torch.full((64,), -1, dtype=torch.int64, device="cuda")
        table = torch.zeros(300, dtype=torch.int32, device="cuda")
        other = torch.zeros(40, dtype=torch.int32, device="cuda")
        ref_lens, ref_table, ref_other = lens.clone(), table.clone(), other.clone()
        commit.begin()
        with input_commit_scope(commit):
            # One slot staged 48 times: the reset to 0 and the restored length, repeated.
            for k in range(48):
                value = 0 if k % 2 == 0 else 1000 + 10 * step + k
                assert stage_scatter(lens, [5, 9 + k % 3], [value, k])
                ref_lens[5] = value
                ref_lens[9 + k % 3] = k
            spans = [(0, 120), (100, 50), (110, 10), (260, 40), (0, 4), (200, 1), (199, 3)]
            for start, size in spans:
                values = [rng.randint(-(1 << 30), 1 << 30) for _ in range(size)]
                src = torch.tensor(values, dtype=torch.int32)
                stage_or_copy(table[start : start + size], src)
                ref_table[start : start + size] = src.cuda()
            values = [rng.randint(0, 99) for _ in range(40)]
            stage_or_copy(other, torch.tensor(values, dtype=torch.int32))
            ref_other.copy_(torch.tensor(values, dtype=torch.int32))
        merged_before = commit.num_merged_runs
        commit.commit()
        # lens[5], lens[9], lens[10], lens[11], table[0:150], table[199:202]; table[260:300]
        # and other are staged once.
        assert commit.num_merged_runs - merged_before == 6
        torch.cuda.synchronize()
        assert torch.equal(lens, ref_lens)
        assert torch.equal(table, ref_table)
        assert torch.equal(other, ref_other)


def _reference_fixup(
    new_tokens,
    lens,
    next_draft,
    slots,
    tokens,
    input_ids,
    ids_start,
    draft_tokens,
    draft_start,
    pos_off,
    pos_start,
    kv_off,
    kv_start,
):
    """The device sequence of _prepare_tp_inputs for requests with a previous step."""
    pos_off *= 0
    kv_off *= 0
    width = tokens - 1
    if not slots:
        return
    previous_slots = torch.tensor(slots, dtype=torch.int, device="cuda")
    num_prev = len(slots)
    flat = new_tokens.transpose(0, 1)[previous_slots, :tokens].flatten()
    input_ids[ids_start : ids_start + num_prev * tokens].copy_(flat)
    if width > 0:
        draft_tokens[draft_start : draft_start + num_prev * width].copy_(
            next_draft[previous_slots, :width].flatten()
        )
    kv_len_offsets = lens - tokens
    pos_indices = torch.tensor(
        [s for s in slots for _ in range(tokens)], dtype=torch.int, device="cuda"
    )
    pos_off[pos_start : pos_start + num_prev * tokens].copy_(lens[pos_indices])
    kv_off[kv_start : kv_start + num_prev].copy_(kv_len_offsets[previous_slots])


@pytest.mark.parametrize("tokens", [1, 2, 7, 8])
@pytest.mark.parametrize("num_prev,num_new", [(0, 2), (1, 0), (2, 0), (2, 1), (37, 3), (256, 0)])
def test_overlap_fixup_matches_gather_sequence(tokens: int, num_prev: int, num_new: int) -> None:
    rng = random.Random(tokens * 1000 + num_prev * 10 + num_new)
    num_slots = num_prev + num_new + 5
    store_tokens = tokens + 1
    store_width = max(tokens, 2)
    new_tokens = torch.randint(0, 160000, (store_tokens, num_slots, 1), dtype=torch.int32).cuda()
    lens = torch.randint(1, tokens + 1, (num_slots,), dtype=torch.int32).cuda()
    next_draft = torch.randint(0, 160000, (num_slots, store_width), dtype=torch.int32).cuda()
    slots = rng.sample(range(num_slots), num_prev)
    max_tokens = (num_prev + num_new) * tokens + 11
    ids_start = num_new * tokens
    draft_start = num_new * (tokens - 1)
    num_extend = num_prev + num_new
    pos_start = (num_extend - num_prev) * tokens
    kv_start = num_extend - num_prev

    def buffers():
        g = torch.Generator().manual_seed(5)
        return [
            torch.randint(-9, 9, (n,), generator=g, dtype=torch.int32).cuda()
            for n in (max_tokens, max_tokens, max_tokens, num_extend + 3)
        ]

    ids_r, draft_r, pos_r, kv_r = buffers()
    _reference_fixup(
        new_tokens,
        lens,
        next_draft,
        slots,
        tokens,
        ids_r,
        ids_start,
        draft_r,
        draft_start,
        pos_r,
        pos_start,
        kv_r,
        kv_start,
    )
    ids_c, draft_c, pos_c, kv_c = buffers()
    commit = InputCommit()
    commit.begin()
    commit.set_overlap_fixup(
        previous_slots=slots,
        tokens_per_req=tokens,
        draft_width=tokens - 1,
        new_tokens=new_tokens,
        new_tokens_lens=lens,
        next_draft_tokens=next_draft,
        input_ids=ids_c,
        input_ids_start=ids_start,
        draft_tokens=draft_c,
        draft_tokens_start=draft_start,
        pos_offsets=pos_c,
        pos_offsets_start=pos_start,
        kv_offsets=kv_c,
        kv_offsets_start=kv_start,
    )
    commit.commit()
    torch.cuda.synchronize()
    for got, ref in ((ids_c, ids_r), (draft_c, draft_r), (pos_c, pos_r), (kv_c, kv_r)):
        assert torch.equal(got, ref)


def test_arena_reuse_waits_and_scope_restores() -> None:
    commit = InputCommit(capacity=256)
    out = torch.zeros(4096, dtype=torch.int64, device="cuda")
    for step in range(8):
        commit.begin()
        with input_commit_scope(commit):
            assert active_input_commit() is commit
            # Grows past the 256-byte arena on the first steps.
            commit.stage(out, np.arange(step, step + out.numel(), dtype=np.int64))
        assert active_input_commit() is None
        commit.commit()
    torch.cuda.synchronize()
    assert torch.equal(out.cpu(), torch.arange(7, 7 + out.numel(), dtype=torch.int64))
    commit.begin()
    commit.abort()
    assert not commit.is_open


def test_stage_or_copy_and_flush_order_like_direct_copies() -> None:
    """Staged copies land at the commit (or at a flush) in the order a device reader between
    them sees with direct copies; outside a scope (or for a source the arena cannot take) the
    helper copies directly."""
    gen = torch.Generator().manual_seed(3)

    def host(n, dtype):
        return torch.randint(-1000, 1000, (n,), generator=gen).to(dtype).pin_memory()

    sources = [host(37, torch.int32), host(36, torch.int64), host(5, torch.float32)]
    for staged in (False, True):
        commit = InputCommit(capacity=64, num_arenas=3) if staged else None
        a = torch.zeros(37, dtype=torch.int32, device="cuda")
        b = torch.zeros(4, 9, dtype=torch.int64, device="cuda")
        c = torch.zeros(5, dtype=torch.float32, device="cuda")
        derived = []
        src = [t.clone().pin_memory() for t in sources]
        for step in range(5):
            if commit is not None:
                commit.begin()
            with input_commit_scope(commit):
                stage_or_copy(a, src[0])
                stage_or_copy(b.view(-1)[:36].view(4, 9), src[1].view(4, 9))
                # A device reader of a staged buffer: flushed first.
                flush_input_commit()
                derived.append((a.long().sum() + b.sum()).item())
                stage_or_copy(c, src[2])
                # A strided destination is copied directly, in stream order.
                b[:, ::2].copy_(torch.full((4, 5), step, dtype=torch.int64), non_blocking=True)
            if commit is not None:
                commit.commit()
            # The direct copies read the pinned sources asynchronously.
            torch.cuda.synchronize()
            for t in src:
                t.add_(1)
        torch.cuda.synchronize()
        if not staged:
            reference = (a.clone(), b.clone(), c.clone(), derived)
        else:
            assert torch.equal(a, reference[0])
            assert torch.equal(b, reference[1])
            assert torch.equal(c, reference[2])
            assert derived == reference[3]
    # No open commit: flush is a no-op and stage_or_copy copies.
    flush_input_commit()
    out = torch.zeros(3, dtype=torch.int32, device="cuda")
    stage_or_copy(out, torch.tensor([1, 2, 3], dtype=torch.int32))
    torch.cuda.synchronize()
    assert out.tolist() == [1, 2, 3]


@pytest.mark.parametrize("dtype", [torch.int64, torch.int32, torch.float32])
def test_stage_scatter_matches_index_copy(dtype) -> None:
    rng = random.Random(7)
    ref = torch.arange(97, device="cuda").to(dtype)
    got = ref.clone()
    commit = InputCommit(capacity=128)
    for _ in range(4):
        slots = rng.sample(range(97), 11)
        values = [rng.randint(-50, 50) for _ in slots]
        ref.index_copy_(
            0,
            torch.tensor(slots, device="cuda"),
            torch.tensor(values, dtype=dtype, device="cuda"),
        )
        assert not stage_scatter(got, slots, values), "staged without an open commit"
        commit.begin()
        with input_commit_scope(commit):
            assert stage_scatter(got, slots, values)
            # 2-D targets are not staged.
            assert not stage_scatter(got.view(1, -1), [0], [1])
        commit.commit()
    torch.cuda.synchronize()
    assert torch.equal(got, ref)


@pytest.mark.parametrize("n,num_slots", [(1, 3), (2, 3), (4, 9), (7, 64)])
@pytest.mark.parametrize("accepted,draft", [(7, 6), (8, 7), (3, 2)])
def test_spec_store_scatter_matches_index_copy(n, num_slots, accepted, draft) -> None:
    gen = torch.Generator().manual_seed(n * 100 + num_slots + accepted)

    def store():
        return SimpleNamespace(
            new_tokens=torch.randint(
                0, 99, (accepted, num_slots, 1), generator=gen, dtype=torch.int32
            ).cuda(),
            next_new_tokens=torch.randint(
                0, 99, (draft + 1, num_slots, 1), generator=gen, dtype=torch.int32
            ).cuda(),
            next_draft_tokens=torch.randint(
                0, 99, (num_slots, draft), generator=gen, dtype=torch.int32
            ).cuda(),
            new_tokens_lens=torch.randint(
                0, 99, (num_slots,), generator=gen, dtype=torch.int32
            ).cuda(),
        )

    ref, got = store(), None
    got = SimpleNamespace(**{k: v.clone() for k, v in vars(ref).items()})
    # Worker outputs are wider row-major buffers sliced to the batch, as in sample_async.
    skip = 1
    out_new = torch.randint(
        0, 1 << 30, (n + 2, accepted + 3), generator=gen, dtype=torch.int32
    ).cuda()
    out_next = torch.randint(
        0, 1 << 30, (n + 2, draft + 1), generator=gen, dtype=torch.int32
    ).cuda()
    out_draft = torch.randint(
        0, 1 << 30, (n + 2, draft + 2), generator=gen, dtype=torch.int32
    ).cuda()
    out_lens = torch.randint(0, 8, (n + 2,), generator=gen, dtype=torch.int32).cuda()
    o_new = out_new[skip : skip + n, :accepted]
    o_next = out_next[skip : skip + n]
    o_draft = out_draft[skip : skip + n, :draft]
    o_lens = out_lens[skip : skip + n]
    slots = torch.randperm(num_slots, generator=gen)[:n].cuda()

    ref.new_tokens.squeeze(-1).T.index_copy_(0, slots, o_new)
    ref.next_new_tokens.squeeze(-1).T.index_copy_(0, slots, o_next)
    ref.new_tokens_lens.index_copy_(0, slots, o_lens)
    ref.next_draft_tokens.index_copy_(0, slots, o_draft)
    assert scatter_spec_store(
        slots,
        got.new_tokens.squeeze(-1),
        got.next_new_tokens.squeeze(-1),
        got.new_tokens_lens,
        got.next_draft_tokens,
        o_new,
        o_next,
        o_lens,
        o_draft,
    )
    torch.cuda.synchronize()
    for key in vars(ref):
        assert torch.equal(getattr(got, key), getattr(ref, key)), key
    # A source of another width does not match the store layout: the caller uses index_copy_.
    assert not scatter_spec_store(
        slots,
        got.new_tokens.squeeze(-1),
        got.next_new_tokens.squeeze(-1),
        got.new_tokens_lens,
        got.next_draft_tokens,
        out_new[skip : skip + n],
        o_next,
        o_lens,
        o_draft,
    )
