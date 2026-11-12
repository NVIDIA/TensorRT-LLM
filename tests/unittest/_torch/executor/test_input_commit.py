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
"""Bit-exactness of the batched input commit and the other host-upload kernels.

Every check compares against the operations the kernels replace: ``Tensor.copy_`` for
staged host->device copies and the small pinned-copy kernel, the overlap-scheduler gather
sequence of ``PyTorchModelEngine._prepare_tp_inputs`` for the fixup, the C++
``copy_batch_block_offsets_to_device`` for the block offsets (one or two tables per
launch), and ``index_copy_`` for the spec sampler's store scatter.
"""

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.pyexecutor.input_commit import (
    InputCommit,
    active_input_commit,
    block_offsets_copy_job,
    copy_batch_block_offsets_from_host,
    copy_block_offsets_jobs,
    copy_small_pinned_pair,
    input_commit_scope,
    scatter_spec_store,
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
    _cpp_block_offsets(host, ref, copy_idx, scales, kv_offset)
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


def _block_offsets_case(seed: int, num_pools: int, num_blocks: int, num_seqs: int):
    gen = torch.Generator().manual_seed(seed)
    src_max_seqs = num_seqs + 3
    host = torch.randint(
        0, 1 << 20, (num_pools, src_max_seqs, 2, num_blocks), generator=gen, dtype=torch.int32
    )
    host[torch.rand(host.shape, generator=gen) < 0.2] = -1
    # A prefix of a pinned buffer, as IndexMapper.get_copy_index returns it.
    copy_idx = torch.randperm(src_max_seqs, generator=gen).to(torch.int32).pin_memory()[:num_seqs]
    scales = torch.randint(1, 97, (num_pools,), generator=gen, dtype=torch.int32)
    kv_offset = torch.randint(0, 1 << 16, (num_pools,), generator=gen, dtype=torch.int32)
    dst = torch.randint(
        -5, 5, (num_pools + 1, num_seqs + 2, 2, num_blocks), generator=gen, dtype=torch.int32
    ).cuda()
    return [host.pin_memory(), dst, copy_idx, scales.pin_memory(), kv_offset.pin_memory()]


@pytest.mark.parametrize(
    "shapes",
    [
        [(2, 976, 2), (1, 4096, 2)],  # target + draft tables
        [(1, 15584, 1), (2, 64, 1)],
        [(2, 976, 4), (1, 1024, 0), (3, 100, 3)],  # an empty job, then an odd job out
    ],
)
def test_block_offsets_jobs_match_separate_copies(shapes) -> None:
    cases = [_block_offsets_case(i + 11, *shape) for i, shape in enumerate(shapes)]
    refs = []
    for host, dst, copy_idx, scales, kv_offset in cases:
        ref = dst.clone()
        if copy_idx.numel():  # nothing to copy for an empty batch
            _cpp_block_offsets(host, ref, copy_idx, scales, kv_offset)
        refs.append(ref)
    jobs = [block_offsets_copy_job(*case) for case in cases]
    assert all(job is not None for job in jobs)
    copy_block_offsets_jobs(jobs)
    torch.cuda.synchronize()
    for case, ref in zip(cases, refs):
        assert torch.equal(case[1], ref)


def test_attention_metadata_pairs_target_and_draft_block_offsets() -> None:
    target = _block_offsets_case(1, 2, 976, 2)
    draft = _block_offsets_case(2, 1, 512, 2)
    refs = []
    for host, dst, copy_idx, scales, kv_offset in (target, draft):
        ref = dst.clone()
        _cpp_block_offsets(host, ref, copy_idx, scales, kv_offset)
        refs.append(ref)

    def manager(case):
        def make_job(dst, request_ids, beam_width, num_contexts, num_seqs):
            assert dst is case[1] and num_seqs == case[2].shape[0]
            return block_offsets_copy_job(*case)

        return SimpleNamespace(block_offsets_copy_job=make_job)

    class _Metadata(TrtllmAttentionMetadata):
        num_seqs = 2  # a plain value instead of the seq_lens-derived property

    md = object.__new__(_Metadata)
    md.__dict__.update(
        kv_cache_manager=manager(target),
        draft_kv_cache_manager=manager(draft),
        kv_cache_block_offsets=target[1],
        draft_kv_cache_block_offsets=draft[1],
        request_ids=[3, 4],
        beam_width=1,
        num_contexts=0,
    )
    assert md._copy_target_and_draft_block_offsets()
    torch.cuda.synchronize()
    assert torch.equal(target[1], refs[0]) and torch.equal(draft[1], refs[1])
    # A manager without the Triton job (e.g. a V1 draft cache) keeps the two copies.
    md.draft_kv_cache_manager = SimpleNamespace()
    assert not md._copy_target_and_draft_block_offsets()


@pytest.mark.parametrize(
    "dtype0,n0,dtype1,n1",
    [
        (torch.int32, 3, torch.bool, 3),
        (torch.int64, 17, torch.int32, 1),
        (torch.int32, 1024, torch.bool, 7),
    ],
)
def test_small_pinned_pair_copy_matches_copy(dtype0, n0, dtype1, n1) -> None:
    rng = random.Random(n0 * 31 + n1)
    srcs = [
        torch.tensor(_random_values(dtype, n, rng), dtype=dtype).pin_memory()
        for dtype, n in ((dtype0, n0), (dtype1, n1))
    ]
    dsts = [torch.zeros(src.shape, dtype=src.dtype, device="cuda") for src in srcs]
    assert copy_small_pinned_pair(dsts[0], srcs[0], dsts[1], srcs[1])
    torch.cuda.synchronize()
    for dst, src in zip(dsts, srcs):
        assert torch.equal(dst.cpu(), src)
    # Pageable source, size mismatch or a large copy: not handled, the caller copies.
    assert not copy_small_pinned_pair(dsts[0], srcs[0].clone(), dsts[1], srcs[1])
    assert not copy_small_pinned_pair(dsts[0][:1], srcs[0], dsts[1], srcs[1])
    big = torch.zeros(2048, dtype=torch.int32).pin_memory()
    assert not copy_small_pinned_pair(big.cuda(), big, dsts[1], srcs[1])


def test_state_index_upload_kernel_matches_copies(monkeypatch: pytest.MonkeyPatch) -> None:
    from tensorrt_llm._torch.pyexecutor import mamba_cache_manager as mcm

    manager = object.__new__(mcm.MambaHybridCacheManagerV2)
    manager.__dict__.update(
        cuda_state_indices=torch.full((3,), -7, dtype=torch.int32, device="cuda"),
        _host_state_indices=torch.tensor([11, 4, 0], dtype=torch.int32).pin_memory(),
        _dummy_request_mask=torch.ones(3, dtype=torch.bool, device="cuda"),
        _dummy_request_mask_host=torch.ones(3, dtype=torch.bool).pin_memory(),
    )
    monkeypatch.setattr(mcm, "gen_host_bundle_kernels_enabled", lambda: False)
    assert not manager._upload_state_indices_and_dummy_mask([False, True])
    monkeypatch.setattr(mcm, "gen_host_bundle_kernels_enabled", lambda: True)
    assert manager._upload_state_indices_and_dummy_mask([False, True])
    torch.cuda.synchronize()
    assert manager.cuda_state_indices.tolist() == [11, 4, 0]
    assert manager._dummy_request_mask.tolist() == [False, True, False]
    # Inside an input commit the uploads are staged by the copy path instead.
    with input_commit_scope(SimpleNamespace()):
        assert not manager._upload_state_indices_and_dummy_mask([False])


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
