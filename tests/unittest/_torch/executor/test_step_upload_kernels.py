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
"""Bit-exactness of the decode-step upload kernels.

Every check compares against the operations the kernels replace: the C++
``copy_batch_block_offsets_to_device`` for the block offsets, ``index_copy_`` for the
speculative sampler's store scatter and ``Tensor.copy_`` / ``DeviceCopyBuffer.update`` for
the state-index uploads.
"""

import random
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._device_copy import DeviceCopyBuffer
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import mamba_cache_manager_v2
from tensorrt_llm._torch.pyexecutor.step_upload_kernels import (
    copy_batch_block_offsets_from_host,
    copy_small_pinned_pair,
    scatter_spec_store,
)
from tensorrt_llm._torch.speculative.spec_sampler_base import SpecSampler

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _cpp_block_offsets(host, dst, copy_idx, scales, kv_offset):
    from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
        copy_batch_block_offsets_to_device,
    )

    copy_batch_block_offsets_to_device(
        host, dst, copy_idx, scales, kv_offset, torch.cuda.current_stream().cuda_stream
    )


@pytest.mark.parametrize("side_stream", [False, True])
@pytest.mark.parametrize(
    "num_pools,num_blocks,num_seqs",
    [(1, 4, 1), (2, 976, 2), (1, 15584, 2), (3, 2052, 7), (2, 1024, 64)],
)
def test_block_offsets_match_cpp_kernel(
    num_pools: int, num_blocks: int, num_seqs: int, side_stream: bool
) -> None:
    gen = torch.Generator().manual_seed(num_pools * 1000 + num_blocks + num_seqs)
    src_max_seqs = num_seqs + 5
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
        -5, 5, (num_pools + 1, dst_max_seqs, 2, num_blocks), generator=gen, dtype=torch.int32
    ).cuda()
    # The C++ kernel takes as many destination pools as the host table has.
    ref = init[:num_pools].clone()
    _cpp_block_offsets(host, ref, copy_idx, scales, kv_offset)
    # The manager passes its execution stream, which need not be the current one.
    stream = torch.cuda.Stream() if side_stream else None
    # The Triton kernel also takes a destination with more pools and leaves the extra ones as
    # they were.
    for dst_pools in (num_pools, num_pools + 1):
        got = init[:dst_pools].clone()
        if stream is not None:
            stream.wait_stream(torch.cuda.current_stream())
        assert copy_batch_block_offsets_from_host(host, got, copy_idx, scales, kv_offset, stream)
        torch.cuda.synchronize()
        assert torch.equal(got[:num_pools], ref)
        assert torch.equal(got[num_pools:], init[num_pools:dst_pools])


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


def test_block_offsets_empty_batch_launches_nothing() -> None:
    host = torch.ones((1, 2, 2, 8), dtype=torch.int32).pin_memory()
    # The manager's copy index is a view of its pinned index-mapper buffer.
    copy_idx = torch.zeros((2,), dtype=torch.int32).pin_memory()[:0]
    ones = torch.ones(1, dtype=torch.int32).pin_memory()
    dst = torch.full((1, 2, 2, 8), 9, dtype=torch.int32, device="cuda")
    assert copy_batch_block_offsets_from_host(host, dst, copy_idx, ones, ones)
    torch.cuda.synchronize()
    assert bool((dst == 9).all())


def _random_store(gen, num_slots, accepted, draft):
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
        new_tokens_lens=torch.randint(0, 99, (num_slots,), generator=gen, dtype=torch.int32).cuda(),
    )


@pytest.mark.parametrize("n,num_slots", [(1, 3), (2, 3), (4, 9), (7, 64)])
@pytest.mark.parametrize("accepted,draft", [(7, 6), (8, 7), (3, 2)])
def test_spec_store_scatter_matches_index_copy(n, num_slots, accepted, draft) -> None:
    gen = torch.Generator().manual_seed(n * 100 + num_slots + accepted)
    ref = _random_store(gen, num_slots, accepted, draft)
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


def test_spec_sampler_fused_scatter_matches_index_copy() -> None:
    """SpecSampler._scatter_to_stores with the switch on and off, over a changing batch.

    The worker's output widths track the runtime draft length (narrower than the stores
    here), and the cached device slot tensor is reused only while the slots repeat.
    """
    num_slots, accepted, draft = 16, 6, 5
    gen = torch.Generator().manual_seed(3)
    init = _random_store(gen, num_slots, accepted, draft)
    samplers = []
    for fused in (False, True):
        sampler = SimpleNamespace(
            store=SimpleNamespace(**{k: v.clone() for k, v in vars(init).items()}),
            _fused_scatter=fused,
            _fused_scatter_slots=None,
            _fused_scatter_slots_device=None,
        )
        samplers.append(sampler)
    for step, slots in enumerate([[3, 1, 7], [3, 1, 7], [3, 7], [0, 3, 7, 12], [0, 3, 7, 12]]):
        num_skip = step % 2
        rows = num_skip + len(slots)
        runtime_draft = draft - (step % 3)
        outputs = {
            "new_tokens": torch.randint(
                0, 1 << 20, (rows, runtime_draft + 1), generator=gen, dtype=torch.int32
            ).cuda(),
            "next_new_tokens": torch.randint(
                0, 1 << 20, (rows, runtime_draft + 1), generator=gen, dtype=torch.int32
            ).cuda(),
            "next_draft_tokens": torch.randint(
                0, 1 << 20, (rows, runtime_draft), generator=gen, dtype=torch.int32
            ).cuda(),
            "new_tokens_lens": torch.randint(
                0, 6, (rows,), generator=gen, dtype=torch.int32
            ).cuda(),
        }
        for sampler in samplers:
            SpecSampler._scatter_to_stores(sampler, outputs, num_skip, slots)
        torch.cuda.synchronize()
        eager, fused = samplers
        for key in vars(eager.store):
            assert torch.equal(getattr(fused.store, key), getattr(eager.store, key)), (step, key)
        assert fused._fused_scatter_slots == slots
        assert eager._fused_scatter_slots is None


def _random_values(dtype: torch.dtype, n: int, rng: random.Random):
    if dtype == torch.bool:
        return [rng.random() < 0.5 for _ in range(n)]
    return [rng.randrange(-(1 << 30), 1 << 30) for _ in range(n)]


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


def _state_index_manager(capacity: int):
    manager = object.__new__(mamba_cache_manager_v2.MambaHybridCacheManagerV2)
    manager.__dict__.update(
        _state_indices_buffer=DeviceCopyBuffer(
            torch.full((capacity,), -7, dtype=torch.int32, device="cuda"), pin_memory=True
        ),
        _host_state_indices=torch.zeros(capacity, dtype=torch.int32).pin_memory(),
        _dummy_request_mask_buffer=DeviceCopyBuffer(
            torch.ones(capacity, dtype=torch.bool, device="cuda"), pin_memory=True
        ),
        _dummy_request_mask_host=torch.ones(capacity, dtype=torch.bool).pin_memory(),
    )
    return manager


def test_state_index_upload_kernel_matches_copies(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _state_index_manager(4)
    manager._host_state_indices[:3] = torch.tensor([11, 4, 0], dtype=torch.int32)
    monkeypatch.setattr(mamba_cache_manager_v2, "state_index_copy_kernel_enabled", lambda: False)
    assert not manager._upload_state_indices_and_dummy_mask([False, True])
    monkeypatch.setattr(mamba_cache_manager_v2, "state_index_copy_kernel_enabled", lambda: True)
    assert manager._upload_state_indices_and_dummy_mask([False, True])
    torch.cuda.synchronize()
    assert manager.cuda_state_indices.tolist() == [11, 4, 0, 0]
    assert manager._dummy_request_mask.tolist() == [False, True, False, False]

    # Unchanged values: nothing is written, as DeviceCopyBuffer.update would skip both.
    manager.cuda_state_indices.fill_(-7)
    manager._dummy_request_mask.fill_(True)
    assert manager._upload_state_indices_and_dummy_mask([False, True])
    torch.cuda.synchronize()
    assert manager.cuda_state_indices.tolist() == [-7] * 4
    assert manager._dummy_request_mask.tolist() == [True] * 4

    # A change in either buffer writes both; later DeviceCopyBuffer updates see the values.
    manager._host_state_indices[:3] = torch.tensor([11, 4, 9], dtype=torch.int32)
    assert manager._upload_state_indices_and_dummy_mask([False, True, False])
    torch.cuda.synchronize()
    assert manager.cuda_state_indices.tolist() == [11, 4, 9, 0]
    assert manager._dummy_request_mask.tolist() == [False, True, False, False]
    assert manager._state_indices_buffer.holds(manager._host_state_indices)
    assert manager._dummy_request_mask_buffer.holds(manager._dummy_request_mask_host)


def test_state_index_upload_falls_back_for_large_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mamba_cache_manager_v2, "state_index_copy_kernel_enabled", lambda: True)
    manager = _state_index_manager(2048)
    assert not manager._upload_state_indices_and_dummy_mask([False])
