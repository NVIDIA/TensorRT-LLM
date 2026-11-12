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
"""Race guards of the batched input commit and the pinned-source kernels.

The kernels read pinned host memory when they RUN, not when they are launched, so every host
buffer needs a lifetime rule: the commit arenas alternate and are guarded by an event, the state
index upload reads fresh pinned snapshots that are held until an event, the paired block-offset
copy needs one copy-index buffer per manager. A consumer kernel that has not started yet is
the dangerous case. Each test delays the consumer with ``torch.cuda._sleep`` queued ahead of it,
so that a missing guard shows as a wrong destination instead of a lucky pass; the negative
controls prove the delay really provokes the race.
"""

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.pyexecutor import input_commit as ic
from tensorrt_llm._torch.pyexecutor import mamba_cache_manager as mcm
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.speculative.interface import SpecMetadata

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# A long delay on the current stream: long compared with the host work of a few commits, short
# enough for a unit test.
DELAY_CYCLES = 400_000_000
N = 256


def _zeros(n: int = N) -> torch.Tensor:
    return torch.zeros(n, dtype=torch.int32, device="cuda")


def _full(value: int, n: int = N) -> np.ndarray:
    return np.full(n, value, dtype=np.int32)


def _new_commit(capacity: int) -> ic.InputCommit:
    # Compile the commit kernel first: a JIT compile between the delay and the commit would let the
    # delay run out on the GPU before the kernel is launched and hide the race.
    commit = ic.InputCommit(capacity=capacity)
    ic.warmup_input_commit_kernels(commit)
    return commit


def _step(commit: ic.InputCommit, dst: torch.Tensor, value: int, delay: bool = False) -> None:
    commit.begin()
    commit.stage(dst, _full(value, dst.numel()))
    if delay:
        torch.cuda._sleep(DELAY_CYCLES)  # the commit kernel queues behind this
    commit.commit()


def test_arena_is_not_overwritten_while_its_commit_kernel_is_pending() -> None:
    commit = _new_commit(4096)
    out = [_zeros(), _zeros(), _zeros()]
    _step(commit, out[0], 11, delay=True)  # arena A, kernel pending
    _step(commit, out[1], 22)  # arena B
    # Arena A again: begin() must wait for the pending kernel of step 1 before the staging below
    # overwrites the descriptors and values that kernel is about to read.
    _step(commit, out[2], 33)
    torch.cuda.synchronize()
    assert [int(t[0]) for t in out] == [11, 22, 33]
    assert all(bool((t == v).all()) for t, v in zip(out, (11, 22, 33)))


def test_negative_control_without_the_arena_guard_the_race_corrupts() -> None:
    commit = _new_commit(4096)
    out = [_zeros(), _zeros(), _zeros()]
    _step(commit, out[0], 11, delay=True)
    _step(commit, out[1], 22)
    commit._events = [None, None]  # what begin() would see without the arena guard
    _step(commit, out[2], 33)
    torch.cuda.synchronize()
    # Step 1's kernel read step 3's descriptors: its own destination was never written.
    assert not bool((out[0] == 11).all())


def test_grow_replaces_only_the_arena_of_the_open_step() -> None:
    commit = _new_commit(256)
    out = [_zeros(32), _zeros(4096), _zeros(32), _zeros(8192)]
    _step(commit, out[0], 5, delay=True)  # small: stays in arena A, kernel pending
    _step(commit, out[1], 6)  # grows arena B while A's kernel still reads A
    _step(commit, out[2], 7)  # arena A once more, after the guard
    _step(commit, out[3], 8)  # grows arena B again
    torch.cuda.synchronize()
    for dst, value in zip(out, (5, 6, 7, 8)):
        assert bool((dst == value).all()), value


def test_grow_rebases_the_staged_host_addresses_of_the_open_step() -> None:
    commit = _new_commit(64)
    out = [_zeros(8), _zeros(2048), _zeros(8)]
    commit.begin()
    commit.stage(out[0], _full(1, 8))  # recorded before the arena is replaced
    commit.stage(out[1], np.arange(2048, dtype=np.int32))  # forces the growth
    commit.stage(out[2], _full(3, 8))
    commit.commit()
    torch.cuda.synchronize()
    assert bool((out[0] == 1).all()) and bool((out[2] == 3).all())
    assert torch.equal(out[1].cpu(), torch.arange(2048, dtype=torch.int32))


def test_abort_after_partial_staging_leaves_no_trace() -> None:
    commit = _new_commit(4096)
    aborted, kept, fix_ids = _zeros(), _zeros(), _zeros(16)
    commit.begin()
    commit.stage(aborted, _full(9))
    commit.set_overlap_fixup(
        previous_slots=[0],
        tokens_per_req=2,
        draft_width=1,
        new_tokens=torch.full((2, 1, 1), 77, dtype=torch.int32, device="cuda"),
        new_tokens_lens=torch.ones(1, dtype=torch.int32, device="cuda"),
        next_draft_tokens=_zeros(1).reshape(1, 1),
        input_ids=fix_ids,
        input_ids_start=0,
        draft_tokens=_zeros(16),
        draft_tokens_start=0,
        pos_offsets=_zeros(2),
        pos_offsets_start=0,
        kv_offsets=_zeros(1),
        kv_offsets_start=0,
    )
    commit.abort()
    assert not commit.is_open
    _step(commit, kept, 4)
    torch.cuda.synchronize()
    assert bool((kept == 4).all())
    assert not bool(aborted.any())  # the aborted copy was never issued
    assert not bool(fix_ids.any())  # nor the stale overlap gather of the aborted step
    # begin() works after an abort and the arena indexing is intact.
    _step(commit, aborted, 6)
    torch.cuda.synchronize()
    assert bool((aborted == 6).all())


def test_abort_runs_the_rollbacks_and_a_commit_drops_them() -> None:
    commit = _new_commit(4096)
    dst = _zeros()
    ran = []
    commit.begin()
    commit.stage(dst, _full(5))
    commit.on_abort(lambda: ran.append("aborted step"))
    commit.abort()
    assert ran == ["aborted step"]
    torch.cuda.synchronize()
    assert not bool(dst.any())  # the staged copy never happened
    # A committed step runs no rollback, and a rollback of one step does not leak into the next.
    commit.begin()
    commit.stage(dst, _full(6))
    commit.on_abort(lambda: ran.append("committed step"))
    commit.commit()
    commit.begin()
    commit.abort()
    torch.cuda.synchronize()
    assert ran == ["aborted step"]
    assert bool((dst == 6).all())


def test_a_failed_commit_runs_the_rollbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FailingKernel:
        def __getitem__(self, grid):
            raise RuntimeError("launch failed")

    commit = _new_commit(4096)
    dst = _zeros()
    ran = []
    commit.begin()
    commit.stage(dst, _full(5))
    commit.on_abort(lambda: ran.append("rolled back"))
    monkeypatch.setattr(ic, "_input_commit_kernel", _FailingKernel())
    with pytest.raises(RuntimeError, match="launch failed"):
        commit.commit()
    assert ran == ["rolled back"]
    assert not commit.is_open
    monkeypatch.undo()
    _step(commit, dst, 7)  # the commit is usable again
    torch.cuda.synchronize()
    assert bool((dst == 7).all())


def _sampling_metadata() -> SimpleNamespace:
    # The sampling-parameter staging of ``SpecMetadata`` on a stand-in that carries the attributes
    # ``populate_sampling_params_for_one_model`` reads (the cache methods are the real ones).
    meta = SimpleNamespace(
        spec_dec_mode=SimpleNamespace(use_one_engine=lambda: True),
        prepare_rejection_sampling_buffers=lambda: None,
        _populate_request_rng_state=lambda requests, normalized: None,
        temperatures=None,
        is_spec_dec_tree=False,
        max_total_draft_tokens=2,
        max_draft_len=2,
        runtime_draft_len=2,
        max_num_requests=4,
        use_rejection_sampling=False,
        batch_slot_ids=None,
        dummy_slot_row=0,
        group_all_greedy_sample=None,
        _force_non_greedy_for_capture=False,
        _sampling_params_signature=[None, None],
    )
    for name in (
        "_scan_one_model_sampling",
        "_sampling_params_buffers_need_update",
        "invalidate_sampling_params_cache",
    ):
        setattr(meta, name, MethodType(getattr(SpecMetadata, name), meta))
    return meta


def _sampling_request(temperature: float, slot: int) -> SimpleNamespace:
    return SimpleNamespace(
        sampling_config=SimpleNamespace(temperature=[temperature], top_k=None, top_p=None),
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        py_seq_slot=slot,
    )


def test_an_aborted_step_does_not_leave_the_sampling_cache_claiming_current_buffers() -> None:
    meta = _sampling_metadata()
    requests = [_sampling_request(0.8, 0), _sampling_request(0.8, 1)]
    commit = _new_commit(1 << 16)

    commit.begin()
    with ic.input_commit_scope(commit):
        SpecMetadata.populate_sampling_params_for_one_model(meta, requests)
    assert meta._sampling_params_signature != [None, None]  # recorded for the staged copies
    commit.abort()  # e.g. a later part of the input preparation raised
    assert meta._sampling_params_signature == [None, None]

    # The same batch again must stage its values again: the buffers still hold the allocation's.
    commit.begin()
    with ic.input_commit_scope(commit):
        SpecMetadata.populate_sampling_params_for_one_model(meta, requests)
    commit.commit()
    torch.cuda.synchronize()
    expected = torch.full((2,), 0.8, dtype=torch.float32)
    assert torch.equal(meta.request_temperatures[:2].cpu(), expected)
    # Two generation requests of one target token and two draft tokens each.
    assert torch.equal(meta.temperatures[:6].cpu(), torch.full((6,), 0.8, dtype=torch.float32))


def test_state_index_upload_reads_a_snapshot_held_until_its_kernel_ran(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcm, "gen_host_bundle_kernels_enabled", lambda: True)
    manager = object.__new__(mcm.MambaHybridCacheManagerV2)
    manager.__dict__.update(
        cuda_state_indices=torch.full((3,), -7, dtype=torch.int32, device="cuda"),
        _host_state_indices=torch.tensor([11, 4, 0], dtype=torch.int32).pin_memory(),
        _dummy_request_mask=torch.ones(3, dtype=torch.bool, device="cuda"),
        _dummy_request_mask_host=torch.ones(3, dtype=torch.bool).pin_memory(),
    )
    assert manager._upload_state_indices_and_dummy_mask([False])  # compiles the kernel
    torch.cuda.synchronize()
    manager._state_upload_snapshots = []
    torch.cuda._sleep(DELAY_CYCLES)  # the upload kernel queues behind this
    assert manager._upload_state_indices_and_dummy_mask([False, True])
    ((snapshots, event),) = manager._state_upload_snapshots
    assert not event.query()  # the kernel has not run: the snapshots must stay alive
    # An admission and the next batch refill the shared host buffers before the first kernel ran.
    manager._host_state_indices.copy_(torch.tensor([99, 98, 97], dtype=torch.int32))
    manager._dummy_request_mask_host.fill_(True)
    # A second upload with the delayed first one still pending accumulates, it does not drop it.
    assert manager._upload_state_indices_and_dummy_mask([True, True, True])
    assert len(manager._state_upload_snapshots) == 2
    torch.cuda.synchronize()
    # The kernels ran in order, each on its own snapshot: the final state is the second upload's.
    assert manager.cuda_state_indices.tolist() == [99, 98, 97]
    assert manager._dummy_request_mask.tolist() == [True, True, True]
    assert event.query()
    # Completed snapshots are released by the next upload.
    manager._host_state_indices.copy_(torch.tensor([1, 2, 3], dtype=torch.int32))
    assert manager._upload_state_indices_and_dummy_mask([False])
    torch.cuda.synchronize()
    assert len(manager._state_upload_snapshots) == 1
    assert manager.cuda_state_indices.tolist() == [1, 2, 3]
    assert manager._dummy_request_mask.tolist() == [False, False, False]


def test_state_index_upload_first_kernel_sees_its_own_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The same race without a second upload to hide it: the host buffer is refilled while the
    # first kernel is still queued, and the device must hold the values of the upload call.
    monkeypatch.setattr(mcm, "gen_host_bundle_kernels_enabled", lambda: True)
    manager = object.__new__(mcm.MambaHybridCacheManagerV2)
    manager.__dict__.update(
        cuda_state_indices=torch.zeros(4, dtype=torch.int32, device="cuda"),
        _host_state_indices=torch.tensor([5, 6, 7, 8], dtype=torch.int32).pin_memory(),
        _dummy_request_mask=torch.zeros(4, dtype=torch.bool, device="cuda"),
        _dummy_request_mask_host=torch.zeros(4, dtype=torch.bool).pin_memory(),
    )
    assert manager._upload_state_indices_and_dummy_mask([False])  # compiles the kernel
    torch.cuda.synchronize()
    manager._host_state_indices.copy_(torch.tensor([5, 6, 7, 8], dtype=torch.int32))
    torch.cuda._sleep(DELAY_CYCLES)
    assert manager._upload_state_indices_and_dummy_mask([False, True, False, True])
    manager._host_state_indices.fill_(-1)  # refilled while the kernel is queued
    manager._dummy_request_mask_host.fill_(False)
    # Were the snapshots released right after the launch, the caching host allocator would hand
    # their blocks to the next pinned allocations of the same sizes.
    junk = [torch.full((4,), -2, dtype=torch.int32).pin_memory() for _ in range(8)]
    junk += [torch.zeros(4, dtype=torch.bool).pin_memory() for _ in range(8)]
    torch.cuda.synchronize()
    assert manager.cuda_state_indices.tolist() == [5, 6, 7, 8]
    assert manager._dummy_request_mask.tolist() == [False, True, False, True]
    del junk


def _case(seed: int, num_pools: int, num_blocks: int, num_seqs: int, copy_idx=None):
    gen = torch.Generator().manual_seed(seed)
    host = torch.randint(
        0, 1 << 20, (num_pools, num_seqs + 1, 2, num_blocks), generator=gen, dtype=torch.int32
    )
    if copy_idx is None:
        copy_idx = torch.arange(num_seqs, dtype=torch.int32).pin_memory()
    scales = torch.ones(num_pools, dtype=torch.int32).pin_memory()
    kv_offset = torch.zeros(num_pools, dtype=torch.int32).pin_memory()
    dst = torch.full((num_pools, num_seqs, 2, num_blocks), -3, dtype=torch.int32, device="cuda")
    return [host.pin_memory(), dst, copy_idx, scales, kv_offset]


def _metadata(target, draft) -> TrtllmAttentionMetadata:
    def manager(case):
        def make_job(dst, request_ids, beam_width, num_contexts, num_seqs):
            return ic.block_offsets_copy_job(*case)

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
    return md


def test_paired_block_offset_copy_needs_one_copy_index_buffer_per_manager() -> None:
    census = ic.bundle_census()
    census.reset()
    census.begin_step(live=True)
    target = _case(1, 2, 256, 2)
    draft = _case(2, 1, 256, 2)
    assert _metadata(target, draft)._copy_target_and_draft_block_offsets()
    assert census.get("block_offsets_pair") == 1

    # Both managers fill their own copy-index buffer at job time and the kernel reads both when it
    # runs: with ONE shared buffer the second fill would overwrite the first. Not paired, nothing
    # launched; the caller then copies each table in its own launch.
    shared = target[2]
    draft_shared = _case(2, 1, 256, 2, copy_idx=shared)
    untouched = draft_shared[1].clone()
    assert not _metadata(target, draft_shared)._copy_target_and_draft_block_offsets()
    torch.cuda.synchronize()
    assert torch.equal(draft_shared[1], untouched)
    assert census.get("block_offsets_pair") == 1 and census.get("block_offsets_pair_fallback") == 1
    census.reset()
