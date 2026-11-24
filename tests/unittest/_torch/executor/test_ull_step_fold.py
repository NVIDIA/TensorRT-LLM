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
"""The decode-step shell's ``step_fold`` (``ull_shell.StepFold``): the folded step prep writes
what the unfolded ops write.

* The KV block-offset copy inside the input commit kernel against the C++
  ``copy_batch_block_offsets_to_device`` (and the standalone Triton copy), over a multi-step
  sequence with requests joining and leaving, block-boundary crossings, unmapped pages,
  staged copies and the overlap fixup in the same commit, and the fold flipping between
  steps; through ``KVCacheManagerV2.copy_batch_block_offsets`` too.
* The mamba state-index upload on the execution stream for generation-only batches.
* The A/B iteration blocks and the gating of every folded site.
"""

import random
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._device_copy import DeviceCopyBuffer
from tensorrt_llm._torch.pyexecutor import input_commit as input_commit_module
from tensorrt_llm._torch.pyexecutor import ull_shell
from tensorrt_llm._torch.pyexecutor.input_commit import (
    InputCommit,
    fold_block_offsets_copy,
    input_commit_step,
    warmup_input_commit_kernels,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import mamba_cache_manager_v2
from tensorrt_llm._torch.pyexecutor.step_upload_kernels import (
    block_offsets_copy_job,
    copy_batch_block_offsets_from_host,
)
from tensorrt_llm._torch.pyexecutor.ull_shell import StepFold

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture
def fold(monkeypatch: pytest.MonkeyPatch) -> StepFold:
    """A switched-on StepFold in place of the process one, with a registered stream."""
    state = StepFold(True)
    state.stream = torch.cuda.current_stream()
    for module in (ull_shell, input_commit_module, mamba_cache_manager_v2):
        monkeypatch.setattr(module, "STEP_FOLD", state)
    return state


def _cpp_block_offsets(host, dst, copy_idx, scales, kv_offset) -> None:
    try:
        from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
            copy_batch_block_offsets_to_device,
        )
    except ImportError:
        assert copy_batch_block_offsets_from_host(host, dst, copy_idx, scales, kv_offset)
        return
    copy_batch_block_offsets_to_device(
        host, dst, copy_idx, scales, kv_offset, torch.cuda.current_stream().cuda_stream
    )


# ---------------------------------------------------------------------------
# StepFold switch state
# ---------------------------------------------------------------------------


def test_step_fold_off_never_active() -> None:
    state = StepFold(False, ab_period=4)
    assert not state.active
    assert not state.start(torch.cuda.current_stream())
    assert state.stream is None
    for i in range(16):
        state.on_iteration(i)
        assert not state.active


def test_step_fold_ab_blocks() -> None:
    state = StepFold(True, ab_period=3)
    assert state.start(torch.cuda.current_stream())
    arms = []
    for i in range(12):
        state.on_iteration(i)
        arms.append(state.active)
    assert arms == [False] * 3 + [True] * 3 + [False] * 3 + [True] * 3
    always = StepFold(True)
    always.on_iteration(5)
    assert always.active
    with pytest.raises(ValueError):
        StepFold(True, ab_period=-1)


def test_step_fold_enable_and_suppress() -> None:
    # The process's fold is off until a model's shell turns it on (ull_shell.adopt).
    assert not ull_shell.StepFold(False).active
    state = StepFold(False, ab_period=2)
    state.enable(True)
    state.on_iteration(2)
    assert state.enabled and state.active
    state.suppress(True)
    assert not state.active
    state.on_iteration(3)
    assert not state.active
    state.suppress(False)
    assert state.active
    state.on_iteration(4)
    assert not state.active  # an A block
    state.enable(False)
    state.on_iteration(6)
    assert not state.active


# ---------------------------------------------------------------------------
# KV block offsets inside the input commit
# ---------------------------------------------------------------------------


class _PageTable:
    """A host page table that evolves like a decode batch's: requests join (new row with a
    few pages) and leave (row freed), every live request grows by 0..3 tokens per step and
    gets a new page when it crosses a block boundary; unmapped tails are -1."""

    def __init__(self, num_pools: int, rows: int, num_blocks: int, block: int, seed: int):
        self.rng = random.Random(seed)
        self.host = torch.full((num_pools, rows, 2, num_blocks), -1, dtype=torch.int32)
        self.host = self.host.pin_memory()
        self.block = block
        self.free_rows = list(range(rows))
        self.live = {}  # row -> tokens
        self.next_page = 1
        self.num_leaves = 0

    def join(self) -> None:
        if not self.free_rows:
            return
        row = self.free_rows.pop(self.rng.randrange(len(self.free_rows)))
        self.host[:, row] = -1
        self.live[row] = 0
        self.grow(row, self.rng.randint(1, 3 * self.block))

    def leave(self) -> None:
        if len(self.live) > 1:
            row = self.rng.choice(sorted(self.live))
            del self.live[row]
            self.free_rows.append(row)
            self.num_leaves += 1

    def grow(self, row: int, tokens: int) -> int:
        before = -(-self.live[row] // self.block)
        self.live[row] += tokens
        after = -(-self.live[row] // self.block)
        for page in range(before, min(after, self.host.shape[-1])):
            for pool in range(self.host.shape[0]):
                self.host[pool, row, 0, page] = self.next_page + pool
                self.host[pool, row, 1, page] = self.next_page + pool
            self.next_page += self.host.shape[0]
        return after - before

    def step(self) -> int:
        crossings = 0
        if self.rng.random() < 0.3:
            self.join()
        if self.rng.random() < 0.2:
            self.leave()
        for row in self.live:
            crossings += self.grow(row, self.rng.randint(0, 3))
        return crossings


def _commit_step(commit, staged, fixup, job) -> None:
    with input_commit_step(commit):
        for dst, values in staged:
            commit.stage(dst, values)
        if fixup is not None:
            commit.set_overlap_fixup(**fixup)
        if job is not None:
            assert fold_block_offsets_copy(*job)


# The C++ copy needs num_blocks * 4 bytes to be a multiple of 16.
@pytest.mark.parametrize("num_pools,num_blocks,block", [(1, 8, 4), (2, 36, 16), (3, 1500, 64)])
@pytest.mark.parametrize("flip", [False, True])
def test_commit_block_offsets_sequence(
    fold: StepFold, num_pools: int, num_blocks: int, block: int, flip: bool
) -> None:
    """Every step of a decode-like sequence: the committed block offsets equal the C++ copy,
    the staged copies and the overlap fixup of the same commit are unaffected, and switching
    the fold off and on between steps (flip) changes nothing."""
    seed = num_pools * 7 + num_blocks + block + flip
    rng = random.Random(seed)
    table = _PageTable(num_pools, 12, num_blocks, block, seed)
    for _ in range(3):
        table.join()
    scales = torch.tensor([2 + p for p in range(num_pools)], dtype=torch.int32).pin_memory()
    kv_offset = torch.tensor([1000 * p for p in range(num_pools)], dtype=torch.int32).pin_memory()
    dst_rows = 8
    dst = torch.full((num_pools, dst_rows, 2, num_blocks), 77, dtype=torch.int32, device="cuda")
    ref = dst.clone()
    staged_dst = torch.zeros(64, dtype=torch.int32, device="cuda")
    tokens, slots_total = 3, 6
    new_tokens = torch.randint(0, 1000, (tokens + 1, slots_total, 1), dtype=torch.int32).cuda()
    lens = torch.randint(1, tokens + 1, (slots_total,), dtype=torch.int32).cuda()
    next_draft = torch.randint(0, 1000, (slots_total, tokens), dtype=torch.int32).cuda()
    commit = InputCommit()
    folded = crossings = 0
    for step in range(40):
        crossings += table.step()
        rows = sorted(table.live)[:dst_rows]
        copy_idx = torch.tensor(rows, dtype=torch.int32).pin_memory()
        job = (table.host, dst, copy_idx, scales, kv_offset, torch.cuda.current_stream())
        values = [rng.randint(-50, 50) for _ in range(staged_dst.numel())]
        prev = rng.sample(range(slots_total), min(len(rows), 3))
        ids = torch.zeros(32, dtype=torch.int32, device="cuda")
        drafts = torch.zeros(32, dtype=torch.int32, device="cuda")
        pos = torch.zeros(16, dtype=torch.int32, device="cuda")
        kv = torch.zeros(8, dtype=torch.int32, device="cuda")
        fixup = dict(
            previous_slots=prev,
            tokens_per_req=tokens,
            draft_width=tokens - 1,
            new_tokens=new_tokens,
            new_tokens_lens=lens,
            next_draft_tokens=next_draft,
            input_ids=ids,
            input_ids_start=0,
            draft_tokens=drafts,
            draft_tokens_start=0,
            pos_offsets=pos,
            pos_offsets_start=0,
            kv_offsets=kv,
            kv_offsets_start=0,
        )
        fold.active = not (flip and step % 2)
        if fold.active:
            _commit_step(commit, [(staged_dst, values)], fixup, job)
            folded += 1
        else:
            assert not fold_block_offsets_copy(*job)
            _cpp_block_offsets(table.host, dst, copy_idx, scales, kv_offset)
            _commit_step(commit, [(staged_dst, values)], fixup, None)
        _cpp_block_offsets(table.host, ref, copy_idx, scales, kv_offset)
        torch.cuda.synchronize()
        assert torch.equal(dst, ref), step
        assert staged_dst.tolist() == values
        prev_t = torch.tensor(prev, dtype=torch.long, device="cuda")
        assert torch.equal(ids[: len(prev) * tokens], new_tokens[:tokens, prev_t, 0].T.flatten())
    assert crossings > 0 and table.num_leaves > 0
    assert fold.counts.get("block_offsets_folded", 0) == folded


def test_commit_block_offsets_verify_and_empty_batch(
    fold: StepFold, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(input_commit_module, "_VERIFY_COMMITS", -1)
    table = _PageTable(2, 4, 16, 8, 1)
    table.join()
    scales = torch.ones(2, dtype=torch.int32).pin_memory()
    kv_offset = torch.ones(2, dtype=torch.int32).pin_memory()
    dst = torch.zeros((2, 4, 2, 16), dtype=torch.int32, device="cuda")
    commit = InputCommit()
    copy_idx = torch.tensor(sorted(table.live), dtype=torch.int32).pin_memory()
    _commit_step(commit, [], None, (table.host, dst, copy_idx, scales, kv_offset, None))
    torch.cuda.synchronize()
    ref = torch.zeros_like(dst)
    _cpp_block_offsets(table.host, ref, copy_idx, scales, kv_offset)
    torch.cuda.synchronize()
    assert torch.equal(dst, ref)
    assert fold.counts["block_offsets_checked"] == 1
    # The fold's own verify count alone checks the first n folded copies.
    monkeypatch.setattr(input_commit_module, "_VERIFY_COMMITS", 0)
    fold.verify = 2
    for _ in range(3):
        _commit_step(commit, [], None, (table.host, dst, copy_idx, scales, kv_offset, None))
    torch.cuda.synchronize()
    assert torch.equal(dst, ref)
    assert fold.counts["block_offsets_checked"] == 3 and fold.num_block_checks == 2
    # An empty batch: the job has no programs and nothing is launched for it. (A 0-row view
    # of pinned storage: a fresh empty tensor has no storage and does not report pinned.)
    empty = torch.zeros(4, dtype=torch.int32).pin_memory()[:0]
    assert empty.is_pinned()
    _commit_step(commit, [], None, (table.host, dst, empty, scales, kv_offset, None))
    torch.cuda.synchronize()
    assert torch.equal(dst, ref)


def test_fold_block_offsets_gating(fold: StepFold) -> None:
    host = torch.zeros((1, 2, 2, 4), dtype=torch.int32).pin_memory()
    dst = torch.zeros((1, 2, 2, 4), dtype=torch.int32, device="cuda")
    idx = torch.zeros(1, dtype=torch.int32).pin_memory()
    ones = torch.ones(1, dtype=torch.int32).pin_memory()
    args = (host, dst, idx, ones, ones)
    assert not fold_block_offsets_copy(*args)  # no commit open
    commit = InputCommit()
    commit.begin()
    with input_commit_module.input_commit_scope(commit):
        fold.active = False
        assert not fold_block_offsets_copy(*args)
        fold.active = True
        assert not fold_block_offsets_copy(*args, torch.cuda.Stream())  # other stream
        assert not fold_block_offsets_copy(host.cuda(), dst, idx, ones, ones)  # bad layout
        assert fold_block_offsets_copy(*args, torch.cuda.current_stream())
        assert not fold_block_offsets_copy(*args)  # one copy per commit
    commit.abort()
    assert commit._block_job is None
    assert fold.counts["block_offsets_folded"] == 1
    assert fold.counts["block_offsets_other_stream"] == 1
    assert fold.counts["block_offsets_not_folded"] == 2


def test_kv_manager_v2_copy_goes_through_commit(fold: StepFold) -> None:
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

    table = _PageTable(2, 6, 40, 16, 5)
    for _ in range(3):
        table.join()
    rows = sorted(table.live)
    copy_idx = torch.tensor(rows, dtype=torch.int32).pin_memory()
    manager = SimpleNamespace(
        _publish_sparse_metadata=lambda: None,
        sparse_metadata_batch=None,
        max_copy_beam_width=1,
        index_mapper=SimpleNamespace(get_copy_index=lambda *a: copy_idx),
        kv_cache_type=None,
        _use_per_layer_page_tables=False,
        _page_table_materializer=SimpleNamespace(uses_device_expansion=False),
        host_kv_cache_block_offsets=table.host,
        index_scales=torch.tensor([3, 5], dtype=torch.int32).pin_memory(),
        kv_offset=torch.tensor([0, 9], dtype=torch.int32).pin_memory(),
        _stream=torch.cuda.current_stream(),
    )
    dst = torch.zeros((2, 4, 2, 40), dtype=torch.int32, device="cuda")
    ref = torch.zeros_like(dst)
    _cpp_block_offsets(table.host, ref, copy_idx, manager.index_scales, manager.kv_offset)
    commit = InputCommit()
    with input_commit_step(commit):
        KVCacheManagerV2.copy_batch_block_offsets(manager, dst, rows, 1, 0, len(rows))
        assert commit._block_job is not None
    torch.cuda.synchronize()
    assert torch.equal(dst, ref)
    # Switch off for the step: the manager copies by itself, no commit involved.
    fold.active = False
    dst.zero_()
    KVCacheManagerV2.copy_batch_block_offsets(manager, dst, rows, 1, 0, len(rows))
    torch.cuda.synchronize()
    assert torch.equal(dst, ref)


def test_warmup_compiles_block_variant(fold: StepFold) -> None:
    commit = InputCommit()
    warmup_input_commit_kernels(commit)
    assert commit.num_commits == 0
    assert block_offsets_copy_job is not None


# ---------------------------------------------------------------------------
# Mamba state-index upload on the execution stream
# ---------------------------------------------------------------------------


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


def test_state_index_upload_on_step_stream(fold: StepFold, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mamba_cache_manager_v2, "state_index_copy_kernel_enabled", lambda: True)
    manager = _state_index_manager(4)
    step_stream = torch.cuda.Stream()
    fold.stream = step_stream
    # Compile the upload kernel first: its JIT would outlast the sleep below.
    manager._host_state_indices[:2] = torch.tensor([1, 2], dtype=torch.int32)
    assert manager._upload_state_indices_and_dummy_mask([True, True])
    torch.cuda.synchronize()
    manager._host_state_indices[:2] = torch.tensor([5, 9], dtype=torch.int32)
    # Hold the step stream so the upload, if it is there, cannot have run yet.
    with torch.cuda.stream(step_stream):
        torch.cuda._sleep(200_000_000)
    assert manager._upload_on_step_stream([False, False], 0)
    torch.cuda.current_stream().synchronize()
    assert not step_stream.query()
    step_stream.synchronize()
    assert manager.cuda_state_indices.tolist() == [5, 9, 0, 0]
    assert manager._dummy_request_mask.tolist() == [False] * 4
    assert fold.counts["state_indices_on_step_stream"] == 1
    # Context batches, the switch off for the step, or no registered stream: not handled.
    assert not manager._upload_on_step_stream([False], 1)
    fold.active = False
    assert not manager._upload_on_step_stream([False], 0)
    fold.active, fold.stream = True, None
    assert not manager._upload_on_step_stream([False], 0)
