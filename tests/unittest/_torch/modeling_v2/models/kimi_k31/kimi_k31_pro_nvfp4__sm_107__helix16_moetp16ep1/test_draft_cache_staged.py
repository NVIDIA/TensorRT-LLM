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
"""The K3.1 ULL target's drafter's managed draft-cache uploads, staged into the step's input commit.

The engine runs ``DFlashWorker.prepare_managed_draft_cache`` inside the step's input commit when
the worker sets ``stages_managed_draft_cache``. Its context-length and position-offset scatters,
the batch-to-slot map, the snapshot slots and the host-built draft block tables are then staged
and written by the commit kernel. Each piece runs twice from the same poisoned device state, once
copying and once staged (commit verify on every commit), over several steps with requests coming
and going; the device buffers must end bitwise equal.
"""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEV = "cuda"
NUM_SLOTS = 9
MAX_BLOCKS = 12


def _worker():
    from tensorrt_llm._device_copy import DeviceCopyBuffer
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1.modeling import (  # noqa: E501
        DFlashWorker,
    )

    w = object.__new__(DFlashWorker)
    w._ctx_len = torch.full((NUM_SLOTS,), -9, dtype=torch.long, device=DEV)
    w._ctx_position_offset = torch.full((NUM_SLOTS,), -8, dtype=torch.long, device=DEV)
    w._ctx_len_host = [0] * NUM_SLOTS
    w._dummy_slot = NUM_SLOTS - 1
    w._req_to_slot = {}
    w._batch_to_slot_buffer = DeviceCopyBuffer(
        torch.full((8,), -7, dtype=torch.long, device=DEV), pin_memory=True
    )
    w._managed_snapshot_slots = torch.full((8,), -6, dtype=torch.long, device=DEV)
    w._ctx_block_tables = torch.full((8, MAX_BLOCKS), -5, dtype=torch.int32, device=DEV)
    w._ctx_block_counts = torch.full((8,), -4, dtype=torch.long, device=DEV)
    w._device_draft_tables = False
    return w


def _manager(tables, counts):
    return SimpleNamespace(
        draft_layer_ids=(61,),
        has_standalone_draft_cache=True,
        get_draft_block_table=lambda rids: tables[: len(rids)].clone(),
        get_draft_num_blocks=lambda rid: counts[rid],
    )


def _step(w, step, rng):
    """One step's managed uploads (the pieces of prepare_managed_draft_cache)."""
    num = 1 + step % 4
    rids = [100 + (step + i) % 6 for i in range(num)]
    w._req_to_slot = {rid: (rid - 100) % (NUM_SLOTS - 1) for rid in rids}
    updates = {w._dummy_slot: 0}
    offsets = {w._dummy_slot: 0}
    for rid in rids[: 1 + step % 2]:
        slot = w._req_to_slot[rid]
        updates[slot] = int(torch.randint(0, 1 << 20, (1,), generator=rng))
        offsets[slot] = int(torch.randint(-64, 64, (1,), generator=rng))
    w._write_ctx_len(updates)
    w._write_slot_values(w._ctx_position_offset, offsets)
    w._rebuild_batch_to_slot(rids)
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.input_staging import (
        stage_or_copy,
    )

    stage_or_copy(
        w._managed_snapshot_slots[:num],
        torch.tensor([w._req_to_slot[r] for r in rids], dtype=torch.long, pin_memory=True),
    )
    tables = torch.randint(0, 1 << 16, (num, MAX_BLOCKS), dtype=torch.int32, generator=rng)
    counts = {rid: int(torch.randint(0, MAX_BLOCKS, (1,), generator=rng)) for rid in rids}
    w._ctx_kv_manager = _manager(tables, counts)
    assert w._refresh_ctx_block_tables(None, num, rids)


def _state(w):
    names = (
        "_ctx_len",
        "_ctx_position_offset",
        "_managed_snapshot_slots",
        "_ctx_block_tables",
        "_ctx_block_counts",
    )
    state = {n: getattr(w, n).clone() for n in names}
    state["batch_to_slot"] = w._batch_to_slot_buffer.tensor.clone()
    state["ctx_len_host"] = list(w._ctx_len_host)
    return state


def test_staged_managed_draft_cache_uploads_match_the_copies(monkeypatch):
    from tensorrt_llm._torch.pyexecutor import input_commit as input_commit_module
    from tensorrt_llm._torch.pyexecutor.input_commit import InputCommit, input_commit_step

    monkeypatch.setattr(input_commit_module, "_VERIFY_COMMITS", -1)
    out = {}
    for staged in (False, True):
        w = _worker()
        rng = torch.Generator().manual_seed(5)
        commit = InputCommit(capacity=1 << 12)
        for step in range(8):
            with input_commit_step(commit if staged else None):
                _step(w, step, rng)
            torch.cuda.synchronize()
        out[staged] = _state(w)
        if staged:
            assert commit.num_commits == 8
    for name, value in out[False].items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, out[True][name]), name
        else:
            assert value == out[True][name], name


def _history_worker(histories):
    from collections import deque

    w = _worker()
    caches = {}

    def get_cache(rid):
        return caches.setdefault(rid, object())

    w._ctx_kv_manager = SimpleNamespace(
        draft_layer_ids=(61,),
        has_standalone_draft_cache=True,
        _draft_dummy_request_ids=set(),
        get_draft_history=lambda rid: histories.get(rid),
    )
    w._ctx_kv_manager.kv_cache_map = type(
        "_CacheMap", (), {"__getitem__": lambda self, rid: get_cache(rid)}
    )()
    w._graph_dummy_id_floor = 1 << 40
    w._free_slots = deque(range(NUM_SLOTS - 1))
    w._managed_cache_bindings = {}
    w._req_ctx_pos = {}
    # The worker's default without a draft window: the whole history is restored.
    w._history_window = None
    return w


def test_staged_managed_history_with_arriving_requests_matches_the_copies(monkeypatch):
    """The engine's ``_prepare_managed_history`` (the target does not override it) while
    requests arrive (a new slot: its length is reset, then restored from the draft history in the
    same step), continue and leave, staged vs copied, with the commit verify on every commit: the
    restored length staged after the reset rewrites the same staged element (the commit's
    last-wins), so the device gets the restored length."""
    from tensorrt_llm._torch.pyexecutor import input_commit as input_commit_module
    from tensorrt_llm._torch.pyexecutor.input_commit import InputCommit, input_commit_step

    monkeypatch.setattr(input_commit_module, "_VERIFY_COMMITS", -1)
    histories = {
        100: SimpleNamespace(valid_length=4096, position=4100),
        101: SimpleNamespace(valid_length=77, position=80),
        102: SimpleNamespace(valid_length=900001, position=900005),
    }
    steps = [[100], [100], [100, 101], [101, 102], [102]]
    out = {}
    for staged in (False, True):
        w = _history_worker(histories)
        commit = InputCommit(capacity=1 << 12)
        restored = []
        for rids in steps:
            with input_commit_step(commit if staged else None):
                restored.append(w._prepare_managed_history(rids))
            torch.cuda.synchronize()
        out[staged] = (_state(w), restored, dict(w._req_to_slot))
    for name, value in out[False][0].items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, out[True][0][name]), name
        else:
            assert value == out[True][0][name], name
    assert out[False][1:] == out[True][1:]
    # The arriving requests' restored lengths are on the device.
    state = out[True][0]
    assert state["_ctx_len"][out[True][2][102]].item() == 900001
