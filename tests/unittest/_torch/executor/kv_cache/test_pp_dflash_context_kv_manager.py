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
"""KVCacheManagerV2 with the layouts of a pipeline-parallel DFlash context server, on one GPU.

A four-stage pipeline-parallel DFlash context server splits a 62-layer
target 15/16/16/15 and puts the drafter's 5 layers (global ids 62-66) on the
last stage only. The four stages' managers are built here in one process, with
the cross-rank collectives answered locally, and checked for:

* the last stage's 20-layer manager (global ids 47-66) allocating a context,
  a generation step with the drafter's block, and freeing it;
* the drafter's block width, generation headroom and block-table width being
  equal on every stage;
* a one-stage manager with all 67 layers behaving alike on any server;
* the pool snapshot a follower reports to the first rank for its admission
  back-pressure (``pp_admission.py``): its slots match the pools, its resume
  room agrees with the resume gate, its charge bounds what a context takes,
  and a lagged admission loop over the four stages with a small last stage
  never fails a follower (the same loop without the ledger does).
"""

from collections import deque
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_kvcm2_integration import _ContextRequest, _run_context, _try_run_context

from tensorrt_llm._torch.disaggregation import pp_dflash_context
from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as kv_cache_v2_module
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheLayerOwner,
    KVCacheLayerSpec,
    KVCacheManagerV2,
    Role,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.pp_admission import Candidate, PPAdmissionLedger, ScheduledChunk
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping

TARGET_LAYERS = 62
DRAFT_LAYERS = 5
PARTITION = [15, 16, 16, 15]
WINDOW = 2048
MAX_SEQ_LEN = 4096
TOKENS_PER_BLOCK = 32
MAX_DRAFT_LEN = 7
BLOCK_WIDTH = MAX_DRAFT_LEN + 1
QUOTA_BYTES = 128 << 20
# Every stage, the last one with the drafter's layers included, holds MAX_SEQ_LEN
# within this quota, so no stage clamps its max_seq_len.
STAGE_QUOTA_BYTES = 512 << 20


@pytest.fixture(autouse=True)
def _server(monkeypatch, enabled: bool = True) -> None:
    """What the model construction of a pipeline-parallel DFlash context server records
    (``pp_dflash_context.set_pp_dflash_context_server``), undone after the test."""
    monkeypatch.setattr(pp_dflash_context, "_pp_dflash_context_server", bool(enabled))


def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()


class _OneProcessGroup:
    """Answers the managers' collectives as if every stage reported the same value."""

    local_world_size = 1

    @staticmethod
    def allreduce(value, op=None):
        return value

    @staticmethod
    def allgather(value):
        return [value]


@pytest.fixture
def stages_in_one_process(monkeypatch):
    monkeypatch.setattr(
        kv_cache_v2_module,
        "Distributed",
        SimpleNamespace(get=lambda mapping: _OneProcessGroup()),
    )


def _target_spec(layer_id: int) -> KVCacheLayerSpec:
    return KVCacheLayerSpec(
        global_layer_id=layer_id,
        owner=KVCacheLayerOwner.TARGET,
        dtype=DataType.FP8,
        num_kv_heads=16,
        head_dim=64,
        buffer_roles=(Role.KEY, Role.VALUE),
        sliding_window_size=None if layer_id % 4 == 1 else WINDOW,
    )


def _draft_spec(index: int) -> KVCacheLayerSpec:
    return replace(
        _target_spec(0),
        global_layer_id=TARGET_LAYERS + index,
        owner=KVCacheLayerOwner.DRAFT,
        dtype=DataType.BF16,
        num_kv_heads=8,
        head_dim=128,
        sliding_window_size=WINDOW,
    )


def _stage_specs(rank: int) -> list:
    start = sum(PARTITION[:rank])
    specs = [_target_spec(layer_id) for layer_id in range(start, start + PARTITION[rank])]
    if rank == len(PARTITION) - 1:
        specs += [_draft_spec(i) for i in range(DRAFT_LAYERS)]
    return specs


def _all_specs() -> list:
    return [_target_spec(layer_id) for layer_id in range(TARGET_LAYERS)] + [
        _draft_spec(i) for i in range(DRAFT_LAYERS)
    ]


def _stage_mapping(rank: int) -> Mapping:
    return Mapping(
        world_size=len(PARTITION),
        rank=rank,
        gpus_per_node=len(PARTITION),
        pp_size=len(PARTITION),
        pp_partition=PARTITION,
    )


def _manager(specs: list, mapping: Mapping, quota_bytes: int = QUOTA_BYTES) -> KVCacheManagerV2:
    return KVCacheManagerV2(
        KvCacheConfig(
            enable_block_reuse=False,
            max_gpu_total_bytes=quota_bytes,
            host_cache_size=0,
            enable_swa_scratch_reuse=False,
        ),
        CacheType.SELF,
        num_layers=TARGET_LAYERS,
        num_kv_heads=16,
        head_dim=64,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=2,
        mapping=mapping,
        dtype=DataType.FP8,
        layer_specs=specs,
        spec_config=DFlashDecodingConfig(max_draft_len=MAX_DRAFT_LEN, speculative_model="draft"),
        vocab_size=4096,
        enable_stats=False,
    )


def _context_then_generation(manager: KVCacheManagerV2, request_id: int, prompt_len: int):
    """Run a context, then one generation step that accepts every draft token."""
    request = _ContextRequest(
        request_id,
        list(range(prompt_len)),
        prompt_len,
        f"conversation-{request_id}",
        use_conversation_params=False,
    )
    _run_context(manager, request)
    generation = SimpleNamespace(
        py_request_id=request.py_request_id,
        py_beam_width=1,
        py_draft_tokens=[0] * MAX_DRAFT_LEN,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        is_dummy_request=False,
        is_disagg_generation_transmission_complete=False,
        context_phase_params=None,
        max_beam_num_tokens=prompt_len + 1,
    )
    assert manager.try_allocate_generation(generation)
    return request


def _assert_draft_pages_cover_the_block(manager: KVCacheManagerV2, request, prompt_len: int):
    furthest_write_end = prompt_len + MAX_DRAFT_LEN + 1 + BLOCK_WIDTH
    for layer_id in manager.draft_layer_ids:
        pages = manager.get_batch_cache_indices([request.py_request_id], layer_idx=layer_id)[0]
        assert len(pages) * TOKENS_PER_BLOCK >= furthest_write_end


def test_last_stage_manager_allocates_appends_and_frees(monkeypatch, stages_in_one_process):
    _server(monkeypatch)
    manager = _manager(_stage_specs(3), _stage_mapping(3))
    # The prompt puts the drafter's furthest write on the first position of a third page.
    prompt_len = 49
    try:
        assert manager.num_layers == TARGET_LAYERS + DRAFT_LAYERS
        assert manager.target_layer_ids == tuple(range(47, 62))
        assert manager.draft_layer_ids == tuple(range(62, 67))
        assert manager.get_draft_kv_cache_view() is manager
        assert manager._kv_alloc_extra_tokens == BLOCK_WIDTH

        for round_index in range(2):
            request = _context_then_generation(manager, 1 + round_index, prompt_len)
            try:
                _assert_draft_pages_cover_the_block(manager, request, prompt_len)
            finally:
                manager.free_resources(request)
            assert not manager.kv_cache_map.keys() - kv_cache_v2_module._RESERVED_REQUEST_IDS
    finally:
        manager.shutdown()


@pytest.mark.parametrize("server", [False, True])
def test_block_width_and_block_tables_across_the_stages(monkeypatch, stages_in_one_process, server):
    _server(monkeypatch, server)
    # The stages run in one process, so the cross-stage token-capacity sync that
    # gives a pipeline's stages one max_seq_len is answered with each stage's own
    # value. A quota that fits MAX_SEQ_LEN on every stage keeps max_seq_len equal,
    # so the block-table widths below differ only by the generation headroom.
    managers = [
        _manager(_stage_specs(rank), _stage_mapping(rank), STAGE_QUOTA_BYTES) for rank in range(4)
    ]
    try:
        assert [m.max_seq_len for m in managers] == [MAX_SEQ_LEN] * 4
        assert [len(m.pp_layers) for m in managers] == [15, 16, 16, 20]
        assert [m.draft_layer_ids for m in managers][:3] == [()] * 3
        assert all(m.get_draft_kv_cache_view() is None for m in managers[:3])
        extras = [m._kv_alloc_extra_tokens for m in managers]
        headrooms = [m._generation_kv_capacity_headroom for m in managers]
        block_table_widths = [m.max_blocks_per_seq for m in managers]
        if server:
            assert extras == [BLOCK_WIDTH] * 4
            assert len(set(headrooms)) == 1
            assert len(set(block_table_widths)) == 1
        else:
            assert extras == [MAX_DRAFT_LEN - 1] * 3 + [BLOCK_WIDTH]
            assert headrooms[0] < headrooms[3]
        # Every stage admits the same request with the drafter's block in its allocation.
        prompt_len = 49
        for manager in managers:
            request = _context_then_generation(manager, 1, prompt_len)
            manager.free_resources(request)
    finally:
        for manager in managers:
            manager.shutdown()


@pytest.mark.parametrize("server", [False, True])
def test_single_stage_manager_with_all_67_layers(monkeypatch, server):
    _server(monkeypatch, server)
    manager = _manager(_all_specs(), Mapping())
    prompt_len = 49
    try:
        assert manager.num_layers == TARGET_LAYERS + DRAFT_LAYERS
        assert manager.draft_layer_ids == tuple(range(62, 67))
        assert manager._kv_alloc_extra_tokens == BLOCK_WIDTH
        request = _context_then_generation(manager, 1, prompt_len)
        try:
            _assert_draft_pages_cover_the_block(manager, request, prompt_len)
        finally:
            manager.free_resources(request)
    finally:
        manager.shutdown()


def _context_request(request_id: int, prompt_len: int) -> _ContextRequest:
    return _ContextRequest(
        request_id,
        list(range(request_id * 7, request_id * 7 + prompt_len)),
        prompt_len,
        f"conversation-{request_id}",
        use_conversation_params=False,
    )


def test_admission_snapshot_matches_the_pools_and_bounds_a_context(
    monkeypatch, stages_in_one_process
):
    _server(monkeypatch)
    manager = _manager(_stage_specs(3), _stage_mapping(3), STAGE_QUOTA_BYTES)
    ledger = PPAdmissionLedger(2, None)
    requests = []
    try:
        snapshot = manager.pp_admission_snapshot()
        assert snapshot.tokens_per_block == TOKENS_PER_BLOCK
        # No host tier: one constraint per GPU pool group, its free + evictable slots.
        assert [c.available for c in snapshot.constraints] == manager.available_pages_per_level()[0]
        windows = {w for c in snapshot.constraints for w in c.window_blocks}
        # The window bound follows the manager's live interval (window plus
        # generation headroom), never below the window plus one block.
        headroom = manager._generation_kv_capacity_headroom
        live_blocks = -(-(WINDOW + headroom - 2) // TOKENS_PER_BLOCK)
        assert windows == {None, max(live_blocks, WINDOW // TOKENS_PER_BLOCK) + 1}
        # The tokens allocated beyond the history come from the manager.
        assert manager._kv_alloc_extra_tokens >= BLOCK_WIDTH
        assert {c.extra_tokens for c in snapshot.constraints} == {manager._kv_alloc_extra_tokens}
        # The pool layout is built once and reused.
        assert manager._pp_admission_layout is not None
        layout = manager._pp_admission_layout
        assert manager.pp_admission_snapshot() == snapshot
        assert manager._pp_admission_layout is layout
        for c in snapshot.constraints:
            # The resume room keeps at least one slot of margin below the gate.
            assert c.resume_room < c.available and c.resume_room <= c.capacity
        for request_id, prompt_len in enumerate([49, 1000, 2047, 2048, 3001, 33], start=10):
            before = manager.pp_admission_snapshot()
            request = _context_request(request_id, min(prompt_len, manager.max_seq_len))
            if not _try_run_context(manager, request):
                manager.free_resources(request)
                break
            requests.append(request)
            after = manager.pp_admission_snapshot()
            # As the executor builds it: the prompt, no draft tokens on a context.
            candidate = Candidate(request_id, 0, request.prompt_len)
            for c_before, c_after in zip(before.constraints, after.constraints):
                taken = c_before.available - c_after.available
                assert taken <= ledger.demand(c_before, candidate, TOKENS_PER_BLOCK), (
                    request.prompt_len,
                    taken,
                    c_before,
                )
            # A non-negative resume room on every pool group means the gate resumes.
            if all(c.resume_room >= 0 for c in after.constraints):
                assert not manager._resume_gate_refuses()
            if manager._resume_gate_refuses():
                assert any(c.resume_room < 0 for c in after.constraints)
        assert requests
    finally:
        for request in requests:
            manager.free_resources(request)
        manager.shutdown()


@pytest.mark.parametrize("ledger_on", [True, False], ids=["ledger", "stock"])
def test_lagged_admission_never_fails_the_small_last_stage(
    monkeypatch, stages_in_one_process, ledger_on
):
    """Four stages in one process; the last stage's quota is a quarter of the
    others'. The first rank admits one whole-prompt context per iteration, the
    followers see it at once and report their pools ``LAG`` iterations later,
    and a slow generation side frees one finished context every ``PULL``
    iterations. With the ledger no follower ever fails; without it the last
    stage does."""
    _server(monkeypatch)
    lag, pull = 2, 3
    quotas = [4 * STAGE_QUOTA_BYTES] * 3 + [STAGE_QUOTA_BYTES]
    managers = [_manager(_stage_specs(r), _stage_mapping(r), quotas[r]) for r in range(4)]
    live = {}
    try:
        prompt_cap = min(m.max_seq_len for m in managers) - BLOCK_WIDTH
        prompts = [min(prompt_cap, 1500 + 731 * (i % 4)) for i in range(48)]
        ledger = PPAdmissionLedger(4, max(prompts)) if ledger_on else None
        waiting = deque(range(len(prompts)))
        finished = deque()
        reports = deque()
        done = 0
        failure = None
        for iteration in range(20 * len(prompts)):
            while reports and reports[0][0] <= iteration - lag:
                reported, snapshots = reports.popleft()
                for stage, wire in enumerate(snapshots, start=1):
                    if ledger_on:
                        ledger.accept(stage, reported, wire)
            held = set()
            if ledger_on:
                candidates = [Candidate(r, 0, prompts[r]) for r in waiting]
                held, _ = ledger.hold(iteration, candidates, list(waiting) + list(live), waiting)
            batch = []
            if waiting and waiting[0] not in held:
                rid = waiting[0]
                leader_request = _context_request(rid, prompts[rid])
                if _try_run_context(managers[0], leader_request):
                    waiting.popleft()
                    batch.append(rid)
                    live[rid] = [leader_request]
                else:
                    managers[0].free_resources(leader_request)
            if ledger_on:
                ledger.record(
                    [ScheduledChunk(r, 0, prompts[r], prompts[r], True) for r in batch],
                    [],
                )
            for rid in batch:
                for stage in range(1, 4):
                    request = _context_request(rid, prompts[rid])
                    if not _try_run_context(managers[stage], request):
                        managers[stage].free_resources(request)
                        failure = (stage, rid)
                        break
                    live[rid].append(request)
                if failure is not None:
                    break
                finished.append(rid)
            if failure is not None:
                break
            if finished and iteration % pull == 0:
                rid = finished.popleft()
                for manager, request in zip(managers, live.pop(rid)):
                    manager.free_resources(request)
                done += 1
            reports.append((iteration, [m.pp_admission_snapshot().to_wire() for m in managers[1:]]))
            if done == len(prompts):
                break
        if ledger_on:
            assert failure is None, failure
            assert done == len(prompts)
            assert ledger.num_held_iterations > 0
        else:
            assert failure is not None and failure[0] == 3, failure
    finally:
        for rid, requests in live.items():
            for manager, request in zip(managers, requests):
                manager.free_resources(request)
        for manager in managers:
            manager.shutdown()
