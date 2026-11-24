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
"""Kimi K3.1: a new context request waits for a prefix another request is prefilling.

Agent fan-outs reach the context servers as forked requests that share their parent's prompt
and arrive together. With ``TRTLLM_KV_V2_WAIT_FOR_INFLIGHT_PREFIX=1`` the KV cache manager keeps
such a request queued while another request still prefills the shared prefix
(``plan_context_waits``), and asks that request for a recurrent-state snapshot at the shared
depth (``_request_snapshot_point``); the waiting request then resumes from it.

These tests run that path on the family KDA + DSA hybrid cache manager of the Rubin targets
(``RubinKdaDsaHybridCacheManager``: KDA layers, and MLA layers with a DSA indexer K-cache) with the
context servers' cache settings, through the V2 scheduler in PyExecutor order. They allocate
real pools. No model runs: the test writes the recurrent states the KDA layers would write, so
that it can tell whose state a request resumes from.
"""

import gc
from collections import defaultdict
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Iterator

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm import (
    rubin_kda_dsa_cache_manager,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_dsa_cache_manager import (
    RubinKdaDsaHybridCacheManager,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_metadata import (
    KDA_PREFILL_CHUNK_SIZE,
)
from tensorrt_llm._torch.modules.kimi_kda import cache_manager as kda_cache_manager
from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import _get_forced_context_chunk_size
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler
from tensorrt_llm.llmapi.llm_args import (
    BlockReuseConfig,
    CapacitySchedulerPolicy,
    ContextChunkingPolicy,
    DeepSeekSparseAttentionConfig,
    KvCacheConfig,
    MambaStateConfig,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import _KVCache

# sm_107 only: l0_gb300 collects ops/ as a directory.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (10, 7),
    reason="Kimi K3.1 Rubin: sm_107",
)

# The context servers' block size.
TOKENS_PER_BLOCK = 64
# Prefix fingerprint unit, at its default.
UNIT = kv_cache_manager_v2._INFLIGHT_PREFIX_UNIT_BLOCKS * TOKENS_PER_BLOCK
# Nine prefix units: deep enough to wait for.
SHARED_LEN = 9 * UNIT
SHARED_BLOCKS = SHARED_LEN // TOKENS_PER_BLOCK
TAIL_LEN = UNIT
MAX_SEQ_LEN = 8192
# The context servers' token budget: a whole prompt fits one step.
STEP_BUDGET = 16384
# A budget below the shared depth: a prompt takes several steps.
CHUNK_BUDGET = 2048
# Layers 0, 2, 4 and 6 are KDA, layers 1, 3, 5 and 7 MLA with a DSA indexer. By default an NVFP4
# KV cache keeps its last two MLA layers in FP8 and needs an NVFP4 MLA layer beside them: with
# four MLA layers the manager holds both kinds, as on the context servers.
KDA_MASK = [True, False] * 4
MLA_MASK = [False, True] * 4
MLA_LAYERS = [1, 3, 5, 7]


def _text_config() -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["KimiLinearForCausalLM"],
        model_type="kimi_linear",
        num_hidden_layers=8,
        kv_lora_rank=32,
        qk_rope_head_dim=32,
        linear_attn_config={
            "head_dim": 8,
            "num_heads": 4,
            "short_conv_kernel_size": 4,
            "kda_layers": [1, 3, 5, 7],
            "full_attn_layers": [2, 4, 6, 8],
        },
        dsa_config={"dsa_layers": [2, 4, 6, 8], "index_topk_freq": 1},
        index_topk_freq=1,
        index_skip_topk_offset=1,
        dtype=torch.bfloat16,
    )


@contextmanager
def _k31_manager(
    *,
    block_reuse: bool = True,
    branch_snapshot: bool = True,
    policy: str = "per_conversation",
    ssm_dtype: torch.dtype = torch.bfloat16,
) -> Iterator[RubinKdaDsaHybridCacheManager]:
    """The context servers' manager by default: an NVFP4 KV cache, FP4 indexer keys, BF16 KDA
    state, block reuse per conversation with branch snapshots, a recurrent snapshot at each
    prompt end. ``ssm_dtype`` is the KDA state dtype (``mamba_ssm_cache_dtype``)."""
    gc.collect()
    torch.cuda.empty_cache()
    manager = RubinKdaDsaHybridCacheManager(
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_num_heads=4,
        mamba_n_groups=4,
        mamba_head_dim=8,
        mamba_num_layers=sum(KDA_MASK),
        mamba_layer_mask=KDA_MASK,
        mamba_cache_dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=ssm_dtype,
        kv_cache_config=KvCacheConfig(
            max_tokens=32768,
            tokens_per_block=TOKENS_PER_BLOCK,
            enable_block_reuse=block_reuse,
            block_reuse_config=BlockReuseConfig(
                policy=policy, max_num_turns=5, enable_branch_snapshot=branch_snapshot
            ),
            mamba_state_config=MambaStateConfig(additional_snapshot_offsets_from_end=[0]),
        ),
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=sum(MLA_MASK),
        num_kv_heads=1,
        head_dim=64,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=4,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.NVFP4,
        layer_mask=MLA_MASK,
        vocab_size=32000,
        sparse_attention_config=DeepSeekSparseAttentionConfig(
            index_n_heads=4, index_head_dim=128, index_topk=64, indexer_k_dtype="fp4"
        ),
        pretrained_config=_text_config(),
    )
    try:
        yield manager
    finally:
        manager.shutdown()
        del manager
        gc.collect()
        torch.cuda.empty_cache()


def _request(
    request_id: int,
    shared_len: int = SHARED_LEN,
    tail_len: int = TAIL_LEN,
    cache_salt: str | None = None,
) -> LlmRequest:
    """``shared_len`` prompt tokens every request starts with, then a tail of its own."""
    tail = [10000 + 5000 * request_id + i for i in range(tail_len)]
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=4,
        input_tokens=list(range(shared_len)) + tail,
        sampling_config=SamplingConfig(),
        is_streaming=False,
        cache_salt=cache_salt,
    )


@pytest.fixture
def wait_for_inflight_prefix(monkeypatch):
    """TRTLLM_KV_V2_WAIT_FOR_INFLIGHT_PREFIX=1, read at import; the manager and the scheduler
    read the constant when they are built."""
    monkeypatch.setattr(kv_cache_manager_v2, "_WAIT_FOR_INFLIGHT_PREFIX", True)


def _midchunk_snapshots(monkeypatch, enabled: bool) -> None:
    """TRTLLM_KDA_MIDCHUNK_SNAPSHOT, on for the context servers; read at import, so set before the
    manager is built. The manager also asks this family's KDA prefill whether it stores state
    snapshots; no prefill runs here, the test writes the states."""
    if enabled and not hasattr(_KVCache, "acquire_ssm_snapshot_slot"):
        pytest.skip("the KV cache manager V2 C++ backend has no pending SSM snapshot pages")
    monkeypatch.setattr(kda_cache_manager, "_KDA_MIDCHUNK_SNAPSHOT", enabled)
    monkeypatch.setattr(
        rubin_kda_dsa_cache_manager, "_kda_prefill_stores_snapshots", lambda: enabled
    )


class _Executor:
    """Scheduler passes over the manager in PyExecutor order, standing in for the KDA layers.

    A step writes a fresh recurrent state for each request: into the snapshot slot the manager
    planned inside the step, if any, and into the request's live slot for the step end.
    ``states`` keeps each by (request id, prompt position); ``resumed`` keeps what a request's
    live slot held before its first step.
    """

    def __init__(self, manager: RubinKdaDsaHybridCacheManager, max_num_tokens: int):
        self.manager = manager
        self.scheduler = KVCacheV2Scheduler(
            max_batch_size=4,
            max_num_tokens=max_num_tokens,
            kv_cache_manager=manager,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
            # The executor's chunking for a hybrid model with block reuse.
            ctx_chunk_config=(ContextChunkingPolicy.FORCE_CHUNK, TOKENS_PER_BLOCK),
            # Without the prefix-aware skip only the prefix wait defers a request.
            enable_prefix_aware_scheduling=False,
        )
        self.chunk_starts = defaultdict(list)
        self.midchunk_points = defaultdict(list)
        self.states = {}
        self.resumed = {}
        self.claimed = {}

    def step(self, requests, inflight=frozenset(), run: bool = True) -> list[int]:
        """One scheduler pass; with ``run`` its context steps run too. Returns the admitted ids."""
        pending = [r for r in requests if r.context_remaining_length > 0]
        self.manager.prepare_expect_snapshot_points(pending)
        admitted = self.scheduler.schedule_request(pending, set(inflight)).context_requests
        self.claimed.update((r.py_request_id, r) for r in admitted)
        if run:
            self.finish(self.prepare(admitted))
        return [r.py_request_id for r in admitted]

    def prepare(self, admitted) -> list:
        """prepare_resources, then the KDA layers' state writes of the step."""
        batch = ScheduledRequests()
        for request in admitted:
            batch.append_context_request(request)
        self.manager.prepare_resources(batch)
        ids = [r.py_request_id for r in admitted]
        live = self.manager.get_state_indices(ids, [False] * len(ids))
        # Fetched by the KDA metadata; a planned snapshot that is not fetched is never committed.
        snapshots = {
            index: (slot, offset)
            for index, slot, offset in self.manager.get_kda_midchunk_snapshots(ids)
        }
        torch.cuda.synchronize()
        for index, request in enumerate(admitted):
            request_id, start = request.py_request_id, request.context_current_position
            if not self.chunk_starts[request_id]:
                self.resumed[request_id] = self._read(live[index])
            self.chunk_starts[request_id].append(start)
            if index in snapshots:
                slot, offset = snapshots[index]
                self.midchunk_points[request_id].append(start + offset)
                self.states[(request_id, start + offset)] = self._write(slot)
            end = start + request.context_chunk_size
            self.states[(request_id, end)] = self._write(live[index])
        torch.cuda.synchronize()
        return [batch, admitted]

    def finish(self, prepared: list) -> None:
        """The end of a step: the context advances, then the cache commits."""
        batch, admitted = prepared
        for request in admitted:
            request.move_to_next_context_chunk()
        self.manager.update_context_resources(batch)

    def _layers(self):
        return zip(self.manager.all_ssm_states, self.manager.all_conv_states)

    def _read(self, slot: int) -> list:
        return [(ssm[slot].clone(), conv[slot].clone()) for ssm, conv in self._layers()]

    def _write(self, slot: int) -> list:
        state = []
        for ssm, conv in self._layers():
            ssm_state = torch.randn(ssm[slot].shape, device=ssm.device).to(ssm.dtype)
            conv_state = torch.randn(conv[slot].shape, device=conv.device).to(conv.dtype)
            ssm[slot], conv[slot] = ssm_state, conv_state
            state.append((ssm_state, conv_state))
        return state

    def free_all(self) -> None:
        for request_id, request in self.claimed.items():
            if request_id in self.manager.kv_cache_map:
                self.manager.free_resources(request)


def _same_state(a: list, b: list) -> bool:
    return len(a) == len(b) and all(
        torch.equal(ssm_a, ssm_b) and torch.equal(conv_a, conv_b)
        for (ssm_a, conv_a), (ssm_b, conv_b) in zip(a, b)
    )


@pytest.mark.parametrize(
    "ssm_dtype", [torch.bfloat16, torch.float32], ids=["bf16-state", "fp32-state"]
)
@pytest.mark.parametrize("midchunk", [True, False], ids=["midchunk-snapshot", "split-step"])
def test_sibling_waits_for_the_owner_and_resumes_from_its_shared_prefix(
    monkeypatch, wait_for_inflight_prefix, midchunk, ssm_dtype
):
    """Owner and sibling arrive in the same pass. The sibling waits, and the owner is asked for a
    snapshot at the shared depth before its first step is sized: its step takes the snapshot
    inside (mid-chunk snapshots, as on the context servers) or ends there. The sibling then
    resumes at that depth from the owner's recurrent state (BF16 or FP32, bit for bit) and on the
    owner's MLA pages."""
    _midchunk_snapshots(monkeypatch, midchunk)
    torch.manual_seed(0)
    with _k31_manager(ssm_dtype=ssm_dtype) as manager:
        assert {ssm.dtype for ssm in manager.all_ssm_states} == {ssm_dtype}
        assert manager.indexer_k_cache_local_layer_mask == MLA_MASK
        assert manager._kda_midchunk_snapshot_alignment == (
            KDA_PREFILL_CHUNK_SIZE if midchunk else None
        )
        executor = _Executor(manager, STEP_BUDGET)
        owner, sibling = _request(1), _request(2)
        try:
            assert executor.step([owner, sibling]) == [1]
            assert manager._requested_snapshot_points == {1: {SHARED_LEN}}
            if midchunk:
                assert owner.context_remaining_length == 0
                assert executor.midchunk_points[1] == [SHARED_LEN]
            else:
                assert owner.context_current_position == SHARED_LEN
            for _ in range(8):
                if owner.context_remaining_length == sibling.context_remaining_length == 0:
                    break
                executor.step([owner, sibling])
            assert owner.context_remaining_length == sibling.context_remaining_length == 0

            assert executor.chunk_starts[1] == ([0] if midchunk else [0, SHARED_LEN])
            assert executor.chunk_starts[2] == [SHARED_LEN]
            index = manager._inflight_prefix
            assert index.deferrals_total == index.points_total == index.points_taken_total == 1
            assert _same_state(executor.resumed[2], executor.states[(1, SHARED_LEN)])
            assert not _same_state(executor.resumed[2], executor.states[(1, owner.prompt_len)])
            # The sibling reuses the owner's pages on every MLA layer, NVFP4 or FP8, and with them
            # the indexer keys, which live in the same pages.
            for layer in MLA_LAYERS:
                owner_pages, sibling_pages = manager.get_batch_cache_indices(
                    [1, 2], layer_idx=layer
                )
                assert owner_pages[:SHARED_BLOCKS] == sibling_pages[:SHARED_BLOCKS]
                assert owner_pages[SHARED_BLOCKS] != sibling_pages[SHARED_BLOCKS]
        finally:
            executor.free_all()


@pytest.mark.parametrize(
    ("manager_config", "enabled", "sibling_salt", "waits"),
    [
        (dict(), True, None, True),
        (dict(policy="per_request"), True, None, True),
        # The owner's snapshot at the shared depth is a branch-snapshot point.
        (dict(branch_snapshot=False), True, None, False),
        (dict(block_reuse=False, branch_snapshot=False), True, None, False),
        (dict(), False, None, False),
        # Requests of another reuse scope share no cache.
        (dict(), True, "tenant-b", False),
    ],
    ids=[
        "context-servers",
        "per-request",
        "no-branch-snapshot",
        "no-block-reuse",
        "wait-off",
        "other-cache-salt",
    ],
)
def test_siblings_wait_only_where_they_can_share_the_inflight_prefix(
    monkeypatch, manager_config, enabled, sibling_salt, waits
):
    monkeypatch.setattr(kv_cache_manager_v2, "_WAIT_FOR_INFLIGHT_PREFIX", enabled)
    _midchunk_snapshots(monkeypatch, True)
    with _k31_manager(**manager_config) as manager:
        executor = _Executor(manager, STEP_BUDGET)
        try:
            admitted = executor.step([_request(1), _request(2, cache_salt=sibling_salt)])
            assert sorted(admitted) == ([1] if waits else [1, 2])
            assert manager._requested_snapshot_points == ({1: {SHARED_LEN}} if waits else {})
        finally:
            executor.free_all()


def test_a_released_owner_no_longer_holds_a_sibling_back(monkeypatch, wait_for_inflight_prefix):
    _midchunk_snapshots(monkeypatch, True)
    with _k31_manager() as manager:
        executor = _Executor(manager, CHUNK_BUDGET)
        owner, sibling = _request(1), _request(2)
        try:
            assert executor.step([owner]) == [1]
            assert executor.step([owner, sibling]) == [1]
            assert manager._requested_snapshot_points == {1: {SHARED_LEN}}
            assert owner.context_current_position < SHARED_LEN

            # The owner is cancelled short of the shared depth and releases its cache.
            manager.free_resources(owner)
            assert manager._requested_snapshot_points == {}
            assert executor.step([sibling]) == [2]
            # Nothing of the shared prefix was committed: the sibling prefills all of it.
            assert executor.chunk_starts[2] == [0]
        finally:
            executor.free_all()


@pytest.mark.parametrize("owner_state", ["before", "passed", "step-in-flight"])
def test_sibling_waits_only_while_the_owner_can_snapshot_the_shared_depth(
    monkeypatch, wait_for_inflight_prefix, owner_state
):
    """The owner can take the snapshot until it has computed past the shared depth; after that
    the sibling no longer waits and prefills the prefix itself."""
    _midchunk_snapshots(monkeypatch, True)
    with _k31_manager() as manager:
        executor = _Executor(manager, CHUNK_BUDGET)
        owner = _request(1, tail_len=3 * CHUNK_BUDGET + CHUNK_BUDGET // 2 - SHARED_LEN)
        sibling = _request(2)
        try:
            for _ in range(2):
                assert executor.step([owner]) == [1]
            assert owner.context_current_position < SHARED_LEN
            inflight, in_flight_step = set(), None
            if owner_state == "passed":
                assert executor.step([owner]) == [1]
                assert owner.context_current_position > SHARED_LEN
            elif owner_state == "step-in-flight":
                # The owner's next step, across the shared depth, was prepared and runs while
                # the scheduler plans the next pass (overlap scheduler).
                assert executor.step([owner], run=False) == [1]
                in_flight_step = executor.prepare([owner])
                inflight = {owner.py_request_id}

            admitted = executor.step([owner, sibling], inflight=inflight, run=False)
            if owner_state == "passed":
                assert 2 in admitted
                assert manager._requested_snapshot_points == {}
                assert sibling.context_current_position == 0
                return
            assert 2 not in admitted
            assert manager._requested_snapshot_points == {1: {SHARED_LEN}}
            if owner_state == "step-in-flight":
                # The snapshot request checks the owner's computed position only, so the sibling
                # also waits behind a step already prepared across the shared depth. That step
                # passes the depth without the snapshot, and the sibling stops waiting. The
                # context servers run without the overlap scheduler: no step is in flight while
                # the scheduler plans the next pass.
                executor.finish(in_flight_step)
                assert owner.context_current_position > SHARED_LEN
                assert 2 in executor.step([owner, sibling], run=False)
                assert sibling.context_current_position == 0
                assert manager._inflight_prefix.points_taken_total == 0
            assert manager._inflight_prefix.deferrals_total == 1
        finally:
            executor.free_all()


def test_same_pass_siblings_wait_for_the_first_one(monkeypatch, wait_for_inflight_prefix):
    """Three requests with one prefix arrive in the same pass. The first does not wait and gets
    the snapshot point before its first step is sized, so that step passes the point and takes
    the snapshot inside; the other two wait for it."""
    _midchunk_snapshots(monkeypatch, True)
    with _k31_manager() as manager:
        first, second, third = _request(1), _request(2), _request(3)
        manager.prepare_expect_snapshot_points([first, second, third])
        assert manager.plan_context_waits([first, second, third]) == {2, 3}
        assert manager._requested_snapshot_points == {1: {SHARED_LEN}}
        assert first.expect_snapshot_points == [SHARED_LEN, first.prompt_len]
        assert first.py_midchunk_pass_point == SHARED_LEN
        assert _get_forced_context_chunk_size(first) == first.prompt_len
        assert manager._inflight_prefix.deferrals_total == 2


def test_waits_for_the_request_sharing_the_deepest_prefix(monkeypatch, wait_for_inflight_prefix):
    """Of two requests prefilling a prefix of the waiting request, it waits for the one sharing
    more of it, and only that one is asked for a snapshot."""
    _midchunk_snapshots(monkeypatch, True)
    deep_len = SHARED_LEN + 2 * UNIT
    with _k31_manager() as manager:
        shallow, deep = _request(1), _request(2, shared_len=deep_len)
        waiter = _request(3, shared_len=deep_len)
        manager.prepare_expect_snapshot_points([shallow, deep, waiter])
        try:
            # Both claimed their caches in an earlier pass.
            assert manager.prepare_context(shallow) and manager.prepare_context(deep)
            assert manager.plan_context_waits([shallow, deep, waiter]) == {3}
            assert manager._requested_snapshot_points == {2: {deep_len}}
        finally:
            for request in (shallow, deep):
                manager.free_resources(request)
