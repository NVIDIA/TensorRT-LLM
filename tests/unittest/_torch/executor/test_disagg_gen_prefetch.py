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
"""Disaggregated generation KV prefetch (TRTLLM_DISAGG_GEN_PREFETCH_REQUESTS).

The KV index pool part (``KVCacheManagerV2.max_admissible_sequences``) is
covered next to the existing capacity formula in
``kv_cache/test_kvcm2_integration.py``.
"""

from collections import deque
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import tensorrt_llm._torch.pyexecutor.model_engine as model_engine_module
import tensorrt_llm._torch.speculative.dflash as dflash_module
from tensorrt_llm._torch.pyexecutor import _util
from tensorrt_llm._torch.pyexecutor.disagg_gen_prefetch import (
    DISAGG_GEN_PREFETCH_REQUESTS_ENV,
    PRIORITIZE_FIRST_TOKEN_ENV,
    get_disagg_gen_prefetch_requests,
    resolve_disagg_gen_prefetch_requests,
)
from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import NoFreeSlotsError
from tensorrt_llm._torch.pyexecutor.scheduler.waiting_queue import FCFSWaitingQueue
from tensorrt_llm._torch.pyexecutor.seq_slot_manager import SeqSlotManager
from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata, DFlashWorker
from tensorrt_llm.bindings.executor import RequestType
from tensorrt_llm.executor.request import DEFAULT_REQUEST_PRIORITY
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_ELIGIBLE = dict(is_disagg=True, enable_attention_dp=False, has_pp=False)


@pytest.fixture(autouse=True)
def _scheduler_option_unset(monkeypatch):
    monkeypatch.delenv(PRIORITIZE_FIRST_TOKEN_ENV, raising=False)


@pytest.fixture
def prefetch_env(monkeypatch):
    def _set(value):
        if value is None:
            monkeypatch.delenv(DISAGG_GEN_PREFETCH_REQUESTS_ENV, raising=False)
        else:
            monkeypatch.setenv(DISAGG_GEN_PREFETCH_REQUESTS_ENV, value)

    return _set


@pytest.mark.parametrize("raw, expected", [(None, 0), ("", 0), ("0", 0), ("1", 1), (" 3 ", 3)])
def test_env_parsing(prefetch_env, raw, expected):
    prefetch_env(raw)
    assert get_disagg_gen_prefetch_requests() == expected


@pytest.mark.parametrize("raw", ["-1", "one", "1.5"])
def test_env_rejects_invalid_values(prefetch_env, raw):
    prefetch_env(raw)
    with pytest.raises(ValueError):
        get_disagg_gen_prefetch_requests()


@pytest.mark.parametrize(
    "override",
    [
        {"is_disagg": False},
        {"enable_attention_dp": True},
        {"has_pp": True},
        {"is_hybrid": True},
        {"has_mrope_delta_cache": True},
        {"kv_cache_manager_is_v2": False},
    ],
)
def test_resolve_ignores_unsupported_executors(prefetch_env, override):
    prefetch_env("2")
    kwargs = {**_ELIGIBLE, **override}
    assert resolve_disagg_gen_prefetch_requests(**kwargs) == 0


def test_resolve_gate(prefetch_env):
    prefetch_env(None)
    assert resolve_disagg_gen_prefetch_requests(**_ELIGIBLE) == 0
    prefetch_env("2")
    assert resolve_disagg_gen_prefetch_requests(**_ELIGIBLE) == 2


@pytest.mark.parametrize("value, prefetch", [("1", 0), ("0", 2), ("true", 2)])
def test_resolve_follows_the_prioritize_first_token_scheduler_option(
    prefetch_env, monkeypatch, value, prefetch
):
    # The scheduler turns the option on only for exactly "1"; prefetch is
    # turned off in exactly the same cases.
    prefetch_env("2")
    monkeypatch.setenv(PRIORITIZE_FIRST_TOKEN_ENV, value)
    assert resolve_disagg_gen_prefetch_requests(**_ELIGIBLE) == prefetch


def _engine_seat_pool(monkeypatch, *, cache_transceiver_config, max_batch_size=2):
    """Run the real ``PyTorchModelEngine.__init__`` up to the seat-pool sizing.

    Returns ``(max_num_seq_slots, disagg_gen_prefetch_requests)``. The gates
    that need a real model are pinned to an aggregated-style non-ADP, non-PP
    V2 engine; ``compute_max_num_sequences`` is the real one, so the pool is
    ``max_batch_size`` plus whatever the prefetch term adds.
    """
    for name in (
        "should_enable_adp_dummy_fixes",
        "should_enable_scheduler_aware_adp_dummy",
        "should_enable_non_overlap_adp_forward_intent",
        "should_enable_overlap_headroom",
    ):
        monkeypatch.setattr(_util, name, Mock(return_value=False))
    monkeypatch.setattr(_util, "resolved_kv_cache_manager_is_v2", Mock(return_value=True))
    for name in ("_configure_deep_gemm_pdl", "create_input_processor"):
        monkeypatch.setattr(model_engine_module, name, Mock())
    monkeypatch.setattr(
        model_engine_module, "resolve_mrope_position_deltas_cache", lambda model: None
    )
    monkeypatch.setattr(model_engine_module, "is_hybrid_linear", lambda config: False)
    monkeypatch.setattr(PyTorchModelEngine, "_validate_breakable_cuda_graph_compatibility", Mock())
    monkeypatch.setattr(PyTorchModelEngine, "__del__", lambda self: None)
    # The seat pool is sized before multimodal scheduling is set up; stop there.
    monkeypatch.setattr(
        model_engine_module.MultimodalItemScheduler,
        "maybe_create",
        Mock(side_effect=RuntimeError("seat pool sized")),
    )
    model = torch.nn.Module()
    model.model_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(model_type="llama"),
        sparse_attention_config=None,
    )
    llm_args = SimpleNamespace(
        encode_only=False,
        mm_encoder_only=False,
        get_runtime_sizes=lambda: (1, 16, 64, max_batch_size),
        encoder_max_batch_size=None,
        encoder_max_num_tokens=None,
        enable_in_graph_sampling=False,
        enable_return_routed_experts=False,
        multimodal_config=SimpleNamespace(video_pruning_rate=None),
        checkpoint_format="HF",
        trust_remote_code=False,
        disable_overlap_scheduler=False,
        kv_cache_config=KvCacheConfig(),
        sparse_attention_config=None,
        cache_transceiver_config=cache_transceiver_config,
    )
    engine = PyTorchModelEngine.__new__(PyTorchModelEngine)
    with pytest.raises(RuntimeError, match="seat pool sized"):
        PyTorchModelEngine.__init__(
            engine,
            model_path="dummy",
            mapping=Mapping(),
            model=model,
            llm_args=llm_args,
            checkpoint_loader=Mock(),
        )
    return engine.max_num_seq_slots, engine.disagg_gen_prefetch_requests


_NIXL = SimpleNamespace(backend="NIXL")


@pytest.mark.parametrize("max_batch_size", [1, 2])
def test_engine_seat_pool_is_unchanged_by_default(monkeypatch, prefetch_env, max_batch_size):
    prefetch_env(None)
    for transceiver in (None, _NIXL):
        assert _engine_seat_pool(
            monkeypatch, cache_transceiver_config=transceiver, max_batch_size=max_batch_size
        ) == (max_batch_size, 0)


@pytest.mark.parametrize("prefetch", [1, 3])
@pytest.mark.parametrize("max_batch_size", [1, 2])
def test_engine_seat_pool_adds_the_prefetch_seats_for_disagg(
    monkeypatch, prefetch_env, max_batch_size, prefetch
):
    prefetch_env(str(prefetch))
    assert _engine_seat_pool(
        monkeypatch, cache_transceiver_config=_NIXL, max_batch_size=max_batch_size
    ) == (max_batch_size + prefetch, prefetch)


def test_engine_seat_pool_ignores_prefetch_without_a_transceiver(monkeypatch, prefetch_env):
    prefetch_env("1")
    assert _engine_seat_pool(monkeypatch, cache_transceiver_config=None) == (2, 0)
    # A transceiver config without a backend is not disaggregated either.
    assert _engine_seat_pool(
        monkeypatch, cache_transceiver_config=SimpleNamespace(backend=None)
    ) == (2, 0)


def test_engine_seat_pool_ignores_prefetch_with_prioritize_first_token(monkeypatch, prefetch_env):
    prefetch_env("1")
    monkeypatch.setenv(PRIORITIZE_FIRST_TOKEN_ENV, "1")
    assert _engine_seat_pool(monkeypatch, cache_transceiver_config=_NIXL) == (2, 0)


def _seq_req(request_id, *, transmission_complete=False):
    """The attributes ``SeqSlotManager.prepare_resources`` reads and writes."""
    return SimpleNamespace(
        request_id=request_id,
        seq_slot=None,
        py_seq_slot=None,
        return_perf_metrics=False,
        is_disagg_generation_init_state=False,
        is_disagg_generation_transmission_complete=transmission_complete,
    )


def _scheduled(*requests):
    return SimpleNamespace(all_requests=lambda: list(requests))


def _overlap_retire_turnover(pool, max_batch_size):
    """Seat ``max_batch_size`` decodes, then schedule a prefetched request.

    Under the overlap scheduler the request that ended on the previous step is
    GENERATION_TO_COMPLETE: it is not scheduled but still owns its sequence
    slot until the next iteration frees it. With prefetch, the request whose KV
    already landed is scheduled in that same iteration.
    """
    manager = SeqSlotManager(max_num_sequences=pool)
    running = [_seq_req(i) for i in range(max_batch_size)]
    manager.prepare_resources(_scheduled(*running))
    prefetched = _seq_req(100, transmission_complete=True)
    manager.prepare_resources(_scheduled(*running[1:], prefetched))
    return manager, running, prefetched


@pytest.mark.parametrize("max_batch_size", [1, 2, 8])
def test_a_seat_pool_of_max_batch_size_cannot_seat_the_prefetched_request(max_batch_size):
    with pytest.raises(NoFreeSlotsError):
        _overlap_retire_turnover(max_batch_size, max_batch_size)


@pytest.mark.parametrize("prefetch", [1, 2])
@pytest.mark.parametrize("max_batch_size", [1, 2, 8])
def test_the_widened_seat_pool_seats_the_prefetched_request(max_batch_size, prefetch):
    pool = max_batch_size + prefetch
    manager, running, prefetched = _overlap_retire_turnover(pool, max_batch_size)
    assert 0 <= prefetched.py_seq_slot < pool
    assert prefetched.py_seq_slot not in {r.py_seq_slot for r in running}
    # The retiring request's slot is returned and reused on later turnovers.
    manager.free_resources(running[0])
    assert len(manager.slot_manager.free_slots) == pool - max_batch_size


def _executor(max_num_active_requests=1, prefetch=0, attention_dp=False):
    ex = object.__new__(PyExecutor)
    ex.enable_attention_dp = attention_dp
    ex.max_num_active_requests = max_num_active_requests
    ex.disagg_gen_prefetch_requests = prefetch
    ex.is_benchmark_disagg = False
    ex._benchmark_fill_phase_active = False
    ex.dist = Mock()
    ex.dist.tp_size = 4
    return ex


@pytest.mark.parametrize("mbs", [1, 2, 3])
def test_admission_capacity_gate(mbs):
    assert _executor(mbs, prefetch=0)._get_request_admission_capacity() == mbs
    assert _executor(mbs, prefetch=1)._get_request_admission_capacity() == mbs + 1
    assert _executor(mbs, prefetch=2)._get_request_admission_capacity() == mbs + 2


def test_admission_capacity_unchanged_without_the_attribute():
    ex = _executor(2)
    del ex.disagg_gen_prefetch_requests
    assert ex._get_request_admission_capacity() == 2


def test_admission_capacity_ignores_prefetch_under_attention_dp():
    ex = _executor(2, prefetch=1, attention_dp=True)
    assert ex._get_request_admission_capacity() == 4 * 2


def _item(req_id, request_type, child_req_ids=None):
    return RequestQueueItem(
        req_id,
        SimpleNamespace(request_type=request_type, priority=DEFAULT_REQUEST_PRIORITY),
        child_req_ids=child_req_ids,
    )


GEN = RequestType.REQUEST_TYPE_GENERATION_ONLY
CTX = RequestType.REQUEST_TYPE_CONTEXT_ONLY


def _pop(ex, items, num_active):
    queue = FCFSWaitingQueue()
    queue.add_requests(items)
    popped = ex._pop_from_waiting_queue(queue, num_active)
    return [i.id for i in popped], [i.id for i in queue]


def test_gate_off_pops_up_to_max_batch_size():
    ex = _executor(1, prefetch=0)
    assert _pop(ex, [_item(1, GEN), _item(2, GEN)], 1) == ([], [1, 2])
    assert _pop(ex, [_item(1, GEN), _item(2, GEN)], 0) == ([1], [2])


def test_prefetch_admits_one_more_generation_request():
    ex = _executor(1, prefetch=1)
    # One request decoding: the next one is admitted to receive its KV.
    assert _pop(ex, [_item(1, GEN), _item(2, GEN)], 1) == ([1], [2])
    # Idle: one regular slot and one prefetch slot.
    assert _pop(ex, [_item(1, GEN), _item(2, GEN), _item(3, GEN)], 0) == ([1, 2], [3])
    # Full including the prefetch slot.
    assert _pop(ex, [_item(1, GEN)], 2) == ([], [1])


def test_prefetch_slots_take_only_generation_only_requests():
    ex = _executor(1, prefetch=2)
    # A context request cannot use a prefetch slot; it and everything after
    # it go back to the front of the queue in order.
    assert _pop(ex, [_item(1, CTX), _item(2, GEN)], 1) == ([], [1, 2])
    assert _pop(ex, [_item(1, CTX), _item(2, CTX), _item(3, GEN)], 0) == ([1], [2, 3])
    assert _pop(ex, [_item(1, GEN), _item(2, CTX), _item(3, GEN)], 0) == ([1], [2, 3])
    assert _pop(ex, [_item(1, GEN), _item(2, GEN), _item(3, CTX)], 0) == ([1, 2], [3])
    # Requests with children never use a prefetch slot.
    assert _pop(ex, [_item(1, GEN, child_req_ids=[9])], 1) == ([], [1])


class _FakeDFlashWorker:
    """The DFlash worker state ``DFlashSpecMetadata.prepare`` reads and writes.

    ``_assign_slot`` is the real method; ``_write_ctx_len`` keeps the host
    mirror (``ctx_len``) that the real one keeps next to the device tensor.
    """

    def __init__(self, max_batch_size):
        self._ctx_buf_inited = True
        self._req_to_slot = {}
        self._req_ctx_pos = {}
        self._free_slots = deque(range(max_batch_size))
        self._dummy_slot = max_batch_size
        self._graph_dummy_id_floor = 1 << 62
        self._batch_to_slot = torch.zeros(max_batch_size, dtype=torch.long)
        self.ctx_len = [0] * (max_batch_size + 1)
        self._assign_slot = MethodType(DFlashWorker._assign_slot, self)

    def _write_ctx_len(self, updates, base_updates=None):
        for slot, value in updates.items():
            self.ctx_len[slot] = value


def _drafter_slots(monkeypatch, max_batch_size, schedule, stored_context=500):
    """Run ``DFlashSpecMetadata.prepare`` over a schedule of generation batches.

    After each step, every request in the batch is given ``stored_context``
    tokens of drafter context, as the real step does. Returns, per step, each
    request's slot and the context length it found when the step started.
    """
    monkeypatch.setattr(dflash_module, "prefer_pinned", lambda: False)
    worker = _FakeDFlashWorker(max_batch_size)
    metadata = object.__new__(DFlashSpecMetadata)
    metadata.batch_indices_cuda = torch.zeros(max_batch_size, dtype=torch.int)
    metadata._dflash_worker = worker
    seen = []
    for batch in schedule:
        metadata.request_ids = list(batch)
        metadata.num_generations = len(batch)
        metadata.prepare()
        slots = worker._batch_to_slot[: len(batch)].tolist()
        seen.append({rid: (slot, worker.ctx_len[slot]) for rid, slot in zip(batch, slots)})
        for slot in slots:
            worker.ctx_len[slot] = stored_context
    return seen


def test_prefetch_turnover_gives_the_drafter_the_slot_state_of_a_plain_turnover(monkeypatch):
    """A prefetched request starts from the same drafter slot state as without prefetch.

    Request 11 ends while request 12 keeps decoding. Without prefetch the next request
    (13) is scheduled after at least one more step (here, one step with
    request 12 alone); with prefetch it is scheduled on the very next step.
    DFlash frees the slots of requests that left the batch before it assigns
    slots to new ones, so request 13 gets the same slot, reset to zero
    context, either way, and request 12 keeps its own slot and context.
    (Request id 0 is the attention-DP dummy, so real ids start above it.)
    """
    today = _drafter_slots(monkeypatch, 2, [[11, 12], [12], [12, 13]])
    prefetch = _drafter_slots(monkeypatch, 2, [[11, 12], [12, 13]])
    assert today[-1] == prefetch[-1]
    slot_13, context_13 = prefetch[-1][13]
    assert context_13 == 0
    assert slot_13 == prefetch[0][11][0]  # the retired request's slot
    assert slot_13 != 2  # not the shared dummy slot
    assert prefetch[-1][12] == (prefetch[0][12][0], 500)


def test_prefetch_turnover_at_max_batch_size_one_reuses_the_slot_from_zero(monkeypatch):
    # With one slot the sequence of drafter batches is [11] then [12] with and
    # without prefetch; request 12 takes request 11's slot at zero context.
    seen = _drafter_slots(monkeypatch, 1, [[11], [12]])
    assert seen[1][12] == (seen[0][11][0], 0)
