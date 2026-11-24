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
"""PyExecutor admission with extra receive slots (TRTLLM_DISAGG_GEN_EXTRA_SLOTS).

A disaggregated generation server may hold up to X generation-only requests beyond
max_batch_size while their KV cache arrives. No GPU required.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.request_utils import (
    get_generation_only_from_waiting_queue,
    is_generation_only_queue_item,
)
from tensorrt_llm._torch.pyexecutor.scheduler.waiting_queue import FCFSWaitingQueue
from tensorrt_llm.bindings.executor import RequestType

pytestmark = pytest.mark.cpu_only

GEN_ONLY = RequestType.REQUEST_TYPE_GENERATION_ONLY
CTX_ONLY = RequestType.REQUEST_TYPE_CONTEXT_ONLY


def item(request_id: int, request_type=GEN_ONLY, child_req_ids=None) -> RequestQueueItem:
    return RequestQueueItem(
        request_id,
        SimpleNamespace(request_type=request_type),
        child_req_ids=child_req_ids,
    )


def queue(*items) -> FCFSWaitingQueue:
    return FCFSWaitingQueue(list(items))


def executor(
    *,
    max_batch_size=2,
    requested=2,
    transceiver=True,
    attention_dp=False,
    has_pp=False,
    benchmark=False,
    admissible=None,
):
    ex = object.__new__(PyExecutor)
    ex.enable_attention_dp = attention_dp
    ex.dist = SimpleNamespace(tp_size=1, has_pp=has_pp)
    ex.max_num_active_requests = max_batch_size
    ex.is_benchmark_disagg = benchmark
    ex.scheduler = SimpleNamespace(gen_extra_slots=requested)
    ex.kv_cache_transceiver = object() if transceiver else None
    ex.kv_cache_manager = SimpleNamespace(
        max_admissible_sequences=(2 * max_batch_size if admissible is None else admissible)
    )
    ex._gen_extra_slots = ex._resolve_gen_extra_slots()
    return ex


def popped(ex, waiting, num_active):
    requests, placement = ex._pop_from_waiting_queue(waiting, num_active)
    assert placement == {}
    return [r.id for r in requests]


# ---------------------------------------------------------------------------
# Resolution of the switch
# ---------------------------------------------------------------------------


def test_off_unless_requested():
    assert executor(requested=0)._gen_extra_slots == 0


def test_on_for_a_disaggregated_server():
    assert executor(requested=2)._gen_extra_slots == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(transceiver=False),
        dict(attention_dp=True),
        dict(has_pp=True),
        dict(benchmark=True),
    ],
    ids=["no-transceiver", "attention-dp", "pipeline-parallel", "benchmark-fill"],
)
def test_ignored_where_unsupported(kwargs):
    assert executor(requested=2, **kwargs)._gen_extra_slots == 0


@pytest.mark.parametrize(("admissible", "expected"), [(4, 2), (3, 1), (2, 0), (100, 5)])
def test_capped_by_the_kv_index_slots(admissible, expected):
    assert executor(requested=5, admissible=admissible)._gen_extra_slots == expected


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------


def test_off_admits_up_to_max_batch_size():
    ex = executor(requested=0)
    waiting = queue(*(item(i) for i in range(1, 6)))
    assert popped(ex, waiting, 0) == [1, 2]
    assert popped(ex, waiting, 2) == []
    assert [r.id for r in waiting] == [3, 4, 5]


@pytest.mark.parametrize(
    ("num_active", "expected"), [(0, [1, 2, 3, 4]), (2, [1, 2]), (3, [1]), (4, [])]
)
def test_on_admits_generation_only_requests_past_max_batch_size(num_active, expected):
    ex = executor(max_batch_size=2, requested=2)
    waiting = queue(*(item(i) for i in range(1, 7)))
    assert popped(ex, waiting, num_active) == expected
    assert [r.id for r in waiting] == [i for i in range(1, 7) if i not in expected]


def test_extra_admission_keeps_queue_order():
    """A context-only (or a parent with children) item at the head stops the extra admission,
    so nothing behind it is admitted ahead of it."""
    ex = executor(max_batch_size=1, requested=3)
    waiting = queue(item(1), item(2), item(3, CTX_ONLY), item(4))
    assert popped(ex, waiting, 0) == [1, 2]
    assert [r.id for r in waiting] == [3, 4]

    waiting = queue(item(5), item(6, child_req_ids=[61]), item(7))
    assert popped(ex, waiting, 1) == [5]
    assert [r.id for r in waiting] == [6, 7]


def test_no_extra_admission_in_warmup():
    ex = executor(max_batch_size=2, requested=2)
    ex.is_warmup = True
    waiting = queue(*(item(i) for i in range(1, 5)))
    assert popped(ex, waiting, 0) == [1, 2]


def test_generation_only_helper():
    waiting = queue(item(1), item(2), item(3, CTX_ONLY))
    assert is_generation_only_queue_item(item(1))
    assert not is_generation_only_queue_item(item(3, CTX_ONLY))
    assert not is_generation_only_queue_item(RequestQueueItem(9, None))
    assert [r.id for r in get_generation_only_from_waiting_queue(waiting, 5)] == [1, 2]
    assert [r.id for r in waiting] == [3]
    assert get_generation_only_from_waiting_queue(waiting, 5) == []
    assert get_generation_only_from_waiting_queue(queue(item(4)), 0) == []
