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
"""The executor side of the KV cache host helper: which forwards open a window, which requests are
claimed ahead, and the release note of gen_init_pause_after_release. No GPU required."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

pytestmark = pytest.mark.cpu_only


def _request(request_id, state=LlmRequestState.CONTEXT_INIT, first_chunk=True, dummy=False):
    return SimpleNamespace(
        py_request_id=request_id,
        state=state,
        is_first_context_chunk=first_chunk,
        is_dummy_request=dummy,
        is_generation_only_request=state == LlmRequestState.GENERATION_IN_PROGRESS,
    )


def _executor(active_requests=()):
    ex = object.__new__(PyExecutor)
    ex._kv_host_helper = Mock()
    ex.kv_cache_manager = Mock()
    ex.active_requests = list(active_requests)
    ex._is_warmup = False
    ex._forward_step_impl = Mock(return_value={"logits": None})
    return ex


def _batch(context=(), generation=()):
    return SimpleNamespace(context_requests=list(context), generation_requests=list(generation))


def test_a_context_only_forward_opens_the_window_and_claims_ahead() -> None:
    scheduled = _request(1)
    waiting = [_request(2), _request(3, first_chunk=False), _request(4, dummy=True), _request(5)]
    started = _request(6, state=LlmRequestState.GENERATION_IN_PROGRESS)
    ex = _executor([scheduled, *waiting, started])
    batch = _batch(context=[scheduled])

    assert ex._forward_step(batch) == {"logits": None}

    ex._kv_host_helper.begin_forward.assert_called_once_with(False)
    ex._kv_host_helper.end_forward.assert_called_once_with()
    (candidates,), _ = ex.kv_cache_manager.claim_context_ahead.call_args
    assert [req.py_request_id for req in candidates] == [2, 5]
    ex._forward_step_impl.assert_called_once_with(batch, None)


@pytest.mark.parametrize("reason", ["generation", "warmup", "torch_compile"])
def test_a_forward_that_may_capture_keeps_the_window_shut(reason) -> None:
    ex = _executor([_request(2)])
    batch = _batch(context=[_request(1)])
    if reason == "generation":
        batch.generation_requests.append(_request(3, state=LlmRequestState.GENERATION_IN_PROGRESS))
    elif reason == "warmup":
        ex._is_warmup = True
    else:
        ex._uses_torch_compile = True

    ex._forward_step(batch)

    ex._kv_host_helper.begin_forward.assert_called_once_with(True)
    ex.kv_cache_manager.claim_context_ahead.assert_not_called()
    ex._kv_host_helper.end_forward.assert_called_once_with()


def test_the_window_closes_when_the_forward_raises() -> None:
    ex = _executor()
    ex._forward_step_impl.side_effect = RuntimeError("forward failed")

    with pytest.raises(RuntimeError, match="forward failed"):
        ex._forward_step(_batch(context=[_request(1)]))

    ex._kv_host_helper.end_forward.assert_called_once_with()


def test_without_a_helper_the_forward_is_unchanged() -> None:
    ex = _executor()
    ex._kv_host_helper = None

    ex._forward_step(_batch(context=[_request(1)]))

    ex.kv_cache_manager.claim_context_ahead.assert_not_called()
    ex._forward_step_impl.assert_called_once()


def _mapping(tp_size=1, attention_dp=False, cp_size=1):
    return SimpleNamespace(
        has_tp=lambda: tp_size > 1,
        has_cp=lambda: cp_size > 1,
        tp_group_shares_requests=not attention_dp,
    )


def _helper_executor(mapping=None, **config):
    ex = object.__new__(PyExecutor)
    ex.dist = SimpleNamespace(mapping=mapping or _mapping())
    ex._is_kv_manager_v2 = True
    ex._defer_generation_only_release = config.pop("defer", True)
    ex.device_id = 0
    ex.kv_cache_manager = Mock()
    ex.scheduler = SimpleNamespace(_joint_draft_manager=config.pop("joint_draft", None))
    ex._execution_config = SimpleNamespace(
        deferred_release_thread=config.pop("thread", True),
        context_claim_ahead=config.pop("claim_ahead", 6),
        context_prefetch_blocks=config.pop("prefetch_blocks", 24000),
        enable_kv_pool_rebalance=config.pop("rebalance", False),
    )
    assert not config
    return ex


def test_the_loop_starts_the_helper_with_the_configured_work() -> None:
    ex = _helper_executor()

    assert ex._start_kv_host_helper() is ex.kv_cache_manager.enable_host_helper.return_value

    ex.kv_cache_manager.enable_host_helper.assert_called_once_with(
        0, releases=True, claim_ahead=6, prefetch_blocks=24000
    )
    assert ex._kv_host_helper is ex.kv_cache_manager.enable_host_helper.return_value


@pytest.mark.parametrize(
    ("config", "releases", "claim_ahead"),
    [
        ({"defer": False}, False, 6),
        ({"thread": False}, False, 6),
        ({"joint_draft": object()}, True, 0),
        ({"rebalance": True}, True, 0),
    ],
)
def test_the_helper_drops_what_the_executor_cannot_support(config, releases, claim_ahead) -> None:
    ex = _helper_executor(**config)

    ex._start_kv_host_helper()

    ex.kv_cache_manager.enable_host_helper.assert_called_once_with(
        0, releases=releases, claim_ahead=claim_ahead, prefetch_blocks=24000
    )


@pytest.mark.parametrize(
    ("mapping", "started"),
    [
        (_mapping(tp_size=8, attention_dp=True), True),  # dep: every rank schedules its own
        (_mapping(tp_size=4), False),  # TP without attention DP schedules in lock step
        (_mapping(cp_size=2), False),
    ],
)
def test_the_helper_stays_off_on_ranks_that_schedule_in_lock_step(mapping, started) -> None:
    ex = _helper_executor(mapping)

    helper = ex._start_kv_host_helper()

    assert ex.kv_cache_manager.enable_host_helper.called is started
    assert (helper is not None) is started
    assert ex._kv_host_helper is helper


def test_a_finished_generation_request_is_noted_for_the_scheduler() -> None:
    ex = object.__new__(PyExecutor)
    ex._free_request_resources = Mock()
    ex._release_multimodal_resources = Mock()
    ex.gather_all_responses = False
    ex.dist = SimpleNamespace(rank=1)
    note = ex._note_generation_release = Mock()

    ex._do_terminate_request(_request(1, state=LlmRequestState.GENERATION_IN_PROGRESS))
    ex._do_terminate_request(_request(2))  # a context request
    ex._do_terminate_request(_request(3, state=LlmRequestState.GENERATION_IN_PROGRESS, dummy=True))

    note.assert_called_once_with()
