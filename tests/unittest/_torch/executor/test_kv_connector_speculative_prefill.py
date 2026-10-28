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
"""The KV connector manager's speculative-prefill-only mode.

The mode lets one-model speculative decoding run next to a connector by
refusing every request that is not context-only work: at admission, where only
the request fails, and again at the connector boundary for anything that
bypasses admission. Dummy requests and a connector that transfers no KV are
never refused.
"""

from collections.abc import Iterator
from contextlib import ExitStack
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import KvCacheConnectorManager
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor, _validate_connector_admission
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

pytestmark = pytest.mark.cpu_only
CONNECTOR_MODULE = "tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector"


class SpecMock(MagicMock):
    supports_attention_dp = True
    supports_speculative_prefill = True


def _request(
    request_id: int,
    *,
    context_only: bool = True,
    generation_only: bool = False,
    dummy: bool = False,
) -> MagicMock:
    """A request whose phase flags are explicit bools.

    The manager compares `is_context_only_request` against True, and a bare
    MagicMock attribute is merely truthy, so every flag is set here.
    """
    req = MagicMock(
        is_dummy_request=dummy,
        is_attention_dp_dummy=dummy,
        is_context_only_request=context_only,
        is_generation_only_request=generation_only,
    )
    req.request_id = request_id
    req.py_request_id = request_id
    req.is_child = False
    req.is_finished = False
    req.state = (
        LlmRequestState.CONTEXT_INIT
        if context_only or not generation_only
        else LlmRequestState.GENERATION_IN_PROGRESS
    )
    req.context_current_position = 0
    req.context_remaining_length = 8
    req.context_chunk_size = 8
    req.get_tokens.return_value = list(range(8))
    req.py_draft_tokens = []
    req.kv_cache_retention_config = None
    req.cache_salt = None
    return req


@pytest.fixture
def exit_stack() -> Iterator[ExitStack]:
    with ExitStack() as stack:
        yield stack


def _manager(
    attention_dp: bool,
    *,
    stack: Optional[ExitStack] = None,
    capacity_only: bool = False,
    speculative_prefill_only: bool = True,
) -> KvCacheConnectorManager:
    """A manager in the speculative-prefill-only mode by default.

    A tensor-parallel manager reaches the leader broadcast on every query, so
    it needs its MPI patches for the whole test: they are entered into
    ``stack``, and the broadcast is an identity that records its calls.
    """
    worker = SpecMock(capacity_only=capacity_only)
    worker.get_finished.return_value = ([], [])
    scheduler = SpecMock()
    scheduler.get_num_new_matched_tokens.return_value = (4, False)
    scheduler.reserve_prefix.return_value = (4, False)
    if attention_dp:
        with patch(f"{CONNECTOR_MODULE}.mpi_rank", return_value=1):
            return KvCacheConnectorManager(
                worker,
                scheduler,
                enable_attention_dp=True,
                speculative_prefill_only=speculative_prefill_only,
            )
    assert stack is not None, "a tensor-parallel manager needs MPI patches for the whole test"
    stack.enter_context(patch(f"{CONNECTOR_MODULE}.mpi_rank", return_value=0))
    stack.enter_context(
        patch(f"{CONNECTOR_MODULE}.mpi_broadcast", side_effect=lambda result, root: result)
    )
    return KvCacheConnectorManager(
        worker, scheduler, speculative_prefill_only=speculative_prefill_only
    )


def _broadcast(manager: KvCacheConnectorManager) -> Mock:
    from tensorrt_llm._torch.pyexecutor.connectors import kv_cache_connector

    assert not manager.enable_attention_dp
    return kv_cache_connector.mpi_broadcast


def test_modes_default_off() -> None:
    with (
        patch(f"{CONNECTOR_MODULE}.mpi_rank", return_value=0),
        patch(f"{CONNECTOR_MODULE}.mpi_broadcast", side_effect=lambda result, root: result),
    ):
        manager = KvCacheConnectorManager(MagicMock(), MagicMock())
    assert manager.speculative_prefill_only is False
    assert manager.enable_attention_dp is False


@pytest.mark.parametrize("undeclared", ["worker", "scheduler"])
def test_constructor_refuses_undeclared_connector(undeclared: str) -> None:
    worker = MagicMock() if undeclared == "worker" else SpecMock()
    scheduler = MagicMock() if undeclared == "scheduler" else SpecMock()
    with patch(f"{CONNECTOR_MODULE}.mpi_rank", return_value=0):
        with pytest.raises(NotImplementedError, match="supports_speculative_prefill"):
            KvCacheConnectorManager(worker, scheduler, speculative_prefill_only=True)


def test_tp_follower_without_scheduler_checks_worker_only() -> None:
    with patch(f"{CONNECTOR_MODULE}.mpi_rank", return_value=1):
        manager = KvCacheConnectorManager(SpecMock(), None, speculative_prefill_only=True)
        assert manager.speculative_prefill_only is True
        with pytest.raises(NotImplementedError, match="supports_speculative_prefill"):
            KvCacheConnectorManager(MagicMock(), None, speculative_prefill_only=True)


@pytest.mark.parametrize("context_only,dummy", [(True, False), (False, True)])
def test_speculative_prefill_allows_context_work_and_adp_dummies(
    context_only: bool, dummy: bool
) -> None:
    manager = _manager(attention_dp=True)
    request = _request(7, context_only=context_only, dummy=dummy)

    num_tokens, _ = manager.query_num_new_matched_tokens(request, 0)

    assert num_tokens == (0 if dummy else 4)
    assert manager.scheduler.get_num_new_matched_tokens.call_count == (0 if dummy else 1)


@pytest.mark.parametrize("attention_dp", [True, False], ids=["adp", "tp"])
@pytest.mark.parametrize("generation_only", [False, True])
def test_speculative_prefill_rejects_generation_before_lookup(
    exit_stack: ExitStack, attention_dp: bool, generation_only: bool
) -> None:
    manager = _manager(attention_dp, stack=exit_stack)
    request = _request(7, context_only=False, generation_only=generation_only)

    with pytest.raises(NotImplementedError, match="context-only"):
        manager.query_num_new_matched_tokens(request, 0)

    manager.scheduler.get_num_new_matched_tokens.assert_not_called()
    if not attention_dp:
        # Every rank raises for the same request; none waits in the broadcast.
        _broadcast(manager).assert_not_called()


@pytest.mark.parametrize("attention_dp", [True, False], ids=["adp", "tp"])
def test_speculative_prefill_rejects_before_prefix_reservation(
    exit_stack: ExitStack, attention_dp: bool
) -> None:
    manager = _manager(attention_dp, stack=exit_stack)
    manager.prefix_reservations_enabled = True

    with pytest.raises(NotImplementedError, match="context-only"):
        manager.reserve_prefix(_request(7, context_only=False), 0)
    manager.scheduler.reserve_prefix.assert_not_called()
    if not attention_dp:
        _broadcast(manager).assert_not_called()

    reservation = manager.reserve_prefix(_request(8), 0)
    assert (reservation.start, reservation.end) == (0, 4)
    manager.scheduler.reserve_prefix.assert_called_once()


@pytest.mark.parametrize("generation_only", [False, True])
def test_speculative_prefill_rejects_generation_before_transfer_scheduling(
    generation_only: bool,
) -> None:
    manager = _manager(attention_dp=True)
    batch = ScheduledRequests()
    batch.context_requests_last_chunk.append(_request(6))
    batch.generation_requests.append(
        _request(7, context_only=False, generation_only=generation_only)
    )

    with patch(f"{CONNECTOR_MODULE}.mpi_rank", return_value=1):
        with pytest.raises(NotImplementedError, match="context-only"):
            manager.build_scheduler_output(batch, MagicMock(spec=KVCacheManagerV2))

    assert manager.scheduler_output_manager.requests == {}
    manager.handle_metadata()
    manager.scheduler.build_connector_meta.assert_not_called()


def test_context_batch_with_adp_dummy_is_reported() -> None:
    manager = _manager(attention_dp=True)
    context = _request(6)
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = [context, _request(900, context_only=False, dummy=True)]
    batch.generation_requests = [_request(901, context_only=False, dummy=True)]
    cache = MagicMock(spec=KVCacheManagerV2)
    cache.get_page_indices_by_layer_group.return_value = [[5]]

    with patch(f"{CONNECTOR_MODULE}.mpi_rank", return_value=1):
        manager.build_scheduler_output(batch, cache)
    manager.handle_metadata()

    output = manager.scheduler.build_connector_meta.call_args.args[0]
    assert [req.request_id for req in output.new_requests] == [6]
    assert output.cached_requests == []


def test_capacity_only_connector_never_refuses() -> None:
    manager = _manager(attention_dp=True, capacity_only=True)
    assert manager.capacity_only is True
    assert manager.speculative_prefill_only is True
    request = _request(7, context_only=False, generation_only=True)

    assert manager.query_num_new_matched_tokens(request, 0) == (0, False)
    assert manager.get_num_new_matched_tokens(request, 0) == 0
    assert manager.reserve_prefix(request, 0) is None
    batch = ScheduledRequests()
    batch.generation_requests.append(request)
    assert manager.build_scheduler_output(batch, MagicMock(spec=KVCacheManagerV2)) is None
    _validate_connector_admission(manager, request)
    manager.scheduler.get_num_new_matched_tokens.assert_not_called()
    manager.scheduler.reserve_prefix.assert_not_called()


@pytest.mark.parametrize("capacity_only", [False, True])
@pytest.mark.parametrize("context_only", [False, True])
@pytest.mark.parametrize("prefill_only", [False, True])
def test_admission_refuses_only_non_context_requests(
    prefill_only: bool, context_only: bool, capacity_only: bool
) -> None:
    """Only the refused request fails; the executor never reaches resource preparation."""
    manager = _manager(
        attention_dp=True, capacity_only=capacity_only, speculative_prefill_only=prefill_only
    )
    executor = object.__new__(PyExecutor)
    executor.kv_connector_manager = manager
    request = _request(7, context_only=context_only, generation_only=not context_only)

    if prefill_only and not context_only and not capacity_only:
        # Nothing else is set on the executor, so this raising proves the
        # admission check is the first thing `_validate_request` does.
        with pytest.raises(ValueError, match="context-only"):
            PyExecutor._validate_request(executor, request)
    else:
        _validate_connector_admission(executor.kv_connector_manager, request)


def test_only_an_explicit_context_only_flag_is_admitted() -> None:
    """A merely truthy `is_context_only_request` is not context-only work."""
    manager = _manager(attention_dp=True)
    request = _request(7)
    request.is_context_only_request = "yes"

    with pytest.raises(ValueError, match="context-only"):
        _validate_connector_admission(manager, request)
    with pytest.raises(NotImplementedError, match="context-only"):
        manager.query_num_new_matched_tokens(request, 0)
    manager.scheduler.get_num_new_matched_tokens.assert_not_called()


@pytest.mark.parametrize(
    "shell",
    [
        SimpleNamespace(),
        SimpleNamespace(kv_connector_manager=None),
        SimpleNamespace(kv_connector_manager=Mock(capacity_only=False)),
        SimpleNamespace(kv_connector_manager=Mock(spec=KvCacheConnectorManager)),
    ],
    ids=["no_attribute", "no_connector", "mock_connector", "spec_mock_connector"],
)
def test_admission_ignores_executor_shells_without_the_mode(shell: SimpleNamespace) -> None:
    request = _request(7, context_only=False, generation_only=True)
    _validate_connector_admission(getattr(shell, "kv_connector_manager", None), request)
