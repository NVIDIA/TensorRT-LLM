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

The mode refuses non-context-only requests at admission and again at the
connector boundary. Dummy requests and capacity-only connectors are exempt.
"""

from contextlib import ExitStack
from typing import Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
from utils.kv_connector import ConnectorAdapterMock, connector_mpi, connector_request

from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import KvCacheConnectorManager
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor, _validate_connector_admission

pytestmark = pytest.mark.cpu_only
CONNECTOR_MODULE = "tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector"


class SpecMock(ConnectorAdapterMock):
    supports_speculative_prefill = True


def _request(
    request_id: int,
    *,
    context_only: bool = True,
    generation_only: bool = False,
    dummy: bool = False,
) -> MagicMock:
    return connector_request(
        request_id,
        context_only=context_only,
        generation_only=generation_only,
        dummy=dummy,
        attention_dp_dummy=dummy,
    )


def _manager(
    attention_dp: bool,
    *,
    stack: Optional[ExitStack] = None,
    capacity_only: bool = False,
    speculative_prefill_only: bool = True,
) -> KvCacheConnectorManager:
    """Build a manager; a tensor-parallel one enters its MPI patches into `stack`."""
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
    stack.enter_context(connector_mpi())
    return KvCacheConnectorManager(
        worker, scheduler, speculative_prefill_only=speculative_prefill_only
    )


def _broadcast(manager: KvCacheConnectorManager) -> Mock:
    from tensorrt_llm._torch.pyexecutor.connectors import kv_cache_connector

    assert not manager.enable_attention_dp
    return kv_cache_connector.mpi_broadcast


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
        # Every rank raises, so none waits in the broadcast.
        _broadcast(manager).assert_not_called()


@pytest.mark.parametrize("capacity_only", [False, True])
@pytest.mark.parametrize("context_only", [False, True])
@pytest.mark.parametrize("prefill_only", [False, True])
def test_admission_refuses_only_non_context_requests(
    prefill_only: bool, context_only: bool, capacity_only: bool
) -> None:
    """Admission refuses only non-context requests, before any other validation."""
    manager = _manager(
        attention_dp=True, capacity_only=capacity_only, speculative_prefill_only=prefill_only
    )
    executor = object.__new__(PyExecutor)
    executor.kv_connector_manager = manager
    request = _request(7, context_only=context_only, generation_only=not context_only)

    if prefill_only and not context_only and not capacity_only:
        # The executor is otherwise empty, so admission must run first.
        with pytest.raises(ValueError, match="context-only"):
            PyExecutor._validate_request(executor, request)
    else:
        _validate_connector_admission(executor.kv_connector_manager, request)
