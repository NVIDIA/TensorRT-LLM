# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native storage ownership follows the existing physical settlement contract."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.disaggregation.native.transfer import RxSession, TxSession
from tensorrt_llm._torch.disaggregation.resource.cache_reuse import _CacheReuseAdapterV2

pytestmark = pytest.mark.cpu_only


def make_session(session_type):
    session = object.__new__(session_type)
    session.lock = threading.Lock()
    session._closed = False
    session._enforce_physical_ownership = True
    session._external_accesses = []
    session._logical_outcomes = SimpleNamespace(terminal=None)
    session._retirement = None
    session._aux_buffer = None
    session.aux_slot = None
    session._sender = None
    session._receiver = None
    session.kv_tasks = []
    session.aux_task = None
    session._kv_tasks = []
    session._aux_physical_owner = None
    return session


@pytest.mark.parametrize("session_type", [TxSession, RxSession])
def test_logical_cancellation_retains_storage_until_physical_owner_drains(session_type):
    session = make_session(session_type)
    owner = SimpleNamespace(resources_drained=False)
    cache = Mock()
    session.retain_external_access(cache, 11, owner)
    cache.expose_external_access.assert_called_once_with(11)
    session._logical_outcomes.terminal = "cancelled"
    assert not session.resources_drained()
    assert not session.close()
    cache.end_external_access.assert_not_called()
    owner.resources_drained = True
    assert session.resources_drained()
    assert session.close()
    cache.end_external_access.assert_called_once_with(11)
    assert session.close()
    cache.end_external_access.assert_called_once_with(11)


@pytest.mark.parametrize("session_type", [TxSession, RxSession])
@pytest.mark.parametrize("rejected", ["closed", "unowned", "terminal", "drained"])
def test_rejected_storage_admission_never_exposes_address(session_type, rejected):
    session = make_session(session_type)
    owner = SimpleNamespace(resources_drained=rejected == "drained")
    session._closed = rejected == "closed"
    session._enforce_physical_ownership = rejected != "unowned"
    session._logical_outcomes.terminal = "error" if rejected == "terminal" else None
    cache = Mock()
    with pytest.raises(RuntimeError, match="physical ownership"):
        session.retain_external_access(cache, 11, owner)
    cache.expose_external_access.assert_not_called()
    assert session._external_accesses == []


@pytest.mark.parametrize("session_type", [TxSession, RxSession])
def test_failed_exposure_removes_session_storage_ownership(session_type):
    session = make_session(session_type)
    owner = SimpleNamespace(resources_drained=False)
    cache = Mock()
    cache.expose_external_access.side_effect = RuntimeError("storage not ready")
    with pytest.raises(RuntimeError, match="not ready"):
        session.retain_external_access(cache, 11, owner)
    assert session._external_accesses == []
    cache.end_external_access.assert_not_called()
    assert session.close()


def test_v2_adapter_forwards_native_storage_ownership():
    cache = Mock()
    adapter = _CacheReuseAdapterV2(SimpleNamespace(kv_cache_map={7: cache}))
    request = SimpleNamespace(py_request_id=7)
    assert adapter.begin_external_read(request, [0, 2], 8) is cache.begin_external_read.return_value
    cache.begin_external_read.assert_called_once_with([0, 2], 8)
    assert (
        adapter.reserve_external_receive(request, 2, 1, 1)
        is cache.reserve_external_receive.return_value
    )
    cache.reserve_external_receive.assert_called_once_with(2, 1, 1)


@pytest.mark.parametrize("session_type", [TxSession, RxSession])
def test_session_retains_multiple_claims_until_every_owner_drains(session_type):
    session = make_session(session_type)
    cache = Mock()
    first = SimpleNamespace(resources_drained=False)
    second = SimpleNamespace(resources_drained=False)
    session.retain_external_access(cache, 11, first)
    session.retain_external_access(cache, 12, second)
    first.resources_drained = True
    assert not session.close()
    cache.end_external_access.assert_not_called()
    second.resources_drained = True
    assert session.close()
    assert [args.args for args in cache.end_external_access.call_args_list] == [(11,), (12,)]
