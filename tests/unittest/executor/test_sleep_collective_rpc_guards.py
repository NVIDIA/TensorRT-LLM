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
"""Pure-Python guard tests for MPI sleep/wakeup and collective_rpc.

No GPU or model weights required; all CUDA/MPI/ZMQ machinery is bypassed
via object.__new__ + manual attribute injection.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Sentinel used as the default sleep_config value in _make_worker so that
# callers who omit sleep_config get a truthy non-None object (simulating a
# configured SleepConfig), while callers who pass sleep_config=None test the
# "feature not enabled" guard.  Module-level placement avoids Ruff B008.
_SLEEP_CONFIG_DEFAULT = object()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_worker(backend="pytorch",
                 world_size=1,
                 sleep_config=_SLEEP_CONFIG_DEFAULT):
    from tensorrt_llm.executor.base_worker import BaseWorker

    w = object.__new__(BaseWorker)
    w._backend = backend
    w._is_pytorch_backend = backend in ("pytorch", "_autodeploy")
    w.llm_args = SimpleNamespace(
        backend=backend,
        parallel_config=SimpleNamespace(world_size=world_size),
        sleep_config=sleep_config,
    )
    return w


def _make_proxy(cls_name, model_world_size=1, rpc_client=None):
    if cls_name == "ipc":
        from tensorrt_llm.executor.proxy import GenerationExecutorProxy as Cls
    else:
        from tensorrt_llm.executor.rpc_proxy import \
            GenerationExecutorRpcProxy as Cls
    p = object.__new__(Cls)
    p.model_world_size = model_world_size
    p.rpc_client = rpc_client
    return p


# ---------------------------------------------------------------------------
# BaseWorker.sleep() / wakeup()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["sleep", "wakeup"])
class TestBaseWorkerSleepGuards:

    def test_wrong_backend_raises(self, method):
        w = _make_worker(backend="tensorrt")
        with pytest.raises(ValueError, match="only available for the PyTorch"):
            getattr(w, method)(["kv_cache"])

    def test_autodeploy_backend_raises(self, method):
        """AutoDeploy must be excluded: allocations aren't tagged under
        sleep_config scopes, so release_with_tag would silently no-op."""
        w = _make_worker(backend="_autodeploy")
        with pytest.raises(ValueError, match="only available for the PyTorch"):
            getattr(w, method)(["kv_cache"])

    def test_missing_sleep_config_raises(self, method):
        w = _make_worker(sleep_config=None)
        with pytest.raises(ValueError, match="Sleep feature is not enabled"):
            getattr(w, method)(["kv_cache"])

    def test_multirank_raises(self, method):
        """world_size > 1 must raise before control_action() is entered."""
        w = _make_worker(world_size=2)
        with pytest.raises(NotImplementedError,
                           match="model_world_size == 1"):
            getattr(w, method)(["kv_cache"])

    def test_backend_checked_before_sleep_config(self, method):
        """Backend check fires even when sleep_config is also absent."""
        w = _make_worker(backend="tensorrt", sleep_config=None)
        with pytest.raises(ValueError, match="only available for the PyTorch"):
            getattr(w, method)(["kv_cache"])

    def test_sleep_config_checked_before_world_size(self, method):
        """sleep_config check fires before world_size so the error is
        actionable (set sleep_config, not 'go to TP1')."""
        w = _make_worker(world_size=2, sleep_config=None)
        with pytest.raises(ValueError, match="Sleep feature is not enabled"):
            getattr(w, method)(["kv_cache"])


# ---------------------------------------------------------------------------
# GenerationExecutorProxy / GenerationExecutorRpcProxy collective_rpc()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ["ipc", "rpc"])
class TestProxyCollectiveRpcGuards:

    def test_raises_for_multirank(self, cls):
        p = _make_proxy(cls, model_world_size=2, rpc_client=MagicMock())
        with pytest.raises(NotImplementedError,
                           match="model_world_size == 1"):
            p.collective_rpc("sleep")

    def test_raises_for_unique_reply_rank(self, cls):
        p = _make_proxy(cls, rpc_client=MagicMock())
        with pytest.raises(NotImplementedError):
            p.collective_rpc("sleep", unique_reply_rank=0)

    def test_raises_for_target_ranks(self, cls):
        p = _make_proxy(cls, rpc_client=MagicMock())
        with pytest.raises(NotImplementedError):
            p.collective_rpc("sleep", target_ranks=[0, 1])

    def test_single_rank_routes_to_rpc_client(self, cls):
        """Blocking call returns [result] and forwards args/kwargs."""
        mock_call = MagicMock()
        mock_call.remote.return_value = "ok"
        mock_client = MagicMock()
        mock_client.my_method.return_value = mock_call

        p = _make_proxy(cls, model_world_size=1, rpc_client=mock_client)
        result = p.collective_rpc("my_method", args=(1,), kwargs={"k": "v"})

        mock_client.my_method.assert_called_once_with(1, k="v")
        assert result == ["ok"]

    def test_single_rank_non_block_returns_future(self, cls):
        """non_block=True returns [Future] without calling .remote()."""
        mock_future = MagicMock()
        mock_call = MagicMock()
        mock_call.remote_future.return_value = mock_future
        mock_client = MagicMock()
        mock_client.my_method.return_value = mock_call

        p = _make_proxy(cls, model_world_size=1, rpc_client=mock_client)
        result = p.collective_rpc("my_method", non_block=True)

        mock_call.remote.assert_not_called()
        assert result == [mock_future]


# IPC proxy additionally validates the rpc_client initialisation guard.
class TestIpcProxyRpcClientGuard:

    def test_raises_when_rpc_client_is_none(self):
        p = _make_proxy("ipc", rpc_client=None)
        with pytest.raises(RuntimeError, match="RPC client is not initialised"):
            p.collective_rpc("sleep")
