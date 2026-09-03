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
"""Tests for CacheTransceiverConfig.backend_params plumbing into the NIXL agent."""

import pytest

import tensorrt_llm._torch.disaggregation.native.transfer as transfer_mod
from tensorrt_llm._torch.disaggregation.native.transfer import (
    _create_nixl_agent,
    _unset_ucx_env_for_engine_config,
)
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig


class _FakeAgent:
    def __init__(self, name, use_prog_thread, num_threads, rank, world_size, **kwargs):
        self.name = name
        self.use_prog_thread = use_prog_thread
        self.num_threads = num_threads
        self.rank = rank
        self.world_size = world_size
        self.kwargs = kwargs


def test_cache_transceiver_config_backend_params_field():
    assert CacheTransceiverConfig().backend_params is None
    config = CacheTransceiverConfig(
        backend_params={"device_list": "mlx5_0", "engine_config": "RNDV_THRESH=8192"}
    )
    assert config.backend_params == {
        "device_list": "mlx5_0",
        "engine_config": "RNDV_THRESH=8192",
    }


def test_cache_transceiver_config_backend_params_normalizes_numbers():
    # Unquoted YAML numbers arrive as int/float; they must not be rejected.
    config = CacheTransceiverConfig(backend_params={"num_workers": 2, "split_batch_size": 2048})
    assert config.backend_params == {"num_workers": "2", "split_batch_size": "2048"}


def test_cache_transceiver_config_backend_params_rejects_non_scalars():
    with pytest.raises(ValueError):
        CacheTransceiverConfig(backend_params={"engine_config": ["RNDV_THRESH=8192"]})


def test_unset_ucx_env_for_engine_config(monkeypatch):
    monkeypatch.setenv("UCX_RNDV_THRESH", "inf")
    monkeypatch.setenv("UCX_TLS", "rc")
    monkeypatch.setenv("UCX_NET_DEVICES", "mlx5_0:1")

    _unset_ucx_env_for_engine_config("RNDV_THRESH=8192,malformed_no_equals,TLS=rc,cuda_copy")

    import os

    assert "UCX_RNDV_THRESH" not in os.environ
    assert "UCX_TLS" not in os.environ
    # Keys not named in engine_config are untouched.
    assert os.environ["UCX_NET_DEVICES"] == "mlx5_0:1"


def test_unset_ucx_env_ignores_unset_and_empty_keys(monkeypatch):
    monkeypatch.delenv("UCX_RNDV_THRESH", raising=False)
    # Must not raise on absent env vars, empty elements, or '=VALUE' entries.
    _unset_ucx_env_for_engine_config("RNDV_THRESH=8192,,=oops")


def test_create_nixl_agent_forwards_backend_params(monkeypatch):
    monkeypatch.setattr(transfer_mod, "NixlTransferAgent", _FakeAgent)
    monkeypatch.setattr(transfer_mod, "use_pure_python_transfer_agent", lambda: False)
    monkeypatch.delenv("TRTLLM_NIXL_NUM_THREADS", raising=False)
    monkeypatch.setenv("TRTLLM_NIXL_SPLIT_BATCH_SIZE", "512")
    monkeypatch.setenv("UCX_RNDV_THRESH", "inf")

    agent = _create_nixl_agent(
        "agent0",
        rank=0,
        world_size=2,
        backend_params={
            "device_list": "mlx5_0",
            "engine_config": "RNDV_THRESH=8192",
            "num_threads": "4",
        },
    )

    assert agent.kwargs["device_list"] == "mlx5_0"
    assert agent.kwargs["engine_config"] == "RNDV_THRESH=8192"
    # backend_params overrides the legacy env-var default.
    assert agent.num_threads == 4
    assert "num_threads" not in agent.kwargs
    # Legacy env default is still honored when not overridden.
    assert agent.kwargs["split_batch_size"] == 512
    # engine_config keys had their UCX_* env vars removed before creation.
    import os

    assert "UCX_RNDV_THRESH" not in os.environ


def test_create_nixl_agent_without_backend_params(monkeypatch):
    monkeypatch.setattr(transfer_mod, "NixlTransferAgent", _FakeAgent)
    monkeypatch.delenv("TRTLLM_NIXL_NUM_THREADS", raising=False)
    monkeypatch.delenv("TRTLLM_NIXL_SPLIT_BATCH_SIZE", raising=False)

    agent = _create_nixl_agent("agent0", rank=0, world_size=1, backend_params=None)

    assert agent.num_threads == 8
    assert agent.kwargs == {}


def test_create_nixl_agent_rejects_reserved_keys(monkeypatch):
    monkeypatch.setattr(transfer_mod, "NixlTransferAgent", _FakeAgent)
    monkeypatch.setattr(transfer_mod, "use_pure_python_transfer_agent", lambda: False)

    with pytest.raises(ValueError, match="reserved keys"):
        _create_nixl_agent("agent0", rank=0, world_size=1, backend_params={"rank": "1"})


def test_create_nixl_agent_pure_python_ignores_backend_params(monkeypatch):
    monkeypatch.setattr(transfer_mod, "NixlTransferAgent", _FakeAgent)
    monkeypatch.setattr(transfer_mod, "use_pure_python_transfer_agent", lambda: True)
    monkeypatch.delenv("TRTLLM_NIXL_NUM_THREADS", raising=False)
    monkeypatch.delenv("TRTLLM_NIXL_SPLIT_BATCH_SIZE", raising=False)
    monkeypatch.setenv("UCX_RNDV_THRESH", "inf")

    agent = _create_nixl_agent(
        "agent0",
        rank=0,
        world_size=1,
        backend_params={"device_list": "mlx5_0", "engine_config": "RNDV_THRESH=8192"},
    )

    # The pure Python agent never forwards backend params to createBackend, so
    # they are dropped with a warning and the UCX env must stay untouched.
    assert agent.kwargs == {}
    import os

    assert os.environ["UCX_RNDV_THRESH"] == "inf"
