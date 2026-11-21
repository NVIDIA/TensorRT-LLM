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
"""CPU-only tests for ordering the transfer-agent bounce receive scatter on the forward stream.

The executor arms the transceiver only after warmup, the transceiver forwards the execution
stream's handle, and the worker reaches the agent only when the transfer-agent bounce is active.
"""

from types import SimpleNamespace

import pytest

transfer = pytest.importorskip("tensorrt_llm._torch.disaggregation.native.transfer")
transceiver = pytest.importorskip("tensorrt_llm._torch.disaggregation.transceiver")
kv_cache_transceiver = pytest.importorskip(
    "tensorrt_llm._torch.disaggregation.kv_cache_transceiver"
)
py_executor = pytest.importorskip("tensorrt_llm._torch.pyexecutor.py_executor")

pytestmark = pytest.mark.cpu_only


class _FakeAgent:
    def __init__(self, bounce_enabled=True, accept=True):
        self.name = "fake-agent"
        self.bounce_enabled = bounce_enabled
        self.accept = accept
        self.calls = []

    def set_scatter_order_stream(self, handle):
        self.calls.append(handle)
        return self.accept


def _worker(agent, agent_buffer_size_mb=1024):
    worker = transfer.TransferWorker.__new__(transfer.TransferWorker)
    worker._agent = agent
    worker._config = SimpleNamespace(agent_buffer_size_mb=agent_buffer_size_mb)
    return worker


@pytest.mark.parametrize(
    "agent, buffer_mb, expected, reached",
    [
        (_FakeAgent(), 1024, True, True),
        (_FakeAgent(), 0, False, False),  # no agent bounce buffer
        (_FakeAgent(bounce_enabled=False), 1024, False, False),  # bounce init failed
        (_FakeAgent(accept=False), 1024, False, True),  # the kill switch refuses
        (SimpleNamespace(name="py-agent", bounce_enabled=True), 1024, False, False),
        (None, 1024, False, False),  # after shutdown
    ],
)
def test_transfer_worker_forwards_only_with_an_active_agent_bounce(
    agent, buffer_mb, expected, reached
):
    assert _worker(agent, buffer_mb).set_scatter_order_stream(0xABC0) is expected
    if hasattr(agent, "calls"):
        assert agent.calls == ([0xABC0] if reached else [])


class _FakeWorker:
    def __init__(self):
        self.calls = []

    def set_scatter_order_stream(self, handle):
        self.calls.append(handle)
        return True


def test_transceiver_forwards_the_execution_stream_handle():
    tx = transceiver.KvCacheTransceiverV2.__new__(transceiver.KvCacheTransceiverV2)
    tx._transfer_worker = _FakeWorker()
    assert tx.set_execution_stream(None) is False
    assert tx.set_execution_stream(SimpleNamespace(cuda_stream=0x7F00)) is True
    assert tx._transfer_worker.calls == [0x7F00]
    base = kv_cache_transceiver.KvCacheTransceiver
    assert base.set_execution_stream(object(), SimpleNamespace(cuda_stream=1)) is False


class _FakeTransceiver:
    def __init__(self):
        self.streams = []

    def set_execution_stream(self, stream):
        self.streams.append(stream)
        return True


@pytest.mark.parametrize(
    "tx, warmup, expected",
    [
        (_FakeTransceiver(), False, True),
        (_FakeTransceiver(), True, False),  # never armed during warmup
        (None, False, False),
        (SimpleNamespace(), False, False),  # a transceiver without order-stream support
    ],
)
def test_executor_arms_the_execution_stream_after_warmup(tx, warmup, expected):
    ex = py_executor.PyExecutor.__new__(py_executor.PyExecutor)
    ex.kv_cache_transceiver = tx
    ex.execution_stream = SimpleNamespace(cuda_stream=0x5150)
    ex._is_warmup = warmup
    assert ex._arm_kv_transfer_order_stream() is expected
    if isinstance(tx, _FakeTransceiver):
        assert tx.streams == ([ex.execution_stream] if expected else [])


def test_cpp_agent_wrapper_forwards_to_binding():
    agent_cpp = pytest.importorskip("tensorrt_llm._torch.disaggregation.nixl._agent_cpp")
    calls = []
    agent = agent_cpp.BindingsNixlTransferAgent.__new__(agent_cpp.BindingsNixlTransferAgent)
    agent._cpp_agent = SimpleNamespace(
        set_scatter_order_stream=lambda handle: calls.append(handle) or True,
    )
    assert agent.set_scatter_order_stream(0x42) is True
    assert calls == [0x42]


@pytest.mark.parametrize(
    "opted_in, params, env, expected",
    [
        (False, None, {}, {}),
        (True, None, {}, {"scatter_on_order_stream": "1", "copy_stream_priority": "default"}),
        (
            True,
            {"copy_stream_priority": "least"},
            {},
            {"scatter_on_order_stream": "1", "copy_stream_priority": "least"},
        ),
        (
            True,
            None,
            {"TRTLLM_NIXL_BOUNCE_SCATTER_ON_ORDER_STREAM": "0"},
            {"copy_stream_priority": "default"},
        ),
    ],
    ids=["not_opted_in", "opted_in", "dict_wins", "variable_wins"],
)
def test_model_bounce_defaults_sit_below_the_dict_and_the_variables(
    monkeypatch, opted_in, params, env, expected
):
    from tensorrt_llm._torch.disaggregation.nixl import bounce_knobs

    for _, env_name in bounce_knobs.ORDER_STREAM_DEFAULTS.values():
        monkeypatch.delenv(env_name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    bounce_knobs.set_model_scatter_on_order_stream(opted_in)
    try:
        assert bounce_knobs.effective_agent_bounce_params(params) == expected
    finally:
        bounce_knobs.set_model_scatter_on_order_stream(False)
