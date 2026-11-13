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

The C++ transport enqueues each received chunk's scatter on the stream handed to it, so the copy
runs between forwards. These tests cover the Python wiring: the executor arms the transceiver only
after warmup, the transceiver forwards the execution stream's handle, and the worker reaches the
agent only when the transfer-agent bounce is active.
"""

import inspect
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.disaggregation.nixl.bounce_knobs import AGENT_BOUNCE_PARAM_KEYS

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


class _FakeWorker:
    def __init__(self, result=True):
        self.calls = []
        self.result = result

    def set_scatter_order_stream(self, handle):
        self.calls.append(handle)
        return self.result


class _FakeTransceiver:
    def __init__(self):
        self.streams = []

    def set_execution_stream(self, stream):
        self.streams.append(stream)
        return True


def test_new_knobs_are_valid_agent_bounce_params():
    assert {
        "scatter_on_order_stream",
        "scatter_copy_engine",
        "copy_stream_priority",
        "max_copy_blocks",
    } <= set(AGENT_BOUNCE_PARAM_KEYS)


class TestTransferWorker:
    def test_forwards_handle_when_agent_bounce_active(self):
        agent = _FakeAgent()
        assert _worker(agent).set_scatter_order_stream(0xABC0) is True
        assert agent.calls == [0xABC0]

    def test_noop_without_agent_bounce_buffer(self):
        agent = _FakeAgent()
        assert _worker(agent, agent_buffer_size_mb=0).set_scatter_order_stream(0xABC0) is False
        assert agent.calls == []

    def test_noop_when_bounce_init_failed(self):
        agent = _FakeAgent(bounce_enabled=False)
        assert _worker(agent).set_scatter_order_stream(0xABC0) is False
        assert agent.calls == []

    def test_noop_for_agent_without_order_stream_support(self):
        agent = SimpleNamespace(name="py-agent", bounce_enabled=True)
        assert _worker(agent).set_scatter_order_stream(0xABC0) is False

    def test_noop_after_shutdown(self):
        assert _worker(None).set_scatter_order_stream(0xABC0) is False

    def test_reports_kill_switch_refusal(self):
        agent = _FakeAgent(accept=False)
        assert _worker(agent).set_scatter_order_stream(0xABC0) is False
        assert agent.calls == [0xABC0]


class TestTransceiver:
    def test_v2_forwards_execution_stream_handle(self):
        tx = transceiver.KvCacheTransceiverV2.__new__(transceiver.KvCacheTransceiverV2)
        tx._transfer_worker = _FakeWorker()
        assert tx.set_execution_stream(SimpleNamespace(cuda_stream=0x7F00)) is True
        assert tx._transfer_worker.calls == [0x7F00]

    def test_v2_ignores_missing_stream(self):
        tx = transceiver.KvCacheTransceiverV2.__new__(transceiver.KvCacheTransceiverV2)
        tx._transfer_worker = _FakeWorker()
        assert tx.set_execution_stream(None) is False
        assert tx._transfer_worker.calls == []

    def test_base_transceiver_is_noop(self):
        base = kv_cache_transceiver.KvCacheTransceiver
        assert base.set_execution_stream(object(), SimpleNamespace(cuda_stream=1)) is False


class TestExecutorArming:
    @staticmethod
    def _executor(tx, warmup):
        ex = py_executor.PyExecutor.__new__(py_executor.PyExecutor)
        ex.kv_cache_transceiver = tx
        ex.execution_stream = SimpleNamespace(cuda_stream=0x5150)
        ex._is_warmup = warmup
        return ex

    def test_arms_with_execution_stream_after_warmup(self):
        tx = _FakeTransceiver()
        ex = self._executor(tx, warmup=False)
        assert ex._arm_kv_transfer_order_stream() is True
        assert tx.streams == [ex.execution_stream]

    def test_not_armed_during_warmup(self):
        tx = _FakeTransceiver()
        assert self._executor(tx, warmup=True)._arm_kv_transfer_order_stream() is False
        assert tx.streams == []

    def test_no_transceiver(self):
        assert self._executor(None, warmup=False)._arm_kv_transfer_order_stream() is False

    def test_transceiver_without_order_stream_support(self):
        ex = self._executor(SimpleNamespace(), warmup=False)
        assert ex._arm_kv_transfer_order_stream() is False

    def test_init_arms_after_warmup_and_graph_capture(self):
        src = inspect.getsource(py_executor.PyExecutor.__init__)
        warmup = src.index("self.model_engine.warmup(")
        warmup_done = src.index("self.is_warmup = False")
        arm = src.index("self._arm_kv_transfer_order_stream()")
        assert warmup < warmup_done < arm


def test_cpp_agent_wrapper_forwards_to_binding():
    agent_cpp = pytest.importorskip("tensorrt_llm._torch.disaggregation.nixl._agent_cpp")
    calls = []
    agent = agent_cpp.BindingsNixlTransferAgent.__new__(agent_cpp.BindingsNixlTransferAgent)
    agent._cpp_agent = SimpleNamespace(
        set_scatter_order_stream=lambda handle: calls.append(handle) or True,
        bounce_ordered_scatter_count=3,
    )
    assert agent.set_scatter_order_stream(0x42) is True
    assert calls == [0x42]
    assert agent.bounce_ordered_scatter_count == 3
