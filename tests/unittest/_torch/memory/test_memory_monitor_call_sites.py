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
"""The executor call sites that feed the memory monitor.

Each wraps a C++ constructor that allocates outside PyTorch with no size
exposed to Python. The readings are faked so that memory appears only while
the wrapped constructor runs; the constructor then fails, which is also the
case that matters most -- a region must record what it saw even when the
code inside it raises.
"""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.memory.monitor import startup

MB = 1 << 20


class _Boom(Exception):
    pass


@pytest.fixture
def fake_memory(monkeypatch):
    """A process whose unexplained memory grows only when ``grow`` is called."""
    monkeypatch.setenv("TLLM_MEM_MONITOR", "1")
    startup.clear()
    state = SimpleNamespace(process=1000 * MB)

    def grow(size):
        state.process += size

    monkeypatch.setattr(
        startup, "_reading", lambda: startup.Reading(state.process, 0, 0), raising=True
    )
    startup.begin_stage("unit")
    yield grow
    startup.clear()


def _region(name):
    return [r for r in startup._regions if r.name == name]


class TestRegions:
    def test_nested_region_is_not_counted_twice(self, fake_memory):
        with startup.region("outer"):
            fake_memory(10 * MB)
            with startup.region("inner"):
                fake_memory(30 * MB)
        assert _region("inner")[0].residual_taken == 30 * MB
        assert _region("outer")[0].residual_taken == 10 * MB

    def test_region_entered_twice_is_one_part(self, fake_memory):
        # One workspace per mapping enters the same region more than once;
        # the summary keys parts by name and would keep only the last.
        for size in (16 * MB, 48 * MB):
            with startup.region("custom allreduce workspace"):
                fake_memory(size)
        parts = startup.residual_breakdown(64 * MB)
        assert dict(parts)["custom allreduce workspace"] == 64 * MB
        assert sum(size for _, size in parts) == 64 * MB
        assert len(parts) == len(dict(parts))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_lora_device_cache_is_named(fake_memory, monkeypatch):
    from tensorrt_llm._torch.pyexecutor import resource_manager
    from tensorrt_llm.llmapi.llm_args import PeftCacheConfig

    def cpp_ctor(**_):
        fake_memory(256 * MB)
        raise _Boom

    monkeypatch.setattr(resource_manager, "PeftCacheManagerCpp", cpp_ctor)
    with pytest.raises(_Boom):
        resource_manager.PeftCacheManager(
            peft_cache_config=PeftCacheConfig(),
            lora_config=Mock(),
            model_config=Mock(),
            lora_model_config=Mock(),
        )
    assert [r.residual_taken for r in _region("LoRA device cache")] == [256 * MB]


def test_kv_transfer_buffers_are_named(fake_memory, monkeypatch):
    from tensorrt_llm._torch.pyexecutor import _util
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager, ResourceManagerType
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
    from tensorrt_llm.mapping import Mapping

    def create_transceiver(*_, **__):
        fake_memory(128 * MB)
        raise _Boom

    for name in ("SimpleScheduler", "SimpleUnifiedScheduler", "KVCacheV2Scheduler"):
        monkeypatch.setattr(_util, name, Mock())
    monkeypatch.setattr(_util, "BindCapacityScheduler", Mock())
    monkeypatch.setattr(_util, "BindMicroBatchScheduler", Mock())
    monkeypatch.setattr(_util, "create_kv_cache_transceiver", create_transceiver)
    manager = Mock(spec=KVCacheManager)
    manager.impl = Mock()
    mapping = Mapping()
    engine = SimpleNamespace(
        spec_config=None,
        mapping=mapping,
        mm_encoder_item_scheduling_enabled=False,
        model=SimpleNamespace(
            model_config=SimpleNamespace(
                pretrained_config=SimpleNamespace(), is_encoder_decoder=False
            ),
        ),
    )
    args = TorchLlmArgs(
        model="/tmp/dummy_model",
        skip_tokenizer_init=True,
        disable_overlap_scheduler=True,
        scheduler_config={},
    )
    with pytest.raises(_Boom):
        _util.create_py_executor_instance(
            dist=Mock(),
            resources={ResourceManagerType.KV_CACHE_MANAGER: manager},
            mapping=mapping,
            llm_args=args,
            scheduler_config=args.scheduler_config,
            ctx_chunk_config=None,
            model_engine=engine,
            start_worker=False,
            sampler=Mock(),
            drafter=None,
            max_seq_len=128,
            max_batch_size=8,
            max_beam_width=1,
            max_num_tokens=8,
        )
    assert [r.residual_taken for r in _region("KV cache transfer buffers")] == [128 * MB]


def test_sleep_listener_reports_a_failed_wakeup(monkeypatch):
    """Rematerializing on wakeup allocates outside PyTorch; its OOM is caught
    by the listener thread, not by ``_handle_errors``."""
    from tensorrt_llm._torch import virtual_memory
    from tensorrt_llm._torch.pyexecutor import py_executor as mod

    def materialize(*_):
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    reported = []
    monkeypatch.setattr(virtual_memory, "materialize_with_tag", materialize)
    monkeypatch.setattr(mod, "monitor_notify_error", reported.append)
    monkeypatch.setattr(mod, "set_thread_local_mpi_comm", lambda *_: None)
    monkeypatch.setattr(mod, "CUASSERT", lambda *_: None)
    monkeypatch.setattr(mod, "cudart", SimpleNamespace(cudaSetDevice=lambda *_: None))
    monkeypatch.setattr(torch.cuda, "set_device", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)

    messages = [
        {"action": mod._SleepWakeupAction.WAKEUP, "tags": ["model"]},
        {"action": mod._SleepWakeupAction.SHUTDOWN},
    ]
    acks = []
    comm = SimpleNamespace(recv=lambda **_: messages.pop(0), send=lambda ack, **_: acks.append(ack))
    executor = mod.PyExecutor.__new__(mod.PyExecutor)
    executor.device_id = 0
    executor.dist = SimpleNamespace(rank=1)
    executor._sleep_wakeup_comm = comm
    executor.control_request_barrier = threading.Event()
    executor.control_request_barrier.set()
    executor.control_action_done = threading.Event()
    executor._active_control_id = None
    executor._run_mnnvl_checkpoint_resources = Mock()

    executor._sleep_wakeup_listener_loop()

    assert acks[0]["status"] == "error"
    assert len(reported) == 1 and "out of memory" in reported[0]
