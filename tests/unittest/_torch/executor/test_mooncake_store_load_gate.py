# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ldfix F2: asynchronous loads start right after a forward launch (TRTLLM_MOONCAKE_STORE_LOAD_AFTER_LAUNCH).

CPU only: the store is the in-process fake from test_mooncake_store_connector.
"""

import threading
import time

import pytest
from test_mooncake_store_connector import (  # noqa: F401 (fixtures)
    fake_store,
    make_layout,
    make_worker,
    staged_copies,
    store_config,
)

from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import worker as worker_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.metadata import (
    PageTransfer,
    RequestTransfers,
)

pytestmark = pytest.mark.cpu_only


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def _transfers(hashes, start=0):
    return [RequestTransfers(1, [PageTransfer(h, 0, start + i) for i, h in enumerate(hashes)])]


# ---- F2: launch gate ----


def test_launch_gate_waits_for_the_next_launch():
    gate = worker_module._LaunchGate(grace=0.0, timeout=5.0, idle=60.0)
    assert gate.wait() == "idle"  # nothing launched yet: no forward to protect
    gate.launched()
    time.sleep(0.01)
    result = []
    waiter = threading.Thread(target=lambda: result.append(gate.wait()))
    waiter.start()
    time.sleep(0.05)
    assert waiter.is_alive()
    gate.launched()
    waiter.join(timeout=2.0)
    assert result == ["launch"]


def test_launch_gate_grace_idle_and_timeout():
    clock = Clock()
    gate = worker_module._LaunchGate(grace=0.03, timeout=0.05, idle=1.0, clock=clock)
    gate.launched()
    clock.now += 0.01
    assert gate.wait() == "grace"
    clock.now += 2.0
    assert gate.wait() == "idle"
    clock.now -= 1.5  # last launch 0.51 s ago: busy, but no launch comes
    assert gate.wait() == "timeout"


def test_launch_gate_is_off_by_default(monkeypatch):
    monkeypatch.delenv(worker_module.LOAD_AFTER_LAUNCH_ENV, raising=False)
    assert worker_module._LaunchGate.from_env() is None
    monkeypatch.setenv(worker_module.LOAD_AFTER_LAUNCH_ENV, "1")
    monkeypatch.setenv(worker_module.LOAD_AFTER_LAUNCH_ENV + "_TIMEOUT_MS", "123")
    gate = worker_module._LaunchGate.from_env()
    assert gate is not None and abs(gate._timeout - 0.123) < 1e-9


@pytest.mark.parametrize("gated", [False, True])
def test_gated_async_load_still_lands(store_config, fake_store, staged_copies, monkeypatch, gated):
    """With the gate on, a parked load waits for the next launch and then lands as before."""
    if gated:
        monkeypatch.setenv(worker_module.LOAD_AFTER_LAUNCH_ENV, "1")
        monkeypatch.setenv(worker_module.LOAD_AFTER_LAUNCH_ENV + "_TIMEOUT_MS", "3000")
    else:
        monkeypatch.delenv(worker_module.LOAD_AFTER_LAUNCH_ENV, raising=False)
    from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.metadata import (
        MooncakeStoreMetadata,
        PrefixLoadTransfers,
    )

    with make_worker(fake_store, layout=make_layout()) as worker:
        assert (worker._launch_gate is not None) == gated
        block_hash = b"\x07" * 16
        fake_store.objects.add(worker._namespaces[0].key(block_hash))
        if gated:
            worker._launch_gate.launched()  # a forward was launched recently, so the gate is armed
            time.sleep(0.05)
        load = PrefixLoadTransfers(11, True, _transfers([block_hash])[0])
        worker.bind_connector_meta(MooncakeStoreMetadata(prefix_loads=[load]))
        worker.start_load_kv(None)
        if gated:
            time.sleep(0.2)
            assert worker.get_finished_prefix_loads() == []  # held for the next launch
            assert fake_store.get_calls == []
            worker.bind_connector_meta(MooncakeStoreMetadata())
            worker.wait_for_save(None)  # the forward launch
        deadline = time.monotonic() + 5.0
        finished = []
        while not finished and time.monotonic() < deadline:
            finished = worker.get_finished_prefix_loads()
            time.sleep(0.01)
        assert finished == [11]
