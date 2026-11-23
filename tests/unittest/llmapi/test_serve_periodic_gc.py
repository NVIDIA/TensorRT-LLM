# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit coverage for trtllm-serve's opt-in periodic GC thread.

TRTLLM_SERVER_GC_COLLECT_INTERVAL_S only acts together with
TRTLLM_SERVER_DISABLE_GC=1; with either unset the server behaves as before.
"""

import gc
import threading
import time
import weakref

import pytest

from tensorrt_llm.commands.serve import _start_periodic_gc_if_requested

pytestmark = pytest.mark.cpu_only

_ENV_VARS = (
    "TRTLLM_SERVER_DISABLE_GC",
    "TRTLLM_SERVER_GC_COLLECT_INTERVAL_S",
    "TRTLLM_SERVER_GC_COLLECT_GENERATION",
)


class _Node:
    """A weak-referenceable object that can form a reference cycle."""

    def __init__(self) -> None:
        self.peer = None


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def gc_disabled():
    """Disable automatic GC for the test, as TRTLLM_SERVER_DISABLE_GC does."""
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def _gc_threads():
    return [t for t in threading.enumerate() if t.name == "trtllm-serve-periodic-gc"]


def _make_garbage_cycle() -> weakref.ref:
    a, b = _Node(), _Node()
    a.peer, b.peer = b, a
    return weakref.ref(a)


def test_off_by_default():
    before = len(_gc_threads())
    assert _start_periodic_gc_if_requested() is None
    assert len(_gc_threads()) == before


def test_off_without_disable_gc(monkeypatch):
    monkeypatch.setenv("TRTLLM_SERVER_GC_COLLECT_INTERVAL_S", "0.01")
    before = len(_gc_threads())
    assert _start_periodic_gc_if_requested() is None
    assert len(_gc_threads()) == before


@pytest.mark.parametrize(
    "name, value",
    [
        ("TRTLLM_SERVER_GC_COLLECT_INTERVAL_S", "abc"),
        ("TRTLLM_SERVER_GC_COLLECT_GENERATION", "3"),
    ],
)
def test_values_are_not_read_without_disable_gc(monkeypatch, name, value):
    """With automatic GC on, the collector variables are not read."""
    monkeypatch.setenv(name, value)
    before = len(_gc_threads())
    assert _start_periodic_gc_if_requested() is None
    assert len(_gc_threads()) == before


@pytest.mark.parametrize("interval", ["", "0", "0.0"])
def test_off_with_zero_interval(monkeypatch, interval):
    monkeypatch.setenv("TRTLLM_SERVER_DISABLE_GC", "1")
    monkeypatch.setenv("TRTLLM_SERVER_GC_COLLECT_INTERVAL_S", interval)
    assert _start_periodic_gc_if_requested() is None


@pytest.mark.parametrize("generation", [None, "0", "1", "2"])
def test_on_collects_cycle(monkeypatch, gc_disabled, generation):
    monkeypatch.setenv("TRTLLM_SERVER_DISABLE_GC", "1")
    monkeypatch.setenv("TRTLLM_SERVER_GC_COLLECT_INTERVAL_S", "0.02")
    if generation is not None:
        monkeypatch.setenv("TRTLLM_SERVER_GC_COLLECT_GENERATION", generation)

    stop = threading.Event()
    thread = _start_periodic_gc_if_requested(stop_event=stop)
    try:
        assert thread is not None
        assert thread.daemon and thread.is_alive()

        ref = _make_garbage_cycle()
        # Unreachable, but only the cyclic collector can free it while
        # automatic GC is disabled.
        assert ref() is not None
        deadline = time.monotonic() + 10.0
        while ref() is not None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ref() is None
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=5.0)
    assert not thread.is_alive()


@pytest.mark.parametrize("interval", ["abc", "-1", "nan", "inf", "1s"])
def test_invalid_interval_raises(monkeypatch, interval):
    monkeypatch.setenv("TRTLLM_SERVER_DISABLE_GC", "1")
    monkeypatch.setenv("TRTLLM_SERVER_GC_COLLECT_INTERVAL_S", interval)
    with pytest.raises(ValueError, match="TRTLLM_SERVER_GC_COLLECT_INTERVAL_S"):
        _start_periodic_gc_if_requested()


@pytest.mark.parametrize("generation", ["3", "-1", "x", "1.0"])
def test_invalid_generation_raises(monkeypatch, generation):
    monkeypatch.setenv("TRTLLM_SERVER_DISABLE_GC", "1")
    monkeypatch.setenv("TRTLLM_SERVER_GC_COLLECT_INTERVAL_S", "1")
    monkeypatch.setenv("TRTLLM_SERVER_GC_COLLECT_GENERATION", generation)
    before = len(_gc_threads())
    with pytest.raises(ValueError, match="TRTLLM_SERVER_GC_COLLECT_GENERATION"):
        _start_periodic_gc_if_requested()
    assert len(_gc_threads()) == before
