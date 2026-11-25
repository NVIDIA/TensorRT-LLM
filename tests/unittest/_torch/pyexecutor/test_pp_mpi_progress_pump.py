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
"""CPU-only tests for the pipeline parallelism idle-time MPI progress pump."""

import queue
import threading
import time

import pytest
from mpi4py import MPI

from tensorrt_llm._torch.pyexecutor import pp_utils
from tensorrt_llm._torch.pyexecutor.pp_utils import (
    DEFAULT_MPI_PROGRESS_POLL_MS,
    MPI_PROGRESS_POLL_MS_ENV_VAR_NAME,
    PPCommTag,
    get_with_mpi_progress,
    make_mpi_progress_exit_hook,
    make_mpi_progress_pump,
    resolve_mpi_progress_poll_interval_ms,
)

pytestmark = pytest.mark.cpu_only


class FakeComm:
    """Records Iprobe calls; raises ``error`` from Iprobe when it is set."""

    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def Iprobe(self, source, tag):
        self.calls.append((source, tag))
        if self.error is not None:
            raise self.error
        return False


@pytest.fixture
def mpi_env(monkeypatch):
    """Enable the pump: default period, MPI on, MPI_THREAD_MULTIPLE."""
    monkeypatch.delenv(MPI_PROGRESS_POLL_MS_ENV_VAR_NAME, raising=False)
    monkeypatch.delenv("TLLM_DISABLE_MPI", raising=False)
    monkeypatch.setattr(MPI, "Query_thread", lambda: MPI.THREAD_MULTIPLE)
    monkeypatch.setattr(MPI, "Is_finalized", lambda: False)
    return monkeypatch


def _make_pump(comm):
    stop_event = threading.Event()
    quiesced_event = threading.Event()
    progress = make_mpi_progress_pump(comm, stop_event, quiesced_event)
    assert progress is not None
    return progress, stop_event, quiesced_event


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, DEFAULT_MPI_PROGRESS_POLL_MS),
        ("abc", DEFAULT_MPI_PROGRESS_POLL_MS),
        ("nan", DEFAULT_MPI_PROGRESS_POLL_MS),
        ("inf", DEFAULT_MPI_PROGRESS_POLL_MS),
        ("0", 0.0),
        ("-1", -1.0),
        ("0.1", 0.5),
        ("5000", 1000.0),
        ("7", 7.0),
    ],
)
def test_resolve_poll_interval(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(MPI_PROGRESS_POLL_MS_ENV_VAR_NAME, raising=False)
    else:
        monkeypatch.setenv(MPI_PROGRESS_POLL_MS_ENV_VAR_NAME, raw)
    assert resolve_mpi_progress_poll_interval_ms() == expected


@pytest.mark.parametrize("raw", ["0", "-1"])
def test_pump_disabled_by_env(mpi_env, raw):
    mpi_env.setenv(MPI_PROGRESS_POLL_MS_ENV_VAR_NAME, raw)
    assert make_mpi_progress_pump(FakeComm(), threading.Event(), threading.Event()) is None


def test_pump_disabled_without_mpi(mpi_env):
    mpi_env.setenv("TLLM_DISABLE_MPI", "1")
    assert make_mpi_progress_pump(FakeComm(), threading.Event(), threading.Event()) is None


def test_pump_disabled_below_thread_multiple(mpi_env):
    mpi_env.setattr(MPI, "Query_thread", lambda: MPI.THREAD_SERIALIZED)
    assert make_mpi_progress_pump(FakeComm(), threading.Event(), threading.Event()) is None


def test_pump_probes_reserved_tag(mpi_env):
    comm = FakeComm()
    (pump, poll_interval_s), _, quiesced_event = _make_pump(comm)
    assert poll_interval_s == DEFAULT_MPI_PROGRESS_POLL_MS / 1000.0
    assert pump() is True
    assert pump() is True
    assert comm.calls == [(MPI.ANY_SOURCE, int(PPCommTag.MPI_PROGRESS_PROBE))] * 2
    assert not quiesced_event.is_set()


def test_pump_stops_without_probing(mpi_env):
    comm = FakeComm()
    (pump, _), stop_event, quiesced_event = _make_pump(comm)
    stop_event.set()
    assert pump() is False
    assert comm.calls == []
    assert quiesced_event.is_set()


def test_pump_stops_after_finalize(mpi_env):
    comm = FakeComm()
    (pump, _), _, quiesced_event = _make_pump(comm)
    mpi_env.setattr(MPI, "Is_finalized", lambda: True)
    assert pump() is False
    assert comm.calls == []
    assert quiesced_event.is_set()


def test_pump_disarms_on_probe_error(mpi_env):
    comm = FakeComm(error=RuntimeError("probe failed"))
    (pump, _), _, quiesced_event = _make_pump(comm)
    assert pump() is False
    assert len(comm.calls) == 1
    assert quiesced_event.is_set()


def test_pump_never_raises_even_if_logging_fails(mpi_env):
    comm = FakeComm(error=RuntimeError("probe failed"))
    (pump, _), _, quiesced_event = _make_pump(comm)

    def _raise(*args, **kwargs):
        raise RuntimeError("logging failed")

    mpi_env.setattr(pp_utils.logger, "error", _raise)
    assert pump() is False
    assert quiesced_event.is_set()


def test_exit_hook_returns_immediately_when_quiesced():
    stop_event = threading.Event()
    quiesced_event = threading.Event()
    quiesced_event.set()
    hook = make_mpi_progress_exit_hook(stop_event, quiesced_event, 0.005)
    start = time.monotonic()
    hook()
    assert stop_event.is_set()
    assert time.monotonic() - start < 0.05


def test_exit_hook_wait_is_bounded():
    stop_event = threading.Event()
    hook = make_mpi_progress_exit_hook(stop_event, threading.Event(), 1.0)
    start = time.monotonic()
    hook()
    elapsed = time.monotonic() - start
    assert stop_event.is_set()
    assert 0.9 <= elapsed < 2.0


def test_get_without_pump_blocks_until_item():
    q = queue.Queue()
    threading.Timer(0.05, q.put, args=("batch",)).start()
    assert get_with_mpi_progress(q, None, threading.Event()) == "batch"


def test_get_pumps_while_waiting():
    q = queue.Queue()
    ticks = []

    def pump():
        ticks.append(1)
        if len(ticks) == 3:
            q.put("batch")
        return True

    stop_event = threading.Event()
    assert get_with_mpi_progress(q, (pump, 0.001), stop_event) == "batch"
    assert len(ticks) >= 3
    assert not stop_event.is_set()


def test_get_returns_shutdown_sentinel():
    q = queue.Queue()
    q.put(None)
    assert get_with_mpi_progress(q, (lambda: True, 0.001), threading.Event()) is None


def test_get_keeps_waiting_after_pump_retires():
    q = queue.Queue()
    ticks = []

    def pump():
        ticks.append(1)
        if len(ticks) == 5:
            q.put("batch")
        return False

    stop_event = threading.Event()
    assert get_with_mpi_progress(q, (pump, 0.001), stop_event) == "batch"
    assert stop_event.is_set()
    assert len(ticks) >= 5
