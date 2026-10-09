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
"""CPU-only tests for the native pipeline parallelism MPI progress thread."""

import time

import pytest
from mpi4py import MPI

from tensorrt_llm._torch.pyexecutor.pp_utils import PPCommTag
from tensorrt_llm._torch.pyexecutor.py_executor import (
    DEFAULT_PP_MPI_PROGRESS_INTERVAL_US,
    PP_MPI_PROGRESS_INTERVAL_US_ENV_VAR_NAME,
    _resolve_pp_mpi_progress_interval_us,
)
from tensorrt_llm.bindings import MpiProgressThread

pytestmark = pytest.mark.cpu_only

needs_thread_multiple = pytest.mark.skipif(
    MPI.Query_thread() < MPI.THREAD_MULTIPLE,
    reason="the MPI progress thread needs MPI_THREAD_MULTIPLE",
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, DEFAULT_PP_MPI_PROGRESS_INTERVAL_US),
        ("abc", DEFAULT_PP_MPI_PROGRESS_INTERVAL_US),
        ("2.5", DEFAULT_PP_MPI_PROGRESS_INTERVAL_US),
        ("0", 0),
        ("-1", -1),
        ("20000", 20000),
    ],
)
def test_resolve_interval(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(PP_MPI_PROGRESS_INTERVAL_US_ENV_VAR_NAME, raising=False)
    else:
        monkeypatch.setenv(PP_MPI_PROGRESS_INTERVAL_US_ENV_VAR_NAME, raw)
    assert _resolve_pp_mpi_progress_interval_us() == expected


def _wait_for_probes(thread, count, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while thread.num_probes < count and time.monotonic() < deadline:
        time.sleep(0.01)
    return thread.num_probes


@needs_thread_multiple
def test_thread_probes_until_stopped():
    thread = MpiProgressThread(MPI.COMM_WORLD.py2f(), 1000, int(PPCommTag.MPI_PROGRESS_PROBE))
    try:
        assert _wait_for_probes(thread, 3) >= 3
        assert thread.is_running
    finally:
        thread.stop()
    assert not thread.is_running
    probes = thread.num_probes
    time.sleep(0.05)
    assert thread.num_probes == probes
    # stop() is idempotent.
    thread.stop()


@needs_thread_multiple
def test_thread_leaves_other_messages_to_the_receiver():
    """The probe never matches, so a message on another tag stays receivable."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    tag = int(PPCommTag.SAMPLE_STATE)
    thread = MpiProgressThread(comm.py2f(), 1000, int(PPCommTag.MPI_PROGRESS_PROBE))
    try:
        request = comm.isend("payload", dest=rank, tag=tag)
        _wait_for_probes(thread, 3)
        assert comm.recv(source=rank, tag=tag) == "payload"
        request.wait()
    finally:
        thread.stop()


@pytest.mark.parametrize("interval_us", [0, -5])
def test_rejects_nonpositive_interval(interval_us):
    with pytest.raises(RuntimeError, match="interval must be positive"):
        MpiProgressThread(MPI.COMM_WORLD.py2f(), interval_us, int(PPCommTag.MPI_PROGRESS_PROBE))
