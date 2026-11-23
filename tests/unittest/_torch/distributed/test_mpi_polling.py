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
"""Bounded receives shared by the sleep/wakeup ACK wait and the memory monitor's rank gather."""

import threading
import time
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.distributed import mpi_polling


class _FakeComm:
    """Point-to-point messages between in-process ranks, mpi4py-shaped."""

    def __init__(self, rank, mailbox, lock):
        self.rank, self._box, self._lock = rank, mailbox, lock

    def send(self, obj, dest, tag):
        with self._lock:
            self._box.setdefault((self.rank, dest, tag), []).append(obj)

    def Iprobe(self, source, tag):
        with self._lock:
            return bool(self._box.get((source, self.rank, tag)))

    def recv(self, source, tag):
        with self._lock:
            return self._box[(source, self.rank, tag)].pop(0)


def _pair():
    box, lock = {}, threading.Lock()
    return _FakeComm(0, box, lock), _FakeComm(1, box, lock)


@pytest.mark.cpu_only
def test_rejected_messages_are_dropped_and_the_first_kept_one_returns():
    rank0, rank1 = _pair()
    for message in ("stale", "late", "current"):
        rank1.send(message, dest=0, tag=7)
    assert mpi_polling.try_recv(rank0, 1, 7, accept=lambda m: m == "current") == (
        True,
        "current",
    ), "stale messages before the current one are dropped"
    assert mpi_polling.try_recv(rank0, 1, 7) == (False, None), "nothing left"

    # A deadline already passed raises, naming what was awaited.
    with pytest.raises(TimeoutError, match="timed out waiting for report from rank 1"):
        mpi_polling.recv_until(rank0, 1, 7, time.monotonic(), poll_interval_s=0.001, what="report")

    # The lower-case object API is used when present.
    lower = SimpleNamespace(iprobe=lambda source, tag: True)
    assert mpi_polling.message_ready(lower, 1, 7), "iprobe is consulted"
    with pytest.raises(RuntimeError):
        mpi_polling.message_ready(object(), 1, 7)


@pytest.mark.cpu_only
def test_sleep_wakeup_acks_for_another_operation_are_dropped(monkeypatch):
    from tensorrt_llm._torch.pyexecutor import py_executor as pe

    rank0, rank1 = _pair()
    warnings = []
    monkeypatch.setattr(pe.logger, "warning", warnings.append)
    ack = pe._SleepWakeupTag.ACK
    rank1.send({"op_id": "old", "phase": "sleep"}, dest=0, tag=ack)
    rank1.send({"op_id": "new", "phase": "prepare"}, dest=0, tag=ack)
    rank1.send({"op_id": "new", "phase": "sleep", "status": "ok"}, dest=0, tag=ack)
    got = pe._recv_sleep_wakeup_ack_until(
        rank0, 1, time.monotonic() + 5, expected_op_id="new", expected_phase="sleep"
    )
    assert got["status"] == "ok", "the matching ACK is returned"
    assert len(warnings) == 2 and all("Ignoring stale sleep/wakeup ACK" in w for w in warnings)
    with pytest.raises(TimeoutError, match="timed out waiting for sleep/wakeup ACK from rank 1"):
        pe._recv_sleep_wakeup_ack(rank0, 1, 0.0)
