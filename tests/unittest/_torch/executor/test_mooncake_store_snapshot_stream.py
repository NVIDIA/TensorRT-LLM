# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Window snapshots batched on a side stream (TRTLLM_MOONCAKE_STORE_SNAPSHOT_STREAM=1).

On the legacy default stream `cudaMemcpyBatchAsync` is refused, so the snapshot of a prompt end's window pages fell
back to one Python-issued copy per page-region run on the executor thread. With the switch the copies go to a side
stream that first waits for the executor stream and that the executor stream then waits for, i.e. the same place in
the stream order. CPU only: fake streams and a fake snapshot ring record the order of operations.
"""

import pytest

from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import worker as wmod
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.metadata import PageTransfer, RequestTransfers

pytestmark = pytest.mark.cpu_only


class FakeStream:
    created = []

    def __init__(self, device=None, handle=None, log=None, name="side"):
        self.device = device
        self.cuda_stream = handle if handle is not None else 0x5000 + len(FakeStream.created)
        self.log = log
        self.name = name
        if handle is None:
            FakeStream.created.append(self)

    def wait_stream(self, other):
        self.log.append(("wait", self.name, other.name))


class FakeRing:
    num_slots = 64

    def __init__(self, log):
        self.log = log
        self.next = 0

    def try_reserve(self, n):
        slots = list(range(self.next, self.next + n))
        self.next += n
        return slots

    def gather_many(self, slots, addresses, sizes, handle):
        self.log.append(("gather", handle, tuple(slots)))
        return [(0x9000 + 0x100 * s, sum(sz)) for s, sz in zip(slots, sizes)]


class FakeAddressing:
    def buffers(self, group, page_index):
        return [0x1000 * page_index, 0x1000 * page_index + 0x800], [0x800, 0x800]


def _worker(log, on):
    w = object.__new__(wmod.MooncakeStoreConnectorWorker)
    w._window_snapshot = FakeRing(log)
    w._life_cycles = {0: object()}
    w._addressing = FakeAddressing()
    w._snapshot_stream_on = on
    w._snapshot_stream = None
    w._rank = 0
    w._save_thread = None
    w._window_snapshot_dropped = 0
    return w


def _saves():
    return [RequestTransfers(1, [PageTransfer(b"\x01" * 16, 0, 7, 0), PageTransfer(b"\x02" * 16, 1, 9, 1),
                                 PageTransfer(b"\x03" * 16, 0, 8, 1)])]


@pytest.fixture(autouse=True)
def _batched(monkeypatch):
    monkeypatch.setattr(wmod, "batch_copy_mode", lambda: "auto")
    FakeStream.created = []
    monkeypatch.setattr(wmod.torch.cuda, "Stream", lambda device=None: FakeStream(device=device, log=_LOG))
    _LOG.clear()


_LOG = []


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(wmod.SNAPSHOT_STREAM_ENV, raising=False)
    assert wmod.SNAPSHOT_STREAM_ENV == "TRTLLM_MOONCAKE_STORE_SNAPSHOT_STREAM"
    w = _worker(_LOG, on=False)
    execs = FakeStream(device="cuda:0", handle=0, log=_LOG, name="exec")
    saves, overrides = w._snapshot_window_pages(_saves(), execs)
    assert _LOG == [("gather", 0, (0, 1))]
    assert FakeStream.created == [] and w._snapshot_stream is None
    assert sorted(overrides) == [(0, 7), (0, 8)]


def test_side_stream_order_on_default_stream():
    w = _worker(_LOG, on=True)
    execs = FakeStream(device="cuda:0", handle=0, log=_LOG, name="exec")
    saves, overrides = w._snapshot_window_pages(_saves(), execs)
    side = FakeStream.created[0]
    assert side.device == "cuda:0"
    assert _LOG == [("wait", "side", "exec"), ("gather", side.cuda_stream, (0, 1)), ("wait", "exec", "side")]
    # same slots and overrides as without the switch
    ref_log = []
    ref = _worker(ref_log, on=False)
    ref_exec = FakeStream(device="cuda:0", handle=0, log=ref_log, name="exec")
    _, ref_overrides = ref._snapshot_window_pages(_saves(), ref_exec)
    assert overrides == ref_overrides and saves == _saves()
    # the side stream is created once and reused
    _LOG.clear()
    w._snapshot_window_pages(_saves(), execs)
    assert len(FakeStream.created) == 1 and _LOG[1][1] == side.cuda_stream


def test_non_default_stream_used_directly():
    w = _worker(_LOG, on=True)
    execs = FakeStream(device="cuda:0", handle=0x77, log=_LOG, name="exec")
    w._snapshot_window_pages(_saves(), execs)
    assert _LOG == [("gather", 0x77, (0, 1))] and FakeStream.created == []

