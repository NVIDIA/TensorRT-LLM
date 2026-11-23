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
"""CPU-only tests for the opt-in role-scoped KV bounce region.

TRTLLM_KV_BOUNCE_DIRECTION selects which bounce regions a transport allocates. The device
allocator, the streams and the scatter worker are replaced by fakes.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.native.bounce import config as bcfg

# impl pulls the CUDA bindings at import; skip when they are absent. Catch only ImportError so a
# genuine bug in the module still fails.
try:
    from tensorrt_llm._torch.disaggregation.native.bounce import impl as btr

    _HAVE_TRANSPORT = True
except ImportError:  # pragma: no cover - CPU-only env without CUDA bindings
    _HAVE_TRANSPORT = False

_ENV = "TRTLLM_KV_BOUNCE_DIRECTION"
_MIB = 1024 * 1024

needs_transport = pytest.mark.skipif(
    not _HAVE_TRANSPORT, reason="bounce impl import needs CUDA bindings"
)


# ---------------------------------------------------------------------------
# env parsing and sizing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, "both"),
        ("", "both"),
        ("both", "both"),
        ("recv", "recv"),
        ("send", "send"),
        (" RECV ", "recv"),
    ],
)
def test_direction_from_env(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(_ENV, raising=False)
    else:
        monkeypatch.setenv(_ENV, raw)
    assert bcfg.direction_from_env() == expected


@pytest.mark.parametrize("raw", ["receive", "1", "gen", "send,recv"])
def test_direction_rejects_bad_value(monkeypatch, raw):
    monkeypatch.setenv(_ENV, raw)
    with pytest.raises(ValueError, match=_ENV):
        bcfg.direction_from_env()


def test_fit_within_free_one_region_doubles_the_budget():
    chunk = 32 * _MIB
    free = 20_000 * _MIB
    want = 64_000 * _MIB  # larger than any budget, so the clamp decides
    two = bcfg.fit_within_free(want, free_bytes=free, chunk_bytes=chunk)
    one = bcfg.fit_within_free(want, free_bytes=free, chunk_bytes=chunk, num_regions=1)
    assert two == (free // 2 // 2 // chunk) * chunk  # the legacy budget is unchanged
    assert one == (free // 2 // chunk) * chunk
    assert one == 2 * two
    # a request below the budget is not changed by the scope
    assert bcfg.fit_within_free(chunk, free_bytes=free, chunk_bytes=chunk, num_regions=1) == chunk


# ---------------------------------------------------------------------------
# transport construction per direction
# ---------------------------------------------------------------------------


class _FakeAlloc:
    """Stand-in for SlotAllocator: no device memory, records what happens to it."""

    instances = []

    def __init__(self, capacity_bytes, phys_chunk_size, name="kv_bounce"):
        self.capacity = capacity_bytes
        self.name = name
        self.reserved = []
        self.released = []
        self.closed = False
        _FakeAlloc.instances.append(self)

    def reserve(self, size, timeout=None):
        self.reserved.append(size)
        return len(self.reserved) - 1, 0x1000

    def release(self, slot_id):
        self.released.append(slot_id)

    def quarantine(self, slot_id, grace_s):
        pass

    def reclaim_expired(self):
        return 0

    def reg_descs(self):
        return f"desc-{self.name}"

    def close(self):
        self.closed = True


def _make(monkeypatch, direction):
    _FakeAlloc.instances = []
    events = {"streams": 0, "scatter_workers": 0, "registered": [], "deregistered": []}

    def _new_stream(self, mode=0):
        events["streams"] += 1
        return events["streams"]

    def _start_scatter_worker(self, name):
        import queue
        import threading

        events["scatter_workers"] += 1
        self._scatter_q = queue.Queue()
        self._stop = threading.Event()
        self._scatter_thread = threading.Thread(target=lambda: None, daemon=True)
        self._scatter_thread.start()

    monkeypatch.setattr(btr, "SlotAllocator", _FakeAlloc)
    monkeypatch.setattr(btr.VmmBounceTransport, "_new_stream", _new_stream)
    monkeypatch.setattr(btr.VmmBounceTransport, "_start_scatter_worker", _start_scatter_worker)
    agent = SimpleNamespace(
        register_memory=events["registered"].append,
        deregister_memory=events["deregistered"].append,
    )
    t = btr.VmmBounceTransport(
        agent,
        device_id=0,
        capacity_bytes=64 * _MIB,
        phys_chunk_size=32 * _MIB,
        block_bytes_per_group=[1024],
        min_blocks=1,
        direction=direction,
    )
    return t, events


def _recv_req(nblocks=4, rid=1, slice_id=0):
    return SimpleNamespace(
        block_ids_per_layer_groups=[np.arange(nblocks, dtype=np.int64)],
        unique_rid=rid,
        slice_id=slice_id,
        bounce_dst_base=None,
    )


def _write_meta(nbytes=4096):
    return SimpleNamespace(
        sizes=np.array([nbytes], dtype=np.int64),
        src_ptrs=np.array([0x10], dtype=np.int64),
        dst_ptrs=np.array([0x20], dtype=np.int64),
        bounce_dst_base=0x9000,
        dst_device_id=0,
        peer_name="gen0",
    )


@needs_transport
def test_both_keeps_two_regions(monkeypatch):
    t, ev = _make(monkeypatch, "both")
    assert t.can_send and t.can_recv
    assert ev["registered"] == ["desc-kv_bounce_send", "desc-kv_bounce_recv"]
    assert ev["streams"] == 1 and ev["scatter_workers"] == 1
    t.close()
    assert ev["deregistered"] == ev["registered"]
    assert all(a.closed for a in _FakeAlloc.instances)


@needs_transport
def test_recv_only(monkeypatch):
    t, ev = _make(monkeypatch, "recv")
    assert not t.can_send and t.can_recv
    assert t._send_alloc is None
    assert ev["registered"] == ["desc-kv_bounce_recv"]
    assert ev["streams"] == 0  # no send stream without a send region
    assert ev["scatter_workers"] == 1
    # the sender side falls back to the per-fragment path
    assert t.build_request(_write_meta()) is None
    t.release_send(None)
    t.release_send(3)  # no send region: a no-op, never raises
    # the receiver side works as before
    req = _recv_req()
    assert t.reserve(req) is True
    assert t.is_bounced((1, 0))
    t.close()
    assert ev["deregistered"] == ["desc-kv_bounce_recv"]


@needs_transport
def test_send_only(monkeypatch):
    t, ev = _make(monkeypatch, "send")
    assert t.can_send and not t.can_recv
    assert t._recv_alloc is None
    assert ev["registered"] == ["desc-kv_bounce_send"]
    assert ev["streams"] == 1
    assert ev["scatter_workers"] == 0 and t._scatter_thread is None
    # the receiver side always falls back, and the recv-state queries answer "not bounced"
    req = _recv_req()
    assert t.reserve(req) is False
    assert t.register_jit(req, writer_ranks=(0,)) is False
    assert t.grant((1, 0), 0, 4096)[0] == 0  # NO
    assert req.bounce_dst_base is None
    assert not t.is_bounced((1, 0))
    assert t.writer_base((1, 0), 0) is None
    assert t.writer_capacity((1, 0), 0) is None
    t.release_idle_reservation((1, 0))
    t.orphan_reservation((1, 0))
    t.abort_publication((1, 0), {0})
    t.record_result((1, 0), 0)
    t.record_failure((1, 0), 0)
    # the sender side gathers into its own region
    send = _FakeAlloc.instances[0]
    monkeypatch.setattr(t, "_gather_blocking", lambda src, wm, total: None)
    monkeypatch.setattr(t, "_make_write", lambda src, wm, total: ("write", src, total))
    built = t.build_request(_write_meta())
    assert built is not None and send.reserved == [4096]
    t.release_send(built[1])
    assert send.released == [built[1]]
    t.close()  # no scatter worker to join
    assert ev["deregistered"] == ["desc-kv_bounce_send"]
    assert send.closed


@needs_transport
def test_transport_rejects_unknown_direction(monkeypatch):
    with pytest.raises(ValueError, match="direction"):
        _make(monkeypatch, "sideways")


@needs_transport
def test_create_bounce_bad_direction_raises_instead_of_disabling(monkeypatch):
    monkeypatch.setenv(_ENV, "receive")

    def _no_build(*args, **kwargs):
        raise AssertionError("from_config must not run for a bad direction")

    monkeypatch.setattr(btr.VmmBounceTransport, "from_config", classmethod(_no_build))
    with pytest.raises(ValueError, match=_ENV):
        btr.create_bounce(SimpleNamespace(), SimpleNamespace(), device_id=0, page_table=None)


@needs_transport
def test_create_bounce_without_config_ignores_direction(monkeypatch):
    monkeypatch.setenv(_ENV, "receive")
    nb = btr.create_bounce(None, None, device_id=0, page_table=None)
    assert isinstance(nb, btr.NoBounceTransport)
    assert nb.can_send is False and nb.can_recv is False


@needs_transport
@pytest.mark.parametrize("direction, regions", [("both", 2), ("recv", 1), ("send", 1)])
def test_from_config_sizes_by_region_count(monkeypatch, direction, regions):
    monkeypatch.setenv(_ENV, direction)
    free = 20_000 * _MIB
    chunk = 32 * _MIB
    monkeypatch.setattr(btr, "cudart", SimpleNamespace(cudaMemGetInfo=lambda: (0, free, 4 * free)))
    monkeypatch.setattr(btr, "CUASSERT", lambda ret: ret[1:])
    captured = {}

    def _init(self, agent, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(btr.VmmBounceTransport, "__init__", _init)
    cfg = bcfg.config_from_size(9216)
    btr.VmmBounceTransport.from_config(None, cfg, device_id=0, block_bytes_per_group=[1024])
    budget = (free // 2 // regions // chunk) * chunk
    assert captured["capacity_bytes"] == min(9216 * _MIB, budget)
    assert captured["direction"] == direction
