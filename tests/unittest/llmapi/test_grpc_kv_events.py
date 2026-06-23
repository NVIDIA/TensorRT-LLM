# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the gRPC KV-cache event converter (CPU-only, no model)."""
import asyncio

import grpc
import pytest

pytest.importorskip("smg_grpc_proto")
from tensorrt_llm.grpc.grpc_servicer import TrtllmServiceServicer
from tensorrt_llm.grpc.kv_events import (convert_batch, convert_event,
                                         convert_events, to_int64)


def test_to_int64_small_positive_unchanged():
    assert to_int64(42) == 42


def test_to_int64_wraps_unsigned_to_signed():
    assert to_int64(2**63) == -(2**63)
    assert to_int64(2**64 - 1) == -1
    assert to_int64(9622824665191806685) == 9622824665191806685 - 2**64


def test_convert_event_stored_single_block():
    ev = {
        "event_id": 1,
        "data": {
            "type": "stored",
            "parent_hash": None,
            "blocks": [{
                "type": "stored_block",
                "block_hash": 9622824665191806685,
                "tokens": [
                    {"type": "unique_token", "token_id": 1, "token_extra_id": 0},
                    {"type": "unique_token", "token_id": 2860, "token_extra_id": 0},
                    {"type": "unique_token", "token_id": 278, "token_extra_id": 0},
                ],
            }],
        },
    }
    out = convert_event(ev, event_id=1)
    assert out.event_id == 1
    assert out.WhichOneof("data") == "stored"
    assert len(out.stored.blocks) == 1
    assert out.stored.blocks[0].block_hash == 9622824665191806685 - 2**64
    assert list(out.stored.blocks[0].token_ids) == [1, 2860, 278]
    assert out.stored.blocks[0].block_size == 3
    assert not out.stored.HasField("parent_block_hash")


def test_convert_event_stored_with_parent():
    ev = {
        "event_id": 4,
        "data": {
            "type": "stored",
            "parent_hash": 9622824665191806685,  # > 2**63, must wrap negative
            "blocks": [{"type": "stored_block", "block_hash": 10,
                        "tokens": [{"token_id": 5, "token_extra_id": 0}]}],
        },
    }
    out = convert_event(ev, event_id=4)
    assert out.stored.parent_block_hash == 9622824665191806685 - 2**64
    assert [b.block_hash for b in out.stored.blocks] == [10]


def test_convert_event_removed():
    # Both hashes are > 2**63 to exercise the unsigned->signed wrap on each.
    ev = {"event_id": 151, "data": {"type": "removed",
          "block_hashes": [10385249491213107555, 12379813738877118345]}}
    out = convert_event(ev, event_id=151)
    assert out.WhichOneof("data") == "removed"
    assert list(out.removed.block_hashes) == [
        10385249491213107555 - 2**64, 12379813738877118345 - 2**64]


def test_convert_event_created_and_updated_skipped():
    created = {"event_id": 0, "data": {"type": "created", "num_blocks_per_cache_level": [2231, 0]}}
    updated = {"event_id": 9, "data": {"type": "updated", "block_hash": 1}}
    assert convert_event(created, 0) is None
    assert convert_event(updated, 9) is None


def test_convert_batch_uses_event_id_and_seq_and_dp_rank():
    ev = {"event_id": 7, "attention_dp_rank": 2,
          "data": {"type": "removed", "block_hashes": [1]}}
    batch = convert_batch(ev, seq_num=3)
    assert batch.sequence_number == 3
    assert batch.dp_rank == 2
    assert len(batch.events) == 1
    assert batch.events[0].event_id == 7


def test_convert_batch_returns_none_for_skipped_event():
    created = {"event_id": 0, "data": {"type": "created", "num_blocks_per_cache_level": [1, 0]}}
    assert convert_batch(created, seq_num=0) is None


def test_convert_events_packs_many_into_one_batch_skipping_created():
    events = [
        {"event_id": 0, "data": {"type": "created", "num_blocks_per_cache_level": [1, 0]}},
        {"event_id": 1, "data": {"type": "stored", "parent_hash": None,
            "blocks": [{"block_hash": 10, "tokens": [{"token_id": 1, "token_extra_id": 0}]}]}},
        {"event_id": 2, "data": {"type": "removed", "block_hashes": [20]}},
    ]
    batch = convert_events(events, seq_num=5)
    assert batch.sequence_number == 5
    assert len(batch.events) == 2  # created skipped, stored + removed packed together
    assert [e.event_id for e in batch.events] == [1, 2]
    assert [e.WhichOneof("data") for e in batch.events] == ["stored", "removed"]


def test_convert_events_all_skipped_returns_none():
    events = [{"event_id": 0, "data": {"type": "created"}},
              {"event_id": 1, "data": {"type": "updated", "block_hash": 1}}]
    assert convert_events(events, seq_num=0) is None


class _Cfg:
    def __init__(self, enable_block_reuse, event_buffer_max_size):
        self.enable_block_reuse = enable_block_reuse
        self.event_buffer_max_size = event_buffer_max_size


class _Args:
    def __init__(self, cfg):
        self.kv_cache_config = cfg


class _FakeLLM:
    def __init__(self, cfg, events):
        self.args = _Args(cfg)
        self._events = events
        self._drained = False

    def get_kv_cache_events(self, timeout=2):
        # One drain returns the whole queued list; subsequent calls are empty.
        if self._drained:
            return []
        self._drained = True
        return list(self._events)


class _FakeRM:
    def __init__(self, llm):
        self.llm = llm


class _Abort(Exception):
    pass


class _FakeCtx:
    """Cancels after the first drain; records abort calls."""
    def __init__(self):
        self._checks = 0
        self.aborted = None
        self.metadata_sent = False

    def cancelled(self):
        # Returns False on the first call (enter loop), True on every subsequent
        # call -- so the loop exits after one full drain of the event queue.
        sent = self._checks > 0
        self._checks += 1
        return sent

    async def send_initial_metadata(self, md):
        self.metadata_sent = True

    async def abort(self, code, details):
        self.aborted = (code, details)
        raise _Abort(details)


async def _collect(servicer, ctx):
    out = []
    async for batch in servicer.SubscribeKvEvents(None, ctx):
        out.append(batch)
    return out


def test_subscribe_batches_drain_into_one_message_skips_created():
    events = [
        {"event_id": 0, "data": {"type": "created", "num_blocks_per_cache_level": [2231, 0]}},
        {"event_id": 1, "data": {"type": "stored", "parent_hash": None,
            "blocks": [{"block_hash": 10, "tokens": [{"token_id": 1, "token_extra_id": 0}]}]}},
        {"event_id": 151, "data": {"type": "removed", "block_hashes": [20]}},
    ]
    servicer = TrtllmServiceServicer(_FakeRM(_FakeLLM(_Cfg(True, 1024), events)))
    ctx = _FakeCtx()
    batches = asyncio.run(_collect(servicer, ctx))
    assert ctx.metadata_sent is True
    # The whole drain cycle is packed into ONE batch message (created skipped).
    assert len(batches) == 1
    assert batches[0].sequence_number == 0
    assert [e.WhichOneof("data") for e in batches[0].events] == ["stored", "removed"]
    assert [e.event_id for e in batches[0].events] == [1, 151]


def test_subscribe_unimplemented_when_events_disabled():
    servicer = TrtllmServiceServicer(_FakeRM(_FakeLLM(_Cfg(False, 0), [])))
    ctx = _FakeCtx()
    with pytest.raises(_Abort):
        asyncio.run(_collect(servicer, ctx))
    assert ctx.aborted[0] == grpc.StatusCode.UNIMPLEMENTED


def test_subscribe_seq_persists_across_reconnects():
    # A second SubscribeKvEvents call on the SAME servicer (a reconnect) must
    # continue the sequence_number, not restart at 0 -- otherwise the consumer's
    # stale-skip (seq <= last_seq) would drop fresh post-reconnect events.
    def stored(h):
        return {"event_id": h, "data": {"type": "stored", "parent_hash": None,
                "blocks": [{"block_hash": h, "tokens": [{"token_id": 1, "token_extra_id": 0}]}]}}
    llm = _FakeLLM(_Cfg(True, 1024), [stored(10), stored(11)])
    servicer = TrtllmServiceServicer(_FakeRM(llm))
    first = asyncio.run(_collect(servicer, _FakeCtx()))
    llm._events = [stored(12), stored(13)]  # new events on the reconnected stream
    llm._drained = False
    second = asyncio.run(_collect(servicer, _FakeCtx()))
    assert [b.sequence_number for b in first] == [0]   # one batch per drain cycle
    assert [b.sequence_number for b in second] == [1]  # continues, not reset to 0
    assert len(second[0].events) == 2
