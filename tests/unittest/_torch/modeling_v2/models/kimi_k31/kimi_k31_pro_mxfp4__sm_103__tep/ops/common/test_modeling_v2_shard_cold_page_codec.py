# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the sharded cold-page codec's encode / decode round trip on one rank.

The codec's restore must reproduce the evicted pages byte for byte. The multi-rank paths (the all-gather, the TP
agreement, the cross-rank page-id match and its fatal cases) need a TP group and are covered by the 8-rank
microbench."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

assert torch.cuda.is_available(), "the sharded cold-page codec requires a CUDA device"


def _single_rank_provider():
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.common import (  # noqa: E501
        shard_cold_page_codec as sc,
    )
    from tensorrt_llm.mapping import Mapping

    return sc, sc.ShardColdPageProvider(
        Mapping(world_size=1, rank=0, tp_size=1), abort_on_error=False
    )


def test_watchdog_times_each_pending_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    sc, provider = _single_rank_provider()
    transport = provider._get_transport()
    provider.close()  # Drive watchdog ticks synchronously with a virtual clock.
    now = 0.0
    monkeypatch.setattr(sc, "time", SimpleNamespace(monotonic=lambda: now))
    first, second = Mock(), Mock()
    first.query.return_value = second.query.return_value = False
    transport.watch(first, (1, 1))
    now = 400.0
    transport.watch(second, (2, 3))
    first.query.return_value = True

    def tick() -> None:
        transport._stop = Mock()
        transport._stop.wait.side_effect = [False, True]
        transport._watch()

    now = 601.0
    tick()
    assert transport.poisoned is None
    now = 1001.0
    tick()
    assert transport.poisoned is not None
    assert "decode (seq, pages) (2, 3) not complete after 600 s" in transport.poisoned


def test_codec_round_trip_single_rank() -> None:
    """encode -> wipe -> decode into other slots reproduces the pages byte for byte; a corrupted trailer is reported
    loudly (raised here, since the test turns the abort off)."""
    sc, provider = _single_rank_provider()
    try:
        _round_trip(sc, provider)
    finally:
        provider.close()


def _round_trip(sc, provider) -> None:
    layers, key_b, idx_b, slots = 4, 36864, 8448, 64
    g = torch.Generator(device="cuda").manual_seed(11)
    key_pool = torch.randint(
        0, 256, (slots, layers * key_b), dtype=torch.uint8, device="cuda", generator=g
    )
    idx_pool = torch.randint(
        0, 256, (slots, layers * idx_b), dtype=torch.uint8, device="cuda", generator=g
    )
    ref_key, ref_idx = key_pool.clone(), idx_pool.clone()

    def hot(base, slot, n):
        return SimpleNamespace(raw_base=base, raw_slot_bytes=slot, raw_bytes=n)

    lifecycle = SimpleNamespace(
        layers={
            layer: {
                "key": hot(key_pool.data_ptr() + layer * key_b, layers * key_b, key_b),
                "index_key": hot(idx_pool.data_ptr() + layer * idx_b, layers * idx_b, idx_b),
            }
            for layer in range(layers)
        }
    )
    state = sc._CodecState(layer_ids=tuple(range(layers)))
    provider.configure(state, [lifecycle])
    cpb = state.plans[0].cold_page_bytes
    assert cpb == layers * (key_b + idx_b) + sc.TRAILER_BYTES
    cold = torch.empty((32, cpb), dtype=torch.uint8, pin_memory=True)
    stream = torch.cuda.Stream()

    def pairs(dst, src):
        return np.ascontiguousarray(
            np.stack([np.asarray(dst, np.int32), np.asarray(src, np.int32)], axis=1)
        )

    hot_src, cold_slots, hot_dst = [3, 9, 20, 41, 5], [7, 0, 31, 12, 4], [50, 51, 60, 2, 33]
    enc = pairs(cold_slots, hot_src)
    provider.encode_cold_pages(
        state, 0, cold.data_ptr(), enc.ctypes.data, len(hot_src), stream.cuda_stream
    )
    stream.synchronize()
    with torch.cuda.stream(stream):
        key_pool[torch.tensor(hot_dst, device="cuda")] = 0
        idx_pool[torch.tensor(hot_dst, device="cuda")] = 0
    dec = pairs(hot_dst, cold_slots)
    provider.decode_cold_pages(
        state, 0, cold.data_ptr(), dec.ctypes.data, len(hot_dst), stream.cuda_stream
    )
    # An encode must never re-record the event that the watchdog uses to time this decode.
    assert all(provider._last_done is not event for event in provider._events)
    stream.synchronize()
    provider._poll_match(blocking=True)
    d, s = torch.tensor(hot_dst, device="cuda"), torch.tensor(hot_src, device="cuda")
    assert torch.equal(key_pool[d], ref_key[s]) and torch.equal(idx_pool[d], ref_idx[s])
    untouched = torch.ones(slots, dtype=torch.bool, device="cuda")
    untouched[d] = False
    assert torch.equal(key_pool[untouched], ref_key[untouched])
    assert torch.equal(idx_pool[untouched], ref_idx[untouched])

    # A cold page whose trailer tag is wrong cannot be matched: the next check raises.
    trailer_off = state.plans[0].trailer_off
    cold[cold_slots[2], trailer_off + 12 : trailer_off + 16] = 0
    provider.decode_cold_pages(
        state, 0, cold.data_ptr(), dec.ctypes.data, len(hot_dst), stream.cuda_stream
    )
    stream.synchronize()
    with pytest.raises(RuntimeError, match="content mismatch"):
        provider._poll_match(blocking=True)
