# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ssm/kda_reset_state_rows op, on a real cache object.

The per-layer fp32 V-first state view ``[slots, 6, 128, 128]`` and bf16 conv view ``[slots, 2304, 3]`` of a real
``KimiK3HybridCacheManager`` (``_kda_cells.build_manager``), each slot row contiguous and the slots strided by the
manager's coalescing, at the slots the manager assigned. The reference is the entry's per-row zero; the kernel
stores zeros, so every comparison is bit for bit.
"""

import zlib

import _kda_cells as kc
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.ssm.kda_reset_state_rows import (
    kda_reset_state_rows,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating

pytestmark = pytest.mark.skipif(not kc.sm100(), reason="certified on SM100 and SM103")

LAYERS = 2
MAX_REQUESTS = 8
NAMES = ("conv", "ssm")


@pytest.fixture(scope="module")
def mgr():
    import tensorrt_llm  # noqa: F401

    m = kc.build_manager(LAYERS, max_batch_size=MAX_REQUESTS, max_tokens=64 * MAX_REQUESTS)
    try:
        yield m
    finally:
        m.shutdown()


@pytest.fixture(scope="module")
def slots(mgr):
    return kc.request_slots(mgr, MAX_REQUESTS, first_id=400)


def _layers(mgr) -> list:
    return [kc.layer_pools(mgr, layer) for layer in range(LAYERS)]


def _fill(pools: list, seed: int) -> None:
    for layer, p in enumerate(pools):
        kc.fill_pools(p, seed + layer)


def _call(fn, p: dict, idx: torch.Tensor, has_init: torch.Tensor, conv: bool):
    return fn(p["ssm"], idx, has_init, p["conv"] if conv else None)


@pytest.mark.parametrize(
    "cell", kda_reset_state_rows.CELLS, ids=[c.why[:40] for c in kda_reset_state_rows.CELLS]
)
def test_reset_on_manager_pools(mgr, slots, cell) -> None:
    """Every layer: the fresh requests' state rows (and conv rows when given) zero, everything else of every layer
    bit for bit as before; the reference gives the same pools."""
    fresh, conv = cell.spec["fresh"], cell.spec["conv"]
    seed = zlib.crc32(cell.why.encode()) % 2**31
    idx = slots[: len(fresh)].flip(0).contiguous()
    has_init = torch.tensor([not f for f in fresh], dtype=torch.bool, device="cuda")
    pools = _layers(mgr)
    with torch.inference_mode():
        _fill(pools, seed)
        for layer, p in enumerate(pools):
            before = [kc.snapshot(q) for q in pools]
            ref = kc.clone_pools({n: p[n] for n in NAMES})
            with validating(kda_reset_state_rows):
                assert _call(kda_reset_state_rows, p, idx, has_init, conv) is None
            _call(kda_reset_state_rows.reference, ref, idx, has_init, conv)
            torch.cuda.synchronize()
            assert all(torch.equal(p[n], ref[n]) for n in NAMES), layer
            for s, f in zip(idx.tolist(), fresh):
                assert torch.equal(
                    p["ssm"][s], torch.zeros_like(p["ssm"][s]) if f else before[layer]["ssm"][s]
                )
                want_conv = (
                    torch.zeros_like(p["conv"][s]) if f and conv else before[layer]["conv"][s]
                )
                assert torch.equal(p["conv"][s], want_conv), (layer, s)
            others = kc.other_slots(p["ssm"].shape[0], idx.tolist())
            assert kc.same_rows(p, before[layer], others, NAMES), layer
            for o, q in enumerate(pools):
                if o != layer:
                    assert all(torch.equal(q[n], before[o][n]) for n in NAMES), (layer, o)


def test_padding_slots_are_skipped(mgr, slots) -> None:
    """Slot indices outside ``[0, slots)`` (the manager's padding: -1 and the pool size) are skipped even when
    marked fresh; the in-range fresh row is zeroed and nothing else changes."""
    pools = _layers(mgr)
    p = pools[0]
    num_slots = p["ssm"].shape[0]
    real = int(slots[0])
    idx = torch.tensor([-1, real, num_slots], dtype=torch.int32, device="cuda")
    has_init = torch.zeros(3, dtype=torch.bool, device="cuda")
    with torch.inference_mode():
        _fill(pools, 77)
        before = [kc.snapshot(q) for q in pools]
        ref = kc.clone_pools({n: p[n] for n in NAMES})
        kda_reset_state_rows(p["ssm"], idx, has_init, p["conv"])
        kda_reset_state_rows.reference(ref["ssm"], idx, has_init, ref["conv"])
        torch.cuda.synchronize()
    assert all(torch.equal(p[n], ref[n]) for n in NAMES)
    assert not p["ssm"][real].any() and not p["conv"][real].any()
    others = kc.other_slots(num_slots, [real])
    assert kc.same_rows(p, before[0], others, NAMES)
    assert all(torch.equal(pools[1][n], before[1][n]) for n in NAMES)


def test_empty_batch_writes_nothing(mgr) -> None:
    pools = _layers(mgr)
    with torch.inference_mode():
        _fill(pools, 79)
        before = [kc.snapshot(q) for q in pools]
        empty = torch.empty(0, dtype=torch.int32, device="cuda")
        kda_reset_state_rows(pools[0]["ssm"], empty, empty.bool(), pools[0]["conv"])
        torch.cuda.synchronize()
    assert all(torch.equal(q[n], b[n]) for q, b in zip(pools, before) for n in NAMES)


def test_graph_replay_matches_eager(mgr, slots) -> None:
    """One reset per layer captured in a CUDA graph, replayed with the slots and flags rewritten in place: the same
    pools as the same resets run eagerly on a copy."""
    pools = _layers(mgr)
    idx = slots[:4].clone()
    flags = torch.zeros(4, dtype=torch.bool, device="cuda")
    schedule = [
        (slots[:4].flip(0), [True, False, True, False]),
        (slots[4:8], [False, True, True, True]),
        (slots[2:6], [False, False, False, False]),
    ]
    with torch.inference_mode():
        _fill(pools, 81)
        eager = [kc.clone_pools({n: p[n] for n in NAMES}) for p in pools]
        start = [kc.clone_pools({n: p[n] for n in NAMES}) for p in pools]
        for s, f in schedule:
            h = torch.tensor([not x for x in f], dtype=torch.bool, device="cuda")
            for e in eager:
                kda_reset_state_rows(e["ssm"], s.contiguous(), h, e["conv"])
        for p in pools:
            kda_reset_state_rows(p["ssm"], idx, flags, p["conv"])  # compile outside the capture
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            for p in pools:
                kda_reset_state_rows(p["ssm"], idx, flags, p["conv"])
        torch.cuda.current_stream().wait_stream(stream)
        for p, s0 in zip(pools, start):
            for n in NAMES:
                p[n].copy_(s0[n])
        for s, f in schedule:
            idx.copy_(s)
            flags.copy_(torch.tensor([not x for x in f], dtype=torch.bool, device="cuda"))
            graph.replay()
        torch.cuda.synchronize()
    for p, e in zip(pools, eager):
        assert all(torch.equal(p[n], e[n]) for n in NAMES)
