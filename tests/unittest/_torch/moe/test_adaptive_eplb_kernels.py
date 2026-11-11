# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the adaptive EPLB route and statistics ops on a GPU, bit for bit against their Python
versions."""

import random

import pytest
import torch
from _torch.moe import adaptive_eplb_reference as reference

from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.policy import (
    ExpertStatistics,
    RouteTables,
    build_route_tables,
    statistics_buffers,
)

TOP_K = 6
needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _random_layout(rng: random.Random, num_experts: int, num_slots: int, ep_size: int) -> list:
    """Every expert at its home slot (rank-major), most free slots another copy of some expert."""
    per_rank, local = num_experts // ep_size, num_slots // ep_size
    layout = [-1] * num_slots
    for expert in range(num_experts):
        layout[(expert // per_rank) * local + expert % per_rank] = expert
    for slot in range(num_slots):
        if layout[slot] < 0 and rng.random() < 0.8:
            layout[slot] = rng.randrange(num_experts)
    return layout


def _random_ids(rng: random.Random, rows: int, num_experts: int, hot: list, wide: bool) -> list:
    """Flat ids, 40 % on the replicated experts, about 1 % outside ``[0, num_experts)``."""
    ids = [
        rng.choice(hot) if hot and rng.random() < 0.4 else rng.randrange(num_experts)
        for _ in range(rows * TOP_K)
    ]
    bad = (-1, num_experts, num_experts + 7, -100) + (((1 << 32) + 5, -(1 << 40)) if wide else ())
    for i in rng.sample(range(len(ids)), max(1, len(ids) // 100)):
        ids[i] = rng.choice(bad)
    return ids


def _dsv4_case(seed: int) -> tuple:
    """A DeepSeek-V4 layer at EP32 (384 experts, 512 slots): layout, shares and 300 rows of ids."""
    rng = random.Random(seed)
    layout = _random_layout(rng, 384, 512, 32)
    shares = [rng.random() if rng.random() > 0.1 else 0.0 for _ in range(512)]
    hot = sorted({e for e in layout if e >= 0 and layout.count(e) > 1})
    ids = [rng.choice(hot) if rng.random() < 0.4 else rng.randrange(384) for _ in range(1800)]
    for i in rng.sample(range(len(ids)), 18):
        ids[i] = rng.choice((-1, 384, 384 + 7))
    return layout, shares, ids


def _tables(counts, starts, slots, cumulative_shares) -> list:
    """The four route tables as int32 tensors on the GPU: the op's table arguments."""
    packed = RouteTables(list(counts), list(starts), list(slots), list(cumulative_shares)).packed()
    table = torch.tensor(packed, dtype=torch.int32, device="cuda")
    return list(table.split([len(counts), len(counts), len(slots), len(slots)]))


def _route(ids, tables, ep_rank, slot_dtype):
    out = torch.empty_like(ids, dtype=slot_dtype)
    torch.ops.trtllm.adaptive_eplb_route_into(ids, *tables, ep_rank, out)
    return out


def _reference_route(ids, tables, ep_rank) -> list:
    """The routes of the flat ``ids`` by the op's Python version."""
    return reference.route(list(ids), *tables, ep_rank)


def _stats(num_experts: int) -> ExpertStatistics:
    on = torch.ones(1, dtype=torch.int32, device="cuda")
    return ExpertStatistics.of(statistics_buffers(1, num_experts, "cuda"), 0, on)


# ------------------------------------------------------------------------------------------------
# Route tables
# ------------------------------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_build_route_tables():
    layout, m32 = [0, 1, 0, 0, 2, 2], (1 << 32) - 1
    tables = build_route_tables(layout, [-1.0, 7.0, 0.0, 0.0, 0.0, 0.0], 3, 6)
    assert tables[:3] == ([3, 1, 2], [0, 3, 4], [0, 2, 3, 1, 4, 5])
    assert tables.cumulative_shares == [round(2**32 / 3), round(2**33 / 3), m32, m32, 1 << 31, m32]
    skew = build_route_tables(layout, [0.5, 1.0, 0.0, 0.5, 2.0, 2.0], 3, 6)
    assert skew.cumulative_shares == [1 << 31, 1 << 31, m32, m32, 1 << 31, m32]
    layout = [0, 1, 2, 0, 1, 3]
    tables = build_route_tables(layout, None, 4, 6)
    assert tables.counts == [2, 2, 1, 1] and tables.slots == [0, 3, 1, 4, 2, 5]
    assert tables.cumulative_shares == [1 << 31, m32, 1 << 31, m32, m32, m32]
    shares = [0.75, 0.5, 1.0, 0.25, 0.5, 1.0]
    assert build_route_tables(layout, shares, 4, 6).cumulative_shares[0] == 3 << 30
    with pytest.raises(ValueError, match="no copy"):
        build_route_tables([0, 1, 2, 0, 1, 2], None, 4, 6)


# ------------------------------------------------------------------------------------------------
# Ops
# ------------------------------------------------------------------------------------------------


@needs_gpu
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
# (384, 576, 64) and (576, 768, 48): tables longer than the kernel's first read of 512 entries.
# The last one has every expert once (no replicas, no shares).
@pytest.mark.parametrize(
    "geometry",
    [
        (384, 512, 32, True),
        (64, 96, 8, True),
        (384, 576, 64, True),
        (576, 768, 48, True),
        (384, 512, 32, False),
    ],
)
def test_route_op_matches_the_reference(id_dtype, slot_dtype, geometry):
    num_experts, num_slots, ep_size, replicas = geometry
    rng = random.Random(num_experts)
    if replicas:
        layout = _random_layout(rng, num_experts, num_slots, ep_size)
        shares = [rng.random() if rng.random() > 0.1 else 0.0 for _ in range(num_slots)]
    else:
        layout = [e if e < num_experts else -1 for e in rng.sample(range(num_slots), num_slots)]
        shares = None
    tables = build_route_tables(layout, shares, num_experts, num_slots)
    device_tables = _tables(*tables)
    hot = [e for e in range(num_experts) if tables.counts[e] > 1]
    for rows in (1, 8, 171, 300, 1100):
        ep_rank = rng.randrange(ep_size)
        ids = _random_ids(rng, rows, num_experts, hot, id_dtype == torch.int64)
        ids_device = torch.tensor(ids, dtype=id_dtype, device="cuda").view(rows, TOP_K)
        got = _route(ids_device, device_tables, ep_rank, slot_dtype)
        assert got.dtype == slot_dtype and tuple(got.shape) == (rows, TOP_K)
        routed = got.view(-1).tolist()
        assert routed == _reference_route(ids, tables, ep_rank)
        # A slot that holds the element's expert, or the sentinel slot.
        for expert, slot in zip(ids, routed):
            assert slot == num_slots or layout[slot] == expert
            assert (slot == num_slots) == (not 0 <= expert < num_experts)


@needs_gpu
def test_route_op_on_unroutable_tables():
    rng = random.Random(11)
    num_experts, num_slots = 384, 512
    counts, starts, slots, _ = build_route_tables(
        _random_layout(rng, num_experts, num_slots, 32), None, num_experts, num_slots
    )
    for expert in rng.sample(range(num_experts), 24):
        kind = rng.randrange(4)
        if kind == 0:
            counts[expert] = 0
        elif kind == 1:
            counts[expert] = -2
        elif kind == 2:
            starts[expert], counts[expert] = num_slots - 1, 3
        else:
            starts[expert] = -4
    cumulative_shares = [rng.randrange(1 << 32) for _ in range(num_slots)]  # not monotone
    device_tables = _tables(counts, starts, slots, cumulative_shares)
    hot = [e for e in range(num_experts) if counts[e] > 1]
    for ep_rank in (0, 13, 31):
        ids = _random_ids(rng, 400, num_experts, hot, True)
        ids_device = torch.tensor(ids, dtype=torch.int64, device="cuda").view(-1, TOP_K)
        got = _route(ids_device, device_tables, ep_rank, torch.int32)
        routed = got.view(-1).tolist()
        assert routed == _reference_route(ids, (counts, starts, slots, cumulative_shares), ep_rank)
        for expert, slot in zip(ids, routed):
            if 0 <= expert < num_experts and not (
                counts[expert] > 0 and 0 <= starts[expert] <= num_slots - counts[expert]
            ):
                assert slot == num_slots


@needs_gpu
def test_a_route_in_a_cuda_graph_follows_its_tables():
    rng = random.Random(5)
    buffer = torch.zeros(2 * (384 + 512), dtype=torch.int32, device="cuda")

    def install(layout, shares):
        tables = build_route_tables(layout, shares, 384, 512)
        buffer.copy_(torch.tensor(tables.packed(), dtype=torch.int32))
        return tables

    install(_random_layout(rng, 384, 512, 32), None)
    ids = _random_ids(rng, 256, 384, [], False)
    ids_device = torch.tensor(ids, dtype=torch.int32, device="cuda").view(-1, TOP_K)
    out = torch.empty_like(ids_device, dtype=torch.int64)
    views = buffer.split([384, 384, 512, 512])
    torch.ops.trtllm.adaptive_eplb_route_into(ids_device, *views, 3, out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops.trtllm.adaptive_eplb_route_into(ids_device, *views, 3, out)
    for _ in range(2):
        tables = install(_random_layout(rng, 384, 512, 32), [rng.random() for _ in range(512)])
        graph.replay()
        torch.cuda.synchronize()
        assert out.view(-1).tolist() == _reference_route(ids, tables, 3)


@needs_gpu
@pytest.mark.parametrize("num_experts", [384, 7, 1000])
def test_record_op_matches_torch(num_experts):
    rng = random.Random(num_experts)
    op, expected = _stats(num_experts), _stats(num_experts)
    for step in range(120):
        if step == 60:  # a new window
            for stats in (op, expected):
                stats.rows.zero_()
                stats.forwards.zero_()
        rows = rng.choice((0, 1, 8, 64, 171, 256, 4096 if step % 40 == 0 else 32))
        wide = step % 2 == 1
        first_call = step % 3 != 2  # every third call is a further chunk of the forward
        ids = _random_ids(rng, rows, num_experts, [], wide) if rows else []
        routes = torch.tensor(ids, dtype=torch.int64 if wide else torch.int32, device="cuda")
        routes = routes.view(rows, TOP_K)
        op.record(routes, first_call)
        reference.record(routes, *expected, first_call)
        assert torch.equal(op.rows.view(torch.int32), expected.rows.view(torch.int32))
        assert torch.equal(op.forwards, expected.forwards)


@needs_gpu
@pytest.mark.parametrize("rows,bucket", [(37, 64), (64, 64), (0, 16), (9000, 9216)])
def test_route_into_op_stages_the_routes_and_records(rows, bucket):
    """One launch writes the slots and gate weights into a bucket's first rows, -1 and 0 into the
    other rows (the columns after top_k untouched), and adds the ids to the statistics; 9000 rows
    are more ids than the statistics count in one block."""
    layout, shares, _ = _dsv4_case(1)
    tables = build_route_tables(layout, shares, 384, 512)
    device_tables = _tables(*tables)
    rng = random.Random(rows)
    hot = [e for e in range(384) if tables.counts[e] > 1]
    ids = _random_ids(rng, rows, 384, hot, False) if rows else []
    ids = torch.tensor(ids, dtype=torch.int32, device="cuda").view(rows, TOP_K)
    weights = torch.rand(rows, TOP_K, device="cuda")
    out = torch.full((bucket, TOP_K), 7, dtype=torch.int64, device="cuda")
    weights_out = torch.full((bucket, TOP_K + 2), 5.0, device="cuda")
    stats, expected = _stats(384), _stats(384)
    torch.ops.trtllm.adaptive_eplb_route_into(
        ids, *device_tables, 3, out, weights, weights_out, *stats, True
    )
    assert out[:rows].tolist() == _route(ids, device_tables, 3, torch.int64).tolist()
    assert (out[rows:] == -1).all()
    assert torch.equal(weights_out[:rows, :TOP_K], weights)
    assert (weights_out[rows:, :TOP_K] == 0).all() and (weights_out[:, TOP_K:] == 5).all()
    reference.record(ids, *expected)
    assert torch.equal(stats.forwards, expected.forwards)
    assert torch.equal(stats.rows.view(torch.int32), expected.rows.view(torch.int32))
    # Without weights the buffer holds exactly the ids' elements (a bucket needs the weights).
    plain = torch.empty(rows, TOP_K, dtype=torch.int32, device="cuda")
    torch.ops.trtllm.adaptive_eplb_route_into(ids, *device_tables, 3, plain)
    assert plain.tolist() == _route(ids, device_tables, 3, torch.int32).tolist()
    if rows < bucket:
        with pytest.raises(RuntimeError, match="exactly"):
            torch.ops.trtllm.adaptive_eplb_route_into(ids, *device_tables, 3, out)


@needs_gpu
@pytest.mark.parametrize("num_experts", [383, 385])
def test_route_into_op_rejects_statistics_of_another_expert_count(num_experts):
    """The statistics rows must have one entry per expert of the tables: the kernel adds the count of
    every routed id at rows[id], so shorter rows would be written past their end."""
    layout, shares, _ = _dsv4_case(1)
    device_tables = _tables(*build_route_tables(layout, shares, 384, 512))
    ids = torch.zeros(8, TOP_K, dtype=torch.int32, device="cuda")
    out = torch.empty(8, TOP_K, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="one entry per expert"):
        torch.ops.trtllm.adaptive_eplb_route_into(
            ids, *device_tables, 3, out, None, None, *_stats(num_experts), True
        )
