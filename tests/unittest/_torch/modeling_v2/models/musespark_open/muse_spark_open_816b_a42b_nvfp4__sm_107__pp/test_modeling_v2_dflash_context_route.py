# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The capture route of the sm107 PP target's DFlash drafter context, on the host only.

``stages.py`` and ``kernels/dflash_context/route.py`` are pure (no torch, no device) and
are loaded from their files. The tests pin, for the drafter's target layers [1, 16, 30, 44,
59] over every split the target serves:

* which captures each stage taps, receives and sends, and which cross each edge;
* the bytes of every stage-to-stage message at 8192 tokens as the buffered PP transport
  packs it (the 15/16/16/15 split: 512 / 768 / 896 MiB);
* the ring refusal, which names the slot size the enlarged boundary needs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[7]
_TARGET = (
    _ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/models/musespark_open"
    "/muse_spark_open_816b_a42b_nvfp4__sm_107__pp"
)


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, _TARGET / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


STAGES = _load("_mso_pp_stages_under_test", "stages.py")
ROUTE = _load("_mso_pp_dflash_route_under_test", "kernels/dflash_context/route.py")

NUM_LAYERS, EVERY_N, PP = 62, 4, 4
HIDDEN = 8192
TARGET_LAYERS = (1, 16, 30, 44, 59)
SPLIT_15_16_16_15 = (15, 16, 16, 15)
ROWS = 8192
MIB = 1 << 20
SERVED = STAGES.served_partitions(NUM_LAYERS, EVERY_N, PP)


def route_of(partition, layers=TARGET_LAYERS):
    ranges = STAGES.stage_ranges(NUM_LAYERS, PP, partition)
    return ROUTE.capture_route(layers, ranges, HIDDEN)


def stage_of(partition, layer) -> int:
    ranges = STAGES.stage_ranges(NUM_LAYERS, PP, partition)
    return next(stage for stage, (first, end) in enumerate(ranges) if first <= layer < end)


def test_the_target_serves_64_splits_and_the_15_16_16_15_one():
    assert len(SERVED) == 64
    assert SPLIT_15_16_16_15 in SERVED


def test_15_16_16_15_split_route():
    route = route_of(SPLIT_15_16_16_15)
    assert [s.layers for s in route.stages] == [(0, 15), (15, 31), (31, 47), (47, 62)]
    assert [route.layers_of(s.own) for s in route.stages] == [(1,), (16, 30), (44,), (59,)]
    assert [route.edge_layers(stage) for stage in range(3)] == [
        (1,),
        (1, 16, 30),
        (1, 16, 30, 44),
    ]
    # Message order: the received groups in stage order, then the stage's own group.
    assert route.stages[2].received == ((0, 1), (1, 3))
    assert route.stages[2].sent == ((0, 1), (1, 3), (3, 4))
    assert route.stages[3].received == ((0, 1), (1, 3), (3, 4))
    assert route.stages[3].sent == ()
    assert route.stages[0].received == ()


def test_15_16_16_15_split_bytes_per_edge_at_8192_tokens():
    route = route_of(SPLIT_15_16_16_15)
    assert route.boundary_bytes(ROWS) == 384 * MIB
    assert [route.edge_bytes(stage, ROWS) // MIB for stage in range(3)] == [512, 768, 896]
    assert route.largest_edge(ROWS) == (2, 896 * MIB)
    shapes = [shape for shape, _ in route.edge_tensors(2, ROWS)]
    assert shapes == [
        (ROWS, HIDDEN),
        (ROWS, HIDDEN),
        (ROWS, HIDDEN),
        (ROWS, 2 * HIDDEN),
        (ROWS, HIDDEN),
    ]
    with pytest.raises(ValueError):
        route.edge_tensors(3, ROWS)


@pytest.mark.parametrize("partition", SERVED, ids=lambda p: "-".join(map(str, p)))
def test_every_served_split(partition):
    """Each capture crosses exactly the edges from its stage to the last one, once per edge,
    in layer order; the last stage holds layer 59 and receives every other capture; each
    message is the 384 MiB boundary plus 128 MiB per capture it carries."""
    route = route_of(partition)
    last = PP - 1
    assert stage_of(partition, 59) == last
    for stage in range(last):
        crossing = route.edge_layers(stage)
        expected = tuple(layer for layer in TARGET_LAYERS if stage_of(partition, layer) <= stage)
        assert crossing == expected
        assert route.edge_bytes(stage, ROWS) == (384 + 128 * len(crossing)) * MIB
    received = [layer for span in route.stages[last].received for layer in route.layers_of(span)]
    assert received == [layer for layer in TARGET_LAYERS if layer != 59]
    owned = [layer for s in route.stages for layer in route.layers_of(s.own)]
    assert owned == list(TARGET_LAYERS)
    for s in route.stages:
        # A group per earlier stage that taps, in stage order.
        assert list(s.received) == [t.own for t in route.stages[: s.stage] if t.taps]


def test_route_table_over_the_served_splits():
    """The largest boundary over every served split is the 896 MiB one: four captures cross
    the edge into the last stage in all 64 splits."""
    largest = {route_of(p).largest_edge(ROWS) for p in SERVED}
    assert largest == {(2, 896 * MIB)}


def test_captures_all_on_the_last_stage_keep_the_bare_boundary():
    route = route_of(SPLIT_15_16_16_15, layers=(50, 59))
    assert [route.edge_bytes(stage, ROWS) // MIB for stage in range(3)] == [384, 384, 384]
    assert route.stages[3].own == (0, 2)


def test_bad_target_layers():
    ranges = STAGES.stage_ranges(NUM_LAYERS, PP, SPLIT_15_16_16_15)
    with pytest.raises(ValueError, match="repeat"):
        ROUTE.capture_route((1, 1, 59), ranges, HIDDEN)
    with pytest.raises(ValueError, match="holds"):
        ROUTE.capture_route((1, 62), ranges, HIDDEN)


def test_ring_refusal_names_the_size_it_needs():
    route = route_of(SPLIT_15_16_16_15)
    assert ROUTE.ring_refusal(route, ROWS, 896 * MIB, edges=[0, 1, 2]) is None
    reason = ROUTE.ring_refusal(route, ROWS, 384 * MIB, edges=[1, 2])
    assert "takes up to 896 MiB" in reason and "go through NCCL" in reason
    assert "2 -> 3" in reason and "[1, 16, 30, 44]" in reason
    # A stage checks the edges its ring carries only.
    assert "takes up to 512 MiB" in ROUTE.ring_refusal(route, ROWS, 384 * MIB, [0])
    assert ROUTE.ring_refusal(route, ROWS, 512 * MIB, [0]) is None
    # Fewer rows need a smaller slot.
    assert ROUTE.ring_refusal(route, 4096, 448 * MIB, [0, 1, 2]) is None
