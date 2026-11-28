# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the megamoe_expert_slots placement helpers: placements, and the slot tables the routing launch reads
(looked up with `_megamoe_reference.megamoe_slots`, the launch's id -> slot arithmetic in plain torch)."""

import pytest
import torch

__extra_import_path__ = ["../.."]  # the family test directory (_megamoe_reference)
from _megamoe_reference import megamoe_slots, megamoe_slots_by_placement  # noqa: E402

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.megamoe_expert_slots import (
    megamoe_check_placement,
    megamoe_default_placement,
    megamoe_slot_table,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
DEV = "cuda"


@pytest.mark.parametrize(
    "experts,slots,ep,rows",
    [(384, 384, 8, 1280), (384, 416, 32, 150), (128, 136, 4, 64)],
    ids=["dep8-one-copy", "dep32-copies", "ep4-copies"],
)
def test_slot_table_routes_as_the_placement(experts, slots, ep, rows):
    """The table looked up on the device gives the slots the placement itself names (copy `h(r) % copies`, rotated by
    the rank), at the deployed geometries and the EP4 test one."""
    ep_rank = 1  # copies are rotated by the rank, so a nonzero rank is the interesting one
    assignments = megamoe_default_placement(experts, slots, ep)
    table = megamoe_slot_table(assignments, experts, ep_rank, DEV)
    gen = torch.Generator(device=DEV).manual_seed(rows)
    ids = torch.randint(0, experts, (rows, 6), device=DEV, generator=gen, dtype=torch.int32)
    got = megamoe_slots(table, ids, experts)
    assert got.dtype == torch.int32
    assert torch.equal(got, megamoe_slots_by_placement(assignments, ids, ep_rank))


def _engine_default(num_experts: int, num_slots: int, ep: int):
    """The engine's initial assignment when no placement is configured (`_init_load_balancer`)."""
    local = num_slots // ep
    return [(r * num_experts // ep + j) % num_experts for r in range(ep) for j in range(local)]


@pytest.mark.parametrize(
    "experts,slots,ep",
    [(384, 416, 32), (384, 384, 8), (384, 384, 32), (128, 136, 4), (384, 480, 16)],
)
def test_default_placement_is_the_engines_and_valid(experts, slots, ep):
    a = megamoe_default_placement(experts, slots, ep)
    assert a == _engine_default(experts, slots, ep)
    megamoe_check_placement(a, experts, ep)
    # every expert held, copies only from the extra slots
    counts = torch.bincount(torch.tensor(a), minlength=experts)
    assert counts.min() >= 1 and counts.sum() == slots


def test_the_deployed_placements():
    """gen-B (EP32, 416 slots): rank r holds experts 12r..12r+12, so each rank's 13th slot is a copy
    of the next rank's first expert, the last wrapping to expert 0. gen-C (EP8, 384): 48 each."""
    b = megamoe_default_placement(384, 416, 32)
    for r in range(32):
        assert b[13 * r : 13 * r + 13] == [(12 * r + j) % 384 for j in range(13)]
    copies = torch.bincount(torch.tensor(b), minlength=384)
    assert [e for e in range(384) if copies[e] == 2] == [12 * r for r in range(32)]
    c = megamoe_default_placement(384, 384, 8)
    assert c == list(range(384))


def test_invalid_placements_are_refused():
    with pytest.raises(ValueError):
        megamoe_default_placement(384, 400, 32)  # 400 % 32 != 0
    with pytest.raises(ValueError):
        megamoe_default_placement(384, 352, 32)  # fewer slots than experts
    good = megamoe_default_placement(128, 136, 4)
    bad = list(good)
    bad[1] = bad[0]  # rank 0 holds an expert twice
    with pytest.raises(ValueError):
        megamoe_check_placement(bad, 128, 4)
    missing = [e if e != 5 else 6 for e in good]
    with pytest.raises(ValueError):
        megamoe_check_placement(missing, 128, 4)


@pytest.mark.parametrize("ep_rank", [0, 1, 3])
def test_one_copy_table_maps_each_expert_to_its_slot(ep_rank):
    perm = torch.randperm(128, generator=torch.Generator().manual_seed(3)).tolist()
    table = megamoe_slot_table(perm, 128, ep_rank, DEV)
    assert table.shape == (128,) and table.dtype == torch.int32
    ids = torch.randint(0, 128, (50, 6), device=DEV, dtype=torch.int32)
    slots = megamoe_slots(table, ids, 128)
    assert slots.dtype == torch.int32
    assert torch.equal(torch.tensor(perm, device=DEV)[slots.long()], ids.long())


@pytest.mark.parametrize("ep_rank", [0, 1, 2, 3])
def test_replica_table_and_hash_match_the_engines_routing_launch(ep_rank):
    from tensorrt_llm._torch.moe.fused_moe.mega_moe.mega_moe_cute_dsl import (
        _replica_slots,
        replica_slot_table,
    )

    a = megamoe_default_placement(128, 136, 4)
    mine = megamoe_slot_table(a, 128, ep_rank, DEV)
    theirs = replica_slot_table(a, 128, ep_rank).to(DEV)
    assert torch.equal(mine, theirs)
    ids = torch.randint(0, 128, (300, 6), device=DEV, dtype=torch.int32)
    assert torch.equal(megamoe_slots(mine, ids, 128), _replica_slots(theirs, ids, 128))


def test_copies_share_the_rows_and_every_slot_holds_the_expert():
    a = megamoe_default_placement(384, 416, 32)
    table = megamoe_slot_table(a, 384, 5, DEV)
    ids = torch.full((4096, 1), 12, dtype=torch.int32, device=DEV)  # expert 12 has two copies
    slots = megamoe_slots(table, ids, 384).flatten()
    held = torch.tensor(a, device=DEV)[slots.long()]
    assert (held == 12).all()
    share = torch.bincount(slots, minlength=416)
    two = share[share > 0]
    assert two.numel() == 2 and 0.4 < two[0].item() / 4096 < 0.6, "the rows split over both copies"


def test_a_captured_lookup_follows_the_in_place_table_rewrite():
    a = megamoe_default_placement(128, 136, 4)
    table = megamoe_slot_table(a, 128, 1, DEV)
    ids = torch.randint(0, 128, (64, 6), device=DEV, dtype=torch.int32)
    out = torch.empty_like(ids)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out.copy_(megamoe_slots(table, ids, 128))
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out.copy_(megamoe_slots(table, ids, 128))
    graph.replay()
    assert torch.equal(torch.tensor(a, device=DEV)[out.long()], ids.long())
    # move experts: rotate rank 0's and rank 1's windows' contents, rewrite the table in place
    moved = list(a)
    moved[0:34], moved[34:68] = a[34:68], a[0:34]
    megamoe_check_placement(moved, 128, 4)
    table.copy_(megamoe_slot_table(moved, 128, 1, DEV))
    graph.replay()
    assert torch.equal(torch.tensor(moved, device=DEV)[out.long()], ids.long())
