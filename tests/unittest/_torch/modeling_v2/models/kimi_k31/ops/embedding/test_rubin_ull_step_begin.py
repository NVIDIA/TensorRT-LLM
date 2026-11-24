# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's step_begin entry: every cell bitwise against the three unfolded
launches (every buffer the prep and the metadata write in place, and the begin's outputs), eager
and under CUDA graph replay with new step inputs."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.embedding.rubin_ull_step_begin import (  # noqa: E501
    step_begin as op,
)

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM10x GPU",
)

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)

VOCAB, HIDDEN, TOKENS = 1000, 7168, 8
PAGE, CP, RANK = 64, 16, 5
MAX_BLOCKS, SLOTS, CAPACITY = 48, 152, 12


def step_inputs(requests: int, overlap: bool, schedule: bool, seed: int) -> SimpleNamespace:
    """One Helix q8 generation step's operands of the three launches, as the engine and the
    target hand them over (the metadata reads the prep's KV lengths, slots and bounds)."""
    g = torch.Generator().manual_seed(seed)
    n = requests * TOKENS

    def ints(lo, hi, *shape):
        return torch.randint(lo, hi, shape, generator=g, dtype=torch.int32).cuda()

    base = torch.randint(2000, 400000, (requests,), generator=g, dtype=torch.int32)
    pos = (base[:, None] + torch.arange(TOKENS, dtype=torch.int32)[None, :]).reshape(-1).cuda()
    position_ids = pos.clone()
    helix_positions = pos.clone()
    pos_offsets = ints(-7, 1, n + 3)
    kv_lens = ints(1000, 30000, requests + 2)
    kv_offsets = ints(-7, 1, requests + 2)
    helix_slots = ints(-9, 9, n + 3)
    helix_bounds = ints(-9, 9, n + 3)
    prep = dict(
        position_ids=position_ids,
        pos_offsets=pos_offsets,
        num_pos=n,
        kv_lens=kv_lens,
        kv_offsets=kv_offsets,
        kv_start=0,
        num_kv=requests,
        has_overlap=overlap,
        helix_positions=helix_positions,
        helix_slots=helix_slots,
        helix_bounds=helix_bounds,
        num_helix=n,
        tokens_per_block=PAGE,
        cp_size=CP,
        cp_rank=RANK,
        kv_bound_start=0,
        tokens_per_seq=TOKENS,
    )
    i64 = dict(dtype=torch.int64, device="cuda")
    meta_args = (
        torch.full((requests,), TOKENS, dtype=torch.int32, device="cuda"),
        kv_lens[:requests],
        ints(0, 5000, requests + 1, MAX_BLOCKS),
        helix_slots,
        helix_bounds,
        ints(-3, 3, n + 2),
        torch.full((n + 2,), -5, **i64),
        torch.full((n + 2,), -5, **i64),
        torch.full((requests + 2,), -5, **i64),
        torch.full((requests + 2,), -5, **i64),
        ints(-3, 3, requests + 1, 16),
        ints(-3, 3, n + 2),
    )
    meta_kwargs = dict(
        num_tokens=n,
        width=TOKENS,
        tokens_per_block=PAGE,
        index_head_dim=128,
        quant_block_size=128,
        data_bytes_per_token=128,
        schedule=ints(-3, 3, SLOTS + 1, 2) if schedule else None,
        schedule_expanded=ints(-3, 3, SLOTS + 1, 2) if schedule else None,
    )
    record = torch.full((CAPACITY, 4), -3, dtype=torch.int32, device="cuda")
    record[:, 0] = ints(0, 8, CAPACITY)
    replay = SimpleNamespace(
        checkpoints=True,
        checkpoint_slot_units=3 * TOKENS + 2,
        checkpoint_state_units=3,
        checkpoint_header_units=1,
        num_speculative_tokens=TOKENS - 1,
        prev_num_accepted_tokens=record[:, 0],
    )
    state = torch.randperm(CAPACITY, generator=g)[:requests].to(torch.int32).cuda()
    ids = ints(0, VOCAB, n)
    table = torch.randn((VOCAB, HIDDEN), generator=g).to(torch.bfloat16).cuda()
    buffers = [v for v in prep.values() if isinstance(v, torch.Tensor)]
    buffers += [t for t in meta_args] + [
        v for v in meta_kwargs.values() if isinstance(v, torch.Tensor)
    ]
    return SimpleNamespace(
        prep=prep,
        meta_args=meta_args,
        meta_kwargs=meta_kwargs,
        replay=replay,
        state=state,
        ids=ids,
        table=table,
        buffers=buffers,
    )


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t


def compare(a: SimpleNamespace, got, b: SimpleNamespace, want) -> list:
    """Names of the buffers and outputs that differ between two runs of the same inputs."""
    bad = [
        f"buffer {i}" for i, (x, y) in enumerate(zip(a.buffers, b.buffers)) if not torch.equal(x, y)
    ]
    for name, g, w in zip(("slots", "accepted", "rows"), got, want):
        if g.dtype != w.dtype or g.shape != w.shape or not torch.equal(_bits(g), _bits(w)):
            bad.append(name)
    return bad


def _call(fn, s: SimpleNamespace):
    return fn(s.prep, s.meta_args, s.meta_kwargs, s.replay, s.state, s.ids, s.table)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    spec = cell.spec
    seed = 7 * spec["requests"] + spec["overlap"] + 2 * spec["schedule"]
    a = step_inputs(spec["requests"], spec["overlap"], spec["schedule"], seed)
    b = step_inputs(spec["requests"], spec["overlap"], spec["schedule"], seed)
    with validating(op):
        got = _call(op, a)
    want = _call(op.reference, b)
    torch.cuda.synchronize()
    assert not compare(a, got, b, want)


@pytest.mark.parametrize("requests", (1, 4))
def test_graph_replay_equals_the_unfolded_launches(requests: int) -> None:
    a = step_inputs(requests, True, True, 11)
    _call(op, a)  # compile outside the capture
    a = step_inputs(requests, True, True, 11)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = _call(op, a)
    for seed in (12, 13, 14):
        fresh = step_inputs(requests, True, True, seed)
        for dst, src in zip(a.buffers, fresh.buffers):
            dst.copy_(src)
        a.ids.copy_(fresh.ids)
        a.table.copy_(fresh.table)
        a.state.copy_(fresh.state)
        a.replay.prev_num_accepted_tokens.copy_(fresh.replay.prev_num_accepted_tokens)
        graph.replay()
        want = _call(op.reference, fresh)
        torch.cuda.synchronize()
        assert not compare(a, got, fresh, want), seed
