# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The cep row split (``cp_rows``), its row gather (``rubin_cp_allgather``) and the KDA Ulysses
all-to-all (``rubin_cp_alltoall``), on CPU.

The entry's cells need 8 or 16 ranks on sm_107 (its ARCHS is empty until a VR200 run drives
them); here every cell drives the entry's reference on a few rows of its shape, and an 8- and a
16-rank simulation of the cep KDA layer's two exchanges (the target's pack and unpack, written
out here on the reference) check that each rank gets its heads of every row and then its rows of
every head.
"""

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm.rubin_cp_rows import (
    cp_row_counts,
    cp_rows,
    rubin_cp_allgather,
    rubin_cp_alltoall,
)

pytestmark = pytest.mark.cpu_only


_WORLDS = (8, 16)


@pytest.mark.parametrize("world", _WORLDS, ids=lambda n: f"cep{n}")
@pytest.mark.parametrize("num_tokens", [0, 1, 7, 8, 9, 15, 16, 17, 8191, 131072])
def test_the_split_covers_every_row_once_in_rank_order(num_tokens, world):
    splits = [cp_rows(num_tokens, world, r) for r in range(world)]
    rows = -(-num_tokens // world)
    assert all(s.rows == rows and s.padded_tokens == rows * world for s in splits)
    covered = [t for s in splits for t in range(s.begin, s.end)]
    assert covered == list(range(num_tokens))
    assert cp_row_counts(num_tokens, world) == tuple(s.live for s in splits)
    # Padding rows are the step's tail: a rank's rows end before the next rank's begin.
    assert all(s.live == rows for s in splits if s.end < num_tokens)


@pytest.mark.parametrize("world", _WORLDS, ids=lambda n: f"cep{n}")
def test_the_split_refuses_a_rank_outside_the_group(world):
    with pytest.raises(ValueError):
        cp_rows(10, world, world)


@pytest.mark.parametrize("cell", rubin_cp_alltoall.CELLS, ids=lambda c: c.why)
def test_the_reference_hands_each_rank_its_chunk_of_every_rank(cell):
    world, (_, width) = cell.spec["world"], cell.spec["shape"]
    rows = 3  # the cell's width and dtype on a few rows: the reference is a copy
    chunks = [
        [
            torch.full((rows, width), 1000 * src + dst, dtype=cell.spec["dtype"])
            for dst in range(world)
        ]
        for src in range(world)
    ]
    for rank in range(world):
        got = rubin_cp_alltoall.reference(chunks, rank)
        assert got.shape == (world, rows, width)
        for src in range(world):
            assert torch.equal(got[src], chunks[src][rank])


@pytest.mark.parametrize("world", _WORLDS, ids=lambda n: f"cep{n}")
def test_the_kda_exchanges_hand_each_rank_its_heads_then_its_rows(world):
    """The cep KDA layer's Ulysses exchanges at N ranks, 96 / N heads each (head dim 4 here)."""
    hc, d, n = 96 // world, 4, 37
    heads, pc = world * hc, hc * d
    rows = -(-n // world)
    # Row t, head h, channel c of q / k / v / g: a value that names all four.
    full = torch.arange(n * 4 * heads * d, dtype=torch.float64).view(n, 4, heads * d)
    beta = torch.arange(n * heads, dtype=torch.float64).view(n, heads) + 0.5
    pad = 4 * pc + hc
    sends = []
    for r in range(world):
        s = cp_rows(n, world, r)
        own = torch.zeros(rows, 4, heads * d, dtype=full.dtype)
        own[: s.live] = full[s.begin : s.end]
        own_beta = torch.zeros(rows, heads, dtype=full.dtype)
        own_beta[: s.live] = beta[s.begin : s.end]
        send = torch.zeros(world, rows, pad, dtype=full.dtype)
        # The target's pack: y[:, :3P].view(rows, 3, world, Pc), g.view(rows, world, Pc), beta.
        qkv = own[:, :3].reshape(rows, 3 * heads * d).view(rows, 3, world, pc).permute(2, 0, 1, 3)
        send[:, :, : 3 * pc] = qkv.reshape(world, rows, 3 * pc)
        send[:, :, 3 * pc : 4 * pc] = own[:, 3].view(rows, world, pc).transpose(0, 1)
        send[:, :, 4 * pc : 4 * pc + hc] = own_beta.view(rows, world, hc).transpose(0, 1)
        sends.append(list(send.unbind(0)))
    outs = []
    for j in range(world):
        got = rubin_cp_alltoall.reference(sends, j).reshape(world * rows, pad)[:n]
        mine = slice(j * pc, (j + 1) * pc)
        for part in range(4):
            assert torch.equal(got[:, part * pc : (part + 1) * pc], full[:, part, mine])
        assert torch.equal(got[:, 4 * pc :], beta[:, j * hc : (j + 1) * hc])
        # Head owner j's output: its heads' q values stand in for the recurrence output.
        o = torch.zeros(world * rows, pc, dtype=full.dtype)
        o[:n] = got[:, :pc]
        outs.append(list(o.view(world, rows, pc).unbind(0)))
    for r in range(world):
        s = cp_rows(n, world, r)
        back = rubin_cp_alltoall.reference(outs, r).transpose(0, 1).reshape(rows, heads * d)
        assert torch.equal(back[: s.live], full[s.begin : s.end, 0])
        assert not back[s.live :].any()


@pytest.mark.parametrize("cell", rubin_cp_allgather.CELLS, ids=lambda c: c.why)
def test_the_gather_returns_every_ranks_rows_in_rank_order(cell):
    world, (rows, width) = cell.spec["world"], cell.spec["shape"]
    rows = min(rows, 3)  # the cell's width and dtype on a few rows: the reference is a copy
    inputs = [torch.full((rows, width), src, dtype=cell.spec["dtype"]) for src in range(world)]
    got = rubin_cp_allgather.reference(inputs, None)
    assert got.shape == (world * rows, width) and got.dtype == cell.spec["dtype"]
    for src in range(world):
        assert torch.equal(got[src * rows : (src + 1) * rows], inputs[src])


def test_the_gather_of_one_rank_is_its_input():
    x = torch.arange(6.0).view(3, 2)
    assert rubin_cp_allgather(x, types.SimpleNamespace(cp_size=1)) is x
