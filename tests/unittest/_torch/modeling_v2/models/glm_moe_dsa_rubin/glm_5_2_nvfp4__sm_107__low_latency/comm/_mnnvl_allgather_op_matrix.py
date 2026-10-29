# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the GLM-5.2 target's mnnvl_allgather entry.

Its own launcher, like ``_mnnvl_hop_op_matrix.py`` (see ``_mnnvl_ranks``):

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _mnnvl_allgather_op_matrix.py

The gather moves bits, so every gate is exact. Beyond the cell: special values (-0.0, infinities,
NaN payloads, the all-ones word), the rotation of the flags and the refill of the previous
buffer with the empty word, several private workspaces side by side (the gather's and the draft
steps'), CUDA graph replays, ranks arriving late, and the guards. The collected entry point is
``test_modeling_v2_mnnvl_allgather_op_matrix.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Tuple

import _mnnvl_ranks as ranks
import torch
import torch.distributed as dist

assert torch.cuda.is_available(), "mnnvl_allgather requires CUDA devices"

DEADLINE_S = 1200
SHARD = 9680
# One Lamport buffer of the logits gather: the gathered logits of up to 54 rows of the
# 154,880-entry BF16 vocabulary.
GATHER_BUFFER_BYTES = 16 * 1024 * 1024
EMPTY_WORD = -1

# Bound in the rank body: the entry and the workspace helpers import tensorrt_llm.
entry: Any = None
ws: Any = None
validating: Any = None
RANK = 0
WORLD = 1
MAPPING: Any = None
WORKSPACE: Any = None


def _shard(rows: int, seed: int, rank: int, shard: int = SHARD) -> torch.Tensor:
    return ranks.normal((rows, shard), seed * 1009 + rank, scale=4.0)


def _gather(x: torch.Tensor, workspace: Any = None) -> torch.Tensor:
    workspace = WORKSPACE if workspace is None else workspace
    return entry.mnnvl_allgather(x, workspace.comm_buffer, workspace.buffer_flags, WORLD)


def _check(out: torch.Tensor, per_rank: List[torch.Tensor], where: str) -> None:
    ranks.assert_bitwise(out, entry.mnnvl_allgather.reference(per_rank), where)


def check_certified_cells() -> None:
    """Every cell: the concatenation bit for bit, the same on every rank."""
    for i, cell in enumerate(entry.mnnvl_allgather.CELLS):
        rows, shard = cell.spec["rows"], cell.spec["shard"]
        seed = 100 + 13 * i
        per_rank = [_shard(rows, seed, r, shard) for r in range(WORLD)]
        with validating(entry.mnnvl_allgather):
            out = entry.mnnvl_allgather(
                per_rank[RANK], WORKSPACE.comm_buffer, WORKSPACE.buffer_flags, WORLD
            )
        assert out.shape == (rows, WORLD * shard) and out.dtype == torch.float32, cell.why
        _check(out, per_rank, cell.why)
        ranks.assert_same_on_every_rank(out, cell.why)
        dist.barrier()


def check_special_values_travel_unchanged() -> None:
    """-0.0, infinities, NaN payloads and subnormals arrive bit for bit; a word of all ones (two
    BF16 NaNs) arrives with its lowest bit cleared, still two NaNs."""
    specials = torch.tensor(
        [0x8000, 0x0000, 0x7F80, 0xFF80, 0x7FC1, 0xFFA5, 0x0001, 0x8001, 0xFFFF, 0xFFFF],
        dtype=torch.int32,
    ).to(torch.int16)
    per_rank = []
    for r in range(WORLD):
        x = _shard(8, 2000, r)
        x.view(torch.int16)[r % 8, : specials.numel()] = specials.cuda()
        per_rank.append(x)
    expected_inputs = []
    for x in per_rank:
        words = x.clone().view(torch.int32)
        words[words == EMPTY_WORD] = EMPTY_WORD - 1
        expected_inputs.append(words.view(torch.bfloat16))
    out = _gather(per_rank[RANK])
    ranks.assert_bitwise(out, entry.mnnvl_allgather.reference(expected_inputs), "special values")


def check_flags_rotate_and_the_previous_buffer_is_emptied() -> None:
    """Each call rotates the flags; the next one fills the buffer it used with the empty word."""
    previous = None
    for step, rows in enumerate((8, 1, 3, 8, 2, 8, 1)):
        seed = 3000 + step
        per_rank = [_shard(rows, seed, r) for r in range(WORLD)]
        before = ranks.lamport_flags(WORKSPACE.buffer_flags)
        _check(_gather(per_rank[RANK]), per_rank, f"step {step}")
        after = ranks.lamport_flags(WORKSPACE.buffer_flags)
        gathered_bytes = rows * WORLD * SHARD * torch.bfloat16.itemsize
        ranks.assert_rotated(before, after, gathered_bytes, f"step {step}")
        if previous is not None:
            index, words = previous
            emptied = WORKSPACE.comm_buffer.view(3, -1)[index, :words]
            assert bool((emptied == EMPTY_WORD).all()), f"step {step}: buffer {index} not emptied"
        previous = (before[0], gathered_bytes // 4)
        dist.barrier()


def check_private_workspaces_side_by_side() -> None:
    """The draft steps' exchange workspaces live beside the gather's; calls alternate freely."""
    # The sizes of the draft_sample_glue workspace (8 rows of 16-byte packets per rank) and of the
    # draft_ehproj_table one (the packet, a BF16 hidden row and the rank's FP32 hidden-half share).
    glue = ws.LamportExchangeWorkspace(MAPPING, 8 * WORLD * 16)
    table = ws.LamportExchangeWorkspace(MAPPING, 8 * WORLD * (16 + 2 * 6144 + 4 * 6144 // WORLD))
    shards = {id(glue): 8, id(table): 6144 // WORLD, id(WORKSPACE): SHARD}
    for step in range(6):
        workspace = (WORKSPACE, glue, table)[step % 3]
        shard = shards[id(workspace)]
        rows = 1 if workspace is glue else 8
        per_rank = [_shard(rows, 4000 + step, r, shard) for r in range(WORLD)]
        _check(_gather(per_rank[RANK], workspace), per_rank, f"workspace {step % 3} step {step}")
    dist.barrier()


def check_cuda_graph_replays() -> None:
    """A captured gather reads its input buffer and the rotating flags at every replay."""
    rows = 8
    x_static = _shard(rows, 5000, RANK).clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _gather(x_static)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _gather(x_static)
    for replay in range(7):
        per_rank = [_shard(rows, 5100 + replay, r) for r in range(WORLD)]
        x_static.copy_(per_rank[RANK])
        torch.cuda.synchronize()
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        _check(out, per_rank, f"replay {replay}")
        dist.barrier()
    del graph
    torch.cuda.synchronize()


def check_late_ranks() -> None:
    """Ranks reach back-to-back gathers at different times."""
    issued: List[Tuple[List[torch.Tensor], torch.Tensor]] = []
    for step in range(6):
        rows = (8, 1, 5)[step % 3]
        per_rank = [_shard(rows, 6000 + step, r) for r in range(WORLD)]
        torch.cuda._sleep(((RANK * 5 + step) % WORLD) * 400_000)
        issued.append((per_rank, _gather(per_rank[RANK])))
    torch.cuda.synchronize()
    for step, (per_rank, out) in enumerate(issued):
        _check(out, per_rank, f"late ranks step {step}")


def check_the_op_refuses_rows_past_the_buffer() -> None:
    """Gathered rows that exceed one Lamport buffer are refused before any rank stores them."""
    rows = GATHER_BUFFER_BYTES // (WORLD * SHARD * torch.bfloat16.itemsize) + 1
    x = torch.zeros((rows, SHARD), dtype=torch.bfloat16, device="cuda")
    try:
        _gather(x)
    except RuntimeError as exc:
        assert "exceed the Lamport buffer" in str(exc), exc
    else:
        raise AssertionError(f"the op took {rows} rows past the buffer")


def check_the_guard_refuses_the_all_reduce_workspace() -> None:
    """The all-reduce workspace marks empty words with -0.0: is_valid refuses it."""
    comm_buffer, buffer_flags = ws.hop_workspace(MAPPING, 8, 6144)
    x = _shard(8, 7000, RANK)
    try:
        with validating(entry.mnnvl_allgather):
            entry.mnnvl_allgather(x, comm_buffer, buffer_flags, WORLD)
    except AssertionError as exc:
        assert "LamportExchangeWorkspace" in str(exc), exc
    else:
        raise AssertionError("is_valid admitted the all-reduce workspace")


def check_the_guard_refuses_flags_of_another_workspace() -> None:
    foreign = WORKSPACE.buffer_flags.clone()
    foreign.view(torch.int32)[2] += 16
    x = _shard(8, 7100, RANK)
    try:
        with validating(entry.mnnvl_allgather):
            entry.mnnvl_allgather(x, WORKSPACE.comm_buffer, foreign, WORLD)
    except AssertionError as exc:
        assert "flags of comm_buffer's workspace" in str(exc), exc
    else:
        raise AssertionError("is_valid admitted foreign flags")


CHECKS = (
    check_certified_cells,
    check_special_values_travel_unchanged,
    check_flags_rotate_and_the_previous_buffer_is_emptied,
    check_private_workspaces_side_by_side,
    check_cuda_graph_replays,
    check_late_ranks,
    check_the_op_refuses_rows_past_the_buffer,
    check_the_guard_refuses_the_all_reduce_workspace,
    check_the_guard_refuses_flags_of_another_workspace,
)


def _run_one_rank() -> None:
    global entry, ws, validating, RANK, WORLD, MAPPING, WORKSPACE
    RANK, WORLD, MAPPING = ranks.init_rank()
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.comm import (  # noqa: E501
        _mnnvl_workspace,
        mnnvl_allgather,
    )

    entry, ws = mnnvl_allgather, _mnnvl_workspace
    WORKSPACE = ws.LamportExchangeWorkspace(MAPPING, GATHER_BUFFER_BYTES)
    # Run as a script, not collected: the shared _validating module is reached the plain way.
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from _validating import validating as _validating_cm

    validating = _validating_cm
    ranks.run_checks(CHECKS, RANK)


if __name__ == "__main__":
    if ranks.WORKER_FLAG in sys.argv:
        _run_one_rank()
    ranks.spawn_ranks(Path(__file__).resolve(), DEADLINE_S)
    print("OK")
