# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Certify the K3.1 ULL target's Helix entries on one GB300 tray (Helix CP over 4 ranks).

Its own launcher (``_k31_ull_comm_ranks``): ``python _helix_matrix.py`` with
``CUDA_VISIBLE_DEVICES`` naming 4 devices. Each rank builds the workspaces exactly as the target
does (``comm.FlatHelix._tail_workspace`` and ``._fifo_workspace`` on a Helix CP mapping), then:

- ``helix_tail.CELLS`` and ``helix_tail_gated.CELLS`` at P = 4, several launches each on the one
  shared tail workspace (the launches alternate its halves): every rank's rows within a few BF16
  ulp of the FP64 reference over all ranks' inputs, the gated rows bitwise the two PyTorch BF16
  kernels applied to the ungated ones;
- a tail captured in a CUDA graph and replayed with fresh inputs;
- ``helix_alltoall_native.CELLS``: every received slot bitwise what its sender sent (zeros and
  ``(-inf, 0)`` for the sender's zero-KV tokens).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _helix_inputs as inputs  # noqa: E402
import _k31_ull_comm_ranks as ranks  # noqa: E402

DEADLINE_S = 900
LAUNCHES = 3

RANK = 0
WORLD = 1
TAIL = None
FIFO = None
ops = None
validating = None


def _tail_args():
    return (
        TAIL.memory.ptr,
        TAIL.memory.rank_stride,
        TAIL.half_bytes,
        TAIL.state,
        TAIL.max_ctas,
        True,
    )


def _all_ranks(spec, seed):
    return [inputs.rank_inputs(spec, seed, r) for r in range(WORLD)]


def _width(spec):
    return inputs.HEADS // spec["cp_size"] * inputs.VALUE


def check_tail_cells() -> None:
    op = ops["helix_tail"]
    w_uv = inputs.w_uv(seed=21)
    for i, cell in enumerate(c for c in op.CELLS if c.spec["cp_size"] == WORLD):
        spec = cell.spec
        for launch in range(LAUNCHES):
            seed = 100 * i + launch
            group = _all_ranks(spec, seed)
            rows, stats, mask = group[RANK]
            out = torch.empty(spec["tokens"], _width(spec), dtype=torch.bfloat16, device="cuda")
            with validating(op):
                op(rows, stats, w_uv, out, mask, RANK, WORLD, *_tail_args())
            torch.cuda.synchronize()
            ref = op.reference(
                [g[0] for g in group], [g[1] for g in group], w_uv, out, [g[2] for g in group],
                RANK, WORLD,
            )  # fmt: skip
            inputs.assert_within_bf16_ulp(out, ref, 3, f"{cell.why}, launch {launch}")
            dist.barrier()


def check_gated_cells() -> None:
    plain_op, op = ops["helix_tail"], ops["helix_tail_gated"]
    w_uv = inputs.w_uv(seed=23)
    for i, cell in enumerate(c for c in op.CELLS if c.spec["cp_size"] == WORLD):
        spec = cell.spec
        for launch in range(LAUNCHES):
            seed = 200 + 100 * i + launch
            group = _all_ranks(spec, seed)
            rows, stats, mask = group[RANK]
            gate = inputs.gate(spec, seed)
            plain = torch.empty(spec["tokens"], _width(spec), dtype=torch.bfloat16, device="cuda")
            out = torch.empty_like(plain)
            plain_op(rows, stats, w_uv, plain, mask, RANK, WORLD, *_tail_args())
            with validating(op):
                op(rows, stats, w_uv, gate, out, mask, RANK, WORLD, *_tail_args())
            torch.cuda.synchronize()
            ranks.assert_bitwise(out, plain * gate.sigmoid(), f"{cell.why}: gated vs torch gate")
            ref = op.reference(
                [g[0] for g in group], [g[1] for g in group], w_uv, gate, out,
                [g[2] for g in group], RANK, WORLD,
            )  # fmt: skip
            inputs.assert_within_bf16_ulp(out, ref, 4, f"{cell.why}, launch {launch}")
            dist.barrier()


def check_tail_graph_replay() -> None:
    """One launch captured, replayed with fresh inputs: the device launch counter advances in
    every replay, so the replays alternate the halves and each reads its own rows."""
    op = ops["helix_tail"]
    spec = dict(tokens=8, cp_size=WORLD, heads=inputs.HEADS, masked=False)
    w_uv = inputs.w_uv(seed=29)
    rows, stats, mask = inputs.rank_inputs(spec, 900, RANK)
    out = torch.empty(spec["tokens"], _width(spec), dtype=torch.bfloat16, device="cuda")
    op(rows, stats, w_uv, out, None, RANK, WORLD, *_tail_args())  # compiles outside capture
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(rows, stats, w_uv, out, None, RANK, WORLD, *_tail_args())
    for rep in range(4):
        group = _all_ranks(spec, 910 + rep)
        rows.copy_(group[RANK][0])
        stats.copy_(group[RANK][1])
        torch.cuda.synchronize()
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        ref = op.reference(
            [g[0] for g in group], [g[1] for g in group], w_uv, out, [None] * WORLD, RANK, WORLD
        )
        inputs.assert_within_bf16_ulp(out, ref, 3, f"graph replay {rep}")
        dist.barrier()


def check_alltoall_cells() -> None:
    op = ops["helix_alltoall_native"]
    workspace, _, cp_rank, cp_size = FIFO
    assert (cp_rank, cp_size) == (RANK, WORLD)
    for i, cell in enumerate(c for c in op.CELLS if c.spec["cp_size"] == WORLD):
        spec = cell.spec
        local = spec["heads"] // WORLD
        for call in range(LAUNCHES):
            seed = 400 + 100 * i + call
            partials, stats, masks = [], [], []
            for r in range(WORLD):
                g = torch.Generator(device="cuda").manual_seed(seed * 131 + r)
                shape = (spec["tokens"], WORLD)
                o = torch.randn(*shape, local * inputs.VALUE, generator=g, device="cuda")
                peak = torch.randn(*shape, local, generator=g, device="cuda") * 4
                total = torch.rand(*shape, local, generator=g, device="cuda") * 50 + 1
                partials.append(o.to(torch.bfloat16))
                stats.append(torch.stack((peak, total), dim=-1).view(*shape, local * 2))
                masks.append(inputs.rank_inputs(spec, seed, r)[2])
            with validating(op):
                got_o, got_s = op(partials[RANK], stats[RANK], workspace, RANK, WORLD, masks[RANK])
            torch.cuda.synchronize()
            ref_o, ref_s = op.reference(partials, stats, workspace, RANK, WORLD, masks)
            ranks.assert_bitwise(got_o, ref_o, f"{cell.why} call {call}: values")
            ranks.assert_bitwise(got_s, ref_s, f"{cell.why} call {call}: stats")
            dist.barrier()


CHECKS = (check_tail_cells, check_gated_cells, check_tail_graph_replay, check_alltoall_cells)


def _worker() -> None:
    global RANK, WORLD, TAIL, FIFO, ops, validating
    RANK, WORLD, mapping = ranks.init_rank(helix=True)
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1 import (  # noqa: E501
        modeling as comm,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm import (  # noqa: E501
        rubin_ull_helix_alltoall_native as helix_alltoall_native,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm import (
        rubin_ull_helix_tail as helix_tail,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm import (
        rubin_ull_helix_tail_gated as helix_tail_gated,
    )

    ops = {
        "helix_tail": helix_tail.helix_tail,
        "helix_tail_gated": helix_tail_gated.helix_tail_gated,
        "helix_alltoall_native": helix_alltoall_native.helix_alltoall_native,
    }
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from _validating import validating as validating_cm

    validating = validating_cm
    FIFO = comm.FlatHelix._fifo_workspace(mapping)
    TAIL = comm.FlatHelix._tail_workspace(mapping)
    ranks.run_checks(CHECKS, RANK)


if __name__ == "__main__":
    if ranks.WORKER_FLAG in sys.argv:
        _worker()
    else:
        ranks.spawn_ranks(Path(__file__).resolve(), DEADLINE_S)
