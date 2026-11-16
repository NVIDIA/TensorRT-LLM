# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Certify `comm/mnnvl_allreduce` over a TP group that spans GB300 trays.

Run as one srun task per rank, never spawned: the cells are world 8, two
trays, and a launcher on one node cannot reach the second. Each task runs

    MASTER_ADDR=<first node> MASTER_PORT=<port> \\
        python _mnnvl_allreduce_op_matrix.py --srun

under ``srun --ntasks=8 --ntasks-per-node=4 --kill-on-bad-exit=1``. The ranks
rendezvous through torch.distributed with MPI disabled, so the entry builds
its workspace on the mapping's TP process group -- the path a Ray-orchestrated
engine takes; under MPI only the communicator that carries the setup
handshake differs.

Not a pytest module, for the same reason as its siblings: the `check_*` bodies
are one ordered sequence inside one job and read rank state that only
`_run_one_rank` binds. Rank 0 prints each cell's measured error as
``[measure]`` lines; `_compare`'s docstring quotes them.
"""

import functools
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _allreduce_op_matrix  # noqa: E402
import _rank_job  # noqa: E402
from _allreduce_op_matrix import _payload, _residual_and_weight  # noqa: E402

GPUS_PER_NODE = 4
"""A GB300 tray. The mapping needs it to see the group span two nodes."""

MAX_ROWS = 8192
"""The row budget the workspace is built for: the engine's max_num_tokens."""

GRAPH_BATCH_SIZES = (1, 8, 13, 128)

# Bound inside the rank body. `mnnvl_allreduce` is the call surface, a partial
# carrying the workspace arguments; `entry` is the module, whose
# `mnnvl_allreduce` instance holds `reference`, `is_valid` and `CELLS`.
mnnvl_allreduce: Any = None
entry: Any = None
validating: Any = None
WORKSPACE_ARGS: dict = {}
RANK = 0
WORLD = 1


def _compare(out, ref) -> None:
    """`comm/allreduce`'s band: 5 ulp per element against the row scale,
    1.25 ulp relative RMS, on both outputs.

    Measured at world 8 over two GB300 trays (bf16, 5 seeds per cell, rows 1
    to 8192), worst case per cell group:

    - hidden 2880: 2.56 ulp per element and 0.88 ulp RMS on `norm_out`,
      1.97 and 0.72 on `residual_out`.
    - hidden 7168: 2.98 ulp per element and 0.84 ulp RMS on `norm_out`,
      2.00 and 0.70 on `residual_out`.

    The same as `comm/allreduce` inside one tray, and flat in the row count --
    this kernel accumulates in fp32 at every size, unlike the NCCL fallback.
    A dropped rank moves the result by tens of ulp;
    `check_the_gate_discriminates_a_wrong_reduction` checks the band refuses
    one.
    """
    _allreduce_op_matrix._compare(out, ref)


def _call(x, residual, weight, eps=1e-5):
    return mnnvl_allreduce(x, residual, norm_weight=weight, eps=eps)


def _reference(rows, hidden, seed, residual, weight, eps=1e-5):
    inputs = [_payload(r, rows, hidden, seed) for r in range(WORLD)]
    return entry.mnnvl_allreduce.reference(inputs, residual, weight, eps)


def _ulp_stats(out, ref):
    """Worst element error against the row scale, and relative RMS, in bf16 ulp."""
    ulp = 2.0**-8
    o, r = out.float().reshape(-1, out.shape[-1]), ref.float().reshape(-1, ref.shape[-1])
    row = r.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    elem = ((o - r).abs() / row).max().item() / ulp
    rms = ((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt().clamp_min(1e-9)).item() / ulp
    return elem, rms


def check_certified_cells() -> None:
    """Every world-matching cell, inside `validating`, against the reference
    and band. Rank 0 also reports the error it measured over several seeds."""
    cells = [c for c in entry.mnnvl_allreduce.CELLS if c.spec["world"] == WORLD]
    assert cells, f"no cell is certified at world size {WORLD}"
    for i, cell in enumerate(cells):
        spec = cell.spec
        worst = [0.0, 0.0, 0.0, 0.0]
        for s in range(5):
            seed = 7000 + 17 * i + 1000 * s
            x = _payload(RANK, spec["rows"], spec["hidden"], seed)
            residual, weight = _residual_and_weight(spec["rows"], spec["hidden"], seed)
            with validating(entry.mnnvl_allreduce):
                out = _call(x, residual, weight, spec["eps"])
            ref = _reference(spec["rows"], spec["hidden"], seed, residual, weight, spec["eps"])
            for k, (o, r) in enumerate(zip(out, ref)):
                e, rms = _ulp_stats(o, r)
                worst[2 * k] = max(worst[2 * k], e)
                worst[2 * k + 1] = max(worst[2 * k + 1], rms)
            try:
                _compare(out, ref)
            except AssertionError as exc:
                raise AssertionError(f"{cell.why}: {exc}") from exc
            dist.barrier()
        if RANK == 0:
            print(
                f"[measure] world={WORLD} hidden={spec['hidden']} rows={spec['rows']} "
                f"norm_out elem={worst[0]:.2f} rms={worst[1]:.3f} "
                f"residual_out elem={worst[2]:.2f} rms={worst[3]:.3f} (ulp, 5 seeds)",
                flush=True,
            )


def check_inputs_are_untouched() -> None:
    x = _payload(RANK, 64, 2880, 21)
    residual, weight = _residual_and_weight(64, 2880, 21)
    x0, r0 = x.clone(), residual.clone()
    _call(x, residual, weight)
    torch.cuda.synchronize()
    assert torch.equal(x, x0) and torch.equal(residual, r0), "the call wrote an input"
    dist.barrier()


def check_cuda_graph_replay_at_engine_batch_sizes() -> None:
    """Captured once per decode batch size, refilled and replayed: the Lamport
    buffers rotate across calls, and a replay must still read fresh data."""
    for rows in GRAPH_BATCH_SIZES:
        x = _payload(RANK, rows, 2880, 31)
        residual, weight = _residual_and_weight(rows, 2880, 31)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                _call(x, residual, weight)
        torch.cuda.current_stream().wait_stream(stream)
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = _call(x, residual, weight)
        for rep in range(3):
            seed = 31 + 100 * (rep + 1) + rows
            x.copy_(_payload(RANK, rows, 2880, seed))
            fresh_res, _ = _residual_and_weight(rows, 2880, seed)
            residual.copy_(fresh_res)
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
            ref = _reference(rows, 2880, seed, residual, weight)
            try:
                _compare(out, ref)
            except AssertionError as exc:
                raise AssertionError(f"graph replay rows={rows} rep={rep}: {exc}") from exc
        dist.barrier()


def check_the_gate_discriminates_a_wrong_reduction() -> None:
    """A reference missing one rank must fail the band."""
    rows = 64
    x = _payload(RANK, rows, 2880, 81)
    residual, weight = _residual_and_weight(rows, 2880, 81)
    out = _call(x, residual, weight)
    short = [_payload(r, rows, 2880, 81) for r in range(WORLD - 1)]
    wrong = entry.mnnvl_allreduce.reference(short, residual, weight, 1e-5)
    try:
        _compare(out, wrong)
    except AssertionError:
        pass
    else:
        raise AssertionError("the band accepted a sum that is missing one rank")
    dist.barrier()


def check_the_guard_refuses_a_call_past_the_workspace() -> None:
    """The kernel never checks the row count against its buffer; the guard
    must. Presented a buffer view too small for the call, it refuses before
    the op runs, so no rank enters the collective."""
    rows = 2048
    x = _payload(RANK, rows, 2880, 91)
    residual, weight = _residual_and_weight(rows, 2880, 91)
    small = WORKSPACE_ARGS["comm_buffer"][:, :1024]
    try:
        with validating(entry.mnnvl_allreduce):
            mnnvl_allreduce(x, residual, norm_weight=weight, eps=1e-5, comm_buffer=small)
    except AssertionError as exc:
        assert "workspace" in str(exc), exc
    else:
        raise AssertionError("the guard admitted a call larger than the workspace")
    dist.barrier()


CHECKS = (
    check_certified_cells,
    check_inputs_are_untouched,
    check_cuda_graph_replay_at_engine_batch_sizes,
    check_the_gate_discriminates_a_wrong_reduction,
    check_the_guard_refuses_a_call_past_the_workspace,
)


def _run_one_rank(device: int) -> int:
    """Body of one srun task. `device` is the node-local index the srun step
    gave this task (see `_rank_job.adopt_srun_env`)."""
    global RANK, WORLD, WORKSPACE_ARGS, mnnvl_allreduce, entry, validating
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(device)
    # cuda:nccl,cpu:gloo, as a Ray worker builds it: the workspace's setup
    # handshake runs on the TP group's CPU backend.
    dist.init_process_group(
        "cuda:nccl,cpu:gloo", init_method="env://", device_id=torch.device("cuda", device)
    )
    RANK, WORLD = rank, world

    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import mnnvl_allreduce as entry
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=world, rank=rank, gpus_per_node=GPUS_PER_NODE, tp_size=world)
    assert mapping.is_multi_node(), "these cells certify a group that spans trays"
    # What a target does in __init__: build the workspace once and keep its
    # arguments. One workspace for every cell, sized by the widest hidden a
    # cell certifies. The partial is only the call surface; the guard is armed
    # on `entry.mnnvl_allreduce` itself.
    widest = max(cell.spec["hidden"] for cell in entry.mnnvl_allreduce.CELLS)
    WORKSPACE_ARGS = entry.mnnvl_workspace_args(
        mapping, max_rows=MAX_ROWS, hidden=widest, dtype=torch.bfloat16
    )
    mnnvl_allreduce = functools.partial(entry.mnnvl_allreduce, **WORKSPACE_ARGS)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from _validating import validating as _validating_cm

    validating = _validating_cm

    for check in CHECKS:
        try:
            check()
        except BaseException:
            import traceback

            print(f"[rank {RANK}] FAILED {check.__name__}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(13)
    dist.barrier()
    torch.cuda.synchronize()
    print(f"[rank {RANK}] {len(CHECKS)} checks passed", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    assert "--srun" in sys.argv, "run one rank per srun task: see the module docstring"
    sys.exit(_run_one_rank(device=_rank_job.adopt_srun_env()))
