# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the sm_107 tp4 target's `symm_logits_gather_softcap`.

Every rank builds the target's workspace (64 rows, the target and draft slots) over the
default process group, as the target does at load, and drives the cells of its own world size
against the entry's reference, the NCCL all-gather and soft-cap chain, bitwise: the gather and
the upcast are copies and the soft-cap is the torch op's own `tanhf`. One draw of each cell
carries every finite bf16 bit pattern. A captured graph then replays the target and draft
gathers back to back on fresh logits, the way a decode step does.

The engine orders a slot's next write behind every peer's last read of it with the step's
all-reduces. Here the reference's NCCL all-gather between two calls does the same.

A collective cannot be exercised in one process. Two jobs, both this file:

- the world-4 cells, on one node. Run plainly, this script is its own launcher: it spawns one
  worker process per device named in CUDA_VISIBLE_DEVICES (four), under a deadline it enforces
  by killing every worker's session, since a collective whose ranks disagree hangs rather than
  raising. The group comes up from the rendezvous address the launcher exports.

      CUDA_VISIBLE_DEVICES=0,1,2,3 python _symm_logits_gather_op_matrix.py

- the world-8 cells, over two nodes of four devices. Started once per rank by an MPI launcher
  with `--mpi-rank-worker`, as the tp8 engine's ranks are: each rank takes the device of its
  node-local rank and the group comes up from rank 0's address broadcast over
  `MPI.COMM_WORLD`. A rank that wedges is ended by faulthandler at `MPI_DEADLINE_S`.

      <launcher> python _symm_logits_gather_op_matrix.py --mpi-rank-worker

Not a pytest module: the checks are one ordered sequence inside a job, reading rank state only
`_run_one_rank` binds. The collected entry point is `test_modeling_v2_symm_logits_gather.py`
beside this file. This file imports only torch and the standard library at module scope; the
entry is imported inside the rank body.
"""

import faulthandler
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

import torch
import torch.distributed as dist

_WORKER_FLAG = "--rank-worker"
_MPI_WORKER_FLAG = "--mpi-rank-worker"
DEADLINE_S = 600
MPI_DEADLINE_S = 600
WORLD_SIZE = 4
MPI_WORLD_SIZE = 8
#: The target's workspace: `_LOGITS_GATHER_ROWS` rows in each of its two slots.
ROWS = 64
TARGET_SLOT, DRAFT_SLOT = 0, 1
DRAWS = 3
GRAPH_REPLAYS = 20

# Bound in the rank body.
op: Any = None
build_workspace: Any = None
RANK = 0
WORLD = 1
WORKSPACES: Dict[int, Any] = {}


def _logits(rows: int, shard: int, seed: int) -> torch.Tensor:
    """One rank's lm_head shard: distinct per (rank, seed), spread over the soft-cap's range."""
    gen = torch.Generator(device="cuda").manual_seed(seed * 1009 + RANK)
    return (torch.randn(rows, shard, device="cuda", generator=gen) * 20).to(torch.bfloat16)


def _with_every_finite_bf16(local: torch.Tensor) -> torch.Tensor:
    """Overwrite the head of `local` with every finite bf16 bit pattern, subnormals and both
    zeros included, rotated per rank so every rank's block holds them in a different place."""
    bits = torch.arange(-32768, 32768, dtype=torch.int32, device="cuda").to(torch.int16)
    values = bits.view(torch.bfloat16)
    values = values[values.isfinite()]
    flat = local.view(-1)
    n = min(flat.numel(), values.numel())
    flat[:n] = torch.roll(values, RANK * 997)[:n]
    return local


def _assert_bitwise(out: torch.Tensor, ref: torch.Tensor, where: str) -> None:
    assert out.dtype == ref.dtype == torch.float32, (where, out.dtype, ref.dtype)
    assert out.shape == ref.shape, (where, out.shape, ref.shape)
    differ = out.view(torch.int32) != ref.view(torch.int32)
    if differ.any():
        first = differ.nonzero()[0].tolist()
        raise AssertionError(
            f"{where}: {int(differ.sum())} of {out.numel()} values differ, first at {first}: "
            f"{out[tuple(first)].item()!r} vs {ref[tuple(first)].item()!r}"
        )


def _mine():
    return [cell for cell in op.CELLS if cell.spec["world"] == WORLD]


def check_every_cell_against_the_reference() -> None:
    for cell in _mine():
        spec = cell.spec
        ws = WORKSPACES[spec["shard"]]
        for slot in (TARGET_SLOT, DRAFT_SLOT):
            for draw in range(DRAWS):
                local = _logits(spec["rows"], spec["shard"], cell.seed_base() + 2 * draw + slot)
                if draw == 0:
                    _with_every_finite_bf16(local)
                out = op(local, ws, slot, spec["cap"])
                ref = op.reference(local, ws, slot, spec["cap"])
                _assert_bitwise(out, ref, f"{cell.why}, slot {slot}, draw {draw}")


def check_one_row_and_a_full_slot() -> None:
    """The edges of what a slot holds: one row, and every row of the workspace."""
    shard = _mine()[0].spec["shard"]
    cap = _mine()[0].spec["cap"]
    for rows in (1, ROWS):
        for slot in (TARGET_SLOT, DRAFT_SLOT):
            local = _logits(rows, shard, 700 + rows + slot)
            out = op(local, WORKSPACES[shard], slot, cap)
            _assert_bitwise(out, op.reference(local, WORKSPACES[shard], slot, cap), f"{rows} rows")


def check_graph_replays_both_slots() -> None:
    """The target's gather then the draft's, captured into one graph and replayed on fresh
    logits each time; the output tensors are the ones the capture returned."""
    shard = _mine()[0].spec["shard"]
    cap = _mine()[0].spec["cap"]
    ws = WORKSPACES[shard]
    target = torch.zeros(16, shard, dtype=torch.bfloat16, device="cuda")
    draft = torch.zeros(12, shard, dtype=torch.bfloat16, device="cuda")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(target, ws, TARGET_SLOT, cap)
        op(draft, ws, DRAFT_SLOT, cap)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        target_out = op(target, ws, TARGET_SLOT, cap)
        draft_out = op(draft, ws, DRAFT_SLOT, cap)
    dist.barrier()
    for replay in range(GRAPH_REPLAYS):
        target.copy_(_logits(16, shard, 900 + 2 * replay))
        draft.copy_(_logits(12, shard, 901 + 2 * replay))
        graph.replay()
        _assert_bitwise(target_out, op.reference(target, ws, TARGET_SLOT, cap), f"replay {replay}")
        _assert_bitwise(draft_out, op.reference(draft, ws, DRAFT_SLOT, cap), f"replay {replay}")


CHECKS = (
    check_every_cell_against_the_reference,
    check_one_row_and_a_full_slot,
    check_graph_replays_both_slots,
)


def _run_one_rank(mpi: bool) -> int:
    """Body of one rank: run every check; a failing rank exits nonzero, and its launcher ends
    the siblings it would otherwise leave waiting in a collective."""
    global RANK, WORLD, op, build_workspace
    if mpi:
        faulthandler.dump_traceback_later(MPI_DEADLINE_S, exit=True)
        # A node's environment can hold a rendezvous address of its own; every rank drops it so
        # all of them take the MPI broadcast of rank 0's address.
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"):
            os.environ.pop(key, None)
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank, world = comm.Get_rank(), comm.Get_size()
        local = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    else:
        os.environ.setdefault("TLLM_DISABLE_MPI", "1")
        rank = local = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local)

    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import _lamport
    from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__tp4.ops.comm import (  # noqa: E501
        symm_logits_gather,
    )

    RANK, WORLD = rank, world
    want = MPI_WORLD_SIZE if mpi else WORLD_SIZE
    assert WORLD == want, f"this job drives the world-{want} cells, got {WORLD} ranks"
    op = symm_logits_gather.symm_logits_gather_softcap
    build_workspace = symm_logits_gather.build_workspace
    # The target's group: the Lamport workspace brings it up before the gather is built.
    _lamport.ensure_process_group(rank, world)
    assert dist.is_initialized() and dist.get_world_size() == world
    for shard in sorted({cell.spec["shard"] for cell in _mine()}):
        WORKSPACES[shard] = build_workspace(ROWS, shard, rank, world)
    torch.cuda.synchronize()
    if RANK == 0:
        dev = torch.cuda.current_device()
        print(
            f"[rank 0] {torch.cuda.get_device_name(dev)}; {WORLD} ranks, "
            f"{len(_mine())} cells, workspaces {sorted(WORKSPACES)}",
            flush=True,
        )

    for check in CHECKS:
        t0 = time.monotonic()
        try:
            check()
        except BaseException:
            print(f"[rank {RANK}] FAILED {check.__name__}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            # Exit hard: a rank that leaves early wedges every sibling in the next collective,
            # and the launcher ends them on this nonzero exit.
            os._exit(13)
        if RANK == 0:
            print(f"[rank 0] {check.__name__}: {time.monotonic() - t0:.1f}s", flush=True)
    dist.barrier()
    torch.cuda.synchronize()
    print(f"[rank {RANK}] {len(CHECKS)} checks passed", flush=True)
    sys.stdout.flush()
    if mpi:
        faulthandler.cancel_dump_traceback_later()
        dist.destroy_process_group()
        comm.Barrier()
        MPI.Finalize()
    # Exit without tearing the symmetric memory down: these are throwaway processes.
    os._exit(0)


def _spawn_ranks() -> None:
    """Start one worker per claimed device; on the first nonzero exit or the deadline, kill
    every worker's session."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the devices this run owns, e.g. 0,1,2,3"
    world_size = len([d for d in visible.split(",") if d.strip()])
    assert world_size == WORLD_SIZE, (
        f"CUDA_VISIBLE_DEVICES names {world_size} device(s); this job needs {WORLD_SIZE}"
    )
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(
        os.environ,
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        WORLD_SIZE=str(world_size),
        TLLM_DISABLE_MPI="1",
    )
    # A launching srun step's PMI namespace, inherited, would register the workers as its
    # participants; they rendezvous through torch.distributed instead.
    env = {k: v for k, v in env.items() if not k.startswith(("PMIX_", "PMI_", "OMPI_", "SLURM_"))}
    print(f"[launcher] spawning {world_size} ranks, rendezvous on :{port}", flush=True)
    children = [
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), _WORKER_FLAG],
            env=dict(env, RANK=str(r)),
            start_new_session=True,
        )
        for r in range(world_size)
    ]

    def _kill_all() -> None:
        for p in children:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for p in children:
            p.wait()

    deadline = time.monotonic() + DEADLINE_S
    live = set(range(world_size))
    failure = None
    while live and failure is None:
        if time.monotonic() > deadline:
            failure = f"did not finish in {DEADLINE_S}s (wedged)"
            break
        for r in sorted(live):
            code = children[r].poll()
            if code is None:
                continue
            live.discard(r)
            if code != 0:
                failure = f"rank {r} exited {code}"
                break
        time.sleep(0.5)
    if failure is not None:
        _kill_all()
        raise AssertionError(f"the {world_size}-rank run failed: {failure}")


if __name__ == "__main__":
    if _MPI_WORKER_FLAG in sys.argv:
        sys.exit(_run_one_rank(mpi=True))
    if _WORKER_FLAG in sys.argv:
        sys.exit(_run_one_rank(mpi=False))
    _spawn_ranks()
    print("OK")
