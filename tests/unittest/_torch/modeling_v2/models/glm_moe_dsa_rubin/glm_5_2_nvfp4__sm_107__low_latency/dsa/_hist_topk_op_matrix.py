# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the GLM hist_topk entry (the key-sharded decode indexer top-k).

A collective cannot be exercised in one process, so this script is its own launcher: run plainly,
it spawns one rank per device named in CUDA_VISIBLE_DEVICES, under a deadline it enforces by
killing every rank's process group. A broken exchange waits (up to the kernel's 120 s peer timeout)
rather than raising, so the deadline is what keeps it from taking the calling run down::

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _hist_topk_op_matrix.py

That run drives the entry's 4-rank cells. Its 16-rank cells (c1) need 16 ranks over 4 nodes: start
this file with ``--rank-worker`` on every task, one per GPU, with ``MASTER_ADDR`` / ``MASTER_PORT``
pointing at rank 0's node and a free port. ``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK`` and
``LOCAL_WORLD_SIZE`` fall back to Slurm's ``SLURM_PROCID``, ``SLURM_NTASKS``, ``SLURM_LOCALID`` and
``SLURM_NTASKS_PER_NODE``::

    srun -N4 --ntasks-per-node=4 env MASTER_ADDR=<node 0> MASTER_PORT=<port> \\
        python _hist_topk_op_matrix.py --rank-worker

Every rank builds the same full logits from a seed, scores nothing itself, and hands the op its own
shard (the columns past its keys are NaN, which would outrank +inf if they were read). The checks:
the cells against ``reference`` (row sets, the order within a row being unspecified), the raw
output bit-identical on all ranks, back-to-back calls through the rotating Lamport buffers, a call
without an exchange leaving the workspace untouched, CUDA-graph replays with new logits and
lengths, and the guard.

Not a pytest module: the ``check_*`` bodies are one fixed sequence inside a single job and share
the rank's workspace. The collected entry point is ``test_modeling_v2_hist_topk_op_matrix.py``,
which starts this file and turns its exit code into an assertion.
"""

import os
import signal
import socket
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Any, List, Tuple

import torch
import torch.distributed as dist

_WORKER_FLAG = "--rank-worker"
DEADLINE_S = 1200

# Bound inside the rank body, never in the launcher: importing the entry pulls in tensorrt_llm,
# which must not create CUDA or torch.distributed state in the launching process.
op: Any = None
canonical_rows: Any = None
shard_size: Any = None
shard_capacity: Any = None
validating: Any = None
WORKSPACE: Any = None
RANK = 0
WORLD = 1

_SPECIAL_BITS = (
    0x7F800000,  # +inf
    -0x00800000,  # -inf (0xFF800000)
    0x00000000,  # +0
    -0x80000000,  # -0 (0x80000000)
    0x7FC00000,  # NaN
    -0x00400000,  # NaN with the sign bit (0xFFC00000)
    0x7FFFFFFF,  # NaN whose key is clamped below the empty word
)


def _full_logits(spec: dict, seed: int) -> torch.Tensor:
    """The request's logits [rows, kv_len], the same on every rank."""
    gen = torch.Generator().manual_seed(seed)
    rows, kv_len, pattern = spec["rows"], spec["kv_len"], spec["pattern"]
    if pattern == "ties":
        logits = torch.randint(0, 64, (rows, kv_len), generator=gen).float()
    elif pattern == "equal":
        logits = torch.ones(rows, kv_len)
    elif pattern == "saturated":
        # Sign, exponent and the top two mantissa bits equal: one top-11-bit bin holds them all.
        logits = 1.0 + torch.rand(rows, kv_len, generator=gen) * 2.0**-10
    else:
        logits = torch.randn(rows, kv_len, generator=gen)
    if pattern == "special":
        values = torch.tensor(_SPECIAL_BITS, dtype=torch.int64).to(torch.int32).view(torch.float32)
        for value in values:
            index = torch.randint(0, kv_len, (rows, 64), generator=gen)
            logits.scatter_(1, index, value.expand(rows, 64).contiguous())
    return logits.cuda()


def _shard(full: torch.Tensor, kv_len: int, rank: int, width: int) -> torch.Tensor:
    """Rank ``rank``'s shard logits [rows, width]: its keys, then NaN."""
    size = shard_size(kv_len, WORLD)
    n = min(max(kv_len - rank * size, 0), size)
    shard = torch.full((full.shape[0], width), float("nan"), device="cuda")
    shard[:, :n] = full[:, rank * size : rank * size + n]
    return shard


def _call(
    full: torch.Tensor, spec: dict, width: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One call of the op on this rank's shard: (out, logits, kv_lens)."""
    kv_len = spec["kv_len"]
    logits = _shard(full, kv_len, RANK, max(width, shard_size(kv_len, WORLD)))
    kv_lens = torch.tensor([kv_len], dtype=torch.int32, device="cuda")
    out = torch.full((spec["rows"], 2048), -7, dtype=torch.int32, device="cuda")
    with validating(op):
        op(
            logits,
            kv_lens,
            out,
            WORKSPACE.comm_buffer,
            WORKSPACE.state,
            spec["next_n"],
            WORLD,
        )
    torch.cuda.synchronize()
    return out, logits, kv_lens


def _verify(full: torch.Tensor, spec: dict, out: torch.Tensor, kv_lens: torch.Tensor, where: str):
    """The row sets against ``reference``; the raw rows bit-identical on every rank."""
    width = shard_size(spec["kv_len"], WORLD)
    shards: List[torch.Tensor] = [_shard(full, spec["kv_len"], r, width) for r in range(WORLD)]
    expected = op.reference(
        shards, kv_lens, out, WORKSPACE.comm_buffer, WORKSPACE.state, spec["next_n"], WORLD
    )
    got = canonical_rows(out)
    if not torch.equal(got, expected):
        bad = (got != expected).any(dim=1).nonzero().flatten().tolist()
        raise AssertionError(f"{where}: rows {bad} hold another set than the reference")
    gathered = [torch.empty_like(out) for _ in range(WORLD)]
    dist.all_gather(gathered, out)
    for rank, rows in enumerate(gathered):
        assert torch.equal(rows, out), f"{where}: rank {rank} wrote other bits than rank {RANK}"
    assert int(WORKSPACE.state[2]) == 0, f"{where}: the op recorded a peer timeout"


def check_certified_cells() -> None:
    ran, skipped = 0, []
    for cell in op.CELLS:
        spec = cell.spec
        if spec["group_size"] != WORLD:
            skipped.append(cell.why)
            continue
        full = _full_logits(spec, zlib.crc32(cell.why.encode()) % 2**31)
        out, _, kv_lens = _call(full, spec)
        _verify(full, spec, out, kv_lens, cell.why)
        ran += 1
    assert ran > 0, f"no cell of hist_topk.CELLS runs at {WORLD} ranks"
    if RANK == 0:
        print(f"[rank 0] {ran} cells at {WORLD} ranks; not run here: {skipped}", flush=True)


def check_back_to_back_calls_rotate_the_lamport_buffers() -> None:
    """Calls with and without an exchange, of every size, one after the other: each exchange
    launch takes the next of the three buffers and clears what the previous one wrote."""
    lengths = (150_000, 1_000, 90_000, 2_051, 4 * 98_304, 60_000, 30_000, 250_000, 1_500, 5_000)
    patterns = ("randn", "randn", "ties", "randn", "saturated", "equal", "special", "randn")
    for step, kv_len in enumerate(lengths):
        spec = dict(kv_len=kv_len, pattern=patterns[step % len(patterns)], rows=8, next_n=8)
        full = _full_logits(spec, 1_000 + step)
        out, _, kv_lens = _call(full, spec)
        _verify(full, spec, out, kv_lens, f"call {step} ({kv_len} keys)")


def check_rows_without_an_exchange_leave_the_workspace_alone() -> None:
    state = WORKSPACE.state.clone()
    buffer = WORKSPACE.comm_buffer.clone()
    spec = dict(kv_len=2_000, pattern="randn", rows=8, next_n=8)
    full = _full_logits(spec, 2_000)
    out, _, kv_lens = _call(full, spec)
    _verify(full, spec, out, kv_lens, "2,000 keys")
    assert torch.equal(WORKSPACE.state, state), "a launch without an exchange moved the state"
    assert torch.equal(WORKSPACE.comm_buffer, buffer), "a launch without an exchange wrote"


def check_cuda_graph_replay_follows_new_logits_and_lengths() -> None:
    """The decode step is one captured graph: replays read the logits and the KV length in place,
    and every replay is an exchange launch of its own."""
    width = shard_capacity(300_000, WORLD)
    first = dict(kv_len=200_000, pattern="randn", rows=8, next_n=8)
    full = _full_logits(first, 3_000)
    out, logits, kv_lens = _call(full, first, width)
    _verify(full, first, out, kv_lens, "graph warm-up")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(logits, kv_lens, out, WORKSPACE.comm_buffer, WORKSPACE.state, 8, WORLD)
    for step, kv_len in enumerate((180_000, 300_000, 1_900, 120_001)):
        spec = dict(kv_len=kv_len, pattern="ties" if step % 2 else "randn", rows=8, next_n=8)
        full = _full_logits(spec, 3_001 + step)
        logits.copy_(_shard(full, kv_len, RANK, width))
        kv_lens.fill_(kv_len)
        out.fill_(-7)
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        _verify(full, spec, out, kv_lens, f"replay {step} ({kv_len} keys)")


def check_guard_refuses_a_batch_of_kv_lens() -> None:
    spec = dict(kv_len=100_000, pattern="randn", rows=8, next_n=8)
    logits = _shard(_full_logits(spec, 4_000), spec["kv_len"], RANK, shard_size(100_000, WORLD))
    kv_lens = torch.tensor([100_000, 7], dtype=torch.int32, device="cuda")
    out = torch.empty(8, 2048, dtype=torch.int32, device="cuda")
    try:
        with validating(op):
            op(logits, kv_lens, out, WORKSPACE.comm_buffer, WORKSPACE.state, 8, WORLD)
    except AssertionError as error:
        assert "kv_lens" in str(error), error
    else:
        raise AssertionError("is_valid admitted the KV lengths of two requests")


CHECKS = (
    check_certified_cells,
    check_back_to_back_calls_rotate_the_lamport_buffers,
    check_rows_without_an_exchange_leave_the_workspace_alone,
    check_cuda_graph_replay_follows_new_logits_and_lengths,
    check_guard_refuses_a_batch_of_kv_lens,
)


def _env_int(name: str, slurm_name: str, default: int) -> int:
    value = os.environ.get(name) or os.environ.get(slurm_name)
    return int(value) if value else default


def _run_one_rank() -> int:
    """Body of one rank: run every check; a failing rank exits nonzero and the launcher kills
    its siblings."""
    global op, canonical_rows, shard_size, shard_capacity, validating, WORKSPACE, RANK, WORLD
    os.environ.setdefault("TLLM_DISABLE_MPI", "1")
    rank = _env_int("RANK", "SLURM_PROCID", 0)
    world = _env_int("WORLD_SIZE", "SLURM_NTASKS", 1)
    local_rank = _env_int("LOCAL_RANK", "SLURM_LOCALID", rank)
    local_world = _env_int("LOCAL_WORLD_SIZE", "SLURM_NTASKS_PER_NODE", world)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "cpu:gloo,cuda:nccl",
        init_method="env://",
        rank=rank,
        world_size=world,
        device_id=torch.device("cuda", local_rank),
    )

    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.dsa import (  # noqa: E501
        _hist_topk,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.dsa import (  # noqa: E501
        hist_topk as entry,
    )
    from tensorrt_llm.mapping import Mapping

    # A spawned worker, not a collected test: the repo's __extra_import_path__ is not in play, so
    # the shared helper is reached the plain way, inside this rank process only.
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from _validating import validating as _validating_cm

    op = entry.hist_topk
    canonical_rows = entry.canonical_rows
    shard_size = _hist_topk.shard_size
    shard_capacity = _hist_topk.shard_capacity
    validating = _validating_cm
    RANK, WORLD = rank, world
    assert 2 <= WORLD <= _hist_topk.MAX_RANKS, f"a {WORLD}-rank group"
    mapping = Mapping(world_size=world, rank=rank, gpus_per_node=local_world, tp_size=world)
    WORKSPACE = _hist_topk.create_hist_topk_workspace(mapping, comm=dist.group.WORLD)

    for check in CHECKS:
        try:
            check()
            dist.barrier()
        except BaseException:
            import traceback

            print(f"[rank {RANK}] FAILED {check.__name__}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            # Exit hard: a rank that leaves a collective early wedges its siblings, and the
            # launcher kills the rest when it sees this exit.
            os._exit(13)
    torch.cuda.synchronize()
    print(f"[rank {RANK}] {len(CHECKS)} checks passed", flush=True)
    # Exit without tearing the process group down: these are throwaway processes.
    sys.stdout.flush()
    os._exit(0)


def _spawn_ranks() -> None:
    """Start one worker process per claimed device on this node; kill them all on the first
    nonzero exit or at the deadline."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the devices this run owns, e.g. 0,1,2,3"
    world_size = len([d for d in visible.split(",") if d.strip()])
    assert world_size >= 2, f"CUDA_VISIBLE_DEVICES names {world_size} device(s); need at least 2"
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(
        os.environ,
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        WORLD_SIZE=str(world_size),
        LOCAL_WORLD_SIZE=str(world_size),
        TLLM_DISABLE_MPI="1",
    )
    # Drop the launching step's PMIx/PMI/OMPI/Slurm namespace from the children: they would
    # otherwise register as participants of that step, and the first hard exit would have the
    # step killed before anything is reported.
    env = {k: v for k, v in env.items() if not k.startswith(("PMIX_", "PMI_", "OMPI_", "SLURM_"))}
    print(f"[launcher] spawning {world_size} ranks, rendezvous on :{port}", flush=True)
    children = [
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), _WORKER_FLAG],
            env=dict(env, RANK=str(r), LOCAL_RANK=str(r)),
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
    if _WORKER_FLAG in sys.argv:
        sys.exit(_run_one_rank())
    _spawn_ranks()
    print("OK")
