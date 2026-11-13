# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU check, 2 ranks (SM107): TP2W_S2_TAIL (``minimax_m31_s2_moe_tail``) and the built-in one-shot Lamport
all-reduce alternating on one ``AllReduce`` workspace, as the decode layer runs them (attention boundary: the built-in
one; MoE boundary: the fused tail).

Every Lamport kernel on the workspace clears the sub-buffer of the call before it by the clear size that call left,
in bytes. A tail call followed by built-in calls must leave its own byte count: leaving less keeps stale words that a
later call on the same sub-buffer takes as already arrived (a wrong sum, or a rank running a call ahead whose writes
the peer's clear then erases: a hang). One rank per round starts each call late, so the other rank polls the late
rank's part of the buffer before it arrives and any stale word is taken.

Small integers keep every sum exact: the fused tail's updated residual (finalize + shared, all-reduce, + residual) and
the built-in all-reduce's output are compared bit for bit with the exact sums.

Its own 2-rank launcher, as ``_m3_allreduce_op_matrix.py`` is; the collected entry point is
``test_modeling_v2_m31_s2_tail_lamport.py``.

    CUDA_VISIBLE_DEVICES=0,1 python _m31_s2_tail_lamport_op_matrix.py
"""

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist

assert torch.cuda.is_available(), "the fused MoE tail requires CUDA devices"

_WORKER_FLAG = "--rank-worker"
DEADLINE_S = 600

HIDDEN = 6144
TOP_K = 4
EPS = 1e-6
# About 10 ms of GPU time: the other rank polls the late rank's part of the buffer before it is written.
LATE_RANK_SLEEP_CYCLES = 20_000_000
# (kind, rows) per call: a large fused tail, two small built-in calls, the large fused tail again, so the last call
# lands on the first call's sub-buffer (four calls per round on a triple buffer: each round starts on another one).
CALLS = (
    ("tail", 224),
    ("stock", 8),
    ("stock", 8),
    ("tail", 224),
    ("stock", 224),
    ("tail", 8),
    ("tail", 8),
)
ROUNDS = 3

RANK = 0
WORLD = 1


def _ints(seed: int, *shape: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(-4, 5, shape, generator=g, device="cuda").to(torch.bfloat16)


def _tail_inputs(step: int, rank: int, rows: int):
    """One rank's routed parts (identity permutation, unit scales) and shared output, all small integers."""
    fc2 = _ints(1000 * step + 10 * rank + 1, rows * TOP_K, HIDDEN)
    shared = _ints(1000 * step + 10 * rank + 2, rows, HIDDEN)
    return fc2, shared


def _single_node_cluster_info() -> None:
    """Ranks spawned without Ray on one node: every rank reports the same node and its own device."""
    from tensorrt_llm._torch.distributed.communicator import TorchDist

    def cluster_info(self):
        if self.cluster_info is None:
            self.cluster_info = [("127.0.0.1", r) for r in range(dist.get_world_size())]
        return self.cluster_info

    TorchDist._get_cluster_info = cluster_info


@torch.inference_mode()
def _run_matrix() -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2 import (
        native_ops,
    )
    from tensorrt_llm._torch.distributed import AllReduce
    from tensorrt_llm.functional import AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    op = getattr(native_ops, "minimax_m31_s2_moe_tail", None)
    assert op is not None, "the build does not carry minimax_m31_s2_moe_tail"
    _single_node_cluster_info()
    mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=RANK, gpus_per_node=WORLD)
    ar = AllReduce(mapping=mapping, strategy=AllReduceStrategy.ONESHOT, dtype=torch.bfloat16)
    ws = ar.workspace
    assert ws is not None, "the one-shot all-reduce has no Lamport workspace"
    norm_w = torch.ones(HIDDEN, device="cuda", dtype=torch.bfloat16)
    for step, (kind, rows) in enumerate(CALLS * ROUNDS):
        late = step // len(CALLS) % WORLD
        residual = _ints(1000 * step + 7, rows, HIDDEN)
        if kind == "tail":
            parts = [_tail_inputs(step, r, rows) for r in range(WORLD)]
            expected = residual.float()
            for fc2, shared in parts:
                expected = expected + fc2.float().view(rows, TOP_K, HIDDEN).sum(1) + shared.float()
            fc2, shared = parts[RANK]
            perm = torch.arange(rows * TOP_K, device="cuda", dtype=torch.int32).view(rows, TOP_K)
            scales = torch.ones(rows, TOP_K, device="cuda", dtype=torch.float32)
            if RANK == late:
                torch.cuda._sleep(LATE_RANK_SLEEP_CYCLES)
            _, updated, _, _ = op(
                fc2, scales, perm, shared, residual, norm_w, ws, RANK, WORLD, EPS, False
            )
            got = updated
        else:
            xs = [_ints(1000 * step + 10 * r + 3, rows, HIDDEN) for r in range(WORLD)]
            expected = sum(x.float() for x in xs)
            if RANK == late:
                torch.cuda._sleep(LATE_RANK_SLEEP_CYCLES)
            got = ar(xs[RANK])
        torch.cuda.synchronize()
        bad = int((got.float() != expected).sum().item())
        assert bad == 0, (
            f"step {step} ({kind}, {rows} rows, late rank {late}): {bad} elements differ from the exact sum"
        )
        dist.barrier()
    if RANK == 0:
        print(f"[s2-lamport] {len(CALLS) * ROUNDS} alternating calls exact", flush=True)


def _run_one_rank() -> int:
    global RANK, WORLD
    os.environ.setdefault("TLLM_DISABLE_MPI", "1")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "cpu:gloo,cuda:nccl", init_method="env://", device_id=torch.device("cuda", rank)
    )
    RANK, WORLD = rank, world
    assert WORLD == 2, f"the fused tail runs at tp2, got {WORLD} ranks"
    try:
        _run_matrix()
    except BaseException:
        import traceback

        print(f"[rank {RANK}] FAILED", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(13)
    dist.barrier()
    torch.cuda.synchronize()
    print(f"[rank {RANK}] passed", flush=True)
    sys.stdout.flush()
    os._exit(0)


def _spawn_ranks() -> None:
    """One worker per claimed device, each in its own session; on the first nonzero exit or the deadline every
    session is killed (a clear-size mismatch can wedge both ranks inside the kernel)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the 2 devices this run owns"
    world_size = len([d for d in visible.split(",") if d.strip()])
    assert world_size == 2, (
        f"CUDA_VISIBLE_DEVICES names {world_size} device(s); the check runs at 2"
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
    env = {k: v for k, v in env.items() if not k.startswith(("PMIX_", "PMI_", "OMPI_", "SLURM_"))}
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
    failure: Optional[str] = None
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
