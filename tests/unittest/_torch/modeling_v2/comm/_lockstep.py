# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the stateful collective matrices (``_mnnvl_allreduce_attn_res_op_matrix.py``): the launcher
with the world size and launcher as parameters, the rank setup, the native-torch reference of Kimi K3's residual
update, and exact-arithmetic payloads.

Started by file path from the launchers (this tree is not a package). Importing it pulls in torch only; the
rank body imports tensorrt_llm.

Launchers (``--launcher``):
  mpirun  this process re-executes the entry under ``mpirun -n <world size>``, one rank per device named in
          CUDA_VISIBLE_DEVICES, under a deadline it enforces by killing the process group (CI on one tray);
  srun    this process is already one of ``<world size>`` ranks started by an external launcher (e.g.
          ``srun -N 4 --ntasks-per-node 4 --mpi=pmix python <entry> --launcher srun --world-size 16`` across trays);
          the launcher owns the deadline. Rank r drives device r % (devices per node).
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import torch

WORKER_FLAG = "--rank-worker"
EMPTY_ROWS_EPS = 1e-6
STACK_DUMP_S = 600
"""A rank still running after this long prints every thread's Python stack (and again every period): a wedged
collective shows where it waits instead of only timing out."""


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--launcher", choices=("mpirun", "srun"), default="mpirun")
    parser.add_argument("--fabric-handle", choices=("auto", "on", "off"), default="auto")
    parser.add_argument(WORKER_FLAG, action="store_true")
    args, _ = parser.parse_known_args(argv)
    return args


def spawn(entry_file: str, args: argparse.Namespace, deadline_s: int) -> None:
    """Re-exec ``entry_file`` under mpirun with ``args.world_size`` ranks (default: one per visible device)."""
    visible = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    assert visible, "set CUDA_VISIBLE_DEVICES to the devices this run owns, e.g. 0,1,2,3"
    world = args.world_size or len(visible)
    assert 2 <= world <= len(visible), (
        f"world size {world} needs 2..{len(visible)} visible devices (one rank per device)"
    )
    command = ["mpirun", "-n", str(world), sys.executable, str(Path(entry_file).resolve()), WORKER_FLAG,
               "--world-size", str(world), "--fabric-handle", args.fabric_handle]  # fmt: skip
    print(f"[launcher] {' '.join(command)}", flush=True)
    process = subprocess.Popen(command, start_new_session=True)
    try:
        code = process.wait(timeout=deadline_s)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise AssertionError(
            f"the {world}-rank run did not finish in {deadline_s}s (wedged)"
        ) from None
    assert code == 0, f"the {world}-rank run exited {code}"


class Rank:
    """This process's rank: the MPI world, its device, the TP Mapping over every rank of the job."""

    def __init__(self, args: argparse.Namespace):
        from mpi4py import MPI

        from tensorrt_llm.mapping import Mapping

        faulthandler.dump_traceback_later(STACK_DUMP_S, repeat=True)
        self.MPI = MPI
        self.comm = MPI.COMM_WORLD
        self.rank, self.world = self.comm.Get_rank(), self.comm.Get_size()
        if args.world_size is not None:
            assert self.world == args.world_size, (
                f"the launcher started {self.world} ranks, --world-size says {args.world_size}"
            )
        assert self.world >= 2, f"a collective needs at least 2 ranks, got {self.world}"
        per_node = torch.cuda.device_count()
        torch.cuda.set_device(self.rank % per_node)
        self.mapping = Mapping(
            world_size=self.world, rank=self.rank, gpus_per_node=per_node, tp_size=self.world
        )
        self.fabric = {"auto": None, "on": True, "off": False}[args.fabric_handle]

    def barrier(self) -> None:
        torch.cuda.synchronize()
        self.comm.Barrier()

    def all_true(self, flag: bool) -> bool:
        return all(self.comm.allgather(bool(flag)))

    def same_on_ranks(self, *tensors: torch.Tensor) -> bool:
        """Bitwise equality of the tensors across ranks (their int16 words summed per row, compared exactly)."""
        mine = [t.contiguous().view(torch.int16).long().sum(dim=-1).cpu() for t in tensors]
        every = self.comm.allgather(mine)
        return all(torch.equal(a, b) for other in every[1:] for a, b in zip(every[0], other))

    def late(self, which: Optional[int], seconds: float = 0.005) -> None:
        """Rank ``which`` (None: nobody) starts its next launch ``seconds`` late."""
        if which == self.rank:
            time.sleep(seconds)


def run_checks(rank: Rank, checks: List[Callable[[], None]]) -> int:
    for check in checks:
        try:
            check()
        except BaseException:
            import traceback

            print(f"[rank {rank.rank}] FAILED {check.__name__}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            # A rank that leaves a collective early wedges every other rank in it.
            rank.comm.Abort(1)
        if rank.rank == 0:
            print(f"[rank 0] passed {check.__name__}", flush=True)
    rank.barrier()
    print(f"[rank {rank.rank}] {len(checks)} checks passed", flush=True)
    return 0


def exact_bf16(gen: torch.Generator, shape, lo: int, hi: int, scale: float) -> torch.Tensor:
    """bf16 integers in [lo, hi) times ``scale`` (a power of two): sums of a few of them are exact in fp32 and
    bf16, so a reference sum does not depend on the summation order."""
    ints = torch.randint(lo, hi, tuple(shape), generator=gen, device="cuda")
    return (ints.float() * scale).bfloat16()


def residual_update_ref(
    updated: torch.Tensor,
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    output_rms_weight: torch.Tensor,
    output_rms_eps: float,
) -> torch.Tensor:
    """Kimi K3's residual update after the all-reduce, in fp32 (HF ``_apply_attn_res`` + RMSNorm): over the
    candidates ``v = [block_residual..., updated]`` score ``sum(rmsnorm(v) * rms_weight * res_weight)``, softmax
    over the candidates, mix, RMSNorm with ``output_rms_weight``; bf16 out."""
    v = torch.cat([block_residual, updated.unsqueeze(0)], dim=0).float()
    rs = (v.square().mean(dim=-1) + rms_eps).rsqrt()
    logits = (v * rs[..., None] * (rms_weight.float() * res_weight.float())).sum(dim=-1)
    probs = torch.softmax(logits, dim=0)
    mixed = (probs[..., None] * v).sum(dim=0).bfloat16().float()
    out = mixed * (mixed.square().mean(dim=-1, keepdim=True) + output_rms_eps).rsqrt()
    return (out * output_rms_weight.float()).bfloat16()


def rel_err(got: torch.Tensor, want: torch.Tensor) -> float:
    return (
        (got.float() - want.float()).abs().max()
        / want.float().abs().max().clamp_min(EMPTY_ROWS_EPS)
    ).item()
