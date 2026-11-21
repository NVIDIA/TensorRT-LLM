# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Certify `ops/comm/m31_allreduce` at world 4: every cell of the entry certified at this world size (the cells naming
``world`` 4 and the tp8 target's cells, which name none: the result does not depend on it) against the entry's
reference, under the target's AUTO strategy, inside `validating`.

The bands are the ones of the MiniMax-M3 tp4 target's matrix (``_m3_allreduce_op_matrix.py``): rank-order fp32 sums
with one BF16 rounding of the sum and one of the norm. A plain sum is within 1 BF16 ulp at the summands' magnitude;
the fused residual sum within 2.5 ulps at its operands' magnitude (a strategy may round the cross-rank sum before
adding the residual), and the normed rows within 2 ulps of the norm of the op's own residual sum.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _m31_allreduce_op_matrix.py
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

assert torch.cuda.is_available(), "m31_allreduce requires CUDA devices"

_WORKER_FLAG = "--rank-worker"
DEADLINE_S = 900
EPS = 1e-6

RANK = 0
WORLD = 1


def _ulps_at(out: torch.Tensor, ref: torch.Tensor, scale: torch.Tensor) -> float:
    """Max |out - ref| in BF16 ulps at the magnitude ``scale``: 2^(e - 8) for |v| in [2^(e-1), 2^e)."""
    _, e = torch.frexp(scale.abs().double())
    ulp = torch.ldexp(torch.ones_like(scale, dtype=torch.float64), torch.clamp(e - 8, min=-133))
    return ((out.double() - ref.double()).abs() / ulp).max().item()


def _inputs(seed: int, m: int, hidden: int):
    """Every rank's [m, H] BF16 partial rows, the residual and the norm weight (the same on every rank)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    xs = [torch.randn(m, hidden, generator=g, device="cuda").bfloat16() for _ in range(WORLD)]
    residual = (torch.randn(m, hidden, generator=g, device="cuda") * 4.0).bfloat16()
    w = (1.0 + 0.25 * torch.randn(hidden, generator=g, device="cuda")).bfloat16()
    return xs, residual, w


def check_certified_cells() -> None:
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceFusionOp, AllReduceParams
    from tensorrt_llm._torch.distributed.communicator import TorchDist
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.comm.m31_allreduce import (  # noqa: E501
        m3_allreduce as op,
    )
    from tensorrt_llm.functional import AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    # tests/unittest/_torch/modeling_v2, where _validating.py lives.
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from _validating import validating

    # TorchDist (the communicator under TLLM_DISABLE_MPI) asks Ray for each rank's node; these ranks run without Ray
    # on one node, so every rank reports the same node and its own device.
    def cluster_info(self):
        if self.cluster_info is None:
            self.cluster_info = [("127.0.0.1", r) for r in range(dist.get_world_size())]
        return self.cluster_info

    TorchDist._get_cluster_info = cluster_info
    mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=RANK, gpus_per_node=WORLD)
    ar = AllReduce(mapping=mapping, strategy=AllReduceStrategy.AUTO, dtype=torch.bfloat16)
    cells = [c for c in op.CELLS if c.spec.get("world", WORLD) == WORLD]
    assert cells, f"no cell is certified at world size {WORLD}"
    for i, cell in enumerate(cells):
        spec = cell.spec
        xs, residual, w = _inputs(200 + i, spec["m"], spec["hidden"])
        params = (
            None
            if spec.get("plain")
            else AllReduceParams(
                fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                residual=residual.clone(),
                norm_weight=w,
                eps=EPS,
                trigger_completion_at_end=False,
            )
        )
        # The reference first: the fused path may write the updated residual into ``params.residual``.
        ref = op.reference(xs, ar, params)
        with validating(op):
            out = op(xs[RANK], ar, params)
        torch.cuda.synchronize()
        total = sum(x.double() for x in xs)
        if params is None:
            err = _ulps_at(out, ref, total)
            assert err <= 1.0, (cell.why, err)
        else:
            err = _ulps_at(out[1], ref[1], torch.maximum(total.abs(), residual.double().abs()))
            assert err <= 2.5, (cell.why, "residual", err)
            h = out[1].double()
            own = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + EPS) * w.double()
            normed = _ulps_at(out[0], own, torch.maximum(out[0].double().abs(), own.abs()))
            assert normed <= 2.0, (cell.why, "normed", normed)
        dist.barrier()
        if RANK == 0:
            print(f"[cell] world={WORLD} {cell.why}: passed", flush=True)


def _run_one_rank(device: int) -> int:
    global RANK, WORLD
    os.environ.setdefault("TLLM_DISABLE_MPI", "1")
    RANK, WORLD = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(device)
    dist.init_process_group(
        "cpu:gloo,cuda:nccl", init_method="env://", device_id=torch.device("cuda", device)
    )
    import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

    try:
        check_certified_cells()
    except BaseException:
        import traceback  # only the rank subprocess's failure path needs it

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
    session is killed."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the devices this run owns"
    world_size = len([d for d in visible.split(",") if d.strip()])
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(os.environ, MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world_size),
               TLLM_DISABLE_MPI="1")  # fmt: skip
    env = {k: v for k, v in env.items() if not k.startswith(("PMIX_", "PMI_", "OMPI_", "SLURM_"))}
    children = []
    try:
        for r in range(world_size):
            children.append(
                subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), _WORKER_FLAG],
                    env=dict(env, RANK=str(r), LOCAL_RANK=str(r)),
                    start_new_session=True,
                )
            )
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
            raise AssertionError(f"the {world_size}-rank run failed: {failure}")
    finally:
        for p in children:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for p in children:
            p.wait()


if __name__ == "__main__":
    if _WORKER_FLAG in sys.argv:
        sys.exit(_run_one_rank(device=int(os.environ["RANK"])))
    _spawn_ranks()
    print("OK")
