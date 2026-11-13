# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the tp2 target's fused MoE tail (``native_ops.minimax_m31_s2_moe_tail``,
TP2W_S2_TAIL), 2 ranks (SM107): bit for bit the chain it fuses,

    m3_moe_finalize -> AllReduce(ONESHOT, RESIDUAL_RMS_NORM) -> mxfp8_quantize(norm_out, swizzled).

Every output is compared: norm_out, residual_out, the MXFP8 bytes and the valid rows' scales; the pad rows' scales
are 0. Inputs: random rows, dropped top-k slots, and rows scaled by 2^-12..2^12; BF16 and FP32 routing weights.

Its own 2-rank launcher, as ``_m3_allreduce_op_matrix.py`` is; the collected entry point is
``test_modeling_v2_m31_s2_moe_tail.py``.

    CUDA_VISIBLE_DEVICES=0,1 python _m31_s2_moe_tail_op_matrix.py
"""

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.distributed as dist

assert torch.cuda.is_available(), "the fused MoE tail requires CUDA devices"

_WORKER_FLAG = "--rank-worker"
DEADLINE_S = 900

HIDDEN, TOPK, EPS = 6144, 4, 1e-6
ROWS = (1, 8, 48, 64, 88, 112, 128, 136, 176, 224)

RANK = 0
WORLD = 1


def _sf_valid_mask(rows: int, dev: torch.device) -> torch.Tensor:
    """The swizzled (128 x 4) scale bytes that belong to the first ``rows`` rows."""
    cols = HIDDEN // 32
    padded = (rows + 127) // 128 * 128
    m = torch.zeros(padded * cols, dtype=torch.bool, device=dev)
    r = torch.arange(padded, device=dev).view(-1, 1).expand(padded, cols)
    c = torch.arange(cols, device=dev).view(1, -1).expand(padded, cols)
    off = (
        (r // 128) * (cols // 4 * 512)
        + (c // 4) * 512
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + (c % 4)
    )
    m[off.reshape(-1)] = (r < rows).reshape(-1)
    return m


# ---------------------------------------------------------------- the matrix


def _single_node_cluster_info() -> None:
    """``TorchDist`` (the communicator under ``TLLM_DISABLE_MPI``) asks Ray for each rank's node and GPU to build its
    node-local group. These ranks are spawned without Ray on one node, so every rank reports the same node and its
    own device."""
    from tensorrt_llm._torch.distributed.communicator import TorchDist

    def cluster_info(self):
        if self.cluster_info is None:
            self.cluster_info = [("127.0.0.1", r) for r in range(dist.get_world_size())]
        return self.cluster_info

    TorchDist._get_cluster_info = cluster_info


def _run_matrix() -> None:
    import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers trtllm::mxfp8_quantize
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2 import (  # noqa: E501
        native_ops,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2.kernels.m3_moe_finalize import (  # noqa: E501
        m3_moe_finalize,
    )
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceFusionOp, AllReduceParams
    from tensorrt_llm.functional import AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    _single_node_cluster_info()
    op = native_ops.minimax_m31_s2_moe_tail
    dev = torch.device("cuda", RANK)
    mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=RANK, gpus_per_node=WORLD)
    ar = AllReduce(mapping=mapping, strategy=AllReduceStrategy.ONESHOT, dtype=torch.bfloat16)
    g0 = torch.Generator(device=dev).manual_seed(7)
    w_norm = (1.0 + 0.1 * torch.randn(HIDDEN, generator=g0, device=dev)).to(torch.bfloat16)
    w_norm[::97] = -w_norm[::97]
    bad: List[tuple] = []
    for rows in ROWS:
        for sdt in (torch.bfloat16, torch.float32):
            for seed in (0, 1, 2):
                g = torch.Generator(device=dev).manual_seed(1000 * seed + 17 * RANK + rows)
                n_perm = rows * TOPK
                fc2 = (
                    torch.randn(((n_perm + 127) // 128 * 128, HIDDEN), generator=g, device=dev)
                    * 0.5
                )
                if seed == 2:  # rows scaled by 2^-12..2^12
                    fc2 = fc2 * torch.pow(
                        2.0,
                        torch.randint(-12, 13, (fc2.shape[0], 1), generator=g, device=dev).float(),
                    )
                fc2 = fc2.to(torch.bfloat16)
                perm = (
                    torch.randperm(n_perm, generator=g, device=dev).to(torch.int32).view(rows, TOPK)
                )
                if seed == 1:  # dropped slots
                    drop = torch.rand((rows, TOPK), generator=g, device=dev) < 0.02
                    perm = torch.where(drop, torch.full_like(perm, -1), perm)
                perm = perm.contiguous()
                scales = torch.softmax(torch.randn((rows, TOPK), generator=g, device=dev), -1).to(
                    sdt
                )
                shared = (torch.randn((rows, HIDDEN), generator=g, device=dev) * 0.5).to(
                    torch.bfloat16
                )
                residual = (torch.randn((rows, HIDDEN), generator=g, device=dev) * 2).to(
                    torch.bfloat16
                )
                o = m3_moe_finalize(fc2, scales.contiguous(), perm, shared)
                n, r = ar(
                    o,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                        residual=residual,
                        norm_weight=w_norm,
                        eps=EPS,
                        trigger_completion_at_end=False,
                    ),
                )
                q, sf = torch.ops.trtllm.mxfp8_quantize(n, True)
                b = op(
                    fc2,
                    scales.contiguous(),
                    perm,
                    shared,
                    residual,
                    w_norm,
                    ar.workspace,
                    RANK,
                    WORLD,
                    EPS,
                    False,
                )
                torch.cuda.synchronize()
                valid = _sf_valid_mask(rows, dev)
                diffs = (
                    int((n.view(torch.int16) != b[0].view(torch.int16)).sum()),
                    int((r.view(torch.int16) != b[1].view(torch.int16)).sum()),
                    int((q.view(torch.uint8) != b[2].view(torch.uint8)).sum()),
                    int((sf[valid] != b[3][valid]).sum()),
                    int((b[3][~valid] != 0).sum()),
                )
                if any(diffs):
                    bad.append((rows, str(sdt), seed, diffs))
                dist.barrier()
    assert not bad, (
        f"(rows, scale dtype, seed, (normed, residual, MXFP8, scales, pad scales) differing): {bad}"
    )


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
    assert WORLD == 2, f"the tp2 target runs at 2 ranks, got {WORLD}"
    try:
        _run_matrix()
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
    session is killed (as ``_m3_allreduce_op_matrix.py`` does)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the 2 devices this run owns"
    world_size = len([d for d in visible.split(",") if d.strip()])
    assert world_size == 2, (
        f"CUDA_VISIBLE_DEVICES names {world_size} device(s); the matrix runs at 2"
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
