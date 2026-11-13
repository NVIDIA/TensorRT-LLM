# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification of OPROJ_PUSH_GEMM's attention boundary, 2 ranks (SM107).

``minimax_m31_ar_prepushed`` is the stock one-shot Lamport AllReduce(RESIDUAL_RMS_NORM) without its push loop, plus the
MXFP8 of the norm. Two producers feed it, and each result must equal the stock chain bit for bit (normed rows,
residual, MXFP8 codes and swizzled scales):

1. a test producer doing exactly the stock push: the rank's all-reduce input with every 32-bit word 0x80000000 (-0, the
   Lamport "empty" marker) replaced by +0, stored into this rank's section of every rank's data slot flag % 3; against
   the stock AllReduce(ONESHOT) + mxfp8_quantize on the same input. Random, -0-word, zero-row and signed-zero inputs at
   1..256 rows;
2. the o_proj GEMM with the Lamport-push epilogue (``kernels/lamport_push_gemm``) at its pinned tactics; against
   FlashInfer's GEMM at the same tactic, then the stock AllReduce(ONESHOT) + mxfp8_quantize.

The reference pins the one-shot Lamport kernel, the one the prepushed boundary replaces (with the AllReduce autotuner
on, AUTO may pick NCCL, which keeps signed zeros where the Lamport push stores -0 words as +0). Both chains run on the
same workspace one after the other, so this also checks that the two kernels keep the workspace's clear-size and flag
protocol consistent.

Its own 2-rank launcher, as ``_m3_allreduce_op_matrix.py`` is; the collected entry point is
``test_modeling_v2_m31_ar_prepushed.py``.

    CUDA_VISIBLE_DEVICES=0,1 python _ar_prepushed_op_matrix.py
"""

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist

assert torch.cuda.is_available(), "the one-shot Lamport all-reduce requires CUDA devices"

_WORKER_FLAG = "--rank-worker"
DEADLINE_S = 1200

H = 6144
K = 4096
EPS = 1e-5
NR = 2
PUSH_ROWS = (1, 49, 104, 128, 176, 224, 256)
PUSH_KINDS = ("randn", "neg0_words", "zero_rows", "signed_zeros")
GEMM_ROWS = (49, 64, 100, 128, 200, 256)
NEG_ZERO_WORD = -(2**31)

RANK = 0
WORLD = 1


class _DevBytes:
    """A uint8 view of raw device memory (a Lamport slot section of this rank or of its peer)."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


def _gen(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def _inputs(rows: int, kind: str, seed: int):
    x = torch.randn(rows, H, device="cuda", generator=_gen(seed)) * 2.0
    if RANK == 1:
        x = x + torch.randn(rows, H, device="cuda", generator=_gen(seed + 7))
    x = x.to(torch.bfloat16)
    res = (torch.randn(rows, H, device="cuda", generator=_gen(seed + 1)) * 4).to(torch.bfloat16)
    if kind == "neg0_words":
        # words whose high bf16 is -0 and low is +0: the push must store them as +0
        x.view(torch.int32)[:, ::5] = NEG_ZERO_WORD
    elif kind == "zero_rows":
        # both ranks contribute zero rows and the residual is +0 / -0: all-zero normed rows (MXFP8 scale 0)
        x[: rows // 3] = 0
        res[: rows // 4] = 0.0
        res[rows // 4 : rows // 3] = -0.0
    elif kind == "signed_zeros":
        x = torch.where(x > 0, torch.zeros_like(x), -torch.zeros_like(x))
        res[: rows // 2] = -0.0
    return x.contiguous(), res.contiguous()


def _neq(a: torch.Tensor, b: torch.Tensor) -> int:
    a = a.contiguous().view(torch.uint8).reshape(-1)
    b = b.contiguous().view(torch.uint8).reshape(-1)
    return -1 if a.numel() != b.numel() else int((a != b).sum().item())


def _stock(ar, x, residual, norm_w):
    from tensorrt_llm._torch.distributed import AllReduceFusionOp, AllReduceParams

    normed, updated = ar(
        x,
        all_reduce_params=AllReduceParams(
            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
            residual=residual,
            norm_weight=norm_w,
            eps=EPS,
            trigger_completion_at_end=False,
        ),
    )[:2]
    q, sf = torch.ops.trtllm.mxfp8_quantize(normed, True)
    return normed, updated, q, sf


def _prepushed(op, ws, shape_like, residual, norm_w):
    rows = residual.shape[0]
    normed, updated = torch.empty_like(residual), torch.empty_like(residual)
    q = torch.empty(rows, H, dtype=torch.uint8, device="cuda")
    sf = torch.empty((rows + 127) // 128 * 128 * (H // 32), dtype=torch.uint8, device="cuda")
    op(RANK, ws, shape_like, residual, updated, normed, norm_w, EPS, False, q, sf)
    return normed, updated, q, sf


def _compare(name: str, got, want, bad: Dict[str, dict]) -> None:
    cell = {
        what: _neq(g, w) for what, g, w in zip(("norm", "residual", "mxfp8", "mx_sf"), got, want)
    }
    if any(cell.values()):
        bad[name] = cell


def _run_matrix() -> None:
    import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers trtllm::mxfp8_quantize
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2 import (  # noqa: E501
        native_ops,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2.kernels import (  # noqa: E501
        lamport_push_gemm as lpg,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.gemm import (
        mxfp8_cute_pinned_tactics as pin,
    )
    from tensorrt_llm._torch.distributed import AllReduce
    from tensorrt_llm._torch.distributed.ops import get_allreduce_workspace
    from tensorrt_llm.functional import AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    _single_node_cluster_info()
    mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=RANK, gpus_per_node=WORLD)
    ar = AllReduce(mapping=mapping, strategy=AllReduceStrategy.ONESHOT, dtype=torch.bfloat16)
    ws = get_allreduce_workspace(mapping)
    op = native_ops.minimax_m31_ar_prepushed
    read = lpg.read_device_i64
    wsp = [int(v) for v in ws.cpu().tolist()]
    comm_size = read(wsp[NR * 3 + 1] + 8)
    flag_addr = wsp[NR * 3] + 8
    norm_w = (torch.rand(H, device="cuda", generator=_gen(3)) + 0.5).to(torch.bfloat16)
    bad: Dict[str, dict] = {}

    # 1. the test producer: the stock push loop
    for rows in PUSH_ROWS:
        for kind in PUSH_KINDS:
            x, r0 = _inputs(rows, kind, 1000 * rows + len(kind))
            want = _stock(ar, x, r0, norm_w)
            torch.cuda.synchronize()
            dist.barrier()
            slot = (read(flag_addr) & 0xFFFFFFFF) % 3
            words = x.view(torch.int32).clone()
            words[words == NEG_ZERO_WORD] = 0
            src = words.view(torch.uint8).reshape(-1)
            for dest in range(NR):
                base = wsp[2 * NR + dest] + slot * comm_size + RANK * src.numel()
                # a view on the owning device (the peer's slot is on the peer GPU); as_tensor(device=) would copy it
                torch.as_tensor(_DevBytes(base, src.numel())).copy_(src)
            torch.cuda.synchronize()
            dist.barrier()
            got = _prepushed(op, ws, x, r0, norm_w)
            torch.cuda.synchronize()
            _compare(f"push r{rows} {kind}", got, want, bad)
            dist.barrier()

    # 2. the Lamport-push o_proj GEMM as the producer
    w_bf16 = (torch.randn(H, K, device="cuda", generator=_gen(11 + RANK)) * 0.05).to(torch.bfloat16)
    w, w_sf = torch.ops.trtllm.mxfp8_quantize(w_bf16, True)
    launcher = lpg.LamportPushGemmLauncher(ws, RANK, H, K, read)
    for rows in GEMM_ROWS:
        tactic = pin.TACTICS[(H, K, max(64, 1 << (rows - 1).bit_length()))]
        x = (torch.randn(rows, K, device="cuda", generator=_gen(rows + 100 * RANK)) * 4).to(
            torch.bfloat16
        )
        x[: rows // 8] = 0
        r0 = (torch.randn(rows, H, device="cuda", generator=_gen(rows + 1)) * 4).to(torch.bfloat16)
        act, act_sf = torch.ops.trtllm.mxfp8_quantize(x, True)
        with pin.engaged_for("_ar_prepushed_op_matrix"):
            y = pin.run(act, act_sf, w, w_sf, torch.bfloat16)
        assert y is not None, f"FlashInfer's runner did not take the pinned tactic {tactic}"
        want = _stock(ar, y, r0, norm_w)
        torch.cuda.synchronize()
        dist.barrier()
        shape_like = torch.empty((rows, H), dtype=torch.bfloat16, device="cuda")
        launcher.run(tactic, act, act_sf, w, w_sf, shape_like)
        got = _prepushed(op, ws, shape_like, r0, norm_w)
        torch.cuda.synchronize()
        _compare(f"gemm r{rows} {tactic}", got, want, bad)
        dist.barrier()

    assert not bad, f"rank {RANK}: the prepushed boundary differs from the stock chain: {bad}"
    if RANK == 0:
        print(
            f"[ar_prepushed] {len(PUSH_ROWS) * len(PUSH_KINDS)} pushed cells and {len(GEMM_ROWS)} GEMM cells "
            "bitwise equal to the stock chain",
            flush=True,
        )


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
    assert WORLD == NR, f"the boundary runs at tp2, got {WORLD} ranks"
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
    assert world_size == NR, (
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
