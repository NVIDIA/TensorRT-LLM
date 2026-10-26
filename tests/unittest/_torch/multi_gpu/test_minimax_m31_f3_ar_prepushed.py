# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""trtllm::minimax_m31_f3_ar_prepushed (MiniMax-M3.1 TP2 F3 boundary) is bit for bit the stock one-shot Lamport
AllReduce(RESIDUAL_RMS_NORM) (+ mxfp8_quantize of the norm) when the producer has pre-pushed its rows.

The serving producer is the o_proj GEMM's epilogue fork (kernels/m31_f3_gemm.py). Here a test producer does exactly
the stock push loop: the rank's allreduce input, with every 32-bit word 0x80000000 (-0, the Lamport "empty" marker)
replaced by +0, stored into this rank's section of every rank's data slot flag % 3. Two ranks; random, -0-word, zero-row
and signed-zero inputs; real-size and padded MXFP8 scale tiles.
"""

import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
import tensorrt_llm._torch.custom_ops  # noqa: F401
from tensorrt_llm._torch.distributed import (
    AllReduce,
    AllReduceFusionOp,
    AllReduceParams,
    AllReduceStrategy,
)
from tensorrt_llm._torch.distributed.ops import get_allreduce_workspace
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

# needed since we reuse the mpi executor pool, first test running will leak a thread
pytestmark = pytest.mark.threadleak(enabled=False)

H = 6144
EPS = 1e-5
NR = 2


class _DevBytes:
    """A uint8 view of raw device memory (the Lamport slots of this rank and its peer)."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


def _inputs(rows: int, kind: str, rank: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, H, device="cuda", generator=g) * 2.0
    if rank == 1:
        x = x + torch.randn(
            rows, H, device="cuda", generator=torch.Generator(device="cuda").manual_seed(seed + 7)
        )
    x = x.to(torch.bfloat16)
    res = (
        torch.randn(
            rows, H, device="cuda", generator=torch.Generator(device="cuda").manual_seed(seed + 1)
        )
        * 4
    ).to(torch.bfloat16)
    if kind == "neg0_words":
        # words whose high bf16 is -0 and low is +0: the push must store them as +0
        w = x.view(torch.int32)
        w[:, ::5] = torch.tensor(-(2**31), dtype=torch.int32, device="cuda")
    elif kind == "zero_rows":
        # both ranks contribute zero rows and the residual is +0 / -0: all-zero normed rows (MXFP8 scale 0)
        x[: rows // 3] = 0
        res[: rows // 4] = 0.0
        res[rows // 4 : rows // 3] = -0.0
    elif kind == "signed_zeros":
        x = torch.where(x > 0, torch.zeros_like(x), -torch.zeros_like(x))
        res[: rows // 2] = -0.0
    return x.contiguous(), res.contiguous()


def _run(rows_list, kinds):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    comm = MPI.COMM_WORLD
    mapping = Mapping(world_size=NR, rank=rank, gpus_per_node=NR, tp_size=NR)
    # The reference is pinned to the one-shot Lamport kernel, the kernel F3 replaces (F3 declines a
    # boundary that is not it). AUTO is not a stable reference here: with the AllReduce autotuner on
    # it can pick NCCL, which keeps signed zeros where the Lamport push stores -0 words as +0
    # (serving runs AR_DET: autotune off).
    ar = AllReduce(mapping=mapping, strategy=AllReduceStrategy.ONESHOT, dtype=torch.bfloat16)
    ws = get_allreduce_workspace(mapping)
    wsp = [int(v) for v in ws.cpu().tolist()]
    # getattr keeps "ops" out of this function's code names: cloudpickle would otherwise pickle the torch.ops module
    # object (a ModuleType subclass it cannot reduce) along with the function.
    trtllm_ops = getattr(torch, "ops").trtllm
    read = trtllm_ops.minimax_m31_f3_read_i64
    comm_size = read(wsp[NR * 3 + 1] + 8)
    flag_addr = wsp[NR * 3] + 8
    nw = (
        torch.rand(H, device="cuda", generator=torch.Generator(device="cuda").manual_seed(3)) + 0.5
    ).to(torch.bfloat16)
    bad = {}
    for rows in rows_list:
        for kind in kinds:
            x, r0 = _inputs(rows, kind, rank, 1000 * rows + len(kind))
            n_s, r_s = ar(
                x,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=r0,
                    norm_weight=nw,
                    eps=EPS,
                    trigger_completion_at_end=False,
                ),
            )[:2]
            q_s, sf_s = trtllm_ops.mxfp8_quantize(n_s, True)
            torch.cuda.synchronize()
            comm.Barrier()
            # the test producer: the stock push loop
            slot = (read(flag_addr) & 0xFFFFFFFF) % 3
            words = x.view(torch.int32).clone()
            words[words == -(2**31)] = 0
            src = words.view(torch.uint8).reshape(-1)
            for dest in range(NR):
                base = wsp[2 * NR + dest] + slot * comm_size + rank * src.numel()
                # a view on the owning device (the peer's slot is on the peer GPU); as_tensor(device=) would copy it
                torch.as_tensor(_DevBytes(base, src.numel())).copy_(src)
            torch.cuda.synchronize()
            comm.Barrier()
            n_f, r_f = torch.empty_like(r0), torch.empty_like(r0)
            q_f = torch.empty(rows, H, dtype=torch.uint8, device="cuda")
            sf_f = torch.empty(
                (rows + 127) // 128 * 128 * (H // 32), dtype=torch.uint8, device="cuda"
            )
            trtllm_ops.minimax_m31_f3_ar_prepushed(
                rank, ws, x, r0, r_f, n_f, nw, EPS, False, q_f, sf_f
            )
            torch.cuda.synchronize()

            def neq(a, b):
                a, b = (
                    a.contiguous().view(torch.uint8).reshape(-1),
                    b.contiguous().view(torch.uint8).reshape(-1),
                )
                return -1 if a.numel() != b.numel() else int((a != b).sum())

            cell = {
                "norm": neq(n_f, n_s),
                "residual": neq(r_f, r_s),
                "mxfp8": neq(q_f, q_s),
                "mx_sf": neq(sf_f, sf_s),
            }
            if any(cell.values()):
                bad[f"r{rows}_{kind}"] = cell
            comm.Barrier()
    return bad


def _run_rank(rows_list, kinds):
    try:
        return _run(rows_list, kinds)
    except Exception:
        traceback.print_exc()
        raise


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.skipif(
    not hasattr(torch.ops.trtllm, "minimax_m31_f3_ar_prepushed"),
    reason="the F3 prepushed all-reduce op is not built",
)
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_f3_ar_prepushed_is_the_stock_oneshot_allreduce(mpi_pool_executor):
    rows_list = [1, 49, 104, 128, 176, 224, 256]
    kinds = ["randn", "neg0_words", "zero_rows", "signed_zeros"]
    results = mpi_pool_executor.map(_run_rank, *zip(*[(rows_list, kinds)] * 2))
    for rank, bad in enumerate(results):
        assert not bad, f"rank {rank}: F3 boundary bytes differ from the stock chain: {bad}"
