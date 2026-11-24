# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test (four ranks, one GB300 or VR200 tray) for ``ops/gemm/rubin_ull_bf16_gemm_push``.

Every rank builds the engine MNNVL AllReduce on the shared workspace, regenerates every rank's
``x`` and ``weight`` from the cell's seed, pushes its product into the call and finishes it with
``rubin_lamport_finish``. Gated bit for bit against the finish's reference over every rank's
payload (the kernel's store-only variant, ``run_gemm``, gives the payload) and against the C++
all-reduce of this rank's plain product run on the same workspace afterwards; the payload against
the entry's FP64 reference within one BF16 rounding of FP32 sums. ``act`` cells (``x`` a gate / up
output ``[M, 2K]``): the payload is also bit for bit the plain kernel's on FlashInfer's
``silu_and_mul`` of ``x``.

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx
environment.
"""

import os
import pickle
import sys
import traceback
import types
import zlib

import cloudpickle
import pytest
import torch
from mpi4py import MPI

WORLD = 4
pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in ((10, 3), (10, 7))
        or torch.cuda.device_count() < WORLD,
        reason="sm_103 / sm_107 kernels on four GPUs (one tray)",
    ),
]

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm.rubin_lamport_push import (  # noqa: E402,E501
    rubin_lamport_finish,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_ull_bf16_gemm_push import (  # noqa: E402,E501
    bf16_gemm_push,
)

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

MIN_COSINE, MAX_RELATIVE_L2 = 0.99999, 1e-3


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


def _setup():
    """This rank's (rank, mapping, MNNVL AllReduce), built once per worker process."""
    os.environ["TLLM_TEST_MNNVL"] = "1"
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
    import tensorrt_llm
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    reg = vars(sys.modules.setdefault("_kimi_k31_push", types.ModuleType("_kimi_k31_push")))
    if "allreduce" not in reg:
        mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=rank)
        reg["mapping"] = mapping
        reg["allreduce"] = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
        )
        assert reg["allreduce"].mnnvl_allreduce is not None
    return rank, reg["mapping"], reg["allreduce"]


def _operands(spec: dict, seed: int, r: int):
    g = torch.Generator(device="cuda").manual_seed(seed * 1009 + r)
    width = spec["k"] * (2 if spec.get("act") else 1)
    x = torch.randn(spec["rows"], width, generator=g, device="cuda").bfloat16()
    w = (torch.randn(spec["n"], spec["k"], generator=g, device="cuda") * 0.02).bfloat16()
    return x, w


def _mismatch(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((a.view(torch.int16) != b.view(torch.int16)).sum())


def _close(a: torch.Tensor, b: torch.Tensor) -> bool:
    x, y = a.float().flatten(), b.float().flatten()
    cosine = torch.nn.functional.cosine_similarity(x, y, dim=0).item()
    rel = ((x - y).norm() / (y.norm() + 1e-12)).item()
    return cosine > MIN_COSINE and rel < MAX_RELATIVE_L2


def _digest(t: torch.Tensor) -> int:
    bits = t.contiguous().view(torch.int16).to(torch.int64).flatten()
    weights = torch.arange(1, bits.numel() + 1, device=bits.device, dtype=torch.int64) % 65521
    return int((bits * weights).sum().item())


def _push_rank(spec: dict, seed: int) -> dict:
    try:
        from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import SplitKTactic

        from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.flashinfer_silu_and_mul import (  # noqa: E501
            flashinfer_silu_and_mul,
        )
        from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.comm.rubin_lamport_ar import (  # noqa: E501
            LamportWorkspace,
            required_bytes,
        )
        from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.spec.rubin_draft_gemm_push import (  # noqa: E501
            run_gemm,
        )
        from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_linear import (  # noqa: E501
            rubin_linear,
        )
        from tensorrt_llm._torch.distributed.ops import get_or_scale_allreduce_mnnvl_workspace

        rank, mapping, allreduce = _setup()
        rows, n = spec["rows"], spec["n"]
        need = max(
            required_bytes(rows, n, WORLD, True),
            required_bytes(rows, n, WORLD, False),
            allreduce.mnnvl_allreduce.get_required_workspace_size(rows, n, WORLD, torch.bfloat16),
        )
        workspace = get_or_scale_allreduce_mnnvl_workspace(
            mapping, torch.bfloat16, buffer_size_bytes=need
        )
        ws = LamportWorkspace.from_mnnvl(workspace, WORLD, rank)
        tactic = spec["tactic"]
        act = spec.get("act", False)
        # Every rank's payload: the kernel's store-only variant on that rank's operands.
        payloads = []
        for r in range(WORLD):
            x, w = _operands(spec, seed, r)
            payloads.append(
                run_gemm(
                    x,
                    w,
                    torch.empty(rows, n, dtype=torch.bfloat16, device="cuda"),
                    SplitKTactic(*tactic[1:]),
                    act=act,
                )
            )
        x, w = _operands(spec, seed, rank)
        # act: the plain kernel on FlashInfer's silu_and_mul of x (the launch the fold replaces).
        xa = flashinfer_silu_and_mul(x) if act else x
        act_plain = run_gemm(
            xa,
            w,
            torch.empty(rows, n, dtype=torch.bfloat16, device="cuda"),
            SplitKTactic(*tactic[1:]),
        )
        out = torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")
        args = (x, w, out, ws, spec["one_shot"], tactic, True, act)
        bf16_gemm_push.is_valid(*args)
        bf16_gemm_push(*args)
        got = torch.empty_like(out)
        finish = (ws, got, spec["one_shot"], True, False, None, False)
        rubin_lamport_finish.is_valid(*finish)
        rubin_lamport_finish(*finish)
        ref = rubin_lamport_finish.reference(payloads, spec["one_shot"], None)
        cpp_out = allreduce(payloads[rank])
        torch.cuda.synchronize()
        return dict(
            rank=rank,
            ref=_mismatch(got, ref),
            cpp=_mismatch(got, cpp_out),
            payload_close=_close(payloads[rank], bf16_gemm_push.reference(*args)),
            payload_cublas=_mismatch(payloads[rank], rubin_linear(xa, w, False)),
            payload_act=_mismatch(payloads[rank], act_plain),
            digest=_digest(got),
        )
    except Exception:
        traceback.print_exc()
        raise


def _run(executor, fn, *args):
    results = list(executor.map(fn, *zip(*[args] * WORLD)))
    return sorted(results, key=lambda r: r["rank"])


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize(
    "cell", bf16_gemm_push.CELLS, ids=[c.why[:48] for c in bf16_gemm_push.CELLS]
)
def test_bf16_gemm_push_cells(mpi_pool_executor, cell) -> None:
    """Every cell of bf16_gemm_push (``.CELLS``): the finish of the pushed products is bit for bit
    the reference and the C++ all-reduce of the plain products, the same bits on every rank; the
    payload is within one rounding of the FP64 product."""
    results = _run(mpi_pool_executor, _push_rank, cell.spec, _seed(cell))
    for r in results:
        assert r["ref"] == 0 and r["cpp"] == 0 and r["payload_close"] and r["payload_act"] == 0, r
    assert len({r["digest"] for r in results}) == 1
    print(f"payload vs cuBLAS mismatches per rank: {[r['payload_cublas'] for r in results]}")
