# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An o_proj that owns its MNNVL all-reduce, fed the MXFP8 form of ``bf16(x * gate)`` (the operand
the Kimi-K3 Helix tail stores): ``forward_prequantized`` pushes from the GEMM like ``forward_gated``
does (low-M and mid-M kernels), left pending for its consumer up to ``in_place_max_rows`` rows, and
is bitwise ``forward_gated(x, gate)``, eagerly and in a CUDA graph replayed with new inputs."""

import os
import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from utils.util import skip_pre_blackwell

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# needed since we reuse the mpi executor pool, first test running will leak a thread
pytestmark = pytest.mark.threadleak(enabled=False)

HIDDEN = 7168


def _input(m, k, seed, rank):
    gen = torch.Generator(device="cuda").manual_seed(seed * 1000 + rank)
    return (torch.randn(m, k, device="cuda", generator=gen) * 0.25).to(torch.bfloat16)


def _gate(m, k, seed, rank):
    gen = torch.Generator(device="cuda").manual_seed(seed * 1000 + rank + 7)
    return torch.randn(m, k, device="cuda", generator=gen).sigmoid().to(torch.bfloat16)


def _operand(x, gate):
    """MXFP8 ``(xq, xs)`` of ``bf16(x * gate)``, scales in the R128c4 layout."""
    from tensorrt_llm._torch.kernels.blackwell import fp8_block_scale_low_m_gemm as low_m

    m, k = x.shape
    xq, xs_rows = low_m.mxfp8_quantize_reference(x * gate)
    xs = torch.full((-(-m // 128) * 128 * (k // 32),), 0x7F, dtype=torch.uint8, device="cuda")
    r = torch.arange(m, device="cuda").unsqueeze(1)
    c = torch.arange(k // 32, device="cuda").unsqueeze(0)
    offsets = ((r // 128) * (k // 128) + c // 4) * 512 + (r % 32) * 16 + (r % 128) // 32 * 4 + c % 4
    xs[offsets] = xs_rows
    return xq, xs


def _materialize(out):
    from tensorrt_llm._torch.distributed.ops import MnnvlWorkspaceResult

    return out.materialize() if isinstance(out, MnnvlWorkspaceResult) else out


def _run_rank(world_size):
    os.environ["TLLM_TEST_MNNVL"] = "1"
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    from tensorrt_llm._torch.distributed.ops import MnnvlPendingResult
    from tensorrt_llm._torch.models import modeling_kimi_linear as kimi
    from tensorrt_llm.mapping import Mapping

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank)
    try:
        mapping = Mapping(world_size=world_size, tp_size=world_size, rank=rank)
        allreduce = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
        )
        bad = []

        def differ(tag, ref, out):
            n = int((ref.view(torch.int16) != out.view(torch.int16)).sum())
            if n:
                bad.append(f"{tag}: {n} values differ")

        for k in (768, 1536):
            gen = torch.Generator(device="cuda").manual_seed(21 + rank)
            weight = (torch.randn((HIDDEN, k), device="cuda", generator=gen) * 0.02).to(
                torch.bfloat16
            )
            w, w_scale = kimi._Fp8BlockScaleWeightReadLinear.quantize_weight(weight)
            for in_place_max_rows in (0, 128):
                o_proj = kimi._Fp8BlockScaleWeightReadLinear(
                    w, w_scale, HIDDEN, allreduce, in_place_max_rows
                )
                o_proj.post_load_weights()
                for m in (1, 8, 14, 32, 64, 65, 96, 128):
                    if not o_proj.takes_prequantized_input(m):
                        continue
                    x, gate = _input(m, k, 9 * m, rank), _gate(m, k, 13 * m, rank)
                    ref = o_proj.forward_gated(x, gate)
                    ref_pending = isinstance(ref, MnnvlPendingResult)
                    ref = _materialize(ref).clone()
                    out = o_proj.forward_prequantized(*_operand(x, gate))
                    if isinstance(out, MnnvlPendingResult) != ref_pending:
                        bad.append(f"k={k} m={m}: pending result differs from forward_gated")
                    differ(f"k={k} m={m} rows={in_place_max_rows}", ref, _materialize(out))

        # One CUDA graph of pushed calls of both forms, replayed with new inputs.
        k, m = 1536, 8
        gen = torch.Generator(device="cuda").manual_seed(31 + rank)
        weight = (torch.randn((HIDDEN, k), device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
        o_proj = kimi._Fp8BlockScaleWeightReadLinear(
            *kimi._Fp8BlockScaleWeightReadLinear.quantize_weight(weight), HIDDEN, allreduce, 128
        )
        o_proj.post_load_weights()
        inputs = [(_input(m, k, 40 + i, rank), _gate(m, k, 50 + i, rank)) for i in range(2)]
        operands = [_operand(x, g) for x, g in inputs]

        def calls():
            res = []
            for (x, g), (xq, xs) in zip(inputs, operands):
                res.append(_materialize(o_proj.forward_gated(x, g)))
                res.append(_materialize(o_proj.forward_prequantized(xq, xs)))
            return res

        calls()
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            results = calls()
        for rep in range(3):
            for i, ((x, g), (xq, xs)) in enumerate(zip(inputs, operands)):
                x.copy_(_input(m, k, 60 + 10 * rep + i, rank))
                g.copy_(_gate(m, k, 70 + 10 * rep + i, rank))
                new_q, new_s = _operand(x, g)
                xq.copy_(new_q)
                xs.copy_(new_s)
            torch.cuda.synchronize()
            comm.Barrier()
            graph.replay()
            torch.cuda.synchronize()
            for i in range(2):
                differ(f"graph rep {rep} call {i}", results[2 * i], results[2 * i + 1])
        del graph
        failures = comm.allgather(bad)
        assert not any(failures), f"rank {rank}: {failures}"
    except Exception:
        traceback.print_exc()
        raise
    return True


@skip_pre_blackwell
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_o_proj_prequantized_push_matches_gated(mpi_pool_executor):
    world_size = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    results = mpi_pool_executor.map(_run_rank, *zip(*[(world_size,)] * world_size))
    for r in results:
        assert r is True
