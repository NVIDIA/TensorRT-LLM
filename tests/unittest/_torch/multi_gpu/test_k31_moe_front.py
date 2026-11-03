# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Kimi K3.1 sharded MoE front (k31_moe_front) against the replicated front it replaces.

The replaced path on every rank: ``dsv3_router_gemm_op`` FP32 router logits, the BF16 latent_down GEMM and
``kimi_k3_noaux_tc_mxfp8_quant`` (top-16 + MXFP8 of the latent). The front sums the same products in another order
(split over ranks and K slices), so the comparison allows last-bit differences of the MXFP8 bytes; the top-16 sets
must match on these inputs, and every rank must return the same bytes (they gather the same head). With the runner
sort, the TRTLLM-Gen MXFP4 runner's routing tables must equal ``k3_route_tail.run_k3_route_runner``'s for the same
top-16 (EP slices of 896 / W and 224 experts). With the shared expert's gate_up rows, the front must return the same
routing / MXFP8 bytes and a SiTU activation within BF16-level error of the separate GEMM + SituAndMul. A CUDA graph
of several calls (both workspace buffers) replayed on new inputs must return what eager calls return.
"""

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

pytestmark = pytest.mark.threadleak(enabled=False)

HIDDEN = 7168
LATENT = 3584
NUM_EXPERTS = 896


def _trt():
    """``torch.ops.trtllm``, looked up by name: an attribute named ``ops`` in a function cloudpickle
    ships by value would make it pickle the ``torch.ops`` object."""
    return getattr(getattr(torch, "ops"), "trtllm")


def _operands(seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    down = (torch.randn(LATENT, HIDDEN, device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    router = (torch.randn(NUM_EXPERTS, HIDDEN, device="cuda", generator=gen) * 0.02).to(
        torch.bfloat16
    )
    bias = (torch.rand(NUM_EXPERTS, device="cuda", generator=gen) - 0.5) * 0.19
    return down, router, bias


def _input(m, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(m, HIDDEN, device="cuda", generator=gen).to(torch.bfloat16)


def _reference(x, down, router, bias, scale):
    logits = _trt().dsv3_router_gemm_op(x, router.t(), bias=None, out_dtype=torch.float32)
    latent = torch.nn.functional.linear(x, down)
    return _trt().kimi_k3_noaux_tc_mxfp8_quant(logits, bias, latent, scale)


def _compare(got, ref):
    """(hard mismatches, MXFP8 byte mismatches, elements): the top-16 sets per token and the weights where the
    order agrees must match; MXFP8 bytes / scales may differ in the last bit of the latent."""
    ids, weights, codes, scales = got[:4]
    r_ids, r_weights, r_codes, r_scales = ref
    hard = 0
    for t in range(ids.shape[0]):
        hard += len(set(ids[t].tolist()) ^ set(r_ids[t].tolist()))
    same = ids == r_ids
    hard += int(((weights.float() - r_weights.float()).abs() > 1e-2)[same].sum())
    soft = int((codes.view(torch.uint8) != r_codes.view(torch.uint8)).sum())
    soft += int((scales != r_scales).sum())
    return hard, soft, codes.numel()


def _sort_differ(got, ref, tile):
    """Mismatches of the runner's routing tables (valid entries only)."""
    e2p, p2t, total, tile_expert, tile_limit, num_tiles = got
    r_e2p, r_p2t, r_total, r_expert, r_limit, r_tiles = ref
    bad = (
        int((e2p != r_e2p).sum())
        + int((total != r_total).sum())
        + int((num_tiles != r_tiles).sum())
    )
    n = int(r_tiles.item())
    if int(num_tiles.item()) != n:
        return bad + 1
    bad += int((tile_expert[:n] != r_expert[:n]).sum()) + int((tile_limit[:n] != r_limit[:n]).sum())
    for t in range(n):
        lo, hi = t * tile, int(r_limit[t])
        bad += int((p2t[lo:hi] != r_p2t[lo:hi]).sum())
    return bad


def _run_rank(world_size):
    os.environ["TLLM_TEST_MNNVL"] = "1"
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
    from tensorrt_llm._torch.moe.kernels.blackwell import k3_route_tail
    from tensorrt_llm._torch.moe.kernels.blackwell import k31_moe_front as front
    from tensorrt_llm.mapping import Mapping

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    gpus_per_node = torch.cuda.device_count()
    torch.cuda.set_device(rank % gpus_per_node)
    try:
        mapping = Mapping(
            world_size=world_size, tp_size=world_size, rank=rank, gpus_per_node=gpus_per_node
        )
        ws = front.head_workspace(mapping)
        down, router, bias = _operands(7)
        weight = front.front_weight(front.head_weight(down, router, rank, world_size))
        scale = 2.5
        hard = soft = elems = 0
        tile = 8
        ep = NUM_EXPERTS // world_size
        layouts = [(rank * ep, ep), ((rank % 4) * 224, 224)]
        for m in list(range(1, front.M_MAX + 1)) + [12, 16, 24, 32]:
            x = _input(m, 100 + m)
            got = front.run_k31_moe_front(x, weight, bias, scale, ws)
            torch.cuda.synchronize()
            h, s, n = _compare(got, _reference(x, down, router, bias, scale))
            hard, soft, elems = hard + h, soft + s, elems + n
            blob = (
                torch.cat([t.reshape(-1).view(torch.uint8) for t in got[:4]])
                .cpu()
                .numpy()
                .tobytes()
            )
            hard += sum(b != blob for b in comm.allgather(blob))
            logits = _trt().dsv3_router_gemm_op(x, router.t(), bias=None, out_dtype=torch.float32)
            for offset, num_local in layouts:
                sorted_ = front.run_k31_moe_front(
                    x, weight, bias, scale, ws, sort=(tile, offset, num_local, NUM_EXPERTS)
                )
                ref = k3_route_tail.run_k3_route_runner(
                    logits, m, bias, scale, offset, num_local, NUM_EXPERTS, tile, True
                )
                torch.cuda.synchronize()
                for a, b in zip(sorted_[:4], got[:4]):
                    hard += int((a.view(torch.uint8) != b.view(torch.uint8)).sum())
                hard += int((ref[0] != sorted_[0]).sum())
                hard += _sort_differ(sorted_[5], ref[2], tile)

        # The shared expert's gate_up + SiTU in the front (this rank's TP shard, up to 16 rows) vs the separate GEMM
        # + SituAndMul: BF16-level differences only (other summation order, approximate tanh / sigmoid).
        from tensorrt_llm._torch.modules.situ import SituAndMul

        inter = 384
        gen = torch.Generator(device="cuda").manual_seed(11 + rank)
        gate_up = (torch.randn(2 * inter, HIDDEN, device="cuda", generator=gen) * 0.02).to(
            torch.bfloat16
        )
        weight_s = front.front_weight(front.head_weight(down, router, rank, world_size), gate_up)
        act = SituAndMul(beta=4.0, linear_beta=25.0)
        for m in (1, 8, 12, 16):
            x = _input(m, 700 + m)
            got = front.run_k31_moe_front(
                x, weight_s, bias, scale, ws, shared_cols=inter, gate_cap=4.0, linear_cap=25.0
            )
            plain = front.run_k31_moe_front(x, weight, bias, scale, ws)
            torch.cuda.synchronize()
            for a, b in zip(got[:4], plain[:4]):
                hard += int((a.view(torch.uint8) != b.view(torch.uint8)).sum())
            ref = act(torch.nn.functional.linear(x, gate_up)).float()
            err = (got[4].float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
            hard += int(err > 2e-2)
            # The shared columns and the runner sort in one launch: same bytes, SiTU and routing tables.
            offset, num_local = layouts[1]
            both = front.run_k31_moe_front(
                x, weight_s, bias, scale, ws, shared_cols=inter, gate_cap=4.0, linear_cap=25.0,
                sort=(tile, offset, num_local, NUM_EXPERTS),
            )  # fmt: skip
            logits = _trt().dsv3_router_gemm_op(x, router.t(), bias=None, out_dtype=torch.float32)
            ref_sort = k3_route_tail.run_k3_route_runner(
                logits, m, bias, scale, offset, num_local, NUM_EXPERTS, tile, True
            )
            torch.cuda.synchronize()
            for a, b in zip(both[:5], got[:5]):
                hard += int((a.view(torch.uint8) != b.view(torch.uint8)).sum())
            hard += _sort_differ(both[5], ref_sort[2], tile)

        # One CUDA graph with several front calls (both workspace buffers; 24 rows = two token halves in one launch),
        # replayed on new inputs.
        m = 24
        xs = [_input(m, 500 + i) for i in range(3)]
        for x in xs:
            front.run_k31_moe_front(x, weight, bias, scale, ws)
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            results = [front.run_k31_moe_front(x, weight, bias, scale, ws) for x in xs]
        for rep in range(3):
            for i, x in enumerate(xs):
                x.copy_(_input(m, 900 + 10 * rep + i))
            torch.cuda.synchronize()
            comm.Barrier()
            graph.replay()
            torch.cuda.synchronize()
            for x, got in zip(xs, results):
                eager = front.run_k31_moe_front(x, weight, bias, scale, ws)
                torch.cuda.synchronize()
                for a, b in zip(got[:4], eager[:4]):
                    hard += int((a.view(torch.uint8) != b.view(torch.uint8)).sum())
                h, s, n = _compare(got, _reference(x, down, router, bias, scale))
                hard, soft, elems = hard + h, soft + s, elems + n
        del graph
        torch.cuda.synchronize()
        if rank == 0:
            print(
                f"[k31_moe_front] MXFP8 bytes / scales differing from the replicated front: {soft} of {elems}"
            )
        assert comm.allreduce(hard) == 0, f"rank {rank}: {hard} hard mismatches"
        assert soft <= elems // 100, f"rank {rank}: {soft} of {elems} MXFP8 bytes differ"
    except Exception:
        traceback.print_exc()
        raise
    return True


@skip_pre_blackwell
@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
def test_k31_moe_front(mpi_pool_executor):
    world_size = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    results = mpi_pool_executor.map(_run_rank, *zip(*[(world_size,)] * world_size))
    for r in results:
        assert r is True
