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
"""Kimi-K3 sharded MoE decode front (``kimi_k3_moe_front``) against the path it replaces.

Per rank: the replicated router logits (``dsv3_router_gemm_op`` per 16-row tile), the routing
kernel on them (``kimi_k3_routing``: top-16 + ``moe_sort`` tables of the rank's routed experts)
and latent_down with its NVFP4 output (``run_low_m_gemm_nvfp4``), also in the publish mode.
Bitwise where the replaced path runs the same arithmetic: routing at every M (per 16-row tile of
the router), the NVFP4 latent against the low-M GEMM at the front's tactic at every M and against
the default NVFP4 route where that runs the low-M kernel. Other differences are printed.
"""

import os
import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from _torch.helpers import dsv3_router_logits
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
LATENT = 3584
NUM_EXPERTS = 896
TOP_K = 16
TILE = 128


def _operands(seed):
    """latent_down (FP8 + the low-M GEMM's packed scale), router weight and bias of all ranks."""
    from tensorrt_llm._torch.kernels.blackwell import fp8_block_scale_low_m_gemm as low_m
    from tensorrt_llm._torch.models.modeling_kimi_linear import _Fp8BlockScaleWeightReadLinear

    gen = torch.Generator(device="cuda").manual_seed(seed)
    weight = (torch.randn(LATENT, HIDDEN, device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    w, w_scale = _Fp8BlockScaleWeightReadLinear.quantize_weight(weight)
    if torch.cuda.get_device_capability() == (10, 7):
        w_scale = low_m.pack_block_scale_from_mxfp8_layout(w_scale, LATENT, HIDDEN)
    router = (torch.randn(NUM_EXPERTS, HIDDEN, device="cuda", generator=gen) * 0.02).to(
        torch.bfloat16
    )
    bias = (torch.rand(NUM_EXPERTS, device="cuda", generator=gen) - 0.5) * 0.19
    return w, w_scale, router, bias


def _input(m, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(m, HIDDEN, device="cuda", generator=gen).to(torch.bfloat16)


def _router_logits(x, router):
    """The replaced path's logits: ``dsv3_router_gemm_op`` per 16-row tile (its MMA kernel)."""
    return dsv3_router_logits(x, router, tile_rows=16)


def _route(logits, bias, offset, num_local):
    from tensorrt_llm._torch.moe.kernels.blackwell import kimi_k3_routing as rt

    m = logits.shape[0]
    ids = torch.empty((m, TOP_K), dtype=torch.int32, device="cuda")
    vals = torch.empty((m, TOP_K), dtype=torch.float32, device="cuda")
    sort = rt.sort_outputs(m, num_local, logits.device)
    rt.run_route_sort(logits, bias, 2.5, offset, num_local, ids, vals, sort)
    return ids, vals, sort


def _nvfp4(x, w, w_scale, gsf, tactic=None):
    from tensorrt_llm._torch.kernels.blackwell import fp8_block_scale_low_m_gemm as low_m

    m = x.shape[0]
    codes = torch.empty((m, LATENT // 2), dtype=torch.uint8, device="cuda")
    scales = torch.empty((m, LATENT // 16), dtype=torch.uint8, device="cuda")
    return low_m.run_low_m_gemm_nvfp4(x, w, w_scale, gsf, codes, scales, tactic=tactic)


def _differ(a, b):
    if a.dtype == torch.float32:
        a, b = a.view(torch.int32), b.view(torch.int32)
    return int((a.reshape(-1) != b.reshape(-1)).sum())


def _sort_differ(got, ref):
    """Mismatches of the ``moe_sort`` tables, valid entries only."""
    tile_expert, tile_limit, e2p, p2e, total, num_tiles = got
    r_expert, r_limit, r_e2p, r_p2e, r_total, r_tiles = ref
    bad = _differ(num_tiles, r_tiles) + _differ(total, r_total) + _differ(e2p, r_e2p)
    n = int(r_tiles.item())
    if int(num_tiles.item()) != n:
        return bad + 1
    bad += _differ(tile_expert[:n], r_expert[:n]) + _differ(tile_limit[:n], r_limit[:n])
    for t in range(n):
        lo, hi = t * TILE, int(r_limit[t])
        bad += _differ(p2e[lo:hi], r_p2e[lo:hi])
    return bad


def _check(x, front_out, router, bias, w, w_scale, gsf, offset, num_local, info):
    from tensorrt_llm._torch.kernels.blackwell import fp8_block_scale_low_m_gemm as low_m
    from tensorrt_llm._torch.moe.kernels.blackwell import kimi_k3_moe_front as front

    m = x.shape[0]
    ids, vals, sort, codes, scales = front_out
    r_ids, r_vals, r_sort = _route(_router_logits(x, router), bias, offset, num_local)
    bad = _differ(ids, r_ids) + _differ(vals, r_vals) + _sort_differ(sort, r_sort)
    q_front, s_front = _nvfp4(x, w, w_scale, gsf, tactic=front.latent_tactic(m))
    bad += _differ(codes, q_front) + _differ(scales, s_front)
    q_cur, s_cur = _nvfp4(x, w, w_scale, gsf)
    default = low_m.nvfp4_out_tactic(m, LATENT, HIDDEN)
    if isinstance(default, low_m.LowMTactic):
        bad += _differ(codes, q_cur) + _differ(scales, s_cur)
    else:
        info.append(
            f"m={m}: NVFP4 latent vs the default route ({default}): "
            f"{_differ(codes, q_cur)} code bytes, {_differ(scales, s_cur)} scales differ"
        )
    return bad


def _run_rank(world_size):
    os.environ["TLLM_TEST_MNNVL"] = "1"
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
    from tensorrt_llm._torch.moe.kernels.blackwell import kimi_k3_moe_front as front
    from tensorrt_llm.mapping import Mapping

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    gpus_per_node = torch.cuda.device_count()
    torch.cuda.set_device(rank % gpus_per_node)
    try:
        mapping = Mapping(
            world_size=world_size, tp_size=world_size, rank=rank, gpus_per_node=gpus_per_node
        )
        ws = front.moe_front_workspace(mapping)
        w, w_scale, router, bias = _operands(7)
        bad = 0
        info = []
        # (local_expert_offset, local_num_experts): EP = world, an EP16-sized slice (TP4 x EP4 at
        # 16 ranks: 224) and EP2 (448 local experts, the full sort capacity).
        ep = NUM_EXPERTS // world_size
        layouts = [(rank * ep, ep), ((rank % 4) * 224, 224), ((rank % 2) * 448, 448)]
        for m in (1, 7, 8, 13, 16, 17, 24, 32):
            x = _input(m, 100 + m)
            gsf = torch.full((1,), 0.37, device="cuda")
            for offset, num_local in layouts:
                plan = front.make_plan(
                    w, w_scale, router, bias, 2.5, world_size, rank, offset, num_local
                )
                out = front.run_moe_front(x, plan, ws, gsf)
                torch.cuda.synchronize()
                bad += _check(x, out, router, bias, w, w_scale, gsf, offset, num_local, info)
                # Publish mode: the same top-k and latent, one release add per token, no tables.
                word = torch.zeros(1, dtype=torch.int32, device="cuda")
                ids, vals, sort, codes, scales = front.run_moe_front(x, plan, ws, gsf, word)
                torch.cuda.synchronize()
                r_ids, r_vals, _ = _route(_router_logits(x, router), bias, offset, num_local)
                q_front, s_front = _nvfp4(x, w, w_scale, gsf, tactic=front.latent_tactic(m))
                bad += int(sort is not None) + abs(int(word.item()) - m)
                bad += _differ(ids, r_ids) + _differ(vals, r_vals)
                bad += _differ(codes, q_front) + _differ(scales, s_front)
            if m > 16:
                cublas = dsv3_router_logits(x, router)
                tiled = _router_logits(x, router)
                ids_c = _route(cublas, bias, 0, ep)[0]
                ids_t = _route(tiled, bias, 0, ep)[0]
                info.append(
                    f"m={m}: logits vs cuBLAS {_differ(tiled, cublas)} differ, top-k ids "
                    f"{_differ(ids_c, ids_t)} differ"
                )

        # One CUDA graph with an even number of front calls (both workspace halves), replayed on
        # new inputs with no barrier: the halves alternate across the eager calls before it, the
        # capture and its replays, which alone order the ranks.
        m = 8
        xs = [_input(m, 500 + i) for i in range(4)]
        gsf = torch.full((1,), 0.25, device="cuda")
        plan = front.make_plan(w, w_scale, router, bias, 2.5, world_size, rank, rank * ep, ep)
        for x in xs:
            front.run_moe_front(x, plan, ws, gsf)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            results = [front.run_moe_front(x, plan, ws, gsf) for x in xs]
        for rep in range(3):
            for i, x in enumerate(xs):
                x.copy_(_input(m, 900 + 10 * rep + i))
            graph.replay()
            torch.cuda.synchronize()
            for x, out in zip(xs, results):
                bad += _check(x, out, router, bias, w, w_scale, gsf, rank * ep, ep, [])
        del graph
        torch.cuda.synchronize()
        if rank == 0:
            for line in info:
                print(f"[kimi_k3_moe_front] {line}")
        assert comm.allreduce(bad) == 0, f"rank {rank}: {bad} values differ from the replaced path"
    except Exception:
        traceback.print_exc()
        raise
    return True


@skip_pre_blackwell
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_kimi_k3_moe_front(mpi_pool_executor):
    world_size = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    results = mpi_pool_executor.map(_run_rank, *zip(*[(world_size,)] * world_size))
    for r in results:
        assert r is True


def test_kimi_k3_moe_front_geometry():
    """Every latent tile has one owner, the slots hold the widest rank's tiles and the logits
    follow the slots inside a half."""
    from tensorrt_llm._torch.moe.kernels.blackwell import kimi_k3_moe_front as front

    for world in (2, 4, 8, 16):
        assert front.supports_world(world)
        tiles = [front.latent_tiles_of(r, world) for r in range(world)]
        assert tiles[0][0] == 0
        for (t0, n0), (t1, _) in zip(tiles, tiles[1:]):
            assert t0 + n0 == t1
        assert sum(n for _, n in tiles) == LATENT // TILE
        assert front.slot_cols(world) == max(n for _, n in tiles) * TILE
        assert front.logits_offset(world) >= front.M_MAX * world * front.slot_cols(world) * 2
        assert front.half_bytes(world) >= front.logits_offset(world) + front.M_MAX * NUM_EXPERTS * 4
    assert not front.supports_world(32)
    assert not front.supports_world(3)
