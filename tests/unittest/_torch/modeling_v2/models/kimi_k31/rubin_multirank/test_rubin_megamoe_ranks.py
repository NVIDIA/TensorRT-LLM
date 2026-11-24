# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""rubin_megamoe's cells over the ranks of one tray (attention DP, EP across the ranks).

Each rank runs its rows of a cell through MegaMoE and compares them with the entry's reference
over every rank's experts and rows. Experts and inputs come from shared seeds, so every rank
builds the whole layer's reference itself. Each rank also checks the bucket every launch took and,
in the shrink cell, that the launch of fewer rows reads no row of the earlier one.
"""

import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

_SM107 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(not _SM107, reason="certified on SM107 only"),
]

HIDDEN, INTER, TOP_K = 3584, 3072, 16
# As the NVFP4 MoE entries' tests: 16 / 2 BF16 ulp (element / RMS).
ELEMENT_ULP, RMS_ULP = 16.0, 2.0


def _entry():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe import rubin_megamoe

    return rubin_megamoe


def _swizzle(s):
    m, c = s.shape
    return s.reshape(m // 128, 4, 32, c // 4, 4).permute(0, 3, 2, 1, 4).reshape(m, c)


def _gate_up16(gate, up):
    rows, cols = gate.shape
    return torch.stack((gate.reshape(-1, 16, cols), up.reshape(-1, 16, cols)), 1).reshape(
        2 * rows, cols
    )


def _experts(rank, local):
    """Rank ``rank``'s experts as the weight table stores them (input scales 1, so alphas are
    ``weight_scale_2`` and the FC2 input scale is 1)."""
    g = torch.Generator(device="cuda").manual_seed(29 + rank)

    def proj(n, k):
        codes = torch.randint(
            0, 256, (local, n, k // 2), generator=g, device="cuda", dtype=torch.uint8
        )
        steps = torch.randint(120, 132, (local, n, k // 16), generator=g, device="cuda").float()
        return codes, (steps / 126.0).to(torch.float8_e4m3fn).view(torch.uint8)

    w1, s1 = proj(INTER, HIDDEN)
    w3, s3 = proj(INTER, HIDDEN)
    w2, s2 = proj(HIDDEN, INTER)
    ws2 = torch.full((local,), 0.03, device="cuda")
    return (
        torch.stack([_gate_up16(w1[e], w3[e]) for e in range(local)]),
        torch.stack([_swizzle(_gate_up16(s1[e], s3[e])) for e in range(local)]),
        w2,
        torch.stack([_swizzle(s2[e]) for e in range(local)]),
        ws2.clone(),
        ws2.clone(),
        torch.ones(local, device="cuda"),
    )


def _rows(rank, n, experts_total, seed):
    """Rank ``rank``'s rows of a call: NVFP4 latent, linear scales, top-16 routing."""
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
        fp4_quantize,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe._rubin_moe_common import (  # noqa: E501
        noaux_topk,
    )

    g = torch.Generator(device="cuda").manual_seed(seed * 100 + rank)
    x = (torch.randn(n, HIDDEN, generator=g, device="cuda") * 0.15).to(torch.bfloat16)
    logits = torch.randn(n, experts_total, generator=g, device="cuda") * 2
    ids, weights = noaux_topk(logits, torch.zeros(experts_total, device="cuda"), 1.0)
    if n == 0:
        return (
            torch.empty(0, HIDDEN // 2, dtype=torch.uint8, device="cuda"),
            torch.empty(0, HIDDEN // 16, dtype=torch.uint8, device="cuda"),
            ids.to(torch.int32),
            weights.float(),
        )
    x4, sf = fp4_quantize(x, torch.ones(1, device="cuda"), 16, False, False)
    return x4, sf.view(n, HIDDEN // 16), ids.to(torch.int32).contiguous(), weights.float()


def _run_rank(spec, why):
    from tensorrt_llm.mapping import Mapping

    mod = _entry()
    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    try:
        local = spec["local"]
        mapping = Mapping(
            world_size=world,
            rank=rank,
            tp_size=world,
            enable_attention_dp=True,
            moe_ep_size=world,
            moe_tp_size=1,
            gpus_per_node=torch.cuda.device_count(),
        )
        ws = mod.rubin_megamoe_workspace(
            mod.megamoe_ep_group(mapping),
            rank,
            world,
            local,
            HIDDEN,
            INTER,
            1024,
            1024 * world,
            torch.device("cuda"),
            rungs=spec["rungs"],
        )
        every = [_experts(r, local) for r in range(world)]
        # Per call: every rank's rows (``counts``: one call with these; ``rows``: calls in turn).
        calls = [list(spec["counts"])] if "counts" in spec else [[n] * world for n in spec["rows"]]
        worst = []
        for c, ns in enumerate(calls):
            inputs = [_rows(r, ns[r], local * world, c + 1) for r in range(world)]
            chunk_max = max(ns)
            out = mod.rubin_megamoe(*inputs[rank], *every[rank], ws, chunk_max, ns[rank])
            torch.cuda.synchronize()
            ref = mod.rubin_megamoe.reference(
                *(list(t) for t in zip(*inputs)),
                *(list(t) for t in zip(*every)),
                ws,
                chunk_max,
                ns[rank],
            )
            if ns[rank]:
                o, r = out.float(), ref
                scale = r.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
                element = ((o - r).abs() / scale).max().item() / 2.0**-8
                rms = ((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()).item() / 2.0**-8
                worst.append((element, rms))
        bad = int(any(e > ELEMENT_ULP or r > RMS_ULP for e, r in worst))
        result = (bad, worst)
    except Exception:
        traceback.print_exc()
        raise
    return result


def _drive(mpi_pool_executor, cell):
    world = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    results = list(mpi_pool_executor.map(_run_rank, *zip(*[(cell.spec, cell.why)] * world)))
    assert all(r[0] == 0 for r in results), f"{cell.why}: (bad, [(element ulp, rms ulp)]) {results}"


def _cells(ranks):
    return [c for c in _entry().RubinMegamoe.CELLS if c.spec.get("ranks", 1) == ranks]


@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
def test_megamoe_cells_over_four_ranks(mpi_pool_executor):
    """RubinMegamoe.CELLS (4 ranks, one tray)."""
    for cell in _cells(4):
        _drive(mpi_pool_executor, cell)
