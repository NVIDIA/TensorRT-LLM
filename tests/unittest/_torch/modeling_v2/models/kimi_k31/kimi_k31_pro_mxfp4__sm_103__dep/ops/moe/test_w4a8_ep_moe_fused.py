# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 expert-parallel routed experts with the combine fused into the FC2 epilogue, one GB300
tray (4 ranks, 224 experts each).

Each rank builds random MXFP4 experts (the stored layout the kernels read) and K3.1 rows (an MXFP8 latent, its scales,
16 expert ids of 896, BF16 routing weights). Every certified cell runs through the op (its guard interposed) and its
torch reference, within the FP32 band of the reference. The fused op must also equal, bit for bit, the path it
replaces on every rank: the one-sided all-to-all dispatch (``ops/moe/ep_dispatch_combine``, the engine's kernels, with
the scheduler's padding of a step with an empty rank), the CuTe W4A8 op (sort, CTA-pair FC1 + FC2, finalize) and the
one-sided combine. Several rounds per case reuse the workspace and its call counts.

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx environment.
"""

import pickle
import sys
import zlib

import cloudpickle
import pytest
import torch
from mpi4py import MPI

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__dep.ops.moe.w4a8_ep_moe_fused import (  # noqa: E501
    k31_w4a8_ep_moe_fused,
)
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

__extra_import_path__ = ["../../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("the fused MoE kernels are CuTe DSL kernels", allow_module_level=True)

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

WORLD = 4
NUM_EXPERTS, TOP_K, HIDDEN, INTER = 896, 16, 3584, 3072
pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in ((10, 0), (10, 3))
        or torch.cuda.device_count() < WORLD,
        reason="one GB300 tray: sm_100 / sm_103 and four GPUs",
    ),
]


op = k31_w4a8_ep_moe_fused

#: The attention-DP target's own step sizes that one tray can drive: one token per rank, and its largest per-rank
#: step (``MAX_ROWS`` over the 16 ranks of DEP16).
TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "kimi_k31/kimi_k31_pro_mxfp4__sm_103__dep": (
        Cell(
            why="one token per rank (low-latency decode)",
            spec=dict(world=WORLD, num_tokens=(1, 1, 1, 1)),
        ),
        Cell(
            why="the target's largest per-rank step: 256 tokens per rank (4096 rows over DEP16)",
            spec=dict(world=WORLD, num_tokens=(256, 256, 256, 256)),
        ),
    ),
}
CELLS = certified_cells(op, TARGET_CELLS)


def _band(out, ref):
    """FP32-accumulated reference vs the kernel: cosine and the max error relative to the row max."""
    o, r = out.double(), ref.double()
    cos = float((o * r).sum() / (o.norm() * r.norm()).clamp_min(1e-300))
    rel = float(((o - r).abs().amax(-1) / r.abs().amax(-1).clamp_min(1e-30)).max())
    return cos, rel


def _cell_case(spec: dict, seed: int) -> dict:
    """One rank of a certified cell: every rank builds every rank's experts from the cell's seed (each expert from its
    own seed, so a placement only moves them), calls the op with its own and checks it against the reference."""
    import tensorrt_llm
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__dep.ops.moe import (
        w4a8_ep_moe_fused as mm,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__dep.ops.moe.expert_placement import (  # noqa: E501
        expert_slots,
        layer_slot_experts,
    )
    from tensorrt_llm.llmapi.llm_args import MoeLoadBalancerConfig
    from tensorrt_llm.mapping import Mapping

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    torch.set_grad_enabled(False)
    world = spec["world"]
    mapping = Mapping(
        world_size=world,
        rank=rank,
        gpus_per_node=4,
        tp_size=world,
        moe_ep_size=world,
        moe_tp_size=1,
        enable_attention_dp=True,
    )
    num_tokens = list(spec["num_tokens"])
    tmax = max(num_tokens)
    per_rank = NUM_EXPERTS // world
    dev = torch.device("cuda")

    def expert(e):
        g = torch.Generator(device="cuda").manual_seed(seed * 100000 + e)

        def u8(*shape, lo=0, hi=256):
            return torch.randint(lo, hi, shape, generator=g, device=dev, dtype=torch.uint8)

        return (
            u8(2 * INTER, HIDDEN // 2),
            u8(2 * INTER * HIDDEN // 32, lo=118, hi=126),
            u8(HIDDEN, INTER // 2),
            u8(HIDDEN * INTER // 32, lo=118, hi=126),
        )

    def stack(experts):
        w = [expert(e) for e in experts]
        return [torch.stack([x[i] for x in w]) for i in range(4)]

    place = None
    slot_experts = list(range(NUM_EXPERTS))
    if spec.get("placed"):
        perm = torch.randperm(NUM_EXPERTS, generator=torch.Generator().manual_seed(seed)).tolist()
        config = MoeLoadBalancerConfig(num_slots=NUM_EXPERTS, initial_global_assignments={1: perm})
        slot_experts = layer_slot_experts(config, 1, NUM_EXPERTS)
        place = torch.tensor(expert_slots(slot_experts), dtype=torch.int32, device=dev)
    group = [stack(slot_experts[r * per_rank : (r + 1) * per_rank]) for r in range(world)]
    g = torch.Generator(device="cuda").manual_seed(seed * 1000 + rank)
    n = num_tokens[rank]
    ids = torch.rand((n, NUM_EXPERTS), generator=g, device=dev).topk(TOP_K, dim=1).indices
    ids = ids.to(torch.int32).contiguous()
    wts = torch.rand((n, TOP_K), generator=g, device=dev) + 0.5
    wts = (wts / wts.sum(1, keepdim=True)).to(torch.bfloat16)
    xq = torch.randint(0, 0x7F, (n, HIDDEN), generator=g, device=dev, dtype=torch.uint8)
    xq = (
        xq | (torch.randint(0, 2, (n, HIDDEN), generator=g, device=dev, dtype=torch.uint8) << 7)
    ).view(torch.float8_e4m3fn)
    xsf = torch.randint(120, 130, (n, HIDDEN // 32), generator=g, device=dev, dtype=torch.uint8)
    real = spec.get("real")
    real_rows = None
    if real is not None:
        real_rows = torch.full((4,), real[rank], dtype=torch.int32, device=dev)
    w1, w1s, w2, w2s = group[rank]
    args = dict(
        ws=mm.get_ep_moe_workspace(mapping, tmax),
        xq=xq,
        xsf=xsf,
        topk_ids=ids,
        topk_weights=wts,
        w1=w1,
        w1_sf=w1s,
        w2=w2,
        w2_sf=w2s,
        local_offset=rank * per_rank,
        local_num=per_rank,
        tmax=tmax,
        real_rows=real_rows,
        place=place,
    )
    with validating(mm.k31_w4a8_ep_moe_fused):
        out = mm.k31_w4a8_ep_moe_fused(**args)
    torch.cuda.synchronize()
    ref = mm.k31_w4a8_ep_moe_fused.reference(
        **{
            **args,
            **{k: [w[i] for w in group] for i, k in enumerate(("w1", "w1_sf", "w2", "w2_sf"))},
        }
    )
    fails = []
    nr = n if real is None else min(n, real[rank])
    if tuple(out.shape) != (n, HIDDEN) or out.dtype is not torch.bfloat16:
        fails.append(("shape", tuple(out.shape), out.dtype))
    elif nr < n and bool(out[nr:].view(torch.int16).any()):
        fails.append(("padding_rows_not_zero",))
    elif nr:
        if not torch.isfinite(ref[:nr].float()).all():
            fails.append(("nonfinite_reference",))
        # MXFP8 requant of an FP32-accumulated intermediate: a flipped E4M3 rounding moves one output by about an ulp
        # of its block's scale.
        cos, rel = _band(out[:nr], ref[:nr])
        if not (cos > 0.99999 and rel < 0.02):
            fails.append(("band", cos, rel))
    return dict(rank=rank, fails=fails)


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_certified_cells(cell, mpi_pool_executor) -> None:
    seed = zlib.crc32(cell.why.encode()) % 2**31
    results = list(mpi_pool_executor.map(_cell_case, *zip(*[(cell.spec, seed)] * WORLD)))
    for r in results:
        assert not r["fails"], r


def _rank_case(num_tokens: list, seed: int, rounds: int, real=None) -> dict:
    import tensorrt_llm
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__dep.ops.moe import (
        w4a8_ep_moe_fused as mm,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.ep_dispatch_combine import (
        K31EpAllToAll,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.w4a8_moe_cute import (
        k31_w4a8_moe_cute,
    )
    from tensorrt_llm.mapping import Mapping

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    torch.set_grad_enabled(False)
    mapping = Mapping(
        world_size=WORLD,
        rank=rank,
        gpus_per_node=4,
        tp_size=WORLD,
        moe_ep_size=WORLD,
        moe_tp_size=1,
        enable_attention_dp=True,
    )
    tmax = max(num_tokens)
    per_rank = NUM_EXPERTS // WORLD
    off = rank * per_rank
    dev = torch.device("cuda")
    g = torch.Generator(device="cuda").manual_seed(seed * 1000 + rank)

    def u8(*shape, lo=0, hi=256):
        return torch.randint(lo, hi, shape, generator=g, device=dev, dtype=torch.uint8)

    w1, w2 = u8(per_rank, 2 * INTER, HIDDEN // 2), u8(per_rank, HIDDEN, INTER // 2)
    w1s = u8(per_rank, 2 * INTER * HIDDEN // 32, lo=118, hi=126)
    w2s = u8(per_rank, HIDDEN * INTER // 32, lo=118, hi=126)
    n = num_tokens[rank]
    ids = torch.rand((n, NUM_EXPERTS), generator=g, device=dev).topk(TOP_K, dim=1).indices
    ids = ids.to(torch.int32).contiguous()
    wts = torch.rand((n, TOP_K), generator=g, device=dev) + 0.5
    wts = (wts / wts.sum(1, keepdim=True)).to(torch.bfloat16)
    # MXFP8 rows: finite E4M3 bytes (no 0x7F / 0xFF NaN codes) and UE8M0 scales near 1
    xq = (u8(n, HIDDEN, hi=0x7F) | (u8(n, HIDDEN, hi=2) << 7)).view(torch.float8_e4m3fn)
    xsf = u8(n, HIDDEN // 32, lo=120, hi=130)
    exchange = K31EpAllToAll(mapping, NUM_EXPERTS, TOP_K, tmax, HIDDEN, torch.bfloat16)
    # The separate path pads every rank to the step's largest count when a rank has no tokens (the routed-expert
    # scheduler's rule): zero rows routed alike, outputs dropped.
    counts, pad = num_tokens, 0
    if tmax and 0 in num_tokens:
        counts, pad = [tmax] * WORLD, tmax - n
    pad_ids = ((torch.arange(TOP_K, device=dev, dtype=torch.int32) * 53) % NUM_EXPERTS).expand(
        pad, TOP_K
    )
    ids_p = torch.cat([ids, pad_ids]).contiguous()
    wts_p = torch.cat([wts, wts.new_full((pad, TOP_K), 1.0 / TOP_K)]).contiguous()
    xq_p = torch.cat([xq, xq.new_zeros(pad, HIDDEN)]).contiguous()
    xsf_p = torch.cat([xsf, xsf.new_zeros(pad, HIDDEN // 32)]).contiguous()
    ws = mm.get_ep_moe_workspace(mapping, tmax)
    dummy = torch.zeros(16, dtype=torch.uint8, device=dev)
    real_rows = None
    if real is not None:
        real_rows = torch.full((4,), real[rank], dtype=torch.int32, device=dev)
    fails = []
    for it in range(rounds):
        xr, xsr, ir, wr = exchange.dispatch(
            xq_p, xsf_p, ids_p, wts_p, counts, True if pad else None
        )
        y = k31_w4a8_moe_cute(
            xr, xsr, ir, wr, w1, w1s, w2, w2s, off, per_rank, 128, dummy, pair=True
        )
        ref = exchange.combine(y, tmax)[:n]
        out = mm.k31_w4a8_ep_moe_fused(
            ws, xq, xsf, ids, wts, w1, w1s, w2, w2s, off, per_rank, tmax, real_rows=real_rows
        )
        torch.cuda.synchronize()
        nr = n if real is None else min(n, real[rank])
        if out.shape != ref.shape or not torch.equal(
            out[:nr].view(torch.int16), ref[:nr].view(torch.int16)
        ):
            fails.append(("fused_vs_separate", it))
        if nr < n and bool(out[nr:].view(torch.int16).any()):
            fails.append(("padding_rows_not_zero", it))
        if it == 0 and n and not torch.isfinite(ref.float()).all():
            fails.append(("nonfinite_reference",))
    return dict(rank=rank, fails=fails)


# Per-rank token counts: K3.1 DEP16-like decode rows per rank (one and two token tiles per expert), and uneven
# ranks including one without tokens (padding rows and a zero-token reduce).
CASES = [
    [48, 48, 48, 48],
    [200, 200, 200, 200],
    [37, 0, 64, 5],
]


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize("case", range(len(CASES)))
def test_fused_matches_separate_path(case, mpi_pool_executor) -> None:
    num_tokens = CASES[case]
    results = list(mpi_pool_executor.map(_rank_case, *zip(*[(num_tokens, 31 + case, 3)] * WORLD)))
    for r in results:
        assert not r["fails"], r


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
def test_padding_rows_skipped(mpi_pool_executor) -> None:
    """A real-row word per rank (the attention-DP CUDA-graph padding is the tail of the step): the real rows are
    bitwise the separate path's (which still runs the padding rows), the padding rows are zero."""
    num_tokens, real = [52, 52, 52, 52], [33, 52, 0, 40]
    results = list(mpi_pool_executor.map(_rank_case, *zip(*[(num_tokens, 77, 2, real)] * WORLD)))
    for r in results:
        assert not r["fails"], r


def _placed_case(num_tokens: list, seed: int) -> dict:
    """The fused op with an expert placement from a static load balancer configuration (a random permutation of the
    experts over the slots) equals the identity placement bit for bit: every rank holds the weights of the experts its
    physical slots name, the routing is the same."""
    import tensorrt_llm
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__dep.ops.moe import (
        w4a8_ep_moe_fused as mm,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__dep.ops.moe.expert_placement import (  # noqa: E501
        expert_slots,
        layer_slot_experts,
    )
    from tensorrt_llm.llmapi.llm_args import MoeLoadBalancerConfig
    from tensorrt_llm.mapping import Mapping

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    torch.set_grad_enabled(False)
    mapping = Mapping(
        world_size=WORLD,
        rank=rank,
        gpus_per_node=4,
        tp_size=WORLD,
        moe_ep_size=WORLD,
        moe_tp_size=1,
        enable_attention_dp=True,
    )
    tmax = max(num_tokens)
    per_rank = NUM_EXPERTS // WORLD
    dev = torch.device("cuda")

    def expert(e):
        g = torch.Generator(device="cuda").manual_seed(seed * 100000 + e)

        def u8(*shape, lo=0, hi=256):
            return torch.randint(lo, hi, shape, generator=g, device=dev, dtype=torch.uint8)

        return (
            u8(2 * INTER, HIDDEN // 2),
            u8(2 * INTER * HIDDEN // 32, lo=118, hi=126),
            u8(HIDDEN, INTER // 2),
            u8(HIDDEN * INTER // 32, lo=118, hi=126),
        )

    def stack(experts):
        w = [expert(e) for e in experts]
        return [torch.stack([x[i] for x in w]) for i in range(4)]

    perm = torch.randperm(NUM_EXPERTS, generator=torch.Generator().manual_seed(seed)).tolist()
    config = MoeLoadBalancerConfig(num_slots=NUM_EXPERTS, initial_global_assignments={1: perm})
    slot_experts = layer_slot_experts(config, 1, NUM_EXPERTS)
    place = expert_slots(slot_experts)
    ident = stack(range(rank * per_rank, (rank + 1) * per_rank))
    placed = stack(slot_experts[rank * per_rank : (rank + 1) * per_rank])
    place_t = torch.tensor(place, dtype=torch.int32, device=dev)
    g = torch.Generator(device="cuda").manual_seed(seed * 1000 + rank)
    n = num_tokens[rank]
    ids = torch.rand((n, NUM_EXPERTS), generator=g, device=dev).topk(TOP_K, dim=1).indices
    ids = ids.to(torch.int32).contiguous()
    wts = torch.rand((n, TOP_K), generator=g, device=dev) + 0.5
    wts = (wts / wts.sum(1, keepdim=True)).to(torch.bfloat16)
    xq = torch.randint(0, 0x7F, (n, HIDDEN), generator=g, device=dev, dtype=torch.uint8)
    xq = (
        xq | (torch.randint(0, 2, (n, HIDDEN), generator=g, device=dev, dtype=torch.uint8) << 7)
    ).view(torch.float8_e4m3fn)
    xsf = torch.randint(120, 130, (n, HIDDEN // 32), generator=g, device=dev, dtype=torch.uint8)
    ws = mm.get_ep_moe_workspace(mapping, tmax)
    off = rank * per_rank
    fails = []
    for it in range(2):
        ref = mm.k31_w4a8_ep_moe_fused(ws, xq, xsf, ids, wts, *ident, off, per_rank, tmax)
        out = mm.k31_w4a8_ep_moe_fused(
            ws, xq, xsf, ids, wts, *placed, off, per_rank, tmax, place=place_t
        )
        torch.cuda.synchronize()
        if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
            fails.append(("placed_vs_identity", it))
    return dict(
        rank=rank,
        fails=fails,
        moved=sum(p // per_rank != e // per_rank for e, p in enumerate(place)),
    )


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
def test_expert_placement_is_bitwise(mpi_pool_executor) -> None:
    results = list(mpi_pool_executor.map(_placed_case, *zip(*[([64, 48, 64, 7], 5)] * WORLD)))
    for r in results:
        assert not r["fails"], r
        assert r["moved"] > 0, r
