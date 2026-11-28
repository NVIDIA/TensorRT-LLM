# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 expert-parallel routed experts with the combine fused into the FC2 epilogue, one GB300
tray (4 ranks, 224 experts each).

Each rank builds random MXFP4 experts (the stored layout the kernels read) and K3.1 rows (an MXFP8 latent, its scales,
16 expert ids of 896, BF16 routing weights). A sample of the certified cells (``sample_cells``) runs through the op
(its guard interposed) and its torch reference, within the FP32 band of the reference.

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx environment.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__dep.ops.moe.w4a8_ep_moe_fused import (  # noqa: E501
    k31_w4a8_ep_moe_fused,
)

__extra_import_path__ = ["../../../../..", "../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _k31_utils import (  # noqa: E402 — needs the path declared above
    dep_mapping,
    on_ranks,
    pickle_by_value,
    run_per_rank,
    sample_cells,
)
from _validating import validating  # noqa: E402 — needs the path declared above

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("the fused MoE kernels are CuTe DSL kernels", allow_module_level=True)

pickle_by_value(__name__)

WORLD = 4
NUM_EXPERTS, TOP_K, HIDDEN, INTER = 896, 16, 3584, 3072
pytestmark = on_ranks(WORLD, 100, 103)


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
SAMPLE = sample_cells(CELLS)[:2]


def _band(out, ref):
    """FP32-accumulated reference vs the kernel: cosine and the max error relative to the row max."""
    o, r = out.double(), ref.double()
    cos = float((o * r).sum() / (o.norm() * r.norm()).clamp_min(1e-300))
    rel = float(((o - r).abs().amax(-1) / r.abs().amax(-1).clamp_min(1e-30)).max())
    return cos, rel


def _stack_experts(experts, seed: int) -> list:
    """The stacked MXFP4 weights ``[w1, w1_sf, w2, w2_sf]`` of ``experts`` in the stored layout the kernels read, each
    expert from its own seed, so a placement only moves them."""
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

    w = [expert(e) for e in experts]
    return [torch.stack([x[i] for x in w]) for i in range(4)]


def _rank_rows(n: int, seed: int, rank: int) -> tuple:
    """One rank's ``n`` K3.1 rows: 16 expert ids of 896, BF16 routing weights, an MXFP8 latent and its scales."""
    dev = torch.device("cuda")
    g = torch.Generator(device="cuda").manual_seed(seed * 1000 + rank)
    ids = torch.rand((n, NUM_EXPERTS), generator=g, device=dev).topk(TOP_K, dim=1).indices
    ids = ids.to(torch.int32).contiguous()
    wts = torch.rand((n, TOP_K), generator=g, device=dev) + 0.5
    wts = (wts / wts.sum(1, keepdim=True)).to(torch.bfloat16)
    xq = torch.randint(0, 0x7F, (n, HIDDEN), generator=g, device=dev, dtype=torch.uint8)
    xq = (
        xq | (torch.randint(0, 2, (n, HIDDEN), generator=g, device=dev, dtype=torch.uint8) << 7)
    ).view(torch.float8_e4m3fn)
    xsf = torch.randint(120, 130, (n, HIDDEN // 32), generator=g, device=dev, dtype=torch.uint8)
    return ids, wts, xq, xsf


def _cell_case(spec: dict, seed: int) -> dict:
    """One rank of a certified cell: every rank builds every rank's experts from the cell's seed (each expert from its
    own seed, so a placement only moves them), calls the op with its own and checks it against the reference."""
    import tensorrt_llm
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__dep.ops.moe import (
        w4a8_ep_moe_fused as mm,
    )
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__dep.ops.moe.expert_placement import (  # noqa: E501
        expert_slots,
        layer_slot_experts,
    )
    from tensorrt_llm.llmapi.llm_args import MoeLoadBalancerConfig

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    torch.set_grad_enabled(False)
    world = spec["world"]
    mapping = dep_mapping(world, rank, gpus_per_node=4)
    num_tokens = list(spec["num_tokens"])
    tmax = max(num_tokens)
    per_rank = NUM_EXPERTS // world
    dev = torch.device("cuda")

    place = None
    slot_experts = list(range(NUM_EXPERTS))
    if spec.get("placed"):
        perm = torch.randperm(NUM_EXPERTS, generator=torch.Generator().manual_seed(seed)).tolist()
        config = MoeLoadBalancerConfig(num_slots=NUM_EXPERTS, initial_global_assignments={1: perm})
        slot_experts = layer_slot_experts(config, 1, NUM_EXPERTS)
        place = torch.tensor(expert_slots(slot_experts), dtype=torch.int32, device=dev)
    group = [
        _stack_experts(slot_experts[r * per_rank : (r + 1) * per_rank], seed) for r in range(world)
    ]
    n = num_tokens[rank]
    ids, wts, xq, xsf = _rank_rows(n, seed, rank)
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


@pytest.mark.gpu4
@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize("cell", SAMPLE, ids=[c.why[:48] for c in SAMPLE])
def test_certified_cells(cell, mpi_pool_executor) -> None:
    seed = zlib.crc32(cell.why.encode()) % 2**31
    results = run_per_rank(mpi_pool_executor, _cell_case, cell.spec, seed, world=WORLD)
    for r in results:
        assert not r["fails"], r
