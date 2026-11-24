# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""rubin_fc12's cells on SM107, each on the operands it describes, against rubin_fc12.reference;
and the family launcher bitwise against the engine runner at the targets' lever values.

The test builds what the entry takes: stored-form NVFP4 experts in the layout the engine's NVFP4
CuteDSL MoE method gives them on SM107, the input quantized as that backend quantizes it, and the
sort tables (the routing tail's for the presorted cells, moe_sort's for the other). One rank is
emulated on one GPU: its window of experts (EP8's 112 at the full intermediate, TP2xEP4's 224 at
half of it) is the whole expert set, routed by top-16 over that window.
"""

import zlib

import _rubin_nvfp4_moe_inputs as moe_inputs
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.moe import (
    rubin_fc12_launch as launch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe._rubin_moe_common import (
    SF_VEC,
    TILE,
    noaux_topk,
    token_ordered_sort,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.rubin_fc12 import (
    rubin_fc12,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying  # noqa: E402,F401 -- path above

_SM107 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
# Certified on SM107 only.
pytestmark = pytest.mark.skipif(
    torch.cuda.is_available() and not _SM107, reason="certified on SM107 only"
)
gpu = pytest.mark.skipif(not _SM107, reason="needs an SM107 GPU")

HIDDEN = 3584  # the latent the routed experts see
TOP_K = 16
SITU = dict(situ_beta=4.0, situ_linear_beta=25.0)
# BF16 ulps of each row's largest output, and of the relative RMS: the catalog
# fp4_block_scale_moe_runner's band for the same NVFP4 two-GEMM chain (it measured 9.97 / 0.88).
ELEMENT_ULP, RMS_ULP = 16.0, 2.0
LEVERS = ("mma_n", "die_sched", "die_min_tokens", "lookahead", "sm_reserve", "die_claim")


def _quantize_input(x: torch.Tensor, g1: torch.Tensor):
    """The backend's input quantization: NVFP4 codes and linear block scales ``[rows, H / 16]``."""
    x4, sf = torch.ops.trtllm.fp4_quantize(x, g1, SF_VEC, False, False)
    return x4.view(torch.float4_e2m1fn_x2), sf.view(x.shape[0], HIDDEN // SF_VEC).view(torch.uint8)


def _tables(ids, weights, local, tile, presorted):
    """``moe_sort``'s six tables for the window: the routing tail's (token order) for a presorted
    cell, else the engine sort's."""
    if presorted:
        return token_ordered_sort(ids, 0, local, tile)
    return torch.ops.trtllm.moe_sort(
        token_selected_experts=ids,
        token_final_scales=weights,
        num_experts=local,
        top_k=TOP_K,
        local_expert_offset=0,
        local_num_experts=local,
        tile_tokens_dim=tile,
    )


def _run(x4, sf, weights, layer, tables, tile, levers, output):
    """One rubin_fc12 launch over the window; ``output`` receives the routed sum."""
    from tensorrt_llm._torch.utils import ActivationType

    tile_expert, tile_limit, e2p, p2e, _, num_tiles = tables
    local = layer["fc1_alpha"].numel()
    rubin_fc12(
        input=x4,
        fc1_weight=layer["fc1_w"],
        input_scale=sf,
        fc1_weight_scale=layer["fc1_s"],
        fc1_alpha=layer["fc1_alpha"],
        tile_idx_to_group_idx=tile_expert,
        tile_idx_to_mn_limit=tile_limit,
        permuted_idx_to_expanded_idx=p2e,
        num_non_exiting_tiles=num_tiles,
        global_sf=layer["g2"],
        fc2_weight=layer["fc2_w"],
        fc2_weight_scale=layer["fc2_s"],
        fc2_alpha=layer["fc2_alpha"],
        output=output,
        token_final_scales=weights,
        expanded_idx_to_permuted_idx=e2p,
        num_experts=local,
        top_k=TOP_K,
        num_local_experts=local,
        local_expert_offset=0,
        tile_size=tile,
        swiglu_limit=float("inf"),
        ep_size=1,
        enable_alltoall=False,
        activation_type=int(ActivationType.SiTu),
        **levers,
        **SITU,
    )
    return output


@gpu
@pytest.mark.parametrize("cell", rubin_fc12.CELLS, ids=[c.why[:40] for c in rubin_fc12.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811 -- fixture resolved by name
    """Each cell tuned under autotune(), then run from the tuned tactic into an output it must zero
    itself. A pinned N is the only N any compiled kernel has. The presorted cells take the routing
    tail's tables at its tile; the other takes moe_sort's at both tiles the runner implements."""
    from tensorrt_llm._torch.autotuner import autotune

    certifying(rubin_fc12)
    spec = cell.spec
    rows, local, mma_n = spec["rows"], spec["local"], spec["mma_n"]
    levers = {k: spec[k] for k in LEVERS if k in spec}
    seed = zlib.crc32(cell.why.encode())
    experts = moe_inputs.experts(local, HIDDEN, spec["inter"], seed)
    layer = moe_inputs.layer(experts, HIDDEN)
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(rows, HIDDEN, generator=g, device="cuda") * 0.15).to(torch.bfloat16)
    logits = torch.randn(rows, local, generator=g, device="cuda") * 2
    ids, weights = noaux_topk(logits, torch.zeros(local, device="cuda"), 1.0)
    x4, sf = _quantize_input(x, layer["g1"])
    ref = rubin_fc12.reference(x, ids, weights, experts, local_expert_offset=0, **SITU)
    for tile in (TILE,) if spec["presorted"] else (128, 256):
        tables = _tables(ids, weights, local, tile, spec["presorted"])
        compiled = set(launch.RubinFc12Runner.kernel_cache)
        with autotune():
            _run(x4, sf, weights, layer, tables, tile, levers, torch.empty_like(x))
        out = _run(x4, sf, weights, layer, tables, tile, levers, torch.full_like(x, float("nan")))
        if mma_n is not None:
            built = {key[3][1] for key in set(launch.RubinFc12Runner.kernel_cache) - compiled}
            assert built <= {mma_n}, f"kernels of N {built} built for a cell pinned to N {mma_n}"
        assert_within_ulp(out, ref.to(torch.bfloat16), element_ulp=ELEMENT_ULP, rms_ulp=RMS_ULP)


@gpu
def test_fc1_rows_are_the_engine_layout():
    """The FC1 row order the engine's NVFP4 CuteDSL MoE method gives one expert on SM107."""
    from tensorrt_llm._torch.moe.fused_moe.quantization import NVFP4CuteDslFusedMoEMethod

    g = torch.Generator(device="cuda").manual_seed(0)
    gate, up = torch.randint(0, 256, (2, 96, 8), generator=g, device="cuda", dtype=torch.uint8)
    engine = NVFP4CuteDslFusedMoEMethod._interleave_fc1_rows(torch.cat([up, gate], dim=0))
    assert torch.equal(moe_inputs.fc1_rows(gate, up), engine)


#: The FC12 lever environment (helix8 / TEP8 generation; see the family README's lever-variable
#: paragraph) and the levers it sets.
_LEVER_ENV = {
    "TRTLLM_K3_FC12_SM_RESERVE": "16",
    "TRTLLM_K3_FC12_DIE_SCHED": "1",
    "TRTLLM_K3_FC12_DIE_SCHED_MIN_TOKENS": "24",
}
_ENV_LEVERS = dict(die_sched=True, die_min_tokens=24, lookahead=1, sm_reserve=16)


@gpu
@pytest.mark.parametrize(
    "rows, local, inter, mma_n",
    [(8, 112, 3072, None), (96, 224, 1536, 256), (64, 224, 1536, 256)],
    ids=["tep8_b1k7", "helix8_b16k5", "helix8_b8k7"],
)
def test_launch_is_bitwise_the_engine_runner_at_the_lever_env(
    monkeypatch, rows, local, inter, mma_n
):
    """Every tactic the launcher offers, run through the family runner (levers as arguments) and
    the engine runner (levers from the lever environment, the tactic given), on the same
    die-alternating weights and inputs: the same bits. Both runs zero their own counters and
    output; the engine run is repeated first, and the compare is bitwise only where the engine
    agrees with itself (the finalize adds with atomics), else within twice the engine's own
    run-to-run spread."""
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as engine
    from tensorrt_llm._torch.die_locality import die_alternating_copy
    from tensorrt_llm._torch.utils import ActivationType

    for name, value in _LEVER_ENV.items():
        monkeypatch.setenv(name, value)
    seed = zlib.crc32(f"{rows}-{local}-{inter}".encode())
    layer = moe_inputs.layer(moe_inputs.experts(local, HIDDEN, inter, seed), HIDDEN)
    for key in ("fc1_w", "fc2_w"):
        placed = die_alternating_copy(layer[key])
        if placed is not None:
            layer[key] = placed
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(rows, HIDDEN, generator=g, device="cuda") * 0.15).to(torch.bfloat16)
    logits = torch.randn(rows, local, generator=g, device="cuda") * 2
    ids, weights = noaux_topk(logits, torch.zeros(local, device="cuda"), 1.0)
    x4, sf = _quantize_input(x, layer["g1"])
    tables = token_ordered_sort(ids, 0, local, TILE)
    tile_expert, tile_limit, e2p, p2e, _, num_tiles = tables
    scalars = dict(
        num_experts=local,
        top_k=TOP_K,
        num_local_experts=local,
        local_expert_offset=0,
        tile_size=TILE,
        scaling_vector_size=SF_VEC,
        swiglu_limit=float("inf"),
        ep_size=1,
        enable_alltoall=False,
        activation_type=ActivationType.SiTu,
        situ_beta=SITU["situ_beta"],
        situ_linear_beta=SITU["situ_linear_beta"],
    )

    def inputs(out):
        return [
            x4,
            layer["fc1_w"],
            sf,
            layer["fc1_s"],
            layer["fc1_alpha"],
            tile_expert,
            tile_limit,
            p2e,
            num_tiles,
            layer["g2"],
            layer["fc2_w"],
            layer["fc2_s"],
            layer["fc2_alpha"],
            out,
            weights,
            e2p,
        ]

    ours = launch.RubinFc12Runner(**{k: v for k, v in scalars.items()}, mma_n=mma_n, **_ENV_LEVERS)
    theirs = engine.Sm107BlockScaledContiguousGroupedGemmFusedFc12Runner(**scalars)
    tactics = ours.get_valid_tactics(inputs(torch.empty_like(x)), None)
    # The engine runner offers both MMA N; the family runner offers the same set without an MMA N
    # and, with the target's MMA N, the part of it at that N.
    unpinned = launch.RubinFc12Runner(**scalars, mma_n=None, **_ENV_LEVERS)
    unpinned = unpinned.get_valid_tactics(inputs(torch.empty_like(x)), None)
    assert unpinned == theirs.get_valid_tactics(inputs(torch.empty_like(x)), None)
    assert tactics and all(t in unpinned for t in tactics)
    for tactic in tactics:
        a = theirs(inputs(torch.full_like(x, float("nan"))), tactic=tactic).clone()
        b = theirs(inputs(torch.full_like(x, float("nan"))), tactic=tactic).clone()
        c = ours(inputs(torch.full_like(x, float("nan"))), tactic=tactic).clone()
        torch.cuda.synchronize()
        if torch.equal(a, b):
            assert torch.equal(c, a), f"tactic {tactic}: not the engine runner's bits"
        else:
            spread = (a.float() - b.float()).abs().max()
            assert (c.float() - a.float()).abs().max() <= 2 * spread, f"tactic {tactic}"
