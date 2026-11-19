# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A sample of rubin_fc12's cells on SM107, each on the operands it describes, against rubin_fc12.reference.

The test builds what the entry takes: stored-form NVFP4 experts in the layout the engine's NVFP4
CuteDSL MoE method gives them on SM107, the input quantized as that backend quantizes it, and the
sort tables (the routing tail's for the presorted cells, moe_sort's for the other) at the 128-row
tile. One rank is
emulated on one GPU: its window of experts (EP8's 112 at the full intermediate, TP2xEP4's 224 at
half of it) is the whole expert set, routed by top-16 over that window.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.moe import rubin_fc12_launch as launch
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe._rubin_moe_common import (
    SF_VEC,
    TILE,
    noaux_topk,
    token_ordered_sort,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.rubin_fc12 import rubin_fc12

__extra_import_path__ = [".", "../../../..", "../.."]
import _rubin_nvfp4_moe_inputs as moe_inputs  # noqa: E402 — needs the path declared above
from _k31_utils import cpu_or_sm, on_sm, sample_cells
from _validating import assert_within_ulp, certifying  # noqa: E402,F401 -- path above

# Certified on SM107 only.
pytestmark = cpu_or_sm(107)
gpu = on_sm(107)

HIDDEN = 3584  # the latent the routed experts see
TOP_K = 16
SITU = dict(situ_beta=4.0, situ_linear_beta=25.0)
# BF16 ulps of each row's largest output, and of the relative RMS: the catalog
# fp4_block_scale_moe_runner's band for the same NVFP4 two-GEMM chain (it measured 9.97 / 0.88).
ELEMENT_ULP, RMS_ULP = 16.0, 2.0
LEVERS = ("mma_n", "die_claim")


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


def _run(x4, sf, weights, layer, tables, levers, output):
    """One rubin_fc12 launch over the window; ``output`` receives the routed sum."""
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
        tile_size=TILE,
        ep_size=1,
        enable_alltoall=False,
        scaling_vector_size=SF_VEC,
        **levers,
        **SITU,
    )
    return output


CELLS = sample_cells(rubin_fc12.CELLS, size_key="rows")[:2]


@gpu
@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:40] for c in CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811 -- fixture resolved by name
    """Each cell tuned under autotune(), then run from the tuned tactic into an output it must zero
    itself. A pinned N is the only N any compiled kernel has. The presorted cells take the routing
    tail's tables; the other takes moe_sort's, both at the runner's 128-row tile."""
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
    tables = _tables(ids, weights, local, TILE, spec["presorted"])
    compiled = set(launch.RubinFc12Runner.kernel_cache)
    with autotune():
        _run(x4, sf, weights, layer, tables, levers, torch.empty_like(x))
    out = _run(x4, sf, weights, layer, tables, levers, torch.full_like(x, float("nan")))
    if mma_n is not None:
        built = {key[3][1] for key in set(launch.RubinFc12Runner.kernel_cache) - compiled}
        assert built <= {mma_n}, f"kernels of N {built} built for a cell pinned to N {mma_n}"
    assert_within_ulp(out, ref.to(torch.bfloat16), element_ulp=ELEMENT_ULP, rms_ulp=RMS_ULP)
