# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""rubin_cutedsl_fc1 and rubin_cutedsl_fc2_finalize, the CuteDSL context path's NVFP4 experts.

On SM103 and SM107: a sample of the cells of the chain sort -> memset -> FC1 -> FC2 finalize over one rank's
window against FC1's reference finalized by FC2's (SM103 runs the engine's Blackwell kernels, whose
FC1 rows come in their own order and whose tiles are 128 and 256).
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.rubin_cutedsl_fc1 import (  # noqa: E501
    rubin_cutedsl_fc1,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.rubin_cutedsl_fc2_finalize import (  # noqa: E501
    rubin_cutedsl_fc2_finalize,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.rubin_moe_output_memset import (  # noqa: E501
    rubin_moe_output_memset,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.rubin_moe_sort import rubin_moe_sort

__extra_import_path__ = [".", "../../../..", "../.."]
import _rubin_nvfp4_moe_inputs as moe_inputs  # noqa: E402 — needs the path declared above
from _k31_utils import cpu_or_sm, on_sm, sample_cells
from _validating import assert_within_ulp, certifying  # noqa: E402,F401 -- path above

pytestmark = cpu_or_sm(103, 107)
gpu_10x = on_sm(103, 107)

#: As rubin_fc12's cells: BF16 ulps of each row's largest output and of the relative RMS, the
#: catalog fp4_block_scale_moe_runner's band for the same NVFP4 two-GEMM chain.
ELEMENT_ULP, RMS_ULP = 16.0, 2.0


def _ulp_check(out, ref):
    assert_within_ulp(out, ref.to(out.dtype), element_ulp=ELEMENT_ULP, rms_ulp=RMS_ULP)


def _run_cell(spec, seed):
    """The four entries in the targets' order over one rank's window of the cell; returns
    ``(output, own)``: ``own`` is FC1's reference finalized by FC2's, over the inputs of the two
    calls (the sort tables the kernels ran on and the memset rows FC2 adds into)."""
    rows, local, inter, tile, top_k, hidden = (
        spec["tokens"],
        spec["local"],
        spec["inter"],
        spec["tile_size"],
        spec["top_k"],
        spec["hidden"],
    )
    offset = spec.get("local_expert_offset", 0)
    total = max(896, offset + local)
    experts = moe_inputs.experts(local, hidden, inter, seed)
    layer = moe_inputs.layer(experts, hidden)
    ids, weights = moe_inputs.route(rows, total, top_k, seed, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(rows, hidden, generator=g, device="cuda") * 0.15).to(torch.bfloat16)
    x4, sf = torch.ops.trtllm.fp4_quantize(x, layer["g1"], 16, False, False)
    sf = sf.view(rows, hidden // 16)
    sort = rubin_moe_sort(ids, weights, total, top_k, offset, local, tile)
    t2e, t2l, e2p, p2e, _, n = sort
    out = torch.full((rows, hidden), float("nan"), dtype=torch.bfloat16, device="cuda")
    rubin_moe_output_memset(out, t2l, e2p, p2e, n, tile, top_k, 8, False)
    zeroed = out.clone()  # FC2 finalizes into ``out`` in place; its reference starts here too
    moe_inputs.fc1_fc2(x4, sf, layer, out, sort, weights, total, top_k, local, offset, tile)
    torch.cuda.synchronize()
    # The entries' references take the BF16 input and the experts as stored (their documented
    # exception); FC1's rows are what FC2's reference reads, as (h, h_sf) are what FC2 reads.
    act = rubin_cutedsl_fc1.reference(
        x, experts, t2e, t2l, p2e, n, top_k=top_k, tile_size=tile, **moe_inputs.SITU
    )
    own = rubin_cutedsl_fc2_finalize.reference(
        act, experts, t2e, t2l, p2e, n, weights, zeroed, top_k=top_k, tile_size=tile
    )
    return out, own


CELLS = sample_cells(rubin_cutedsl_fc1.CELLS, size_key="tokens")[:2]


@gpu_10x
@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:40] for c in CELLS])
def test_fc1_cells(cell, certifying):  # noqa: F811
    """FC1 is checked through FC2: its output only exists in the swizzled NVFP4 form
    rubin_cutedsl_fc2_finalize reads, so the chain is held to FC1's reference finalized by
    FC2's. The two entries certify the same specs, so each cell certifies both."""
    assert [c.spec for c in rubin_cutedsl_fc1.CELLS] == [
        c.spec for c in rubin_cutedsl_fc2_finalize.CELLS
    ], "FC1 and FC2 finalize no longer share their cells: certify FC2's own"
    moe_inputs.skip_tiles_the_gpu_lacks(cell.spec["tile_size"])
    certifying(rubin_cutedsl_fc1, rubin_cutedsl_fc2_finalize)
    _ulp_check(*_run_cell(cell.spec, zlib.crc32(cell.why.encode())))
