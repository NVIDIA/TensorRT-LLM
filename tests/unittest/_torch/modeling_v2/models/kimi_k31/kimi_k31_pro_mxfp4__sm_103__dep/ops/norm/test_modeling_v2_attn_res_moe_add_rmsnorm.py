# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 attention-DP target's attn_res_moe_add_rmsnorm entry: the pre-attention mix that
first adds the routed and shared MoE outputs to the residual, on a sample of its cells (``sample_cells``)."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__dep.ops.norm.attn_res_moe_add_rmsnorm import (  # noqa: E501
    k31_attn_res_moe_add_rmsnorm as op,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.moe_residual_add import (  # noqa: E501
    k31_moe_residual_add,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.norm.attn_res_rmsnorm import (  # noqa: E501
    k31_attn_res_rmsnorm,
)

__extra_import_path__ = ["../../../../..", "../../..", "../../../ops/norm"]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _k31_attn_res_inputs import OUT_EPS, RMS_EPS, mix_inputs  # noqa: E402
from _k31_utils import sample_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("the attn_res entries run a CuTe DSL kernel", allow_module_level=True)
assert torch.cuda.is_available(), "the attn_res entries require a CUDA device"

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)
SAMPLE = sample_cells(CELLS, size_key="rows")[:2]


@pytest.mark.parametrize("cell", SAMPLE, ids=[c.why[:44] for c in SAMPLE])
def test_is_bitwise_to_add_then_mix(cell) -> None:
    """The fold the attention-DP target runs is the bits of k31_moe_residual_add followed by attn_res_rmsnorm."""
    prefix, routed, block, weights = mix_inputs(
        cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**29
    )
    shared = (routed.float() * -0.5 + 0.25).to(torch.bfloat16)
    with validating(op):
        updated, out = op(prefix, routed, shared, block, *weights, RMS_EPS, OUT_EPS)
    ref_updated = k31_moe_residual_add(prefix, routed, shared)
    ref_out = k31_attn_res_rmsnorm(ref_updated, block, *weights, RMS_EPS, OUT_EPS)
    assert torch.equal(updated.view(torch.int16), ref_updated.view(torch.int16))
    assert torch.equal(out.view(torch.int16), ref_out.view(torch.int16))
