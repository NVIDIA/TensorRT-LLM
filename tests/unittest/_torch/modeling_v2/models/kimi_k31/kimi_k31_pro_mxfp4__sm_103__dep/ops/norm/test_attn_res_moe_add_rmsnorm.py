# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 attention-DP target's attn_res_moe_add_rmsnorm entry: the pre-attention mix that
first adds the routed and shared MoE outputs to the residual."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__dep.ops.norm.attn_res_moe_add_rmsnorm import (  # noqa: E501
    k31_attn_res_moe_add_rmsnorm as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.moe_residual_add import (  # noqa: E501
    k31_moe_residual_add,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm.attn_res_rmsnorm import (  # noqa: E501
    k31_attn_res_rmsnorm,
)
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

__extra_import_path__ = ["../../../../..", "../../../ops/norm"]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _k31_attn_res_inputs import OUT_EPS, RMS_EPS, assert_mix_close, mix_inputs  # noqa: E402
from _validating import validating  # noqa: E402 — needs the path declared above

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("the attn_res entries run a CuTe DSL kernel", allow_module_level=True)
assert torch.cuda.is_available(), "the attn_res entries require a CUDA device"

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    prefix, routed, block, weights = mix_inputs(
        cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**29
    )
    shared = (routed.float() * -0.5 + 0.25).to(torch.bfloat16)
    args = (prefix, routed, shared, block, *weights, RMS_EPS, OUT_EPS)
    with validating(op):
        updated, out = op(*args)
    ref_updated, ref_out = op.reference(*args)
    assert torch.equal(updated.view(torch.int16), ref_updated.view(torch.int16))
    assert_mix_close(out, ref_out, weights[2])


@pytest.mark.parametrize("rows", [8, 64, 160, 191])
@pytest.mark.parametrize("candidates", [2, 6, 9])
def test_is_bitwise_to_add_then_mix(rows, candidates) -> None:
    """The fold the attention-DP target runs is the bits of k31_moe_residual_add followed by attn_res_rmsnorm."""
    prefix, routed, block, weights = mix_inputs(
        dict(rows=rows, candidates=candidates), seed=rows * 16 + candidates
    )
    shared = (routed.float() * -0.5 + 0.25).to(torch.bfloat16)
    updated, out = op(prefix, routed, shared, block, *weights, RMS_EPS, OUT_EPS)
    ref_updated = k31_moe_residual_add(prefix, routed, shared)
    ref_out = k31_attn_res_rmsnorm(ref_updated, block, *weights, RMS_EPS, OUT_EPS)
    assert torch.equal(updated.view(torch.int16), ref_updated.view(torch.int16))
    assert torch.equal(out.view(torch.int16), ref_out.view(torch.int16))
