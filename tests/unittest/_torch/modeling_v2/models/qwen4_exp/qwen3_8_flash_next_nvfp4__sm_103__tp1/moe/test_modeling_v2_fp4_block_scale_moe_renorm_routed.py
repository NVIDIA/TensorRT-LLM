# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for ops/moe/fp4_block_scale_moe_renorm_routed.

Two claims per cell: the logits-routed call is **bitwise** the composition it
replaces (`renorm_moe_routing_op` -> pre-routed `fp4_block_scale_moe_runner`),
and it sits inside the pre-routed entry's band against this entry's own
fp32 reference. The NVFP4 operand construction (layout, activations, scalars,
the band) uses ordinary shared test helpers. Beyond the cells: the refusal
of an uncertified (num_experts, top_k) pair.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.moe.fp4_block_scale_moe_renorm_routed import (  # noqa: E501
    CERTIFIED_PAIRS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.moe.fp4_block_scale_moe_renorm_routed import (  # noqa: E501
    fp4_block_scale_moe_renorm_routed as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.moe.renorm_moe_routing_op import (  # noqa: E501
    renorm_moe_routing_op,
)

__extra_import_path__ = ["../../../.."]
import _fp4_moe_test_utils as H  # noqa: E402
from _fp4_moe_test_utils import many_expert_layer  # noqa: E402,F401 — fixture by name
from _precision import true_fp32_matmul  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "fp4_block_scale_moe_renorm_routed requires a CUDA device"

DEV = "cuda"
G1 = 137.0  # activation global scale, as the pre-routed entry's test draws it


def _routed(args, data, sf, logits, scalars, spec, top_k=None):
    return op(
        logits,
        data,
        sf.view(torch.float8_e4m3fn),
        args["gemm1_weights"],
        args["gemm1_weights_scale"],
        args["gemm2_weights"],
        args["gemm2_weights_scale"],
        *scalars,
        spec["num_experts"],
        spec["top_k"] if top_k is None else top_k,
        spec["intermediate"],
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying, many_expert_layer):  # noqa: F811
    certifying(op)
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    args, ref, data, sf, x, logits, ids, wts, g2, (s1, sg, s2) = H.renorm_operands(
        many_expert_layer(), spec, seed, renorm_moe_routing_op
    )

    y = _routed(args, data, sf, logits, (s1, sg, s2), spec)
    assert y.shape == (spec["tokens"], spec["hidden"]) and y.dtype == torch.bfloat16

    # 1. Bitwise the composition it replaces, on identical operands.
    y_pre = H._call(
        data,
        sf,
        args,
        spec["num_experts"],
        spec["top_k"],
        topk_ids=ids,
        topk_weights=wts,
        output1_scale_scalar=s1,
        output1_scale_gate_scalar=sg,
        output2_scale_scalar=s2,
    )[0]
    differing = (y.view(torch.int16) != y_pre.view(torch.int16)).any(dim=1).sum().item()
    assert differing == 0, f"{differing} of {spec['tokens']} rows differ from the composition"

    # 2. Its own reference, through the pre-routed entry's gate.
    with true_fp32_matmul():
        exp = op.reference(logits, x, spec["top_k"], ref["up"], ref["gate"], ref["down"], g2)
    H._compare(y, exp.to(torch.bfloat16))


def test_uncertified_pair_refused(many_expert_layer):  # noqa: F811
    spec = op.CELLS[0].spec
    assert (spec["num_experts"], 9) not in CERTIFIED_PAIRS
    args, _, data, sf, _, logits, _, _, _, scalars = H.renorm_operands(
        many_expert_layer(), spec, 13, renorm_moe_routing_op
    )
    with validating(op), pytest.raises(AssertionError, match="certified Renormalize"):
        _routed(args, data, sf, logits, scalars, spec, top_k=9)
