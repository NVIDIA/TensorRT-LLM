# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for ops/moe/fp4_moe_renorm_cute_dsl.

The experts use the shared test helpers' drawn NVFP4 codes and
power-of-two scales, so every weight is exact; this entry's layout is built
from their raw per-expert tensors by the target's own `weights.py` transforms,
so the test also pins those. Per cell: inside the routed entry's band against
the fp32 reference. Beyond the cells: the guard against a per-expert FC2 input
scale.
"""

import zlib
from collections.abc import Iterator

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1 import (  # noqa: E501
    weights as W,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.moe.fp4_moe_renorm_cute_dsl import (  # noqa: E501
    fp4_moe_renorm_cute_dsl as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.moe.renorm_moe_routing_op import (  # noqa: E501
    renorm_moe_routing_op,
)

__extra_import_path__ = ["../../../.."]
import _fp4_moe_test_utils as H  # noqa: E402
from _fp4_moe_test_utils import many_expert_layer  # noqa: E402,F401 — fixture by name
from _precision import true_fp32_matmul  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "fp4_moe_renorm_cute_dsl requires a CUDA device"

DEV = "cuda"
G1 = 137.0  # activation global scale, as the pre-routed entry's test draws it


@pytest.fixture(scope="module")
def cute_dsl_experts(many_expert_layer) -> Iterator[tuple[torch.Tensor, ...]]:  # noqa: F811
    """This entry's operands from the raw expert tensors, through `weights.py`."""
    _, ref = many_expert_layer()
    up_p, gt_p, up_s, gt_s, dn_p, dn_s = ref["raw"]
    e = up_p.shape[0]
    fc1_w = torch.stack([W._expert_fc1_w_cd((up_p[i], gt_p[i]), None) for i in range(e)])
    fc1_s = torch.stack([W._expert_fc1_s_cd((up_s[i], gt_s[i]), None) for i in range(e)])
    fc2_w = dn_p.contiguous()
    fc2_s = torch.stack([W._expert_fc2_s_cd(dn_s[i], None) for i in range(e)])
    yield fc1_w, fc1_s, fc2_w, fc2_s


def _cute_dsl(experts, data, sf, logits, g2, spec):
    fc1_w, fc1_s, fc2_w, fc2_s = experts
    s1, sg, s2 = H._scalars(spec["num_experts"], G1, g2)
    gsf = torch.tensor([g2], dtype=torch.float32, device=DEV)
    return op(
        logits,
        data,
        sf,
        fc1_w,
        fc1_s,
        sg,
        fc2_w,
        fc2_s,
        s2,
        gsf,
        spec["num_experts"],
        spec["top_k"],
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying, many_expert_layer, cute_dsl_experts):  # noqa: F811
    certifying(op)
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    args, ref, data, sf, x, logits, _, _, g2, _ = H.renorm_operands(
        many_expert_layer(), spec, seed, renorm_moe_routing_op
    )
    y = _cute_dsl(cute_dsl_experts, data, sf, logits, g2, spec)
    assert y.shape == (spec["tokens"], spec["hidden"]) and y.dtype == torch.bfloat16
    with true_fp32_matmul():
        exp = op.reference(logits, x, spec["top_k"], ref["up"], ref["gate"], ref["down"], g2)
    H._compare(y, exp.to(torch.bfloat16))


def test_guard_refuses_a_per_expert_fc2_scale(many_expert_layer, cute_dsl_experts):  # noqa: F811
    spec = {**op.CELLS[0].spec, "tokens": 1024}
    args, ref, data, sf, x, logits, _, _, g2, _ = H.renorm_operands(
        many_expert_layer(), spec, 9, renorm_moe_routing_op
    )
    fc1_w, fc1_s, fc2_w, fc2_s = cute_dsl_experts
    s1, sg, s2 = H._scalars(spec["num_experts"], G1, g2)
    with validating(op), pytest.raises(AssertionError):
        op(
            logits,
            data,
            sf,
            fc1_w,
            fc1_s,
            sg,
            fc2_w,
            fc2_s,
            s2,
            s2,
            spec["num_experts"],
            spec["top_k"],
        )
