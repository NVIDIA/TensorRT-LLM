# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for ops/moe/fp4_moe_renorm_cute_dsl.

The experts are the pre-routed entry's test's (drawn NVFP4 codes and
power-of-two scales, so every weight is exact); this entry's layout is built
from their raw per-expert tensors by the target's own `weights.py` transforms,
so the test also pins those. Per cell: inside the routed entry's band against
the fp32 reference. Beyond the cells: the guard against a per-expert FC2 input
scale.
"""

import importlib.util
import zlib
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.renorm_moe_routing_op import (  # noqa: E501
    renorm_moe_routing_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1 import (  # noqa: E501
    weights as W,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.moe.fp4_moe_renorm_cute_dsl import (  # noqa: E501
    fp4_moe_renorm_cute_dsl as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "fp4_moe_renorm_cute_dsl requires a CUDA device"

DEV = "cuda"
G1 = 137.0  # activation global scale, as the pre-routed entry's test draws it


def _helpers():
    path = (
        Path(__file__).resolve().parents[4]
        / "moe"
        / "test_modeling_v2_fp4_block_scale_moe_runner.py"
    )
    spec = importlib.util.spec_from_file_location("_fp4_moe_runner_test_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _helpers()
_CD = None


def _cute_dsl_experts(ref):
    """This entry's operands from the raw expert tensors, through `weights.py`."""
    global _CD
    if _CD is None:
        up_p, gt_p, up_s, gt_s, dn_p, dn_s = ref["raw"]
        e = up_p.shape[0]
        fc1_w = torch.stack([W._expert_fc1_w_cd((up_p[i], gt_p[i]), None) for i in range(e)])
        fc1_s = torch.stack([W._expert_fc1_s_cd((up_s[i], gt_s[i]), None) for i in range(e)])
        fc2_w = dn_p.contiguous()
        fc2_s = torch.stack([W._expert_fc2_s_cd(dn_s[i], None) for i in range(e)])
        _CD = (fc1_w, fc1_s, fc2_w, fc2_s)
    return _CD


def _operands(spec, seed):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    args, ref = H._many_experts()
    data, sf, x = H._rand_activation(spec["tokens"], spec["hidden"], G1, gen)
    logits = torch.randn(spec["tokens"], spec["num_experts"], generator=gen, device=DEV).to(
        spec["logits_dtype"]
    )
    ids, _ = renorm_moe_routing_op(logits, spec["top_k"], torch.bfloat16)
    g2 = H._calibrate_g2(x, ids, ref)
    return args, ref, data, sf, x, logits, g2


def _cute_dsl(ref, data, sf, logits, g2, spec):
    fc1_w, fc1_s, fc2_w, fc2_s = _cute_dsl_experts(ref)
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
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    args, ref, data, sf, x, logits, g2 = _operands(spec, seed)
    y = _cute_dsl(ref, data, sf, logits, g2, spec)
    assert y.shape == (spec["tokens"], spec["hidden"]) and y.dtype == torch.bfloat16
    exp = op.reference(logits, x, spec["top_k"], ref["up"], ref["gate"], ref["down"], g2)
    H._compare(y, exp.to(torch.bfloat16))


def test_guard_refuses_a_per_expert_fc2_scale():
    spec = {**op.CELLS[0].spec, "tokens": 1024}
    args, ref, data, sf, x, logits, g2 = _operands(spec, 9)
    fc1_w, fc1_s, fc2_w, fc2_s = _cute_dsl_experts(ref)
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
