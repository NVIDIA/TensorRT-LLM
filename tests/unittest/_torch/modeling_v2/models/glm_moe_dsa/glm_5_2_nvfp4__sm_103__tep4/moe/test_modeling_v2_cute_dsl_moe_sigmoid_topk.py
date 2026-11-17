# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_moe_sigmoid_topk entry."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel needs an sm_103 GPU", allow_module_level=True)

from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe import (  # noqa: E402, E501
    cute_dsl_moe_sigmoid_topk as entry,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

op = entry.cute_dsl_moe_sigmoid_topk
EXPERTS = entry.EXPERTS


@pytest.fixture(scope="module")
def compiled():
    return entry.compile()


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(spec["tokens"], EXPERTS, generator=g, device="cuda") * 3
    bias = torch.randn(EXPERTS, generator=g, device="cuda") * 0.1
    if spec.get("ties"):
        # Two experts with equal selection scores on every token, above every other
        # expert's (the bias is the largest by far): the lower id has to come first.
        logits[:, 10] = logits[:, 200] = 9.0
        bias[10] = bias[200] = 1.0
    return logits, bias


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, compiled, certifying) -> None:  # noqa: F811 — fixture by name
    """The ids exactly, in order. The weights within 2e-6 relative: the kernel
    normalizes its fp32 sum of the 8 picked scores in fp32 (a few fp32 ulps), the
    reference in fp64. A wrong pick, a biased weight or a missed renormalization is
    off by orders of magnitude more."""
    certifying(op)
    spec = cell.spec
    logits, bias = _build(spec, zlib.crc32(cell.why.encode()) % 2**31)
    args = (logits, bias, spec["n_group"], spec["topk_group"], spec["topk"], spec["scale"])
    assert entry.takes(*args[:5])
    w, ids = op(*args, compiled)
    ref_w, ref_ids = op.reference(*args, compiled)
    assert ids.dtype is torch.int32 and w.dtype is torch.float32
    assert torch.equal(ids, ref_ids), "expert ids differ from the reference"
    rel = ((w.double() - ref_w.double()).abs() / ref_w.double().abs().clamp_min(1e-30)).max()
    assert float(rel) <= 2e-6, f"weights {float(rel):.2e} relative from the reference"
    if spec.get("ties"):
        assert bool((ids[:, :2] == torch.tensor([10, 200], device="cuda", dtype=torch.int32)).all())


def test_matches_the_stock_op_it_replaces(compiled) -> None:
    """Same ids as noaux_tc_op on the token counts the target sends here."""
    from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe import (  # noqa: E501
        noaux_tc_op as stock,
    )

    for tokens in (65, 2048):
        logits, bias = _build(dict(tokens=tokens), tokens)
        w, ids = op(logits, bias, 1, 1, 8, 2.5, compiled)
        sw, sids = stock.noaux_tc_op(logits, bias, 1, 1, 8, 2.5)
        assert torch.equal(ids, sids.to(torch.int32))
        assert torch.allclose(w, sw.float(), rtol=2e-6, atol=0)


def test_is_valid_refuses_a_grouped_routing(compiled) -> None:
    logits, bias = _build(dict(tokens=8), 1)
    with pytest.raises(AssertionError, match="one group"):
        with validating(op):
            op(logits, bias, 8, 4, 8, 2.5, compiled)
