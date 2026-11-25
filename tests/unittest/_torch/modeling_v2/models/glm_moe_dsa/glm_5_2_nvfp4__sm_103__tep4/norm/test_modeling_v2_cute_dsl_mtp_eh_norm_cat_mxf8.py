# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_mtp_eh_norm_cat_mxf8 entry.

The operand must be byte for byte the activation quant of the bf16 path it replaces: the two
hidden-width RMSNorms (`cute_dsl_hidden_rmsnorm`) concatenated."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel needs an sm_103 GPU", allow_module_level=True)

from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.kernels.common import (  # noqa: E501
    self_attn_prefill_act_quant_mxf8 as quant,
)  # noqa: E402, E501
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.norm import (  # noqa: E402, E501
    cute_dsl_hidden_rmsnorm as hidden_norm,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.norm import (  # noqa: E402, E501
    cute_dsl_mtp_eh_norm_cat_mxf8 as entry,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

op = entry.cute_dsl_mtp_eh_norm_cat_mxf8
_EPS = 1e-5


@pytest.fixture(scope="module")
def compiled():
    return entry.compile(torch.cuda.current_device())


@pytest.fixture(scope="module")
def compiled_norm():
    return hidden_norm.compile()


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, compiled, compiled_norm) -> None:
    rows = cell.spec["rows"]
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    e = (2.0 * torch.randn(rows, entry.HIDDEN, generator=g, device="cuda")).to(torch.bfloat16)
    h = (0.5 * torch.randn(rows, entry.HIDDEN, generator=g, device="cuda")).to(torch.bfloat16)
    we = (1.0 + 0.2 * torch.randn(entry.HIDDEN, generator=g, device="cuda")).to(torch.bfloat16)
    wh = (1.0 + 0.2 * torch.randn(entry.HIDDEN, generator=g, device="cuda")).to(torch.bfloat16)
    q, sf = entry.operand(rows, e.device)
    with validating(op):
        got_q, got_sf = op(e, we, h, wh, _EPS, q, sf, compiled)
    torch.cuda.synchronize()
    assert got_q is q and got_sf is sf
    # The bf16 path it replaces: the two hidden-width norms, the concat, the activation quant.
    cat = torch.cat(
        [
            hidden_norm.cute_dsl_hidden_rmsnorm(e, we, _EPS, compiled_norm),
            hidden_norm.cute_dsl_hidden_rmsnorm(h, wh, _EPS, compiled_norm),
        ],
        -1,
    )
    want_q, want_sf = quant.reference(cat)
    assert torch.equal(q, want_q), f"{int((q != want_q).sum())} operand bytes differ"
    got = quant.dequantize(q, sf, rows, entry.K_TOTAL)
    assert torch.equal(got, quant.dequantize(want_q, want_sf, rows, entry.K_TOTAL)), "scales differ"
    # Against the fp32 norm: a bf16 value one ulp off moves at most its own e4m3 code.
    ref_q, _ = op.reference(e, we, h, wh, _EPS, q, sf, compiled)
    frac = float((q != ref_q).float().mean())
    assert frac < 1e-3, f"{frac:.2e} of the bytes differ from the fp32 reference's"


def test_is_valid_refuses_mismatched_row_counts(compiled) -> None:
    e = torch.zeros(80, entry.HIDDEN, dtype=torch.bfloat16, device="cuda")
    h = torch.zeros(79, entry.HIDDEN, dtype=torch.bfloat16, device="cuda")
    w = torch.ones(entry.HIDDEN, dtype=torch.bfloat16, device="cuda")
    q, sf = entry.operand(80, e.device)
    with pytest.raises(AssertionError, match="row blocks"):
        with validating(op):
            op(e, w, h, w, _EPS, q, sf, compiled)
