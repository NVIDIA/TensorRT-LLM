# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_shared_up_swiglu_mxf8 entry.

The fused launch must write exactly the bytes of the pair it replaces -- the plain
`shared_up_mxf8` GEMM on the non-interleaved weight into bf16 `[T, 1024]`, then the rows
SwiGLU -- and stay within bf16 rounding of the fp64 reference on the same MXF8 operands."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel needs an sm_103 GPU", allow_module_level=True)

from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.kernels.common import (  # noqa: E501
    moe_sigmoid_topk_swiglu_rows as rows_swiglu,
)  # noqa: E402, E501
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.kernels.common import (  # noqa: E501
    self_attn_prefill_act_quant_mxf8 as quant,
)  # noqa: E402, E501
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.kernels.gemm import (  # noqa: E501
    self_attn_prefill_projections_mxf8 as proj,
)  # noqa: E402, E501
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm import (  # noqa: E402, E501
    cute_dsl_shared_up_swiglu_mxf8 as entry,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — path declared above

op = entry.cute_dsl_shared_up_swiglu_mxf8


@pytest.fixture(scope="module")
def compiled():
    return entry.compile(torch.cuda.current_device())


@pytest.fixture(scope="module")
def plain():
    """The pair the fused launch replaces: the plain MXF8 GEMM and the rows SwiGLU."""
    return proj.build("shared_up_mxf8", torch.cuda.current_device()), rows_swiglu.compile_swiglu()


@pytest.fixture(scope="module")
def weights():
    g = torch.Generator(device="cuda").manual_seed(11)
    w = (0.02 * torch.randn(entry.GATE_UP, entry.HIDDEN, generator=g, device="cuda")).to(
        torch.bfloat16
    )
    return w, entry.prepare_weight(w), proj.quantize_weight(w)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, compiled, plain, weights) -> None:
    rows = cell.spec["rows"]
    _, w8_il, w8 = weights
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    x = torch.randn(rows, entry.HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    pre = None
    if cell.spec.get("pre"):
        pre = quant.reference(x)
    out = torch.full((rows, entry.OUT), float("nan"), dtype=torch.bfloat16, device="cuda")
    with validating(op):
        got = op(x, w8_il, out, compiled, pre=pre)
    torch.cuda.synchronize()
    assert got is out
    # Byte for byte the plain GEMM into bf16 [T, 1024] followed by the rows SwiGLU.
    gemm, sw = plain
    gu = torch.empty(rows, entry.GATE_UP, dtype=torch.bfloat16, device="cuda")
    gemm(x, w8, gu, pre=pre)
    want = rows_swiglu.swiglu(sw, gu)
    torch.cuda.synchronize()
    assert torch.equal(out, want), f"{int((out != want).sum())} outputs differ from the pair"
    # The fp64 SwiGLU on the same MXF8 operands. The GEMM's fp32 sums and the reference's
    # differ in order, so a gate or up value may round to the neighbouring bf16 before the
    # product: up to 4 ulp on such an element against the row scale, 1 ulp RMS.
    assert_within_ulp(
        out, op.reference(x, w8_il, out, compiled, pre=pre), element_ulp=4.0, rms_ulp=1.0
    )


def test_is_valid_refuses_the_plain_weight_shape(compiled) -> None:
    x = torch.zeros(600, entry.HIDDEN, dtype=torch.bfloat16, device="cuda")
    w = proj.quantize_weight(torch.zeros(512, entry.HIDDEN, dtype=torch.bfloat16, device="cuda"))
    out = torch.empty(600, entry.OUT, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="prepare_weight"):
        with validating(op):
            op(x, w, out, compiled)
