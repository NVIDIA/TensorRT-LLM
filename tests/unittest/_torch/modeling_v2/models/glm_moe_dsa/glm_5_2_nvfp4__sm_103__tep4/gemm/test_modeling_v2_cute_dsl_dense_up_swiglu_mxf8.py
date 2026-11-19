# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_dense_up_swiglu_mxf8 entry.

The fused launch must write exactly the bytes of the pair it replaces -- the plain
`dense_gate_up_mxf8` GEMM on the non-interleaved weight into bf16 `[T, 6144]`, then
`cute_dsl_swiglu_mxf8` -- and stay within one e4m3 code on rare elements of the fp64
reference on the same MXF8 operands."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel needs an sm_103 GPU", allow_module_level=True)

from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.kernels.common import (  # noqa: E501
    self_attn_prefill_act_quant_mxf8 as quant,
)  # noqa: E402, E501
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.kernels.gemm import (  # noqa: E501
    self_attn_prefill_projections_mxf8 as proj,
)  # noqa: E402, E501
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.activation import (  # noqa: E402, E501
    cute_dsl_swiglu_mxf8 as swiglu8,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm import (  # noqa: E402, E501
    cute_dsl_dense_up_swiglu_mxf8 as entry,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

op = entry.cute_dsl_dense_up_swiglu_mxf8


@pytest.fixture(scope="module")
def compiled():
    return entry.compile(torch.cuda.current_device())


@pytest.fixture(scope="module")
def plain():
    """The pair the fused launch replaces: the plain MXF8 GEMM and the SwiGLU into MXF8."""
    dev = torch.cuda.current_device()
    return proj.build("dense_gate_up_mxf8", dev), swiglu8.compile(dev)


@pytest.fixture(scope="module")
def weights():
    g = torch.Generator(device="cuda").manual_seed(13)
    w = (0.02 * torch.randn(entry.GATE_UP, entry.HIDDEN, generator=g, device="cuda")).to(
        torch.bfloat16
    )
    return entry.prepare_weight(w), proj.quantize_weight(w)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, compiled, plain, weights) -> None:
    rows = cell.spec["rows"]
    w8_il, w8 = weights
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    x = torch.randn(rows, entry.HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    pre = quant.reference(x) if cell.spec.get("pre") else None
    q, sf = entry.operand(rows, x.device)
    q.fill_(0x7F)
    with validating(op):
        got_q, got_sf = op(x, w8_il, q, sf, compiled, pre=pre)
    torch.cuda.synchronize()
    assert got_q is q and got_sf is sf
    # Byte for byte the plain GEMM into bf16 [T, 6144] followed by the SwiGLU into MXF8.
    gemm, sw = plain
    gu = torch.empty(rows, entry.GATE_UP, dtype=torch.bfloat16, device="cuda")
    gemm(x, w8, gu, pre=pre)
    want_q, want_sf = swiglu8.operand(rows, entry.OUT, x.device)
    swiglu8.cute_dsl_swiglu_mxf8(gu, want_q, want_sf, sw)
    torch.cuda.synchronize()
    assert torch.equal(q, want_q), f"{int((q != want_q).sum())} operand bytes differ from the pair"
    n = entry.sf_bytes(rows, entry.OUT)
    assert torch.equal(sf[:n], want_sf[:n]), "scale bytes differ from the pair's"
    # The fp64 SwiGLU on the same MXF8 operands: the GEMM's fp32 sums and the reference's
    # differ in order and the kernel's sigmoid is approximate, so a rare element's bf16 value,
    # and with it its e4m3 code, may move by one.
    ref_q, _ = op.reference(x, w8_il, q, sf, compiled, pre=pre)
    frac = float((q != ref_q).float().mean())
    assert frac < 1e-2, f"{frac:.2e} of the bytes differ from the fp64 reference's"


def test_is_valid_refuses_the_plain_weight_shape(compiled) -> None:
    x = torch.zeros(600, entry.HIDDEN, dtype=torch.bfloat16, device="cuda")
    w = proj.quantize_weight(torch.zeros(512, entry.HIDDEN, dtype=torch.bfloat16, device="cuda"))
    with pytest.raises(AssertionError, match="prepare_weight"):
        with validating(op):
            op(x, w, *entry.operand(600, x.device), compiled)
