# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_swiglu_mxf8 entry.

The operand must be the activation quant's bytes of what the bf16 dense SwiGLU
(`cute_dsl_dense_swiglu`, the op this one replaces on the MXF8 path) writes: byte for byte.
Against the fp64 reference only the bf16 rounding of the approximate sigmoid may differ."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel needs an sm_103 GPU", allow_module_level=True)

from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.kernels.common import (  # noqa: E501
    self_attn_prefill_act_quant_mxf8 as quant,
)  # noqa: E402, E501
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.activation import (  # noqa: E402, E501
    cute_dsl_dense_swiglu as dense,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.activation import (  # noqa: E402, E501
    cute_dsl_swiglu_mxf8 as entry,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

op = entry.cute_dsl_swiglu_mxf8


@pytest.fixture(scope="module")
def compiled():
    return entry.compile(torch.cuda.current_device())


@pytest.fixture(scope="module")
def compiled_dense():
    return dense.compile()


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, compiled, compiled_dense) -> None:
    rows, width = cell.spec["rows"], cell.spec["width"]
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    # The gate|up output scale: a few units, so the sigmoid sees both tails.
    source = (3.0 * torch.randn(rows, 2 * width, generator=g, device="cuda")).to(torch.bfloat16)
    q, sf = entry.operand(rows, width, source.device)
    with validating(op):
        got_q, got_sf = op(source, q, sf, compiled)
    torch.cuda.synchronize()
    assert got_q is q and got_sf is sf
    # Byte for byte the activation quant of the bf16 dense SwiGLU's output.
    act = dense.cute_dsl_dense_swiglu(source, compiled_dense)
    want_q, want_sf = quant.reference(act)
    assert torch.equal(q, want_q), f"{int((q != want_q).sum())} operand bytes differ"
    got = quant.dequantize(q, sf, rows, width)
    want = quant.dequantize(want_q, want_sf, rows, width)
    assert torch.equal(got, want), "block scales differ from the activation quant's"
    # Against the fp64 SwiGLU: the approximate sigmoid moves a rare bf16 value by one ulp,
    # which moves at most that element's e4m3 code.
    ref_q, _ = op.reference(source, q, sf, compiled)
    frac = float((q != ref_q).float().mean())
    assert frac < 1e-3, f"{frac:.2e} of the bytes differ from the fp64 reference's"


def test_is_valid_refuses_a_short_scale_buffer(compiled) -> None:
    source = torch.zeros(200, 6144, dtype=torch.bfloat16, device="cuda")
    q = torch.empty(200, 3072, dtype=torch.uint8, device="cuda")
    sf = torch.empty(entry.sf_bytes(128, 3072), dtype=torch.uint8, device="cuda")
    with pytest.raises(AssertionError, match="at least"):
        with validating(op):
            op(source, q, sf, compiled)
