# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_projection_mxf8_decode entry.

A producer-written operand (`pre`) holds the activation quant's bytes, so the GEMM alone
must give the quant + GEMM pair's output bitwise."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel needs an sm_103 GPU", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.kernels.common import (  # noqa: E501
    self_attn_prefill_act_quant_mxf8 as quant,
)  # noqa: E402, E501
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm import (  # noqa: E402, E501
    cute_dsl_projection_mxf8_decode as entry,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — path declared above

op = entry.cute_dsl_projection_mxf8_decode
_SHAPES = {"q_b": (4096, 2048), "o_proj": (6144, 4096)}
_RUNNERS: dict = {}


def _weight(kernel: str, g: torch.Generator) -> torch.Tensor:
    n, k = _SHAPES[kernel]
    return (torch.randn(n, k, generator=g, device="cuda") * k**-0.5).to(torch.bfloat16)


def _runner(kernel: str) -> entry.DecodeMxf8Runner:
    if kernel not in _RUNNERS:
        g = torch.Generator(device="cuda").manual_seed(1)
        _RUNNERS[kernel] = entry.build(kernel, entry._kernel.quantize_weight(_weight(kernel, g)))
    return _RUNNERS[kernel]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    kernel, rows = cell.spec["kernel"], cell.spec["rows"]
    runner = _runner(kernel)
    n, k = _SHAPES[kernel]
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    weight = entry._kernel.quantize_weight(_weight(kernel, g))
    x = torch.randn(rows, k, generator=g, device="cuda").to(torch.bfloat16)
    pre = bool(cell.spec.get("pre"))
    if pre:
        # The producer's bytes: the activation quant's, written into the launcher's operand.
        xq, sf = quant.reference(x)
        runner.operand[0][:rows].copy_(xq)
        need = quant.sf_bytes(rows, k)
        runner.operand[1][:need].copy_(sf[:need])
    out = torch.full((rows, n), float("nan"), dtype=torch.bfloat16, device="cuda")
    with validating(op):
        assert op(x, weight, out, runner, pre=pre) is out
    torch.cuda.synchronize()
    ref = op.reference(x, weight, out, runner, pre=pre)
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)
    if pre:
        plain = torch.full_like(out, float("nan"))
        op(x, weight, plain, runner)
        torch.cuda.synchronize()
        assert torch.equal(out, plain), "the producer's operand must give the quant path's output"


def test_shift_scales_is_the_scaled_weight() -> None:
    g = torch.Generator(device="cuda").manual_seed(5)
    w = _weight("q_b", g)
    shifted = entry.shift_scales(entry._kernel.quantize_weight(w), 3)
    direct = entry._kernel.quantize_weight((w.float() * 8).to(torch.bfloat16))
    assert torch.equal(shifted.q, direct.q) and torch.equal(shifted.sf, direct.sf)
