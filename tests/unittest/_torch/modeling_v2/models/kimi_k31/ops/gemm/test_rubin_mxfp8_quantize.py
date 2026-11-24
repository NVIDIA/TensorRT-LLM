# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the rubin_mxfp8_quantize entry: the activation quantization of rubin_mxfp8_linear
above the low-M GEMM's rows.

Every cell (the K3.1 prefill widths at 256..8192 rows) runs the entry against its reference and
against the C++ ``mxfp8_quantize`` it replaces; both must match byte for byte.
"""

import pytest
import torch
from _rubin_gemm_util import activation, generator, requires_sm107

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_mxfp8_quantize import (  # noqa: E501
    rubin_mxfp8_quantize as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

pytestmark = requires_sm107


def _same(got, want) -> bool:
    return torch.equal(got[0].view(torch.uint8), want[0].view(torch.uint8)) and torch.equal(
        got[1], want[1]
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:56] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture resolved by name
    certifying(op)
    x = activation(cell.spec["m"], cell.spec["k"], generator(cell.why))
    out = op(x)
    assert _same(out, op.reference(x))
    assert _same(out, torch.ops.trtllm.mxfp8_quantize(x, True, 32))


def test_guard_refuses_a_partial_block() -> None:
    x = torch.zeros(256, 48, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="32-value"):
        with validating(op):
            op(x)


def test_guard_refuses_a_strided_input() -> None:
    x = torch.zeros(256, 512, dtype=torch.bfloat16, device="cuda")[:, :256]
    with pytest.raises(AssertionError, match="contiguous"):
        with validating(op):
            op(x)
