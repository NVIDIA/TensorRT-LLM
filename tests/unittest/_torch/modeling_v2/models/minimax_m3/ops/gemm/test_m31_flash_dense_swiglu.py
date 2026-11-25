# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gemm/m31_dense_swiglu entry of the MiniMax-M3.1-Flash sm_107 targets: the dense MLP's swigluoai
C++ op (trtllm::minimax_m3_sm107_tp4_swiglu_oai) at each declared cell, bit for bit the Triton silu_and_mul its
``reference`` runs, with and without the programmatic launch."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.gemm.m31_dense_swiglu import (  # noqa: E501
    ALPHA,
    BETA,
    LIMIT,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.gemm.m31_dense_swiglu import (  # noqa: E501
    m3_dense_swiglu as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("the sm_107 targets run on SM107 only", allow_module_level=True)
if not hasattr(torch.ops.trtllm, "minimax_m3_sm107_tp4_swiglu_oai"):
    pytest.skip(
        "needs the TP4 op library with minimax_m3_sm107_tp4_swiglu_oai", allow_module_level=True
    )

# The targets share their cells; each distinct one runs once.
CELLS = list({repr(c): c for c in op.CELLS}.values())


@pytest.mark.parametrize("pdl", [True, False], ids=["pdl", "plain"])
@pytest.mark.parametrize("cell", CELLS, ids=[c.why for c in CELLS])
def test_certified_cells(cell, pdl: bool) -> None:
    """gate_up rows in a GEMV output's range plus values past the clamps."""
    rows, d = cell.spec["rows"], cell.spec["d"]
    g = torch.Generator(device="cuda").manual_seed(3000 + rows)
    x = (3.0 * torch.randn(rows, 2 * d, generator=g, device="cuda")).to(torch.bfloat16)
    x[:, :8] = torch.tensor([8.0, -8.0, 7.0, -7.0, 0.0, -0.0, 50.0, -50.0], dtype=torch.bfloat16)
    x[:, d : d + 8] = x[:, :8]
    args = (x, LIMIT, ALPHA, BETA, pdl)
    with validating(op):
        out = op(*args)
    assert out is not None, "the op declined a contiguous bf16 [rows, 2d] input"
    ref = op.reference(*args)
    bad = (out.view(torch.int16) != ref.view(torch.int16)).nonzero().flatten()
    assert bad.numel() == 0, f"{bad.numel()} mismatches, first at {bad[:4].tolist()}"
