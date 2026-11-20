# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the activation/m31_silu_and_mul entry of the MiniMax-M3.1-Flash sm_107 targets: SwiGLU-OAI and plain
SwiGLU, each target's cells with the ``native_vec`` it passes (tp2 False, tp4 and tp8 their SILU_VEC switch)."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.activation.m31_silu_and_mul import (  # noqa: E501
    m31_silu_and_mul as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("the sm_107 targets run on SM107 only", allow_module_level=True)

OAI = (7.0, 1.702, 1.0)
CELLS = [(tp, cell) for tp, cells in op.CELLS_BY_TP.items() for cell in cells]


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    # Gate and up spread past the clamp (|x| up to ~12) so both clamps engage.
    x = (torch.randn(spec["m"], 2 * spec["inter"], generator=g, device="cuda") * 4.0).to(
        torch.bfloat16
    )
    return (x, *OAI) if spec["oai"] else (x, None, None, None)


@pytest.mark.parametrize("tp,cell", CELLS, ids=[f"tp{tp}-{c.why[:44]}" for tp, c in CELLS])
def test_certified_cells(tp, cell) -> None:
    """fp32 arithmetic and one bf16 rounding on both sides (the native op is bit for bit the Triton kernel): the
    fast sigmoid flips the bf16 rounding of an element by at most one step."""
    args = (*_build(cell.spec, zlib.crc32(cell.why.encode()) % 2**31), tp != 2)
    with validating(op):
        out = op(*args)
    assert out.shape == (cell.spec["m"], cell.spec["inter"]) and out.dtype == torch.bfloat16
    assert_within_ulp(out, op.reference(*args), element_ulp=1.0, rms_ulp=0.1)
