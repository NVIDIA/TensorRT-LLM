# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fused all-reduce boundary's binding refuses operands its kernel cannot read.

Each case breaks one operand of an otherwise valid call; the binding must raise before the launch,
so one process and one device are enough (a call that launched would wait for the other ranks).
The collective itself is certified by ``_mhc_ar_boundary_op_matrix.py``.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4 import (  # noqa: E501
    native_ops,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.comm import (  # noqa: E501
    mhc_ar_boundary as boundary,
)
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    get_sm_version() != 100, reason="the glm5_next fp8 tp4 entries are certified on sm_100"
)

ROWS, MULT, HIDDEN = 4, boundary.MULT, boundary.HIDDEN
MIX = (2 + MULT) * MULT


def _call() -> dict:
    """A call with every operand at the shape, dtype and layout the kernel is compiled for."""

    def empty(*shape, dtype=torch.float32):
        return torch.empty(shape, dtype=dtype, device="cuda")

    bf16 = torch.bfloat16
    return dict(
        partial=empty(ROWS, HIDDEN, dtype=bf16),
        workspace=torch.zeros(3 * boundary.WORLD + 2, dtype=torch.int64, device="cuda"),
        rank=0,
        residual=empty(ROWS, MULT, HIDDEN, dtype=bf16),
        post=empty(ROWS, MULT),
        comb=empty(ROWS, MULT, MULT),
        fn=empty(MIX, MULT * HIDDEN),
        scale=empty(3),
        base=empty(MIX),
        norm_weight=empty(HIDDEN, dtype=bf16),
        sinkhorn_iters=20,
        norm_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_eps=1e-6,
        post_mult=2.0,
        layer_input_eps=1e-5,
        residual_out=empty(ROWS, MULT, HIDDEN, dtype=bf16),
        post_out=empty(ROWS, MULT),
        comb_out=empty(ROWS, MULT, MULT),
        layer_input=empty(ROWS, HIDDEN, dtype=bf16),
        pdl=False,
    )


def _misaligned(*shape, dtype):
    """A contiguous tensor that starts 2 elements past a 16-byte boundary."""
    numel = 1
    for size in shape:
        numel *= size
    return torch.empty(numel + 8, dtype=dtype, device="cuda")[2 : 2 + numel].view(*shape)


_BAD = {
    "partial": lambda: torch.empty(ROWS, HIDDEN, dtype=torch.float16, device="cuda"),
    "workspace": lambda: torch.zeros(3 * boundary.WORLD + 1, dtype=torch.int64, device="cuda"),
    "rank": lambda: boundary.WORLD,
    "residual": lambda: torch.empty(ROWS, MULT + 1, HIDDEN, dtype=torch.bfloat16, device="cuda"),
    "post": lambda: torch.empty(ROWS * MULT, device="cuda"),
    "comb": lambda: torch.empty(ROWS, MULT, MULT, device="cuda").transpose(1, 2),
    "fn": lambda: _misaligned(MIX, MULT * HIDDEN, dtype=torch.float32),
    "scale": lambda: torch.empty(3, dtype=torch.bfloat16, device="cuda"),
    "base": lambda: torch.empty(MIX - 1, device="cuda"),
    "norm_weight": lambda: torch.empty(HIDDEN, device="cuda"),
    "sinkhorn_iters": lambda: 0,
    "residual_out": lambda: _misaligned(ROWS, MULT, HIDDEN, dtype=torch.bfloat16),
    "post_out": lambda: torch.empty(ROWS, MULT, dtype=torch.bfloat16, device="cuda"),
    "comb_out": lambda: torch.empty(ROWS, MULT * MULT, device="cuda"),
    "layer_input": lambda: torch.empty(ROWS + 1, HIDDEN, dtype=torch.bfloat16, device="cuda"),
}


@pytest.mark.parametrize("name", list(_BAD))
def test_rejects_an_operand_the_kernel_cannot_read(name):
    call = _call()
    call[name] = _BAD[name]()
    with pytest.raises(RuntimeError, match=f"mhc_ar_boundary: {name} must"):
        native_ops.mhc_ar_boundary(*call.values())
