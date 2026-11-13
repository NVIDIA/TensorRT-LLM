# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family's trtllm::k3_rms_norm_gated_token_major is a copy of trtllm::rms_norm_gated_token_major: same bits.

Both ops are registered in one process (the built-in one by importing tensorrt_llm._torch.modules.mamba.
layernorm_gated), and every cell of the rms_norm_gated_token_major OpWrapper must give identical outputs. The cells
reach all three kernels: the token-major multi-row kernel, the row-level multi-row kernel after a gate copy
(head-strided gate) and the generic one-pass kernel (row length not a power of two, or above 256).
"""

import zlib

import pytest
import torch

import tensorrt_llm._torch.modules.mamba.layernorm_gated  # noqa: F401  (registers the comparison op)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.norm.rms_norm_gated_token_major import (
    rms_norm_gated_token_major,
)

assert torch.cuda.is_available(), "k3_rms_norm_gated_token_major requires a CUDA device"
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="certified on sm_100 and sm_103",
)


def _inputs(cell):
    """The rms_norm_gated_token_major op test's inputs for ``cell``."""
    spec = cell.spec
    torch.manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    tokens, heads, n, dtype = spec["tokens"], spec["heads"], spec["width"], spec["dtype"]
    weight = torch.randn(n, dtype=dtype, device="cuda")
    x = torch.randn(tokens * heads, n, dtype=dtype, device="cuda")
    if spec.get("head_strided"):
        z = torch.randn(tokens, heads, 2 * n, dtype=dtype, device="cuda")[:, :, :n]
    else:
        wide = torch.randn(tokens, heads * n + spec.get("extra", 0), dtype=dtype, device="cuda")
        z = wide[:, : heads * n].view(tokens, heads, n)
    fp8_scale = None
    if "fp8_scale" in spec:
        fp8_scale = torch.tensor(spec["fp8_scale"], dtype=torch.float32, device="cuda")
    return x, z, weight, spec["eps"], fp8_scale, spec["gate_activation"]


@pytest.mark.parametrize(
    "cell",
    rms_norm_gated_token_major.CELLS,
    ids=[c.why[:40] for c in rms_norm_gated_token_major.CELLS],
)
def test_bitwise_equal_to_builtin_op(cell) -> None:
    args = _inputs(cell)
    family = torch.ops.trtllm.k3_rms_norm_gated_token_major(*args)
    builtin = torch.ops.trtllm.rms_norm_gated_token_major(*args)
    assert family.shape == builtin.shape and family.dtype == builtin.dtype
    assert torch.equal(family.view(torch.uint8), builtin.view(torch.uint8)), cell.why
