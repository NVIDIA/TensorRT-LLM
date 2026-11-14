# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen3.8-27b sm_120 tp1 flashinfer_gemma_fused_add_rmsnorm entry.

The cells against the fp32 reference: the residual bitwise (both sides round one exact
fp32 sum of two bf16 values once), the normalized output in torch's default bf16 band.
Then the aliasing guard.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.norm.flashinfer_gemma_fused_add_rmsnorm import (  # noqa: E501
    flashinfer_gemma_fused_add_rmsnorm as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the qwen3.8-27b sm_120 tp1 entries are certified on sm_120"
)


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


def _build(spec, seed):
    """`(x, residual, weight, eps)`: rows at magnitudes spread over e^[-6, 5] (the residual
    stream grows across layers; the attention output does not), one all-zero 16-column
    block and one all -0.0 block per row, and a weight with entries at exactly -1 (so
    `1 + w == 0`) and at x20."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows, hidden = spec["rows"], spec["hidden"]
    scale = torch.empty(rows, 1, device="cuda").uniform_(-6.0, 5.0, generator=g).exp()
    x = (torch.randn(rows, hidden, generator=g, device="cuda") * scale).to(spec["dtype"])
    residual = (torch.randn(rows, hidden, generator=g, device="cuda") * 4 * scale).to(spec["dtype"])
    x[:, 32:48] = 0.0
    residual[:, 32:48] = 0.0
    x[:, 80:96] = -0.0
    residual[:, 80:96] = -0.0
    weight = torch.randn(hidden, generator=g, device="cuda") * 0.3
    weight[torch.randperm(hidden, generator=g, device="cuda")[: hidden // 100]] = -1.0
    weight[torch.randperm(hidden, generator=g, device="cuda")[: hidden // 100]] *= 20.0
    return x, residual, weight.to(spec["dtype"]), spec["eps"]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    x, residual, weight, eps = _build(cell.spec, seed=_seed(cell.why))
    ref_x, ref_residual = op.reference(x, residual, weight, eps)

    out_x, out_residual = op(x, residual, weight, eps)

    assert out_x is x and out_residual is residual, "the op writes in place"
    assert torch.equal(residual.view(torch.int16), ref_residual.view(torch.int16))
    torch.testing.assert_close(x, ref_x)


def test_guard_refuses_aliased_buffers() -> None:
    """One buffer passed as both: the op writes the sum, then the output, over it."""
    x, _, weight, eps = _build(op.CELLS[0].spec, seed=_seed("alias"))
    with pytest.raises(AssertionError, match="distinct buffers"):
        with validating(op):
            op(x, x, weight, eps)
