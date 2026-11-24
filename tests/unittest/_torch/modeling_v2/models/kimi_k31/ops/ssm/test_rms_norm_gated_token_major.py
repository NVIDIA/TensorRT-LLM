# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 rms_norm_gated_token_major entry.

Each cell builds the KDA layer's prefill output gate: x the contiguous [T * H, 128] core of the prefill, z the
[T, H, 128] view of the fused [q | k | v | g] projection's gate columns (or a dense gate), the bf16 o_norm weight,
sigmoid gate. Output within one bf16 rounding of an fp32 evaluation; inputs untouched.
"""

import pytest
import torch
from _kda_ops_util import HEAD_DIM, bits, generator, requires_sm103

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rms_norm_gated_token_major import (
    rms_norm_gated_token_major as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

pytestmark = requires_sm103


def _inputs(spec, gen):
    tokens, heads, width = spec["tokens"], spec["heads"], spec["width"]
    dim = heads * width
    x = torch.randn(tokens * heads, width, generator=gen, device="cuda").to(torch.bfloat16)
    columns = spec["gate_columns_of"]
    # The gate is the last section of a [T, columns * D] projection output.
    projection = torch.randn(tokens, columns * dim, generator=gen, device="cuda").to(torch.bfloat16)
    z = projection[:, (columns - 1) * dim :].reshape(tokens, heads, width)
    weight = (1 + 0.1 * torch.randn(width, generator=gen, device="cuda")).to(torch.bfloat16)
    return x, z, weight, spec["eps"]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:60] for c in op.CELLS])
@torch.inference_mode()
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    x, z, weight, eps = _inputs(cell.spec, generator(cell))
    if cell.spec["gate_columns_of"] > 1 and cell.spec["tokens"] > 1:
        assert z.stride(0) == cell.spec["gate_columns_of"] * z.shape[1] * HEAD_DIM, (
            "not the strided gate view"
        )
    x_before, z_before = x.clone(), z.clone()
    out = op(x, z, weight, eps, None, "sigmoid")
    ref = op.reference(x, z, weight, eps, None, "sigmoid")
    assert out.shape == x.shape and out.dtype == x.dtype and out.is_contiguous()
    assert out.data_ptr() not in (x.data_ptr(), z.data_ptr()), "the op must return a new tensor"
    # fp32 statistics in both; one rounding to bf16 at the end (plus the kernel's rsqrt): one bf16 ulp
    # of the output's scale, relative and absolute.
    torch.testing.assert_close(out, ref, rtol=2**-7, atol=2**-7)
    assert torch.equal(bits(x), bits(x_before)) and torch.equal(bits(z), bits(z_before)), (
        "inputs mutated"
    )


def test_guard_refuses_a_per_column_fp8_scale() -> None:
    """A multi-element fp8 scale is read only at its first element: a plausible, wrong quantization."""
    gen = torch.Generator(device="cuda").manual_seed(3)
    x, z, weight, eps = _inputs(
        dict(tokens=4, heads=6, width=128, eps=1e-5, gate_columns_of=1), gen
    )
    scale = torch.ones(128, dtype=torch.float32, device="cuda")
    with pytest.raises(AssertionError, match="exactly one element"):
        with validating(op):
            op(x, z, weight, eps, scale, "sigmoid")
