# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the spark2_5 headwise output gate op.

The entry owns what is certified: `CELLS` (both published geometries),
`reference` and `is_valid`. This file turns a cell spec into tensors and
drives the guard's rejection cases.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.spark2_5.spark2_5_dense__sm_10x__default.ops.attention.headwise_output_gate import (  # noqa: E501
    headwise_output_gate as op,
)

__extra_import_path__ = ["../../.."]  # noqa: F841 -- repo's file-scoped import hook

from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "the headwise output gate test requires a CUDA device"


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    tokens, num_heads, head_dim, dtype = (
        spec["tokens"],
        spec["num_heads"],
        spec["head_dim"],
        spec["dtype"],
    )
    attention_output = torch.randn(tokens, num_heads * head_dim, generator=g, device="cuda").to(
        dtype
    )
    gate = torch.randn(tokens, num_heads, generator=g, device="cuda").to(dtype)
    return attention_output, gate, num_heads, head_dim


# The op's own CELLS already carry both published geometries (1.7B: 8 heads,
# 4B: 16 heads, head_dim 256 each); no separate target cells to add.


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by name
    certifying(op)
    attention_output, gate, num_heads, head_dim = _build(
        cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31
    )
    out = op(attention_output, gate, num_heads, head_dim)
    assert out.shape == attention_output.shape and out.dtype == attention_output.dtype
    assert out.data_ptr() != attention_output.data_ptr(), "the op must return a new tensor"
    torch.testing.assert_close(
        out, op.reference(attention_output, gate, num_heads, head_dim), atol=2e-2, rtol=2e-2
    )


def test_guard_refuses_a_wrong_head_count() -> None:
    """A gate whose head count happens to divide the output width broadcasts
    silently; the guard is what refuses it."""
    attention_output = torch.randn(4, 8 * 256, device="cuda").to(torch.bfloat16)
    gate = torch.randn(4, 4, device="cuda").to(torch.bfloat16)  # 4 heads, not 8
    with pytest.raises(AssertionError, match=r"\[T, num_heads\]"):
        with validating(op):
            op(attention_output, gate, 8, 256)


def test_guard_refuses_mixed_dtypes() -> None:
    attention_output = torch.randn(4, 8 * 256, device="cuda").to(torch.bfloat16)
    gate = torch.randn(4, 8, device="cuda").to(torch.float32)
    with pytest.raises(AssertionError, match="dtype"):
        with validating(op):
            op(attention_output, gate, 8, 256)


def test_gate_semantics_are_per_head_sigmoid() -> None:
    """Pin the math against a hand computation: one sigmoid scalar per head,
    fp32 sigmoid, broadcast over head_dim -- not per-element, not softplus."""
    torch.manual_seed(0)
    tokens, num_heads, head_dim = 3, 2, 4
    attention_output = torch.randn(tokens, num_heads * head_dim, device="cuda")
    gate = torch.randn(tokens, num_heads, device="cuda")

    out = op(attention_output, gate, num_heads, head_dim)

    expected = (
        attention_output.view(tokens, num_heads, head_dim) * torch.sigmoid(gate).unsqueeze(-1)
    ).view(tokens, num_heads * head_dim)
    torch.testing.assert_close(out, expected)
