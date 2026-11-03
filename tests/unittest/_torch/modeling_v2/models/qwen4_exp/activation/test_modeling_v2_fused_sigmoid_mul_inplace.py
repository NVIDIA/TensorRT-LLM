# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fused_sigmoid_mul_inplace catalog entry: every cell against
the entry's reference with nothing written past `attention_output`'s last row."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.activation.fused_sigmoid_mul_inplace import (  # noqa: E501
    fused_sigmoid_mul_inplace as op,
)

__extra_import_path__ = ["../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "fused_sigmoid_mul_inplace requires a CUDA device"

#: Elements past `attention_output`'s last row, inside the buffer it views: a
#: masked lane that stored anyway would land here (one kernel block is 2048 wide).
_TAIL = 2048


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def _build(spec, seed):
    """`(out, out_buffer, gate_buffer, gate)` for one spec.

    `out` is the compact `[T, hidden]` head of `out_buffer`, which runs `_TAIL`
    elements past its last row. `gate` is a view of `gate_buffer`: the second
    half of each `[T, heads, 2 * D]` q_proj head (`gate="interleaved"`), or the
    2-D gate window of the target's fused projection row (`gate="fused_row"`:
    `row_stride` columns, the gate at `gate_col`; tp1's 14080 / 7168 where the
    spec names neither).
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    t, heads, d, dtype = spec["tokens"], spec["heads"], spec["head_dim"], spec["dtype"]
    hidden = heads * d

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda").to(dtype)

    out_buffer = randn(t * hidden + _TAIL)
    out = out_buffer[: t * hidden].view(t, hidden)
    if spec["gate"] == "interleaved":
        gate_buffer = randn(t, heads, 2 * d)
        gate = gate_buffer[:, :, d:]
    else:
        assert spec["gate"] == "fused_row", spec["gate"]
        # The target's fused input projection row [q | k | v | gate | index scratch]:
        # the gate a 2-D [T, hidden] window at column qkv_width.
        row_stride, gate_col = spec.get("row_stride", 14080), spec.get("gate_col", 7168)
        gate_buffer = randn(t, row_stride)
        gate = gate_buffer[:, gate_col : gate_col + hidden]
    return out, out_buffer, gate_buffer, gate


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    out, out_buffer, gate_buffer, gate = _build(cell.spec, seed=_seed(cell.why))
    gate_before = gate_buffer.clone()
    tail_before = out_buffer[out.numel() :].clone()
    expected = op.reference(out, gate)

    result = op(out, gate)

    assert result.data_ptr() == out.data_ptr() and result.shape == out.shape
    torch.testing.assert_close(result, expected)
    assert torch.equal(gate_buffer, gate_before)
    assert torch.equal(out_buffer[out.numel() :], tail_before), "a masked lane stored"
