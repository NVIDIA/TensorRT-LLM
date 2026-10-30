# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fused_sigmoid_mul_inplace entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.activation.fused_sigmoid_mul_inplace__mtp import (  # noqa: E501
    fused_sigmoid_mul_inplace as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "fused_sigmoid_mul_inplace requires a CUDA device"


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def _build(spec, seed):
    """`(out, gate_buffer, gate)` for one spec.

    `gate` is a view of `gate_buffer`: the second half of each
    `[T, heads, 2 * D]` q_proj head, or the 2-D gate window of the target's
    fused projection row.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    t, heads, d, dtype = spec["tokens"], spec["heads"], spec["head_dim"], spec["dtype"]
    hidden = heads * d

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda").to(dtype)

    out = randn(t, hidden)
    if spec["gate"] == "interleaved":
        gate_buffer = randn(t, heads, 2 * d)
        gate = gate_buffer[:, :, d:]
    else:
        assert spec["gate"] == "fused_row", spec["gate"]
        # The target's fused input projection row [q | k | v | gate | index | pad]:
        # 14080 columns, the gate a 2-D [T, hidden] window at column 7168.
        gate_buffer = randn(t, 14080)
        gate = gate_buffer[:, 7168 : 7168 + hidden]
    return out, gate_buffer, gate


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    out, gate_buffer, gate = _build(cell.spec, seed=_seed(cell.why))
    gate_before = gate_buffer.clone()
    expected = op.reference(out, gate)

    result = op(out, gate)

    assert result.data_ptr() == out.data_ptr() and result.shape == out.shape
    torch.testing.assert_close(result, expected)
    assert torch.equal(gate_buffer, gate_before)


def test_metadata_misuse_raises() -> None:
    """A gate with a strided last dim is refused loudly, as is every layout
    violation; hence the entry has no `is_valid`."""
    g = torch.Generator(device="cuda").manual_seed(8)
    attention_output = torch.randn(4, 128, generator=g, device="cuda").to(torch.bfloat16)
    gate = torch.randn(4, 256, generator=g, device="cuda").to(torch.bfloat16)[:, ::2]
    with pytest.raises(AssertionError):
        op(attention_output, gate)
