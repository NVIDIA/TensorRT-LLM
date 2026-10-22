# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fused_sigmoid_mul_inplace entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.activation.fused_sigmoid_mul_inplace import (  # noqa: E501
    fused_sigmoid_mul_inplace as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "fused_sigmoid_mul_inplace requires a CUDA device"


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def _build(spec, seed):
    """`(buffer, out, region, gate_buffer, gate)` for one spec.

    `out` is `region(buffer)`: the whole buffer, a padded window of it, or every
    other row. `gate` is a view of `gate_buffer` -- for the checkpoint layout the
    second half of each `[T, heads, 2 * D]` q_proj head.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    t, heads, d, dtype = spec["tokens"], spec["heads"], spec["head_dim"], spec["dtype"]
    hidden = heads * d

    def randn(*shape, dt=dtype):
        return torch.randn(*shape, generator=g, device="cuda").to(dt)

    layout = spec.get("out", "contiguous")
    if layout == "window":
        buffer = randn(t + 2, hidden + 2048)

        def region(x):
            return x[:t, :hidden]
    elif layout == "every_other_row":
        buffer = randn(2 * t, hidden)

        def region(x):
            return x[::2]
    else:
        buffer = randn(t, hidden)

        def region(x):
            return x

    gate_layout = spec["gate"]
    if gate_layout == "interleaved":
        gate_buffer = randn(t, heads, 2 * d)
        gate = gate_buffer[:, :, d:]
    elif gate_layout == "packed":
        gate_buffer = gate = randn(t, heads, d)
    else:
        gate_buffer = gate = randn(t, hidden, dt=spec.get("gate_dtype", dtype))
    return buffer, region(buffer), region, gate_buffer, gate


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    buffer, out, region, gate_buffer, gate = _build(cell.spec, seed=_seed(cell.why))
    if cell.spec["gate"] == "interleaved" and cell.spec["tokens"] > 1:
        assert gate.stride() == (
            cell.spec["heads"] * 2 * cell.spec["head_dim"],
            2 * cell.spec["head_dim"],
            1,
        )
    buffer_before, gate_before = buffer.clone(), gate_buffer.clone()
    expected = op.reference(out, gate)

    result = op(out, gate)

    assert result.data_ptr() == out.data_ptr() and result.shape == out.shape
    torch.testing.assert_close(result, expected)
    # In place: everything outside the [T, hidden] window, and the gate, bitwise unchanged.
    touched = torch.zeros_like(buffer, dtype=torch.bool)
    region(touched).fill_(True)
    assert torch.equal(buffer[~touched], buffer_before[~touched])
    assert torch.equal(gate_buffer, gate_before)


def test_the_gate_half_is_read_through_its_own_strides() -> None:
    """The strided view and its compacted copy agree bitwise; the q half of the
    same buffer does not -- so the cells can tell the two halves apart."""
    spec = dict(op.CELLS[1].spec)
    _, out, _, q_and_gate, gate = _build(spec, seed=7)
    q_half = q_and_gate[:, :, : spec["head_dim"]]

    from_strided = op(out.clone(), gate)
    from_compacted = op(out.clone(), gate.contiguous())
    from_wrong_half = op(out.clone(), q_half)

    assert torch.equal(from_strided, from_compacted)
    torch.testing.assert_close(from_strided, op.reference(out, gate))
    assert (from_wrong_half.float() - from_strided.float()).abs().max().item() > 0.5


@pytest.mark.parametrize(
    "case",
    [
        "output_3d",
        "output_last_stride",
        "gate_numel_mismatch",
        "gate_token_mismatch",
        "gate_2d_shape_mismatch",
        "gate_1d",
        "gate_4d",
        "gate_last_stride",
    ],
)
def test_metadata_misuse_raises(case: str) -> None:
    """Every layout violation is loud, which is why the entry has no `is_valid`."""
    g = torch.Generator(device="cuda").manual_seed(8)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda").to(torch.bfloat16)

    attention_output, gate = randn(4, 128), randn(4, 2, 64)
    if case == "output_3d":
        attention_output = randn(4, 2, 64)
    elif case == "output_last_stride":
        attention_output = randn(4, 256)[:, ::2]
    elif case == "gate_numel_mismatch":
        gate = randn(4, 2, 32)
    elif case == "gate_token_mismatch":
        gate = randn(3, 2, 64)
    elif case == "gate_2d_shape_mismatch":
        gate = randn(4, 64)
    elif case == "gate_1d":
        gate = randn(512)
    elif case == "gate_4d":
        gate = randn(4, 2, 2, 32)
    elif case == "gate_last_stride":
        gate = randn(4, 256)[:, ::2]

    with pytest.raises(AssertionError):
        op(attention_output, gate)


def test_a_cpu_tensor_is_rejected_by_triton() -> None:
    attention_output = torch.randn(4, 128, dtype=torch.bfloat16)
    gate = torch.randn(4, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="Pointer argument cannot be accessed from Triton"):
        op(attention_output, gate)
