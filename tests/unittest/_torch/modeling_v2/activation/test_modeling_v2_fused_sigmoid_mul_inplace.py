# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fused_sigmoid_mul_inplace catalog entry: every cell against
the entry's reference with nothing written past `attention_output`'s last row,
graph capture of the unified target's fused-row and MTP forms, the band refusing
a wrong gate read, and the loud layout refusal."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.fused_sigmoid_mul_inplace import (  # noqa: E501
    fused_sigmoid_mul_inplace as op,
)

__extra_import_path__ = [".."]
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


def _mtp_spec(tokens: int) -> dict:
    """The unified MTP layer's 6-head interleaved form at `tokens` rows."""
    spec = next(
        c.spec for c in op.CELLS if c.spec["gate"] == "interleaved" and c.spec["heads"] == 6
    )
    return dict(spec, tokens=tokens)


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


def test_cuda_graph_capture_and_replay() -> None:
    """The decode call sites run captured: the tp4 / tep4 form (6 heads off the
    4352-wide row, two 1024-wide blocks a row) replays on refilled inputs."""
    spec = next(c.spec for c in op.CELLS if c.spec.get("row_stride") == 4352)
    spec = dict(spec, tokens=8)
    out, _, gate_buffer, gate = _build(spec, seed=5)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(out, gate)  # JIT compile outside capture
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(out, gate)

    for seed in (11, 12):
        fresh_out, _, fresh_gate_buffer, _ = _build(spec, seed=seed)
        out.copy_(fresh_out)
        gate_buffer.copy_(fresh_gate_buffer)
        expected = op.reference(out, gate)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("tokens", [1, 128, 1024])
def test_mtp_gate_graph_capture_and_replay(tokens) -> None:
    """The MTP drafter and verify run captured: the 6-head `[T, 6, 512]` gate half
    replays on refilled inputs, on both block widths (1024 below 1024 rows, 2048 from)."""
    spec = _mtp_spec(tokens)
    out, _, gate_buffer, gate = _build(spec, seed=5)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(out, gate)  # JIT compile outside capture
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(out, gate)

    for seed in (11, 12):
        fresh_out, _, fresh_gate_buffer, _ = _build(spec, seed=seed)
        out.copy_(fresh_out)
        gate_buffer.copy_(fresh_gate_buffer)
        expected = op.reference(out, gate)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("tokens", [1, 1024])
def test_the_gate_discriminates_a_wrong_gate_read(tokens) -> None:
    """The default bf16 band refuses the two reads a wrong per-head indexing gives:
    the q half of each head, and the first `hidden` elements of the flat
    `[T, 6 * 512]` row (head stride ignored)."""
    spec = _mtp_spec(tokens)
    hidden = spec["heads"] * spec["head_dim"]
    out, _, gate_buffer, gate = _build(spec, seed=_seed(f"wrong read {tokens}"))
    original = out.clone()
    op(out, gate)
    for name, wrong in (
        ("q half", gate_buffer[:, :, : spec["head_dim"]]),
        ("flat row", gate_buffer.reshape(tokens, -1)[:, :hidden]),
    ):
        with pytest.raises(AssertionError):
            torch.testing.assert_close(out, op.reference(original, wrong))
        assert (out.float() - op.reference(original, wrong).float()).abs().max() > 0.5, name


def test_metadata_misuse_raises() -> None:
    """A gate with a strided last dim is refused loudly, as is every layout
    violation; hence the entry has no `is_valid`."""
    g = torch.Generator(device="cuda").manual_seed(8)
    attention_output = torch.randn(4, 128, generator=g, device="cuda").to(torch.bfloat16)
    gate = torch.randn(4, 256, generator=g, device="cuda").to(torch.bfloat16)[:, ::2]
    with pytest.raises(AssertionError):
        op(attention_output, gate)
