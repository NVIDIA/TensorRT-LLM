# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fused_sigmoid_mul_inplace catalog entry.

Cells are gated against the entry's fp32-once reference in torch's default
bf16 band. The Step-5 cells also gate against Step-5's own formulation, which
rounds `sigmoid(g)` to bf16 before the multiply: the op's single rounding and
that double rounding differ by at most one bf16 ulp per element. The exact
product `a * s` and `a * bf16(s)` are within `|a * s| * 2^-9` of each other
(half a bf16 ulp of `s`, relative), which is under one ulp of the product, so
their roundings land at most one ulp apart.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.fused_sigmoid_mul_inplace import (  # noqa: E501
    fused_sigmoid_mul_inplace as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "fused_sigmoid_mul_inplace requires a CUDA device"

#: Spread of the synthetic per-head gate logits. N(0, 4^2) reaches |g| > 8
#: on about 5% of heads, so near-closed (sigmoid ~ 3e-4) and near-open heads
#: are both driven, not only the linear middle of the sigmoid.
LOGIT_STD = 4.0

#: The Step-5 decode site is captured at these padded batches; the widest
#: (dp, 64 heads x 256) geometry is the one replayed here.
CAPTURED_TOKENS = (1, 128)


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def _build(spec, seed):
    """`(out, gate_buffer, gate)` for one spec.

    `gate` is a view of `gate_buffer`: the second half of each
    `[T, heads, 2 * D]` q_proj head, the 2-D gate window of qwen4_exp's fused
    projection row, or Step-5's per-head logits `[T, heads]` materialised to a
    contiguous `[T, heads, D]` gate (`gate_buffer` is then the logits). With
    `padded_from`, each head's lanes from that column on are zero in `out`, as
    the attention output of a head zero-padded 192 -> 256 is.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    t, heads, d, dtype = spec["tokens"], spec["heads"], spec["head_dim"], spec["dtype"]
    hidden = heads * d

    def randn(*shape, std=1.0):
        return (torch.randn(*shape, generator=g, device="cuda") * std).to(dtype)

    out = randn(t, hidden)
    if spec.get("padded_from") is not None:
        out.view(t, heads, d)[:, :, spec["padded_from"] :] = 0
    if spec["gate"] == "interleaved":
        gate_buffer = randn(t, heads, 2 * d)
        gate = gate_buffer[:, :, d:]
    elif spec["gate"] == "per_head_materialized":
        gate_buffer = randn(t, heads, std=LOGIT_STD)
        gate = gate_buffer.unsqueeze(-1).expand(t, heads, d).contiguous()
    else:
        assert spec["gate"] == "fused_row", spec["gate"]
        # qwen4_exp's fused input projection row [q | k | v | gate | index | pad]:
        # 14080 columns, the gate a 2-D [T, hidden] window at column 7168.
        gate_buffer = randn(t, 14080)
        gate = gate_buffer[:, 7168 : 7168 + hidden]
    return out, gate_buffer, gate


def _dp_padded_spec(tokens: int) -> dict:
    """Step-5's widest gate site: attention DP, 64 heads padded to 256."""
    return dict(
        tokens=tokens,
        heads=64,
        head_dim=256,
        padded_from=192,
        dtype=torch.bfloat16,
        gate="per_head_materialized",
    )


def _ordered_bits(x: torch.Tensor) -> torch.Tensor:
    """bf16 bit patterns mapped to integers that are monotone in value (+0 == -0)."""
    bits = x.view(torch.int16).to(torch.int32)
    return torch.where(bits < 0, -(bits & 0x7FFF), bits)


def _ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (_ordered_bits(a) - _ordered_bits(b)).abs()


def _step5_double_rounded(attention_output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Step-5's formulation: `sigmoid(g)` stored in bf16, then a bf16 multiply."""
    num_tokens, hidden = attention_output.shape
    s = torch.sigmoid(gate.reshape(num_tokens, hidden))
    assert s.dtype == torch.bfloat16
    return (attention_output.float() * s.float()).to(attention_output.dtype)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    out, gate_buffer, gate = _build(cell.spec, seed=_seed(cell.why))
    gate_before = gate_buffer.clone()
    gate_view_before = gate.clone()
    expected = op.reference(out, gate)
    step5 = _step5_double_rounded(out, gate)

    result = op(out, gate)

    assert result.data_ptr() == out.data_ptr() and result.shape == out.shape
    torch.testing.assert_close(result, expected)
    assert torch.equal(gate_buffer, gate_before)
    assert torch.equal(gate, gate_view_before)
    if cell.spec["gate"] == "per_head_materialized":
        distance = _ulp_distance(result, step5)
        assert int(distance.max()) <= 1, (
            f"{int((distance > 1).sum())} elements more than one ulp from the double-rounded form"
        )
    pad = cell.spec.get("padded_from")
    if pad is not None:
        t, heads, d = cell.spec["tokens"], cell.spec["heads"], cell.spec["head_dim"]
        assert torch.count_nonzero(result.view(t, heads, d)[:, :, pad:]) == 0


@pytest.mark.parametrize("tokens", CAPTURED_TOKENS)
def test_cuda_graph_capture_and_replay(tokens, certifying) -> None:  # noqa: F811
    """The decode site is captured, so a replay must compute the refilled operands.

    Warm up on a side stream (the first call JIT-compiles), capture, refill
    both operands in place, replay, and check the replay against the
    reference of the new operands and against an eager call.
    """
    certifying(op)
    spec = _dp_padded_spec(tokens)
    out, _, gate = _build(spec, seed=_seed(f"capture {tokens}"))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            op(out, gate)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = op(out, gate)

    fresh_out, _, fresh_gate = _build(spec, seed=_seed(f"replay {tokens}"))
    out.copy_(fresh_out)
    gate.copy_(fresh_gate)
    expected = op.reference(fresh_out, fresh_gate)
    graph.replay()
    torch.cuda.synchronize()

    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(result, expected)
    eager = fresh_out.clone()
    op(eager, fresh_gate)
    assert torch.equal(result, eager), "graph replay differs from the eager call"


def test_head_shifted_gate_leaves_the_band() -> None:
    """A gate shifted by one head lands well outside the cells' default band.

    That is the wrong head mapping a misplaced slice or reshape would produce,
    so the per-head read is under test.
    """
    spec = _dp_padded_spec(128)
    out, _, gate = _build(spec, seed=_seed("head shift"))
    expected = op.reference(out, gate)
    shifted = gate.roll(1, dims=1).contiguous()

    op(out, shifted)

    max_abs = float((out.float() - expected.float()).abs().max())
    assert max_abs > 0.5, f"a one-head shift moved the result only {max_abs}"
    with pytest.raises(AssertionError):
        torch.testing.assert_close(out, expected)


def test_metadata_misuse_raises() -> None:
    """A gate with a strided last dim is refused loudly.

    So is every layout violation; hence the entry has no `is_valid`.
    """
    g = torch.Generator(device="cuda").manual_seed(8)
    attention_output = torch.randn(4, 128, generator=g, device="cuda").to(torch.bfloat16)
    gate = torch.randn(4, 256, generator=g, device="cuda").to(torch.bfloat16)[:, ::2]
    with pytest.raises(AssertionError):
        op(attention_output, gate)


def test_broadcast_head_gate_is_refused_loudly() -> None:
    """A broadcast per-head gate view is refused before anything is written.

    Step-5's per-head logits expanded to `[T, H, D]` (last stride 0) trip the
    op's own assert; the caller has to materialise the gate.
    """
    g = torch.Generator(device="cuda").manual_seed(9)
    tokens, heads, d = 4, 16, 192
    attention_output = torch.randn(tokens, heads * d, generator=g, device="cuda").to(torch.bfloat16)
    logits = torch.randn(tokens, heads, generator=g, device="cuda").to(torch.bfloat16)
    before = attention_output.clone()
    with pytest.raises(AssertionError):
        op(attention_output, logits.unsqueeze(-1).expand(tokens, heads, d))
    assert torch.equal(attention_output, before)
