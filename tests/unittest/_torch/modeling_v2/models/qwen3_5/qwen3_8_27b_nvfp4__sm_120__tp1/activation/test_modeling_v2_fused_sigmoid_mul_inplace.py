# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fused_sigmoid_mul_inplace entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.activation.fused_sigmoid_mul_inplace import (  # noqa: E501
    fused_sigmoid_mul_inplace as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", ".."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the Qwen3.8-27B sm_120 tp1 entries are certified on sm_120"
)


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def _build(spec, seed):
    """`(out, gate_buffer, gate)` for one spec: `gate` is the 3-D gate view of the
    interleaved `[T, 14336]` qkv projection in `gate_buffer`."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    t, heads, d, dtype = spec["tokens"], spec["heads"], spec["head_dim"], spec["dtype"]
    hidden = heads * d

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda").to(dtype)

    out = randn(t, hidden)
    assert spec["gate"] == "qkv_view", spec["gate"]
    # [q0, g0, q1, g1, ..., K, V]: 24 interleaved q/gate heads of 256, then 4 K and
    # 4 V heads -- the gate the g_i halves, strides (14336, 512, 1).
    gate_buffer = randn(t, 2 * hidden + 2 * 4 * d)
    gate = gate_buffer[:, : 2 * hidden].unflatten(1, (heads, 2 * d))[:, :, d:]
    # A size-1 dimension's stride is never used, and torch reports any value for it.
    assert gate.stride()[1:] == (2 * d, 1)
    assert t == 1 or gate.stride(0) == 2 * hidden + 8 * d
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


def test_cuda_graph_replay_reads_the_operands_at_replay() -> None:
    """The target runs the gate inside its decode graph: a replay must read the gate
    view and the attention output as they are when it replays."""
    spec = next(c.spec for c in op.CELLS if c.spec["gate"] == "qkv_view" and c.spec["tokens"] > 1)
    out, gate_buffer, gate = _build(spec, seed=_seed("graph"))
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(out.clone(), gate)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(out, gate)
    for seed in ("replay 1", "replay 2"):
        fresh_out, fresh_buffer, _ = _build(spec, seed=_seed(seed))
        out.copy_(fresh_out)
        gate_buffer.copy_(fresh_buffer)
        expected = op.reference(out, gate)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected)
