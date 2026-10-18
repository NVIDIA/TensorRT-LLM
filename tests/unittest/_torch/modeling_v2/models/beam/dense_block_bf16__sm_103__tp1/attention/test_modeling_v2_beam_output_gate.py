# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the beam_output_gate entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.beam.dense_block_bf16__sm_103__tp1.ops.attention.beam_output_gate import (  # noqa: E501
    beam_output_gate as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "beam_output_gate requires a CUDA device"


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    t, h, d, dtype = spec["tokens"], spec["num_heads"], spec["head_dim"], spec["dtype"]
    attention_output = torch.randn(t, h, d, generator=g, device="cuda").to(dtype)
    gate = (2.0 * torch.randn(t, h, d, generator=g, device="cuda")).to(dtype)
    gate_bias = (0.75 * torch.randn(h, generator=g, device="cuda")).to(dtype)
    return attention_output, gate, gate_bias


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """Bitwise: each activation-dtype step of the op is an fp32 computation rounded once,
    which is what the reference spells out."""
    certifying(op)
    attention_output, gate, gate_bias = _build(
        cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31
    )
    out = op(attention_output, gate, gate_bias)
    assert out.shape == attention_output.shape and out.dtype == attention_output.dtype
    torch.testing.assert_close(
        out, op.reference(attention_output, gate, gate_bias), rtol=0.0, atol=0.0
    )


def test_rounding_between_steps_is_observable() -> None:
    """One fp32 chain rounded once is a different answer on these inputs, so the cell
    gate above distinguishes the source's per-step rounding from a fused gate."""
    attention_output, gate, gate_bias = _build(
        dict(tokens=129, num_heads=48, head_dim=128, dtype=torch.bfloat16), seed=7
    )
    fused = (
        attention_output.float() * torch.sigmoid(gate.float() + gate_bias.float().unsqueeze(-1))
    ).to(torch.bfloat16)
    assert not torch.equal(op(attention_output, gate, gate_bias), fused)


@pytest.mark.parametrize(
    "case,match",
    [
        ("bias_length_one", "gate_bias must be"),
        ("fp32_bias", "share one dtype"),
        ("gate_shape", "must both be"),
    ],
)
def test_guard_refuses_inputs_the_op_answers_wrongly(case, match) -> None:
    attention_output, gate, gate_bias = _build(
        dict(tokens=17, num_heads=48, head_dim=128, dtype=torch.bfloat16), seed=11
    )
    if case == "bias_length_one":
        gate_bias = gate_bias[:1]
    elif case == "fp32_bias":
        gate_bias = gate_bias.float()
    else:
        gate = gate[:, :1]
    with pytest.raises(AssertionError, match=match):
        with validating(op):
            op(attention_output, gate, gate_bias)
