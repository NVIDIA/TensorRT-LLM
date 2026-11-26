# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the spark2_5 gated erf-GELU activation op."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.spark2_5.ops.activation.gelu_erf_and_mul import (
    gelu_erf_and_mul as op,
)

__extra_import_path__ = ["../../.."]  # noqa: F841 -- repo's file-scoped import hook

from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "the gelu_erf_and_mul test requires a CUDA device"


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    gate_up = torch.randn(spec["tokens"], 2 * spec["inter"], generator=g, device="cuda").to(
        spec["dtype"]
    )
    return (gate_up,)


CELLS = certified_cells(op, {})


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by name
    certifying(op)
    (gate_up,) = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(gate_up)
    assert out.shape == (gate_up.shape[0], gate_up.shape[1] // 2)
    assert out.dtype == gate_up.dtype
    assert out.data_ptr() != gate_up.data_ptr(), "the op must return a new tensor"
    torch.testing.assert_close(out, op.reference(gate_up), atol=2e-2, rtol=2e-2)


def test_guard_refuses_an_odd_width() -> None:
    gate_up = torch.randn(4, 13, device="cuda").to(torch.bfloat16)
    with pytest.raises(AssertionError, match="even"):
        with validating(op):
            op(gate_up)


def test_activation_is_erf_not_tanh() -> None:
    """The reference MLP runs ACT2FN['gelu'] (exact erf); the tanh
    approximation differs by more than bf16 noise on the activation scale."""
    torch.manual_seed(0)
    gate_up = torch.randn(64, 2 * 128, device="cuda") * 4.0

    out = op(gate_up)

    gate, up = gate_up.chunk(2, dim=-1)
    erf = torch.nn.functional.gelu(gate) * up
    tanh = torch.nn.functional.gelu(gate, approximate="tanh") * up
    torch.testing.assert_close(out, erf)
    assert not torch.allclose(out, tanh, atol=1e-4, rtol=1e-4)
