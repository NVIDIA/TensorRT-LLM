# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for this target's fused_sigmoid_gate_mul_add variant
(ops/moe/fused_sigmoid_gate_mul_add__r12).

The entry owns the contract. This file turns a cell into operands, locks both
sides of the in-place write, and holds the claims no cell carries: the empty
batch and the loud rejection of a wrong-length gate.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.moe.fused_sigmoid_gate_mul_add__r12 import (  # noqa: E501
    fused_sigmoid_gate_mul_add as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "fused_sigmoid_gate_mul_add requires a CUDA device"

DEV = "cuda"
HIDDEN = 2560


def _operands(spec, seed):
    """`(final, gate, shared)`: the in-place combine's operands."""
    t, h, dt = spec["tokens"], spec["hidden"], spec["dtype"]
    assert spec["dest"] == "inplace", spec["dest"]
    g = torch.Generator(device=DEV).manual_seed(seed)

    def rand(*shape, dtype=dt):
        return torch.randn(*shape, device=DEV, generator=g).to(dtype)

    final, shared = rand(t, h), rand(t, h)
    gate = rand(t, 1, dtype=spec["gate_dtype"])
    if spec["gate_layout"] == "strided":
        wide = rand(t, 4, dtype=spec["gate_dtype"])
        wide[:, 2:3] = gate
        gate = wide[:, 2:3]
        assert not gate.is_contiguous()
    else:
        assert spec["gate_layout"] == "column", spec["gate_layout"]
    return final, gate, shared


# ── the certified cells ───────────────────────────────────────────────────


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    final, gate, shared = _operands(cell.spec, zlib.crc32(cell.why.encode()))
    before = {k: v.clone() for k, v in dict(final=final, gate=gate, shared=shared).items()}
    ref = op.reference(before["final"], gate, shared)

    out = op(final, gate, shared)

    assert out.data_ptr() == final.data_ptr(), "the op returns what it wrote"
    assert out.dtype == final.dtype and out.shape == final.shape
    torch.testing.assert_close(out, ref)
    assert torch.equal(gate, before["gate"]) and torch.equal(shared, before["shared"])


# ── what a cell cannot carry ──────────────────────────────────────────────


def test_zero_tokens_is_a_no_op():
    final = torch.empty(0, HIDDEN, dtype=torch.bfloat16, device=DEV)
    shared = torch.empty(0, HIDDEN, dtype=torch.bfloat16, device=DEV)
    gate = torch.empty(0, 1, dtype=torch.bfloat16, device=DEV)
    returned = op(final, gate, shared)
    assert returned.shape == (0, HIDDEN) and returned.data_ptr() == final.data_ptr()


def test_a_gate_of_the_wrong_length_raises() -> None:
    """Metadata violations are loud, which is why the entry has no guard."""
    bf = dict(dtype=torch.bfloat16, device=DEV)
    final, shared = torch.randn(4, 128, **bf), torch.randn(4, 128, **bf)
    with pytest.raises(AssertionError):
        op(final, torch.randn(3, 1, **bf), shared)
