# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the beam_neox_rope entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.beam_neox_rope import (
    beam_neox_rope as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "beam_neox_rope requires a CUDA device"


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    t, d, dtype = spec["tokens"], spec["head_dim"], spec["dtype"]
    q = torch.randn(t, spec["num_heads"], d, generator=g, device="cuda").to(dtype)
    k = torch.randn(t, spec["num_kv_heads"], d, generator=g, device="cuda").to(dtype)
    return q, k, torch.arange(t, device="cuda")


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """Bitwise: the op and its reference make the same fp32 operations in the same order
    and round once, so any difference is a changed formula, not accumulation noise."""
    certifying(op)
    spec = cell.spec
    q, k, positions = _build(spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    q_out, k_out = op(q, k, positions, spec["theta"], spec["factor"])
    q_ref, k_ref = op.reference(q, k, positions, spec["theta"], spec["factor"])
    for out, ref, src in ((q_out, q_ref, q), (k_out, k_ref, k)):
        assert out.shape == src.shape and out.dtype == src.dtype
        assert out.data_ptr() != src.data_ptr(), "the op must return a new tensor"
        torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0)
    if spec["tokens"] == 1:
        assert torch.equal(q_out, q) and torch.equal(k_out, k), "position 0 must rotate by 0"
    else:
        assert not torch.equal(q_out[1:], q[1:]), "positions past 0 must rotate"


def test_halves_rotate_together_not_interleaved_pairs() -> None:
    """A unit vector in dimension 0 at position p goes to (cos p, ..., sin p at D/2, ...):
    NeoX pairs dimension i with i + D/2, not with i + 1."""
    d = 128
    q = torch.zeros(2, 1, d, dtype=torch.float32, device="cuda")
    q[:, 0, 0] = 1.0
    k = q.clone()
    positions = torch.arange(2, device="cuda")
    q_out, _ = op(q, k, positions, 10000.0, 1.0)
    assert q_out[1, 0, 0].item() == pytest.approx(torch.cos(torch.tensor(1.0)).item(), abs=1e-6)
    assert q_out[1, 0, d // 2].item() == pytest.approx(
        torch.sin(torch.tensor(1.0)).item(), abs=1e-6
    )
    assert q_out[1, 0, 1].item() == 0.0


@pytest.mark.parametrize(
    "case,match",
    [
        ("positions_broadcast", "positions must be"),
        ("float_positions", "integer token positions"),
        ("odd_head_dim", "even head_dim"),
    ],
)
def test_guard_refuses_inputs_the_op_answers_wrongly(case, match) -> None:
    q = torch.randn(17, 48, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(17, 8, 128, dtype=torch.bfloat16, device="cuda")
    positions = torch.arange(17, device="cuda")
    if case == "positions_broadcast":
        positions = positions[:1]
    elif case == "float_positions":
        positions = positions.float()
    else:
        q, k = q[..., :127], k[..., :127]
    with pytest.raises(AssertionError, match=match):
        with validating(op):
            op(q, k, positions, 10000.0, 1.0)
