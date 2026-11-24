# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's dense0_situ_gate_up entry: every cell against the FP64 reference on
operands whose gate/up sums are exact, the padding columns of ``hidden`` and ``grid_out`` untouched."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.gemm import (
    rubin_ull_k3_dense0 as k3_dense0,  # noqa: E501
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_ull_dense0_situ_gate_up import (  # noqa: E501
    BETA,
    INTER,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_ull_dense0_situ_gate_up import (  # noqa: E501
    dense0_situ_gate_up as op,
)

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the split-K CuTe DSL kernel needs an SM10x GPU",
)

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)

HIDDEN = k3_dense0.HIDDEN
PAD = k3_dense0.pad_k(INTER)
POISON = -7.0


def _build(spec, seed):
    """Operands whose gate/up sums are exact in FP32 in any order: integer activations in
    [-2, 2] and weights in [-2, 2] / 64, so every partial sum is a multiple of 2^-6 far inside
    FP32's 24 bits. The kernel and the reference then round the same gate/up values to BF16, and
    the comparison sees only the activation and the output rounding. The products land at a
    standard deviation of ~2.6: the activation's curved range and the soft cap's knee."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = spec["rows"]

    def ints(*shape):
        return torch.randint(-2, 3, shape, generator=g, device="cuda").to(torch.bfloat16)

    x = ints(rows, HIDDEN)
    gate, up = ints(INTER, HIDDEN) / 64, ints(INTER, HIDDEN) / 64
    weight = k3_dense0.fuse_gate_up(gate, up)
    hidden = torch.full((rows, PAD), POISON, dtype=torch.bfloat16, device="cuda")
    grid_out = torch.full((rows, 2 * INTER), POISON, dtype=torch.bfloat16, device="cuda")
    tactic = k3_dense0.SplitKTactic(*spec["tactic"])
    return x, weight, hidden, grid_out, tactic, BETA, spec["linear_beta"]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    x, weight, hidden, grid_out, tactic, beta, linear_beta = args
    with validating(op):
        result = op(*args, True)
    torch.cuda.synchronize()
    assert result is hidden
    ref = op.reference(*args, True).double()
    got = hidden[:, :INTER].double()
    # Same BF16 gate/up on both sides (exact sums): the kernel's FP32 activation (ex2.approx tanh
    # and sigmoid, a few FP32 ulps) rounded to BF16 against the FP64 one rounded to BF16 -- one BF16
    # rounding each, 2^-8 relative apiece.
    err = (got - ref).abs()
    band = 2.0**-7 * ref.abs() + 2.0**-120
    assert bool(torch.isfinite(got).all())
    assert bool((err <= band).all()), f"max err / band {(err / band).max().item():.3f}"
    assert bool((hidden[:, INTER:] == POISON).all()), "wrote the padding columns"
    assert bool((grid_out == POISON).all()), "wrote grid_out"


def test_is_valid_refuses_a_strided_x() -> None:
    args = list(_build(CELLS[0].spec, seed=1))
    wide = torch.zeros(args[0].shape[0], 2 * HIDDEN, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="contiguous"):
        op.is_valid(wide[:, :HIDDEN], *args[1:], True)
