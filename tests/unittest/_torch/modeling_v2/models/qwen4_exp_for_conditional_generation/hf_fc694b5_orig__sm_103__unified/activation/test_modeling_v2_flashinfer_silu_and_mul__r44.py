# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_silu_and_mul catalog entry, as this target's local variant r44.

The entry owns what is certified. This file turns a cell's spec into tensors
and drives the inputs the guard exists to refuse.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.activation.flashinfer_silu_and_mul__r44 import (  # noqa: E501
    flashinfer_silu_and_mul as op,
)

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "flashinfer_silu_and_mul requires a CUDA device"


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(spec["shape"], generator=g, device="cuda").to(spec["dtype"])


# The shapes each shipped target calls this entry with, by target directory
# under `models/`. The entry's own `CELLS` are the op's whoever calls it.
TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "deepseek_v3/r1_0528_nvfp4__sm_10x__default": (
        Cell(
            why="deepseek default at attention TP8: the shared expert's gate+up, 2 x 2048/8",
            spec=dict(shape=(16, 512), dtype=torch.bfloat16),
        ),
    ),
}

CELLS = certified_cells(op, TARGET_CELLS)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:40] for c in CELLS])
def test_certified_cells(cell) -> None:
    x = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    with validating(op):
        out = op(x)
    assert out.shape == (*x.shape[:-1], x.shape[-1] // 2)
    assert out.data_ptr() != x.data_ptr(), "the op must return a new tensor"
    torch.testing.assert_close(out, op.reference(x))


@pytest.mark.parametrize(
    "width,match",
    [
        (4095, "must be even"),
        # 24 bf16 elements is 48 bytes per row, which the op's own row check
        # accepts; the half is 24 bytes, so the second half starts mid-vector.
        (24, "16-byte vectors"),
        # The only width that reaches the third assert: for the half to be a
        # whole number of 16-byte vectors *and* smaller than one, it has to be
        # zero. Worth a case anyway -- it is the difference between a guard
        # that is unreachable and one that is merely narrow.
        (0, "at least one 16-byte vector"),
    ],
)
def test_guard_refuses_widths_the_kernel_faults_on(width, match) -> None:
    """A misaligned half kills the CUDA context rather than raising.

    That is why these are `is_valid` and not left to the op: the failure is
    unrecoverable and its message names neither the op nor the shape.
    """
    x = torch.randn(4, width, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match=match):
        with validating(op):
            op(x)
