# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the reset_recurrent_state_rows entry."""

import zlib
from typing import Optional

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.reset_recurrent_state_rows import (  # noqa: E501
    CONV_SHAPE,
    REC_SHAPE,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.reset_recurrent_state_rows import (  # noqa: E501
    reset_recurrent_state_rows as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "reset_recurrent_state_rows requires a CUDA device"


def _compare(
    out: tuple[torch.Tensor, Optional[torch.Tensor]],
    ref: tuple[torch.Tensor, Optional[torch.Tensor]],
) -> None:
    """Bitwise, both pools whole: every element is its old value or +0.0,
    so a tolerance could only hide a wrong row -- and comparing the whole
    pool is the lock on every slot the batch did not name. A zeroed row
    must be +0.0 by bit pattern; `torch.equal` alone would accept -0.0."""
    for got, want in zip(out, ref):
        assert (got is None) == (want is None)
        if got is None:
            continue
        assert got.dtype == want.dtype and got.shape == want.shape
        assert torch.equal(got, want), "pool differs from the reference"
        zero = want.reshape(want.shape[0], -1).eq(0).all(dim=1)
        if zero.any():
            bits = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[got.element_size()]
            assert (got[zero].contiguous().view(bits) == 0).all(), "a reset row is not +0.0"


DEV = "cuda"
_LAYERS = 3


def _build(spec, seed):
    """`__call__`'s arguments, plus the backing allocations a layer view views."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    slots = spec["slots"]
    rec_dtype = spec.get("rec_dtype", torch.float32)
    layer_view = spec.get("layout", "plain") == "layer_view"

    def randn(*shape, dtype):
        return torch.randn(*shape, generator=g, device=DEV).to(dtype)

    backing = {}
    if layer_view:
        backing["rec"] = randn(_LAYERS, slots, *REC_SHAPE, dtype=rec_dtype)
        backing["conv"] = randn(_LAYERS, slots, *CONV_SHAPE, dtype=torch.bfloat16)
        rec, conv = backing["rec"][1], backing["conv"][1]
    else:
        rec = randn(slots, *REC_SHAPE, dtype=rec_dtype)
        conv = randn(slots, *CONV_SHAPE, dtype=torch.bfloat16)

    indices = torch.tensor(spec["indices"], dtype=torch.int32, device=DEV)
    flags = torch.tensor(spec["flags"], dtype=torch.bool, device=DEV)
    return [rec, indices, flags, conv], backing


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    args, backing = _build(spec, seed=zlib.crc32(cell.why.encode()))
    rec, indices, flags, conv = args
    before = {name: t.clone() for name, t in backing.items()}
    indices_before, flags_before = indices.clone(), flags.clone()
    ref = op.reference(*args)

    out = op(*args)

    assert out[0] is rec and out[1] is conv, "returns the pools it was handed"
    _compare(out, ref)
    assert torch.equal(indices, indices_before) and torch.equal(flags, flags_before)

    named = {s for s in indices.tolist() if 0 <= s < spec["slots"]}
    reset = {s for s, f in zip(indices.tolist(), flags.tolist()) if not f and s in named}
    if spec["slots"] > 64:
        # Non-vacuity: the engine-width cells reset at least one row, so the
        # whole-pool compare is also the lock on the ~430 unnamed slots.
        assert reset
    if spec.get("layout") == "layer_view":
        for name in backing:
            for other in (0, 2):
                assert torch.equal(backing[name][other], before[name][other]), (name, other)


def _pools():
    torch.manual_seed(7)
    rec = torch.randn(6, *REC_SHAPE[1:], dtype=torch.float32, device=DEV)
    conv = torch.randn(6, *CONV_SHAPE, dtype=torch.bfloat16, device=DEV)
    return rec, conv


def test_guard_refuses_index_tensors_the_op_misreads() -> None:
    """Unguarded, a stride-2 `state_indices` view is read through its base pointer
    as a flat buffer: indices [0, 1, 2] arrive as [0, 9, 1], so rows 0 and 1 are reset
    and row 2 is not (measured). The guard refuses it."""
    rec, conv = _pools()
    indices = torch.tensor([0, 9, 1, 9, 2, 9], dtype=torch.int32, device=DEV)[::2]
    flags = torch.zeros(3, dtype=torch.bool, device=DEV)
    with pytest.raises(AssertionError, match="state_indices must be a contiguous 1-D tensor"):
        with validating(op):
            op(rec, indices, flags, conv)
