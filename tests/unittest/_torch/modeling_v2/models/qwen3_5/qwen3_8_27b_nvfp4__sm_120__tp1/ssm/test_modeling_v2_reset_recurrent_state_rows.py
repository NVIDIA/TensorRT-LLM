# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the reset_recurrent_state_rows entry."""

import zlib
from typing import Optional

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.ssm.reset_recurrent_state_rows import (  # noqa: E501
    COALESCED_LAYERS,
    CONV_SHAPE,
    REC_SHAPE,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.ssm.reset_recurrent_state_rows import (  # noqa: E501
    reset_recurrent_state_rows as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", ".."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the Qwen3.8-27B sm_120 tp1 entries are certified on sm_120"
)


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


def _build(spec, seed):
    """`__call__`'s arguments, plus the backing allocations a layer view views."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    slots = spec["slots"]
    rec_dtype = spec.get("rec_dtype", torch.float32)
    rec_shape = spec.get("rec_shape", REC_SHAPE)
    layout = spec.get("layout", "plain")

    def randn(*shape, dtype):
        if dtype == torch.int8:  # random bytes: int16 codes and fp32 scale bits alike
            return torch.randint(-128, 128, shape, generator=g, device=DEV, dtype=dtype)
        return torch.randn(*shape, generator=g, device=DEV).to(dtype)

    backing = {}
    if layout == "coalesced":
        # The cache manager v2's view: every GDN layer's state of a slot in one page.
        backing["rec"] = randn(slots, COALESCED_LAYERS, *rec_shape, dtype=rec_dtype)
        backing["conv"] = randn(slots, COALESCED_LAYERS, *CONV_SHAPE, dtype=torch.bfloat16)
        rec, conv = backing["rec"][:, 0], backing["conv"][:, 0]
    else:
        rec = randn(slots, *rec_shape, dtype=rec_dtype)
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
    if spec.get("layout") == "coalesced":
        # The other layers' states between this layer's rows stay bitwise.
        for name in backing:
            assert torch.equal(backing[name][:, 1:], before[name][:, 1:]), name
        assert reset, "the cell resets at least one row"


def _pools():
    torch.manual_seed(7)
    rec = torch.randint(-128, 128, (6, *REC_SHAPE), dtype=torch.int8, device=DEV)
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
