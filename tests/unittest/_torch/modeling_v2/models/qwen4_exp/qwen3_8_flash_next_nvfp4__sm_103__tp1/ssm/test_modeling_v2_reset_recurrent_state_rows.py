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
    """`__call__`'s arguments, plus the backing allocations a layout views."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    slots = spec["slots"]
    rec_shape, conv_shape = spec.get("rec_shape", REC_SHAPE), spec.get("conv_shape", CONV_SHAPE)
    rec_dtype = spec.get("rec_dtype", torch.float32)
    layout = spec.get("layout", "plain")

    def randn(*shape, dtype):
        return torch.randn(*shape, generator=g, device=DEV).to(dtype)

    backing = {}
    if layout == "row_padding":
        row = 1
        for d in rec_shape:
            row *= d
        backing["rec"] = randn(slots, 2, row, dtype=rec_dtype)
        rec = backing["rec"][:, 0]
        assert rec.stride() == (2 * row, 1)
    elif layout == "layer_view":
        backing["rec"] = randn(_LAYERS, slots, *rec_shape, dtype=rec_dtype)
        rec = backing["rec"][1]
    else:
        rec = randn(slots, *rec_shape, dtype=rec_dtype)
    conv = None
    if spec.get("conv", True):
        if layout == "layer_view":
            backing["conv"] = randn(_LAYERS, slots, *conv_shape, dtype=torch.bfloat16)
            conv = backing["conv"][1]
        else:
            conv = randn(slots, *conv_shape, dtype=torch.bfloat16)

    if spec["indices"] == "perm":
        indices = torch.randperm(slots, generator=g, device=DEV).to(torch.int32)
        flags = torch.rand(slots, generator=g, device=DEV) < 0.5
    else:
        indices = torch.tensor(
            spec["indices"], dtype=spec.get("index_dtype", torch.int32), device=DEV
        )
        flags = torch.tensor(spec["flags"], dtype=spec.get("flag_dtype", torch.bool), device=DEV)
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
    if spec.get("layout") == "row_padding":
        assert torch.equal(backing["rec"][:, 1], before["rec"][:, 1]), "padding lane written"
    if spec.get("layout") == "layer_view":
        for name in backing:
            for other in (0, 2):
                assert torch.equal(backing[name][other], before[name][other]), (name, other)
    if spec["indices"] == "perm":
        # The cell discriminates "reset slot s" from "reset row i".
        by_position = {i for i, f in enumerate(flags.tolist()) if not f}
        assert reset != by_position, "the permutation failed to separate the two readings"


def test_the_following_kernel_sees_the_zeros() -> None:
    """The launch is ordered on the current stream: no explicit sync is needed."""
    torch.manual_seed(5)
    rec = torch.randn(4, *REC_SHAPE, dtype=torch.float32, device=DEV)
    state_indices = torch.arange(4, dtype=torch.int32, device=DEV)
    has_initial_states = torch.zeros(4, dtype=torch.bool, device=DEV)
    for _ in range(20):
        rec.normal_()
        op(rec, state_indices, has_initial_states)
        assert rec.sum().item() == 0.0


@pytest.mark.parametrize(
    "case,message",
    [
        ("empty_pool", "recurrent state pool must have at least one cache line"),
        ("slot_count_mismatch", "recurrent and convolution pools must have the same slot count"),
        ("flag_shape_mismatch", "has_initial_states and state_indices must have matching shapes"),
        ("recurrent_rows_noncontiguous", "recurrent state rows must be contiguous"),
        ("recurrent_rows_overlapping", "recurrent state rows must not overlap"),
        ("conv_rows_noncontiguous", "convolution state rows must be contiguous"),
        ("conv_rows_overlapping", "convolution state rows must not overlap"),
        ("cpu_pool", "Pointer argument cannot be accessed from Triton"),
        ("cpu_indices", "Pointer argument cannot be accessed from Triton"),
    ],
)
def test_op_rejects_layout_violations_loudly(case: str, message: str) -> None:
    """Why `is_valid` guards only the index tensors: the rest raises itself."""
    torch.manual_seed(6)
    rec = torch.randn(2, 4, dtype=torch.float32, device=DEV)
    conv = torch.randn(2, 4, dtype=torch.bfloat16, device=DEV)
    state_indices = torch.tensor([0], dtype=torch.int32, device=DEV)
    has_initial_states = torch.tensor([False], device=DEV)
    if case == "empty_pool":
        rec, conv = torch.empty(0, 4, dtype=torch.float32, device=DEV), None
    elif case == "slot_count_mismatch":
        conv = torch.randn(3, 4, dtype=torch.bfloat16, device=DEV)
    elif case == "flag_shape_mismatch":
        has_initial_states = torch.tensor([False, True], device=DEV)
    elif case == "recurrent_rows_noncontiguous":
        rec = torch.randn(2, 8, dtype=torch.float32, device=DEV)[:, ::2]
    elif case == "recurrent_rows_overlapping":
        rec = torch.as_strided(torch.randn(6, dtype=torch.float32, device=DEV), (2, 4), (2, 1))
        conv = None
    elif case == "conv_rows_noncontiguous":
        conv = torch.randn(2, 8, dtype=torch.bfloat16, device=DEV)[:, ::2]
    elif case == "conv_rows_overlapping":
        conv = torch.as_strided(torch.randn(6, dtype=torch.bfloat16, device=DEV), (2, 4), (2, 1))
    elif case == "cpu_pool":
        rec, conv = rec.cpu(), None
        state_indices, has_initial_states = state_indices.cpu(), has_initial_states.cpu()
    elif case == "cpu_indices":
        conv = None
        state_indices, has_initial_states = state_indices.cpu(), has_initial_states.cpu()
    with pytest.raises(ValueError, match=message):
        with validating(op):
            op(rec, state_indices, has_initial_states, conv)


def _pools():
    torch.manual_seed(7)
    rec = torch.randn(6, *REC_SHAPE[1:], dtype=torch.float32, device=DEV)
    conv = torch.randn(6, *CONV_SHAPE, dtype=torch.bfloat16, device=DEV)
    return rec, conv


def test_guard_hazard_strided_index_tensors() -> None:
    """Unguarded, a stride-2 view is read through its base pointer: indices
    [0, 1, 2] reset [0, 1], and flags [T, T, T] reset row 1. No error."""
    good_indices = torch.tensor([0, 1, 2], dtype=torch.int32, device=DEV)
    good_flags = torch.zeros(3, dtype=torch.bool, device=DEV)
    strided_indices = torch.tensor([0, 9, 1, 9, 2, 9], dtype=torch.int32, device=DEV)[::2]
    strided_flags = torch.tensor([True, False] * 3, device=DEV)[::2]
    for indices, flags in ((strided_indices, good_flags), (good_indices, strided_flags)):
        rec, conv = _pools()
        # The reference reads the tensors by value, which is what a caller meant.
        want = op.reference(rec, indices, flags, conv)
        got = op(rec, indices, flags, conv)
        with pytest.raises(AssertionError, match="pool differs"):
            _compare(got, want)


def test_guard_hazard_two_d_index_tensors() -> None:
    """Unguarded, a matching [2, 3] pair passes the op's shape check and only
    its first shape[0] = 2 of 6 requests are considered."""
    rec, conv = _pools()
    indices = torch.arange(6, dtype=torch.int32, device=DEV).reshape(2, 3)
    flags = torch.zeros(2, 3, dtype=torch.bool, device=DEV)
    want = op.reference(rec, indices.reshape(-1), flags.reshape(-1), conv)
    got = op(rec, indices, flags, conv)
    with pytest.raises(AssertionError, match="pool differs"):
        _compare(got, want)


@pytest.mark.parametrize("which", ["state_indices", "has_initial_states"])
@pytest.mark.parametrize("how", ["strided", "two_d"])
def test_guard_refuses_index_tensors_the_op_misreads(which: str, how: str) -> None:
    rec, conv = _pools()
    indices = torch.zeros(3, dtype=torch.int32, device=DEV)
    flags = torch.zeros(3, dtype=torch.bool, device=DEV)
    if how == "strided":
        bad_indices = torch.tensor([0, 9, 1, 9, 2, 9], dtype=torch.int32, device=DEV)[::2]
        bad_flags = torch.zeros(6, dtype=torch.bool, device=DEV)[::2]
    else:
        bad_indices = torch.arange(6, dtype=torch.int32, device=DEV).reshape(2, 3)
        bad_flags = torch.zeros(2, 3, dtype=torch.bool, device=DEV)
    if which == "state_indices":
        indices = bad_indices
    else:
        flags = bad_flags
    with pytest.raises(AssertionError, match=f"{which} must be a contiguous 1-D tensor"):
        with validating(op):
            op(rec, indices, flags, conv)
