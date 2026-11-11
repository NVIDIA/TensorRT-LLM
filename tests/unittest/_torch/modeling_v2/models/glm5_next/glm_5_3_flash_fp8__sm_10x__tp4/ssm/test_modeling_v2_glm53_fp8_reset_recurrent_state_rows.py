# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ssm/reset_recurrent_state_rows operation of the glm5_next FP8 target.

The pools are views into wider per-slot records, as the hybrid cache manager lays them out, so a write past a
slot's row shows up in the record's padding. The op writes zeros or nothing, so every comparison is exact.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.ssm.reset_recurrent_state_rows import (  # noqa: E501
    reset_recurrent_state_rows as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 -- path above; fixture by name

pytestmark = pytest.mark.skipif(get_sm_version() != 100, reason="certified on sm_100")

#: GLM-5.3-Flash's KDA at TP4: 64 heads over 4 ranks, K = V = 128.
HEADS, HEAD_DIM = 16, 128
STATE_ROW = HEADS * HEAD_DIM * HEAD_DIM
#: Elements past each slot's row in its record (the other layers' rows in the real manager).
SLOT_PAD = 4096

# The resets the shipped target issues before each KDA prefill, by target directory under `models/`.
CELLS: tuple[Cell, ...] = op.CELLS + (
    Cell(
        why="a 13,312-token prompt's prefill: one fresh request on the default bf16 pool",
        spec=dict(
            slots=8,
            indices=(5,),
            has_init=(False,),
            state_dtype=torch.bfloat16,
        ),
    ),
    Cell(
        why="a chunked-prefill step: the continuing chunk keeps its row, two fresh prompts are zeroed",
        spec=dict(
            slots=8,
            indices=(6, 1, 4),
            has_init=(True, False, False),
            state_dtype=torch.bfloat16,
        ),
    ),
    Cell(
        why="the same step on an fp32 pool (kv_cache_config.mamba_ssm_cache_dtype float32)",
        spec=dict(
            slots=8,
            indices=(6, 1, 4),
            has_init=(True, False, False),
            state_dtype=torch.float32,
        ),
    ),
)


def _build(spec: dict, seed: int) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """``(records, args)``: the per-slot records, nonzero everywhere, and the op's arguments viewing them."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    slots = spec["slots"]
    records = torch.randn(slots, STATE_ROW + SLOT_PAD, generator=gen, device="cuda").to(
        spec["state_dtype"]
    )
    states = records[:, :STATE_ROW].view(slots, HEADS, HEAD_DIM, HEAD_DIM)
    indices = torch.tensor(
        spec["indices"], dtype=spec.get("index_dtype", torch.int32), device="cuda"
    )
    has_init = torch.tensor(spec["has_init"], dtype=torch.bool, device="cuda")
    return records, (states, indices, has_init)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 -- pytest resolves the fixture by this name
    """The fresh rows are zero, every other row and all record padding bit-identical (the op writes 0.0 or
    nothing, so there is no band)."""
    certifying(op)
    records, args = _build(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    before = records.clone()
    want_states = op.reference(*args)
    assert op(*args) is None
    assert torch.equal(args[0], want_states)
    assert torch.equal(records[:, STATE_ROW:], before[:, STATE_ROW:]), (
        "the op wrote past a slot's row"
    )


def test_empty_batch_launches_nothing() -> None:
    """No request, no launch: the pools are untouched (a zero-row grid is not launched at all)."""
    records, args = _build(
        dict(slots=4, indices=(), has_init=(), state_dtype=torch.float32), seed=3
    )
    before = records.clone()
    op(*args)
    assert torch.equal(records, before)


def test_guard_refuses_a_flag_per_slot() -> None:
    """``has_initial_states`` is read by request index: a per-slot flag vector of another length is taken and
    zeroes rows by the wrong requests' flags."""
    _, (states, indices, _) = _build(
        dict(slots=8, indices=(6, 1), has_init=(True, False), state_dtype=torch.float32), seed=4
    )
    per_slot = torch.zeros(8, dtype=torch.bool, device="cuda")
    with pytest.raises(AssertionError, match="one flag per request"):
        with validating(op):
            op(states, indices, per_slot)


def test_guard_refuses_overlapping_rows() -> None:
    """A pool view whose slot stride is shorter than its row makes zeroing one slot clear part of the next."""
    flat = torch.randn(4 * 1024, device="cuda")
    overlapping = flat.as_strided((4, 2, 1024), (512, 1024, 1))
    indices = torch.tensor([1], dtype=torch.int32, device="cuda")
    has_init = torch.tensor([False], device="cuda")
    with pytest.raises(AssertionError, match="must not overlap"):
        with validating(op):
            op(overlapping, indices, has_init)
