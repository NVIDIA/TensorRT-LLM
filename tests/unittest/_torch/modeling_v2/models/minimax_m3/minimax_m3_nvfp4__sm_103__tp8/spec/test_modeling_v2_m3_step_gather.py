# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_step_gather entry (the overlap scheduler's step-input gathers in one launch)."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.spec.m3_step_gather import (  # noqa: E501
    m3_step_gather as op,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_spec_inputs import VOCAB, gen, seed_of  # noqa: E402 — needs the path declared above
from _validating import certifying, validating  # noqa: E402, F401

assert torch.cuda.is_available(), "m3_step_gather requires a CUDA device"

#: Rows of the engine's flat buffers (the max batch), and their stale fill.
MAX_ROWS = 8
STALE = -4


def _case(spec, seed):
    g = gen(seed)
    p, t, d, nslots, b0 = spec["p"], spec["t"], spec["d"], spec["slots"], spec["b0"]
    i32 = dict(dtype=torch.int32, device="cuda")
    new_tok = torch.randint(0, VOCAB, (t, nslots, 1), generator=g, device="cuda", dtype=torch.int32)
    if spec.get("two_d"):
        new_tok = new_tok.squeeze(-1)
    next_draft = torch.randint(0, VOCAB, (nslots, d), generator=g, device="cuda", dtype=torch.int32)
    lens = torch.randint(1, t + 1, (nslots,), generator=g, device="cuda", dtype=torch.int32)
    slots = torch.randperm(nslots, generator=g, device="cuda")[:p].to(torch.int32)
    ppi = slots.repeat_interleave(t)
    ids = torch.full((MAX_ROWS * t,), STALE, **i32)
    draft = torch.full((MAX_ROWS * d,), STALE, **i32)
    pos_off = torch.full((MAX_ROWS * t,), STALE, **i32)
    kv_off = torch.full((MAX_ROWS,), STALE, **i32)
    return (
        new_tok,
        next_draft,
        lens,
        slots,
        ppi,
        p,
        t,
        d,
        ids,
        spec["ids0"],
        draft,
        spec["d0"],
        pos_off,
        kv_off,
        b0,
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """Index copies: exact, including the zero-filled offsets and the untouched rest of ids / draft."""
    args = _case(cell.spec, seed_of(cell.why))
    certifying(op)
    ref = op.reference(*args)
    op(*args)
    torch.cuda.synchronize()
    for name, got in (
        ("ids", args[8]),
        ("draft", args[10]),
        ("pos_off", args[12]),
        ("kv_off", args[13]),
    ):
        assert torch.equal(got, ref[name]), name


def test_guard_refuses_a_slot_past_the_store() -> None:
    args = list(_case(op.CELLS[0].spec, 1))
    args[3] = args[3].clone()
    args[3][0] = 8
    with pytest.raises(AssertionError, match="slots must index the store"):
        with validating(op):
            op(*args)
