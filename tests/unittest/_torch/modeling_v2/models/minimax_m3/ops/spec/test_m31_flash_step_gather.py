# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the spec/m31_step_gather entry of the MiniMax-M3.1-Flash sm_107 targets (the overlap scheduler's
step-input gathers in one launch)."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.spec.m31_step_gather import (  # noqa: E501
    m3_step_gather as op,
)

__extra_import_path__ = ["../../../..", "../.."]
from _m3_test_inputs import gen, seed_of  # noqa: E402
from _validating import certifying  # noqa: E402, F401

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("the sm_107 targets run on SM107 only", allow_module_level=True)

VOCAB = 200064
#: Rows of the engine's flat buffers (the max batch), and their stale fill.
MAX_ROWS = 8
STALE = -4
# The targets share most cells; each distinct one runs once.
CELLS = list({repr(c): c for c in op.CELLS}.values())


def _case(spec, seed):
    g = gen(seed)
    p, t, d, nslots = spec["p"], spec["t"], spec["d"], spec["slots"]
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
    return (new_tok, next_draft, lens, slots, ppi, p, t, d, ids, spec["ids0"], draft, spec["d0"], pos_off,
            kv_off, spec["b0"])  # fmt: skip


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """Index copies: exact, including the zero-filled offsets and the untouched rest of ids / draft."""
    args = _case(cell.spec, seed_of(cell.why))
    certifying(op)
    ref = op.reference(*args)
    op(*args)
    torch.cuda.synchronize()
    for name, i in (("ids", 8), ("draft", 10), ("pos_off", 12), ("kv_off", 13)):
        assert torch.equal(args[i], ref[name]), name
