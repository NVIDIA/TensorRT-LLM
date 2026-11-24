# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's begin_fold entry: every cell bitwise against the plain-torch
reference (slots, accepted counts and embedding rows), graph padding rows included."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.embedding.rubin_ull_begin_fold import (  # noqa: E501
    begin_fold as op,
)

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM10x GPU",
)

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)

VOCAB = 1000
HIDDEN = 7168
CAPACITY = 12
TOKENS = 8


def _replay(units):
    record = torch.full((CAPACITY, 4), -3, dtype=torch.int32, device="cuda")
    for slot, count in ((3, 0), (5, 7), (0, 2), (9, 6), (CAPACITY - 1, 0)):
        record[slot, 0] = count
    header, slot_units, state_units = units[2], units[0], units[1]
    return SimpleNamespace(
        checkpoints=True,
        checkpoint_slot_units=slot_units,
        checkpoint_state_units=state_units,
        checkpoint_header_units=header,
        num_speculative_tokens=TOKENS - 1,
        prev_num_accepted_tokens=record[:, 0],  # a strided view, as the manager's
    )


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    spec = cell.spec
    gen = torch.Generator().manual_seed(spec["batch"] * 31 + spec["width"])
    table = torch.randn((VOCAB, HIDDEN), generator=gen).to(torch.bfloat16).cuda()
    live = [3, 5, 0, 9][: max(spec["batch"] - 1, 1)]
    state = torch.tensor(
        live + [CAPACITY - 1] * (spec["batch"] - len(live)), dtype=torch.int32, device="cuda"
    )
    rows = spec["batch"] * spec["width"]
    ids = torch.randint(0, VOCAB, (rows,), generator=gen).to(getattr(torch, spec["ids_dtype"]))
    ids[len(live) * spec["width"] :] = 0  # padding rows: the dummy request's ids
    ids[-1] = VOCAB + 3 if rows > len(live) * spec["width"] else ids[-1]  # one outside the table
    ids = ids.cuda()
    replay = _replay(spec["units"])
    with validating(op):
        got = op(replay, state, ids, table)
    torch.cuda.synchronize()
    want = op.reference(replay, state, ids, table)
    for name, g, w in zip(("slots", "accepted", "rows"), got, want):
        assert g.dtype == w.dtype and g.shape == w.shape, name
        assert torch.equal(_bits(g), _bits(w)), f"{name} differs"


def test_is_valid_refuses_inputs_the_fold_does_not_take() -> None:
    table = torch.zeros((VOCAB, HIDDEN), dtype=torch.bfloat16, device="cuda")
    replay = _replay(CELLS[0].spec["units"])
    state = torch.zeros(1, dtype=torch.int32, device="cuda")
    ids = torch.zeros(TOKENS, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="contiguous"):
        op.is_valid(replay, state, ids[None], table)
    with pytest.raises(ValueError, match="contiguous"):
        op.is_valid(replay, state, ids, table.t().contiguous().t())
    with pytest.raises(ValueError, match="checkpoint mode"):
        op.is_valid(SimpleNamespace(**{**vars(replay), "checkpoints": False}), state, ids, table)
