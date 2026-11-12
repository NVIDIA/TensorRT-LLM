# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_slot_scatter entry (the spec sampler's per-slot stores in one launch)."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.spec.m3_slot_scatter import (  # noqa: E501
    m3_slot_scatter as op,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_spec_inputs import VOCAB, gen, seed_of  # noqa: E402 — needs the path declared above
from _validating import certifying, validating  # noqa: E402, F401

assert torch.cuda.is_available(), "m3_slot_scatter requires a CUDA device"

STALE = -4


def _case(spec, seed):
    """Worker outputs and the store's transposed [slots, width] views of its [width, slots, 1] buffers."""
    g = gen(seed)
    n, nslots = spec["n"], spec["slots"]

    def rnd(shape):
        return torch.randint(0, VOCAB, shape, generator=g, device="cuda", dtype=torch.int32)

    store_new = torch.full((spec["store_w"], nslots, 1), STALE, dtype=torch.int32, device="cuda")
    store_next = torch.full((spec["store_w"], nslots, 1), STALE, dtype=torch.int32, device="cuda")
    store_lens = torch.full((nslots,), STALE, dtype=torch.int32, device="cuda")
    store_draft = torch.full((nslots, spec["store_d"]), STALE, dtype=torch.int32, device="cuda")
    slots = torch.randperm(nslots, generator=g, device="cuda")[:n]
    lens = torch.randint(1, 5, (n,), generator=g, device="cuda", dtype=torch.int32)
    return (slots, rnd((n, spec["src_w"])), store_new.squeeze(-1).T, rnd((n, spec["src_w"])),
            store_next.squeeze(-1).T, lens, store_lens, rnd((n, spec["src_d"])), store_draft)  # fmt: skip


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """Index copies: exact, zero padding included, other slots untouched."""
    args = _case(cell.spec, seed_of(cell.why))
    certifying(op)
    ref = op.reference(*args)
    op(*args)
    torch.cuda.synchronize()
    for name, got in (("da", args[2]), ("db", args[4]), ("dl", args[6]), ("dd", args[8])):
        assert torch.equal(got, ref[name]), name


def test_guard_refuses_repeated_slots() -> None:
    args = list(_case(op.CELLS[0].spec, 1))
    args[0] = args[0].clone()
    args[0][1] = args[0][0]
    with pytest.raises(AssertionError, match="distinct"):
        with validating(op):
            op(*args)
