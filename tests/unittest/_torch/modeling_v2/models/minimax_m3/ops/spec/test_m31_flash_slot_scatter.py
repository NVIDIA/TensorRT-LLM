# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the spec/m31_slot_scatter entry of the MiniMax-M3.1-Flash sm_107 targets (the spec sampler's
per-slot stores in one launch)."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.spec.m31_slot_scatter import (  # noqa: E501
    m3_slot_scatter as op,
)

__extra_import_path__ = ["../../../..", "../.."]
from _m3_test_inputs import gen, seed_of  # noqa: E402
from _validating import certifying  # noqa: E402, F401

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("the sm_107 targets run on SM107 only", allow_module_level=True)

VOCAB = 200064
STALE = -4
# Every target certifies the same cells; each distinct one runs once.
CELLS = list({repr(c): c for c in op.CELLS}.values())


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


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """Index copies: exact, zero padding included, other slots untouched."""
    args = _case(cell.spec, seed_of(cell.why))
    certifying(op)
    ref = op.reference(*args)
    op(*args)
    torch.cuda.synchronize()
    for name, got in (("da", args[2]), ("db", args[4]), ("dl", args[6]), ("dd", args[8])):
        assert torch.equal(got, ref[name]), name
