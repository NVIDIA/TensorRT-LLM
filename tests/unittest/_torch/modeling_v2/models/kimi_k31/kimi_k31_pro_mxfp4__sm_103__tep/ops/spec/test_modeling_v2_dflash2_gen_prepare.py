# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the k31_dflash2_gen_prepare entry on a sample of its cells (``sample_cells``): every output and
in-place update bitwise against the torch path it replaces."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.spec.dflash2_gen_prepare import (  # noqa: E501
    k31_dflash2_gen_prepare as op,
)

__extra_import_path__ = ["../../../../..", "../../.."]
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: F401 — needs the path declared above

pytestmark = on_sm(103, reason="the K3.1 DFlash2 gen-prepare entry is certified on sm_103")

VOCAB = 163840
HIDDEN = 7168
PAGE = 32
_EMBED = {}


def _embed():
    if not _EMBED:
        g = torch.Generator(device="cuda").manual_seed(3)
        _EMBED["w"] = torch.randn(VOCAB, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
        _EMBED["mask"] = torch.randn(HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    return _EMBED["w"], _EMBED["mask"]


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    n, width, block = spec["n"], spec["width"], spec["block"]
    num_slots, pages = spec["slots"], spec["pages"]
    ring_rows = pages * PAGE if spec["ring"] else 0
    max_ctx = 1 << 20 if spec["ring"] else pages * PAGE - max(width, block)
    dummy = num_slots - 1
    perm = torch.randperm(num_slots - 1, generator=g, device="cuda")[:n]
    slots = perm.long()
    ctx_hi = max_ctx if spec["ring"] else max_ctx - width
    ctx_len = torch.randint(0, ctx_hi, (num_slots,), generator=g, device="cuda", dtype=torch.long)
    num_accepted = torch.randint(1, width + 1, (n,), generator=g, device="cuda", dtype=torch.int32)
    accepted = torch.randint(0, VOCAB, (n, width), generator=g, device="cuda", dtype=torch.int32)
    # A page table that is not the identity (pages of other slots, shuffled).
    table = torch.randperm(num_slots * pages, generator=g, device="cuda").to(torch.int32)
    table = table.view(num_slots, pages).contiguous()
    if spec["special"] == "edges":
        slots[-2:] = dummy  # two padding rows on the dummy slot
        ctx_len[slots[0]] = max_ctx  # a context at the cap
        num_accepted[1] = 0
        accepted[2, :] = -1
        num_accepted[2] = 3
    seq_lens = torch.zeros(n, dtype=torch.int32, device="cuda")
    w, mask = _embed()
    return (
        [num_accepted, accepted, slots, ctx_len, table, w, mask, seq_lens],
        dict(block=block, max_ctx=max_ctx, dummy_slot=dummy, ring_rows=ring_rows, page_size=PAGE),
    )


def _clone(args):
    return [a.clone() if i in (3, 7) else a for i, a in enumerate(args)]


def _check(args, kw):
    ref_args = _clone(args)
    got = op(*args, **kw)
    expected = op.reference(*ref_args, **kw)
    for name, g_t, e_t in zip(expected._fields, got, expected):
        assert g_t.dtype == e_t.dtype and g_t.shape == e_t.shape, name
        if name == "noise":
            assert torch.equal(g_t.view(torch.int16), e_t.view(torch.int16)), name
        else:
            assert torch.equal(g_t, e_t), name
    assert torch.equal(args[3], ref_args[3]), "ctx_len"
    assert torch.equal(args[7], ref_args[7]), "seq_lens"


CELLS = sample_cells(op.CELLS, size_key="n")[:2]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    args, kw = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    certifying(op)
    _check(args, kw)
