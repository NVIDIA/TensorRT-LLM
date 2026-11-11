# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the glm5_next fp8 tp4 entry kpool_expand."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.attention.kpool_expand import (  # noqa: E501
    kpool_expand,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 100, reason="the glm5_next fp8 tp4 entries are certified on sm_100"
)

DEVICE = "cuda"
TPB, KPOOL, SELECT_K, OUT_WIDTH, PAGES = 32, 4, 512, 2112, 4100
# A coalesced latent pool: this layer's 32 rows of a slot start at row 32 of 96.
BASE_ROW, ROWS_PER_SLOT = 32, 96


def _schedule(spec: dict):
    """``(visible int64 [N], request_ids int32 [N] or None, tables)`` of a cell.

    Generation cells: request ``b``'s KV length after the step is ``kv - b % 4``. With ``T > 1``
    its rows are positions ``kv_b - T .. kv_b - 1`` and read table ``1 + b`` by request id (row
    0 is a context request's table, as in a mixed step); with ``T = 1`` row ``b`` reads table
    ``b`` of the generation tables, without request ids. Context cells: the packed rows of each
    ``(cached, new)`` request, restricted to ``rows``, reading their request's table.
    """
    if "context" in spec:
        visible, request_ids = [], []
        for i, (cached, new) in enumerate(spec["context"]):
            visible.append(torch.arange(cached + 1, cached + new + 1))
            request_ids.append(torch.full((new,), i, dtype=torch.int32))
        lo, hi = spec["rows"]
        visible = torch.cat(visible)[lo:hi].to(DEVICE)
        return visible, torch.cat(request_ids)[lo:hi].to(DEVICE), len(spec["context"])
    requests, tokens = spec["requests"], spec["tokens"]
    kv = spec["kv"] - torch.arange(requests, device=DEVICE) % 4
    if tokens == 1:
        return kv, None, requests
    visible = (kv[:, None] - tokens + 1 + torch.arange(tokens, device=DEVICE)).reshape(-1)
    request_ids = (1 + torch.arange(requests, device=DEVICE, dtype=torch.int32)).repeat_interleave(
        tokens
    )
    return visible, request_ids, 1 + requests


def _top_k_like(visible: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Selections shaped like the decode top-k's: a row with at most ``SELECT_K`` candidate
    pools lists them in order then ``-1``; a longer row lists ``SELECT_K`` distinct candidates
    in random order."""
    num_cand = visible // KPOOL
    width = max(int(num_cand.max()), SELECT_K)
    keys = torch.rand(visible.shape[0], width, generator=g, device=DEVICE)
    keys = torch.where(torch.arange(width, device=DEVICE) < num_cand[:, None], keys, -1.0)
    selected = keys.topk(SELECT_K, dim=1).indices.to(torch.int32)
    ident = torch.arange(SELECT_K, device=DEVICE, dtype=torch.int32)
    short = num_cand <= SELECT_K
    selected[short] = torch.where(ident < num_cand[short, None], ident, -1).to(torch.int32)
    return selected


def _args(spec: dict, seed: int):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    visible, request_ids, num_tables = _schedule(spec)
    pages = spec.get("pages", PAGES)
    tables = torch.randint(0, 1 << 20, (num_tables, pages), generator=g, device=DEVICE)
    selected = _top_k_like(visible, g)
    # The kernel's own validity mask: a -1 and an id past the candidates in a long row.
    long_rows = (visible // KPOOL > SELECT_K).nonzero().flatten()
    if long_rows.numel():
        r = int(long_rows[0])
        selected[r, 5] = int(visible[r] // KPOOL) + 7
        selected[r, 9] = -1
    args = (selected, visible, tables, TPB)
    kwargs = dict(
        base_row=BASE_ROW,
        rows_per_slot=ROWS_PER_SLOT,
        kpool=KPOOL,
        out_width=OUT_WIDTH,
        request_ids=request_ids,
    )
    return args, kwargs


@pytest.mark.parametrize("cell", kpool_expand.CELLS, ids=[c.why[:44] for c in kpool_expand.CELLS])
def test_kpool_expand_matches_reference(cell, certifying):  # noqa: F811
    """The row ids equal the reference's exactly (integer arithmetic): members of every valid
    selected pool, the tail, -1 for invalid pools and the padding."""
    certifying(kpool_expand)
    args, kwargs = _args(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    out = kpool_expand(*args, **kwargs)
    ref = kpool_expand.reference(*args, **kwargs)
    assert out.shape == ref.shape == (args[0].shape[0], OUT_WIDTH)
    assert torch.equal(out, ref)


def test_guard_refuses_visible_lengths_past_the_table():
    """A visible position past a table's last page reads the next table's pages as its own."""
    selected = torch.full((1, SELECT_K), -1, dtype=torch.int32, device=DEVICE)
    tables = torch.zeros(2, 4, dtype=torch.int64, device=DEVICE)
    kv_lens = torch.tensor([4 * TPB + 3], device=DEVICE)
    with pytest.raises(AssertionError, match="exceeds"):
        with validating(kpool_expand):
            kpool_expand(
                selected,
                kv_lens,
                tables,
                TPB,
                base_row=0,
                rows_per_slot=TPB,
                kpool=KPOOL,
                out_width=OUT_WIDTH,
            )
