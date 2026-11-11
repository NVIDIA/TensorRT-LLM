# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the glm5_next fp8 tp4 entry kpool_update."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.attention.kpool_update import (  # noqa: E501
    kpool_update,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 100, reason="the glm5_next fp8 tp4 entries are certified on sm_100"
)

DEVICE = "cuda"
TPB, KPOOL, HD, PAGES = 32, 4, 128, 4100


def _rows(spec: dict):
    """``(positions, request_ids, pages each table needs)`` of a cell.

    Generation cells: request ``b``'s KV length after the step is ``kv - b % 4``; its ``T``
    rows are positions ``kv_b - T .. kv_b - 1`` (``[B, T]``, or ``[B]`` when ``T = 1``) of
    generation table ``b``, whose pools reach back to the first member of the first row's
    pool. Context cells: the pool-final positions each ``(cached, new)`` request completes in
    the step (one row per pool, with request ids), as the module path's context schedule
    lists them; their pools span every page of the request.
    """
    if "context" in spec:
        positions, request_ids, pages = [], [], []
        for i, (cached, new) in enumerate(spec["context"]):
            first_final = cached + (-(cached + 1)) % KPOOL
            finals = list(range(first_final, cached + new, KPOOL))
            positions += finals
            request_ids += [i] * len(finals)
            pages.append(range(0, -(-(cached + new) // TPB)))
        return (
            torch.tensor(positions, dtype=torch.int64, device=DEVICE),
            torch.tensor(request_ids, dtype=torch.int32, device=DEVICE),
            pages,
        )
    requests, tokens = spec["requests"], spec["tokens"]
    kv = spec["kv"] - torch.arange(requests, device=DEVICE) % 4
    positions = (kv[:, None] - tokens + torch.arange(tokens, device=DEVICE)).squeeze(1)
    pages = [
        range((length - tokens) // KPOOL * KPOOL // TPB, (length - 1) // TPB + 1)
        for length in kv.tolist()
    ]
    return positions, None, pages


def _args(spec: dict, seed: int):
    """The op's arguments and the storage behind the pool view.

    Every page a table's rows reach gets its own slot (the other pages point at slot 0, which
    no row reaches). The pool is a coalesced view: the second of two subpages of every slot,
    rows padded by 64 columns, as a layer's indexer pool sits in the cache's pages.
    Generation tables follow one context request's table, so the op reads a row slice of the
    step's tables, like the module path's generation slice.
    """
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    positions, request_ids, pages = _rows(spec)
    lead = 0 if request_ids is not None else 1
    tables = torch.zeros(lead + len(pages), spec.get("pages", PAGES), dtype=torch.int64)
    slots = 1 + sum(len(p) for p in pages)
    order = (1 + torch.randperm(slots - 1, generator=torch.Generator().manual_seed(seed))).tolist()
    for t, table_pages in enumerate(pages, start=lead):
        for page in table_pages:
            tables[t, page] = order.pop()
    tables = tables.to(DEVICE)
    storage = torch.randn(slots, 2, TPB, 3 * HD + 64, generator=g, device=DEVICE)
    storage = storage.to(torch.bfloat16)
    pool = storage[:, 1, :, : 3 * HD]
    ape = (torch.randn(KPOOL, HD, generator=g, device=DEVICE) * 0.5).to(torch.bfloat16)
    args = (pool, tables[lead:], positions, ape, TPB)
    return args, dict(head_dim=HD, kpool=KPOOL, request_ids=request_ids), storage


def _key_scale(pool, tables, positions, request_ids):
    """Per row: the ``(slot, row)`` of its pool's first member and the largest ``|k|`` among the
    pool's members the row sees, the scale of the key it writes."""
    rows_per_table = positions.shape[1] if positions.dim() == 2 else 1
    pos = positions.reshape(-1)
    order = torch.arange(pos.shape[0], device=DEVICE)
    table = order // rows_per_table if request_ids is None else request_ids.long()
    member = pos[:, None] // KPOOL * KPOOL + torch.arange(KPOOL, device=DEVICE)
    slot = tables[table[:, None], (member // TPB).clamp(max=tables.shape[1] - 1)]
    k = pool[slot, member % TPB, :HD].float().abs().amax(dim=-1)
    k = torch.where(member <= pos[:, None], k, 0.0).amax(dim=-1)
    return slot[:, 0], member[:, 0] % TPB, k


@pytest.mark.parametrize("cell", kpool_update.CELLS, ids=[c.why[:44] for c in kpool_update.CELLS])
def test_kpool_update_matches_reference(cell, certifying):  # noqa: F811
    """Only the pool-key columns of refreshed pools' first-member rows change, and each new
    key is within ``2**-6 * K + 2**-7 * |key|`` of the reference's, ``K`` the largest ``|k|``
    the last refreshing row sees in the pool.

    Both sides round the same quantities to BF16 at the same points, but their exponentials
    and sums differ in the last FP32 bits, which can move a probability or a product to the
    neighbouring BF16 value. One step of a probability ``p`` moves a term by at most ``2**-7 *
    p * |k|``, the product's rounding as much again: at most ``2**-6 * sum_m p_m |k_m| <= 2**-6
    * K`` over the members. The key's own rounding can take one more step, ``2**-7 * |key|``.
    """
    certifying(kpool_update)
    args, kwargs, storage = _args(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    before = storage.clone()
    ref_storage = storage.clone()
    assert kpool_update(*args, **kwargs) is None
    kpool_update.reference(ref_storage[:, 1, :, : 3 * HD], *args[1:], **kwargs)
    # Nothing but the pool-key columns of the layer's rows moves, on either side.
    for s in (storage, ref_storage):
        assert torch.equal(s[:, 0], before[:, 0])
        assert torch.equal(s[:, 1, :, : 2 * HD], before[:, 1, :, : 2 * HD])
        assert torch.equal(s[:, 1, :, 3 * HD :], before[:, 1, :, 3 * HD :])
    keys = (s[:, 1, :, 2 * HD : 3 * HD] for s in (storage, ref_storage, before))
    got, want, old = keys
    slot, row, scale = _key_scale(before[:, 1, :, : 3 * HD], *args[1:3], kwargs["request_ids"])
    dest = slot * TPB + row
    written = torch.zeros(got.shape[0] * TPB, dtype=torch.bool, device=DEVICE)
    written[dest] = True
    written = written.view(got.shape[:2])
    assert torch.equal(got[~written], old[~written])
    assert torch.equal(want[~written], old[~written])
    # The last row on a pool sees all the members the others do: the largest scale is its.
    tol = torch.zeros(got.shape[0] * TPB, device=DEVICE)
    tol = tol.scatter_reduce(0, dest, 2**-6 * scale, reduce="amax").view(got.shape[:2])
    err = (got.float() - want.float()).abs()
    band = tol[..., None] + 2**-7 * torch.maximum(got.float().abs(), want.float().abs())
    assert bool((err <= band).all()), f"largest key error {float(err.max()):.3e}"


def test_guard_refuses_two_rows_on_one_pool():
    """Packed context rows run one program each: two rows on one pool race for its key."""
    args, kwargs, _ = _args(dict(context=((0, 64),), pages=8), seed=0)
    pool, tables, _, ape, tpb = args
    positions = torch.tensor([5, 6], dtype=torch.int64, device=DEVICE)
    kwargs["request_ids"] = torch.zeros(2, dtype=torch.int32, device=DEVICE)
    with pytest.raises(AssertionError, match="two rows refresh one pool"):
        with validating(kpool_update):
            kpool_update(pool, tables, positions, ape, tpb, **kwargs)
