# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The MLA latent of a cep group under distributed KV (DKV): owned-row writes and a Top-K read.

Under DKV (``KVCacheManagerV2(dkv=True)``) the N ranks of a cep group store each
request's latent once in the ``page_split`` layout: global page ``g`` (``tokens_per_block`` tokens)
lives on rank ``g % N``, in
that rank's slot of DKV block ``g // N``. Every rank still computes every new token's latent
(the row all-gather), so:

* Writes: a rank appends only the new tokens it owns, at their rank-local positions
  (``Dkv.local_len``); a chunk's owned tokens are contiguous there, so the paged append runs on
  rank-local lengths over the rank's own block table. With owner writes (``owner_write``) the rows
  come to the owner from the ranks that computed them (one push per layer) instead of from a
  row all-gather, and after a fence each reader copies the new rows of its requests from the
  owners' pools into its read buffer's head (``rubin_dkv_rows_read``).
* Reads: each DSA layer's sparse MLA reads a local buffer. Its head holds the step's new tokens in
  pages (every rank appends every new token there, as a replicated cache would hold them); behind
  them, the history rows the layer's Top-K of this rank's rows selects, copied once each from their
  owners through the peer mapping (``rubin_dkv_topk_read``). The Top-K is remapped to rows of the
  buffer. In-kernel reads of 576-byte rows from other ranks ran the sparse MLA ~19x slower in our
  measurements, hence the local copy; only the selected rows travel.

``dkv_plan`` builds a step's plan once: the writes, the new-token pages, and per history token of
the requests with rows on this rank its row in the peer mapping (``hist_src``, on the device).
"""

from __future__ import annotations

import itertools
from typing import Any, Mapping, NamedTuple, Optional, Sequence

import numpy as np
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.dkv import peer_region_tensor


class DkvPlan(NamedTuple):
    """One step's DKV work for the context requests of the batch."""

    # Writes: rows of the step's new tokens this rank owns, and their rank-local lengths.
    owned_rows: torch.Tensor  # [owned] int64, rows of the step's [num_tokens] latent
    cu_cached: torch.Tensor  # [num_contexts + 1] int64, rank-local cached tokens
    cu_kv: torch.Tensor  # [num_contexts + 1] int64, rank-local cached + new tokens
    max_new: int  # the longest request's owned new tokens
    # Reads. The new tokens: each context request's pages of its new tokens, the head of a layer's
    # read buffer (page 0 stands for the history blocks: nothing reads it).
    pages: int  # new-token pages, page 0 included
    table: torch.Tensor  # [num_contexts, width] int32, the new-token page of each request page
    offsets: torch.Tensor  # [1, num_contexts, 2, width] int32, the table as block offsets
    # The history the sparse MLA may read: the cached tokens of the requests with rows on this
    # rank (``hist_len`` 0 for the others), history token ``hist_start[r] + p`` for position p.
    hist_start: torch.Tensor  # [num_contexts] int32
    hist_len: torch.Tensor  # [num_contexts] int32
    hist_src: torch.Tensor  # [history tokens] int64: row in the peer mapping, layer offset 0


def dkv_plan(
    dkv: Any,
    peer_pages: Sequence[Sequence[Sequence[int]]],
    cached: Sequence[int],
    new: Sequence[int],
    rows_per_rank: int,
    page_scale: int,
    device: Any,
    needed: Optional[Sequence[bool]] = None,
    width: int = 1,
) -> DkvPlan:
    """Plan a step's owned-row writes and Top-K reads.

    Only the requests with rows on this rank (``needed``; default all) get a history: the
    contiguous row split keeps a request on as few ranks as possible, and a rank never reads the
    history of a request it computes no row of. Every request's new tokens get pages (the append
    writes every new token of the step).

    Args:
        DKV: The group (``Dkv``).
        peer_pages: Per context request, every rank's slot ids by DKV block
            (``Dkv.peer_pages``).
        cached: Per context request, its cached tokens before the step.
        new: Per context request, its new tokens of the step.
        rows_per_rank: Pages of one layer in one rank's copy in the peer mapping.
        page_scale: The latent pool's pages per slot (its coalesced layers).
        device: Where the plan's tensors live.
        needed: Per context request, whether this rank computes rows of it this step.
        width: The new-token table's columns, at least (the Top-K remap kernel specializes on
            the table width, so a fixed width avoids loading a kernel per layer for each new one).
    """
    tpb, size, rank = dkv.tokens_per_block, dkv.size, dkv.rank
    owned, cu_cached, cu_kv, max_new, row = [], [0], [0], 0, 0
    for c, n in zip(cached, new):
        position = c
        while position < c + n:
            end = min((position // tpb + 1) * tpb, c + n)
            if (position // tpb) % size == rank:
                owned.extend(range(row + position - c, row + end - c))
            position = end
        lo, hi = dkv.local_len(c), dkv.local_len(c + n)
        cu_cached.append(cu_cached[-1] + lo)
        cu_kv.append(cu_kv[-1] + hi)
        max_new = max(max_new, hi - lo)
        row += n
    table, first = new_token_table(cached, new, tpb, width)
    blocks, width = table.shape
    offsets = table.view(1, blocks, 1, width).expand(1, blocks, 2, width).contiguous()

    lens = [c if needed is None or needed[r] else 0 for r, c in enumerate(cached)]
    starts = [0, *itertools.accumulate(lens)][: len(lens)]

    def put(values, dtype):
        return torch.tensor(values, dtype=dtype).to(device, non_blocking=True)

    return DkvPlan(
        owned_rows=put(owned, torch.int64),
        cu_cached=put(cu_cached, torch.int64),
        cu_kv=put(cu_kv, torch.int64),
        max_new=max_new,
        pages=first,
        table=table.to(device, non_blocking=True),
        offsets=offsets.to(device, non_blocking=True),
        hist_start=put(starts, torch.int32),
        hist_len=put(lens, torch.int32),
        hist_src=_history_rows(
            peer_pages, lens, starts, tpb, size, rows_per_rank, page_scale, device
        ),
    )


def new_token_table(
    cached: Sequence[int], new: Sequence[int], tpb: int, width: int = 1
) -> tuple[torch.Tensor, int]:
    """The read buffer's new-token pages (host int32 ``[num_contexts, width]``: each request page's
    page of the buffer, 0 for the pages before the step's) and the pages in all (page 0 included)."""
    blocks = [-(-(c + n) // tpb) for c, n in zip(cached, new)]
    width = max(width, max(blocks, default=1))
    table = torch.zeros(len(blocks), width, dtype=torch.int32)
    first = 1  # page 0: the history blocks
    for r, (c, b) in enumerate(zip(cached, blocks)):
        start = c // tpb
        table[r, start:b] = torch.arange(first, first + b - start, dtype=torch.int32)
        first += b - start
    return table, first


def _history_rows(
    peer_pages: Sequence[Sequence[Sequence[int]]],
    lens: Sequence[int],
    starts: Sequence[int],
    tpb: int,
    size: int,
    rows_per_rank: int,
    page_scale: int,
    device: Any,
) -> torch.Tensor:
    """Each history token's latent row in the peer mapping at layer offset 0 (int64, on the
    device): position ``p`` of request ``r`` is on rank ``o = (p // tpb) % N``, in that rank's slot
    ``s`` of DKV block ``p // (tpb N)``, row ``p % tpb``: row ``(o * rows_per_rank + s *
    page_scale) * tpb + p % tpb``."""
    total = sum(lens)
    if total == 0:
        return torch.empty(0, dtype=torch.int64, device=device)
    reads = [r for r, n in enumerate(lens) if n]
    width = max(len(peer_pages[r][o]) for r in reads for o in range(size))
    slots = np.zeros((len(lens), size, width), dtype=np.int64)
    for r in reads:
        for o, ids in enumerate(peer_pages[r]):
            slots[r, o, : len(ids)] = ids
    slots = torch.from_numpy(slots).to(device, non_blocking=True)
    count = torch.tensor(lens, dtype=torch.int64).to(device, non_blocking=True)
    first = torch.tensor(starts, dtype=torch.int64).to(device, non_blocking=True)
    req = torch.repeat_interleave(torch.arange(len(lens), device=device), count, output_size=total)
    pos = torch.arange(total, dtype=torch.int64, device=device) - first[req]
    page = pos // tpb
    owner = page % size
    slot = slots[req, owner, page // size]
    return (owner * rows_per_rank + slot * page_scale) * tpb + pos % tpb


class OwnerWrite(NamedTuple):
    """One step's owner writes: this rank's new rows (its contiguous rows ``[begin, end)`` of the
    step) to the owners of their pages, the owners' appends, and the new rows a reader copies.

    ``send_order`` permutes this rank's live rows so that each owner's rows are contiguous, in
    owner order (and step row order within an owner); owner ``d`` gets them at row ``d0`` of its
    received rows, which hold its owned new tokens in step row order (``DkvPlan.owned_rows``).
    """

    send_order: torch.Tensor  # [live] int64: local rows, owner-major
    blocks: tuple  # per owner: (s0, d0, n) of the push, rows of send_order / received rows
    recv: tuple  # per rank: its owned new tokens (rows it receives)
    # The new tokens a reader copies from their owners into its read buffer's head: every new
    # position of the requests with rows on this rank (the head then holds what the all-gather's
    # append wrote there). Row in the peer mapping at layer offset 0 (``new_src``), head row
    # (``DkvPlan.table`` pages, ``new_dst``), the identity selection and its count (device).
    new_src: torch.Tensor  # [copied] int64
    new_dst: torch.Tensor  # [copied] int64
    new_pos: torch.Tensor  # [copied] int64: arange
    new_count: torch.Tensor  # [1] int32


def row_owners(cached: Sequence[int], new: Sequence[int], tpb: int, size: int) -> np.ndarray:
    """The owning rank of each of the step's new tokens, in step row order (int64 numpy):
    position ``p`` of a request is on rank ``(p // tpb) % size`` (``page_split``)."""
    parts = [
        ((np.arange(c, c + n, dtype=np.int64) // tpb) % size) for c, n in zip(cached, new) if n
    ]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)


def owner_write(
    dkv: Any,
    peer_pages: Sequence[Sequence[Sequence[int]]],
    cached: Sequence[int],
    new: Sequence[int],
    rows: int,
    needed: Sequence[bool],
    rows_per_rank: int,
    page_scale: int,
    device: Any,
    width: int = 1,
) -> OwnerWrite:
    """Plan a step's owner writes from this rank (``dkv.rank``) of a cep group whose ranks hold
    ``rows`` rows each of the step (rank ``r``: ``[r * rows, (r + 1) * rows)``), and the new rows it
    reads back from their owners into its read buffer's head (``new_token_table`` of ``width``, as
    ``dkv_plan``; ``needed``: per context request, whether this rank holds rows of it)."""
    tpb, size, rank = dkv.tokens_per_block, dkv.size, dkv.rank
    owners = row_owners(cached, new, tpb, size)
    total = owners.shape[0]
    begin, end = min(rank * rows, total), min((rank + 1) * rows, total)
    # counts[s, d]: rows of rank s owned by rank d.
    counts = np.zeros((size, size), dtype=np.int64)
    for s in range(size):
        lo, hi = min(s * rows, total), min((s + 1) * rows, total)
        counts[s] = np.bincount(owners[lo:hi], minlength=size)
    order = np.argsort(owners[begin:end], kind="stable")
    s0 = np.concatenate(([0], np.cumsum(counts[rank])[:-1]))
    d0 = counts[:rank].sum(axis=0)

    host_table = new_token_table(cached, new, tpb, width)[0].numpy()
    dst, src = [], []
    for r, (c, n) in enumerate(zip(cached, new)):
        if n and needed[r]:
            pos = np.arange(c, c + n, dtype=np.int64)
            page = pos // tpb
            dst.append(host_table[r, page].astype(np.int64) * tpb + pos % tpb)
            slots = np.zeros((size, max(len(t) for t in peer_pages[r])), dtype=np.int64)
            for o, ids in enumerate(peer_pages[r]):
                slots[o, : len(ids)] = ids
            owner = page % size
            src.append(
                (owner * rows_per_rank + slots[owner, page // size] * page_scale) * tpb + pos % tpb
            )
    src = np.concatenate(src) if src else np.zeros(0, dtype=np.int64)
    dst = np.concatenate(dst) if dst else np.zeros(0, dtype=np.int64)

    def put(values, dtype=torch.int64):
        return torch.from_numpy(np.asarray(values)).to(dtype).to(device, non_blocking=True)

    return OwnerWrite(
        send_order=put(order.astype(np.int64)),
        blocks=tuple((int(a), int(b), int(n)) for a, b, n in zip(s0, d0, counts[rank])),
        recv=tuple(int(x) for x in counts.sum(axis=0)),
        new_src=put(src),
        new_dst=put(dst),
        new_pos=put(np.arange(src.shape[0], dtype=np.int64)),
        new_count=put(np.array([src.shape[0]]), torch.int32),
    )


class IndexSlots(NamedTuple):
    """Sharded index keys: a step's byte slots (the paged index-key layout: a page holds its tokens'
    FP8 keys, then their FP32 scales)."""

    insert_fp8: torch.Tensor  # [owned] int64: the owned new tokens' keys, in this rank's layer view
    insert_scale: torch.Tensor  # [owned] int64: their scales
    full_fp8: torch.Tensor  # [sum of kv lengths] int64: every token's key, in the mirror view
    full_scale: torch.Tensor  # [sum of kv lengths] int64


def index_slots(
    dkv: Any,
    peer_pages: Sequence[Sequence[Sequence[int]]],
    cached: Sequence[int],
    new: Sequence[int],
    slot_bytes: int,
    rank_bytes: int,
    head_dim: int,
    device: Any,
) -> IndexSlots:
    """The byte slots of sharded index keys for a step's context requests.

    A token at global position ``p`` lives on rank ``o = (p // tpb) % N``, in that rank's slot ``s``
    of DKV block ``p // (tpb * N)``, row ``p % tpb``: its key at ``s * slot_bytes + row *
    head_dim`` of a layer's view and its scale at ``s * slot_bytes + tpb * head_dim + row * 4``.
    Inserts address this rank's own view (the owned new tokens, in step row order, as
    ``DkvPlan.owned_rows``); the full-KV slots address a view of every rank's copy of the layer
    (``o * rank_bytes`` further), in the order of the step's full-KV mapping (each context request's
    tokens ``[0, cached + new)``).
    """
    tpb, size, rank = dkv.tokens_per_block, dkv.size, dkv.rank
    scale_base = tpb * head_dim
    ins_fp8, ins_scale = [], []
    for r, (c, n) in enumerate(zip(cached, new)):
        mine = peer_pages[r][rank]
        for p in range(c, c + n):
            if (p // tpb) % size == rank:
                base = mine[p // (tpb * size)] * slot_bytes
                ins_fp8.append(base + (p % tpb) * head_dim)
                ins_scale.append(base + scale_base + (p % tpb) * 4)
    full_fp8, full_scale = [], []
    for r, (c, n) in enumerate(zip(cached, new)):
        kv = c + n
        if kv == 0:
            continue
        width = max(len(t) for t in peer_pages[r])
        table = torch.full((size, width), 0, dtype=torch.int64)
        for o, slots in enumerate(peer_pages[r]):
            table[o, : len(slots)] = torch.tensor(slots, dtype=torch.int64)
        table = table.to(device, non_blocking=True)
        pos = torch.arange(kv, dtype=torch.int64, device=device)
        page, row = pos // tpb, pos % tpb
        owner = page % size
        base = owner * rank_bytes + table[owner, page // size] * slot_bytes
        full_fp8.append(base + row * head_dim)
        full_scale.append(base + scale_base + row * 4)

    def put(values):
        return torch.tensor(values, dtype=torch.int64).to(device, non_blocking=True)

    def cat(parts):
        if not parts:
            return torch.empty(0, dtype=torch.int64, device=device)
        return torch.cat(parts)

    return IndexSlots(put(ins_fp8), put(ins_scale), cat(full_fp8), cat(full_scale))


def index_mirror_views(peers: Any, own_base: int, views: Mapping[int, torch.Tensor]) -> dict:
    """Each layer's index-key view over every rank's copy (``peers``: the peer mapping of the
    index-key pool, rank ``o``'s copy ``o * stride`` in; ``own_base``: where this rank's mapped
    region starts), at the layer view's offset, as ``[pages, tpb, 1, bytes per token]``: a slot of
    ``index_slots`` then addresses any rank's key."""
    flat = peer_region_tensor(peers)
    out = {}
    for layer, view in views.items():
        _, tpb, _, per_token = view.shape
        page = tpb * per_token
        offset = view.data_ptr() - own_base
        pages = (flat.numel() - offset) // page
        out[layer] = flat[offset : offset + pages * page].view(pages, tpb, 1, per_token)
    return out


def peer_tokens(peers: Any, row_bytes: int) -> torch.Tensor:
    """Every rank's latent pool through the peer mapping as token rows ``[rows, row_bytes]``
    (uint8): rank ``o``'s copy starts at row ``o * rows_per_rank * tpb``."""
    return peer_region_tensor(peers).view(-1, row_bytes)


def read_pointers(kv: torch.Tensor, num_local_layers: int) -> tuple:
    """The attention op's pool pointers and layer mapping of a layer's read buffer: one pool, every
    layer at offset 0 (host tensors, as the cache manager's)."""
    pointers = torch.tensor([[kv.data_ptr(), 0]], dtype=torch.int64)
    mapping = torch.zeros(num_local_layers, 2, dtype=torch.int32)
    return pointers, mapping
