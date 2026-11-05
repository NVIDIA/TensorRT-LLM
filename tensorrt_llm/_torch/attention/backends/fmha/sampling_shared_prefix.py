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

"""Shared-prompt generation attention for ``n``-sample requests with speculative decoding (SM90).

The children of an ``n``-sample request (``beam_width == 1``) are independent requests that map
the same leading prompt pages when KV-cache block reuse lets them share the parent's blocks. The
regular decode kernel reads that prompt once per child; for a long prompt the shared prefix
dominates the step.

This library splits a generation step into two parts and merges them by log-sum-exp. Rows whose
block tables start with identical pages form a group; the group's common full pages are read once
for all of its queries (``Fa4Prefix``, FlashAttention-4, no mask: every shared token precedes every
query), and every row attends its own remaining pages (``Fa4Tail``, causal). The step's K/V is
written into the row's own pages at ``kv_len - q_len + j``, exactly as the regular kernel does, so
the two paths are interchangeable step by step. Kernels: ``sampling_shared_prefix_fa4``.

Scope: generation-only batches with a linear draft (``q_len = 1 + K`` tokens per row, ``q_len >=
MIN_Q_LEN``) of a dense GQA/MHA model on SM90, bf16 paged KV cache of KV cache manager V2, no
sliding window, sinks, custom masks, beam search or tree drafts. Every other batch is declined and
the regular kernel runs. The draft layers run on the same metadata with the draft KV manager
swapped in and are declined.

Planning (grouping, the prefix request tables) is host work in ``TrtllmAttentionMetadata.prepare``
(``plan_step``); the groups change only when the batch's requests or their shared pages do. Each
batch size owns fixed-address device tables, refreshed with a few asynchronous copies when the
groups change, so a CUDA graph captured with this library serves any batch of its size: rows
without a shared prefix, and sharing groups beyond the ``MAX_GROUPS_ENV`` capacity, are attended
through the tail call alone. A sharing group is kept only if the size gate (``COST_MODEL``) says
reading its prefix once pays; an eager batch without any such group is declined.

Under the overlap scheduler the host KV length can exceed the true (device) length by up to K, so
the host page tables only bound the work: the tail tables and the write positions come from the
device KV lengths. Rejected drafts need no handling, their stale K/V is overwritten by the next step.

RoPE: a module that applies RoPE itself hands no ``pos_embd_params`` to the backend and the library
sees rotated Q/K. A module that fuses RoPE into the attention op (Qwen2, Llama, ...) hands them
over, and the library then rotates Q and the step's K while writing them (``rope_write_kv_paged``),
reading the op's own cos/sin table at the device position ``kv_len - q_len + j``. Supported for
NeoX-style rotation, no scaling or linear scaling and a full or partial rotary dim that is a power
of two; every other rotary variant is declined and the regular kernel runs.

Opt in with ``TLLM_FMHA_LIBS=+sampling_shared_prefix``.
"""

from __future__ import annotations

import dataclasses
import math
import os
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    PredefinedAttentionMask,
)
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantMode

from .interface import FmhaPhase
from .phased import FmhaParams, PhasedFmha
from .sampling_shared_prefix_fa4 import (
    Fa4Prefix,
    Fa4Tail,
    fa4_prefix_splits,
    fa4_sampling_attention,
    fa4_supported,
)

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )

LIB_NAME = "sampling_shared_prefix"
"""Registry name."""

SAMPLING_KEY = LIB_NAME + "_sampling"
"""Key of the per-metadata plan state in ``TrtllmAttentionMetadata.fmha_plan_caches``."""

MIN_Q_LEN = 4
"""Fewest query tokens per row (1 + draft tokens). The FA4 prefix tile needs the queries of a whole
group; the library is measured with drafts of 3 and 8 tokens."""
MAX_GROUPS_ENV = "TLLM_SAMPLING_SHARED_PREFIX_MAX_GROUPS"
DEFAULT_MAX_GROUPS = 8
"""Sharing groups the prefix call of a batch size can hold (capped at half the rows). FA4 launches
blocks for every padded request, so the padded group count stays small; the rows of further groups
are attended through the tail call alone."""
MIN_GROUP_ENV = "TLLM_SAMPLING_SHARED_PREFIX_MIN_GROUP"
DEFAULT_MIN_GROUP = 2
"""Fewest rows of a sharing group."""
MIN_SHARED_ENV = "TLLM_SAMPLING_SHARED_PREFIX_MIN_SHARED_TOKENS"
"""Fixed smallest shared prefix (tokens) of a sharing group; replaces the cost model. 0 keeps
every group, a very large value none."""

COST_MODEL = dict(mha_min_saved_bytes=8.0e6, gqa_min_saved_bytes=0.0)
"""Size gate of a sharing group on H100: the group shares its prefix when the KV bytes it saves per
layer, ``(rows - 1) * shared tokens * kv_bytes_per_token`` (``group_saved_bytes``), reach the
threshold of its attention type (MHA: as many KV heads as query heads; GQA otherwise).

Measured per layer with CUDA graphs on H100 (SM90): the library with every group sharing against
the same launches with every group demoted to the tail, 888 cases of 4-120 rows in groups of 2-80
rows, 32-6,400 shared and 256-2,048 private tokens, 3 and 8 draft tokens, for MHA 20/20 and GQA
28/4 heads of 128. With MHA, sharing loses up to 12 us per layer on small groups with short
prefixes (2-12 rows, up to a few hundred tokens) and saves up to 700 us on wide groups with long
ones; the threshold that loses least summed over the MHA cases is 7.2-10.2 MB (8 MB: 54 us over
444 cases, at most 7.8 us in one case).
GQA reads a fifth of the bytes per token and its savings and losses are small (-4 to +110 us);
sharing every group loses least (74 us over 444 cases, at most 4.2 us). Other head layouts are
not measured and take the rule of their attention type."""


def _is_v2_manager(kv_cache_manager) -> bool:
    return hasattr(kv_cache_manager, "index_mapper")


def _page_layout(metadata: "TrtllmAttentionMetadata", layer_idx: int) -> tuple[int, int, int]:
    """(pool index, scale, divisor) of a layer of a V2 manager: the page of the
    layer's pool view (``get_buffers``) that a host block-table entry names is
    ``entry * scale // divisor``.

    V2 entries are base page indices that the device-side copy multiplies by the
    pool's index scale, and ``get_buffers`` folds K and V into one page.
    """
    manager = metadata.kv_cache_manager
    pool_index = int(manager.layer_to_pool_mapping_dict[manager.layer_offsets[layer_idx]])
    return pool_index, manager.get_layer_page_index_scale(layer_idx), int(manager.kv_factor)


def _table_rows(metadata: "TrtllmAttentionMetadata", num_rows: int) -> torch.Tensor:
    """Row of the host block table of each generation row (context-free batch).

    The V2 manager keeps the table indexed by the request's slot in its index mapper.
    """
    manager = metadata.kv_cache_manager
    request_ids = metadata.request_ids
    if request_ids is None or len(request_ids) < num_rows:
        raise RuntimeError(f"{LIB_NAME}: the batch's request ids are required with KV manager V2.")
    beam_stride = int(manager.max_beam_width)
    return torch.tensor(
        [manager.index_mapper.get_index(request_ids[i]) * beam_stride for i in range(num_rows)],
        dtype=torch.int64,
    )


def _kv_manager_supported(kv_cache_manager) -> bool:
    """Whether the manager exposes the per-layer pool view and the host
    block-offset table this library reads."""
    if kv_cache_manager is None:
        return False
    for attr in ("get_buffers", "host_kv_cache_block_offsets", "kv_factor", "tokens_per_block"):
        if not hasattr(kv_cache_manager, attr):
            return False
    if getattr(kv_cache_manager, "kv_factor", 0) != 2:
        return False
    # The layer-first pool layout (recurrent-state caches) encodes offsets
    # differently; those managers never run dense attention anyway.
    if getattr(kv_cache_manager, "is_linear_attention", False):
        return False
    # With per-layer page tables (SWA scratch reuse, mixed page sizes or storage layouts in one
    # lifecycle group) the block-offset table is indexed by local layer, not by the lifecycle
    # group this library reads, and the layers of a group do not share one table.
    if getattr(kv_cache_manager, "_use_per_layer_page_tables", False):
        return False
    return True


def has_active_plan(metadata: "TrtllmAttentionMetadata") -> bool:
    """Whether ``prepare`` planned the current batch for this library.

    The FMHA selection cache keys on batch shape alone, so this state is part
    of its key: without it, a selection made on a step that had no plan would
    be reused on steps that do.
    """
    state = metadata.fmha_plan_caches.get(SAMPLING_KEY)
    return state is not None and state.plan is not None


def plan_step(metadata: "TrtllmAttentionMetadata") -> None:
    """Per-step host planning, called from ``TrtllmAttentionMetadata.prepare``.

    Groups the rows that share leading prompt pages and derives the prefix tables
    from the host block offsets the manager just filled. Runs before any forward
    of the step, so under CUDA graphs it precedes capture and replay.
    """
    state = metadata.fmha_plan_caches.get(SAMPLING_KEY)
    if state is not None:
        state.plan = None
    if sampling_metadata_eligible(metadata):
        _plan_sampling_step(metadata)
        state = metadata.fmha_plan_caches.get(SAMPLING_KEY)
    if state is not None and state.captured and state.plan is None:
        # A graph captured with this library would replay stale tables.
        raise RuntimeError(
            f"{LIB_NAME}: CUDA graph metadata captured with the n-sample path no longer "
            "satisfies its preconditions."
        )


# --------------------------------------------------------------------------- #
# Grouping
# --------------------------------------------------------------------------- #
@dataclass
class _Grouping:
    """Row grouping of one batch composition and the prefix tables derived from it.

    Rows whose block tables start with the same full pages form a sharing group
    that reads those pages once (``prefix``); every other row joins a single
    placeholder group whose prefix call runs over one token and is masked out
    of the merge. The prefix tables are in ``perm`` order (the rows of a group
    are contiguous there).

    A grouping stays valid while the batch keeps the same rows, the block
    table still holds every page it read (each row's pages up to its last
    one), and each row keeps the same number of full pages before its first
    token of the step: block tables only grow, so nothing it holds can change
    otherwise. The page check catches a row whose pages changed without any
    length changing: a new request in a recycled slot, or pages released and
    allocated again (preemption, resume from a lower cache tier). When rows
    cross a page boundary the groups stay valid unless a group's shared run
    was cut by its members' full pages (``capped_keys``) and that limit moved.
    """

    version: int
    """Changes whenever the groups or the prefix tables change."""
    num_rows: int
    table_rows: np.ndarray
    """[rows] int64: host block-table row of each batch row."""
    private_from: np.ndarray
    """[rows] int64: index of the page holding the row's first token of this step.
    Pages before it may be shared."""
    q_len: int
    """Query tokens per row: 1 + draft tokens."""
    last_pages: np.ndarray
    """[rows] int64: index of the page holding the row's last token of this step."""
    num_groups: int
    """Groups handed to the prefix call, placeholder and padding groups included."""
    num_sharing: int
    """Groups that share at least one page."""
    num_placeholder_rows: int
    shared_tokens: int
    """Prompt tokens read once per group instead of once per row, summed over rows."""
    perm: np.ndarray
    """[rows] int64: batch row of each grouped position."""
    is_identity: bool
    key_of_row: np.ndarray
    """[rows] int64: first-page key of each row."""
    capped_keys: np.ndarray
    """Keys of multi-row first pages whose members also agree on the page after
    their shared run, so the run may grow with their smallest full-page count."""
    key_caps: np.ndarray
    """[keys] int64: smallest full-page count of each key's rows."""
    shared_g: np.ndarray
    """[rows] int64, grouped order: shared pages of the row's group."""
    qo_indptr: np.ndarray
    """[num_groups + 1] int32: query-token range of each group."""
    prefix_indptr: np.ndarray
    """[num_groups + 1] int32: page range of each group in ``prefix_indices``."""
    prefix_indices: np.ndarray
    """int32: the groups' shared pages, then one placeholder page per non-sharing group."""
    prefix_last_page_len: np.ndarray
    """[num_groups] int32: tokens in each group's last prefix page."""
    prefix_empty: np.ndarray
    """[rows] bool, grouped order: the row sits in the placeholder group."""


def _indptr(counts: np.ndarray) -> np.ndarray:
    indptr = np.zeros(counts.size + 1, dtype=np.int32)
    np.cumsum(counts, out=indptr[1:])
    return indptr


def _build_grouping(
    version: int,
    table_rows: np.ndarray,
    pages_all: np.ndarray,
    kv_lens: np.ndarray,
    tokens_per_block: int,
    *,
    max_sharing_groups: Optional[int],
    padded_groups: Optional[int],
    q_len: int,
    group_pays: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
) -> _Grouping:
    """Group rows by common leading pages and build the prefix tables.

    Two rows share a prefix when their block tables start with the same pool
    page. A group's shared pages are the longest common run of full pages that
    lie entirely before every member's first token of the step; everything after
    that run stays private to the row.

    Args:
        pages_all: [rows, pages] int64 page tables covering every row's pages.
        kv_lens: [rows] int64, >= q_len; the KV length including this step's
            ``q_len`` tokens (the host value; under speculative decoding it may
            exceed the true length by up to the draft length).
        q_len: query tokens per row.
        max_sharing_groups: keep at most this many sharing groups (those that
            save the most page reads; ties go to the lower first page); the
            rows of the others join the placeholder group. None keeps all.
        padded_groups: with a value, the prefix call always gets exactly this
            many groups: the sharing groups, the placeholder group (possibly
            without rows) and empty padding groups. Without one, the
            placeholder group exists only when it has rows.
        group_pays: ``(rows, shared tokens)`` of the candidate groups -> which of
            them share; the rows of the others join the placeholder group.
    """
    i64 = np.int64
    pages_all = np.asarray(pages_all, dtype=i64)
    kv_lens = np.asarray(kv_lens, dtype=i64)
    table_rows = np.asarray(table_rows, dtype=i64)
    rows, width = pages_all.shape
    row_ids = np.arange(rows, dtype=i64)
    last_pages = (kv_lens + tokens_per_block - 1) // tokens_per_block - 1
    private_from = np.maximum(kv_lens - q_len, 0) // tokens_per_block

    first_pages = pages_all[:, 0].copy()
    _, leader, key_of_row, members = np.unique(
        first_pages, return_index=True, return_inverse=True, return_counts=True
    )
    key_of_row = key_of_row.reshape(-1)
    num_keys = leader.size
    cap = np.full(num_keys, width, dtype=i64)
    np.minimum.at(cap, key_of_row, private_from)
    multi = members > 1
    shared = np.zeros(num_keys, dtype=i64)
    capped_keys = np.zeros(0, dtype=i64)
    if multi.any():
        # Compare one page past the cap as well (the page of the step's first token,
        # which every row has): a run that also covers it may grow with the cap.
        span = int(cap[multi].max()) + 1
        cols = pages_all[:, :span]
        same = cols == cols[leader[key_of_row]]
        same &= np.arange(span, dtype=i64) <= cap[key_of_row][:, None]
        # Length of each row's leading run of pages equal to its group leader's.
        run = np.where(same.all(axis=1), span, np.argmin(same, axis=1))
        common = np.full(num_keys, span, dtype=i64)
        np.minimum.at(common, key_of_row, run)
        shared = np.where(multi, np.minimum(common, cap), 0)
        capped_keys = np.flatnonzero(multi & (common > cap))

    sharing = np.flatnonzero(shared > 0)
    if group_pays is not None and sharing.size:
        sharing = sharing[group_pays(members[sharing], shared[sharing] * tokens_per_block)]
    if max_sharing_groups is not None and sharing.size > max_sharing_groups:
        saved = shared[sharing] * (members[sharing] - 1)
        sharing = sharing[np.argsort(-saved, kind="stable")[:max_sharing_groups]]
    # Sharing groups in order of their first row; the placeholder group last.
    sharing = sharing[np.argsort(leader[sharing], kind="stable")]
    num_sharing = int(sharing.size)
    rank = np.full(num_keys, num_sharing, dtype=i64)
    rank[sharing] = np.arange(num_sharing, dtype=i64)
    row_group = rank[key_of_row]
    in_sharing = row_group < num_sharing
    row_shared = np.where(in_sharing, shared[key_of_row], 0)
    perm = np.argsort(row_group, kind="stable")

    group_sizes = members[sharing]
    group_shared = shared[sharing]
    num_placeholder_rows = rows - int(group_sizes.sum())
    if num_sharing:
        lead_pages = pages_all[leader[sharing], : int(group_shared.max())]
        sharing_pages = lead_pages[np.arange(lead_pages.shape[1]) < group_shared[:, None]]
    else:
        sharing_pages = np.zeros(0, dtype=i64)
    # Any valid page serves the placeholder: its output is masked out.
    placeholder_page = (
        int(first_pages[perm[rows - num_placeholder_rows]])
        if num_placeholder_rows
        else int(first_pages[0])
    )
    if padded_groups is None:
        extra = 1 if num_placeholder_rows else 0
        sizes = np.concatenate([group_sizes, np.full(extra, num_placeholder_rows, dtype=i64)])
    else:
        extra = padded_groups - num_sharing
        if extra < 1:
            raise RuntimeError(
                f"{LIB_NAME}: {num_sharing} sharing groups leave no room for the placeholder "
                f"group in {padded_groups} padded groups."
            )
        sizes = np.concatenate(
            [group_sizes, [num_placeholder_rows], np.zeros(extra - 1, dtype=i64)]
        )
    return _Grouping(
        version=version,
        num_rows=rows,
        table_rows=table_rows,
        private_from=private_from,
        q_len=q_len,
        last_pages=last_pages,
        num_groups=num_sharing + extra,
        num_sharing=num_sharing,
        num_placeholder_rows=num_placeholder_rows,
        shared_tokens=int((group_shared * group_sizes).sum()) * tokens_per_block,
        perm=perm,
        is_identity=bool((perm == row_ids).all()),
        key_of_row=key_of_row,
        capped_keys=capped_keys,
        key_caps=cap,
        shared_g=row_shared[perm],
        qo_indptr=_indptr(sizes * q_len),
        prefix_indptr=_indptr(np.concatenate([group_shared, np.ones(extra, dtype=i64)])),
        prefix_indices=np.concatenate(
            [sharing_pages, np.full(extra, placeholder_page, dtype=i64)]
        ).astype(np.int32),
        prefix_last_page_len=np.concatenate(
            [np.full(num_sharing, tokens_per_block, dtype=np.int32), np.ones(extra, dtype=np.int32)]
        ),
        prefix_empty=~in_sharing[perm],
    )


def _advance(
    grouping: _Grouping, private_from: np.ndarray, last_pages: np.ndarray
) -> Optional[_Grouping]:
    """``grouping`` after its rows crossed a page boundary, or None when the groups
    themselves may have changed (a capped shared run may have grown)."""
    if grouping.capped_keys.size:
        caps = grouping.key_caps.copy()
        caps[grouping.capped_keys] = int(last_pages.max()) + 1
        np.minimum.at(caps, grouping.key_of_row, private_from)
        if not np.array_equal(caps[grouping.capped_keys], grouping.key_caps[grouping.capped_keys]):
            return None
    return dataclasses.replace(grouping, private_from=private_from, last_pages=last_pages)


def _prefix_tables(
    grouping: _Grouping,
    tokens_per_block: int,
    chunk_tokens: int,
    num_requests: int,
    max_pages: int,
    splits: int = 1,
) -> tuple:
    """Request tables of the prefix call (``Fa4Prefix``).

    Every group with query rows becomes one request per ``chunk_tokens`` query tokens, each
    over the group's shared pages; the rest of the ``num_requests`` entries are empty. With
    ``splits`` > 1 the queries are replicated ``splits`` times (copy ``s`` at query offset
    ``s * tokens``) and copy ``s`` of a request attends only the ``s``-th of ``splits``
    near-equal runs of the group's pages (possibly empty).
    Returns ``(block_tables, seq_lens, cum_q, cum_kv)`` as int32 arrays.
    """
    from .flash_attn4_generation import MIN_SPLIT_TOKENS

    qo = grouping.qo_indptr
    pages_ptr = grouping.prefix_indptr
    pages = grouping.prefix_indices
    last = grouping.prefix_last_page_len
    block_tables = np.zeros((num_requests, max_pages), dtype=np.int32)
    seq_lens = np.zeros((num_requests,), dtype=np.int32)
    cum_q = np.zeros((num_requests + 1,), dtype=np.int32)
    cum_kv = np.zeros((num_requests + 1,), dtype=np.int32)
    request = 0
    tokens = int(qo[grouping.num_groups])
    for split in range(splits):
        for group in range(grouping.num_groups):
            q0, q1 = int(qo[group]), int(qo[group + 1])
            if q1 == q0:
                continue
            first, stop = int(pages_ptr[group]), int(pages_ptr[group + 1])
            count = stop - first
            kv_len = (count - 1) * tokens_per_block + int(last[group]) if count else 0
            if splits > 1:
                # Runs of at least MIN_SPLIT_TOKENS: a short prompt uses fewer (non-empty) runs.
                per = max(-(-count // splits), -(-MIN_SPLIT_TOKENS // tokens_per_block))
                lo = min(split * per, count)
                hi = min(lo + per, count)
                first, stop = first + lo, first + hi
                # Only the group's last page may be partly filled.
                kv_len = max(min(kv_len - lo * tokens_per_block, (hi - lo) * tokens_per_block), 0)
                count = hi - lo
            for start in range(q0, q1, chunk_tokens):
                if request >= num_requests:
                    raise RuntimeError(
                        f"{LIB_NAME}: the step needs more prefix requests than the "
                        f"{num_requests} its buffers hold."
                    )
                block_tables[request, :count] = pages[first:stop]
                seq_lens[request] = kv_len
                cum_q[request + 1] = split * tokens + min(start + chunk_tokens, q1)
                cum_kv[request + 1] = cum_kv[request] + kv_len
                request += 1
    cum_q[request + 1 :] = cum_q[request]
    cum_kv[request + 1 :] = cum_kv[request]
    return block_tables, seq_lens, cum_q, cum_kv


@dataclass
class _StepBuffers:
    """Fixed-address step inputs of one batch size.

    The grouping's row order, shared page counts, placeholder mask and the prefix
    request tables live in device tensors that a captured graph reads; ``upload``
    stages them in page-locked buffers and copies them asynchronously whenever the
    groups change (``prepare`` issues it, so it precedes the replay on the stream).
    The tail tables are rebuilt on the device every step (``Fa4Tail.build_tables``).
    """

    rows: int
    num_groups: int
    prefix: Fa4Prefix
    tail: Fa4Tail
    perm: torch.Tensor
    """[rows] int64: batch row of each grouped position."""
    shared: torch.Tensor
    """[rows] int32, grouped order: shared pages of the row's group."""
    empty: torch.Tensor
    """[rows] bool, grouped order: the row's prefix state is a placeholder."""
    staging: tuple
    copied: torch.cuda.Event
    pending: bool = False
    uploaded_version: int = -1

    @classmethod
    def create(
        cls,
        rows: int,
        q_len: int,
        shape: tuple,
        max_groups: int,
        max_pages: int,
        tokens_per_block: int,
        device: torch.device,
    ) -> "_StepBuffers":
        from .flash_attn4_generation import generation_splits

        num_heads, num_kv_heads, head_dim = shape
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        num_groups = min(max_groups, rows // 2) + 1
        tokens = rows * q_len
        splits = fa4_prefix_splits(rows, q_len, num_heads, num_kv_heads, num_sms)
        prefix = Fa4Prefix(
            tokens,
            num_heads,
            head_dim,
            q_len,
            Fa4Prefix.num_requests_for(num_groups, tokens, q_len) * splits,
            max_pages,
            torch.bfloat16,
            device,
            tokens_per_block,
            splits=splits,
        )
        tail = Fa4Tail(
            rows,
            q_len,
            num_heads,
            head_dim,
            max_pages,
            torch.bfloat16,
            device,
            splits=generation_splits(rows, num_heads, num_kv_heads, q_len, num_sms),
        )
        perm = torch.zeros((rows,), dtype=torch.int64, device=device)
        shared = torch.zeros((rows,), dtype=torch.int32, device=device)
        empty = torch.zeros((rows,), dtype=torch.bool, device=device)
        pinned = prefer_pinned()
        staging = tuple(
            torch.zeros(t.shape, dtype=t.dtype, pin_memory=pinned)
            for t in (perm, shared, empty, *prefix.table_arrays())
        )
        return cls(
            rows=rows,
            num_groups=num_groups,
            prefix=prefix,
            tail=tail,
            perm=perm,
            shared=shared,
            empty=empty,
            staging=staging,
            copied=torch.cuda.Event(),
        )

    def upload(self, grouping: _Grouping) -> None:
        """Stage the grouping's tables and copy them to the device buffers."""
        if self.pending:
            # The previous copies out of the staging buffers must have run.
            self.copied.synchronize()
        prefix = self.prefix
        host = (
            grouping.perm,
            grouping.shared_g,
            grouping.prefix_empty,
            *_prefix_tables(
                grouping,
                prefix.tokens_per_block,
                prefix.chunk_tokens,
                prefix.num_requests,
                prefix.block_tables.shape[1],
                splits=prefix.splits,
            ),
        )
        devices = (self.perm, self.shared, self.empty, *prefix.table_arrays())
        for staging, device, values in zip(self.staging, devices, host):
            staging.numpy()[...] = values
            device.copy_(staging, non_blocking=True)
        self.copied.record()
        self.pending = True
        self.uploaded_version = grouping.version


@dataclass
class _SamplingPlan:
    """Host-side description of one n-sample generation step, shared by all layers."""

    step: int
    grouping: _Grouping

    @property
    def num_rows(self) -> int:
        return self.grouping.num_rows

    @property
    def q_len(self) -> int:
        return self.grouping.q_len


@dataclass
class _SamplingState:
    pool_index: int
    page_scale: int
    page_div: int
    num_pages: int
    manager_ref: Optional[weakref.ref] = None
    """The KV-cache manager the plan describes. The draft layers of one-model
    speculative decoding run on this metadata with the draft manager swapped in;
    they must not use the target's plan."""
    buffers: Optional[_StepBuffers] = None
    plan: Optional[_SamplingPlan] = None
    grouping: Optional[_Grouping] = None
    grouping_pages: Optional[np.ndarray] = None
    """[rows, pages] host block-table entries ``grouping`` was built from."""
    table_rows_key: Optional[tuple] = None
    table_rows: Optional[np.ndarray] = None
    step_counter: int = 0
    grouping_counter: int = 0
    captured: bool = False
    tables_step: int = -1
    """Step whose tail tables ``Fa4Tail.build_tables`` last wrote."""
    tables_captured: bool = False
    logged: bool = False
    ran: bool = False


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return int(value)


_model_shape: Optional[tuple] = None
"""(heads, kv_heads, head_dim) of the attention layers that can use the library, set
when a layer reports itself available; the buffers and the cost model need it."""


def group_saved_bytes(rows: np.ndarray, tokens: np.ndarray, shape: tuple) -> np.ndarray:
    """KV bytes per layer that sharing groups of ``rows`` rows and ``tokens`` shared tokens do not
    read again: every row but one would read the shared prefix (K and V, bf16)."""
    _, num_kv_heads, head_dim = shape
    rows = np.asarray(rows, dtype=np.float64)
    tokens = np.asarray(tokens, dtype=np.float64)
    return (rows - 1) * tokens * (4 * num_kv_heads * head_dim)


def _min_saved_bytes(shape: tuple) -> float:
    num_heads, num_kv_heads, _ = shape
    key = "mha_min_saved_bytes" if num_heads == num_kv_heads else "gqa_min_saved_bytes"
    return COST_MODEL[key]


def _group_filter(q_len: int) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Which candidate groups ``(rows, shared tokens)`` share: the env threshold if set, else the
    cost model (``COST_MODEL``); never a group below the minimum size."""
    min_group = _int_env(MIN_GROUP_ENV, DEFAULT_MIN_GROUP)
    fixed = os.environ.get(MIN_SHARED_ENV, "").strip()
    shape = _model_shape

    def pays(rows: np.ndarray, tokens: np.ndarray) -> np.ndarray:
        wide = rows >= min_group
        if fixed:
            return wide & (tokens >= int(fixed))
        return wide & (group_saved_bytes(rows, tokens, shape) >= _min_saved_bytes(shape))

    return pays


def sampling_metadata_eligible(metadata: "TrtllmAttentionMetadata") -> bool:
    """Batch-level preconditions of the n-sample path (KV-cache manager V2 only).

    They depend only on the batch shape and the model configuration, so they
    hold on every replay of a CUDA graph that was captured with this path.
    """
    return (
        metadata.beam_width == 1
        and metadata.num_generations > 1
        and metadata.num_contexts == 0
        and not metadata.is_cross
        and not metadata.enable_helix
        # use_spec_decoding is not tested: the model forward leaves it set after
        # every step, so it says nothing about the next one. A tree draft is
        # declined, a linear draft is the q_len > 1 path.
        and not metadata.is_spec_dec_tree
        and not metadata.is_spec_dec_dynamic_tree
        and metadata.kv_cache_block_offsets is not None
        and metadata.kv_cache_params is not None
        and metadata.kv_cache_params.use_cache
        and _is_v2_manager(metadata.kv_cache_manager)
        and _kv_manager_supported(metadata.kv_cache_manager)
    )


def _sampling_q_len(metadata: "TrtllmAttentionMetadata", rows: int) -> int:
    """Query tokens per generation row of a context-free batch, or 0 when the
    rows do not all carry the same number (the batch is declined then).

    Speculative decoding gives every generation request ``1 + draft tokens``
    tokens; the executor keeps that count uniform across a batch. Eager batches
    are checked against the per-request lengths, CUDA graph batches are uniform
    by construction.
    """
    tokens = metadata.num_tokens
    if rows <= 0 or tokens % rows:
        return 0
    q_len = tokens // rows
    if q_len > 1 and not metadata.is_cuda_graph:
        seq_lens = metadata.seq_lens
        if seq_lens is None or not bool((seq_lens[:rows] == q_len).all()):
            return 0
    return q_len


def _sampling_table_rows(
    state: _SamplingState, metadata: "TrtllmAttentionMetadata", rows: int
) -> np.ndarray:
    """``_table_rows`` cached while the batch keeps the same requests."""
    request_ids = metadata.request_ids
    cache_key = tuple(request_ids[:rows]) if request_ids is not None else None
    if cache_key is None or cache_key != state.table_rows_key:
        state.table_rows = _table_rows(metadata, rows).numpy()
        state.table_rows_key = cache_key
    assert state.table_rows is not None
    return state.table_rows


def _pages_unchanged(
    saved: Optional[np.ndarray], offsets: np.ndarray, table_rows: np.ndarray, last_pages: np.ndarray
) -> bool:
    """Whether the host block table still holds every page a grouping read.

    Args:
        saved: [rows, pages] block-table entries the grouping was built from.
        offsets: [table rows, max pages] the host block table of the pool.
        table_rows: [rows] block-table row of each batch row.
        last_pages: [rows] the grouping's last page index of each row; entries past it
            were not read and may change.
    """
    if saved is None or saved.shape[0] != table_rows.size:
        return False
    width = saved.shape[1]
    read = np.arange(width) <= last_pages[:, None]
    return bool(((offsets[table_rows, :width] == saved) | ~read).all())


def _plan_sampling_step(metadata: "TrtllmAttentionMetadata") -> None:
    """Plan an n-sample generation step: group the rows and derive the prefix tables.

    An eager batch whose sharing groups do not pay is declined (the regular kernel
    runs). A CUDA graph batch is always planned, because its graph is captured
    with this path: rows without a shared prefix take the placeholder group.
    """
    rows = metadata.num_generations
    q_len = _sampling_q_len(metadata, rows)
    if q_len < MIN_Q_LEN or _model_shape is None:
        return
    device = metadata.kv_cache_block_offsets.device
    if not fa4_supported(_model_shape[2], torch.bfloat16, device):
        return
    state = metadata.fmha_plan_caches.get(SAMPLING_KEY)
    tokens_per_block = int(metadata.tokens_per_block)
    host_offsets = metadata.host_kv_cache_block_offsets
    assert host_offsets is not None
    if state is None:
        kv_cache_manager = metadata.kv_cache_manager
        first_layer = min(kv_cache_manager.layer_offsets)
        pool_index, page_scale, page_div = _page_layout(metadata, first_layer)
        pool = kv_cache_manager.get_buffers(first_layer, kv_layout="HND")
        state = _SamplingState(
            pool_index=pool_index,
            page_scale=page_scale,
            page_div=page_div,
            num_pages=int(pool.size(0)),
            manager_ref=weakref.ref(kv_cache_manager),
        )
        metadata.fmha_plan_caches[SAMPLING_KEY] = state
    eager = not metadata.is_cuda_graph
    buffers = state.buffers
    if buffers is not None and (rows != buffers.rows or q_len != buffers.tail.q_len):
        if not eager:
            raise RuntimeError(
                f"{LIB_NAME}: CUDA graph metadata planned for {buffers.rows} rows got {rows}."
            )
        buffers = None  # an eager batch changed shape: new fixed-address buffers below
    if buffers is None:
        buffers = state.buffers = _StepBuffers.create(
            rows,
            q_len,
            _model_shape,
            _int_env(MAX_GROUPS_ENV, DEFAULT_MAX_GROUPS),
            int(host_offsets.size(-1)),
            tokens_per_block,
            device,
        )
        state.grouping = None
    state.step_counter += 1

    kv_lens = np.maximum(metadata.kv_lens_runtime[:rows].numpy(), q_len).astype(np.int64)
    table_rows = _sampling_table_rows(state, metadata, rows)
    # Pages that may be shared end before the row's first token of this step; the
    # row's own pages run to the page of its last token.
    private_from = (kv_lens - q_len) // tokens_per_block
    last_pages = (kv_lens - 1) // tokens_per_block
    offsets = host_offsets.numpy()[state.pool_index, :, 0, :]
    grouping = state.grouping
    same_rows = (
        grouping is not None
        and grouping.num_rows == rows
        and grouping.q_len == q_len
        and np.array_equal(grouping.table_rows, table_rows)
        and _pages_unchanged(state.grouping_pages, offsets, table_rows, grouping.last_pages)
    )
    if (
        not same_rows
        or not np.array_equal(grouping.private_from, private_from)
        or not np.array_equal(grouping.last_pages, last_pages)
    ):
        width = int(last_pages.max()) + 1
        # Fancy indexing copies: the snapshot does not follow later table writes.
        host_pages = offsets[table_rows, :width]
        advanced = _advance(grouping, private_from, last_pages) if same_rows else None
        if advanced is None:
            pages_all = np.clip(
                host_pages.astype(np.int64) * state.page_scale // state.page_div,
                0,
                state.num_pages - 1,
            )
            state.grouping_counter += 1
            advanced = _build_grouping(
                state.grouping_counter,
                table_rows,
                pages_all,
                kv_lens,
                tokens_per_block,
                max_sharing_groups=buffers.num_groups - 1,
                padded_groups=buffers.num_groups,
                q_len=q_len,
                group_pays=_group_filter(q_len),
            )
        grouping = state.grouping = advanced
        state.grouping_pages = host_pages
    if eager and grouping.num_sharing == 0:
        # No group passes the size gate: the library would only add its fixed cost.
        return

    state.plan = _SamplingPlan(step=state.step_counter, grouping=grouping)
    if grouping.version != buffers.uploaded_version:
        buffers.upload(grouping)
    if not state.logged:
        state.logged = True
        logger.info(
            f"{LIB_NAME}: n-sample step with {rows} rows in {grouping.num_sharing} sharing "
            f"groups ({grouping.num_placeholder_rows} rows without a shared prefix), "
            f"{grouping.shared_tokens} prompt tokens read once per group, "
            f"{buffers.num_groups} prefix groups"
        )


def _rotary_dim(attn: "TrtllmAttention") -> int:
    return int(attn.rope_params.dim)


def _fused_rope_supported(attn: "TrtllmAttention") -> bool:
    """Whether the rotary embedding the attention op would apply is one that
    ``rope_write_kv_paged`` reproduces (see the module docstring)."""
    if int(attn.position_embedding_type) != int(PositionEmbeddingType.rope_gpt_neox):
        return False
    rope = attn.rope_params
    if rope.scale_type not in (RotaryScalingType.none, RotaryScalingType.linear):
        return False
    if rope.duplicate_data or getattr(attn, "rotary_cos_sin", None) is None:
        return False
    dim = _rotary_dim(attn)
    half = dim // 2
    if dim <= 0 or dim % 2 or half & (half - 1) or dim > attn.head_dim:
        return False
    # The table is [positions][dim] float32 on the device.
    table = attn.rotary_cos_sin
    return table.dtype == torch.float32 and table.numel() % dim == 0


class SamplingSharedPrefixFmha(PhasedFmha):
    """Generation-phase library for n-sample requests; see the module docstring."""

    supports_workspace_reclamation = False

    def __init__(self, attn: "TrtllmAttention"):
        super().__init__(attn)
        self._pool_views: dict[int, torch.Tensor] = {}
        # (pool index, scale, divisor) of this layer; fixed for the manager lifetime.
        self._layout: Optional[tuple[int, int, int]] = None
        # The KV-cache manager the cached pool views and layout belong to. The
        # executor replaces the manager after memory estimation, and CUDA graph
        # warmup runs this library with the estimation manager first.
        self._manager_ref: Optional[weakref.ref] = None

    def _bind_manager(self, metadata: "TrtllmAttentionMetadata") -> None:
        manager = metadata.kv_cache_manager
        if self._manager_ref is None or self._manager_ref() is not manager:
            self._pool_views.clear()
            self._layout = None
            self._manager_ref = weakref.ref(manager)

    @classmethod
    def _is_available(cls, attn: "TrtllmAttention") -> bool:
        if (
            getattr(attn, "is_mla_enable", False)
            or getattr(attn, "sparse_params", None) is not None
        ):
            return False
        if getattr(attn, "predicted_tokens_per_seq", 1) != 1:
            return False
        if getattr(attn, "attention_chunk_size", None):
            return False
        head_dim = getattr(attn, "head_dim", 0)
        num_heads = getattr(attn, "num_heads", 0)
        num_kv_heads = getattr(attn, "num_kv_heads", 0) or 0
        if head_dim not in (64, 128):
            return False
        if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
            return False
        if QuantMode(getattr(attn, "quant_mode", 0)).has_kv_cache_quant():
            return False
        position_type = PositionEmbeddingType(getattr(attn, "position_embedding_type", 0))
        if position_type.is_rope():
            if not _fused_rope_supported(attn):
                return False
        elif position_type != PositionEmbeddingType.learned_absolute:
            # ALiBi, relative, GLM and deferred embeddings are applied inside the attention op,
            # which this library bypasses.
            return False
        global _model_shape
        if _model_shape is None:  # the target layers come first; draft layers share the shape
            _model_shape = (num_heads, num_kv_heads, head_dim)
        return True

    def _is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> bool:
        if phase == FmhaPhase.CONTEXT:
            return False
        if not sampling_metadata_eligible(metadata):
            return False
        if PositionEmbeddingType(getattr(self.attn, "position_embedding_type", 0)).is_rope():
            # The rotary table must cover every position the batch can reach.
            max_seq_len = metadata.max_seq_len
            if max_seq_len is None or max_seq_len > (
                self.attn.rotary_cos_sin.numel() // _rotary_dim(self.attn)
            ):
                return False
        if forward_args.attention_input_type == AttentionInputType.context_only:
            return False
        if not (forward_args.is_fused_qkv and k is None and v is None):
            return False
        if not forward_args.update_kv_cache:
            return False
        if forward_args.attention_mask != PredefinedAttentionMask.CAUSAL:
            return False
        if forward_args.attention_mask_data is not None or forward_args.attention_sinks is not None:
            return False
        if forward_args.relative_attention_bias is not None:
            return False
        if forward_args.out_scale is not None or forward_args.output_sf is not None:
            return False
        if q.dtype != torch.bfloat16:
            return False
        window = forward_args.attention_window_size
        if (
            window is not None
            and metadata.max_seq_len is not None
            and window < metadata.max_seq_len
        ):
            return False
        sampling_state = metadata.fmha_plan_caches.get(SAMPLING_KEY)
        # prepare() declines batches the library does not serve.
        if sampling_state is None or sampling_state.plan is None:
            return False
        # The plan describes the target model's KV cache. The draft layers of
        # one-model speculative decoding run on this metadata with the draft
        # manager swapped in (and with different token counts); they take the
        # regular kernel.
        manager_ref = sampling_state.manager_ref
        if manager_ref is None or manager_ref() is not metadata.kv_cache_manager:
            return False
        plan = sampling_state.plan
        return q.shape[0] == plan.num_rows * plan.q_len

    def _pool_view(self, metadata: "TrtllmAttentionMetadata") -> torch.Tensor:
        self._bind_manager(metadata)
        layer_idx = self.attn.layer_idx
        pool = self._pool_views.get(layer_idx)
        if pool is None:
            pool = metadata.kv_cache_manager.get_buffers(layer_idx, kv_layout="HND")
            if pool is None:
                raise RuntimeError(f"{type(self).__name__}: layer {layer_idx} has no KV pool.")
            self._pool_views[layer_idx] = pool
        return pool

    def run_generation(self, params: FmhaParams) -> None:
        qkv = params.qkv_input
        if qkv is None:
            raise RuntimeError(f"{type(self).__name__} requires packed QKV input.")
        attn = params.attn
        metadata = params.meta
        state: Optional[_SamplingState] = metadata.fmha_plan_caches.get(SAMPLING_KEY)
        if state is None or state.plan is None or state.buffers is None:
            raise RuntimeError(
                f"{type(self).__name__}: no n-sample step plan; "
                "TrtllmAttentionMetadata.prepare() must run before the forward."
            )
        plan = state.plan
        buffers = state.buffers
        num_heads = attn.num_heads
        num_kv_heads = attn.num_kv_heads
        head_dim = attn.head_dim
        rows = plan.num_rows
        q_len = plan.q_len
        tokens = params.num_tokens
        if tokens != rows * q_len:
            raise RuntimeError(
                f"{type(self).__name__}: {tokens} generation tokens do not match the planned "
                f"{rows} rows of {q_len}."
            )
        self._bind_manager(metadata)
        if self._layout is None:
            self._layout = _page_layout(metadata, attn.layer_idx)
        if self._layout != (state.pool_index, state.page_scale, state.page_div):
            raise RuntimeError(
                f"{type(self).__name__} supports a single KV pool layout per model; layer "
                f"{attn.layer_idx} uses (pool, scale, divisor) {self._layout} but the plan was "
                f"built for {(state.pool_index, state.page_scale, state.page_div)}."
            )
        tail = buffers.tail
        if tail.out.shape[1:] != (num_heads, head_dim):
            raise RuntimeError(
                f"{type(self).__name__}: the step buffers were sized for (heads, head_dim) "
                f"{tuple(tail.out.shape[1:])} (the first model that reported the library available) "
                f"but layer {attn.layer_idx} has {(num_heads, head_dim)}."
            )
        pool = self._pool_view(metadata)
        kv_lens = metadata.kv_lens_cuda_runtime[:rows]
        capturing = torch.cuda.is_current_stream_capturing()
        # Once per step, and recorded by the capture itself: an eager warmup forward of
        # the same step does not put it in the graph.
        if state.tables_step != plan.step or (capturing and not state.tables_captured):
            tail.build_tables(
                metadata.kv_cache_block_offsets[state.pool_index, :rows, 0],
                buffers.perm,
                buffers.shared,
                kv_lens,
                int(metadata.kv_cache_manager.kv_factor),
                int(metadata.tokens_per_block),
            )
            state.tables_step = plan.step
            state.tables_captured = capturing
        if capturing:
            state.captured = True

        from .flash_attn4_generation import rope_write_kv_paged

        rope = PositionEmbeddingType(attn.position_embedding_type).is_rope()
        # RoPE on Q (in place) and K, and the step's K/V into the rows' private pages, in one
        # launch; only the pool is read for K/V from here on.
        rope_write_kv_paged(
            qkv,
            attn.rotary_cos_sin if rope else None,
            pool,
            tail.table,
            kv_lens,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rotary_dim=_rotary_dim(attn) if rope else head_dim,
            q_len=q_len,
            page_lens=tail.tail_lens,
        )
        output = params.output
        if output is None:
            raise RuntimeError(f"{type(self).__name__} requires an output tensor.")
        fa4_sampling_attention(
            pool,
            qkv[:, : num_heads * head_dim].view(tokens, num_heads, head_dim),
            None,
            None,
            output.view(tokens, num_heads, head_dim),
            perm=buffers.perm,
            prefix_empty=buffers.empty,
            prefix=buffers.prefix,
            tail=tail,
            sm_scale=1.0 / (math.sqrt(head_dim) * attn.q_scaling),
        )
        if not state.ran:
            state.ran = True
            logger.info(f"{LIB_NAME}: first n-sample attention call (layer {attn.layer_idx})")


__all__ = [
    "LIB_NAME",
    "SamplingSharedPrefixFmha",
    "group_saved_bytes",
    "has_active_plan",
    "plan_step",
    "sampling_metadata_eligible",
]
