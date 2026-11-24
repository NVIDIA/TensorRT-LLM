# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helix attention over CP-owned target and draft pages."""

import math
from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell, OpWrapper
from tensorrt_llm._torch.attention.attention import _helix_post_process, _helix_zero_kv_mask
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.distributed.ops import cp_allgather

from ...runtime.draft_attention import HelixDraftShard, combine_helix_partials


def helix_gen_zero_kv_mask(
    attn_metadata: AttentionMetadata, num_tokens: int
) -> Optional[torch.Tensor]:
    """Per-token mask of the ``num_tokens`` generation tokens this CP rank owns no KV for.

    Used by the Helix generation combine (``_helix_post_process``) to make those rows a
    no-op. It only depends on the step's metadata, so callers may compute it ahead of the
    attention (see ``MLA.helix_zero_kv_mask_fn``).
    """
    helix_kv_bounds = getattr(attn_metadata, "helix_kv_bounds", None)
    if helix_kv_bounds is not None and getattr(attn_metadata, "_helix_spec_tokens_valid", False):
        # Speculative verify groups: KV ownership is per-TOKEN. A rank
        # owning only the tail page of a group has zero visible KV for
        # the group's leading tokens while its per-sequence kv_len is
        # nonzero, so the per-sequence mask below misses those rows.
        # Their decode rows are fully masked with a finite sentinel,
        # making partial_o an average over (possibly uninitialized)
        # pool values; the combine multiplies by corr = 0 and
        # 0 * NaN would poison the token on every CP rank — sanitize
        # by the per-token bound instead.
        return helix_kv_bounds[:num_tokens] == 0
    # MLA processes only the generation token slice, so build the mask
    # from [num_contexts, num_seqs). Skip this expansion when the more
    # precise per-token bounds above are valid.
    return _helix_zero_kv_mask(
        attn_metadata,
        num_tokens,
        seq_start=attn_metadata.num_contexts,
        num_seqs=attn_metadata.num_generations,
    )


class HelixTargetCombine(OpWrapper):
    """Drop padded query heads, exchange CP partials, and return this rank's head shard."""

    ARCHS = frozenset()
    CELLS = (
        Cell(
            why="96 real heads padded to 128 before a CP4 exchange",
            spec=dict(heads=96, padded=128, cp=4),
        ),
    )
    note = "Uses the engine Helix exchange, including empty-shard sanitization."

    def __call__(self, partial, stats, mapping, heads, latent_dim, metadata, stream, events):
        rows = partial.shape[0]
        partial = partial.view(rows, -1, latent_dim)[:, :heads].contiguous().view(rows, -1)
        stats = stats[:, :heads].contiguous()
        empty = helix_gen_zero_kv_mask(metadata, rows)
        sparse_empty = getattr(metadata, "helix_sparse_empty_rows", None)
        if sparse_empty is not None:
            sparse_empty = sparse_empty(rows)
            if sparse_empty is not None:
                empty = sparse_empty if empty is None else empty | sparse_empty
        return _helix_post_process(
            partial,
            stats,
            mapping,
            heads // mapping.cp_size,
            latent_dim,
            stream,
            events,
            zero_kv_mask=empty,
            compile_nccl_reformat=False,
        )

    def reference(self, partials, stats, cp_rank):
        """Combine rank-ordered normalized partials with natural-log maxima and exp sums."""
        maxima, sums = stats[..., 0], stats[..., 1]
        maximum = maxima.amax(dim=0, keepdim=True)
        weights = (maxima - maximum).exp().nan_to_num(0) * sums
        weights = weights / weights.sum(dim=0).clamp_min(torch.finfo(torch.float32).tiny)
        output = (partials.float().nan_to_num(0) * weights[..., None]).sum(dim=0)
        width = output.shape[1] // partials.shape[0]
        return output[:, cp_rank * width : (cp_rank + 1) * width].to(partials.dtype)


class HelixDraftAttention(OpWrapper):
    """Sliding-window MLA over manager-owned context shards and a transient proposed block.

    Query is [batch * block, heads, latent + rope]. Pool is [pages, page_size, latent + rope].
    Only window rows are gathered; the persistent cache never has a replicated full-history arena.
    """

    ARCHS = frozenset()
    CELLS = tuple(
        Cell(
            why="CP4 empty shards, partial pages and window crossing", spec=dict(block=width, cp=4)
        )
        for width in (5, 8)
    )
    note = "FP32 sliding-window attention over CP-owned history and a replicated proposed block."

    def __call__(
        self, query, pool, tables, lengths, block_kv, block, window, scale, latent_dim, mapping
    ):
        batch = lengths.numel()
        device = query.device
        past = lengths[:, None] + torch.arange(1 - window, 0, device=device)
        future = lengths[:, None] + torch.arange(block, device=device)
        shard = HelixDraftShard(mapping.cp_size, mapping.cp_rank, pool.shape[1])
        slots, owned = shard.local_slots(past.clamp_min(0))
        valid = owned & (past >= 0)
        slots = slots.masked_fill(~valid, 0)
        pages = tables.gather(1, slots // pool.shape[1])
        # Convert before indexing for FP8 pools, whose advanced indexing is not supported everywhere.
        if pool.dtype == torch.float8_e4m3fn:
            context = (
                pool.view(torch.uint8)[pages, slots % pool.shape[1]]
                .view(torch.float8_e4m3fn)
                .float()
            )
        else:
            context = pool[pages, slots % pool.shape[1]].float()
        context.masked_fill_(~valid[..., None], 0)
        kv = torch.cat((context, block_kv.float().view(batch, block, -1)), dim=1)
        positions = torch.cat((past, future), dim=1)
        _, block_owned = shard.local_slots(future)
        visible = (positions[:, None] >= 0) & (positions[:, None] <= future[:, :, None])
        visible &= positions[:, None] > future[:, :, None] - window
        visible &= torch.cat((owned, block_owned), dim=1)[:, None]
        scores = (
            torch.einsum("bqhd,bkd->bqhk", query.float().view(batch, block, query.shape[1], -1), kv)
            * scale
        )
        scores.masked_fill_(~visible[:, :, None], -torch.inf)
        probs = scores.softmax(-1).nan_to_num(0)
        partial = (
            torch.einsum("bqhk,bkd->bqhd", probs, kv[..., :latent_dim])
            .reshape(query.shape[0], query.shape[1], latent_dim)
            .to(query.dtype)
        )
        lse = (scores.logsumexp(-1) / math.log(2)).reshape(query.shape[:2])
        return combine_helix_partials(
            cp_allgather(partial[None], mapping, dim=0), cp_allgather(lse[None], mapping, dim=0)
        )

    def reference(self, query, full_history, lengths, block_kv, block, window, scale, latent_dim):
        """Dense per-request attention, with no page layout or CP collectives."""
        outputs = []
        for row, history in enumerate(full_history):
            kv = torch.cat((history.float(), block_kv[row * block : (row + 1) * block].float()))
            qs = query[row * block : (row + 1) * block].float()
            for offset in range(block):
                end = int(lengths[row]) + offset + 1
                keys = kv[max(0, end - window) : end]
                probs = (qs[offset] @ keys.T * scale).softmax(-1)
                outputs.append(probs @ keys[:, :latent_dim])
        return torch.stack(outputs).to(query.dtype)


helix_target_combine = HelixTargetCombine()
helix_draft_attention = HelixDraftAttention()
