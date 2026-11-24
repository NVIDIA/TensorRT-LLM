# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helix attention over CP-owned target and draft pages.

The general target's Helix context parallelism follows the engine's Helix support, functional only: the causal LM's
mapping swap (:func:`setup_helix_mappings`), the engine attention backend of each MLA layer
(:func:`create_helix_attention`, :func:`helix_mla_attention`) and the CP combines below. Only a deployment with Helix CP
(``mapping.has_cp_helix()``) imports this module.
"""

import math

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.dsa_attention import (
    _write_identity_rope_values,
)
from tensorrt_llm._torch.attention.attention import _helix_post_process
from tensorrt_llm._torch.attention.backends.interface import AttentionForwardArgs
from tensorrt_llm._torch.attention.backends.utils import create_attention
from tensorrt_llm._torch.attention.mla import helix_gen_zero_kv_mask
from tensorrt_llm._torch.distributed.ops import cp_allgather
from tensorrt_llm._torch.speculative.dflash_attention import HelixDraftShard, combine_helix_partials


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
    note = "Eager FP32 attention and CP all-gather; CUDA graphs are not supported."

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


def setup_helix_mappings(model, model_config, text_config, spec_config) -> None:
    """The engine's Helix mapping swap for the Kimi K3 causal LM, on ``model`` (the target's causal LM).

    ``KimiLinearForCausalLM._setup_helix_mappings`` checks the Helix preconditions, swaps ``model_config.mapping`` for
    the repurposed one (CP ranks become TP ranks: every layer but the MLA attention is built on it) and records the CP
    original in ``model.mapping_with_cp`` (restored after construction) and ``model_config._helix_mapping_with_cp``
    (read by the MLA layers), and the repurposed one in ``model._repurposed_tp_mapping`` (the weight loader's TP rank).
    """
    from tensorrt_llm._torch.models.modeling_kimi_linear import KimiLinearForCausalLM

    KimiLinearForCausalLM._setup_helix_mappings(model, model_config, text_config, spec_config)


def create_helix_attention(
    backend_name: str, layer_idx: int, num_heads: int, head_dim: int, **kwargs
):
    """The attention of a Helix MLA layer: the engine's backend for the layer's DSA parameters
    (``attention.backends.utils.create_attention``), which attends over this rank's CP-owned pages.

    Adapted to the family MLA in two ways. K3.1 is NoPE: the backend's RoPE table holds the identity and keeps it
    when it grows (``_ensure_rope_table_size``, also bound as ``ensure_rope_table_size``, the name the MLA's generation
    RoPE calls). And its indexer takes the family DSA ops' calls (:func:`_take_family_indexer_calls`).
    """
    backend = create_attention(backend_name, layer_idx, num_heads, head_dim, **kwargs)
    _install_identity_rope_table(backend)
    if getattr(backend, "indexer", None) is not None:
        _take_family_indexer_calls(backend.indexer)
    return backend


def _install_identity_rope_table(backend) -> None:
    """Identity values (cos 1, sin 0) in ``backend``'s RoPE table, its shape kept, and kept through every resize."""
    if backend.rotary_cos_sin is None:
        raise RuntimeError(
            "the Helix MLA backend has no RoPE table; check pos_embd_params has a rope dim > 0"
        )
    _write_identity_rope_values(backend.rotary_cos_sin)
    resize = backend._ensure_rope_table_size

    def identity_preserving_resize(required_max_positions: int) -> None:
        if required_max_positions <= backend.rope_params.max_positions:
            return
        resize(required_max_positions)
        _write_identity_rope_values(backend.rotary_cos_sin)

    backend._ensure_rope_table_size = identity_preserving_resize
    backend.ensure_rope_table_size = identity_preserving_resize


def _take_family_indexer_calls(indexer) -> None:
    """Give the engine indexer the family DSA ops' calls.

    The family ops call ``pre_indexer_proj(qr, hidden_states)`` for the intermediates ``[q_fp8, k_fp8, k_scale,
    weights]`` and pass those to ``forward_from_projected``. The engine's also take the token positions, which a
    position-free indexer (K3.1's, ``index_rope_dim`` 0) does not read, and carry the query scale that only FP4 index
    keys read (K3.1's are FP8).
    """
    project = indexer.pre_indexer_proj
    predict = indexer.forward_from_projected

    def pre_indexer_proj(qr, hidden_states):
        q_fp8, k_fp8, k_scale, weights, _ = project(qr, hidden_states, None)
        return q_fp8, k_fp8, k_scale, weights

    def forward_from_projected(metadata, hidden_states, indexer_intermediates, is_generation=None):
        return predict(metadata, hidden_states, [*indexer_intermediates, None], is_generation)

    indexer.pre_indexer_proj = pre_indexer_proj
    indexer.forward_from_projected = forward_from_projected


def helix_mla_attention(mla, q: torch.Tensor, metadata, **kwargs) -> torch.Tensor:
    """``mla``'s absorbed attention on a Helix rank: ``[rows, num_heads_tp_cp * kv_lora_rank]``.

    The engine backend (``mla.mqa``, :func:`create_helix_attention`) attends over this rank's CP-owned pages and
    returns its softmax statistics; :data:`helix_target_combine` exchanges the CP partials and returns this rank's head
    shard. ``kwargs`` are the backend's forward arguments (``AttentionForwardArgs`` fields).
    """
    stats = torch.empty((q.shape[0], mla.num_heads_attn, 2), device=q.device, dtype=torch.float32)
    partial = mla.mqa.forward(
        q,
        None,
        None,
        metadata,
        forward_args=AttentionForwardArgs(softmax_stats_tensor=stats, **kwargs),
    )
    return helix_target_combine(
        partial,
        stats,
        mla.mapping,
        mla.num_heads_tp,
        mla.kv_lora_rank,
        metadata,
        mla.aux_stream,
        mla.ln_events,
    )
