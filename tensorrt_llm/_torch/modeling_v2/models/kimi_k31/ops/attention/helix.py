# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helix attention over CP-owned target and draft pages.

The general target's Helix context parallelism follows the engine's Helix support for Kimi K3, functional only: the
causal LM's mapping swap (:func:`setup_helix_mappings`), the engine attention backend of each MLA layer
(:func:`create_helix_attention`, :func:`helix_mla_attention`) and the CP combines below. The mapping swap, the target
combine and its zero-KV masks are the engine's Kimi K3 / MLA code, copied here; the attention backend is the engine's.
The engine's DSA metadata runs Helix through ``HelixDSAMetadataMixin`` (:class:`K31HelixDSAMetadata`, declared by
:class:`K31HelixDSABackend`), and the engine indexer's decode Top-K takes the Helix merge of the CP ranks' selections
(:func:`_take_helix_decode_top_k`). Only a deployment with Helix CP (``mapping.has_cp_helix()``) imports this module.
"""

from __future__ import annotations

import copy
import functools
import math
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionForwardArgs
from tensorrt_llm._torch.attention.backends.sparse.dsa.backend import DSATrtllmAttention
from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import Indexer
from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import DSAtrtllmAttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.attention.backends.utils import create_attention
from tensorrt_llm._torch.distributed import HelixAllToAllNative, alltoall_helix
from tensorrt_llm._torch.distributed.ops import cp_allgather
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention._rubin_dsa_indexer_plan import (
    compute_slot_mappings,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_attention import (
    _write_identity_rope_values,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_metadata import (
    HelixDSAMetadataMixin,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.get_paged_mqa_logits_metadata import (
    SCHEDULE_BLOCK_KV,
    get_paged_mqa_logits_metadata,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.rubin_merge_helix_topk import (
    rubin_merge_helix_topk,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.helix_draft_shard import (
    HelixDraftShard,
    combine_helix_partials,
)
from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig
    from tensorrt_llm.mapping import Mapping


def _helix_zero_kv_mask(
    attn_metadata: AttentionMetadata,
    num_tokens: int,
    *,
    seq_start: int = 0,
    num_seqs: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Return a per-token bool mask marking tokens with zero local KV on this CP rank.

    These tokens belong to sequences for which this rank owns no KV blocks. Since
    kv_lens_cuda is per-sequence, it is expanded to per-token using the
    seq_lens_cuda query lengths, which stays correct when a sequence spans
    multiple tokens (e.g. speculative decoding). The buffers are static and
    output_size is passed to repeat_interleave, so the mask is CUDA-graph safe.
    Returns None when a length buffer is unavailable.

    Args:
        attn_metadata: Attention metadata holding the length buffers.
        num_tokens: Number of tokens covered by partial_o; used as the static
            output_size of the per-token expansion.
        seq_start: Index of the first sequence covered by partial_o. Defaults to
            0 for generation-only callers; MLA passes num_contexts to select the
            generation slice.
        num_seqs: Number of sequences covered by partial_o. Defaults to all
            sequences after seq_start.
    """
    kv_lens = getattr(attn_metadata, "kv_lens_cuda", None)
    seq_lens = getattr(attn_metadata, "seq_lens_cuda", None)
    if kv_lens is None or seq_lens is None:
        return None
    if num_seqs is None:
        num_seqs = attn_metadata.num_seqs - seq_start
    seq_slice = slice(seq_start, seq_start + num_seqs)
    per_seq_mask = kv_lens[seq_slice] == 0
    return torch.repeat_interleave(per_seq_mask, seq_lens[seq_slice], output_size=num_tokens)


def helix_gen_zero_kv_mask(
    attn_metadata: AttentionMetadata, num_tokens: int
) -> Optional[torch.Tensor]:
    """Per-token mask of the ``num_tokens`` generation tokens this CP rank owns no KV for.

    Used by the Helix generation combine (``_helix_post_process``) to make those rows a
    no-op. It only depends on the step's metadata, so callers may compute it ahead of the
    attention.
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


def _helix_sanitize_empty_kv(
    partial_o: torch.Tensor,
    softmax_stats: torch.Tensor,
    zero_kv_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Force zero-local-KV rows to a no-op contribution for the Helix combine.

    A CP rank that owns no KV blocks for a token attends to zero keys, so the
    attention kernel normalizes by a zero softmax sum and yields NaN. The combine
    weights each rank by sum * exp(max - global_max), so such rows must contribute
    softmax_stats of (-inf, 0) and a zeroed partial_o to act as a no-op. The rows
    are selected by zero_kv_mask, which is robust regardless of what the kernel
    wrote. Passing None disables sanitization.

    Args:
        partial_o: Partial attention output, shape [num_tokens, ...].
        softmax_stats: Per (token, head) (max, sum), shape [num_tokens, num_heads, 2].
        zero_kv_mask: Bool tensor of shape [num_tokens], True where this rank has
            zero local KV.
    """
    if zero_kv_mask is None:
        return partial_o, softmax_stats
    num_tokens = partial_o.shape[0]
    mask = zero_kv_mask[:num_tokens].view(-1, 1)
    # masked_fill overwrites masked rows regardless of their current (possibly NaN)
    # value and is CUDA-graph safe due to static shapes.
    partial_o = partial_o.masked_fill(mask, 0.0)
    sm_max = softmax_stats[..., 0].masked_fill(mask, float("-inf"))
    sm_sum = softmax_stats[..., 1].masked_fill(mask, 0.0)
    softmax_stats = torch.stack([sm_max, sm_sum], dim=-1)
    return partial_o, softmax_stats


def _helix_nccl_pre_alltoall_eager(
    partial_o: torch.Tensor,
    softmax_stats: torch.Tensor,
    zero_kv_mask: Optional[torch.Tensor],
    cp_size: int,
) -> List[torch.Tensor]:
    """Sanitize zero-local-KV rows and reformat into the alltoall send layout.

    ``_helix_sanitize_empty_kv`` followed by the transpose and split into the
    all-to-all layout. Both live in one compiled region
    so inductor folds the fill into the transposed store instead of writing
    ``partial_o`` and reading it straight back. Dynamo inlines the call, so the
    sanitize has exactly one definition and the fusion is unaffected.

    """
    partial_o, softmax_stats = _helix_sanitize_empty_kv(partial_o, softmax_stats, zero_kv_mask)
    chunks = []
    for t in (partial_o, softmax_stats):
        t = t.transpose(1, 0).contiguous()
        chunks.extend(torch.split(t, t.shape[0] // cp_size))
    return chunks


def _helix_nccl_post_alltoall_eager(gathered: List[torch.Tensor]) -> List[torch.Tensor]:
    """Reformat the gathered partials into the helix_post_process layout."""
    return [t.transpose(1, 2).contiguous() for t in gathered]


def _helix_post_process(
    partial_o: torch.Tensor,
    softmax_stats: torch.Tensor,
    mapping: Mapping,
    num_heads_tp_cp: int,
    value_dim: int,
    aux_stream: Optional[torch.cuda.Stream] = None,
    ln_events: Optional[list] = None,
    zero_kv_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Helix CP post-processing: all-to-all exchange and combine partial
    attention outputs across CP ranks.

    This is shared by both MHA (Attention) and MLA modules.  The only
    dimension that differs between the two callers is *value_dim*
    (``head_dim`` for MHA; for MLA ``kv_lora_rank``, or ``v_head_dim`` when
    the V up-projection runs before the exchange).

    zero_kv_mask marks tokens this CP rank owns no KV for; those rows are forced
    to a no-op contribution before the exchange, exactly once per backend:
      NCCL     in _helix_nccl_pre_alltoall_eager, with the reformat
      fifo v2  in the all-to-all sender, while the entry is in shared memory
      fifo v1  here, via _helix_sanitize_empty_kv

    When *aux_stream* and *ln_events* are provided the two
    ``.contiguous()`` calls in the FIFO-v1 path are overlapped on
    separate CUDA streams for better performance.

    The NCCL path's reformats run eagerly: the MLA layers meet new shapes in the middle of a multi-rank forward,
    where a compile loads a module while the other ranks already wait in the NCCL all-to-all, and that load can
    block every rank.
    """
    if mapping.cp_config.get("use_nccl_for_alltoall", True):
        chunks = _helix_nccl_pre_alltoall_eager(
            partial_o, softmax_stats, zero_kv_mask, mapping.cp_size
        )
        gathered = alltoall_helix(chunks, mapping.cp_group)
        gathered = _helix_nccl_post_alltoall_eager(gathered)
        return torch.ops.trtllm.helix_post_process(gathered[0], gathered[1], 1.0)
    else:
        # FIFO-based implementation using MNNVL workspace.
        helix = HelixAllToAllNative.get(mapping)
        num_tokens = partial_o.shape[0]
        cp_size = mapping.cp_size
        fifo_version = mapping.cp_config.get("fifo_version", 2)

        if fifo_version == 1:
            partial_o, softmax_stats = _helix_sanitize_empty_kv(
                partial_o, softmax_stats, zero_kv_mask
            )

            def reshape_o():
                return (
                    partial_o.view(num_tokens, cp_size, num_heads_tp_cp, value_dim)
                    .transpose(1, 2)
                    .contiguous()
                )

            def reshape_s():
                return (
                    softmax_stats.view(num_tokens, cp_size, num_heads_tp_cp, 2)
                    .transpose(1, 2)
                    .contiguous()
                )

            if aux_stream is not None and ln_events is not None:
                partial_o, softmax_stats = maybe_execute_in_parallel(
                    reshape_o,
                    reshape_s,
                    ln_events[0],
                    ln_events[1],
                    aux_stream,
                )
            else:
                partial_o = reshape_o()
                softmax_stats = reshape_s()

            partial_o_out, softmax_stats_out = helix.alltoall_native(partial_o, softmax_stats)
            return torch.ops.trtllm.helix_post_process_native(
                partial_o_out, softmax_stats_out, 1.0, 2
            )
        else:
            # fifo_v2: one entry is one token, and the sender already streams
            # every byte through shared memory, so the sanitize rides along
            # there instead of 6 separate elementwise kernels (~40% of the block).
            partial_o = partial_o.view(num_tokens, cp_size, num_heads_tp_cp * value_dim)
            softmax_stats = softmax_stats.view(num_tokens, cp_size, num_heads_tp_cp * 2)
            partial_o_out, softmax_stats_out = helix.alltoall_native(
                partial_o,
                softmax_stats,
                None if zero_kv_mask is None else zero_kv_mask[:num_tokens],
            )
            gathered_o = partial_o_out.view(num_tokens, cp_size, num_heads_tp_cp, value_dim)
            gathered_stats = softmax_stats_out.view(num_tokens, cp_size, num_heads_tp_cp, 2)
            return torch.ops.trtllm.helix_post_process_native(gathered_o, gathered_stats, 1.0, 1)


def _setup_helix_mappings(
    self: "torch.nn.Module",
    model_config: ModelConfig,
    cfg: "PretrainedConfig",
    spec_config: Optional["DecodingBaseConfig"],
) -> None:
    """Validate helix preconditions and stage the dual-mapping swap.

    DeepseekV3 pattern: the MLA layers keep the CP original; everything
    else is built against the repurposed mapping (CP ranks become TP
    ranks). Sets ``mapping_with_cp`` (restored after construction) and
    ``_repurposed_tp_mapping`` (load_weights shard selection); both stay
    None outside helix.
    """
    self.mapping_with_cp = None
    self._repurposed_tp_mapping = None
    if not model_config.mapping.has_cp_helix():
        return
    # Helix + attention DP = helix_a2a: KDA keeps head-TP state across the
    # CP group (every CP rank holds the same requests) and only the MoE
    # layers run attention-DP on ceil(T / cp) token rows each.
    if model_config.mapping.enable_attention_dp and model_config.mapping.tp_size != 1:
        raise NotImplementedError(
            "Kimi K3 helix with enable_attention_dp requires tp_size=1 "
            f"(got tp_size={model_config.mapping.tp_size}): KDA and MLA are "
            "head-TP over the CP group, so there is no request-level DP to add."
        )
    if spec_config is not None:
        # Standalone block drafters use the V2 superblock ledger for
        # page ownership and CTX-to-GEN transfer.
        if not (spec_config.spec_dec_mode.is_dspark() or spec_config.spec_dec_mode.is_dflash()):
            raise ValueError(
                "Kimi K3 helix supports speculative decoding only with "
                f"DSpark or MLA DFlash2 (standalone drafter); got "
                f"{spec_config.decoding_type!r}."
            )
        # The SpeculationGate acceptance-rate trip permanently disables
        # speculation mid-flight while enable_spec_decode stays True;
        # in-flight helix requests then fall into the plain generation
        # loop whose position math (total_input_len_cp +
        # py_decoding_iter - 1) is stale once any draft token was
        # accepted -> silently wrong RoPE positions and KV slots. Reject
        # the trip wires until that loop is helix-group aware.
        if (
            spec_config.acceptance_rate_window_size is not None
            or spec_config.acceptance_rate_threshold is not None
        ):
            raise ValueError(
                "Kimi K3 helix does not support the speculation "
                "acceptance-rate gate (acceptance_rate_window_size / "
                "acceptance_rate_threshold): dynamically disabling "
                "speculation mid-flight leaves helix requests on a "
                "single-token position formula."
            )
        # max_concurrency is the same trip wire by another name: the
        # drafter re-evaluates should_use_spec_decode on every scheduling
        # iteration and flips enable_spec_decode off as soon as the active
        # batch exceeds the cap. In-flight helix requests then take the
        # plain generation loop, whose position formula counts ITERATIONS
        # (total_input_len_cp + py_decoding_iter - 1) rather than
        # committed tokens, so it is stale by however many draft tokens
        # were accepted -- a wrong RoPE position, and across a ledger page
        # boundary a KV write to the wrong CP rank. Mirror the drafter's
        # own "unset" test (Drafter.should_use_spec_decode returns True
        # when max_concurrency is None) so an unset value is not rejected.
        if spec_config.max_concurrency is not None:
            raise ValueError(
                "Kimi K3 helix does not support the speculation "
                "concurrency cutoff (max_concurrency): disabling "
                "speculation above the cap leaves in-flight helix "
                "requests on a position formula that assumes one "
                "committed token per iteration, which accepted draft "
                "tokens break."
            )
        # draft_len_schedule is the user-facing alternative to
        # max_concurrency (llm_args rejects setting both) and reaches the
        # same end by a route that does not go through
        # should_use_spec_decode at all: py_executor turns speculation off
        # directly once the schedule yields draft_len 0 for the active
        # batch size. Guarding only max_concurrency would leave this door
        # open. Skip the schedule that llm_args synthesized from
        # max_concurrency, so a config that set only that field raises the
        # message above naming the field the user actually wrote.
        if (
            spec_config.draft_len_schedule is not None
            and not spec_config._translated_from_max_concurrency
        ):
            raise ValueError(
                "Kimi K3 helix does not support the dynamic draft-length "
                "schedule (draft_len_schedule): a batch size past the "
                "last entry drops the draft length to 0 and turns "
                "speculation off mid-run, leaving in-flight helix "
                "requests on a position formula that assumes one "
                "committed token per iteration."
            )
    cp = model_config.mapping.cp_size
    repurposed_tp = model_config.mapping.tp_size * cp
    if cfg.num_attention_heads % repurposed_tp != 0:
        raise ValueError(
            f"Kimi K3 helix requires tp_size*cp_size ({repurposed_tp}) "
            f"to divide the MLA head count ({cfg.num_attention_heads})."
        )
    # MoE splits apply to the repurposed tp*cp group (helix
    # moe_world_size = tp*cp); default EP-only. The Mapping constructor
    # skips its product check when both sizes are 1, so validate here.
    moe_ep = repurposed_tp
    if model_config.mapping.moe_tp_ep_user_specified:
        moe_tp = model_config.mapping.moe_tp_size
        moe_ep = model_config.mapping.moe_ep_size
        if moe_tp * moe_ep != repurposed_tp:
            raise ValueError(
                f"Kimi K3 helix: moe_tensor_parallel_size ({moe_tp}) x "
                f"moe_expert_parallel_size ({moe_ep}) must equal "
                f"tp_size*cp_size ({repurposed_tp}): MoE runs on the "
                "repurposed tp*cp group."
            )
    if cfg.num_experts and cfg.num_experts % moe_ep != 0:
        raise ValueError(
            f"Kimi K3 helix requires the MoE EP size ({moe_ep}) to "
            f"divide the routed expert count ({cfg.num_experts}): each "
            "EP rank of the repurposed tp*cp group holds whole experts."
        )
    self.mapping_with_cp = copy.deepcopy(model_config.mapping)
    repurposed = model_config.mapping.repurpose_helix_cp_to_tp()
    # repurpose passes resolved moe sizes, which the Mapping constructor
    # mistakes for user-specified values; restore the flag.
    repurposed.moe_tp_ep_user_specified = self.mapping_with_cp.moe_tp_ep_user_specified
    # load_weights shard selection must use this tp_rank; the restored
    # CP original's tp_rank is 0 on every rank.
    self._repurposed_tp_mapping = repurposed
    model_config._frozen = False
    model_config.mapping = repurposed
    # Side-channel for the MLA layers (avoids threading a kwarg through
    # every intermediate signature).
    model_config._helix_mapping_with_cp = self.mapping_with_cp
    model_config._frozen = True


class HelixTargetCombine(OpWrapper):
    """Drop padded query heads, exchange CP partials, and return this rank's head shard."""

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
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


class K31HelixDSAMetadata(HelixDSAMetadataMixin, DSAtrtllmAttentionMetadata):
    """The DSA metadata of the general target under Helix CP: the engine's (``DSAtrtllmAttentionMetadata``), which the
    engine attention backend of the Helix MLA layers reads, with Helix CP (``HelixDSAMetadataMixin``) and, by
    lifecycle point:

    * prepare input: ``_helix_prepare_step``, the step preparation on the rank-local KV lengths, and
      ``_helix_plan_verify_rows``, a verify step's per-token rows;
    * every KV-length refresh (after the input preparation's corrections and before each draft step):
      ``_apply_helix_gen_ownership``, the generation rows' indexer-K slots on the owning rank only.

    The engine methods it overrides only adapt to those methods, and are the engine's without Helix: ``prepare``,
    ``prepare_for_spec_decode`` (with ``use_cute_dsl_paged_mqa_logits``) and ``on_update_kv_lens``.
    """

    #: Set while ``prepare_for_spec_decode`` plans a Helix verify step, whose rows the CuTe DSL paged logits cannot
    #: score.
    _dsl_logits_off = False

    # --- prepare input ---

    def _helix_prepare_step(self) -> None:
        """The engine's step preparation (``DSAtrtllmAttentionMetadata.prepare``) on the rank-local KV lengths of
        ``_helix_local_kv_lens`` instead of the engine's Helix lengths, then the Helix sparse rows."""
        self._validate_ragged_verification()
        if self.sparse_offload_state is not None:
            self._validate_dsa_sparse_offload_batch()
            self.kv_cache_manager.prepare_sparse_batch(self.request_ids)
        TrtllmAttentionMetadata.prepare(self)
        self._invalidate_pool_view_cache()

        # Get kv lengths
        assert self.kv_cache_params.use_cache is True, "DSA requires use_cache to be True"
        cached_token_lens = torch.tensor(
            self.kv_cache_params.num_cached_tokens_per_seq,
            dtype=torch.int,
            device="cpu",
        )
        kv_lens = self._helix_local_kv_lens(cached_token_lens)

        kv_lens = self._prepare_device_window_kv_lens(kv_lens)

        # For mla_rope_append_paged_kv_assign_q
        self.prepare_for_mla_rope_append(cached_token_lens, kv_lens)

        # Prepare to support skip indexer
        self.prepare_for_skip_indexer(kv_lens)

        # For indices conversion
        self.prepare_for_indices_conversion()

        # For indexer k cache
        self.prepare_for_indexer_k_cache()

        # For spec decode
        self.prepare_for_spec_decode(kv_lens)

        # Prepare metadata for indexer
        Indexer.prepare(metadata=self)

        if self.sparse_offload_state is not None:
            self.kv_cache_manager.prepare_sparse_offload(
                self.sparse_offload_state, self.request_ids, beam_width=self.beam_width
            )
        self._prepare_helix_sparse_rows(kv_lens)

    def prepare(self):
        if not self.enable_helix:
            super().prepare()
            return
        self._helix_prepare_step()

    def _helix_plan_verify_rows(self, kv_lens: torch.Tensor) -> None:
        """A Helix verify step's per-token (expanded) rows, when the engine's planning chose none: each verify token
        has its own rank-local KV bound (``helix_kv_bounds``), which the in-kernel next_n stagger cannot express. The
        rows start at the step's lengths and block table; the ownership pass sets their lengths and schedule from the
        bounds."""
        if self.use_expanded_buffers_for_mtp:
            return
        self.use_expanded_buffers_for_mtp = True
        gen_kv_lens = self.get_indexer_kv_lens(kv_lens[self.num_contexts : self.num_seqs])
        gen_kv_lens_expanded, num_tokens = self.expand_per_gen_token(gen_kv_lens)
        self.kv_lens_expanded_host[:num_tokens].copy_(gen_kv_lens_expanded)
        self.kv_lens_expanded_cuda[:num_tokens].copy_(
            self.kv_lens_expanded_host[:num_tokens], non_blocking=True
        )
        if self.kv_cache_manager is not None and self.num_generations > 0:
            max_len = self.host_indexer_k_cache_block_offsets.shape[1]
            gen_blocks = self.host_indexer_k_cache_block_offsets[
                self.num_contexts : self.num_seqs, :max_len
            ]
            expanded_blocks, _ = self.expand_per_gen_token(gen_blocks, dim=0)
            self.host_block_table_expanded[:num_tokens, :max_len].copy_(expanded_blocks)
            self.block_table_expanded[:num_tokens].copy_(
                self.host_block_table_expanded[:num_tokens], non_blocking=True
            )
            self.block_table_expanded.clamp_(min=0)

    @property
    def use_cute_dsl_paged_mqa_logits(self) -> bool:
        return not self._dsl_logits_off and super().use_cute_dsl_paged_mqa_logits

    def prepare_for_spec_decode(self, kv_lens: torch.Tensor):
        """The engine's verify planning; a Helix verify step plans without the CuTe DSL paged logits, which cannot
        score per-token bounds, and on per-token rows (``_helix_plan_verify_rows``)."""
        if not (self.enable_helix and self._helix_spec_tokens_valid):
            super().prepare_for_spec_decode(kv_lens)
            return
        self._dsl_logits_off = True
        try:
            super().prepare_for_spec_decode(kv_lens)
        finally:
            self._dsl_logits_off = False
        self._helix_plan_verify_rows(kv_lens)

    # --- every KV-length refresh ---

    def on_update_kv_lens(self, skip_indexer_schedule: bool = False) -> None:
        super().on_update_kv_lens(skip_indexer_schedule=skip_indexer_schedule)
        if self.enable_helix and self.kv_cache_manager is not None and self.num_generations > 0:
            self._apply_helix_gen_ownership()

    def _apply_helix_gen_ownership(self) -> None:
        """Give generation rows this CP rank's indexer-K slot, or -1 (skipped) for tokens another rank owns; verify
        groups score one row per token against ``helix_kv_bounds``. Device-only with no host sync, so it can be
        captured."""
        num_ctx_tokens = self.num_ctx_tokens
        num_gen_tokens = self.num_tokens - num_ctx_tokens
        if num_gen_tokens <= 0:
            return
        if self._helix_spec_tokens_valid:
            local = self.helix_local_slots[:num_gen_tokens].to(torch.int64)
            owned = local >= 0
            req_indices = self.req_idx_per_token[num_ctx_tokens : self.num_tokens].to(torch.int64)
        else:
            # One token per generation row; the active rank appends it locally.
            gen_kv_lens = self.kv_lens_cuda[self.num_contexts : self.num_seqs].to(torch.int64)
            local = gen_kv_lens - 1
            owned = ~self.helix_is_inactive_rank[: self.num_generations]
            req_indices = torch.arange(
                self.num_contexts, self.num_seqs, dtype=torch.int64, device=local.device
            )
        index_head_dim = self.kv_cache_manager.index_head_dim
        use_fp4 = getattr(self.kv_cache_manager, "use_fp4", False)
        fp8_indices, scale_indices = compute_slot_mappings(
            local.clamp(min=0),
            self.indexer_k_cache_block_offsets,
            req_indices,
            index_head_dim,
            self._tokens_per_block,
            self.indexer_quant_block_size,
            data_bytes_per_token=index_head_dim // 2 if use_fp4 else index_head_dim,
        )
        gen_rows = slice(num_ctx_tokens, self.num_tokens)
        self.slot_mapping_fp8[gen_rows] = torch.where(owned, fp8_indices, -1)
        self.slot_mapping_scale[gen_rows] = torch.where(owned, scale_indices, -1)

        if self._helix_spec_tokens_valid and self.use_expanded_buffers_for_mtp:
            # A rank may see no KV for a token (bound 0); scoring keeps at least
            # one row slot so the schedule never sees an all-empty batch, and
            # the indexer drops those rows' candidates (helix_spec_row_kv_lens).
            bounds = self.helix_kv_bounds[:num_gen_tokens]
            self.kv_lens_expanded_cuda[:num_gen_tokens].copy_(bounds.clamp(min=1))
            get_paged_mqa_logits_metadata(
                self.kv_lens_expanded_cuda[:num_gen_tokens].view(-1, 1),
                SCHEDULE_BLOCK_KV,
                self.num_sms,
                self.scheduler_metadata_buffer_expanded,
            )


class K31HelixDSABackend(DSATrtllmAttention):
    """The DSA backend class the general targets declare under Helix CP (``RuntimeHooks.attention_backend_cls``): the
    Helix DSA metadata (:class:`K31HelixDSAMetadata`); every other member is the engine backend class's, whose instances
    the MLA layers attend through (:func:`create_helix_attention`)."""

    Metadata = K31HelixDSAMetadata


def setup_helix_mappings(model, model_config, text_config, spec_config) -> None:
    """The Helix mapping swap of the Kimi K3 causal LM (:func:`_setup_helix_mappings`), on ``model`` (the target's
    causal LM).

    It checks the Helix preconditions, swaps ``model_config.mapping`` for
    the repurposed one (CP ranks become TP ranks: every layer but the MLA attention is built on it) and records the CP
    original in ``model.mapping_with_cp`` (restored after construction) and ``model_config._helix_mapping_with_cp``
    (read by the MLA layers), and the repurposed one in ``model._repurposed_tp_mapping`` (the weight loader's TP rank).
    """
    _setup_helix_mappings(model, model_config, text_config, spec_config)


def create_helix_attention(
    backend_name: str, layer_idx: int, num_heads: int, head_dim: int, **kwargs
):
    """The attention of a Helix MLA layer: the engine's backend for the layer's DSA parameters
    (``attention.backends.utils.create_attention``), which attends over this rank's CP-owned pages.

    Adapted to the family MLA in three ways. K3.1 is NoPE: the backend's RoPE table holds the identity and keeps it
    when it grows (``_ensure_rope_table_size``, also bound as ``ensure_rope_table_size``, the name the MLA's generation
    RoPE calls). Its indexer takes the rotated channels and key-norm epsilon of the family DSA parameters
    (:func:`_take_family_indexer_settings`), the family DSA ops' calls (:func:`_take_family_indexer_calls`) and the
    Helix decode Top-K (:func:`_take_helix_decode_top_k`).
    """
    backend = create_attention(backend_name, layer_idx, num_heads, head_dim, **kwargs)
    _install_identity_rope_table(backend)
    if getattr(backend, "indexer", None) is not None:
        _take_family_indexer_settings(backend.indexer, kwargs.get("sparse_params"))
        _take_family_indexer_calls(backend.indexer)
        _take_helix_decode_top_k(backend.indexer)
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
    # The targets' MLA generation RoPE calls ``mqa.ensure_rope_table_size`` (the family DSA backend's name).
    backend.ensure_rope_table_size = identity_preserving_resize


def _take_family_indexer_settings(indexer, sparse_params) -> None:
    """The indexer settings the family DSA parameters (``K31DSAParams``) state, on the engine indexer, which is built
    with the MLA's rotated channels (``qk_rope_head_dim``) and DeepSeek-V3.2's key-norm epsilon (1e-6).

    ``index_k_norm_eps``, when set, is the key LayerNorm's epsilon. ``index_rope_dim`` 0 is a position-free indexer
    (K3.1's): no RoPE, and :func:`_position_free_qk_projection` passes each head's query and key unrotated, as two
    halves, to the fused concatenate + quantize. Any other set value must be the MLA's own rotated channels.
    """
    k_norm_eps = getattr(sparse_params, "index_k_norm_eps", None)
    if k_norm_eps is not None:
        indexer.k_norm.variance_epsilon = k_norm_eps
    rope_dim = getattr(sparse_params, "index_rope_dim", None)
    if rope_dim == 0:
        indexer.rope_dim = 0
        indexer.rotary_emb = None
        indexer._qk_projection_and_rope = functools.partial(_position_free_qk_projection, indexer)
    elif rope_dim is not None and rope_dim != indexer.rope_dim:
        raise ValueError(
            f"index_rope_dim {rope_dim}: the Helix MLA indexer rotates no channels or the MLA's "
            f"{indexer.rope_dim}"
        )


def _position_free_qk_projection(
    indexer, qr: torch.Tensor, indexer_k: torch.Tensor, position_ids: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The engine indexer's ``_qk_projection_and_rope`` without RoPE: ``(q_pe, q_nope, k_pe, k_nope)``, the two
    halves of each projected query head and of the normed key. ``position_ids`` is not read."""
    q = indexer.wq_b(qr)
    k = indexer.k_norm(indexer_k)
    q = q.view(-1, indexer.n_heads, indexer.head_dim)
    half = indexer.head_dim // 2
    q_pe, q_nope = q.split([half, indexer.head_dim - half], dim=-1)
    k_pe, k_nope = k.split([half, indexer.head_dim - half], dim=-1)
    return q_pe, q_nope, k_pe, k_nope


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


def _take_helix_decode_top_k(indexer) -> None:
    """Give the engine indexer's decode Top-K the Helix merge of the CP ranks' selections.

    Under Helix each CP rank holds a slice of every sequence's KV cache, so its indexer only ranks local tokens. The
    global Top-K is a subset of the union of the local Top-Ks, so gathering every rank's candidate scores and taking
    the Top-K of that union reproduces the selection of an unsharded indexer (``rubin_merge_helix_topk``). On a decode
    call the indexer's ``top_k`` (``_helix_decode_top_k``):

    * scores a Helix verify step one row per token against its rank-local bound (``helix_spec_row_kv_lens``), as
      ragged rows are scored, and drops the candidates of a token this rank holds no KV for;
    * keeps this rank's share of each row's global Top-K, kept entries first and the rest -1;
    * presents each row to the token-sparse MLA with exactly its kept entries (``set_helix_sparse_row_lens``).

    A Helix verify step's rows carry per-token bounds the CuTe DSL paged logits cannot take, so those calls score
    with DeepGEMM's paged logits on the per-token schedule. Prefill calls are the engine's. ``forward_from_projected``
    hands each call's metadata to the Top-K and sets the logits kernel for the call.
    """
    top_k = indexer.top_k
    select = top_k.forward
    predict = indexer.forward_from_projected
    dsl_logits = indexer.use_cute_dsl_paged_mqa_logits
    call = {"metadata": None}

    def forward_from_projected(metadata, hidden_states, indexer_intermediates, is_generation=None):
        num_gen_tokens = metadata.num_tokens - metadata.num_ctx_tokens
        indexer.use_cute_dsl_paged_mqa_logits = (
            dsl_logits and metadata.helix_spec_row_kv_lens(num_gen_tokens) is None
        )
        call["metadata"] = metadata
        try:
            return predict(metadata, hidden_states, indexer_intermediates, is_generation)
        finally:
            call["metadata"] = None
            indexer.use_cute_dsl_paged_mqa_logits = dsl_logits

    top_k.forward = functools.partial(_helix_decode_top_k, select, call)
    indexer.forward_from_projected = forward_from_projected


def _helix_decode_top_k(
    select,
    call: dict,
    scores: torch.Tensor,
    output_indices: torch.Tensor,
    *,
    is_prefill: bool,
    row_kv_lens: Optional[torch.Tensor] = None,
    next_n: int = 1,
    **kwargs,
) -> torch.Tensor:
    """The engine Top-K (``select``) of one indexer call, with the Helix bounds and merge on a decode call (see
    :func:`_take_helix_decode_top_k`). ``scores`` are the call's local indexer logits, ``output_indices`` its
    ``[num_gen_tokens, k]`` int32 rows."""
    metadata = call["metadata"]
    if is_prefill or metadata is None:
        return select(
            scores,
            output_indices,
            is_prefill=is_prefill,
            row_kv_lens=row_kv_lens,
            next_n=next_n,
            **kwargs,
        )
    bounds = metadata.helix_spec_row_kv_lens(output_indices.shape[0])
    if bounds is not None:
        # Helix verify tokens carry per-token rank-local bounds, like ragged rows.
        row_kv_lens, next_n = bounds, 1
    result = select(
        scores, output_indices, is_prefill=False, row_kv_lens=row_kv_lens, next_n=next_n, **kwargs
    )
    mapping = metadata.mapping
    if metadata.enable_helix and mapping.cp_size > 1:
        if bounds is not None:
            # Tokens that see no KV on this rank were scored over a placeholder slot; they must not offer candidates.
            output_indices.masked_fill_((bounds == 0).unsqueeze(1), -1)
        valid = output_indices >= 0
        candidates = torch.gather(scores.float(), 1, output_indices.clamp(min=0).long())
        candidates = candidates.masked_fill(~valid, float("-inf"))
        gathered = cp_allgather(candidates.unsqueeze(0), mapping, dim=0)
        rubin_merge_helix_topk(gathered, output_indices, mapping.cp_rank)
        metadata.set_helix_sparse_row_lens(output_indices)
    return result


def helix_mla_attention(mla, q: torch.Tensor, metadata, **kwargs) -> torch.Tensor:
    """``mla``'s absorbed attention on a Helix rank: ``[rows, num_heads_tp_cp * kv_lora_rank]``.

    The engine backend (``mla.mqa``, :func:`create_helix_attention`) attends over this rank's CP-owned pages and
    returns its softmax statistics; :data:`helix_target_combine` exchanges the CP partials and returns this rank's head
    shard. ``kwargs`` are the backend's forward arguments (``AttentionForwardArgs`` fields).
    """
    stats = torch.empty((q.shape[0], mla.num_heads_attn, 2), device=q.device, dtype=torch.float32)
    # A token-sparse MLA call: under Helix the backend attends one row per generation token, each as long as this
    # rank's kept Top-K entries (``K31HelixDSAMetadata.token_major_gen_view``).
    with metadata.token_sparse_call():
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
