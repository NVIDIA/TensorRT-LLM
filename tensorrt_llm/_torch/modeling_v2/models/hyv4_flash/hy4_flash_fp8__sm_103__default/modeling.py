# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HY4-Flash text core (42 gated-MLA layers, iHC streams, Engram, latent MoE) and its shell.

Context is unabsorbed MLA (explicit K/V) and dense over the keys it is given: a sparse layer's
context attends the whole chunk rather than its top-k. Decode is absorbed MLA over the paged pool;
only sparse layers pass a selection.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modeling_v2.catalog.attention.load_paged_kv_cache_for_mla import (
    load_paged_kv_cache_for_mla,
)
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks
from tensorrt_llm._utils import mpi_disabled

from .. import cache_manager
from ..engram_host import EngramHostTable
from . import vision as _vision
from . import weights as _weights
from .dims import dims
from .ops.activation.flashinfer_silu_and_mul__r18 import flashinfer_silu_and_mul
from .ops.activation.gated_attn_output import gated_attn_output
from .ops.attention.convert_req_index_to_global import convert_req_index_to_global
from .ops.attention.indexer_pool import indexer_pool
from .ops.attention.indexer_select import indexer_select
from .ops.attention.mla_rope_append_paged_kv_assign_q__r14 import mla_rope_append_paged_kv_assign_q
from .ops.attention.mla_rope_generation__mla_decode_prologue import mla_rope_generation
from .ops.attention.thop_attention__mla_windowed import MLA_CONTEXT, MLA_GENERATION, thop_attention
from .ops.comm.allgather__r20 import allgather
from .ops.comm.allreduce__r19 import PLAIN, allreduce, allreduce_group_args
from .ops.comm.reducescatter__r21 import reducescatter
from .ops.engram.engram_branch import engram_branch
from .ops.engram.engram_fp8_roundtrip import engram_fp8_roundtrip
from .ops.gemm.block_fp8_gemm import block_fp8_gemm
from .ops.gemm.bmm_out__r16 import bmm_out
from .ops.gemm.cublas_mm__r11 import cublas_mm
from .ops.mhc.ihc_mapping import ihc_mapping
from .ops.moe.fp8_block_scale_moe_runner import fp8_block_scale_moe_runner
from .ops.norm.flashinfer_rmsnorm__r12 import flashinfer_rmsnorm
from .ops.quantization.fp8_quantize_1x128 import fp8_quantize_1x128

#: The engine's attention arguments this target does not drive.
_INERT = dict(
    output_sf=None,
    max_context_q_len_override=None,
    cache_indirection=None,
    kv_scale_orig_quant=None,
    kv_scale_quant_orig=None,
    out_scale=None,
    rotary_inv_freq=None,
    block_ids_per_seq=None,
    beam_width=1,
    mask_type=1,
    position_embedding_type=8,
    rope_scale_type=0,
    rope_scale=1.0,
    rope_short_m_scale=1.0,
    rope_long_m_scale=1.0,
    use_paged_context_fmha=False,
    chunked_prefill_buffer_batch_size=None,
    rope_append=True,
    mrope_rotary_cos_sin=None,
    mrope_position_deltas=None,
    helix_position_offsets=None,
    helix_is_inactive_rank=None,
    attention_chunk_size=None,
    softmax_stats_tensor=None,
    is_spec_decoding_enabled=False,
    use_spec_decoding=False,
    is_spec_dec_tree=False,
    spec_decoding_generation_lengths=None,
    spec_decoding_position_offsets=None,
    spec_decoding_packed_mask=None,
    spec_decoding_bl_tree_mask_offset=None,
    spec_decoding_bl_tree_mask=None,
    spec_bl_tree_first_sparse_mask_offset_kv=None,
    sparse_kv_indices=None,
    sparse_kv_offsets=None,
    sparse_attn_offsets=None,
    sparse_attn_indices_block_size=0,
    sparse_attn_kv_lens=None,
)


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return flashinfer_rmsnorm(x, weight, eps)


def _hc_residual(post: torch.Tensor, block: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """HF's `post * block + residual`, carried in fp32 and rounded once."""
    return (post.float().unsqueeze(-1) * block.float().unsqueeze(-2) + residual.float()).to(
        residual.dtype
    )


def _recent_rows(positions: torch.Tensor, width: int) -> torch.Tensor:
    """`[T, width]` int32: each row's own key position and the `width - 1` before it, `-1` past 0.

    The sparse decode kernel assumes a valid prefix followed by `-1` padding.
    """
    back = torch.arange(width, dtype=torch.int32, device=positions.device)
    return (positions.to(torch.int32)[:, None] - back[None, :]).clamp(min=-1)


def _row_batch(md) -> torch.Tensor:
    """`[T]`: the request each row of the step belongs to."""
    return md.req_idx_per_token[: int(md.num_tokens)].to(torch.int64)


def _picks_rows(md) -> torch.Tensor:
    """`[N]` int64: the rows whose selections the draft layers read.

    Every generation row (which verify row is accepted is not known yet) and the last row of
    each context request.
    """
    rows = _row_batch(md)
    total = int(md.num_tokens)
    num_ctx = int(md.num_contexts)
    tc = int(md.num_ctx_tokens) if num_ctx else 0
    picked = []
    if tc:
        counts = torch.zeros(num_ctx, dtype=torch.int64, device=rows.device)
        counts.scatter_add_(0, rows[:tc], torch.ones(tc, dtype=torch.int64, device=rows.device))
        picked.append(counts.cumsum(0) - 1)
    if total > tc:
        picked.append(torch.arange(tc, total, dtype=torch.int64, device=rows.device))
    if not picked:
        return torch.empty(0, dtype=torch.int64, device=rows.device)
    return torch.cat(picked)


def _ngram_tokens(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    row_batch: torch.Tensor,
    max_ngram: int,
    pad_id: int,
) -> torch.Tensor:
    """`[T, max_ngram]`: column `d` is the token at position `p - d` from this step, else pad.

    A predecessor counts only when it is the previous row of the same request at `p - d`.
    """
    tokens = input_ids.to(torch.int64).view(-1)
    positions = position_ids.to(torch.int64).view(-1)
    rows = row_batch.to(torch.int64)
    columns = [tokens]
    for offset in range(1, max_ngram):
        previous = torch.full_like(tokens, pad_id)
        previous_position = torch.full_like(positions, -1)
        previous_row = torch.full_like(rows, -1)
        if tokens.numel() > offset:
            previous[offset:] = tokens[:-offset]
            previous_position[offset:] = positions[:-offset]
            previous_row[offset:] = rows[:-offset]
        adjacent = (
            (positions >= offset)
            & (previous_position == positions - offset)
            & (previous_row == rows)
        )
        columns.append(torch.where(adjacent, previous, pad_id))
    return torch.stack(columns, dim=-1)


class EngramHistory:
    """Per-request token lookback across steps, so a decode row hashes its real n-gram.

    A request's ring row is its first encoded KV block id (stable and unique while live; graph
    padding rows have zero KV length) modulo a prime; each row records its key and each cell its
    position, so a collision, a recycled row or an overwritten cell reads pad, never another
    request's token. Device ops only, so a captured decode graph replays it.
    """

    #: Prime, so the encoded block stride cannot alias it.
    ROWS = 1021

    def __init__(self, core: "Hyv4FlashCore") -> None:
        self.core = core
        self._tok: torch.Tensor | None = None
        self._pos: torch.Tensor | None = None
        self._key: torch.Tensor | None = None

    def _state(
        self, device: torch.device, width: int, pad_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tok = self._tok
        if tok is None or tok.device != device or tok.shape[1] < width:
            self._tok = torch.full((self.ROWS, width), pad_id, dtype=torch.int64, device=device)
            self._pos = torch.full((self.ROWS, width), -1, dtype=torch.int64, device=device)
            self._key = torch.full((self.ROWS,), -1, dtype=torch.int64, device=device)
        return self._tok, self._pos, self._key

    def lookback(
        self,
        md,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        row_batch: torch.Tensor,
        pad_id: int,
    ) -> torch.Tensor | None:
        """`[T, max_ngram - 1]` tokens before each row (pad where unknown); then records this step."""
        d = self.core.dims
        depth = d.engram_max_ngram - 1
        offsets = getattr(md, "kv_cache_block_offsets", None)
        if depth <= 0 or offsets is None:
            return None
        tokens = input_ids.to(torch.int64).view(-1)
        positions = position_ids.to(torch.int64).view(-1)
        rows = row_batch.to(torch.int64)
        tok, pos, key_of = self._state(tokens.device, int(d.max_num_tokens), pad_id)
        width = tok.shape[1]
        own = offsets[0][rows, 0, 0].to(torch.int64)
        live = md.kv_lens_cuda_runtime[rows] > 0
        own = torch.where(live, own, torch.full_like(own, -1))
        key = own.clamp(min=0) % self.ROWS
        columns = []
        for step in range(1, depth + 1):
            target = positions - step
            slot = target.clamp(min=0) % width
            cell = tok[key, slot]
            valid = (pos[key, slot] == target) & (key_of[key] == own) & (target >= 0)
            columns.append(torch.where(valid, cell, torch.full_like(cell, pad_id)))
        # Columns are position modulo the step token budget, so one step's rows never share a cell.
        slot = positions.clamp(min=0) % width
        tok[key, slot] = torch.where(live, tokens, tok[key, slot])
        pos[key, slot] = torch.where(live, positions, pos[key, slot])
        key_of.scatter_(0, key, torch.where(live, own, key_of[key]))
        return torch.stack(columns, dim=-1)


class Indexer:
    """One sparse layer's compressed indexer (HF `HYV4FlashCompressedIndexer`)."""

    def __init__(self, core: "Hyv4FlashCore", layer: int) -> None:
        self.core = core
        self.layer = layer
        self.d = core.dims

    def pool_and_select(
        self,
        q_resid: torch.Tensor,
        x: torch.Tensor,
        md,
        row_batch: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """`[T, index_topk]` request-local key positions, valid prefix then `-1`."""
        d, w, i = self.d, self.core.w, self.layer
        num_seqs = int(md.num_seqs)
        # HF's short circuit: at or below `index_topk` keys every key is visible. Host lengths:
        # a device read would sync, which graph capture forbids.
        if int(md.kv_lens_runtime[:num_seqs].max()) <= d.index_topk:
            return _recent_rows(positions, d.index_topk)
        token_keys = _rms(
            cublas_mm(x, w[f"l{i}_idx_token_key"].t()), w[f"l{i}_idx_token_key_norm"], d.rms_eps
        )
        # `indexer_pool` reads `[B, tokens, head_dim]`: lay the step's keys out per request,
        # zero-padded. Graph padding rows go to a trash slot (a boolean index would sync).
        rows = row_batch.to(torch.int64)
        live = (rows >= 0) & (rows < num_seqs)
        batch = rows.clamp(0, num_seqs - 1)
        per_seq = torch.zeros(num_seqs, dtype=torch.int64, device=token_keys.device)
        per_seq.scatter_add_(0, batch, live.to(torch.int64))
        starts = per_seq.cumsum(0) - per_seq
        slot = torch.arange(token_keys.shape[0], device=token_keys.device) - starts[batch]
        width = max(int(md.max_ctx_seq_len), 1)
        flat = token_keys.new_zeros(num_seqs * width + 1, token_keys.shape[-1])
        trash = torch.full_like(batch, num_seqs * width)
        dest = torch.where(live & (slot >= 0) & (slot < width), batch * width + slot, trash)
        flat.scatter_(0, dest.unsqueeze(1).expand(-1, token_keys.shape[-1]), token_keys)
        keys = flat[: num_seqs * width].view(num_seqs, width, token_keys.shape[-1])
        group_keys = indexer_pool(
            keys,
            w[f"l{i}_idx_kv_proj"],
            w[f"l{i}_idx_gate_proj"],
            w[f"l{i}_idx_ape"],
            w[f"l{i}_idx_key_norm"],
            d.rms_eps,
            kpool=d.index_kpool,
        )
        query = cublas_mm(q_resid, w[f"l{i}_idx_query"].t()).view(
            -1, d.index_n_heads, d.index_head_dim
        )
        head_weights = cublas_mm(x, w[f"l{i}_idx_head_weight"].t()).float() * d.index_scale
        packed, _, _ = indexer_select(
            query,
            group_keys,
            head_weights,
            batch,
            positions,
            ratio=d.index_kpool,
            index_topk=d.index_topk,
        )
        # The pool holds only this step's keys (no token-key cache yet), so a row whose prefix
        # is partly cached (every decode row, every chunk after the first) attends its most
        # recent `index_topk` keys instead.
        recent = _recent_rows(positions, d.index_topk)
        full = (slot + 1) >= (positions.to(torch.int64) + 1)
        return torch.where(full[:, None], packed, recent)


class EngramLayer:
    """One Engram layer: hashed n-gram rows gathered from a pinned host table, then a gated branch."""

    def __init__(self, core: "Hyv4FlashCore", layer: int) -> None:
        self.core = core
        self.layer = layer
        self.d = core.dims
        self.table: EngramHostTable = core.engram_tables[layer]

    def hash_ngrams(self, ngram_tokens: torch.Tensor) -> torch.Tensor:
        """`[T, num_heads]` global table rows (HF `hash_ngrams`, per-head row offsets added)."""
        d, w = self.d, self.core.w
        i = self.layer
        table = w[f"l{i}_engram_lookup"]
        ids = ngram_tokens.to(torch.int64)
        compressed = table[ids.clamp(max=d.engram_table_size - 1)]
        compressed = torch.where(
            ids < d.engram_table_size, compressed, table[d.engram_pad_id]
        ).long()
        mixed = compressed * self.core.hash_multipliers[i]
        orders = d.engram_max_ngram - 1
        per_order = d.engram_num_heads // orders
        rows = []
        for k in range(d.engram_num_heads):
            order = k // per_order
            h = mixed[:, 0]
            for offset in range(1, order + 2):
                h = torch.bitwise_xor(h, mixed[:, offset])
            rows.append((h % self.core.head_moduli[i][k]) + d.engram_head_offsets[i][k])
        return torch.stack(rows, dim=-1)

    def forward(
        self, streams: torch.Tensor, ngram_tokens: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """`[T, hc, hidden] -> [T, hc, hidden]`: the branch added to each stream."""
        d, w, i = self.d, self.core.w, self.layer
        rows = self.hash_ngrams(ngram_tokens)
        head_dim = d.engram_head_dim
        gathered = self.table.gather(rows.reshape(-1)).view(-1, d.engram_num_heads, head_dim)
        if d.engram_fp8:
            gathered = engram_fp8_roundtrip(gathered)
        # One RMSNorm per hash head (HF `lookup_per_head_norms`), not one over the concatenation.
        norms = w[f"l{i}_engram_head_norm"]
        gathered = torch.stack(
            [_rms(gathered[:, k], norms[k], d.rms_eps) for k in range(d.engram_num_heads)],
            dim=1,
        ).view(-1, d.engram_num_heads * head_dim)
        return engram_branch(
            streams,
            gathered,
            w[f"l{i}_engram_key_w"],
            w[f"l{i}_engram_key_b"],
            w[f"l{i}_engram_norm_key"],
            w[f"l{i}_engram_norm_query"],
            w[f"l{i}_engram_value_w"],
            w[f"l{i}_engram_value_b"],
            w[f"l{i}_engram_pre_gamma"],
            w[f"l{i}_engram_hc_scale"],
            mask,
        )


class LatentMoe:
    """Route on the hidden, run this rank's expert window on the latent, add the shared expert.

    Attention TP: every rank holds every row, so one allreduce completes the routed and the
    shared partials. Attention DP: the group's rows are gathered for the window and the routed
    sum is reduce-scattered back; the shared expert is whole and added locally, once.
    """

    def __init__(self, core: "Hyv4FlashCore", layer: int) -> None:
        self.core = core
        self.layer = layer
        self.d = core.dims

    def _routed(self, x: torch.Tensor) -> torch.Tensor:
        d, w, i = self.d, self.core.w, self.layer
        latent = cublas_mm(x, w[f"l{i}_pre_expert"].t())
        latent_q, latent_s = fp8_quantize_1x128(latent)
        routed = fp8_block_scale_moe_runner(
            routing_logits=cublas_mm(x, w[f"l{i}_router"].t()),
            routing_bias=w[f"l{i}_router_bias"],
            hidden_states=latent_q,
            hidden_states_scale=latent_s,
            gemm1_weights=w[f"l{i}_experts_gate_up"],
            gemm1_weights_scale=w[f"l{i}_experts_gate_up_scale"],
            gemm2_weights=w[f"l{i}_experts_down"],
            gemm2_weights_scale=w[f"l{i}_experts_down_scale"],
            num_experts=d.num_experts,
            top_k=d.top_k,
            num_fused_shared_experts=0,
            n_group=d.n_group,
            topk_group=d.topk_group,
            intermediate_size=d.moe_inter,
            local_expert_offset=d.expert_offset,
            local_num_experts=d.local_experts,
            routed_scaling_factor=d.routed_scale,
            routing_method_type=2,  # DeepSeekV3: sigmoid, bias add, top-k, renorm
            topk_weights=None,
            topk_ids=None,
            act_type=0,  # SwiGLU
            gemm1_clamp_limit=d.swiglu_limit,
            output=None,
            tune_max_num_tokens=d.max_num_tokens,
            use_dp=d.attention_dp,
        )
        return cublas_mm(routed, w[f"l{i}_post_expert"].t())

    def _shared(self, x: torch.Tensor) -> torch.Tensor:
        w, i = self.core.w, self.layer
        return cublas_mm(
            flashinfer_silu_and_mul(
                cublas_mm(
                    x,
                    torch.cat([w[f"l{i}_shared_gate"], w[f"l{i}_shared_up"]], dim=0).t(),
                )
            ),
            w[f"l{i}_shared_down"].t(),
        )

    def forward(self, x: torch.Tensor, md) -> torch.Tensor:
        d, w, i = self.d, self.core.w, self.layer
        if self.core._dp_exchange():
            sizes = self.core._dp_row_sizes(md, x.shape[0])
            union = allgather(x, sizes, self.core._dp_group, self.core._dp_pg)
            out = reducescatter(
                self._routed(union), sizes, self.core._dp_group, self.core._dp_pg
            ) + self._shared(x)
        else:
            out = self._routed(x) + self._shared(x)
            if d.moe_ep > 1 or d.mlp_tp > 1:
                out = self.core._allreduce(out)
        return _rms(out, w[f"l{i}_combine_output_norm"], d.rms_eps)


class MlaLayer:
    """One gated-MLA layer: `context_rows` (prefill) or `generation_partial` (decode), then `epilogue`.

    The host (the core or the drafter) provides `dims`, `w`, `gen_width`, `_pool_row` and
    `_allreduce`.
    """

    def __init__(self, core: "Hyv4FlashCore", layer: int) -> None:
        self.core = core
        self.layer = layer
        self.d = core.dims
        self.geom = core.dims.layers[layer]
        self.indexer = Indexer(core, layer) if self.geom.sparse else None
        #: Generation rows per request: 1, or the draft block width under speculation.
        self.gen_width = core.gen_width

    def _absorbed_q(self, q: torch.Tensor, q_roped_tail: torch.Tensor) -> torch.Tensor:
        """`[T, heads_local, 512 + 64]`: `q_nope @ k_b` with the roped tail appended."""
        d, w, i = self.d, self.core.w, self.layer
        q_nope = q[..., : d.nope]
        k_b = w[f"l{i}_k_b_proj"]
        out = torch.empty(
            q.shape[0],
            self.geom.heads_local,
            d.kv_lora + d.rope_dim,
            dtype=q.dtype,
            device=q.device,
        )
        bmm_out(q_nope.transpose(0, 1), k_b, out[:, :, : d.kv_lora].transpose(0, 1))
        out[:, :, d.kv_lora :] = q_roped_tail
        return out

    def _window(self, md) -> int:
        """The layer's window, else the pool's max sequence length (0 would mean a tiny window)."""
        return self.geom.window or self._pool_max(md)

    def _pool_max(self, md) -> int:
        return int(getattr(md, "max_seq_len", self.d.max_seq_len) or self.d.max_seq_len)

    def context_rows(
        self,
        q: torch.Tensor,
        latent: torch.Tensor,
        md,
        rope: dict,
    ) -> torch.Tensor:
        """Prefill: rope and append this chunk's latent rows, then unabsorbed attention.

        The context entry needs explicit K/V: the fused (absorbed) form leaves the output
        unwritten in a context call.
        """
        d, w, i = self.d, self.core.w, self.layer
        hl = self.geom.heads_local
        num_tokens = q.shape[0]
        # K/V span each request's cached prefix too; q and out hold this step's rows only.
        num_kv = int(md.host_total_kv_lens[0])
        roped = torch.empty(num_tokens, hl, d.qk_dim, dtype=q.dtype, device=q.device)
        roped.copy_(q)
        mla_rope_append_paged_kv_assign_q(
            roped.view(num_tokens, hl * d.qk_dim),  # roped in place
            latent,
            md.num_contexts,
            md.ctx_cached_token_indptr,
            md.ctx_kv_indptr,
            int(md.max_ctx_seq_len),
            rope["rotary_cos_sin"],
            hl,  # head_num
            d.nope,  # nope_size
            d.rope_dim,  # rope_size
            d.kv_lora,  # lora_size
            md.kv_cache_block_offsets,
            md.host_kv_cache_pool_pointers,
            md.host_kv_cache_pool_mapping,
            None,  # kv_scale_orig_quant
            0,  # residual_dim
            self.core._pool_row(md, i),  # layer_idx
            d.tokens_per_block,
            self._window(md),
            1,  # beam_width
            0,  # quant_mode
        )
        # Read back `[normed compressed_kv | rope(k_pe)]`: the roped K rope half comes from here.
        ckv, k_pe = load_paged_kv_cache_for_mla(
            out_dtype=q.dtype,
            num_contexts=md.num_contexts,
            num_ctx_kv_tokens=num_kv,
            max_ctx_kv_len=int(md.max_ctx_kv_len),
            cu_ctx_kv_lens=md.ctx_kv_indptr,
            kv_cache_block_offsets=md.kv_cache_block_offsets,
            host_kv_cache_pool_pointers=md.host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=md.host_kv_cache_pool_mapping,
            kv_scale_quant_orig=None,
            layer_idx=self.core._pool_row(md, i),
            kv_lora_rank=d.kv_lora,
            qk_rope_head_dim=d.rope_dim,
            tokens_per_block=d.tokens_per_block,
            attention_window_size=self._window(md),
            beam_width=1,
            quant_mode=0,
        )
        # V is a split view of the packed GEMM output: the context FMHA reads V's row stride
        # as the packed width.
        kv = cublas_mm(ckv, w[f"l{i}_kv_b_proj"].t())
        k = torch.empty(num_kv, hl, d.qk_dim, dtype=q.dtype, device=q.device)
        k[..., : d.nope].copy_(kv[:, : hl * d.nope].view(num_kv, hl, d.nope))
        k[..., d.nope :].copy_(k_pe.unsqueeze(1))
        out = torch.empty(num_tokens, hl * d.v_dim, dtype=q.dtype, device=q.device)
        thop_attention(
            q=roped.view(num_tokens, hl * d.qk_dim),
            k=k.view(num_kv, hl * d.qk_dim),
            v=kv[:, hl * d.nope :],
            output=out,
            is_fused_qkv=False,
            num_kv_heads=hl,  # context runs as MHA
            head_size=d.qk_dim,
            v_head_dim=d.v_dim,
            attention_input_type=MLA_CONTEXT,
            num_contexts=int(md.num_contexts),
            num_ctx_tokens=num_tokens,
            latent_cache=None,
            q_pe=None,
            sparse_attn_indices=None,
            num_sparse_topk=None,
            **self._step_args(md, rope),
            **_INERT,
        )
        return out

    def generation_partial(
        self,
        q: torch.Tensor,
        latent: torch.Tensor,
        md,
        rope: dict,
        packed: torch.Tensor | None,
    ) -> torch.Tensor:
        """Decode: rope the absorbed query and append the row, latent MQA, then `v_b` expansion."""
        d, w, i = self.d, self.core.w, self.layer
        num_tokens = q.shape[0]
        q_pe = q[..., d.nope :]
        absorbed = self._absorbed_q(q, q_pe)
        device = q.device
        # Written by the prologue, read back by the attention entry of the same step.
        cu_q = torch.empty(num_tokens + 1, dtype=torch.int32, device=device)
        cu_kv = torch.empty(num_tokens + 1, dtype=torch.int32, device=device)
        counter = torch.empty(1, dtype=torch.uint32, device=device)
        mla_rope_generation(
            absorbed.view(num_tokens, -1),  # fused_q, roped in place
            q_pe,
            latent,
            rope["rotary_cos_sin"],
            cu_q,
            cu_kv,
            counter,
            None,  # mla_bmm1_scale
            None,  # mla_bmm2_scale
            None,  # quant_q_buffer
            md.kv_lens_cuda_runtime,  # sequence_length
            md.kv_lens_runtime,  # host_past_key_value_lengths
            md.prompt_lens_cpu_runtime,  # host_context_lengths
            md.num_contexts,
            md.kv_cache_block_offsets,
            md.host_kv_cache_pool_pointers,
            md.host_kv_cache_pool_mapping,
            None,  # kv_scale_orig_quant
            None,  # kv_scale_quant_orig
            None,  # kv_cache_scale_orig_quant
            None,  # out_scale
            None,  # block_ids_per_seq
            [None, None],  # helix_tensor_params
            self.gen_width,  # predicted_tokens_per_seq
            self.core._pool_row(md, self.layer),  # layer_idx
            self.geom.heads_local,  # num_heads
            1,  # num_kv_heads
            d.lat_dim,  # head_size
            0,  # residual_dim
            d.tokens_per_block,
            self._window(md),
            1,  # beam_width
            0,  # quant_mode
            d.q_scaling,
            d.q_lora,
            d.kv_lora,
            d.nope,
            d.rope_dim,
            d.v_dim,  # v_head_dim
            True,  # rope_append
            None,  # kv_norm_weight
            d.rms_eps,  # kv_norm_eps
            False,  # precomputed_cu_seqlens
            False,  # precomputed_fmha_scheduler
            False,  # kv_only
            False,  # kv_done_elsewhere
            None,  # quant_scale_qkv
        )
        lat = torch.empty(
            num_tokens,
            self.geom.heads_local * d.kv_lora,
            dtype=q.dtype,
            device=q.device,
        )
        thop_attention(
            q=absorbed.contiguous(),
            k=None,
            v=None,
            output=lat,
            is_fused_qkv=True,
            num_kv_heads=1,
            head_size=d.lat_dim,
            v_head_dim=d.kv_lora,
            attention_input_type=MLA_GENERATION,
            num_contexts=int(md.num_contexts),
            num_ctx_tokens=0,
            sparse_attn_indices=packed,
            num_sparse_topk=None if packed is None else int(packed.shape[1]),
            latent_cache=latent,
            q_pe=q_pe,
            cu_q_seqlens=cu_q,
            cu_kv_seqlens=cu_kv,
            fmha_scheduler_counter=counter,
            **self._generation_args(md, rope, packed),
            **_INERT,
        )
        hl = self.geom.heads_local
        out = torch.empty(num_tokens, hl, d.v_dim, dtype=q.dtype, device=q.device)
        bmm_out(
            lat.view(num_tokens, hl, d.kv_lora).transpose(0, 1),
            w[f"l{i}_v_b_proj"],
            out.transpose(0, 1),
        )
        return out.view(num_tokens, hl * d.v_dim)

    def _step_args(self, md, rope: dict) -> dict:
        """The step-level arguments both MLA calls share."""
        d = self.d
        return dict(
            workspace=md.effective_workspace,
            sequence_length=md.kv_lens_cuda_runtime,
            host_past_key_value_lengths=md.kv_lens_runtime,
            host_total_kv_lens=md.host_total_kv_lens,
            context_lengths=md.prompt_lens_cuda_runtime,
            host_context_lengths=md.prompt_lens_cpu_runtime,
            host_request_types=md.host_request_types_runtime,
            kv_cache_block_offsets=md.kv_cache_block_offsets,
            host_kv_cache_pool_pointers=md.host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=md.host_kv_cache_pool_mapping,
            rotary_cos_sin=rope["rotary_cos_sin"],
            attention_sinks=self.core.w[f"l{self.layer}_sink"],
            update_kv_cache=True,
            predicted_tokens_per_seq=1,
            local_layer_idx=self.core._pool_row(md, self.layer),
            num_heads=self.geom.heads_local,
            tokens_per_block=d.tokens_per_block,
            max_num_requests=int(getattr(md, "max_num_requests", d.max_batch) or d.max_batch),
            max_context_length=int(
                getattr(md, "max_context_length", d.max_context_len) or d.max_context_len
            ),
            max_seq_len=int(getattr(md, "max_seq_len", d.max_seq_len) or d.max_seq_len),
            attention_window_size=self._window(md),
            quant_mode=0,
            q_scaling=d.q_scaling,
            rope_dim=d.rope_dim,
            rope_base=self.geom.theta,
            rope_max_positions=d.max_seq_len,
            rope_original_max_positions=d.max_seq_len,
            is_mla_enable=True,
            q_lora_rank=d.q_lora,
            kv_lora_rank=d.kv_lora,
            qk_nope_head_dim=d.nope,
            qk_rope_head_dim=d.rope_dim,
        )

    def _generation_args(self, md, rope: dict, packed: torch.Tensor | None) -> dict:
        """`_step_args` for the generation entry, with a step-invariant host KV bound.

        A captured CUDA graph freezes the host KV length at capture, so it must not move with
        the step: the selection width on a sparse layer, the pool's max length on a windowed
        one (any smaller value could take the dense mask past the window).
        """
        args = self._step_args(md, rope)
        args["predicted_tokens_per_seq"] = self.gen_width
        bound = int(packed.shape[1]) if packed is not None else self._pool_max(md)
        args["host_past_key_value_lengths"] = torch.full_like(md.kv_lens_runtime, bound)
        return args

    def epilogue(self, attn: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Sigmoid gate, `o_proj` and `post_proj_norm`, over the `v_dim`-wide output."""
        d, w, i = self.d, self.core.w, self.layer
        gated = gated_attn_output(attn, self._linear(x, f"l{i}_linear_gate"))
        out = self._linear(gated, f"l{i}_o_proj")
        if d.attn_tp > 1:
            out = self.core._allreduce(out)
        return _rms(out, w[f"l{i}_post_proj_norm"], d.rms_eps)

    def _linear(self, x: torch.Tensor, weight: str) -> torch.Tensor:
        """Block-FP8 where the weight has a scale, bf16 otherwise (layer 0)."""
        w = self.core.w
        scale = f"{weight}_scale"
        if scale in w:
            q, s = fp8_quantize_1x128(x)
            return block_fp8_gemm(q, w[weight], s, w[scale], output_dtype=x.dtype)
        return cublas_mm(x, w[weight].t())


class Hyv4FlashCore(DecoderModel):
    """The 42-layer text core."""

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        m = model_config.mapping
        assert m.pp_size == 1, "pipeline parallelism is not implemented"
        assert m.cp_size == 1, "context parallelism is not implemented"
        assert m.moe_cluster_size == 1, "moe_cluster_size is not implemented"
        assert m.enable_attention_dp or m.attn_tp_size == m.tp_size, (
            f"attention TP {m.attn_tp_size} differs from TP {m.tp_size}"
        )
        assert not m.enable_attention_dp or m.moe_tp_size * m.moe_ep_size == m.tp_size, (
            f"attention DP needs moe_tp {m.moe_tp_size} x moe_ep {m.moe_ep_size} == TP {m.tp_size}"
        )

        self.dims = dims(self)
        self.w = _weights.MODEL_WEIGHTS.declare(self.dims)
        self.engram_tables = _weights.MODEL_WEIGHTS.engram_tables(self)
        # A speculative verify step processes `tokens_per_gen_step` rows per request.
        spec_config = getattr(model_config, "spec_config", None)
        self.gen_width = int(spec_config.tokens_per_gen_step) if spec_config is not None else 1
        self._ar_args = dict(
            allreduce_group_args(m),
            op=PLAIN,
            residual=None,
            norm_weight=None,
            scale=None,
            bias=None,
            eps=self.dims.rms_eps,
            trigger_completion_at_end=False,
        )
        self._dp_group = list(m.tp_group)
        self._dp_pg = m.tp_group_pg.boxed() if mpi_disabled() else None
        self._rope: dict[float, dict] | None = None
        self._rope_positions = 0
        self._engram_history = EngramHistory(self)
        # This step's vision rows and the placeholder rows they replace, parked by the shell.
        self._vision: tuple[torch.Tensor, torch.Tensor] | None = None

    def _rope_tables(self, device, positions: int, theta: float) -> dict:
        """The GPT-J rope table the MLA ops read: pairs `(2p, 2p + 1)`, first half duplicated."""
        rope_dim = self.dims.rope_dim
        half = rope_dim // 2
        inv = theta ** (-2.0 * torch.arange(half, dtype=torch.float64) / rope_dim)
        ang = torch.arange(positions, dtype=torch.float64)[:, None] * inv[None, :]
        cos, sin = ang.cos(), ang.sin()
        table = torch.empty(positions, rope_dim, 2, dtype=torch.float64)
        table[:, :half, 0] = cos
        table[:, half:, 0] = cos
        table[:, :half, 1] = sin
        table[:, half:, 1] = sin
        return {
            "rotary_cos_sin": table.reshape(1, positions * rope_dim * 2)
            .float()
            .to(device)
            .contiguous(),
        }

    def post_load_weights(self) -> None:
        cfg = self.model_config.pretrained_config
        device = self.w["final_norm"].device
        # Grown on demand in `forward`, never built for the config's 1M maximum up front.
        self._rope_positions = min(int(cfg.max_position_embeddings), 8192)
        self._rope = {
            theta: self._rope_tables(device, self._rope_positions, theta)
            for theta in sorted({g.theta for g in self.dims.layers})
        }
        self._engram_constants()

    def _engram_constants(self) -> None:
        """HF's `draw_hash_multipliers` (the checkpoint stores none) and the per-head moduli.

        The bound is the loaded lookup map's compressed vocabulary, as HF re-draws after loading.
        """
        import numpy as np

        d = self.dims
        self.hash_multipliers: dict[int, torch.Tensor] = {}
        self.head_moduli: dict[int, list[int]] = {}
        for i in d.engram_layers:
            sizes = d.engram_head_vocab_sizes[d.engram_ids[d.engram_layers.index(i)]]
            self.head_moduli[i] = list(sizes)
            lookup = self.w[f"l{i}_engram_lookup"]
            compressed_vocab = int(lookup.max()) + 1
            half_bound = max(1, np.iinfo(np.int64).max // compressed_vocab // 2)
            rng = np.random.default_rng(
                d.engram_seed + 10007 * d.engram_ids[d.engram_layers.index(i)]
            )
            drawn = rng.integers(0, half_bound, size=d.engram_max_ngram, dtype=np.int64)
            self.hash_multipliers[i] = torch.tensor(
                (drawn * 2 + 1).tolist(), dtype=torch.int64, device=lookup.device
            )

    def _allreduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.dims.world == 1:
            return x
        return allreduce(x, **self._ar_args)

    def _dp_exchange(self) -> bool:
        """Whether the step's rows are split over the group (attention DP, group > 1)."""
        return self.dims.attention_dp and len(self._dp_group) > 1

    def _dp_row_sizes(self, md, rows: int) -> list[int] | None:
        """The group's per-rank row counts in group order, or None when they are all equal."""
        counts = getattr(md, "all_rank_num_tokens", None)
        assert counts is not None, (
            "an attention-DP step carries the group's row counts (`all_rank_num_tokens`)"
        )
        counts = [int(n) for n in counts]
        assert len(counts) == len(self._dp_group), (counts, self._dp_group)
        rank = self._dp_group.index(self.dims.rank)
        assert counts[rank] == rows, (
            f"rank {self.dims.rank} holds {rows} rows but all_rank_num_tokens says "
            f"{counts[rank]} ({counts}); the MoE's round trip would gather the wrong split"
        )
        return None if min(counts) == max(counts) else counts

    def _pool_row(self, md, layer: int) -> int:
        """The layer's index in the primary KV pool (decoder order)."""
        return layer

    def _pool_rows(self, md, layer: int, packed: torch.Tensor) -> torch.Tensor:
        """Request-local key positions -> primary-pool token rows (`-1` stays `-1`)."""
        manager = md.kv_cache_manager
        scale, layer_offset = manager.get_primary_pool_page_index_params(self._pool_row(md, layer))
        # The block table was built with the manager's page size, not the dims'.
        tokens_per_block = int(manager.tokens_per_block)
        return convert_req_index_to_global(
            md.req_idx_per_token[: packed.shape[0]],
            md.block_table,
            packed,
            tokens_per_block,
            int(packed.shape[1]),
            int(scale) * tokens_per_block,
            int(layer_offset),
        )

    def _layer_forward(
        self,
        md,
        i: int,
        streams: torch.Tensor,
        row_batch: torch.Tensor,
        position_ids: torch.Tensor,
        ngram_tokens: torch.Tensor | None,
        engram_mask: torch.Tensor | None,
        rope: dict,
        picks: cache_manager.DraftPicks | None = None,
        picks_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One decoder layer over `[T, hc, hidden]` residual streams."""
        d, w = self.dims, self.w
        geom = d.layers[i]

        # HF adds the memory before the layer's hyper-connection reads the streams.
        if i in d.engram_layers:
            streams = streams + EngramLayer(self, i).forward(streams, ngram_tokens, engram_mask)
        residual = streams
        h, gates = ihc_mapping(
            streams,
            w[f"l{i}_hc_attn_fn"],
            w[f"l{i}_hc_attn_base"],
            w[f"l{i}_hc_attn_scale"],
            d.hc_magnitude,
            d.hc_eps,
            d.rms_eps,
        )
        h = _rms(h, w[f"l{i}_input_layernorm"], d.rms_eps)

        q_resid = _rms(cublas_mm(h, w[f"l{i}_q_a_proj"].t()), w[f"l{i}_q_a_layernorm"], d.rms_eps)
        kv = cublas_mm(h, w[f"l{i}_kv_a_proj_with_mqa"].t())
        latent = kv[:, : d.kv_lora + d.rope_dim]
        latent[:, : d.kv_lora] = _rms(latent[:, : d.kv_lora], w[f"l{i}_kv_a_layernorm"], d.rms_eps)
        q = self._expand_q(q_resid, i)
        layer = MlaLayer(self, i)
        # Only a sparse layer hands decode a selection; a windowed one decodes through the dense
        # windowed entry.
        packed = None
        if geom.sparse:
            selected = layer.indexer.pool_and_select(q_resid, h, md, row_batch, position_ids)
            packed = self._pool_rows(md, i, selected)
            if picks is not None:
                picks.write(i, packed, picks_rows)
        # A step's rows are the context requests' first, then one block per decoding request.
        num_ctx = int(md.num_contexts)
        tc = int(md.num_ctx_tokens) if num_ctx else 0
        gen = h.shape[0] - tc
        if tc and gen:
            attn = torch.cat(
                [
                    layer.context_rows(q[:tc], latent[:tc], md, rope),
                    layer.generation_partial(
                        q[tc:],
                        latent[tc:],
                        md,
                        rope,
                        None if packed is None else packed[tc:],
                    ),
                ],
                dim=0,
            )
        elif tc:
            attn = layer.context_rows(q, latent, md, rope)
        else:
            attn = layer.generation_partial(q, latent, md, rope, packed)
        attn = layer.epilogue(attn, h)
        streams = _hc_residual(gates[..., d.hc_mult :], attn, residual)

        residual = streams
        h, gates = ihc_mapping(
            streams,
            w[f"l{i}_hc_mlp_fn"],
            w[f"l{i}_hc_mlp_base"],
            w[f"l{i}_hc_mlp_scale"],
            d.hc_magnitude,
            d.hc_eps,
            d.rms_eps,
        )
        h = _rms(h, w[f"l{i}_post_attention_layernorm"], d.rms_eps)
        if i < d.first_dense:
            h = self._dense_mlp(h, i)
            h = _rms(h, w[f"l{i}_post_fc2_norm"], d.rms_eps)
        else:
            h = LatentMoe(self, i).forward(h, md)
        return _hc_residual(gates[..., d.hc_mult :], h, residual)

    def _expand_q(self, q_resid: torch.Tensor, i: int) -> torch.Tensor:
        """`[T, heads_local, qk_dim]`: the unabsorbed query, block-FP8 `q_b` past the dense layer."""
        d, w = self.dims, self.w
        geom = d.layers[i]
        if i >= d.first_dense:
            q, s = fp8_quantize_1x128(q_resid)
            flat = block_fp8_gemm(
                q, w[f"l{i}_q_b_proj"], s, w[f"l{i}_q_b_proj_scale"], output_dtype=q_resid.dtype
            )
        else:
            flat = cublas_mm(q_resid, w[f"l{i}_q_b_proj"].t())
        return flat.view(-1, geom.heads_local, d.qk_dim)

    def _dense_mlp(self, x: torch.Tensor, i: int) -> torch.Tensor:
        w, d = self.w, self.dims
        up = flashinfer_silu_and_mul(
            cublas_mm(x, torch.cat([w[f"l{i}_dense_gate"], w[f"l{i}_dense_up"]], dim=0).t())
        )
        out = cublas_mm(up, w[f"l{i}_dense_down"].t())
        if d.mlp_tp > 1:
            out = self._allreduce(out)
        return out

    def _head(self, streams: torch.Tensor) -> torch.Tensor:
        """`hc_head` then `model.norm`: the streams collapse to one hidden state."""
        d, w = self.dims, self.w
        h, _ = ihc_mapping(
            streams,
            w["hc_head_fn"],
            w["hc_head_base"],
            w["hc_head_scale"],
            d.hc_magnitude,
            d.hc_eps,
            d.rms_eps,
        )
        return _rms(h, w["final_norm"], d.rms_eps)

    def forward(
        self,
        attn_metadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        md = attn_metadata
        # The engine hands MRoPE models `[3, 1, N]` position ids; text positions are on every axis.
        if position_ids is not None and position_ids.dim() > 1:
            position_ids = position_ids[0].reshape(-1)
        wanted = int(getattr(md, "max_seq_len", 0) or 0)
        if wanted > self._rope_positions:
            self._rope_positions = min(
                wanted, int(self.model_config.pretrained_config.max_position_embeddings)
            )
            self._rope = {
                theta: self._rope_tables(self.w["final_norm"].device, self._rope_positions, theta)
                for theta in self._rope
            }
        if inputs_embeds is None:
            inputs_embeds = torch.nn.functional.embedding(input_ids, self.w["embed"])
        vision, self._vision = self._vision, None
        if vision is not None:
            rows, mm_token_indices = vision
            inputs_embeds = inputs_embeds.index_copy(
                0, mm_token_indices, rows.to(inputs_embeds.dtype)
            )
        d = self.dims
        streams = inputs_embeds.unsqueeze(1).expand(-1, d.hc_mult, -1).contiguous()
        row_batch = _row_batch(md)
        ngram_tokens = engram_mask = None
        if d.engram_layers:
            # In-step predecessors first; the history fills the pads before the step's first row.
            ngram_tokens = _ngram_tokens(
                input_ids, position_ids, row_batch, d.engram_max_ngram, d.engram_pad_id
            )
            history = self._engram_history.lookback(
                md, input_ids, position_ids, row_batch, d.engram_pad_id
            )
            if history is not None:
                ngram_tokens[:, 1:] = torch.where(
                    ngram_tokens[:, 1:] == d.engram_pad_id, history, ngram_tokens[:, 1:]
                )
            engram_mask = input_ids.view(-1) != d.engram_pad_id
        picks = None
        picks_rows = None
        if d.draft_source_layers:
            # DFly2's draft layers read the sparse layers' picks back from the cache manager.
            picks = md.kv_cache_manager.draft_picks(
                d.draft_source_layers,
                d.index_topk,
                rows=d.max_num_tokens,
                device=streams.device,
            )
            picks_rows = _picks_rows(md)
            if int(md.num_tokens) > picks.history.shape[1]:
                raise ValueError(
                    f"this step carries {int(md.num_tokens)} rows and the picks state holds "
                    f"{picks.history.shape[1]}"
                )
        for i in range(d.num_layers):
            streams = self._layer_forward(
                md,
                i,
                streams,
                row_batch,
                position_ids,
                ngram_tokens,
                engram_mask,
                self._rope[d.layers[i].theta],
                picks,
                picks_rows,
            )
        return self._head(streams)


class ModelingV2Hy4FlashFp8Sm103Default(DecoderModelForCausalLM):
    """The shell: lm_head, logits processor, vision tower and, under one-engine DFly2, the drafter.

    The drafter is five `model.dfly_layers.*` layers of this checkpoint (`draft_config` is None),
    run by `Hyv4FlashDFlashWorker` over the target's sparse KV pools.
    """

    runtime_hooks = RuntimeHooks(kv_cache_manager_cls=cache_manager.Hyv4FlashCacheManager)

    #: The worker verifies through gathered logits, not the vocab-shard fast paths.
    spec_decode_fast_paths: bool = False

    def __init__(self, model_config: ModelConfig):
        cfg = model_config.pretrained_config
        text = getattr(cfg, "text_config", cfg)
        super().__init__(
            Hyv4FlashCore(model_config),
            config=model_config,
            hidden_size=text.hidden_size,
            vocab_size=text.vocab_size,
        )
        self.spec_config = getattr(model_config, "spec_config", None)
        self.draft_model = None
        self.draft_config = None
        self.spec_worker = None
        self.use_separate_draft_kv_cache = False
        self.layer_idx = -1
        if self.spec_config is not None:
            self._build_drafter(model_config)
        # Built lazily: `__init__` runs under meta init.
        self._mm_token_ids_value = (int(cfg.image_token_id), int(cfg.video_token_id))
        self._mm_token_ids_cache: torch.Tensor | None = None
        # Replicated on every rank, built in `post_load_weights`; None for a config without one.
        self.vision_tower: _vision.VisionTower | None = None

    @property
    def multimodal_token_ids(self) -> torch.Tensor:
        """The placeholder ids whose rows the tower's embeddings replace."""
        if self._mm_token_ids_cache is None:
            self._mm_token_ids_cache = torch.tensor(self._mm_token_ids_value, dtype=torch.int32)
        return self._mm_token_ids_cache

    @property
    def multimodal_data_device_paths(self) -> list[str]:
        return list(_vision.DEVICE_PATHS)

    def validate_request(self, request) -> None:
        _vision.validate_request(self.vision_tower, request)

    @classmethod
    def get_model_defaults(cls, llm_args):
        return {"lm_head_output_dtype": "float32"}

    @classmethod
    def get_preferred_kv_cache_manager_version(cls, pretrained_config):
        return "V2"

    def _build_drafter(self, model_config: ModelConfig) -> None:
        spec_config = self.spec_config
        if not (
            spec_config.spec_dec_mode.is_dflash() and spec_config.spec_dec_mode.use_one_engine()
        ):
            raise ValueError(
                "This target serves the checkpoint's DFly2 drafter in one engine, got "
                f"{spec_config.spec_dec_mode=} with {spec_config.uses_external_draft_model=}"
            )
        from .draft import Hyv4FlashDraft
        from .spec_worker import Hyv4FlashDFlashWorker

        drafter = Hyv4FlashDraft(self.model)
        drafter.logits_processor = self.logits_processor
        drafter.lm_head = self.lm_head
        self.draft_model = drafter
        self.epilogue.append(drafter)

        from tensorrt_llm._torch.speculative import should_use_separate_draft_kv_cache

        if should_use_separate_draft_kv_cache(spec_config):
            raise ValueError(
                "The DFly2 drafter keeps no KV cache of its own (it writes the target sparse "
                "layers' pools); serve it with dflash attention_backend VANILLA."
            )

        worker = Hyv4FlashDFlashWorker(
            spec_config, model_config.mapping, self.use_separate_draft_kv_cache
        )
        worker.set_draft_model(drafter)
        worker.set_target_model(self)
        self.spec_worker = worker
        self.epilogue.append(worker)

    def load_draft_weights(self, weights, weight_mapper=None) -> None:
        self.draft_model.load_weights(weights=weights, weight_mapper=weight_mapper)

    def set_guided_decoder(self, guided_decoder) -> bool:
        if self.spec_worker is None:
            return False
        return self.spec_worker.set_guided_decoder(guided_decoder)

    def load_weights(self, weights, *args, **kwargs):
        self.vision_tower = None
        _weights.MODEL_WEIGHTS.load(self, weights)

    def forward(
        self,
        attn_metadata,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        return_context_logits: bool = False,
        spec_metadata=None,
        lora_params: dict | None = None,
        **kwargs,
    ):
        """The media first (the inherited forward drops `multimodal_params`), then the target,
        then, under speculation, the worker's acceptance and drafting."""
        self.model._vision = _vision.multimodal_embeds(self.vision_tower, kwargs)
        if self.spec_worker is None:
            return super().forward(
                attn_metadata,
                input_ids,
                position_ids,
                inputs_embeds,
                return_context_logits,
                spec_metadata,
                lora_params,
                **kwargs,
            )

        hidden = self.model(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            lora_params=lora_params,
        )
        if spec_metadata is not None and spec_metadata.is_layer_capture(self.layer_idx):
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden)
        if attn_metadata.padded_num_tokens is not None:
            # Drop graph padding: every row index the drafter reads is in the unpadded layout.
            num_tokens = attn_metadata.num_tokens
            hidden = hidden[:num_tokens]
            input_ids = input_ids[:num_tokens] if input_ids is not None else None
            position_ids = position_ids[..., :num_tokens] if position_ids is not None else None

        # Full fp32 logits on every rank: this worker declines the vocab-shard selection.
        logits = self.spec_worker.target_logits(
            hidden[spec_metadata.gather_ids],
            self.lm_head,
            self.logits_processor,
            attn_metadata,
            spec_metadata,
            self.draft_model,
        )
        return self.spec_worker(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden,
            logits=logits,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=self.draft_model,
            resource_manager=kwargs.get("resource_manager"),
        )

    def post_load_weights(self):
        super().post_load_weights()
        self.model.post_load_weights()
        if self.model.dims.vision is None:
            return
        self.vision_tower = _vision.VisionTower(self.model.w, self.model.dims.vision)
        # CPU-only load checks build the shell over host weights; nothing to warm up there.
        if self.lm_head.weight.is_cuda:
            self.vision_tower.warm_up()
