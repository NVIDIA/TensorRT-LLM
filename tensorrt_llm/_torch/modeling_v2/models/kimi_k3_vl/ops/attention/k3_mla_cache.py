# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 MLA latent-cache operations over the engine's KV cache, per cache state.

Every call takes the role's :class:`~.k3_mla_state.K3MlaAttnState` and the step's attention metadata (pools,
block offsets, lengths; the engine owns the KV cache manager).

FP8 / BF16 latent cache (the attention op's paged MLA cache; K3's FP8 layers 91 / 92):

* :func:`mla_rope_append_paged_kv_assign_q`: rotate Q and the chunk's K-pe (identity RoPE) and append the
  chunk's latent rows to the cache (``trtllm::mla_rope_append_paged_kv_assign_q``);
* :func:`load_paged_kv_cache_for_mla` / :func:`load_chunked_kv_cache_for_mla`: read the whole cached
  context, or one chunked-prefill partition, back as BF16 latent + K-pe;
* :func:`merge_attention_for_mla`: merge a partition's output into the running output by softmax stats;
* :func:`mla_rope_generation`: the generation cache append, query rotation and FP8 query quantization
  before the decode (``trtllm::mla_rope_generation``).

NVFP4 latent cache (the FP4 MLA cache of the KV cache manager; K3's other MLA layers):

* :func:`append_fp4_mla_context_cache`: rotate Q and K-pe and append one context chunk (fused scatter);
* :func:`fp4_mla_rope_generation`: the generation scatter with the fused FP4 query quantization;
* :func:`load_chunked_kv_cache_for_mla` reads a partition back through the FP4 chunked gather;
* :func:`fp4_mla_context_attention`: context attention without a cached prefix: the FP8 context FMHA over
  a per-step FP8 scratch cache, beside the FP4 scatter of the chunk on the scratch's cache stream.

The FP4 MLA kernels and their launch code are the family copies in ``_k3_fp4_mla_cache`` /
``_k3_fp4_mla_kernels``; the FP4 cache layout, metadata and plan helpers they read are the KV cache
manager's contract (``attention/backends/fp4_mla/{config,layout,metadata,packed_plan,state}``).
"""

from __future__ import annotations

import copy
from typing import Optional

import torch

from tensorrt_llm.quantization.mode import QuantMode

from ._k3_fp4_mla_cache import (
    _execute_fp8_context_with_cache_update,
    _Fp8MlaContextScratch,
    _get_fp8_mla_context_metadata,
    can_fuse_fp4_mla_q_quant,
    load_fp4_mla_chunked_kv_cache,
    require_fp4_mla_fp8_context_support,
    scatter_fp4_mla_kv_cache,
)
from .k3_mla_state import K3MlaAttnState

# The FP8 context scratch this family keeps on the KV cache manager (one per manager, shared by its layers).
_K3_FP8_CONTEXT_SCRATCH_ATTR = "_k3_fp4_mla_fp8_context_scratch"

# ---------------------------------------------------------------------------------------------
# FP8 / BF16 latent cache.
# ---------------------------------------------------------------------------------------------


def mla_rope_append_paged_kv_assign_q(
    state: K3MlaAttnState, q: torch.Tensor, latent_cache: torch.Tensor, md
) -> None:
    """Rotate Q (``[T, H * 192]``, in place) and the chunk's K-pe, and append ``latent_cache``
    (``[T, 576]``) to this layer's paged cache; Helix owner-only prefill appends the owned slots only."""
    manager = md.kv_cache_manager
    assert manager is not None
    state._ensure_rope_table_size(manager.max_seq_len)
    torch.ops.trtllm.mla_rope_append_paged_kv_assign_q(
        q,
        latent_cache,
        md.num_contexts,
        md.ctx_cached_token_indptr,
        md.ctx_kv_indptr,
        md.max_ctx_seq_len,
        state.rotary_cos_sin,
        state.num_heads,
        state.qk_nope_head_dim,
        state.qk_rope_head_dim,
        state.kv_lora_rank,
        md.kv_cache_block_offsets,
        manager.kv_cache_pool_pointers,
        manager.kv_cache_pool_mapping,
        None,  # kv_scale_orig_quant
        0,  # residual_dim
        state.get_local_layer_idx(md),
        manager.tokens_per_block,
        manager.max_seq_len,
        1,  # beam_width
        state.quant_mode,
        md.helix_context_local_slots,
    )


def load_paged_kv_cache_for_mla(
    state: K3MlaAttnState, md, out_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """The context requests' whole cached KV (prefix and current chunk) as ``(latent [N, 512],
    K-pe [N, 64])`` in ``out_dtype``."""
    assert out_dtype in (torch.float16, torch.bfloat16, torch.float32)
    manager = md.kv_cache_manager
    assert manager is not None
    assert md.max_ctx_kv_len > 0
    assert md.num_ctx_cached_tokens + md.num_ctx_tokens == md.host_ctx_kv_indptr[md.num_contexts]
    return torch.ops.trtllm.load_paged_kv_cache_for_mla(
        out_dtype,
        md.num_contexts,
        md.num_ctx_cached_tokens + md.num_ctx_tokens,
        md.max_ctx_kv_len,
        md.ctx_kv_indptr,
        md.kv_cache_block_offsets,
        manager.kv_cache_pool_pointers,
        manager.kv_cache_pool_mapping,
        None,  # kv_scale_quant_orig
        state.get_local_layer_idx(md),
        state.kv_lora_rank,
        state.qk_rope_head_dim,
        manager.tokens_per_block,
        manager.max_seq_len,
        1,  # beam_width
        state.quant_mode,
    )


def load_chunked_kv_cache_for_mla(
    state: K3MlaAttnState,
    md,
    num_ctx_cached_tokens: int,
    cu_chunked_seq_len: torch.Tensor,
    chunked_global_offset: torch.Tensor,
    chunked_max_seq_len: int,
    out_dtype: torch.dtype,
    helix_plan=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One chunked-prefill partition of the cached prefix as ``(latent [N, 512], K-pe [N, 64])`` in
    ``out_dtype``; an FP4 cache reads it through the FP4 chunked gather, a Helix plan gathers every CP
    rank's share."""
    assert out_dtype in (torch.float16, torch.bfloat16, torch.float32)
    manager = md.kv_cache_manager
    assert manager is not None
    if state.uses_fp4_mla_attention:
        return load_fp4_mla_chunked_kv_cache(
            md,
            state.layer_idx,
            num_ctx_cached_tokens=num_ctx_cached_tokens,
            cu_chunked_seq_len=cu_chunked_seq_len,
            chunked_global_offset=chunked_global_offset,
            chunked_max_seq_len=chunked_max_seq_len,
            out_dtype=out_dtype,
            kv_lora_rank=state.kv_lora_rank,
            qk_rope_head_dim=state.qk_rope_head_dim,
        )
    if helix_plan is not None:
        from tensorrt_llm._torch.distributed import cp_allgather

        rank = manager.mapping.cp_rank
        local_tokens = helix_plan.rank_sizes[rank]
        if local_tokens:
            local_kv, local_pe = load_chunked_kv_cache_for_mla(
                state,
                md,
                local_tokens,
                helix_plan.cu_seq_lens[rank],
                helix_plan.local_offsets[rank],
                helix_plan.max_local_seq_len,
                out_dtype,
            )
            local = torch.cat((local_kv, local_pe), dim=-1)
        else:
            local = torch.empty(
                (0, state.kv_lora_rank + state.qk_rope_head_dim),
                dtype=out_dtype,
                device=cu_chunked_seq_len.device,
            )
        # Every rank enters the collective with the same global plan.
        gathered = cp_allgather(local, manager.mapping, dim=0, sizes=helix_plan.rank_sizes)
        assert isinstance(gathered, torch.Tensor)
        return torch.ops.trtllm.reorder_helix_mla_kv(
            gathered,
            cu_chunked_seq_len,
            chunked_global_offset,
            helix_plan.cu_seq_lens,
            helix_plan.local_offsets,
            helix_plan.rank_offsets,
            manager.tokens_per_block,
            state.kv_lora_rank,
            state.qk_rope_head_dim,
        )
    if md.max_ctx_cached_token_len == 0:
        return (
            torch.empty((0, state.kv_lora_rank), dtype=out_dtype, device=cu_chunked_seq_len.device),
            torch.empty(
                (0, state.qk_rope_head_dim), dtype=out_dtype, device=cu_chunked_seq_len.device
            ),
        )
    return torch.ops.trtllm.load_chunked_kv_cache_for_mla(
        out_dtype,
        md.num_contexts,
        num_ctx_cached_tokens,
        cu_chunked_seq_len,
        chunked_global_offset,
        md.kv_cache_block_offsets,
        manager.kv_cache_pool_pointers,
        manager.kv_cache_pool_mapping,
        None,  # kv_scale_quant_orig
        state.get_local_layer_idx(md),
        state.kv_lora_rank,
        state.qk_rope_head_dim,
        manager.tokens_per_block,
        chunked_max_seq_len,
        manager.max_seq_len,
        1,  # beam_width
        state.quant_mode,
    )


def merge_attention_for_mla(
    state: K3MlaAttnState,
    merged_attn: torch.Tensor,
    temp_attn: torch.Tensor,
    softmax_stats: torch.Tensor,
    temp_softmax_stats: torch.Tensor,
    merge_op: torch.Tensor,
    md,
) -> None:
    """Merge one partition's output ``temp_attn`` (and stats) into ``merged_attn`` per ``merge_op``."""
    assert md.kv_cache_manager is not None
    torch.ops.trtllm.merge_chunked_attention_for_mla(
        merged_attn,
        temp_attn,
        softmax_stats,
        temp_softmax_stats,
        md.num_contexts,
        md.ctx_uncached_token_indptr,
        md.max_ctx_seq_len,
        merge_op,
        state.num_heads,
        state.v_head_dim,
    )


def mla_rope_generation(
    state: K3MlaAttnState,
    fused_q: torch.Tensor,
    q_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    md,
    cu_q_seqlens: torch.Tensor,
    cu_kv_seqlens: torch.Tensor,
    fmha_scheduler_counter: torch.Tensor,
    mla_bmm1_scale: Optional[torch.Tensor],
    mla_bmm2_scale: Optional[torch.Tensor],
    quant_q_buffer: Optional[torch.Tensor],
    *,
    fuse_fp4_q_quant: bool = False,
) -> None:
    """The generation cache update before the decode: rotate ``q_pe`` (identity) into ``fused_q``'s K-pe
    half, append ``latent_cache`` (``[T, 576]``) and, on a quantized cache, write the quantized query
    (FP8: ``quant_q_buffer`` and the BMM scales; FP4: :func:`fp4_mla_rope_generation`)."""
    assert md.kv_cache_manager is not None
    state._ensure_rope_table_size(md.max_seq_len)
    if state.uses_fp4_mla_attention:
        fp4_mla_rope_generation(
            state, fused_q, q_pe, latent_cache, md, fuse_q_quant=fuse_fp4_q_quant
        )
        return
    helix_tensor_params = [md.helix_position_offsets, md.helix_is_inactive_rank]
    if md._helix_spec_tokens_valid:
        # Speculative verify groups: per-token KV write slots (-1 = not this rank's position).
        helix_tensor_params.append(md.helix_local_slots)
    manager = md.kv_cache_manager
    torch.ops.trtllm.mla_rope_generation(
        fused_q,
        q_pe,
        latent_cache,
        state.rotary_cos_sin,
        cu_q_seqlens,
        cu_kv_seqlens,
        fmha_scheduler_counter,
        mla_bmm1_scale,
        mla_bmm2_scale,
        quant_q_buffer,
        md.kv_lens_cuda_runtime,
        md.kv_lens_runtime,
        md.prompt_lens_cpu_runtime,
        md.num_contexts,
        md.kv_cache_block_offsets,
        manager.kv_cache_pool_pointers,
        manager.kv_cache_pool_mapping,
        None,  # kv_scale_orig_quant
        None,  # kv_scale_quant_orig
        None,  # kv_cache_scale_orig_quant (sparse NVFP4 MLA only)
        None,  # out_scale
        md.block_ids_per_seq,
        helix_tensor_params,
        state.predicted_tokens_per_seq,
        state.get_local_layer_idx(md),
        state.num_heads,
        state.num_kv_heads,
        state.head_dim,
        getattr(manager, "mla_kv_cache_residual_dim", 0),
        manager.tokens_per_block,
        md.max_seq_len,  # attention_window_size
        md.beam_width,
        state.quant_mode,
        state.q_scaling,
        state.q_lora_rank,
        state.kv_lora_rank,
        state.qk_nope_head_dim,
        state.qk_rope_head_dim,
        state.v_head_dim,
        state.rope_append,
        None,  # kv_norm_weight
        1e-6,  # kv_norm_eps
        False,  # precomputed_cu_seqlens
        False,  # precomputed_fmha_scheduler
        False,  # kv_only
        False,  # kv_done_elsewhere
        None,  # quant_scale_qkv
        False,  # q_rope_applied
    )


# ---------------------------------------------------------------------------------------------
# NVFP4 latent cache.
# ---------------------------------------------------------------------------------------------


def append_fp4_mla_context_cache(
    state: K3MlaAttnState, q: torch.Tensor, latent_cache: torch.Tensor, md
) -> None:
    """Rotate Q and K-pe (identity RoPE) and append one context chunk to the FP4 MLA cache."""
    if not state.uses_fp4_mla_attention:
        raise RuntimeError("FP4 MLA context append requires dense FP4 MLA attention.")
    if md.fp4_mla_state is None or md.fp4_mla_state.positions is None:
        raise RuntimeError("FP4 MLA context append requires token positions.")
    state._ensure_rope_table_size(md.max_seq_len)
    hp_pool_updated = scatter_fp4_mla_kv_cache(
        md,
        latent_cache[: md.num_ctx_tokens],
        state.layer_idx,
        token_offset=0,
        phase="context",
        local_layer=state.get_fp4_mla_local_layer_idx(md),
        v_head_dim=state.kv_lora_rank,
        rotary_cos_sin=state.rotary_cos_sin,
        q_context=q,
        q_nope_head_dim=state.qk_nope_head_dim,
    )
    if not hp_pool_updated:
        raise RuntimeError("Fused FP4 MLA context scatter did not update the HP pool.")


def fp4_mla_rope_generation(
    state: K3MlaAttnState,
    fused_q: torch.Tensor,
    q_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    md,
    *,
    fuse_q_quant: bool = False,
) -> None:
    """The FP4 generation scatter: rotate ``q_pe`` into ``fused_q``, append ``latent_cache`` and quantize
    the query into the FP4 Q buffers on the metadata's FP4 MLA state."""
    if q_pe.shape[-1] != state.qk_rope_head_dim:
        raise RuntimeError(
            f"FP4 MLA q_pe last dimension must be {state.qk_rope_head_dim}, got {q_pe.shape[-1]}."
        )
    if latent_cache.shape[-1] != state.kv_lora_rank + state.qk_rope_head_dim:
        raise RuntimeError(
            "FP4 MLA latent_cache last dimension must be "
            f"{state.kv_lora_rank + state.qk_rope_head_dim}, got {latent_cache.shape[-1]}."
        )
    if not fuse_q_quant:
        raise RuntimeError("FP4 MLA generation requires fused Q quantization.")
    if not state.rope_duplicate_data:
        raise RuntimeError("FP4 MLA requires RopeParams.duplicate_data=True.")
    if not (
        state.uses_fp4_mla_attention and can_fuse_fp4_mla_q_quant(md, fused_q, q_pe, latent_cache)
    ):
        raise RuntimeError("FP4 MLA fused Q quantization eligibility changed before launch.")
    if (
        state.rotary_cos_sin is None
        or q_pe.dtype != torch.bfloat16
        or fused_q.dtype != torch.bfloat16
        or latent_cache.dtype != torch.bfloat16
        or state.rotary_cos_sin.dtype != torch.float32
    ):
        raise RuntimeError(
            "FP4 MLA generation requires fused BF16 RoPE/cache update with an FP32 rotary table."
        )
    hp_pool_updated = scatter_fp4_mla_kv_cache(
        md,
        latent_cache,
        state.layer_idx,
        token_offset=getattr(md, "num_ctx_tokens", 0),
        phase="generation",
        local_layer=state.get_fp4_mla_local_layer_idx(md),
        v_head_dim=state.kv_lora_rank,
        rotary_cos_sin=state.rotary_cos_sin,
        q_pe=q_pe,
        q_rope_out=fused_q[..., state.kv_lora_rank :],
        q_quant_input=fused_q,
        helix_position_offsets=md.helix_position_offsets,
        helix_is_inactive_rank=md.helix_is_inactive_rank,
    )
    if not hp_pool_updated:
        raise RuntimeError("Fused FP4 MLA RoPE/cache scatter did not update the HP pool.")


def _fp8_context_state(state: K3MlaAttnState) -> K3MlaAttnState:
    """The FP8 view of an FP4 layer's context role over the one-layer FP8 scratch cache."""
    fp8_state = getattr(state, "_fp8_context_view", None)
    if fp8_state is None:
        fp8_state = copy.copy(state)
        fp8_state.quant_mode = QuantMode(int(QuantMode(0).set_fp8_kv_cache()))
        fp8_state.has_fp4_kv_cache = False
        fp8_state.has_fp8_kv_cache = True
        fp8_state.local_layer_idx = 0
        fp8_state.layer_idx = 0
        state._fp8_context_view = fp8_state
    fp8_state.rotary_inv_freq = state.rotary_inv_freq
    fp8_state.rotary_cos_sin = state.rotary_cos_sin
    fp8_state.rope_max_positions = state.rope_max_positions
    return fp8_state


def _fp8_context_scratch(md, q: torch.Tensor, state: K3MlaAttnState) -> _Fp8MlaContextScratch:
    """The KV cache manager's FP8 context scratch (created by the first layer of the first step)."""
    manager = md.kv_cache_manager
    if manager is None:
        raise RuntimeError("FP8 MLA context scratch requires a KV cache manager.")
    head_dim = state.kv_lora_rank + state.qk_rope_head_dim
    scratch = getattr(manager, _K3_FP8_CONTEXT_SCRATCH_ATTR, None)
    if scratch is None:
        scratch = _Fp8MlaContextScratch.create(md, device=q.device, head_dim=head_dim)
        setattr(manager, _K3_FP8_CONTEXT_SCRATCH_ATTR, scratch)
    else:
        assert scratch.matches(md, device=q.device, head_dim=head_dim), (
            "FP8 MLA context scratch geometry must be shared by all layers of its KV manager."
        )
    return scratch


def fp4_mla_context_attention(
    state: K3MlaAttnState,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    md,
    output: torch.Tensor,
    latent_cache: torch.Tensor,
) -> torch.Tensor:
    """Context attention of an FP4-cache layer without a cached prefix (``q`` / ``k`` ``[T, H * 192]``,
    ``v`` ``[T, H * 128]``, ``latent_cache`` ``[T, 576]``) into ``output``: the FP8 context FMHA over the
    FP8 scratch cache (the op appends the chunk's latent rows there, rotated at their absolute positions)
    and, on the scratch's cache stream, the FP4 scatter of the chunk into the FP4 cache."""
    from .k3_mla_attention import mla_context_attention

    if md.fp4_mla_state is None or md.fp4_mla_state.hp_pool is None:
        raise RuntimeError("FP4 MLA requires prepared high-precision KV state.")
    if md.fp4_mla_state.positions is None:
        raise RuntimeError("FP4 MLA context requires token positions.")
    if md.num_contexts <= 0:
        raise RuntimeError("FP4 MLA context requires context requests.")
    if getattr(md, "num_ctx_cached_tokens", 0) != 0:
        raise NotImplementedError("FP4 MLA does not support cached-context prefill.")
    num_tokens = md.num_ctx_tokens
    if q.shape[0] != num_tokens or k.shape[0] != num_tokens or v.shape[0] != num_tokens:
        raise RuntimeError("FP4 MLA context Q/K/V token counts must match num_ctx_tokens.")
    require_fp4_mla_fp8_context_support()
    # As for every MLA attention call: separate Q / K / V, and a workspace the call may keep state in.
    md.use_paged_context_fmha = False
    md.workspace_reclaimable = False
    state._ensure_rope_table_size(md.max_seq_len)
    output = output.view(num_tokens, -1)
    local_layer = state.get_fp4_mla_local_layer_idx(md)
    phase_latent = latent_cache[:num_tokens]

    def update_fp4_cache() -> None:
        hp_pool_updated = scatter_fp4_mla_kv_cache(
            md,
            phase_latent,
            state.layer_idx,
            token_offset=0,
            phase="context",
            local_layer=local_layer,
            v_head_dim=state.kv_lora_rank,
            rotary_cos_sin=state.rotary_cos_sin,
        )
        if not hp_pool_updated:
            raise RuntimeError("Fused FP4 MLA context scatter did not update the HP pool.")

    scratch = _fp8_context_scratch(md, q, state)
    fp8_state = _fp8_context_state(state)
    fp8_md = _get_fp8_mla_context_metadata(md, scratch)

    def run_fp8_context() -> None:
        mla_context_attention(fp8_state, q, k, v, fp8_md, output, latent_cache=latent_cache)

    _execute_fp8_context_with_cache_update(
        run_fp8_context,
        update_fp4_cache,
        scratch.cache_stream,
        scratch.cache_start_event,
        scratch.cache_done_event,
    )
    return output


__all__ = [
    "append_fp4_mla_context_cache",
    "fp4_mla_context_attention",
    "fp4_mla_rope_generation",
    "load_chunked_kv_cache_for_mla",
    "load_paged_kv_cache_for_mla",
    "merge_attention_for_mla",
    "mla_rope_append_paged_kv_assign_q",
    "mla_rope_generation",
]
