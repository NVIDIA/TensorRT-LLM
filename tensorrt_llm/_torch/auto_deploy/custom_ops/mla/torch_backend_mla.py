"""Custom ops for MultiHead Latent Attention (MLA) with FlashInfer-compatible cache.

This module provides:
- torch_cached_mla_with_cache: cached backend op
- MultiHeadLatentAttention: attention descriptor

FlashInfer MLA Cache Layout:
    mla_cache: [max_batch, max_seq, kv_lora_rank + qk_rope_head_dim]
    - No num_heads dimension (MLA-specific optimization)
    - compressed_kv_cached = mla_cache[:, :, :kv_lora_rank]  (zero-copy slice)
    - kpe_cached = mla_cache[:, :, kv_lora_rank:]  (zero-copy slice)

The implementation uses:
- Prefill: Expand compressed_kv -> full K, V, compute normal attention
- Generate: Weight absorption for efficiency (Q @ W^T instead of expanding cached KV)

Reference: https://docs.flashinfer.ai/tutorials/kv_layout.html#mla-page-layout
"""

import math
from typing import List, Optional

import torch
from torch._ops import OpOverloadPacket
from torch.fx import Node

from .....llmapi.llm_args import KvCacheConfig
from ..attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    Constant,
    MHACallable,
    ResourceHandlerDict,
    UnpagedResourceHandler,
)


def _update_mla_cache(
    compressed_kv: torch.Tensor,  # [total_tokens, kv_lora_rank]
    kpe: torch.Tensor,  # [total_tokens, qk_rope_head_dim]
    mla_cache: torch.Tensor,  # [max_batch, max_seq, kv_lora_rank + qk_rope_head_dim]
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    seq_start: torch.Tensor,
    kv_lora_rank: int,
) -> None:
    """Update FlashInfer MLA cache with compressed_kv and kpe values.

    FlashInfer MLA cache layout: [max_batch, max_seq, kv_lora_rank + qk_rope_head_dim]
    - First kv_lora_rank dims: compressed KV latent (before kv_b_proj)
    - Last qk_rope_head_dim dims: key positional encoding
    """
    for idx in range(seq_len.shape[0]):
        start = seq_start[idx].item()
        length = seq_len[idx].item()
        cache_idx = slot_idx[idx].item()
        pos = input_pos[idx].item()

        # Update compressed_kv portion
        mla_cache[cache_idx, pos : pos + length, :kv_lora_rank] = compressed_kv[
            start : start + length
        ]
        # Update kpe portion
        mla_cache[cache_idx, pos : pos + length, kv_lora_rank:] = kpe[start : start + length]


def _torch_mla_generate_with_absorption(
    q_nope: torch.Tensor,  # [B, 1, N, qk_nope_head_dim]
    q_pe: torch.Tensor,  # [B, 1, N, qk_rope_head_dim]
    compressed_kv: torch.Tensor,  # [B, 1, kv_lora_rank]
    kpe: torch.Tensor,  # [B, 1, 1, qk_rope_head_dim]
    kv_b_proj_weight: torch.Tensor,  # [N * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
    mla_cache: torch.Tensor,  # [max_batch, max_seq, kv_lora_rank + qk_rope_head_dim]
    slot_idx: torch.Tensor,
    input_pos: torch.Tensor,
    scale: float,
    kv_lora_rank: int,
    num_heads: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
    out: torch.Tensor,
) -> None:
    """Generate-only MLA attention with weight absorption.

    Weight absorption: Instead of expanding all cached KV, we absorb kv_b_proj into Q.
    Q_absorbed = Q_nope @ W_k^T  where W_k is the k_nope portion of kv_b_proj_weight

    This avoids expanding potentially thousands of cached tokens.
    """
    b = q_nope.shape[0]

    # Extract k_nope and v portions from kv_b_proj_weight
    # kv_b_proj_weight: [N * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
    # Reshape to [N, qk_nope_head_dim + v_head_dim, kv_lora_rank]
    weight_reshaped = kv_b_proj_weight.view(num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)
    w_k_nope = weight_reshaped[:, :qk_nope_head_dim, :]  # [N, qk_nope_head_dim, kv_lora_rank]
    w_v = weight_reshaped[:, qk_nope_head_dim:, :]  # [N, v_head_dim, kv_lora_rank]

    # Update cache with new tokens
    compressed_kv_flat = compressed_kv.squeeze(1)  # [B, kv_lora_rank]
    kpe_flat = kpe.squeeze(1).squeeze(1)  # [B, qk_rope_head_dim]

    for i in range(b):
        cache_idx = slot_idx[i].item()
        pos = input_pos[i].item()
        mla_cache[cache_idx, pos, :kv_lora_rank] = compressed_kv_flat[i]
        mla_cache[cache_idx, pos, kv_lora_rank:] = kpe_flat[i]

    # Compute attention for each sequence using weight absorption
    for i in range(b):
        cache_idx = slot_idx[i].item()
        pos = input_pos[i].item()

        # Get query for this sequence: [N, qk_nope_head_dim], [N, qk_rope_head_dim]
        q_nope_i = q_nope[i, 0]  # [N, qk_nope_head_dim]
        q_pe_i = q_pe[i, 0]  # [N, qk_rope_head_dim]

        # Retrieve cached data up to current position
        cached_data = mla_cache[cache_idx, : pos + 1]  # [seq_len, kv_lora_rank + qk_rope_head_dim]
        compressed_kv_cached = cached_data[:, :kv_lora_rank]  # [seq_len, kv_lora_rank]
        kpe_cached = cached_data[:, kv_lora_rank:]  # [seq_len, qk_rope_head_dim]

        # =====================================================================
        # Weight absorption for Q_nope part
        # =====================================================================
        # q_absorbed = q_nope @ w_k_nope^T  (absorb k_nope projection into query)
        # q_nope_i: [N, qk_nope_head_dim]
        # w_k_nope: [N, qk_nope_head_dim, kv_lora_rank]
        # q_absorbed: [N, kv_lora_rank]
        q_absorbed = torch.einsum("nd,ndk->nk", q_nope_i, w_k_nope)

        # Attention scores from absorbed Q and compressed KV
        # Compute in fp32 to match FlashInfer's use_fp16_qk_reduction=False
        # q_absorbed: [N, kv_lora_rank], compressed_kv_cached: [seq_len, kv_lora_rank]
        # scores_nope: [N, seq_len]
        scores_nope = torch.matmul(q_absorbed.float(), compressed_kv_cached.float().t())

        # =====================================================================
        # Q_pe part - standard attention with kpe
        # =====================================================================
        # q_pe_i: [N, qk_rope_head_dim], kpe_cached: [seq_len, qk_rope_head_dim]
        # scores_pe: [N, seq_len]
        scores_pe = torch.matmul(q_pe_i.float(), kpe_cached.float().t())

        # Combined attention scores (already in fp32)
        attn_scores = (scores_nope + scores_pe) * scale  # [N, seq_len]

        # Softmax (already in fp32, convert back to input dtype)
        attn_weights = torch.softmax(attn_scores, dim=-1).to(q_nope.dtype)  # [N, seq_len]

        # =====================================================================
        # Compute output with absorbed value projection
        # =====================================================================
        # v_out = attn_weights @ compressed_kv @ w_v^T
        # First: weighted_kv = attn_weights @ compressed_kv_cached -> [N, kv_lora_rank]
        weighted_kv = torch.matmul(attn_weights, compressed_kv_cached)  # [N, kv_lora_rank]

        # Then: attn_out = weighted_kv @ w_v^T -> [N, v_head_dim]
        # w_v: [N, v_head_dim, kv_lora_rank]
        # weighted_kv: [N, kv_lora_rank]
        attn_out = torch.einsum("nk,nvk->nv", weighted_kv, w_v)  # [N, v_head_dim]

        out[i] = attn_out


def _torch_mla_context_with_expansion(
    q_nope: torch.Tensor,  # [total_tokens, N, qk_nope_head_dim]
    q_pe: torch.Tensor,  # [total_tokens, N, qk_rope_head_dim]
    compressed_kv: torch.Tensor,  # [total_tokens, kv_lora_rank]
    kpe: torch.Tensor,  # [total_tokens, 1, qk_rope_head_dim]
    kv_b_proj_weight: torch.Tensor,  # [N * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
    mla_cache: torch.Tensor,  # [max_batch, max_seq, kv_lora_rank + qk_rope_head_dim]
    input_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    seq_len: torch.Tensor,
    seq_start: torch.Tensor,
    scale: float,
    kv_lora_rank: int,
    num_heads: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
    out: torch.Tensor,
) -> None:
    """Context MLA attention with kv_b_proj expansion.

    For prefill, we expand compressed_kv using kv_b_proj_weight and compute
    standard attention. This is more efficient than absorption for prefill
    since we only expand the current tokens, not the full cache.
    """

    # Flatten kpe: [total_tokens, 1, qk_rope_head_dim] -> [total_tokens, qk_rope_head_dim]
    kpe_flat = kpe.squeeze(1)

    # Update cache first with compressed representation
    _update_mla_cache(
        compressed_kv,
        kpe_flat,
        mla_cache,
        seq_len,
        input_pos,
        slot_idx,
        seq_start,
        kv_lora_rank,
    )

    # Compute attention for each sequence
    attn_outputs = []
    for idx in range(seq_len.shape[0]):
        seq_len_i = seq_len[idx].item()
        input_pos_i = input_pos[idx].item()
        slot_idx_i = slot_idx[idx].item()
        seq_start_i = seq_start[idx].item()

        if seq_len_i == 0:
            continue

        # Get query for this sequence
        q_nope_seq = q_nope[
            seq_start_i : seq_start_i + seq_len_i
        ]  # [seq_len_i, N, qk_nope_head_dim]
        q_pe_seq = q_pe[seq_start_i : seq_start_i + seq_len_i]  # [seq_len_i, N, qk_rope_head_dim]

        # Get cached data for attention (includes just-added tokens)
        kv_seq_len = input_pos_i + seq_len_i
        cached_data = mla_cache[
            slot_idx_i, :kv_seq_len
        ]  # [kv_seq_len, kv_lora_rank + qk_rope_head_dim]
        compressed_kv_cached = cached_data[:, :kv_lora_rank]  # [kv_seq_len, kv_lora_rank]
        kpe_cached = cached_data[:, kv_lora_rank:]  # [kv_seq_len, qk_rope_head_dim]

        # =====================================================================
        # Expand compressed_kv using kv_b_proj_weight for this sequence
        # =====================================================================
        # compressed_kv_cached: [kv_seq_len, kv_lora_rank]
        # kv_b_proj_weight: [N * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
        # kv_expanded: [kv_seq_len, N * (qk_nope_head_dim + v_head_dim)]
        kv_expanded = torch.matmul(compressed_kv_cached, kv_b_proj_weight.t())

        # Reshape to [kv_seq_len, N, qk_nope_head_dim + v_head_dim]
        kv_expanded = kv_expanded.view(kv_seq_len, num_heads, qk_nope_head_dim + v_head_dim)

        # Split into k_nope and v
        k_nope_expanded = kv_expanded[:, :, :qk_nope_head_dim]  # [kv_seq_len, N, qk_nope_head_dim]
        v_expanded = kv_expanded[:, :, qk_nope_head_dim:]  # [kv_seq_len, N, v_head_dim]

        # Expand kpe to all heads
        kpe_expanded = kpe_cached.unsqueeze(1).expand(
            -1, num_heads, -1
        )  # [kv_seq_len, N, qk_rope_head_dim]

        # Construct full query and key
        query_full = torch.cat([q_nope_seq, q_pe_seq], dim=-1)  # [seq_len_i, N, qk_head_dim]
        key_full = torch.cat(
            [k_nope_expanded, kpe_expanded], dim=-1
        )  # [kv_seq_len, N, qk_head_dim]

        # Transpose for attention: [1, N, seq_len, head_dim]
        query_t = query_full.transpose(0, 1).unsqueeze(0)  # [1, N, seq_len_i, qk_head_dim]
        key_t = key_full.transpose(0, 1).unsqueeze(0)  # [1, N, kv_seq_len, qk_head_dim]

        # Compute attention scores in fp32 to match FlashInfer's use_fp16_qk_reduction=False
        # FlashInfer uses fp32 accumulation for QK^T, so we do the same for numerical consistency
        attn_scores = (
            torch.matmul(query_t.float(), key_t.float().transpose(-2, -1)) * scale
        )  # [1, N, seq_len_i, kv_seq_len] in fp32

        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len_i, kv_seq_len, device=q_nope.device, dtype=torch.bool),
            diagonal=kv_seq_len - seq_len_i + 1,
        )
        attn_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Softmax (already in fp32, convert back to input dtype)
        attn_weights = torch.softmax(attn_scores, dim=-1).to(q_nope.dtype)

        # Value: [1, N, kv_seq_len, v_head_dim]
        v_t = v_expanded.transpose(0, 1).unsqueeze(0)

        # Compute output
        attn_out = torch.matmul(attn_weights, v_t)  # [1, N, seq_len_i, v_head_dim]
        attn_out = attn_out[0].transpose(0, 1)  # [seq_len_i, N, v_head_dim]

        attn_outputs.append(attn_out)

    # Concatenate all outputs
    if len(attn_outputs) == 0:
        out.zero_()
    elif len(attn_outputs) == 1:
        out.copy_(attn_outputs[0])
    else:
        out.copy_(torch.cat(attn_outputs, dim=0))


@torch.library.custom_op("auto_deploy::torch_cached_mla_with_cache", mutates_args=())
def torch_backend_mla_with_cache(
    # 5 tensor args (get_num_qkv_args = 5)
    q_nope: torch.Tensor,  # [B, S, N, qk_nope_head_dim]
    q_pe: torch.Tensor,  # [B, S, N, qk_rope_head_dim]
    compressed_kv: torch.Tensor,  # [B, S, kv_lora_rank]
    kpe: torch.Tensor,  # [B, S, 1, qk_rope_head_dim]
    kv_b_proj_weight: torch.Tensor,  # [N * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
    # Standard metadata
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    cu_seqlen: torch.Tensor,
    # Cache (FlashInfer layout)
    mla_cache: torch.Tensor,  # [max_batch, max_seq, kv_lora_rank + qk_rope_head_dim]
    # Constants
    scale: Optional[float] = None,
    kv_lora_rank: int = 512,
) -> torch.Tensor:
    """Torch backend MLA with FlashInfer-compatible compressed cache.

    FlashInfer MLA Cache Layout:
        mla_cache: [max_batch, max_seq, kv_lora_rank + qk_rope_head_dim]
        - compressed_kv = mla_cache[:, :, :kv_lora_rank]  (zero-copy slice)
        - kpe = mla_cache[:, :, kv_lora_rank:]  (zero-copy slice)

    Prefill (context): Expand compressed_kv, compute normal attention
    Generate (decode): Use weight absorption for efficiency
    """
    # Get dimensions
    b, s = q_nope.shape[:2]
    num_heads = q_nope.shape[2]
    qk_nope_head_dim = q_nope.shape[3]
    qk_rope_head_dim = q_pe.shape[3]
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    # Infer v_head_dim from kv_b_proj_weight
    out_features = kv_b_proj_weight.shape[0]
    kv_head_dim = out_features // num_heads
    v_head_dim = kv_head_dim - qk_nope_head_dim

    # Get cleaned up metadata
    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode
    seq_len = seq_len[:num_seq]
    input_pos = input_pos[:num_seq]
    slot_idx = slot_idx[:num_seq]
    seq_start = cu_seqlen[:num_seq]

    # Set scale
    if scale is None:
        scale = 1.0 / math.sqrt(qk_head_dim)

    # Define output shape: [B, S, N, v_head_dim]
    output_shape = (b, s, num_heads, v_head_dim)

    if s == 1:
        # =====================================================================
        # Generate phase: Use weight absorption
        # =====================================================================
        y = q_nope.new_empty(b, num_heads, v_head_dim).contiguous()

        _torch_mla_generate_with_absorption(
            q_nope,
            q_pe,
            compressed_kv,
            kpe,
            kv_b_proj_weight,
            mla_cache,
            slot_idx,
            input_pos,
            scale,
            kv_lora_rank,
            num_heads,
            qk_nope_head_dim,
            v_head_dim,
            y,
        )

        return y.unsqueeze(1)  # [B, 1, N, v_head_dim]
    else:
        # =====================================================================
        # Context phase: Expand and compute normal attention
        # =====================================================================
        bs_view = (b * s,)

        q_nope_flat = q_nope.contiguous().view(*bs_view, num_heads, qk_nope_head_dim)
        q_pe_flat = q_pe.contiguous().view(*bs_view, num_heads, qk_rope_head_dim)
        compressed_kv_flat = compressed_kv.contiguous().view(*bs_view, kv_lora_rank)
        kpe_flat = kpe.contiguous().view(*bs_view, 1, qk_rope_head_dim)

        y = q_nope.new_empty(*bs_view, num_heads, v_head_dim).contiguous()

        _torch_mla_context_with_expansion(
            q_nope_flat,
            q_pe_flat,
            compressed_kv_flat,
            kpe_flat,
            kv_b_proj_weight,
            mla_cache,
            input_pos,
            slot_idx,
            seq_len,
            seq_start,
            scale,
            kv_lora_rank,
            num_heads,
            qk_nope_head_dim,
            v_head_dim,
            y,
        )

        return y.view(*output_shape)


@torch_backend_mla_with_cache.register_fake
def torch_backend_mla_with_cache_fake(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    compressed_kv: torch.Tensor,
    kpe: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    cu_seqlen: torch.Tensor,
    mla_cache: torch.Tensor,
    scale: Optional[float] = None,
    kv_lora_rank: int = 512,
) -> torch.Tensor:
    """Fake implementation for torch_backend_mla_with_cache."""
    num_heads = q_nope.shape[2]
    qk_nope_head_dim = q_nope.shape[-1]
    out_features = kv_b_proj_weight.shape[0]
    kv_head_dim = out_features // num_heads
    v_head_dim = kv_head_dim - qk_nope_head_dim

    return q_nope.new_empty(
        q_nope.shape[0], q_nope.shape[1], q_nope.shape[2], v_head_dim
    ).contiguous()


@AttentionRegistry.register("torch_mla")
class MultiHeadLatentAttention(AttentionDescriptor):
    """Attention descriptor for Multi-head Latent Attention (MLA).

    This descriptor uses FlashInfer-compatible compressed cache:
    - torch_mla: source op that expands compressed_kv for attention
    - torch_cached_mla_with_cache: cached op with absorption for generate

    FlashInfer MLA Cache Layout:
        mla_cache: [max_batch, max_seq, head_dim_ckv + head_dim_kpe]
        - No num_heads dimension (MLA-specific optimization)
        - ckv_cached = mla_cache[:, :, :head_dim_ckv]  (zero-copy slice)
        - kpe_cached = mla_cache[:, :, head_dim_ckv:]  (zero-copy slice)

    Reference: https://docs.flashinfer.ai/tutorials/kv_layout.html#mla-page-layout
    """

    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        """Get the attention layout expected by the backend."""
        return "bsnd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        """Get the number of tensor arguments expected by the source op."""
        return 5  # q_nope, q_pe, compressed_kv, kpe, kv_b_proj_weight

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        """Get the source attention op that we target for replacement."""
        return torch.ops.auto_deploy.torch_mla

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        """Get the cached attention op."""
        return torch.ops.auto_deploy.torch_cached_mla_with_cache.default

    @classmethod
    def get_standard_metadata_args(cls) -> List[str]:
        """Get the list of standard metadata arguments."""
        return ["batch_info_host", "seq_len", "input_pos", "slot_idx", "cu_seqlen"]

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
        """Get cache initializers using FlashInfer MLA cache layout."""
        # Extract dimensions from source node args
        # torch_mla signature: q_nope, q_pe, compressed_kv, kpe, kv_b_proj_weight, ...
        compressed_kv_fake = source_attn_node.args[2].meta["val"]
        kpe_fake = source_attn_node.args[3].meta["val"]

        # Get dimensions
        # compressed_kv: [B, S, kv_lora_rank]
        # kpe: [B, S, 1, qk_rope_head_dim]
        kv_lora_rank = compressed_kv_fake.shape[-1]
        qk_rope_head_dim = kpe_fake.shape[-1]

        # FlashInfer MLA cache: [max_batch, max_seq, kv_lora_rank + qk_rope_head_dim]
        # No num_heads dimension - this is the key MLA optimization
        return {
            "mla_cache": UnpagedResourceHandler(
                kv_lora_rank + qk_rope_head_dim,
                dtype=cls.resolve_cache_dtype(cache_config.dtype, compressed_kv_fake.dtype),
            ),
        }

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        """Get constants to pass to the cached attention op."""
        # Extract kv_lora_rank for cache slicing
        compressed_kv_fake = source_attn_node.args[2].meta["val"]
        kv_lora_rank = compressed_kv_fake.shape[-1]

        # Get scale from kwargs
        scale = source_attn_node.kwargs.get("scale", None)

        return [scale, kv_lora_rank]
