import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

import tensorrt_llm
import tensorrt_llm.bindings
import tensorrt_llm.quantization.utils.fp8_utils as fp8_utils
from tensorrt_llm._torch.attention_backend.interface import (
    MLAParams, PositionalEmbeddingParams)
from tensorrt_llm._torch.attention_backend.trtllm import (
    TrtllmAttention, TrtllmAttentionMetadata)
from tensorrt_llm._torch.modules.layer_norm import LayerNorm
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.rotary_embedding import RotaryEmbedding
from tensorrt_llm._torch.pyexecutor.llm_request import (
    LlmRequestState, get_draft_token_length)
from tensorrt_llm._torch.pyexecutor.resource_manager import (BlockManager,
                                                             KVCacheManager)
from tensorrt_llm._utils import get_size_in_bytes
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.bindings.internal.batch_manager import \
    CacheType as CacheTypeCpp
from tensorrt_llm import deep_gemm
from tensorrt_llm.deep_gemm import fp8_mqa_logits, fp8_paged_mqa_logits, get_paged_mqa_logits_metadata
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

ModelConfig = tensorrt_llm.bindings.ModelConfig


def compute_cu_seqlen_kv_bounds_nocache(
    seq_lens: torch.Tensor,
    num_contexts: int,
    num_ctx_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute attention window bounds for batched sequences with causal attention.

    Args:
        seq_lens: Sequence lengths [num_contexts], dtype=torch.int32
        num_contexts: Number of sequences in the batch
        num_ctx_tokens: Total number of tokens across all sequences

    Returns:
        cu_seqlen_ks: Start index in KV for each token [num_ctx_tokens]
        cu_seqlen_ke: End index (exclusive) in KV for each token [num_ctx_tokens]
    """
    device = seq_lens.device
    # Cumulative sequence offsets: where each sequence starts
    cu_seq_offsets = torch.cat([
        torch.zeros(1, device=device, dtype=torch.int32),
        torch.cumsum(seq_lens, dim=0)
    ])  # [num_contexts + 1]

    # Map each token to its batch: [0,0,...,0, 1,1,...,1, ..., B-1,B-1,...,B-1]
    batch_ids = torch.repeat_interleave(
        torch.arange(num_contexts, device=device, dtype=torch.int32),
        seq_lens)  # [num_ctx_tokens]

    # Each token's KV window starts at its sequence's start
    cu_seqlen_ks = cu_seq_offsets[batch_ids]  # [num_ctx_tokens]

    # Compute local position within each sequence (0-based)
    global_positions = torch.arange(num_ctx_tokens,
                                    device=device,
                                    dtype=torch.int32)
    local_positions = global_positions - torch.repeat_interleave(
        cu_seq_offsets[:-1], seq_lens)

    # Causal: token at local position j attends to [seq_start, seq_start + j + 1)
    cu_seqlen_ke = cu_seqlen_ks + local_positions + 1  # [num_ctx_tokens]

    return cu_seqlen_ks, cu_seqlen_ke


class DSAtrtllmAttentionMetadata(TrtllmAttentionMetadata):
    # Store reference to indexer for preparation stage
    indexer: Optional["Indexer"] = None

    def __post_init__(self):
        super().__post_init__()
        self.low_rank_cache_block_offsets = torch.empty(
            [
                self.max_num_sequences,
                self.kv_cache_manager.max_low_rank_blocks_per_seq
            ],
            dtype=torch.int32,
            device='cuda',
        )

    def prepare(self):
        super().prepare()
        if self.kv_cache_manager is not None:
            self.host_low_rank_cache_block_offsets = \
                self.kv_cache_manager.get_low_rank_block_offsets(self.request_ids)
            self.low_rank_cache_block_offsets[:self.num_seqs].copy_(
                self.host_low_rank_cache_block_offsets[:self.num_seqs],
                non_blocking=True)

        # Prepare DSA indexer if available
        if self.indexer is not None:
            self.indexer.prepare(self)


class Indexer(nn.Module):

    def __init__(self, quant_config: Optional[QuantConfig],
                 pos_embd_params: Optional[PositionalEmbeddingParams],
                 mla_params: Optional[MLAParams],
                 skip_create_weights_in_init: bool,
                 sparse_attention_config: "SparseAttentionConfig",
                 dtype: Optional[torch.dtype],
                 layer_idx: int = 0):
        super().__init__()
        self.hidden_size = mla_params.hidden_size
        self.q_lora_rank = mla_params.q_lora_rank
        self.rope_dim = mla_params.qk_rope_head_dim
        self.n_heads = sparse_attention_config.index_n_heads  # 64
        self.head_dim = sparse_attention_config.index_head_dim  # 128
        self.index_topk = sparse_attention_config.index_topk  # 2048
        self.layer_idx = layer_idx
        self.cache_manager = None  # Will be set by DSATrtllmAttention

        self.wq_b = Linear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights_in_init=skip_create_weights_in_init,
            use_custom_cublas_mm=True)
        self.wk = Linear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights_in_init=skip_create_weights_in_init,
            use_custom_cublas_mm=True)
        self.k_norm = LayerNorm(hidden_size=self.head_dim,
                                eps=1e-6,
                                dtype=torch.float32)
        self.weights_proj = Linear(
            self.hidden_size,
            self.n_heads,
            bias=False,
            dtype=torch.get_default_dtype(),
            quant_config=None,
            skip_create_weights_in_init=skip_create_weights_in_init,
            use_custom_cublas_mm=True)

        self.rotary_emb = RotaryEmbedding(
            pos_embd_params.rope,
            head_dim=self.rope_dim,
            is_neox=pos_embd_params.is_neox,
        )

        self.softmax_scale = self.head_dim**-0.5
        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128
        self._tokens_per_block = None  # Will be retrieved from cache_manager
        # Separate slot mappings for non-interleaved layout (flat byte indices)
        self.slot_mapping_fp8 = None    # [total_tokens] flat byte indices for FP8 data start
        self.slot_mapping_scale = None  # [total_tokens] flat byte indices for scale start

        # TODO: Consolidate these parameters into decoder metadata.
        # Preparation stage buffers
        self.num_sms = deep_gemm.get_num_sms()
        self.scheduler_metadata_buffer = torch.empty((self.num_sms + 1, 2),
                                                dtype=torch.int32,
                                                device='cuda')
        # Decode phase metadata
        self.decode_context_lens = None  # Context lengths for decode phase
        self.decode_block_table = None   # Block table for decode phase

    def set_cache_manager(self, cache_manager):
        """Set the cache manager reference for KV cache operations."""
        self.cache_manager = cache_manager
        if cache_manager is not None:
            self._tokens_per_block = cache_manager.tokens_per_block

    @property
    def tokens_per_block(self):
        """Get tokens_per_block from cache_manager or use default."""
        if self._tokens_per_block is not None:
            return self._tokens_per_block
        if self.cache_manager is not None:
            return self.cache_manager.tokens_per_block
        return None

    def prepare(self, metadata: TrtllmAttentionMetadata):
        """
        Prepare indexer for the forward pass.
        This should be called during metadata.prepare() stage.

        - Computes slot_mapping for KV cache updates
        - Prepares schedule_metadata for fp8_paged_mqa_logits
        - Stores generation request IDs for decode phase
        """
        if self.cache_manager is None:
            return
        num_contexts = metadata.num_contexts
        num_generations = metadata.num_generations


        # Prepare for decode phase if there are generation requests
        if num_generations > 0:
            # Get context_lens for generation requests (total tokens including history + new token)
            context_lens = metadata.kv_lens[num_contexts:num_contexts + num_generations].to(torch.int32)

            # Get generation request IDs
            gen_request_ids = metadata.request_ids[num_contexts:num_contexts + num_generations]

            # Get block table for generation requests
            block_table = self.cache_manager.get_indexer_k_block_offsets(gen_request_ids)

            # Prepare schedule metadata for fp8_paged_mqa_logits
            # This is a preprocessing step that computes scheduling information for the kernel
            blocksize = self.tokens_per_block
            self.scheduler_metadata_buffer[:] = get_paged_mqa_logits_metadata(
                context_lens, blocksize, self.num_sms
            )

            # Store decode metadata for use in forward pass
            self.decode_context_lens = context_lens
            self.decode_block_table = block_table
        else:
            # Clear decode metadata if no generation requests
            self.decode_context_lens = None
            self.decode_block_table = None

        # Compute slot_mapping for all requests (both context and generation)
        # This maps each token to its flat cache position for vectorized KV cache updates
        request_ids = metadata.request_ids
        seq_lens = metadata.seq_lens

        # num_tokens_per_request is the same for both context and generation
        num_tokens_per_request = seq_lens

        # start_positions: where to start inserting tokens for each request
        cached_tokens = metadata.kv_cache_params.num_cached_tokens_per_seq
        start_positions = torch.tensor(cached_tokens, dtype=torch.int32)

        # Compute the flat slot mapping for all tokens (separate FP8 and scale offsets)
        self._compute_slot_mapping(request_ids, start_positions, num_tokens_per_request)

    def _compute_slot_mapping(self, request_ids: List[int],
                              start_positions: torch.Tensor,
                              num_tokens_per_request: torch.Tensor) -> None:
        """
        Compute flat byte-level slot mappings for indexer k cache updates
        Computes separate flat indices for FP8 data and scales.

        Args:
            request_ids: List of request IDs
            start_positions: Starting position in cache for each request
            num_tokens_per_request: Number of tokens to insert for each request

        Sets:
            self.slot_mapping_fp8: [total_tokens] - flat byte index for FP8 data start
            self.slot_mapping_scale: [total_tokens] - flat byte index for scale start
        """
        block_offsets = self.cache_manager.get_indexer_k_block_offsets(request_ids)
        total_tokens = num_tokens_per_request.sum().item()

        head_dim = self.head_dim
        scale_size = head_dim // self.quant_block_size * 4  # float32 = 4 bytes
        block_stride = self.tokens_per_block * (head_dim + scale_size)  # Bytes per block
        scale_base_offset = self.tokens_per_block * head_dim  # Offset to scale region in block

        # Preallocate slot mappings: [total_tokens] flat byte indices
        self.slot_mapping_fp8 = torch.full((total_tokens,), -1, dtype=torch.int64, device=start_positions.device)
        self.slot_mapping_scale = torch.full((total_tokens,), -1, dtype=torch.int64, device=start_positions.device)

        token_idx = 0
        for req_idx, req_id in enumerate(request_ids):
            num_tokens = num_tokens_per_request[req_idx].item()
            start_pos = start_positions[req_idx].item()

            # Compute slots for each token in this request
            for local_token_idx in range(num_tokens):
                global_pos = start_pos + local_token_idx
                block_idx_in_seq = global_pos // self.tokens_per_block
                pos_in_block = global_pos % self.tokens_per_block

                # Get physical block ID
                if block_idx_in_seq < block_offsets.shape[1]:
                    block_id = block_offsets[req_idx, block_idx_in_seq].item()
                    if block_id >= 0:
                        # Flat byte index for FP8 data
                        fp8_flat_idx = block_id * block_stride + pos_in_block * head_dim
                        self.slot_mapping_fp8[token_idx] = fp8_flat_idx

                        # Flat byte index for scale
                        scale_flat_idx = block_id * block_stride + scale_base_offset + pos_in_block * scale_size
                        self.slot_mapping_scale[token_idx] = scale_flat_idx

                token_idx += 1

        self.slot_mapping_fp8 = self.slot_mapping_fp8.cuda(non_blocking=True)
        self.slot_mapping_scale = self.slot_mapping_scale.cuda(non_blocking=True)


    def _update_k_cache(self, k_fp8: torch.Tensor, k_scale: torch.Tensor) -> None:
        """
        Insert/append k values and scales into the indexer k cache using pre-computed slot mappings.
        Uses flat byte indices with vectorized scatter.

        Note: slot_mapping_fp8 and slot_mapping_scale should be pre-computed in prepare() stage
              and should only contain valid indices.

        Args:
            k_fp8: FP8 quantized k tensor, shape [total_tokens, head_dim]
            k_scale: Scaling factors, shape [total_tokens, head_dim // quant_block_size]
        """
        if self.cache_manager is None or self.slot_mapping_fp8 is None:
            return

        k_cache = self.cache_manager.get_indexer_k_cache_buffers(self.layer_idx)
        k_cache_flat = k_cache.view(-1)  # Flatten to 1D for byte-level indexing

        num_tokens = k_fp8.shape[0]
        head_dim = k_fp8.shape[1]
        scale_size = k_scale.shape[1] * 4  # Convert to bytes (float32 = 4 bytes)

        # Convert to bytes: flatten first, then view as uint8, then reshape
        k_fp8_bytes = k_fp8.contiguous().view(-1).view(torch.uint8).view(num_tokens, head_dim)
        k_scale_bytes = k_scale.contiguous().view(-1).view(torch.uint8).view(num_tokens, scale_size)

        # Scatter FP8 data
        flat_indices_fp8 = self.slot_mapping_fp8[:num_tokens]  # [num_tokens] start indices
        byte_offsets = torch.arange(head_dim, device=k_cache.device).unsqueeze(0)  # [1, head_dim]
        scatter_indices_fp8 = flat_indices_fp8.unsqueeze(1) + byte_offsets  # [num_tokens, head_dim]
        k_cache_flat[scatter_indices_fp8] = k_fp8_bytes

        # Scatter scales
        flat_indices_scale = self.slot_mapping_scale[:num_tokens]  # [num_tokens] start indices
        byte_offsets = torch.arange(scale_size, device=k_cache.device).unsqueeze(0)  # [1, scale_size]
        scatter_indices_scale = flat_indices_scale.unsqueeze(1) + byte_offsets  # [num_tokens, scale_size]
        k_cache_flat[scatter_indices_scale] = k_scale_bytes

    def sparse_attn_indexer(
        self,
        metadata: TrtllmAttentionMetadata,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:

        num_contexts = metadata.num_contexts
        num_generations = metadata.num_generations
        num_ctx_tokens = metadata.num_ctx_tokens
        num_tokens = metadata.num_tokens

        has_decode = num_generations > 0
        has_prefill = num_contexts > 0
        num_gen_tokens = num_tokens - num_ctx_tokens

        topk_indices_buffer = torch.empty(
            (hidden_states.shape[0], self.index_topk),
            dtype=torch.int32,
            device=hidden_states.device)
        topk_indices_buffer[:hidden_states.shape[0]] = -1

        # Quantize k and store into indexer k cache
        k_fp8, k_scale = torch.ops.trtllm.fp8_quantize_1x128(k[:num_tokens])
        self._update_k_cache(k_fp8, k_scale)

        if has_prefill:
            # Compute attention window bounds for each query token in batched sequences
            # cu_seqlen_ks[i]: start index in global KV for query token i
            # cu_seqlen_ke[i]: end index (exclusive) in global KV for query token i
            seq_lens = metadata.seq_lens_cuda[:num_contexts].to(torch.int32)

            # FIXME: better way to retrieve cu_seqlen_ks and cu_seqlen_ke from kv cache or attn metadata?
            # TODO: maybe move this to prepare() stage
            cu_seqlen_ks, cu_seqlen_ke = compute_cu_seqlen_kv_bounds_nocache(
                seq_lens, num_contexts, num_ctx_tokens)

            logits = fp8_mqa_logits(
                q_fp8[:num_ctx_tokens, ...],
                (k_fp8[:num_ctx_tokens, ...], k_scale[:num_ctx_tokens, ...]),
                weights[:num_ctx_tokens, ...],
                cu_seqlen_ks,
                cu_seqlen_ke,
            )
            topk_indices = logits.topk(min(self.index_topk, logits.shape[-1]),
                                       dim=-1)[1]
            topk_indices -= cu_seqlen_ks[:, None]
            mask_lo = topk_indices >= 0
            mask_hi = topk_indices - (cu_seqlen_ke - cu_seqlen_ks)[:, None] < 0
            mask = torch.full_like(topk_indices,
                                   False,
                                   dtype=torch.bool,
                                   device=topk_indices.device)
            mask = mask_lo & mask_hi
            topk_indices = topk_indices.masked_fill(~mask, -1)
            topk_indices_buffer[:num_ctx_tokens, :topk_indices.
                                shape[-1]] = topk_indices.to(dtype=torch.int32)

        if has_decode:
            max_seq_len = self.cache_manager.max_seq_len
            # Get decode lengths per request (from seq_lens) for validation
            gen_seq_lens = metadata.seq_lens[num_contexts:num_contexts + num_generations]
            max_decode_len = gen_seq_lens.max().item()
            min_decode_len = gen_seq_lens.min().item()
            assert max_decode_len == min_decode_len, "max_decode_len != min_decode_len, we need padding"

            # Reshape q for decode phase: [num_gen_tokens, ...] -> [batch_size, next_n, ...]
            q_decode = q_fp8[num_ctx_tokens:num_ctx_tokens + num_gen_tokens, ...]
            q_decode = q_decode.view(num_generations, -1, *q_fp8.shape[1:])
            batch_size = q_decode.shape[0]
            next_n = q_decode.shape[1]
            assert num_gen_tokens == batch_size * next_n
            weights_decode = weights[num_ctx_tokens:num_ctx_tokens + num_gen_tokens, ...]

            # Get k cache and call fp8_paged_mqa_logits with prepared decode metadata
            k_cache = self.cache_manager.get_indexer_k_cache_buffers(self.layer_idx) # [num_blocks, tokens_per_block, 1, head_dim + scale_size]
            logits_decode = fp8_paged_mqa_logits(
                q_decode,
                k_cache,
                weights_decode,
                self.decode_context_lens,  # context_lens prepared in prepare()
                self.decode_block_table,    # block_table prepared in prepare()
                self.scheduler_metadata_buffer,
                max_seq_len
            )
            # padded
            positions = torch.arange(max_seq_len,
                                    device=q_decode.device).unsqueeze(0).expand(
                                        num_gen_tokens, -1)
            row_indices = torch.arange(num_gen_tokens,
                                    device=q_decode.device) // next_n
            next_n_offset = torch.arange(
                num_gen_tokens,
                device=q_decode.device) % next_n
            index_end_pos = (self.decode_context_lens[row_indices] - next_n +
                            next_n_offset).unsqueeze(1)

            # index_end_pos: [B * N, 1]
            mask = positions <= index_end_pos
            # mask: [B * N, L]
            logits_decode = logits_decode.masked_fill(~mask, float('-inf'))
            topk_indices_decode = logits_decode.topk(self.index_topk,
                                    dim=-1)[1].to(torch.int32)  # [B * N, K]
            # ensure we don't set indices for the top k
            # that is out of range(masked already)
            # this will happen if context length is shorter than K
            topk_indices_decode[topk_indices_decode > index_end_pos] = -1

            # Store in buffer
            topk_indices_buffer[num_ctx_tokens:num_ctx_tokens + num_gen_tokens,
                               :topk_indices_decode.shape[-1]] = topk_indices_decode.to(dtype=torch.int32)

        return topk_indices_buffer

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                metadata: TrtllmAttentionMetadata,
                hidden_states: Optional[torch.Tensor] = None,
                qr: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None):
        q = self.wq_b(qr)
        q = q.view(-1, self.n_heads, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1)

        k = self.wk(hidden_states)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1)

        q_pe, k_pe = self.rotary_emb(position_ids, [q_pe, k_pe])
        q = torch.cat([q_pe, q_nope], dim=-1)
        k = torch.cat([k_pe.squeeze(1), k_nope], dim=-1)

        # we only quant q here since k quant is fused with cache insertion
        q = q.view(-1, self.head_dim)
        q_fp8, q_scale = fp8_utils.per_token_quant_and_transform(
            q, self.quant_block_size, scale_ue8m0=self.scale_fmt == "ue8m0")
        q_fp8 = q_fp8.view(-1, self.n_heads, self.head_dim)
        q_scale = q_scale.view(-1, self.n_heads, 1)

        weights = self.weights_proj(hidden_states)
        weights = weights.unsqueeze(
            -1) * q_scale * self.softmax_scale * self.n_heads**-0.5
        weights = weights.squeeze(-1)

        topk_indices_buffer = self.sparse_attn_indexer(metadata, hidden_states, q_fp8, k, weights)
        # TODO: from topk_indices_buffer ([num_tokens, index_topk]), retrieve sparse_attn_indices, sparse_attn_offsets

        return None, None  # sparse_attn_indices, sparse_attn_offsets


class DSATrtllmAttention(TrtllmAttention, nn.Module):
    Metadata = DSAtrtllmAttentionMetadata

    def __init__(
            self,
            layer_idx: int,
            num_heads: int,
            head_dim: int,
            num_kv_heads: Optional[int] = None,
            quant_config: Optional[QuantConfig] = None,
            q_scaling: Optional[float] = None,
            pos_embd_params: Optional[PositionalEmbeddingParams] = None,
            mla_params: Optional[MLAParams] = None,
            skip_create_weights_in_init: bool = False,
            attention_chunk_size: Optional[int] = None,
            sparse_attention_config: Optional["SparseAttentionConfig"] = None,
            dtype: Optional[torch.dtype] = None,
            **kwargs):
        TrtllmAttention.__init__(
            self,
            layer_idx,
            num_heads,
            head_dim,
            sparse_attention_config=sparse_attention_config,
            num_kv_heads=num_kv_heads,
            quant_config=quant_config,
            q_scaling=q_scaling,
            pos_embd_params=pos_embd_params,
            mla_params=mla_params,
            skip_create_weights_in_init=skip_create_weights_in_init,
            attention_chunk_size=attention_chunk_size,
            **kwargs)
        nn.Module.__init__(self)

        self.indexer = Indexer(quant_config, pos_embd_params, mla_params,
                               skip_create_weights_in_init,
                               sparse_attention_config, dtype, layer_idx)

    def sparse_attn_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: DSAtrtllmAttentionMetadata,
        hidden_states: Optional[torch.Tensor] = None,
        qr: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Set index cache manager reference if not already set
        if self.indexer.cache_manager is None and metadata.kv_cache_manager is not None:
            self.indexer.set_cache_manager(metadata.kv_cache_manager)

        # Dynamically append indexer into DSA Attention metadata, to call indexer.prepare() during metadata.prepare() stage
        # TODO: this might not be here, maybe during attn creation, we append indexer into metadata
        if metadata.indexer is None:
            metadata.indexer = self.indexer

        return self.indexer(q, k, metadata, hidden_states, qr, position_ids)

    def sparse_kv_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: TrtllmAttentionMetadata,
        hidden_states: Optional[torch.Tensor] = None,
        qr: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return None, None


class DSACacheManager(KVCacheManager):

    def __init__(
        self,
        kv_cache_config: KvCacheConfig,
        kv_cache_type: CacheTypeCpp,
        *,
        num_layers: int,
        num_kv_heads: Union[int, List[Optional[int]]],
        head_dim: int,
        tokens_per_block: int,
        # Note that max_seq_len is not necessarily equal to kv_cache_config.num_tokens.
        # It's derived from the model's BuildConfig for consistency with the C++ backend.
        max_seq_len: int,
        max_batch_size: int,
        mapping: Mapping,
        dtype: DataType = DataType.HALF,
        spec_config: Optional["DecodingBaseConfig"] = None,
        layer_mask: Optional[List[bool]] = None,
        max_num_tokens: int = 8192,
        model_config: Optional[ModelConfig] = None,
        max_beam_width: int = 1,
        sparse_attn_config: Optional["SparseAttentionConfig"] = None,
        **kwargs,
    ) -> None:

        assert not kv_cache_config.enable_block_reuse, "DSA cache requires block reuse to be disabled in KV cache config"
        self.quant_block_size = 128
        self.index_head_dim = sparse_attn_config.index_head_dim

        super().__init__(
            kv_cache_config,
            kv_cache_type,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=mapping,
            dtype=dtype,
            spec_config=spec_config,
            layer_mask=layer_mask,
            max_num_tokens=max_num_tokens,
            model_config=model_config,
            max_beam_width=max_beam_width,
            **kwargs,
        )

        # Per layer low rank cache pool
        self.num_blocks = self.blocks_in_primary_pool
        self.low_rank_cache_pool_per_layer = [
            torch.empty(
                (self.num_blocks, tokens_per_block, 1, self.index_head_dim +
                 self.index_head_dim // self.quant_block_size * 4),
                device="cuda",
                dtype=torch.uint8) for _ in range(self.num_local_layers)
        ]
        self.max_low_rank_blocks_per_seq = self.num_blocks

        # Block manager to manage the low rank cache blocks for each request. Different layers share the
        # same block ids.
        self.low_rank_cache_manager = BlockManager(self.num_local_layers, self.num_blocks,
                                                   tokens_per_block)

        # Indexer K cache pool for DSA attention
        # Shape: [num_blocks, tokens_per_block * (index_head_dim + scale_size)]
        # Non-interleaved layout: [fp8_tok0 | fp8_tok1 | ... | scale_tok0 | scale_tok1 | ...]
        # Store FP8-quantized k values from the indexer
        scale_size = self.index_head_dim // self.quant_block_size * 4
        self.indexer_k_cache_pool_per_layer = [
            torch.empty(
                (self.num_blocks, tokens_per_block * (self.index_head_dim + scale_size)),
                device="cuda",
                dtype=torch.uint8) for _ in range(self.num_local_layers)
        ]

    def add_dummy_requests(
        self,
        request_ids: List[int],
        token_nums: Optional[List[int]] = None,
        is_gen: bool = False,
        prepare_resource: bool = True,
        max_num_draft_tokens: int = 0,
        use_mrope: bool = False,
        max_beam_width: int = 1,
    ):
        requests = super().add_dummy_requests(
            request_ids=request_ids,
            token_nums=token_nums,
            is_gen=is_gen,
            prepare_resource=prepare_resource,
            max_num_draft_tokens=max_num_draft_tokens,
            use_mrope=use_mrope,
            max_beam_width=max_beam_width,
        )
        if prepare_resource:
            for req in requests:
                request_id = req.py_request_id
                self.low_rank_cache_manager.add_tokens(request_id,
                                                       req.max_beam_num_tokens)
        return requests

    def get_low_rank_buffers(self, layer_idx: int):
        return self.low_rank_cache_pool_per_layer[layer_idx]

    def get_low_rank_block_offsets(self,
                                   request_ids: List[int]) -> torch.Tensor:
        return self.low_rank_cache_manager.get_block_offsets(request_ids)

    def get_indexer_k_cache_buffers(self, layer_idx: int):
        """Get indexer k cache buffer from a specific layer pool."""
        block_size = self.low_rank_cache_manager.tokens_per_block
        per_token_size = self.index_head_dim + self.index_head_dim // self.quant_block_size * 4
        return self.indexer_k_cache_pool_per_layer[layer_idx].view(self.num_blocks, block_size, 1, per_token_size)

    def get_indexer_k_block_offsets(self, request_ids: List[int]) -> torch.Tensor:
        """Get block offsets for indexer k cache """
        return self.low_rank_cache_manager.get_block_offsets(request_ids)

    def prepare_resources(self, scheduled_batch):
        """
        Prepare resources for both low rank cache and indexer K cache.
        """
        super().prepare_resources(scheduled_batch)
        context_batch = scheduled_batch.context_requests
        generation_batch = scheduled_batch.generation_requests

        # Allocate blocks for context requests
        for req in context_batch:
            request_id = req.py_request_id
            if req.is_first_context_chunk:
                self.low_rank_cache_manager.add_tokens(request_id, req.prompt_len)
                # TODO: Add support for num_extra_kv_tokens
                # for _ in range(self.num_extra_kv_tokens):
                #     self.low_rank_cache_manager.add_tokens(request_id, 1)

                # Add draft tokens
                draft_token_len = get_draft_token_length(req)
                if draft_token_len > 0:
                    self.low_rank_cache_manager.add_tokens(request_id, draft_token_len)

        # Allocate blocks for generation requests
        for req in generation_batch:
            request_id = req.py_request_id
            self.low_rank_cache_manager.add_tokens(request_id, 1)
            # Add draft tokens
            draft_token_len = get_draft_token_length(req)
            if draft_token_len > 0:
                self.low_rank_cache_manager.add_tokens(request_id, draft_token_len)

        # TODO: Support beam search, current assume beam_width=1

    def update_resources(self, scheduled_batch):
        for request in scheduled_batch.context_requests:
            if request.state != LlmRequestState.GENERATION_COMPLETE:
                seq_len = request.get_num_tokens(0)
                rewind_len = max(seq_len - 1 - self.prompt_budget, 0)
                self.rewind_kv_cache(request, rewind_len)
                self.low_rank_cache_manager.rewind_cache(request, rewind_len)

    def free_resources(self, request):
        super().free_resources(request)
        self.low_rank_cache_manager.free_resources(request)

    @staticmethod
    def get_cache_size_per_token(model_config: ModelConfig, mapping: Mapping):
        sparse_attn_config = model_config.sparse_attention_config
        quant_block_size = 128

        # get kv cache dtype bytes
        mem_per_token = 2
        quant_config = model_config.quant_config
        if quant_config is not None and quant_config.quant_mode.has_fp8_kv_cache(
        ):
            mem_per_token = 1

        # get num key value heads
        num_key_value_heads = 1

        # get head dim
        tp_size = 1 if mapping.enable_attention_dp else mapping.tp_size
        head_dim = sparse_attn_config.index_head_dim
        head_dim = head_dim * num_key_value_heads // tp_size

        # provide at least 1 layer to prevent division by zero cache size
        num_attention_layers = max(
            len(mapping.pp_layers(model_config.get_num_attention_layers())), 1)
        mem_per_token *= num_attention_layers * head_dim

        # 1 for K, others for low rank cache
        kv_factor = 1 + (head_dim + head_dim // quant_block_size * 4) / head_dim
        mem_per_token *= kv_factor
        return mem_per_token

    def get_cache_bytes_per_token(self):
        # 1 for K, others for low rank cache
        kv_factor = self.kv_factor + (
            self.index_head_dim + self.index_head_dim // self.quant_block_size *
            4) / self.index_head_dim
        cache_size_per_token = math.ceil(
            kv_factor * sum(self.num_kv_heads_per_layer) * self.head_dim)

        if self.dtype not in (DataType.FP8, DataType.HALF, DataType.BF16,
                              DataType.FLOAT, DataType.NVFP4):
            raise ValueError(f'Cannot support {self.dtype} KV cache.')

        cache_size_bytes_per_token = get_size_in_bytes(cache_size_per_token,
                                                       self.dtype)
        if self.dtype == DataType.NVFP4:
            cache_size_bytes_per_token += self.calculate_scaling_factor_size_bytes(
                cache_size_per_token,
                quant_vector_size=16,
                scaling_factor_dtype=DataType.FP8)
        return cache_size_bytes_per_token
