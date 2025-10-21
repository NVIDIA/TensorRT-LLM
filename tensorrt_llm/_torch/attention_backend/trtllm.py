import math
import os
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch

if TYPE_CHECKING:
    from ..speculative.utils import SpecDecodingTensor

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.functional import AttentionMaskType
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig

from ..utils import (compute_swizzled_sf_shape, get_global_attrs,
                     get_model_extra_attrs)
from .interface import (AttentionBackend, AttentionInputType, AttentionMask,
                        AttentionMetadata, KVCacheParams, MLAParams,
                        PositionalEmbeddingParams, PredefinedAttentionMask,
                        RopeParams)


@dataclass(kw_only=True, init=False)
class TrtllmAttentionWrapper:
    sequence_length: torch.Tensor
    host_past_key_value_lengths: torch.Tensor
    host_total_kv_lens: torch.Tensor
    context_lengths: torch.Tensor
    host_context_lengths: torch.Tensor
    host_request_types: torch.Tensor
    kv_cache_block_offsets: torch.Tensor
    host_kv_cache_block_offsets: torch.Tensor
    host_kv_cache_pool_pointers: torch.Tensor
    host_kv_cache_pool_mapping: torch.Tensor
    workspace: Optional[torch.Tensor]
    cache_indirection: Optional[torch.Tensor]
    kv_scale_orig_quant: Optional[torch.Tensor]
    kv_scale_quant_orig: Optional[torch.Tensor]
    out_scale: Optional[torch.Tensor]
    rotary_inv_freq: Optional[torch.Tensor]
    rotary_cos_sin: Optional[torch.Tensor]
    layer_idx: int
    num_heads: int
    num_kv_heads: int
    head_size: int
    tokens_per_block: int
    max_num_requests: int
    max_context_length: int
    attention_window_size: int
    sink_token_length: int
    beam_width: int
    predicted_tokens_per_seq: int
    quant_mode: int
    position_embedding_type: int
    rotary_embedding_dim: int
    rotary_embedding_base: float
    rotary_embedding_scale_type: int
    rotary_embedding_scale: float
    rotary_embedding_short_m_scale: float
    rotary_embedding_long_m_scale: float
    rotary_embedding_max_positions: int
    rotary_embedding_original_max_positions: int
    use_paged_context_fmha: bool
    is_mla_enable: bool
    q_lora_rank: Optional[int]
    kv_lora_rank: Optional[int]
    qk_rope_head_dim: Optional[int]
    qk_nope_head_dim: Optional[int]
    v_head_dim: Optional[int]
    chunked_prefill_buffer_batch_size: Optional[int]
    attention_chunk_size: Optional[int]
    softmax_stats_tensor: Optional[torch.Tensor]
    use_spec_decoding: bool
    is_spec_dec_tree: bool
    spec_decoding_position_offsets: Optional[torch.Tensor]
    spec_decoding_packed_mask: Optional[torch.Tensor]
    spec_decoding_generation_lengths: Optional[torch.Tensor]
    kwargs: dict

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        num_kv_heads: Optional[int] = None,
        pos_embd_params: Optional[PositionalEmbeddingParams] = None,
        q_scaling: Optional[float] = None,
        mla_params: Optional[MLAParams] = None,
        attention_chunk_size: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize the attention wrapper.
        Args:
            num_heads (int): The number of query heads.
            head_dim (int): The size of each attention head (hidden_size // num_heads).
            num_kv_heads (int): The number of kv heads. Defaults to num_heads if None.
            pos_embd_params (PositionalEmbeddingParams): Optional parameters defining how positional embedding should be applied.
        """
        rope_params = None
        if pos_embd_params is not None:
            rope_params = pos_embd_params.rope
        rope_params = rope_params or RopeParams()
        self.rope_params = rope_params

        self.is_mla_enable = mla_params is not None
        self.q_scaling = q_scaling or 1.0
        self.predicted_tokens_per_seq = 1
        self.attention_chunk_size = attention_chunk_size

        if self.is_mla_enable:
            self.q_lora_rank = mla_params.q_lora_rank
            self.kv_lora_rank = mla_params.kv_lora_rank
            self.qk_nope_head_dim = mla_params.qk_nope_head_dim
            self.qk_rope_head_dim = mla_params.qk_rope_head_dim
            self.v_head_dim = mla_params.v_head_dim
            self.predicted_tokens_per_seq = mla_params.predicted_tokens_per_seq
        else:
            self.q_lora_rank = None
            self.kv_lora_rank = None
            self.qk_nope_head_dim = None
            self.qk_rope_head_dim = None
            self.v_head_dim = None

        self.rotary_inv_freq, self.rotary_cos_sin = self.rope_params.create_rope_const_params(
        )

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_size = head_size
        self.position_embedding_type = int(
            pos_embd_params.type) if pos_embd_params is not None else 0
        self.rotary_embedding_dim = rope_params.dim
        self.rotary_embedding_base = rope_params.theta
        self.rotary_embedding_scale_type = int(rope_params.scale_type)
        self.rotary_embedding_scale = rope_params.scale
        self.rotary_embedding_short_m_scale = rope_params.short_m_scale
        self.rotary_embedding_long_m_scale = rope_params.long_m_scale
        self.rotary_embedding_max_positions = rope_params.max_positions
        self.rotary_embedding_original_max_positions = rope_params.original_max_positions
        self.kwargs = {}
        self.kwargs.update(kwargs)

    def update_quant_config(self, quant_config: Optional[QuantConfig] = None):
        quant_config = quant_config or QuantConfig()
        self.quant_mode = int(quant_config.layer_quant_mode)

    def plan(
        self,
        *,
        layer_idx: int = 0,
        tokens_per_block: Optional[int] = None,
        max_num_requests: int = 0,
        max_sequence_length: int = 0,
        max_context_length: int = 0,
        attention_window_size: Optional[int] = None,
        sink_token_length: int = 0,
        beam_width: int = 1,
        sequence_length: torch.Tensor = ...,
        host_past_key_value_lengths: torch.Tensor = ...,
        host_total_kv_lens: torch.Tensor = ...,
        context_lengths: torch.Tensor = ...,
        host_context_lengths: torch.Tensor = ...,
        host_request_types: torch.Tensor = ...,
        kv_cache_block_offsets: Optional[torch.Tensor] = None,
        host_kv_cache_block_offsets: Optional[torch.Tensor] = None,
        host_kv_cache_pool_pointers: Optional[torch.Tensor] = None,
        host_kv_cache_pool_mapping: Optional[torch.Tensor] = None,
        block_ids_per_seq: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        cache_indirection: Optional[torch.Tensor] = None,
        kv_scale_orig_quant: Optional[torch.Tensor] = None,
        kv_scale_quant_orig: Optional[torch.Tensor] = None,
        out_scale: Optional[torch.Tensor] = None,
        out_scale_sf: Optional[torch.Tensor] = None,
        kv_scales_sf: Optional[torch.Tensor] = None,
        kv_scales_sf_inv: Optional[torch.Tensor] = None,
        use_nvfp4_output: bool = False,
        use_paged_context_fmha: bool = False,
        attention_input_type: AttentionInputType = AttentionInputType.mixed,
        latent_cache: Optional[torch.Tensor] = None,
        q_pe: Optional[torch.Tensor] = None,
        mrope_config: Optional[dict] = None,
        softmax_stats_tensor: Optional[torch.Tensor] = None,
        helix_position_offsets: Optional[torch.Tensor] = None,
        is_spec_decoding_enabled: bool = False,
        use_spec_decoding: bool = False,
        is_spec_dec_tree: bool = False,
        spec_decoding_position_offsets: Optional[torch.Tensor] = None,
        spec_decoding_packed_mask: Optional[torch.Tensor] = None,
        spec_decoding_generation_lengths: Optional[torch.Tensor] = None,
        attention_sinks: Optional[torch.Tensor] = None,
        chunked_prefill_buffer_batch_size: int = 1,
        sparse_kv_indices: Optional[torch.Tensor] = None,
        sparse_kv_offsets: Optional[torch.Tensor] = None,
        sparse_attn_indices: Optional[torch.Tensor] = None,
        sparse_attn_offsets: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Plan the attention operation.
        Call this method without arguments can reset the planned states.
        For required arguments, can use ellipsis (...) as default value to represent invalid states.
        Args:
            layer_idx (int): The index of the attention layer in the model.
            tokens_per_block (int): Token number per KV cache block.
            max_num_requests (int): Max request number per batch.
            max_sequence_length (int): Max sequence length.
            max_context_length (int): Max context length per context-phase sequence.
            attention_window_size (int): Max token number attended in windowed attention.
            sink_token_length (int): Sink token number in StreamingLLM.
            beam_width (int): Beam width in beam search.
            sequence_length (torch.Tensor): The length of each sequence with shape (batch_size) on GPU.
            host_past_key_value_lengths (torch.Tensor): Same as sequence_length, but on CPU.
            host_total_kv_lens (torch.Tensor): The tensor to store the total KV lens for context requests and generation requests, with shape (2) on CPU.
            context_lengths (torch.Tensor): The context-phase sequence length of each request with shape (batch_size) on GPU.
            host_context_lengths (torch.Tensor): Same as context_lengths, but on CPU.
            host_request_types (torch.Tensor): The tensor that indicates whether a request is in context or generation phase, with shape (batch_size) on CPU.
            kv_cache_block_offsets (torch.Tensor): The offsets to the blocks inside KV cache pools on GPU, its shape is (num_pools, max_batch_size * max_beam_width, 2, max_blocks_per_sequence), one for each block. If kv_cache_block_offsets, host_kv_cache_block_offsets, host_kv_cache_pool_pointers, host_kv_cache_pool_mapping are all None, the attention will be no cache attention.
            host_kv_cache_block_offsets (torch.Tensor): Same as kv_cache_block_offsets, but on CPU.
            host_kv_cache_pool_pointers (torch.Tensor): The pointers to the KV cache pools on CPU, its shape is (num_pools, 2), one for primary pool in GPU memory, one for secondary pool in CPU memory.
            host_kv_cache_pool_mapping (torch.Tensor): The index of the pool used by each attention layer on CPU, its shape is (num_local_attention_layers). The local attention layers mean all attention layers in the current PP stage in the pipeline parallelism case.
            workspace (torch.Tensor): An optional workspace tensor on GPU.
            cache_indirection (torch.Tensor): A tensor for beam search on GPU, its shape is (batch_size, beam_width, max_seqlen), for a sequence si, a beam bi and a token ti, the element cache_indirection[si][bi][ti] is an integer between 0 and beam_width-1 that indicates which path in the beam to read the K and V elements from in the KV cache.
            kv_scale_orig_quant (torch.Tensor): The tensor to store the scaling factor for quantization to INT8/FP8 in the KV cache, with shape (1) on GPU.
            kv_scale_quant_orig (torch.Tensor): The tensor to store the scaling factor for dequantization from INT8/FP8 in the KV cache, with shape (1) on GPU.
            out_scale (torch.Tensor): The tensor to store the scaling factor to quantize output, with shape (1) on GPU.
            out_scale_sf (torch.Tensor): The tensor to store the global scale for NVFP4 scaling factors, with shape (1) on GPU.
            kv_scales_sf (torch.Tensor): The tensor to store the global scale for KV NVFP4 scaling factors, with shape (2) on GPU.
            kv_scales_sf_inv (torch.Tensor): The tensor to store the inverse of the global scale for KV NVFP4 scaling factors, with shape (2) on GPU.
            use_paged_context_fmha (bool): Sets the mPagedContextFMHA attribute in the op runner.
            mrope_config (dict): The dictionary containing the mRope configuration.
            softmax_stats_tensor (torch.Tensor): The tensor to store the softmax statistics (max/sum)
            helix_position_offsets (torch.Tensor): The tensor to store the helix position offsets, with shape (num_tokens) on GPU.
            attention_sinks (torch.Tensor): The attention sinks (additional value in the denominator of the softmax) with shape of (num_heads_q) on GPU.
            chunked_prefill_buffer_batch_size (int): used for malloc buffer for k and v in fp8 context mla. the max input kv length is not max_num_tokens in this case. It is chunked_prefill_buffer_batch_size * max_num_tokens.
            sparse_kv_indices (torch.Tensor): The sparse indices for the KV cache, with shape of (num_heads_kv, num_sparse_tokens) on GPU.
            sparse_kv_offsets (torch.Tensor): The batch offsets for the sparse KV indices, with shape of (num_contexts + 1) on GPU.
            sparse_attn_indices (torch.Tensor): The sparse indices for the attention layer, with shape of (num_heads_kv, num_sparse_tokens) on GPU.
            sparse_attn_offsets (torch.Tensor): The batch offsets for the sparse attention indices, with shape of (num_generations + 1) on GPU.
        """
        self.layer_idx = layer_idx
        self.tokens_per_block = tokens_per_block
        self.max_num_requests = max_num_requests
        self.max_context_length = max_context_length
        self.attention_window_size = attention_window_size or max_sequence_length
        self.sink_token_length = sink_token_length
        self.beam_width = beam_width
        self.sequence_length = sequence_length
        self.host_past_key_value_lengths = host_past_key_value_lengths
        self.host_total_kv_lens = host_total_kv_lens
        self.context_lengths = context_lengths
        self.host_context_lengths = host_context_lengths
        self.host_request_types = host_request_types
        self.kv_cache_block_offsets = kv_cache_block_offsets
        self.host_kv_cache_block_offsets = host_kv_cache_block_offsets
        self.host_kv_cache_pool_pointers = host_kv_cache_pool_pointers
        self.host_kv_cache_pool_mapping = host_kv_cache_pool_mapping
        self.workspace = workspace
        self.cache_indirection = cache_indirection
        self.kv_scale_orig_quant = kv_scale_orig_quant if kv_scales_sf_inv is None else kv_scales_sf_inv
        self.kv_scale_quant_orig = kv_scale_quant_orig if kv_scales_sf is None else kv_scales_sf
        self.out_scale = out_scale
        self.out_scale_sf = out_scale_sf
        self.use_paged_context_fmha = use_paged_context_fmha
        self.use_nvfp4_output = use_nvfp4_output
        self.attention_input_type = int(attention_input_type)
        self.latent_cache = latent_cache
        self.q_pe = q_pe
        self.mrope_rotary_cos_sin = mrope_config.get(
            'mrope_rotary_cos_sin') if mrope_config is not None else None
        self.mrope_position_deltas = mrope_config.get(
            'mrope_position_deltas') if mrope_config is not None else None
        self.block_ids_per_seq = block_ids_per_seq
        self.softmax_stats_tensor = softmax_stats_tensor
        self.helix_position_offsets = helix_position_offsets
        self.attention_sinks = attention_sinks
        self.sparse_kv_indices = sparse_kv_indices
        self.sparse_kv_offsets = sparse_kv_offsets
        self.sparse_attn_indices = sparse_attn_indices
        self.sparse_attn_offsets = sparse_attn_offsets
        if max_sequence_length > self.rope_params.max_positions:
            self.rope_params.max_positions = max_sequence_length
            self.rotary_inv_freq, self.rotary_cos_sin = self.rope_params.create_rope_const_params(
            )
        self.is_spec_decoding_enabled = is_spec_decoding_enabled
        self.use_spec_decoding = use_spec_decoding
        self.is_spec_dec_tree = is_spec_dec_tree
        self.spec_decoding_position_offsets = spec_decoding_position_offsets
        self.spec_decoding_packed_mask = spec_decoding_packed_mask
        self.spec_decoding_generation_lengths = spec_decoding_generation_lengths
        self.chunked_prefill_buffer_batch_size = chunked_prefill_buffer_batch_size
        self.kwargs.update(kwargs)

    def create_output(self, q: torch.Tensor, out_dtype: torch.dtype):
        num_tokens = q.size(0)
        attention_input_type = (AttentionInputType(self.attention_input_type)
                                if self.attention_input_type is not None else
                                AttentionInputType.mixed)
        if out_dtype is None:
            out_dtype = q.dtype
        is_gen_only = attention_input_type == AttentionInputType.generation_only
        v_head_size = self.head_size
        if self.is_mla_enable:
            v_head_size = self.kv_lora_rank if is_gen_only else self.v_head_dim
        if out_dtype == torch.uint8:
            num_nvfp4_elements_per_container = 2
            scaling_vector_size = 16
            size_per_token = self.num_heads * v_head_size
            output = q.new_empty(
                (num_tokens,
                 size_per_token // num_nvfp4_elements_per_container),
                dtype=torch.uint8)
            # Create a sf (scaling factors) tensor for NVFP4 (use INT8 as the container dtype).
            output_sf = q.new_empty(compute_swizzled_sf_shape(
                num_tokens, size_per_token // scaling_vector_size),
                                    dtype=torch.uint8)
        else:
            output = q.new_empty((num_tokens, self.num_heads * v_head_size),
                                 dtype=out_dtype)
            output_sf = None
        return output, output_sf

    def run(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
        output_sf: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = None,
        is_fused_qkv: bool = True,
        update_kv_cache: bool = True,
        attention_mask: AttentionMask = PredefinedAttentionMask.CAUSAL,
    ):
        """
        Run the attention operation.
        Args:
            q (torch.Tensor): Query tensor with shape (num_tokens, num_heads * head_dim) or QKV tensor with shape (num_tokens, (num_heads + 2 * num_kv_heads) * head_dim).
            k (Optional[torch.Tensor]): Key tensor with shape (num_tokens, num_kv_heads * head_dim) or None if QKV tensor is provided.
            v (Optional[torch.Tensor]): Value tensor with shape (num_tokens, num_kv_heads * head_dim) or None if QKV tensor is provided.
            out_dtype (Optional[torch.dtype]): Output data type if provided.
            is_fused_qkv (bool): Whether QKV tensor is provided.
            update_kv_cache (bool): Whether KV cache is updated.
            attention_mask (AttentionMask): Attention mask. See definition of AttentionMask for accepted types. Defaults to predefined causal mask.
        Returns:
            torch.Tensor with shape (num_tokens, num_heads * head_dim).
        """
        if len(self.kwargs) > 0:
            logger.warning(
                f"unknown arguments {list(self.kwargs.keys())} in attention wrapper"
            )
        assert (is_fused_qkv and k is None
                and v is None) or (not is_fused_qkv and k is not None
                                   and v is not None)

        if not self.is_mla_enable:
            if is_fused_qkv:
                qkv_hidden_size = (self.num_heads +
                                   2 * self.num_kv_heads) * self.head_size
                assert q.shape[1] == qkv_hidden_size
            else:
                q_hidden_size = self.num_heads * self.head_size
                assert q.shape[1] == q_hidden_size
                if update_kv_cache:
                    kv_hidden_size = self.num_kv_heads * self.head_size
                    assert k.shape[1] == kv_hidden_size
                    assert v.shape[1] == kv_hidden_size
            num_tokens = q.shape[0]
            if k is not None:
                assert k.shape[0] == num_tokens
                assert v.shape[0] == num_tokens
            batch_size = self.sequence_length.shape[0]
            assert self.host_past_key_value_lengths.shape[0] == batch_size
            assert self.context_lengths.shape[0] == batch_size
            assert self.host_context_lengths.shape[0] == batch_size
            assert self.host_request_types.shape[0] == batch_size

            if attention_mask == PredefinedAttentionMask.CAUSAL:
                mask_type = AttentionMaskType.causal
            elif attention_mask == PredefinedAttentionMask.FULL:
                mask_type = AttentionMaskType.padding
            else:
                raise ValueError("Unexpected attention mask type")
        else:
            if self.attention_input_type == AttentionInputType.context_only:
                assert not is_fused_qkv
                qkv_hidden_size = self.num_heads * (self.qk_nope_head_dim +
                                                    self.qk_rope_head_dim)
            elif self.attention_input_type == AttentionInputType.generation_only:
                assert is_fused_qkv
                qkv_hidden_size = self.num_heads * (self.kv_lora_rank +
                                                    self.qk_rope_head_dim)
            else:
                raise ValueError(
                    "In MLA, TrtllmAttention can only support context_only or generation_only, not mixed."
                )
            assert q.shape[
                1] == qkv_hidden_size, f"q.shape[1] must be equal to qkv_hidden_size, got {q.shape[1]=}, {qkv_hidden_size=}"

            batch_size = self.sequence_length.shape[0]
            assert self.host_past_key_value_lengths.shape[0] == batch_size
            assert self.context_lengths.shape[0] == batch_size
            assert self.host_context_lengths.shape[0] == batch_size
            assert self.host_request_types.shape[0] == batch_size

            if attention_mask == PredefinedAttentionMask.CAUSAL:
                mask_type = AttentionMaskType.causal
            elif attention_mask == PredefinedAttentionMask.FULL:
                mask_type = AttentionMaskType.padding
            else:
                raise ValueError("Unexpected attention mask type")

        if output is None:
            assert output_sf is None
            output, output_sf = self.create_output(q, out_dtype)
        else:
            # output is provided, expect output_sf be provided as well if has NVFP4 output.
            assert out_dtype is None or out_dtype != torch.uint8 or output_sf is not None

        # packing parameters to avoid maxing out 64 arguments
        rotary_embedding_scales = [
            self.rotary_embedding_scale, self.rotary_embedding_short_m_scale,
            self.rotary_embedding_long_m_scale
        ]
        rotary_embedding_max_position_info = [
            self.rotary_embedding_max_positions,
            self.rotary_embedding_original_max_positions
        ]
        spec_decoding_bool_params = [
            self.is_spec_decoding_enabled, self.use_spec_decoding,
            self.is_spec_dec_tree
        ]
        spec_decoding_tensor_params = [
            self.spec_decoding_generation_lengths,
            self.spec_decoding_position_offsets, self.spec_decoding_packed_mask
        ]
        mla_tensor_params = [self.helix_position_offsets]
        sparse_attention_params = [
            self.sparse_kv_indices,
            self.sparse_kv_offsets,
            self.sparse_attn_indices,
            self.sparse_attn_offsets,
        ]

        thop.attention(
            q,
            k,
            v,
            output,
            output_sf,
            out_dtype,
            self.workspace,
            self.sequence_length,
            self.host_past_key_value_lengths,
            self.host_total_kv_lens,
            self.context_lengths,
            self.host_context_lengths,
            self.host_request_types,
            self.kv_cache_block_offsets,
            self.host_kv_cache_block_offsets,
            self.host_kv_cache_pool_pointers,
            self.host_kv_cache_pool_mapping,
            self.cache_indirection,
            self.kv_scale_orig_quant,
            self.kv_scale_quant_orig,
            self.out_scale_sf if self.use_nvfp4_output else self.out_scale,
            self.rotary_inv_freq,
            self.rotary_cos_sin,
            self.latent_cache,
            self.q_pe,
            self.block_ids_per_seq,
            self.attention_sinks,
            is_fused_qkv,
            update_kv_cache,
            self.predicted_tokens_per_seq,
            self.layer_idx,
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            self.tokens_per_block,
            self.max_num_requests,
            self.max_context_length,
            self.attention_window_size,
            self.sink_token_length,
            self.beam_width,
            int(mask_type),
            self.quant_mode,
            self.q_scaling,
            self.position_embedding_type,
            self.rotary_embedding_dim,
            self.rotary_embedding_base,
            self.rotary_embedding_scale_type,
            rotary_embedding_scales,
            rotary_embedding_max_position_info,
            self.use_paged_context_fmha,
            self.attention_input_type,
            self.is_mla_enable,
            self.chunked_prefill_buffer_batch_size,
            self.q_lora_rank,
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.mrope_rotary_cos_sin,
            self.mrope_position_deltas,
            mla_tensor_params,
            self.attention_chunk_size,
            self.softmax_stats_tensor,
            spec_decoding_bool_params,
            spec_decoding_tensor_params,
            sparse_attention_params,
        )
        # reset the planned states (especially tensors) to avoid memory leak
        self.plan()
        return output, output_sf

    def is_nvfp4_output_kernel_available(
        self,
        *,
        tokens_per_block: Optional[int] = None,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.
        CAUSAL,
        use_paged_context_fmha: bool = False,
        is_mla_enable: bool = False,
        **kwargs,
    ):
        """
        Runtime check whether the NVFP4 output kernel is available.
        Args:
            tokens_per_block (int): Token number per KV cache block.
            attention_mask (PredefinedAttentionMask): The attention mask type.
            use_paged_context_fmha (bool): Whether to use paged context FMHA.
            is_mla_enable (bool): Whether to use MLA.
        """
        if attention_mask == PredefinedAttentionMask.CAUSAL:
            mask_type = AttentionMaskType.causal
        elif attention_mask == PredefinedAttentionMask.FULL:
            mask_type = AttentionMaskType.padding
        else:
            raise ValueError("Unexpected attention mask type")

        return torch.ops.trtllm.attention_supports_nvfp4_output(
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            tokens_per_block,
            int(mask_type),
            self.quant_mode,
            use_paged_context_fmha,
            is_mla_enable,
        )


@dataclass(kw_only=True)
class TrtllmAttentionMetadata(AttentionMetadata):
    workspace: Optional[torch.Tensor] = None

    # TrtllmAttention needs to know the beam width to access to the cache indirection buffer,
    # when beam search is enabled.
    beam_width: int = 1

    # TrtllmAttention needs to know the max sequence length.
    # Implemented as a property to support no cache mode.
    max_seq_len: Optional[int]

    # Storage for internal max_seq_len value
    _max_seq_len_storage: Optional[int] = field(default=None,
                                                init=True,
                                                repr=False)

    # Flags to enable spec-dec mode (multi-query mode) in TRTLLM XQA Kernels
    # spec decoding mode can be enabled for non-TRTLLM-gen kernels (pre-Blackwell XQA kernels)
    # is_spec_decoding_enabled specifies if spec-dec mode is supported for the entire runtime.
    is_spec_decoding_enabled: bool = False
    # use_spec_decoding determines if the attention layer should be run in spec-dec mode at the specific step / layer.
    use_spec_decoding: bool = False

    # if spec-dec tree is a tree or a chain (linear tree)
    is_spec_dec_tree: bool = False
    # if spec-dec tree wouldn't be changed at all, the mask won't be computed every step.
    is_spec_dec_dynamic_tree: bool = False

    # parameters required for spec-dec mode
    spec_decoding_position_offsets: Optional[torch.Tensor] = None
    spec_decoding_packed_mask: Optional[torch.Tensor] = None
    spec_decoding_generation_lengths: Optional[torch.Tensor] = None

    @property
    def max_seq_len(self) -> int:
        """
        Returns the max sequence length.
        If the attention uses KV cache, it will return max_seq_len from the KV cache manager.
        If the attention is no cache, max_seq_len should be set manually by user.
        """
        if self.kv_cache_manager is not None:
            return self.kv_cache_manager.max_seq_len
        else:
            assert self._max_seq_len_storage is not None, "max_seq_len should be set for no kv cache attention"
            return self._max_seq_len_storage

    @max_seq_len.setter
    def max_seq_len(self, value: int) -> None:
        """
        Set the max sequence length for no cache attention.
        """
        self._max_seq_len_storage = value

    @property
    def tokens_per_block(self) -> Optional[int]:
        """
        Returns the number of tokens per block from the KV cache manager.
        """
        return self.kv_cache_manager.tokens_per_block if self.kv_cache_manager is not None else None

    @property
    def host_kv_cache_pool_pointers(self) -> Optional[torch.Tensor]:
        """
        Returns the host KV cache pool pointers from the KV cache manager if KV cache manager is not None.
        """
        return self.kv_cache_manager.kv_cache_pool_pointers if self.kv_cache_manager is not None else None

    @property
    def host_kv_cache_pool_mapping(self) -> Optional[torch.Tensor]:
        """
        Returns the host KV cache pool mapping from the KV cache manager if KV cache manager is not None.
        """
        return self.kv_cache_manager.kv_cache_pool_mapping if self.kv_cache_manager is not None else None

    def __post_init__(self) -> None:
        super().__post_init__()
        self._post_init_with_buffers(self.cuda_graph_buffers)

    def _post_init_with_buffers(self, buffers) -> None:

        # Set a default value, as max_num_sequences is not always set.
        if self.max_num_sequences is None:
            self.max_num_sequences = self.max_num_requests

        capture_graph = torch.cuda.is_current_stream_capturing()

        def get_empty(tensor_shape: list[int], dtype: torch.dtype,
                      cache_name: str) -> torch.Tensor:
            """
            Finds a compatible, reusable buffer from a cache or creates a new one.

            This function searches for a pre-allocated tensor (buffer) that can be
            reused for an operation involving a tensor with the shape of `tensor_shape`.

            The compatibility rules are: The buffer's total elements must be >= tensor_shape's.

            If a compatible buffer is found, it's returned immediately. Otherwise, a new
            buffer is allocated on the 'cuda' device with the give properties of 'tensor_shape' and 'dtype'.

            Args:
                tensor_shape: The required shape.
                dtype: The required dtype.
                cache_name: The key for the specific list of buffers to search in.
            Returns:
                An existing compatible buffer or a newly created one.
            """
            if buffers is None:
                return torch.zeros(tensor_shape, device='cuda', dtype=dtype)

            return buffers.get_buffer(tensor_shape, dtype, cache_name,
                                      capture_graph)

        def get_empty_like(like_tensor: torch.Tensor,
                           cache_name: str) -> torch.Tensor:
            return get_empty(like_tensor.shape,
                             cache_name=cache_name,
                             dtype=like_tensor.dtype)

        self.prompt_lens_cuda = get_empty(
            (self.max_num_sequences, ),
            cache_name="prompt_lens_cuda",
            dtype=torch.int,
        )
        self.prompt_lens_cpu = torch.empty_like(
            self.prompt_lens_cuda,
            device='cpu',
            pin_memory=True,
        )
        self.kv_lens_cuda = get_empty_like(self.prompt_lens_cuda,
                                           cache_name="kv_lens_cuda")
        self.kv_lens = torch.empty_like(self.kv_lens_cuda,
                                        device='cpu',
                                        pin_memory=True)
        self.host_total_kv_lens = torch.empty(2, device='cpu', dtype=torch.int)
        self.host_request_types = torch.empty_like(self.prompt_lens_cpu)

        # For debugging, can use it to call the wrapper's plan function
        if self.workspace is None:
            self.workspace = torch.empty(
                (0, ),
                device='cuda',
                dtype=torch.int8,
            )
        if self.kv_cache_manager is not None:
            self.kv_cache_block_offsets = get_empty(
                [
                    self.kv_cache_manager.num_pools, self.max_num_sequences, 2,
                    self.kv_cache_manager.max_blocks_per_seq
                ],
                cache_name="kv_cache_block_offsets",
                dtype=torch.int32,
            )
            self.host_kv_cache_block_offsets = torch.empty_like(
                self.kv_cache_block_offsets,
                device='cpu',
                pin_memory=True,
            )
            self.block_ids_per_seq = None
            self.kv_block_ids_per_seq = None
            if self.enable_flash_mla:
                self.block_ids_per_seq = get_empty(
                    [
                        self.kv_cache_manager.max_batch_size,
                        self.kv_cache_manager.max_blocks_per_seq
                    ],
                    cache_name="block_ids_per_seq",
                    dtype=torch.int32,
                )
                self.kv_block_ids_per_seq = get_empty(
                    [
                        self.kv_cache_manager.max_batch_size,
                        self.kv_cache_manager.max_blocks_per_seq
                    ],
                    cache_name="kv_block_ids_per_seq",
                    dtype=torch.int32,
                )
            if self.enable_context_mla_with_cached_kv:
                # for kv cache reuse/chunked context in MLA
                self.ctx_cached_token_indptr = get_empty(
                    (self.max_num_requests + 1, ),
                    cache_name="ctx_cached_token_indptr",
                    dtype=torch.int64,
                )
                self.host_ctx_cached_token_indptr = torch.zeros_like(
                    self.ctx_cached_token_indptr,
                    device='cpu',
                    pin_memory=True,
                )
                self.ctx_uncached_token_indptr = get_empty(
                    (self.max_num_requests + 1, ),
                    cache_name="ctx_uncached_token_indptr",
                    dtype=torch.int64,
                )
                self.host_ctx_uncached_token_indptr = torch.zeros_like(
                    self.ctx_uncached_token_indptr,
                    device='cpu',
                    pin_memory=True,
                )
                # context full seqlens include cached tokens and uncached tokens
                self.ctx_kv_indptr = get_empty(
                    (self.max_num_requests + 1, ),
                    cache_name="ctx_kv_indptr",
                    dtype=torch.int64,
                )
                self.host_ctx_kv_indptr = torch.zeros_like(
                    self.ctx_kv_indptr,
                    device='cpu',
                    pin_memory=True,
                )

    def on_update_kv_lens(self):
        # After changing the kv_lens/kv_lens_cuda, we may need to update other metadata.
        # Especially for the changes in the _preprocess_inputs() of model_engine.py.
        pass

    def prepare(self) -> None:
        extra_attrs = get_model_extra_attrs()
        # If model extra attrs is set, attention_metadata is setup in executor.
        if extra_attrs is None:
            get_global_attrs().attention_metadata = weakref.ref(self)
        if self.kv_cache_manager is None:
            # Convert the attention metadata to a TRT-LLM no cache attention metadata.
            assert self.kv_cache_manager is None, "no cache attention should not have KV cache manager"
            assert self._max_seq_len_storage is not None, "max_seq_len should be set for no cache attention"

            # setting kv cache params
            self.kv_cache_params = KVCacheParams(use_cache=False, )

            # trtllm attn metadata prepare() requires this
            self.prompt_lens = self.context_lens

            # set params that are used in wrapper.plan()
            self.kv_cache_block_offsets = None
            self.host_kv_cache_block_offsets = None
            self.block_ids_per_seq = None

        prompt_lens = torch.tensor(
            self.prompt_lens,
            dtype=torch.int,
            device='cpu',
        )
        self.prompt_lens_cpu[:self.num_seqs].copy_(prompt_lens)
        self.prompt_lens_cuda[:self.num_seqs].copy_(
            self.prompt_lens_cpu[:self.num_seqs], non_blocking=True)

        # number of tokens in the kv cache for each sequence in the batch
        cached_token_lens = torch.tensor(
            self.kv_cache_params.num_cached_tokens_per_seq,
            dtype=torch.int,
            device='cpu',
        ) if self.kv_cache_params.use_cache else None

        if self.enable_flash_mla:
            self.prepare_flash_mla()
        # number of tokens needed in the kv cache for each sequence after the next pass
        kv_lens = cached_token_lens + self.seq_lens_kv if cached_token_lens is not None else self.seq_lens_kv
        # self.kv_lens is the valid kv cache length, while the self.kv_lens_cuda is
        # the sequence length including the cached tokens and the input tokens.
        self.kv_lens[:self.num_seqs].copy_(
            kv_lens + self.kv_cache_params.num_extra_kv_tokens)
        self.kv_lens_cuda[:self.num_seqs].copy_(
            kv_lens[:self.num_seqs].pin_memory(), non_blocking=True)
        # total kv lens for context requests and generation requests, without extra tokens
        self.host_total_kv_lens[0] = kv_lens[:self.num_contexts].sum().item()
        self.host_total_kv_lens[1] = kv_lens[self.num_contexts:self.
                                             num_seqs].sum().item()
        self.host_request_types[:self.num_contexts].fill_(0)
        self.host_request_types[self.num_contexts:self.num_seqs].fill_(1)

        # prepare for kv cache reuse/chunked context in MLA
        if self.enable_context_mla_with_cached_kv:
            self.prepare_context_mla_with_cached_kv(cached_token_lens, kv_lens)

        # kv block offsets
        assert self.request_ids is not None
        if self.kv_cache_manager is not None:
            # Copy blocks for all context requests
            self.kv_cache_manager.impl.copy_batch_block_offsets(
                self.host_kv_cache_block_offsets,
                self.request_ids[:self.num_contexts], 1, 0)
            # Copy blocks for all generation requests
            self.kv_cache_manager.impl.copy_batch_block_offsets(
                self.host_kv_cache_block_offsets,
                self.request_ids[self.num_contexts:], self.beam_width,
                self.num_contexts)
            self.kv_cache_block_offsets[:, :self.num_seqs].copy_(
                self.host_kv_cache_block_offsets[:, :self.num_seqs],
                non_blocking=True)

            error_message = (
                f"The max KV cache length of input sequences ({self.kv_lens[:self.num_seqs].max()}) "
                f"exceeds the KV cache manager's maximum supported length "
                f"({self.kv_cache_manager.max_seq_len}).")

            assert self.kv_lens[:self.num_seqs].max(
            ) <= self.kv_cache_manager.max_seq_len, error_message

        self.kv_lens_cuda_runtime = self.kv_lens_cuda[:self.num_seqs]
        self.kv_lens_runtime = self.kv_lens[:self.num_seqs]
        self.prompt_lens_cuda_runtime = self.prompt_lens_cuda[:self.num_seqs]
        self.prompt_lens_cpu_runtime = self.prompt_lens_cpu[:self.num_seqs]
        self.host_request_types_runtime = self.host_request_types[:self.
                                                                  num_seqs]

    def prepare_flash_mla(self) -> None:
        block_ids_per_seq = self.kv_cache_manager.get_block_ids_per_seq(
            self.request_ids).pin_memory()
        num_blocks = block_ids_per_seq.shape[1]
        self.kv_block_ids_per_seq.fill_(0)
        self.kv_block_ids_per_seq[:self.num_seqs, :num_blocks].copy_(
            block_ids_per_seq, non_blocking=True)
        self.block_ids_per_seq.fill_(0)
        self.block_ids_per_seq[:self.num_generations, :num_blocks].copy_(
            block_ids_per_seq[self.num_contexts:], non_blocking=True)

        self.kv_lens_cuda_runtime = self.kv_lens_cuda[:self.num_seqs]
        self.kv_lens_runtime = self.kv_lens[:self.num_seqs]
        self.prompt_lens_cuda_runtime = self.prompt_lens_cuda[:self.num_seqs]
        self.prompt_lens_cpu_runtime = self.prompt_lens_cpu[:self.num_seqs]
        self.host_request_types_runtime = self.host_request_types[:self.
                                                                  num_seqs]

    def pre_process_for_chunked_prefill(
        self,
        chunked_seq_len: torch.Tensor,
        chunked_global_offset: torch.
        Tensor,  # [chunked_loop_num + 1, num_contexts]
        cu_chunked_seq_len: torch.Tensor,
        merge_op_tensor: torch.Tensor,
        max_chunk_len_per_loop: list[int],
        chunked_loop_num: int,
    ) -> None:
        """
        Pre-process the MLA layer for chunked prefill.
        This method is called before the forward pass to prepare the MLA layer for chunked prefill.
        """
        num_contexts = self.num_contexts
        chunk_size = self.runtime_features.chunk_size
        chunk_batch_size = self.runtime_features.chunked_prefill_buffer_batch_size
        total_chunk_size = chunk_size * chunk_batch_size
        remain_buffer_len = total_chunk_size
        current_batch_idx = 0
        max_chunk_len_per_loop.clear()
        max_chunk_len = 0
        # cal chunked_seq_len
        for batch_idx in range(num_contexts):
            cached_kv_len = self.kv_cache_params.num_cached_tokens_per_seq[
                batch_idx]
            while cached_kv_len > 0:
                used_buffer_len = min(remain_buffer_len, cached_kv_len)
                chunked_seq_len[current_batch_idx, batch_idx] = used_buffer_len
                max_chunk_len = max(max_chunk_len, used_buffer_len)
                remain_buffer_len -= used_buffer_len
                cached_kv_len -= used_buffer_len
                chunked_global_offset[
                    current_batch_idx + 1, batch_idx] = chunked_global_offset[
                        current_batch_idx,
                        batch_idx] + chunked_seq_len[current_batch_idx,
                                                     batch_idx]
                if remain_buffer_len == 0:
                    current_batch_idx += 1
                    remain_buffer_len = total_chunk_size
                    max_chunk_len_per_loop.append(max_chunk_len)
                    max_chunk_len = 0
        if len(max_chunk_len_per_loop) < chunked_loop_num:
            max_chunk_len_per_loop.append(max_chunk_len)
        assert len(
            max_chunk_len_per_loop
        ) == chunked_loop_num, f"max_chunk_len_per_loop size {len(max_chunk_len_per_loop)} != chunked_loop_num {chunked_loop_num}"
        for loop_idx in range(chunked_loop_num):
            cu_chunked_seq_len[loop_idx, 0] = 0
            torch.cumsum(chunked_seq_len[loop_idx, :num_contexts],
                         dim=0,
                         dtype=torch.int64,
                         out=cu_chunked_seq_len[loop_idx, 1:num_contexts + 1])
            for s in range(num_contexts):
                if chunked_seq_len[loop_idx, s] > 0 and (
                        loop_idx == 0 or chunked_seq_len[loop_idx - 1, s] == 0):
                    merge_op_tensor[loop_idx, s] = 2  # copy only
                elif chunked_seq_len[loop_idx, s] > 0:
                    merge_op_tensor[loop_idx, s] = 1  # merge
                else:
                    merge_op_tensor[loop_idx, s] = 0  # skip

        # set merge op for last attn
        for s in range(num_contexts):
            if self.kv_cache_params.num_cached_tokens_per_seq[s] == 0:
                merge_op_tensor[chunked_loop_num, s] = 2  # copy only
            else:
                merge_op_tensor[chunked_loop_num, s] = 1  # merge

    def prepare_context_mla_with_cached_kv(self,
                                           cached_token_lens: torch.Tensor,
                                           kv_lens: torch.Tensor) -> None:
        if self.num_contexts > 0:
            self.num_ctx_cached_tokens = cached_token_lens[:self.
                                                           num_contexts].sum(
                                                           ).item()
            self.max_ctx_cached_token_len = cached_token_lens[:self.
                                                              num_contexts].max(
                                                              ).item()
            self.max_ctx_kv_len = kv_lens[:self.num_contexts].max().item()
            self.max_ctx_seq_len = self.seq_lens[:self.num_contexts].max().item(
            )
            # determine the number of loop
            # currently we assume that the chunk size is the same as the max_num_tokens
            if self.runtime_features.chunked_prefill:
                chunk_size = self.runtime_features.chunk_size
                chunk_batch_size = self.runtime_features.chunked_prefill_buffer_batch_size
                total_chunk_size = chunk_size * chunk_batch_size
                self.chunked_loop_num = math.ceil(self.num_ctx_cached_tokens /
                                                  total_chunk_size)
                self.chunked_seq_len = torch.zeros(
                    (self.chunked_loop_num, self.num_seqs),
                    dtype=torch.int,
                    device='cuda',
                )
                self.host_chunked_seq_len = torch.zeros_like(
                    self.chunked_seq_len,
                    device='cpu',
                    pin_memory=True,
                )
                self.cu_chunked_seq_len = torch.zeros(
                    (self.chunked_loop_num, self.num_contexts + 1),
                    dtype=torch.int64,
                    device='cuda',
                )
                self.host_cu_chunked_seq_len = torch.zeros_like(
                    self.cu_chunked_seq_len,
                    device='cpu',
                    pin_memory=True,
                )
                self.chunked_global_offset = torch.zeros(
                    (self.chunked_loop_num + 1, self.num_contexts),
                    dtype=torch.int64,
                    device='cuda',
                )
                self.host_chunked_global_offset = torch.zeros_like(
                    self.chunked_global_offset,
                    device='cpu',
                    pin_memory=True,
                )
                self.max_chunk_len_per_loop = []
                # For last chunk we use the uncached kv
                self.merge_op_tensor = torch.empty(
                    (self.chunked_loop_num + 1, self.num_contexts),
                    dtype=torch.int64,
                    device='cuda',
                )
                self.host_merge_op_tensor = torch.empty_like(
                    self.merge_op_tensor,
                    device='cpu',
                    pin_memory=True,
                )

                self.pre_process_for_chunked_prefill(
                    chunked_seq_len=self.host_chunked_seq_len,
                    chunked_global_offset=self.host_chunked_global_offset,
                    cu_chunked_seq_len=self.host_cu_chunked_seq_len,
                    merge_op_tensor=self.host_merge_op_tensor,
                    max_chunk_len_per_loop=self.max_chunk_len_per_loop,
                    chunked_loop_num=self.chunked_loop_num)
                self.chunked_seq_len.copy_(self.host_chunked_seq_len,
                                           non_blocking=True)
                self.cu_chunked_seq_len.copy_(self.host_cu_chunked_seq_len,
                                              non_blocking=True)
                self.merge_op_tensor.copy_(self.host_merge_op_tensor,
                                           non_blocking=True)
                self.chunked_global_offset.copy_(
                    self.host_chunked_global_offset, non_blocking=True)
        else:
            self.num_ctx_cached_tokens = 0
            self.max_ctx_cached_token_len = 0
            self.max_ctx_kv_len = 0
            self.max_ctx_seq_len = 0
        torch.cumsum(cached_token_lens[:self.num_contexts],
                     dim=0,
                     dtype=torch.int64,
                     out=self.host_ctx_cached_token_indptr[1:self.num_contexts +
                                                           1])
        self.ctx_cached_token_indptr[:self.num_contexts + 1].copy_(
            self.host_ctx_cached_token_indptr[:self.num_contexts + 1],
            non_blocking=True)
        torch.cumsum(
            self.seq_lens[:self.num_contexts],
            dim=0,
            dtype=torch.int64,
            out=self.host_ctx_uncached_token_indptr[1:self.num_contexts + 1])
        self.ctx_uncached_token_indptr[:self.num_contexts + 1].copy_(
            self.host_ctx_uncached_token_indptr[:self.num_contexts + 1],
            non_blocking=True)
        torch.cumsum(kv_lens[:self.num_contexts],
                     dim=0,
                     dtype=torch.int64,
                     out=self.host_ctx_kv_indptr[1:self.num_contexts + 1])
        self.ctx_kv_indptr[:self.num_contexts + 1].copy_(
            self.host_ctx_kv_indptr[:self.num_contexts + 1], non_blocking=True)

    def update_spec_dec_param(
        self,
        is_spec_decoding_enabled,
        is_spec_dec_tree,
        is_spec_dec_dynamic_tree,
        max_draft_tokens,
        spec_decoding_tensor: Optional['SpecDecodingTensor'] = None,
    ):

        if spec_decoding_tensor is not None:
            spec_decoding_position_offsets = spec_decoding_tensor.position_offsets
            spec_decoding_packed_mask = spec_decoding_tensor.packed_mask
            spec_decoding_generation_lengths = spec_decoding_tensor.generation_lengths
        else:
            spec_decoding_position_offsets = None
            spec_decoding_packed_mask = None
            spec_decoding_generation_lengths = None
        # spec_dec mode should only be enabled for pre-Blackwell machines and when there's a spec-dec tree.
        self.is_spec_decoding_enabled = is_spec_decoding_enabled and get_sm_version(
        ) < 100

        if get_sm_version() >= 100:
            if is_spec_dec_tree or is_spec_dec_dynamic_tree:
                assert not is_spec_dec_tree, "Spec-dec tree is not supported on this machine. Please use a pre-Blackwell machine for a spec-dec tree."
                assert not is_spec_dec_dynamic_tree, "Spec-dec dynamic tree is not supported on this machine. Please use a pre-Blackwell machine for a spec-dec dynamic tree."

        # use_spec_decoding is default to true by default, change in runtime by layers / requests
        self.use_spec_decoding = self.is_spec_decoding_enabled

        self.is_spec_dec_tree = is_spec_dec_tree
        self.is_spec_dec_dynamic_tree = is_spec_dec_dynamic_tree

        # Parameters can be fixed and not changed during runtime if the
        if self.is_spec_decoding_enabled:
            self.spec_decoding_position_offsets = torch.empty(
                [self.max_num_requests, max_draft_tokens + 1],
                dtype=torch.int,
                device='cuda',
            )

            self.spec_decoding_packed_mask = torch.empty(
                [
                    self.max_num_requests, max_draft_tokens + 1,
                    math.ceil((max_draft_tokens + 1) / 32)
                ],
                dtype=torch.int,
                device='cuda',
            )

            self.spec_decoding_generation_lengths = torch.empty(
                [self.max_num_requests],
                dtype=torch.int,
                device='cuda',
            )

            if self.is_spec_dec_dynamic_tree:
                assert spec_decoding_position_offsets is not None, "spec_decoding_position_offsets is required for dynamic tree"
                assert spec_decoding_packed_mask is not None, "spec_decoding_packed_mask is required for dynamic tree"
                self.spec_decoding_position_offsets.copy_(
                    spec_decoding_position_offsets, non_blocking=True)
                self.spec_decoding_packed_mask.copy_(spec_decoding_packed_mask,
                                                     non_blocking=True)
                if spec_decoding_generation_lengths is not None:
                    self.spec_decoding_generation_lengths.copy_(
                        spec_decoding_generation_lengths, non_blocking=True)
                else:
                    self.generate_spec_decoding_generation_length(
                        max_draft_tokens=max_draft_tokens)
            else:
                # Populate the mask that won't change during inference phase.
                self.generate_spec_decoding_position_offsets(
                    max_draft_tokens=max_draft_tokens)
                self.generate_spec_decoding_packed_mask(
                    max_draft_tokens=max_draft_tokens)
                self.generate_spec_decoding_generation_length(
                    max_draft_tokens=max_draft_tokens)

    def generate_spec_decoding_position_offsets(self, max_draft_tokens):
        position_offset = torch.arange(max_draft_tokens + 1,
                                       dtype=torch.int,
                                       device='cpu',
                                       pin_memory=True)

        # fill all the batches with same position offset
        self.spec_decoding_position_offsets.copy_(position_offset,
                                                  non_blocking=True)

    def generate_spec_decoding_packed_mask(self, max_draft_tokens):
        dummy_idx = torch.arange(max_draft_tokens + 1)
        spec_decoding_packed_mask = torch.pow(2, dummy_idx + 1) - 1
        self.spec_decoding_packed_mask[:, :, 0].copy_(spec_decoding_packed_mask,
                                                      non_blocking=True)

    def generate_spec_decoding_generation_length(self, max_draft_tokens):
        spec_decoding_generation_length = torch.full((self.max_num_requests, ),
                                                     max_draft_tokens + 1)
        self.spec_decoding_generation_lengths[:self.max_num_requests].copy_(
            spec_decoding_generation_length, non_blocking=True)


class TrtllmAttention(AttentionBackend[TrtllmAttentionMetadata]):

    Metadata = TrtllmAttentionMetadata

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
        **kwargs,
    ):
        """
        Initialize the backend.
        Args:
            layer_idx (int): The index of the attention layer in the model.
            num_heads (int): The number of query heads.
            head_dim (int): The size of each attention head (hidden_size // num_heads).
            num_kv_heads (int): The number of kv heads. Defaults to num_heads if None.
            quant_config (QuantConfig): Optional quantization configuration. If None, no quantization is applied.
            q_scaling (float): Scaling factor for QK. Defaults to 1.0 if None.
            pos_embd_params (PositionalEmbeddingParams): Optional parameters defining how positional embedding should be applied.
                                                         If None, positional embedding should be applied by the model before calling the backend.
                                                         Otherwise, the backend is in-charge of applying positional embedding and may cache K without embedding it first.
            mla_params (MLAParams): Optional parameters for MLA. If None, MLA is not enabled.
        """
        super().__init__(layer_idx,
                         num_heads,
                         head_dim,
                         num_kv_heads,
                         quant_config,
                         q_scaling=q_scaling,
                         pos_embd_params=pos_embd_params,
                         mla_params=mla_params,
                         **kwargs)

        self.wrapper = TrtllmAttentionWrapper(
            num_heads,
            head_dim,
            num_kv_heads,
            pos_embd_params=pos_embd_params,
            q_scaling=q_scaling,
            mla_params=mla_params,
            attention_chunk_size=attention_chunk_size,
        )

        self.is_mla_enable = mla_params is not None
        self.mla_params = mla_params or MLAParams()
        self.v_head_dim = self.mla_params.v_head_dim if self.is_mla_enable else head_dim
        self.kv_cache_scaling_factor = torch.ones(1,
                                                  dtype=torch.float32,
                                                  device='cuda')
        self.kv_scale_quant_orig = self.kv_cache_scaling_factor
        self.kv_scale_orig_quant = 1.0 / self.kv_cache_scaling_factor
        if not skip_create_weights_in_init:
            self.update_quant_config(self.quant_config)

    def update_quant_config(self, new_quant_config: Optional[QuantConfig]):
        self.quant_config = new_quant_config
        self.wrapper.update_quant_config(self.quant_config)

        self.has_fp8_qdq = self.has_fp8_kv_cache = self.has_nvfp4 = False
        if self.quant_config is not None:
            self.has_fp8_kv_cache = self.quant_config.layer_quant_mode.has_fp8_kv_cache(
            )
            self.has_fp4_kv_cache = self.quant_config.layer_quant_mode.has_fp4_kv_cache(
            )

            self.has_fp8_qdq = self.quant_config.layer_quant_mode.has_fp8_qdq()
            self.has_fp8_block_wise = self.quant_config.layer_quant_mode.has_fp8_block_scales(
            )
            self.has_fp8_rowwise = self.quant_config.layer_quant_mode.has_fp8_rowwise(
            )
            self.has_nvfp4 = self.quant_config.layer_quant_mode.has_nvfp4()
            self.has_w4a8_nvfp4_fp8 = self.quant_config.layer_quant_mode.has_w4a8_nvfp4_fp8(
            )

    def get_local_layer_idx(self, metadata: TrtllmAttentionMetadata) -> int:
        if metadata.kv_cache_manager is None:
            return self.layer_idx
        else:
            return metadata.kv_cache_manager.layer_offsets[self.layer_idx]

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: TrtllmAttentionMetadata,
        out_scale: Optional[torch.Tensor] = None,
        out_scale_sf: Optional[torch.Tensor] = None,
        kv_scales_sf: Optional[torch.Tensor] = None,
        kv_scales_sf_inv: Optional[torch.Tensor] = None,
        *,
        attention_mask: AttentionMask = PredefinedAttentionMask.CAUSAL,
        attention_input_type: AttentionInputType = AttentionInputType.mixed,
        latent_cache: Optional[torch.Tensor] = None,
        q_pe: Optional[torch.Tensor] = None,
        mrope_config: Optional[dict] = None,
        attention_window_size: Optional[int] = None,
        softmax_stats_tensor: Optional[torch.Tensor] = None,
        helix_position_offsets: Optional[torch.Tensor] = None,
        enable_attn_nvfp4_output: bool = True,
        output: Optional[torch.Tensor] = None,
        output_sf: Optional[torch.Tensor] = None,
        attention_sinks: Optional[torch.Tensor] = None,
        chunked_prefill_buffer_batch_size: int = 1,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        assert isinstance(
            metadata,
            TrtllmAttentionMetadata,
        )
        assert not metadata.is_cross, "TRT-LLM Attention does not support cross attention yet."

        use_paged_context_fmha = (
            metadata.runtime_features.chunked_prefill
            or metadata.runtime_features.cache_reuse
            or metadata.runtime_features.has_speculative_draft_tokens
        ) if metadata.runtime_features else False

        if self.is_mla_enable:
            # Context MLA uses separate qkv instead of paged_context_fmha
            use_paged_context_fmha = False

        use_nvfp4_output = False
        if enable_attn_nvfp4_output and self.has_nvfp4 and self.support_nvfp4_output(
        ):
            # Runtime check whether the NVFP4 output kernel is available.
            use_nvfp4_output = self.wrapper.is_nvfp4_output_kernel_available(
                tokens_per_block=metadata.tokens_per_block,
                attention_mask=attention_mask,
                use_paged_context_fmha=use_paged_context_fmha,
                is_mla_enable=self.is_mla_enable,
            )

        sparse_kv_indices, sparse_kv_offsets, sparse_attn_indices, sparse_attn_offsets = None, None, None, None
        if self.sparse_attention_config is not None:
            sparse_kv_indices, sparse_kv_offsets = self.sparse_kv_predict(
                q, k, metadata, **kwargs)
            sparse_attn_indices, sparse_attn_offsets = self.sparse_attn_predict(
                q, k, metadata, **kwargs)

        self.wrapper.plan(
            layer_idx=self.get_local_layer_idx(metadata),
            tokens_per_block=metadata.tokens_per_block,
            max_num_requests=metadata.max_num_requests,
            max_sequence_length=metadata.max_seq_len,
            max_context_length=min(metadata.max_seq_len - 1,
                                   metadata.max_num_tokens),
            attention_window_size=attention_window_size,
            sink_token_length=0,
            beam_width=metadata.beam_width,
            sequence_length=metadata.kv_lens_cuda_runtime,
            host_past_key_value_lengths=metadata.kv_lens_runtime,
            host_total_kv_lens=metadata.host_total_kv_lens,
            context_lengths=metadata.prompt_lens_cuda_runtime,
            host_context_lengths=metadata.prompt_lens_cpu_runtime,
            host_request_types=metadata.host_request_types_runtime,
            kv_cache_block_offsets=metadata.kv_cache_block_offsets,
            host_kv_cache_block_offsets=metadata.host_kv_cache_block_offsets,
            host_kv_cache_pool_pointers=metadata.host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=metadata.host_kv_cache_pool_mapping,
            block_ids_per_seq=metadata.block_ids_per_seq,
            workspace=metadata.
            workspace,  # re-enable it, if pass None to it, fp8 mla will encounter invalid cuda free issue.
            cache_indirection=metadata.cache_indirection,
            kv_scale_orig_quant=self.kv_scale_orig_quant,
            kv_scale_quant_orig=self.kv_scale_quant_orig,
            out_scale=out_scale,
            out_scale_sf=out_scale_sf,
            kv_scales_sf=kv_scales_sf,
            kv_scales_sf_inv=kv_scales_sf_inv,
            use_nvfp4_output=use_nvfp4_output,
            use_paged_context_fmha=use_paged_context_fmha,
            attention_input_type=attention_input_type,
            latent_cache=latent_cache,
            q_pe=q_pe,
            mrope_config=mrope_config,
            softmax_stats_tensor=softmax_stats_tensor,
            helix_position_offsets=helix_position_offsets,
            is_spec_decoding_enabled=metadata.is_spec_decoding_enabled,
            use_spec_decoding=metadata.use_spec_decoding,
            is_spec_dec_tree=metadata.is_spec_dec_tree,
            spec_decoding_position_offsets=metadata.
            spec_decoding_position_offsets,
            spec_decoding_packed_mask=metadata.spec_decoding_packed_mask,
            spec_decoding_generation_lengths=metadata.
            spec_decoding_generation_lengths,
            attention_sinks=attention_sinks,
            chunked_prefill_buffer_batch_size=chunked_prefill_buffer_batch_size,
            sparse_kv_indices=sparse_kv_indices,
            sparse_kv_offsets=sparse_kv_offsets,
            sparse_attn_indices=sparse_attn_indices,
            sparse_attn_offsets=sparse_attn_offsets,
        )
        out_dtype = None
        if out_scale is not None:
            if use_nvfp4_output:
                # Use UINT8 as the container dtype for NVFP4.
                out_dtype = torch.uint8
            elif (self.has_fp8_qdq or self.has_nvfp4 or self.has_fp8_block_wise
                  or self.has_fp8_rowwise
                  or self.has_w4a8_nvfp4_fp8) and (self.has_fp8_kv_cache
                                                   or self.has_fp4_kv_cache):
                # TODO(qijun): revisit fp8_context_fmha logic
                out_dtype = torch.float8_e4m3fn

        output, output_sf = self.wrapper.run(
            q,
            k,
            v,
            output=output,
            output_sf=output_sf,
            out_dtype=out_dtype,
            is_fused_qkv=not metadata.is_cross and k is None,
            update_kv_cache=not metadata.is_cross or k is not None,
            attention_mask=attention_mask)

        if use_nvfp4_output:
            return output, output_sf

        return output

    @classmethod
    def support_fused_rope(cls) -> bool:
        return True

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return True

    @classmethod
    def support_mla(cls) -> bool:
        return True

    @classmethod
    def support_nvfp4_output(cls) -> bool:
        # Default enabled, but allow manual disabling through `TRTLLM_ENABLE_ATTENTION_NVFP4_OUTPUT=0`
        return os.environ.get("TRTLLM_ENABLE_ATTENTION_NVFP4_OUTPUT",
                              "1") == "1"

    def has_cached_kv_for_mla_context(
        self,
        metadata: TrtllmAttentionMetadata,
    ) -> bool:
        return (self.is_mla_enable and metadata.kv_cache_manager is not None
                and metadata.enable_context_mla_with_cached_kv
                and metadata.num_ctx_cached_tokens > 0)

    def is_chunked_prefill_for_mla_context(
        self,
        metadata: TrtllmAttentionMetadata,
    ) -> bool:
        return (self.is_mla_enable and metadata.kv_cache_manager is not None
                and metadata.enable_context_mla_with_cached_kv
                and metadata.num_ctx_cached_tokens > 0
                and metadata.runtime_features.chunked_prefill)

    def load_paged_kv_cache_for_mla(
        self,
        metadata: TrtllmAttentionMetadata,
        out_dtype: torch.dtype,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert out_dtype in [torch.float16, torch.bfloat16, torch.float32]
        assert self.is_mla_enable and self.mla_params is not None
        assert metadata.kv_cache_manager is not None
        assert metadata.max_ctx_kv_len > 0
        assert metadata.num_ctx_cached_tokens + metadata.num_ctx_tokens == metadata.host_ctx_kv_indptr[
            metadata.num_contexts]

        sink_token_length = 0
        beam_width = 1

        compressed_kv, k_pe = torch.ops.trtllm.load_paged_kv_cache_for_mla(
            out_dtype,
            metadata.num_contexts,
            metadata.num_ctx_cached_tokens + metadata.num_ctx_tokens,
            metadata.max_ctx_kv_len,
            metadata.ctx_kv_indptr,
            metadata.kv_cache_block_offsets,
            metadata.host_kv_cache_block_offsets,
            metadata.kv_cache_manager.kv_cache_pool_pointers,
            metadata.kv_cache_manager.kv_cache_pool_mapping,
            self.kv_scale_orig_quant,
            self.kv_scale_quant_orig,
            self.get_local_layer_idx(metadata),
            self.mla_params.kv_lora_rank,
            self.mla_params.qk_rope_head_dim,
            metadata.kv_cache_manager.tokens_per_block,
            metadata.kv_cache_manager.max_seq_len,
            sink_token_length,
            beam_width,
            self.wrapper.quant_mode,
        )

        return compressed_kv, k_pe

    def load_chunked_kv_cache_for_mla(
        self,
        metadata: TrtllmAttentionMetadata,
        num_ctx_cached_tokens: int,
        cu_chunked_seq_len: torch.Tensor,
        chunked_global_offset: torch.Tensor,
        chunked_max_seq_len: int,
        out_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert out_dtype in [torch.float16, torch.bfloat16, torch.float32]
        assert self.is_mla_enable and self.mla_params is not None
        assert metadata.kv_cache_manager is not None

        if metadata.max_ctx_cached_token_len == 0:
            empty_kv = torch.empty((0, self.mla_params.kv_lora_rank),
                                   dtype=out_dtype,
                                   device=cu_chunked_seq_len.device)
            empty_k_pe = torch.empty((0, self.mla_params.qk_rope_head_dim),
                                     dtype=out_dtype,
                                     device=cu_chunked_seq_len.device)
            return empty_kv, empty_k_pe

        sink_token_length = 0
        beam_width = 1

        output_kv, output_k_pe = torch.ops.trtllm.load_chunked_kv_cache_for_mla(
            out_dtype,
            metadata.num_contexts,
            num_ctx_cached_tokens,
            cu_chunked_seq_len,
            chunked_global_offset,
            metadata.kv_cache_block_offsets,
            metadata.kv_cache_manager.kv_cache_pool_pointers,
            metadata.kv_cache_manager.kv_cache_pool_mapping,
            self.kv_scale_orig_quant,
            self.kv_scale_quant_orig,
            self.get_local_layer_idx(metadata),
            self.mla_params.kv_lora_rank,
            self.mla_params.qk_rope_head_dim,
            metadata.kv_cache_manager.tokens_per_block,
            chunked_max_seq_len,
            metadata.kv_cache_manager.max_seq_len,
            sink_token_length,
            beam_width,
            self.wrapper.quant_mode,
        )
        return output_kv, output_k_pe

    def mla_rope_append_paged_kv_assign_q(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        metadata: TrtllmAttentionMetadata,
        **kwargs,
    ) -> None:
        assert self.is_mla_enable and self.mla_params is not None
        assert metadata.kv_cache_manager is not None

        sink_token_length = 0
        beam_width = 1

        torch.ops.trtllm.mla_rope_append_paged_kv_assign_q(
            q,
            latent_cache,
            metadata.num_contexts,
            metadata.ctx_cached_token_indptr,
            metadata.ctx_kv_indptr,
            metadata.max_ctx_seq_len,
            self.wrapper.rotary_cos_sin,
            self.num_heads,
            self.mla_params.qk_nope_head_dim,
            self.mla_params.qk_rope_head_dim,
            self.mla_params.kv_lora_rank,
            metadata.kv_cache_block_offsets,
            metadata.host_kv_cache_block_offsets,
            metadata.kv_cache_manager.kv_cache_pool_pointers,
            metadata.kv_cache_manager.kv_cache_pool_mapping,
            self.kv_scale_orig_quant,
            self.kv_scale_quant_orig,
            self.get_local_layer_idx(metadata),
            metadata.kv_cache_manager.tokens_per_block,
            metadata.kv_cache_manager.max_seq_len,
            sink_token_length,
            beam_width,
            self.wrapper.quant_mode,
        )

    def merge_attention_for_mla(
        self,
        merged_attn: torch.Tensor,
        temp_attn: torch.Tensor,
        softmax_stats: torch.Tensor,
        temp_softmax_stats: torch.Tensor,
        merge_op: torch.Tensor,
        metadata: TrtllmAttentionMetadata,
    ) -> None:
        assert self.is_mla_enable and self.mla_params is not None
        assert metadata.kv_cache_manager is not None

        torch.ops.trtllm.merge_chunked_attention_for_mla(
            merged_attn,
            temp_attn,
            softmax_stats,
            temp_softmax_stats,
            metadata.num_contexts,
            metadata.ctx_uncached_token_indptr,  # cu_q_seq_len
            metadata.max_ctx_seq_len,  # max_q_seq_len
            merge_op,
            self.num_heads,
            self.mla_params.v_head_dim,
        )

    def sparse_attn_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: TrtllmAttentionMetadata,
        **kwargs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
            Predict sparse attn indices. It's implemented in the derived class.
        """
        raise NotImplementedError

    def sparse_kv_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: TrtllmAttentionMetadata,
        **kwargs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
            Predict sparse kv indices. It's implemented in the derived class.
        """
        raise NotImplementedError
