import math
import os
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from ..speculative.utils import SpecDecodingTensor
    from ..speculative.interface import SpecMetadata
    from ..speculative.spec_tree_manager import SpecTreeManager

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.functional import AttentionMaskType
from tensorrt_llm.llmapi import SkipSoftmaxAttentionConfig
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
    spec_decoding_bl_tree_mask_offset: Optional[torch.Tensor]
    spec_decoding_bl_tree_mask: Optional[torch.Tensor]
    spec_bl_tree_first_sparse_mask_offset_kv: Optional[torch.Tensor]
    helix_position_offsets: Optional[torch.Tensor]
    helix_is_inactive_rank: Optional[torch.Tensor]
    attention_input_type: Optional[torch.Tensor]
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
        self.attention_input_type = None
        self.kwargs = {}
        self.kwargs.update(kwargs)
        self.skip_softmax_stat = torch.zeros(2,
                                             dtype=torch.uint32,
                                             device='cuda')
        # Default disabled, but allow manual enabling through `TRTLLM_PRINT_SKIP_SOFTMAX_STAT=1`
        self.print_skip_softmax_stat = os.environ.get(
            "TRTLLM_PRINT_SKIP_SOFTMAX_STAT", "0") == "1"

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
        is_spec_decoding_enabled: bool = False,
        use_spec_decoding: bool = False,
        is_spec_dec_tree: bool = False,
        spec_decoding_position_offsets: Optional[torch.Tensor] = None,
        spec_decoding_packed_mask: Optional[torch.Tensor] = None,
        spec_decoding_generation_lengths: Optional[torch.Tensor] = None,
        spec_decoding_bl_tree_mask_offset: Optional[torch.Tensor] = None,
        spec_decoding_bl_tree_mask: Optional[torch.Tensor] = None,
        spec_bl_tree_first_sparse_mask_offset_kv: Optional[torch.Tensor] = None,
        attention_sinks: Optional[torch.Tensor] = None,
        chunked_prefill_buffer_batch_size: int = 1,
        sparse_kv_indices: Optional[torch.Tensor] = None,
        sparse_kv_offsets: Optional[torch.Tensor] = None,
        sparse_attn_indices: Optional[torch.Tensor] = None,
        sparse_attn_offsets: Optional[torch.Tensor] = None,
        sparse_attn_indices_block_size: int = 1,
        sparse_mla_topk: int = 0,
        skip_softmax_threshold_scale_factor_prefill: Optional[float] = None,
        skip_softmax_threshold_scale_factor_decode: Optional[float] = None,
        helix_position_offsets: Optional[torch.Tensor] = None,
        helix_is_inactive_rank: Optional[torch.Tensor] = None,
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
            kv_cache_block_offsets (torch.Tensor): The offsets to the blocks inside KV cache pools on GPU, its shape is (num_pools, max_batch_size * max_beam_width, 2, max_blocks_per_sequence), one for each block. If kv_cache_block_offsets, host_kv_cache_pool_pointers, host_kv_cache_pool_mapping are all None, the attention will be no cache attention.
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
            attention_sinks (torch.Tensor): The attention sinks (additional value in the denominator of the softmax) with shape of (num_heads_q) on GPU.
            chunked_prefill_buffer_batch_size (int): used for malloc buffer for k and v in fp8 context mla. the max input kv length is not max_num_tokens in this case. It is chunked_prefill_buffer_batch_size * max_num_tokens.
            sparse_kv_indices (torch.Tensor): The sparse indices for the KV cache, with shape of (num_heads_kv, num_sparse_tokens) on GPU.
            sparse_kv_offsets (torch.Tensor): The batch offsets for the sparse KV indices, with shape of (num_contexts + 1) on GPU.
            sparse_attn_indices (torch.Tensor): The sparse indices for the attention layer, with shape of (num_heads_kv, num_sparse_tokens) on GPU.
            sparse_attn_offsets (torch.Tensor): The batch offsets for the sparse attention indices, with shape of (num_generations + 1) on GPU.
            sparse_attn_indices_block_size (int): The granularity of the sparse attention indices, used by block sparse attention.
            sparse_mla_topk (int): The topk for the sparse MLA, used by DSA attention.
            skip_softmax_threshold_scale_factor_prefill (float): The scale factor for the skip softmax threshold in prefill phase.
            skip_softmax_threshold_scale_factor_decode (float): The scale factor for the skip softmax threshold in decode phase.
            helix_position_offsets (torch.Tensor): The tensor to store the helix position offsets, with shape (num_tokens) on GPU.
            helix_is_inactive_rank (torch.Tensor): For Helix: whether the current rank is inactive, with shape (batch_size) on GPU.
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
        self.attention_sinks = attention_sinks
        self.sparse_kv_indices = sparse_kv_indices
        self.sparse_kv_offsets = sparse_kv_offsets
        self.sparse_attn_indices = sparse_attn_indices
        self.sparse_attn_offsets = sparse_attn_offsets
        self.sparse_attn_indices_block_size = sparse_attn_indices_block_size
        self.sparse_mla_topk = sparse_mla_topk
        self.helix_position_offsets = helix_position_offsets
        self.helix_is_inactive_rank = helix_is_inactive_rank

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
        self.spec_decoding_bl_tree_mask_offset = spec_decoding_bl_tree_mask_offset
        self.spec_decoding_bl_tree_mask = spec_decoding_bl_tree_mask
        self.spec_bl_tree_first_sparse_mask_offset_kv = spec_bl_tree_first_sparse_mask_offset_kv
        self.chunked_prefill_buffer_batch_size = chunked_prefill_buffer_batch_size
        self.skip_softmax_threshold_scale_factor_prefill = skip_softmax_threshold_scale_factor_prefill
        self.skip_softmax_threshold_scale_factor_decode = skip_softmax_threshold_scale_factor_decode
        self.kwargs.update(kwargs)

    def create_output(
        self,
        q: torch.Tensor,
        out_dtype: Optional[torch.dtype],
        use_nvfp4_output: bool,
        is_gen_only: bool,
    ):
        num_tokens = q.size(0)
        if out_dtype is None:
            out_dtype = q.dtype
        v_head_size = self.head_size
        if self.is_mla_enable:
            v_head_size = self.kv_lora_rank if is_gen_only else self.v_head_dim
        if use_nvfp4_output:
            num_nvfp4_elements_per_container = 2
            scaling_vector_size = 16
            size_per_token = self.num_heads * v_head_size
            output = q.new_empty(
                (num_tokens,
                 size_per_token // num_nvfp4_elements_per_container),
                dtype=torch.uint8)
            # Create a sf (scaling factors) tensor for NVFP4 (use INT8 as the container dtype).
            padded_row, padded_col = compute_swizzled_sf_shape(
                num_tokens, size_per_token // scaling_vector_size)
            output_sf = q.new_empty(padded_row * padded_col, dtype=torch.uint8)
            return [output, output_sf]
        else:
            return [
                q.new_empty((num_tokens, self.num_heads * v_head_size),
                            dtype=out_dtype)
            ]

    def run(
        self,
        q: torch.Tensor,
        output: torch.Tensor,
        output_sf: Optional[torch.Tensor] = None,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        is_fused_qkv: bool = True,
        update_kv_cache: bool = True,
        attention_mask: AttentionMask = PredefinedAttentionMask.CAUSAL,
        cu_q_seqlens: Optional[torch.Tensor] = None,
        cu_kv_seqlens: Optional[torch.Tensor] = None,
        fmha_scheduler_counter: Optional[torch.Tensor] = None,
        mla_bmm1_scale: Optional[torch.Tensor] = None,
        mla_bmm2_scale: Optional[torch.Tensor] = None,
        quant_q_buffer: Optional[torch.Tensor] = None,
    ):
        """
        Run the attention operation.
        Args:
            q (torch.Tensor): Query tensor with shape (num_tokens, num_heads * head_dim) or QKV tensor with shape (num_tokens, (num_heads + 2 * num_kv_heads) * head_dim).
            output (torch.Tensor): Output tensor with shape.
            output_sf (Optional[torch.Tensor]): Output scaling factors tensor.
            k (Optional[torch.Tensor]): Key tensor with shape (num_tokens, num_kv_heads * head_dim) or None if QKV tensor is provided.
            v (Optional[torch.Tensor]): Value tensor with shape (num_tokens, num_kv_heads * head_dim) or None if QKV tensor is provided.
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
            # For DSA, use the same qkv hidden size for context and generation phases
            is_sparse_attn = self.sparse_attn_indices is not None and self.sparse_attn_indices.numel(
            ) > 0
            if self.attention_input_type == AttentionInputType.context_only and is_sparse_attn:
                assert is_fused_qkv
                qkv_hidden_size = self.num_heads * (self.kv_lora_rank +
                                                    self.qk_rope_head_dim)
            elif self.attention_input_type == AttentionInputType.context_only:
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
        if self.is_sm_version_trtllm_gen_kernel(sm=get_sm_version()):
            spec_decoding_tensor_params.append(
                self.spec_decoding_bl_tree_mask_offset)
            spec_decoding_tensor_params.append(self.spec_decoding_bl_tree_mask)
            spec_decoding_tensor_params.append(
                self.spec_bl_tree_first_sparse_mask_offset_kv)
        mla_tensor_params = [
            self.helix_position_offsets, self.helix_is_inactive_rank
        ]

        if self.print_skip_softmax_stat:
            self.skip_softmax_stat.zero_()

        thop.attention(
            q,
            k,
            v,
            output,
            output_sf,
            self.workspace,
            self.sequence_length,
            self.host_past_key_value_lengths,
            self.host_total_kv_lens,
            self.context_lengths,
            self.host_context_lengths,
            self.host_request_types,
            self.kv_cache_block_offsets,
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
            self.sparse_kv_indices,
            self.sparse_kv_offsets,
            self.sparse_attn_indices,
            self.sparse_attn_offsets,
            self.sparse_attn_indices_block_size,
            self.sparse_mla_topk,
            self.skip_softmax_threshold_scale_factor_prefill,
            self.skip_softmax_threshold_scale_factor_decode,
            self.skip_softmax_stat,
            cu_q_seqlens,
            cu_kv_seqlens,
            fmha_scheduler_counter,
            mla_bmm1_scale,
            mla_bmm2_scale,
            quant_q_buffer,
        )

        if self.print_skip_softmax_stat:
            (total_blocks, skipped_blocks) = self.skip_softmax_stat
            if total_blocks != 0:
                print(
                    f"SKIP_SOFTMAX_STAT: layer{self.layer_idx}: {skipped_blocks} / {total_blocks}"
                    f" = {skipped_blocks / total_blocks * 100: .2f}%")

        # reset the planned states (especially tensors) to avoid memory leak
        self.plan()

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

    def is_sm_version_trtllm_gen_kernel(self, sm):
        return not (sm < 100 or sm in [120, 121])


@dataclass(kw_only=True)
class TrtllmAttentionMetadata(AttentionMetadata):
    workspace: Optional[torch.Tensor] = None
    cuda_graph_workspace: Optional[torch.Tensor] = None

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
    spec_decoding_bl_tree_mask_offset: Optional[torch.Tensor] = None
    spec_decoding_bl_tree_mask: Optional[torch.Tensor] = None
    spec_bl_tree_first_sparse_mask_offset_kv: Optional[torch.Tensor] = None

    # Flag to enable helix parallelism.
    enable_helix: bool = False

    # Global position ids of tokens for each sequence in the batch. Given
    # each helix rank has only a subset of tokens for a sequence, we compute
    # a global position id for each token here.
    helix_position_offsets: Optional[torch.Tensor] = None
    helix_position_offsets_cpu: Optional[torch.Tensor] = None

    # Whether the current rank is inactive for helix parallelism.
    # In helix parallelism, only the active rank appends KV cache for the query token
    # and attends to the previously cached tokens as well as the query token. Inactive
    # ranks do not append KV cache for the query token and attend to the previously
    # cached tokens only.
    helix_is_inactive_rank: Optional[torch.Tensor] = None
    helix_is_inactive_rank_cpu: Optional[torch.Tensor] = None

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
        self.enable_helix = self.mapping.has_cp_helix(
        ) if self.mapping is not None else False
        self._post_init_with_buffers(self.cuda_graph_buffers)

    def _post_init_with_buffers(self, buffers) -> None:

        # Set a default value, as max_num_sequences is not always set.
        if self.max_num_sequences is None:
            self.max_num_sequences = self.max_num_requests

        capture_graph = self.is_cuda_graph

        self.prompt_lens_cuda = self.get_empty(
            buffers,
            (self.max_num_sequences, ),
            cache_name="prompt_lens_cuda",
            dtype=torch.int,
            capture_graph=capture_graph,
        )
        self.prompt_lens_cpu = torch.empty_like(
            self.prompt_lens_cuda,
            device='cpu',
            pin_memory=True,
        )
        self.kv_lens_cuda = self.get_empty_like(
            buffers,
            self.prompt_lens_cuda,
            cache_name="kv_lens_cuda",
            capture_graph=capture_graph,
        )
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

        if self.cuda_graph_workspace is None:
            self.cuda_graph_workspace = torch.empty(
                (0, ),
                device='cuda',
                dtype=torch.int8,
            )

        if self.kv_cache_manager is not None:
            self.kv_cache_block_offsets = self.get_empty(
                buffers,
                [
                    self.kv_cache_manager.num_pools, self.max_num_sequences, 2,
                    self.kv_cache_manager.max_blocks_per_seq
                ],
                cache_name="kv_cache_block_offsets",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            self.block_ids_per_seq = None
            self.kv_block_ids_per_seq = None
            if self.enable_flash_mla:
                self.block_ids_per_seq = self.get_empty(
                    buffers,
                    [
                        self.kv_cache_manager.max_batch_size,
                        self.kv_cache_manager.max_blocks_per_seq
                    ],
                    cache_name="block_ids_per_seq",
                    dtype=torch.int32,
                    capture_graph=capture_graph,
                )
                self.kv_block_ids_per_seq = self.get_empty(
                    buffers,
                    [
                        self.kv_cache_manager.max_batch_size,
                        self.kv_cache_manager.max_blocks_per_seq
                    ],
                    cache_name="kv_block_ids_per_seq",
                    dtype=torch.int32,
                    capture_graph=capture_graph,
                )
            if self.enable_context_mla_with_cached_kv:
                # for kv cache reuse/chunked context in MLA
                self.ctx_cached_token_indptr = self.get_empty(
                    buffers,
                    (self.max_num_requests + 1, ),
                    cache_name="ctx_cached_token_indptr",
                    dtype=torch.int64,
                    capture_graph=capture_graph,
                )
                self.host_ctx_cached_token_indptr = torch.zeros_like(
                    self.ctx_cached_token_indptr,
                    device='cpu',
                    pin_memory=True,
                )
                self.ctx_uncached_token_indptr = self.get_empty(
                    buffers,
                    (self.max_num_requests + 1, ),
                    cache_name="ctx_uncached_token_indptr",
                    dtype=torch.int64,
                    capture_graph=capture_graph,
                )
                self.host_ctx_uncached_token_indptr = torch.zeros_like(
                    self.ctx_uncached_token_indptr,
                    device='cpu',
                    pin_memory=True,
                )
                # context full seqlens include cached tokens and uncached tokens
                self.ctx_kv_indptr = self.get_empty(
                    buffers,
                    (self.max_num_requests + 1, ),
                    cache_name="ctx_kv_indptr",
                    dtype=torch.int64,
                    capture_graph=capture_graph,
                )
                self.host_ctx_kv_indptr = torch.zeros_like(
                    self.ctx_kv_indptr,
                    device='cpu',
                    pin_memory=True,
                )

        # Allocate static buffers for helix parallelism support.
        if self.enable_helix:
            self.helix_position_offsets = self.get_empty(
                buffers,
                (self.max_num_tokens, ),
                cache_name="helix_position_offsets",
                dtype=torch.int,
                capture_graph=capture_graph,
            )
            self.helix_position_offsets_cpu = torch.empty_like(
                self.helix_position_offsets,
                device='cpu',
                pin_memory=True,
            )
            self.helix_is_inactive_rank = self.get_empty(
                buffers,
                (self.max_num_sequences, ),
                cache_name="helix_is_inactive_rank",
                dtype=torch.bool,
                capture_graph=capture_graph,
            )
            self.helix_is_inactive_rank_cpu = torch.empty_like(
                self.helix_is_inactive_rank,
                device='cpu',
                pin_memory=True,
            )

    def on_update_kv_lens(self):
        # After changing the kv_lens/kv_lens_cuda, we may need to update other metadata.
        # Especially for the changes in the _preprocess_inputs() of model_engine.py.
        pass

    def update_helix_param(
        self,
        helix_position_offsets: List[int],
        helix_is_inactive_rank: List[bool],
    ) -> None:
        """
        Update helix parameters by copying into static buffers for CUDA graph compatibility.

        Args:
            helix_position_offsets: Position offsets for helix parallelism with shape (num_tokens,).
            helix_is_inactive_rank: Whether the current rank is inactive with shape (batch_size,).
        """
        if helix_position_offsets is not None and self.helix_position_offsets is not None:
            num_tokens = len(helix_position_offsets)
            self.helix_position_offsets_cpu[:num_tokens].copy_(
                torch.tensor(helix_position_offsets, dtype=torch.int))
            self.helix_position_offsets[:num_tokens].copy_(
                self.helix_position_offsets_cpu[:num_tokens], non_blocking=True)

        if helix_is_inactive_rank is not None and self.helix_is_inactive_rank is not None:
            batch_size = len(helix_is_inactive_rank)
            self.helix_is_inactive_rank_cpu[:batch_size].copy_(
                torch.tensor(helix_is_inactive_rank, dtype=torch.bool))
            self.helix_is_inactive_rank[:batch_size].copy_(
                self.helix_is_inactive_rank_cpu[:batch_size], non_blocking=True)

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

        # number of tokens needed in the kv cache for each sequence after the next pass.
        if self.enable_helix:
            # If helix is inactive, attend to the previously cached tokens only.
            assert cached_token_lens is not None, "cached_token_lens should be set for helix"
            active_rank = ~self.helix_is_inactive_rank_cpu[:self.num_seqs]
            kv_lens = cached_token_lens.clone()
            kv_lens[active_rank] += self.seq_lens_kv[active_rank]
        else:
            kv_lens = cached_token_lens + \
                self.seq_lens_kv if cached_token_lens is not None else self.seq_lens_kv

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
            self.kv_cache_manager.copy_batch_block_offsets(
                self.kv_cache_block_offsets, self.request_ids, self.beam_width,
                self.num_contexts, self.num_seqs)

            error_message = (
                f"The max KV cache length of input sequences ({self.kv_lens[:self.num_seqs].max()}) "
                f"exceeds the KV cache manager's maximum supported length "
                f"({self.kv_cache_manager.max_seq_len}).")

            assert self.kv_lens[:self.num_seqs].max(
            ) <= self.kv_cache_manager.max_seq_len, error_message

        self.kv_lens_cuda_runtime = self.kv_lens_cuda[:self.num_seqs]
        # Don't use self.kv_lens here because it includes extra tokens.
        # Use actual KV length (without extra tokens) for kv_lens_runtime,
        # which becomes host_past_key_value_lengths and eventually mMaxSeqLenKv.
        self.kv_lens_runtime = kv_lens[:self.num_seqs]
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

    def compute_max_num_custom_mask_tiles_kv_upper_bound(
            self, max_seq_len_kv, min_first_sparse_mask_offset_kv,
            tile_size_kv_per_cta) -> int:
        """
        Compute the conservative upper bound of numCustomMaskTilesKv.

        Args:
            max_seq_len_kv (int): The maximum seqLenKv in the batch
            min_first_sparse_mask_offset_kv (int): The minimum firstSparseMaskOffsetKv in the batch
            tile_size_kv_per_cta (int): tileSizeKvPerCta value
        """
        first_sparse_tile_offset = min_first_sparse_mask_offset_kv // tile_size_kv_per_cta
        num_tiles_kv_total = math.ceil(max_seq_len_kv / tile_size_kv_per_cta)
        max_num_custom_mask_tiles_kv = num_tiles_kv_total - first_sparse_tile_offset
        return max_num_custom_mask_tiles_kv

    def spec_decoding_param_prepare_for_blackwell(self) -> None:
        """
        Prepare the blackwell parameters for the speculative decoding (Medusa and Eagle) generation-phase attention kernels.
        """
        self.spec_decoding_bl_tree_mask_offset = torch.zeros(
            [self.max_num_requests],
            dtype=torch.int64,
            device='cuda',
        )
        max_kv_len = self.kv_lens[:self.num_seqs].max()
        assert self.kv_lens_cuda[:self.
                                 num_seqs] >= self._seq_lens_cuda[:self.
                                                                  num_seqs], "kv_lens should be greater than seq_lens,please run prepare() first"

        # Only support seq_lens are equal in one batch
        seq_lens_slice = self.seq_lens[:self.num_seqs]
        assert seq_lens_slice.min() == seq_lens_slice.max(), \
            f"All elements in seq_lens must be equal in one batch, but got min={seq_lens_slice.min()}, max={seq_lens_slice.max()}"

        self.spec_bl_tree_first_sparse_mask_offset_kv = (
            self.kv_lens_cuda[:self.num_seqs] -
            self._seq_lens_cuda[:self.num_seqs]).to(torch.int32)
        min_first_sparse_mask_offset_kv = self.spec_bl_tree_first_sparse_mask_offset_kv.min(
        )
        # tile_size_kv * tile_size_q * num_instances_q * num_instances_kv is the largest value that is used in the trtllm-gen kernels
        tile_size_kv = 128
        tile_size_q = 128
        # num_instances_q * num_instances_kv <= 2
        num_instances_q = 1
        num_instances_kv = 2
        tile_size_kv_per_cta = tile_size_kv * num_instances_kv
        tile_size_q_per_cta = tile_size_q * num_instances_q
        max_num_custom_mask_tiles_kv = self.compute_max_num_custom_mask_tiles_kv_upper_bound(
            max_kv_len, min_first_sparse_mask_offset_kv, tile_size_kv_per_cta)
        max_num_tiles_q = math.ceil(
            (self.seq_lens[:self.num_seqs].max() * self.num_heads_per_kv) /
            tile_size_q_per_cta)
        mask_size = int(self.max_num_requests * max_num_tiles_q *
                        max_num_custom_mask_tiles_kv * num_instances_q *
                        num_instances_kv * tile_size_q * tile_size_kv / 32)
        self.spec_decoding_bl_tree_mask = torch.zeros(
            mask_size,
            dtype=torch.uint32,
            device='cuda',
        )

    def update_spec_dec_param(
        self,
        batch_size,
        is_spec_decoding_enabled,
        is_spec_dec_tree,
        is_spec_dec_dynamic_tree,
        max_draft_len,
        max_total_draft_tokens,
        model_is_wrapped: bool = False,
        spec_metadata: Optional['SpecMetadata'] = None,
        spec_tree_manager: Optional['SpecTreeManager'] = None,
        spec_decoding_tensor: Optional['SpecDecodingTensor'] = None,
    ) -> None:
        '''
        Update the spec-dec parameters for the TRTLLM attention layer.
        Args:
            batch_size: int, the number of requests in the batch.
            is_spec_decoding_enabled: bool, whether the attention need to be spec_decoding mode, which is determined by attention_need_spec_dec_mode() function.
            is_spec_dec_tree: bool, whether the spec-dec mode is a tree, i.e., static tree or dynamic tree. For linear-tree, it is always False.
            is_spec_dec_dynamic_tree: bool, whether using dynamic tree.
            max_draft_len: int, the number of the draft layers.
            max_total_draft_tokens: int, the number of all nodes in the tree (except the root).
            model_is_wrapped: Optional[bool] = False, whether the drafter model is wrapped (i.e, CDL).
            spec_metadata: Optional['SpecMetadata'] = None, the metadata of the spec-dec.
            spec_tree_manager: Optional['SpecTreeManager'] = None, the spec_tree_manager for draft token tree.
            spec_decoding_tensor: Optional['SpecDecodingTensor'] = None, the spec_decoding_tensor for draft token tree.
        '''
        if spec_decoding_tensor is not None:
            spec_decoding_position_offsets = spec_decoding_tensor.position_offsets
            spec_decoding_packed_mask = spec_decoding_tensor.packed_mask
            spec_decoding_generation_lengths = spec_decoding_tensor.generation_lengths
        else:
            spec_decoding_position_offsets = None
            spec_decoding_packed_mask = None
            spec_decoding_generation_lengths = None

        # spec_dec mode should only be enabled for non-sm100 machines and when there's a spec-dec tree.
        self.is_spec_decoding_enabled = is_spec_decoding_enabled and (
            not self.is_sm_version_trtllm_gen_kernel(sm=get_sm_version()))

        self.is_spec_dec_tree = spec_tree_manager is not None
        self.is_spec_dec_dynamic_tree = spec_tree_manager is not None and spec_tree_manager.use_dynamic_tree

        if self.is_sm_version_trtllm_gen_kernel(sm=get_sm_version()):
            if self.is_spec_dec_tree or self.is_spec_dec_dynamic_tree:
                assert not self.is_spec_dec_tree, "Spec-dec tree is not supported on this machine. Please use a pre-Blackwell machine for a spec-dec tree."

        # use_spec_decoding is default to true by default, change in runtime by layers / requests
        self.use_spec_decoding = self.is_spec_decoding_enabled

        self.is_spec_dec_tree = is_spec_dec_tree
        self.is_spec_dec_dynamic_tree = is_spec_dec_dynamic_tree

        # Parameters can be fixed and not changed during runtime if the
        if self.is_spec_decoding_enabled:
            # These buffers are accessed more like removing input padding,
            # rather than using max_total_draft_tokens + 1 as the offset between different requests.
            if self.spec_decoding_position_offsets is None:
                self.spec_decoding_position_offsets = torch.empty(
                    [self.max_num_requests, max_total_draft_tokens + 1],
                    dtype=torch.int,
                    device='cuda',
                )
            if self.spec_decoding_packed_mask is None:
                self.spec_decoding_packed_mask = torch.empty(
                    [
                        self.max_num_requests, max_total_draft_tokens + 1,
                        math.ceil((max_total_draft_tokens + 1) / 32)
                    ],
                    dtype=torch.int,
                    device='cuda',
                )
            if self.spec_decoding_generation_lengths is None:
                self.spec_decoding_generation_lengths = torch.empty(
                    [self.max_num_requests],
                    dtype=torch.int,
                    device='cuda',
                )

            if self.is_sm_version_trtllm_gen_kernel(sm=get_sm_version()):
                self.spec_decoding_param_prepare_for_blackwell()
            else:
                self.spec_decoding_bl_tree_mask_offset = None
                self.spec_decoding_bl_tree_mask = None
                self.spec_bl_tree_first_sparse_mask_offset_kv = None

            # Case 1: dynamic tree
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
                        max_draft_len=max_total_draft_tokens)

            # Case 2/3: static tree
            elif self.is_spec_dec_tree and not self.is_spec_dec_dynamic_tree and spec_metadata is not None:
                assert spec_metadata.spec_dec_mode.is_eagle3(
                ), "Tree decoding is only supported for Eagle3 now"

                is_target_model = not getattr(spec_metadata, 'is_draft_model',
                                              False)

                # Case 2: static tree and target model
                if is_target_model:
                    # For the target model, we update the spec-dec parameters with the spec_tree_manager, which is prepared in advance.
                    self.spec_decoding_position_offsets[:batch_size, :].copy_(
                        spec_tree_manager.spec_dec_position_offsets[0, :],
                        non_blocking=True)
                    self.spec_decoding_packed_mask[:batch_size, :, :].copy_(
                        spec_tree_manager.spec_dec_packed_mask[0, :, :],
                        non_blocking=True)
                    self.spec_decoding_generation_lengths[:batch_size].fill_(
                        spec_tree_manager.max_total_draft_tokens + 1)

                # Case 3: static tree and the first drafter layer
                else:
                    assert model_is_wrapped == True, "The drafter model should be wrapped"
                    # The first drafter layer will take the padded tokens as input (padding to the max_draft_len + 1)
                    # But the spec-dec parameters are still in the shape of max_total_draft_tokens + 1.
                    # Considering that these spec-dec params are accessed consecutively (without padding) in the attention Op,
                    # we need to write them consecutively when setting them.
                    # For the next drafter layers, we will prepare these spec-dec params in the drafting loops.
                    # position_offsets
                    position_offset = torch.arange(
                        max_draft_len + 1,
                        dtype=torch.int,
                        device='cpu',
                        pin_memory=True).repeat(batch_size)
                    self.spec_decoding_position_offsets.reshape(
                        -1)[:(max_draft_len + 1) * batch_size].copy_(
                            position_offset, non_blocking=True)
                    # packed_mask
                    dummy_idx = torch.arange(max_draft_len + 1)
                    spec_decoding_packed_mask = torch.pow(
                        2, dummy_idx + 1) - 1  # [max_draft_len + 1]
                    spec_decoding_packed_mask = spec_decoding_packed_mask.repeat(
                        batch_size)  # [batch_size * (max_draft_len + 1)]
                    self.spec_decoding_packed_mask.reshape(
                        -1)[:(max_draft_len + 1) * batch_size].copy_(
                            spec_decoding_packed_mask, non_blocking=True)
                    # generation_lengths
                    self.generate_spec_decoding_generation_length(
                        max_draft_len=max_draft_len)

            # Case 4: linear tree
            else:
                assert max_draft_len == max_total_draft_tokens, "max_draft_len should be equal to max_total_draft_tokens for linear tree"
                # Prepare for the linear-tree.
                # Populate the mask that won't change during inference phase.
                self.generate_spec_decoding_position_offsets(
                    max_draft_len=max_draft_len)
                self.generate_spec_decoding_packed_mask(
                    max_draft_len=max_draft_len)
                self.generate_spec_decoding_generation_length(
                    max_draft_len=max_draft_len)

    def generate_spec_decoding_position_offsets(self, max_draft_len):
        position_offset = torch.arange(max_draft_len + 1,
                                       dtype=torch.int,
                                       device='cpu',
                                       pin_memory=True)
        # fill all the batches with same position offset
        self.spec_decoding_position_offsets.copy_(position_offset,
                                                  non_blocking=True)

    def generate_spec_decoding_packed_mask(self, max_draft_len):
        num_blocks = math.ceil((max_draft_len + 1) / 32)
        tmp_max_draft_len = max_draft_len + 1
        for block_idx in range(num_blocks):
            if tmp_max_draft_len < 0:
                break
            dummy_idx = torch.arange(min(32, tmp_max_draft_len))
            spec_decoding_packed_mask = torch.pow(2, dummy_idx + 1) - 1
            self.spec_decoding_packed_mask[:, :, block_idx].copy_(
                spec_decoding_packed_mask, non_blocking=True)
            tmp_max_draft_len -= 32

    def generate_spec_decoding_generation_length(self, max_draft_len):
        spec_decoding_generation_length = torch.full((self.max_num_requests, ),
                                                     max_draft_len + 1)
        self.spec_decoding_generation_lengths[:self.max_num_requests].copy_(
            spec_decoding_generation_length, non_blocking=True)

    def is_sm_version_trtllm_gen_kernel(self, sm):
        return not (sm < 100 or sm in [120, 121])


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

    def use_nvfp4_output(
        self,
        metadata: TrtllmAttentionMetadata,
        attention_mask: AttentionMask,
    ) -> bool:
        # Not running NVFP4
        if not self.has_nvfp4:
            return False

        # Default enabled, but allow manual disabling through `TRTLLM_ENABLE_ATTENTION_NVFP4_OUTPUT=0`
        if not os.environ.get("TRTLLM_ENABLE_ATTENTION_NVFP4_OUTPUT",
                              "1") == "1":
            return False

        use_paged_context_fmha = (
            metadata.runtime_features.chunked_prefill
            or metadata.runtime_features.cache_reuse
            or metadata.runtime_features.has_speculative_draft_tokens
        ) if metadata.runtime_features else False

        # This is a workaround for https://nvbugs/5624818
        # Paged context FMHA is forced on SM90 for correctness
        if get_sm_version() == 90:
            use_paged_context_fmha = True

        return self.wrapper.is_nvfp4_output_kernel_available(
            tokens_per_block=metadata.tokens_per_block,
            attention_mask=attention_mask,
            use_paged_context_fmha=use_paged_context_fmha,
            is_mla_enable=self.is_mla_enable,
        )

    def get_quantize_output_dtype(
            self, use_nvfp4_output: bool) -> Optional[torch.dtype]:
        if use_nvfp4_output:
            # Use UINT8 as the container dtype for NVFP4.
            return torch.uint8
        elif (self.has_fp8_qdq or self.has_nvfp4 or self.has_fp8_block_wise
              or self.has_fp8_rowwise
              or self.has_w4a8_nvfp4_fp8) and (self.has_fp8_kv_cache
                                               or self.has_fp4_kv_cache):
            return torch.float8_e4m3fn
        return None

    def create_output(self, q, *, is_quantize_output: bool,
                      metadata: TrtllmAttentionMetadata,
                      attention_mask: AttentionMask, is_gen_only: bool,
                      **kwargs) -> List[torch.Tensor]:
        use_nvfp4_output = False
        out_dtype = None
        if is_quantize_output:
            use_nvfp4_output = self.use_nvfp4_output(metadata, attention_mask)
            out_dtype = self.get_quantize_output_dtype(use_nvfp4_output)

        return self.wrapper.create_output(q, out_dtype, use_nvfp4_output,
                                          is_gen_only)

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: TrtllmAttentionMetadata,
        output: Optional[torch.Tensor] = None,
        output_sf: Optional[torch.Tensor] = None,
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
        enable_attn_nvfp4_output: bool = True,
        attention_sinks: Optional[torch.Tensor] = None,
        chunked_prefill_buffer_batch_size: int = 1,
        cu_q_seqlens: Optional[torch.Tensor] = None,
        cu_kv_seqlens: Optional[torch.Tensor] = None,
        fmha_scheduler_counter: Optional[torch.Tensor] = None,
        mla_bmm1_scale: Optional[torch.Tensor] = None,
        mla_bmm2_scale: Optional[torch.Tensor] = None,
        quant_q_buffer: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Execute the attention operation.
        Args:
            q (torch.Tensor): Query tensor or QKV tensor.
            k (Optional[torch.Tensor]): Key tensor or None if QKV tensor is provided.
            v (Optional[torch.Tensor]): Value tensor or None if QKV tensor is provided.
            metadata (TrtllmAttentionMetadata): Metadata for the attention operation.
            output (Optional[torch.Tensor]): Output tensor to store the attention output.
            output_sf (Optional[torch.Tensor]): Output scale factor tensor for NVFP4.
            out_scale (Optional[torch.Tensor]): Scale factor tensor for quantizing output.
            out_scale_sf (Optional[torch.Tensor]): Global scale factor tensor for NVFP4 for quantizingoutput.
            kv_scales_sf (Optional[torch.Tensor]): KV scale factor tensor.
            kv_scales_sf_inv (Optional[torch.Tensor]): KV scale factor inverse tensor.
            attention_mask (AttentionMask): Attention mask.
            attention_input_type (AttentionInputType): Attention input type.
            latent_cache (Optional[torch.Tensor]): Latent cache tensor.
            q_pe (Optional[torch.Tensor]): Q position embedding tensor.
            mrope_config (Optional[dict]): Mrope configuration.
            attention_window_size (Optional[int]): Attention window size.
            softmax_stats_tensor (Optional[torch.Tensor]): Softmax statistics tensor.
            helix_position_offsets (Optional[torch.Tensor]): Helix position offsets tensor.
            attention_sinks (Optional[torch.Tensor]): Attention sinks tensor.
            chunked_prefill_buffer_batch_size (int): Chunked prefill buffer batch size.
        """
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

        # This is a workaround for https://nvbugs/5624818
        # Paged context FMHA is forced on SM90 for correctness
        if get_sm_version() == 90:
            use_paged_context_fmha = True

        if self.is_mla_enable:
            # Context MLA uses separate qkv instead of paged_context_fmha
            use_paged_context_fmha = False

        if output is None:
            # Output is not provided.
            is_gen_only = attention_input_type == AttentionInputType.generation_only
            outputs = self.create_output(
                q,
                is_quantize_output=out_scale is not None,
                metadata=metadata,
                attention_mask=attention_mask,
                use_paged_context_fmha=use_paged_context_fmha,
                is_mla_enable=self.is_mla_enable,
                is_gen_only=is_gen_only,
            )

            output = outputs[0]
            output_sf = outputs[1] if len(outputs) == 2 else None

        sparse_kv_indices, sparse_kv_offsets, sparse_attn_indices, sparse_attn_offsets = None, None, None, None
        sparse_attn_indices_block_size = 1
        skip_softmax_threshold_scale_factor_prefill = None
        skip_softmax_threshold_scale_factor_decode = None
        if self.sparse_attention_config is not None:
            if isinstance(self.sparse_attention_config,
                          SkipSoftmaxAttentionConfig):
                skip_softmax_threshold_scale_factor_prefill = self.sparse_attention_config.threshold_scale_factor_prefill
                skip_softmax_threshold_scale_factor_decode = self.sparse_attention_config.threshold_scale_factor_decode

            else:
                sparse_kv_indices, sparse_kv_offsets = self.sparse_kv_predict(
                    q, k, metadata, **kwargs)
                sparse_attn_indices, sparse_attn_offsets = self.sparse_attn_predict(
                    q, k, metadata, **kwargs)
                sparse_attn_indices_block_size = self.sparse_attention_config.get_indices_block_size(
                )

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
            host_kv_cache_pool_pointers=metadata.host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=metadata.host_kv_cache_pool_mapping,
            block_ids_per_seq=metadata.block_ids_per_seq,
            # re-enable it, if pass None to it, fp8 mla will encounter invalid cuda free issue.
            workspace=metadata.workspace
            if not metadata.is_cuda_graph else metadata.cuda_graph_workspace,
            cache_indirection=metadata.cache_indirection,
            kv_scale_orig_quant=self.kv_scale_orig_quant,
            kv_scale_quant_orig=self.kv_scale_quant_orig,
            out_scale=out_scale,
            out_scale_sf=out_scale_sf,
            kv_scales_sf=kv_scales_sf,
            kv_scales_sf_inv=kv_scales_sf_inv,
            use_nvfp4_output=output_sf
            is not None,  # NVFP4 output will setup output_sf tensor
            use_paged_context_fmha=use_paged_context_fmha,
            attention_input_type=attention_input_type,
            latent_cache=latent_cache,
            q_pe=q_pe,
            mrope_config=mrope_config,
            softmax_stats_tensor=softmax_stats_tensor,
            is_spec_decoding_enabled=metadata.is_spec_decoding_enabled,
            use_spec_decoding=metadata.use_spec_decoding,
            is_spec_dec_tree=metadata.is_spec_dec_tree,
            spec_decoding_position_offsets=metadata.
            spec_decoding_position_offsets,
            spec_decoding_packed_mask=metadata.spec_decoding_packed_mask,
            spec_decoding_generation_lengths=metadata.
            spec_decoding_generation_lengths,
            spec_decoding_bl_tree_mask_offset=metadata.
            spec_decoding_bl_tree_mask_offset,
            spec_decoding_bl_tree_mask=metadata.spec_decoding_bl_tree_mask,
            spec_bl_tree_first_sparse_mask_offset_kv=metadata.
            spec_bl_tree_first_sparse_mask_offset_kv,
            attention_sinks=attention_sinks,
            chunked_prefill_buffer_batch_size=chunked_prefill_buffer_batch_size,
            sparse_kv_indices=sparse_kv_indices,
            sparse_kv_offsets=sparse_kv_offsets,
            sparse_attn_indices=sparse_attn_indices,
            sparse_attn_offsets=sparse_attn_offsets,
            sparse_attn_indices_block_size=sparse_attn_indices_block_size,
            sparse_mla_topk=metadata.sparse_mla_topk if hasattr(
                metadata, 'sparse_mla_topk') else 0,
            skip_softmax_threshold_scale_factor_prefill=
            skip_softmax_threshold_scale_factor_prefill,
            skip_softmax_threshold_scale_factor_decode=
            skip_softmax_threshold_scale_factor_decode,
            helix_position_offsets=metadata.helix_position_offsets,
            helix_is_inactive_rank=metadata.helix_is_inactive_rank,
        )

        self.wrapper.run(q,
                         output,
                         output_sf,
                         k,
                         v,
                         is_fused_qkv=not metadata.is_cross and k is None,
                         update_kv_cache=not metadata.is_cross or k is not None,
                         attention_mask=attention_mask,
                         cu_q_seqlens=cu_q_seqlens,
                         cu_kv_seqlens=cu_kv_seqlens,
                         fmha_scheduler_counter=fmha_scheduler_counter,
                         mla_bmm1_scale=mla_bmm1_scale,
                         mla_bmm2_scale=mla_bmm2_scale,
                         quant_q_buffer=quant_q_buffer)

        if output_sf is None:
            return output
        else:
            return output, output_sf

    @classmethod
    def support_fused_rope(cls) -> bool:
        return True

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return True

    @classmethod
    def support_mla(cls) -> bool:
        return True

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

    def mla_rope_generation(
        self,
        fused_q: torch.Tensor,
        q_pe: torch.Tensor,
        latent_cache: torch.Tensor,
        metadata: TrtllmAttentionMetadata,
        cu_q_seqlens: torch.Tensor,
        cu_kv_seqlens: torch.Tensor,
        fmha_scheduler_counter: torch.Tensor,
        mla_bmm1_scale: torch.Tensor,
        mla_bmm2_scale: torch.Tensor,
        quant_q_buffer: torch.Tensor,
        out_scale: Optional[torch.Tensor] = None,
    ) -> None:
        """
            fused_q (torch.Tensor): The tensor to store the fused q, with shape (num_tokens, num_heads, kv_lora_rank + qk_rope_head_dim) on GPU.
            q_pe (torch.Tensor): The tensor to store the q_pe, with shape (num_tokens, num_heads, qk_rope_head_dim) on GPU.
            latent_cache (torch.Tensor): The tensor to store the latent cache, with shape (num_tokens, kv_lora_rank + qk_rope_head_dim) on GPU.
            cu_q_seqlens (torch.Tensor): The tensor to store the cu_q_seqlens, with shape (num_seqs + 1) on GPU.
            cu_kv_seqlens (torch.Tensor): The tensor to store the cu_kv_seqlens, with shape (num_seqs + 1) on GPU.
            fmha_scheduler_counter (torch.Tensor): The tensor to store the fmha_scheduler_counter, with shape (1) on GPU.
            mla_bmm1_scale (torch.Tensor): The tensor to store the mla_bmm1_scale, with shape (2) on GPU.
            mla_bmm2_scale (torch.Tensor): The tensor to store the mla_bmm2_scale, with shape (1) on GPU.
            quant_q_buffer (torch.Tensor): The tensor to store the quant_q_buffer, with shape (tokens, num_heads, kv_lora_rank + qk_rope_head_dim) on GPU.
            helix_position_offsets (torch.Tensor): The tensor to store the helix position offsets, with shape (num_tokens) on GPU.
            out_scale (torch.Tensor): The tensor to store the out_scale, with shape (1) on GPU.
        """

        assert self.is_mla_enable and self.mla_params is not None
        assert metadata.kv_cache_manager is not None
        sink_token_length = 0

        mla_tensor_params = [
            metadata.helix_position_offsets, metadata.helix_is_inactive_rank
        ]

        torch.ops.trtllm.mla_rope_generation(
            fused_q,
            q_pe,
            latent_cache,
            self.wrapper.rotary_cos_sin,
            cu_q_seqlens,
            cu_kv_seqlens,
            fmha_scheduler_counter,
            mla_bmm1_scale,
            mla_bmm2_scale,
            quant_q_buffer,
            metadata.kv_lens_cuda_runtime,  # sequence_length
            metadata.kv_lens_runtime,  # host_past_key_value_lengths
            metadata.prompt_lens_cpu_runtime,  # host_context_lengths,
            metadata.num_contexts,
            metadata.kv_cache_block_offsets,
            metadata.kv_cache_manager.kv_cache_pool_pointers,
            metadata.kv_cache_manager.kv_cache_pool_mapping,
            self.kv_scale_orig_quant,
            self.kv_scale_quant_orig,
            out_scale,
            metadata.block_ids_per_seq,
            mla_tensor_params,
            self.wrapper.predicted_tokens_per_seq,
            self.get_local_layer_idx(metadata),
            self.wrapper.num_heads,
            self.wrapper.num_kv_heads,
            self.wrapper.head_size,
            metadata.kv_cache_manager.tokens_per_block,
            metadata.max_seq_len,  # attention_window_size
            sink_token_length,
            metadata.beam_width,
            self.wrapper.quant_mode,
            self.wrapper.q_scaling,
            self.wrapper.q_lora_rank,
            self.wrapper.kv_lora_rank,
            self.wrapper.qk_nope_head_dim,
            self.wrapper.qk_rope_head_dim,
            self.wrapper.v_head_dim,
        )
