import math
from dataclasses import dataclass
from typing import Optional

import torch

from tensorrt_llm._torch.attention_backend.interface import (
    AttentionBackend, AttentionMask, AttentionMetadata, MLAParams,
    PositionalEmbeddingParams, PredefinedAttentionMask, RopeParams)
from tensorrt_llm.functional import (AttentionMaskType, RopeEmbeddingUtils,
                                     RotaryScalingType)
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig


@dataclass(kw_only=True, init=False)
class TrtllmAttentionWrapper:
    sequence_length: torch.Tensor
    host_past_key_value_lengths: torch.Tensor
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
    is_mla_enable: Optional[bool]
    q_b_proj: Optional[torch.Tensor]
    kv_b_proj: Optional[torch.Tensor]
    k_b_proj_trans: Optional[torch.Tensor]
    q_lora_rank: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    v_head_dim: int
    kwargs: dict

    def __init__(
            self,
            layer_idx: int,
            num_heads: int,
            head_size: int,
            num_kv_heads: Optional[int] = None,
            pos_embd_params: Optional[PositionalEmbeddingParams] = None,
            quant_config: Optional[QuantConfig] = None,
            is_mla_enable: Optional[bool] = False,
            mla_params: Optional[MLAParams] = MLAParams(),
            **kwargs,
    ):
        """
        Initialize the attention wrapper.
        Args:
            layer_idx (int): The index of the attention layer in the model.
            num_heads (int): The number of query heads.
            head_dim (int): The size of each attention head (hidden_size // num_heads).
            num_kv_heads (int): The number of kv heads. Defaults to num_heads if None.
            pos_embd_params (PositionalEmbeddingParams): Optional parameters defining how positional embedding should be applied.
            quant_config (QuantConfig): Optional quantization configuration. If None, no quantization is applied.
        """
        if pos_embd_params is not None:
            rope_params = pos_embd_params.rope
        else:
            self.rotary_inv_freq = None
            self.rotary_cos_sin = None
            rope_params = RopeParams()

        self.is_mla_enable = is_mla_enable
        self.q_lora_rank = mla_params.q_lora_rank
        self.kv_lora_rank = mla_params.kv_lora_rank
        self.qk_nope_head_dim = mla_params.qk_nope_head_dim
        self.qk_rope_head_dim = mla_params.qk_rope_head_dim
        self.v_head_dim = mla_params.v_head_dim
        self.q_scaling = 1.0

        if is_mla_enable:
            self.rotary_embedding_dim = 0

            def yarn_get_mscale(scale=1, mscale=1):
                if scale <= 1:
                    return 1.0
                return 0.1 * mscale * math.log(scale) + 1.0

            mscale_all_dim = rope_params.mscale_all_dim
            scaling_factor = rope_params.scale
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.q_scaling = 1.0 / (mscale * mscale)

            rope_cos_sin = RopeEmbeddingUtils.create_sinusoidal_positions_for_deepseek_attention_plugin(
                rope_params.max_positions,
                self.qk_rope_head_dim,
                rope_params.theta,
                rope_params.scale,
                rope_params.original_max_positions,
                rope_params.beta_fast,
                rope_params.beta_slow,
                rope_params.mscale,
                rope_params.mscale_all_dim,
            )
            self.rotary_cos_sin = torch.tensor(rope_cos_sin,
                                               dtype=torch.float32,
                                               device="cuda")
            self.rotary_inv_freq = None
            self.rotary_embedding_scale_type = RotaryScalingType.none
            self.rotary_embedding_scale = 1.0
        else:
            self.rotary_inv_freq, self.rotary_cos_sin = rope_params.create_rope_const_params(
            )
            self.rotary_embedding_dim = rope_params.dim
            self.rotary_embedding_scale_type = int(rope_params.scale_type)
            self.rotary_embedding_scale = rope_params.scale

        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_size = head_size
        quant_config = quant_config or QuantConfig()
        self.quant_mode = int(quant_config.layer_quant_mode)
        self.position_embedding_type = int(
            pos_embd_params.type) if pos_embd_params is not None else 0
        self.rotary_embedding_base = rope_params.theta
        self.rotary_embedding_short_m_scale = rope_params.short_m_scale
        self.rotary_embedding_long_m_scale = rope_params.long_m_scale
        self.rotary_embedding_max_positions = rope_params.max_positions
        self.rotary_embedding_original_max_positions = rope_params.original_max_positions
        self.kwargs = {}
        self.kwargs.update(kwargs)

    def plan(
        self,
        *,
        tokens_per_block: int,
        max_num_requests: int,
        max_context_length: int,
        attention_window_size: Optional[int] = None,
        sink_token_length: int = 0,
        beam_width: int = 1,
        sequence_length: torch.Tensor,
        host_past_key_value_lengths: torch.Tensor,
        context_lengths: torch.Tensor,
        host_context_lengths: torch.Tensor,
        host_request_types: torch.Tensor,
        kv_cache_block_offsets: torch.Tensor,
        host_kv_cache_block_offsets: torch.Tensor,
        host_kv_cache_pool_pointers: torch.Tensor,
        host_kv_cache_pool_mapping: torch.Tensor,
        workspace: Optional[torch.Tensor] = None,
        cache_indirection: Optional[torch.Tensor] = None,
        kv_scale_orig_quant: Optional[torch.Tensor] = None,
        kv_scale_quant_orig: Optional[torch.Tensor] = None,
        out_scale: Optional[torch.Tensor] = None,
        use_paged_context_fmha: bool = False,
        q_b_proj: Optional[torch.Tensor] = None,
        kv_b_proj: Optional[torch.Tensor] = None,
        k_b_proj_trans: Optional[torch.Tensor] = None,
        q_b_proj_scale: Optional[torch.Tensor] = None,
        kv_b_proj_scale: Optional[torch.Tensor] = None,
        k_b_proj_trans_scale: Optional[torch.Tensor] = None,
        is_fp8_block_scaling_enabled: bool = False,
        **kwargs,
    ):
        """
        Plan the attention operation.
        Args:
            tokens_per_block (int): Token number per KV cache block.
            max_num_requests (int): Max request number per batch.
            max_context_length (int): Max context length per context-phase sequence.
            attention_window_size (int): Max token number attended in windowed attention.
            sink_token_length (int): Sink token number in StreamingLLM.
            beam_width (int): Beam width in beam search.
            sequence_length (torch.Tensor): The length of each sequence with shape (batch_size) on GPU.
            host_past_key_value_lengths (torch.Tensor): Same as sequence_length, but on CPU.
            context_lengths (torch.Tensor): The context-phase sequence length of each request with shape (batch_size) on GPU.
            host_context_lengths (torch.Tensor): Same as context_lengths, but on CPU.
            host_request_types (torch.Tensor): The tensor that indicates whether a request is in context or generation phase, with shape (batch_size) on CPU.
            kv_cache_block_offsets (torch.Tensor): The offsets to the blocks inside KV cache pools on GPU, its shape is (num_pools, max_batch_size * max_beam_width, 2, max_blocks_per_sequence), one for each block.
            host_kv_cache_block_offsets (torch.Tensor): Same as kv_cache_block_offsets, but on CPU.
            host_kv_cache_pool_pointers (torch.Tensor): The pointers to the KV cache pools on CPU, its shape is (num_pools, 2), one for primary pool in GPU memory, one for secondary pool in CPU memory.
            host_kv_cache_pool_mapping (torch.Tensor): The index of the pool used by each attention layer on CPU, its shape is (num_local_attention_layers). The local attention layers mean all attention layers in the current PP stage in the pipeline parallelism case.
            workspace (torch.Tensor): An optional workspace tensor on GPU.
            cache_indirection (torch.Tensor): A tensor for beam search on GPU, its shape is (batch_size, beam_width, max_seqlen), for a sequence si, a beam bi and a token ti, the element cache_indirection[si][bi][ti] is an integer between 0 and beam_width-1 that indicates which path in the beam to read the K and V elements from in the KV cache.
            kv_scale_orig_quant (torch.Tensor): The tensor to store the scaling factor for quantization to INT8/FP8 in the KV cache, with shape (1) on GPU.
            kv_scale_quant_orig (torch.Tensor): The tensor to store the scaling factor for dequantization from INT8/FP8 in the KV cache, with shape (1) on GPU.
            out_scale (torch.Tensor): The tensor to store the scaling factor to quantize output, with shape (1) on GPU.
            use_paged_context_fmha (bool): Sets the mPagedContextFMHA attribute in the op runner.
        """
        self.tokens_per_block = tokens_per_block
        self.max_num_requests = max_num_requests
        self.max_context_length = max_context_length
        self.attention_window_size = attention_window_size or max_context_length
        self.sink_token_length = sink_token_length
        self.beam_width = beam_width
        self.sequence_length = sequence_length
        self.host_past_key_value_lengths = host_past_key_value_lengths
        self.context_lengths = context_lengths
        self.host_context_lengths = host_context_lengths
        self.host_request_types = host_request_types
        self.kv_cache_block_offsets = kv_cache_block_offsets
        self.host_kv_cache_block_offsets = host_kv_cache_block_offsets
        self.host_kv_cache_pool_pointers = host_kv_cache_pool_pointers
        self.host_kv_cache_pool_mapping = host_kv_cache_pool_mapping
        self.workspace = workspace
        self.cache_indirection = cache_indirection
        self.kv_scale_orig_quant = kv_scale_orig_quant
        self.kv_scale_quant_orig = kv_scale_quant_orig
        self.out_scale = out_scale
        self.use_paged_context_fmha = use_paged_context_fmha
        self.q_b_proj = q_b_proj
        self.kv_b_proj = kv_b_proj
        self.k_b_proj_trans = k_b_proj_trans
        self.q_b_proj_scale = q_b_proj_scale
        self.kv_b_proj_scale = kv_b_proj_scale
        self.k_b_proj_trans_scale = k_b_proj_trans_scale
        self.is_fp8_block_scaling_enabled = is_fp8_block_scaling_enabled
        self.kwargs.update(kwargs)

    def run(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
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
            assert is_fused_qkv
            qkv_hidden_size = self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim
            assert q.shape[1] == qkv_hidden_size
            num_tokens = q.shape[0]
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

        output = torch.ops.trtllm.attention(
            q,
            k,
            v,
            out_dtype,
            self.workspace,
            self.sequence_length,
            self.host_past_key_value_lengths,
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
            self.out_scale,
            self.rotary_inv_freq,
            self.rotary_cos_sin,
            self.q_b_proj,
            self.kv_b_proj,
            self.k_b_proj_trans,
            self.q_b_proj_scale,
            self.kv_b_proj_scale,
            self.k_b_proj_trans_scale,
            is_fused_qkv,
            update_kv_cache,
            self.layer_idx,
            self.num_heads,
            1 if self.is_mla_enable else self.num_kv_heads,
            self.kv_lora_rank +
            self.qk_rope_head_dim if self.is_mla_enable else self.head_size,
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
            self.rotary_embedding_scale,
            self.rotary_embedding_short_m_scale,
            self.rotary_embedding_long_m_scale,
            self.rotary_embedding_max_positions,
            self.rotary_embedding_original_max_positions,
            self.use_paged_context_fmha,
            self.is_mla_enable,
            self.q_lora_rank,
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.is_fp8_block_scaling_enabled,
        )
        return output


@dataclass(kw_only=True)
class TrtllmAttentionMetadata(AttentionMetadata):
    workspace: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.prompt_lens_cuda = torch.empty(
            (self.max_num_requests, ),
            device='cuda',
            dtype=torch.int,
        )
        self.prompt_lens_cpu = torch.empty_like(
            self.prompt_lens_cuda,
            device='cpu',
            pin_memory=True,
        )
        self.kv_lens_cuda = torch.empty_like(self.prompt_lens_cuda)
        self.kv_lens = torch.empty_like(self.kv_lens_cuda,
                                        device='cpu',
                                        pin_memory=True)
        self.host_request_types = torch.empty_like(self.prompt_lens_cpu)
        # For debugging, can use it to call the wrapper's plan function
        if self.workspace is None:
            self.workspace = torch.empty(
                (0, ),
                device='cuda',
                dtype=torch.int8,
            )
        if self.kv_cache_manager is not None:
            self.kv_cache_block_offsets = torch.empty(
                [
                    self.kv_cache_manager.num_pools, self.max_num_requests, 2,
                    self.kv_cache_manager.max_blocks_per_seq
                ],
                dtype=torch.int32,
                device='cuda',
            )
            self.host_kv_cache_block_offsets = torch.empty_like(
                self.kv_cache_block_offsets,
                device='cpu',
                pin_memory=True,
            )

    def prepare(self) -> None:
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
        )

        # number of tokens needed in the kv cache for each sequence after the next pass
        kv_lens = cached_token_lens + self.seq_lens_kv
        self.kv_lens[:self.num_seqs].copy_(kv_lens)
        self.kv_lens_cuda[:self.num_seqs].copy_(self.kv_lens[:self.num_seqs],
                                                non_blocking=True)
        self.host_request_types[:self.num_contexts].fill_(0)
        self.host_request_types[self.num_contexts:self.num_seqs].fill_(1)

        # kv block offsets
        assert self.request_ids is not None
        if self.kv_cache_manager is not None:
            self.kv_cache_manager.impl.copy_batch_block_offsets(
                self.host_kv_cache_block_offsets, self.request_ids)
            self.kv_cache_block_offsets[:, :self.num_seqs].copy_(
                self.host_kv_cache_block_offsets[:, :self.num_seqs],
                non_blocking=True)


class TrtllmAttention(AttentionBackend[TrtllmAttentionMetadata]):

    Metadata = TrtllmAttentionMetadata

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        pos_embd_params: Optional[PositionalEmbeddingParams] = None,
        quant_config: Optional[QuantConfig] = None,
        is_mla_enable: Optional[bool] = False,
        mla_params: Optional[MLAParams] = None,
    ):
        """
        Initialize the backend.
        Args:
            layer_idx (int): The index of the attention layer in the model.
            num_heads (int): The number of query heads.
            head_dim (int): The size of each attention head (hidden_size // num_heads).
            num_kv_heads (int): The number of kv heads. Defaults to num_heads if None.
            pos_embd_params (PositionalEmbeddingParams): Optional parameters defining how positional embedding should be applied.
                                                         If None, positional embedding should be applied by the model before calling the backend.
                                                         Otherwise, the backend is in-charge of applying positional embedding and may cache K without embedding it first.
            quant_config (QuantConfig): Optional quantization configuration. If None, no quantization is applied.
        """
        super().__init__(layer_idx, num_heads, head_dim, num_kv_heads,
                         quant_config)
        self.wrapper = TrtllmAttentionWrapper(
            layer_idx,
            num_heads,
            head_dim,
            num_kv_heads,
            pos_embd_params=pos_embd_params,
            quant_config=quant_config,
            is_mla_enable=is_mla_enable,
            mla_params=mla_params,
        )

        self.is_mla_enable = is_mla_enable
        self.mla_params = mla_params

        self.kv_cache_scaling_factor = torch.tensor(
            [1.0],
            dtype=torch.float32,
            device='cuda',
        )
        self.kv_scale_quant_orig = self.kv_cache_scaling_factor
        self.kv_scale_orig_quant = 1.0 / self.kv_scale_quant_orig
        self.has_fp8_qdq = self.has_fp8_kv_cache = self.has_nvfp4 = False
        self.is_fp8_block_scaling_enabled = False
        if self.quant_config:
            self.has_fp8_qdq = self.quant_config.layer_quant_mode.has_fp8_qdq()
            self.has_nvfp4 = self.quant_config.layer_quant_mode.has_nvfp4()
            self.has_fp8_kv_cache = self.quant_config.layer_quant_mode.has_fp8_kv_cache(
            )
            self.has_nvfp4 = self.quant_config.layer_quant_mode.has_nvfp4()
            self.is_fp8_block_scaling_enabled = self.quant_config.layer_quant_mode.has_fp8_block_scales(
            )

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: TrtllmAttentionMetadata,
        out_scale: Optional[torch.Tensor] = None,
        *,
        attention_mask: AttentionMask = PredefinedAttentionMask.CAUSAL,
        q_b_proj: Optional[torch.Tensor] = None,
        kv_b_proj: Optional[torch.Tensor] = None,
        k_b_proj_trans: Optional[torch.Tensor] = None,
        q_b_proj_scale: Optional[torch.Tensor] = None,
        kv_b_proj_scale: Optional[torch.Tensor] = None,
        k_b_proj_trans_scale: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # This is only for memory estimation for now.
        # NOTE: this method is not accurate while it works for most scenario.
        if metadata is None or metadata.kv_cache_manager is None:
            if not self.is_mla_enable:
                num_heads = self.wrapper.num_heads
                num_kv_heads = self.wrapper.num_kv_heads
                head_dim = q.shape[-1]
                q_size = int(head_dim * num_heads /
                             (num_heads + 2 * num_kv_heads))
                kv_size = int(head_dim * num_kv_heads /
                              (num_heads + 2 * num_kv_heads))
                q, k, v = q.split([q_size, kv_size, kv_size], dim=-1)
                head_dim = int(q_size / num_heads)
                assert head_dim == self.wrapper.head_size, "given head dim is not equal with head dim in q."
                q = q.reshape(1, -1, num_heads, head_dim).contiguous()
                k = k.reshape(1, -1, num_kv_heads, head_dim).contiguous()
                v = v.reshape(1, -1, num_kv_heads, head_dim).contiguous()
                return AttentionBackend.dummy_forward(q, k, v)
            else:
                # now the dummy_forward doesn't support mla, it's not accurate
                num_heads = self.wrapper.num_heads
                num_kv_heads = self.wrapper.num_kv_heads
                head_dim = self.mla_params.qk_nope_head_dim + self.mla_params.qk_rope_head_dim
                seq_len = int(q.numel() / q.shape[-1])
                q = torch.randn(1,
                                seq_len,
                                num_heads,
                                head_dim,
                                dtype=q.dtype,
                                device=q.device)
                k = torch.randn(1,
                                seq_len,
                                num_kv_heads,
                                head_dim,
                                dtype=q.dtype,
                                device=q.device)
                v = torch.randn(1,
                                seq_len,
                                num_kv_heads,
                                head_dim,
                                dtype=q.dtype,
                                device=q.device)
                return AttentionBackend.dummy_forward(
                    q, k, v)[..., :num_kv_heads *
                             self.mla_params.v_head_dim].contiguous()

        assert isinstance(
            metadata,
            TrtllmAttentionMetadata,
        )
        assert not metadata.is_cross, "TRT-LLM Attention does not support cross attention yet."

        use_paged_context_fmha = (metadata.runtime_features.chunked_prefill
                                  or metadata.runtime_features.cache_reuse)

        if use_paged_context_fmha and self.has_fp8_kv_cache:
            # NOTE: W4A8_AWQ can be included too, exclude for now since
            # we don't use int4 in PyTorch
            if not (self.has_fp8_qdq or self.has_nvfp4):
                raise RuntimeError(
                    "When FP8 KV cache is being used, paged context FMHA cannot be used without "
                    "FP8 attention.")

        num_seqs = metadata.num_seqs
        self.wrapper.plan(
            tokens_per_block=metadata.kv_cache_manager.tokens_per_block,
            max_num_requests=metadata.kv_cache_manager.max_batch_size,
            max_context_length=metadata.kv_cache_manager.max_seq_len,
            attention_window_size=None,
            sink_token_length=0,
            beam_width=1,
            sequence_length=metadata.kv_lens_cuda[:num_seqs],
            host_past_key_value_lengths=metadata.kv_lens[:num_seqs],
            context_lengths=metadata.prompt_lens_cuda[:num_seqs],
            host_context_lengths=metadata.prompt_lens_cpu[:num_seqs],
            host_request_types=metadata.host_request_types[:num_seqs],
            kv_cache_block_offsets=metadata.kv_cache_block_offsets,
            host_kv_cache_block_offsets=metadata.host_kv_cache_block_offsets,
            host_kv_cache_pool_pointers=metadata.kv_cache_manager.
            kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=metadata.kv_cache_manager.
            kv_cache_pool_mapping,
            workspace=None,
            cache_indirection=None,
            kv_scale_orig_quant=self.kv_scale_orig_quant,
            kv_scale_quant_orig=self.kv_scale_quant_orig,
            out_scale=out_scale,
            use_paged_context_fmha=use_paged_context_fmha,
            q_b_proj=q_b_proj,
            kv_b_proj=kv_b_proj,
            k_b_proj_trans=k_b_proj_trans,
            q_b_proj_scale=q_b_proj_scale,
            kv_b_proj_scale=kv_b_proj_scale,
            k_b_proj_trans_scale=k_b_proj_trans_scale,
            is_fp8_block_scaling_enabled=self.is_fp8_block_scaling_enabled,
        )
        out_dtype = None
        if out_scale is not None:
            if (self.has_fp8_qdq or self.has_nvfp4) and self.has_fp8_kv_cache:
                # TODO(qijun): revisit fp8_context_fmha logic
                out_dtype = torch.float8_e4m3fn

        output = self.wrapper.run(q,
                                  k,
                                  v,
                                  out_dtype=out_dtype,
                                  is_fused_qkv=not metadata.is_cross
                                  and k is None,
                                  update_kv_cache=not metadata.is_cross
                                  or k is not None,
                                  attention_mask=attention_mask)
        return output
