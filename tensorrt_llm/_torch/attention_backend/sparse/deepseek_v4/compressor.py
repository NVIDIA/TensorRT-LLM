from enum import IntEnum
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_llm._torch.attention_backend.interface import MLAParams, PositionalEmbeddingParams
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.modules.rotary_embedding import RotaryEmbedding

if TYPE_CHECKING:
    from .deepseek_v4 import DeepseekV4TrtllmAttentionMetadata


class KVCacheDtype(IntEnum):
    """KV cache dtype/layout preset (values match C++ cache_scale_type parameter).

    The store dtype and scale layout are implied by this value:
      - NONE:              keeps the input dtype (bf16/fp32, decided by the
                           caller's tensor element size).
      - FP8_PERTENSOR:     1 byte per value (FP8 E4M3) with implicit scale=1.
      - FP8_BLOCKWISE:     1 byte per value + 1 fp32 scale per 128 values.
      - MXFP4_BLOCKWISE:   packed FP4 (½ byte per value) + 1 UE8M0 byte per
                           32 values.

    Storage size in bytes per logical element is therefore::

        size_per_value = {
            NONE: elem_bytes,  # caller-side
            FP8_PERTENSOR: 1,
            FP8_BLOCKWISE: 1 + 4 / 128,  # data + fp32 scale
            MXFP4_BLOCKWISE: 0.5 + 1 / 32,  # nibble + ue8m0 byte
        }[kv_cache_dtype]
    """

    NONE = 0
    FP8_PERTENSOR = 1  # FP8 E4M3 with implicit scale=1
    FP8_BLOCKWISE = 2  # FP8 E4M3 with per-128 fp32 scales
    MXFP4_BLOCKWISE = 3  # packed FP4 E2M1 with per-32 UE8M0 scales


_KV_CACHE_DTYPE_MAP = {
    "default": KVCacheDtype.NONE,
    "bf16": KVCacheDtype.NONE,
    "fp8_pertensor": KVCacheDtype.FP8_PERTENSOR,
    "fp8_blockwise": KVCacheDtype.FP8_BLOCKWISE,
    "mxfp4": KVCacheDtype.MXFP4_BLOCKWISE,
}


def resolve_kv_cache_dtype(kv_cache_dtype: Union[str, KVCacheDtype]) -> KVCacheDtype:
    if isinstance(kv_cache_dtype, str):
        return _KV_CACHE_DTYPE_MAP[kv_cache_dtype]
    return kv_cache_dtype


class Compressor(nn.Module):
    """KV compressor using Triton kernels with paged memory management.

    Args:
        mla_params: MLA parameters containing hidden_size and head dimensions
        layer_idx: Layer index for cache management
        compress_ratio: Compression ratio (e.g., 4 compresses 4 tokens into 1)
        norm_eps: RMSNorm epsilon
        skip_create_weights_in_init: Whether to skip weight initialization
        pos_embd_params: Positional embedding parameters for RoPE
        dtype: Data type for computation
        kv_cache_dtype: Cache preset string or KVCacheDtype.
        rotate_activation: Whether to apply Hadamard transform in postprocessing (False to skip)
    """

    def __init__(
        self,
        mla_params: MLAParams,
        layer_idx: int,
        compress_ratio: int,
        norm_eps: float,
        skip_create_weights_in_init: bool,
        pos_embd_params: PositionalEmbeddingParams,
        dtype: Optional[torch.dtype] = torch.bfloat16,
        kv_cache_dtype: Union[str, KVCacheDtype] = KVCacheDtype.NONE,
        is_indexer: bool = False,
        rotate_activation: bool = False,
    ):
        super().__init__()
        # Dimensions
        self.dim = mla_params.hidden_size
        self.head_dim = mla_params.qk_rope_head_dim + mla_params.qk_nope_head_dim
        self.rope_head_dim = mla_params.qk_rope_head_dim
        self.nope_head_dim = mla_params.qk_nope_head_dim

        # Compression config
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.state_dim = 2 * self.head_dim if self.overlap else self.head_dim

        # Cache config
        self.layer_idx = layer_idx
        self.kv_cache_dtype: KVCacheDtype = resolve_kv_cache_dtype(kv_cache_dtype)
        self.is_indexer = is_indexer
        self.rotate_activation = rotate_activation

        # Modules
        self.wkv_gate = Linear(
            self.dim,
            self.state_dim * 2,
            bias=False,
            dtype=dtype,
            quant_config=None,
            skip_create_weights_in_init=skip_create_weights_in_init,
            use_custom_cublas_mm=True,
        )
        self.norm = RMSNorm(hidden_size=self.head_dim, eps=norm_eps, dtype=dtype)
        self.rotary_emb = RotaryEmbedding(
            pos_embd_params.rope,
            head_dim=self.rope_head_dim,
            is_neox=pos_embd_params.is_neox,
        )

        # Learnable absolute positional encoding for compression
        self.ape = nn.Parameter(torch.empty(compress_ratio, self.state_dim, dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        metadata: "DeepseekV4TrtllmAttentionMetadata",
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass for paged KV compression.

        Args:
            x: Input tensor [num_tokens, dim]
            metadata: Attention metadata with cache info

        Returns:
            (kv_data, scale) tuple:
            - default / fp8_pertensor main compressor: (kv_comp, None)
            - default indexer:                         (kv_out, None)  bf16
            - fp8_blockwise indexer:                   (fp8_output, fp32 scale)
            - mxfp4 indexer:                           (fp4_output, ue8m0 scale)
            - no compressed tokens:                    (None, None)
        """
        # Import at runtime to avoid circular dependency
        from .deepseek_v4 import DeepseekV4AttentionType

        # Extract metadata
        num_contexts = metadata.num_contexts
        num_generations = metadata.num_generations
        num_ctx_tokens = metadata.num_ctx_tokens
        bsz = num_contexts + num_generations

        # Determine attention types based on whether this is an indexer compressor
        if self.is_indexer:
            compress_type = DeepseekV4AttentionType.INDEXER_COMPRESS
            state_type = DeepseekV4AttentionType.INDEXER_COMPRESSOR_STATE
            score_type = DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE
        else:
            compress_type = DeepseekV4AttentionType.COMPRESS
            state_type = DeepseekV4AttentionType.COMPRESSOR_STATE
            score_type = DeepseekV4AttentionType.COMPRESSOR_SCORE

        # Get cache buffers
        kv_cache = metadata.kv_cache_manager.get_buffers(self.layer_idx, compress_type)
        paged_kv_state = metadata.kv_cache_manager.get_buffers(self.layer_idx, state_type)
        paged_score_state = metadata.kv_cache_manager.get_buffers(self.layer_idx, score_type)

        # Get block tables
        block_table = metadata.block_tables[(self.compress_ratio, compress_type)]
        block_table_kv_state = metadata.block_tables[(self.compress_ratio, state_type)]
        block_table_score_state = metadata.block_tables[(self.compress_ratio, score_type)]

        # Get tokens_per_block from cache manager
        # state_tokens_per_block: for state/score caches (used in compress kernels)
        # compress_tokens_per_block: for compressed KV cache (used in scatter)
        state_tokens_per_block = metadata.kv_cache_manager.tokens_per_block
        compress_tokens_per_block = metadata.kv_cache_manager.compressed_block_sizes[self.layer_idx]

        # Get compression metadata
        cu_new_comp_kv = metadata.cu_new_comp_kv_cuda[self.compress_ratio]
        kv_lens = metadata.kv_lens_cuda_runtime
        total_num_comp_tokens = metadata.num_total_compressed_tokens[self.compress_ratio]
        num_comp_tokens = metadata.new_comp_kv_lens_cuda[self.compress_ratio][:bsz]
        max_ctx_comp_kv_lens = metadata.max_ctx_compressed_tokens[self.compress_ratio]

        # Project input to KV and score in the checkpoint dtype. The compressor
        # kernels accept bf16 or fp32 kv_score and convert values to fp32
        # internally for state updates and online-softmax accumulation.
        kv_score = F.linear(x.to(self.wkv_gate.weight.dtype), self.wkv_gate.weight)

        # Allocate output buffer
        kv_comp = torch.empty(total_num_comp_tokens, self.head_dim, device=x.device, dtype=x.dtype)

        # Run compression kernels
        if num_contexts > 0:
            torch.ops.trtllm.compressor_prefill_reduction(
                kv_score[:num_ctx_tokens],
                self.ape,
                paged_kv_state,
                paged_score_state,
                block_table_kv_state[:num_contexts],
                block_table_score_state[:num_contexts],
                kv_comp,
                kv_lens[:num_contexts],
                metadata.cached_token_lens_cuda[:num_contexts],
                metadata.cu_seq_lens_cuda,
                cu_new_comp_kv[: num_contexts + 1],
                num_contexts,
                state_tokens_per_block,
                self.head_dim,
                self.compress_ratio,
                max_ctx_comp_kv_lens,
            )

        if num_generations > 0:
            gen_kv_lens = kv_lens[num_contexts:]
            next_n = metadata.num_gen_tokens_per_seq
            # Pass full kv_score (not sliced) with the generation portion of
            # cu_seq_lens so the kernel reads at the correct absolute offsets.
            torch.ops.trtllm.compressor_paged_kv_compress(
                kv_score,
                self.ape,
                paged_kv_state,
                paged_score_state,
                block_table_kv_state[num_contexts:],
                block_table_score_state[num_contexts:],
                kv_comp,
                gen_kv_lens,
                metadata.cu_seq_lens_cuda[num_contexts:],
                cu_new_comp_kv[num_contexts:],
                num_generations,
                state_tokens_per_block,
                self.head_dim,
                self.compress_ratio,
                next_n,
            )

        # Scatter to cache with appropriate quantization (all modes fused)
        start_pos = metadata.past_kv_lens_cuda[self.compress_ratio][:bsz]
        total_tokens = kv_comp.shape[0]

        # Allocate optional returned postprocess buffers for indexer paths.
        kv_out = None
        quant_output = None
        scale_output = None
        if self.is_indexer:
            if self.kv_cache_dtype == KVCacheDtype.NONE:
                kv_out = torch.empty_like(kv_comp)
            elif self.kv_cache_dtype == KVCacheDtype.FP8_BLOCKWISE:
                num_scale_blocks = self.head_dim // 128
                quant_output = torch.empty(
                    total_tokens, self.head_dim, dtype=torch.uint8, device=kv_comp.device
                )
                scale_output = torch.empty(
                    total_tokens, num_scale_blocks, dtype=torch.float32, device=kv_comp.device
                )
            elif self.kv_cache_dtype == KVCacheDtype.MXFP4_BLOCKWISE:
                num_scale_blocks = self.head_dim // 32
                quant_output = torch.empty(
                    total_tokens, self.head_dim // 2, dtype=torch.uint8, device=kv_comp.device
                )
                scale_output = torch.empty(
                    total_tokens, num_scale_blocks, dtype=torch.uint8, device=kv_comp.device
                )

        position_ids = metadata.compressed_position_ids_cuda[self.compress_ratio][:total_tokens]
        compressed_mask = metadata.compressed_mask_cuda[self.compress_ratio][:total_tokens]

        # Fused postprocess + scatter: RMSNorm + RoPE + Hadamard + paged cache write
        torch.ops.trtllm.compressor_postprocess_scatter(
            kv_comp,
            kv_out,
            self.norm.weight,
            self.norm.variance_epsilon,
            self.rotary_emb.rotary_cos_sin,
            position_ids,
            self.nope_head_dim,
            self.rope_head_dim,
            kv_cache,
            num_comp_tokens,
            cu_new_comp_kv,
            start_pos,
            block_table,
            compressed_mask,
            compress_tokens_per_block,
            int(self.kv_cache_dtype),
            self.rotate_activation,
            quant_output,
            scale_output,
        )

        if quant_output is not None:
            if self.kv_cache_dtype == KVCacheDtype.MXFP4_BLOCKWISE:
                return quant_output.view(torch.float4_e2m1fn_x2), scale_output
            return quant_output.view(torch.float8_e4m3fn), scale_output
        if kv_out is not None:
            return kv_out, None
        return kv_comp, None
