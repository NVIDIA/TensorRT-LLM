import math
from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional, Set, Tuple

import torch

from tensorrt_llm._torch.attention_backend.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    MLAParams,
    PositionalEmbeddingParams,
)
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention, TrtllmAttentionMetadata
from tensorrt_llm._torch.modules.linear import Linear  # noqa: E402  (avoid cycle)
from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel
from tensorrt_llm._torch.modules.rotary_embedding import RotaryEmbedding
from tensorrt_llm._torch.utils import maybe_compile
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.utils import fp8_utils

from ..dsa import DSAtrtllmAttentionMetadata, Indexer, rotate_activation
from ..kernel import deepseek_v4_local_to_global_indices
from .compressor import Compressor, KVCacheDtype, resolve_kv_cache_dtype

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import SparseAttentionConfig

DEEPSEEK_V4_SPARSE_RATIO = 4
DEEPSEEK_V4_OVERLAP_COMPRESSOR_RATIO = 4


class DeepseekV4AttentionType(Enum):
    SWA = 0
    COMPRESS = 1
    COMPRESSOR_STATE = 2
    COMPRESSOR_SCORE = 3
    INDEXER_COMPRESS = 4
    INDEXER_COMPRESSOR_STATE = 5
    INDEXER_COMPRESSOR_SCORE = 6


def is_overlap_compressor(compress_ratio: int) -> bool:
    """
    Check if the compressor of the given layer is working in the overlap mode.
    """
    return compress_ratio == DEEPSEEK_V4_OVERLAP_COMPRESSOR_RATIO


def is_sparse_layer(compress_ratio: int) -> bool:
    """
    Check if the given layer is a sparse layer.
    """
    return compress_ratio == DEEPSEEK_V4_SPARSE_RATIO


def is_compress_layer(compress_ratio: int) -> bool:
    """
    Check if the given layer is a compress layer.
    """
    return compress_ratio > 1


def compress_ratio_has_attention(compress_ratio: int, attn_type: DeepseekV4AttentionType) -> bool:
    """
    Check if the given compress ratio has the given attention type.
    """
    is_sparse = is_sparse_layer(compress_ratio)
    is_compress = is_compress_layer(compress_ratio)

    if attn_type == DeepseekV4AttentionType.SWA:
        return True
    elif attn_type == DeepseekV4AttentionType.COMPRESS:
        return is_compress
    elif attn_type == DeepseekV4AttentionType.COMPRESSOR_STATE:
        return is_compress
    elif attn_type == DeepseekV4AttentionType.COMPRESSOR_SCORE:
        return is_compress
    elif attn_type == DeepseekV4AttentionType.INDEXER_COMPRESS:
        return is_sparse
    elif attn_type == DeepseekV4AttentionType.INDEXER_COMPRESSOR_STATE:
        return is_sparse
    elif attn_type == DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE:
        return is_sparse


def get_attn_dim(
    head_dim: int, index_head_dim: int, compress_ratio: int, attn_type: DeepseekV4AttentionType
) -> int:
    """
    Get the dimension of the attention type for a specific layer.
    """
    state_factor = 2 if is_overlap_compressor(compress_ratio) else 1
    if attn_type == DeepseekV4AttentionType.SWA:
        return head_dim
    elif attn_type == DeepseekV4AttentionType.COMPRESS:
        return head_dim
    elif attn_type == DeepseekV4AttentionType.COMPRESSOR_STATE:
        return state_factor * head_dim
    elif attn_type == DeepseekV4AttentionType.COMPRESSOR_SCORE:
        return state_factor * head_dim
    elif attn_type == DeepseekV4AttentionType.INDEXER_COMPRESS:
        return index_head_dim
    elif attn_type == DeepseekV4AttentionType.INDEXER_COMPRESSOR_STATE:
        return state_factor * index_head_dim
    elif attn_type == DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE:
        return state_factor * index_head_dim


def get_token_bytes(
    head_dim: int,
    index_head_dim: int,
    compress_ratio: int,
    attn_type: DeepseekV4AttentionType,
    has_fp8_kv_cache: bool,
    indexer_k_cache_dtype: str = "fp8_blockwise",
) -> int:
    """
    Get the token bytes for a specific layer and attention type.

    Args:
        head_dim: The head dimension
        index_head_dim: The index head dimension
        compress_ratio: The compress ratio
        attn_type: The attention type
        has_fp8_kv_cache: Whether the KV cache uses FP8 quantization
        indexer_k_cache_dtype: Indexer compressor cache preset string
            (controls INDEXER_COMPRESS dtype + scale layout independently
            from has_fp8_kv_cache, which controls the main attention path).

    Returns:
        The number of bytes per token, including scaling factor
    """
    if not compress_ratio_has_attention(compress_ratio, attn_type):
        raise ValueError(
            f"Layer with compress ratio {compress_ratio} does not have attention type {attn_type}"
        )

    attn_dim = get_attn_dim(head_dim, index_head_dim, compress_ratio, attn_type)

    # Default dtype is bfloat16 (2 bytes), or fp8 (1 byte) when FP8 kv cache is enabled
    dtype_bytes = 1 if has_fp8_kv_cache else 2
    # (indexer) compressor state and score always use float32
    if attn_type in [
        DeepseekV4AttentionType.COMPRESSOR_STATE,
        DeepseekV4AttentionType.COMPRESSOR_SCORE,
        DeepseekV4AttentionType.INDEXER_COMPRESSOR_STATE,
        DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE,
    ]:
        dtype_bytes = 4  # (indexer) compressor state and score use float32
    # Indexer cache always packs data + per-block scales into one row.  Only
    # FP8 blockwise and MXFP4 are valid here -- bf16 / fp8_pertensor
    # are reserved for the main-attention compressor.
    if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESS:
        indexer_cache_dtype = resolve_kv_cache_dtype(indexer_k_cache_dtype)
        if indexer_cache_dtype == KVCacheDtype.FP8_BLOCKWISE:
            # 1 byte per value + 1 fp32 scale per 128 values.
            return attn_dim + index_head_dim // 128 * 4
        if indexer_cache_dtype == KVCacheDtype.MXFP4_BLOCKWISE:
            # ½ byte per value + 1 ue8m0 byte per 32 values.
            return index_head_dim // 2 + index_head_dim // 32
        raise ValueError(
            f"Unsupported indexer_k_cache_dtype {indexer_k_cache_dtype!r}; "
            "expected 'fp8_blockwise' or 'mxfp4'."
        )

    return attn_dim * dtype_bytes


class DeepseekV4TrtllmAttentionMetadata(DSAtrtllmAttentionMetadata):
    # The set of compress ratios for the layers
    compress_ratio_set: Set[int]
    # The set of (compress ratio, attention type) for the layers
    attention_type_set: Set[Tuple[int, DeepseekV4AttentionType]]
    # The number of total compressed tokens for each compress ratio
    num_total_compressed_tokens: Dict[int, int] = {}
    # The max number of context compressed tokens for each compress ratio
    max_ctx_compressed_tokens: Dict[int, int] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __post_init__(self):
        super().__post_init__()
        assert self.sparse_attention_config.window_size == 128, (
            f"Dual-pool sparse MLA requires window_size == 128, which equals to the"
            f"TileSizeKV of the FMHA kernel. (got {self.sparse_attention_config.window_size})."
        )
        capture_graph = self.is_cuda_graph
        self.compress_ratio_set = set(self.compress_ratios)
        # Cache a sorted list for deterministic iteration order across
        # torch.compile traces. Avoids repeated set→list conversion and
        # guarantees stable specialization keys.
        self._compress_ratios_sorted = sorted(self.compress_ratio_set)
        _supported_ratios = {1, 4, 128}
        _unsupported = self.compress_ratio_set - _supported_ratios
        assert not _unsupported, (
            f"Unsupported compress ratios {_unsupported}. Only {_supported_ratios} are supported."
        )

        attention_types = []
        for compress_ratio in self.compress_ratio_set:
            if compress_ratio == 1:
                attention_types.append((1, DeepseekV4AttentionType.SWA))
            elif compress_ratio == 4:
                attention_types.append((1, DeepseekV4AttentionType.SWA))
                attention_types.append((compress_ratio, DeepseekV4AttentionType.COMPRESS))
                attention_types.append((compress_ratio, DeepseekV4AttentionType.COMPRESSOR_STATE))
                attention_types.append((compress_ratio, DeepseekV4AttentionType.COMPRESSOR_SCORE))
                attention_types.append((compress_ratio, DeepseekV4AttentionType.INDEXER_COMPRESS))
                attention_types.append(
                    (compress_ratio, DeepseekV4AttentionType.INDEXER_COMPRESSOR_STATE)
                )
                attention_types.append(
                    (compress_ratio, DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE)
                )
            else:
                attention_types.append((1, DeepseekV4AttentionType.SWA))
                attention_types.append((compress_ratio, DeepseekV4AttentionType.COMPRESS))
                attention_types.append((compress_ratio, DeepseekV4AttentionType.COMPRESSOR_STATE))
                attention_types.append((compress_ratio, DeepseekV4AttentionType.COMPRESSOR_SCORE))
        self.attention_type_set = set(attention_types)

        # Create buffers for the compressor
        # cu_seq_lens_cuda is the cumulative sequence lengths for the requests
        self.cu_seq_lens_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences + 1,),
            dtype=torch.int,
            cache_name="cu_seq_lens_cuda",
            capture_graph=capture_graph,
        )
        self.cu_seq_lens = torch.empty_like(
            self.cu_seq_lens_cuda, device="cpu", pin_memory=prefer_pinned()
        )
        self.cu_seq_lens[0] = 0

        # new_comp_kv_lens_cuda is the number of new compressed tokens for the requests
        self.new_comp_kv_lens_cuda = {
            compress_ratio: self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_sequences,),
                dtype=torch.int,
                cache_name=f"new_comp_kv_lens_cuda_{compress_ratio}",
                capture_graph=capture_graph,
            )
            for compress_ratio in self.compress_ratio_set
        }

        # cu_new_comp_kv_cuda is the cumulative number of new compressed tokens for the requests
        self.cu_new_comp_kv_cuda = {
            compress_ratio: self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_sequences + 1,),
                dtype=torch.int,
                cache_name=f"cu_new_comp_kv_cuda_{compress_ratio}",
                capture_graph=capture_graph,
            )
            for compress_ratio in self.compress_ratio_set
        }

        # compressed_kv_lens_cuda is the number of compressed tokens for the requests
        self.compressed_kv_lens_cuda = {
            compress_ratio: self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_sequences,),
                dtype=torch.int,
                cache_name=f"compressed_kv_lens_cuda_{compress_ratio}",
                capture_graph=capture_graph,
            )
            for compress_ratio in self.compress_ratio_set
        }

        # past_kv_lens_cuda is the number of past compressed tokens for the requests
        self.past_kv_lens_cuda = {
            compress_ratio: self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_sequences,),
                dtype=torch.int,
                cache_name=f"past_kv_lens_cuda_{compress_ratio}",
                capture_graph=capture_graph,
            )
            for compress_ratio in self.compress_ratio_set
        }

        # compressed_position_ids_cuda is the compressed position ids for the requests
        self.compressed_position_ids_cuda = {
            compress_ratio: self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_tokens,),
                dtype=torch.int,
                cache_name=f"compressed_position_ids_cuda_{compress_ratio}",
                capture_graph=capture_graph,
            )
            for compress_ratio in self.compress_ratio_set
        }

        # compressed_mask_cuda: per-token bool mask for postprocess scatter.
        # Precomputed from new_comp_kv_lens to skip padded generation slots.
        self.compressed_mask_cuda = {
            compress_ratio: self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_tokens,),
                dtype=torch.bool,
                cache_name=f"compressed_mask_cuda_{compress_ratio}",
                capture_graph=capture_graph,
            )
            for compress_ratio in self.compress_ratio_set
        }
        self.compressed_mask = {
            compress_ratio: torch.empty_like(
                self.compressed_mask_cuda[compress_ratio], device="cpu", pin_memory=prefer_pinned()
            )
            for compress_ratio in self.compress_ratio_set
        }

        # empty topk indices buffer with all -1s in the tensor
        self.empty_topk_indices_buffer = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens, self.sparse_mla_topk),
            cache_name="empty_topk_indices_buffer",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.empty_topk_indices_buffer.fill_(-1)

        # SWA local indices
        self.swa_local_indices_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens, self.sparse_attention_config.window_size),
            cache_name="swa_local_indices_cuda",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )

        # Compute max_compressed_indices for CUDA graph compatibility.
        # For ratio=4, the indexer selects index_topk compressed tokens.
        # For ratio=128, use max_seq_len / 128 rounded up to next power of 2
        raw_128 = math.ceil(self.max_seq_len / 128)
        po2_128 = 1 << (raw_128 - 1).bit_length() if raw_128 > 0 else 1
        self.max_compressed_indices = {
            1: 0,  # No compressed indices
            4: self.sparse_mla_topk,  # index_topk from indexer
            128: po2_128,  # All compressed tokens, rounded to power of 2
        }

        # Compressed local indices for compress_ratio=128
        # Note: ratio=4 uses dynamic topk_indices from indexer, so we only pre-allocate for ratio=128
        self.compressed_local_indices_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens, self.max_compressed_indices[128]),
            cache_name="compressed_local_indices_cuda",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )

        self.block_tables = {
            attention_type: self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_sequences, self.kv_cache_manager.max_blocks_per_seq),
                cache_name=f"block_tables_{attention_type}",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            for attention_type in self.attention_type_set
        }
        self.host_block_tables = {
            attention_type: torch.empty_like(
                self.block_tables[attention_type], device="cpu", pin_memory=prefer_pinned()
            )
            for attention_type in self.attention_type_set
        }

        # sparse_mla_topk_lens: actual token count per token for each compress_ratio (SWA + compressed)
        # Shape: [max_num_tokens] per compress_ratio
        self.sparse_mla_topk_lens = {
            compress_ratio: self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_tokens,),
                cache_name=f"sparse_mla_topk_lens_{compress_ratio}",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            for compress_ratio in self.compress_ratio_set
        }

        # cached_token_lens_cuda: number of tokens already cached per request
        self.cached_token_lens_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences,),
            cache_name="cached_token_lens_cuda",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.cached_token_lens_cpu = torch.empty_like(
            self.cached_token_lens_cuda, device="cpu", pin_memory=prefer_pinned()
        )

        # Cache buffer data pointers are constant after KV cache allocation,
        # so compute them once during initialization instead of every prepare().
        self._init_cache_buffer_data_pointers()

    def prepare_for_indexer_k_cache(self):
        """Optimized: bulk tensor copy instead of per-row Python loop.

        Note: must use num_contexts=0 for get_batch_attn_offset because the
        indexer k cache always uses generation-style copy indices regardless
        of request type.
        """
        num_seqs = self.num_seqs
        offsets = self.kv_cache_manager.get_batch_attn_offset(
            self.request_ids,
            1,  # beam_width
            0,  # num_contexts=0 (indexer always uses gen-style copy index)
            num_seqs,
            DeepseekV4AttentionType.INDEXER_COMPRESS,
            DEEPSEEK_V4_SPARSE_RATIO,
        )
        num_cols = offsets.shape[1]
        self.host_indexer_k_cache_block_offsets[:num_seqs, :num_cols] = offsets[:num_seqs]
        self.indexer_k_cache_block_offsets[:num_seqs].copy_(
            self.host_indexer_k_cache_block_offsets[:num_seqs],
            non_blocking=True,
        )

    def prepare_for_block_tables(self):
        """Prepare block tables for all attention types.

        Delegates offset computation to DeepseekV4CacheManager.get_batch_block_offsets
        (single get_copy_index call, deduplicated by pool), then copies to device.
        """
        num_seqs = self.num_seqs
        offsets_map = self.kv_cache_manager.get_batch_block_offsets(
            self.request_ids, self.num_contexts, self.attention_type_set
        )

        # Phase 1: write host buffers.
        for key, offsets in offsets_map.items():
            self.host_block_tables[key][:num_seqs] = offsets[:num_seqs]

        # Phase 2: batch H2D copies.
        for key in self.attention_type_set:
            self.block_tables[key][:num_seqs].copy_(
                self.host_block_tables[key][:num_seqs], non_blocking=True
            )

    def prepare_for_deepseek_v4_indices(self, token_positions=None):
        """Prepare SWA/compressed local indices and sparse_mla_topk_lens."""
        window_size = self.sparse_attention_config.window_size
        device = self.swa_local_indices_cuda.device

        if token_positions is None:
            # Initial prepare() path — build token_positions from CPU data.
            num_tokens = self.num_tokens
            num_requests = self.num_seqs

            # cu_seq_lens_cuda must already be populated before this call
            cu_seq_lens = self.cu_seq_lens_cuda[: num_requests + 1]
            cached_tokens = self.cached_token_lens_cuda[:num_requests]

            token_idx = torch.arange(num_tokens, dtype=torch.int32, device=device)
            req_idx = torch.searchsorted(cu_seq_lens[1:].to(torch.int32), token_idx, right=True)
            offsets = token_idx - cu_seq_lens[req_idx].to(torch.int32)
            token_positions = cached_tokens[req_idx].to(torch.int32) + offsets

        self._prepare_deepseek_v4_indices_compiled(
            token_positions,
            window_size,
            self.max_compressed_indices[128],
            self.sparse_mla_topk,
            self.swa_local_indices_cuda,
            self.compressed_local_indices_cuda,
            self.sparse_mla_topk_lens,
            self._compress_ratios_sorted,
        )

    @staticmethod
    @maybe_compile(dynamic=True, options={"max-autotune": True})
    def _prepare_deepseek_v4_indices_compiled(
        token_positions: torch.Tensor,
        window_size: int,
        max_compressed_indices_128: int,
        sparse_mla_topk: int,
        swa_local_indices_buf: torch.Tensor,
        compressed_local_indices_buf: torch.Tensor,
        sparse_mla_topk_lens_bufs: Dict[int, torch.Tensor],
        compress_ratios: list,
    ):
        """Build SWA indices, compressed indices, and topk_lens in one fused graph."""
        device = token_positions.device
        num_tokens = token_positions.shape[0]

        # ── SWA local indices ──
        positions = token_positions.unsqueeze(1)  # [num_tokens, 1]
        swa_offsets = torch.arange(window_size, dtype=torch.int32, device=device)
        swa_start = (positions - window_size + 1).clamp(min=0)
        swa_indices = swa_start + swa_offsets
        swa_indices = torch.where(swa_indices > positions, -1, swa_indices).to(torch.int32)
        swa_local_indices_buf[:num_tokens] = swa_indices

        # ── Compressed local indices (ratio=128) ──
        # Hardcoded 128: compressed_local_indices only applies to ratio=128
        # layers. This is a design constraint — see compress_ratio_set
        # validation in __post_init__ which restricts ratios to {1, 4, 128}.
        num_valid = (token_positions + 1) // 128
        comp_col = torch.arange(max_compressed_indices_128, dtype=torch.int32, device=device)
        valid_mask = comp_col.unsqueeze(0) < num_valid.unsqueeze(1)
        comp_indices = torch.where(
            valid_mask,
            comp_col.unsqueeze(0).expand(num_tokens, -1),
            torch.full(
                (num_tokens, max_compressed_indices_128), -1, dtype=torch.int32, device=device
            ),
        )
        compressed_local_indices_buf[:num_tokens] = comp_indices

        # ── sparse_mla_topk_lens per compress_ratio ──
        # Dual-pool layout:
        # - compress_ratio=1: min(kv_len, window_size) — actual valid SWA count
        # - compress_ratio=4: window_size + min(kv_len // 4, sparse_mla_topk)
        #   (fixed 128 SWA slots + compressed valid count)
        # - compress_ratio=128: window_size + kv_len // 128
        #   (fixed 128 SWA slots + compressed valid count)
        # For ratio>1, the SWA region always occupies exactly window_size (128)
        # slots. Invalid SWA positions are padded with -1 in the index buffer.
        kv_lens = token_positions + 1
        for compress_ratio in compress_ratios:
            if compress_ratio == 1:
                total_count = kv_lens.clamp(max=window_size)
            elif compress_ratio == 4:
                compressed_count = (kv_lens // compress_ratio).clamp(max=sparse_mla_topk)
                total_count = window_size + compressed_count
            elif compress_ratio == 128:
                compressed_count = kv_lens // compress_ratio
                total_count = window_size + compressed_count
            else:
                raise ValueError(f"Unsupported compress_ratio: {compress_ratio}")
            sparse_mla_topk_lens_bufs[compress_ratio][:num_tokens] = total_count.to(torch.int32)

    def _init_cache_buffer_data_pointers(self):
        # If MTP is enabled, enlarge the compress ratios by max_draft_tokens - 1
        extend_compress_ratios = self.compress_ratios + [self.compress_ratios[-1]] * (
            self.max_draft_tokens - 1
        )
        self.swa_buffer_ptrs = {
            layer_idx: self.kv_cache_manager.get_buffers(
                layer_idx, DeepseekV4AttentionType.SWA
            ).data_ptr()
            for layer_idx in self.kv_cache_manager.pp_layers
        }
        self.compressed_buffer_ptrs = {
            layer_idx: self.kv_cache_manager.get_buffers(
                layer_idx, DeepseekV4AttentionType.COMPRESS
            ).data_ptr()
            for layer_idx in self.kv_cache_manager.pp_layers
            if is_compress_layer(extend_compress_ratios[layer_idx])
        }

        # Per-ratio base pointer for sparse MLA global-index conversion.
        # Ratio 1 is the SWA pool; ratios greater than 1 are compressed pools.
        self.sparse_mla_base_ptrs = {
            1: self.kv_cache_manager.swa_pool_ptr,
        }
        for ratio, compress_pool_ptr in self.kv_cache_manager.compress_pool_ptrs.items():
            self.sparse_mla_base_ptrs[ratio] = compress_pool_ptr

    def prepare(self):
        TrtllmAttentionMetadata.prepare(self)

        num_requests = self.num_contexts + self.num_generations
        num_gen_tokens = self.num_tokens - self.num_ctx_tokens

        self.cached_token_lens_cpu[:num_requests] = torch.tensor(
            self.kv_cache_params.num_cached_tokens_per_seq[:num_requests],
            dtype=torch.int32,
        )
        cached_token_lens = self.cached_token_lens_cpu
        kv_lens = cached_token_lens[:num_requests] + self.seq_lens_kv[:num_requests]

        self.cached_token_lens_cuda[:num_requests].copy_(
            cached_token_lens[:num_requests], non_blocking=True
        )

        # Prepare cu_seq_lens early — needed by prepare_for_deepseek_v4_indices
        self.cu_seq_lens[1 : num_requests + 1] = self.seq_lens.cumsum(0)
        self.cu_seq_lens_cuda[: num_requests + 1].copy_(
            self.cu_seq_lens[: num_requests + 1], non_blocking=True
        )

        # For indices conversion
        self.prepare_for_indices_conversion()

        has_sparse_layers = DEEPSEEK_V4_SPARSE_RATIO in self.compress_ratio_set

        # For indexer k cache (only needed when sparse layers exist)
        if has_sparse_layers:
            self.prepare_for_indexer_k_cache()

        # For spec decode
        self.prepare_for_spec_decode(kv_lens)

        # For block offsets
        self.prepare_for_block_tables()

        # For DeepSeek-V4 indices
        self.prepare_for_deepseek_v4_indices()

        # Prepare metadata for indexer (only needed when sparse layers exist)
        if has_sparse_layers:
            DeepseekV4Indexer.prepare(metadata=self)

        # --- Per-ratio metadata ---
        # 1) CPU-side: compute scalar metadata (num_total_compressed_tokens, etc.)
        # 2) CUDA-side: fill *_cuda buffers via prepare_compressed_kv_metadata()
        num_gen_tokens_per_seq = (
            num_gen_tokens // self.num_generations if self.num_generations > 0 else 0
        )
        self.num_gen_tokens_per_seq = num_gen_tokens_per_seq
        num_contexts = self.num_contexts
        num_generations = self.num_generations
        kv_lens_slice = kv_lens[:num_requests]
        cached_slice = cached_token_lens[:num_requests]

        if num_contexts > 0:
            # Prefill path: need per-request tensor ops for ctx scalar metadata.
            for compress_ratio in self.compress_ratio_set:
                new_comp_kv_lens = kv_lens_slice // compress_ratio - cached_slice // compress_ratio
                cu_new = new_comp_kv_lens.cumsum(0)
                num_ctx_compressed_tokens = cu_new[num_contexts - 1].item()
                num_gen_compressed_tokens = num_generations * (
                    (num_gen_tokens_per_seq + compress_ratio - 1) // compress_ratio
                )
                self.num_total_compressed_tokens[compress_ratio] = (
                    num_ctx_compressed_tokens + num_gen_compressed_tokens
                )
                self.max_ctx_compressed_tokens[compress_ratio] = (
                    new_comp_kv_lens[:num_contexts].max().item()
                )
        else:
            # Decode-only: scalars depend only on num_generations and
            # num_gen_tokens_per_seq, no per-request tensor ops needed.
            for compress_ratio in self.compress_ratio_set:
                self.num_total_compressed_tokens[compress_ratio] = num_generations * (
                    (num_gen_tokens_per_seq + compress_ratio - 1) // compress_ratio
                )
                self.max_ctx_compressed_tokens[compress_ratio] = 0

        # 2) CUDA-side: fill *_cuda buffers on device.
        kv_lens_cuda = (
            self.cached_token_lens_cuda[:num_requests] + self._seq_lens_cuda[:num_requests]
        )
        cached_tokens_cuda = self.cached_token_lens_cuda[:num_requests]
        self.prepare_compressed_kv_metadata(kv_lens_cuda, cached_tokens_cuda)

        self._compute_compressed_mask(
            self.new_comp_kv_lens_cuda,
            self.cu_new_comp_kv_cuda,
            self.compressed_mask_cuda,
            num_requests,
            self.num_total_compressed_tokens,
            self._compress_ratios_sorted,
        )

    def prepare_compressed_kv_metadata(
        self,
        kv_lens: torch.Tensor,
        cached_tokens: torch.Tensor,
    ):
        """Compute per-ratio compressed KV lens and position IDs on device.

        Shared by prepare() and on_update_kv_lens() to avoid duplicated logic.

        Args:
            kv_lens: Total KV lengths per request (device tensor, [batch_size]).
            cached_tokens: Cached token counts per request (device tensor, [batch_size]).
        """
        batch_size = kv_lens.shape[0]
        num_contexts = self.num_contexts
        num_generations = self.num_generations

        self._compute_per_ratio_kv_lens(
            kv_lens,
            cached_tokens,
            batch_size,
            self.compressed_kv_lens_cuda,
            self.past_kv_lens_cuda,
            self.new_comp_kv_lens_cuda,
            self.cu_new_comp_kv_cuda,
            self._compress_ratios_sorted,
        )

        if num_contexts > 0:
            self._compute_ctx_compressed_position_ids(
                self.past_kv_lens_cuda,
                self.cu_new_comp_kv_cuda,
                self.compressed_position_ids_cuda,
                num_contexts,
                self._compress_ratios_sorted,
            )

        if self.num_gen_tokens_per_seq > 0 and num_generations > 0:
            # Extract output_offset as Python int per ratio to avoid
            # tensor-scalar slice inside compiled function.
            # For decode-only batches (num_contexts == 0), offset is 0.
            gen_output_offsets = {
                r: self.cu_new_comp_kv_cuda[r][num_contexts].item() if num_contexts > 0 else 0
                for r in self._compress_ratios_sorted
            }
            self._compute_gen_compressed_position_ids(
                self.past_kv_lens_cuda,
                self.compressed_position_ids_cuda,
                num_contexts,
                num_generations,
                self.num_gen_tokens_per_seq,
                self._compress_ratios_sorted,
                gen_output_offsets,
            )

    def on_update_kv_lens(self):
        """Recompute kv-lens-dependent DeepSeek-V4 metadata on device."""
        super().on_update_kv_lens()

        batch_size = self.num_seqs
        num_tokens = self.num_tokens
        kv_lens = self.kv_lens_cuda[:batch_size]
        seq_lens = self._seq_lens_cuda[:batch_size]
        cached_tokens = kv_lens - seq_lens

        num_gen_tokens = num_tokens - self.num_ctx_tokens
        self.num_gen_tokens_per_seq = (
            num_gen_tokens // self.num_generations if self.num_generations > 0 else 0
        )

        self.prepare_compressed_kv_metadata(kv_lens, cached_tokens)

        self._compute_compressed_mask(
            self.new_comp_kv_lens_cuda,
            self.cu_new_comp_kv_cuda,
            self.compressed_mask_cuda,
            batch_size,
            self.num_total_compressed_tokens,
            self._compress_ratios_sorted,
        )

        token_positions = self._compute_token_positions(
            seq_lens,
            cached_tokens,
            batch_size,
            num_tokens,
            self.cu_seq_lens_cuda,
            self.req_idx_per_token,
        )

        self.prepare_for_deepseek_v4_indices(token_positions)

    @staticmethod
    @maybe_compile(dynamic=True, options={"max-autotune": True})
    def _compute_per_ratio_kv_lens(
        kv_lens: torch.Tensor,
        cached_tokens: torch.Tensor,
        batch_size: int,
        compressed_kv_lens_bufs: Dict[int, torch.Tensor],
        past_kv_lens_bufs: Dict[int, torch.Tensor],
        new_comp_kv_lens_bufs: Dict[int, torch.Tensor],
        cu_new_comp_kv_bufs: Dict[int, torch.Tensor],
        compress_ratios: list,
    ):
        """Compute per-ratio compressed/past/new kv lens and cu_new_comp."""
        for compress_ratio in compress_ratios:
            compressed_kv = (kv_lens // compress_ratio).to(torch.int)
            compressed_kv_lens_bufs[compress_ratio][:batch_size] = compressed_kv

            past_kv = (cached_tokens // compress_ratio).to(torch.int)
            past_kv_lens_bufs[compress_ratio][:batch_size] = past_kv

            new_comp = compressed_kv - past_kv
            new_comp_kv_lens_bufs[compress_ratio][:batch_size] = new_comp

            cu_new_comp = cu_new_comp_kv_bufs[compress_ratio]
            cu_new_comp[: batch_size + 1] = torch.nn.functional.pad(
                torch.cumsum(new_comp, dim=0), (1, 0)
            )

    @staticmethod
    def _compute_token_positions(
        seq_lens: torch.Tensor,
        cached_tokens: torch.Tensor,
        batch_size: int,
        num_tokens: int,
        cu_seq_lens_buf: torch.Tensor,
        req_idx_per_token_buf: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cu_seq_lens, req_idx_per_token, and token_positions (eager)."""
        device = seq_lens.device

        # cu_seq_lens
        cu_seq_lens_buf[: batch_size + 1] = torch.nn.functional.pad(
            torch.cumsum(seq_lens.to(torch.int), dim=0), (1, 0)
        )

        # req_idx_per_token via searchsorted
        token_idx = torch.arange(num_tokens, dtype=torch.int32, device=device)
        req_idx = torch.searchsorted(
            cu_seq_lens_buf[1 : batch_size + 1].to(torch.int32), token_idx, right=True
        )
        req_idx_per_token_buf[:num_tokens] = req_idx

        # token positions
        base_pos = cached_tokens[req_idx].to(torch.int32)
        offsets = token_idx - cu_seq_lens_buf[req_idx].to(torch.int32)
        return base_pos + offsets

    @staticmethod
    @maybe_compile(dynamic=True, options={"max-autotune": True})
    def _compute_gen_compressed_position_ids(
        past_kv_lens_bufs: Dict[int, torch.Tensor],
        compressed_position_ids_bufs: Dict[int, torch.Tensor],
        num_contexts: int,
        num_generations: int,
        num_gen_tokens_per_seq: int,
        compress_ratios: list,
        gen_output_offsets: Dict[int, int],
    ):
        """Generation compressed position IDs.

        gen_output_offsets: dict mapping compress_ratio -> Python int offset,
        pre-extracted by the caller to avoid tensor-scalar .item() inside
        compiled code.  0 for decode-only batches."""
        device = past_kv_lens_bufs[compress_ratios[0]].device
        batch_size = num_contexts + num_generations
        for compress_ratio in compress_ratios:
            gen_past = past_kv_lens_bufs[compress_ratio][num_contexts:batch_size]
            new_gen_comp = (num_gen_tokens_per_seq + compress_ratio - 1) // compress_ratio
            gen_offsets = torch.arange(new_gen_comp, dtype=torch.int32, device=device)
            gen_pos = gen_past.unsqueeze(1) + gen_offsets.unsqueeze(0)
            gen_comp = num_generations * new_gen_comp
            result = (gen_pos.reshape(-1) * compress_ratio).to(torch.int)
            output_offset = gen_output_offsets[compress_ratio]
            compressed_position_ids_bufs[compress_ratio][
                output_offset : output_offset + gen_comp
            ] = result

    @staticmethod
    @maybe_compile(dynamic=True, options={"max-autotune": True})
    def _compute_compressed_mask(
        new_comp_kv_lens_bufs: Dict[int, torch.Tensor],
        cu_new_comp_kv_bufs: Dict[int, torch.Tensor],
        compressed_mask_bufs: Dict[int, torch.Tensor],
        batch_size: int,
        num_total_compressed_tokens: Dict[int, int],
        compress_ratios: list,
    ):
        """Compute per-token compressed_mask on device (graph-safe, no .item()).

        For each token in [0, total_tokens), determine which sequence it
        belongs to via searchsorted on cu_new_comp_kv, then compare its
        within-sequence offset against the actual new_comp_kv_len.
        Context tokens (offset < actual) are always True.
        Generation tokens whose offset >= actual new_comp are padding → False.
        """
        device = new_comp_kv_lens_bufs[compress_ratios[0]].device
        for compress_ratio in compress_ratios:
            total_tokens = num_total_compressed_tokens[compress_ratio]
            new_comp = new_comp_kv_lens_bufs[compress_ratio][:batch_size]
            cu = cu_new_comp_kv_bufs[compress_ratio][: batch_size + 1]

            token_idx = torch.arange(total_tokens, dtype=torch.int32, device=device)
            seq_idx = torch.searchsorted(cu[1:], token_idx, right=True)
            seq_idx = seq_idx.clamp_(max=batch_size - 1)
            offset_in_seq = token_idx - cu[seq_idx]
            compressed_mask_bufs[compress_ratio][:total_tokens] = offset_in_seq < new_comp[seq_idx]

    @staticmethod
    def _compute_ctx_compressed_position_ids(
        past_kv_lens_bufs: Dict[int, torch.Tensor],
        cu_new_comp_kv_bufs: Dict[int, torch.Tensor],
        compressed_position_ids_bufs: Dict[int, torch.Tensor],
        num_contexts: int,
        compress_ratios: list,
    ):
        """Context-only compressed position IDs (eager, data-dependent shapes)."""
        device = past_kv_lens_bufs[compress_ratios[0]].device
        for compress_ratio in compress_ratios:
            past_kv = past_kv_lens_bufs[compress_ratio]
            cu_new_comp = cu_new_comp_kv_bufs[compress_ratio]

            total_ctx_comp = cu_new_comp[num_contexts]
            ctx_idx = torch.arange(total_ctx_comp, dtype=torch.int32, device=device)
            ctx_cu = cu_new_comp[: num_contexts + 1].to(torch.int32)
            ctx_req = torch.searchsorted(ctx_cu[1:], ctx_idx, right=True)
            ctx_offset = ctx_idx - ctx_cu[ctx_req]
            compressed_position_ids_bufs[compress_ratio][:total_ctx_comp] = (
                (past_kv[:num_contexts][ctx_req] + ctx_offset) * compress_ratio
            ).to(torch.int)


class DeepseekV4Indexer(Indexer):
    def __init__(
        self,
        quant_config: Optional[QuantConfig],
        pos_embd_params: Optional[PositionalEmbeddingParams],
        mla_params: Optional[MLAParams],
        skip_create_weights_in_init: bool,
        sparse_attention_config: "SparseAttentionConfig",
        dtype: Optional[torch.dtype],
        compress_ratio: int = 1,
        layer_idx: int = 0,
        aux_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__(
            quant_config,
            pos_embd_params,
            mla_params,
            skip_create_weights_in_init,
            sparse_attention_config,
            dtype,
            compress_ratio,
            layer_idx,
            aux_stream,
        )
        # Override base Indexer.weights_proj to bf16 (matches V4 checkpoint).
        self.weights_proj = Linear(
            self.hidden_size,
            self.n_heads,
            bias=False,
            dtype=dtype,
            quant_config=None,
            skip_create_weights_in_init=skip_create_weights_in_init,
            use_custom_cublas_mm=True,
        )
        self.rotary_emb = RotaryEmbedding(
            pos_embd_params.rope,
            head_dim=self.rope_dim,
            is_neox=False,
        )
        rms_norm_eps = 1e-6
        index_head_dim = sparse_attention_config.index_head_dim
        indexer_mla_params = MLAParams(
            hidden_size=mla_params.hidden_size,
            qk_rope_head_dim=mla_params.qk_rope_head_dim,
            qk_nope_head_dim=index_head_dim - mla_params.qk_rope_head_dim,
        )
        self.indexer_k_cache_dtype = getattr(
            sparse_attention_config, "indexer_k_cache_dtype", "fp8_blockwise"
        )
        self.indexer_cache_dtype = resolve_kv_cache_dtype(self.indexer_k_cache_dtype)
        self.compressor = Compressor(
            indexer_mla_params,
            layer_idx,
            compress_ratio,
            rms_norm_eps,
            skip_create_weights_in_init,
            pos_embd_params,
            dtype=dtype,
            kv_cache_dtype=self.indexer_k_cache_dtype,
            is_indexer=True,
            rotate_activation=True,
        )

    def post_load_weights(self):
        # V4 does not use the V3 fused fp32 wk+weights_proj GEMM, and the
        # base concat would now hit an fp32/bf16 dtype mismatch.
        return

    def _qk_projection_and_rope(self, qr: torch.Tensor, position_ids: torch.Tensor):
        """Project Q and apply RoPE.

        Returns q with layout [num_tokens, n_heads, head_dim] where
        head_dim = nope_dim + rope_dim, RoPE already applied in-place.
        """
        q = self.wq_b(qr)
        q = q.view(-1, self.n_heads, self.head_dim)
        # Fused in-place RoPE on the rope portion of each head
        nope_dim = self.head_dim - self.rope_dim
        torch.ops.trtllm.mla_rope_inplace(
            q,
            position_ids.view(-1),
            self.rotary_emb.rotary_cos_sin,
            self.n_heads,
            nope_dim,
            self.rope_dim,
            False,
            self.rotary_emb.is_neox,
        )
        return q

    def _update_k_cache(self, k_fp8, k_scale, metadata):
        """Overwrite the fused compressor's FP8 cache with reference-style FP4-QAT values."""
        super()._update_k_cache(k_fp8, k_scale, metadata)

    def forward(
        self,
        qr: torch.Tensor,
        hidden_states: torch.Tensor,
        metadata: DeepseekV4TrtllmAttentionMetadata,
        position_ids: torch.Tensor,
    ):
        if self.indexer_cache_dtype not in (
            KVCacheDtype.FP8_PERTENSOR,
            KVCacheDtype.FP8_BLOCKWISE,
        ):
            raise NotImplementedError(
                "DeepSeek-V4 indexer BMM currently consumes FP8 K. "
                f"Indexer cache preset {self.indexer_k_cache_dtype!r} is supported by "
                "the compressor/cache scatter path, but needs the matching BF16/FP4 "
                "indexer BMM path before end-to-end indexer execution can use it."
            )
        # compress k
        k_fp8, k_scale = self.compressor(hidden_states, metadata)

        # multi-stream q proj/rope and weights proj
        q, weights = maybe_execute_in_parallel(
            lambda: self._qk_projection_and_rope(qr, position_ids),
            lambda: self.weights_proj(hidden_states),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )

        # Rotate + quantize (layout matches compressor K: [nope|pe])
        q = rotate_activation(q)
        q = q.view(-1, self.head_dim)
        q_fp8, q_scale = fp8_utils.fp8_quantize_1x128_sf_transpose(
            q, use_ue8m0=self.scale_fmt == "ue8m0"
        )
        q_fp8 = q_fp8.view(-1, self.n_heads, self.head_dim)
        q_scale = q_scale.view(-1, self.n_heads, 1)

        # weights scale
        weights = self._weight_scale(weights, q_scale)

        # If there are no compressed tokens, return an topk indices buffer with all -1s in the tensor.
        if k_fp8 is None:
            topk_indices = metadata.empty_topk_indices_buffer[: hidden_states.shape[0]]
        else:
            topk_indices = self.sparse_attn_indexer(
                metadata, hidden_states, q_fp8, k_fp8, k_scale, weights
            )
        return topk_indices


class DeepseekV4TrtllmAttention(TrtllmAttention):
    Metadata = DeepseekV4TrtllmAttentionMetadata

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
        aux_stream: Optional[torch.cuda.Stream] = None,
        **kwargs,
    ):
        assert sparse_attention_config is not None, (
            "sparse_attention_config is required for DeepseekV4TrtllmAttention and cannot be None"
        )
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
            **kwargs,
        )

        self.compress_ratio = sparse_attention_config.compress_ratios[layer_idx]

        if self.compress_ratio == 4:
            self.indexer = DeepseekV4Indexer(
                quant_config,
                pos_embd_params,
                mla_params,
                skip_create_weights_in_init,
                sparse_attention_config,
                dtype,
                self.compress_ratio,
                layer_idx,
                aux_stream,
            )

        if self.compress_ratio > 1:
            rms_norm_eps = 1e-6
            has_fp8_kv_cache = False
            if quant_config is not None:
                has_fp8_kv_cache = quant_config.layer_quant_mode.has_fp8_kv_cache()
            kv_cache_dtype = "fp8_pertensor" if has_fp8_kv_cache else "default"
            self.compressor = Compressor(
                mla_params,
                layer_idx,
                self.compress_ratio,
                rms_norm_eps,
                skip_create_weights_in_init,
                pos_embd_params,
                kv_cache_dtype=kv_cache_dtype,
                dtype=dtype,
                rotate_activation=False,
            )

    def forward(self, *args, **kwargs):
        attn_sink = getattr(self, "attn_sink", None)
        if attn_sink is not None and kwargs.get("attention_sinks") is None:
            kwargs["attention_sinks"] = attn_sink.data
        return super().forward(*args, **kwargs)

    def sparse_attn_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: DeepseekV4TrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Convert local indices (SWA + compressed) to global pool indices."""
        layer_idx = self.layer_idx
        kv_cache_manager = metadata.kv_cache_manager
        attention_input_type = forward_args.attention_input_type

        swa_pool_base_ptr = metadata.sparse_mla_base_ptrs[1]

        # Get cached buffer pointers
        swa_buffer_ptr = metadata.swa_buffer_ptrs[layer_idx]

        # Token stride
        index_head_dim = self.sparse_attention_config.index_head_dim
        has_fp8_kv_cache = False
        if self.quant_config is not None:
            has_fp8_kv_cache = self.quant_config.layer_quant_mode.has_fp8_kv_cache()
        token_stride = get_token_bytes(
            self.head_dim,
            index_head_dim,
            self.compress_ratio,
            DeepseekV4AttentionType.SWA,
            has_fp8_kv_cache,
        )

        # Select token range based on phase
        if attention_input_type == AttentionInputType.context_only:
            start_idx = 0
            end_idx = metadata.num_ctx_tokens
        elif attention_input_type == AttentionInputType.generation_only:
            start_idx = metadata.num_ctx_tokens
            end_idx = metadata.num_tokens
        else:
            start_idx = 0
            end_idx = metadata.num_tokens

        # Use global req_id directly
        req_id = metadata.req_idx_per_token[start_idx:end_idx]
        swa_local_indices = metadata.swa_local_indices_cuda[start_idx:end_idx]
        block_table_swa = metadata.block_tables[(1, DeepseekV4AttentionType.SWA)]

        if self.compress_ratio > 1:
            compressed_buffer_ptr = metadata.compressed_buffer_ptrs[layer_idx]
            compress_pool_base_ptr = metadata.sparse_mla_base_ptrs[self.compress_ratio]
            block_table_compressed = metadata.block_tables[
                (self.compress_ratio, DeepseekV4AttentionType.COMPRESS)
            ]
            if self.compress_ratio == 4:
                topk_indices = forward_args.topk_indices
                assert topk_indices is not None, "topk_indices is required when compress_ratio=4"
                compressed_local_indices = topk_indices
            else:
                compressed_local_indices = metadata.compressed_local_indices_cuda[start_idx:end_idx]
        else:
            compressed_buffer_ptr = 0
            compress_pool_base_ptr = 0
            block_table_compressed = None
            compressed_local_indices = None

        global_indices = deepseek_v4_local_to_global_indices(
            req_id=req_id,
            block_table_swa=block_table_swa,
            swa_local_indices=swa_local_indices,
            swa_pool_base_ptr=swa_pool_base_ptr,
            swa_buffer_ptr=swa_buffer_ptr,
            tokens_per_block=kv_cache_manager.tokens_per_block,
            token_stride=token_stride,
            block_table_compressed=block_table_compressed,
            compressed_local_indices=compressed_local_indices,
            compress_pool_base_ptr=compress_pool_base_ptr,
            compressed_buffer_ptr=compressed_buffer_ptr,
            compress_ratio=self.compress_ratio,
            num_compressed_indices=metadata.max_compressed_indices[self.compress_ratio],
        )

        return global_indices, None

    def sparse_kv_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: DeepseekV4TrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return None, None
