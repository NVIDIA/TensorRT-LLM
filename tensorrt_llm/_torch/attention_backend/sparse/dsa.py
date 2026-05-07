"""Dense Sparse Attention (DSA) backend for TRT-LLM with indexer-based TopK selection."""
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionForwardArgs, MLAParams, PositionalEmbeddingParams)
from tensorrt_llm._torch.attention_backend.trtllm import (
    TrtllmAttention, TrtllmAttentionMetadata)
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.distributed.ops import allgather
from tensorrt_llm._torch.modules.layer_norm import LayerNorm
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.multi_stream_utils import \
    maybe_execute_in_parallel
from tensorrt_llm._torch.modules.rotary_embedding import RotaryEmbedding
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.utils import maybe_compile
from tensorrt_llm._utils import get_size_in_bytes, get_sm_version, prefer_pinned
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.bindings.internal.batch_manager import \
    CacheType as CacheTypeCpp
from tensorrt_llm.deep_gemm import (fp8_fp4_mqa_logits,
                                    fp8_fp4_paged_mqa_logits, fp8_mqa_logits,
                                    fp8_paged_mqa_logits,
                                    get_paged_mqa_logits_metadata)
from tensorrt_llm.llmapi.llm_args import SparseAttentionConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

ModelConfig = tensorrt_llm.bindings.ModelConfig

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig

# Optional import: fast-hadamard-transform causes CI build issues (requires wheel+torch pre-installed)
try:
    from fast_hadamard_transform import hadamard_transform
    HAS_FAST_HADAMARD = True
except ImportError:
    hadamard_transform = None
    HAS_FAST_HADAMARD = False


def _compute_slot_mappings(
    global_positions: torch.Tensor,
    block_offsets: torch.Tensor,
    req_indices: torch.Tensor,
    head_dim: int,
    tokens_per_block: int,
    quant_block_size: int,
    data_bytes_per_token: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute flat byte indices for indexer K data and scales from global token positions.

    Shared by Indexer.prepare() (CPU) and on_update_kv_lens() (GPU) to avoid
    duplicating the slot mapping arithmetic.

    Args:
        global_positions: Per-token absolute position in the KV sequence.
        block_offsets: [num_seqs, max_blocks_per_seq] block offset table.
        req_indices: Per-token request index.
        head_dim: Indexer head dimension (used for the scale-size formula).
        tokens_per_block: Tokens stored per cache block.
        quant_block_size: Quantization block size.
        data_bytes_per_token: Bytes of quantized data per token in the cache
            pool. FP8 stores one byte per element (= head_dim). FP4 packs two
            E2M1 codes per byte (= head_dim // 2). Defaults to ``head_dim``
            when unset, preserving the FP8 layout for callers that haven't
            threaded the FP4 dtype through.

    Returns:
        (fp8_indices, scale_indices): Flat byte offsets into the cache pool.
    """
    if data_bytes_per_token is None:
        data_bytes_per_token = head_dim
    scale_size = head_dim // quant_block_size * 4  # float32 = 4 bytes
    block_stride = tokens_per_block * (data_bytes_per_token + scale_size)
    scale_base_offset = tokens_per_block * data_bytes_per_token

    block_indices_in_seq = global_positions // tokens_per_block
    pos_in_blocks = global_positions % tokens_per_block

    max_blocks = block_offsets.shape[1]
    if block_indices_in_seq.is_cuda:
        # Clamp to prevent OOB from stale token-to-seq mappings during
        # CUDA graph capture/replay with MTP + DSA.
        block_indices_in_seq = block_indices_in_seq.clamp(0, max_blocks - 1)
    else:
        assert (block_indices_in_seq < max_blocks).all(), \
            f"Block index out of bounds: max={max_blocks}, got indices up to {block_indices_in_seq.max().item()}"

    block_ids = block_offsets[req_indices, block_indices_in_seq].to(torch.int64)

    fp8_indices = block_ids * block_stride + pos_in_blocks * data_bytes_per_token
    scale_indices = (block_ids * block_stride + scale_base_offset +
                     pos_in_blocks * scale_size)
    return fp8_indices, scale_indices


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply Hadamard rotation to activation tensor for DSA sparse attention."""
    assert x.dtype == torch.bfloat16

    if not HAS_FAST_HADAMARD:
        # Fallback: skip transformation (acceptable for test/dev)
        logger.warning_once(
            "fast-hadamard-transform not available. DSA sparse attention will skip "
            "hadamard transformation. Install with: "
            "pip install git+https://github.com/Dao-AILab/fast-hadamard-transform.git",
            key="fast_hadamard_import_missing")
        return x

    hidden_size = x.size(-1)
    assert (hidden_size & (hidden_size - 1)
            ) == 0, "Hidden size must be a power of 2 for Hadamard transform."
    return hadamard_transform(x, scale=hidden_size**-0.5)


def transform_local_topk_and_prepare_pool_view(
    topk_indices: torch.Tensor,
    attn_metadata: "DSAtrtllmAttentionMetadata",
    layer_idx: int,
    is_generation: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert local topk indices to global pool indices and prepare KV pool.

    Uses cached values from attn_metadata._ensure_pool_view_cached()
    to avoid redundant Python/CUDA overhead across layers.
    """
    assert topk_indices.dtype == torch.int32

    attn_metadata._ensure_pool_view_cached()

    if is_generation:
        block_table = attn_metadata._cached_block_table_gen
        req_idx = attn_metadata._cached_req_idx_gen
    else:
        block_table = attn_metadata._cached_block_table_ctx
        req_idx = attn_metadata._cached_req_idx_ctx

    global_indices = torch.ops.trtllm.convert_req_index_to_global(
        req_idx,
        block_table,
        topk_indices,
        attn_metadata._cached_tokens_per_block,
        topk_indices.shape[1],
        attn_metadata._cached_stride_factor,
        layer_idx,
    )

    return global_indices, attn_metadata._cached_pool_view


def split_prefill_chunks(
    seq_lens: torch.Tensor,
    max_chunk_size: int,
    start_idx: int = 0,
) -> List[List[Tuple[int, int, int, int]]]:
    """
    Split prefill requests into chunks based on max_chunk_size.
    Supports two-level chunking:
    1. Request-boundary chunking: group multiple small requests into one chunk
    2. Intra-request chunking: split large requests into multiple Q-block chunks

    Args:
        seq_lens: Sequence lengths for all requests
        max_chunk_size: Maximum number of tokens per chunk
        start_idx: Starting index for prefill requests

    Returns:
        List of chunk groups, where each group is a list of chunk specs.
        Each chunk spec is (req_idx, token_start_in_req, token_end_in_req, req_cum_start)

        - For multi-request chunks: group contains multiple specs (one per request)
        - For intra-request chunks: each Q-block is a separate group with single spec
    """
    chunk_groups = []
    num_reqs = len(seq_lens)

    current_req = start_idx
    # Compute cumulative token positions
    query_start_loc_cpu = torch.cat([
        torch.zeros(1, dtype=torch.int32, device='cpu'),
        seq_lens.cumsum(dim=0).to(torch.int32)
    ])

    while current_req < num_reqs:
        seq_len = seq_lens[current_req].item()
        req_cum_start = query_start_loc_cpu[current_req].item()

        if seq_len <= max_chunk_size:
            # This request fits in one chunk - try to pack with others
            current_size = seq_len
            chunk_specs = [(current_req, 0, seq_len, req_cum_start)]
            next_req = current_req + 1

            # Try to add more requests to this chunk
            while next_req < num_reqs:
                next_seq_len = seq_lens[next_req].item()
                if next_seq_len > max_chunk_size:
                    # Next request is large, stop packing
                    break
                if current_size + next_seq_len <= max_chunk_size:
                    next_cum_start = query_start_loc_cpu[next_req].item()
                    chunk_specs.append(
                        (next_req, 0, next_seq_len, next_cum_start))
                    current_size += next_seq_len
                    next_req += 1
                else:
                    break

            # Add as one multi-request chunk group
            chunk_groups.append(chunk_specs)
            current_req = next_req
        else:
            # Large request - split into Q-blocks
            # Each Q-block is a separate chunk group (processed in separate iteration)
            num_q_blocks = (seq_len + max_chunk_size - 1) // max_chunk_size
            for q_block_idx in range(num_q_blocks):
                token_start = q_block_idx * max_chunk_size
                token_end = min(token_start + max_chunk_size, seq_len)
                q_block_spec = [(current_req, token_start, token_end,
                                 req_cum_start)]
                chunk_groups.append(q_block_spec)

            current_req += 1

    return chunk_groups


def compute_cu_seqlen_kv_bounds_with_cache(
    seq_lens: torch.Tensor,
    num_contexts: int,
    num_ctx_tokens: int,
    cached_token_lens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute attention window bounds for batched sequences with causal attention,
    accounting for cached KV tokens.

    Args:
        seq_lens: current token lengths [num_contexts], dtype=torch.int32
        num_contexts: Number of sequences in the batch
        num_ctx_tokens: Total number of context tokens across all sequences in current batch
        cached_token_lens: Cached KV token lengths [num_contexts], dtype=torch.int32 (optional)

    Returns:
        cu_seqlen_ks: Start index in KV for each Q token [num_ctx_tokens]
        cu_seqlen_ke: End index (exclusive) in KV for each Q token [num_ctx_tokens]
    """
    device = seq_lens.device
    # Total KV lengths per request
    kv_lens = seq_lens if cached_token_lens is None else cached_token_lens + seq_lens  # [num_contexts]

    # Cumulative KV offsets: where each request's KV sequence starts in global KV space
    cu_kv_offsets = torch.cat([
        torch.zeros(1, device=device, dtype=torch.int32),
        torch.cumsum(kv_lens, dim=0).to(torch.int32)
    ])  # [num_contexts + 1]

    # Map each Q token to its request: [0,0,...,0, 1,1,...,1, ..., B-1,B-1,...,B-1]
    batch_ids = torch.repeat_interleave(
        torch.arange(num_contexts, device=device, dtype=torch.int32),
        seq_lens)  # [num_ctx_tokens]

    # Each Q token's KV window starts at its request's KV sequence start
    cu_seqlen_ks = cu_kv_offsets[batch_ids]  # [num_ctx_tokens]

    # Compute local Q position within each request (0-based, relative to current batch context tokens)
    cu_q_offsets = torch.cat([
        torch.zeros(1, device=device, dtype=torch.int32),
        torch.cumsum(seq_lens, dim=0).to(torch.int32)
    ])  # [num_contexts + 1]

    global_q_positions = torch.arange(num_ctx_tokens,
                                      device=device,
                                      dtype=torch.int32)
    local_q_positions = global_q_positions - torch.repeat_interleave(
        cu_q_offsets[:-1], seq_lens)  # [num_ctx_tokens]

    if cached_token_lens is not None:
        cached_per_token = torch.repeat_interleave(cached_token_lens,
                                                   seq_lens)  # [num_ctx_tokens]
        cu_seqlen_ke = cu_seqlen_ks + cached_per_token + local_q_positions + 1  # [num_ctx_tokens]
    else:
        cu_seqlen_ke = cu_seqlen_ks + local_q_positions + 1  # [num_ctx_tokens]

    return cu_seqlen_ks, cu_seqlen_ke


@dataclass
class IndexerPrefillChunkMetadata:
    """Metadata for a single prefill chunk in the indexer"""
    cu_seqlen_ks: torch.Tensor  # Attention window start for each token
    cu_seqlen_ke: torch.Tensor  # Attention window end for each token
    token_start: int  # Q token start index in batch
    token_end: int  # Q token end index in batch
    k_token_start: int  # K token start index in batch
    k_token_end: int  # K token end index in batch


class DSAtrtllmAttentionMetadata(TrtllmAttentionMetadata):
    """Attention metadata for DSA (Dense Sparse Attention) with indexer state."""

    # Store reference to indexer for preparation stage
    indexer: Optional["Indexer"] = None
    # Chunked prefill metadata for indexer (prefill-only, no CUDA graph needed)
    indexer_prefill_chunks: Optional[List[IndexerPrefillChunkMetadata]] = None
    # Max chunk size for two-level chunking:
    # 1. Request-level: Pack multiple small requests into one chunk (up to indexer_max_chunk_size)
    # 2. Intra-request: Split large requests into Q-blocks when seq_len > max_chunk_size
    indexer_max_chunk_size: int
    # Topk for sparse MLA
    num_sparse_topk: int
    # max number of draft tokens
    max_draft_tokens: int = 0
    # Enable indexer skip for short sequences
    enable_indexer_skip: bool = False
    # Whether skip the indexer for context requests
    skip_indexer_for_ctx_reqs: bool = False
    # Whether skip the indexer for generation requests
    skip_indexer_for_gen_reqs: bool = False
    # Whether to use the expanded buffers for MTP support
    use_expanded_buffers_for_mtp: bool = False

    def __init__(self, *args, **kwargs):
        """Initialize DSA metadata with SM count and indexer chunk size."""
        self.num_sms = tensorrt_llm.deep_gemm.get_num_sms()
        # Cached step-invariant values for transform_local_topk_and_prepare_pool_view.
        # These are recomputed once per step in _ensure_pool_view_cached() and
        # reused across all layers to avoid redundant Python/CUDA overhead.
        # Initialized here as plain instance attributes (not class-level
        # annotations) to stay invisible to dataclass/torch.compile introspection.
        self._pool_cache_valid = False
        self._cached_kv_mgr_id = 0
        self._cached_pool_view = None
        self._cached_stride_factor = 0
        self._cached_tokens_per_block = 0
        self._cached_block_table_ctx = None
        self._cached_block_table_gen = None
        self._cached_req_idx_ctx = None
        self._cached_req_idx_gen = None
        super().__init__(*args, **kwargs)
        if self.sparse_attention_config.indexer_max_chunk_size is not None:
            self.indexer_max_chunk_size = self.sparse_attention_config.indexer_max_chunk_size
        else:
            self.indexer_max_chunk_size = 32768  # Default to 32K tokens for the indexer

    def __post_init__(self):
        """Allocate indexer K-cache buffers and heuristic TopK metadata."""
        super().__post_init__()
        assert isinstance(self.kv_cache_manager, DSACacheManager), \
            f"DSAtrtllmAttentionMetadata requires DSACacheManager, got {type(self.kv_cache_manager)}"

        self.num_sparse_topk = self.sparse_attention_config.index_topk
        self.enable_indexer_skip = self.sparse_attention_config.skip_indexer_for_short_seqs
        capture_graph = self.is_cuda_graph

        self.indexer_k_cache_block_offsets = self.get_empty(
            self.cuda_graph_buffers,
            [self.max_num_sequences, self.kv_cache_manager.max_blocks_per_seq],
            cache_name="indexer_k_cache_block_offsets",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.host_indexer_k_cache_block_offsets = torch.zeros_like(
            self.indexer_k_cache_block_offsets,
            device='cpu',
            pin_memory=prefer_pinned(),
        )

        if not self.enable_context_mla_with_cached_kv:
            self.ctx_cached_token_indptr = self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_requests + 1, ),
                cache_name="ctx_cached_token_indptr",
                dtype=torch.int64,
                capture_graph=capture_graph,
            )
            self.host_ctx_cached_token_indptr = torch.zeros_like(
                self.ctx_cached_token_indptr,
                device='cpu',
                pin_memory=prefer_pinned(),
            )
            self.ctx_kv_indptr = self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_requests + 1, ),
                cache_name="ctx_kv_indptr",
                dtype=torch.int64,
                capture_graph=capture_graph,
            )
            self.host_ctx_kv_indptr = torch.zeros_like(
                self.ctx_kv_indptr,
                device='cpu',
                pin_memory=prefer_pinned(),
            )

        # Only when MLA chunked prefill is enabled, we need to gather the full KV for indexer's logit computation.
        # These buffers will be allocated dynamically in Indexer.prepare() based on actual total_kv_len to save memory.
        if self.enable_context_mla_with_cached_kv:
            self.slot_mapping_fp8_fullkv = None
            self.slot_mapping_scale_fullkv = None

        # New generation buffers for dsa
        self.gen_cached_token_indptr = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_requests + 1, ),
            cache_name="gen_cached_token_indptr",
            dtype=torch.int64,
            capture_graph=capture_graph,
        )
        self.host_gen_cached_token_indptr = torch.zeros_like(
            self.gen_cached_token_indptr,
            device='cpu',
            pin_memory=prefer_pinned(),
        )
        self.gen_kv_indptr = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_requests + 1, ),
            cache_name="gen_kv_indptr",
            dtype=torch.int64,
            capture_graph=capture_graph,
        )
        self.host_gen_kv_indptr = torch.zeros_like(
            self.gen_kv_indptr,
            device='cpu',
            pin_memory=prefer_pinned(),
        )
        # Indexer metadata
        # Separate slot mappings for non-interleaved layout (flat byte indices)
        self.slot_mapping_fp8 = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens, ),
            cache_name="slot_mapping_fp8",
            dtype=torch.int64,
            capture_graph=capture_graph,
        )
        self.host_slot_mapping_fp8 = torch.zeros_like(
            self.slot_mapping_fp8,
            device='cpu',
            pin_memory=prefer_pinned(),
        )
        self.slot_mapping_scale = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens, ),
            cache_name="slot_mapping_scale",
            dtype=torch.int64,
            capture_graph=capture_graph,
        )
        self.host_slot_mapping_scale = torch.zeros_like(
            self.slot_mapping_scale,
            device='cpu',
            pin_memory=prefer_pinned(),
        )
        # Per-token request index buffer for topk_indices conversion
        self.req_idx_per_token = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens, ),
            cache_name="req_idx_per_token",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        # Block table for topk_indices conversion (shared for context and generation)
        self.block_table = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_requests, self.kv_cache_manager.max_blocks_per_seq),
            cache_name="block_table",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.scheduler_metadata_buffer = self.get_empty(
            self.cuda_graph_buffers,
            (self.num_sms + 1, 2),
            cache_name="scheduler_metadata_buffer",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        # When MTP runs without the expanded-tokens path, the same forward step
        # alternates between full-window calls (next_n == 1 + max_draft_tokens)
        # and per-token draft calls (next_n == 1). The 2D DeepGEMM metadata
        # API encodes next_n into the schedule, so the precomputed schedule
        # for one shape cannot be reused for the other. Maintain a second
        # buffer holding the schedule for the full next_n window; the draft
        # path keeps using `scheduler_metadata_buffer`. Always allocate (a
        # few KB) so transitions in `update_spec_dec_param` don't have to
        # special-case its existence.
        self.scheduler_metadata_buffer_full_next_n = self.get_empty(
            self.cuda_graph_buffers,
            (self.num_sms + 1, 2),
            cache_name="scheduler_metadata_buffer_full_next_n",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        # Pre-allocated 2D kv_lens buffer for the new DeepGEMM 2D context_lens
        # API. Shape: (max_num_sequences, 1 + max_draft_tokens). Each row
        # broadcasts the same kv_len across next_n positions; kernel reads a
        # slice per forward. Avoids per-forward .expand().contiguous()
        # allocations that would break CUDA graphs.
        self._create_kv_lens_2d_buffer(capture_graph=capture_graph)
        self.cu_seqlen_ks = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens, ),
            cache_name="cu_seqlen_ks",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.cu_seqlen_ke = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens, ),
            cache_name="cu_seqlen_ke",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        # Topk indices buffer to support skip indexer for requests with short sequence lengths
        if self.enable_indexer_skip:
            self.topk_indices_buffer = self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_tokens, self.num_sparse_topk),
                cache_name="topk_indices_buffer",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            self.host_topk_indices_buffer = torch.zeros_like(
                self.topk_indices_buffer,
                device='cpu',
                pin_memory=prefer_pinned(),
            )
        # Per-layer persistent buffers for heuristic TopK pre_idx.
        # Indexed by [local_layer_idx, generation_position, :].
        # The graph captures reads/writes on these stable-address buffers;
        # each replay's write becomes the next replay's read (feedback loop).
        self.enable_heuristic_topk = (
            self.sparse_attention_config.enable_heuristic_topk
            and get_sm_version() >= 100)
        if self.enable_heuristic_topk:
            num_local_layers = self.kv_cache_manager.num_local_layers
            self.heuristic_prev_topk = self.get_empty(
                self.cuda_graph_buffers,
                (num_local_layers, self.max_num_sequences,
                 self.num_sparse_topk),
                cache_name="heuristic_prev_topk",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            # Zero-initialize so the first decode step's pre_idx (kernel
            # adds +1 offset) points to index 1 — a valid but benign candidate.
            # Without this, uninitialized memory produces random hint indices.
            self.heuristic_prev_topk.zero_()
            # Scratch buffer for heuristic TopK kernel output values.
            # Pre-allocated with stable address for CUDA Graph compatibility
            # (replaces cudaMallocAsync/cudaFreeAsync inside the kernel launcher).
            # Shape: [max_gen_tokens, topK] where max_gen_tokens = max_batch * (1 + max_draft).
            max_gen_tokens = self.max_num_sequences * (1 +
                                                       self.max_draft_tokens)
            self.heuristic_scratch_values = self.get_empty(
                self.cuda_graph_buffers,
                (max_gen_tokens, self.num_sparse_topk),
                cache_name="heuristic_scratch_values",
                dtype=torch.float32,
                capture_graph=capture_graph,
            )

        # Create expanded buffers for MTP support
        self.create_expanded_buffers(capture_graph=capture_graph)

    def _create_kv_lens_2d_buffer(self, capture_graph=False):
        """Pre-allocated buffer for the DeepGEMM 2D context_lens API.

        Avoids per-forward .expand().contiguous() allocations that break CUDA
        graphs. The buffer is written in-place via .copy_() inside
        on_update_kv_lens so its address stays stable across replays.
        """
        self.kv_lens_cuda_2d = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences, 1 + self.max_draft_tokens),
            cache_name="kv_lens_cuda_2d",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )

    # TODO: remove these expanded buffers when fp8_paged_mqa_logits supports an arbitrary number of MTP draft tokens.
    def create_expanded_buffers(self, capture_graph=False):
        """Create expanded KV-length and block-table buffers for speculative decoding."""
        self.kv_lens_expanded_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences * (1 + self.max_draft_tokens), ),
            cache_name="kv_lens_expanded_cuda",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.kv_lens_expanded_host = torch.zeros_like(
            self.kv_lens_expanded_cuda,
            device='cpu',
            pin_memory=prefer_pinned(),
        )
        self.block_table_expanded = self.get_empty(
            self.cuda_graph_buffers,
            [
                self.max_num_sequences * (1 + self.max_draft_tokens),
                self.kv_cache_manager.max_blocks_per_seq
            ],
            cache_name="block_table_expanded",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.host_block_table_expanded = torch.zeros_like(
            self.block_table_expanded,
            device='cpu',
            pin_memory=prefer_pinned(),
        )
        self.scheduler_metadata_buffer_expanded = self.get_empty(
            self.cuda_graph_buffers,
            (self.num_sms + 1, 2),
            cache_name="scheduler_metadata_buffer_expanded",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )

    # This function is only used to create the expanded buffers when the max_draft_tokens is changed.
    # TODO: remove this function once fp8_paged_mqa_logits supports an arbitrary number of MTP draft tokens.
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
    ):
        """Update speculative decoding parameters and create expanded buffers."""
        super().update_spec_dec_param(batch_size, is_spec_decoding_enabled,
                                      is_spec_dec_tree,
                                      is_spec_dec_dynamic_tree, max_draft_len,
                                      max_total_draft_tokens, model_is_wrapped,
                                      spec_metadata, spec_tree_manager)
        self.max_draft_tokens = max_draft_len
        capture_graph = self.is_cuda_graph
        if self.kv_lens_cuda_2d.shape[1] != 1 + self.max_draft_tokens:
            self._create_kv_lens_2d_buffer(capture_graph=capture_graph)
        init_shape = self.kv_lens_expanded_host.shape[0]
        if self.max_num_sequences * (1 + self.max_draft_tokens) != init_shape:
            self.create_expanded_buffers(capture_graph=capture_graph)
            # Resize heuristic scratch buffer for new max_draft_tokens.
            if self.enable_heuristic_topk:
                max_gen_tokens = self.max_num_sequences * (
                    1 + self.max_draft_tokens)
                self.heuristic_scratch_values = self.get_empty(
                    self.cuda_graph_buffers,
                    (max_gen_tokens, self.num_sparse_topk),
                    cache_name="heuristic_scratch_values",
                    dtype=torch.float32,
                    capture_graph=capture_graph,
                )

    def _invalidate_pool_view_cache(self):
        """Invalidate the cached pool view and related step-invariant values.

        Must be called at the start of each forward step (in prepare()) so that
        _ensure_pool_view_cached() recomputes them for the new batch.
        """
        self._pool_cache_valid = False

    def _ensure_pool_view_cached(self):
        """Compute and cache values used by
        transform_local_topk_and_prepare_pool_view().

        These values (pool view, stride factor, block table slices, request
        index slices) are constant across all layers sharing the same KV pool
        and batch dimensions within a forward pass. Caching them avoids
        redundant Python/CUDA overhead per layer.

        Safety: _invalidate_pool_view_cache() is called unconditionally at the
        start of every step (prepare() and on_update_kv_lens()), so the boolean
        flag is always cleared before the first per-layer call within a step.
        """
        if self._pool_cache_valid and self._cached_kv_mgr_id == id(
                self.kv_cache_manager):
            return

        pool = self.kv_cache_manager.get_unique_primary_pool()
        kv_cache_manager = self.kv_cache_manager
        num_blocks, num_layers, _, _ = pool.shape
        self._cached_tokens_per_block = kv_cache_manager.tokens_per_block
        head_dim = kv_cache_manager.head_dim
        self._cached_pool_view = pool.squeeze(2).view(-1, 1, head_dim)
        self._cached_stride_factor = (num_layers *
                                      self._cached_tokens_per_block)
        self._cached_block_table_ctx = self.block_table[:self.num_contexts]
        self._cached_block_table_gen = self.block_table[self.num_contexts:self.
                                                        num_seqs]
        self._cached_req_idx_ctx = self.req_idx_per_token[:self.num_ctx_tokens]
        self._cached_req_idx_gen = (
            self.req_idx_per_token[self.num_ctx_tokens:self.num_tokens] -
            self.num_contexts)
        self._cached_kv_mgr_id = id(kv_cache_manager)
        self._pool_cache_valid = True

    @maybe_compile(dynamic=True)
    def _get_dense_topk_indices(self, seq_lens, kv_lens, num_tokens):
        device = kv_lens.device
        past_kv_lens = kv_lens - seq_lens
        # get position ids
        seq_ends = torch.cumsum(seq_lens, dim=0)
        seq_starts = seq_ends - seq_lens
        per_seq_offsets = past_kv_lens - seq_starts  # Shape: [batch_size]
        global_indices = torch.arange(num_tokens, device=device)
        batch_indices = torch.searchsorted(seq_ends,
                                           global_indices,
                                           side='right')
        repeated_offsets = per_seq_offsets[batch_indices]
        position_ids = global_indices + repeated_offsets
        # get the dense topk indices with causal mask
        range_row = torch.arange(self.num_sparse_topk, device=device)
        mask = range_row <= position_ids.unsqueeze(1)
        return torch.where(mask, range_row, -1)

    def prepare_dense_topk_indices(self,
                                   kv_lens,
                                   device=False):  # device=False means use CPU
        """Prepare dense TopK indices for short sequences that skip the indexer."""

        if self.num_contexts > 0 and self.skip_indexer_for_ctx_reqs:
            ctx_range = slice(self.num_ctx_tokens)
            if device:
                self.topk_indices_buffer[ctx_range, :].copy_(
                    self._get_dense_topk_indices(
                        self.seq_lens_cuda[:self.num_contexts],
                        kv_lens[:self.num_contexts], self.num_ctx_tokens),
                    non_blocking=True)
            else:
                self.host_topk_indices_buffer[
                    ctx_range, :] = self._get_dense_topk_indices(
                        self.seq_lens[:self.num_contexts],
                        kv_lens[:self.num_contexts], self.num_ctx_tokens)
                self.topk_indices_buffer[ctx_range, :].copy_(
                    self.host_topk_indices_buffer[ctx_range, :],
                    non_blocking=True)

        if self.num_generations > 0 and self.skip_indexer_for_gen_reqs:
            gen_range = slice(self.num_ctx_tokens, self.num_tokens)
            if device:
                self.topk_indices_buffer[gen_range, :].copy_(
                    self._get_dense_topk_indices(
                        self.seq_lens_cuda[self.num_contexts:self.num_seqs],
                        kv_lens[self.num_contexts:self.num_seqs],
                        self.num_tokens - self.num_ctx_tokens),
                    non_blocking=True)
            else:
                self.host_topk_indices_buffer[
                    gen_range, :] = self._get_dense_topk_indices(
                        self.seq_lens[self.num_contexts:self.num_seqs],
                        kv_lens[self.num_contexts:self.num_seqs],
                        self.num_tokens - self.num_ctx_tokens)
                self.topk_indices_buffer[gen_range, :].copy_(
                    self.host_topk_indices_buffer[gen_range, :],
                    non_blocking=True)

    def _get_pool_block_indices(self) -> torch.Tensor:
        """Extract memory pool block indices from host_kv_cache_block_offsets.

        The C++ setOffsets() encodes offsets as:
            encoded = memPoolBlockIndex * numLayers * kvFactor
        For SELFKONLY (MLA/DSA), kvFactor=1, so:
            memPoolBlockIndex = encoded // num_local_layers

        Returns a (num_seqs, max_blocks_per_seq) int32 CPU tensor with valid
        pool indices clamped to [0, blocks_in_primary_pool - 1].
        """
        num_local_layers = self.kv_cache_manager.num_local_layers
        max_pool_idx = self.kv_cache_manager.blocks_in_primary_pool - 1
        # DSA uses SELFKONLY mode where only key cache is stored (kv_factor=1).
        # host_kv_cache_block_offsets shape: (num_pools, max_batch*beam, 2, max_blocks_per_seq)
        # Note: dim=2 is always 2 in the tensor layout (K and V slots), but for
        # SELFKONLY only the K slot (index 0) contains valid data.
        assert self.kv_cache_manager.kv_factor == 1, \
            f"DSA requires SELFKONLY mode (kv_factor=1), got kv_factor={self.kv_cache_manager.kv_factor}"
        # Pool 0, first num_seqs entries, field 0 (key offsets)
        encoded = self.kv_cache_manager.host_kv_cache_block_offsets[
            0, :self.num_seqs, 0, :]
        pool_indices = encoded // num_local_layers
        # Clamp for safety: handles garbage padding from torch.empty in uninitialized slots
        pool_indices = pool_indices.clamp(min=0,
                                          max=max_pool_idx).to(torch.int32)
        return pool_indices

    def prepare(self):
        """Prepare DSA metadata: compute slot mappings, block tables, and prefill chunks."""
        super().prepare()
        self._invalidate_pool_view_cache()

        # Get kv lengths
        assert self.kv_cache_params.use_cache is True, "DSA requires use_cache to be True"
        cached_token_lens = torch.tensor(
            self.kv_cache_params.num_cached_tokens_per_seq,
            dtype=torch.int,
            device='cpu',
        )
        if self.enable_helix:
            # For Helix CP, inactive ranks only attend to previously cached
            # tokens (no new token appended), while active ranks add new tokens.
            # This mirrors the kv_lens logic in TrtllmAttentionMetadata.prepare().
            active_rank = ~self.helix_is_inactive_rank_cpu[:self.num_seqs]
            kv_lens = cached_token_lens.clone()
            kv_lens[active_rank] += self.seq_lens_kv[active_rank]
        else:
            kv_lens = cached_token_lens + self.seq_lens_kv

        # Prepare to support skip indexer
        num_extra_kv_tokens = self.kv_cache_params.num_extra_kv_tokens
        if self.num_contexts > 0 and self.enable_indexer_skip:
            # Minus the number of extra KV tokens because when using one-model MTP, the
            # draft layers needs more KV tokens for the next draft forwards.
            self.skip_indexer_for_ctx_reqs = kv_lens[:self.num_contexts].max(
            ).item() <= self.num_sparse_topk - num_extra_kv_tokens
        else:
            self.skip_indexer_for_ctx_reqs = False

        if self.num_generations > 0 and self.enable_indexer_skip:
            # Minus the number of extra KV tokens because when using one-model MTP, the
            # draft layers needs more KV tokens for the next draft forwards.
            self.skip_indexer_for_gen_reqs = kv_lens[
                self.num_contexts:self.num_seqs].max().item(
                ) <= self.num_sparse_topk - num_extra_kv_tokens
        else:
            self.skip_indexer_for_gen_reqs = False
        self.prepare_dense_topk_indices(kv_lens)

        # Build indexer_k_cache_block_offsets using pool block indices derived
        # from host_kv_cache_block_offsets (populated by super().prepare()).
        # This correctly resolves block IDs to memory pool indices, which is
        # required when host cache offload is enabled (block IDs != pool indices
        # for onboarded secondary blocks).
        if self.kv_cache_manager is not None:
            pool_indices = self._get_pool_block_indices()
            self.host_indexer_k_cache_block_offsets[:self.num_seqs].copy_(
                pool_indices)
            self.indexer_k_cache_block_offsets[:self.num_seqs].copy_(
                self.host_indexer_k_cache_block_offsets[:self.num_seqs],
                non_blocking=True)
            # Safety clamp: prevent OOB from CUDA graph padding entries which
            # may contain stale negative or out-of-range values after block
            # eviction/onboarding with host cache offload.
            self.indexer_k_cache_block_offsets.clamp_(min=0)

        # Build req_idx_per_token for topk_indices conversion
        host_req_idx_per_token = torch.repeat_interleave(torch.arange(
            self.num_seqs, dtype=torch.int32),
                                                         self.seq_lens,
                                                         dim=0)
        self.req_idx_per_token[:self.num_tokens].copy_(host_req_idx_per_token,
                                                       non_blocking=True)

        # Build block_table for topk_indices conversion (actual block allocation)
        if self.kv_cache_manager is not None:
            tokens_per_block = self.kv_cache_manager.tokens_per_block
            num_blocks_per_seq = (kv_lens[:self.num_seqs] + tokens_per_block -
                                  1) // tokens_per_block
            max_blocks_used = num_blocks_per_seq.max().item(
            ) if self.num_seqs > 0 else 1
            # pool_indices already has correct values; set padding to -1
            host_block_table = pool_indices[:, :max_blocks_used].clone()
            for i in range(self.num_seqs):
                if num_blocks_per_seq[i] < max_blocks_used:
                    host_block_table[i, num_blocks_per_seq[i]:] = -1
            # Copy to GPU
            self.block_table[:self.num_seqs, :max_blocks_used].copy_(
                host_block_table, non_blocking=True)

        # For mla_rope_append_paged_kv_assign_q
        if self.num_contexts > 0:
            self.num_ctx_cached_tokens = cached_token_lens[:self.
                                                           num_contexts].sum(
                                                           ).item()
            self.max_ctx_kv_len = kv_lens[:self.num_contexts].max().item()
            self.max_ctx_seq_len = self.seq_lens[:self.num_contexts].max().item(
            )
            # context cached token indptr
            torch.cumsum(
                cached_token_lens[:self.num_contexts],
                dim=0,
                dtype=torch.int64,
                out=self.host_ctx_cached_token_indptr[1:self.num_contexts + 1])
            self.ctx_cached_token_indptr[:self.num_contexts + 1].copy_(
                self.host_ctx_cached_token_indptr[:self.num_contexts + 1],
                non_blocking=True)
            # context kv indptr
            torch.cumsum(kv_lens[:self.num_contexts],
                         dim=0,
                         dtype=torch.int64,
                         out=self.host_ctx_kv_indptr[1:self.num_contexts + 1])
            self.ctx_kv_indptr[:self.num_contexts + 1].copy_(
                self.host_ctx_kv_indptr[:self.num_contexts + 1],
                non_blocking=True)
        else:
            self.num_ctx_cached_tokens = 0
            self.max_ctx_kv_len = 0
            self.max_ctx_seq_len = 0

        if self.num_generations > 0:
            self.max_gen_seq_len = self.seq_lens[self.num_contexts:self.
                                                 num_seqs].max().item()
            # generation cached token indptr
            torch.cumsum(
                cached_token_lens[self.num_contexts:self.num_seqs],
                dim=0,
                dtype=torch.int64,
                out=self.host_gen_cached_token_indptr[1:self.num_generations +
                                                      1])
            self.gen_cached_token_indptr[:self.num_generations + 1].copy_(
                self.host_gen_cached_token_indptr[:self.num_generations + 1],
                non_blocking=True)
            # generation kv indptr
            torch.cumsum(kv_lens[self.num_contexts:self.num_seqs],
                         dim=0,
                         dtype=torch.int64,
                         out=self.host_gen_kv_indptr[1:self.num_generations +
                                                     1])
            self.gen_kv_indptr[:self.num_generations + 1].copy_(
                self.host_gen_kv_indptr[:self.num_generations + 1],
                non_blocking=True)
        else:
            self.max_gen_seq_len = 0

        # Because the fp8_paged_mqa_logits only supports seq_len == 1/2/4 (i.e., max_draft_tokens == 0/1/3) on sm100, and
        # seq_len == 1/2 (i.e., max_draft_tokens == 0/1) on sm90, for other cases, we need to flatten the q tensor and
        # expand the kv_lens and block_table for MTP support.
        # TODO:
        # - No distinction between sm90 and sm100 is needed once MTP3 is supported on sm90.
        # - Remove this once fp8_paged_mqa_logits supports an arbitrary number of MTP draft tokens.
        self.use_expanded_buffers_for_mtp = (
            (self.max_draft_tokens > 1 and get_sm_version() == 90)
            or ((self.max_draft_tokens == 2 or self.max_draft_tokens > 3)
                and get_sm_version() >= 100))
        if self.use_expanded_buffers_for_mtp:
            # Expand kv_lens_cuda (only generation)
            num_tokens = self.num_generations * (1 + self.max_draft_tokens)
            gen_kv_lens = kv_lens[self.num_contexts:self.num_seqs]
            gen_kv_lens_expanded = torch.stack([gen_kv_lens] *
                                               (1 + self.max_draft_tokens),
                                               dim=0)
            gen_kv_lens_expanded = gen_kv_lens_expanded.transpose(
                0, 1).contiguous().flatten()
            self.kv_lens_expanded_host[:num_tokens].copy_(gen_kv_lens_expanded)
            self.kv_lens_expanded_cuda[:num_tokens].copy_(
                self.kv_lens_expanded_host[:num_tokens], non_blocking=True)

            # Expand indexer_k_cache_block_offsets (only generation)
            # host_indexer_k_cache_block_offsets already contains correct pool
            # indices from _get_pool_block_indices() above.
            if self.kv_cache_manager is not None and self.num_generations > 0:
                max_len = self.host_indexer_k_cache_block_offsets.shape[1]
                gen_block_tensor = self.host_indexer_k_cache_block_offsets[
                    self.num_contexts:self.num_seqs, :max_len]
                expanded_blocks = gen_block_tensor.repeat_interleave(
                    1 + self.max_draft_tokens, dim=0)
                self.host_block_table_expanded[:num_tokens, :max_len].copy_(
                    expanded_blocks, non_blocking=True)
                self.block_table_expanded[:num_tokens].copy_(
                    self.host_block_table_expanded[:num_tokens],
                    non_blocking=True)
                self.block_table_expanded.clamp_(min=0)

        # Prepare metadata for indexer
        Indexer.prepare(metadata=self)

    def on_update_kv_lens(self):
        """Refresh indexer slot mappings after KV lengths change at runtime."""
        # After changing the kv_lens/kv_lens_cuda, we may need to update other metadatas.
        # Especially for the changes in the _preprocess_inputs() of model_engine.py.
        #
        # NOTE:
        # In overlap scheduler + speculative decoding, kv_lens_cuda can be corrected at runtime
        # (inside _preprocess_inputs) to account for variable accepted tokens. The indexer
        # slot_mapping_* buffers also depend on these effective cached lengths. If we do not
        # refresh slot mappings here, indexer K-cache updates can be written with stale offsets.

        # _preprocess_inputs() also uses this as a general hook to "invalidate per-forward-pass
        # caches so they are recomputed (and captured) on every _forward_step". Invalidate the
        # pool_view cache here so it is recomputed on the next
        # transform_local_topk_and_prepare_pool_view() call.
        self._invalidate_pool_view_cache()

        if self.kv_cache_manager is not None and self.num_tokens > 0:
            seq_lens = self.seq_lens_cuda[:self.num_seqs]
            # Runtime cached lengths after overlap/spec-dec correction.
            start_positions = self.kv_lens_cuda[:self.num_seqs] - seq_lens

            # Reuse request-per-token mapping prepared in metadata.prepare().
            # This avoids repeat_interleave in graph-capture mode.
            req_indices = self.req_idx_per_token[:self.num_tokens].to(
                dtype=torch.int64)
            seq_starts = torch.cumsum(
                seq_lens, dim=0, dtype=torch.int64) - seq_lens.to(torch.int64)
            token_offsets = torch.arange(
                self.num_tokens, device=seq_lens.device,
                dtype=torch.int64) - seq_starts[req_indices]

            global_positions = start_positions[req_indices] + token_offsets
            # Under FP4 the indexer cache stores two E2M1 codes per byte, so
            # the per-token data footprint is head_dim // 2; otherwise it is
            # head_dim (one FP8 byte per element). Feed the real byte count
            # into _compute_slot_mappings so scatter/gather see offsets that
            # match the pool layout produced by createIndexerKCachePools.
            use_fp4 = self.kv_cache_manager.use_fp4
            index_head_dim = self.kv_cache_manager.index_head_dim
            data_bytes_per_token = index_head_dim // 2 if use_fp4 else index_head_dim
            fp8_indices, scale_indices = _compute_slot_mappings(
                global_positions,
                self.indexer_k_cache_block_offsets,
                req_indices,
                index_head_dim,
                self.kv_cache_manager.tokens_per_block,
                self.kv_cache_manager.quant_block_size,
                data_bytes_per_token=data_bytes_per_token,
            )
            self.slot_mapping_fp8[:self.num_tokens] = fp8_indices
            self.slot_mapping_scale[:self.num_tokens] = scale_indices

        if self.num_generations > 0:
            torch.cumsum(
                self.kv_lens_cuda[self.num_contexts:self.
                                  num_seqs],  # num_contexts should be 0
                dim=0,
                dtype=torch.int64,
                out=self.gen_kv_indptr[1:self.num_generations + 1])
            torch.cumsum(
                (self.kv_lens_cuda[self.num_contexts:self.num_seqs] -
                 self.seq_lens_cuda[self.num_contexts:self.num_seqs]),
                dim=0,
                dtype=torch.int64,
                out=self.gen_cached_token_indptr[1:self.num_generations + 1])
            # Write 2D kv_lens in-place (broadcast same kv_len across next_n
            # positions). .expand() returns a view and .copy_() writes into the
            # pre-allocated destination, so this is CUDA-graph-friendly.
            gen_kv_lens = self.kv_lens_cuda[self.num_contexts:self.num_seqs]
            next_n_cap = self.kv_lens_cuda_2d.shape[1]
            self.kv_lens_cuda_2d[:self.num_generations, :next_n_cap].copy_(
                gen_kv_lens.unsqueeze(-1).expand(-1, next_n_cap))
            # Build the next_n=1 schedule (used by MTP draft layers and any
            # non-MTP forward). Reshape the contiguous gen slice of
            # kv_lens_cuda to (num_gen, 1) — slicing kv_lens_cuda_2d's first
            # column would be non-contiguous and would fail the metadata
            # kernel's is_contiguous assertion.
            context_lens_next_n1 = gen_kv_lens.view(-1, 1)
            scheduler_metadata_buffer = get_paged_mqa_logits_metadata(
                context_lens_next_n1, self.kv_cache_manager.tokens_per_block,
                self.num_sms)
            self.scheduler_metadata_buffer.copy_(scheduler_metadata_buffer,
                                                 non_blocking=True)
            # When MTP is on without the expanded-tokens path, also populate
            # the full-next_n schedule for the main forward call. The metadata
            # kernel reads next_n from context_lens.size(1), so we must pass
            # the wider slice here.
            if (self.max_draft_tokens > 0
                    and not self.use_expanded_buffers_for_mtp):
                context_lens_full_next_n = self.kv_lens_cuda_2d[:self.
                                                                num_generations, :
                                                                next_n_cap]
                scheduler_metadata_buffer_full_next_n = get_paged_mqa_logits_metadata(
                    context_lens_full_next_n,
                    self.kv_cache_manager.tokens_per_block, self.num_sms)
                self.scheduler_metadata_buffer_full_next_n.copy_(
                    scheduler_metadata_buffer_full_next_n, non_blocking=True)
            if self.use_expanded_buffers_for_mtp:
                num_draft_tokens = 1 + self.max_draft_tokens
                num_tokens = self.num_generations * num_draft_tokens
                kv_lens_expanded = torch.stack([gen_kv_lens] * num_draft_tokens,
                                               dim=0)
                self.kv_lens_expanded_cuda[:num_tokens] = \
                    kv_lens_expanded.transpose(0, 1).contiguous().flatten()
                # New API requires 2D; each expanded token becomes a (1,) row.
                kv_lens_expanded_2d = self.kv_lens_expanded_cuda[:
                                                                 num_tokens].view(
                                                                     -1, 1)
                scheduler_metadata_buffer_expanded = get_paged_mqa_logits_metadata(
                    kv_lens_expanded_2d, self.kv_cache_manager.tokens_per_block,
                    self.num_sms)
                self.scheduler_metadata_buffer_expanded.copy_(
                    scheduler_metadata_buffer_expanded, non_blocking=True)
        self.prepare_dense_topk_indices(self.kv_lens_cuda, device=True)

    def update_for_spec_dec(self):
        """Reset context/generation counters and refresh slot mappings for speculative decoding."""
        super().update_for_spec_dec()
        # host
        self.max_ctx_kv_len = 0
        self.num_ctx_cached_tokens = 0
        self.max_gen_seq_len = 1

        # device
        self.on_update_kv_lens()


@maybe_compile(dynamic=True)
def _scale(weights: torch.Tensor, q_scale: torch.Tensor,
           s: float) -> torch.Tensor:
    """Scale attention weights by quantization scale and constant factor."""
    return weights * q_scale.squeeze(-1) * s


@maybe_compile(dynamic=True)
def _to_float(hidden_states: torch.Tensor) -> torch.Tensor:
    """Cast hidden states to float32 for TF32 GEMM computation."""
    return hidden_states.float()


@contextmanager
def _tf32_matmul_enabled():
    """Temporarily enable TF32 tensor cores for FP32 matmul in this scope.

    Forces PyTorch/cuBLASLt to use CUBLAS_COMPUTE_32F_FAST_TF32, which
    guarantees TF32 tensor cores. Plain CUBLAS_COMPUTE_32F (used by
    torch.ops.trtllm.cublas_mm) falls back to SIMT SGEMM on CUDA cores
    based on cuBLASLt heuristics for small M.
    """
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


class Indexer(nn.Module):
    """DSA sparse attention indexer that selects top-K KV cache entries per token."""

    def __init__(self,
                 quant_config: Optional[QuantConfig],
                 pos_embd_params: Optional[PositionalEmbeddingParams],
                 mla_params: Optional[MLAParams],
                 skip_create_weights_in_init: bool,
                 sparse_attention_config: "SparseAttentionConfig",
                 dtype: Optional[torch.dtype],
                 layer_idx: int = 0,
                 aux_stream: Optional[torch.cuda.Stream] = None):
        """Initialize indexer with projection weights, norms, and TopK configuration."""
        super().__init__()
        self.hidden_size = mla_params.hidden_size
        self.q_lora_rank = mla_params.q_lora_rank
        self.rope_dim = mla_params.qk_rope_head_dim
        self.n_heads = sparse_attention_config.index_n_heads  # 64
        self.head_dim = sparse_attention_config.index_head_dim  # 128
        self.index_topk = sparse_attention_config.index_topk  # 2048
        self.layer_idx = layer_idx

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
            dtype=torch.float32,
            quant_config=None,
            skip_create_weights_in_init=skip_create_weights_in_init,
            use_custom_cublas_mm=True)
        self.k_norm = LayerNorm(hidden_size=self.head_dim, eps=1e-6)
        self.weights_proj = Linear(
            self.hidden_size,
            self.n_heads,
            bias=False,
            dtype=torch.float32,
            quant_config=None,
            skip_create_weights_in_init=skip_create_weights_in_init,
            use_custom_cublas_mm=True)

        # Fused wk + weights_proj weight for single F.linear FP32 GEMM under allow_tf32.
        # Maps to TF32 tensor cores on Ampere+.
        self._fused_wk_wp_weight: Optional[torch.Tensor] = None

        indexer_rope_interleave = getattr(sparse_attention_config,
                                          'indexer_rope_interleave', False)
        self.rotary_emb = RotaryEmbedding(
            pos_embd_params.rope,
            head_dim=self.rope_dim,
            is_neox=not indexer_rope_interleave,
        )

        self.softmax_scale = self.head_dim**-0.5
        # TODO: make it configurable from hf config
        self.scale_fmt = "ue8m0"
        # indexer_k_dtype controls both Q and K precision. DeepGEMM's
        # fp8_fp4_mqa_logits / fp8_fp4_paged_mqa_logits kernels only dispatch
        # to FP4xFP4 or FP8xFP8 (no mixed-precision variant). The DeepGEMM
        # kernel asserts SM100 + head_dim=128 at launch time under FP4.
        self.use_fp4 = sparse_attention_config.indexer_k_dtype == "fp4"
        self.aux_stream = aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]
        self.use_cute_dsl_topk = (sparse_attention_config.use_cute_dsl_topk
                                  and IS_CUTLASS_DSL_AVAILABLE)
        self.weight_scale_factor = self.softmax_scale * self.n_heads**-0.5

        self._enable_heuristic_topk = (
            sparse_attention_config.enable_heuristic_topk
            and get_sm_version() >= 100)

        if self.use_cute_dsl_topk and layer_idx == 0:
            from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops

            # the dtype of topk input tensor, which is float32 now.
            # Note, need to update it if the dtype of topk input tensor is changed.
            cute_dsl_custom_ops.warmup_cute_dsl_indexer_topk(
                dtype=torch.float32, top_k=self.index_topk)

        if self._enable_heuristic_topk and layer_idx == 0:
            # Populate static caches (sm_count, L2 cache size) inside the C++
            # Scheme X dispatcher before any CUDA Graph capture so the host
            # attribute queries do not end up frozen into a captured graph.
            from tensorrt_llm._torch.custom_ops import cpp_custom_ops
            cpp_custom_ops.warmup_heuristic_topk_decode(top_k=self.index_topk)

    def post_load_weights(self):
        """Fuse wk + weights_proj into single FP32 weight for F.linear GEMM under allow_tf32 (TF32 tensor cores on Ampere+)."""
        # wk: [head_dim, hidden_size] + weights_proj: [n_heads, hidden_size]
        # → fused: [head_dim + n_heads, hidden_size]
        self._fused_wk_wp_weight = torch.cat(
            [self.wk.weight.data, self.weights_proj.weight.data], dim=0)

    @staticmethod
    def prepare_one_prefill_chunk(
        metadata: DSAtrtllmAttentionMetadata,
        chunk_specs: List[Tuple[int, int, int, int]],
    ) -> IndexerPrefillChunkMetadata:
        """
        Build metadata for one prefill chunk for indexer forward pass.
        Handles both multi-request chunks and intra-request Q-block chunks.

        Args:
            metadata: Attention metadata
            chunk_specs: List of (req_idx, token_start_in_req, token_end_in_req, req_cum_start)
                        - token_start_in_req, token_end_in_req are indices into current batch context tokens
                        - For multi-request: multiple specs from different requests (full requests)
                        - For intra-request: single spec from one request's Q-block

        Note: Cached token counts are derived from metadata.host_ctx_cached_token_indptr
        """
        device = metadata.cu_seqlen_ks.device
        if len(chunk_specs) == 1:
            # Single request or intra-request Q-block
            req_idx, token_start_in_req, token_end_in_req, req_cum_start = chunk_specs[
                0]
            num_q_tokens = token_end_in_req - token_start_in_req

            # Get cached token count for this request from metadata
            num_cached = (
                metadata.host_ctx_cached_token_indptr[req_idx + 1] -
                metadata.host_ctx_cached_token_indptr[req_idx]).item()

            # For intra-request chunks: Q block attends to all previous K in the request
            # Q tokens [token_start_in_req:token_end_in_req] within the request's current tokens
            # K tokens [0:num_cached + token_end_in_req] within the request (causal attention)
            cu_seqlen_ks = torch.zeros(num_q_tokens,
                                       dtype=torch.int32,
                                       device='cpu')
            cu_seqlen_ke = torch.arange(token_start_in_req + 1,
                                        token_end_in_req + 1,
                                        dtype=torch.int32,
                                        device='cpu') + num_cached

            # Q token range in batch (indices into context tokens in the current batch)
            token_start = req_cum_start + token_start_in_req
            token_end = req_cum_start + token_end_in_req

            # K token range: index into full KV slot mapping (cached + current batch context tokens)
            kv_offset_in_extended = metadata.host_ctx_kv_indptr[req_idx].item()
            total_kv_for_req = num_cached + token_end_in_req
            k_token_start = kv_offset_in_extended
            k_token_end = kv_offset_in_extended + total_kv_for_req

        else:
            # Multi-request chunk: batch multiple full requests together
            # Extract sequence lengths for these requests
            req_seq_lens = []
            req_cached_lens = []
            first_req_idx = chunk_specs[0][0]

            for spec in chunk_specs:
                req_idx, token_start_in_req, token_end_in_req, _ = spec
                req_seq_lens.append(token_end_in_req - token_start_in_req)
                # Get cached token count from metadata
                num_cached = (
                    metadata.host_ctx_cached_token_indptr[req_idx + 1] -
                    metadata.host_ctx_cached_token_indptr[req_idx]).item()
                req_cached_lens.append(num_cached)

            req_seq_lens_tensor = torch.tensor(req_seq_lens,
                                               dtype=torch.int32,
                                               device='cpu')
            req_cached_lens_tensor = torch.tensor(req_cached_lens,
                                                  dtype=torch.int32,
                                                  device='cpu')
            num_q_tokens = sum(req_seq_lens)

            # Compute causal attention bounds for batched requests
            cu_seqlen_ks, cu_seqlen_ke = compute_cu_seqlen_kv_bounds_with_cache(
                req_seq_lens_tensor, len(chunk_specs), num_q_tokens,
                req_cached_lens_tensor)

            # Global Q token ranges (indices into ctx tokens in the current batch)
            token_start = chunk_specs[0][3]  # req_cum_start of first request
            token_end = token_start + num_q_tokens

            # K token range: index into full kv slot mapping (cached + current ctx tokens within the batch)
            kv_offset_in_extended = metadata.host_ctx_kv_indptr[
                first_req_idx].item()
            total_kv_len = sum(req_seq_lens_tensor +
                               req_cached_lens_tensor).item()
            k_token_start = kv_offset_in_extended
            k_token_end = kv_offset_in_extended + total_kv_len

        assert cu_seqlen_ks.shape[0] == num_q_tokens == token_end - token_start, \
            f"Indexer.prepare_one_prefill_chunk - cu_seqlen_ks length mismatch: {cu_seqlen_ks.shape[0]} != {num_q_tokens}"
        assert cu_seqlen_ke.shape[0] == num_q_tokens == token_end - token_start, \
            f"Indexer.prepare_one_prefill_chunk - cu_seqlen_ke length mismatch: {cu_seqlen_ke.shape[0]} != {num_q_tokens}"

        return IndexerPrefillChunkMetadata(
            cu_seqlen_ks=cu_seqlen_ks.to(device, non_blocking=True),
            cu_seqlen_ke=cu_seqlen_ke.to(device, non_blocking=True),
            token_start=token_start,
            token_end=token_end,
            k_token_start=k_token_start,
            k_token_end=k_token_end,
        )

    @staticmethod
    def recompute_slot_mappings(metadata: DSAtrtllmAttentionMetadata):
        """Recompute only slot_mapping_fp8/scale from the current block offsets.

        This is the subset of prepare() that maps each token to its flat cache
        position.  It is safe to call in isolation (e.g. during draft KV-cache
        replay) because it only touches slot-mapping buffers and reads
        block-offset / sequence metadata that the caller has already set up.
        """
        kv_cache_manager = metadata.kv_cache_manager
        if kv_cache_manager is None or not hasattr(kv_cache_manager,
                                                   'index_head_dim'):
            return

        seq_lens = metadata.seq_lens
        head_dim = kv_cache_manager.index_head_dim
        tokens_per_block = kv_cache_manager.tokens_per_block
        quant_block_size = kv_cache_manager.quant_block_size
        use_fp4 = kv_cache_manager.use_fp4
        # FP4 packs two E2M1 codes per byte; FP8 stores one byte per element.
        data_bytes_per_token = head_dim // 2 if use_fp4 else head_dim
        cached_tokens = metadata.kv_cache_params.num_cached_tokens_per_seq
        total_tokens = seq_lens.sum().item()

        start_positions = torch.tensor(cached_tokens, dtype=torch.int32)
        batch_size = len(metadata.request_ids)

        req_indices = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.int64, device='cpu'), seq_lens)

        token_offsets = torch.cat([
            torch.arange(seq_lens[i].item(), dtype=torch.int64, device='cpu')
            for i in range(batch_size)
        ])

        global_positions = start_positions[req_indices] + token_offsets

        fp8_flat_indices, scale_flat_indices = _compute_slot_mappings(
            global_positions,
            metadata.host_indexer_k_cache_block_offsets,
            req_indices,
            head_dim,
            tokens_per_block,
            quant_block_size,
            data_bytes_per_token=data_bytes_per_token,
        )

        metadata.host_slot_mapping_fp8[:total_tokens] = fp8_flat_indices
        metadata.host_slot_mapping_scale[:total_tokens] = scale_flat_indices

        metadata.slot_mapping_fp8[:total_tokens].copy_(
            metadata.host_slot_mapping_fp8[:total_tokens], non_blocking=True)
        metadata.slot_mapping_scale[:total_tokens].copy_(
            metadata.host_slot_mapping_scale[:total_tokens], non_blocking=True)

    @staticmethod
    def prepare(metadata: DSAtrtllmAttentionMetadata):
        """
        Prepare indexer for the forward pass.
        This should be called during metadata.prepare() stage.

        - Computes slot_mapping for KV cache updates
        - Prepares schedule_metadata for fp8_paged_mqa_logits
        - Stores generation request IDs for decode phase
        """
        kv_cache_manager = metadata.kv_cache_manager
        num_contexts = metadata.num_contexts
        num_generations = metadata.num_generations
        num_ctx_tokens = metadata.num_ctx_tokens
        seq_lens = metadata.seq_lens
        tokens_per_block = kv_cache_manager.tokens_per_block

        # Prepare for prefill phase if there are context requests
        if num_contexts > 0:
            # Compute attention window bounds for each query token in batched sequences
            # cu_seqlen_ks[i]: start index in global KV for query token i
            # cu_seqlen_ke[i]: end index (exclusive) in global KV for query token i
            host_seq_lens = seq_lens[:num_contexts]
            cached_tokens = metadata.kv_cache_params.num_cached_tokens_per_seq
            host_cached_tokens = torch.tensor(cached_tokens[:num_contexts],
                                              dtype=torch.int32,
                                              device='cpu')

            # When MLA chunked prefill is active, it already handles chunking
            # Indexer should just process the current MLA chunk as a single chunk
            has_mla_chunked_prefill = (
                metadata.enable_context_mla_with_cached_kv
                and metadata.runtime_features.chunked_prefill)

            if has_mla_chunked_prefill:
                # MLA chunked prefill is active - use single-chunk pattern for
                # indexer prefill chunks.
                chunk_specs = [(i, 0, host_seq_lens[i].item(),
                                host_seq_lens[:i].sum().item() if i > 0 else 0)
                               for i in range(num_contexts)]
                metadata.indexer_prefill_chunks = [
                    Indexer.prepare_one_prefill_chunk(
                        metadata,
                        chunk_specs,
                    )
                ]
            else:
                # Use indexer's own chunking logic to prevent L^2 complexity of indexer MQA logits computation for long sequences.
                # This is only used when MLA chunked prefill is not enabled.
                chunk_groups = split_prefill_chunks(
                    host_seq_lens,
                    metadata.indexer_max_chunk_size,
                    start_idx=0,
                )

                if len(chunk_groups
                       ) > 1 or metadata.enable_context_mla_with_cached_kv:
                    metadata.indexer_prefill_chunks = [
                        Indexer.prepare_one_prefill_chunk(
                            metadata,
                            chunk_specs,
                        ) for chunk_specs in chunk_groups
                    ]
                else:
                    metadata.indexer_prefill_chunks = None

            host_cu_seqlen_ks, host_cu_seqlen_ke = compute_cu_seqlen_kv_bounds_with_cache(
                host_seq_lens, num_contexts, num_ctx_tokens, host_cached_tokens)

            metadata.cu_seqlen_ks[:num_ctx_tokens].copy_(host_cu_seqlen_ks,
                                                         non_blocking=True)
            metadata.cu_seqlen_ke[:num_ctx_tokens].copy_(host_cu_seqlen_ke,
                                                         non_blocking=True)

        # Prepare for decode phase if there are generation requests
        if num_generations > 0:
            # Prepare schedule metadata for fp8_paged_mqa_logits
            # This is a preprocessing step that computes scheduling information for the kernel
            if not metadata.use_expanded_buffers_for_mtp:
                # Write 2D kv_lens (broadcast same kv_len across next_n positions).
                gen_seq_lens = metadata.kv_lens_cuda_runtime[
                    num_contexts:num_contexts + num_generations]
                next_n_cap = metadata.kv_lens_cuda_2d.shape[1]
                metadata.kv_lens_cuda_2d[:num_generations, :next_n_cap].copy_(
                    gen_seq_lens.unsqueeze(-1).expand(-1, next_n_cap))
                # Build the next_n=1 schedule (used by MTP draft layers).
                # Use the contiguous 1D gen slice reshaped to (num_gen, 1);
                # slicing kv_lens_cuda_2d's first column would be a strided
                # view that fails the metadata kernel's contiguous assertion.
                context_lens_next_n1 = gen_seq_lens.view(-1, 1)
                scheduler_metadata_buffer = get_paged_mqa_logits_metadata(
                    context_lens_next_n1, tokens_per_block, metadata.num_sms)
                metadata.scheduler_metadata_buffer.copy_(
                    scheduler_metadata_buffer, non_blocking=True)
                # MTP main forward uses next_n = 1 + max_draft_tokens; build
                # a separate schedule because the metadata kernel reads next_n
                # from context_lens.size(1).
                if metadata.max_draft_tokens > 0:
                    context_lens_full_next_n = metadata.kv_lens_cuda_2d[:
                                                                        num_generations, :
                                                                        next_n_cap]
                    scheduler_metadata_buffer_full_next_n = get_paged_mqa_logits_metadata(
                        context_lens_full_next_n, tokens_per_block,
                        metadata.num_sms)
                    metadata.scheduler_metadata_buffer_full_next_n.copy_(
                        scheduler_metadata_buffer_full_next_n,
                        non_blocking=True)
            else:
                # Expand schedule metadata buffer (only generation). The new
                # DeepGEMM API requires 2D; each expanded token becomes a (1,)
                # row.
                num_tokens = metadata.num_generations * (
                    1 + metadata.max_draft_tokens)
                kv_lens_expanded_2d = metadata.kv_lens_expanded_cuda[:
                                                                     num_tokens].view(
                                                                         -1, 1)
                scheduler_metadata_buffer_expanded = get_paged_mqa_logits_metadata(
                    kv_lens_expanded_2d, tokens_per_block, metadata.num_sms)
                metadata.scheduler_metadata_buffer_expanded.copy_(
                    scheduler_metadata_buffer_expanded, non_blocking=True)

        # Compute slot_mapping for all requests (both context and generation)
        Indexer.recompute_slot_mappings(metadata)

        # When chunked prefill or KVCache reuse is enabled, we need to gather the full KV for indexer's logit computation.
        # Indexer's own chunking does not need full KV gathering, instead it gathers only the current chunk with loop-based gathering.
        _need_full_kv_gathering = num_contexts > 0 and metadata.enable_context_mla_with_cached_kv
        if _need_full_kv_gathering:
            head_dim = kv_cache_manager.index_head_dim
            quant_block_size = kv_cache_manager.quant_block_size
            use_fp4 = kv_cache_manager.use_fp4
            data_bytes_per_token = head_dim // 2 if use_fp4 else head_dim
            cached_tokens = metadata.kv_cache_params.num_cached_tokens_per_seq
            start_positions = torch.tensor(cached_tokens, dtype=torch.int32)

            total_kv_len = metadata.host_ctx_kv_indptr[num_contexts].item()
            total_kv_per_request = seq_lens[:
                                            num_contexts] + start_positions[:
                                                                            num_contexts]
            host_slot_mapping_fp8_fullkv = torch.empty(
                total_kv_len, dtype=torch.int64, pin_memory=prefer_pinned())
            host_slot_mapping_scale_fullkv = torch.empty(
                total_kv_len, dtype=torch.int64, pin_memory=prefer_pinned())

            fullkv_req_indices = torch.repeat_interleave(
                torch.arange(num_contexts, dtype=torch.int64, device='cpu'),
                total_kv_per_request)

            kv_positions = torch.cat([
                torch.arange(total_kv_per_request[i].item(),
                             dtype=torch.int64,
                             device='cpu') for i in range(num_contexts)
            ])

            fp8_flat_indices, scale_flat_indices = _compute_slot_mappings(
                kv_positions,
                metadata.host_indexer_k_cache_block_offsets,
                fullkv_req_indices,
                head_dim,
                tokens_per_block,
                quant_block_size,
                data_bytes_per_token=data_bytes_per_token,
            )

            host_slot_mapping_fp8_fullkv[:total_kv_len] = fp8_flat_indices
            host_slot_mapping_scale_fullkv[:total_kv_len] = scale_flat_indices

            assert len(fp8_flat_indices) == total_kv_len, \
                f"host_slot_mapping_fp8_fullkv/host_slot_mapping_scale_fullkv length mismatch: {len(fp8_flat_indices)} != total_kv_len={total_kv_len}"

            # Store extended mappings for indexer full KV gathering
            metadata.slot_mapping_fp8_fullkv = host_slot_mapping_fp8_fullkv.cuda(
                non_blocking=True)
            metadata.slot_mapping_scale_fullkv = host_slot_mapping_scale_fullkv.cuda(
                non_blocking=True)
        else:
            metadata.slot_mapping_fp8_fullkv = metadata.slot_mapping_fp8
            metadata.slot_mapping_scale_fullkv = metadata.slot_mapping_scale

    def _update_k_cache(self, k_fp8: torch.Tensor, k_scale: torch.Tensor,
                        metadata: DSAtrtllmAttentionMetadata) -> None:
        """
        Insert/append k values and scales into the indexer k cache using pre-computed slot mappings.
        Uses flat byte indices with vectorized scatter.

        Args:
            k_fp8: FP8 quantized k tensor, shape [total_tokens, head_dim]
            k_scale: Scaling factors, shape [total_tokens, head_dim // quant_block_size]
        """
        if metadata.kv_cache_manager is None or metadata.slot_mapping_fp8 is None:
            return

        k_cache = metadata.kv_cache_manager.get_indexer_k_cache_buffers(
            self.layer_idx)

        num_tokens = k_fp8.shape[0]

        # The C++ op reinterprets k_fp8 (FP8) and k_scale (float32) as raw
        # bytes internally and only reads the first num_tokens entries from
        # the slot mapping buffers, avoiding Python-side view/slice overhead.
        torch.ops.trtllm.indexer_k_cache_scatter_op(k_fp8, k_scale, k_cache,
                                                    metadata.slot_mapping_fp8,
                                                    metadata.slot_mapping_scale,
                                                    num_tokens)

    def _call_mqa_logits(self, q_fp8: torch.Tensor, k_fp8: torch.Tensor,
                         k_scale: torch.Tensor, weights: torch.Tensor,
                         cu_seqlen_ks: torch.Tensor, cu_seqlen_ke: torch.Tensor,
                         q_scale: Optional[torch.Tensor]) -> torch.Tensor:
        """Dispatch to fp8_mqa_logits or fp8_fp4_mqa_logits based on use_fp4.

        For FP4 the gather output is typed as FP8 for historical reasons;
        reinterpret the bytes as the expected int8 / int32 layouts. DeepGEMM
        asserts kv_sf is 1D in both modes, so flatten the scale here.
        """
        if self.use_fp4:
            k_fp4_bytes = k_fp8.view(torch.int8)
            k_scale_int32 = k_scale.view(torch.int32).reshape(-1)
            # q_scale arrives here as (chunk_tokens, n_heads, 1) — fused_cat_fp4
            # emits one int32 per (token, head) carrying four UE8M0 exponents,
            # pre_indexer_proj reshapes back to (N, n_heads, 1), and
            # sparse_attn_indexer chunk-slices on axis 0. The DeepGEMM FP4
            # kernel asserts q_sf is 2D, so collapse the trailing unit axis.
            q_scale_2d = q_scale.reshape(-1, self.n_heads)
            return fp8_fp4_mqa_logits(
                (q_fp8, q_scale_2d),
                (k_fp4_bytes, k_scale_int32),
                weights,
                cu_seqlen_ks,
                cu_seqlen_ke,
            )
        return fp8_mqa_logits(q_fp8, (k_fp8, k_scale.reshape(-1)), weights,
                              cu_seqlen_ks, cu_seqlen_ke)

    def _call_paged_mqa_logits(self, q_decode: torch.Tensor,
                               k_cache: torch.Tensor,
                               weights_decode: torch.Tensor,
                               context_lens: torch.Tensor,
                               block_table: torch.Tensor,
                               scheduler_metadata_buffer: torch.Tensor,
                               max_seq_len: int,
                               q_scale: Optional[torch.Tensor]) -> torch.Tensor:
        """Dispatch to fp8_paged_mqa_logits or fp8_fp4_paged_mqa_logits."""
        if self.use_fp4:
            return fp8_fp4_paged_mqa_logits(
                (q_decode, q_scale), k_cache, weights_decode, context_lens,
                block_table, scheduler_metadata_buffer, max_seq_len)
        return fp8_paged_mqa_logits(q_decode, k_cache, weights_decode,
                                    context_lens, block_table,
                                    scheduler_metadata_buffer, max_seq_len)

    def sparse_attn_indexer(
        self,
        metadata: DSAtrtllmAttentionMetadata,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k_fp8: torch.Tensor,
        k_scale: torch.Tensor,
        weights: torch.Tensor,
        use_custom_topk: bool = True,
        q_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the indexer TopK kernel for both prefill and decode phases.

        q_scale is only consumed by the FP4 dispatch; FP8 path ignores it.
        """
        # DSACacheManager hardcodes quant_block_size=128 (see its __init__);
        # FP4 uses per-block-32 UE8M0 scales but packs four of them into one
        # int32 so the cache-layout contribution is the same 4 bytes/token as
        # FP8 per-block-128 at head_dim=128.
        assert metadata.kv_cache_manager is None or \
            metadata.kv_cache_manager.quant_block_size == 128, \
            f"Unexpected quant_block_size {metadata.kv_cache_manager.quant_block_size if metadata.kv_cache_manager else 'N/A'}"
        # Update the indexer k cache before prefill chunks gather from it.
        self._update_k_cache(k_fp8, k_scale, metadata)

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
        if not use_custom_topk:
            topk_indices_buffer[:hidden_states.shape[0]] = -1

        if has_prefill and not metadata.skip_indexer_for_ctx_reqs:
            # Use chunked prefill to reduce memory footprint
            if metadata.indexer_prefill_chunks is not None:

                # Default to 8192 if sparse_attention_config is not available (e.g., in unit tests)
                q_split_threshold = metadata.sparse_attention_config.q_split_threshold if metadata.sparse_attention_config is not None else 8192
                q_split_eligible = q_split_threshold >= 0 and metadata.mapping is not None and not metadata.mapping.enable_attention_dp and metadata.mapping.tp_size > 1

                if q_split_eligible:
                    tp_rank = metadata.mapping.tp_rank
                    tp_size = metadata.mapping.tp_size

                k_cache_4d = metadata.kv_cache_manager.get_indexer_k_cache_buffers(
                    self.layer_idx)

                gather_head_dim = self.head_dim // 2 if self.use_fp4 else self.head_dim
                for chunk in metadata.indexer_prefill_chunks:
                    num_k_tokens = chunk.k_token_end - chunk.k_token_start
                    chunk_k_fp8, chunk_k_scale = torch.ops.trtllm.indexer_k_cache_gather_op(
                        k_cache_4d, metadata.slot_mapping_fp8_fullkv,
                        metadata.slot_mapping_scale_fullkv, chunk.k_token_start,
                        num_k_tokens, gather_head_dim)

                    chunk_num_token = chunk.token_end - chunk.token_start
                    apply_q_split = q_split_eligible and chunk_num_token >= q_split_threshold
                    if apply_q_split:
                        chunk_q_start = chunk_num_token * tp_rank // tp_size
                        chunk_q_end = chunk_num_token * (tp_rank + 1) // tp_size
                    else:
                        chunk_q_start = 0
                        chunk_q_end = chunk_num_token

                    global_q_start = chunk.token_start + chunk_q_start
                    global_q_end = chunk.token_start + chunk_q_end

                    chunk_q_scale = q_scale[global_q_start:global_q_end,
                                            ...] if self.use_fp4 else None
                    logits = self._call_mqa_logits(
                        q_fp8[global_q_start:global_q_end, ...],
                        chunk_k_fp8,
                        chunk_k_scale,
                        weights[global_q_start:global_q_end, ...],
                        chunk.cu_seqlen_ks[chunk_q_start:chunk_q_end],
                        chunk.cu_seqlen_ke[chunk_q_start:chunk_q_end],
                        chunk_q_scale,
                    )
                    if use_custom_topk:
                        torch.ops.trtllm.indexer_topk_prefill(
                            logits,
                            chunk.cu_seqlen_ks[chunk_q_start:chunk_q_end],
                            chunk.cu_seqlen_ke[chunk_q_start:chunk_q_end],
                            topk_indices_buffer[global_q_start:global_q_end, :])
                    else:
                        topk_indices = logits.topk(min(self.index_topk,
                                                       logits.shape[-1]),
                                                   dim=-1)[1]
                        topk_indices -= chunk.cu_seqlen_ks[
                            chunk_q_start:chunk_q_end][:, None]

                        mask_lo = topk_indices >= 0
                        mask_hi = topk_indices - (
                            chunk.cu_seqlen_ke[chunk_q_start:chunk_q_end] -
                            chunk.cu_seqlen_ks[chunk_q_start:chunk_q_end]
                        )[:, None] < 0
                        mask = mask_lo & mask_hi

                        # local indices per sequence
                        topk_indices = topk_indices.masked_fill(~mask, -1)

                        topk_indices_buffer[
                            global_q_start:global_q_end, :topk_indices.
                            shape[-1]] = topk_indices.to(dtype=torch.int32)

                    if apply_q_split:
                        q_sizes = [(r + 1) * chunk_num_token // tp_size -
                                   r * chunk_num_token // tp_size
                                   for r in range(tp_size)]
                        topk_indices_buffer[
                            chunk.token_start:chunk.token_end, :] = allgather(
                                topk_indices_buffer[
                                    global_q_start:global_q_end, :],
                                metadata.mapping,
                                dim=0,
                                sizes=q_sizes)
            else:
                # Fallback: single-pass indexer prefill (TODO: remove this once chunked prefill is fully tested)
                cu_seqlen_ks = metadata.cu_seqlen_ks[:num_ctx_tokens]
                cu_seqlen_ke = metadata.cu_seqlen_ke[:num_ctx_tokens]

                ctx_q_scale = q_scale[:num_ctx_tokens,
                                      ...] if self.use_fp4 else None
                logits = self._call_mqa_logits(
                    q_fp8[:num_ctx_tokens, ...],
                    k_fp8[:num_ctx_tokens, ...],
                    k_scale[:num_ctx_tokens, ...],
                    weights[:num_ctx_tokens, ...],
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                    ctx_q_scale,
                )
                if use_custom_topk:
                    torch.ops.trtllm.indexer_topk_prefill(
                        logits, cu_seqlen_ks, cu_seqlen_ke,
                        topk_indices_buffer[:num_ctx_tokens, :])
                else:
                    topk_indices = logits.topk(min(self.index_topk,
                                                   logits.shape[-1]),
                                               dim=-1)[1]
                    topk_indices -= cu_seqlen_ks[:, None]
                    mask_lo = topk_indices >= 0
                    mask_hi = topk_indices - (cu_seqlen_ke -
                                              cu_seqlen_ks)[:, None] < 0
                    mask = mask_lo & mask_hi

                    # local indices per sequence
                    topk_indices = topk_indices.masked_fill(~mask, -1)
                    topk_indices_buffer[:num_ctx_tokens, :topk_indices.
                                        shape[-1]] = topk_indices.to(
                                            dtype=torch.int32)
        elif has_prefill and metadata.skip_indexer_for_ctx_reqs:
            # Fill topk_indices_buffer with pre-defined dense topk indices
            topk_indices_buffer[:num_ctx_tokens, :] = \
                metadata.topk_indices_buffer[:num_ctx_tokens, :]

        if has_decode and not metadata.skip_indexer_for_gen_reqs:
            max_seq_len = metadata.kv_cache_manager.max_seq_len
            # Get decode lengths per request (from seq_lens) for validation
            gen_seq_lens = metadata.seq_lens[num_contexts:num_contexts +
                                             num_generations]
            max_decode_len = gen_seq_lens.max().item()
            min_decode_len = gen_seq_lens.min().item()
            assert max_decode_len == min_decode_len, "max_decode_len != min_decode_len, we need padding"

            # Reshape q for decode phase: [num_gen_tokens, ...] -> [batch_size, next_n, ...]
            q_decode = q_fp8[num_ctx_tokens:num_ctx_tokens + num_gen_tokens,
                             ...]
            batch_size = num_generations
            next_n = num_gen_tokens // num_generations
            # Because fp8_paged_mqa_logits can only support next_n == 1/2/4 on sm100, and
            # next_n == 1/2 on sm90, for other next_n, we need to flatten the q_decode tensor
            # and expand the corresponding metadata.
            if not metadata.use_expanded_buffers_for_mtp or next_n == 1:
                q_decode = q_decode.view(num_generations, -1, *q_fp8.shape[1:])
                # 2D context_lens slice from the pre-allocated buffer; matches
                # q_decode's (batch, next_n) layout required by the new
                # DeepGEMM paged MQA logits API.
                context_lens = metadata.kv_lens_cuda_2d[:num_generations, :
                                                        next_n].contiguous()
                block_table = metadata.indexer_k_cache_block_offsets[
                    num_contexts:num_contexts + num_generations]
                # The 2D-context_lens metadata kernel encodes next_n into the
                # schedule (via num_next_n_atoms). MTP forwards alternate
                # between the full-window call (next_n == 1+max_draft_tokens)
                # and per-token draft calls (next_n == 1), so we must select
                # the buffer that was populated for this next_n.
                if next_n == 1:
                    scheduler_metadata_buffer = metadata.scheduler_metadata_buffer
                else:
                    scheduler_metadata_buffer = metadata.scheduler_metadata_buffer_full_next_n
            else:
                q_decode = q_decode.view(-1, 1, *q_fp8.shape[1:])
                num_tokens = q_decode.shape[0]
                # New API requires 2D; each expanded token becomes a (1,) row.
                context_lens = metadata.kv_lens_expanded_cuda[:num_tokens].view(
                    -1, 1)
                block_table = metadata.block_table_expanded[:num_tokens]
                scheduler_metadata_buffer = metadata.scheduler_metadata_buffer_expanded

            assert num_gen_tokens == batch_size * next_n
            weights_decode = weights[num_ctx_tokens:num_ctx_tokens +
                                     num_gen_tokens, ...]

            # Get k cache and call fp8_paged_mqa_logits with prepared decode metadata
            # [num_blocks, tokens_per_block, 1, head_dim + scale_size]
            k_cache = metadata.kv_cache_manager.get_indexer_k_cache_buffers(
                self.layer_idx)

            decode_q_scale = q_scale[num_ctx_tokens:num_ctx_tokens +
                                     num_gen_tokens,
                                     ...] if self.use_fp4 else None
            if self.use_fp4:
                # q_decode shape is either (num_generations, next_n, n_heads,
                # head_dim/2) [non-expanded] or (batch*next_n, 1, n_heads,
                # head_dim/2) [expanded]. Match q_scale's batch/next_n dims.
                decode_q_scale = decode_q_scale.view(q_decode.shape[0],
                                                     q_decode.shape[1],
                                                     self.n_heads)
            logits_decode = self._call_paged_mqa_logits(
                q_decode, k_cache, weights_decode, context_lens, block_table,
                scheduler_metadata_buffer, max_seq_len, decode_q_scale)

            if use_custom_topk:
                # Kernel expects kv_lens (total cache length), not seq_lens (new tokens)
                # This is because rowEnd = seq_len - next_n + offset + 1
                gen_kv_lens_cuda = metadata.kv_lens_cuda_runtime[
                    num_contexts:num_contexts + num_generations]

                pre_idx = None
                heuristic_scratch = None
                if self._enable_heuristic_topk:
                    local_layer = metadata.kv_cache_manager.layer_offsets[
                        self.layer_idx]
                    # Pass prev_topk directly; the +1 temporal offset is
                    # handled inside the C++ kernel (preIdxOffset += 1).
                    pre_idx = metadata.heuristic_prev_topk[
                        local_layer, :num_generations]
                    heuristic_scratch = \
                        metadata.heuristic_scratch_values[
                            :num_gen_tokens]

                # CuTE DSL top-k allocates O(num_gen_tokens * kv_len) global
                # memory. Beyond 256 tokens the extra memory becomes significant,
                # so we cap it at 256 for now and fall back to the CUDA C++
                # indexer_topk_decode. This limit can be removed if GPU memory
                # is not a bottleneck.
                if self.use_cute_dsl_topk and num_gen_tokens <= 256:
                    torch.ops.trtllm.cute_dsl_indexer_topk_decode(
                        logits_decode, gen_kv_lens_cuda,
                        topk_indices_buffer[num_ctx_tokens:num_ctx_tokens +
                                            num_gen_tokens, :], self.index_topk,
                        next_n)
                else:
                    torch.ops.trtllm.indexer_topk_decode(
                        logits_decode,
                        gen_kv_lens_cuda,
                        topk_indices_buffer[num_ctx_tokens:num_ctx_tokens +
                                            num_gen_tokens, :],
                        next_n,
                        self.index_topk,
                        pre_idx=pre_idx,
                        heuristic_scratch=heuristic_scratch)
            else:
                # padded
                positions = torch.arange(
                    max_seq_len, device=q_decode.device).unsqueeze(0).expand(
                        num_gen_tokens, -1)
                row_indices = torch.arange(num_gen_tokens,
                                           device=q_decode.device) // next_n
                next_n_offset = torch.arange(num_gen_tokens,
                                             device=q_decode.device) % next_n
                index_end_pos = (
                    metadata.kv_lens_cuda_runtime[num_contexts + row_indices] -
                    next_n + next_n_offset).unsqueeze(1)
                # index_end_pos: [B * N, 1]
                mask = positions <= index_end_pos
                # mask: [B * N, L]
                logits_decode = logits_decode.masked_fill(~mask, float('-inf'))
                topk_indices_decode = logits_decode.topk(
                    min(self.index_topk, logits_decode.shape[-1]),
                    dim=-1)[1].to(torch.int32)  # [B * N, K]
                # ensure we don't set indices for the top k
                # that is out of range(masked already)
                # this will happen if context length is shorter than K
                mask_decode = topk_indices_decode <= index_end_pos

                # local indices per sequence
                topk_indices_decode = topk_indices_decode.masked_fill(
                    ~mask_decode, -1)
                # Store in buffer
                topk_indices_buffer[num_ctx_tokens:num_ctx_tokens +
                                    num_gen_tokens, :topk_indices_decode.
                                    shape[-1]] = topk_indices_decode.to(
                                        dtype=torch.int32)

            if self._enable_heuristic_topk:
                local_layer = metadata.kv_cache_manager.layer_offsets[
                    self.layer_idx]
                decode_topk = topk_indices_buffer[
                    num_ctx_tokens:num_ctx_tokens + num_gen_tokens]
                last_mtp_topk = decode_topk[next_n - 1::next_n]
                metadata.heuristic_prev_topk[
                    local_layer, :num_generations].copy_(last_mtp_topk)

        elif has_decode and metadata.skip_indexer_for_gen_reqs:
            # Fill topk_indices_buffer with pre-defined dense topk indices
            topk_indices_buffer[num_ctx_tokens:num_tokens, :] = \
                metadata.topk_indices_buffer[num_ctx_tokens:num_tokens, :]
        return topk_indices_buffer

    def _weight_scale(self, weights: torch.Tensor,
                      q_scale: torch.Tensor) -> torch.Tensor:
        """Apply quantization scale to indexer attention weights."""
        weights = _scale(weights, q_scale, self.weight_scale_factor)
        return weights

    def _qk_projection_and_rope(self, qr: torch.Tensor, indexer_k: torch.Tensor,
                                position_ids: torch.Tensor):
        """Project Q/K and apply RoPE"""
        q = self.wq_b(qr)
        k = self.k_norm(indexer_k)
        q = q.view(-1, self.n_heads, self.head_dim)
        q_pe, q_nope = q.split([self.rope_dim, self.head_dim - self.rope_dim],
                               dim=-1)
        k_pe, k_nope = k.split([self.rope_dim, self.head_dim - self.rope_dim],
                               dim=-1)
        q_pe, k_pe = self.rotary_emb(position_ids, [q_pe, k_pe.unsqueeze(1)])
        k_pe = k_pe[:, 0, :]
        return q_pe, q_nope, k_pe, k_nope

    def _prep_q_or_k(self, qk_pe: torch.Tensor, qk_nope: torch.Tensor):
        """Concatenate and quantize for Q or K.

        FP8 mode: fused cat + FP8 quantize via CUDA kernel.
        FP4 mode: fused cat + per-block-32 FP4 E2M1 quantize via CUDA kernel.
        The returned packed bytes are int8 (two FP4 codes per byte) and the
        scale is int32 (four UE8M0 exponents packed little-endian).
        """
        if self.use_fp4:
            return torch.ops.trtllm.fused_cat_fp4(qk_pe, qk_nope)
        fp8_out, scale = torch.ops.trtllm.fused_cat_fp8(
            qk_pe, qk_nope, self.scale_fmt == "ue8m0")
        return fp8_out, scale

    def pre_indexer_proj(
        self, qr: torch.Tensor, hidden_states: torch.Tensor,
        position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        """Pure token-wise projections (CUDA-graph-capturable).

        Runs cublas_mm, qk_projection_and_rope, FP8 quantize, and weight
        scaling.  Does NOT touch the k cache or any batch-specific metadata,
        so this can safely run inside a captured CUDA graph partition.

        Returns (q_fp_bytes, k_fp_bytes, k_scale, weights, q_scale). The last
        tensor is only consumed by the FP4 kernel dispatch; the FP8 path
        ignores it. It is returned unconditionally so the two-op CUDA graph
        split in MLA.forward_dsa_proj sees a stable signature.
        """
        assert self._fused_wk_wp_weight is not None, \
            "post_load_weights() must be called before forward()"
        hidden_float = _to_float(hidden_states)
        with _tf32_matmul_enabled():
            # F.linear computes input @ weight.T internally; no explicit .t() needed.
            # _fused_wk_wp_weight is [head_dim + n_heads, hidden_size] (nn.Linear convention).
            # Goes through PyTorch's cuBLAS handle which respects allow_tf32 and
            # dispatches CUBLAS_COMPUTE_32F_FAST_TF32, unlike torch.ops.trtllm.cublas_mm
            # which uses its own handle and always falls back to CUDA-core SGEMM.
            fused_out = F.linear(hidden_float, self._fused_wk_wp_weight)
        indexer_k, weights = fused_out.split([self.head_dim, self.n_heads],
                                             dim=-1)
        # Cast indexer_k back to model dtype for downstream ops (k_norm, RoPE, FP8 quantize)
        indexer_k = indexer_k.to(hidden_states.dtype)

        q_pe, q_nope, k_pe, k_nope = self._qk_projection_and_rope(
            qr, indexer_k, position_ids)
        q, k = maybe_execute_in_parallel(
            lambda: self._prep_q_or_k(q_pe, q_nope),
            lambda: self._prep_q_or_k(k_pe, k_nope),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )
        q_fp8, q_scale = q
        k_fp8, k_scale = k
        if self.use_fp4:
            # FP4 packs two codes per byte, so the trailing dim is head_dim // 2.
            # fused_cat_fp4 flattens the leading dims to M=N*n_heads; restore
            # the (N, n_heads, ...) shape so downstream slicing in
            # sparse_attn_indexer (which indexes by token) lines up with
            # q_fp8. The DeepGEMM FP4 kernel applies the per-block q_scale
            # internally, so weights carry only softmax_scale * n_heads^-0.5.
            q_fp8 = q_fp8.view(-1, self.n_heads, self.head_dim // 2)
            q_scale = q_scale.view(-1, self.n_heads, 1)
            weights = weights * self.weight_scale_factor
        else:
            q_fp8 = q_fp8.view(-1, self.n_heads, self.head_dim)
            q_scale = q_scale.view(-1, self.n_heads, 1)
            weights = self._weight_scale(weights, q_scale)

        return q_fp8, k_fp8, k_scale, weights, q_scale

    @torch.inference_mode()
    def forward(self, qr: torch.Tensor, hidden_states: torch.Tensor,
                metadata: DSAtrtllmAttentionMetadata,
                position_ids: torch.Tensor):
        q_fp8, k_fp8, k_scale, weights, q_scale = self.pre_indexer_proj(
            qr, hidden_states, position_ids)

        # Return topk indices buffer for sparse attention [num_tokens, index_topk]
        return self.sparse_attn_indexer(metadata,
                                        hidden_states,
                                        q_fp8,
                                        k_fp8,
                                        k_scale,
                                        weights,
                                        q_scale=q_scale)


class DSATrtllmAttention(TrtllmAttention):
    """TRT-LLM attention layer with DSA sparse indexer for MLA models."""

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
            aux_stream: Optional[torch.cuda.Stream] = None,
            **kwargs):
        """Initialize DSA attention with an Indexer sub-module for sparse TopK selection."""
        if sparse_attention_config is None:
            raise ValueError(
                "sparse_attention_config is required for DSATrtllmAttention and cannot be None"
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
            **kwargs)

        self.indexer = Indexer(quant_config, pos_embd_params, mla_params,
                               skip_create_weights_in_init,
                               sparse_attention_config, dtype, layer_idx,
                               aux_stream)

    def sparse_attn_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: DSAtrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Transform local TopK indices to global paged KV cache indices."""
        # Transform the local topk indices to global topk indices in paged kv cache
        topk_indices_global, _ = transform_local_topk_and_prepare_pool_view(
            forward_args.topk_indices, metadata,
            self.get_local_layer_idx(metadata), forward_args.is_generation)

        # TODO: Use sparse_attn_indexer to predict the indices for DSA attention
        # return self.indexer(q, k, metadata, hidden_states, qr, position_ids)
        return topk_indices_global, None

    def sparse_kv_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: DSAtrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """No-op KV prediction; DSA uses indexer-based selection instead."""
        return None, None

    def mla_rope_append_paged_kv_assign_q(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        metadata: DSAtrtllmAttentionMetadata,
        is_generation: bool = False,
        **kwargs,
    ) -> None:
        """Apply RoPE, append latent cache to paged KV, and assign query for MLA."""
        if is_generation:
            cached_token_indptr = metadata.gen_cached_token_indptr
            kv_indptr = metadata.gen_kv_indptr
            num_seqs = metadata.num_generations
            max_seq_len = metadata.max_gen_seq_len
            block_offsets = metadata.kv_cache_block_offsets[:, metadata.
                                                            num_contexts:]
        else:
            cached_token_indptr = metadata.ctx_cached_token_indptr
            kv_indptr = metadata.ctx_kv_indptr
            num_seqs = metadata.num_contexts
            max_seq_len = metadata.max_ctx_seq_len
            block_offsets = metadata.kv_cache_block_offsets
        assert self.is_mla_enable and self.mla_params is not None
        assert metadata.kv_cache_manager is not None

        sink_token_length = 0
        beam_width = 1

        torch.ops.trtllm.mla_rope_append_paged_kv_assign_q(
            q,
            latent_cache,
            num_seqs,
            cached_token_indptr,
            kv_indptr,
            max_seq_len,
            self.rotary_cos_sin,
            self.num_heads,
            self.mla_params.qk_nope_head_dim,
            self.mla_params.qk_rope_head_dim,
            self.mla_params.kv_lora_rank,
            block_offsets,
            metadata.kv_cache_manager.kv_cache_pool_pointers,
            metadata.kv_cache_manager.kv_cache_pool_mapping,
            self.kv_scale_orig_quant,
            self.kv_scale_quant_orig,
            self.get_local_layer_idx(metadata),
            metadata.kv_cache_manager.tokens_per_block,
            metadata.kv_cache_manager.max_seq_len,
            sink_token_length,
            beam_width,
            self.quant_mode,
        )


class DSACacheManager(KVCacheManager):
    """KV cache manager for DSA with additional indexer K-cache pools."""

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
        sparse_attn_config: "SparseAttentionConfig",
        **kwargs,
    ) -> None:
        """Initialize cache manager with indexer K-cache pool per layer."""
        self.quant_block_size = 128
        self.index_head_dim = sparse_attn_config.index_head_dim
        # FP4 mode packs the indexer K cache as head_dim/2 data bytes + 4
        # scale bytes (vs. head_dim + 4 for FP8). The C++ WindowBlockManager
        # allocates the pool with this smaller stride when the flag is set.
        self.use_fp4 = sparse_attn_config.indexer_k_dtype == "fp4"

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
            enable_indexer_k_cache=True,
            indexer_k_cache_quant_block_size=128,
            indexer_k_cache_index_head_dim=self.index_head_dim,
            indexer_k_cache_use_fp4=self.use_fp4,
            **kwargs,
        )
        self.num_blocks = self.blocks_in_primary_pool

        # Indexer K cache pool for DSA attention
        # Shape: [num_blocks, self.tokens_per_block * (index_head_dim + scale_size)]
        # Non-interleaved layout: [fp8_tok0 | fp8_tok1 | ... | scale_tok0 | scale_tok1 | ...]
        # Store FP8-quantized k values from the indexer
        self.indexer_k_cache_pool_per_layer = [
            self.get_indexer_k_cache_pool_data(layer_idx)
            for layer_idx in range(self.num_local_layers)
        ]

    def get_indexer_k_cache_buffers(self, layer_idx: int):
        """Get indexer k cache buffer from a specific layer pool."""
        block_size = self.tokens_per_block
        data_bytes = self.index_head_dim // 2 if self.use_fp4 else self.index_head_dim
        per_token_size = data_bytes + self.index_head_dim // self.quant_block_size * 4
        layer_offset = self.layer_offsets[layer_idx]
        return self.indexer_k_cache_pool_per_layer[layer_offset].view(
            self.num_blocks, block_size, 1, per_token_size)

    def shutdown(self):
        """Release indexer K-cache pool references before C++ buffer cleanup."""
        # Clear Python references BEFORE C++ frees the underlying CUDA buffers
        self.indexer_k_cache_pool_per_layer = []
        super().shutdown()

    @staticmethod
    def get_cache_size_per_token(model_config: ModelConfig,
                                 mapping: Mapping,
                                 num_layers: Optional[int] = None,
                                 **kwargs):
        """Estimate total cache bytes per token including indexer K-cache overhead."""
        config = model_config.pretrained_config
        sparse_attn_config = model_config.sparse_attention_config
        index_head_dim = sparse_attn_config.index_head_dim
        quant_block_size = 128
        # Under FP4 the indexer stores two E2M1 codes per byte, so the
        # per-token data footprint halves (132 B -> 68 B at index_head_dim=128);
        # the scale bytes are unchanged (4 per token, one int32 holding four
        # UE8M0 exponents at quant_block_size=32 after packing).
        use_fp4 = sparse_attn_config.indexer_k_dtype == "fp4"
        indexer_data_dim = index_head_dim // 2 if use_fp4 else index_head_dim

        # get kv cache dtype bytes
        mem_per_token = 2
        quant_config = model_config.quant_config
        if quant_config is not None and quant_config.quant_mode.has_fp8_kv_cache(
        ):
            mem_per_token = 1

        # get head dim
        head_dim = config.kv_lora_rank + config.qk_rope_head_dim

        num_attention_layers = KVCacheManager._resolve_num_attention_layers(
            model_config, mapping, num_layers)
        mem_per_token *= num_attention_layers * head_dim

        # 1 for K, others for indexer K cache
        head_dim_factor = (indexer_data_dim +
                           index_head_dim // quant_block_size * 4) / head_dim
        kv_factor = 1 + head_dim_factor
        mem_per_token *= kv_factor
        return mem_per_token

    def get_cache_bytes_per_token(self):
        """Compute actual cache bytes per token from instance configuration."""
        # self.kv_factor for K, others for indexer K cache.
        # Under FP4 the indexer data portion is halved (two E2M1 codes per
        # byte); scale bytes are unchanged.
        indexer_data_dim = self.index_head_dim // 2 if self.use_fp4 else self.index_head_dim
        head_dim_factor = (indexer_data_dim + self.index_head_dim //
                           self.quant_block_size * 4) / self.head_dim
        kv_factor = self.kv_factor + head_dim_factor
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
