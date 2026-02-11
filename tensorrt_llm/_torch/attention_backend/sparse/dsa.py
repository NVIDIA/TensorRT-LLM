import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
import torch.nn as nn

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.attention_backend.interface import (
    MLAParams, PositionalEmbeddingParams)
from tensorrt_llm._torch.attention_backend.trtllm import (
    TrtllmAttention, TrtllmAttentionMetadata)
from tensorrt_llm._torch.modules.layer_norm import LayerNorm
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.multi_stream_utils import \
    maybe_execute_in_parallel
from tensorrt_llm._torch.modules.rotary_embedding import RotaryEmbedding
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.utils import maybe_compile, maybe_compiled_cat
from tensorrt_llm._utils import get_size_in_bytes, get_sm_version
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.bindings.internal.batch_manager import \
    CacheType as CacheTypeCpp
from tensorrt_llm.deep_gemm import (fp8_mqa_logits, fp8_paged_mqa_logits,
                                    get_paged_mqa_logits_metadata)
from tensorrt_llm.llmapi.llm_args import SparseAttentionConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.utils import fp8_utils

from .kernel import triton_convert_req_index_to_global_index

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


def _unravel_indices(flat_indices: torch.Tensor,
                     shape: Tuple[int, ...]) -> Tuple[torch.Tensor, ...]:
    """
    Unravel indices into multiple dimensions.
    """
    d3 = shape[3]
    i3 = flat_indices % d3
    flat_indices = flat_indices // d3
    d2 = shape[2]
    i2 = flat_indices % d2
    flat_indices = flat_indices // d2
    d1 = shape[1]
    i1 = flat_indices % d1
    flat_indices = flat_indices // d1
    i0 = flat_indices
    return i0, i1, i2, i3


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
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
    """
    Convert local topk indices to global pool indices and prepare KV pool.
    Auto-detects stride and handles both contiguous/strided layouts.

    Args:
        topk_indices: [num_tokens, NUM_TOPK]
        attn_metadata: Metadata with block_table and request mappings
        kv_cache_manager: KV cache manager
        layer_idx: Layer index
        is_generation: Generation vs context phase

    Returns:
        (global_indices, kv_pool):
            - global_indices: [num_tokens, NUM_TOPK]
            - kv_pool: [total_tokens, 1, head_dim]
    """
    assert topk_indices.dtype == torch.int32

    # Get all layer KV cache pool: [num_blocks, num_layers, kv_factor, blockSize]
    kv_cache_manager = attn_metadata.kv_cache_manager
    all_layer_kv_pool = kv_cache_manager.get_unique_primary_pool(
    )  # [num_blocks, num_layers, kv_factor, blockSize]
    num_blocks, num_layers, _, _ = all_layer_kv_pool.shape
    tokens_per_block = kv_cache_manager.tokens_per_block
    head_dim = kv_cache_manager.head_dim
    assert all_layer_kv_pool.is_contiguous(
    ), "all_layer_kv_pool should be contiguous"
    all_layer_kv_pool = all_layer_kv_pool.squeeze(2).view(-1, 1, head_dim)
    stride_factor = num_layers * tokens_per_block

    # Get block_table and request indices for this phase
    if is_generation:
        block_table = attn_metadata.block_table[
            attn_metadata.num_contexts:attn_metadata.num_seqs]
        req_idx = attn_metadata.req_idx_per_token[
            attn_metadata.num_ctx_tokens:attn_metadata.num_tokens]
        req_idx = req_idx - attn_metadata.num_contexts
    else:
        block_table = attn_metadata.block_table[:attn_metadata.num_contexts]
        req_idx = attn_metadata.req_idx_per_token[:attn_metadata.num_ctx_tokens]

    # Convert to global indices
    global_indices = triton_convert_req_index_to_global_index(
        req_idx,
        block_table,
        topk_indices,
        BLOCK_SIZE=tokens_per_block,
        NUM_TOPK_TOKENS=topk_indices.shape[1],
        stride_factor=stride_factor,
        layer_id=layer_idx,
    )

    return global_indices, all_layer_kv_pool


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
    # Store reference to indexer for preparation stage
    indexer: Optional["Indexer"] = None
    # Chunked prefill metadata for indexer (prefill-only, no CUDA graph needed)
    indexer_prefill_chunks: Optional[List[IndexerPrefillChunkMetadata]] = None
    # Max chunk size for two-level chunking:
    # 1. Request-level: Pack multiple small requests into one chunk (up to indexer_max_chunk_size)
    # 2. Intra-request: Split large requests into Q-blocks when seq_len > max_chunk_size
    indexer_max_chunk_size: int
    # Topk for sparse MLA
    sparse_mla_topk: int
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
        self.num_sms = tensorrt_llm.deep_gemm.get_num_sms()
        super().__init__(*args, **kwargs)
        if self.sparse_attention_config.indexer_max_chunk_size is not None:
            self.indexer_max_chunk_size = self.sparse_attention_config.indexer_max_chunk_size
        else:
            self.indexer_max_chunk_size = 32768  # Default to 32K tokens for the indexer

    def __post_init__(self):
        super().__post_init__()

        self.sparse_mla_topk = self.sparse_attention_config.index_topk
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
            pin_memory=True,
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
                pin_memory=True,
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
                pin_memory=True,
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
            pin_memory=True,
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
            pin_memory=True,
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
            pin_memory=True,
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
            pin_memory=True,
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
                (self.max_num_tokens, self.sparse_mla_topk),
                cache_name="topk_indices_buffer",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            self.host_topk_indices_buffer = torch.zeros_like(
                self.topk_indices_buffer,
                device='cpu',
                pin_memory=True,
            )
        # Create expanded buffers for MTP support
        self.create_expanded_buffers(capture_graph=capture_graph)

    # TODO: remove these expanded buffers when fp8_paged_mqa_logits supports an arbitrary number of MTP draft tokens.
    def create_expanded_buffers(self, capture_graph=False):
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
            pin_memory=True,
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
            pin_memory=True,
        )
        self.scheduler_metadata_buffer_expanded = self.get_empty(
            self.cuda_graph_buffers,
            (self.num_sms + 1, 2),
            cache_name="scheduler_metadata_buffer_expanded",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        # The fp8_paged_mqa_logits kernel needs different layout of the metadata buffer for MTP=3.
        if self.max_draft_tokens == 3:
            self.scheduler_metadata_buffer_mtp3 = self.get_empty(
                self.cuda_graph_buffers,
                (self.num_sms // 2 + 1, 2),
                cache_name="scheduler_metadata_buffer_mtp3",
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
        spec_decoding_tensor: Optional['SpecDecodingTensor'] = None,
    ):
        super().update_spec_dec_param(batch_size, is_spec_decoding_enabled,
                                      is_spec_dec_tree,
                                      is_spec_dec_dynamic_tree, max_draft_len,
                                      max_total_draft_tokens, model_is_wrapped,
                                      spec_metadata, spec_tree_manager,
                                      spec_decoding_tensor)
        self.max_draft_tokens = max_draft_len
        init_shape = self.kv_lens_expanded_host.shape[0]
        if self.max_num_sequences * (1 + self.max_draft_tokens) != init_shape:
            capture_graph = self.is_cuda_graph
            self.create_expanded_buffers(capture_graph=capture_graph)

    def prepare_dense_topk_indices(self,
                                   kv_lens,
                                   device=False):  # device=False means use CPU

        @maybe_compile(dynamic=True)
        def _get_dense_topk_indices(seq_lens, kv_lens, num_tokens):
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
            range_row = torch.arange(self.sparse_mla_topk, device=device)
            mask = range_row <= position_ids.unsqueeze(1)
            return torch.where(mask, range_row, -1)

        if self.num_contexts > 0 and self.skip_indexer_for_ctx_reqs:
            ctx_range = slice(self.num_ctx_tokens)
            if device:
                self.topk_indices_buffer[ctx_range, :].copy_(
                    _get_dense_topk_indices(
                        self.seq_lens_cuda[:self.num_contexts],
                        kv_lens[:self.num_contexts], self.num_ctx_tokens),
                    non_blocking=True)
            else:
                self.host_topk_indices_buffer[
                    ctx_range, :] = _get_dense_topk_indices(
                        self.seq_lens[:self.num_contexts],
                        kv_lens[:self.num_contexts], self.num_ctx_tokens)
                self.topk_indices_buffer[ctx_range, :].copy_(
                    self.host_topk_indices_buffer[ctx_range, :],
                    non_blocking=True)

        if self.num_generations > 0 and self.skip_indexer_for_gen_reqs:
            gen_range = slice(self.num_ctx_tokens, self.num_tokens)
            if device:
                self.topk_indices_buffer[gen_range, :].copy_(
                    _get_dense_topk_indices(
                        self.seq_lens_cuda[self.num_contexts:self.num_seqs],
                        kv_lens[self.num_contexts:self.num_seqs],
                        self.num_tokens - self.num_ctx_tokens),
                    non_blocking=True)
            else:
                self.host_topk_indices_buffer[
                    gen_range, :] = _get_dense_topk_indices(
                        self.seq_lens[self.num_contexts:self.num_seqs],
                        kv_lens[self.num_contexts:self.num_seqs],
                        self.num_tokens - self.num_ctx_tokens)
                self.topk_indices_buffer[gen_range, :].copy_(
                    self.host_topk_indices_buffer[gen_range, :],
                    non_blocking=True)

    def prepare(self):
        super().prepare()

        # Get kv lengths
        assert self.kv_cache_params.use_cache is True, "DSA requires use_cache to be True"
        cached_token_lens = torch.tensor(
            self.kv_cache_params.num_cached_tokens_per_seq,
            dtype=torch.int,
            device='cpu',
        )
        kv_lens = cached_token_lens + self.seq_lens_kv

        # Prepare to support skip indexer
        num_extra_kv_tokens = self.kv_cache_params.num_extra_kv_tokens
        if self.num_contexts > 0 and self.enable_indexer_skip:
            # Minus the number of extra KV tokens because when using one-model MTP, the
            # draft layers needs more KV tokens for the next draft forwards.
            self.skip_indexer_for_ctx_reqs = kv_lens[:self.num_contexts].max(
            ).item() <= self.sparse_mla_topk - num_extra_kv_tokens
        else:
            self.skip_indexer_for_ctx_reqs = False

        if self.num_generations > 0 and self.enable_indexer_skip:
            # Minus the number of extra KV tokens because when using one-model MTP, the
            # draft layers needs more KV tokens for the next draft forwards.
            self.skip_indexer_for_gen_reqs = kv_lens[
                self.num_contexts:self.num_seqs].max().item(
                ) <= self.sparse_mla_topk - num_extra_kv_tokens
        else:
            self.skip_indexer_for_gen_reqs = False
        self.prepare_dense_topk_indices(kv_lens)

        # Build indexer_k_cache_block_offsets
        if self.kv_cache_manager is not None:
            block_ids = self.kv_cache_manager.get_batch_cache_indices(
                self.request_ids)
            for i in range(len(block_ids)):
                self.host_indexer_k_cache_block_offsets[
                    i, :len(block_ids[i])].copy_(
                        torch.tensor(block_ids[i],
                                     dtype=torch.int32,
                                     device='cpu'))
            self.indexer_k_cache_block_offsets[:self.num_seqs].copy_(
                self.host_indexer_k_cache_block_offsets[:self.num_seqs],
                non_blocking=True)

        # Build req_idx_per_token for topk_indices conversion
        host_req_idx_per_token = torch.repeat_interleave(torch.arange(
            self.num_seqs, dtype=torch.int32),
                                                         self.seq_lens,
                                                         dim=0)
        self.req_idx_per_token[:self.num_tokens].copy_(host_req_idx_per_token,
                                                       non_blocking=True)

        # Build block_table for topk_indices conversion (actual block allocation)
        block_ids_all = self.kv_cache_manager.get_batch_cache_indices(
            self.request_ids[:self.num_seqs])
        max_blocks_used = max(len(b)
                              for b in block_ids_all) if block_ids_all else 1
        host_block_table = torch.full((self.num_seqs, max_blocks_used),
                                      -1,
                                      dtype=torch.int32)
        for i, blocks in enumerate(block_ids_all):
            if len(blocks) > 0:
                host_block_table[i, :len(blocks)] = torch.tensor(
                    blocks, dtype=torch.int32)
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
            if self.kv_cache_manager is not None:
                block_ids = self.kv_cache_manager.get_batch_cache_indices(
                    self.request_ids)
                gen_block_ids = block_ids[self.num_contexts:]
                if len(gen_block_ids) > 0:
                    # Find max length and create padded tensor
                    max_len = max(len(bid) for bid in gen_block_ids)
                    gen_block_tensor = self.host_indexer_k_cache_block_offsets[
                        self.num_contexts:self.num_seqs, :max_len]
                    expanded_blocks = gen_block_tensor.repeat_interleave(
                        1 + self.max_draft_tokens, dim=0)
                    self.host_block_table_expanded[:num_tokens, :max_len].copy_(
                        expanded_blocks, non_blocking=True)
                    self.block_table_expanded[:num_tokens].copy_(
                        self.host_block_table_expanded[:num_tokens],
                        non_blocking=True)

        # Prepare metadata for indexer
        Indexer.prepare(metadata=self)

    def on_update_kv_lens(self):
        # After changing the kv_lens/kv_lens_cuda, we may need to update other metadatas.
        # Especially for the changes in the _preprocess_inputs() of model_engine.py.
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
            scheduler_metadata_buffer = get_paged_mqa_logits_metadata(
                self.kv_lens_cuda[self.num_contexts:self.num_seqs],
                self.kv_cache_manager.tokens_per_block, self.num_sms)
            self.scheduler_metadata_buffer.copy_(scheduler_metadata_buffer,
                                                 non_blocking=True)
            if self.use_expanded_buffers_for_mtp:
                num_draft_tokens = 1 + self.max_draft_tokens
                num_tokens = self.num_generations * num_draft_tokens
                gen_kv_lens = self.kv_lens_cuda[self.num_contexts:self.num_seqs]
                kv_lens_expanded = torch.stack([gen_kv_lens] * num_draft_tokens,
                                               dim=0)
                self.kv_lens_expanded_cuda[:num_tokens] = \
                    kv_lens_expanded.transpose(0, 1).contiguous().flatten()
                # Expand schedule metadata buffer (only generation)
                kv_lens_expanded = self.kv_lens_expanded_cuda[:num_tokens]
                scheduler_metadata_buffer_expanded = get_paged_mqa_logits_metadata(
                    kv_lens_expanded, self.kv_cache_manager.tokens_per_block,
                    self.num_sms)
                self.scheduler_metadata_buffer_expanded.copy_(
                    scheduler_metadata_buffer_expanded, non_blocking=True)
            elif self.max_draft_tokens == 3:
                scheduler_metadata_buffer_mtp3 = get_paged_mqa_logits_metadata(
                    self.kv_lens_cuda[self.num_contexts:self.num_seqs],
                    self.kv_cache_manager.tokens_per_block, self.num_sms // 2)
                self.scheduler_metadata_buffer_mtp3.copy_(
                    scheduler_metadata_buffer_mtp3, non_blocking=True)
        self.prepare_dense_topk_indices(self.kv_lens_cuda, device=True)

    def update_for_spec_dec(self):
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
    return weights * q_scale.squeeze(-1) * s


@maybe_compile(dynamic=True)
def _to_float(hidden_states: torch.Tensor) -> torch.Tensor:
    return hidden_states.float()


class Indexer(nn.Module):

    def __init__(self,
                 quant_config: Optional[QuantConfig],
                 pos_embd_params: Optional[PositionalEmbeddingParams],
                 mla_params: Optional[MLAParams],
                 skip_create_weights_in_init: bool,
                 sparse_attention_config: "SparseAttentionConfig",
                 dtype: Optional[torch.dtype],
                 layer_idx: int = 0,
                 aux_stream: Optional[torch.cuda.Stream] = None):
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
            dtype=dtype,
            quant_config=quant_config,
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

        self.rotary_emb = RotaryEmbedding(
            pos_embd_params.rope,
            head_dim=self.rope_dim,
            # RoPE in indexer is not interleaved
            is_neox=True,
        )

        self.softmax_scale = self.head_dim**-0.5
        # TODO: make it configurable from hf config
        self.scale_fmt = "ue8m0"
        self.aux_stream = aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]
        self.weight_scale_factor = self.softmax_scale * self.n_heads**-0.5

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
    def prepare(metadata: DSAtrtllmAttentionMetadata):
        """
        Prepare indexer for the forward pass.
        This should be called during metadata.prepare() stage.

        - Computes slot_mapping for KV cache updates
        - Prepares schedule_metadata for fp8_paged_mqa_logits
        - Stores generation request IDs for decode phase
        """
        num_contexts = metadata.num_contexts
        num_generations = metadata.num_generations
        num_ctx_tokens = metadata.num_ctx_tokens
        request_ids = metadata.request_ids
        seq_lens = metadata.seq_lens
        head_dim = metadata.kv_cache_manager.index_head_dim
        tokens_per_block = metadata.kv_cache_manager.tokens_per_block
        quant_block_size = metadata.kv_cache_manager.quant_block_size
        cached_tokens = metadata.kv_cache_params.num_cached_tokens_per_seq
        total_tokens = seq_lens.sum().item()

        # Prepare for prefill phase if there are context requests
        if num_contexts > 0:
            # Compute attention window bounds for each query token in batched sequences
            # cu_seqlen_ks[i]: start index in global KV for query token i
            # cu_seqlen_ke[i]: end index (exclusive) in global KV for query token i
            host_seq_lens = seq_lens[:num_contexts]
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
                gen_seq_lens = metadata.kv_lens_cuda_runtime[
                    num_contexts:num_contexts + num_generations]
                scheduler_metadata_buffer = get_paged_mqa_logits_metadata(
                    gen_seq_lens, tokens_per_block, metadata.num_sms)
                metadata.scheduler_metadata_buffer.copy_(
                    scheduler_metadata_buffer, non_blocking=True)
                if metadata.max_draft_tokens == 3:
                    scheduler_metadata_buffer_mtp3 = get_paged_mqa_logits_metadata(
                        gen_seq_lens, tokens_per_block, metadata.num_sms // 2)
                    metadata.scheduler_metadata_buffer_mtp3.copy_(
                        scheduler_metadata_buffer_mtp3, non_blocking=True)
            else:
                # Expand schedule metadata buffer (only generation)
                num_tokens = metadata.num_generations * (
                    1 + metadata.max_draft_tokens)
                kv_lens_expanded = metadata.kv_lens_expanded_cuda[:num_tokens]
                scheduler_metadata_buffer_expanded = get_paged_mqa_logits_metadata(
                    kv_lens_expanded, tokens_per_block, metadata.num_sms)
                metadata.scheduler_metadata_buffer_expanded.copy_(
                    scheduler_metadata_buffer_expanded, non_blocking=True)

        # Compute slot_mapping for all requests (both context and generation)
        # This maps each token to its flat cache position for vectorized KV cache updates
        start_positions = torch.tensor(cached_tokens, dtype=torch.int32)
        scale_size = head_dim // quant_block_size * 4  # float32 = 4 bytes
        block_stride = tokens_per_block * (head_dim + scale_size
                                           )  # Bytes per block
        scale_base_offset = tokens_per_block * head_dim  # Offset to scale region in block

        batch_size = len(request_ids)

        req_indices = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.int64, device='cpu'), seq_lens)

        token_offsets = torch.cat([
            torch.arange(seq_lens[i].item(), dtype=torch.int64, device='cpu')
            for i in range(batch_size)
        ])

        # Compute global positions for all tokens in the batch
        global_positions = start_positions[req_indices] + token_offsets

        # Block indices/pos for all tokens in the batch
        block_indices_in_seq = global_positions // tokens_per_block
        pos_in_blocks = global_positions % tokens_per_block

        max_blocks = metadata.host_indexer_k_cache_block_offsets.shape[1]
        assert (block_indices_in_seq < max_blocks).all(), \
            f"Block index out of bounds: max={max_blocks}, got indices up to {block_indices_in_seq.max().item()}"

        # Gather block IDs
        block_ids = metadata.host_indexer_k_cache_block_offsets[
            req_indices, block_indices_in_seq]

        assert (block_ids >= 0).all(), \
            f"Unallocated block (block_id < 0) found at positions {torch.where(block_ids < 0)[0].tolist()}"

        # Compute flat indices for all tokens in the batch
        fp8_flat_indices = block_ids * block_stride + pos_in_blocks * head_dim
        scale_flat_indices = block_ids * block_stride + scale_base_offset + pos_in_blocks * scale_size

        metadata.host_slot_mapping_fp8[:total_tokens] = fp8_flat_indices
        metadata.host_slot_mapping_scale[:total_tokens] = scale_flat_indices

        metadata.slot_mapping_fp8[:total_tokens].copy_(
            metadata.host_slot_mapping_fp8[:total_tokens], non_blocking=True)
        metadata.slot_mapping_scale[:total_tokens].copy_(
            metadata.host_slot_mapping_scale[:total_tokens], non_blocking=True)

        # When chunked prefill or KVCache reuse is enabled, we need to gather the full KV for indexer's logit computation.
        # Indexer's own chunking does not need full KV gathering, instead it gathers only the current chunk with loop-based gathering.
        _need_full_kv_gathering = num_contexts > 0 and metadata.enable_context_mla_with_cached_kv
        if _need_full_kv_gathering:
            total_kv_len = metadata.host_ctx_kv_indptr[num_contexts].item()
            total_kv_per_request = seq_lens[:
                                            num_contexts] + start_positions[:
                                                                            num_contexts]
            host_slot_mapping_fp8_fullkv = torch.empty(total_kv_len,
                                                       dtype=torch.int64,
                                                       pin_memory=True)
            host_slot_mapping_scale_fullkv = torch.empty(total_kv_len,
                                                         dtype=torch.int64,
                                                         pin_memory=True)

            req_indices = torch.repeat_interleave(
                torch.arange(num_contexts, dtype=torch.int64, device='cpu'),
                total_kv_per_request)

            kv_positions = torch.cat([
                torch.arange(total_kv_per_request[i].item(),
                             dtype=torch.int64,
                             device='cpu') for i in range(num_contexts)
            ])

            block_indices_in_seq = kv_positions // tokens_per_block
            pos_in_blocks = kv_positions % tokens_per_block

            max_blocks = metadata.host_indexer_k_cache_block_offsets.shape[1]
            assert (block_indices_in_seq < max_blocks).all(), \
                f"Block index out of bounds: max={max_blocks}, got indices up to {block_indices_in_seq.max().item()}"

            # Gather block IDs
            block_ids = metadata.host_indexer_k_cache_block_offsets[
                req_indices, block_indices_in_seq]

            assert (block_ids >= 0).all(), \
                f"Unallocated block (block_id < 0) found at positions {torch.where(block_ids < 0)[0].tolist()}"

            # Compute flat indices for all kv slots in the batch
            fp8_flat_indices = block_ids * block_stride + pos_in_blocks * head_dim
            scale_flat_indices = block_ids * block_stride + scale_base_offset + pos_in_blocks * scale_size

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

        # [num_blocks, block_size, 1, per_token_size ]
        k_cache = metadata.kv_cache_manager.get_indexer_k_cache_buffers(
            self.layer_idx)

        num_tokens = k_fp8.shape[0]
        head_dim = k_fp8.shape[1]
        scale_size = k_scale.shape[1] * 4  # Convert to bytes (float32 = 4 bytes)

        # Convert to bytes: flatten first, then view as uint8, then reshape
        k_fp8_bytes = k_fp8.view(-1).view(torch.uint8).view(
            num_tokens, head_dim)

        # k_scale: for single-element tensors, contiguous() may be no-op
        # Fix stride(-1) for byte-level view
        k_scale_flat = k_scale.view(-1)
        if k_scale_flat.stride(-1) != 1:
            k_scale_flat = torch.as_strided(k_scale_flat.contiguous(),
                                            size=(k_scale_flat.numel(), ),
                                            stride=(1, ))
        k_scale_bytes = k_scale_flat.view(torch.uint8).view(
            num_tokens, scale_size)

        # Use CUDA kernel to scatter FP8 and scale bytes into cache
        flat_indices_fp8 = metadata.slot_mapping_fp8[:num_tokens]
        flat_indices_scale = metadata.slot_mapping_scale[:num_tokens]
        torch.ops.trtllm.indexer_k_cache_scatter_op(k_fp8_bytes, k_scale_bytes,
                                                    k_cache, flat_indices_fp8,
                                                    flat_indices_scale)

    def _gather_k_cache_for_chunk(
        self,
        metadata: DSAtrtllmAttentionMetadata,
        chunk: IndexerPrefillChunkMetadata,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gather K values from indexer cache for a specific chunk.

        Uses pre-computed extended slot mappings that cover cached + current batch context tokens.
        chunk.k_token_start/k_token_end directly index into the extended slot mapping.

        Args:
            metadata: Attention metadata
            chunk: Chunk metadata with k_token_start/end as indices into extended slot mapping

        Returns:
            k_fp8: FP8 quantized k tensor, shape [num_k_tokens, head_dim]
            k_scale: Scaling factors, shape [num_k_tokens, 1]
        """
        assert metadata.slot_mapping_fp8_fullkv is not None, \
            "_gather_k_cache_for_chunk requires extended slot mappings (only available with cached tokens)"

        k_cache = metadata.kv_cache_manager.get_indexer_k_cache_buffers(
            self.layer_idx)

        head_dim = self.head_dim
        scale_size = 4  # float32 = 4 bytes

        # Extract slot mappings using chunk's k_token_start/end
        # These indices point directly into the extended slot mapping array
        k_token_start = chunk.k_token_start
        k_token_end = chunk.k_token_end
        num_k_tokens = k_token_end - k_token_start

        slot_mapping_fp8_chunk = metadata.slot_mapping_fp8_fullkv[
            k_token_start:k_token_end]
        slot_mapping_scale_chunk = metadata.slot_mapping_scale_fullkv[
            k_token_start:k_token_end]

        # Vectorized gather using pre-computed slot mappings
        # Gather FP8 data
        byte_offsets_fp8 = torch.arange(
            head_dim, device=k_cache.device).unsqueeze(0)  # [1, head_dim]
        gather_indices_fp8 = slot_mapping_fp8_chunk.unsqueeze(
            1) + byte_offsets_fp8  # [num_k_tokens, head_dim]
        gather_indices_fp8 = _unravel_indices(gather_indices_fp8, k_cache.shape)
        k_fp8_bytes = k_cache[gather_indices_fp8]
        k_fp8 = k_fp8_bytes.view(torch.float8_e4m3fn).view(
            num_k_tokens, head_dim)

        # Gather scale data
        byte_offsets_scale = torch.arange(
            scale_size, device=k_cache.device).unsqueeze(0)  # [1, 4]
        gather_indices_scale = slot_mapping_scale_chunk.unsqueeze(
            1) + byte_offsets_scale  # [num_k_tokens, 4]
        gather_indices_scale = _unravel_indices(gather_indices_scale,
                                                k_cache.shape)
        k_scale_bytes = k_cache[gather_indices_scale]
        k_scale = k_scale_bytes.view(torch.float32).view(num_k_tokens, 1)

        return k_fp8, k_scale

    def sparse_attn_indexer(
        self,
        metadata: DSAtrtllmAttentionMetadata,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k_fp8: torch.Tensor,
        k_scale: torch.Tensor,
        weights: torch.Tensor,
        use_custom_topk: bool = True,
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
        if not use_custom_topk:
            topk_indices_buffer[:hidden_states.shape[0]] = -1

        if has_prefill and not metadata.skip_indexer_for_ctx_reqs:
            # Use chunked prefill to reduce memory footprint
            if metadata.indexer_prefill_chunks is not None:
                for chunk in metadata.indexer_prefill_chunks:
                    # Gather K from cache for this chunk (dual to _update_k_cache)
                    chunk_k_fp8, chunk_k_scale = self._gather_k_cache_for_chunk(
                        metadata, chunk)
                    logits = fp8_mqa_logits(
                        q_fp8[chunk.token_start:chunk.token_end, ...],
                        (chunk_k_fp8, chunk_k_scale),
                        weights[chunk.token_start:chunk.token_end, ...],
                        chunk.cu_seqlen_ks,
                        chunk.cu_seqlen_ke,
                    )
                    if use_custom_topk:
                        torch.ops.trtllm.indexer_topk_prefill(
                            logits, chunk.cu_seqlen_ks, chunk.cu_seqlen_ke,
                            topk_indices_buffer[
                                chunk.token_start:chunk.token_end, :])
                    else:
                        topk_indices = logits.topk(min(self.index_topk,
                                                       logits.shape[-1]),
                                                   dim=-1)[1]
                        topk_indices -= chunk.cu_seqlen_ks[:, None]

                        mask_lo = topk_indices >= 0
                        mask_hi = topk_indices - (chunk.cu_seqlen_ke -
                                                  chunk.cu_seqlen_ks)[:,
                                                                      None] < 0
                        mask = mask_lo & mask_hi

                        # local indices per sequence
                        topk_indices = topk_indices.masked_fill(~mask, -1)

                        topk_indices_buffer[
                            chunk.token_start:chunk.token_end, :topk_indices.
                            shape[-1]] = topk_indices.to(dtype=torch.int32)
            else:
                # Fallback: single-pass indexer prefill (TODO: remove this once chunked prefill is fully tested)
                cu_seqlen_ks = metadata.cu_seqlen_ks[:num_ctx_tokens]
                cu_seqlen_ke = metadata.cu_seqlen_ke[:num_ctx_tokens]

                logits = fp8_mqa_logits(
                    q_fp8[:num_ctx_tokens, ...],
                    (k_fp8[:num_ctx_tokens, ...], k_scale[:num_ctx_tokens,
                                                          ...]),
                    weights[:num_ctx_tokens, ...],
                    cu_seqlen_ks,
                    cu_seqlen_ke,
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
                context_lens = metadata.kv_lens_cuda_runtime[
                    num_contexts:num_contexts + num_generations]
                block_table = metadata.indexer_k_cache_block_offsets[
                    num_contexts:num_contexts + num_generations]
                if q_decode.shape[1] == 4:
                    scheduler_metadata_buffer = metadata.scheduler_metadata_buffer_mtp3
                else:
                    scheduler_metadata_buffer = metadata.scheduler_metadata_buffer
            else:
                q_decode = q_decode.view(-1, 1, *q_fp8.shape[1:])
                num_tokens = q_decode.shape[0]
                context_lens = metadata.kv_lens_expanded_cuda[:num_tokens]
                block_table = metadata.block_table_expanded[:num_tokens]
                scheduler_metadata_buffer = metadata.scheduler_metadata_buffer_expanded

            assert num_gen_tokens == batch_size * next_n
            weights_decode = weights[num_ctx_tokens:num_ctx_tokens +
                                     num_gen_tokens, ...]

            # Get k cache and call fp8_paged_mqa_logits with prepared decode metadata
            # [num_blocks, tokens_per_block, 1, head_dim + scale_size]
            k_cache = metadata.kv_cache_manager.get_indexer_k_cache_buffers(
                self.layer_idx)
            logits_decode = fp8_paged_mqa_logits(q_decode, k_cache,
                                                 weights_decode, context_lens,
                                                 block_table,
                                                 scheduler_metadata_buffer,
                                                 max_seq_len)

            if use_custom_topk:
                # Kernel expects kv_lens (total cache length), not seq_lens (new tokens)
                # This is because rowEnd = seq_len - next_n + offset + 1
                gen_kv_lens_cuda = metadata.kv_lens_cuda_runtime[
                    num_contexts:num_contexts + num_generations]
                torch.ops.trtllm.indexer_topk_decode(
                    logits_decode, gen_kv_lens_cuda,
                    topk_indices_buffer[num_ctx_tokens:num_ctx_tokens +
                                        num_gen_tokens, :], next_n)
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
        elif has_decode and metadata.skip_indexer_for_gen_reqs:
            # Fill topk_indices_buffer with pre-defined dense topk indices
            topk_indices_buffer[num_ctx_tokens:num_tokens, :] = \
                metadata.topk_indices_buffer[num_ctx_tokens:num_tokens, :]
        return topk_indices_buffer

    def _weight_scale(self, weights: torch.Tensor,
                      q_scale: torch.Tensor) -> torch.Tensor:
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
        """Concatenate, rotate, and FP8 quantize for Q or K"""
        q_or_k = maybe_compiled_cat([qk_pe, qk_nope], dim=-1)
        q_or_k = rotate_activation(q_or_k)
        q_or_k = q_or_k.view(-1, self.head_dim)
        q_or_k = fp8_utils.fp8_quantize_1x128_sf_transpose(
            q_or_k, use_ue8m0=self.scale_fmt == "ue8m0")
        return q_or_k

    @torch.inference_mode()
    def forward(self, qr: torch.Tensor, hidden_states: torch.Tensor,
                metadata: DSAtrtllmAttentionMetadata,
                position_ids: torch.Tensor, indexer_k: torch.Tensor):
        quant_block_size = metadata.kv_cache_manager.quant_block_size
        assert quant_block_size == 128, "Only support quant_block_size = 128 for now"

        q_and_k, weights = maybe_execute_in_parallel(
            lambda: self._qk_projection_and_rope(qr, indexer_k, position_ids),
            lambda: self.weights_proj(_to_float(hidden_states)),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )
        q_pe, q_nope, k_pe, k_nope = q_and_k
        q, k = maybe_execute_in_parallel(
            lambda: self._prep_q_or_k(q_pe, q_nope),
            lambda: self._prep_q_or_k(k_pe, k_nope),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )
        q_fp8, q_scale = q
        k_fp8, k_scale = k
        q_fp8 = q_fp8.view(-1, self.n_heads, self.head_dim)
        q_scale = q_scale.view(-1, self.n_heads, 1)

        weights, _ = maybe_execute_in_parallel(
            lambda: self._weight_scale(weights, q_scale),
            lambda: self._update_k_cache(
                k_fp8, k_scale, metadata),  # store k_fp8 and k_scale in k cache
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )

        # Return topk indices buffer for sparse attention [num_tokens, index_topk]
        return self.sparse_attn_indexer(metadata, hidden_states, q_fp8, k_fp8,
                                        k_scale, weights)


class DSATrtllmAttention(TrtllmAttention):
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
        hidden_states: Optional[torch.Tensor] = None,
        qr: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        is_generation: bool = True,
        **kwargs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Transform the local topk indices to global topk indices in paged kv cache
        topk_indices_global, _ = transform_local_topk_and_prepare_pool_view(
            topk_indices, metadata, self.get_local_layer_idx(metadata),
            is_generation)

        # TODO: Use sparse_attn_indexer to predict the indices for DSA attention
        # return self.indexer(q, k, metadata, hidden_states, qr, position_ids)
        return topk_indices_global, None

    def sparse_kv_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: DSAtrtllmAttentionMetadata,
        hidden_states: Optional[torch.Tensor] = None,
        qr: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return None, None

    def mla_rope_append_paged_kv_assign_q(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        metadata: DSAtrtllmAttentionMetadata,
        is_generation: bool = False,
        **kwargs,
    ) -> None:
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
            self.wrapper.rotary_cos_sin,
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
            self.wrapper.quant_mode,
        )


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
        sparse_attn_config: "SparseAttentionConfig",
        **kwargs,
    ) -> None:
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
            enable_indexer_k_cache=True,
            indexer_k_cache_quant_block_size=128,
            indexer_k_cache_index_head_dim=self.index_head_dim,
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
        per_token_size = self.index_head_dim + self.index_head_dim // self.quant_block_size * 4
        layer_offset = self.layer_offsets[layer_idx]
        return self.indexer_k_cache_pool_per_layer[layer_offset].view(
            self.num_blocks, block_size, 1, per_token_size)

    def shutdown(self):
        # Clear Python references BEFORE C++ frees the underlying CUDA buffers
        self.indexer_k_cache_pool_per_layer = []
        super().shutdown()

    @staticmethod
    def get_cache_size_per_token(model_config: ModelConfig, mapping: Mapping,
                                 **kwargs):
        config = model_config.pretrained_config
        sparse_attn_config = model_config.sparse_attention_config
        index_head_dim = sparse_attn_config.index_head_dim
        quant_block_size = 128

        # get kv cache dtype bytes
        mem_per_token = 2
        quant_config = model_config.quant_config
        if quant_config is not None and quant_config.quant_mode.has_fp8_kv_cache(
        ):
            mem_per_token = 1

        # get head dim
        head_dim = config.kv_lora_rank + config.qk_rope_head_dim

        # provide at least 1 layer to prevent division by zero cache size
        num_attention_layers = max(
            len(mapping.pp_layers(model_config.get_num_attention_layers())), 1)
        mem_per_token *= num_attention_layers * head_dim

        # 1 for K, others for indexer K cache
        head_dim_factor = (index_head_dim +
                           index_head_dim // quant_block_size * 4) / head_dim
        kv_factor = 1 + head_dim_factor
        mem_per_token *= kv_factor
        return mem_per_token

    def get_cache_bytes_per_token(self):
        # self.kv_factor for K, others for indexer K cache
        head_dim_factor = (self.index_head_dim + self.index_head_dim //
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
