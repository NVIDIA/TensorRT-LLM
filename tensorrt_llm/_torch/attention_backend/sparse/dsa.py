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
from tensorrt_llm._torch.utils import maybe_compile
from tensorrt_llm._utils import get_size_in_bytes
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
    assert num_layers == kv_cache_manager.num_local_layers, "PP is not enabled yet for DeepSeek V3.2"
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
        torch.cumsum(seq_lens, dim=0).to(torch.int32)
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


@dataclass
class IndexerPrefillChunkMetadata:
    """Metadata for a single prefill chunk in the indexer"""
    cu_seqlen_ks: torch.Tensor  # Attention window start for each token
    cu_seqlen_ke: torch.Tensor  # Attention window end for each token
    token_start: int  # Q token start index in batch
    token_end: int  # Q token end index in batch
    k_token_start: int  # K token start index in batch
    k_token_end: int  # K token end index in batch
    ctx_kv_offsets: torch.Tensor  # Offsets for converting local to global topk indices


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

    def __init__(self, *args, **kwargs):
        self.num_sms = tensorrt_llm.deep_gemm.get_num_sms()
        super().__init__(*args, **kwargs)
        if self.sparse_attention_config.indexer_max_chunk_size is not None:
            self.indexer_max_chunk_size = self.sparse_attention_config.indexer_max_chunk_size
        else:
            self.indexer_max_chunk_size = 32768  # Default to 32K tokens for the indexer
        self.sparse_mla_topk = self.sparse_attention_config.index_topk

    def __post_init__(self):
        super().__post_init__()

        capture_graph = torch.cuda.is_current_stream_capturing()

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

        # For mla_rope_append_paged_kv_assign_q
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

    def prepare(self):
        super().prepare()
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
        assert self.kv_cache_params.use_cache is True, "DSA requires use_cache to be True"
        cached_token_lens = torch.tensor(
            self.kv_cache_params.num_cached_tokens_per_seq,
            dtype=torch.int,
            device='cpu',
        )
        kv_lens = cached_token_lens + self.seq_lens_kv

        if self.num_contexts > 0 and not self.enable_context_mla_with_cached_kv:
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
        elif self.num_contexts == 0:
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

        # Because the fp8_paged_mqa_logits only supports seq_len == 1 or 2, so it cannot support
        # MTP > 1. To handle this, when MTP > 1, we flatten the q tensor and expand the kv_lens and
        # block_table for to use the fp8_paged_mqa_logits.
        # TODO: remove this when fp8_paged_mqa_logits supports MTP > 1.
        if self.max_draft_tokens > 1:
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
            tokens_per_block = self.kv_cache_manager.indexer_k_cache_tokens_per_block
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
                tokens_per_block, self.num_sms)
            self.scheduler_metadata_buffer.copy_(scheduler_metadata_buffer,
                                                 non_blocking=True)

    def update_for_spec_dec(self):
        super().update_for_spec_dec()
        self.kv_cache_manager.indexer_k_cache_tokens_per_block
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
            dtype=dtype,
            quant_config=None,
            skip_create_weights_in_init=skip_create_weights_in_init,
            use_custom_cublas_mm=True)

        self.rotary_emb = RotaryEmbedding(
            pos_embd_params.rope,
            head_dim=self.rope_dim,
            is_neox=pos_embd_params.is_neox,
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
        seq_lens_cpu: torch.Tensor,
    ) -> IndexerPrefillChunkMetadata:
        """
        Build metadata for one prefill chunk for indexer forward pass.
        Handles both multi-request chunks and intra-request Q-block chunks.

        Args:
            metadata: Attention metadata
            chunk_specs: List of (req_idx, token_start_in_req, token_end_in_req, req_cum_start)
                        - For multi-request: multiple specs from different requests
                        - For intra-request: single spec from one request's Q-block
            seq_lens_cpu: Sequence lengths for all requests
        """
        device = metadata.cu_seqlen_ks.device

        if len(chunk_specs) == 1:
            # Single request or intra-request Q-block
            req_idx, token_start_in_req, token_end_in_req, req_cum_start = chunk_specs[
                0]
            num_q_tokens = token_end_in_req - token_start_in_req

            # For intra-request chunks: Q block attends to all previous K in the request
            # Q tokens [token_start_in_req:token_end_in_req] within the request
            # K tokens [0:token_end_in_req] within the request (causal attention)

            cu_seqlen_ks = torch.zeros(num_q_tokens,
                                       dtype=torch.int32,
                                       device='cpu')
            cu_seqlen_ke = torch.arange(token_start_in_req + 1,
                                        token_end_in_req + 1,
                                        dtype=torch.int32,
                                        device='cpu')

            # Q token range in batch
            token_start = req_cum_start + token_start_in_req
            token_end = req_cum_start + token_end_in_req

            # K token range in batch: from request start to Q block end
            k_token_start = req_cum_start
            k_token_end = req_cum_start + token_end_in_req

            kv_offset = metadata.host_ctx_kv_indptr[req_idx].item()
            ctx_kv_offsets = torch.full((num_q_tokens, 1),
                                        kv_offset,
                                        dtype=torch.int32,
                                        device='cpu')
        else:
            # Multi-request chunk: batch multiple full requests together
            # Extract sequence lengths for these requests
            req_seq_lens = []
            for spec in chunk_specs:
                req_idx, token_start_in_req, token_end_in_req, _ = spec
                assert token_start_in_req == 0, "Multi-request chunk must contain full requests"
                assert token_end_in_req == seq_lens_cpu[req_idx].item(), \
                    "Multi-request chunk must contain full requests"
                req_seq_lens.append(token_end_in_req)

            req_seq_lens_tensor = torch.tensor(req_seq_lens,
                                               dtype=torch.int32,
                                               device='cpu')
            num_q_tokens = sum(req_seq_lens)

            # Compute causal attention bounds for batched requests
            cu_seqlen_ks, cu_seqlen_ke = compute_cu_seqlen_kv_bounds_nocache(
                req_seq_lens_tensor, len(chunk_specs), num_q_tokens)

            # Global Q and K token ranges (same for multi-request chunks)
            token_start = chunk_specs[0][3]  # req_cum_start of first request
            token_end = chunk_specs[-1][3] + chunk_specs[-1][
                2]  # last req start + length
            k_token_start = token_start
            k_token_end = token_end

            # KV offsets for each request
            ctx_kv_offsets_list = []
            for spec in chunk_specs:
                req_idx, _, token_end_in_req, _ = spec
                kv_offset = metadata.host_ctx_kv_indptr[req_idx].item()
                ctx_kv_offsets_list.append(
                    torch.full((token_end_in_req, ),
                               kv_offset,
                               dtype=torch.int32,
                               device='cpu'))
            ctx_kv_offsets = torch.cat(ctx_kv_offsets_list).unsqueeze(1)

        return IndexerPrefillChunkMetadata(
            cu_seqlen_ks=cu_seqlen_ks.to(device, non_blocking=True),
            cu_seqlen_ke=cu_seqlen_ke.to(device, non_blocking=True),
            token_start=token_start,
            token_end=token_end,
            k_token_start=k_token_start,
            k_token_end=k_token_end,
            ctx_kv_offsets=ctx_kv_offsets.to(device, non_blocking=True),
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
        tokens_per_block = metadata.kv_cache_manager.indexer_k_cache_tokens_per_block
        quant_block_size = metadata.kv_cache_manager.quant_block_size
        cached_tokens = metadata.kv_cache_params.num_cached_tokens_per_seq
        total_tokens = seq_lens.sum().item()

        # Prepare for prefill phase if there are context requests
        if num_contexts > 0:
            # Compute attention window bounds for each query token in batched sequences
            # cu_seqlen_ks[i]: start index in global KV for query token i
            # cu_seqlen_ke[i]: end index (exclusive) in global KV for query token i
            host_seq_lens = seq_lens[:num_contexts]

            # Two-level chunking: request-boundary + intra-request
            chunk_groups = split_prefill_chunks(
                host_seq_lens,
                metadata.indexer_max_chunk_size,
                start_idx=0,
            )

            # Enable chunking when num_chunk > 1
            if len(chunk_groups) > 1:
                metadata.indexer_prefill_chunks = [
                    Indexer.prepare_one_prefill_chunk(
                        metadata,
                        chunk_specs,
                        host_seq_lens,
                    ) for chunk_specs in chunk_groups
                ]
            else:
                # Single chunk - use non-chunked fallback path
                metadata.indexer_prefill_chunks = None

            # TODO: Still compute global cu_seqlen bounds for fallback/debugging.
            # Remove this when indexer chunked prefill is fully tested.
            # Still compute global cu_seqlen bounds for fallback/debugging
            host_cu_seqlen_ks, host_cu_seqlen_ke = compute_cu_seqlen_kv_bounds_nocache(
                host_seq_lens, num_contexts, num_ctx_tokens)
            metadata.cu_seqlen_ks[:num_ctx_tokens].copy_(host_cu_seqlen_ks,
                                                         non_blocking=True)
            metadata.cu_seqlen_ke[:num_ctx_tokens].copy_(host_cu_seqlen_ke,
                                                         non_blocking=True)

        # Prepare for decode phase if there are generation requests
        if num_generations > 0:
            # Prepare schedule metadata for fp8_paged_mqa_logits
            # This is a preprocessing step that computes scheduling information for the kernel
            if metadata.max_draft_tokens <= 1:
                gen_seq_lens = metadata.kv_lens_cuda_runtime[
                    num_contexts:num_contexts + num_generations]
                scheduler_metadata_buffer = get_paged_mqa_logits_metadata(
                    gen_seq_lens, tokens_per_block, metadata.num_sms)
                metadata.scheduler_metadata_buffer.copy_(
                    scheduler_metadata_buffer, non_blocking=True)
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

        token_idx = 0
        for req_idx, req_id in enumerate(request_ids):
            num_tokens = seq_lens[req_idx].item()
            start_pos = start_positions[req_idx].item()
            # Compute slots for each token in this request
            for local_token_idx in range(num_tokens):
                global_pos = start_pos + local_token_idx
                block_idx_in_seq = global_pos // tokens_per_block
                pos_in_block = global_pos % tokens_per_block
                # Get physical block ID
                if block_idx_in_seq < metadata.host_indexer_k_cache_block_offsets.shape[
                        1]:
                    block_id = metadata.host_indexer_k_cache_block_offsets[
                        req_idx, block_idx_in_seq].item()
                    if block_id >= 0:
                        # Flat byte index for FP8 data
                        fp8_flat_idx = block_id * block_stride + pos_in_block * head_dim
                        metadata.host_slot_mapping_fp8[token_idx] = fp8_flat_idx

                        # Flat byte index for scale
                        scale_flat_idx = block_id * block_stride + scale_base_offset + pos_in_block * scale_size
                        metadata.host_slot_mapping_scale[
                            token_idx] = scale_flat_idx
                token_idx += 1

        metadata.slot_mapping_fp8[:total_tokens].copy_(
            metadata.host_slot_mapping_fp8[:total_tokens], non_blocking=True)
        metadata.slot_mapping_scale[:total_tokens].copy_(
            metadata.host_slot_mapping_scale[:total_tokens], non_blocking=True)

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

        Uses pre-computed slot_mapping for efficient vectorized gather.
        For intra-request Q-blocks, gathers all previous K tokens in the request.

        Args:
            metadata: Attention metadata
            chunk: Chunk metadata containing token ranges

        Returns:
            k_fp8: FP8 quantized k tensor, shape [num_k_tokens, head_dim]
            k_scale: Scaling factors, shape [num_k_tokens, 1]
        """
        k_cache = metadata.kv_cache_manager.get_indexer_k_cache_buffers(
            self.layer_idx)

        head_dim = self.head_dim
        scale_size = 4  # float32 = 4 bytes

        # Extract slot mappings for K token range
        k_token_start = chunk.k_token_start
        k_token_end = chunk.k_token_end
        num_k_tokens = k_token_end - k_token_start

        slot_mapping_fp8_chunk = metadata.slot_mapping_fp8[
            k_token_start:k_token_end]
        slot_mapping_scale_chunk = metadata.slot_mapping_scale[
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
        use_custom_topk = use_custom_topk and self.index_topk == 2048  #@TODO: Do we need to add a warning?
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

        if has_prefill:
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
                        torch.ops.trtllm.indexer_topk_prefill_op(
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
                    torch.ops.trtllm.indexer_topk_prefill_op(
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

        if has_decode:
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
            # Because fp8_paged_mqa_logits cannot support next_n > 2, we need to flatten the q_decode tensor
            # and expand the corresponding metadata.
            if next_n <= 2:
                q_decode = q_decode.view(num_generations, -1, *q_fp8.shape[1:])
                context_lens = metadata.kv_lens_cuda_runtime[
                    num_contexts:num_contexts + num_generations]
                block_table = metadata.indexer_k_cache_block_offsets[
                    num_contexts:num_contexts + num_generations]
                scheduler_metadata_buffer = metadata.scheduler_metadata_buffer
            else:
                q_decode = q_decode.view(-1, 1, *q_fp8.shape[1:])
                num_tokens = num_generations * (1 + metadata.max_draft_tokens)
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
                torch.ops.trtllm.indexer_topk_decode_op(
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
        return topk_indices_buffer

    def weight_scale(self, hidden_states: torch.Tensor,
                     indexer_weights: Optional[torch.Tensor],
                     q_scale: torch.Tensor) -> torch.Tensor:
        weights = indexer_weights if indexer_weights is not None else self.weights_proj(
            hidden_states)
        weights = _scale(weights, q_scale, self.weight_scale_factor)
        return weights

    @torch.inference_mode()
    def forward(self, qr: torch.Tensor, hidden_states: torch.Tensor,
                metadata: DSAtrtllmAttentionMetadata,
                position_ids: torch.Tensor, indexer_k: Optional[torch.Tensor],
                indexer_weights: Optional[torch.Tensor]):
        quant_block_size = metadata.kv_cache_manager.quant_block_size
        assert quant_block_size == 128, "Only support quant_block_size = 128 for now"

        if indexer_k is not None:
            q, k = maybe_execute_in_parallel(
                lambda: self.wq_b(
                    qr),  # TODO: fuse wq_b and move this outside of the indexer
                lambda: self.k_norm(indexer_k),
                self.ln_events[0],
                self.ln_events[1],
                self.aux_stream,
            )
        else:
            q, k = maybe_execute_in_parallel(
                lambda: self.wq_b(qr),
                lambda: self.k_norm(self.wk(hidden_states)),
                self.ln_events[0],
                self.ln_events[1],
                self.aux_stream,
            )

        # q/k rope + possible fast_hadamard_transform
        q = q.view(-1, self.n_heads, self.head_dim)

        q, k = maybe_execute_in_parallel(
            lambda: torch.split(
                q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1),
            lambda: torch.split(
                k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )

        q_pe, q_nope = q
        k_pe, k_nope = k
        q_pe, k_pe = self.rotary_emb(position_ids, [q_pe, k_pe.unsqueeze(1)])

        k_pe = k_pe[:, 0, :]

        def _prep_q_or_k(qk_pe, qk_nope):
            q_or_k = torch.cat([qk_pe, qk_nope], dim=-1)
            q_or_k = rotate_activation(q_or_k)
            q_or_k = q_or_k.view(-1, self.head_dim)
            q_or_k = fp8_utils.fp8_quantize_1x128_sf_transpose(
                q_or_k, use_ue8m0=self.scale_fmt == "ue8m0")
            return q_or_k

        q, k = maybe_execute_in_parallel(
            lambda: _prep_q_or_k(q_pe, q_nope),
            lambda: _prep_q_or_k(k_pe, k_nope),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )

        q_fp8, q_scale = q
        k_fp8, k_scale = k
        q_fp8 = q_fp8.view(-1, self.n_heads, self.head_dim)
        q_scale = q_scale.view(-1, self.n_heads, 1)

        weights, _ = maybe_execute_in_parallel(
            lambda: self.weight_scale(hidden_states, indexer_weights, q_scale),
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
            topk_indices, metadata, self.layer_idx, is_generation)

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
            host_block_offsets = metadata.host_kv_cache_block_offsets[:,
                                                                      metadata.
                                                                      num_contexts:]
        else:
            cached_token_indptr = metadata.ctx_cached_token_indptr
            kv_indptr = metadata.ctx_kv_indptr
            num_seqs = metadata.num_contexts
            max_seq_len = metadata.max_ctx_seq_len
            block_offsets = metadata.kv_cache_block_offsets
            host_block_offsets = metadata.host_kv_cache_block_offsets
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
            host_block_offsets,
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

        if kv_cache_config.enable_block_reuse:
            raise NotImplementedError(
                "DSA indexer K-cache manager does not support block reuse yet")
        self.quant_block_size = 128
        self.index_head_dim = sparse_attn_config.index_head_dim
        # Use a fixed tokens_per_block for indexer k cache due to DG kernel constraints
        self.indexer_k_cache_tokens_per_block = 64
        assert self.indexer_k_cache_tokens_per_block == tokens_per_block, "tokens_per_block must be set to 64 for DeepSeek v3.2"

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
        # Shape: [num_blocks, self.indexer_k_cache_tokens_per_block * (index_head_dim + scale_size)]
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

    @staticmethod
    def get_cache_size_per_token(model_config: ModelConfig, mapping: Mapping,
                                 **kwargs):
        config = model_config.pretrained_config
        sparse_attn_config = model_config.sparse_attention_config
        index_head_dim = sparse_attn_config.index_head_dim
        tokens_per_block = kwargs['tokens_per_block']
        quant_block_size = 128
        indexer_k_cache_tokens_per_block = 64

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
        tokens_per_block_factor = indexer_k_cache_tokens_per_block / tokens_per_block
        kv_factor = 1 + head_dim_factor * tokens_per_block_factor
        mem_per_token *= kv_factor
        return mem_per_token

    def get_cache_bytes_per_token(self):
        # self.kv_factor for K, others for indexer K cache
        head_dim_factor = (self.index_head_dim + self.index_head_dim //
                           self.quant_block_size * 4) / self.head_dim
        tokens_per_block_factor = self.indexer_k_cache_tokens_per_block / self.tokens_per_block
        kv_factor = self.kv_factor + head_dim_factor * tokens_per_block_factor
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
