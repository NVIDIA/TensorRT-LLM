# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense Sparse Attention (DSA) backend for TRT-LLM with indexer-based TopK selection."""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.attention_backend.interface import MLAParams, PositionalEmbeddingParams
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.distributed.ops import allgather
from tensorrt_llm._torch.modules.layer_norm import LayerNorm
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel
from tensorrt_llm._torch.modules.rotary_embedding import RotaryEmbedding
from tensorrt_llm._torch.utils import Fp4QuantizedTensor, maybe_compile
from tensorrt_llm._utils import get_sm_version, maybe_pin_memory, prefer_pinned
from tensorrt_llm.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    fp8_mqa_logits,
    fp8_paged_mqa_logits,
    get_paged_mqa_logits_metadata,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig

from .params import DSAParams

ModelConfig = tensorrt_llm.bindings.ModelConfig

# Cap the per-call indexer MQA-logits transient (in elements). fp8_mqa_logits
# allocates its [q x kv] logits output via torch.empty; the KV dimension is the
# full (compressed) context and is unbounded, so for a large query chunk on a
# long-context prefill this single allocation can reach tens of GB. Under
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True such an allocation can stall
# indefinitely in cuMemCreate on the longest-context (attention_dp laggard) rank
# -> GPU idle -> peers block at the next MoE all-to-all -> watchdog hang. Tiling
# the query dimension caps the transient to q_tile x kv with identical results
# (each query row's logits/top-k are independent). Override via env if needed.
_INDEXER_MQA_LOGITS_ELEM_BUDGET = int(
    os.environ.get("TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET", 1 << 31)
)

if TYPE_CHECKING:
    from .metadata import DSAtrtllmAttentionMetadata

# Optional import: fast-hadamard-transform causes CI build issues (requires wheel+torch pre-installed)
try:
    from fast_hadamard_transform import hadamard_transform

    HAS_FAST_HADAMARD = True
except ImportError:
    hadamard_transform = None
    HAS_FAST_HADAMARD = False

# Idempotency guard for warmup_heuristic_topk_decode — keyed by
# (device_index, top_k, hint_size, num_cols). Prevents repeated allocations
# and synchronizations when multiple Indexer modules invoke the warmup with
# the same parameters during model construction.
_HEURISTIC_TOPK_WARMUP_DONE: Set[Tuple[int, int, int, int]] = set()
_HEURISTIC_TOPK_WARMUP_LOCK = threading.Lock()
_DG_SCHEDULE_BLOCK_KV = 64


def warmup_heuristic_topk_decode(
    top_k: int = 2048, hint_size: int = 2048, num_cols: int = 4096
) -> None:
    """Pre-initialize cached hardware attributes in the C++ Scheme X dispatcher.

    The dispatcher inside ``invokeIndexerTopKDecode`` lazily queries
    ``cudaDeviceGetAttribute`` for ``MultiProcessorCount`` and
    ``L2CacheSize`` on its first call. Those host-side queries must not
    be issued during ``cudaStreamBeginCapture / EndCapture``: the values
    captured there become frozen into the graph and cannot be refreshed
    across replays on a different device.

    This warmup issues one small heuristic decode call so the static
    caches are populated before any CUDA Graph capture begins. Must be
    called from the Indexer setup hook (``layer_idx == 0``) when
    ``enable_heuristic_topk`` is true.

    Repeated invocations with the same ``(device, top_k, hint_size,
    num_cols)`` key are short-circuited so that constructing many Indexer
    modules in the same process does not re-allocate scratch tensors or
    issue redundant synchronizations.
    """
    key = (torch.cuda.current_device(), top_k, hint_size, num_cols)
    with _HEURISTIC_TOPK_WARMUP_LOCK:
        if key in _HEURISTIC_TOPK_WARMUP_DONE:
            return
        _HEURISTIC_TOPK_WARMUP_DONE.add(key)

    device = torch.device("cuda")
    logits = torch.zeros((1, num_cols), dtype=torch.float32, device=device)
    seq_lens = torch.tensor([num_cols], dtype=torch.int32, device=device)
    indices = torch.empty((1, top_k), dtype=torch.int32, device=device)
    pre_idx = torch.zeros((1, hint_size), dtype=torch.int32, device=device)
    scratch = torch.empty((top_k,), dtype=torch.float32, device=device)
    # The default warmup geometry (num_cols=4096) falls below kSeqSmall=12288
    # and routes to the Radix path with blocks_per_row=2 (num_rows=1 sweeps
    # bp ∈ [2, maxByCols=2]). The cpp op rejects blocks_per_row > 1 without
    # caller-owned radix aux scratch, so supply worst-case (kMaxBlocksPerRowDecode=10)
    # buffers here. Cost is negligible (~80 KB) and the warmup is a one-shot.
    _radix_max_bp = 10
    radix_aux_indices = torch.empty((1, _radix_max_bp, top_k), dtype=torch.int32, device=device)
    radix_aux_logits = torch.empty((1, _radix_max_bp, top_k), dtype=torch.float32, device=device)
    torch.ops.trtllm.indexer_topk_decode(
        logits,
        seq_lens,
        indices,
        1,
        top_k,
        pre_idx=pre_idx,
        heuristic_scratch=scratch,
        radix_aux_indices=radix_aux_indices,
        radix_aux_logits=radix_aux_logits,
    )
    torch.cuda.synchronize()


def _pick_dsl_expand(
    next_n: int,
    num_sms: int,
    batch_size: int = 0,
    max_ctx: int = 0,
    kernel_atoms: Tuple[int, ...] = (1, 2, 3),
) -> Tuple[int, int]:
    """Pick (expand_factor, effective_next_n) for the DSL paged kernel
    using a wave-aware strategy. Used by both FP4 and FP8 DSL paths.

    The DSL kernel natively supports ``effective_next_n ∈ kernel_atoms``
    (FP4: ``(1, 2, 3)``; FP8: ``(1, 2, 3, 4)``). For ``next_n`` not natively
    supported or when SM utilization can be improved, reshape
    ``[B, next_n, ...]`` -> ``[B * expand_factor, effective_next_n, ...]``
    caller-side.

    Strategy: enumerate ``(expand_factor, effective_next_n)`` pairs with
    ``expand_factor * effective_next_n == next_n`` and ``effective_next_n
    in kernel_atoms``. Score each by ``(waves, -expand_factor)`` where
    ``waves = ceil(B * expand_factor * ceil(max_ctx/256) / num_sms)``.
    Pick min waves; on tie, prefer LARGER expand_factor (more SMs busy per
    wave; pays HBM cost of expand_factor x KV re-reads).

    When ``batch_size == 0`` or ``max_ctx == 0`` (workload unknown), fall
    back to the legacy HBM-minimizing heuristic: largest effective_next_n
    that divides next_n cleanly (still constrained to ``kernel_atoms``).

    Examples (wave-aware, num_sms=148 [B200], SPLIT_KV=256 tokens):
        FP4, next_n=4, B=1,  ctx=4096   -> (4, 1): ntask=64<148, 1 wave, max factor
        FP4, next_n=4, B=32, ctx=4096   -> (2, 2): ntask=1024>148, multi-wave, min factor
        FP4, next_n=2, B=1,  ctx=4096   -> (2, 1): wave-tie, larger factor
        FP8, next_n=4, B=1,  ctx=4096   -> (4, 1): kernel_atoms incl. 4 doesn't change small-B pick
    """
    # Legacy fallback when workload is unknown.
    if batch_size <= 0 or max_ctx <= 0:
        for eff in sorted(kernel_atoms, reverse=True):
            if next_n % eff == 0:
                return next_n // eff, eff
        return next_n, 1

    SPLIT_KV_TOKENS = 256
    cands = []
    for eff in kernel_atoms:
        if next_n % eff == 0:
            factor = next_n // eff
            ntask = batch_size * factor * ((max_ctx + SPLIT_KV_TOKENS - 1) // SPLIT_KV_TOKENS)
            waves = (ntask + num_sms - 1) // num_sms
            cands.append((waves, factor, eff))
    if not cands:
        return next_n, 1
    cands.sort(key=lambda x: (x[0], -x[1]))  # min waves, max factor
    _, factor, eff = cands[0]
    return factor, eff


def _compute_slot_mappings(
    global_positions: torch.Tensor,
    block_offsets: torch.Tensor,
    req_indices: torch.Tensor,
    head_dim: int,
    tokens_per_block: int,
    quant_block_size: int,
    data_bytes_per_token: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute flat byte indices for FP8/FP4 data and scales from global token positions.

    Shared by Indexer.prepare() (CPU) and on_update_kv_lens() (GPU) to avoid
    duplicating the slot mapping arithmetic.

    Args:
        global_positions: Per-token absolute position in the KV sequence.
        block_offsets: [num_seqs, max_blocks_per_seq] block offset table.
        req_indices: Per-token request index.
        head_dim: Indexer head dimension (logical element count).
        tokens_per_block: Tokens stored per cache block.
        quant_block_size: Quantization block size.
        data_bytes_per_token: Bytes of quantized data per token in the cache
            (default: head_dim, i.e. one byte per FP8 value). For MXFP4 this
            should be ``head_dim // 2`` since two E2M1 codes pack into one byte.
            The scale layout is identical at head_dim=128: 4 bytes per token
            (one float32 for FP8, four packed UE8M0 exponents for FP4).

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
        assert (block_indices_in_seq < max_blocks).all(), (
            f"Block index out of bounds: max={max_blocks}, got indices up to {block_indices_in_seq.max().item()}"
        )

    block_ids = block_offsets[req_indices, block_indices_in_seq].to(torch.int64)

    fp8_indices = block_ids * block_stride + pos_in_blocks * data_bytes_per_token
    scale_indices = block_ids * block_stride + scale_base_offset + pos_in_blocks * scale_size
    return fp8_indices, scale_indices


def _unravel_indices(
    flat_indices: torch.Tensor, shape: Tuple[int, ...]
) -> Tuple[torch.Tensor, ...]:
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
    """Apply Hadamard rotation to activation tensor for DSA sparse attention."""
    assert x.dtype == torch.bfloat16

    if not HAS_FAST_HADAMARD:
        # Fallback: skip transformation (acceptable for test/dev)
        logger.warning_once(
            "fast-hadamard-transform not available. Sparse MLA will skip "
            "hadamard transformation. Install with: "
            "pip install git+https://github.com/Dao-AILab/fast-hadamard-transform.git",
            key="fast_hadamard_import_missing",
        )
        return x

    hidden_size = x.size(-1)
    assert (hidden_size & (hidden_size - 1)) == 0, (
        "Hidden size must be a power of 2 for Hadamard transform."
    )
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
    query_start_loc_cpu = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device="cpu"), seq_lens.cumsum(dim=0).to(torch.int32)]
    )

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
                    chunk_specs.append((next_req, 0, next_seq_len, next_cum_start))
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
                q_block_spec = [(current_req, token_start, token_end, req_cum_start)]
                chunk_groups.append(q_block_spec)

            current_req += 1

    return chunk_groups


# Shrink the indexer prefill chunk size for very long requests to bound the
# fp8(_fp4)_mqa_logits activation memory (~ chunk_size * K_compressed), keyed on
# the largest compressed KV length in the batch. Entries are
# (k_compressed_lower_bound_exclusive, chunk_size); >512K -> 8K, [256K, 512K]
# -> 16K, otherwise unchanged.
_INDEXER_CHUNK_SIZE_HEURISTIC = (
    (512 * 1024, 8 * 1024),
    (256 * 1024 - 1, 16 * 1024),
)


def select_indexer_chunk_size(configured_chunk_size: int, max_k_compressed: int) -> int:
    """Pick the indexer prefill chunk size from the batch's largest K_compressed.

    Only reduces ``configured_chunk_size`` (never increases it).
    """
    for threshold, chunk_size in _INDEXER_CHUNK_SIZE_HEURISTIC:
        if max_k_compressed > threshold:
            return min(configured_chunk_size, chunk_size)
    return configured_chunk_size


def _select_indexer_compress_ratio(compress_ratios: List[int]) -> int:
    if 4 in compress_ratios:
        return 4
    if 1 in compress_ratios:
        return 1
    return 0


def _effective_compress_ratio_divisor(compress_ratio: int) -> int:
    return compress_ratio if compress_ratio > 1 else 1


def compute_cu_seqlen_kv_bounds_with_cache(
    seq_lens: torch.Tensor,
    num_contexts: int,
    num_ctx_tokens: int,
    cached_token_lens: Optional[torch.Tensor] = None,
    kv_lens: Optional[torch.Tensor] = None,
    compress_ratio: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute attention window bounds for batched sequences with causal attention,
    accounting for cached KV tokens.

    Args:
        seq_lens: current token lengths [num_contexts], dtype=torch.int32
        num_contexts: Number of sequences in the batch
        num_ctx_tokens: Total number of context tokens across all sequences in current batch
        cached_token_lens: Cached KV token lengths [num_contexts], dtype=torch.int32 (optional)
        kv_lens: KV token lengths [num_contexts], dtype=torch.int32 (optional)
        compress_ratio: Compression ratio for KV tokens

    Returns:
        cu_seqlen_ks: Start index in KV for each Q token [num_ctx_tokens]
        cu_seqlen_ke: End index (exclusive) in KV for each Q token [num_ctx_tokens]
    """
    device = seq_lens.device
    # Total KV lengths per request
    if kv_lens is None:
        kv_lens = (
            seq_lens if cached_token_lens is None else cached_token_lens + seq_lens
        )  # [num_contexts]
        kv_lens = kv_lens // compress_ratio

    # Cumulative KV offsets: where each request's KV sequence starts in global KV space
    cu_kv_offsets = torch.cat(
        [
            torch.zeros(1, device=device, dtype=torch.int32),
            torch.cumsum(kv_lens, dim=0).to(torch.int32),
        ]
    )  # [num_contexts + 1]

    # Map each Q token to its request: [0,0,...,0, 1,1,...,1, ..., B-1,B-1,...,B-1]
    batch_ids = torch.repeat_interleave(
        torch.arange(num_contexts, device=device, dtype=torch.int32), seq_lens
    )  # [num_ctx_tokens]

    # Each Q token's KV window starts at its request's KV sequence start
    cu_seqlen_ks = cu_kv_offsets[batch_ids]  # [num_ctx_tokens]

    # Compute local Q position within each request (0-based, relative to current batch context tokens)
    cu_q_offsets = torch.cat(
        [
            torch.zeros(1, device=device, dtype=torch.int32),
            torch.cumsum(seq_lens, dim=0).to(torch.int32),
        ]
    )  # [num_contexts + 1]

    global_q_positions = torch.arange(num_ctx_tokens, device=device, dtype=torch.int32)
    local_q_positions = global_q_positions - torch.repeat_interleave(
        cu_q_offsets[:-1], seq_lens
    )  # [num_ctx_tokens]

    if cached_token_lens is not None:
        cached_per_token = torch.repeat_interleave(cached_token_lens, seq_lens)  # [num_ctx_tokens]
        cu_seqlen_ke = (
            cu_seqlen_ks + (cached_per_token + local_q_positions + 1) // compress_ratio
        )  # [num_ctx_tokens]
    else:
        cu_seqlen_ke = cu_seqlen_ks + (local_q_positions + 1) // compress_ratio  # [num_ctx_tokens]

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


@maybe_compile(dynamic=True)
def _scale(weights: torch.Tensor, q_scale: torch.Tensor, s: float) -> torch.Tensor:
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


@dataclass
class IndexerParams:
    """
    Parameters for indexer.
    """

    num_contexts: int
    num_generations: int
    num_ctx_tokens: int
    head_dim: int
    quant_block_size: int
    tokens_per_block: int
    compress_ratio: int
    request_ids: List[int]
    num_past_tokens: List[int]
    seq_lens: torch.Tensor
    # Bytes of quantized data per token in the indexer K cache; defaults to
    # head_dim (one FP8 byte per element). For MXFP4 use head_dim // 2.
    data_bytes_per_token: Optional[int] = None

    def __post_init__(self):
        # Pre-compute frequently used tensors once instead of on every property access
        num_past_tokens_tensor = torch.tensor(self.num_past_tokens, dtype=torch.int32)
        self._num_past_tokens_tensor = num_past_tokens_tensor
        compress_ratio = self.compress_ratio
        self._cached_kv_tokens = num_past_tokens_tensor // compress_ratio
        self._all_kv_tokens = (self.seq_lens + num_past_tokens_tensor) // compress_ratio
        self._new_kv_tokens = self._all_kv_tokens - self._cached_kv_tokens
        self._kv_lens = self._all_kv_tokens
        if self.data_bytes_per_token is None:
            self.data_bytes_per_token = self.head_dim
        self._scale_size = self.head_dim // self.quant_block_size * 4
        self._block_stride = self.tokens_per_block * (self.data_bytes_per_token + self._scale_size)

    @property
    def batch_size(self):
        return len(self.request_ids)

    @property
    def kv_lens(self):
        return self._kv_lens

    @property
    def total_tokens(self):
        return self.seq_lens.sum().item()

    @property
    def cached_kv_tokens(self):
        return self._cached_kv_tokens

    @property
    def all_kv_tokens(self):
        return self._all_kv_tokens

    @property
    def new_kv_tokens(self):
        return self._new_kv_tokens

    @property
    def scale_size(self):
        return self._scale_size

    @property
    def block_stride(self):
        return self._block_stride


class Indexer(nn.Module):
    """DSA sparse attention indexer that selects top-K KV cache entries per token."""

    def __init__(
        self,
        quant_config: Optional[QuantConfig],
        pos_embd_params: Optional[PositionalEmbeddingParams],
        mla_params: Optional[MLAParams],
        skip_create_weights_in_init: bool,
        sparse_params: DSAParams,
        dtype: Optional[torch.dtype],
        compress_ratio: int = 1,
        layer_idx: int = 0,
        aux_stream: Optional[torch.cuda.Stream] = None,
    ):
        """Initialize indexer with projection weights, norms, and TopK configuration."""
        super().__init__()
        self.hidden_size = mla_params.hidden_size
        self.q_lora_rank = mla_params.q_lora_rank
        self.rope_dim = mla_params.qk_rope_head_dim
        self.n_heads = sparse_params.index_n_heads  # 64
        self.head_dim = sparse_params.index_head_dim  # 128
        self.index_topk = sparse_params.index_topk  # 2048
        self.layer_idx = layer_idx
        self.compress_ratio = compress_ratio

        self.wq_b = Linear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights_in_init=skip_create_weights_in_init,
            use_custom_cublas_mm=True,
        )
        self.wk = Linear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            dtype=torch.float32,
            quant_config=None,
            skip_create_weights_in_init=skip_create_weights_in_init,
            use_custom_cublas_mm=True,
        )
        self.k_norm = LayerNorm(hidden_size=self.head_dim, eps=1e-6)
        self.weights_proj = Linear(
            self.hidden_size,
            self.n_heads,
            bias=False,
            dtype=torch.float32,
            quant_config=None,
            skip_create_weights_in_init=skip_create_weights_in_init,
            use_custom_cublas_mm=True,
        )

        # Fused wk + weights_proj weight for single F.linear FP32 GEMM under allow_tf32.
        # Maps to TF32 tensor cores on Ampere+.
        self._fused_wk_wp_weight: Optional[torch.Tensor] = None

        indexer_rope_interleave = sparse_params.indexer_rope_interleave
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
        self.use_fp4 = sparse_params.indexer_k_dtype == "fp4"
        self.aux_stream = aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]
        self.use_cute_dsl_topk = sparse_params.use_cute_dsl_topk and IS_CUTLASS_DSL_AVAILABLE
        self.use_cute_dsl_paged_mqa_logits = (
            sparse_params.use_cute_dsl_paged_mqa_logits and IS_CUTLASS_DSL_AVAILABLE
        )
        self.weight_scale_factor = self.softmax_scale * self.n_heads**-0.5

        self._enable_heuristic_topk = (
            sparse_params.enable_heuristic_topk and get_sm_version() >= 100
        )

        if (self.use_cute_dsl_topk or self.use_cute_dsl_paged_mqa_logits) and layer_idx == 0:
            from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops

            if self.use_cute_dsl_topk and not self._enable_heuristic_topk:
                # the dtype of topk input tensor, which is float32 now.
                # Note, need to update it if the dtype of topk input tensor is changed.
                cute_dsl_custom_ops.warmup_cute_dsl_indexer_topk(
                    dtype=torch.float32, top_k=self.index_topk
                )

        if self._enable_heuristic_topk and layer_idx == 0:
            # Populate static caches (sm_count, L2 cache size) inside the C++
            # Scheme X dispatcher before any CUDA Graph capture so the host
            # attribute queries do not end up frozen into a captured graph.
            warmup_heuristic_topk_decode(top_k=self.index_topk)

        # Fused wk + weights_proj weight for single FP32 cuBLAS GEMM
        # (populated in cache_derived_state; maps to TF32 tensor cores on Ampere+)
        self._fused_wk_wp_weight: Optional[torch.Tensor] = None

    def cache_derived_state(self) -> None:
        """Fuse wk and weights_proj for F.linear with TF32 tensor cores on Ampere+."""
        # wk: [head_dim, hidden_size] + weights_proj: [n_heads, hidden_size]
        # → fused: [head_dim + n_heads, hidden_size]
        self._fused_wk_wp_weight = torch.cat(
            [self.wk.weight.data, self.weights_proj.weight.data], dim=0
        )

    def post_load_weights(self) -> None:
        self.cache_derived_state()

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
        compress_ratio = _effective_compress_ratio_divisor(
            _select_indexer_compress_ratio(metadata.compress_ratios)
        )
        if len(chunk_specs) == 1:
            # Single request or intra-request Q-block
            req_idx, token_start_in_req, token_end_in_req, req_cum_start = chunk_specs[0]
            num_q_tokens = token_end_in_req - token_start_in_req

            # Get cached token count for this request from metadata
            num_cached = (
                metadata.host_ctx_cached_token_indptr[req_idx + 1]
                - metadata.host_ctx_cached_token_indptr[req_idx]
            ).item()

            # Total compressed KV tokens for this request
            req_kv_len = (num_cached + token_end_in_req) // compress_ratio

            # For intra-request chunks: Q block attends to all previous K in the request
            # Q tokens [token_start_in_req:token_end_in_req] within the request's current tokens
            # K tokens [0:req_kv_len] in compressed KV space
            cu_seqlen_ks = torch.zeros(num_q_tokens, dtype=torch.int32, device="cpu")
            cu_seqlen_ke = (
                torch.arange(
                    token_start_in_req + 1, token_end_in_req + 1, dtype=torch.int32, device="cpu"
                )
                + num_cached
            ) // compress_ratio

            # Q token range in batch (indices into context tokens in the current batch)
            token_start = req_cum_start + token_start_in_req
            token_end = req_cum_start + token_end_in_req

            # K token range: index into full KV slot mapping in compressed KV space
            # For req_idx=0 the offset is 0; for other indices compute compressed cumulative offset
            kv_offset_in_extended = sum(
                (metadata.host_ctx_kv_indptr[j + 1] - metadata.host_ctx_kv_indptr[j]).item()
                // compress_ratio
                for j in range(req_idx)
            )
            total_kv_for_req = req_kv_len
            k_token_start = kv_offset_in_extended
            k_token_end = kv_offset_in_extended + total_kv_for_req

        else:
            # Multi-request chunk: batch multiple full requests together
            # Extract sequence lengths for these requests
            req_seq_lens = []
            req_num_past_tokens = []
            req_kv_lens = []
            first_req_idx = chunk_specs[0][0]

            for spec in chunk_specs:
                req_idx, token_start_in_req, token_end_in_req, _ = spec
                req_seq_lens.append(token_end_in_req - token_start_in_req)
                # Get cached token count from metadata
                num_past_tokens = (
                    metadata.host_ctx_cached_token_indptr[req_idx + 1]
                    - metadata.host_ctx_cached_token_indptr[req_idx]
                ).item()
                req_num_past_tokens.append(num_past_tokens)
                req_kv_lens.append((num_past_tokens + req_seq_lens[-1]) // compress_ratio)

            req_seq_lens_tensor = torch.tensor(req_seq_lens, dtype=torch.int32, device="cpu")
            req_num_past_tokens_tensor = torch.tensor(
                req_num_past_tokens, dtype=torch.int32, device="cpu"
            )
            req_kv_lens_tensor = torch.tensor(req_kv_lens, dtype=torch.int32, device="cpu")
            num_q_tokens = sum(req_seq_lens)

            # Compute causal attention bounds for batched requests
            cu_seqlen_ks, cu_seqlen_ke = compute_cu_seqlen_kv_bounds_with_cache(
                req_seq_lens_tensor,
                len(chunk_specs),
                num_q_tokens,
                req_num_past_tokens_tensor,
                req_kv_lens_tensor,
                compress_ratio,
            )

            # Global Q token ranges (indices into ctx tokens in the current batch)
            token_start = chunk_specs[0][3]  # req_cum_start of first request
            token_end = token_start + num_q_tokens

            # K token range: index into full kv slot mapping (cached + current ctx tokens within the batch)
            # Must use compressed offsets
            kv_offset_in_extended = sum(
                (metadata.host_ctx_kv_indptr[j + 1] - metadata.host_ctx_kv_indptr[j]).item()
                // compress_ratio
                for j in range(first_req_idx)
            )
            total_kv_len = sum(req_kv_lens)
            k_token_start = kv_offset_in_extended
            k_token_end = kv_offset_in_extended + total_kv_len

        assert cu_seqlen_ks.shape[0] == num_q_tokens == token_end - token_start, (
            "Indexer.prepare_one_prefill_chunk - cu_seqlen_ks length mismatch: "
            f"{cu_seqlen_ks.shape[0]} != {num_q_tokens}"
        )
        assert cu_seqlen_ke.shape[0] == num_q_tokens == token_end - token_start, (
            "Indexer.prepare_one_prefill_chunk - cu_seqlen_ke length mismatch: "
            f"{cu_seqlen_ke.shape[0]} != {num_q_tokens}"
        )

        return IndexerPrefillChunkMetadata(
            cu_seqlen_ks=maybe_pin_memory(cu_seqlen_ks).to(device, non_blocking=True),
            cu_seqlen_ke=maybe_pin_memory(cu_seqlen_ke).to(device, non_blocking=True),
            token_start=token_start,
            token_end=token_end,
            k_token_start=k_token_start,
            k_token_end=k_token_end,
        )

    @staticmethod
    def build_indexer_params(metadata: DSAtrtllmAttentionMetadata) -> Optional[IndexerParams]:
        kv_cache_manager = metadata.kv_cache_manager
        if kv_cache_manager is None or not hasattr(kv_cache_manager, "index_head_dim"):
            return None

        head_dim = metadata.indexer_head_dim
        data_bytes_per_token = (
            head_dim // 2 if getattr(kv_cache_manager, "use_fp4", False) else head_dim
        )
        compress_ratio = _effective_compress_ratio_divisor(
            _select_indexer_compress_ratio(metadata.compress_ratios)
        )
        return IndexerParams(
            num_contexts=metadata.num_contexts,
            num_generations=metadata.num_generations,
            num_ctx_tokens=metadata.num_ctx_tokens,
            head_dim=head_dim,
            quant_block_size=metadata.indexer_quant_block_size,
            tokens_per_block=metadata._tokens_per_block,
            compress_ratio=compress_ratio,
            request_ids=metadata.request_ids,
            num_past_tokens=metadata.kv_cache_params.num_cached_tokens_per_seq,
            seq_lens=metadata.seq_lens,
            data_bytes_per_token=data_bytes_per_token,
        )

    @staticmethod
    def recompute_slot_mappings(
        metadata: DSAtrtllmAttentionMetadata, indexer_params: Optional[IndexerParams] = None
    ):
        """Recompute slot_mapping_fp8/scale from current block offsets.

        This is the subset of prepare_for_update_k_cache() that maps each new
        compressed KV token to its flat cache position. It is safe to call in
        isolation after a caller has swapped the active KV cache manager and
        block-offset buffers, such as during draft KV-cache replay.
        """
        if indexer_params is None:
            indexer_params = Indexer.build_indexer_params(metadata)
            if indexer_params is None:
                return

        batch_size = indexer_params.batch_size
        tokens_per_block = indexer_params.tokens_per_block
        head_dim = indexer_params.head_dim
        new_kv_tokens = indexer_params.new_kv_tokens
        total_new_kv_tokens = new_kv_tokens.sum().item()
        data_bytes_per_token = indexer_params.data_bytes_per_token

        # Compute global positions for all kv tokens in the batch (fully vectorized)
        req_indices = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.int64, device="cpu"), new_kv_tokens
        )
        # Vectorized token_offsets: arange(total) - cumulative start per request
        cu_new_kv = torch.zeros(batch_size + 1, dtype=torch.int64, device="cpu")
        cu_new_kv[1:] = new_kv_tokens.to(torch.int64).cumsum(0)
        token_offsets = torch.arange(
            total_new_kv_tokens, dtype=torch.int64, device="cpu"
        ) - cu_new_kv[:-1].repeat_interleave(new_kv_tokens)
        global_positions = indexer_params.cached_kv_tokens[req_indices] + token_offsets

        # Block indices/pos for all kv tokens in the batch
        block_indices_in_seq = global_positions // tokens_per_block

        max_blocks = metadata.host_indexer_k_cache_block_offsets.shape[1]
        assert (block_indices_in_seq < max_blocks).all(), (
            f"Block index out of bounds: max={max_blocks}, got indices up to {block_indices_in_seq.max().item()}"
        )

        fp8_flat_indices, scale_flat_indices = _compute_slot_mappings(
            global_positions,
            metadata.host_indexer_k_cache_block_offsets,
            req_indices,
            head_dim,
            tokens_per_block,
            indexer_params.quant_block_size,
            data_bytes_per_token=data_bytes_per_token,
        )

        metadata.host_slot_mapping_fp8[:total_new_kv_tokens] = fp8_flat_indices
        metadata.host_slot_mapping_scale[:total_new_kv_tokens] = scale_flat_indices

        metadata.slot_mapping_fp8[:total_new_kv_tokens].copy_(
            metadata.host_slot_mapping_fp8[:total_new_kv_tokens], non_blocking=True
        )
        metadata.slot_mapping_scale[:total_new_kv_tokens].copy_(
            metadata.host_slot_mapping_scale[:total_new_kv_tokens], non_blocking=True
        )

    @staticmethod
    def prepare_for_update_k_cache(
        metadata: DSAtrtllmAttentionMetadata, indexer_params: IndexerParams
    ):
        """
        Prepare indexer for the update_k_cache stage.

        Compute slot_mapping for all requests (both context and generation)
        This maps each token to its flat cache position for vectorized KV cache updates
        """
        Indexer.recompute_slot_mappings(metadata, indexer_params)

    @staticmethod
    def prepare_for_chunked_prefill(
        metadata: DSAtrtllmAttentionMetadata, indexer_params: IndexerParams
    ):
        """
        Prepare indexer for the chunked prefill.
        """
        num_contexts = indexer_params.num_contexts
        seq_lens = indexer_params.seq_lens
        tokens_per_block = indexer_params.tokens_per_block
        head_dim = indexer_params.head_dim

        # When MLA chunked prefill is active, it already handles chunking
        # Indexer should just process the current MLA chunk as a single chunk
        has_mla_chunked_prefill = (
            metadata.enable_context_mla_with_cached_kv and metadata.runtime_features.chunked_prefill
        )
        if has_mla_chunked_prefill:
            chunk_specs = [
                (i, 0, seq_lens[i].item(), seq_lens[:i].sum().item() if i > 0 else 0)
                for i in range(num_contexts)
            ]
            metadata.indexer_prefill_chunks = [
                Indexer.prepare_one_prefill_chunk(
                    metadata,
                    chunk_specs,
                )
            ]
        else:
            # Use indexer's chunking to avoid quadratic indexer MQA logits
            # computation for long sequences.
            # This is only used when MLA chunked prefill is not enabled.
            # Adapt chunk size to the batch's largest K_compressed (see
            # select_indexer_chunk_size).
            max_k_compressed = (
                int(indexer_params.kv_lens[:num_contexts].max().item()) if num_contexts > 0 else 0
            )
            effective_chunk_size = select_indexer_chunk_size(
                metadata.indexer_max_chunk_size, max_k_compressed
            )
            chunk_groups = split_prefill_chunks(
                seq_lens[:num_contexts],
                effective_chunk_size,
                start_idx=0,
            )

            if len(chunk_groups) > 1 or metadata.enable_context_mla_with_cached_kv:
                metadata.indexer_prefill_chunks = [
                    Indexer.prepare_one_prefill_chunk(
                        metadata,
                        chunk_specs,
                    )
                    for chunk_specs in chunk_groups
                ]
            else:
                metadata.indexer_prefill_chunks = None

        # Chunked prefill and KV-cache reuse require the full KV for indexer
        # logits. The indexer's own chunking gathers only the current chunk.
        if metadata.enable_context_mla_with_cached_kv:
            # Use kv_lens which correctly computes (raw_past + seq_lens) // compress_ratio.
            total_kv_per_request = indexer_params.kv_lens[:num_contexts]
            total_kv_len = total_kv_per_request.sum().item()
            host_slot_mapping_fp8_fullkv = torch.empty(
                total_kv_len, dtype=torch.int64, pin_memory=prefer_pinned()
            )
            host_slot_mapping_scale_fullkv = torch.empty(
                total_kv_len, dtype=torch.int64, pin_memory=prefer_pinned()
            )

            req_indices = torch.repeat_interleave(
                torch.arange(num_contexts, dtype=torch.int64, device="cpu"), total_kv_per_request
            )

            cu_kv = torch.zeros(num_contexts + 1, dtype=torch.int64, device="cpu")
            cu_kv[1:] = total_kv_per_request.to(torch.int64).cumsum(0)
            kv_positions = torch.arange(total_kv_len, dtype=torch.int64, device="cpu") - cu_kv[
                :-1
            ].repeat_interleave(total_kv_per_request)

            fp8_flat_indices, scale_flat_indices = _compute_slot_mappings(
                kv_positions,
                metadata.host_indexer_k_cache_block_offsets,
                req_indices,
                head_dim,
                tokens_per_block,
                indexer_params.quant_block_size,
                data_bytes_per_token=head_dim // 2
                if metadata.kv_cache_manager.use_fp4
                else head_dim,
            )

            host_slot_mapping_fp8_fullkv[:total_kv_len] = fp8_flat_indices
            host_slot_mapping_scale_fullkv[:total_kv_len] = scale_flat_indices

            assert len(fp8_flat_indices) == total_kv_len, (
                "host_slot_mapping_fp8_fullkv/host_slot_mapping_scale_fullkv "
                f"length mismatch: {len(fp8_flat_indices)} != total_kv_len={total_kv_len}"
            )

            # Store extended mappings for indexer full KV gathering
            metadata.slot_mapping_fp8_fullkv = host_slot_mapping_fp8_fullkv.cuda(non_blocking=True)
            metadata.slot_mapping_scale_fullkv = host_slot_mapping_scale_fullkv.cuda(
                non_blocking=True
            )
        else:
            metadata.slot_mapping_fp8_fullkv = metadata.slot_mapping_fp8
            metadata.slot_mapping_scale_fullkv = metadata.slot_mapping_scale

    @staticmethod
    def prepare_scheduler_metadata(metadata: DSAtrtllmAttentionMetadata):
        """
        Prepare scheduler metadata for the DeepGEMM decode MQA kernel.
        """
        num_contexts = metadata.num_contexts
        num_generations = metadata.num_generations
        if not metadata.use_expanded_buffers_for_mtp:
            gen_seq_lens = metadata.get_indexer_kv_lens(
                metadata.kv_lens_cuda_runtime[num_contexts : num_contexts + num_generations]
            )
            metadata.gen_indexer_kv_lens_cuda_runtime = gen_seq_lens
            next_n_cap = metadata.kv_lens_cuda_2d.shape[1]
            metadata.kv_lens_cuda_2d[:num_generations, :next_n_cap].copy_(
                gen_seq_lens.unsqueeze(-1).expand(-1, next_n_cap)
            )
            scheduler_metadata_buffer = get_paged_mqa_logits_metadata(
                gen_seq_lens.view(-1, 1), _DG_SCHEDULE_BLOCK_KV, metadata.num_sms
            )
            metadata.scheduler_metadata_buffer.copy_(scheduler_metadata_buffer, non_blocking=True)
            if metadata.max_draft_tokens > 0:
                scheduler_metadata_buffer_full_next_n = get_paged_mqa_logits_metadata(
                    metadata.kv_lens_cuda_2d[:num_generations, :next_n_cap],
                    _DG_SCHEDULE_BLOCK_KV,
                    metadata.num_sms,
                )
                metadata.scheduler_metadata_buffer_full_next_n.copy_(
                    scheduler_metadata_buffer_full_next_n, non_blocking=True
                )
        else:
            # Expand schedule metadata buffer (only generation). The DeepGEMM
            # API requires 2D; each expanded token becomes a (1,) row.
            num_tokens = metadata.num_generations * (1 + metadata.max_draft_tokens)
            scheduler_metadata_buffer_expanded = get_paged_mqa_logits_metadata(
                metadata.kv_lens_expanded_cuda[:num_tokens].view(-1, 1),
                _DG_SCHEDULE_BLOCK_KV,
                metadata.num_sms,
            )
            metadata.scheduler_metadata_buffer_expanded.copy_(
                scheduler_metadata_buffer_expanded, non_blocking=True
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

        # Skip indexer preparation if the kv_cache_manager doesn't have index_head_dim.
        # This can happen when the metadata is being used with a draft KV cache manager
        # during MTP speculative decoding, which uses a regular KVCacheManager instead
        # of DSACacheManager.
        indexer_params = Indexer.build_indexer_params(metadata)
        if indexer_params is None:
            return

        num_contexts = metadata.num_contexts
        num_generations = metadata.num_generations
        num_ctx_tokens = metadata.num_ctx_tokens
        seq_lens = metadata.seq_lens
        compress_ratio = _effective_compress_ratio_divisor(
            _select_indexer_compress_ratio(metadata.compress_ratios)
        )
        # Store compressed KV token count for context requests
        metadata.num_ctx_kv_tokens = indexer_params.new_kv_tokens[:num_contexts].sum().item()

        # Prepare for update_k_cache
        Indexer.prepare_for_update_k_cache(metadata, indexer_params)

        # Prepare for prefill phase if there are context requests
        if num_contexts > 0:
            # Compute attention window bounds for each query token in batched sequences
            # cu_seqlen_ks[i]: start index in global KV for query token i
            # cu_seqlen_ke[i]: end index (exclusive) in global KV for query token i
            host_seq_lens = seq_lens[:num_contexts]
            host_num_past_tokens = indexer_params._num_past_tokens_tensor[:num_contexts]
            host_kv_lens = indexer_params.kv_lens[:num_contexts]
            host_cu_seqlen_ks, host_cu_seqlen_ke = compute_cu_seqlen_kv_bounds_with_cache(
                host_seq_lens,
                num_contexts,
                num_ctx_tokens,
                host_num_past_tokens,
                host_kv_lens,
                compress_ratio,
            )

            metadata.cu_seqlen_ks[:num_ctx_tokens].copy_(
                maybe_pin_memory(host_cu_seqlen_ks), non_blocking=True
            )
            metadata.cu_seqlen_ke[:num_ctx_tokens].copy_(
                maybe_pin_memory(host_cu_seqlen_ke), non_blocking=True
            )
            Indexer.prepare_for_chunked_prefill(metadata, indexer_params)

        # Prepare for decode phase if there are generation requests
        if num_generations > 0:
            # Prepare schedule metadata for fp8_paged_mqa_logits
            # This is a preprocessing step that computes scheduling information for the kernel
            Indexer.prepare_scheduler_metadata(metadata)

    def _update_k_cache(
        self, k_fp8: torch.Tensor, k_scale: torch.Tensor, metadata: DSAtrtllmAttentionMetadata
    ) -> None:
        """
        Insert/append k values and scales into the indexer k cache using pre-computed slot mappings.
        Uses flat byte indices with vectorized scatter.

        Args:
            k_fp8: FP8 quantized k tensor, shape [total_tokens, head_dim]
            k_scale: Scaling factors, shape [total_tokens, head_dim // quant_block_size]
        """
        if metadata.kv_cache_manager is None or metadata.slot_mapping_fp8 is None:
            return

        k_cache = metadata.kv_cache_manager.get_indexer_k_cache_buffers(self.layer_idx)

        num_tokens = k_fp8.shape[0]

        # The C++ op reinterprets k_fp8 (FP8) and k_scale (float32) as raw
        # bytes internally and only reads the first num_tokens entries from
        # the slot mapping buffers, avoiding Python-side view/slice overhead.
        if k_scale.element_size() == 1:
            # The op expects 4 byte elements.
            k_scale = k_scale.view(torch.int32)
        torch.ops.trtllm.indexer_k_cache_scatter_op(
            k_fp8,
            k_scale,
            k_cache,
            metadata.slot_mapping_fp8,
            metadata.slot_mapping_scale,
            num_tokens,
        )

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
        assert metadata.slot_mapping_fp8_fullkv is not None, (
            "_gather_k_cache_for_chunk requires extended slot mappings (only available with cached tokens)"
        )

        k_cache = metadata.kv_cache_manager.get_indexer_k_cache_buffers(self.layer_idx)

        head_dim = self.head_dim
        scale_size = 4  # float32 = 4 bytes

        # Extract slot mappings using chunk's k_token_start/end
        # These indices point directly into the extended slot mapping array
        k_token_start = chunk.k_token_start
        k_token_end = chunk.k_token_end
        num_k_tokens = k_token_end - k_token_start

        slot_mapping_fp8_chunk = metadata.slot_mapping_fp8_fullkv[k_token_start:k_token_end]
        slot_mapping_scale_chunk = metadata.slot_mapping_scale_fullkv[k_token_start:k_token_end]

        # Vectorized gather using pre-computed slot mappings
        # Gather FP8 data
        byte_offsets_fp8 = torch.arange(head_dim, device=k_cache.device).unsqueeze(
            0
        )  # [1, head_dim]
        gather_indices_fp8 = (
            slot_mapping_fp8_chunk.unsqueeze(1) + byte_offsets_fp8
        )  # [num_k_tokens, head_dim]
        assert (gather_indices_fp8 >= k_cache.numel()).sum() == 0, "Out-of-bounds access detected"
        gather_indices_fp8 = _unravel_indices(gather_indices_fp8, k_cache.shape)
        k_fp8_bytes = k_cache[gather_indices_fp8]
        k_fp8 = k_fp8_bytes.view(torch.float8_e4m3fn).view(num_k_tokens, head_dim)

        # Gather scale data
        byte_offsets_scale = torch.arange(scale_size, device=k_cache.device).unsqueeze(0)  # [1, 4]
        gather_indices_scale = (
            slot_mapping_scale_chunk.unsqueeze(1) + byte_offsets_scale
        )  # [num_k_tokens, 4]
        assert (gather_indices_scale >= k_cache.numel()).sum() == 0, "Out-of-bounds access detected"
        gather_indices_scale = _unravel_indices(gather_indices_scale, k_cache.shape)
        k_scale_bytes = k_cache[gather_indices_scale]
        k_scale = k_scale_bytes.view(torch.float32).view(num_k_tokens, 1)

        return k_fp8, k_scale

    def _call_mqa_logits(
        self,
        q_fp8: torch.Tensor,
        k_fp8: torch.Tensor,
        k_scale: torch.Tensor,
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        q_scale: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Dispatch fp8_mqa_logits vs fp8_fp4_mqa_logits based on use_fp4.

        For FP4 the gather output keeps the legacy float8_e4m3fn dtype for
        API compatibility; reinterpret the bytes as the int8 / int32 layout
        the DeepGEMM kernel expects. The scale tensor is collapsed to 1D for
        the kv side and 2D for the q side per the kernel's asserts.
        """
        if self.use_fp4:
            k_fp4_bytes = k_fp8.view(torch.int8)
            k_scale_int32 = k_scale.view(torch.int32).reshape(-1)
            # q_scale arrives as (chunk_tokens, n_heads, 1); the FP4 kernel
            # asserts q_sf is 2D so collapse the trailing unit axis.
            q_scale_2d = q_scale.reshape(-1, self.n_heads)
            return fp8_fp4_mqa_logits(
                (q_fp8, q_scale_2d),
                (k_fp4_bytes, k_scale_int32),
                weights,
                cu_seqlen_ks,
                cu_seqlen_ke,
            )
        return fp8_mqa_logits(
            q_fp8, (k_fp8, k_scale.reshape(-1)), weights, cu_seqlen_ks, cu_seqlen_ke
        )

    def _call_paged_mqa_logits(
        self,
        q_decode: torch.Tensor,
        k_cache: torch.Tensor,
        weights_decode: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        scheduler_metadata_buffer: torch.Tensor,
        max_seq_len: int,
        q_scale: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Dispatch fp8_paged_mqa_logits vs fp8_fp4_paged_mqa_logits."""
        if self.use_fp4:
            return fp8_fp4_paged_mqa_logits(
                (q_decode, q_scale),
                k_cache,
                weights_decode,
                context_lens,
                block_table,
                scheduler_metadata_buffer,
                max_seq_len,
            )
        return fp8_paged_mqa_logits(
            q_decode,
            k_cache,
            weights_decode,
            context_lens,
            block_table,
            scheduler_metadata_buffer,
            max_seq_len,
        )

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
        is_generation: Optional[bool] = None,
    ) -> torch.Tensor:
        """Run the indexer TopK kernel for one phase or the full batch.

        When ``is_generation`` is ``None``, the inputs contain the full mixed
        batch. Otherwise they contain only the selected context or generation
        phase. ``q_scale`` is only consumed by the FP4 dispatch.
        """
        # DSACacheManager / DeepseekV4CacheManager force quant_block_size to
        # 128 (FP8 path) or 32 (MXFP4 path); both round-trip to the same
        # 4-byte scale word per token at index_head_dim=128, so the slot
        # mapping arithmetic is unchanged in either mode.
        assert metadata.kv_cache_manager is None or metadata.kv_cache_manager.quant_block_size in (
            32,
            128,
        ), (
            "Unexpected quant_block_size "
            f"{metadata.kv_cache_manager.quant_block_size if metadata.kv_cache_manager else 'N/A'}"
        )
        num_contexts = metadata.num_contexts
        num_generations = metadata.num_generations
        num_ctx_tokens = metadata.num_ctx_tokens
        num_tokens = metadata.num_tokens

        num_gen_tokens = num_tokens - num_ctx_tokens
        if is_generation is None:
            has_prefill = num_contexts > 0
            has_decode = num_generations > 0
            token_offset = num_ctx_tokens
            cache_name = "indexer_topk_out_buffer"
        else:
            has_prefill = not is_generation and num_contexts > 0
            has_decode = is_generation and num_generations > 0
            token_offset = 0
            expected_tokens = num_gen_tokens if is_generation else num_ctx_tokens
            assert hidden_states.shape[0] == expected_tokens, (
                "Phase-specific DSA prediction received "
                f"{hidden_states.shape[0]} tokens, expected {expected_tokens}."
            )
            cache_name = (
                "indexer_topk_out_buffer_gen" if is_generation else "indexer_topk_out_buffer_ctx"
            )

        topk_indices_buffer = metadata.get_empty(
            metadata.cuda_graph_buffers,
            (hidden_states.shape[0], self.index_topk),
            cache_name=cache_name,
            dtype=torch.int32,
            capture_graph=metadata.is_cuda_graph,
        )
        if not use_custom_topk:
            topk_indices_buffer[: hidden_states.shape[0]] = -1

        if has_prefill and not metadata.skip_indexer_for_ctx_reqs:
            # Use chunked prefill to reduce memory footprint
            if metadata.indexer_prefill_chunks is not None:
                sparse_metadata_params = metadata.sparse_metadata_params
                q_split_threshold = (
                    sparse_metadata_params.q_split_threshold
                    if sparse_metadata_params is not None
                    else 8192
                )
                q_split_eligible = (
                    q_split_threshold >= 0
                    and metadata.mapping is not None
                    and not metadata.mapping.enable_attention_dp
                    and metadata.mapping.tp_size > 1
                )

                if q_split_eligible:
                    tp_rank = metadata.mapping.tp_rank
                    tp_size = metadata.mapping.tp_size

                k_cache_4d = metadata.kv_cache_manager.get_indexer_k_cache_buffers(self.layer_idx)
                # FP4 packs two codes per byte so the gathered row holds half
                # as many bytes as in the FP8 path. The scale (4 bytes) is the
                # same in both modes because FP4 packs four UE8M0 exponents
                # into one int32 to match FP8's float32 scale width.
                gather_head_dim = self.head_dim // 2 if self.use_fp4 else self.head_dim

                for chunk in metadata.indexer_prefill_chunks:
                    # Skip chunks with no compressed KV tokens (e.g., warmup
                    # sequences shorter than compress_ratio produce zero KV).
                    if chunk.k_token_start >= chunk.k_token_end:
                        topk_indices_buffer[chunk.token_start : chunk.token_end, :].fill_(-1)
                        continue
                    num_k_tokens = chunk.k_token_end - chunk.k_token_start
                    chunk_k_fp8, chunk_k_scale = torch.ops.trtllm.indexer_k_cache_gather_op(
                        k_cache_4d,
                        metadata.slot_mapping_fp8_fullkv,
                        metadata.slot_mapping_scale_fullkv,
                        chunk.k_token_start,
                        num_k_tokens,
                        gather_head_dim,
                    )

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

                    # Tile the query dimension so each fp8_mqa_logits call
                    # allocates at most [q_tile x num_k_tokens] instead of the
                    # full [local_q x num_k_tokens] (which can reach tens of GB
                    # on a long context and stall cuMemCreate under
                    # expandable_segments -> engine hang; see
                    # _INDEXER_MQA_LOGITS_ELEM_BUDGET). Results are identical:
                    # each query row's logits/top-k are independent and the KV
                    # (chunk_k_fp8) is unchanged across tiles, so the per-call
                    # allocation is the same size and the caching allocator
                    # reuses one block (peak ~= one tile, no extra sync).
                    local_q_len = chunk_q_end - chunk_q_start
                    q_tile = max(
                        1, min(local_q_len, _INDEXER_MQA_LOGITS_ELEM_BUDGET // max(1, num_k_tokens))
                    )
                    for tile_off in range(0, local_q_len, q_tile):
                        c0 = chunk_q_start + tile_off
                        c1 = min(c0 + q_tile, chunk_q_end)
                        g0 = chunk.token_start + c0
                        g1 = chunk.token_start + c1
                        tile_q_scale = q_scale[g0:g1, ...] if self.use_fp4 else None
                        logits = self._call_mqa_logits(
                            q_fp8[g0:g1, ...],
                            chunk_k_fp8,
                            chunk_k_scale,
                            weights[g0:g1, ...],
                            chunk.cu_seqlen_ks[c0:c1],
                            chunk.cu_seqlen_ke[c0:c1],
                            tile_q_scale,
                        )
                        if use_custom_topk:
                            torch.ops.trtllm.indexer_topk_prefill(
                                logits,
                                chunk.cu_seqlen_ks[c0:c1],
                                chunk.cu_seqlen_ke[c0:c1],
                                topk_indices_buffer[g0:g1, :],
                                self.index_topk,
                            )
                        else:
                            topk_indices = logits.topk(
                                min(self.index_topk, logits.shape[-1]), dim=-1
                            )[1]
                            topk_indices -= chunk.cu_seqlen_ks[c0:c1][:, None]

                            mask_lo = topk_indices >= 0
                            mask_hi = (
                                topk_indices
                                - (chunk.cu_seqlen_ke[c0:c1] - chunk.cu_seqlen_ks[c0:c1])[:, None]
                                < 0
                            )
                            mask = mask_lo & mask_hi

                            # local indices per sequence
                            topk_indices = topk_indices.masked_fill(~mask, -1)

                            topk_indices_buffer[g0:g1, : topk_indices.shape[-1]] = topk_indices.to(
                                dtype=torch.int32
                            )

                    if apply_q_split:
                        q_sizes = [
                            (r + 1) * chunk_num_token // tp_size - r * chunk_num_token // tp_size
                            for r in range(tp_size)
                        ]
                        topk_indices_buffer[chunk.token_start : chunk.token_end, :] = allgather(
                            topk_indices_buffer[global_q_start:global_q_end, :],
                            metadata.mapping,
                            dim=0,
                            sizes=q_sizes,
                        )
            elif metadata.num_ctx_kv_tokens == 0:
                # No compressed KV tokens — fill with -1 (no valid indices)
                topk_indices_buffer[:num_ctx_tokens, :].fill_(-1)
            else:
                # Fallback: single-pass indexer prefill (TODO: remove this once chunked prefill is fully tested)
                num_ctx_kv_tokens = metadata.num_ctx_kv_tokens
                cu_seqlen_ks = metadata.cu_seqlen_ks[:num_ctx_tokens]
                cu_seqlen_ke = metadata.cu_seqlen_ke[:num_ctx_tokens]

                ctx_q_scale = q_scale[:num_ctx_tokens, ...] if self.use_fp4 else None
                logits = self._call_mqa_logits(
                    q_fp8[:num_ctx_tokens, ...],
                    k_fp8[:num_ctx_kv_tokens, ...],
                    k_scale[:num_ctx_kv_tokens, ...],
                    weights[:num_ctx_tokens, ...],
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                    ctx_q_scale,
                )
                if use_custom_topk:
                    torch.ops.trtllm.indexer_topk_prefill(
                        logits,
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        topk_indices_buffer[:num_ctx_tokens, :],
                        self.index_topk,
                    )
                else:
                    topk_indices = logits.topk(min(self.index_topk, logits.shape[-1]), dim=-1)[1]
                    topk_indices -= cu_seqlen_ks[:, None]
                    mask_lo = topk_indices >= 0
                    mask_hi = topk_indices - (cu_seqlen_ke - cu_seqlen_ks)[:, None] < 0
                    mask = mask_lo & mask_hi

                    # local indices per sequence
                    topk_indices = topk_indices.masked_fill(~mask, -1)
                    topk_indices_buffer[:num_ctx_tokens, : topk_indices.shape[-1]] = (
                        topk_indices.to(dtype=torch.int32)
                    )
        elif has_prefill and metadata.skip_indexer_for_ctx_reqs:
            # Fill topk_indices_buffer with pre-defined dense topk indices
            topk_indices_buffer[:num_ctx_tokens, :] = metadata.topk_indices_buffer[
                :num_ctx_tokens, :
            ]

        # Prefill→decode GVR handoff: seed each finishing-prefill sequence's
        # heuristic_prev_topk slot with its own last-context-token top-K, so
        # the FIRST decode step of that sequence gets a warm-started preIdx
        # (~60-75% set-overlap with the eventual decode top-K on this
        # workload) instead of the all-zero / all-(-1) cold start that the
        # default `heuristic_prev_topk.zero_()` initialization leaves behind.
        # Without this, GVR P2 secant on decode step 0 runs from a benign
        # but uninformative seed (kernel +1 offset on zeros → all indices
        # point at compressed-token position 1), wasting iterations.
        # Slot convention (mirrors the existing decode write-back at the
        # bottom of the decode block): new gens from finishing prefill
        # append after currently-active gens, i.e., slots
        # [num_generations : num_generations + num_contexts].
        if self._enable_heuristic_topk and has_prefill and not metadata.skip_indexer_for_ctx_reqs:
            local_layer = metadata.kv_cache_manager.layer_offsets[self.layer_idx]
            ctx_seq_lens = metadata.seq_lens[:num_contexts]
            # Per-sequence last context-token offset (exclusive cumsum minus 1).
            last_ctx_idx = (torch.cumsum(ctx_seq_lens, dim=0) - 1).to(dtype=torch.long)
            metadata.heuristic_prev_topk[
                local_layer, num_generations : num_generations + num_contexts
            ].copy_(topk_indices_buffer[last_ctx_idx, :])

        if has_decode and not metadata.skip_indexer_for_gen_reqs:
            # Get decode lengths per request (from seq_lens) for validation
            gen_seq_lens = metadata.seq_lens[num_contexts : num_contexts + num_generations]
            max_decode_len = gen_seq_lens.max().item()
            min_decode_len = gen_seq_lens.min().item()
            assert max_decode_len == min_decode_len, (
                "max_decode_len != min_decode_len, we need padding"
            )

            # Reshape q for decode phase: [num_gen_tokens, ...] -> [batch_size, next_n, ...]
            q_decode = q_fp8[token_offset : token_offset + num_gen_tokens, ...]
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
                context_lens = metadata.kv_lens_cuda_2d[:num_generations, :next_n].contiguous()
                block_table = metadata.indexer_k_cache_block_offsets[
                    num_contexts : num_contexts + num_generations
                ]
                # The 2D-context_lens metadata kernel encodes next_n into the
                # schedule (via num_next_n_atoms). MTP forwards alternate
                # between the full-window call (next_n == 1+max_draft_tokens)
                # and per-token draft calls (next_n == 1), so we must select
                # the buffer that was populated for this next_n. The DSL path
                # uses its own schedule buffer (built with num_next_n_atoms=1
                # via a (num_gen, 1) input shape) and overrides this below.
                if next_n == 1:
                    scheduler_metadata_buffer = metadata.scheduler_metadata_buffer
                else:
                    scheduler_metadata_buffer = metadata.scheduler_metadata_buffer_full_next_n
            else:
                q_decode = q_decode.view(-1, 1, *q_fp8.shape[1:])
                num_tokens = q_decode.shape[0]
                # New API requires 2D; each expanded token becomes a (1,) row.
                context_lens = metadata.kv_lens_expanded_cuda[:num_tokens].view(-1, 1)
                block_table = metadata.block_table_expanded[:num_tokens]
                scheduler_metadata_buffer = metadata.scheduler_metadata_buffer_expanded

            assert num_gen_tokens == batch_size * next_n
            weights_decode = weights[token_offset : token_offset + num_gen_tokens, ...]

            # Get k cache and call fp8_paged_mqa_logits / fp8_fp4_paged_mqa_logits
            # with prepared decode metadata.
            # [num_blocks, tokens_per_block, 1, head_dim + scale_size]
            k_cache = metadata.kv_cache_manager.get_indexer_k_cache_buffers(self.layer_idx)
            indexer_max_seq_len = metadata.get_indexer_max_seq_len()

            if self.use_cute_dsl_paged_mqa_logits:
                # DSL kernel design: 1 atom per q (atom = real next_n positions),
                # kNumNextNAtoms = 1 for any real next_n. The matching schedule
                # is `scheduler_metadata_buffer` — built in `Indexer.prepare()`
                # with a (num_gen, 1) input shape, which makes DeepGEMM's wrapper
                # compute `num_next_n_atoms = 1`. (DeepGEMM uses the same buffer
                # for its own next_n=1 kernel; DSL piggy-backs on it for all
                # real next_n values.) All next_n positions of a batch share
                # the same KV length on this path (kv_lens_cuda_2d broadcasts),
                # so passing the 1D contiguous kv_lens slice for context_lens
                # avoids materializing a 2D contiguous tensor per call.
                dsl_context_lens = metadata.gen_indexer_kv_lens_cuda_runtime
                assert dsl_context_lens is not None
                # Wave-aware atom-split: the picker in `_pick_dsl_expand` caches
                # (factor, atom) on metadata with invariant
                # `factor * atom == 1 + max_draft_tokens` (the target/verify-time
                # next_n). MTPEagle reuses the same metadata for its multi-step
                # draft loop; after i=0 it mutates seq_lens to 1, so i≥1
                # iterations run with next_n=1. The reshape
                # `(num_gen, next_n, ...) -> (num_gen*factor, atom, ...)` is only
                # valid when the caller actually supplies next_n == factor * atom
                # tokens; gate here so i≥1 draft calls fall back to the
                # kernel-native next_n=1 path.
                dsl_atom_split = (
                    metadata.dsl_expand_factor > 1
                    and next_n == metadata.dsl_expand_factor * metadata.dsl_atom
                )
                if self.use_fp4:
                    # FP4 DSL signature splits DG's (q, sf_q) tuple into two
                    # separate args and requires q.dtype == uint8 (q_decode
                    # came in via the FP8 plumbing as int8; reinterpret with
                    # no copy). sf_q is the q_scale slice reshaped to
                    # (B, next_n, H) int32 -- mirrors the non-DSL FP4 branch.
                    decode_q_scale = q_scale[token_offset : token_offset + num_gen_tokens, ...]
                    decode_q_scale = decode_q_scale.view(
                        q_decode.shape[0], q_decode.shape[1], self.n_heads
                    )
                    dsl_q = q_decode.view(torch.uint8)
                    dsl_block_table = block_table
                    dsl_schedule_meta = metadata.scheduler_metadata_buffer
                    if dsl_atom_split:
                        factor = metadata.dsl_expand_factor
                        eff_next_n = metadata.dsl_atom
                        exp_B = num_generations * factor
                        dsl_q = dsl_q.reshape(exp_B, eff_next_n, self.n_heads, self.head_dim // 2)
                        decode_q_scale = decode_q_scale.reshape(exp_B, eff_next_n, self.n_heads)
                        dsl_context_lens = metadata.kv_lens_expanded_cuda[:exp_B]
                        dsl_block_table = metadata.block_table_expanded[:exp_B]
                        dsl_schedule_meta = metadata.scheduler_metadata_buffer_expanded

                    logits_decode = torch.ops.trtllm.cute_dsl_fp4_paged_mqa_logits(
                        dsl_q,
                        decode_q_scale,
                        k_cache,
                        weights_decode,
                        dsl_context_lens,
                        dsl_block_table,
                        dsl_schedule_meta,
                        indexer_max_seq_len,
                    )
                else:
                    # FP8 DSL kernel natively supports next_n ∈ {1, 2, 3, 4}.
                    # Atom-split benefits small-batch / low-ntask configs by
                    # raising SM utilization at the cost of factorx KV HBM
                    # re-reads. Context lengths are indexer KV lengths, which
                    # may be compressed for DeepSeek-V4.
                    dsl_q = q_decode
                    fp8_ctx_lens = dsl_context_lens
                    fp8_block_table = block_table
                    fp8_schedule_meta = metadata.scheduler_metadata_buffer
                    if dsl_atom_split:
                        factor = metadata.dsl_expand_factor
                        atom = metadata.dsl_atom
                        exp_B = num_generations * factor
                        dsl_q = q_decode.reshape(exp_B, atom, self.n_heads, self.head_dim)
                        fp8_ctx_lens = metadata.kv_lens_expanded_cuda[:exp_B]
                        fp8_block_table = metadata.block_table_expanded[:exp_B]
                        fp8_schedule_meta = metadata.scheduler_metadata_buffer_expanded
                    logits_decode = torch.ops.trtllm.cute_dsl_fp8_paged_mqa_logits(
                        dsl_q,
                        k_cache,
                        weights_decode,
                        fp8_ctx_lens,
                        fp8_block_table,
                        fp8_schedule_meta,
                        indexer_max_seq_len,
                    )
            else:
                decode_q_scale = (
                    q_scale[token_offset : token_offset + num_gen_tokens, ...]
                    if self.use_fp4
                    else None
                )
                if self.use_fp4:
                    # q_decode shape is either (num_generations, next_n, n_heads,
                    # head_dim/2) [non-expanded] or (batch*next_n, 1, n_heads,
                    # head_dim/2) [expanded]. Match q_scale's batch/next_n dims.
                    decode_q_scale = decode_q_scale.view(
                        q_decode.shape[0], q_decode.shape[1], self.n_heads
                    )
                logits_decode = self._call_paged_mqa_logits(
                    q_decode,
                    k_cache,
                    weights_decode,
                    context_lens,
                    block_table,
                    scheduler_metadata_buffer,
                    indexer_max_seq_len,
                    decode_q_scale,
                )

            if use_custom_topk:
                # Kernel expects kv_lens (total cache length), not seq_lens (new tokens)
                # This is because rowEnd = seq_len - next_n + offset + 1
                gen_kv_lens_cuda = metadata.kv_lens_cuda_runtime[
                    num_contexts : num_contexts + num_generations
                ]

                pre_idx = None
                heuristic_scratch = None
                if self._enable_heuristic_topk:
                    local_layer = metadata.kv_cache_manager.layer_offsets[self.layer_idx]
                    # Pass prev_topk directly; the +1 temporal offset is
                    # handled inside the C++ kernel (preIdxOffset += 1).
                    pre_idx = metadata.heuristic_prev_topk[local_layer, :num_generations]
                    if not metadata.use_cute_dsl_topk:
                        heuristic_scratch = metadata.heuristic_scratch_values[:num_gen_tokens]

                if self.use_cute_dsl_topk and self._enable_heuristic_topk:
                    torch.ops.trtllm.cute_dsl_gvr_topk_decode(
                        logits_decode,
                        pre_idx,
                        gen_kv_lens_cuda,
                        topk_indices_buffer[token_offset : token_offset + num_gen_tokens, :],
                        self.index_topk,
                        next_n=next_n,
                        compress_ratio=self.compress_ratio,
                        max_seq_len=indexer_max_seq_len,
                        order_row=metadata.kv_lens_row_reorder,
                    )
                # The radix DSL path uses O(num_gen_tokens * kv_len) memory.
                elif (
                    self.use_cute_dsl_topk
                    and num_gen_tokens <= 256
                    and (self.compress_ratio == 1 or next_n == 1)
                ):
                    torch.ops.trtllm.cute_dsl_indexer_topk_decode(
                        logits_decode,
                        context_lens if self.compress_ratio > 1 else gen_kv_lens_cuda,
                        topk_indices_buffer[token_offset : token_offset + num_gen_tokens, :],
                        self.index_topk,
                        next_n,
                    )
                else:
                    torch.ops.trtllm.indexer_topk_decode(
                        logits_decode,
                        gen_kv_lens_cuda,
                        topk_indices_buffer[token_offset : token_offset + num_gen_tokens, :],
                        next_n,
                        self.index_topk,
                        pre_idx=pre_idx,
                        heuristic_scratch=heuristic_scratch,
                        compress_ratio=self.compress_ratio,
                        radix_aux_indices=metadata.radix_aux_indices,
                        radix_aux_logits=metadata.radix_aux_logits,
                    )
            else:
                # padded
                positions = (
                    torch.arange(logits_decode.shape[-1], device=q_decode.device)
                    .unsqueeze(0)
                    .expand(num_gen_tokens, -1)
                )
                row_indices = torch.arange(num_gen_tokens, device=q_decode.device) // next_n
                next_n_offset = torch.arange(num_gen_tokens, device=q_decode.device) % next_n
                index_end_pos = (context_lens[row_indices] - next_n + next_n_offset).unsqueeze(1)
                # index_end_pos: [B * N, 1]
                mask = positions <= index_end_pos
                # mask: [B * N, L]
                logits_decode = logits_decode.masked_fill(~mask, float("-inf"))
                topk_indices_decode = logits_decode.topk(
                    min(self.index_topk, logits_decode.shape[-1]), dim=-1
                )[1].to(torch.int32)  # [B * N, K]
                # ensure we don't set indices for the top k
                # that is out of range(masked already)
                # this will happen if context length is shorter than K
                mask_decode = topk_indices_decode <= index_end_pos

                # local indices per sequence
                topk_indices_decode = topk_indices_decode.masked_fill(~mask_decode, -1)
                # Store in buffer
                topk_indices_buffer[
                    token_offset : token_offset + num_gen_tokens, : topk_indices_decode.shape[-1]
                ] = topk_indices_decode.to(dtype=torch.int32)

            if self._enable_heuristic_topk:
                local_layer = metadata.kv_cache_manager.layer_offsets[self.layer_idx]
                decode_topk = topk_indices_buffer[token_offset : token_offset + num_gen_tokens]
                last_mtp_topk = decode_topk[next_n - 1 :: next_n]
                metadata.heuristic_prev_topk[local_layer, :num_generations].copy_(last_mtp_topk)

        elif has_decode and metadata.skip_indexer_for_gen_reqs:
            # Fill topk_indices_buffer with pre-defined dense topk indices
            topk_indices_buffer[token_offset : token_offset + num_gen_tokens, :] = (
                metadata.topk_indices_buffer[num_ctx_tokens:num_tokens, :]
            )
        return topk_indices_buffer

    def _weight_scale(self, weights: torch.Tensor, q_scale: torch.Tensor) -> torch.Tensor:
        """Apply quantization scale to indexer attention weights."""
        weights = _scale(weights, q_scale, self.weight_scale_factor)
        return weights

    def _qk_projection_and_rope(
        self, qr: torch.Tensor, indexer_k: torch.Tensor, position_ids: torch.Tensor
    ):
        """Project Q/K and apply RoPE"""
        q = self.wq_b(qr)
        k = self.k_norm(indexer_k)
        q = q.view(-1, self.n_heads, self.head_dim)
        q_pe, q_nope = q.split([self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
        k_pe, k_nope = k.split([self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
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
        fp8_out, scale = torch.ops.trtllm.fused_cat_fp8(qk_pe, qk_nope, self.scale_fmt == "ue8m0")
        return fp8_out, scale

    def pre_indexer_proj(
        self, qr: torch.Tensor, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pure token-wise projections (CUDA-graph-capturable).

        Runs cublas_mm, qk_projection_and_rope, FP8/FP4 quantize, and weight
        scaling.  Does NOT touch the k cache or any batch-specific metadata,
        so this can safely run inside a captured CUDA graph partition.

        Returns (q_fp_bytes, k_fp_bytes, k_scale, weights, q_scale). The last
        tensor is only consumed by the FP4 kernel dispatch; the FP8 path
        ignores it. It is returned unconditionally so the two-op CUDA graph
        split in dsa.module.forward_dsa_proj sees a stable signature.
        """
        assert self._fused_wk_wp_weight is not None, (
            "cache_derived_state() must be called before forward()"
        )
        # When the boundary fusion pre-quantized the next layer's kv_a_proj
        # NVFP4 input, hidden_states is an Fp4QuantizedTensor that also carries
        # the BF16 post-RMSNorm value. Use that BF16 view here -- the indexer
        # weight is BF16 and the matmul needs a float input.
        if isinstance(hidden_states, Fp4QuantizedTensor):
            assert hidden_states.unquantized_hidden_states is not None, (
                "pre_indexer_proj received Fp4QuantizedTensor without bf16 view; "
                "the producer fusion must request return_norm_out=True"
            )
            hidden_states_bf = hidden_states.unquantized_hidden_states
        else:
            hidden_states_bf = hidden_states
        hidden_float = _to_float(hidden_states_bf)
        with _tf32_matmul_enabled():
            # F.linear computes input @ weight.T internally; no explicit .t() needed.
            # _fused_wk_wp_weight is [head_dim + n_heads, hidden_size] (nn.Linear convention).
            # Goes through PyTorch's cuBLAS handle which respects allow_tf32 and
            # dispatches CUBLAS_COMPUTE_32F_FAST_TF32, unlike torch.ops.trtllm.cublas_mm
            # which uses its own handle and always falls back to CUDA-core SGEMM.
            fused_out = F.linear(hidden_float, self._fused_wk_wp_weight)
        indexer_k, weights = fused_out.split([self.head_dim, self.n_heads], dim=-1)
        # Cast indexer_k back to model dtype for downstream ops (k_norm, RoPE, FP8 quantize)
        indexer_k = indexer_k.to(hidden_states_bf.dtype)

        q_pe, q_nope, k_pe, k_nope = self._qk_projection_and_rope(qr, indexer_k, position_ids)
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
            # DeepGEMM's fp8_fp4_(paged_)mqa_logits asserts
            # `weights.scalar_type() == kFloat`. Unlike the FP8 branch (whose
            # `_weight_scale` multiplies by the fp32 `q_scale` tensor and so
            # implicitly upcasts), this branch's only multiplier is a Python
            # float — `bf16 * float` stays bf16 and trips the kernel assert.
            # Cast explicitly so this path doesn't silently rely on an
            # upstream `_to_float`.
            weights = weights.float() * self.weight_scale_factor
        else:
            q_fp8 = q_fp8.view(-1, self.n_heads, self.head_dim)
            q_scale = q_scale.view(-1, self.n_heads, 1)
            weights = self._weight_scale(weights, q_scale)

        return q_fp8, k_fp8, k_scale, weights, q_scale

    def forward_from_projected(
        self,
        metadata: DSAtrtllmAttentionMetadata,
        hidden_states: torch.Tensor,
        indexer_intermediates: List[torch.Tensor],
        is_generation: Optional[bool] = None,
    ) -> torch.Tensor:
        """Run sparse prediction from graph-captured indexer projections.

        Projected inputs cover the full batch. Phase-specific prediction slices
        Q-side inputs here while retaining the full-batch K inputs needed by
        context prediction.
        """
        if is_generation is None:
            phase_start = 0
            phase_end = metadata.num_tokens
        elif is_generation:
            phase_start = metadata.num_ctx_tokens
            phase_end = metadata.num_tokens
        else:
            phase_start = 0
            phase_end = metadata.num_ctx_tokens

        q_fp8, k_fp8, k_scale, weights, q_scale = indexer_intermediates
        q_fp8_phase = q_fp8[phase_start:phase_end]
        weights_phase = weights[phase_start:phase_end]
        q_scale_phase = q_scale[phase_start:phase_end] if q_scale is not None else None

        return self.sparse_attn_indexer(
            metadata,
            hidden_states,
            q_fp8_phase,
            k_fp8,
            k_scale,
            weights_phase,
            q_scale=q_scale_phase,
            is_generation=is_generation,
        )

    @torch.inference_mode()
    def forward(
        self,
        qr: torch.Tensor,
        hidden_states: torch.Tensor,
        metadata: DSAtrtllmAttentionMetadata,
        position_ids: torch.Tensor,
    ):
        q_fp8, k_fp8, k_scale, weights, q_scale = self.pre_indexer_proj(
            qr, hidden_states, position_ids
        )
        indexer_intermediates = [q_fp8, k_fp8, k_scale, weights, q_scale]
        self._update_k_cache(k_fp8, k_scale, metadata)

        # Return topk indices buffer for sparse attention [num_tokens, index_topk]
        return self.forward_from_projected(metadata, hidden_states, indexer_intermediates)
