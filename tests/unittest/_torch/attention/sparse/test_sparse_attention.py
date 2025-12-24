"""
Unit tests for sparse attention with TrtllmAttention backend.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pytest
import torch

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.sparse.kernel import (
    triton_convert_req_index_to_global_index,
)
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention, TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping

ATOL = 1e-2
RTOL = 1e-2


@dataclass(kw_only=True, frozen=False)
class SparseScenario:
    """Base configuration for sparse attention tests."""

    dtype: torch.dtype = torch.float16
    kvcache_dtype: torch.dtype = torch.float16
    num_layers: int = 1
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    page_size: int = 32
    num_pages: int = 16
    batch_size: int = 4

    @property
    def num_kv_groups(self) -> int:
        return self.num_heads // self.num_kv_heads

    @property
    def kv_cache_len(self) -> int:
        return self.page_size * self.num_pages

    @property
    def max_num_pages(self) -> int:
        return self.batch_size * self.num_pages


@dataclass(kw_only=True, frozen=False)
class SparseContextScenario(SparseScenario):
    """Configuration for context phase tests with sparse kv cache write."""

    seq_lens: Tuple[int, ...] = (128,)
    sparse_ratio: float = 0.5

    def __post_init__(self):
        if len(self.seq_lens) != self.batch_size:
            raise ValueError(
                f"seq_lens length {len(self.seq_lens)} must match batch_size {self.batch_size}"
            )

    @property
    def max_seq_len(self) -> int:
        return max(self.seq_lens)

    @property
    def nnz_q(self) -> int:
        return sum(self.seq_lens)


@dataclass(kw_only=True, frozen=False)
class SparseGenerationScenario(SparseScenario):
    """Configuration for generation phase tests with sparse attention."""

    past_kv_lens: Tuple[int, ...] = (256,)
    num_contexts: int = 0
    sparse_ratio: float = 0.5

    def __post_init__(self):
        if len(self.past_kv_lens) != self.batch_size:
            raise ValueError(
                f"past_kv_lens length {len(self.past_kv_lens)} must match batch_size {self.batch_size}"
            )

    @property
    def num_generations(self) -> int:
        return self.batch_size - self.num_contexts

    @property
    def max_past_kv_len(self) -> int:
        return max(self.past_kv_lens)

    @property
    def nnz_q(self) -> int:
        return self.num_generations


class MockSparseAttentionConfig:
    def get_indices_block_size(self) -> int:
        return 1


class TestSparseAttention(TrtllmAttention):
    """TrtllmAttention subclass for testing with predetermined sparse indices."""

    def __init__(
        self,
        *args,
        sparse_kv_indices: Optional[torch.Tensor] = None,
        sparse_kv_offsets: Optional[torch.Tensor] = None,
        sparse_attn_indices: Optional[torch.Tensor] = None,
        sparse_attn_offsets: Optional[torch.Tensor] = None,
        sparse_attn_ctx_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        kwargs["sparse_attention_config"] = MockSparseAttentionConfig()
        kwargs["pos_embd_params"] = None
        super().__init__(*args, **kwargs)
        self._sparse_kv_indices = sparse_kv_indices
        self._sparse_kv_offsets = sparse_kv_offsets
        self._sparse_attn_indices = sparse_attn_indices
        self._sparse_attn_offsets = sparse_attn_offsets
        self._sparse_attn_ctx_indices = sparse_attn_ctx_indices

    def sparse_kv_predict(self, q, k, metadata, **kwargs):
        return self._sparse_kv_indices, self._sparse_kv_offsets

    def sparse_attn_predict(self, q, k, metadata, **kwargs):
        return self._sparse_attn_indices, self._sparse_attn_offsets

    def sparse_attn_ctx_predict(self, q, k, metadata, **kwargs):
        return self._sparse_attn_ctx_indices


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat kv heads to match query heads."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _get_kv_cache_dtype(dtype: torch.dtype):
    """Convert torch dtype to TensorRT-LLM dtype."""
    if dtype == torch.float16:
        return tensorrt_llm.bindings.DataType.HALF
    elif dtype == torch.bfloat16:
        return tensorrt_llm.bindings.DataType.BF16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def create_kv_cache_manager(
    s: SparseScenario, kv_cache: Optional[torch.Tensor] = None
) -> KVCacheManager:
    """Create kv cache manager for testing."""
    kv_cache_config = KvCacheConfig(max_tokens=s.max_num_pages * s.page_size)
    mapping = Mapping(world_size=1, tp_size=1, rank=0)

    manager = KVCacheManager(
        kv_cache_config,
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=s.num_layers,
        num_kv_heads=s.num_kv_heads,
        head_dim=s.head_dim,
        tokens_per_block=s.page_size,
        max_seq_len=s.max_num_pages * s.page_size,
        max_batch_size=s.batch_size,
        mapping=mapping,
        dtype=_get_kv_cache_dtype(s.kvcache_dtype),
    )

    if kv_cache is not None:
        for i in range(s.num_layers):
            manager.get_buffers(i, kv_layout="HND").copy_(kv_cache[i])

    return manager


def _generate_sparse_indices_for_batch(
    seq_len: int,
    sparse_ratio: float,
    num_kv_heads: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate sparse indices for a single batch with per-head patterns."""
    sparse_len = max(1, int(seq_len * sparse_ratio))
    batch_indices = []
    for _ in range(num_kv_heads):
        indices = torch.randperm(seq_len, device=device)[:sparse_len].sort().values
        batch_indices.append(indices)
    return torch.stack(batch_indices, dim=0)


def generate_sparse_kv_indices(
    s: SparseContextScenario, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate sparse kv indices for context phase."""
    all_indices = []
    offsets = [0]

    for seq_len in s.seq_lens:
        batch_indices = _generate_sparse_indices_for_batch(
            seq_len, s.sparse_ratio, s.num_kv_heads, device
        )
        all_indices.append(batch_indices)
        offsets.append(offsets[-1] + batch_indices.shape[1])

    indices = torch.cat(all_indices, dim=1).int()
    offsets = torch.tensor(offsets, dtype=torch.int32, device=device)
    return indices, offsets


def generate_sparse_attn_indices(
    s: SparseGenerationScenario, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate sparse attention indices for generation phase."""
    all_indices = []
    offsets = [0]

    for gen_idx in range(s.num_generations):
        batch_idx = s.num_contexts + gen_idx
        past_kv_len = s.past_kv_lens[batch_idx]
        batch_indices = _generate_sparse_indices_for_batch(
            past_kv_len, s.sparse_ratio, s.num_kv_heads, device
        )
        all_indices.append(batch_indices)
        offsets.append(offsets[-1] + batch_indices.shape[1])

    indices = torch.cat(all_indices, dim=1).int()
    offsets = torch.tensor(offsets, dtype=torch.int32, device=device)
    return indices, offsets


def generate_sparse_attn_ctx_indices(
    s: SparseContextScenario, device: torch.device
) -> torch.Tensor:
    """
    Generate sparse attention context indices for context phase.
    Returns: [num_kv_heads, num_tokens, num_sparse_tokens] with -1 padding.
    """
    all_batch_indices = []
    token_offset = 0

    for seq_len in s.seq_lens:
        batch_indices = []
        for token_idx in range(seq_len):
            # Each token can attend to all previous tokens including itself due to causal mask
            available_kv_len = token_idx + 1
            sparse_len = max(1, int(available_kv_len * s.sparse_ratio))

            per_head_indices = []
            for _ in range(s.num_kv_heads):
                # TODO: check whether the indices are sorted
                indices = torch.randperm(available_kv_len, device=device)[:sparse_len].sort().values
                per_head_indices.append(indices)

            batch_indices.append(per_head_indices)

        all_batch_indices.append(batch_indices)
        token_offset += seq_len

    # Flatten all batches together and find max sparse length
    flattened_indices = []
    for batch in all_batch_indices:
        flattened_indices.extend(batch)

    # Find the maximum sparse length across all tokens and heads
    max_sparse_len = max(
        max(indices.shape[0] for indices in token_indices) for token_indices in flattened_indices
    )

    # Build the tensor with -1 padding
    total_tokens = sum(s.seq_lens)
    result = torch.full(
        (s.num_kv_heads, total_tokens, max_sparse_len), -1, dtype=torch.int32, device=device
    )

    for token_idx, token_indices in enumerate(flattened_indices):
        for head_idx in range(s.num_kv_heads):
            indices = token_indices[head_idx]
            result[head_idx, token_idx, : len(indices)] = indices

    return result


def convert_sparse_attn_ctx_indices_to_global(
    sparse_attn_ctx_indices: torch.Tensor,
    metadata: TrtllmAttentionMetadata,
    layer_idx: int = 0,
    kv_factor: int = 2,
) -> torch.Tensor:
    """
    Convert local sparse_attn_ctx_indices to global KV cache pool indices.
    """
    num_kv_heads, num_tokens, num_sparse_tokens = sparse_attn_ctx_indices.shape
    device = sparse_attn_ctx_indices.device

    tokens_per_block = metadata.kv_cache_manager.tokens_per_block
    num_layers = metadata.kv_cache_manager.num_layers
    stride_factor = num_layers * tokens_per_block * kv_factor * num_kv_heads

    # Build req_idx_per_token
    num_contexts = metadata.num_contexts
    seq_lens = (
        metadata.seq_lens[:num_contexts]
        if hasattr(metadata.seq_lens, "__getitem__")
        else metadata.seq_lens
    )
    host_req_idx_per_token = torch.repeat_interleave(
        torch.arange(num_contexts, dtype=torch.int32), seq_lens, dim=0
    )
    req_idx_per_token = host_req_idx_per_token.to(device)

    # Build block_table
    block_ids_all = metadata.kv_cache_manager.get_batch_cache_indices(
        metadata.request_ids[:num_contexts]
    )

    max_blocks_used = max(len(b) for b in block_ids_all) if block_ids_all else 1

    host_block_table = torch.full((num_contexts, max_blocks_used), -1, dtype=torch.int32)
    for i, blocks in enumerate(block_ids_all):
        if len(blocks) > 0:
            host_block_table[i, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)

    block_table = host_block_table.to(device)

    # Convert to global
    global_indices = triton_convert_req_index_to_global_index(
        req_idx_per_token,
        block_table,
        sparse_attn_ctx_indices,
        BLOCK_SIZE=tokens_per_block,
        NUM_TOPK_TOKENS=num_sparse_tokens,
        BLOCK_N=64,
        stride_factor=stride_factor,
        layer_id=layer_idx,
        num_kv_heads=num_kv_heads,
        kv_factor=kv_factor,
    )

    return global_indices


def _extract_batch_tensors(
    tensor: torch.Tensor, offset: int, length: int, shape_per_token: Tuple
) -> torch.Tensor:
    """Extract and reshape tensors for a specific batch."""
    return tensor[offset : offset + length].view(length, *shape_per_token)


def build_expected_sparse_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    sparse_kv_indices: torch.Tensor,
    sparse_kv_offsets: torch.Tensor,
    s: SparseContextScenario,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Build expected sparse K and V values based on sparse indices."""
    expected_kvs = []
    token_offset = 0

    for batch_idx, seq_len in enumerate(s.seq_lens):
        sparse_len = max(1, int(seq_len * s.sparse_ratio))
        k_batch = _extract_batch_tensors(k, token_offset, seq_len, (s.num_kv_heads, s.head_dim))
        v_batch = _extract_batch_tensors(v, token_offset, seq_len, (s.num_kv_heads, s.head_dim))

        expected_k = torch.zeros(
            sparse_len, s.num_kv_heads, s.head_dim, device=k.device, dtype=k.dtype
        )
        expected_v = torch.zeros_like(expected_k)

        start, end = sparse_kv_offsets[batch_idx].item(), sparse_kv_offsets[batch_idx + 1].item()
        for head_idx in range(s.num_kv_heads):
            indices = sparse_kv_indices[head_idx, start:end]
            expected_k[:, head_idx] = k_batch[indices, head_idx]
            expected_v[:, head_idx] = v_batch[indices, head_idx]

        expected_kvs.append((expected_k, expected_v))
        token_offset += seq_len

    return expected_kvs


def _extract_tokens_from_cache(
    kv_buffer: torch.Tensor,
    block_ids: List[int],
    num_tokens: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract tokens from paged kv cache."""
    device = kv_buffer.device
    k_cache = torch.zeros(num_tokens, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.zeros_like(k_cache)

    for token_idx in range(num_tokens):
        block_idx = token_idx // page_size
        offset_in_block = token_idx % page_size
        block_id = block_ids[block_idx]

        for head_idx in range(num_kv_heads):
            k_cache[token_idx, head_idx] = kv_buffer[block_id, 0, head_idx, offset_in_block, :].to(
                dtype
            )
            v_cache[token_idx, head_idx] = kv_buffer[block_id, 1, head_idx, offset_in_block, :].to(
                dtype
            )

    return k_cache, v_cache


def extract_kv_from_paged_cache(
    kv_cache_manager: KVCacheManager,
    request_ids: List[int],
    sparse_kv_offsets: torch.Tensor,
    s: SparseContextScenario,
    dtype: torch.dtype,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Extract K and V values from paged kv cache."""
    kv_buffer = kv_cache_manager.get_buffers(0, kv_layout="HND")
    kv_caches = []

    for batch_idx in range(s.batch_size):
        num_sparse_tokens = (
            sparse_kv_offsets[batch_idx + 1].item() - sparse_kv_offsets[batch_idx].item()
        )
        block_ids = kv_cache_manager.get_block_ids_per_seq([request_ids[batch_idx]])[0]
        k_cache, v_cache = _extract_tokens_from_cache(
            kv_buffer, block_ids, num_sparse_tokens, s.num_kv_heads, s.head_dim, s.page_size, dtype
        )
        kv_caches.append((k_cache, v_cache))

    return kv_caches


def _compute_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_kv_groups: int,
) -> torch.Tensor:
    """Compute causal attention for a single batch."""
    seq_len = q.shape[2]
    head_dim = q.shape[3]

    k_expanded = repeat_kv(k, num_kv_groups)
    v_expanded = repeat_kv(v, num_kv_groups)

    attn_weights = torch.matmul(q, k_expanded.transpose(-1, -2)) / math.sqrt(head_dim)
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=q.device), diagonal=1
    )
    attn_weights = attn_weights + causal_mask
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        q.dtype
    )
    output = torch.matmul(attn_weights, v_expanded)

    return output


def reference_context_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: SparseContextScenario,
) -> torch.Tensor:
    """Reference implementation for context phase."""
    outputs = []
    token_offset = 0

    for seq_len in s.seq_lens:
        q_batch = _extract_batch_tensors(q, token_offset, seq_len, (s.num_heads, s.head_dim))
        k_batch = _extract_batch_tensors(k, token_offset, seq_len, (s.num_kv_heads, s.head_dim))
        v_batch = _extract_batch_tensors(v, token_offset, seq_len, (s.num_kv_heads, s.head_dim))

        q_batch = q_batch.view(1, seq_len, s.num_heads, s.head_dim).transpose(1, 2)
        k_batch = k_batch.view(1, seq_len, s.num_kv_heads, s.head_dim).transpose(1, 2)
        v_batch = v_batch.view(1, seq_len, s.num_kv_heads, s.head_dim).transpose(1, 2)

        output_batch = _compute_causal_attention(q_batch, k_batch, v_batch, s.num_kv_groups)
        output_batch = output_batch.transpose(1, 2).reshape(seq_len, s.num_heads * s.head_dim)
        outputs.append(output_batch)
        token_offset += seq_len

    return torch.cat(outputs, dim=0)


def reference_context_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sparse_attn_ctx_indices: torch.Tensor,
    s: SparseContextScenario,
) -> torch.Tensor:
    """
    Reference implementation for context phase with sparse attention.
    Uses mask-based approach for each KV head.
    """
    total_tokens = sum(s.seq_lens)
    device = q.device
    dtype = q.dtype

    # Reshape inputs: [num_tokens, num_heads, head_dim]
    q_reshaped = q.view(total_tokens, s.num_heads, s.head_dim)
    k_reshaped = k.view(total_tokens, s.num_kv_heads, s.head_dim)
    v_reshaped = v.view(total_tokens, s.num_kv_heads, s.head_dim)

    outputs = []
    token_offset = 0

    for seq_len in s.seq_lens:
        q_batch = q_reshaped[
            token_offset : token_offset + seq_len
        ]  # [seq_len, num_heads, head_dim]
        k_batch = k_reshaped[
            token_offset : token_offset + seq_len
        ]  # [seq_len, num_kv_heads, head_dim]
        v_batch = v_reshaped[
            token_offset : token_offset + seq_len
        ]  # [seq_len, num_kv_heads, head_dim]

        batch_output = []

        # Process each KV head
        for kv_head_idx in range(s.num_kv_heads):
            k_head = k_batch[:, kv_head_idx, :]
            v_head = v_batch[:, kv_head_idx, :]

            # Build sparse mask for this head
            sparse_mask = torch.full(
                (seq_len, seq_len), float("-inf"), device=device, dtype=torch.float32
            )

            for token_idx in range(seq_len):
                global_token_idx = token_offset + token_idx
                # Get sparse indices for this token: [num_sparse_tokens]
                indices = sparse_attn_ctx_indices[kv_head_idx, global_token_idx]
                # Filter out -1 padding
                valid_indices = indices[indices >= 0]
                # Set mask values to 0 for valid positions
                sparse_mask[token_idx, valid_indices] = 0.0

            # Apply causal mask on top of sparse mask
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=torch.float32),
                diagonal=1,
            )
            combined_mask = sparse_mask + causal_mask

            # Process each query head in this KV group
            for group_idx in range(s.num_kv_groups):
                q_head_idx = kv_head_idx * s.num_kv_groups + group_idx
                q_head = q_batch[:, q_head_idx, :]  # [seq_len, head_dim]

                attn_scores = torch.matmul(q_head, k_head.T) / math.sqrt(s.head_dim)
                attn_scores = attn_scores + combined_mask
                attn_weights = torch.nn.functional.softmax(
                    attn_scores, dim=-1, dtype=torch.float32
                ).to(dtype)

                out_head = torch.matmul(attn_weights, v_head)
                batch_output.append(out_head)

        # Concatenate all heads: [seq_len, num_heads, head_dim] -> [seq_len, num_heads * head_dim]
        batch_output = torch.stack(batch_output, dim=1)
        batch_output = batch_output.reshape(seq_len, s.num_heads * s.head_dim)
        outputs.append(batch_output)

        token_offset += seq_len

    return torch.cat(outputs, dim=0)


def _get_selected_pages_tokens(
    token_indices: torch.Tensor,
    page_size: int,
    kv_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert token indices to page indices and gather all tokens from selected pages."""
    if len(token_indices) == 0:
        return torch.tensor([], dtype=torch.long, device=device)

    page_indices = torch.unique((token_indices // page_size).sort().values)
    selected_tokens = []

    for page_idx in page_indices:
        token_start = page_idx * page_size
        token_end = min(token_start + page_size, kv_len)
        selected_tokens.append(torch.arange(token_start, token_end, device=device))

    return (
        torch.cat(selected_tokens)
        if selected_tokens
        else torch.tensor([], dtype=torch.long, device=device)
    )


def _compute_sparse_attention_per_head(
    q_head: torch.Tensor,
    k_sparse: torch.Tensor,
    v_sparse: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """Compute attention for a single query head."""
    if len(k_sparse) == 0:
        return torch.zeros(head_dim, device=q_head.device, dtype=q_head.dtype)

    attn_weights = torch.matmul(q_head, k_sparse.T) / math.sqrt(head_dim)
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        q_head.dtype
    )
    return torch.matmul(attn_weights, v_sparse)


def reference_generation_sparse_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    sparse_attn_indices: torch.Tensor,
    sparse_attn_offsets: torch.Tensor,
    s: SparseGenerationScenario,
) -> torch.Tensor:
    """Reference implementation for generation phase with sparse attention at page granularity."""
    outputs = []

    for gen_idx in range(s.num_generations):
        batch_idx = s.num_contexts + gen_idx
        past_kv_len = s.past_kv_lens[batch_idx]
        kv_len = past_kv_len + 1

        k_full = k_cache[batch_idx, :kv_len].clone()
        v_full = v_cache[batch_idx, :kv_len].clone()
        k_full[past_kv_len] = k_new[gen_idx].view(s.num_kv_heads, s.head_dim)
        v_full[past_kv_len] = v_new[gen_idx].view(s.num_kv_heads, s.head_dim)

        start, end = sparse_attn_offsets[gen_idx].item(), sparse_attn_offsets[gen_idx + 1].item()
        q_batch = q[gen_idx].view(s.num_heads, s.head_dim)
        head_outputs = []

        for kv_head_idx in range(s.num_kv_heads):
            kv_head_indices = sparse_attn_indices[kv_head_idx, start:end]
            selected_token_indices = _get_selected_pages_tokens(
                kv_head_indices, s.page_size, kv_len, q.device
            )

            if len(selected_token_indices) == 0:
                head_outputs.extend(
                    [torch.zeros(s.head_dim, device=q.device, dtype=q.dtype)] * s.num_kv_groups
                )
                continue

            k_sparse = k_full[selected_token_indices, kv_head_idx, :]
            v_sparse = v_full[selected_token_indices, kv_head_idx, :]

            for group_idx in range(s.num_kv_groups):
                q_head_idx = kv_head_idx * s.num_kv_groups + group_idx
                out_head = _compute_sparse_attention_per_head(
                    q_batch[q_head_idx], k_sparse, v_sparse, s.head_dim
                )
                head_outputs.append(out_head)

        outputs.append(torch.cat(head_outputs, dim=0))

    return torch.stack(outputs, dim=0)


def _setup_context_test(s: SparseContextScenario):
    """Setup common components for context test."""
    device = torch.device("cuda")
    torch.manual_seed(42)

    q = torch.randn(s.nnz_q, s.num_heads * s.head_dim, device=device, dtype=s.dtype)
    k = torch.randn(s.nnz_q, s.num_kv_heads * s.head_dim, device=device, dtype=s.dtype)
    v = torch.randn(s.nnz_q, s.num_kv_heads * s.head_dim, device=device, dtype=s.dtype)

    sparse_kv_indices, sparse_kv_offsets = generate_sparse_kv_indices(s, device)

    kv_cache = torch.zeros(
        s.num_layers,
        s.max_num_pages,
        2,
        s.num_kv_heads,
        s.page_size,
        s.head_dim,
        device=device,
        dtype=s.kvcache_dtype,
    )
    kv_cache_manager = create_kv_cache_manager(s, kv_cache)

    request_ids = list(range(s.batch_size))
    kv_cache_manager.add_dummy_requests(request_ids, list(s.seq_lens))

    metadata = TrtllmAttentionMetadata(
        num_contexts=s.batch_size,
        kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0] * s.batch_size),
        seq_lens=torch.tensor(s.seq_lens, dtype=torch.int32),
        max_num_requests=s.batch_size,
        max_num_tokens=s.nnz_q,
        kv_cache_manager=kv_cache_manager,
        request_ids=request_ids,
        prompt_lens=list(s.seq_lens),
    )
    metadata.prepare()

    attention = TestSparseAttention(
        layer_idx=0,
        num_heads=s.num_heads,
        head_dim=s.head_dim,
        num_kv_heads=s.num_kv_heads,
        sparse_kv_indices=sparse_kv_indices,
        sparse_kv_offsets=sparse_kv_offsets,
    )

    return (
        device,
        q,
        k,
        v,
        sparse_kv_indices,
        sparse_kv_offsets,
        kv_cache_manager,
        request_ids,
        metadata,
        attention,
    )


def _setup_generation_test(s: SparseGenerationScenario):
    """Setup common components for generation test."""
    device = torch.device("cuda")
    torch.manual_seed(42)

    token_nums = [past_len + 1 for past_len in s.past_kv_lens]

    q = torch.randn(s.num_generations, s.num_heads * s.head_dim, device=device, dtype=s.dtype)
    k_new = torch.randn(
        s.num_generations, s.num_kv_heads * s.head_dim, device=device, dtype=s.dtype
    )
    v_new = torch.randn(
        s.num_generations, s.num_kv_heads * s.head_dim, device=device, dtype=s.dtype
    )

    sparse_attn_indices, sparse_attn_offsets = generate_sparse_attn_indices(s, device)

    kv_cache = torch.randn(
        s.num_layers,
        s.max_num_pages,
        2,
        s.num_kv_heads,
        s.page_size,
        s.head_dim,
        device=device,
        dtype=s.kvcache_dtype,
    )
    kv_cache_manager = create_kv_cache_manager(s, kv_cache)

    request_ids = list(range(s.batch_size))
    kv_cache_manager.add_dummy_requests(request_ids, token_nums)

    metadata = TrtllmAttentionMetadata(
        num_contexts=s.num_contexts,
        kv_cache_params=KVCacheParams(
            use_cache=True, num_cached_tokens_per_seq=list(s.past_kv_lens)
        ),
        seq_lens=torch.tensor([1] * s.num_generations).int(),
        max_num_requests=s.batch_size,
        max_num_tokens=s.num_generations,
        kv_cache_manager=kv_cache_manager,
        request_ids=request_ids,
        prompt_lens=list(s.past_kv_lens),
    )
    metadata.prepare()

    attention = TestSparseAttention(
        layer_idx=0,
        num_heads=s.num_heads,
        head_dim=s.head_dim,
        num_kv_heads=s.num_kv_heads,
        sparse_attn_indices=sparse_attn_indices,
        sparse_attn_offsets=sparse_attn_offsets,
    )

    return (
        device,
        q,
        k_new,
        v_new,
        sparse_attn_indices,
        sparse_attn_offsets,
        kv_cache_manager,
        request_ids,
        metadata,
        attention,
    )


def _build_reference_kv_cache(
    kv_cache_manager, request_ids, s: SparseGenerationScenario, device, dtype
):
    """Build reference K, V cache from paged format."""
    k_cache_ref = torch.zeros(
        s.batch_size, s.kv_cache_len, s.num_kv_heads, s.head_dim, device=device, dtype=dtype
    )
    v_cache_ref = torch.zeros_like(k_cache_ref)

    kv_buffer = kv_cache_manager.get_buffers(0, kv_layout="HND")
    for batch_idx, past_kv_len in enumerate(s.past_kv_lens):
        block_ids = kv_cache_manager.get_block_ids_per_seq([request_ids[batch_idx]])[0]
        for block_local_idx, block_id in enumerate(block_ids):
            token_start = block_local_idx * s.page_size
            token_end = min(token_start + s.page_size, past_kv_len)
            tokens_in_block = token_end - token_start

            for head_idx in range(s.num_kv_heads):
                k_cache_ref[batch_idx, token_start:token_end, head_idx] = kv_buffer[
                    block_id, 0, head_idx, :tokens_in_block, :
                ].to(dtype)
                v_cache_ref[batch_idx, token_start:token_end, head_idx] = kv_buffer[
                    block_id, 1, head_idx, :tokens_in_block, :
                ].to(dtype)

    return k_cache_ref, v_cache_ref


@pytest.mark.parametrize(
    "s",
    [
        SparseContextScenario(batch_size=2, seq_lens=(48, 64), sparse_ratio=0.5, num_pages=8),
        SparseContextScenario(
            batch_size=4, seq_lens=(96, 112, 128, 144), sparse_ratio=0.25, num_pages=16
        ),
        SparseContextScenario(batch_size=1, seq_lens=(256,), sparse_ratio=0.75, num_pages=8),
        SparseContextScenario(batch_size=3, seq_lens=(64, 96, 128), sparse_ratio=0.4, num_pages=12),
    ],
    ids=["batch2_var_seq", "batch4_var_seq", "batch1_seq256", "batch3_var_seq"],
)
def test_context_sparse_kv(s: SparseContextScenario):
    """Test context phase with sparse kv cache write."""
    (
        device,
        q,
        k,
        v,
        sparse_kv_indices,
        sparse_kv_offsets,
        kv_cache_manager,
        request_ids,
        metadata,
        attention,
    ) = _setup_context_test(s)

    ref_output = reference_context_attention(q.clone(), k.clone(), v.clone(), s)
    expected_kvs = build_expected_sparse_kv(
        k.clone(), v.clone(), sparse_kv_indices, sparse_kv_offsets, s
    )

    qkv = torch.cat([q, k, v], dim=1)
    output = attention.forward(qkv, None, None, metadata)

    assert output.shape == ref_output.shape, f"Shape mismatch: {output.shape} vs {ref_output.shape}"
    torch.testing.assert_close(output, ref_output, atol=ATOL, rtol=RTOL)
    print(f"Context sparse kv attention output test passed: {s}")

    actual_kvs = extract_kv_from_paged_cache(
        kv_cache_manager, request_ids, sparse_kv_offsets, s, s.dtype
    )

    for batch_idx in range(s.batch_size):
        actual_k, actual_v = actual_kvs[batch_idx]
        expected_k, expected_v = expected_kvs[batch_idx]
        torch.testing.assert_close(
            actual_k,
            expected_k,
            atol=ATOL,
            rtol=RTOL,
            msg=f"K cache mismatch for batch {batch_idx} after sparse compaction",
        )
        torch.testing.assert_close(
            actual_v,
            expected_v,
            atol=ATOL,
            rtol=RTOL,
            msg=f"V cache mismatch for batch {batch_idx} after sparse compaction",
        )

    print(f"Context sparse kv cache content test passed: {s}")
    kv_cache_manager.shutdown()


@pytest.mark.parametrize(
    "s",
    [
        SparseGenerationScenario(
            batch_size=2, past_kv_lens=(96, 128), sparse_ratio=0.5, num_pages=16
        ),
        SparseGenerationScenario(
            batch_size=4, past_kv_lens=(192, 224, 256, 288), sparse_ratio=0.25, num_pages=32
        ),
        SparseGenerationScenario(batch_size=1, past_kv_lens=(64,), sparse_ratio=0.75, num_pages=8),
        SparseGenerationScenario(
            batch_size=3, past_kv_lens=(128, 160, 192), sparse_ratio=0.4, num_pages=24
        ),
    ],
    ids=["batch2_var_kv", "batch4_var_kv", "batch1_kv64", "batch3_var_kv"],
)
def test_generation_sparse_attention(s: SparseGenerationScenario):
    """Test generation phase with sparse attention computation."""
    (
        device,
        q,
        k_new,
        v_new,
        sparse_attn_indices,
        sparse_attn_offsets,
        kv_cache_manager,
        request_ids,
        metadata,
        attention,
    ) = _setup_generation_test(s)

    k_cache_ref, v_cache_ref = _build_reference_kv_cache(
        kv_cache_manager, request_ids, s, device, s.dtype
    )
    ref_sparse_output = reference_generation_sparse_attention(
        q, k_cache_ref, v_cache_ref, k_new, v_new, sparse_attn_indices, sparse_attn_offsets, s
    )

    qkv = torch.cat([q, k_new, v_new], dim=1)
    output = attention.forward(qkv, None, None, metadata)

    expected_shape = (s.num_generations, s.num_heads * s.head_dim)
    assert output.shape == expected_shape, f"Shape mismatch: {output.shape} vs {expected_shape}"
    assert torch.isfinite(output).all(), "Output contains non-finite values"

    torch.testing.assert_close(output, ref_sparse_output, atol=ATOL, rtol=RTOL)
    print(f"Generation sparse attention test passed: {s}")
    kv_cache_manager.shutdown()


@pytest.mark.parametrize(
    "s",
    [
        SparseContextScenario(
            batch_size=2,
            seq_lens=(128, 64),
            sparse_ratio=0.5,
            num_pages=8,
            num_kv_heads=1,
            num_heads=8,
            head_dim=128,
        ),
    ],
)
def test_context_sparse_attention_mqa(s: SparseContextScenario):
    """Test context phase with sparse attention using sparse_attn_ctx_indices (MQA setup)."""
    device, q, k, v, _, _, kv_cache_manager, request_ids, metadata, _ = _setup_context_test(s)

    # Generate sparse attention context indices
    sparse_attn_ctx_indices = generate_sparse_attn_ctx_indices(s, device)

    # Convert to global indices for attentionOp
    global_sparse_attn_ctx_indices = convert_sparse_attn_ctx_indices_to_global(
        sparse_attn_ctx_indices, metadata, layer_idx=0
    )

    # Compute reference output using local indices
    ref_output = reference_context_sparse_attention(
        q.clone(), k.clone(), v.clone(), sparse_attn_ctx_indices, s
    )

    # Verify reference output shape
    total_tokens = sum(s.seq_lens)
    expected_shape = (total_tokens, s.num_heads * s.head_dim)
    assert ref_output.shape == expected_shape, (
        f"Reference output shape mismatch: {ref_output.shape} vs {expected_shape}"
    )
    assert torch.isfinite(ref_output).all(), "Reference output contains non-finite values"

    print(f"Context sparse attention MQA reference test passed: {s}")

    attention = TestSparseAttention(
        layer_idx=0,
        num_heads=s.num_heads,
        head_dim=s.head_dim,
        num_kv_heads=s.num_kv_heads,
        sparse_attn_ctx_indices=global_sparse_attn_ctx_indices,  # Use global indices here
    )

    qkv = torch.cat([q, k, v], dim=1)
    output = attention.forward(qkv, None, None, metadata)
    torch.testing.assert_close(output, ref_output, atol=ATOL, rtol=RTOL)
    print(f"Context sparse attention MQA forward test passed: {s}")

    kv_cache_manager.shutdown()


if __name__ == "__main__":
    s = SparseContextScenario(
        batch_size=2,
        seq_lens=(128, 64),
        sparse_ratio=0.5,
        num_pages=8,
        num_kv_heads=1,
        num_heads=8,
        head_dim=128,
    )
    test_context_sparse_attention_mqa(s)
