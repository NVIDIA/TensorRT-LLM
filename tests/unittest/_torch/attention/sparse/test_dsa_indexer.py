"""
Test suite for DeepGEMM indexer kernels and some related utilities.

This file tests:
1. fp8_mqa_logits operation from the DeepGEMM library
2. fp8_paged_mqa_logits operation with paged KV cache
3. compute_cu_seqlen_kv_bounds utility for batched causal attention
"""

import random
from unittest.mock import Mock, patch

import pytest
import torch
from utils.util import check_accuracy, skip_pre_hopper

from tensorrt_llm import deep_gemm
from tensorrt_llm._torch.attention_backend.interface import (
    PositionalEmbeddingParams, RopeParams)
from tensorrt_llm._torch.attention_backend.sparse.dsa import (
    DSACacheManager, DSAtrtllmAttentionMetadata, Indexer,
    compute_cu_seqlen_kv_bounds_with_cache, split_prefill_chunks)
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.bindings.internal.batch_manager import \
    CacheType as CacheTypeCpp
from tensorrt_llm.deep_gemm import fp8_paged_mqa_logits
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization.utils import fp8_utils


def has_deep_gemm():
    try:
        return deep_gemm is not None
    except Exception:
        return False


def _ceil_to_ue8m0(x: torch.Tensor):
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b


def create_dsa_cache_manager(
    batch_size: int,
    head_dim: int,
    tokens_per_block: int,
    max_seq_len: int,
    num_layers: int = 1,
):
    """Helper to create a DSACacheManager for testing."""

    # Create a minimal sparse attention config
    class SparseAttentionConfig:

        def __init__(self, index_head_dim, index_n_heads, index_topk):
            self.index_head_dim = index_head_dim
            self.index_n_heads = index_n_heads
            self.index_topk = index_topk
            self.prompt_budget = 1024

    sparse_attn_config = SparseAttentionConfig(
        index_head_dim=head_dim,
        index_n_heads=32,  # Default number of heads for indexer
        index_topk=2048)

    # Create KV cache config
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=False,
        max_tokens=max_seq_len * batch_size,
    )

    # Create mapping (single GPU, no parallelism)
    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)

    # Create cache manager
    # Use SELFKONLY for DSA (similar to MLA usage in _util.py)
    cache_manager = DSACacheManager(
        kv_cache_config=kv_cache_config,
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=num_layers,
        num_kv_heads=1,  # MQA
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=DataType.HALF,
        sparse_attn_config=sparse_attn_config,
    )

    return cache_manager, sparse_attn_config


def create_indexer(sparse_attn_config, layer_idx=0):
    """Helper to create an Indexer for testing."""
    # Create RopeParams
    rope_params = RopeParams(
        dim=64,  # qk_rope_head_dim
        theta=10000.0,
        max_positions=4096)

    # Create PositionalEmbeddingParams with required 'type' argument
    pos_embd_params = PositionalEmbeddingParams(
        type=PositionEmbeddingType.rope_gpt_neox,
        rope=rope_params,
        is_neox=True)

    # Create MLAParams
    class MLAParams:

        def __init__(self, head_dim):
            self.hidden_size = 4096  # Example hidden size
            self.q_lora_rank = 512  # Example q_lora_rank
            self.qk_rope_head_dim = 64

    mla_params = MLAParams(sparse_attn_config.index_head_dim)

    # Mock RotaryEmbedding since we're only testing cache management, not rope functionality
    with patch(
            'tensorrt_llm._torch.attention_backend.sparse.dsa.RotaryEmbedding'
    ) as mock_rope:
        # Create a mock instance with a simple forward method
        mock_rope_instance = Mock()
        mock_rope_instance.forward = Mock(
            side_effect=lambda pos_ids, tensors: tensors)
        mock_rope.return_value = mock_rope_instance

        indexer = Indexer(
            quant_config=None,
            pos_embd_params=pos_embd_params,
            mla_params=mla_params,
            skip_create_weights_in_init=True,  # Skip weight creation for test
            sparse_attention_config=sparse_attn_config,
            dtype=torch.bfloat16,
            layer_idx=layer_idx)

    return indexer


def _calc_diff(x: torch.Tensor, y: torch.Tensor):
    """Return a global difference metric for unit tests.

    DeepGEMM kernels on Blackwell/B200 currently exhibit noticeable per-element
    error, causing ``torch.testing.assert_close`` to fail.  Instead of checking
    every element, we compute a cosine-style similarity over the whole tensor
    and report ``1 - sim``.  Once kernel accuracy improves this helper can be
    removed.
    """

    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def per_custom_dims_cast_to_fp8(x: torch.Tensor, dims, use_ue8m0=False):
    """
    Cast tensor to FP8 per custom dimensions.
    For kv, we quantize along dimension 0 (sequence dimension).
    """
    excluded_dims = tuple([i for i in range(x.dim()) if i not in set(dims)])
    x_amax = x.abs().float().amax(dim=excluded_dims, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = _ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled, sf.squeeze()


def _ref_fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
):
    """
    Reference implementation of fp8_paged_mqa_logits (optimized version).

    Args:
        q: [batch_size, next_n, num_heads, head_dim]
        kv_cache: [num_blocks, block_size, 1, head_dim]
        weights: [batch_size * next_n, num_heads]
        context_lens: [batch_size]
        block_tables: [batch_size, max_num_blocks]
        max_model_len: Maximum sequence length

    Returns:
        logits: [batch_size * next_n, max_model_len]
    """
    batch_size, next_n, _, _ = q.size()
    _, block_size, _, _ = kv_cache.size()

    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )

    context_lens_list = context_lens.tolist()

    for i in range(batch_size):
        context_len = context_lens_list[i]

        # Query positions: [context_len - next_n, ..., context_len - 1]
        q_offsets = torch.arange(context_len - next_n,
                                 context_len,
                                 device="cuda")

        # Transpose weights for this sequence: [num_heads, next_n]
        weight_slice = (weights[i * next_n:(i + 1) * next_n, :].transpose(
            0, 1).contiguous())

        # Process each block in the sequence
        for block_rk in range(cdiv(context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_cache[block_idx]

            # Key positions in this block
            k_offsets = torch.arange(
                block_rk * block_size,
                (block_rk + 1) * block_size,
                device="cuda",
            )

            # Causal mask: k_pos < context_len AND k_pos <= q_pos
            mask = (k_offsets[None, :] < context_len) & (k_offsets[None, :]
                                                         <= q_offsets[:, None])

            # Compute attention scores: [num_heads, next_n, block_size]
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                    logits.dtype),
                float("-inf"),
            )

            # Apply ReLU, multiply by weights, and sum over heads
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)  # [next_n, block_size]

            # Write to output with additional causal mask
            logits[
                i * next_n:(i + 1) * next_n,
                block_rk * block_size:(block_rk + 1) * block_size,
            ] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s,
                            float("-inf"))

    return logits


def _ref_fp8_mqa_logits(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
):
    """
    Reference implementation of fp8_mqa_logits.
    out_ij = q[i, :, :] @ kv[j, :] # [num_heads]
    out_ij = out_ij.relu() * weights[i, :]
    out_ij = out_ij.sum()  # Scalar

    Args:
        q: [seq_len, num_heads, head_dim]
        kv: [seq_len_kv, head_dim]
        weights: [seq_len, num_heads]
        cu_seqlen_ks: [seq_len]
        cu_seqlen_ke: [seq_len]

    Returns:
        logits: [seq_len, seq_len_kv]
    """

    seq_len_kv = kv.shape[0]

    k = kv
    q = q.float()
    k = k.float()

    mask_lo = (torch.arange(0, seq_len_kv, device="cuda")[None, :]
               >= cu_seqlen_ks[:, None])
    mask_hi = (torch.arange(0, seq_len_kv, device="cuda")[None, :]
               < cu_seqlen_ke[:, None])
    mask = mask_lo & mask_hi

    score = torch.einsum("mhd,nd->hmn", q, k)
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))

    return logits


@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@skip_pre_hopper
def test_deepgemm_fp8_mqa_logits_basic():
    """
    Basic test for deepgemm.fp8_mqa_logits kernel.
    Tests the disable_cp path with simple validation.
    """
    torch.manual_seed(0)

    num_heads, head_dim = 64, 128
    seq_len = 2048
    seq_len_kv = 4096
    #[seq_len, num_heads, head_dim]
    q = torch.randn(
        seq_len,
        num_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    #[seq_len_kv, head_dim] -> num_head = 1
    kv = torch.randn(
        seq_len_kv,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    #[seq_len, num_heads]
    weights = torch.randn(
        seq_len,
        num_heads,
        device="cuda",
        dtype=torch.float32,
    )
    # ks[i] -> ke[i] for each q[i]
    ks = torch.zeros(seq_len, dtype=torch.int, device="cuda")
    ke = torch.arange(seq_len, dtype=torch.int,
                      device="cuda") + (seq_len_kv - seq_len)

    # Convert to FP8
    q_fp8 = q.to(torch.float8_e4m3fn)
    kv_fp8 = per_custom_dims_cast_to_fp8(kv, (0, ), False)
    logits = deep_gemm.fp8_mqa_logits(q_fp8, kv_fp8, weights, ks,
                                      ke)  # -> [seq_len, seq_len_kv]

    # Basic sanity checks
    assert logits.shape == (seq_len, seq_len_kv), \
        f"Expected shape ({seq_len}, {seq_len_kv}), got {logits.shape}"
    assert logits.dtype == torch.float32, \
        f"Expected dtype torch.float32, got {logits.dtype}"

    ref_logits = _ref_fp8_mqa_logits(
        q=q,
        kv=kv,
        weights=weights,
        cu_seqlen_ks=ks,
        cu_seqlen_ke=ke,
    )

    ref_neginf_mask = ref_logits == float("-inf")
    neginf_mask = logits == float("-inf")
    assert torch.equal(neginf_mask, ref_neginf_mask)

    ref_logits = ref_logits.masked_fill(ref_neginf_mask, 0)
    logits = logits.masked_fill(neginf_mask, 0)

    diff = _calc_diff(logits, ref_logits)
    assert diff < 1e-3, f"{diff=}"  # check for cosine similarity
    check_accuracy(logits, ref_logits, atol=1e-2, rtol=2e-1,
                   percent=0.9)  # double check for per-element similarity


def _create_mock_metadata(request_ids,
                          batch_size,
                          num_contexts,
                          num_generations,
                          seq_lens,
                          kv_lens,
                          num_cached_tokens,
                          cache_manager,
                          num_ctx_tokens,
                          num_tokens,
                          indexer_max_chunk_size=8194,
                          max_draft_tokens=0,
                          enable_context_mla_with_cached_kv=False,
                          index_topk=2048,
                          enable_indexer_skip=False):
    """Helper to create mock metadata for testing."""

    class MockKVCacheParams:

        def __init__(self):
            self.num_cached_tokens_per_seq = num_cached_tokens

    class MockMetadata(DSAtrtllmAttentionMetadata):

        def __init__(self):
            self.num_sms = deep_gemm.get_num_sms()
            self.request_ids = request_ids
            self.num_contexts = num_contexts
            self.num_generations = num_generations
            self._num_seqs = num_contexts + num_generations
            self.max_draft_tokens = max_draft_tokens
            self.sparse_mla_topk = index_topk
            self.enable_indexer_skip = enable_indexer_skip
            # Keep seq_lens on CPU for split_prefill_chunks and other CPU operations
            # CUDA kernels will convert to CUDA as needed
            self.seq_lens = seq_lens.cpu() if seq_lens.is_cuda else seq_lens
            self.kv_lens = kv_lens
            self.kv_cache_params = MockKVCacheParams()
            self.kv_cache_manager = cache_manager
            self.kv_lens_cuda_runtime = kv_lens.cuda()
            self.indexer_k_cache_block_offsets = torch.zeros(
                [batch_size, cache_manager.max_blocks_per_seq],
                device='cuda',
                dtype=torch.int32,
            )
            self.host_indexer_k_cache_block_offsets = torch.zeros_like(
                self.indexer_k_cache_block_offsets,
                device='cpu',
                pin_memory=True,
            )
            block_ids = cache_manager.get_batch_cache_indices(request_ids)
            for i in range(len(block_ids)):
                self.host_indexer_k_cache_block_offsets[
                    i, :len(block_ids[i])] = torch.tensor(block_ids[i],
                                                          dtype=torch.int32)
            self.indexer_k_cache_block_offsets[:batch_size].copy_(
                self.host_indexer_k_cache_block_offsets[:batch_size],
                non_blocking=True)

            self.slot_mapping_fp8 = torch.zeros((num_tokens, ),
                                                device='cuda',
                                                dtype=torch.int64)
            self.slot_mapping_scale = torch.zeros((num_tokens, ),
                                                  device='cuda',
                                                  dtype=torch.int64)
            self.scheduler_metadata_buffer = torch.zeros((self.num_sms + 1, 2),
                                                         device='cuda',
                                                         dtype=torch.int32)
            self.cu_seqlen_ks = torch.zeros((num_tokens, ),
                                            device='cuda',
                                            dtype=torch.int32)
            self.cu_seqlen_ke = torch.zeros((num_tokens, ),
                                            device='cuda',
                                            dtype=torch.int32)
            self.ctx_kv_offsets = torch.zeros((num_tokens, 1),
                                              device='cuda',
                                              dtype=torch.int32)
            self.gen_kv_offsets = torch.zeros((num_tokens, 1),
                                              device='cuda',
                                              dtype=torch.int32)
            self.host_slot_mapping_fp8 = torch.zeros_like(self.slot_mapping_fp8,
                                                          device='cpu',
                                                          pin_memory=True)
            self.host_slot_mapping_scale = torch.zeros_like(
                self.slot_mapping_scale, device='cpu', pin_memory=True)
            self.host_ctx_kv_indptr = torch.zeros((num_contexts + 1, ),
                                                  device='cpu',
                                                  pin_memory=True,
                                                  dtype=torch.int64)
            self.host_gen_kv_indptr = torch.zeros((num_generations + 1, ),
                                                  device='cpu',
                                                  pin_memory=True,
                                                  dtype=torch.int64)
            # Add host_ctx_cached_token_indptr for prepare_one_prefill_chunk
            self.host_ctx_cached_token_indptr = torch.zeros(
                (num_contexts + 1, ),
                device='cpu',
                pin_memory=True,
                dtype=torch.int64)
            self._num_ctx_tokens = num_ctx_tokens
            self._num_tokens = num_tokens
            self.num_gen_tokens = num_tokens - num_ctx_tokens
            # Also set private attributes used by DSAtrtllmAttentionMetadata
            self._num_contexts = num_contexts
            self._num_generations = num_generations
            self._num_ctx_tokens = num_ctx_tokens
            self._num_tokens = num_tokens

            torch.cumsum(kv_lens[:num_contexts],
                         dim=0,
                         dtype=torch.int64,
                         out=self.host_ctx_kv_indptr[1:num_contexts + 1])
            torch.cumsum(kv_lens[num_contexts:num_contexts + num_generations],
                         dim=0,
                         dtype=torch.int64,
                         out=self.host_gen_kv_indptr[1:num_generations + 1])
            # Compute host_ctx_cached_token_indptr from num_cached_tokens
            if num_contexts > 0:
                cached_lens = torch.tensor(num_cached_tokens[:num_contexts],
                                           dtype=torch.int64)
                torch.cumsum(
                    cached_lens,
                    dim=0,
                    dtype=torch.int64,
                    out=self.host_ctx_cached_token_indptr[1:num_contexts + 1])

            # Add indexer-specific attributes
            self.indexer_max_chunk_size = indexer_max_chunk_size
            self.indexer_prefill_chunks = None

            self.enable_context_mla_with_cached_kv = enable_context_mla_with_cached_kv

            # Add runtime_features for chunked prefill detection
            class RuntimeFeatures:

                def __init__(self):
                    self.chunked_prefill = enable_context_mla_with_cached_kv
                    self.cache_reuse = False
                    self.has_speculative_draft_tokens = False
                    self.chunk_size = 512
                    self.chunked_prefill_buffer_batch_size = 4

            self.runtime_features = RuntimeFeatures()

            # Add expanded buffers for MTP support
            self.use_expanded_buffers_for_mtp = (
                (self.max_draft_tokens > 1 and get_sm_version() == 90)
                or ((self.max_draft_tokens == 2 or self.max_draft_tokens > 3)
                    and get_sm_version() >= 100))
            self.kv_lens_expanded_cuda = torch.zeros(
                (self.num_seqs * (1 + self.max_draft_tokens), ),
                device='cuda',
                dtype=torch.int32)
            self.kv_lens_expanded_host = torch.zeros_like(
                self.kv_lens_expanded_cuda, device='cpu', pin_memory=True)
            self.block_table_expanded = torch.zeros(
                (self.num_seqs * (1 + self.max_draft_tokens),
                 self.kv_cache_manager.max_blocks_per_seq),
                device='cuda',
                dtype=torch.int32)
            self.host_block_table_expanded = torch.zeros_like(
                self.block_table_expanded, device='cpu', pin_memory=True)
            self.scheduler_metadata_buffer_expanded = torch.zeros(
                (self.num_sms + 1, 2), device='cuda', dtype=torch.int32)
            if self.max_draft_tokens == 3:
                self.scheduler_metadata_buffer_mtp3 = torch.zeros(
                    (self.num_sms // 2 + 1, 2),
                    device='cuda',
                    dtype=torch.int32)
            if self.use_expanded_buffers_for_mtp:
                gen_kv_lens = kv_lens[num_contexts:self.num_seqs]
                gen_kv_lens_expanded = torch.stack([gen_kv_lens] *
                                                   (1 + self.max_draft_tokens),
                                                   dim=0)
                gen_kv_lens_expanded = gen_kv_lens_expanded.transpose(
                    0, 1).contiguous().flatten()
                self.kv_lens_expanded_host[:self.num_gen_tokens].copy_(
                    gen_kv_lens_expanded)
                self.kv_lens_expanded_cuda[:self.num_gen_tokens].copy_(
                    self.kv_lens_expanded_host[:self.num_gen_tokens],
                    non_blocking=True)

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
                        self.host_block_table_expanded[:self.num_gen_tokens, :
                                                       max_len].copy_(
                                                           expanded_blocks,
                                                           non_blocking=True)
                        self.block_table_expanded[:self.num_gen_tokens].copy_(
                            self.host_block_table_expanded[:self.
                                                           num_gen_tokens],
                            non_blocking=True)

            # Add skip indexer attributes
            self.topk_indices_buffer = torch.zeros(
                (num_tokens, self.sparse_mla_topk),
                device='cuda',
                dtype=torch.int32)

            if self.num_contexts > 0 and self.enable_indexer_skip:
                self.skip_indexer_for_ctx_reqs = kv_lens[:self.num_contexts].max(
                ).item() <= self.sparse_mla_topk
            else:
                self.skip_indexer_for_ctx_reqs = False

            if self.num_generations > 0 and self.enable_indexer_skip:
                self.max_draft_tokens + 1
                self.skip_indexer_for_gen_reqs = kv_lens[
                    self.num_contexts:self.num_seqs].max().item(
                    ) <= self.sparse_mla_topk
            else:
                self.skip_indexer_for_gen_reqs = False
            self.prepare_dense_topk_indices(self.kv_lens_cuda_runtime,
                                            device=True)

        @property
        def num_seqs(self) -> int:
            """
            The number of sequences in the batch.
            """
            return self._num_seqs

    return MockMetadata()


def validate_topk_indices(topk_indices_0, topk_indices_1, total_tokens):
    """
    Validate the similarity between two topk indices.
    """
    num_exact_matches = 0
    total_similarity = 0.0
    min_similarity = 1.0

    for token_idx in range(total_tokens):
        valid_0 = topk_indices_0[token_idx][topk_indices_0[token_idx] != -1]
        valid_1 = topk_indices_1[token_idx][topk_indices_1[token_idx] != -1]

        if torch.equal(valid_0, valid_1):
            num_exact_matches += 1
            similarity = 1.0
            total_similarity += similarity
        else:
            valid_0_set = set(valid_0.cpu().tolist())
            valid_1_set = set(valid_1.cpu().tolist())
            intersection = len(valid_0_set & valid_1_set)
            union = len(valid_0_set | valid_1_set)
            similarity = intersection / union if union > 0 else 0.0
            total_similarity += similarity

        # Track min similarity
        min_similarity = min(min_similarity, similarity)

    return num_exact_matches, total_similarity, min_similarity


@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@skip_pre_hopper
def test_indexer_k_cache_scatter_custom_op():
    """
    Direct comparison: CUDA kernel vs Python reference for k_cache scatter.

    This test ensures the new CUDA kernel indexer_k_cache_scatter_op produces
    exactly the same results as the Python scatter implementation.
    """
    torch.manual_seed(123)

    # Test parameters
    head_dim = 128
    block_size = 64
    batch_size = 3
    num_tokens = 96  # 3 requests × 32 tokens each
    max_seq_len = 512

    # Use different layers for CUDA vs Python to test non-contiguous handling
    layer_idx_cuda = 1  # CUDA kernel writes to layer 0
    layer_idx_python = 2  # Python reference writes to layer 1

    # Create cache manager with multiple layers
    cache_manager, sparse_attn_config = create_dsa_cache_manager(
        batch_size=batch_size,
        head_dim=head_dim,
        tokens_per_block=block_size,
        max_seq_len=max_seq_len,
        num_layers=3)  # Multi-layer pool for non-contiguous test

    # Allocate blocks
    request_ids = list(range(batch_size))
    tokens_per_req = [32, 32, 32]
    cache_manager.add_dummy_requests(request_ids,
                                     tokens_per_req,
                                     is_gen=False,
                                     prepare_resource=True)

    # Create metadata
    metadata = _create_mock_metadata(
        request_ids,
        batch_size,
        num_contexts=batch_size,
        num_generations=0,
        seq_lens=torch.tensor(tokens_per_req, dtype=torch.int32),
        kv_lens=torch.tensor(tokens_per_req, dtype=torch.int32),
        num_cached_tokens=[0] * batch_size,
        cache_manager=cache_manager,
        num_ctx_tokens=num_tokens,
        num_tokens=num_tokens,
    )

    from tensorrt_llm._torch.attention_backend.sparse.dsa import Indexer
    Indexer.prepare(metadata)

    # Generate test data
    k_original = torch.randn((num_tokens, head_dim),
                             device="cuda",
                             dtype=torch.bfloat16)
    k_fp8, k_scale = fp8_utils.fp8_quantize_1x128_sf_transpose(k_original)

    # Prepare byte-level data
    scale_size = k_scale.shape[1] * 4
    k_fp8_bytes = k_fp8.view(-1).view(torch.uint8).view(num_tokens, head_dim)
    k_scale_flat = k_scale.view(-1)
    if k_scale_flat.stride(-1) != 1:
        k_scale_flat = torch.as_strided(k_scale_flat.contiguous(),
                                        size=(k_scale_flat.numel(), ),
                                        stride=(1, ))
    k_scale_bytes = k_scale_flat.view(torch.uint8).view(num_tokens, scale_size)

    flat_indices_fp8 = metadata.slot_mapping_fp8[:num_tokens]
    flat_indices_scale = metadata.slot_mapping_scale[:num_tokens]

    # ========== Use Different Layers for CUDA vs Python ==========
    # Simple approach: use layer 0 for CUDA, layer 1 for Python
    # Both get the same input data, but write to different layers
    # Then we extract and compare the outputs from each layer

    # Get k_cache for CUDA path (layer 0)
    k_cache_cuda = cache_manager.get_indexer_k_cache_buffers(layer_idx_cuda)
    k_cache_cuda.zero_()

    # Get k_cache for Python path (layer 1)
    k_cache_python = cache_manager.get_indexer_k_cache_buffers(layer_idx_python)
    k_cache_python.zero_()

    # Print cache properties
    print(f"\n=== Cache Properties ===")
    print(f"  CUDA (layer {layer_idx_cuda}):")
    print(f"    Shape: {k_cache_cuda.shape}")
    print(f"    Stride: {k_cache_cuda.stride()}")
    print(f"    is_contiguous: {k_cache_cuda.is_contiguous()}")
    print(f"  Python (layer {layer_idx_python}):")
    print(f"    Shape: {k_cache_python.shape}")
    print(f"    Stride: {k_cache_python.stride()}")
    print(f"    is_contiguous: {k_cache_python.is_contiguous()}")

    # ========== Path 1: CUDA Kernel ==========
    print(f"\n=== Path 1: CUDA Kernel ===")
    torch.ops.trtllm.indexer_k_cache_scatter_op(k_fp8_bytes, k_scale_bytes,
                                                k_cache_cuda, flat_indices_fp8,
                                                flat_indices_scale)
    torch.cuda.synchronize()
    print(f"✓ CUDA kernel completed")

    # ========== Path 2: Python Reference ==========
    print(f"\n=== Path 2: Python Reference ===")

    def _unravel_indices(flat_indices, shape):
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

    # Scatter FP8 data
    byte_offsets = torch.arange(head_dim,
                                device=k_cache_python.device).unsqueeze(0)
    scatter_indices_fp8 = flat_indices_fp8.unsqueeze(1) + byte_offsets
    scatter_indices_fp8 = _unravel_indices(scatter_indices_fp8,
                                           k_cache_python.shape)
    k_cache_python[scatter_indices_fp8] = k_fp8_bytes

    # Scatter scale data
    byte_offsets = torch.arange(scale_size,
                                device=k_cache_python.device).unsqueeze(0)
    scatter_indices_scale = flat_indices_scale.unsqueeze(1) + byte_offsets
    scatter_indices_scale = _unravel_indices(scatter_indices_scale,
                                             k_cache_python.shape)
    k_cache_python[scatter_indices_scale] = k_scale_bytes

    # ========== Validation: Byte-for-Byte Comparison ==========
    print(f"\n=== Validation ===")

    total_bytes = k_cache_cuda.numel()

    # Compare entire cache tensors
    if torch.equal(k_cache_cuda, k_cache_python):
        print(f"✅ PERFECT MATCH! CUDA and Python produce identical cache")
        print(f"  Total bytes compared: {total_bytes}")
        print(
            f"  Tokens: {num_tokens}, head_dim: {head_dim}, block_size: {block_size}"
        )
    else:
        # Find differences
        diff_mask = k_cache_cuda != k_cache_python
        num_diffs = diff_mask.sum().item()

        print(
            f"⚠️  Found {num_diffs}/{total_bytes} byte differences ({100*num_diffs/total_bytes:.4f}%)"
        )

        # Show first few differences
        diff_indices = torch.nonzero(diff_mask.view(-1))[:5]
        for idx in diff_indices:
            flat_idx = idx.item()
            print(
                f"  Byte {flat_idx}: CUDA={k_cache_cuda.view(-1)[flat_idx].item()}, "
                f"Python={k_cache_python.view(-1)[flat_idx].item()}")

        # Fail the test
        raise AssertionError(
            "CUDA kernel produced different results than Python reference")


@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@skip_pre_hopper
def test_fp8_k_cache_roundtrip():
    """Verify FP8 quantization scales survive write/read cycle for multiple requests."""
    torch.manual_seed(42)

    # Setup with 2 requests, each spanning multiple blocks
    batch_size = 2
    head_dim, block_size = 128, 64
    max_seq_len = 512
    num_tokens_per_req = [150, 100]  # Request 0: 3 blocks, Request 1: 2 blocks

    cache_manager, sparse_attn_config = create_dsa_cache_manager(
        batch_size=batch_size,
        head_dim=head_dim,
        tokens_per_block=block_size,
        max_seq_len=max_seq_len,
        num_layers=1)
    indexer = create_indexer(sparse_attn_config, layer_idx=0)

    # Allocate blocks for both requests
    request_ids = [0, 1]
    cache_manager.add_dummy_requests(request_ids,
                                     num_tokens_per_req,
                                     is_gen=False,
                                     prepare_resource=True)

    # Prepare and write data for each request
    total_tokens = sum(num_tokens_per_req)
    metadata = _create_mock_metadata(
        request_ids,
        batch_size,
        num_contexts=batch_size,
        num_generations=0,
        seq_lens=torch.tensor(num_tokens_per_req, dtype=torch.int32),
        kv_lens=torch.tensor(num_tokens_per_req, dtype=torch.int32),
        num_cached_tokens=[0] * batch_size,
        cache_manager=cache_manager,
        num_ctx_tokens=total_tokens,
        num_tokens=total_tokens,
    )
    Indexer.prepare(metadata)

    # Generate unique patterns for each request and quantize
    k_original = torch.randn((total_tokens, head_dim),
                             device="cuda",
                             dtype=torch.bfloat16)
    k_fp8, k_scale = fp8_utils.fp8_quantize_1x128_sf_transpose(k_original)

    # Write to cache
    indexer._update_k_cache(k_fp8, k_scale, metadata)

    # Verify scales for both requests
    cache_flat = cache_manager.indexer_k_cache_pool_per_layer[
        0]  # [num_blocks, flat_bytes]

    scale_offset = block_size * head_dim  # Scales start after FP8 data
    scale_size = 4  # float32
    original_scales = k_scale.cpu().numpy()

    # Verify scales for all requests
    global_token_idx = 0
    for req_idx, num_tokens in enumerate(num_tokens_per_req):
        for local_token_idx in range(num_tokens):
            # Compute block location
            block_idx_in_seq = local_token_idx // block_size
            pos_in_block = local_token_idx % block_size
            block_id = metadata.host_indexer_k_cache_block_offsets[
                req_idx, block_idx_in_seq].item()

            # Extract stored scale
            scale_bytes = cache_flat[block_id, scale_offset +
                                     pos_in_block * scale_size:scale_offset +
                                     (pos_in_block + 1) * scale_size]
            stored_scale = scale_bytes.view(torch.float32).item()

            # Compare with original
            orig_scale = original_scales[global_token_idx]
            assert abs(orig_scale - stored_scale) < 1e-6, \
                f"Request {req_idx}, token {local_token_idx} (block {block_idx_in_seq}, pos {pos_in_block}): " \
                f"scale mismatch (orig={orig_scale:.6f}, stored={stored_scale:.6f})"

            global_token_idx += 1


@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@skip_pre_hopper
@pytest.mark.parametrize("batch_size,next_n", [(4, 1), (2, 2), (4, 3), (4, 4)])
def test_indexer_decode_with_paged_kv_cache(batch_size, next_n):
    """
    Test FP8 paged KV cache with two-phase workflow and variable context lengths.

    Validates:
    - Variable-length sequences in the same batch
    - Slot mapping computation for non-interleaved FP8 cache layout
    - Vectorized scatter operations for FP8 data and scales
    - Block table management for paged attention
    - Multi-token generation phase (similar to prepare_resources)
    - Kernel accuracy vs reference implementation
    """
    torch.manual_seed(123)
    random.seed(123)

    # Test parameters
    heads, head_dim = 32, 128
    block_size = 64
    avg_context_len = 2048
    num_gen_tokens = next_n  # Number of tokens to generate per sequence
    max_model_len = 4096
    layer_idx = 0

    # Generate variable context lengths per sequence
    context_lens_context = torch.randint(int(0.7 * avg_context_len),
                                         int(1.4 * avg_context_len),
                                         (batch_size, ),
                                         dtype=torch.int32,
                                         device="cpu")

    # Final lengths after generation phase
    final_lens = context_lens_context + num_gen_tokens
    max_seq_len = final_lens.max().item()

    print(f"\n=== Test Config ===")
    print(
        f"  Batch: {batch_size}, Next_N: {next_n}, Heads: {heads}, Head_dim: {head_dim}"
    )
    print(f"  Context lengths: {context_lens_context.tolist()}")
    print(f"  Final lengths: {final_lens.tolist()}")
    print(f"  Max sequence length: {max_seq_len}")

    # Setup: Create cache manager and indexer
    cache_manager, sparse_attn_config = create_dsa_cache_manager(
        batch_size=batch_size,
        head_dim=head_dim,
        tokens_per_block=block_size,
        max_seq_len=max_model_len,
        num_layers=1)
    indexer = create_indexer(sparse_attn_config, layer_idx=layer_idx)

    # Allocate blocks for all sequences (max final length)
    request_ids = list(range(batch_size))
    cache_manager.add_dummy_requests(request_ids=request_ids,
                                     token_nums=final_lens.tolist(),
                                     is_gen=False,
                                     prepare_resource=True)

    # Generate test data with variable lengths
    total_context_tokens = context_lens_context.sum().item()
    q = torch.randn((batch_size, next_n, heads, head_dim),
                    device="cuda",
                    dtype=torch.bfloat16)
    weights = torch.randn((batch_size * next_n, heads),
                          device="cuda",
                          dtype=torch.float32)
    k_context_bf16 = torch.randn((total_context_tokens, head_dim),
                                 device="cuda",
                                 dtype=torch.bfloat16)
    k_gen_bf16 = torch.randn((batch_size * num_gen_tokens, head_dim),
                             device="cuda",
                             dtype=torch.bfloat16)

    # Phase 1: Write context tokens (variable per sequence) as FP8
    print(f"\n=== Phase 1: Context (variable tokens/seq) ===")
    metadata_context = _create_mock_metadata(
        request_ids,
        batch_size,
        num_contexts=batch_size,
        num_generations=0,
        seq_lens=context_lens_context.clone(),
        kv_lens=context_lens_context.clone(),
        num_cached_tokens=[0] * batch_size,
        cache_manager=cache_manager,
        num_ctx_tokens=total_context_tokens,
        num_tokens=total_context_tokens,
        max_draft_tokens=next_n - 1,
    )
    Indexer.prepare(metadata_context)

    k_context_fp8, k_context_scale = fp8_utils.fp8_quantize_1x128_sf_transpose(
        k_context_bf16)

    indexer._update_k_cache(k_context_fp8, k_context_scale, metadata_context)
    print(f"✓ Wrote {total_context_tokens} FP8 context tokens to cache")

    # Phase 2: Write generation tokens (next_n per sequence) as FP8
    # Similar to prepare_resources: add_token() for each new token
    print(f"\n=== Phase 2: Generation ({num_gen_tokens} tokens/seq) ===")
    metadata_gen = _create_mock_metadata(
        request_ids,
        batch_size,
        num_contexts=0,
        num_generations=batch_size,
        seq_lens=torch.tensor([num_gen_tokens] * batch_size,
                              dtype=torch.int32,
                              device='cpu'),
        kv_lens=final_lens.clone(),
        num_cached_tokens=context_lens_context.tolist(),
        cache_manager=cache_manager,
        num_ctx_tokens=0,
        num_tokens=batch_size * num_gen_tokens,
        max_draft_tokens=next_n - 1,
    )
    Indexer.prepare(metadata_gen)

    k_gen_fp8, k_gen_scale = fp8_utils.fp8_quantize_1x128_sf_transpose(
        k_gen_bf16)
    indexer._update_k_cache(k_gen_fp8, k_gen_scale, metadata_gen)
    print(
        f"✓ Wrote {batch_size * num_gen_tokens} FP8 generation tokens to cache")

    # Run kernel: FP8 paged MQA with actual cache
    print(f"\n=== Kernel Execution ===")
    kv_cache_fp8_pool = cache_manager.get_indexer_k_cache_buffers(layer_idx)
    q_fp8 = q.to(torch.float8_e4m3fn)

    if not metadata_gen.use_expanded_buffers_for_mtp:
        q_fp8 = q_fp8
        context_lens = metadata_gen.kv_lens_cuda_runtime[0:batch_size]
        block_table = metadata_gen.indexer_k_cache_block_offsets[0:batch_size]
        if q_fp8.shape[1] == 4:
            scheduler_metadata_buffer = metadata_gen.scheduler_metadata_buffer_mtp3
        else:
            scheduler_metadata_buffer = metadata_gen.scheduler_metadata_buffer
    else:
        q_fp8 = q_fp8.view(-1, 1, *q_fp8.shape[2:])
        num_tokens = batch_size * next_n
        context_lens = metadata_gen.kv_lens_expanded_cuda[:num_tokens]
        block_table = metadata_gen.block_table_expanded[:num_tokens]
        scheduler_metadata_buffer = metadata_gen.scheduler_metadata_buffer_expanded

    logits = fp8_paged_mqa_logits(q_fp8, kv_cache_fp8_pool, weights,
                                  context_lens, block_table,
                                  scheduler_metadata_buffer, max_model_len)
    print(f"✓ Kernel output shape: {logits.shape}")

    # Reference: Reconstruct BF16 cache from original values
    print(f"\n=== Reference Computation ===")
    num_blocks = kv_cache_fp8_pool.shape[0]
    kv_cache_bf16 = torch.zeros((num_blocks, block_size, 1, head_dim),
                                device="cuda",
                                dtype=torch.bfloat16)

    # Populate cache with variable-length sequences
    context_offset = 0
    gen_offset = 0
    for seq_idx in range(batch_size):
        seq_context_len = context_lens_context[seq_idx].item()

        # Write context tokens
        for token_pos in range(seq_context_len):
            block_idx = token_pos // block_size
            pos_in_block = token_pos % block_size
            physical_block_id = metadata_gen.indexer_k_cache_block_offsets[
                seq_idx, block_idx].item()
            if physical_block_id >= 0:
                kv_cache_bf16[physical_block_id, pos_in_block, 0, :] = \
                    k_context_bf16[context_offset + token_pos]

        # Write generation tokens
        for gen_token_idx in range(num_gen_tokens):
            token_pos = seq_context_len + gen_token_idx
            block_idx = token_pos // block_size
            pos_in_block = token_pos % block_size
            physical_block_id = metadata_gen.indexer_k_cache_block_offsets[
                seq_idx, block_idx].item()
            if physical_block_id >= 0:
                kv_cache_bf16[physical_block_id, pos_in_block, 0, :] = \
                    k_gen_bf16[gen_offset + gen_token_idx]

        context_offset += seq_context_len
        gen_offset += num_gen_tokens

    ref_logits = _ref_fp8_paged_mqa_logits(
        q, kv_cache_bf16, weights,
        metadata_gen.kv_lens_cuda_runtime[0:batch_size],
        metadata_gen.indexer_k_cache_block_offsets, max_model_len)
    print(f"✓ Reference output shape: {ref_logits.shape}")

    # Validate: Compare masked outputs (handle variable lengths and next_n)
    print(f"\n=== Validation ===")
    context_lens_cuda = metadata_gen.kv_lens_cuda_runtime  # [batch_size]

    # Expand context lens for each query: each sequence has next_n queries
    # Query at position i (where i = 0..next_n-1) attends to tokens up to (context_len - next_n + i)
    positions = torch.arange(max_model_len,
                             device="cuda").unsqueeze(0)  # [1, max_model_len]

    # For each query, compute its end position
    # Shape: [batch_size * next_n]
    row_indices = torch.arange(batch_size * next_n,
                               device="cuda") // next_n  # Which sequence
    next_n_offset = torch.arange(
        batch_size * next_n,
        device="cuda") % next_n  # Query offset within sequence
    query_end_positions = context_lens_cuda[
        row_indices] - next_n + next_n_offset  # [batch_size * next_n]

    # Create mask: positions <= query_end_position
    # Shape: [batch_size * next_n, max_model_len]
    mask = positions <= query_end_positions.unsqueeze(1)

    diff = _calc_diff(logits.masked_fill(~mask, 0),
                      ref_logits.masked_fill(~mask, 0))

    assert diff < 1e-3, f"Accuracy check failed: {diff=}"
    print(f"✅ Test passed! Accuracy: {diff:.6f} < 1e-3")
    print(
        f"   Total cache tokens: {final_lens.sum().item()}, Avg: {final_lens.float().mean():.1f}"
    )


def test_compute_cu_seqlen_bounds_nocache():
    """Simple test case with 2 sequences."""
    seq_lens = torch.tensor([3, 4], dtype=torch.int32, device="cuda")
    num_contexts = 2
    num_ctx_tokens = 7

    cu_seqlen_ks, cu_seqlen_ke = compute_cu_seqlen_kv_bounds_with_cache(
        seq_lens, num_contexts, num_ctx_tokens, None)

    # Expected results:
    # Seq 0: tokens [0,1,2], KV [0,1,2]
    #   Token 0: [0, 1)
    #   Token 1: [0, 2)
    #   Token 2: [0, 3)
    # Seq 1: tokens [3,4,5,6], KV [3,4,5,6]
    #   Token 3: [3, 4)
    #   Token 4: [3, 5)
    #   Token 5: [3, 6)
    #   Token 6: [3, 7)

    expected_ks = torch.tensor([0, 0, 0, 3, 3, 3, 3],
                               dtype=torch.int32,
                               device="cuda")
    expected_ke = torch.tensor([1, 2, 3, 4, 5, 6, 7],
                               dtype=torch.int32,
                               device="cuda")

    assert torch.equal(cu_seqlen_ks, expected_ks), \
        f"cu_seqlen_ks mismatch:\nGot:      {cu_seqlen_ks.tolist()}\nExpected: {expected_ks.tolist()}"
    assert torch.equal(cu_seqlen_ke, expected_ke), \
        f"cu_seqlen_ke mismatch:\nGot:      {cu_seqlen_ke.tolist()}\nExpected: {expected_ke.tolist()}"


def test_compute_cu_seqlen_bounds_with_cache():
    """
    Test case with 2 sequences using chunked prefill (with cached tokens).

    Scenario:
    - Seq 0: 2 cached tokens, 3 new tokens being added
    - Seq 1: 1 cached token, 4 new tokens being added
    """
    # New tokens being added in this chunk
    seq_lens = torch.tensor([3, 4], dtype=torch.int32, device="cuda")

    # Previously cached tokens
    cached_token_lens = torch.tensor([2, 1], dtype=torch.int32, device="cuda")

    num_contexts = 2
    num_ctx_tokens = 7  # 3 + 4 new tokens total

    cu_seqlen_ks, cu_seqlen_ke = compute_cu_seqlen_kv_bounds_with_cache(
        seq_lens, num_contexts, num_ctx_tokens, cached_token_lens)

    # Expected results:
    #
    # Seq 0: KV has [cached: 0,1] + [new: 2,3,4] = total 5 KV tokens, global range [0:5]
    #   New Q token 0 (local pos 0): attends to cached(2) + self(1) = [0, 3)
    #   New Q token 1 (local pos 1): attends to cached(2) + 0,1(2) = [0, 4)
    #   New Q token 2 (local pos 2): attends to cached(2) + 0,1,2(3) = [0, 5)
    #
    # Seq 1: KV has [cached: 0] + [new: 1,2,3,4] = total 5 KV tokens, global range [5:10]
    #   New Q token 0 (local pos 0, global Q pos 3): attends to cached(1) + self(1) = [5, 7)
    #   New Q token 1 (local pos 1, global Q pos 4): attends to cached(1) + 0,1(2) = [5, 8)
    #   New Q token 2 (local pos 2, global Q pos 5): attends to cached(1) + 0,1,2(3) = [5, 9)
    #   New Q token 3 (local pos 3, global Q pos 6): attends to cached(1) + 0,1,2,3(4) = [5, 10)

    expected_ks = torch.tensor([0, 0, 0, 5, 5, 5, 5],
                               dtype=torch.int32,
                               device="cuda")
    expected_ke = torch.tensor([3, 4, 5, 7, 8, 9, 10],
                               dtype=torch.int32,
                               device="cuda")

    assert torch.equal(cu_seqlen_ks, expected_ks), \
        f"cu_seqlen_ks mismatch:\nGot:      {cu_seqlen_ks.tolist()}\nExpected: {expected_ks.tolist()}"
    assert torch.equal(cu_seqlen_ke, expected_ke), \
        f"cu_seqlen_ke mismatch:\nGot:      {cu_seqlen_ke.tolist()}\nExpected: {expected_ke.tolist()}"


def test_compute_cu_seqlen_bounds_with_cache_edge_cases():
    """Additional edge case tests for chunked prefill with cache."""

    # Case 1: No cached tokens (should behave similarly to nocache version but with global KV offsets)
    seq_lens = torch.tensor([2, 3], dtype=torch.int32, device="cuda")
    cached_token_lens = torch.tensor([0, 0], dtype=torch.int32, device="cuda")
    num_contexts = 2
    num_ctx_tokens = 5

    cu_seqlen_ks, cu_seqlen_ke = compute_cu_seqlen_kv_bounds_with_cache(
        seq_lens, num_contexts, num_ctx_tokens, cached_token_lens)

    # Seq 0: KV [0:2], Q tokens attend to [0,1), [0,2)
    # Seq 1: KV [2:5], Q tokens attend to [2,3), [2,4), [2,5)
    expected_ks = torch.tensor([0, 0, 2, 2, 2],
                               dtype=torch.int32,
                               device="cuda")
    expected_ke = torch.tensor([1, 2, 3, 4, 5],
                               dtype=torch.int32,
                               device="cuda")

    assert torch.equal(cu_seqlen_ks, expected_ks), \
        f"Case 1 - cu_seqlen_ks mismatch:\nGot: {cu_seqlen_ks.tolist()}\nExpected: {expected_ks.tolist()}"
    assert torch.equal(cu_seqlen_ke, expected_ke), \
        f"Case 1 - cu_seqlen_ke mismatch:\nGot: {cu_seqlen_ke.tolist()}\nExpected: {expected_ke.tolist()}"

    # Case 2: Single new token per sequence
    seq_lens = torch.tensor([1, 1], dtype=torch.int32, device="cuda")
    cached_token_lens = torch.tensor([5, 3], dtype=torch.int32, device="cuda")
    num_contexts = 2
    num_ctx_tokens = 2

    cu_seqlen_ks, cu_seqlen_ke = compute_cu_seqlen_kv_bounds_with_cache(
        seq_lens, num_contexts, num_ctx_tokens, cached_token_lens)

    # Seq 0: 5 cached + 1 new = 6 total KV, global [0:6]
    #   New Q token 0: attends to cached(5) + self(1) = [0, 6)
    # Seq 1: 3 cached + 1 new = 4 total KV, global [6:10]
    #   New Q token 0: attends to cached(3) + self(1) = [6, 10)
    expected_ks = torch.tensor([0, 6], dtype=torch.int32, device="cuda")
    expected_ke = torch.tensor([6, 10], dtype=torch.int32, device="cuda")

    assert torch.equal(cu_seqlen_ks, expected_ks), \
        f"Case 2 - cu_seqlen_ks mismatch:\nGot: {cu_seqlen_ks.tolist()}\nExpected: {expected_ks.tolist()}"
    assert torch.equal(cu_seqlen_ke, expected_ke), \
        f"Case 2 - cu_seqlen_ke mismatch:\nGot: {cu_seqlen_ke.tolist()}\nExpected: {expected_ke.tolist()}"

    # Case 3: Different cached amounts across sequences
    seq_lens = torch.tensor([2, 1, 3], dtype=torch.int32, device="cuda")
    cached_token_lens = torch.tensor([10, 0, 5],
                                     dtype=torch.int32,
                                     device="cuda")
    num_contexts = 3
    num_ctx_tokens = 6

    cu_seqlen_ks, cu_seqlen_ke = compute_cu_seqlen_kv_bounds_with_cache(
        seq_lens, num_contexts, num_ctx_tokens, cached_token_lens)

    # Seq 0: 10 cached + 2 new = 12 total KV, global [0:12]
    #   Q token 0 (local 0): cached(10) + self(1) = [0, 11)
    #   Q token 1 (local 1): cached(10) + 0,1(2) = [0, 12)
    # Seq 1: 0 cached + 1 new = 1 total KV, global [12:13]
    #   Q token 0 (local 0): cached(0) + self(1) = [12, 13)
    # Seq 2: 5 cached + 3 new = 8 total KV, global [13:21]
    #   Q token 0 (local 0): cached(5) + self(1) = [13, 19)
    #   Q token 1 (local 1): cached(5) + 0,1(2) = [13, 20)
    #   Q token 2 (local 2): cached(5) + 0,1,2(3) = [13, 21)
    expected_ks = torch.tensor([0, 0, 12, 13, 13, 13],
                               dtype=torch.int32,
                               device="cuda")
    expected_ke = torch.tensor([11, 12, 13, 19, 20, 21],
                               dtype=torch.int32,
                               device="cuda")

    assert torch.equal(cu_seqlen_ks, expected_ks), \
        f"Case 3 - cu_seqlen_ks mismatch:\nGot: {cu_seqlen_ks.tolist()}\nExpected: {expected_ks.tolist()}"
    assert torch.equal(cu_seqlen_ke, expected_ke), \
        f"Case 3 - cu_seqlen_ke mismatch:\nGot: {cu_seqlen_ke.tolist()}\nExpected: {expected_ke.tolist()}"

    # Case 4: All tokens cached, single new token
    seq_lens = torch.tensor([1], dtype=torch.int32, device="cuda")
    cached_token_lens = torch.tensor([100], dtype=torch.int32, device="cuda")
    num_contexts = 1
    num_ctx_tokens = 1

    cu_seqlen_ks, cu_seqlen_ke = compute_cu_seqlen_kv_bounds_with_cache(
        seq_lens, num_contexts, num_ctx_tokens, cached_token_lens)

    # Seq 0: 100 cached + 1 new = 101 total KV
    #   Q token 0: attends to all = [0, 101)
    expected_ks = torch.tensor([0], dtype=torch.int32, device="cuda")
    expected_ke = torch.tensor([101], dtype=torch.int32, device="cuda")

    assert torch.equal(cu_seqlen_ks, expected_ks), \
        f"Case 4 - cu_seqlen_ks mismatch:\nGot: {cu_seqlen_ks.tolist()}\nExpected: {expected_ks.tolist()}"
    assert torch.equal(cu_seqlen_ke, expected_ke), \
        f"Case 4 - cu_seqlen_ke mismatch:\nGot: {cu_seqlen_ke.tolist()}\nExpected: {expected_ke.tolist()}"


def test_compute_cu_seqlen_bounds_with_cache_properties():
    """Property-based tests to verify invariants of the with_cache function."""

    for trial in range(10):
        num_contexts = random.randint(1, 5)
        seq_lens = torch.randint(1,
                                 10, (num_contexts, ),
                                 dtype=torch.int32,
                                 device="cuda")
        cached_token_lens = torch.randint(0,
                                          20, (num_contexts, ),
                                          dtype=torch.int32,
                                          device="cuda")
        num_ctx_tokens = seq_lens.sum().item()

        cu_seqlen_ks, cu_seqlen_ke = compute_cu_seqlen_kv_bounds_with_cache(
            seq_lens, num_contexts, num_ctx_tokens, cached_token_lens)

        # Property 1: Output length matches num_ctx_tokens
        assert cu_seqlen_ks.shape[0] == num_ctx_tokens, \
            f"Trial {trial}: ks length mismatch: {cu_seqlen_ks.shape[0]} != {num_ctx_tokens}"
        assert cu_seqlen_ke.shape[0] == num_ctx_tokens, \
            f"Trial {trial}: ke length mismatch: {cu_seqlen_ke.shape[0]} != {num_ctx_tokens}"

        # Property 2: End > Start for all tokens
        assert torch.all(cu_seqlen_ke > cu_seqlen_ks), \
            f"Trial {trial}: End indices must be greater than start indices"

        # Property 3: End indices are strictly increasing within each sequence
        offset = 0
        for seq_idx in range(num_contexts):
            seq_len = seq_lens[seq_idx].item()
            seq_ke = cu_seqlen_ke[offset:offset + seq_len]

            # Each token should attend to one more KV position than previous
            if seq_len > 1:
                diffs = seq_ke[1:] - seq_ke[:-1]
                assert torch.all(diffs == 1), \
                    f"Trial {trial}: Seq {seq_idx} - consecutive ke diffs should be 1, got {diffs.tolist()}"

            offset += seq_len

        # Property 4: First token in each sequence has correct attention window size
        offset = 0
        for seq_idx in range(num_contexts):
            seq_len = seq_lens[seq_idx].item()
            cached_len = cached_token_lens[seq_idx].item()

            # First new Q token should attend to all cached + itself
            expected_window_size = cached_len + 1
            actual_window_size = (cu_seqlen_ke[offset] -
                                  cu_seqlen_ks[offset]).item()

            assert actual_window_size == expected_window_size, \
                f"Trial {trial}: Seq {seq_idx} first token: expected window {expected_window_size}, got {actual_window_size}"

            offset += seq_len

        # Property 5: KV sequences are non-overlapping and contiguous in global space
        kv_lens = cached_token_lens + seq_lens
        expected_total_kv = kv_lens.sum().item()

        # Last token of last sequence should end at total KV count
        assert cu_seqlen_ke[-1].item() == expected_total_kv, \
            f"Trial {trial}: Last ke should equal total KV count: {cu_seqlen_ke[-1].item()} != {expected_total_kv}"


@pytest.mark.parametrize(
    "max_chunk_size,seq_lens,start_idx,expected_specs",
    [
        # Small requests - single chunk
        (1000, [200, 300, 400], 0, [[(0, 0, 200, 0), (1, 0, 300, 200),
                                     (2, 0, 400, 500)]]),
        # Small requests - multiple chunks
        (1000, [400, 500, 300, 600], 0, [[(0, 0, 400, 0), (1, 0, 500, 400)],
                                         [(2, 0, 300, 900),
                                          (3, 0, 600, 1200)]]),
        # Large request - intra-request chunking
        (1000, [2500], 0, [[(0, 0, 1000, 0)], [(0, 1000, 2000, 0)],
                           [(0, 2000, 2500, 0)]]),
        # Mixed: small + large + small
        (1000, [400, 2500, 300], 0, [[
            (0, 0, 400, 0)
        ], [(1, 0, 1000, 400)], [(1, 1000, 2000, 400)], [(1, 2000, 2500, 400)],
                                     [(2, 0, 300, 2900)]]),
        # Edge case: exact chunk size
        (1000, [1000], 0, [[(0, 0, 1000, 0)]]),
        # Edge case: non-zero start index
        (1000, [100, 200, 300], 1, [[(1, 0, 200, 100), (2, 0, 300, 300)]]),
    ],
    ids=[
        "small_single", "small_multi", "large_chunked", "mixed", "exact_size",
        "non_zero_start"
    ])
def test_split_prefill_chunks(max_chunk_size, seq_lens, start_idx,
                              expected_specs):
    """
    Test split_prefill_chunks covering:
    - Request-level chunking (small requests)
    - Intra-request Q-block chunking (large requests)
    - Mixed scenarios
    - Edge cases
    """
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)
    chunk_groups = split_prefill_chunks(seq_lens_tensor,
                                        max_chunk_size,
                                        start_idx=start_idx)

    assert len(chunk_groups) == len(expected_specs), \
        f"Expected {len(expected_specs)} chunks, got {len(chunk_groups)}"

    for i, expected in enumerate(expected_specs):
        assert chunk_groups[i] == expected, \
            f"Chunk {i} mismatch:\nGot:      {chunk_groups[i]}\nExpected: {expected}"

    print(f"✅ test_split_prefill_chunks passed")


@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@skip_pre_hopper
@pytest.mark.parametrize(
    "chunk_size,seq_lens_list,chunking_type",
    [
        # Request-level chunking: small requests that fit individually but need chunking as batch
        (512, None, "request_level"),  # Random seq lens, 5 requests
        (1024, None, "request_level"),  # Random seq lens, 4 requests
        (2048, None, "request_level"),  # Random seq lens, 4 requests

        # Two-level chunking: intra-request Q-block splitting for large requests
        (1024, [512, 2500, 800], "two_level"),  # Middle request needs Q-blocks
        (1500, [3200, 1200], "two_level"),  # First request needs 3 Q-blocks
        (1000, [400, 600, 2200, 500
                ], "two_level"),  # Mixed: request 2 needs Q-blocks
    ],
)
def test_indexer_chunked_prefill(chunk_size, seq_lens_list, chunking_type):
    """
    Tests for indexer chunked prefill:
    1. Request-level chunking: Multiple small requests packed together
    2. Two-level chunking: Large requests split into Q-blocks + request-level chunking

    Validates that chunked and non-chunked (single-pass) execution produce identical topk indices.
    Tests:
    - Chunk metadata creation and usage
    - K-cache gathering for chunks
    - Variable-length sequence handling
    - Q-block splitting for large requests
    - Correct attention scope per Q-block
    """
    torch.manual_seed(42)
    random.seed(42)

    # Test parameters
    heads, head_dim = 32, 128
    block_size = 64
    index_topk = 2048
    max_model_len = 16384
    layer_idx = 0

    # Generate sequence lengths based on chunking type
    if chunking_type == "request_level":
        # Generate variable seq lens for request-level chunking
        # Each request fits in chunk, but batch exceeds chunk_size
        batch_size = 5 if chunk_size == 512 else 4
        seq_lens_list = []
        for _ in range(batch_size):
            seq_len = random.randint(int(0.3 * chunk_size),
                                     int(0.9 * chunk_size))
            seq_lens_list.append(seq_len)
    else:
        # Two-level chunking: use provided seq_lens with large requests
        batch_size = len(seq_lens_list)

    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device='cpu')
    total_tokens = seq_lens.sum().item()
    max_seq_len = seq_lens.max().item()

    print(f"\n=== Test Config: {chunking_type} ===")
    print(f"  Batch: {batch_size}, Chunk size: {chunk_size}")
    print(f"  Sequence lengths: {seq_lens_list}")
    print(f"  Total tokens: {total_tokens}, Max seq len: {max_seq_len}")

    # Identify large requests for two-level chunking
    if chunking_type == "two_level":
        large_requests = [(i, seq_len)
                          for i, seq_len in enumerate(seq_lens_list)
                          if seq_len > chunk_size]
        if large_requests:
            print(f"  Large requests (Q-block splitting):")
            for req_idx, seq_len in large_requests:
                num_q_blocks = (seq_len + chunk_size - 1) // chunk_size
                print(
                    f"    Request {req_idx}: {seq_len} tokens → {num_q_blocks} Q-blocks"
                )

    # Create cache manager and indexer
    cache_manager, sparse_attn_config = create_dsa_cache_manager(
        batch_size=batch_size,
        head_dim=head_dim,
        tokens_per_block=block_size,
        max_seq_len=max_model_len,
        num_layers=1)
    sparse_attn_config.index_topk = index_topk
    indexer = create_indexer(sparse_attn_config, layer_idx=layer_idx)

    # Allocate blocks for all sequences
    request_ids = list(range(batch_size))
    cache_manager.add_dummy_requests(request_ids=request_ids,
                                     token_nums=seq_lens_list,
                                     is_gen=False,
                                     prepare_resource=True)

    # Generate test data
    q = torch.randn((total_tokens, heads, head_dim),
                    device="cuda",
                    dtype=torch.bfloat16)
    k = torch.randn((total_tokens, head_dim),
                    device="cuda",
                    dtype=torch.bfloat16)
    weights = torch.randn((total_tokens, heads),
                          device="cuda",
                          dtype=torch.float32)
    hidden_states = torch.randn((total_tokens, 4096),
                                device="cuda",
                                dtype=torch.bfloat16)

    # Quantize inputs
    q_fp8 = q.to(torch.float8_e4m3fn)
    k_fp8, k_scale = fp8_utils.fp8_quantize_1x128_sf_transpose(k)

    # ========== Test Path 1: Chunked Prefill ==========
    print(f"\n=== Chunked Path ===")

    metadata_chunked = _create_mock_metadata(
        request_ids,
        batch_size,
        num_contexts=batch_size,
        num_generations=0,
        seq_lens=seq_lens.clone(),
        kv_lens=seq_lens.clone(),
        num_cached_tokens=[0] * batch_size,
        cache_manager=cache_manager,
        num_ctx_tokens=total_tokens,
        num_tokens=total_tokens,
        indexer_max_chunk_size=chunk_size,
    )

    Indexer.prepare(metadata_chunked)

    assert metadata_chunked.indexer_prefill_chunks is not None
    num_chunks = len(metadata_chunked.indexer_prefill_chunks)
    print(f"✓ Created {num_chunks} chunks")

    # Print chunk details
    for i, chunk in enumerate(metadata_chunked.indexer_prefill_chunks):
        num_q = chunk.token_end - chunk.token_start
        num_k = chunk.k_token_end - chunk.k_token_start
        print(
            f"  Chunk {i}: Q[{chunk.token_start}:{chunk.token_end}] ({num_q} tokens), "
            f"K[{chunk.k_token_start}:{chunk.k_token_end}] ({num_k} tokens)")

    indexer._update_k_cache(k_fp8, k_scale, metadata_chunked)
    topk_indices_chunked = indexer.sparse_attn_indexer(metadata_chunked,
                                                       hidden_states, q_fp8,
                                                       k_fp8, k_scale, weights)

    print(f"✓ Chunked execution completed, shape: {topk_indices_chunked.shape}")

    # ========== Test Path 2: Non-chunked Baseline ==========
    print(f"\n=== Non-chunked Baseline ===")

    metadata_baseline = _create_mock_metadata(
        request_ids,
        batch_size,
        num_contexts=batch_size,
        num_generations=0,
        seq_lens=seq_lens.clone(),
        kv_lens=seq_lens.clone(),
        num_cached_tokens=[0] * batch_size,
        cache_manager=cache_manager,
        num_ctx_tokens=total_tokens,
        num_tokens=total_tokens,
        indexer_max_chunk_size=max_model_len,
    )

    Indexer.prepare(metadata_baseline)

    if metadata_baseline.indexer_prefill_chunks is not None:
        num_baseline_chunks = len(metadata_baseline.indexer_prefill_chunks)
        print(
            f"✓ Created {num_baseline_chunks} chunk(s) (effectively non-chunked)"
        )

    indexer._update_k_cache(k_fp8, k_scale, metadata_baseline)
    topk_indices_baseline = indexer.sparse_attn_indexer(metadata_baseline,
                                                        hidden_states, q_fp8,
                                                        k_fp8, k_scale, weights)

    print(
        f"✓ Non-chunked execution completed, shape: {topk_indices_baseline.shape}"
    )

    # ========== Validation ==========
    print(f"\n=== Validation ===")

    # Use Jaccard similarity to handle ties (multiple indices with same value)
    num_exact_matches = 0
    num_high_similarity = 0
    total_similarity = 0.0

    for token_idx in range(total_tokens):
        chunked_indices = topk_indices_chunked[token_idx]
        baseline_indices = topk_indices_baseline[token_idx]

        # Filter out -1 (invalid) indices
        chunked_valid = chunked_indices[chunked_indices != -1]
        baseline_valid = baseline_indices[baseline_indices != -1]

        # Check if exactly the same
        if torch.equal(chunked_valid, baseline_valid):
            num_exact_matches += 1
            total_similarity += 1.0
            continue

        # Calculate set-based similarity (Jaccard index) to handle ties
        if chunked_valid.shape[0] > 0 or baseline_valid.shape[0] > 0:
            chunked_set = set(chunked_valid.cpu().tolist())
            baseline_set = set(baseline_valid.cpu().tolist())

            intersection = len(chunked_set & baseline_set)
            union = len(chunked_set | baseline_set)
            similarity = intersection / union if union > 0 else 0.0
            total_similarity += similarity

            if similarity >= 0.95:
                num_high_similarity += 1

    # Calculate statistics
    avg_similarity = total_similarity / total_tokens
    exact_match_ratio = num_exact_matches / total_tokens
    high_similarity_ratio = (num_exact_matches +
                             num_high_similarity) / total_tokens

    print(f"  Results:")
    print(
        f"    Exact matches: {num_exact_matches}/{total_tokens} ({exact_match_ratio:.1%})"
    )
    print(f"    High similarity (>=95%): {num_high_similarity} additional")
    print(f"    Overall high similarity ratio: {high_similarity_ratio:.1%}")
    print(f"    Average Jaccard similarity: {avg_similarity:.4f}")

    # Detailed mismatch analysis for low similarity cases
    if avg_similarity < 0.95:
        low_sim_count = 0
        cumulative_lens = torch.cat([torch.tensor([0]), seq_lens.cumsum(0)])

        for token_idx in range(total_tokens):
            chunked_indices = topk_indices_chunked[token_idx]
            baseline_indices = topk_indices_baseline[token_idx]

            chunked_valid = chunked_indices[chunked_indices != -1]
            baseline_valid = baseline_indices[baseline_indices != -1]

            if not torch.equal(chunked_valid, baseline_valid):
                chunked_set = set(chunked_valid.cpu().tolist())
                baseline_set = set(baseline_valid.cpu().tolist())
                intersection = len(chunked_set & baseline_set)
                union = len(chunked_set | baseline_set)
                similarity = intersection / union if union > 0 else 0.0

                if similarity < 0.9:
                    if low_sim_count < 5:  # Show first 5 low similarity cases
                        # Find which request this token belongs to
                        req_idx = (cumulative_lens
                                   <= token_idx).sum().item() - 1
                        local_token_idx = token_idx - cumulative_lens[
                            req_idx].item()

                        print(
                            f"    Token {token_idx} (req {req_idx}, local pos {local_token_idx}): "
                            f"similarity {similarity:.3f}")
                        print(
                            f"      Chunked size: {len(chunked_set)}, Baseline size: {len(baseline_set)}"
                        )
                        print(
                            f"      Intersection: {intersection}, Union: {union}"
                        )
                    low_sim_count += 1

        if low_sim_count > 5:
            print(
                f"    ... and {low_sim_count - 5} more tokens with low similarity"
            )

    # Use Jaccard similarity threshold instead of exact match
    assert avg_similarity >= 0.9, \
        f"Chunked and non-chunked results differ significantly: avg similarity {avg_similarity:.4f} < 0.9"

    print(
        f"\n✅ Test passed! {chunking_type} chunking produces highly similar results"
    )
    print(f"   Config: chunk_size={chunk_size}, num_chunks={num_chunks}, "
          f"batch={batch_size}, seq_lens={seq_lens_list}")


@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@skip_pre_hopper
@pytest.mark.parametrize("batch_size", [1, 16, 64])
@pytest.mark.parametrize("next_n", [1, 2, 4])
@pytest.mark.parametrize("index_topk", [2048])
@pytest.mark.parametrize("seq_len_range", [(2048, 8192), (512, 1024)])
def test_indexer_decode_custom_vs_fallback(batch_size, next_n, index_topk,
                                           seq_len_range):
    """
    Test that use_custom_topk=True and use_custom_topk=False produce identical results
    in the decode phase of sparse_attn_indexer.

    This test validates:
    1. Custom CUDA top-k kernel (indexer_topk_decode) correctness
    2. Consistency with PyTorch fallback implementation
    3. Handling of decode scenarios with next_n > 1 (speculative decoding)
    4. Proper masking and handling of variable-length sequences

    Test scenarios:
    - Different batch sizes
    - Different next_n values (1, 2, 4 for speculative decode)
    - Variable sequence lengths (90% >= 2048 to test realistic long sequences)
    - Short sequences (512, 1024) to test the indexer skip functionality
    """
    torch.manual_seed(42)
    random.seed(42)

    # Test parameters
    heads, head_dim = 32, 128
    block_size = 64
    max_model_len = 16384
    layer_idx = 0
    min_seq_len, max_seq_len = seq_len_range
    enable_indexer_skip = max_seq_len <= 2048

    # Generate KV cache lengths
    if enable_indexer_skip:
        kv_lens = torch.randint(min_seq_len,
                                max_seq_len, (batch_size, ),
                                dtype=torch.int32)
    else:
        # (90% >= 2048 to test realistic scenarios)
        kv_lens = torch.zeros(batch_size, dtype=torch.int32)
        is_long = torch.rand(batch_size) < 0.9

        num_long = is_long.sum().item()
        if num_long > 0:
            long_min = max(2048, min_seq_len)
            long_max = max(long_min + 1, max_seq_len)
            kv_lens[is_long] = torch.randint(long_min,
                                             long_max, (num_long, ),
                                             dtype=torch.int32)

        num_short = (~is_long).sum().item()
        if num_short > 0:
            short_max = min(2048, max_seq_len)
            if short_max > min_seq_len:
                kv_lens[~is_long] = torch.randint(min_seq_len,
                                                  short_max, (num_short, ),
                                                  dtype=torch.int32)
            else:
                kv_lens[~is_long] = torch.randint(max(2048, min_seq_len),
                                                  max(2049, max_seq_len),
                                                  (num_short, ),
                                                  dtype=torch.int32)

    seq_lens = torch.full((batch_size, ), next_n, dtype=torch.int32)
    num_gen_tokens = batch_size * next_n
    num_cached_tokens = kv_lens.tolist()

    # Create cache manager and indexer
    cache_manager, sparse_attn_config = create_dsa_cache_manager(
        batch_size=batch_size,
        head_dim=head_dim,
        tokens_per_block=block_size,
        max_seq_len=max_model_len,
        num_layers=1)
    sparse_attn_config.index_topk = index_topk
    indexer = create_indexer(sparse_attn_config, layer_idx=layer_idx)

    # Allocate blocks for all sequences (including historical + new tokens)
    request_ids = list(range(batch_size))
    final_lens = kv_lens + next_n  # Historical + new decode tokens
    cache_manager.add_dummy_requests(request_ids=request_ids,
                                     token_nums=final_lens.tolist(),
                                     is_gen=False,
                                     prepare_resource=True)

    # Populate KV cache with historical context
    total_context_tokens = kv_lens.sum().item()
    k_context_bf16 = torch.randn((total_context_tokens, head_dim),
                                 device="cuda",
                                 dtype=torch.bfloat16)
    k_context_fp8, k_context_scale = fp8_utils.fp8_quantize_1x128_sf_transpose(
        k_context_bf16)

    metadata_context = _create_mock_metadata(
        request_ids=request_ids,
        batch_size=batch_size,
        num_contexts=batch_size,
        num_generations=0,
        seq_lens=kv_lens.clone(),
        kv_lens=kv_lens.clone(),
        num_cached_tokens=[0] * batch_size,
        cache_manager=cache_manager,
        num_ctx_tokens=total_context_tokens,
        num_tokens=total_context_tokens,
        max_draft_tokens=next_n - 1,
    )
    Indexer.prepare(metadata_context)
    indexer._update_k_cache(k_context_fp8, k_context_scale, metadata_context)

    # Generate decode phase test data
    q = torch.randn((num_gen_tokens, heads, head_dim),
                    device="cuda",
                    dtype=torch.bfloat16)
    k_gen_bf16 = torch.randn((num_gen_tokens, head_dim),
                             device="cuda",
                             dtype=torch.bfloat16)
    weights = torch.randn((num_gen_tokens, heads),
                          device="cuda",
                          dtype=torch.float32)
    hidden_states = torch.randn((num_gen_tokens, 4096),
                                device="cuda",
                                dtype=torch.bfloat16)

    q_fp8 = q.to(torch.float8_e4m3fn)
    k_fp8, k_scale = fp8_utils.fp8_quantize_1x128_sf_transpose(k_gen_bf16)

    metadata_gen_write = _create_mock_metadata(
        request_ids=request_ids,
        batch_size=batch_size,
        num_contexts=0,
        num_generations=batch_size,
        seq_lens=seq_lens.clone(),
        kv_lens=final_lens.clone(),
        num_cached_tokens=num_cached_tokens,
        cache_manager=cache_manager,
        num_ctx_tokens=0,
        num_tokens=num_gen_tokens,
        max_draft_tokens=next_n - 1,
    )
    Indexer.prepare(metadata_gen_write)
    indexer._update_k_cache(k_fp8, k_scale, metadata_gen_write)

    # Test with custom CUDA kernel
    metadata_custom = _create_mock_metadata(request_ids,
                                            batch_size,
                                            0,
                                            batch_size,
                                            seq_lens.clone(),
                                            final_lens.clone(),
                                            num_cached_tokens,
                                            cache_manager,
                                            0,
                                            num_gen_tokens,
                                            max_model_len,
                                            max_draft_tokens=next_n - 1)

    Indexer.prepare(metadata_custom)
    indexer._update_k_cache(k_fp8, k_scale, metadata_custom)

    try:
        topk_indices_custom = indexer.sparse_attn_indexer(metadata_custom,
                                                          hidden_states,
                                                          q_fp8,
                                                          k_fp8,
                                                          k_scale,
                                                          weights,
                                                          use_custom_topk=True)
    except Exception as e:
        pytest.skip(f"Custom topk not available: {e}")

    # Test with PyTorch fallback
    metadata_fallback = _create_mock_metadata(request_ids,
                                              batch_size,
                                              0,
                                              batch_size,
                                              seq_lens.clone(),
                                              final_lens.clone(),
                                              num_cached_tokens,
                                              cache_manager,
                                              0,
                                              num_gen_tokens,
                                              max_model_len,
                                              max_draft_tokens=next_n - 1)

    Indexer.prepare(metadata_fallback)
    indexer._update_k_cache(k_fp8, k_scale, metadata_fallback)
    topk_indices_fallback = indexer.sparse_attn_indexer(metadata_fallback,
                                                        hidden_states,
                                                        q_fp8,
                                                        k_fp8,
                                                        k_scale,
                                                        weights,
                                                        use_custom_topk=False)

    # Test with indexer skip enabled
    if enable_indexer_skip:
        metadata_skip = _create_mock_metadata(request_ids,
                                              batch_size,
                                              0,
                                              batch_size,
                                              seq_lens.clone(),
                                              final_lens.clone(),
                                              num_cached_tokens,
                                              cache_manager,
                                              0,
                                              num_gen_tokens,
                                              max_model_len,
                                              max_draft_tokens=next_n - 1,
                                              enable_indexer_skip=True)

        Indexer.prepare(metadata_skip)
        indexer._update_k_cache(k_fp8, k_scale, metadata_skip)

        try:
            topk_indices_skip = indexer.sparse_attn_indexer(
                metadata_skip,
                hidden_states,
                q_fp8,
                k_fp8,
                k_scale,
                weights,
                use_custom_topk=True)
        except Exception as e:
            raise RuntimeError(f"Error when testing indexer skip: {e}")

    # Validation
    ## Custom vs fallback
    num_ctx_tokens = 0
    custom_decode = topk_indices_custom[num_ctx_tokens:num_ctx_tokens +
                                        num_gen_tokens, :]
    fallback_decode = topk_indices_fallback[num_ctx_tokens:num_ctx_tokens +
                                            num_gen_tokens, :]
    num_exact_matches, total_similarity, _ = validate_topk_indices(
        custom_decode, fallback_decode, num_gen_tokens)
    avg_similarity = total_similarity / num_gen_tokens
    assert avg_similarity >= 0.95, \
        f"Decode custom vs fallback differ: avg similarity {avg_similarity:.4f} < 0.95"
    ## Custom vs skip
    if enable_indexer_skip:
        skip_decode = topk_indices_skip[num_ctx_tokens:num_ctx_tokens +
                                        num_gen_tokens, :]
        num_exact_matches, total_similarity, _ = validate_topk_indices(
            custom_decode, skip_decode, num_gen_tokens)
        avg_similarity = total_similarity / num_gen_tokens
        assert avg_similarity >= 0.95, \
            f"Decode custom vs skip differ: avg similarity {avg_similarity:.4f} < 0.95"


@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@skip_pre_hopper
@pytest.mark.parametrize("batch_size", [4, 16])
@pytest.mark.parametrize("index_topk", [2048])
@pytest.mark.parametrize("chunk_size", [1024, 2048])
def test_indexer_prefill_chunked_custom_vs_fallback(batch_size, index_topk,
                                                    chunk_size):
    """
    Test chunked prefill: use_custom_topk=True vs use_custom_topk=False
    with metadata.indexer_prefill_chunks != None.

    This test validates:
    1. Custom CUDA top-k kernel (indexer_topk_prefill) correctness in chunked mode
    2. Consistency with PyTorch fallback implementation
    3. Proper handling of multiple chunks
    4. Correct masking and local index computation per chunk
    """
    torch.manual_seed(42)
    random.seed(42)

    # Test parameters
    heads, head_dim = 32, 128
    block_size = 64
    max_model_len = 16384
    layer_idx = 0

    # Generate variable sequence lengths to trigger chunking
    min_seq_len = chunk_size // 2
    max_seq_len = chunk_size * 3
    seq_lens = torch.randint(min_seq_len,
                             max_seq_len, (batch_size, ),
                             dtype=torch.int32)
    total_tokens = seq_lens.sum().item()

    # Create cache manager and indexer
    cache_manager, sparse_attn_config = create_dsa_cache_manager(
        batch_size=batch_size,
        head_dim=head_dim,
        tokens_per_block=block_size,
        max_seq_len=max_model_len,
        num_layers=1)
    sparse_attn_config.index_topk = index_topk
    indexer = create_indexer(sparse_attn_config, layer_idx=layer_idx)

    # Allocate cache blocks
    request_ids = list(range(batch_size))
    cache_manager.add_dummy_requests(request_ids=request_ids,
                                     token_nums=seq_lens.tolist(),
                                     is_gen=False,
                                     prepare_resource=True)

    # Generate test data
    q = torch.randn((total_tokens, heads, head_dim),
                    device="cuda",
                    dtype=torch.bfloat16)
    k = torch.randn((total_tokens, head_dim),
                    device="cuda",
                    dtype=torch.bfloat16)
    weights = torch.randn((total_tokens, heads),
                          device="cuda",
                          dtype=torch.float32)
    hidden_states = torch.randn((total_tokens, 4096),
                                device="cuda",
                                dtype=torch.bfloat16)

    q_fp8 = q.to(torch.float8_e4m3fn)
    k_fp8, k_scale = fp8_utils.fp8_quantize_1x128_sf_transpose(k)

    # Test with custom CUDA kernel
    metadata_custom = _create_mock_metadata(request_ids, batch_size,
                                            batch_size, 0, seq_lens.clone(),
                                            seq_lens.clone(), [0] * batch_size,
                                            cache_manager, total_tokens,
                                            total_tokens, chunk_size)

    Indexer.prepare(metadata_custom)
    indexer._update_k_cache(k_fp8, k_scale, metadata_custom)

    assert metadata_custom.indexer_prefill_chunks is not None

    try:
        topk_indices_custom = indexer.sparse_attn_indexer(metadata_custom,
                                                          hidden_states,
                                                          q_fp8,
                                                          k_fp8,
                                                          k_scale,
                                                          weights,
                                                          use_custom_topk=True)
    except Exception as e:
        pytest.skip(f"Custom topk not available: {e}")

    # Test with PyTorch fallback
    metadata_fallback = _create_mock_metadata(request_ids, batch_size,
                                              batch_size, 0, seq_lens.clone(),
                                              seq_lens.clone(),
                                              [0] * batch_size, cache_manager,
                                              total_tokens, total_tokens,
                                              chunk_size)

    Indexer.prepare(metadata_fallback)
    indexer._update_k_cache(k_fp8, k_scale, metadata_fallback)
    topk_indices_fallback = indexer.sparse_attn_indexer(metadata_fallback,
                                                        hidden_states,
                                                        q_fp8,
                                                        k_fp8,
                                                        k_scale,
                                                        weights,
                                                        use_custom_topk=False)

    # Validation
    num_exact_matches, total_similarity, _ = validate_topk_indices(
        topk_indices_custom, topk_indices_fallback, total_tokens)
    avg_similarity = total_similarity / total_tokens
    assert avg_similarity >= 0.95, \
        f"Chunked prefill differ: avg similarity {avg_similarity:.4f} < 0.95"


@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@skip_pre_hopper
@pytest.mark.parametrize("batch_size", [4, 16])
@pytest.mark.parametrize("index_topk", [2048])
@pytest.mark.parametrize("seq_len_range", [(1, 512)])
def test_indexer_prefill_single_pass_custom_vs_fallback(batch_size, index_topk,
                                                        seq_len_range):
    """
    Test single-pass prefill: use_custom_topk=True vs use_custom_topk=False
    with metadata.indexer_prefill_chunks == None (else branch).
    """
    torch.manual_seed(42)
    random.seed(42)

    heads, head_dim = 32, 128
    block_size = 64
    max_model_len = 16384
    layer_idx = 0
    min_seq_len, max_seq_len = seq_len_range

    # Generate variable context lengths per sequence
    seq_lens = torch.randint(min_seq_len,
                             max_seq_len, (batch_size, ),
                             dtype=torch.int32)
    total_tokens = seq_lens.sum().item()

    # Create cache manager and indexer
    cache_manager, sparse_attn_config = create_dsa_cache_manager(
        batch_size=batch_size,
        head_dim=head_dim,
        tokens_per_block=block_size,
        max_seq_len=max_model_len,
        num_layers=1)
    sparse_attn_config.index_topk = index_topk
    indexer = create_indexer(sparse_attn_config, layer_idx=layer_idx)

    request_ids = list(range(batch_size))
    cache_manager.add_dummy_requests(request_ids=request_ids,
                                     token_nums=seq_lens.tolist(),
                                     is_gen=False,
                                     prepare_resource=True)

    # Generate test data
    q = torch.randn((total_tokens, heads, head_dim),
                    device="cuda",
                    dtype=torch.bfloat16)
    k = torch.randn((total_tokens, head_dim),
                    device="cuda",
                    dtype=torch.bfloat16)
    weights = torch.randn((total_tokens, heads),
                          device="cuda",
                          dtype=torch.float32)
    hidden_states = torch.randn((total_tokens, 4096),
                                device="cuda",
                                dtype=torch.bfloat16)

    q_fp8 = q.to(torch.float8_e4m3fn)
    k_fp8, k_scale = fp8_utils.fp8_quantize_1x128_sf_transpose(k)

    # Test with custom CUDA kernel
    metadata_custom = _create_mock_metadata(request_ids, batch_size,
                                            batch_size, 0, seq_lens.clone(),
                                            seq_lens.clone(), [0] * batch_size,
                                            cache_manager, total_tokens,
                                            total_tokens, max_model_len)

    Indexer.prepare(metadata_custom)
    indexer._update_k_cache(k_fp8, k_scale, metadata_custom)
    # Force single-pass path by setting indexer_prefill_chunks to None
    metadata_custom.indexer_prefill_chunks = None

    try:
        topk_indices_custom = indexer.sparse_attn_indexer(metadata_custom,
                                                          hidden_states,
                                                          q_fp8,
                                                          k_fp8,
                                                          k_scale,
                                                          weights,
                                                          use_custom_topk=True)
    except Exception as e:
        pytest.skip(f"Custom topk not available: {e}")

    # Test with PyTorch fallback
    metadata_fallback = _create_mock_metadata(request_ids, batch_size,
                                              batch_size, 0, seq_lens.clone(),
                                              seq_lens.clone(),
                                              [0] * batch_size, cache_manager,
                                              total_tokens, total_tokens,
                                              max_model_len)

    Indexer.prepare(metadata_fallback)
    indexer._update_k_cache(k_fp8, k_scale, metadata_fallback)
    # Force single-pass path by setting indexer_prefill_chunks to None
    metadata_fallback.indexer_prefill_chunks = None

    topk_indices_fallback = indexer.sparse_attn_indexer(metadata_fallback,
                                                        hidden_states,
                                                        q_fp8,
                                                        k_fp8,
                                                        k_scale,
                                                        weights,
                                                        use_custom_topk=False)

    # Test with indexer skip enabled
    metadata_skip = _create_mock_metadata(request_ids,
                                          batch_size,
                                          batch_size,
                                          0,
                                          seq_lens.clone(),
                                          seq_lens.clone(), [0] * batch_size,
                                          cache_manager,
                                          total_tokens,
                                          total_tokens,
                                          max_model_len,
                                          enable_indexer_skip=True)
    Indexer.prepare(metadata_skip)
    indexer._update_k_cache(k_fp8, k_scale, metadata_skip)
    metadata_skip.indexer_prefill_chunks = None

    try:
        topk_indices_skip = indexer.sparse_attn_indexer(metadata_skip,
                                                        hidden_states,
                                                        q_fp8,
                                                        k_fp8,
                                                        k_scale,
                                                        weights,
                                                        use_custom_topk=True)
    except Exception as e:
        raise RuntimeError(f"Indexer skip not available: {e}")

    # Validation
    ## Custom vs fallback
    num_exact_matches, total_similarity, _ = validate_topk_indices(
        topk_indices_custom, topk_indices_fallback, total_tokens)
    avg_similarity = total_similarity / total_tokens
    assert avg_similarity >= 0.95, \
        f"Single-pass prefill differ: avg similarity {avg_similarity:.4f} < 0.95"
    ## Custom vs skip
    num_exact_matches, total_similarity, _ = validate_topk_indices(
        topk_indices_custom, topk_indices_skip, total_tokens)
    avg_similarity = total_similarity / total_tokens
    assert avg_similarity >= 0.95, \
        f"Single-pass prefill differ: avg similarity {avg_similarity:.4f} < 0.95"


@skip_pre_hopper
@pytest.mark.parametrize("enable_indexer_skip", [True, False])
def test_indexer_topk_multi_request_with_different_cache(enable_indexer_skip):
    """
    Test that custom topk kernel handles multi-request batches with different cached amounts.
    """
    torch.manual_seed(42)

    # Parameters matching your problematic case
    batch_size = 2
    heads, head_dim = 64, 128
    block_size = 64
    index_topk = 2048
    max_model_len = 10240
    layer_idx = 0

    # Critical: different cached amounts
    seq_lens = [256, 237]  # NEW tokens
    if enable_indexer_skip:
        cached_tokens = [256, 584]  # Req0: no cache, Req1: short cache
    else:
        cached_tokens = [0, 3584]  # Req0: no cache, Req1: large cache
    total_kv_lens = [seq_lens[i] + cached_tokens[i] for i in range(batch_size)]
    total_tokens = sum(seq_lens)

    print(f"\n=== Test: Multi-request with different cache ===")
    print(
        f"  Req0: {cached_tokens[0]} cached + {seq_lens[0]} new = {total_kv_lens[0]} total"
    )
    print(
        f"  Req1: {cached_tokens[1]} cached + {seq_lens[1]} new = {total_kv_lens[1]} total"
    )

    # Create cache manager
    cache_manager, sparse_attn_config = create_dsa_cache_manager(
        batch_size=batch_size,
        head_dim=head_dim,
        tokens_per_block=block_size,
        max_seq_len=max_model_len,
        num_layers=1)
    sparse_attn_config.index_topk = index_topk
    indexer = create_indexer(sparse_attn_config, layer_idx=layer_idx)

    # Allocate blocks
    request_ids = [0, 1]
    cache_manager.add_dummy_requests(request_ids,
                                     total_kv_lens,
                                     is_gen=False,
                                     prepare_resource=True)

    # Generate test data
    q = torch.randn((total_tokens, heads, head_dim),
                    device="cuda",
                    dtype=torch.bfloat16)
    k = torch.randn((total_tokens, head_dim),
                    device="cuda",
                    dtype=torch.bfloat16)
    weights = torch.randn((total_tokens, heads),
                          device="cuda",
                          dtype=torch.float32)
    hidden_states = torch.randn((total_tokens, 4096),
                                device="cuda",
                                dtype=torch.bfloat16)

    q_fp8 = q.to(torch.float8_e4m3fn)
    k_fp8, k_scale = fp8_utils.fp8_quantize_1x128_sf_transpose(k)

    # Create metadata with chunked prefill enabled
    metadata = _create_mock_metadata(request_ids,
                                     batch_size,
                                     batch_size,
                                     0,
                                     torch.tensor(seq_lens, dtype=torch.int32),
                                     torch.tensor(total_kv_lens,
                                                  dtype=torch.int32),
                                     cached_tokens,
                                     cache_manager,
                                     total_tokens,
                                     total_tokens,
                                     indexer_max_chunk_size=32768,
                                     enable_context_mla_with_cached_kv=True)

    Indexer.prepare(metadata)
    indexer._update_k_cache(k_fp8, k_scale, metadata)

    # Test custom kernel
    topk_custom = indexer.sparse_attn_indexer(metadata,
                                              hidden_states,
                                              q_fp8,
                                              k_fp8,
                                              k_scale,
                                              weights,
                                              use_custom_topk=True)

    # Test fallback
    topk_fallback = indexer.sparse_attn_indexer(metadata,
                                                hidden_states,
                                                q_fp8,
                                                k_fp8,
                                                k_scale,
                                                weights,
                                                use_custom_topk=False)

    # Test with indexer skip enabled
    if enable_indexer_skip:
        metadata_skip = _create_mock_metadata(
            request_ids,
            batch_size,
            batch_size,
            0,
            torch.tensor(seq_lens, dtype=torch.int32),
            torch.tensor(total_kv_lens, dtype=torch.int32),
            cached_tokens,
            cache_manager,
            total_tokens,
            total_tokens,
            indexer_max_chunk_size=32768,
            enable_context_mla_with_cached_kv=True,
            enable_indexer_skip=True)
        Indexer.prepare(metadata_skip)
        indexer._update_k_cache(k_fp8, k_scale, metadata_skip)
        topk_indices_skip = indexer.sparse_attn_indexer(metadata_skip,
                                                        hidden_states,
                                                        q_fp8,
                                                        k_fp8,
                                                        k_scale,
                                                        weights,
                                                        use_custom_topk=True)

    # Validate: custom and fallback should match
    print(f"\n=== Validation ===")

    print(f"Checking for invalid negative indices:")
    for tok_id in [0, 255, 256, 492]:  # First/last of each request
        num_valid_custom = (topk_custom[tok_id] >= 0).sum().item()
        num_valid_fallback = (topk_fallback[tok_id] >= 0).sum().item()
        min_val_custom = topk_custom[tok_id].min().item()
        min_val_fallback = topk_fallback[tok_id].min().item()

        has_invalid = min_val_custom < -1
        if has_invalid or num_valid_custom != num_valid_fallback:
            print(
                f"  Token {tok_id}: custom={num_valid_custom}, fallback={num_valid_fallback}"
            )
            print(
                f"    Custom min={min_val_custom}, Fallback min={min_val_fallback}"
            )
            if has_invalid:
                print(
                    f"    ⚠️ INVALID: Custom has negative indices < -1 (kernel bug!)"
                )

    # Check tokens with large windows (>= 2048) should have exactly 2048 valid indices
    print(f"\n=== Check: Large windows must have 2048 valid ===")
    from tensorrt_llm._torch.attention_backend.sparse.dsa import \
        compute_cu_seqlen_kv_bounds_with_cache
    host_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device='cpu')
    host_cached = torch.tensor(cached_tokens, dtype=torch.int32, device='cpu')
    cu_ks, cu_ke = compute_cu_seqlen_kv_bounds_with_cache(
        host_seq_lens, batch_size, total_tokens, host_cached)

    for tok_id in range(total_tokens):
        window_size = (cu_ke[tok_id] - cu_ks[tok_id]).item()
        if window_size >= index_topk:
            num_valid = (topk_custom[tok_id] >= 0).sum().item()
            num_valid_fallback = (topk_fallback[tok_id] >= 0).sum().item()
            assert num_valid_fallback == index_topk, \
                f"[Fallback] Token {tok_id}: window={window_size}, but only {num_valid_fallback}/{index_topk} valid indices"
            assert num_valid == index_topk, \
                f"[Custom Topk] Token {tok_id}: window={window_size}, but only {num_valid}/{index_topk} valid indices"

    print(f"  ✓ All large-window tokens have {index_topk} valid indices")

    # Validation
    num_exact_matches, total_similarity, min_similarity = validate_topk_indices(
        topk_custom, topk_fallback, total_tokens)
    avg_similarity = total_similarity / total_tokens
    print(f"  Exact matches: {num_exact_matches}/{total_tokens}")
    print(
        f"  Similarity - Min: {min_similarity:.4f}, Avg: {avg_similarity:.4f}")

    assert avg_similarity >= 0.95, \
        f"Custom vs fallback differ: avg similarity {avg_similarity:.4f} < 0.95"

    if enable_indexer_skip:
        num_exact_matches, total_similarity, min_similarity = validate_topk_indices(
            topk_custom, topk_indices_skip, total_tokens)
        avg_similarity = total_similarity / total_tokens
        print(f"  Exact matches: {num_exact_matches}/{total_tokens}")
        print(
            f"  Similarity - Min: {min_similarity:.4f}, Avg: {avg_similarity:.4f}"
        )
        assert avg_similarity >= 0.95, \
            f"Custom vs indexer skip differ: avg similarity {avg_similarity:.4f} < 0.95"
