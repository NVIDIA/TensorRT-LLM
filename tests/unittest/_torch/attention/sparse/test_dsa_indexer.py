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
from utils.util import check_accuracy, getSMVersion

from tensorrt_llm import deep_gemm
from tensorrt_llm._torch.attention_backend.interface import (
    PositionalEmbeddingParams, RopeParams)
from tensorrt_llm._torch.attention_backend.sparse.dsa import (
    DSACacheManager, Indexer, compute_cu_seqlen_kv_bounds_nocache,
    split_prefill_chunks)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.bindings.internal.batch_manager import \
    CacheType as CacheTypeCpp
from tensorrt_llm.deep_gemm import fp8_paged_mqa_logits
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.mapping import Mapping


def has_deep_gemm():
    try:
        return deep_gemm is not None
    except:
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
    # Note: max_attention_window expects list[int] (one per layer)
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=False,
        max_tokens=max_seq_len * batch_size,
        max_attention_window=[max_seq_len] *
        num_layers,  # List of max window per layer
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
@pytest.mark.skipif(getSMVersion() < 90,
                    reason="fp8_mqa_logits is only supported in SM90 and SM100")
def test_deepgemm_fp8_mqa_logits_basic():
    """
    Basic test for deepgemm.fp8_mqa_logits kernel.
    Tests the disable_cp path with simple validation.
    """
    torch.manual_seed(0)

    num_heads, head_dim = 32, 128
    seq_len = 512
    seq_len_kv = 1024
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
    ke = torch.arange(seq_len, dtype=torch.int, device="cuda") + (
        seq_len_kv - seq_len) + 1  # +1 for exclusive end

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


def _create_mock_metadata(request_ids, batch_size, num_contexts,
                          num_generations, seq_lens, kv_lens, num_cached_tokens,
                          cache_manager, num_ctx_tokens, num_tokens):
    """Helper to create mock metadata for testing."""

    class MockKVCacheParams:

        def __init__(self):
            self.num_cached_tokens_per_seq = num_cached_tokens

    class MockMetadata:

        def __init__(self):
            self.num_sms = deep_gemm.get_num_sms()
            self.request_ids = request_ids
            self.num_contexts = num_contexts
            self.num_generations = num_generations
            self.seq_lens = seq_lens
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
            self.kv_cache_manager.copy_indexer_k_cache_offsets(
                self.request_ids, self.host_indexer_k_cache_block_offsets)
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
            self.num_ctx_tokens = num_ctx_tokens
            self.num_tokens = num_tokens
            self.num_ctx_cached_tokens_for_spec_dec = self.num_ctx_tokens - self.num_contexts
            torch.cumsum(kv_lens[:num_contexts],
                         dim=0,
                         dtype=torch.int64,
                         out=self.host_ctx_kv_indptr[1:num_contexts + 1])
            torch.cumsum(kv_lens[num_contexts:num_contexts + num_generations],
                         dim=0,
                         dtype=torch.int64,
                         out=self.host_gen_kv_indptr[1:num_generations + 1])

            # Add indexer-specific attributes
            self.indexer_max_chunk_size = 8194
            self.indexer_prefill_chunks = None

    return MockMetadata()


@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@pytest.mark.skipif(
    getSMVersion() < 90,
    reason="fp8_paged_mqa_logits is only supported in SM90 and SM100")
@pytest.mark.parametrize("batch_size,next_n", [(4, 1), (2, 2)])
def test_indexer_paged_kv_cache_block_management(batch_size, next_n):
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
    )
    Indexer.prepare(metadata_context)

    k_context_fp8, k_context_scale = torch.ops.trtllm.fp8_quantize_1x128(
        k_context_bf16)
    k_context_scale = k_context_scale.contiguous().transpose(0, 1)
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
    )
    Indexer.prepare(metadata_gen)

    k_gen_fp8, k_gen_scale = torch.ops.trtllm.fp8_quantize_1x128(k_gen_bf16)
    k_gen_scale = k_gen_scale.contiguous().transpose(0, 1)
    indexer._update_k_cache(k_gen_fp8, k_gen_scale, metadata_gen)
    print(
        f"✓ Wrote {batch_size * num_gen_tokens} FP8 generation tokens to cache")

    # Run kernel: FP8 paged MQA with actual cache
    print(f"\n=== Kernel Execution ===")
    kv_cache_fp8_pool = cache_manager.get_indexer_k_cache_buffers(layer_idx)
    q_fp8 = q.to(torch.float8_e4m3fn)

    logits = fp8_paged_mqa_logits(
        q_fp8, kv_cache_fp8_pool, weights,
        metadata_gen.kv_lens_cuda_runtime[0:batch_size],
        metadata_gen.indexer_k_cache_block_offsets,
        metadata_gen.scheduler_metadata_buffer, max_model_len)
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
        final_lens[seq_idx].item()

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


@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@pytest.mark.skipif(
    getSMVersion() < 90,
    reason="fp8_paged_mqa_logits is only supported in SM90 and SM100")
def test_decode_phase_masking():
    """
    Test the decode phase masking logic in sparse_attn_indexer.

    This test validates:
    1. Position tensor creation with actual_max_len instead of max_seq_len
    2. Proper masking based on context lengths
    3. TopK selection respects masked positions
    """
    torch.manual_seed(42)

    # Setup parameters
    batch_size = 3
    next_n = 2  # e.g., speculative decoding with 2 draft tokens
    num_gen_tokens = batch_size * next_n
    index_topk = 32

    # Simulate different context lengths per sequence
    # Seq 0: 10 tokens context, Seq 1: 15 tokens, Seq 2: 8 tokens
    context_lens = torch.tensor([10, 15, 8], dtype=torch.int32, device="cuda")
    actual_max_len = context_lens.max().item()  # 15

    # Simulate logits from fp8_paged_mqa_logits: [B*N, max_model_len]
    # We'll use actual_max_len for efficiency
    logits_decode = torch.randn(num_gen_tokens,
                                actual_max_len,
                                dtype=torch.float32,
                                device="cuda")

    # Apply the masking logic from the decode branch
    positions = torch.arange(actual_max_len, device="cuda").unsqueeze(0).expand(
        num_gen_tokens, -1)

    row_indices = torch.arange(num_gen_tokens, device="cuda") // next_n
    next_n_offset = torch.arange(num_gen_tokens, device="cuda") % next_n

    # For each token, compute valid context end position
    # Token i in sequence j can attend to positions [0, context_len[j] - next_n + offset[i]]
    index_end_pos = (context_lens[row_indices] - next_n +
                     next_n_offset).unsqueeze(1)

    # Validate index_end_pos shape and values
    assert index_end_pos.shape == (num_gen_tokens, 1)
    expected_end_pos = torch.tensor(
        [
            [8],  # Seq 0, token 0: 10 - 2 + 0 = 8
            [9],  # Seq 0, token 1: 10 - 2 + 1 = 9
            [13],  # Seq 1, token 0: 15 - 2 + 0 = 13
            [14],  # Seq 1, token 1: 15 - 2 + 1 = 14
            [6],  # Seq 2, token 0: 8 - 2 + 0 = 6
            [7],  # Seq 2, token 1: 8 - 2 + 1 = 7
        ],
        dtype=torch.int32,
        device="cuda")
    assert torch.equal(index_end_pos, expected_end_pos), \
        f"index_end_pos mismatch:\nGot:\n{index_end_pos}\nExpected:\n{expected_end_pos}"

    # Apply mask: positions <= index_end_pos
    mask = positions <= index_end_pos
    logits_masked = logits_decode.masked_fill(~mask, float('-inf'))

    # Verify masking correctness
    # For token 0 of seq 0 (index_end_pos=8), positions 0-8 should be valid, 9-14 masked
    assert not torch.isinf(
        logits_masked[0, :9]).any(), "Valid positions should not be -inf"
    assert torch.isinf(
        logits_masked[0, 9:]).all(), "Invalid positions should be -inf"

    # For token 1 of seq 1 (index_end_pos=14), positions 0-14 should be valid
    assert not torch.isinf(
        logits_masked[3, :15]).any(), "All positions should be valid"

    # For token 0 of seq 2 (index_end_pos=6), positions 0-6 valid, 7-14 masked
    assert not torch.isinf(
        logits_masked[4, :7]).any(), "Valid positions should not be -inf"
    assert torch.isinf(
        logits_masked[4, 7:]).all(), "Invalid positions should be -inf"

    # Test topk selection
    topk_k = min(index_topk, actual_max_len)
    topk_values, topk_indices = logits_masked.topk(topk_k, dim=-1)

    # Verify topk indices are within valid range OR correspond to masked (-inf) positions
    for i in range(num_gen_tokens):
        valid_end = index_end_pos[i, 0].item()
        num_valid_positions = valid_end + 1  # Positions [0, valid_end] inclusive

        # Check that non-inf topk values correspond to valid indices
        valid_topk_mask = ~torch.isinf(topk_values[i])
        valid_topk_indices = topk_indices[i][valid_topk_mask]

        # All non-masked topk indices should be within valid range
        assert (valid_topk_indices <= valid_end).all(), \
            f"Token {i}: valid topk indices {valid_topk_indices} exceed valid range [0, {valid_end}]"

        # If we have fewer valid positions than topk_k, expect some -inf values
        if num_valid_positions < topk_k:
            num_inf = torch.isinf(topk_values[i]).sum().item()
            expected_inf = topk_k - num_valid_positions
            assert num_inf == expected_inf, \
                f"Token {i}: expected {expected_inf} masked values, got {num_inf}"

    # Edge case: if context is shorter than topk_k, handle gracefully
    # Simulate sequence with very short context (4 valid positions, request 10)
    short_context_lens = torch.tensor([5], dtype=torch.int32, device="cuda")
    short_logits = torch.randn(1, 10, dtype=torch.float32, device="cuda")

    short_positions = torch.arange(10, device="cuda").unsqueeze(0)
    short_index_end_pos = torch.tensor([[3]], dtype=torch.int32,
                                       device="cuda")  # Only 4 valid positions
    short_mask = short_positions <= short_index_end_pos
    short_logits_masked = short_logits.masked_fill(~short_mask, float('-inf'))

    # Request topk_k=10 but only 4 valid positions
    short_topk_values, short_topk_indices = short_logits_masked.topk(10, dim=-1)

    # First 4 should be non-inf with valid indices, rest should be -inf
    assert not torch.isinf(
        short_topk_values[0, :4]).any(), "First 4 should be valid values"
    assert torch.isinf(short_topk_values[0, 4:]).all(), "Last 6 should be -inf"
    assert (short_topk_indices[0, :4]
            <= 3).all(), "Valid topk indices should be within range [0, 3]"


@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@pytest.mark.skipif(getSMVersion() < 90, reason="FP8 operations require SM90+")
def test_fp8_scale_roundtrip():
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
    k_fp8, k_scale = torch.ops.trtllm.fp8_quantize_1x128(k_original)
    k_scale = k_scale.contiguous().transpose(0, 1)

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


def test_split_prefill_chunks_small_requests():
    """Test request-level chunking: multiple small requests packed together."""
    max_chunk_size = 1000

    # Case 1: All requests fit in one chunk
    seq_lens = torch.tensor([200, 300, 400], dtype=torch.int32)
    chunk_groups = split_prefill_chunks(seq_lens, max_chunk_size, start_idx=0)

    # Should get 1 chunk group with 3 specs (one per request)
    assert len(
        chunk_groups) == 1, f"Expected 1 chunk group, got {len(chunk_groups)}"
    assert len(
        chunk_groups[0]
    ) == 3, f"Expected 3 requests in group, got {len(chunk_groups[0])}"

    # Verify chunk specs
    expected_specs = [
        (0, 0, 200, 0),  # req 0: tokens [0:200], cumulative start = 0
        (1, 0, 300, 200),  # req 1: tokens [0:300], cumulative start = 200
        (2, 0, 400, 500),  # req 2: tokens [0:400], cumulative start = 500
    ]
    assert chunk_groups[0] == expected_specs, \
        f"Chunk specs mismatch:\nGot:      {chunk_groups[0]}\nExpected: {expected_specs}"

    # Case 2: Requests need multiple chunks
    seq_lens = torch.tensor([400, 500, 300, 600], dtype=torch.int32)
    chunk_groups = split_prefill_chunks(seq_lens, max_chunk_size, start_idx=0)

    # First 2 requests fit in chunk 0 (400+500=900 < 1000)
    # Requests 2 and 3 fit in chunk 1 (300+600=900 < 1000)
    assert len(
        chunk_groups) == 2, f"Expected 2 chunk groups, got {len(chunk_groups)}"

    expected_group_0 = [(0, 0, 400, 0), (1, 0, 500, 400)]
    expected_group_1 = [(2, 0, 300, 900), (3, 0, 600, 1200)]

    assert chunk_groups[0] == expected_group_0
    assert chunk_groups[1] == expected_group_1

    print("✅ test_split_prefill_chunks_small_requests passed")


def test_split_prefill_chunks_large_request():
    """Test intra-request chunking: large request split into Q-blocks."""
    max_chunk_size = 1000

    # Case 1: Single large request
    seq_lens = torch.tensor([2500], dtype=torch.int32)
    chunk_groups = split_prefill_chunks(seq_lens, max_chunk_size, start_idx=0)

    # Should get 3 chunk groups (one per Q-block)
    assert len(
        chunk_groups) == 3, f"Expected 3 chunk groups, got {len(chunk_groups)}"

    # Each Q-block is a separate chunk group with 1 spec
    assert len(chunk_groups[0]) == 1
    assert chunk_groups[0][0] == (0, 0, 1000, 0)  # Q-block 0: tokens [0:1000]

    assert len(chunk_groups[1]) == 1
    assert chunk_groups[1][0] == (0, 1000, 2000, 0
                                  )  # Q-block 1: tokens [1000:2000]

    assert len(chunk_groups[2]) == 1
    assert chunk_groups[2][0] == (0, 2000, 2500, 0
                                  )  # Q-block 2: tokens [2000:2500]

    # Case 2: Exactly 2x max_chunk_size
    seq_lens = torch.tensor([2000], dtype=torch.int32)
    chunk_groups = split_prefill_chunks(seq_lens, max_chunk_size, start_idx=0)

    assert len(chunk_groups) == 2  # 2 Q-blocks
    assert len(chunk_groups[0]) == 1
    assert chunk_groups[0][0] == (0, 0, 1000, 0)
    assert len(chunk_groups[1]) == 1
    assert chunk_groups[1][0] == (0, 1000, 2000, 0)

    # Case 3: Slightly over max_chunk_size
    seq_lens = torch.tensor([1001], dtype=torch.int32)
    chunk_groups = split_prefill_chunks(seq_lens, max_chunk_size, start_idx=0)

    assert len(chunk_groups) == 2  # 2 Q-blocks
    assert len(chunk_groups[0]) == 1
    assert chunk_groups[0][0] == (0, 0, 1000, 0)
    assert len(chunk_groups[1]) == 1
    assert chunk_groups[1][0] == (0, 1000, 1001, 0)

    print("✅ test_split_prefill_chunks_large_request passed")


def test_split_prefill_chunks_mixed():
    """Test mixed scenario: small requests + large request."""
    max_chunk_size = 1000

    # Small request, large request, small request
    seq_lens = torch.tensor([400, 2500, 300], dtype=torch.int32)
    chunk_groups = split_prefill_chunks(seq_lens, max_chunk_size, start_idx=0)

    # Expected:
    # - Group 0: [req 0] (small request)
    # - Group 1: [req 1, Q-block 0] (large request, first Q-block)
    # - Group 2: [req 1, Q-block 1] (large request, second Q-block)
    # - Group 3: [req 1, Q-block 2] (large request, third Q-block)
    # - Group 4: [req 2] (small request)

    assert len(
        chunk_groups) == 5, f"Expected 5 chunk groups, got {len(chunk_groups)}"

    # Group 0: small request 0
    assert len(chunk_groups[0]) == 1
    assert chunk_groups[0][0] == (0, 0, 400, 0)

    # Group 1-3: large request 1 split into 3 Q-blocks (each is separate group)
    assert len(chunk_groups[1]) == 1
    assert chunk_groups[1][0] == (1, 0, 1000, 400)

    assert len(chunk_groups[2]) == 1
    assert chunk_groups[2][0] == (1, 1000, 2000, 400)

    assert len(chunk_groups[3]) == 1
    assert chunk_groups[3][0] == (1, 2000, 2500, 400)

    # Group 4: small request 2
    assert len(chunk_groups[4]) == 1
    assert chunk_groups[4][0] == (2, 0, 300, 2900)

    print("✅ test_split_prefill_chunks_mixed passed")


def test_split_prefill_chunks_edge_cases():
    """Test edge cases for split_prefill_chunks."""
    max_chunk_size = 1000

    # Case 1: Single request exactly at max_chunk_size
    seq_lens = torch.tensor([1000], dtype=torch.int32)
    chunk_groups = split_prefill_chunks(seq_lens, max_chunk_size, start_idx=0)

    assert len(chunk_groups) == 1
    assert len(chunk_groups[0]) == 1
    assert chunk_groups[0][0] == (0, 0, 1000, 0)

    # Case 2: Multiple requests summing to exactly max_chunk_size
    seq_lens = torch.tensor([300, 400, 300], dtype=torch.int32)
    chunk_groups = split_prefill_chunks(seq_lens, max_chunk_size, start_idx=0)

    assert len(chunk_groups) == 1
    assert len(chunk_groups[0]) == 3

    # Case 3: Very large request (many Q-blocks)
    seq_lens = torch.tensor([5500], dtype=torch.int32)
    chunk_groups = split_prefill_chunks(seq_lens, max_chunk_size, start_idx=0)

    assert len(chunk_groups) == 6  # 5500 / 1000 = 5.5 -> 6 Q-blocks
    # Each Q-block is a separate chunk group
    for i in range(6):
        assert len(chunk_groups[i]) == 1

    # Verify last block has remaining tokens
    last_block = chunk_groups[-1][0]
    assert last_block == (0, 5000, 5500, 0)

    # Case 4: Small request that doesn't fit with previous
    seq_lens = torch.tensor([900, 200], dtype=torch.int32)
    chunk_groups = split_prefill_chunks(seq_lens, max_chunk_size, start_idx=0)

    # 900 + 200 = 1100 > 1000, so split into 2 groups
    assert len(chunk_groups) == 2
    assert len(chunk_groups[0]) == 1
    assert chunk_groups[0][0] == (0, 0, 900, 0)
    assert len(chunk_groups[1]) == 1
    assert chunk_groups[1][0] == (1, 0, 200, 900)

    # Case 5: Start from non-zero index
    seq_lens = torch.tensor([100, 200, 300, 400], dtype=torch.int32)
    chunk_groups = split_prefill_chunks(seq_lens, max_chunk_size, start_idx=1)

    # Should start from request 1 (200 tokens)
    assert len(chunk_groups) == 1
    assert len(chunk_groups[0]) == 3
    assert chunk_groups[0][0][0] == 1  # First request should be index 1
    assert chunk_groups[0][0][
        3] == 100  # Cumulative start includes skipped request 0

    print("✅ test_split_prefill_chunks_edge_cases passed")


def test_split_prefill_chunks_realistic_32k():
    """Test realistic scenario with 32k max_chunk_size."""
    max_chunk_size = 32768

    # Realistic batch: mix of short, medium, and long requests
    seq_lens = torch.tensor(
        [
            1024,  # Short
            8192,  # Medium
            16384,  # Medium-long
            40000,  # Long (exceeds limit, needs Q-blocks)
            2048,  # Short
            65536,  # Very long (needs 2 Q-blocks)
        ],
        dtype=torch.int32)

    chunk_groups = split_prefill_chunks(seq_lens, max_chunk_size, start_idx=0)

    # Expected grouping:
    # Group 0: [req0(1024), req1(8192), req2(16384)] = 25600 < 32768
    # Group 1: [req3, Q-block0(32768)]
    # Group 2: [req3, Q-block1(7232)]
    # Group 3: [req4(2048)]
    # Group 4: [req5, Q-block0(32768)]
    # Group 5: [req5, Q-block1(32768)]

    print(f"Number of chunk groups: {len(chunk_groups)}")
    for i, group in enumerate(chunk_groups):
        total_q_tokens = sum(spec[2] - spec[1] for spec in group)
        print(f"  Group {i}: {len(group)} spec(s), {total_q_tokens} Q tokens")
        for spec in group:
            req_idx, token_start, token_end, cum_start = spec
            print(
                f"    Req {req_idx}: Q[{token_start}:{token_end}], cum_start={cum_start}"
            )

    # Verify first group packs small requests
    assert len(chunk_groups) > 0
    assert len(chunk_groups[0]) == 3
    total_tokens_group0 = sum(spec[2] - spec[1] for spec in chunk_groups[0])
    assert total_tokens_group0 == 25600

    # Find group with req 3 (40k tokens)
    req3_groups = [g for g in chunk_groups if any(spec[0] == 3 for spec in g)]
    assert len(req3_groups) == 2  # Should be split into 2 Q-blocks

    # Find groups with req 5 (65536 tokens)
    req5_groups = [g for g in chunk_groups if any(spec[0] == 5 for spec in g)]
    assert len(req5_groups) == 2  # Should be split into 2 Q-blocks

    print("✅ test_split_prefill_chunks_realistic_32k passed")


def test_compute_cu_seqlen_bounds_nocache():
    """Simple test case with 2 sequences."""
    seq_lens = torch.tensor([3, 4], dtype=torch.int32, device="cuda")
    num_contexts = 2
    num_ctx_tokens = 7

    cu_seqlen_ks, cu_seqlen_ke = compute_cu_seqlen_kv_bounds_nocache(
        seq_lens, num_contexts, num_ctx_tokens)

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


@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@pytest.mark.skipif(getSMVersion() < 90, reason="FP8 operations require SM90+")
@pytest.mark.parametrize(
    "chunk_size,batch_size",
    [
        (
            512, 5
        ),  # Small chunks, multiple chunks - 5 requests to ensure > 512 tokens total
        (1024, 4),  # Medium chunks - 4 requests to ensure > 1024 tokens total
        (2048, 4),  # Large chunks - 4 requests to ensure > 2048 tokens total
    ])
def test_indexer_chunked_prefill(chunk_size, batch_size):
    """
    Test chunked prefill for indexer by directly calling sparse_attn_indexer.

    Compares results with indexer_prefill_chunks set vs None (fallback path).

    This test validates:
    1. Chunked path and non-chunked path produce identical topk indices
    2. _gather_k_cache_for_chunk works correctly
    3. Chunk metadata is correctly built and used
    4. Variable-length sequences work across chunks
    """
    torch.manual_seed(42)
    random.seed(42)

    # Test parameters
    heads, head_dim = 32, 128
    block_size = 64
    index_topk = 2048
    max_model_len = 8192
    layer_idx = 0

    # Generate variable sequence lengths
    # Each request must be <= chunk_size (request-boundary chunking)
    # But total batch should exceed chunk_size to force multiple chunks
    seq_lens_list = []
    for _ in range(batch_size):
        # Random length between 0.3x and 0.9x chunk_size
        # This ensures: individual requests fit, but batch needs chunking
        seq_len = random.randint(int(0.3 * chunk_size), int(0.9 * chunk_size))
        seq_lens_list.append(seq_len)

    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device='cpu')
    total_tokens = seq_lens.sum().item()
    max_seq_len = seq_lens.max().item()

    print(f"\n=== Test Config ===")
    print(f"  Batch: {batch_size}, Chunk size: {chunk_size}")
    print(f"  Sequence lengths: {seq_lens_list}")
    print(f"  Total tokens: {total_tokens}, Max seq len: {max_seq_len}")
    print(f"  Expected chunks: ~{total_tokens // chunk_size + 1}")

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
    k_fp8, k_scale = torch.ops.trtllm.fp8_quantize_1x128(k)
    k_scale = k_scale.contiguous().transpose(0, 1)

    # ========== Test 1: Chunked Prefill ==========
    print(f"\n=== Test 1: Chunked Prefill (via sparse_attn_indexer) ===")

    # Create fresh metadata for chunked test
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
    )

    # Enable chunking by setting indexer_max_chunk_size
    metadata_chunked.indexer_max_chunk_size = chunk_size

    # Prepare metadata - this will create chunks
    Indexer.prepare(metadata_chunked)

    # Verify chunks were created
    assert metadata_chunked.indexer_prefill_chunks is not None, \
        "Chunked prefill should create chunk metadata"
    num_chunks = len(metadata_chunked.indexer_prefill_chunks)
    print(
        f"✓ Created {num_chunks} chunks (total_tokens={total_tokens}, chunk_size={chunk_size})"
    )

    # Print chunk info
    for i, chunk in enumerate(metadata_chunked.indexer_prefill_chunks):
        num_tokens_in_chunk = chunk.token_end - chunk.token_start
        print(
            f"  Chunk {i}: tokens [{chunk.token_start}:{chunk.token_end}] ({num_tokens_in_chunk} tokens)"
        )

    # Call sparse_attn_indexer directly (uses chunked path)
    topk_indices_chunked = indexer.sparse_attn_indexer(
        metadata_chunked,
        hidden_states,
        q_fp8,
        k,
        weights,
    )

    print(f"✓ Chunked sparse_attn_indexer completed")
    print(f"  Output shape: {topk_indices_chunked.shape}")

    # ========== Test 2: Non-chunked Baseline ==========
    print(f"\n=== Test 2: Non-chunked Baseline (via sparse_attn_indexer) ===")

    # Create fresh metadata for non-chunked test
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
    )
    # disable chunking by using the default indexer_max_chunk_size

    # Prepare metadata - this will create a single chunk containing all requests
    Indexer.prepare(metadata_baseline)

    # Verify only one chunk created (all requests in single chunk = effectively no chunking)
    if metadata_baseline.indexer_prefill_chunks is not None:
        num_baseline_chunks = len(metadata_baseline.indexer_prefill_chunks)
        print(f"✓ Created {num_baseline_chunks} chunk(s)")
        if num_baseline_chunks == 1:
            print(
                f"  Single chunk [0:{total_tokens}] - effectively non-chunked")
        # Note: Even with large chunk size, we create at least 1 chunk
        # The "fallback path" in sparse_attn_indexer is when indexer_prefill_chunks is None
        # But after calling prepare(), it's always a list (possibly with 1 chunk)
    else:
        print(f"✓ No chunks created - using fallback path")

    # Call sparse_attn_indexer directly (uses fallback path)
    topk_indices_baseline = indexer.sparse_attn_indexer(
        metadata_baseline,
        hidden_states,
        q_fp8,
        k,
        weights,
    )

    print(f"✓ Non-chunked sparse_attn_indexer completed")
    print(f"  Output shape: {topk_indices_baseline.shape}")

    # ========== Validation ==========
    print(f"\n=== Validation ===")

    # Compare topk indices
    match_count = (topk_indices_chunked == topk_indices_baseline).sum().item()
    total_elements = total_tokens * index_topk
    match_ratio = match_count / total_elements

    print(f"  Match ratio: {match_ratio:.4f} ({match_count}/{total_elements})")

    # Check per-token accuracy
    per_token_match = (topk_indices_chunked == topk_indices_baseline).all(dim=1)
    num_perfect_tokens = per_token_match.sum().item()
    print(
        f"  Perfect token matches: {num_perfect_tokens}/{total_tokens} ({num_perfect_tokens/total_tokens:.2%})"
    )

    # Detailed mismatch analysis for debugging
    if match_ratio < 1.0:
        mismatch_tokens = (~per_token_match).nonzero(as_tuple=True)[0]
        print(f"  Tokens with mismatches: {len(mismatch_tokens)}")
        if len(mismatch_tokens) > 0:
            # Show first few mismatches
            for i in range(min(3, len(mismatch_tokens))):
                token_idx = mismatch_tokens[i].item()
                chunked_topk = topk_indices_chunked[token_idx]
                baseline_topk = topk_indices_baseline[token_idx]
                diff_mask = chunked_topk != baseline_topk
                num_diffs = diff_mask.sum().item()
                print(
                    f"    Token {token_idx}: {num_diffs}/{index_topk} indices differ"
                )

    # Should be identical since we use same K cache and same compute
    # Allow tiny tolerance for potential numerical differences
    assert match_ratio >= 0.99, \
        f"Chunked and non-chunked results differ: {match_ratio:.4f} < 0.99"

    print(
        f"✅ Test passed! Chunked and non-chunked paths produce consistent results"
    )
    print(
        f"   Chunk size: {chunk_size}, Num chunks: {num_chunks}, Batch: {batch_size}"
    )


@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@pytest.mark.skipif(getSMVersion() < 90, reason="FP8 operations require SM90+")
@pytest.mark.parametrize(
    "chunk_size,seq_lens_list",
    [
        # Test intra-request Q-block chunking
        (1024, [512, 2500, 800]),  # Middle request needs Q-block splitting
        (2048, [1000, 5000, 1500]),  # Middle request needs 3 Q-blocks
        (1500, [3200, 1200]),  # First request needs Q-block splitting
        # Test mixed: small requests + large request
        (1000, [400, 600, 2200, 500]),  # Request 2 needs Q-block splitting
    ])
def test_indexer_two_level_chunking(chunk_size, seq_lens_list):
    """
    Test two-level chunking: request-level + intra-request Q-block chunking.

    This test validates that:
    1. Large requests exceeding chunk_size are split into Q-blocks
    2. Each Q-block correctly attends to all previous K tokens in the request
    3. Results are identical to non-chunked execution
    4. Chunked and non-chunked paths produce the same topk indices
    """
    torch.manual_seed(42)
    random.seed(42)

    # Test parameters
    heads, head_dim = 32, 128
    block_size = 64
    index_topk = 2048
    max_model_len = 16384
    layer_idx = 0
    batch_size = len(seq_lens_list)

    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device='cpu')
    total_tokens = seq_lens.sum().item()
    max_seq_len = seq_lens.max().item()

    print(f"\n=== Test Config: Two-Level Chunking ===")
    print(f"  Batch: {batch_size}, Chunk size: {chunk_size}")
    print(f"  Sequence lengths: {seq_lens_list}")
    print(f"  Total tokens: {total_tokens}, Max seq len: {max_seq_len}")

    # Identify which requests need Q-block splitting
    large_requests = [(i, seq_len) for i, seq_len in enumerate(seq_lens_list)
                      if seq_len > chunk_size]
    if large_requests:
        print(f"  Large requests (need Q-block splitting):")
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
    k_fp8, k_scale = torch.ops.trtllm.fp8_quantize_1x128(k)
    k_scale = k_scale.contiguous().transpose(0, 1)

    # ========== Test 1: Two-Level Chunked Prefill ==========
    print(f"\n=== Test 1: Two-Level Chunked Prefill ===")

    # Create fresh metadata for chunked test
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
    )

    # Enable two-level chunking
    metadata_chunked.indexer_max_chunk_size = chunk_size

    # Prepare metadata - this will create chunks with Q-block splitting
    Indexer.prepare(metadata_chunked)

    # Verify chunks were created
    assert metadata_chunked.indexer_prefill_chunks is not None, \
        "Chunked prefill should create chunk metadata"
    num_chunks = len(metadata_chunked.indexer_prefill_chunks)
    print(f"✓ Created {num_chunks} chunks")

    # Print detailed chunk info
    for i, chunk in enumerate(metadata_chunked.indexer_prefill_chunks):
        num_q_tokens = chunk.token_end - chunk.token_start
        num_k_tokens = chunk.k_token_end - chunk.k_token_start
        print(
            f"  Chunk {i}: Q[{chunk.token_start}:{chunk.token_end}] ({num_q_tokens} tokens), "
            f"K[{chunk.k_token_start}:{chunk.k_token_end}] ({num_k_tokens} tokens)"
        )

    # Call sparse_attn_indexer directly (uses two-level chunked path)
    # Note: sparse_attn_indexer internally calls _update_k_cache, so no need to call it explicitly
    topk_indices_chunked = indexer.sparse_attn_indexer(
        metadata_chunked,
        hidden_states,
        q_fp8,
        k,
        weights,
    )

    print(f"✓ Two-level chunked sparse_attn_indexer completed")
    print(f"  Output shape: {topk_indices_chunked.shape}")

    # ========== Test 2: Non-chunked Baseline ==========
    print(f"\n=== Test 2: Non-chunked Baseline ===")

    # Create fresh metadata for non-chunked test
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
    )

    # Use very large chunk size to effectively disable chunking
    metadata_baseline.indexer_max_chunk_size = max_model_len

    # Prepare metadata
    Indexer.prepare(metadata_baseline)

    if metadata_baseline.indexer_prefill_chunks is not None:
        num_baseline_chunks = len(metadata_baseline.indexer_prefill_chunks)
        print(
            f"✓ Created {num_baseline_chunks} chunk(s) (effectively non-chunked)"
        )

    # Call sparse_attn_indexer directly
    # Note: sparse_attn_indexer internally calls _update_k_cache, so no need to call it explicitly
    topk_indices_baseline = indexer.sparse_attn_indexer(
        metadata_baseline,
        hidden_states,
        q_fp8,
        k,
        weights,
    )

    print(f"✓ Non-chunked sparse_attn_indexer completed")
    print(f"  Output shape: {topk_indices_baseline.shape}")

    # ========== Validation ==========
    print(f"\n=== Validation ===")

    # Compare topk indices
    match_count = (topk_indices_chunked == topk_indices_baseline).sum().item()
    total_elements = total_tokens * index_topk
    match_ratio = match_count / total_elements

    print(f"  Match ratio: {match_ratio:.4f} ({match_count}/{total_elements})")

    # Check per-token accuracy
    per_token_match = (topk_indices_chunked == topk_indices_baseline).all(dim=1)
    num_perfect_tokens = per_token_match.sum().item()
    print(f"  Perfect token matches: {num_perfect_tokens}/{total_tokens} "
          f"({num_perfect_tokens/total_tokens:.2%})")

    # Detailed mismatch analysis per request
    if match_ratio < 1.0:
        mismatch_tokens = (~per_token_match).nonzero(as_tuple=True)[0]
        print(f"  Tokens with mismatches: {len(mismatch_tokens)}")

        # Group mismatches by request
        cumulative_lens = torch.cat([torch.tensor([0]), seq_lens.cumsum(0)])
        for req_idx in range(batch_size):
            req_start = cumulative_lens[req_idx].item()
            req_end = cumulative_lens[req_idx + 1].item()
            req_mismatches = mismatch_tokens[(mismatch_tokens >= req_start)
                                             & (mismatch_tokens < req_end)]
            if len(req_mismatches) > 0:
                print(f"    Request {req_idx} (len={seq_lens_list[req_idx]}): "
                      f"{len(req_mismatches)} mismatched tokens")
                # Show first few
                for i in range(min(2, len(req_mismatches))):
                    token_idx = req_mismatches[i].item()
                    diff_mask = topk_indices_chunked[
                        token_idx] != topk_indices_baseline[token_idx]
                    num_diffs = diff_mask.sum().item()
                    print(
                        f"      Token {token_idx}: {num_diffs}/{index_topk} indices differ"
                    )

    # Should be identical since we use same K cache and same compute
    # Allow tiny tolerance for potential numerical differences
    assert match_ratio >= 0.99, \
        f"Two-level chunked and non-chunked results differ: {match_ratio:.4f} < 0.99"

    print(f"✅ Test passed! Two-level chunking produces consistent results")
    print(
        f"   Chunk size: {chunk_size}, Num chunks: {num_chunks}, Batch: {batch_size}"
    )
    print(f"   Seq lens: {seq_lens_list}")
