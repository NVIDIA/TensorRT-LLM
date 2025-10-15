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
                          cache_manager, num_ctx_tokens, num_tokens,
                          indexer_max_chunk_size=8194):
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
            self.indexer_max_chunk_size = indexer_max_chunk_size
            self.indexer_prefill_chunks = None

    return MockMetadata()

@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@pytest.mark.skipif(getSMVersion() < 90, reason="FP8 operations require SM90+")
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


@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@pytest.mark.skipif(
    getSMVersion() < 90,
    reason="fp8_paged_mqa_logits is only supported in SM90 and SM100")
@pytest.mark.parametrize("batch_size,next_n", [(4, 1), (2, 2)])
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


@pytest.mark.parametrize(
    "max_chunk_size,seq_lens,start_idx,expected_specs",
    [
        # Small requests - single chunk
        (1000, [200, 300, 400], 0, [
            [(0, 0, 200, 0), (1, 0, 300, 200), (2, 0, 400, 500)]
        ]),
        # Small requests - multiple chunks
        (1000, [400, 500, 300, 600], 0, [
            [(0, 0, 400, 0), (1, 0, 500, 400)],
            [(2, 0, 300, 900), (3, 0, 600, 1200)]
        ]),
        # Large request - intra-request chunking
        (1000, [2500], 0, [
            [(0, 0, 1000, 0)],
            [(0, 1000, 2000, 0)],
            [(0, 2000, 2500, 0)]
        ]),
        # Mixed: small + large + small
        (1000, [400, 2500, 300], 0, [
            [(0, 0, 400, 0)],
            [(1, 0, 1000, 400)],
            [(1, 1000, 2000, 400)],
            [(1, 2000, 2500, 400)],
            [(2, 0, 300, 2900)]
        ]),
        # Edge case: exact chunk size
        (1000, [1000], 0, [
            [(0, 0, 1000, 0)]
        ]),
        # Edge case: non-zero start index
        (1000, [100, 200, 300], 1, [
            [(1, 0, 200, 100), (2, 0, 300, 300)]
        ]),
    ],
    ids=["small_single", "small_multi", "large_chunked", "mixed", "exact_size", "non_zero_start"]
)
def test_split_prefill_chunks(max_chunk_size, seq_lens, start_idx, expected_specs):
    """
    Test split_prefill_chunks covering:
    - Request-level chunking (small requests)
    - Intra-request Q-block chunking (large requests)
    - Mixed scenarios
    - Edge cases
    """
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)
    chunk_groups = split_prefill_chunks(seq_lens_tensor, max_chunk_size, start_idx=start_idx)

    assert len(chunk_groups) == len(expected_specs), \
        f"Expected {len(expected_specs)} chunks, got {len(chunk_groups)}"

    for i, expected in enumerate(expected_specs):
        assert chunk_groups[i] == expected, \
            f"Chunk {i} mismatch:\nGot:      {chunk_groups[i]}\nExpected: {expected}"

    print(f"✅ test_split_prefill_chunks passed")


@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@pytest.mark.skipif(getSMVersion() < 90, reason="FP8 operations require SM90+")
@pytest.mark.parametrize(
    "chunk_size,seq_lens_list,chunking_type",
    [
        # Request-level chunking: small requests that fit individually but need chunking as batch
        (512, None, "request_level"),   # Random seq lens, 5 requests
        (1024, None, "request_level"),  # Random seq lens, 4 requests
        (2048, None, "request_level"),  # Random seq lens, 4 requests

        # Two-level chunking: intra-request Q-block splitting for large requests
        (1024, [512, 2500, 800], "two_level"),        # Middle request needs Q-blocks
        (1500, [3200, 1200], "two_level"),            # First request needs 3 Q-blocks
        (1000, [400, 600, 2200, 500], "two_level"),   # Mixed: request 2 needs Q-blocks
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
            seq_len = random.randint(int(0.3 * chunk_size), int(0.9 * chunk_size))
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
        large_requests = [(i, seq_len) for i, seq_len in enumerate(seq_lens_list)
                          if seq_len > chunk_size]
        if large_requests:
            print(f"  Large requests (Q-block splitting):")
            for req_idx, seq_len in large_requests:
                num_q_blocks = (seq_len + chunk_size - 1) // chunk_size
                print(f"    Request {req_idx}: {seq_len} tokens → {num_q_blocks} Q-blocks")

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
    q = torch.randn((total_tokens, heads, head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((total_tokens, head_dim), device="cuda", dtype=torch.bfloat16)
    weights = torch.randn((total_tokens, heads), device="cuda", dtype=torch.float32)
    hidden_states = torch.randn((total_tokens, 4096), device="cuda", dtype=torch.bfloat16)

    # Quantize inputs
    q_fp8 = q.to(torch.float8_e4m3fn)
    k_fp8, k_scale = torch.ops.trtllm.fp8_quantize_1x128(k)
    k_scale = k_scale.contiguous().transpose(0, 1)

    # ========== Test Path 1: Chunked Prefill ==========
    print(f"\n=== Chunked Path ===")

    metadata_chunked = _create_mock_metadata(
        request_ids, batch_size,
        num_contexts=batch_size, num_generations=0,
        seq_lens=seq_lens.clone(), kv_lens=seq_lens.clone(),
        num_cached_tokens=[0] * batch_size,
        cache_manager=cache_manager,
        num_ctx_tokens=total_tokens, num_tokens=total_tokens,
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
        print(f"  Chunk {i}: Q[{chunk.token_start}:{chunk.token_end}] ({num_q} tokens), "
              f"K[{chunk.k_token_start}:{chunk.k_token_end}] ({num_k} tokens)")

    topk_indices_chunked = indexer.sparse_attn_indexer(
        metadata_chunked, hidden_states, q_fp8, k, weights)

    print(f"✓ Chunked execution completed, shape: {topk_indices_chunked.shape}")

    # ========== Test Path 2: Non-chunked Baseline ==========
    print(f"\n=== Non-chunked Baseline ===")

    metadata_baseline = _create_mock_metadata(
        request_ids, batch_size,
        num_contexts=batch_size, num_generations=0,
        seq_lens=seq_lens.clone(), kv_lens=seq_lens.clone(),
        num_cached_tokens=[0] * batch_size,
        cache_manager=cache_manager,
        num_ctx_tokens=total_tokens, num_tokens=total_tokens,
        indexer_max_chunk_size=max_model_len,
    )

    Indexer.prepare(metadata_baseline)

    if metadata_baseline.indexer_prefill_chunks is not None:
        num_baseline_chunks = len(metadata_baseline.indexer_prefill_chunks)
        print(f"✓ Created {num_baseline_chunks} chunk(s) (effectively non-chunked)")

    topk_indices_baseline = indexer.sparse_attn_indexer(
        metadata_baseline, hidden_states, q_fp8, k, weights)

    print(f"✓ Non-chunked execution completed, shape: {topk_indices_baseline.shape}")

    # ========== Validation ==========
    print(f"\n=== Validation ===")

    match_count = (topk_indices_chunked == topk_indices_baseline).sum().item()
    total_elements = total_tokens * index_topk
    match_ratio = match_count / total_elements

    print(f"  Match ratio: {match_ratio:.4f} ({match_count}/{total_elements})")

    per_token_match = (topk_indices_chunked == topk_indices_baseline).all(dim=1)
    num_perfect_tokens = per_token_match.sum().item()
    print(f"  Perfect token matches: {num_perfect_tokens}/{total_tokens} "
          f"({num_perfect_tokens/total_tokens:.2%})")

    # Detailed mismatch analysis
    if match_ratio < 1.0:
        mismatch_tokens = (~per_token_match).nonzero(as_tuple=True)[0]
        print(f"  Tokens with mismatches: {len(mismatch_tokens)}")

        # Group by request for two-level chunking
        if chunking_type == "two_level":
            cumulative_lens = torch.cat([torch.tensor([0]), seq_lens.cumsum(0)])
            for req_idx in range(batch_size):
                req_start = cumulative_lens[req_idx].item()
                req_end = cumulative_lens[req_idx + 1].item()
                req_mismatches = mismatch_tokens[(mismatch_tokens >= req_start)
                                                 & (mismatch_tokens < req_end)]
                if len(req_mismatches) > 0:
                    print(f"    Request {req_idx} (len={seq_lens_list[req_idx]}): "
                          f"{len(req_mismatches)} mismatches")
        else:
            # Show first few mismatches
            for i in range(min(3, len(mismatch_tokens))):
                token_idx = mismatch_tokens[i].item()
                diff_count = (topk_indices_chunked[token_idx] !=
                             topk_indices_baseline[token_idx]).sum().item()
                print(f"    Token {token_idx}: {diff_count}/{index_topk} indices differ")

    assert match_ratio >= 0.99, \
        f"Chunked and non-chunked results differ: {match_ratio:.4f} < 0.99"

    print(f"✅ Test passed! {chunking_type} chunking produces consistent results")
    print(f"   Config: chunk_size={chunk_size}, num_chunks={num_chunks}, "
          f"batch={batch_size}, seq_lens={seq_lens_list}")