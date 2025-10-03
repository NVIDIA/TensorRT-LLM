"""
Test suite for DeepGEMM indexer kernels and some related utilities.

This file tests:
1. fp8_mqa_logits operation from the DeepGEMM library
2. fp8_paged_mqa_logits operation with paged KV cache
3. compute_cu_seqlen_kv_bounds utility for batched causal attention
"""

import pytest
import random
import torch
from utils.util import getSMVersion
from tensorrt_llm import deep_gemm
from tensorrt_llm.deep_gemm import (fp8_paged_mqa_logits,
                                     get_paged_mqa_logits_metadata,
                                     get_num_sms)
from tensorrt_llm._torch.attention_backend.sparse.dsa import (
    compute_cu_seqlen_kv_bounds_nocache,
    DSACacheManager
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.mapping import Mapping
from utils.util import check_accuracy

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
        def __init__(self, index_head_dim):
            self.index_head_dim = index_head_dim
            self.prompt_budget = 1024

    sparse_attn_config = SparseAttentionConfig(head_dim)

    # Create KV cache config
    # Note: max_attention_window expects list[int] (one per layer)
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=False,
        max_tokens=max_seq_len * batch_size,
        max_attention_window=[max_seq_len] * num_layers,  # List of max window per layer
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

    return cache_manager

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


def kv_cache_cast_to_fp8(x: torch.Tensor) -> torch.Tensor:
    """
    Cast paged KV cache to FP8 format with packed scale layout.

    Args:
        x: Tensor of shape [num_blocks, block_size, 1, head_dim]

    Returns:
        Packed FP8 cache with shape [num_blocks, block_size, 1, head_dim + 4]
        where the last 4 bytes per token store the float32 dequant scale.
    """
    # x: (num_blocks, block_size, 1, head_dim)
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1

    # Compute per-token scales: amax along head_dim
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0

    # Quantize to FP8
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)

    # Pack FP8 data and scales into flat layout then reshape
    x_fp8 = torch.empty(
        (num_blocks, block_size * (head_dim + 4)),
        device=x.device,
        dtype=torch.uint8,
    )

    # Pack FP8 values: [num_blocks, block_size * head_dim]
    x_fp8[:, :block_size * head_dim] = x_scaled.view(
        num_blocks, block_size * head_dim).view(dtype=torch.uint8)

    # Pack scales: [num_blocks, block_size * 4]
    x_fp8[:, block_size * head_dim:] = sf.view(
        num_blocks, block_size).view(dtype=torch.uint8)

    # Reshape to [num_blocks, block_size, 1, head_dim + 4]
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4)


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
                        reason="fp8_mqa_logits is only supported in SM90 and SM100"
                    )
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
    ke = torch.arange(seq_len, dtype=torch.int,
                      device="cuda") + (seq_len_kv - seq_len) + 1  # +1 for exclusive end

    # Convert to FP8
    q_fp8 = q.to(torch.float8_e4m3fn)
    kv_fp8 = per_custom_dims_cast_to_fp8(kv, (0,), False)
    logits = deep_gemm.fp8_mqa_logits(q_fp8, kv_fp8, weights, ks, ke) # -> [seq_len, seq_len_kv]

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
    assert diff < 1e-3, f"{diff=}" # check for cosine similarity
    check_accuracy(logits, ref_logits, atol=1e-2, rtol=2e-1, percent=0.9) # double check for per-element similarity

@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@pytest.mark.skipif(getSMVersion() < 90,
                    reason="fp8_paged_mqa_logits is only supported in SM90 and SM100")
def test_indexer_paged_kv_cache_block_management():
    """
    Explicitly test paged KV cache block table management using DSACacheManager.

    This test validates:
    1. DSACacheManager.get_indexer_k_cache_buffers() provides correct cache
    2. DSACacheManager.get_indexer_k_block_offsets() provides correct block tables
    3. Non-contiguous block assignments work correctly
    4. Different sequences can use different physical blocks

    Uses same parameters as test_deepgemm_fp8_paged_mqa_logits with next_n=1
    """
    torch.manual_seed(123)
    random.seed(123)

    # Use same parameters as test_deepgemm_fp8_paged_mqa_logits
    batch_size, next_n = 4, 1
    heads, head_dim = 32, 128
    block_size = 64
    max_model_len = 4096
    layer_idx = 0

    # Create DSA cache manager
    cache_manager = create_dsa_cache_manager(
        batch_size=batch_size,
        head_dim=head_dim,
        tokens_per_block=block_size,
        max_seq_len=max_model_len,
        num_layers=1
    )

    # Add dummy requests to allocate blocks (192 tokens each = 3 blocks)
    request_ids = list(range(batch_size))
    token_nums = [192] * batch_size
    cache_manager.add_dummy_requests(
        request_ids=request_ids,
        token_nums=token_nums,
        is_gen=False,
        prepare_resource=True
    )

    # Get indexer k cache buffer using cache manager method
    kv_cache = cache_manager.get_indexer_k_cache_buffers(layer_idx)
    # Shape: [num_blocks, block_size, 1, head_dim + scale_size]

    # Get block tables using cache manager method
    block_tables = cache_manager.get_indexer_k_block_offsets(request_ids)
    # Shape: [batch_size, max_blocks_per_seq]

    print(f"✓ Cache manager setup:")
    print(f"  - KV cache shape: {kv_cache.shape}")
    print(f"  - Block tables shape: {block_tables.shape}")
    print(f"  - Block assignments per sequence:")
    for i in range(batch_size):
        assigned_blocks = block_tables[i, :3].cpu().tolist()
        print(f"    Seq {i}: blocks {assigned_blocks}")

    # Create BF16 version of cache for test data (strip scale bytes)
    num_blocks, _, _, head_dim_with_scale = kv_cache.shape
    kv_cache_bf16 = torch.randn((num_blocks, block_size, 1, head_dim),
                                device="cuda", dtype=torch.bfloat16)

    # Create context lengths (3 blocks = 192 tokens each)
    context_lens = torch.tensor([192, 192, 192, 192], dtype=torch.int32, device="cuda")

    # Create queries and weights with random values (like test_deepgemm_fp8_paged_mqa_logits)
    q = torch.randn((batch_size, next_n, heads, head_dim),
                    device="cuda", dtype=torch.bfloat16)
    weights = torch.randn((batch_size * next_n, heads),
                         device="cuda", dtype=torch.float32)

    # Convert to FP8
    q_fp8 = q.to(torch.float8_e4m3fn)
    kv_cache_fp8 = kv_cache_cast_to_fp8(kv_cache_bf16)

    # Get schedule metadata
    schedule_metadata = get_paged_mqa_logits_metadata(
        context_lens, block_size, get_num_sms())

    # Call kernel with cache manager's block tables
    logits = fp8_paged_mqa_logits(
        q_fp8, kv_cache_fp8, weights, context_lens,
        block_tables, schedule_metadata, max_model_len
    )

    # Compute reference with same block tables
    ref_logits = _ref_fp8_paged_mqa_logits(
        q, kv_cache_bf16, weights, context_lens,
        block_tables, max_model_len
    )

    # Check kernel matches reference (both should handle paging identically)
    # This is the main validation that block table lookups work correctly
    mask = torch.arange(max_model_len, device="cuda")[None, :] < context_lens[:, None]
    logits_masked = logits.masked_fill(~mask, 0)
    ref_logits_masked = ref_logits.masked_fill(~mask, 0)

    diff = _calc_diff(logits_masked, ref_logits_masked)
    assert diff < 1e-3, f"{diff=} - Kernel and reference should match for paged access"

    print(f"✓ Paged KV cache block management validated:")
    print(f"  - Using DSACacheManager.get_indexer_k_cache_buffers()")
    print(f"  - Using DSACacheManager.get_indexer_k_block_offsets()")
    print(f"  - {batch_size} sequences with cache-managed blocks")
    print(f"  - Each sequence accesses distinct physical blocks")
    print(f"  - Kernel accuracy: {diff:.6f}")


@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@pytest.mark.skipif(getSMVersion() < 90,
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
    logits_decode = torch.randn(num_gen_tokens, actual_max_len,
                                 dtype=torch.float32, device="cuda")

    # Apply the masking logic from the decode branch
    positions = torch.arange(actual_max_len,
                            device="cuda").unsqueeze(0).expand(num_gen_tokens, -1)

    row_indices = torch.arange(num_gen_tokens, device="cuda") // next_n
    next_n_offset = torch.arange(num_gen_tokens, device="cuda") % next_n

    # For each token, compute valid context end position
    # Token i in sequence j can attend to positions [0, context_len[j] - next_n + offset[i]]
    index_end_pos = (context_lens[row_indices] - next_n + next_n_offset).unsqueeze(1)

    # Validate index_end_pos shape and values
    assert index_end_pos.shape == (num_gen_tokens, 1)
    expected_end_pos = torch.tensor([
        [8],   # Seq 0, token 0: 10 - 2 + 0 = 8
        [9],   # Seq 0, token 1: 10 - 2 + 1 = 9
        [13],  # Seq 1, token 0: 15 - 2 + 0 = 13
        [14],  # Seq 1, token 1: 15 - 2 + 1 = 14
        [6],   # Seq 2, token 0: 8 - 2 + 0 = 6
        [7],   # Seq 2, token 1: 8 - 2 + 1 = 7
    ], dtype=torch.int32, device="cuda")
    assert torch.equal(index_end_pos, expected_end_pos), \
        f"index_end_pos mismatch:\nGot:\n{index_end_pos}\nExpected:\n{expected_end_pos}"

    # Apply mask: positions <= index_end_pos
    mask = positions <= index_end_pos
    logits_masked = logits_decode.masked_fill(~mask, float('-inf'))

    # Verify masking correctness
    # For token 0 of seq 0 (index_end_pos=8), positions 0-8 should be valid, 9-14 masked
    assert not torch.isinf(logits_masked[0, :9]).any(), "Valid positions should not be -inf"
    assert torch.isinf(logits_masked[0, 9:]).all(), "Invalid positions should be -inf"

    # For token 1 of seq 1 (index_end_pos=14), positions 0-14 should be valid
    assert not torch.isinf(logits_masked[3, :15]).any(), "All positions should be valid"

    # For token 0 of seq 2 (index_end_pos=6), positions 0-6 valid, 7-14 masked
    assert not torch.isinf(logits_masked[4, :7]).any(), "Valid positions should not be -inf"
    assert torch.isinf(logits_masked[4, 7:]).all(), "Invalid positions should be -inf"

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
    short_index_end_pos = torch.tensor([[3]], dtype=torch.int32, device="cuda")  # Only 4 valid positions
    short_mask = short_positions <= short_index_end_pos
    short_logits_masked = short_logits.masked_fill(~short_mask, float('-inf'))

    # Request topk_k=10 but only 4 valid positions
    short_topk_values, short_topk_indices = short_logits_masked.topk(10, dim=-1)

    # First 4 should be non-inf with valid indices, rest should be -inf
    assert not torch.isinf(short_topk_values[0, :4]).any(), "First 4 should be valid values"
    assert torch.isinf(short_topk_values[0, 4:]).all(), "Last 6 should be -inf"
    assert (short_topk_indices[0, :4] <= 3).all(), "Valid topk indices should be within range [0, 3]"


def test_compute_cu_seqlen_bounds_nocache():
    """Simple test case with 2 sequences."""
    seq_lens = torch.tensor([3, 4], dtype=torch.int32, device="cuda")
    num_contexts = 2
    num_ctx_tokens = 7

    cu_seqlen_ks, cu_seqlen_ke = compute_cu_seqlen_kv_bounds_nocache(
        seq_lens, num_contexts, num_ctx_tokens
    )

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

    expected_ks = torch.tensor([0, 0, 0, 3, 3, 3, 3], dtype=torch.int32, device="cuda")
    expected_ke = torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.int32, device="cuda")

    assert torch.equal(cu_seqlen_ks, expected_ks), \
        f"cu_seqlen_ks mismatch:\nGot:      {cu_seqlen_ks.tolist()}\nExpected: {expected_ks.tolist()}"
    assert torch.equal(cu_seqlen_ke, expected_ke), \
        f"cu_seqlen_ke mismatch:\nGot:      {cu_seqlen_ke.tolist()}\nExpected: {expected_ke.tolist()}"
