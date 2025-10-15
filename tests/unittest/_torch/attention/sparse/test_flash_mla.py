"""
Test suite for FlashMLA sparse attention kernels.

Adapted from flash-mla/tests/test_flash_mla_prefill.py
Tests basic sparse MLA forward pass to verify kernels are working correctly.
"""

import math

import pytest
import torch
from utils.util import getSMVersion


def has_flash_mla():
    """Check if FlashMLA module is available."""
    try:
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not has_flash_mla(), reason="FlashMLA not available")
@pytest.mark.skipif(
    getSMVersion() < 90,
    reason="FlashMLA requires SM90 (Hopper) or SM100 (Blackwell)")
@pytest.mark.parametrize(
    "seq_len_q,seq_len_kv,topk",
    [
        (62, 128, 128),  # Small test case
        (128, 256, 128),  # Medium
        (128, 512, 256),  # Larger topk
    ])
def test_flash_mla_sparse_fwd(seq_len_q, seq_len_kv, topk):
    """
    Test FlashMLA sparse attention forward kernel.

    Args:
        seq_len_q: Query sequence length
        seq_len_kv: Key-Value sequence length
        topk: Number of tokens to attend to (must be multiple of 128)
    """
    from tensorrt_llm.flash_mla import flash_mla_sparse_fwd

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Fixed parameters matching FlashMLA's kernel requirements
    # These are hardware-specific, not arbitrary choices
    batch_size = 1
    num_heads_q = 128  # Fixed requirement for kernel (B_H parameter)
    num_heads_kv = 1  # MLA uses 1 KV head
    head_dim_qk = 576  # DeepSeek MLA standard
    head_dim_v = 512  # Fixed requirement (only 512 supported)

    # Generate test inputs
    # Q: [b, s_q, h_q, d_qk]
    q = torch.randn(batch_size,
                    seq_len_q,
                    num_heads_q,
                    head_dim_qk,
                    dtype=torch.bfloat16,
                    device='cuda') / 10.0
    q.clamp_(-10, 10)

    # KV: [b, s_kv, h_kv, d_qk]
    kv = torch.randn(batch_size,
                     seq_len_kv,
                     num_heads_kv,
                     head_dim_qk,
                     dtype=torch.bfloat16,
                     device='cuda') / 10.0
    kv.clamp_(-10, 10)

    # Indices: [b, s_q, h_kv, topk] - which KV tokens each Q attends to
    indices = torch.randint(0,
                            seq_len_kv,
                            (batch_size, seq_len_q, num_heads_kv, topk),
                            dtype=torch.int32,
                            device='cuda')

    softmax_scale = 1.0 / math.sqrt(head_dim_qk)

    # Run FlashMLA sparse forward (API expects no batch dimension)
    output, max_logits, lse = flash_mla_sparse_fwd(
        q.squeeze(0),  # [s_q, h_q, d_qk]
        kv.squeeze(0),  # [s_kv, h_kv, d_qk]
        indices.squeeze(0),  # [s_q, h_kv, topk]
        sm_scale=softmax_scale)

    # Validate outputs
    assert output.shape == (seq_len_q, num_heads_q, head_dim_v), \
        f"Output shape mismatch: expected [{seq_len_q}, {num_heads_q}, {head_dim_v}], got {output.shape}"
    assert output.dtype == torch.bfloat16, \
        f"Output dtype mismatch: expected torch.bfloat16, got {output.dtype}"

    assert max_logits.shape == (seq_len_q, num_heads_q), \
        f"Max logits shape mismatch: got {max_logits.shape}"
    assert max_logits.dtype == torch.float32, \
        f"Max logits dtype mismatch: got {max_logits.dtype}"

    assert lse.shape == (seq_len_q, num_heads_q), \
        f"LSE shape mismatch: got {lse.shape}"
    assert lse.dtype == torch.float32, \
        f"LSE dtype mismatch: got {lse.dtype}"

    # Numerical validity checks
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    assert not torch.isnan(max_logits).any(), "Max logits contains NaN"
    assert not torch.isnan(lse).any(), "LSE contains NaN"
