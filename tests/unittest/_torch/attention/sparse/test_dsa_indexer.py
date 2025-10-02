"""
Test suite for DeepGEMM indexer kernels and some related utilities.

This file tests:
1. fp8_mqa_logits operation from the DeepGEMM library
2. compute_cu_seqlen_kv_bounds utility for batched causal attention
"""

import pytest
import torch
from utils.util import getSMVersion
from tensorrt_llm import deep_gemm
from tensorrt_llm._torch.attention_backend.sparse.dsa import compute_cu_seqlen_kv_bounds_nocache
from utils.util import check_accuracy

def has_deep_gemm():
    try:
        return deep_gemm is not None
    except:
        return False

def _ceil_to_ue8m0(x: torch.Tensor):
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))

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

# TODO: add test for fp8_paged_mqa_logits for kv cache support


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
