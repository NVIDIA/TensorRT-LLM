# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention_backend import interface as attention_backend_interface
from tensorrt_llm._torch.attention_backend import utils as attention_backend_utils


def _cuda_cc():
    if torch.cuda.is_available():
        return torch.cuda.get_device_capability()
    else:
        return -1, -1


def _repeat_kv(hidden_states: torch.Tensor, gqa_groups: int) -> torch.Tensor:
    bsz, n_kv_heads, seqlen, head_dim = hidden_states.shape
    if gqa_groups == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        bsz, n_kv_heads, gqa_groups, seqlen, head_dim
    )
    return hidden_states.reshape(bsz, n_kv_heads * gqa_groups, seqlen, head_dim)


def _sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    q_seq = q.view(-1, seq_len, num_heads, head_dim).transpose(1, 2)
    k_seq = k.view(-1, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v_seq = v.view(-1, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    if num_heads > num_kv_heads:
        gqa_groups = num_heads // num_kv_heads
        k_seq = _repeat_kv(k_seq, gqa_groups)
        v_seq = _repeat_kv(v_seq, gqa_groups)

    out = F.scaled_dot_product_attention(q_seq, k_seq, v_seq, is_causal=False)
    return out.transpose(1, 2).contiguous().view(-1, num_heads * head_dim)


def _test_attention_trtllm_sage(
    num_heads: int = 16,
    num_kv_heads: int = 16,
    head_dim: int = 128,
    batch_size=1,
    seq_len: int = 1024,
    amp_mul: float = 3.2,
    amp_mul_v: Optional[float] = None,
    sage_attn_qk_int8: bool = False,
    sage_attn_num_elts_per_blk_k: Optional[float] = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor, float, float, float]:
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    device = "cuda"
    in_dtype = torch.bfloat16

    q = torch.rand(batch_size * seq_len, num_heads * head_dim, device=device, dtype=in_dtype)
    k = torch.rand(batch_size * seq_len, num_kv_heads * head_dim, device=device, dtype=in_dtype)
    v = torch.rand(batch_size * seq_len, num_kv_heads * head_dim, device=device, dtype=in_dtype)

    # Extra fluctuations
    if amp_mul_v is None:
        amp_mul_v = amp_mul
    q = q * ((torch.rand_like(q) - 0.5) * amp_mul).exp()
    k = k * ((torch.rand_like(k) - 0.5) * amp_mul).exp()
    v = v * ((torch.rand_like(v) - 0.5) * amp_mul_v).exp()

    # Obtain Op and run
    attention_cls = attention_backend_utils.get_attention_backend("TRTLLM")
    attention = attention_cls(
        layer_idx=0,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    metadata = attention_cls.Metadata(
        max_num_requests=batch_size,
        max_num_tokens=seq_len,
        kv_cache_manager=None,
        runtime_features=None,
    )
    metadata.seq_lens = torch.tensor([seq_len] * batch_size, dtype=torch.int32)
    metadata.request_ids = torch.tensor([0] * batch_size, dtype=torch.int32)
    metadata.num_contexts = batch_size
    metadata.max_seq_len = seq_len
    metadata.prepare()

    mask_type = attention_backend_interface.PredefinedAttentionMask.FULL
    out_tllm = torch.empty(
        (batch_size * seq_len, num_heads * head_dim), device=device, dtype=out_dtype
    )

    # Attention kwargs
    attn_kwargs = {
        "output": out_tllm,
        "attention_mask": mask_type,
    }

    # SageAttention separate-QKV requires these block sizes.
    if sage_attn_num_elts_per_blk_k is None:
        sage_attn_num_elts_per_blk_k = (16 if sage_attn_qk_int8 else 1,)
    attn_kwargs.update(
        {
            "sage_attn_num_elts_per_blk_q": 1,
            "sage_attn_num_elts_per_blk_k": sage_attn_num_elts_per_blk_k,
            "sage_attn_num_elts_per_blk_v": 1,
            "sage_attn_qk_int8": sage_attn_qk_int8,
        }
    )

    out_tllm = attention.forward(
        q,
        k,
        v,
        metadata,
        **attn_kwargs,
    )
    if isinstance(out_tllm, tuple):
        out_tllm = out_tllm[0]

    out_native = _sdpa_reference(
        q.to(torch.bfloat16),
        k.to(torch.bfloat16),
        v.to(torch.bfloat16),
        seq_len,
        num_heads,
        num_kv_heads,
        head_dim,
    )
    out_tllm = out_tllm.to(torch.bfloat16)

    max_abs = (out_tllm - out_native).abs().max().item()
    mean_abs = (out_tllm - out_native).abs().mean().item()
    cos_sim = F.cosine_similarity(
        out_tllm.reshape(-1).float(), out_native.reshape(-1).float(), dim=0
    ).item()

    return out_tllm, out_native, max_abs, mean_abs, cos_sim


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for TRTLLM attention.")
@pytest.mark.skipif(
    _cuda_cc()[0] != 10, reason="TRTLLM SageAttention test requires CUDA major version 10."
)
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("gqa_groups", [1, 2, 4])
@pytest.mark.parametrize("num_heads", [4, 12, 16])
@pytest.mark.parametrize("seq_len", [128, 256, 1024, 8192])
@pytest.mark.parametrize(
    "out_dtype,sage_attn_qk_int8,sage_attn_num_elts_per_blk_k,atol,rtol",
    [
        (torch.bfloat16, True, 4, 1e-1, 4e-2),
        (torch.bfloat16, True, 16, 5e-1, 5e-1),
        (torch.float8_e4m3fn, True, 4, 3e-1, 2e-1),
        (torch.float8_e4m3fn, False, 1, 3e-1, 2e-1),
    ],
)
def test_attention_trtllm_sage(
    seq_len: int,
    num_heads: int,
    gqa_groups: int,
    batch_size: int,
    out_dtype: torch.dtype,
    sage_attn_qk_int8: bool,
    sage_attn_num_elts_per_blk_k: int,
    atol: float,
    rtol: float,
):
    if sage_attn_qk_int8 and _cuda_cc()[1] == 3:
        pytest.skip("SM103 does not have Int8 Tensor Cores.")

    out_tllm, out_native, max_abs, mean_abs, cos_sim = _test_attention_trtllm_sage(
        num_heads=num_heads,
        num_kv_heads=num_heads // gqa_groups,
        head_dim=128,
        seq_len=seq_len,
        batch_size=batch_size,
        amp_mul=3.2,
        sage_attn_qk_int8=sage_attn_qk_int8,
        sage_attn_num_elts_per_blk_k=sage_attn_num_elts_per_blk_k,
        out_dtype=out_dtype,
    )

    assert out_tllm.shape == out_native.shape, "Shape mismatch"
    assert torch.isfinite(out_native).all(), "Inf / NaN detected in Torch SDPA"
    assert torch.isfinite(out_tllm).all(), "Inf / NaN detected in TRTLLM attention"

    print("\nResults:")
    print(f"  Output shape: {out_tllm.shape}")
    print(f"  Max absolute difference: {max_abs:.6f}")
    print(f"  Mean absolute difference: {mean_abs:.6f}")
    print(f"  Cosine similarity: {cos_sim:.6f}")

    assert cos_sim > 0.990, f"Cosine similarity {cos_sim:.6f} below threshold"
    torch.testing.assert_close(out_tllm, out_native, atol=atol, rtol=rtol)
