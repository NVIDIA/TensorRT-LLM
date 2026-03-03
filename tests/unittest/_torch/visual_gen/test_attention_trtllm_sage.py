# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention_backend import interface as attention_backend_interface
from tensorrt_llm._torch.attention_backend import utils as attention_backend_utils


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    bsz, n_kv_heads, seqlen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, n_kv_heads, n_rep, seqlen, head_dim)
    return hidden_states.reshape(bsz, n_kv_heads * n_rep, seqlen, head_dim)


def _sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    q_seq = q.view(seq_len, num_heads, head_dim).transpose(0, 1).unsqueeze(0)
    k_seq = k.view(seq_len, num_kv_heads, head_dim).transpose(0, 1).unsqueeze(0)
    v_seq = v.view(seq_len, num_kv_heads, head_dim).transpose(0, 1).unsqueeze(0)
    if num_heads > num_kv_heads:
        n_rep = num_heads // num_kv_heads
        k_seq = _repeat_kv(k_seq, n_rep)
        v_seq = _repeat_kv(v_seq, n_rep)

    out = F.scaled_dot_product_attention(q_seq, k_seq, v_seq, is_causal=False)
    return out.squeeze(0).transpose(0, 1).contiguous().view(seq_len, num_heads * head_dim)


def run_once(
    num_heads: int = 16,
    num_kv_heads: int = 16,
    head_dim: int = 128,
    seq_len: int = 1024,
    amp_mul: float = 3.2,
    amp_mul_v: Optional[float] = None,
    sage_attn_qk_int8: bool = False,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor, float, float, float]:
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    device = "cuda"
    in_dtype = torch.bfloat16

    q = torch.rand(seq_len, num_heads * head_dim, device=device, dtype=in_dtype)
    k = torch.rand(seq_len, num_kv_heads * head_dim, device=device, dtype=in_dtype)
    v = torch.rand(seq_len, num_kv_heads * head_dim, device=device, dtype=in_dtype)

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
        max_num_requests=1,
        max_num_tokens=seq_len,
        kv_cache_manager=None,
        mapping=None,
        runtime_features=None,
    )
    metadata.seq_lens = torch.tensor([seq_len], dtype=torch.int32)
    metadata.num_contexts = 1
    metadata.request_ids = torch.tensor([0], dtype=torch.int32)
    metadata.max_seq_len = seq_len
    metadata.prepare()

    mask_type = attention_backend_interface.PredefinedAttentionMask.FULL
    output = torch.empty((seq_len, num_heads * head_dim), device=device, dtype=out_dtype)

    # Attention kwargs
    attn_kwargs = {
        "output": output,
        "attention_mask": mask_type,
    }

    # SageAttention separate-QKV requires these block sizes.
    attn_kwargs.update({
        "sage_attn_num_elts_per_blk_q": 1,
        "sage_attn_num_elts_per_blk_k": 4 if sage_attn_qk_int8 else 1,
        "sage_attn_num_elts_per_blk_v": 1,
        "sage_attn_qk_int8": sage_attn_qk_int8,
    })

    trtllm_out = attention.forward(
        q,
        k,
        v,
        metadata,
        **attn_kwargs,
    )
    if isinstance(trtllm_out, tuple):
        trtllm_out = trtllm_out[0]

    ref_out = _sdpa_reference(
        q.to(torch.bfloat16),
        k.to(torch.bfloat16),
        v.to(torch.bfloat16),
        seq_len,
        num_heads,
        num_kv_heads,
        head_dim,
    )
    trtllm_out = trtllm_out.to(torch.bfloat16)

    max_abs = (trtllm_out - ref_out).abs().max().item()
    mean_abs = (trtllm_out - ref_out).abs().mean().item()
    cos_sim = F.cosine_similarity(trtllm_out.reshape(-1).float(), ref_out.reshape(-1).float(), dim=0).item()

    return trtllm_out, ref_out, max_abs, mean_abs, cos_sim


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for TRTLLM attention.")
@pytest.mark.parametrize("num_heads", [4, 12, 16])
@pytest.mark.parametrize("seq_len", [128, 256, 1024, 8192])
@pytest.mark.parametrize(
    "out_dtype,sage_attn_qk_int8,atol,rtol",
    [
        (torch.float8_e4m3fn, False, 3e-1, 2e-1),
        (torch.float8_e4m3fn, True, 3e-1, 2e-1),
        (torch.bfloat16, True, 1e-1, 4e-2),
    ],
)
def test_attention_trtllm_sage(
    num_heads: int,
    seq_len: int,
    out_dtype: torch.dtype,
    sage_attn_qk_int8: bool,
    atol: float,
    rtol: float,
):
    print("\n" + "=" * 60)
    print(
        "Testing TRTLLM Separate-QKV SageAttention "
        f"(num_heads={num_heads}, seq_len={seq_len}, qk_int8={sage_attn_qk_int8}, out_dtype={out_dtype})"
    )
    print("=" * 60)

    trtllm_out, ref_out, max_abs, mean_abs, cos_sim = run_once(
        num_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim=128,
        seq_len=seq_len,
        amp_mul=3.2,
        sage_attn_qk_int8=sage_attn_qk_int8,
        out_dtype=out_dtype,
    )

    assert trtllm_out.shape == ref_out.shape
    assert torch.isfinite(trtllm_out).all()
    assert torch.isfinite(ref_out).all()

    print("\nResults:")
    print(f"  Output shape: {trtllm_out.shape}")
    print(f"  Max absolute difference: {max_abs:.6f}")
    print(f"  Mean absolute difference: {mean_abs:.6f}")
    print(f"  Cosine similarity: {cos_sim:.6f}")

    assert cos_sim > 0.90, "Cosine similarity check failed"
    torch.testing.assert_close(trtllm_out, ref_out, atol=atol, rtol=rtol)
