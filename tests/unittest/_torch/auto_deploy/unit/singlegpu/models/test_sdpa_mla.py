import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorrt_llm._torch.auto_deploy  # noqa: F401


def naive_attn(q_nope, q_pe, compressed_kv, k_pe, wkv_b, softmax_scale):
    bsz = q_nope.shape[0]
    q_len = q_nope.shape[2]
    num_heads = q_nope.shape[1]
    qk_nope_head_dim = q_nope.shape[-1]
    v_head_dim = wkv_b.weight.shape[-1] - qk_nope_head_dim
    qk_head_dim = qk_nope_head_dim + q_pe.shape[-1]
    k_pe = k_pe.view(bsz, q_len, 1, q_pe.shape[-1]).transpose(1, 2)

    # Up project compressed_kv
    kv = (
        wkv_b(compressed_kv)
        .view(bsz, q_len, num_heads, qk_nope_head_dim + v_head_dim)
        .transpose(1, 2)
    )

    k_nope, value_states = torch.split(kv, [qk_nope_head_dim, v_head_dim], dim=-1)

    query_states = k_pe.new_empty(bsz, num_heads, q_len, qk_head_dim)
    query_states[:, :, :, :qk_nope_head_dim] = q_nope
    query_states[:, :, :, qk_nope_head_dim:] = q_pe

    key_states = k_pe.new_empty(bsz, num_heads, q_len, qk_head_dim)
    key_states[:, :, :, :qk_nope_head_dim] = k_nope
    key_states[:, :, :, qk_nope_head_dim:] = k_pe

    x = F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=None,
        is_causal=False,
        dropout_p=0.0,
        scale=softmax_scale,
    ).transpose(1, 2)

    return x


def mla_attn(q_nope, q_pe, compressed_kv, k_pe, wkv_b, softmax_scale):
    num_heads = q_nope.shape[1]
    qk_nope_head_dim = q_nope.shape[-1]
    v_head_dim = wkv_b.weight.shape[-1] - qk_nope_head_dim
    kv_lora_rank = compressed_kv.shape[-1]

    # Down project q_nope
    wkv_b_weight = wkv_b.weight.view(num_heads, -1, kv_lora_rank)
    q_nope_proj = torch.einsum("bhsd,hdc->bhsc", q_nope, wkv_b_weight[:, :qk_nope_head_dim])

    # MLA ref operation
    x = torch.ops.auto_deploy.torch_attention_deepseek_mla(
        q_nope_proj, q_pe, compressed_kv, k_pe, None, softmax_scale
    )

    # Up project attention scores
    x = torch.einsum("bshc,hdc->bshd", x, wkv_b_weight[:, -v_head_dim:])
    return x


def test_attn():
    # Define test configurations
    kv_lora_rank = 4
    bsz = 2
    q_len = 6
    v_head_dim = 2
    qk_nope_head_dim = 2
    qk_rope_head_dim = 1
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    num_heads = 4
    softmax_scale = qk_head_dim**-0.5

    # Generate inputs
    q_nope = torch.randn(bsz, num_heads, q_len, qk_nope_head_dim)
    q_pe = torch.randn(bsz, num_heads, q_len, qk_rope_head_dim)
    compressed_kv = torch.randn(bsz, q_len, kv_lora_rank)
    k_pe = torch.randn(bsz, q_len, qk_rope_head_dim)

    # Define w_kv_b projection matrix
    wkv_b = nn.Linear(
        kv_lora_rank, num_heads * (qk_head_dim - qk_rope_head_dim + v_head_dim), bias=False
    )

    # Compute naive attention
    out_naive = naive_attn(q_nope, q_pe, compressed_kv, k_pe, wkv_b, softmax_scale)

    # Compute MLA attention
    out_mla = mla_attn(q_nope, q_pe, compressed_kv, k_pe, wkv_b, softmax_scale)

    # Check if the two outputs are close
    assert torch.allclose(out_naive, out_mla, rtol=1e-5, atol=1e-5)
