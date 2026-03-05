# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
import torch.nn.functional as F


@torch.inference_mode()
def torch_ref_dit_qk_norm_rope(
    qkv,
    num_heads_q,
    num_heads_k,
    num_heads_v,  # Unused — V is passed through unchanged. Kept for API parity with CUDA kernel.
    head_dim,
    eps,
    q_weight,
    k_weight,
    q_add_weight,
    k_add_weight,
    cos_emb,
    sin_emb,
    num_txt_tokens,
):
    """
    PyTorch reference implementation of fused DiT QK Norm + RoPE.

    Applies per-head RMSNorm (with dual-stream weight selection) and
    interleaved RoPE using precomputed cos/sin embeddings.

    Args:
        qkv: [num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]
        cos_emb: [num_tokens, head_dim] float32, precomputed cos
        sin_emb: [num_tokens, head_dim] float32, precomputed sin
        num_txt_tokens: text boundary; tokens [0, num_txt) use add_weights
    """
    num_tokens = qkv.shape[0]
    q_size = num_heads_q * head_dim
    k_size = num_heads_k * head_dim

    q = qkv[:, :q_size].clone()
    k = qkv[:, q_size : q_size + k_size].clone()
    v = qkv[:, q_size + k_size :].clone()

    # Per-head RMSNorm: reshape to [T, H, D], norm over last dim
    q_4d = q.view(num_tokens, num_heads_q, head_dim)
    k_4d = k.view(num_tokens, num_heads_k, head_dim)

    if num_txt_tokens > 0 and q_add_weight is not None:
        # Dual-stream: text tokens use add_weights, image tokens use primary
        txt_q = F.rms_norm(q_4d[:num_txt_tokens].float(), (head_dim,), q_add_weight.float(), eps)
        img_q = F.rms_norm(q_4d[num_txt_tokens:].float(), (head_dim,), q_weight.float(), eps)
        q_4d = torch.cat([txt_q, img_q], dim=0).to(q.dtype)

        txt_k = F.rms_norm(k_4d[:num_txt_tokens].float(), (head_dim,), k_add_weight.float(), eps)
        img_k = F.rms_norm(k_4d[num_txt_tokens:].float(), (head_dim,), k_weight.float(), eps)
        k_4d = torch.cat([txt_k, img_k], dim=0).to(k.dtype)
    else:
        q_4d = F.rms_norm(q_4d.float(), (head_dim,), q_weight.float(), eps).to(q.dtype)
        k_4d = F.rms_norm(k_4d.float(), (head_dim,), k_weight.float(), eps).to(k.dtype)

    q = q_4d.reshape(num_tokens, q_size)
    k = k_4d.reshape(num_tokens, k_size)

    # Interleaved RoPE with precomputed cos/sin.
    # Expand cos/sin [T, D] to match per-head: [T, H*D]
    cos_q = cos_emb.unsqueeze(1).expand(-1, num_heads_q, -1).reshape(num_tokens, q_size)
    sin_q = sin_emb.unsqueeze(1).expand(-1, num_heads_q, -1).reshape(num_tokens, q_size)
    cos_k = cos_emb.unsqueeze(1).expand(-1, num_heads_k, -1).reshape(num_tokens, k_size)
    sin_k = sin_emb.unsqueeze(1).expand(-1, num_heads_k, -1).reshape(num_tokens, k_size)

    # Interleaved rotation: pair (element[2i], element[2i+1])
    q_rot = torch.empty_like(q, dtype=torch.float32)
    q_rot[:, 0::2] = -q[:, 1::2].float()
    q_rot[:, 1::2] = q[:, 0::2].float()
    q = (q.float() * cos_q + q_rot * sin_q).to(qkv.dtype)

    k_rot = torch.empty_like(k, dtype=torch.float32)
    k_rot[:, 0::2] = -k[:, 1::2].float()
    k_rot[:, 1::2] = k[:, 0::2].float()
    k = (k.float() * cos_k + k_rot * sin_k).to(qkv.dtype)

    return torch.cat([q, k, v], dim=1)


def _generate_cos_sin(num_tokens, head_dim, device):
    """Generate paired cos/sin embeddings matching FLUX's repeat_interleave format."""
    # Generate D/2 random frequencies and repeat-interleave to get D values
    half_dim = head_dim // 2
    freqs = torch.randn(num_tokens, half_dim, device=device, dtype=torch.float32)
    freqs = freqs.repeat_interleave(2, dim=-1)  # [T, D] with paired values
    return freqs.cos(), freqs.sin()


head_dims = [64, 128, 256]
# FLUX uses equal Q/K/V heads
num_heads_list = [
    24,  # FLUX.1
    48,  # Larger variant
]
num_tokens_list = [1, 64, 512]
dual_stream_configs = [
    (-1, False),  # No dual-stream
    (128, True),  # 128 text tokens
    (256, True),  # 256 text tokens
]
dtypes = [torch.bfloat16]


@pytest.mark.parametrize("head_dim", head_dims)
@pytest.mark.parametrize("num_heads", num_heads_list)
@pytest.mark.parametrize("num_tokens", num_tokens_list)
@pytest.mark.parametrize("dual_stream_config", dual_stream_configs)
@pytest.mark.parametrize("dtype", dtypes)
def test_fused_dit_qk_norm_rope(head_dim, num_heads, num_tokens, dual_stream_config, dtype):
    """
    Test fused DiT QK Norm + RoPE kernel against PyTorch reference.

    Verifies:
    1. Per-head RMSNorm with correct weight selection (primary vs add)
    2. Interleaved RoPE with precomputed cos/sin
    3. V portion unchanged
    4. Dual-stream weight switching at text boundary
    """
    device = "cuda"
    num_txt_tokens, has_add_weights = dual_stream_config

    # Skip invalid: text tokens can't exceed total tokens
    if num_txt_tokens >= num_tokens:
        pytest.skip("num_txt_tokens >= num_tokens")

    # FLUX has equal Q/K/V heads
    num_heads_q = num_heads
    num_heads_k = num_heads
    num_heads_v = num_heads

    hidden_size = (num_heads_q + num_heads_k + num_heads_v) * head_dim

    torch.random.manual_seed(42)
    qkv = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    qkv_copy = qkv.clone()

    # RMSNorm weights
    q_weight = torch.randn(head_dim, dtype=dtype, device=device) * 5.0
    k_weight = torch.randn(head_dim, dtype=dtype, device=device) * 5.0

    q_add_weight = None
    k_add_weight = None
    if has_add_weights:
        q_add_weight = torch.randn(head_dim, dtype=dtype, device=device) * 5.0
        k_add_weight = torch.randn(head_dim, dtype=dtype, device=device) * 5.0

    eps = 1e-6

    # Precomputed cos/sin embeddings
    cos_emb, sin_emb = _generate_cos_sin(num_tokens, head_dim, device)

    # Run fused kernel (in-place)
    torch.ops.trtllm.fused_dit_qk_norm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        q_add_weight,
        k_add_weight,
        cos_emb,
        sin_emb,
        num_txt_tokens,
    )
    output = qkv

    # Compute reference
    ref_output = torch_ref_dit_qk_norm_rope(
        qkv_copy,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        q_add_weight,
        k_add_weight,
        cos_emb,
        sin_emb,
        num_txt_tokens,
    )

    torch.testing.assert_close(
        output,
        ref_output,
        rtol=1e-2,
        atol=5e-3,
    )


@pytest.mark.parametrize("head_dim", [64, 128])
def test_fused_dit_qk_norm_rope_v_unchanged(head_dim):
    """Verify that the V portion of QKV is not modified by the kernel."""
    device = "cuda"
    num_heads = 24
    num_tokens = 64
    hidden_size = 3 * num_heads * head_dim

    torch.random.manual_seed(0)
    qkv = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    v_size = num_heads * head_dim
    v_original = qkv[:, -v_size:].clone()

    q_weight = torch.randn(head_dim, dtype=torch.bfloat16, device=device)
    k_weight = torch.randn(head_dim, dtype=torch.bfloat16, device=device)
    cos_emb, sin_emb = _generate_cos_sin(num_tokens, head_dim, device)

    torch.ops.trtllm.fused_dit_qk_norm_rope(
        qkv,
        num_heads,
        num_heads,
        num_heads,
        head_dim,
        1e-6,
        q_weight,
        k_weight,
        None,
        None,
        cos_emb,
        sin_emb,
        -1,
    )

    torch.testing.assert_close(qkv[:, -v_size:], v_original, rtol=0, atol=0)
