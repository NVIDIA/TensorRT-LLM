# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from transformers import PretrainedConfig, Qwen3Config

from tensorrt_llm._torch.attention.backends.interface import PositionalEmbeddingParams, RopeParams
from tensorrt_llm._torch.attention.qk_norm_attention import QKNormRoPEAttention
from tensorrt_llm._torch.attention.rotary_embedding import MRotaryEmbedding, RotaryEmbedding
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm.functional import PositionEmbeddingType


@pytest.mark.parametrize(
    "dtype,split_input,is_qk_norm",
    [
        pytest.param(torch.float16, False, True, id="fp16-packed"),
        pytest.param(torch.float16, True, True, id="fp16-split"),
        pytest.param(torch.bfloat16, False, True, id="bf16-packed"),
        pytest.param(torch.bfloat16, True, True, id="bf16-split"),
        pytest.param(torch.float16, False, False, id="fp16-rope-only"),
    ],
)
@torch.inference_mode()
def test_qk_norm_rope_attention_dtype_dispatch(
    dtype: torch.dtype, split_input: bool, is_qk_norm: bool
) -> None:
    head_dim, num_heads, num_kv_heads = 128, 8, 2
    eps, theta, max_positions = 1e-6, 1_000_000.0, 256
    config = PretrainedConfig(torch_dtype=dtype, head_dim=head_dim, rms_norm_eps=eps)
    with torch.device("cuda"):
        attention = QKNormRoPEAttention(
            hidden_size=num_heads * head_dim,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            max_position_embeddings=max_positions,
            bias=False,
            dtype=dtype,
            config=ModelConfig(pretrained_config=config, skip_create_weights_in_init=True),
            pos_embd_params=PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gpt_neox,
                rope=RopeParams(dim=head_dim, theta=theta, max_positions=max_positions),
            ),
            is_qk_norm=is_qk_norm,
        )
    assert attention.fuse_qk_norm_rope and not attention.rope_fusion
    generator = torch.Generator(device="cuda").manual_seed(0)
    qkv = torch.randn(
        7,
        (num_heads + 2 * num_kv_heads) * head_dim,
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    positions = torch.tensor([0, 1, 7, 31, 100, 127, 255], device="cuda", dtype=torch.int32)
    if is_qk_norm:
        attention.q_norm.weight.uniform_(0.5, 1.5, generator=generator)
        attention.k_norm.weight.uniform_(0.5, 1.5, generator=generator)

    with patch.object(attention, "apply_qk_norm_rope", wraps=attention.apply_qk_norm_rope) as fused:
        if split_input:
            inputs = attention.split_qkv(qkv.clone())
            output = attention.apply_rope(*inputs, positions)
        else:
            q, k, v, gate = attention.preprocess_qkv(qkv.clone(), positions)
            assert gate is None
            output = (q, k, v)
        assert fused.call_count == int(dtype == torch.bfloat16)
    actual_q, actual_k, actual_v = attention.split_qkv(*output)
    q, k, v = attention.split_qkv(qkv)
    torch.testing.assert_close(actual_v, v, rtol=0, atol=0)
    if dtype == torch.bfloat16:
        expected, _, _ = attention.apply_qk_norm_rope(qkv.clone(), positions)
        torch.testing.assert_close(
            torch.cat((actual_q, actual_k, actual_v), dim=-1), expected, rtol=0, atol=0
        )
        return

    inverse_frequency = theta ** (
        -torch.arange(0, head_dim, 2, device="cuda", dtype=torch.float32) / head_dim
    )
    angles = positions.float().unsqueeze(-1) * inverse_frequency
    cos, sin = angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)

    def reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x = x.reshape(7, -1, head_dim).float()
        if is_qk_norm:
            x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * weight.float()
            # The unfused normalization writes FP16 before RoPE reads it.
            x = x.to(dtype).float()
        first, second = x.chunk(2, dim=-1)
        return (
            torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)
            .to(dtype)
            .reshape(7, -1)
        )

    torch.testing.assert_close(
        actual_q, reference(q, attention.q_norm.weight), rtol=2e-3, atol=2e-3
    )
    torch.testing.assert_close(
        actual_k, reference(k, attention.k_norm.weight), rtol=2e-3, atol=2e-3
    )


@pytest.mark.parametrize("rope_type", ["yarn", "mrope"])
@pytest.mark.parametrize("fuse_qk_norm_rope", [True, False])
@torch.inference_mode()
def test_qk_norm_rope_attention_scaled_fallback(rope_type: str, fuse_qk_norm_rope: bool) -> None:
    rope_scaling = {"rope_type": rope_type, "factor": 1.0}
    if rope_type == "yarn":
        rope_scaling.update(
            factor=2.0, attention_factor=1.7, truncate=False, original_max_position_embeddings=128
        )
    config = Qwen3Config(
        torch_dtype=torch.float16,
        hidden_size=1024,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=128,
        rms_norm_eps=1e-6,
        max_position_embeddings=256,
        rope_theta=1_000_000.0,
        rope_scaling=rope_scaling,
    )
    with torch.device("cpu"):
        attention = QKNormRoPEAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=2,
            max_position_embeddings=config.max_position_embeddings,
            bias=False,
            dtype=torch.float16,
            fuse_qk_norm_rope=fuse_qk_norm_rope,
            rope_fusion=False,
            config=ModelConfig(pretrained_config=config, skip_create_weights_in_init=True),
            pos_embd_params=PositionalEmbeddingParams(
                type=PositionEmbeddingType.from_string(rope_type),
                rope=RopeParams.from_config(config),
                mrope_section=[16, 24, 24] if rope_type == "mrope" else None,
                mrope_interleaved=rope_type == "mrope",
            ),
        )
    attention = attention.cuda()
    assert attention.q_norm.weight.is_cuda and attention.k_norm.weight.is_cuda
    assert attention.rotary_emb.rotary_cos_sin.is_cuda
    qkv = torch.ones(7, 1536, device="cuda", dtype=torch.float16)
    original = qkv.clone()
    positions = torch.arange(7, device="cuda", dtype=torch.int32)
    if rope_type == "mrope":
        positions = positions.reshape(1, 1, -1).expand(3, 1, -1).contiguous()
    with (
        patch.object(attention, "apply_qk_norm", wraps=attention.apply_qk_norm) as norm,
        patch.object(attention, "apply_qk_norm_rope", wraps=attention.apply_qk_norm_rope) as fused,
        patch.object(attention.rotary_emb, "forward", wraps=attention.rotary_emb.forward) as rotary,
    ):
        if not fuse_qk_norm_rope:
            attention.preprocess_qkv(qkv, positions)
            norm.assert_called_once()
            fused.assert_not_called()
            rotary.assert_called_once()
            return
        with pytest.raises(NotImplementedError, match="only supports unscaled ordinary RoPE"):
            attention.preprocess_qkv(qkv, positions)
    norm.assert_not_called()
    fused.assert_not_called()
    rotary.assert_not_called()
    torch.testing.assert_close(qkv, original, rtol=0, atol=0)


@pytest.mark.cpu_only
@pytest.mark.parametrize("producer", ["norm_rope", "main_kv", "horizontal"])
def test_fp8_producer_meta_keeps_dynamic_num_tokens(producer: str) -> None:
    """All FP8 fake kernels must retain the symbolic token dimension."""
    with FakeTensorMode(shape_env=ShapeEnv()) as mode:
        num_tokens = mode.shape_env.create_unbacked_symint()
        qkv = torch.empty((num_tokens, 1280), dtype=torch.bfloat16)
        positions = torch.empty((num_tokens,), dtype=torch.int32)
        slots = torch.empty((num_tokens,), dtype=torch.int32)
        weight = torch.empty(128, dtype=torch.bfloat16)
        kv_cache = torch.empty((2, 2, 1, 128, 128), dtype=torch.float8_e4m3fn)
        if producer == "norm_rope":
            output = torch.ops.trtllm.fused_qk_norm_rope_to_fp8(
                qkv,
                8,
                1,
                1,
                128,
                64,
                1e-5,
                weight,
                weight,
                10000.0,
                True,
                positions,
                1.0,
                0.0,
                0.0,
                1.0,
                True,
                True,
                False,
                0,
                0,
            )
            outputs, tails = [output], [(1280,)]
        elif producer == "main_kv":
            output = torch.ops.trtllm.minimax_m3_fp8_qk_norm_rope_kv_insert(
                qkv,
                kv_cache,
                slots,
                8,
                1,
                1,
                128,
                64,
                1e-5,
                weight,
                weight,
                10000.0,
                True,
                positions,
            )
            outputs, tails = [output], [(8, 128)]
        else:
            packed = torch.empty((num_tokens, 1536), dtype=torch.bfloat16)
            index_cache = torch.empty((2, 1, 128, 128), dtype=torch.float8_e4m3fn)
            rope = torch.empty((16, 2, 32), dtype=torch.float32)
            outputs = torch.ops.trtllm.minimax_m3_fp8_qkv_indexer_norm_rope_kv_insert(
                packed,
                kv_cache,
                index_cache,
                slots,
                8,
                1,
                1,
                128,
                64,
                1e-5,
                weight,
                weight,
                weight,
                weight,
                rope,
                positions,
            )
            tails = [(8, 128), (1, 128)]

    for output, tail in zip(outputs, tails):
        assert isinstance(output.shape[0], torch.SymInt)
        assert output.shape[0].node.expr == num_tokens.node.expr
        assert output.shape[1:] == tail
        assert output.dtype == torch.float8_e4m3fn


@torch.inference_mode()
def torch_ref_rms_norm_rope(
    qkv,
    num_heads_q,
    num_heads_k,
    num_heads_v,
    head_dim,
    rotary_dim,
    eps,
    q_weight,
    k_weight,
    base,
    is_neox,
    position_ids,
    use_gemma=False,
):
    """
    PyTorch reference implementation of RMSNorm+RoPE for verification.

    Uses TensorRT-LLM's own RMSNorm and RotaryEmbedding modules to ensure consistency
    with the expected behavior of the fused kernel.

    Args:
        qkv: Combined QKV tensor of shape [num_tokens, hidden_size]
        num_heads_q: Number of query heads
        num_heads_k: Number of key heads
        num_heads_v: Number of value heads (unused for normalization/RoPE but needed for tensor splitting)
        head_dim: Dimension of each head
        rotary_dim: Dimension for RoPE
        eps: Epsilon value for RMS normalization
        q_weight: RMSNorm weights for query [head_dim]
        k_weight: RMSNorm weights for key [head_dim]
        base: Base value for RoPE calculations
        is_neox: Whether to use NeoX style RoPE
        position_ids: Position IDs for RoPE of shape [num_tokens]
        use_gemma: Whether QK norm is Gemma-style RMSNorm (scale by (1 + weight))

    Returns:
        Combined tensor with Q and K parts normalized and RoPE applied
    """
    # Get input shape information
    num_tokens = qkv.shape[0]
    hidden_size = qkv.shape[1]

    # Calculate dimensions for Q, K, V segments
    q_size = num_heads_q * head_dim
    k_size = num_heads_k * head_dim
    v_size = num_heads_v * head_dim

    # Verify dimensions match
    assert hidden_size == q_size + k_size + v_size, (
        f"Hidden size {hidden_size} doesn't match Q+K+V dimensions {q_size + k_size + v_size}"
    )

    # Split the tensor into Q, K, V parts
    q = qkv[:, :q_size]
    k = qkv[:, q_size : q_size + k_size]
    v = qkv[:, q_size + k_size :]

    # Create and apply RMSNorm modules with custom weights
    q_norm = (
        RMSNorm(hidden_size=head_dim, eps=eps, use_gemma=use_gemma).to(qkv.device).to(qkv.dtype)
    )
    k_norm = (
        RMSNorm(hidden_size=head_dim, eps=eps, use_gemma=use_gemma).to(qkv.device).to(qkv.dtype)
    )

    # Set the weights to the provided weights
    q_norm.weight.data.copy_(q_weight)
    k_norm.weight.data.copy_(k_weight)

    # Apply RMSNorm to Q and K
    q_normalized = q_norm(q.reshape(num_tokens * num_heads_q, head_dim)).reshape(num_tokens, q_size)
    k_normalized = k_norm(k.reshape(num_tokens * num_heads_k, head_dim)).reshape(num_tokens, k_size)

    # Create and apply RotaryEmbedding module
    rope_params = RopeParams(
        dim=rotary_dim,  # Set the rotary dimension
        theta=base,  # Base value for RoPE calculations
        max_positions=8192,  # Large enough for any reasonable hidden size
    )
    rotary_emb = RotaryEmbedding(rope_params=rope_params, head_dim=head_dim, is_neox=is_neox).to(
        qkv.device
    )

    # Apply RoPE to the normalized Q and K
    [q_rope, k_rope] = rotary_emb(position_ids, [q_normalized, k_normalized])

    # Combine Q, K, V back together
    result = torch.cat([q_rope, k_rope, v], dim=1)

    return result


head_dims = [64, 128]
# (Q heads, K heads, V heads)
num_heads_groups = [
    (16, 8, 8),  # Qwen3-0.6B, Qwen3-1.7B
    (32, 8, 8),  # Qwen3-4B, Qwen3-8B, Qwen3-30B-A3B
    (40, 8, 8),  # Qwen3-14B
    (64, 8, 8),  # Qwen3-32B, Qwen3-235B-A22B
    (24, 8, 8),  # GLM 4.6
]
num_tokens_list = [1, 3, 8, 32, 256]
is_neox_list = [False, True]
partial_rotary_factor_list = [1.0, 0.5]
dtypes = [torch.bfloat16]  # TODO: support float16


@pytest.mark.parametrize("head_dim", head_dims)
@pytest.mark.parametrize("num_heads_group", num_heads_groups)
@pytest.mark.parametrize("num_tokens", num_tokens_list)
@pytest.mark.parametrize("is_neox", is_neox_list)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("partial_rotary_factor", partial_rotary_factor_list)
def test_fused_qk_norm_rope(
    head_dim, num_heads_group, num_tokens, partial_rotary_factor, is_neox, dtype
):
    """
    Test the fused QK RMSNorm + RoPE operation with various configurations.

    This test verifies that the fused kernel correctly applies:
    1. RMSNorm to both query (Q) and key (K) portions of the QKV tensor
    2. Rotary Position Embeddings (RoPE) to the normalized Q and K
    3. Leaves the value (V) portion unchanged

    Args:
        head_dim: Dimension of each attention head
        num_heads_group: Tuple of (num_heads_q, num_heads_k, num_heads_v)
        num_tokens: Number of tokens to process
        dtype: Data type (float16 or bfloat16)
    """
    device = "cuda"
    torch_dtype = dtype

    # Unpack head counts
    num_heads_q, num_heads_k, num_heads_v = num_heads_group

    # Calculate total hidden dimension
    hidden_size = (num_heads_q + num_heads_k + num_heads_v) * head_dim

    # Generate random inputs directly as 2D [num_tokens, hidden_size]
    torch.random.manual_seed(0)
    qkv = torch.randn(num_tokens, hidden_size, dtype=torch_dtype, device=device)
    qkv_copy = qkv.clone()

    # Generate position IDs with +100 offset to test decoding scenarios
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device=device) + 100

    # Generate random weights for RMSNorm
    q_weight = torch.randn(head_dim, dtype=torch_dtype, device=device) * 5.0
    k_weight = torch.randn(head_dim, dtype=torch_dtype, device=device) * 5.0

    # Set RMSNorm and RoPE parameters
    eps = 1e-5
    base = 10000.0

    factor, low, high, attention_factor = 1.0, 0, 0, 1.0
    rotary_dim = int(head_dim * partial_rotary_factor)
    # Run the custom fusedQKNormRope operation
    torch.ops.trtllm.fused_qk_norm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        rotary_dim,
        eps,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids,
        factor,
        low,
        high,
        attention_factor,
        True,
        False,  # use_gemma (standard RMSNorm reference below)
        False,  # use_mrope (plain RoPE)
        0,  # mrope_section1 (unused when use_mrope=False)
        0,  # mrope_section2
    )
    output = qkv  # This op is inplace

    # Compute reference output using TensorRT LLM modules
    ref_output = torch_ref_rms_norm_rope(
        qkv_copy,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        rotary_dim,
        eps,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids,
    )

    # Compare outputs from custom kernel vs reference implementation
    torch.testing.assert_close(
        output,
        ref_output,
        rtol=5e-2,
        atol=1e-1,
    )


@torch.inference_mode()
def torch_ref_gemma_mrope(
    qkv,
    num_heads_q,
    num_heads_k,
    num_heads_v,
    head_dim,
    rotary_dim,
    eps,
    q_weight,
    k_weight,
    base,
    is_neox,
    position_ids_3d,
    mrope_section,
):
    """Reference for the Gemma-RMSNorm + interleaved-mRoPE fused path.

    Mirrors apply_qk_norm_rope's fused branch: Gemma RMSNorm (scale by
    (1 + weight)) on Q/K, then interleaved mRoPE via MRotaryEmbedding.
    """
    num_tokens = qkv.shape[0]
    q_size = num_heads_q * head_dim
    k_size = num_heads_k * head_dim
    q = qkv[:, :q_size]
    k = qkv[:, q_size : q_size + k_size]
    v = qkv[:, q_size + k_size :]

    q_norm = RMSNorm(hidden_size=head_dim, eps=eps, use_gemma=True).to(qkv.device).to(qkv.dtype)
    k_norm = RMSNorm(hidden_size=head_dim, eps=eps, use_gemma=True).to(qkv.device).to(qkv.dtype)
    q_norm.weight.data.copy_(q_weight)
    k_norm.weight.data.copy_(k_weight)
    q_n = q_norm(q.reshape(num_tokens * num_heads_q, head_dim)).reshape(num_tokens, q_size)
    k_n = k_norm(k.reshape(num_tokens * num_heads_k, head_dim)).reshape(num_tokens, k_size)

    rope_params = RopeParams(dim=rotary_dim, theta=base, max_positions=8192)
    rotary_emb = MRotaryEmbedding(
        rope_params=rope_params,
        head_dim=head_dim,
        mrope_section=mrope_section,
        is_neox=is_neox,
        mrope_interleaved=True,
    ).to(qkv.device)
    [q_rope, k_rope] = rotary_emb(position_ids_3d, [q_n, k_n])
    return torch.cat([q_rope, k_rope, v], dim=1)


@pytest.mark.skip(
    reason="WIP: standalone MRotaryEmbedding reference shape handling needs "
    "fixing. The kernel's Gemma + interleaved-mRoPE path is validated "
    "end-to-end by the Qwen3.5 accuracy test."
)
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("num_heads_group", [(16, 8, 8), (32, 8, 8)])
@pytest.mark.parametrize("num_tokens", [1, 3, 8, 256])
@pytest.mark.parametrize("is_neox", [False, True])
@pytest.mark.parametrize("partial_rotary_factor", [1.0, 0.5])
def test_fused_qk_norm_rope_gemma_mrope(
    head_dim, num_heads_group, num_tokens, partial_rotary_factor, is_neox
):
    """Cover the Gemma-RMSNorm + interleaved-mRoPE fused path (Qwen3.5)."""
    device = "cuda"
    dtype = torch.bfloat16
    num_heads_q, num_heads_k, num_heads_v = num_heads_group
    hidden_size = (num_heads_q + num_heads_k + num_heads_v) * head_dim

    torch.random.manual_seed(0)
    qkv = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    qkv_copy = qkv.clone()

    # 3D position_ids [3, num_tokens] with distinct t/h/w to exercise the
    # interleaved section selection (identical components would hide bugs).
    base_pos = torch.arange(num_tokens, dtype=torch.int32, device=device)
    position_ids_3d = torch.stack(
        [base_pos + 100, base_pos + 50, base_pos + 10], dim=0
    ).contiguous()

    q_weight = torch.randn(head_dim, dtype=dtype, device=device) * 5.0
    k_weight = torch.randn(head_dim, dtype=dtype, device=device) * 5.0

    eps = 1e-5
    base = 10000.0
    factor, low, high, attention_factor = 1.0, 0, 0, 1.0
    rotary_dim = int(head_dim * partial_rotary_factor)
    half = rotary_dim // 2
    s = half // 3
    mrope_section = [half - 2 * s, s, s]  # sums to half (rotary_dim/2)

    torch.ops.trtllm.fused_qk_norm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        rotary_dim,
        eps,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids_3d,
        factor,
        low,
        high,
        attention_factor,
        True,  # is_qk_norm
        True,  # use_gemma
        True,  # use_mrope
        mrope_section[1],
        mrope_section[2],
    )
    output = qkv

    ref_output = torch_ref_gemma_mrope(
        qkv_copy,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        rotary_dim,
        eps,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids_3d,
        mrope_section,
    )

    torch.testing.assert_close(output, ref_output, rtol=5e-2, atol=1e-1)


_FP8_NUM_HEADS_GROUPS = (
    (16, 8, 8),
    (32, 8, 8),
    (8, 1, 1),  # MiniMax-M3 sharded GQA (num_heads=8, num_kv_heads=1)
)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("num_heads_group", _FP8_NUM_HEADS_GROUPS)
@pytest.mark.parametrize("num_tokens", [1, 3, 8, 256])
@pytest.mark.parametrize("is_neox", [False, True])
@pytest.mark.parametrize("partial_rotary_factor", [1.0, 0.5])
# MiniMax-M3 stores every RMSNorm as a Gemma norm, so use_gemma=True is what ships.
@pytest.mark.parametrize("use_gemma", [False, True])
def test_fused_qk_norm_rope_to_fp8(
    head_dim, num_heads_group, num_tokens, partial_rotary_factor, is_neox, use_gemma
):
    """Test the FP8 out-variant of fused QK RMSNorm + RoPE.

    Checks that the input is left untouched, the output is FP8 E4M3 of the same
    shape, V is a bit-exact E4M3 cast of the input V, and Q/K match the BF16
    reference after dequantization.
    """
    device = "cuda"
    dtype = torch.bfloat16
    num_heads_q, num_heads_k, num_heads_v = num_heads_group
    hidden_size = (num_heads_q + num_heads_k + num_heads_v) * head_dim

    torch.random.manual_seed(0)
    qkv = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    qkv_ref = qkv.clone()

    position_ids = torch.arange(num_tokens, dtype=torch.int32, device=device) + 100
    q_weight = torch.randn(head_dim, dtype=dtype, device=device) * 5.0
    k_weight = torch.randn(head_dim, dtype=dtype, device=device) * 5.0

    eps = 1e-5
    base = 10000.0
    factor, low, high, attention_factor = 1.0, 0, 0, 1.0
    rotary_dim = int(head_dim * partial_rotary_factor)

    out_fp8 = torch.ops.trtllm.fused_qk_norm_rope_to_fp8(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        rotary_dim,
        eps,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids,
        factor,
        low,
        high,
        attention_factor,
        True,  # is_qk_norm
        use_gemma,
        False,  # use_mrope (plain RoPE)
        0,  # mrope_section1
        0,  # mrope_section2
    )

    assert out_fp8.dtype == torch.float8_e4m3fn
    assert tuple(out_fp8.shape) == (num_tokens, hidden_size)
    # Out-of-place: the BF16 input must be left byte-for-byte unchanged.
    torch.testing.assert_close(qkv, qkv_ref, rtol=0.0, atol=0.0)

    # V is copy-cast only, so it pins down the kernel's FP8 rounding against
    # torch's. The inputs stay well inside E4M3 range, so the two casts'
    # clamp/NaN behaviors don't diverge.
    v_start = (num_heads_q + num_heads_k) * head_dim
    torch.testing.assert_close(
        out_fp8[:, v_start:].float(),
        qkv_ref[:, v_start:].to(torch.float8_e4m3fn).float(),
        rtol=0.0,
        atol=0.0,
    )

    ref_output = torch_ref_rms_norm_rope(
        qkv_ref,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        rotary_dim,
        eps,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids,
        use_gemma=use_gemma,
    )

    # Dominated by the single E4M3 rounding: 3 mantissa bits give a worst-case
    # relative error of 2^-4 = 6.25%, plus the ~2^-9 the BF16 reference carries.
    torch.testing.assert_close(
        out_fp8.float(),
        ref_output.float(),
        rtol=0.07,
        atol=0.1,
    )
