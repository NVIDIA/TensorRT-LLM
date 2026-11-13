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
"""Rotary embedding of the n-sample shared-prefix library.

The library rotates Q and the step's K itself when the model fuses RoPE into the attention op
(Qwen2, Llama, ...); these tests check the rotation against an fp32 reference around page
boundaries and the cases the library declines. The EAGLE-3 end-to-end test in
``test_sampling_shared_prefix_fmha.py`` runs Qwen2's fused RoPE through the library.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha.flash_attn4_kernels import fused_rope_supported
from tensorrt_llm._torch.attention.backends.fmha.sampling_shared_prefix import (
    SamplingSharedPrefixFmha,
)
from tensorrt_llm._torch.attention.backends.interface import RopeParams
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

TPB = 32


def _rope_params(dim, theta=10000.0, scale_type=RotaryScalingType.none, scale=1.0, positions=4096):
    return RopeParams(
        dim=dim, theta=theta, scale_type=scale_type, scale=scale, max_positions=positions
    )


def _reference_rope(x, positions, rotary_dim, theta, scale):
    """HF-style NeoX rotation in fp32. ``x``: [tokens, heads, head_dim]."""
    half = rotary_dim // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, rotary_dim, 2, device=x.device, dtype=torch.float64) / rotary_dim)
    )
    angle = positions.double()[:, None] * inv_freq[None, :] / scale
    cos = torch.cos(angle).float()[:, None, :]
    sin = torch.sin(angle).float()[:, None, :]
    xf = x.float()
    out = xf.clone()
    x1, x2 = xf[..., :half], xf[..., half:rotary_dim]
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:rotary_dim] = x2 * cos + x1 * sin
    return out


def _make_qkv(tokens, num_heads, num_kv_heads, head_dim, seed, row_pad=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    width = (num_heads + 2 * num_kv_heads) * head_dim
    buf = torch.randn(tokens, width + row_pad, generator=gen, device="cuda").bfloat16()
    return buf[:, :width]  # a strided view, like a slice of a larger activation


def _rope_case_kv_lens(q_len):
    # Device KV lengths around page boundaries, including the first token of a
    # page (kv_len - q_len + j on a multiple of the page size) and position 0.
    return [q_len, q_len + 1, TPB, TPB + 1, 2 * TPB - 1, 2 * TPB, 2 * TPB + q_len, 1000, 3999]


@needs_gpu
@pytest.mark.parametrize(
    "num_heads, num_kv_heads, head_dim, rotary_dim, q_len, scale_type, scale",
    [
        (20, 20, 128, 128, 9, RotaryScalingType.none, 1.0),
        (14, 2, 64, 64, 4, RotaryScalingType.none, 1.0),
        (14, 2, 128, 64, 4, RotaryScalingType.linear, 4.0),
    ],
)
def test_rope_write_kv_paged_matches_fp32_reference(
    num_heads, num_kv_heads, head_dim, rotary_dim, q_len, scale_type, scale
):
    """The library's RoPE launch (``rope_write_kv_paged`` over the rows' private pages) against an
    HF-style fp32 rotation at the absolute positions ``kv_len - q_len + j``, around page
    boundaries: Q rotated in place, the rotated K and the V written at slot ``page_len - q_len + j``
    of the private pages, dimensions past the rotary dim untouched."""
    from tensorrt_llm._torch.attention.backends.fmha.flash_attn4_kernels import rope_write_kv_paged

    theta = 1000000.0
    _, table = _rope_params(rotary_dim, theta, scale_type, scale).create_rope_const_params()
    kv_lens_list = _rope_case_kv_lens(q_len)
    rows = len(kv_lens_list)
    # Two shared pages precede each row's private pages (but not for rows shorter than them).
    shared = [2 * TPB if n - q_len >= 2 * TPB else 0 for n in kv_lens_list]
    page_lens_list = [n - s for n, s in zip(kv_lens_list, shared)]
    kv_lens = torch.tensor(kv_lens_list, dtype=torch.int32, device="cuda")
    page_lens = torch.tensor(page_lens_list, dtype=torch.int32, device="cuda")
    width = max(-(-n // TPB) for n in page_lens_list)
    page_table = torch.arange(rows * width, dtype=torch.int32, device="cuda").view(rows, width)
    pool = torch.zeros(rows * width, 2, num_kv_heads, TPB, head_dim, device="cuda").bfloat16()
    qkv = _make_qkv(rows * q_len, num_heads, num_kv_heads, head_dim, seed=3, row_pad=8)
    original = qkv.clone()
    rope_write_kv_paged(
        qkv,
        table,
        pool,
        page_table,
        kv_lens,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        q_len=q_len,
        page_lens=page_lens,
    )
    torch.cuda.synchronize()
    positions = (kv_lens.long()[:, None] - q_len + torch.arange(q_len, device="cuda")).reshape(-1)
    q_size, kv_size = num_heads * head_dim, num_kv_heads * head_dim
    tokens = rows * q_len
    q_before = original[:, :q_size].view(tokens, num_heads, head_dim)
    q_after = qkv[:, :q_size].view(tokens, num_heads, head_dim)
    expected_q = _reference_rope(q_before, positions, rotary_dim, theta, scale)
    # One bf16 rounding of an fp32 rotation.
    torch.testing.assert_close(q_after.float(), expected_q, atol=2e-2, rtol=1e-2)
    assert torch.equal(q_after[..., rotary_dim:], q_before[..., rotary_dim:])
    k_before = original[:, q_size : q_size + kv_size].view(tokens, num_kv_heads, head_dim)
    v_before = original[:, q_size + kv_size :].view(tokens, num_kv_heads, head_dim)
    expected_k = _reference_rope(k_before, positions, rotary_dim, theta, scale)
    for r in range(rows):
        for j in range(q_len):
            slot = page_lens_list[r] - q_len + j
            page = int(page_table[r, slot // TPB])
            k_cached = pool[page, 0, :, slot % TPB].float()
            torch.testing.assert_close(k_cached, expected_k[r * q_len + j], atol=2e-2, rtol=1e-2)
            assert torch.equal(pool[page, 1, :, slot % TPB], v_before[r * q_len + j])


def _fake_attn(
    position_type=PositionEmbeddingType.rope_gpt_neox, rope=None, head_dim=64, table=True, **extra
):
    rope = rope if rope is not None else _rope_params(head_dim)
    attrs = dict(
        is_mla_enable=False,
        sparse_params=None,
        predicted_tokens_per_seq=1,
        attention_chunk_size=None,
        head_dim=head_dim,
        num_heads=14,
        num_kv_heads=2,
        quant_mode=0,
        position_embedding_type=int(position_type),
        rope_params=rope,
        rotary_cos_sin=(
            torch.zeros(rope.max_positions * rope.dim, dtype=torch.float32, device="cuda")
            if table
            else None
        ),
    )
    attrs.update(extra)
    return SimpleNamespace(**attrs)


@needs_gpu
def test_library_accepts_supported_rope_and_declines_the_rest():
    ok = RotaryScalingType.none
    assert SamplingSharedPrefixFmha._is_available(_fake_attn(rope=_rope_params(64, scale_type=ok)))
    assert SamplingSharedPrefixFmha._is_available(
        _fake_attn(rope=_rope_params(64, scale_type=RotaryScalingType.linear, scale=2.0))
    )
    # Partial rotary dim.
    assert SamplingSharedPrefixFmha._is_available(_fake_attn(head_dim=128, rope=_rope_params(64)))
    # A module that rotates by itself hands no pos_embd_params.
    assert SamplingSharedPrefixFmha._is_available(
        _fake_attn(position_type=PositionEmbeddingType.learned_absolute)
    )

    for scale_type in (
        RotaryScalingType.yarn,
        RotaryScalingType.llama3,
        RotaryScalingType.dynamic,
        RotaryScalingType.longrope,
        RotaryScalingType.mrope,
    ):
        attn = _fake_attn(rope=_rope_params(64, scale_type=scale_type, scale=2.0))
        assert not fused_rope_supported(attn), scale_type
        assert not SamplingSharedPrefixFmha._is_available(attn), scale_type
    for position_type in (
        PositionEmbeddingType.rope_gptj,
        PositionEmbeddingType.long_rope,
        PositionEmbeddingType.mrope,
        PositionEmbeddingType.yarn,
        PositionEmbeddingType.alibi,
        PositionEmbeddingType.alibi_with_scale,
        PositionEmbeddingType.relative,
        PositionEmbeddingType.chatglm,
        PositionEmbeddingType.deferred,
    ):
        assert not SamplingSharedPrefixFmha._is_available(_fake_attn(position_type=position_type))
    # No table, rotary dim past the head or not a power of two.
    assert not SamplingSharedPrefixFmha._is_available(_fake_attn(table=False))
    assert not SamplingSharedPrefixFmha._is_available(
        _fake_attn(head_dim=64, rope=_rope_params(128))
    )
    assert not SamplingSharedPrefixFmha._is_available(
        _fake_attn(head_dim=128, rope=_rope_params(96))
    )
