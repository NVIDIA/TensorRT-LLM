# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionPatchEmbed

from tensorrt_llm._torch.models import modeling_qwen2vl
from tensorrt_llm._torch.models.modeling_qwen2vl import (
    Qwen2_5_VisionModel,
    Qwen2_5_VLVisionAttention,
)


@pytest.mark.cpu_only
def test_qwen2_5_vision_patch_projection_matches_conv3d(monkeypatch) -> None:
    patch_embed = Qwen2_5_VisionPatchEmbed(
        patch_size=4,
        temporal_patch_size=2,
        in_channels=3,
        embed_dim=8,
    )
    pixel_values = torch.randn(6, 3 * 2 * 4 * 4)
    expected = patch_embed(pixel_values)
    # The model must reuse the Conv3d parameters as a GEMM, not run the Conv3d.
    patch_embed.proj.register_forward_pre_hook(
        lambda module, args: pytest.fail("patch projection ran the Conv3d")
    )

    vision = Qwen2_5_VisionModel.__new__(Qwen2_5_VisionModel)
    torch.nn.Module.__init__(vision)
    vision.patch_embed = patch_embed
    vision.spatial_merge_unit = 1
    vision._rope_position_ids_buffer = None
    vision.full_attn_metadata = object()
    vision.window_attn_metadata = object()
    vision._full_attn_max_seq_len = 6
    vision._window_attn_max_seq_len = 6
    vision.fullatt_block_indexes = []
    vision.blocks = torch.nn.ModuleList()
    vision.merger = torch.nn.Identity()
    vision.get_rotary_pos_emb_window_data = lambda grid: (
        [torch.empty(6, 0)],
        [torch.empty(6, 0)],
        [torch.arange(6)],
        [6],
    )
    vision.prepare_attn_metadata = lambda seq_lens, metadata, **kwargs: metadata
    monkeypatch.setattr(
        modeling_qwen2vl,
        "async_tensor_h2d",
        lambda tensor, *, dtype, device: tensor.to(dtype=dtype, device=device),
    )

    actual = vision(pixel_values, torch.tensor([[1, 2, 3]]))

    torch.testing.assert_close(actual, expected)


@pytest.mark.cpu_only
def test_qwen2_5_vision_attention_reuses_fused_qkv(monkeypatch) -> None:
    # Force the PyTorch RoPE fallback; FlashInfer and flash_attn already
    # rotate Q/K in place.
    monkeypatch.setattr(modeling_qwen2vl, "_flash_attn_apply_rotary", None)

    attention = Qwen2_5_VLVisionAttention.__new__(Qwen2_5_VLVisionAttention)
    torch.nn.Module.__init__(attention)
    attention.head_dim = 4
    attention.q_size = 8
    attention.kv_size = 8
    attention.support_fused_qkv = True
    attention.layer_idx = 0

    fused_qkv = torch.randn(3, 24)
    q, k, _ = fused_qkv.split(8, dim=-1)
    original_q = q.reshape(3, 2, 4).clone()
    original_k = k.reshape(3, 2, 4).clone()
    cos = torch.randn(3, 2)
    sin = torch.randn(3, 2)
    expected_q = modeling_qwen2vl.RotaryEmbedding.apply_rotary_pos_emb(
        original_q.unsqueeze(0), cos, sin, unsqueeze_dim=1
    ).squeeze(0)
    expected_k = modeling_qwen2vl.RotaryEmbedding.apply_rotary_pos_emb(
        original_k.unsqueeze(0), cos, sin, unsqueeze_dim=1
    ).squeeze(0)
    attention.qkv_proj = lambda hidden_states: fused_qkv
    forwarded = {}

    def forward_impl(**kwargs):
        forwarded.update(kwargs)
        return kwargs["q"]

    attention.forward_impl = forward_impl
    attention.o_proj = lambda output, layer_idx: output

    output = attention.forward(
        torch.empty(3, 8),
        attn_metadata=object(),
        position_embeddings=(cos, sin),
    )

    actual_q, actual_k, _ = fused_qkv.split(8, dim=-1)
    torch.testing.assert_close(actual_q.reshape_as(expected_q), expected_q)
    torch.testing.assert_close(actual_k.reshape_as(expected_k), expected_k)
    assert forwarded["q"] is fused_qkv
    assert forwarded["k"] is None
    assert forwarded["v"] is None
    assert output is fused_qkv
