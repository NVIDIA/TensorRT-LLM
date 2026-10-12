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


def _make_vision_attention(
    fused_qkv: torch.Tensor, *, head_dim: int, support_fused_qkv: bool
) -> tuple[Qwen2_5_VLVisionAttention, dict[str, object]]:
    attention = Qwen2_5_VLVisionAttention.__new__(Qwen2_5_VLVisionAttention)
    torch.nn.Module.__init__(attention)
    attention.head_dim = head_dim
    attention.q_size = fused_qkv.shape[-1] // 3
    attention.kv_size = attention.q_size
    attention.support_fused_qkv = support_fused_qkv
    attention.layer_idx = 0
    attention.qkv_proj = lambda hidden_states: fused_qkv
    forwarded = {}

    def forward_impl(**kwargs) -> torch.Tensor:
        forwarded.update(kwargs)
        return kwargs["q"]

    attention.forward_impl = forward_impl
    attention.o_proj = lambda output, layer_idx: output
    return attention, forwarded


@pytest.mark.cpu_only
@pytest.mark.parametrize("support_fused_qkv", [True, False])
@pytest.mark.parametrize("rope_impl", ["torch", "flash_attn", "flashinfer"])
def test_qwen2_5_vision_attention_reuses_fused_qkv(
    monkeypatch: pytest.MonkeyPatch, support_fused_qkv: bool, rope_impl: str
) -> None:
    # CPU stubs exercise kernel routing and mutation contracts, not CUDA kernels.
    monkeypatch.setattr(modeling_qwen2vl, "_flash_attn_apply_rotary", None)
    monkeypatch.setattr(modeling_qwen2vl, "IS_FLASHINFER_AVAILABLE", rope_impl == "flashinfer")
    head_dim = 64 if rope_impl == "flashinfer" else 4
    q_size = 2 * head_dim
    fused_qkv = torch.randn(3, 3 * q_size)
    attention, forwarded = _make_vision_attention(
        fused_qkv, head_dim=head_dim, support_fused_qkv=support_fused_qkv
    )
    q, k, original_v = attention.split_qkv(fused_qkv)
    original_q = q.reshape(3, 2, head_dim).clone()
    original_k = k.reshape(3, 2, head_dim).clone()
    original_v = original_v.clone()
    cos = torch.randn(3, head_dim // 2)
    sin = torch.randn(3, head_dim // 2)
    position_ids = torch.arange(3, dtype=torch.int32)
    kernel_calls = []

    def flash_attn_rope(
        tensor: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        interleaved: bool,
        inplace: bool,
    ) -> torch.Tensor:
        assert not interleaved
        assert inplace
        kernel_calls.append(tensor)
        tensor.copy_(
            modeling_qwen2vl.RotaryEmbedding.apply_rotary_pos_emb(tensor, cos, sin, unsqueeze_dim=1)
        )
        return tensor

    def flashinfer_rope(
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        head_size: int,
        cos_sin_cache: torch.Tensor,
        *,
        is_neox: bool,
    ) -> None:
        assert positions is position_ids
        assert head_size == 64
        assert is_neox
        assert cos_sin_cache.dtype == torch.float32
        assert cos_sin_cache.is_contiguous()
        torch.testing.assert_close(cos_sin_cache, torch.cat((cos, sin), dim=-1))
        kernel_calls.append(query)
        for tensor in (query, key):
            tensor.copy_(
                modeling_qwen2vl.RotaryEmbedding.apply_rotary_pos_emb(
                    tensor.view(1, 3, 2, head_dim), cos, sin, unsqueeze_dim=1
                ).reshape_as(tensor)
            )

    if rope_impl == "flash_attn":
        monkeypatch.setattr(modeling_qwen2vl, "_flash_attn_apply_rotary", flash_attn_rope)
    elif rope_impl == "flashinfer":
        monkeypatch.setattr(
            modeling_qwen2vl,
            "flashinfer_apply_rope_with_cos_sin_cache_inplace",
            flashinfer_rope,
            raising=False,
        )

    expected_q = modeling_qwen2vl.RotaryEmbedding.apply_rotary_pos_emb(
        original_q.unsqueeze(0), cos, sin, unsqueeze_dim=1
    ).squeeze(0)
    expected_k = modeling_qwen2vl.RotaryEmbedding.apply_rotary_pos_emb(
        original_k.unsqueeze(0), cos, sin, unsqueeze_dim=1
    ).squeeze(0)
    output = attention.forward(
        torch.empty(3, q_size),
        attn_metadata=object(),
        position_ids=position_ids,
        position_embeddings=(cos, sin),
    )

    actual_q, actual_k, actual_v = attention.split_qkv(fused_qkv)
    torch.testing.assert_close(actual_q.reshape_as(expected_q), expected_q)
    torch.testing.assert_close(actual_k.reshape_as(expected_k), expected_k)
    torch.testing.assert_close(actual_v, original_v)
    assert len(kernel_calls) == {"torch": 0, "flash_attn": 2, "flashinfer": 1}[rope_impl]
    if support_fused_qkv:
        assert forwarded["q"] is fused_qkv
        assert forwarded["k"] is None
        assert forwarded["v"] is None
        assert output is fused_qkv
    else:
        assert forwarded["q"].is_set_to(actual_q)
        assert forwarded["k"].is_set_to(actual_k)
        assert forwarded["v"].is_set_to(actual_v)
        assert output.is_set_to(actual_q)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "returned_view",
    ["equivalent", "fresh_q", "fresh_k", "fresh_v", "offset", "stride", "dtype", "mutated_stride"],
)
def test_qwen2_5_vision_attention_checks_rotated_qkv_views(returned_view: str) -> None:
    fused_qkv = torch.randn(3, 24)
    attention, forwarded = _make_vision_attention(fused_qkv, head_dim=4, support_fused_qkv=True)
    returned_qkv = []

    def apply_rope(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        position_ids: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if returned_view == "equivalent":
            q, k, v = q.view_as(q), k.view_as(k), v.view_as(v)
        elif returned_view == "fresh_q":
            q = q + 1
        elif returned_view == "fresh_k":
            k = k + 1
        elif returned_view == "fresh_v":
            v = v + 1
        elif returned_view == "offset":
            q = q.as_strided(q.shape, q.stride(), q.storage_offset() + 1)
        elif returned_view == "stride":
            q = q.as_strided(q.shape, (q.stride(0), 0))
        elif returned_view == "dtype":
            q = q.view(torch.int32)
        elif returned_view == "mutated_stride":
            q.as_strided_(q.shape, (q.stride(0), 0))
        returned_qkv.extend((q, k, v))
        return q, k, v

    attention.apply_rope = apply_rope
    output = attention.forward(torch.empty(3, 8), attn_metadata=object())

    if returned_view == "equivalent":
        assert output is fused_qkv
    else:
        assert output is not fused_qkv
        torch.testing.assert_close(output, torch.cat(returned_qkv, dim=-1))
    assert forwarded["q"] is output
    assert forwarded["k"] is None
    assert forwarded["v"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not modeling_qwen2vl.IS_FLASHINFER_AVAILABLE, reason="FlashInfer required")
@pytest.mark.parametrize("support_fused_qkv", [True, False])
def test_qwen2_5_vision_attention_flashinfer_inplace_cuda(
    monkeypatch: pytest.MonkeyPatch, support_fused_qkv: bool
) -> None:
    head_dim = 64
    fused_qkv = torch.randn(3, 6 * head_dim, device="cuda", dtype=torch.float16)
    original_qkv = fused_qkv.clone()
    attention, forwarded = _make_vision_attention(
        fused_qkv, head_dim=head_dim, support_fused_qkv=support_fused_qkv
    )
    q, k, v = attention.split_qkv(fused_qkv)
    original_v = v.clone()
    cos = torch.randn(3, head_dim // 2, device="cuda", dtype=torch.float32)
    sin = torch.randn_like(cos)
    expected_q = (
        modeling_qwen2vl.RotaryEmbedding.apply_rotary_pos_emb(
            q.float().view(1, 3, 2, head_dim), cos, sin, unsqueeze_dim=1
        )
        .reshape_as(q)
        .to(q.dtype)
    )
    expected_k = (
        modeling_qwen2vl.RotaryEmbedding.apply_rotary_pos_emb(
            k.float().view(1, 3, 2, head_dim), cos, sin, unsqueeze_dim=1
        )
        .reshape_as(k)
        .to(k.dtype)
    )
    original_rope = modeling_qwen2vl.flashinfer_apply_rope_with_cos_sin_cache_inplace
    kernel_calls = []

    def checked_rope(*args, **kwargs) -> None:
        original_rope(*args, **kwargs)
        kernel_calls.append(True)

    monkeypatch.setattr(modeling_qwen2vl, "_flash_attn_apply_rotary", None)
    monkeypatch.setattr(
        modeling_qwen2vl, "flashinfer_apply_rope_with_cos_sin_cache_inplace", checked_rope
    )
    output = attention.forward(
        torch.empty(3, 2 * head_dim, device="cuda", dtype=torch.float16),
        attn_metadata=object(),
        position_ids=torch.arange(3, device="cuda", dtype=torch.int32),
        position_embeddings=(cos, sin),
    )

    assert len(kernel_calls) == 1
    torch.testing.assert_close(q, expected_q, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(k, expected_k, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(v, original_v)
    if support_fused_qkv:
        assert output is fused_qkv
        assert forwarded["k"] is None
        assert forwarded["v"] is None
    else:
        assert output.is_set_to(q)
        assert forwarded["k"].is_set_to(k)
        assert forwarded["v"].is_set_to(v)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        attention.forward(
            torch.empty(3, 2 * head_dim, device="cuda", dtype=torch.float16),
            attn_metadata=object(),
            position_ids=torch.arange(3, device="cuda", dtype=torch.int32),
            position_embeddings=(cos, sin),
        )
    assert len(kernel_calls) == 2
    fused_qkv.copy_(original_qkv)
    graph.replay()
    torch.testing.assert_close(q, expected_q, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(k, expected_k, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(v, original_v)
