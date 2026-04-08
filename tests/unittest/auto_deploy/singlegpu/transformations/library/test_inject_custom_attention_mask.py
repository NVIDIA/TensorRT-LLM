# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for semantic multimodal mask lowering during cached-attention insertion."""

import pytest
import torch

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op
from tensorrt_llm.llmapi.llm_args import KvCacheConfig


class SpanMaskedAttentionModel(torch.nn.Module):
    """Minimal model whose source graph uses a semantic multimodal mask op."""

    def __init__(self, hidden_size: int = 8, num_heads: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.unsqueeze(-1).expand(-1, -1, self.hidden_size).to(torch.float32)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        mm_token_positions: torch.Tensor,
        mm_token_lengths: torch.Tensor,
        mm_item_cu_seqlen: torch.Tensor,
    ) -> torch.Tensor:
        del position_ids
        x = self._embed(input_ids)
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        attn_mask = torch.ops.auto_deploy.gemma4_multimodal_mask.default(
            input_ids,
            mm_token_positions,
            mm_token_lengths,
            mm_item_cu_seqlen,
        )
        return torch.ops.auto_deploy.torch_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=False,
            layout="bsnd",
        )


def _create_seq_info() -> CachedSequenceInterface:
    return CachedSequenceInterface(
        max_seq_len=16,
        max_batch_size=4,
        device="cpu",
        kv_cache_config=KvCacheConfig(tokens_per_block=16),
    )


@torch.inference_mode()
def test_source_graph_uses_semantic_multimodal_mask_op():
    model = SpanMaskedAttentionModel().eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64)
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.int64).repeat(input_ids.shape[0], 1)
    mm_token_positions = torch.tensor([1, 3], dtype=torch.int32)
    mm_token_lengths = torch.tensor([2, 2], dtype=torch.int32)
    mm_item_cu_seqlen = torch.tensor([0, 2], dtype=torch.int32)

    gm = torch_export_to_gm(
        model,
        args=(
            input_ids,
            position_ids,
            mm_token_positions,
            mm_token_lengths,
            mm_item_cu_seqlen,
        ),
        clone=True,
    )

    semantic_nodes = [
        node for node in gm.graph.nodes if is_op(node, torch.ops.auto_deploy.gemma4_multimodal_mask)
    ]
    assert len(semantic_nodes) == 1


@torch.inference_mode()
def test_insert_cached_attention_lowers_semantic_mask_for_torch_backend():
    model = SpanMaskedAttentionModel().eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64)
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.int64).repeat(input_ids.shape[0], 1)
    mm_token_positions = torch.tensor([1, 3], dtype=torch.int32)
    mm_token_lengths = torch.tensor([2, 2], dtype=torch.int32)
    mm_item_cu_seqlen = torch.tensor([0, 2], dtype=torch.int32)

    gm = torch_export_to_gm(
        model,
        args=(
            input_ids,
            position_ids,
            mm_token_positions,
            mm_token_lengths,
            mm_item_cu_seqlen,
        ),
        clone=True,
    )
    cm = _create_seq_info()
    gm_transformed = InferenceOptimizer(
        None,
        {
            "insert_cached_attention": {
                "stage": "cache_init",
                "backend": "torch",
            },
        },
    )(cm, gm)

    assert not any(
        is_op(node, torch.ops.auto_deploy.torch_attention) for node in gm_transformed.graph.nodes
    )
    assert any(
        is_op(node, torch.ops.auto_deploy.gemma4_prepare_multimodal_mask)
        for node in gm_transformed.graph.nodes
    )

    cached_nodes = [
        node
        for node in gm_transformed.graph.nodes
        if is_op(node, torch.ops.auto_deploy.torch_cached_attention_with_cache)
    ]
    assert len(cached_nodes) == 1
    assert is_op(cached_nodes[0].args[-1], torch.ops.auto_deploy.gemma4_prepare_multimodal_mask)


@torch.inference_mode()
def test_insert_cached_attention_lowers_semantic_mask_for_triton_paged_backend():
    model = SpanMaskedAttentionModel().eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64)
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.int64).repeat(input_ids.shape[0], 1)
    mm_token_positions = torch.tensor([1, 3], dtype=torch.int32)
    mm_token_lengths = torch.tensor([2, 2], dtype=torch.int32)
    mm_item_cu_seqlen = torch.tensor([0, 2], dtype=torch.int32)

    gm = torch_export_to_gm(
        model,
        args=(
            input_ids,
            position_ids,
            mm_token_positions,
            mm_token_lengths,
            mm_item_cu_seqlen,
        ),
        clone=True,
    )
    cm = _create_seq_info()
    gm_transformed = InferenceOptimizer(
        None,
        {
            "insert_cached_attention": {
                "stage": "cache_init",
                "backend": "triton_paged",
            },
        },
    )(cm, gm)

    assert any(
        is_op(node, torch.ops.auto_deploy.gemma4_prepare_multimodal_mask)
        for node in gm_transformed.graph.nodes
    )
    cached_nodes = [
        node
        for node in gm_transformed.graph.nodes
        if is_op(node, torch.ops.auto_deploy.triton_paged_mha_with_cache)
    ]
    assert len(cached_nodes) == 1
    assert is_op(cached_nodes[0].args[-1], torch.ops.auto_deploy.gemma4_prepare_multimodal_mask)


@torch.inference_mode()
def test_insert_cached_attention_rejects_unsupported_semantic_mask_backend():
    model = SpanMaskedAttentionModel().eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64)
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.int64).repeat(input_ids.shape[0], 1)
    mm_token_positions = torch.tensor([1, 3], dtype=torch.int32)
    mm_token_lengths = torch.tensor([2, 2], dtype=torch.int32)
    mm_item_cu_seqlen = torch.tensor([0, 2], dtype=torch.int32)

    gm = torch_export_to_gm(
        model,
        args=(
            input_ids,
            position_ids,
            mm_token_positions,
            mm_token_lengths,
            mm_item_cu_seqlen,
        ),
        clone=True,
    )
    cm = _create_seq_info()

    with pytest.raises(
        RuntimeError,
        match=(
            "Cached attention backend 'flashinfer' does not support lowering semantic mask op"
            ".*gemma4_multimodal_mask.*Supported backends: torch, triton_paged"
        ),
    ):
        InferenceOptimizer(
            None,
            {
                "insert_cached_attention": {
                    "stage": "cache_init",
                    "backend": "flashinfer",
                },
            },
        )(cm, gm)
