# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for custom attention mask injection into torch_attention nodes."""

from types import SimpleNamespace

import torch

import tensorrt_llm._torch.auto_deploy.custom_ops.attention.torch_attention  # noqa: F401
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.attention_mask_provider import (
    AttentionMaskProviderRegistry,
)
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op


class DualAttentionModel(torch.nn.Module):
    """Minimal model with two torch_attention calls sharing one custom mask."""

    def __init__(self, hidden_size: int = 8, num_heads: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj_1 = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj_1 = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj_1 = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_proj_2 = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj_2 = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj_2 = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.unsqueeze(-1).expand(-1, -1, self.hidden_size).to(torch.float32)

    def _run_attention(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        def _project(q_proj, k_proj, v_proj):
            q = q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
            v = v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
            return torch.ops.auto_deploy.torch_attention(
                q, k, v, attn_mask=attn_mask, is_causal=False, layout="bsnd"
            )

        return _project(self.q_proj_1, self.k_proj_1, self.v_proj_1) + _project(
            self.q_proj_2, self.k_proj_2, self.v_proj_2
        )

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        del position_ids
        return self._run_attention(self._embed(input_ids), attn_mask=None)

    def forward_with_mask(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        return self._run_attention(self._embed(input_ids), attn_mask=attn_mask)


class DummyFactory:
    def _get_model_config(self):
        return SimpleNamespace(model_type="unit_test_mask_model"), {}


class Gemma4Factory:
    def _get_model_config(self):
        return SimpleNamespace(model_type="gemma4"), {}


def _build_segment_mask(segment_ids: torch.Tensor) -> torch.Tensor:
    same_segment = segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)
    return same_segment.unsqueeze(1)


def _build_token_type_mask(token_type_ids: torch.Tensor) -> torch.Tensor:
    non_text = token_type_ids != 0
    prev = torch.cat(
        [
            torch.zeros(token_type_ids.shape[0], 1, dtype=token_type_ids.dtype),
            token_type_ids[:, :-1],
        ],
        dim=1,
    )
    blob_starts = non_text & (token_type_ids != prev)
    blob_ids = torch.cumsum(blob_starts.to(torch.int64), dim=1)
    token_blob_ids = torch.where(non_text, blob_ids, torch.zeros_like(blob_ids))
    media_mask = (token_blob_ids.unsqueeze(2) == token_blob_ids.unsqueeze(1)) & (
        token_blob_ids.unsqueeze(2) != 0
    )
    positions = torch.arange(token_type_ids.shape[1])
    causal_mask = positions.unsqueeze(0) <= positions.unsqueeze(1)
    return (causal_mask.unsqueeze(0) | media_mask).unsqueeze(1)


_provider_build_counts = {"mask": 0}


@AttentionMaskProviderRegistry.register("unit_test_mask_model", "torch_attention")
def _segment_mask_provider(ctx, source_attn_node):
    del source_attn_node

    def _builder():
        _provider_build_counts["mask"] += 1
        segment_ids = ctx.add_or_retrieve_input(
            "segment_ids",
            activate_arg=False,
            val=torch.zeros(2, 4, dtype=torch.int64),
        )
        seg_q = ctx.gm.graph.call_function(torch.ops.aten.unsqueeze.default, args=(segment_ids, 2))
        seg_k = ctx.gm.graph.call_function(torch.ops.aten.unsqueeze.default, args=(segment_ids, 1))
        same_segment = ctx.gm.graph.call_function(torch.ops.aten.eq.Tensor, args=(seg_q, seg_k))
        return ctx.gm.graph.call_function(torch.ops.aten.unsqueeze.default, args=(same_segment, 1))

    return ctx.get_or_create_cached_node("segment_ids_mask", _builder)


@torch.inference_mode()
def test_inject_custom_attention_mask():
    model = DualAttentionModel().eval()
    input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.int64)
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.int64).repeat(input_ids.shape[0], 1)
    segment_ids = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 2]], dtype=torch.int64)

    gm = torch_export_to_gm(model, args=(input_ids, position_ids), clone=True)

    _provider_build_counts["mask"] = 0
    gm_transformed = InferenceOptimizer(
        DummyFactory(),
        {
            "inject_custom_attention_mask": {
                "stage": "pattern_matcher",
                "backend": "torch_attention",
            },
        },
    )(None, gm)

    attn_nodes = [
        node
        for node in gm_transformed.graph.nodes
        if is_op(node, torch.ops.auto_deploy.torch_attention)
    ]
    assert len(attn_nodes) == 2
    assert _provider_build_counts["mask"] == 1

    mask_nodes = [
        node.args[3] if len(node.args) > 3 else node.kwargs["attn_mask"] for node in attn_nodes
    ]
    assert mask_nodes[0] is mask_nodes[1]

    placeholder_targets = {
        node.target for node in gm_transformed.graph.nodes if node.op == "placeholder"
    }
    assert "segment_ids" in placeholder_targets

    expected_mask = _build_segment_mask(segment_ids)
    expected = model.forward_with_mask(input_ids, expected_mask)
    actual = gm_transformed(input_ids, position_ids, segment_ids=segment_ids)

    torch.testing.assert_close(actual, expected)


@torch.inference_mode()
def test_inject_gemma4_custom_attention_mask_for_torch_backend():
    model = DualAttentionModel().eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64)
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.int64).repeat(input_ids.shape[0], 1)
    token_type_ids = torch.tensor([[0, 1, 1, 2, 2]], dtype=torch.int64)

    gm = torch_export_to_gm(model, args=(input_ids, position_ids), clone=True)
    gm_transformed = InferenceOptimizer(
        Gemma4Factory(),
        {
            "inject_custom_attention_mask": {
                "stage": "pattern_matcher",
                "backend": "torch",
            },
        },
    )(None, gm)

    attn_nodes = [
        node
        for node in gm_transformed.graph.nodes
        if is_op(node, torch.ops.auto_deploy.torch_attention)
    ]
    assert len(attn_nodes) == 2
    assert all(
        (node.args[3] if len(node.args) > 3 else node.kwargs["attn_mask"]) is not None
        for node in attn_nodes
    )

    expected_mask = _build_token_type_mask(token_type_ids)
    expected = model.forward_with_mask(input_ids, expected_mask)
    actual = gm_transformed(input_ids, position_ids, token_type_ids=token_type_ids)
    torch.testing.assert_close(actual, expected)


@torch.inference_mode()
def test_inject_gemma4_custom_attention_mask_for_triton_paged_backend():
    model = DualAttentionModel().eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64)
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.int64).repeat(input_ids.shape[0], 1)
    token_type_ids = torch.tensor([[0, 1, 1, 2, 2]], dtype=torch.int64)

    gm = torch_export_to_gm(model, args=(input_ids, position_ids), clone=True)
    gm_transformed = InferenceOptimizer(
        Gemma4Factory(),
        {
            "inject_custom_attention_mask": {
                "stage": "pattern_matcher",
                "backend": "triton_paged",
            },
        },
    )(None, gm)

    attn_nodes = [
        node
        for node in gm_transformed.graph.nodes
        if is_op(node, torch.ops.auto_deploy.torch_attention)
    ]
    assert len(attn_nodes) == 2
    assert all(
        (node.args[3] if len(node.args) > 3 else node.kwargs["attn_mask"]) is not None
        for node in attn_nodes
    )

    expected_mask = _build_token_type_mask(token_type_ids)
    expected = model.forward_with_mask(input_ids, expected_mask)
    actual = gm_transformed(input_ids, position_ids, token_type_ids=token_type_ids)
    torch.testing.assert_close(actual, expected)
