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

"""
Tests for fuse_rope_attention transformation.
"""

from collections import namedtuple

import pytest
import torch
from torch.export import Dim

# Import modules to register custom ops (torch.ops.auto_deploy.*)
import tensorrt_llm._torch.auto_deploy.custom_ops.torch_attention  # noqa: F401
import tensorrt_llm._torch.auto_deploy.custom_ops.torch_rope  # noqa: F401
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer

torch.manual_seed(0)


def _get_cos_sin(x: torch.Tensor, head_dim):
    """Precompute cos and sin for RoPE."""
    batch_size = x.size(0)
    seq_len = x.size(1)
    cos = x.new_zeros(batch_size, seq_len, head_dim)
    sin = x.new_zeros(batch_size, seq_len, head_dim)
    return cos, sin


class RopeAttentionModel(torch.nn.Module):
    """
    Model that implements the rope + attention pattern that fuse_rope_attention expects.

    Pattern:
    1. q_proj, k_proj, v_proj
    2. view to [batch, seq, num_heads, head_dim]
    3. contiguous
    4. torch_rope_with_explicit_cos_sin for q and k
    5. torch_attention with layout="bsnd"
    """

    def __init__(
        self,
        head_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        max_seq_len: int = 128,
    ):
        super().__init__()
        hidden_size = head_dim * num_q_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_q_heads
        self.max_seq_len = max_seq_len

        # Linear projections
        self.q_proj = torch.nn.Linear(hidden_size, num_q_heads * self.head_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_q_heads * self.head_dim, hidden_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,  # Required for export API compatibility
    ) -> torch.Tensor:
        x = input_ids.unsqueeze(-1).expand(-1, -1, self.hidden_size).to(torch.float16)
        batch_size, seq_len, _ = x.shape

        # Generate q, k, v: [batch, seq, hidden_size] -> [batch, seq, num_heads * head_dim]
        q = self.q_proj(x)  # [batch, seq, num_q_heads * head_dim]
        k = self.k_proj(x)  # [batch, seq, num_kv_heads * head_dim]
        v = self.v_proj(x)  # [batch, seq, num_kv_heads * head_dim]

        # View to [batch, seq, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_q_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply contiguous (this is part of the pattern)
        q = torch.ops.aten.contiguous(q)
        k = torch.ops.aten.contiguous(k)

        # Precompute cos and sin for RoPE
        cos, sin = _get_cos_sin(x, self.head_dim)

        # Apply RoPE with unsqueeze_dim=2 for BSND layout
        # NOTE(yoco): This should be (q, k) according to the definition of torch_rope_with_explicit_cos_sin.
        # However, the actual graph is (k, q), Need further investigation.
        q_rot, k_rot = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(k, q, cos, sin, 2)

        # Apply attention with layout="bsnd"
        attn_output = torch.ops.auto_deploy.torch_attention(
            k_rot,
            q_rot,
            v,
            None,  # attn_mask
            0.0,  # dropout_p
            True,  # is_causal
            None,  # scale
            None,  # sinks
            None,  # sliding_window
            None,  # logit_cap
            "bsnd",  # layout
        )

        # Reshape back and apply output projection
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)
        return output

    def _apply_rope_manual(self, q, k, cos, sin):
        """Manual rope application for fallback/comparison."""

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        cos = cos.unsqueeze(2)  # unsqueeze_dim=2 for BSND
        sin = sin.unsqueeze(2)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def get_dynamic_shapes(self):
        return [
            {0: Dim("batch_size", max=8), 1: Dim("seq_len", max=128)},
            {0: Dim("batch_size", max=8), 1: Dim("seq_len", max=128)},
        ]


def convert_placeholder_meta_to_cuda(gm: torch.fx.GraphModule):
    """Convert placeholder meta to cuda."""
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            node.meta["val"] = node.meta["val"].to("cuda")


def _run_test(
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    batch_size: int,
    seq_len: int,
):
    """Helper function to run the transformation test."""
    model = RopeAttentionModel(
        head_dim=head_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
    ).to("cuda", torch.float16)
    input_ids = torch.randint(0, 10, (batch_size, seq_len), device="cuda", dtype=torch.int32)
    position_ids = torch.randint(0, 10, (batch_size, seq_len), device="cuda", dtype=torch.int32)
    dynamic_shapes = model.get_dynamic_shapes()

    # Export to graph module
    args = (input_ids, position_ids)
    gm = torch_export_to_gm(model, args=args, dynamic_shapes=dynamic_shapes, clone=True)

    # Model config object, include num_attention_heads and hidden_size
    class Factory:
        def _get_model_config(self):
            ModelConfig = namedtuple(
                "ModelConfig", ["num_attention_heads", "hidden_size", "num_key_value_heads"]
            )
            return ModelConfig(
                num_attention_heads=num_q_heads,
                hidden_size=head_dim * num_q_heads,
                num_key_value_heads=num_kv_heads,
            ), None

    # Apply fuse_rope_attention transformation

    optimizer = InferenceOptimizer(
        Factory(),
        {
            "fuse_rope_attention": {
                "stage": "pattern_matcher",
            },
        },
    )
    convert_placeholder_meta_to_cuda(gm)
    gm_transformed = optimizer(None, gm)
    gm_transformed.to("cuda")

    fused_nodes = gm_transformed.graph.find_nodes(
        op="call_function", target=torch.ops.auto_deploy.torch_onnx_attention_plugin.default
    )
    assert len(fused_nodes) == 1, "Expected 1 AttentionPlugin node, got {len(fused_nodes)}"
    input_nodes = gm_transformed.graph.find_nodes(op="placeholder")
    assert len(input_nodes) == 5, "Expected 5 input nodes, got {len(input_nodes)}"
    input_nodes_targets = {node.target for node in input_nodes}
    assert input_nodes_targets == {
        "input_ids",
        "context_lengths",
        "rope_rotary_cos_sin",
        "kvcache_start_index",
        "past_key_values_0",
    }
    out_node = gm_transformed.graph.find_nodes(op="output")[0]
    assert len(out_node.args[0]) == 2, "Expected 2 output nodes, got {len(out_node.args[0])}"


@pytest.mark.parametrize(
    "head_dim,num_q_heads,num_kv_heads,batch_size,seq_len",
    [
        # GQA (Grouped-Query Attention): num_q_heads > num_kv_heads
        pytest.param(64, 14, 2, 2, 16, id="gqa_ratio_7"),
    ],
)
@torch.inference_mode()
def test_fuse_rope_attention(
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    batch_size: int,
    seq_len: int,
):
    """Test fuse_rope_attention transformation with various configurations."""
    _run_test(
        head_dim=head_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        batch_size=batch_size,
        seq_len=seq_len,
    )
