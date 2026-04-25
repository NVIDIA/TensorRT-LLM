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

import pytest
import torch
import torch.nn as nn

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4Config,
)
from tensorrt_llm._torch.auto_deploy.transform.interface import SharedConfig, Stages
from tensorrt_llm._torch.auto_deploy.transform.library.sharding_ir import ApplyShardingHints
from tensorrt_llm._torch.auto_deploy.utils.dist_config import DistConfig
from tensorrt_llm._torch.auto_deploy.utils.node_utils import extract_op_args, is_op


class _SparseAttentionShardingFixture(nn.Module):
    def __init__(self, num_heads: int = 8) -> None:
        super().__init__()
        self.attn_sink = nn.Parameter(torch.arange(num_heads, dtype=torch.float32))

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        topk_idxs: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention.default(
            q,
            kv,
            self.attn_sink,
            topk_idxs,
            0.5,
            enable_sharding=True,
            layer_type="mla",
        )


class _SparseAttentionV2ShardingFixture(nn.Module):
    def __init__(self, num_heads: int = 8) -> None:
        super().__init__()
        self.attn_sink = nn.Parameter(torch.arange(num_heads, dtype=torch.float32))

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        topk_idxs: torch.Tensor,
        compressor_kv: torch.Tensor,
        compressor_gate: torch.Tensor,
        compressor_ape: torch.Tensor,
        compressor_norm_weight: torch.Tensor,
        freqs_cis_table: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_v2.default(
            q,
            kv,
            self.attn_sink,
            topk_idxs,
            compressor_kv,
            compressor_gate,
            compressor_ape,
            compressor_norm_weight,
            freqs_cis_table,
            position_ids,
            0.5,
            enable_sharding=True,
            layer_type="mla",
            window_size=3,
            compress_ratio=0,
            max_compressed_len=None,
            head_dim=4,
            rope_dim=2,
        )


def _trace_with_meta(module: nn.Module) -> torch.fx.GraphModule:
    gm = torch.fx.symbolic_trace(module)
    params_and_buffers = dict(gm.named_parameters())
    params_and_buffers.update(dict(gm.named_buffers()))
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if node.target == "q":
                node.meta["val"] = torch.empty((2, 3, 8, 4), dtype=torch.bfloat16)
            elif node.target == "kv":
                node.meta["val"] = torch.empty((2, 5, 4), dtype=torch.bfloat16)
            elif node.target == "topk_idxs":
                node.meta["val"] = torch.empty((2, 3, 2), dtype=torch.int32)
            elif node.target in {"compressor_kv", "compressor_gate"}:
                node.meta["val"] = torch.empty((2, 5, 0), dtype=torch.bfloat16)
            elif node.target == "compressor_ape":
                node.meta["val"] = torch.empty((0, 0), dtype=torch.bfloat16)
            elif node.target == "compressor_norm_weight":
                node.meta["val"] = torch.empty((0,), dtype=torch.bfloat16)
            elif node.target == "freqs_cis_table":
                node.meta["val"] = torch.empty((8, 1), dtype=torch.complex64)
            elif node.target == "position_ids":
                node.meta["val"] = torch.empty((2, 5), dtype=torch.long)
        elif node.op == "get_attr" and node.target in params_and_buffers:
            node.meta["val"] = params_and_buffers[node.target].detach()
        elif is_op(
            node,
            (
                torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention,
                torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_v2,
            ),
        ):
            node.meta["val"] = torch.empty((2, 3, 8, 4), dtype=torch.bfloat16)
    return gm


def _run_ir_sharding(
    gm: torch.fx.GraphModule,
    *,
    rank: int,
    world_size: int,
) -> tuple[torch.fx.GraphModule, object]:
    transform = ApplyShardingHints.from_kwargs(stage=Stages.SHARDING)
    shared_config = SharedConfig(
        local_rank=rank,
        world_size=world_size,
        dist_config=DistConfig(
            world_size=world_size,
            rank=rank,
            tp_size=world_size,
            moe_ep_size=world_size,
        ),
    )
    return transform._apply(gm, None, None, shared_config)


@pytest.mark.parametrize(
    "fixture_cls",
    [
        pytest.param(_SparseAttentionShardingFixture, id="legacy-source-op"),
        pytest.param(_SparseAttentionV2ShardingFixture, id="v2-source-op"),
    ],
)
def test_deepseek_v4_sparse_attention_shards_attn_sink_without_collective(
    fixture_cls: type[nn.Module],
) -> None:
    gm = _trace_with_meta(fixture_cls())

    transformed, info = _run_ir_sharding(gm, rank=1, world_size=2)

    assert info.num_matches == 1
    torch.testing.assert_close(
        transformed.attn_sink,
        torch.tensor([4.0, 5.0, 6.0, 7.0]),
        rtol=0,
        atol=0,
    )
    assert not any(
        node.op == "call_function" and "all_reduce" in str(node.target)
        for node in transformed.graph.nodes
    )

    load_result = transformed.load_state_dict(
        {"attn_sink": torch.arange(8, dtype=torch.float32).add(10)},
        strict=False,
    )

    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []
    torch.testing.assert_close(
        transformed.attn_sink,
        torch.tensor([14.0, 15.0, 16.0, 17.0]),
        rtol=0,
        atol=0,
    )


def test_deepseek_v4_attention_emits_head_group_views_and_sparse_attention_hint() -> None:
    config = DeepseekV4Config(
        vocab_size=16,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        head_dim=4,
        q_lora_rank=8,
        qk_rope_head_dim=2,
        o_groups=2,
        o_lora_rank=4,
        sliding_window=2,
        compress_ratios=(0,),
        ad_rope_cache_len=8,
    )
    attention = DeepseekV4Attention(config, layer_idx=0)
    gm = torch_export_to_gm(
        attention,
        args=(
            torch.randn(1, 4, 16, dtype=torch.bfloat16),
            torch.arange(4, dtype=torch.long).unsqueeze(0),
        ),
    )

    view_nodes = [node for node in gm.graph.nodes if is_op(node, torch.ops.auto_deploy.view)]
    sparse_node = next(
        node
        for node in gm.graph.nodes
        if is_op(node, torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_v2)
    )

    assert len(view_nodes) >= 2
    assert any(extract_op_args(node, "tp_scaled_dim")[0] == 2 for node in view_nodes)
    assert extract_op_args(sparse_node, "enable_sharding")[0] is True
    assert any(is_op(node, torch.ops.auto_deploy.all_reduce) for node in gm.graph.nodes)
