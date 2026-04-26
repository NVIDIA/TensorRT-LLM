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
    _fake_fp4_activation_quant_dequant,
)
from tensorrt_llm._torch.auto_deploy.transform.interface import SharedConfig, Stages
from tensorrt_llm._torch.auto_deploy.transform.library.quantization import (
    FineGrainedFP8LinearQuantization,
)
from tensorrt_llm._torch.auto_deploy.transform.library.sharding_ir import ApplyShardingHints
from tensorrt_llm._torch.auto_deploy.utils._graph import run_shape_prop
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


class _IndexerFp4ViewShapePropFixture(nn.Module):
    def __init__(self, num_heads: int = 64, head_dim: int = 128) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = q.shape
        q = torch.ops.auto_deploy.view(
            q,
            [batch_size, seq_len, self.num_heads, self.head_dim],
            tp_scaled_dim=2,
            layer_type="mla",
        )
        return _fake_fp4_activation_quant_dequant(q)


class _DummyFactory:
    def __init__(self, quant_config: dict[str, object]) -> None:
        self._quant_config = quant_config

    def get_quant_config(self) -> dict[str, object]:
        return self._quant_config


class _DeepSeekV4AttentionTp8Fixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seq_len = 4
        self.hidden_size = 4096
        self.head_dim = 512
        self.o_lora_rank = 128
        self.local_wo_a_group_dim = 4096
        self.layers = nn.ModuleList([nn.Module()])
        config = DeepseekV4Config(
            vocab_size=16,
            hidden_size=self.hidden_size,
            num_hidden_layers=1,
            num_attention_heads=64,
            head_dim=self.head_dim,
            q_lora_rank=32,
            qk_rope_head_dim=64,
            o_groups=8,
            o_lora_rank=self.o_lora_rank,
            sliding_window=128,
            compress_ratios=(4,),
            index_n_heads=64,
            index_head_dim=128,
            index_topk=512,
            ad_rope_cache_len=8,
            ad_compress_max_seq_len=8192,
        )
        self.layers[0].attn = DeepseekV4Attention(config, layer_idx=0)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return self.layers[0].attn(x, position_ids)


class _DeepSeekV4GroupedWoAUnsupportedTpFixture(nn.Module):
    def __init__(
        self,
        num_groups: int = 4,
        o_lora_rank: int = 128,
        group_dim: int = 16,
    ) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.o_lora_rank = o_lora_rank
        self.group_dim = group_dim
        self.wo_a = nn.Module()
        weight = torch.arange(
            num_groups * o_lora_rank * group_dim,
            dtype=torch.float32,
        )
        weight = weight.reshape(num_groups * o_lora_rank, group_dim)
        self.wo_a.weight = nn.Parameter(weight.to(torch.float8_e4m3fn), requires_grad=False)
        self.wo_a.register_buffer(
            "weight_scale_inv",
            torch.ones(num_groups, 1, dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_fake_quant_deepseek_v4_wo_a_grouped_finegrained_fp8_linear.default(
            x,
            self.wo_a.weight,
            None,
            [self.wo_a.weight_scale_inv],
            layer_type="mla",
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


def _set_placeholder_meta(gm: torch.fx.GraphModule, target: str, value: torch.Tensor) -> None:
    for node in gm.graph.nodes:
        if node.op != "placeholder" or node.target != target:
            continue
        fake_mode = getattr(node.meta.get("val"), "fake_mode", None)
        node.meta["val"] = (
            fake_mode.from_tensor(value, static_shapes=True) if fake_mode is not None else value
        )
        return
    raise AssertionError(f"placeholder {target!r} not found")


def _trace_grouped_wo_a_with_meta(
    module: _DeepSeekV4GroupedWoAUnsupportedTpFixture,
    input_group_count: int | None = None,
) -> torch.fx.GraphModule:
    gm = torch.fx.symbolic_trace(module)
    params_and_buffers = dict(gm.named_parameters())
    params_and_buffers.update(dict(gm.named_buffers()))
    input_shape = (1, 2, input_group_count or module.num_groups, module.group_dim)
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            node.meta["val"] = torch.empty(input_shape, dtype=torch.bfloat16)
        elif node.op == "get_attr" and node.target in params_and_buffers:
            node.meta["val"] = params_and_buffers[node.target].detach()
        elif is_op(
            node,
            torch.ops.auto_deploy.torch_fake_quant_deepseek_v4_wo_a_grouped_finegrained_fp8_linear,
        ):
            node.meta["val"] = torch.empty(
                (*input_shape[:-1], module.o_lora_rank),
                dtype=torch.bfloat16,
            )
    return gm


def _run_finegrained_transform(
    gm: torch.fx.GraphModule,
    quant_config: dict[str, object],
) -> tuple[torch.fx.GraphModule, object]:
    transform = FineGrainedFP8LinearQuantization.from_kwargs(stage=Stages.PATTERN_MATCHER)
    return transform._apply(gm, None, _DummyFactory(quant_config), SharedConfig())


def _deepseek_v4_quant_config() -> dict[str, object]:
    return {
        "quant_method": "deepseek_v4_fp8",
        "linear_quant_method": "finegrained_fp8",
        "weight_block_size": [128, 128],
        "exclude_modules": [
            "embed",
            "head",
            "*.ffn.gate",
            "*.attn.compressor",
            "*.attn.indexer.compressor",
            "*.attn.indexer.weights_proj",
            "*.norm",
            "*.hc_*",
            "*.attn_sink",
            "mtp.*",
        ],
    }


def _node_depends_on(node: torch.fx.Node, target: torch.fx.Node) -> bool:
    def _child_nodes(value: object) -> list[torch.fx.Node]:
        if isinstance(value, torch.fx.Node):
            return [value]
        if isinstance(value, (list, tuple)):
            nodes = []
            for item in value:
                nodes.extend(_child_nodes(item))
            return nodes
        if isinstance(value, dict):
            nodes = []
            for item in value.values():
                nodes.extend(_child_nodes(item))
            return nodes
        return []

    pending = [node]
    seen = set()
    while pending:
        current = pending.pop()
        if current is target:
            return True
        if current in seen:
            continue
        seen.add(current)
        pending.extend(_child_nodes(current.args))
        pending.extend(_child_nodes(current.kwargs))
    return False


def _is_dist_all_reduce_node(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and "dist_all_reduce" in str(node.target)


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


def test_deepseek_v4_sparse_attention_tp8_shards_local_heads_and_attn_sink() -> None:
    gm = _trace_with_meta(_SparseAttentionShardingFixture(num_heads=64))

    transformed, info = _run_ir_sharding(gm, rank=5, world_size=8)

    assert info.num_matches == 1
    torch.testing.assert_close(
        transformed.attn_sink,
        torch.arange(40, 48, dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    sparse_node = next(
        node
        for node in transformed.graph.nodes
        if is_op(node, torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention)
    )
    q_node = sparse_node.args[0]
    assert q_node.meta["val"].shape[-2] == 8
    assert not any(_is_dist_all_reduce_node(node) for node in transformed.graph.nodes)


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


def test_deepseek_v4_indexer_fp4_shape_prop_after_tp_sharding() -> None:
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
        compress_ratios=(4,),
        index_n_heads=4,
        index_head_dim=32,
        index_topk=2,
        ad_rope_cache_len=8,
        ad_compress_max_seq_len=8,
    )
    attention = DeepseekV4Attention(config, layer_idx=0)
    gm = torch_export_to_gm(
        attention,
        args=(
            torch.randn(1, 8, 16, dtype=torch.bfloat16),
            torch.arange(8, dtype=torch.long).unsqueeze(0),
        ),
    )

    transformed, info = _run_ir_sharding(gm, rank=1, world_size=2)

    assert info.num_matches > 0
    assert transformed.get_parameter("indexer.wq_b.weight").shape == (64, 8)


def test_deepseek_v4_indexer_fp4_quant_uses_local_shape_after_tp_sharding() -> None:
    batch_size = 1
    seq_len = 8
    full_heads = 64
    head_dim = 128
    tp_size = 8
    local_heads = full_heads // tp_size
    fixture = _IndexerFp4ViewShapePropFixture(full_heads, head_dim)
    gm = torch_export_to_gm(
        fixture,
        args=(torch.randn(batch_size, seq_len, full_heads * head_dim, dtype=torch.bfloat16),),
    )

    transformed, info = _run_ir_sharding(gm, rank=1, world_size=tp_size)
    local_q = torch.randn(batch_size, seq_len, local_heads * head_dim, dtype=torch.bfloat16)
    _set_placeholder_meta(transformed, "q", local_q)

    assert info.num_matches == 1
    run_shape_prop(transformed)
    transformed.recompile()
    assert transformed(local_q).shape == (batch_size, seq_len, local_heads, head_dim)


def test_deepseek_v4_attention_tp8_ratio4_sharding_places_groups_and_collectives() -> None:
    model = _DeepSeekV4AttentionTp8Fixture().to(torch.bfloat16)
    gm = torch_export_to_gm(
        model,
        args=(
            torch.randn(1, model.seq_len, model.hidden_size, dtype=torch.bfloat16),
            torch.arange(model.seq_len, dtype=torch.long).unsqueeze(0),
        ),
    )

    quantized, quant_info = _run_finegrained_transform(gm, _deepseek_v4_quant_config())
    transformed, shard_info = _run_ir_sharding(quantized, rank=5, world_size=8)

    assert quant_info.num_matches == 6
    assert shard_info.num_matches >= 10

    attn = transformed.get_submodule("layers.0.attn")
    assert attn.attn_sink.shape == (8,)
    assert attn.wq_b.weight.shape[0] // model.head_dim == 8
    assert attn.wo_a.weight.shape == (model.o_lora_rank, model.local_wo_a_group_dim)
    assert attn.wo_a.weight_scale_inv.shape == (1, model.local_wo_a_group_dim // 128)
    assert attn.wo_b.weight.shape == (4096, model.o_lora_rank)
    assert attn.indexer.weights_proj.weight.shape == (8, 4096)

    grouped_node = next(
        node
        for node in transformed.graph.nodes
        if is_op(
            node,
            torch.ops.auto_deploy.torch_fake_quant_deepseek_v4_wo_a_grouped_finegrained_fp8_linear,
        )
    )
    local_group_view = grouped_node.args[0]
    assert local_group_view.args[1] == [1, model.seq_len, 1, -1]

    sparse_node = next(
        node
        for node in transformed.graph.nodes
        if is_op(node, torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_v2)
    )
    [
        q,
        topk_idxs,
        window_size,
        compress_ratio,
        max_compressed_len,
        head_dim,
        rope_dim,
    ] = extract_op_args(
        sparse_node,
        "q",
        "topk_idxs",
        "window_size",
        "compress_ratio",
        "max_compressed_len",
        "head_dim",
        "rope_dim",
    )
    assert tuple(q.meta["val"].shape) == (1, model.seq_len, 8, 512)
    assert tuple(topk_idxs.meta["val"].shape) == (1, model.seq_len, 640)
    assert window_size == 128
    assert compress_ratio == 4
    assert max_compressed_len == 2048
    assert head_dim == 512
    assert rope_dim == 64

    wo_b_node = next(
        node
        for node in transformed.graph.nodes
        if is_op(node, torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear)
        and node.args[1].target == "layers.0.attn.wo_b.weight"
    )
    assert _node_depends_on(wo_b_node.args[0], grouped_node)
    assert any(
        _is_dist_all_reduce_node(node) and node.args[0] is wo_b_node
        for node in transformed.graph.nodes
    )

    topk_node = next(
        node
        for node in transformed.graph.nodes
        if node.op == "call_function"
        and node.target == torch.ops.aten.topk.default
        and "indexer" in node.name
    )
    indexer_score_reduce = next(
        node
        for node in transformed.graph.nodes
        if _is_dist_all_reduce_node(node) and _node_depends_on(topk_node.args[0], node)
    )
    assert "indexer" in indexer_score_reduce.name


def test_deepseek_v4_grouped_wo_a_rejects_tp8_when_o_groups_smaller_than_tp_size() -> None:
    gm = _trace_grouped_wo_a_with_meta(_DeepSeekV4GroupedWoAUnsupportedTpFixture())

    with pytest.raises(AssertionError, match="o_groups >= tp_size"):
        _run_ir_sharding(gm, rank=0, world_size=8)


def test_deepseek_v4_grouped_wo_a_accepts_tp8_local_group_metadata() -> None:
    module = _DeepSeekV4GroupedWoAUnsupportedTpFixture(
        num_groups=8,
        o_lora_rank=128,
        group_dim=64,
    )
    gm = _trace_grouped_wo_a_with_meta(module, input_group_count=1)

    transformed, info = _run_ir_sharding(gm, rank=5, world_size=8)

    assert info.num_matches == 1
    assert transformed.wo_a.weight.shape == (module.o_lora_rank, module.group_dim)
    assert transformed.wo_a.weight_scale_inv.shape == (1, 1)
    grouped_node = next(
        node
        for node in transformed.graph.nodes
        if is_op(
            node,
            torch.ops.auto_deploy.torch_fake_quant_deepseek_v4_wo_a_grouped_finegrained_fp8_linear,
        )
    )
    assert grouped_node.args[0].op == "placeholder"
    assert not any(
        node.op == "call_function" and node.target == torch.ops.aten.slice.Tensor
        for node in transformed.graph.nodes
    )
