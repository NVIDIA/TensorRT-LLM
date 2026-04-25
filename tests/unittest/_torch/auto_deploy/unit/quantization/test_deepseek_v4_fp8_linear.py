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
import torch.nn.functional as F

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4Config,
)
from tensorrt_llm._torch.auto_deploy.transform.interface import SharedConfig, Stages
from tensorrt_llm._torch.auto_deploy.transform.library.quantization import (
    FineGrainedFP8LinearQuantization,
    FP8LinearQuantizationFromConfig,
)
from tensorrt_llm._torch.auto_deploy.transform.library.sharding_ir import ApplyShardingHints
from tensorrt_llm._torch.auto_deploy.utils.dist_config import DistConfig
from tensorrt_llm._torch.auto_deploy.utils.e8m0 import e8m0_to_fp32
from tensorrt_llm._torch.auto_deploy.utils.fp8_dequant import dequant_fp8_weight_two_dim_block_grid
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

_HAS_E8M0 = hasattr(torch, "float8_e8m0fnu")
_requires_e8m0 = pytest.mark.skipif(
    not _HAS_E8M0,
    reason="torch.float8_e8m0fnu is not available in this PyTorch build",
)


class _DummyFactory:
    def __init__(self, quant_config: dict[str, object]) -> None:
        self._quant_config = quant_config

    def get_quant_config(self) -> dict[str, object]:
        return self._quant_config


class _DeepSeekV4LinearFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Module()])
        self.layers[0].attn = nn.Module()
        self.layers[0].attn.wq_a = nn.Linear(16, 8, bias=False)
        self.layers[0].attn.compressor = nn.Module()
        self.layers[0].attn.compressor.wkv = nn.Linear(16, 8, bias=False)
        self.layers[0].ffn = nn.Module()
        self.layers[0].ffn.gate = nn.Linear(16, 4, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.ops.auto_deploy.torch_linear_simple.default(
                x, self.layers[0].attn.wq_a.weight, None
            ),
            torch.ops.auto_deploy.torch_linear_simple.default(
                x, self.layers[0].attn.compressor.wkv.weight, None
            ),
            torch.ops.auto_deploy.torch_linear_simple.default(
                x, self.layers[0].ffn.gate.weight, None
            ),
        )


class _GenericLinearFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(16, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_linear_simple.default(x, self.proj.weight, None)


class _DeepSeekV4WoAFixture(nn.Module):
    def __init__(
        self,
        num_groups: int = 2,
        o_lora_rank: int = 3,
        group_dim: int = 5,
        include_unrelated_einsum: bool = False,
    ) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.o_lora_rank = o_lora_rank
        self.include_unrelated_einsum = include_unrelated_einsum
        self.layers = nn.ModuleList([nn.Module()])
        self.layers[0].attn = nn.Module()
        self.layers[0].attn.wo_a = nn.Linear(group_dim, num_groups * o_lora_rank, bias=False)
        if include_unrelated_einsum:
            self.layers[0].attn.other = nn.Linear(group_dim, num_groups * o_lora_rank, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        o = x.view(x.shape[0], x.shape[1], self.num_groups, -1)
        wo_a = self.layers[0].attn.wo_a.weight.view(self.num_groups, self.o_lora_rank, -1)
        output = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        if not self.include_unrelated_einsum:
            return output

        other = self.layers[0].attn.other.weight.view(self.num_groups, self.o_lora_rank, -1)
        return output, torch.einsum("bsgd,grd->bsgr", o, other)


class _DeepSeekV4WoAShardingFixture(nn.Module):
    def __init__(
        self,
        num_groups: int = 4,
        o_lora_rank: int = 128,
        group_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.o_lora_rank = o_lora_rank
        self.group_dim = group_dim
        self.layers = nn.ModuleList([nn.Module()])
        self.layers[0].attn = nn.Module()
        self.layers[0].attn.wo_a = nn.Module()
        weight = torch.arange(
            num_groups * o_lora_rank * group_dim,
            dtype=torch.float32,
        )
        weight = (
            weight.remainder(31)
            .sub(15)
            .div(16)
            .reshape(
                num_groups * o_lora_rank,
                group_dim,
            )
        )
        self.layers[0].attn.wo_a.weight = nn.Parameter(
            weight.to(torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.layers[0].attn.wo_a.register_buffer(
            "weight_scale_inv",
            torch.arange(num_groups, dtype=torch.float32).reshape(num_groups, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_fake_quant_deepseek_v4_wo_a_grouped_finegrained_fp8_linear.default(
            x,
            self.layers[0].attn.wo_a.weight,
            None,
            [self.layers[0].attn.wo_a.weight_scale_inv],
            layer_type="mla",
        )


class _DeepSeekV4WoALocalViewShardingFixture(_DeepSeekV4WoAShardingFixture):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_view = torch.ops.auto_deploy.view(
            x,
            [1, 2, self.num_groups, self.group_dim],
            tp_scaled_dim=2,
            layer_type="mla",
        )
        return torch.ops.auto_deploy.torch_fake_quant_deepseek_v4_wo_a_grouped_finegrained_fp8_linear.default(
            local_view,
            self.layers[0].attn.wo_a.weight,
            None,
            [self.layers[0].attn.wo_a.weight_scale_inv],
            layer_type="mla",
        )


class _DeepSeekV4AttentionWoBShardingFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        config = DeepseekV4Config(
            vocab_size=16,
            hidden_size=128,
            num_hidden_layers=1,
            num_attention_heads=8,
            head_dim=16,
            q_lora_rank=32,
            qk_rope_head_dim=4,
            o_groups=8,
            o_lora_rank=128,
            sliding_window=2,
            compress_ratios=(0,),
            ad_rope_cache_len=16,
        )
        self.layers = nn.ModuleList([nn.Module()])
        self.layers[0].attn = DeepseekV4Attention(config, layer_idx=0)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return self.layers[0].attn(x, position_ids)


def _trace_with_weight_meta(module: nn.Module) -> torch.fx.GraphModule:
    gm = torch.fx.symbolic_trace(module)
    params_and_buffers = dict(gm.named_parameters())
    params_and_buffers.update(dict(gm.named_buffers()))
    for node in gm.graph.nodes:
        if node.op == "get_attr" and node.target in params_and_buffers:
            node.meta["val"] = params_and_buffers[node.target].detach()
    return gm


def _run_ir_sharding(
    gm: torch.fx.GraphModule,
    *,
    rank: int,
    world_size: int,
    shard_layers: list[str] | None = None,
) -> tuple[torch.fx.GraphModule, object]:
    transform = ApplyShardingHints.from_kwargs(stage=Stages.SHARDING)
    transform.config.shard_layers = shard_layers
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


def _run_finegrained_transform(
    gm: torch.fx.GraphModule,
    quant_config: dict[str, object],
) -> tuple[torch.fx.GraphModule, object]:
    transform = FineGrainedFP8LinearQuantization.from_kwargs(stage=Stages.PATTERN_MATCHER)
    return transform._apply(gm, None, _DummyFactory(quant_config), SharedConfig())


def _load_finegrained_state_dict(state_dict: dict[str, torch.Tensor], weight_name: str) -> None:
    transform = FineGrainedFP8LinearQuantization.from_kwargs(stage=Stages.PATTERN_MATCHER)
    transform.load_hook(state_dict, "", weight_name=weight_name)


def _depends_on(node: torch.fx.Node, target: torch.fx.Node) -> bool:
    def _node_args(value):
        if isinstance(value, torch.fx.Node):
            return [value]
        if isinstance(value, (list, tuple)):
            nodes = []
            for item in value:
                nodes.extend(_node_args(item))
            return nodes
        if isinstance(value, dict):
            nodes = []
            for item in value.values():
                nodes.extend(_node_args(item))
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
        pending.extend(_node_args(current.args))
        pending.extend(_node_args(current.kwargs))
    return False


def _lower_sharding_views_to_aten(gm: torch.fx.GraphModule) -> None:
    for node in gm.graph.nodes:
        if is_op(node, torch.ops.auto_deploy.view):
            node.target = torch.ops.aten.view.default
            node.args = (node.args[0], node.args[1])
            node.kwargs = {}
    gm.graph.lint()
    gm.recompile()


def _fp8_weight(shape: tuple[int, int]) -> torch.Tensor:
    return torch.empty(shape, dtype=torch.float8_e4m3fn, device="cpu")


@_requires_e8m0
def _e8m0_from_bytes(raw_bytes: list[int], shape: tuple[int, ...]) -> torch.Tensor:
    return (
        torch.tensor(raw_bytes, dtype=torch.uint8, device="cpu")
        .view(torch.float8_e8m0fnu)
        .view(shape)
    )


def test_deepseek_v4_scale_alias_loads_into_weight_scale_inv() -> None:
    weight_name = "layers.0.attn.wq_a.weight"
    scale_alias = "layers.0.attn.wq_a.scale"
    scale_buffer = "layers.0.attn.wq_a.weight_scale_inv"
    scale = torch.ones((8, 32), dtype=torch.bfloat16, device="cpu")
    state_dict = {
        weight_name: _fp8_weight((1024, 4096)),
        scale_alias: scale,
    }

    _load_finegrained_state_dict(state_dict, weight_name)

    assert scale_alias not in state_dict
    assert state_dict[scale_buffer] is scale
    assert state_dict[scale_buffer].shape == (8, 32)


@_requires_e8m0
def test_deepseek_v4_e8m0_scale_alias_decodes_to_fp32() -> None:
    weight_name = "layers.0.attn.wq_a.weight"
    scale_alias = "layers.0.attn.wq_a.scale"
    scale = _e8m0_from_bytes([126, 127, 128, 129, 130, 131], (2, 3))
    state_dict = {
        weight_name: _fp8_weight((129, 257)),
        scale_alias: scale,
    }

    _load_finegrained_state_dict(state_dict, weight_name)

    actual = state_dict["layers.0.attn.wq_a.weight_scale_inv"]
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, e8m0_to_fp32(scale), rtol=0, atol=0)


@pytest.mark.parametrize(
    ("weight_shape", "scale_shape"),
    [
        ((1024, 4096), (8, 32)),
        ((129, 257), (2, 3)),
    ],
)
def test_deepseek_v4_scale_shape_validation_accepts_ceil_shapes(
    weight_shape: tuple[int, int],
    scale_shape: tuple[int, int],
) -> None:
    weight_name = "layers.0.attn.wq_a.weight"
    state_dict = {
        weight_name: _fp8_weight(weight_shape),
        "layers.0.attn.wq_a.scale": torch.ones(scale_shape, dtype=torch.float32),
    }

    _load_finegrained_state_dict(state_dict, weight_name)

    assert state_dict["layers.0.attn.wq_a.weight_scale_inv"].shape == scale_shape


def test_deepseek_v4_scale_shape_validation_rejects_mismatches() -> None:
    weight_name = "layers.0.attn.wq_a.weight"
    state_dict = {
        weight_name: _fp8_weight((129, 257)),
        "layers.0.attn.wq_a.scale": torch.ones((1, 3), dtype=torch.float32),
    }

    with pytest.raises(ValueError, match=r"expected shape \(2, 3\)"):
        _load_finegrained_state_dict(state_dict, weight_name)


def test_deepseek_v4_transform_quantizes_only_dense_fp8_linear_paths() -> None:
    model = _DeepSeekV4LinearFixture().to(torch.bfloat16)
    gm = _trace_with_weight_meta(model)

    transformed, info = _run_finegrained_transform(gm, _deepseek_v4_quant_config())

    assert info.num_matches == 1
    assert transformed.get_submodule("layers.0.attn.wq_a").weight.dtype == torch.float8_e4m3fn
    assert transformed.get_submodule("layers.0.attn.wq_a").weight_scale_inv.shape == (1, 1)
    assert transformed.get_submodule("layers.0.ffn.gate").weight.dtype == torch.bfloat16
    assert transformed.get_submodule("layers.0.attn.compressor.wkv").weight.dtype == torch.bfloat16

    quantized_nodes = [
        node
        for node in transformed.graph.nodes
        if is_op(node, torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear)
    ]
    plain_nodes = [
        node
        for node in transformed.graph.nodes
        if is_op(node, torch.ops.auto_deploy.torch_linear_simple)
    ]
    assert len(quantized_nodes) == 1
    assert len(plain_nodes) == 2


def test_deepseek_v4_wo_a_grouped_transform_registers_buffer_and_loads_scale_alias() -> None:
    model = _DeepSeekV4WoAFixture().to(torch.bfloat16)
    gm = _trace_with_weight_meta(model)

    transformed, info = _run_finegrained_transform(gm, _deepseek_v4_quant_config())

    assert info.num_matches == 1
    wo_a = transformed.get_submodule("layers.0.attn.wo_a")
    assert wo_a.weight.dtype == torch.float8_e4m3fn
    assert wo_a.weight_scale_inv.shape == (1, 1)

    scale = torch.full((1, 1), 3.0, dtype=torch.bfloat16)
    state_dict = {
        "layers.0.attn.wo_a.weight": _fp8_weight((6, 5)),
        "layers.0.attn.wo_a.scale": scale,
    }

    missing, unexpected = transformed.load_state_dict(state_dict, strict=False)

    assert not missing
    assert not unexpected
    torch.testing.assert_close(wo_a.weight_scale_inv, scale, rtol=0, atol=0)


def test_deepseek_v4_wo_a_grouped_transform_rewrites_only_scoped_einsum() -> None:
    model = _DeepSeekV4WoAFixture(include_unrelated_einsum=True).to(torch.bfloat16)
    gm = _trace_with_weight_meta(model)

    transformed, info = _run_finegrained_transform(gm, _deepseek_v4_quant_config())

    assert info.num_matches == 1
    assert transformed.get_submodule("layers.0.attn.wo_a").weight.dtype == torch.float8_e4m3fn
    assert transformed.get_submodule("layers.0.attn.other").weight.dtype == torch.bfloat16

    grouped_nodes = [
        node
        for node in transformed.graph.nodes
        if is_op(
            node,
            torch.ops.auto_deploy.torch_fake_quant_deepseek_v4_wo_a_grouped_finegrained_fp8_linear,
        )
    ]
    remaining_einsum_nodes = [
        node
        for node in transformed.graph.nodes
        if node.op == "call_function" and node.target in {torch.einsum, torch.functional.einsum}
    ]
    assert len(grouped_nodes) == 1
    assert grouped_nodes[0].kwargs["layer_type"] == "mla"
    assert len(remaining_einsum_nodes) == 1


def test_deepseek_v4_wo_a_grouped_op_matches_reference_on_ragged_blocks() -> None:
    batch_size, seq_len, num_groups, o_lora_rank, group_dim = 1, 2, 3, 43, 130
    weight_rows = num_groups * o_lora_rank
    input_tensor = (
        torch.arange(batch_size * seq_len * num_groups * group_dim, dtype=torch.float32)
        .reshape(batch_size, seq_len, num_groups, group_dim)
        .remainder(7)
        .sub(3)
        .div(8)
    )
    weight = (
        torch.arange(weight_rows * group_dim, dtype=torch.float32)
        .reshape(weight_rows, group_dim)
        .remainder(17)
        .sub(8)
        .to(torch.float8_e4m3fn)
    )
    weight_scale = torch.tensor(
        [[0.25, 0.5], [1.5, 2.0]],
        dtype=torch.float32,
    )

    grouped_op = (
        torch.ops.auto_deploy.torch_fake_quant_deepseek_v4_wo_a_grouped_finegrained_fp8_linear
    )
    actual = grouped_op.default(input_tensor, weight, None, [weight_scale], layer_type="mla")

    scale_expanded = weight_scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    scale_expanded = scale_expanded[:weight_rows, :group_dim]
    expected_weight = weight.to(torch.float32) * scale_expanded
    expected = torch.einsum(
        "bsgd,grd->bsgr",
        input_tensor,
        expected_weight.view(num_groups, o_lora_rank, group_dim),
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_deepseek_v4_wo_a_grouped_sharding_splits_by_output_group_and_scale_rows() -> None:
    model = _DeepSeekV4WoAShardingFixture()
    gm = _trace_with_weight_meta(model)
    input_shape = (1, 2, model.num_groups, model.group_dim)
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            node.meta["val"] = torch.empty(input_shape, dtype=torch.bfloat16)
        elif is_op(
            node,
            torch.ops.auto_deploy.torch_fake_quant_deepseek_v4_wo_a_grouped_finegrained_fp8_linear,
        ):
            node.meta["val"] = torch.empty(
                (*input_shape[:-1], model.o_lora_rank),
                dtype=torch.bfloat16,
            )

    transformed, info = _run_ir_sharding(gm, rank=1, world_size=2)

    assert info.num_matches == 1
    wo_a = transformed.get_submodule("layers.0.attn.wo_a")
    assert wo_a.weight.shape == (2 * model.o_lora_rank, model.group_dim)
    assert wo_a.weight_scale_inv.shape == (2, 1)
    torch.testing.assert_close(
        wo_a.weight_scale_inv,
        torch.tensor([[2.0], [3.0]], dtype=torch.float32),
        rtol=0,
        atol=0,
    )

    grouped_node = next(
        node
        for node in transformed.graph.nodes
        if is_op(
            node,
            torch.ops.auto_deploy.torch_fake_quant_deepseek_v4_wo_a_grouped_finegrained_fp8_linear,
        )
    )
    local_input = grouped_node.args[0]
    assert local_input.target == torch.ops.aten.slice.Tensor
    assert local_input.args[1:5] == (2, 2, 4, 1)

    full_weight = (
        torch.arange(
            model.num_groups * model.o_lora_rank * model.group_dim,
            dtype=torch.float32,
        )
        .remainder(29)
        .sub(14)
        .div(16)
        .reshape(model.num_groups * model.o_lora_rank, model.group_dim)
        .to(torch.float8_e4m3fn)
    )
    full_scale = torch.arange(model.num_groups, dtype=torch.float32).reshape(model.num_groups, 1)
    full_scale = full_scale.add(10)
    load_result = transformed.load_state_dict(
        {
            "layers.0.attn.wo_a.weight": full_weight,
            "layers.0.attn.wo_a.weight_scale_inv": full_scale,
        },
        strict=False,
    )

    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []
    expected_weight = full_weight.reshape(
        model.num_groups,
        model.o_lora_rank,
        model.group_dim,
    )[2:4].reshape(2 * model.o_lora_rank, model.group_dim)
    torch.testing.assert_close(wo_a.weight.float(), expected_weight.float(), rtol=0, atol=0)
    torch.testing.assert_close(
        wo_a.weight_scale_inv,
        torch.tensor([[12.0], [13.0]], dtype=torch.float32),
        rtol=0,
        atol=0,
    )


def test_deepseek_v4_wo_a_grouped_sharding_skips_input_slice_after_local_group_view() -> None:
    model = _DeepSeekV4WoALocalViewShardingFixture()
    gm = _trace_with_weight_meta(model)
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            node.meta["val"] = torch.empty((1, 2, model.num_groups * model.group_dim))
        elif is_op(node, torch.ops.auto_deploy.view):
            node.meta["val"] = torch.empty((1, 2, model.num_groups, model.group_dim))
        elif is_op(
            node,
            torch.ops.auto_deploy.torch_fake_quant_deepseek_v4_wo_a_grouped_finegrained_fp8_linear,
        ):
            node.meta["val"] = torch.empty(
                (1, 2, model.num_groups, model.o_lora_rank),
                dtype=torch.bfloat16,
            )

    transformed, info = _run_ir_sharding(gm, rank=1, world_size=2)

    assert info.num_matches == 2
    view_node = next(
        node for node in transformed.graph.nodes if is_op(node, torch.ops.auto_deploy.view)
    )
    assert view_node.args[1] == [1, 2, -1, model.group_dim]
    grouped_node = next(
        node
        for node in transformed.graph.nodes
        if is_op(
            node,
            torch.ops.auto_deploy.torch_fake_quant_deepseek_v4_wo_a_grouped_finegrained_fp8_linear,
        )
    )
    assert grouped_node.args[0] is view_node
    assert not any(
        node.op == "call_function" and node.target == torch.ops.aten.slice.Tensor
        for node in transformed.graph.nodes
    )
    assert transformed.get_submodule("layers.0.attn.wo_a").weight.shape == (
        2 * model.o_lora_rank,
        model.group_dim,
    )


@pytest.mark.parametrize(
    "lower_views_to_aten",
    [False, True],
    ids=["auto-deploy-view", "aten-view"],
)
def test_deepseek_v4_attention_wo_a_sharding_feeds_rowwise_wo_b_local_features(
    lower_views_to_aten: bool,
) -> None:
    model = _DeepSeekV4AttentionWoBShardingFixture().to(torch.bfloat16)
    gm = torch_export_to_gm(
        model,
        args=(
            torch.randn(1, 4, 128, dtype=torch.bfloat16),
            torch.arange(4, dtype=torch.long).unsqueeze(0),
        ),
    )
    if lower_views_to_aten:
        _lower_sharding_views_to_aten(gm)

    quantized, quant_info = _run_finegrained_transform(gm, _deepseek_v4_quant_config())
    transformed, shard_info = _run_ir_sharding(
        quantized,
        rank=1,
        world_size=8,
        shard_layers=["mla"],
    )

    assert quant_info.num_matches == 5
    assert shard_info.num_matches >= 6
    wq_b = transformed.get_submodule("layers.0.attn.wq_b")
    wo_a = transformed.get_submodule("layers.0.attn.wo_a")
    wo_b = transformed.get_submodule("layers.0.attn.wo_b")
    assert wq_b.weight.shape == (16, 32)
    assert wq_b.weight_scale_inv.shape == (1, 1)
    assert wo_a.weight.dtype == torch.float8_e4m3fn
    assert wo_a.weight.shape == (128, 16)
    assert wo_a.weight_scale_inv.shape == (1, 1)
    assert wo_b.weight.shape == (128, 128)
    assert wo_b.weight_scale_inv.shape == (1, 1)
    assert transformed.get_parameter("layers.0.attn.attn_sink").shape == (1,)

    q_view_node = next(
        node
        for node in transformed.graph.nodes
        if (
            is_op(node, (torch.ops.auto_deploy.view, torch.ops.aten.view))
            or node.target == torch.ops.aten.view.default
        )
        and node.args[1][-1] == model.layers[0].attn.head_dim
    )
    assert q_view_node.args[1] == [1, 4, -1, 16]
    sparse_node = next(
        node
        for node in transformed.graph.nodes
        if is_op(node, torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_v2)
    )
    assert _depends_on(sparse_node.args[0], q_view_node)

    grouped_node = next(
        node
        for node in transformed.graph.nodes
        if is_op(
            node,
            torch.ops.auto_deploy.torch_fake_quant_deepseek_v4_wo_a_grouped_finegrained_fp8_linear,
        )
    )
    local_group_view = grouped_node.args[0]
    assert local_group_view.args[1] == [1, 4, 1, -1]
    wo_b_node = next(
        node
        for node in transformed.graph.nodes
        if is_op(node, torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear)
        and node.args[1].target == "layers.0.attn.wo_b.weight"
    )
    assert wo_b_node.args[0].args[0] is grouped_node
    assert wo_a.weight.shape[0] // local_group_view.args[1][2] == wo_b.weight.shape[1]
    assert any(
        node.op == "call_function"
        and "dist_all_reduce" in str(node.target)
        and node.args[0] is wo_b_node
        for node in transformed.graph.nodes
    )


def test_fp8_weight_dequant_preserves_fixed_block_scale_for_ragged_shapes() -> None:
    weight = torch.ones((129, 130), dtype=torch.float32).to(torch.float8_e4m3fn)
    weight_scale = torch.tensor(
        [[1.0, 2.0], [4.0, 8.0]],
        dtype=torch.float32,
    )

    actual = dequant_fp8_weight_two_dim_block_grid(
        weight,
        weight_scale,
        block_n=128,
        block_k=128,
        dtype=torch.float32,
    )

    assert actual[65, 0].item() == 1.0
    assert actual[0, 128].item() == 2.0
    assert actual[128, 0].item() == 4.0
    assert actual[128, 128].item() == 8.0


def test_non_deepseek_finegrained_fp8_still_uses_weight_scale_inv() -> None:
    weight_name = "proj.weight"
    state_dict = {
        weight_name: _fp8_weight((8, 16)),
        "proj.weight_scale_inv": torch.ones((1, 1), dtype=torch.bfloat16),
        "proj.scale": torch.full((1, 1), 2.0, dtype=torch.bfloat16),
    }

    _load_finegrained_state_dict(state_dict, weight_name)

    torch.testing.assert_close(
        state_dict["proj.weight_scale_inv"],
        torch.ones((1, 1), dtype=torch.bfloat16),
        rtol=0,
        atol=0,
    )
    assert "proj.scale" in state_dict

    gm = _trace_with_weight_meta(_GenericLinearFixture().to(torch.bfloat16))
    transformed, info = _run_finegrained_transform(
        gm, {"quant_method": "fp8", "weight_block_size": [128, 128]}
    )

    assert info.num_matches == 1
    assert transformed.proj.weight.dtype == torch.float8_e4m3fn


def test_generic_fp8_transform_skips_weight_block_size_configs() -> None:
    gm = _trace_with_weight_meta(_GenericLinearFixture().to(torch.bfloat16))
    transform = FP8LinearQuantizationFromConfig.from_kwargs(stage=Stages.PATTERN_MATCHER)

    transformed, info = transform._apply(
        gm,
        None,
        _DummyFactory({"quant_method": "fp8", "weight_block_size": [128, 128]}),
        SharedConfig(),
    )

    assert info.skipped
    assert transformed.proj.weight.dtype == torch.bfloat16


@_requires_e8m0
def test_trtllm_finegrained_fp8_linear_decodes_e8m0_for_fallback() -> None:
    input_tensor = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.bfloat16)
    weight = torch.tensor(
        [[1.0, 2.0, -1.0, -2.0], [0.5, -0.5, 1.0, -1.0]],
        dtype=torch.float32,
    ).to(torch.float8_e4m3fn)
    bias = torch.tensor([0.25, -0.5], dtype=torch.bfloat16)
    weight_scale = _e8m0_from_bytes([127], (1, 1))

    actual = torch.ops.auto_deploy.trtllm_finegrained_fp8_linear.default(
        input_tensor,
        weight,
        bias,
        weight_scale,
    )

    expected_weight = weight.to(torch.bfloat16) * e8m0_to_fp32(weight_scale).to(torch.bfloat16)
    expected = F.linear(input_tensor, expected_weight, bias)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
