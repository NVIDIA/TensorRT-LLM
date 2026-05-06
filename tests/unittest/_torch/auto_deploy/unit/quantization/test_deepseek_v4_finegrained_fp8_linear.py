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
from torch.fx import Graph, GraphModule, Node

import tensorrt_llm._torch.auto_deploy.custom_ops.quantization.torch_quant  # noqa: F401
from tensorrt_llm._torch.auto_deploy.models.quant_checkpoint_layout import (
    FineGrainedFP8CheckpointLayout,
    QuantCheckpointLayoutRegistry,
    QuantizedCheckpointLayout,
)
from tensorrt_llm._torch.auto_deploy.transform.interface import (
    SharedConfig,
    Stages,
    TransformConfig,
)
from tensorrt_llm._torch.auto_deploy.transform.library.quantization import (
    FineGrainedFP8LinearQuantization,
)
from tensorrt_llm._torch.auto_deploy.transform.library.sharding_ir import ShardableNode
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

_FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)
_E8M0_DTYPE = getattr(torch, "float8_e8m0fnu", None)

pytestmark = pytest.mark.skipif(_FP8_DTYPE is None, reason="Requires torch.float8_e4m3fn")

_ALLOWED_WEIGHT_NAMES = (
    "layers.0.attn.wq_a.weight",
    "layers.0.attn.wq_b.weight",
    "layers.0.attn.wkv.weight",
    "layers.0.attn.wo_a.weight",
    "layers.0.attn.wo_b.weight",
    "layers.0.attn.indexer.wq_b.weight",
    "layers.0.ffn.shared_experts.w1.weight",
    "layers.0.ffn.shared_experts.w2.weight",
    "layers.0.ffn.shared_experts.w3.weight",
)
_SKIPPED_WEIGHT_NAMES = (
    "layers.0.attn.compressor.wkv.weight",
    "layers.0.attn.indexer.compressor.wkv.weight",
    "layers.0.attn.indexer.weights_proj.weight",
    "layers.0.ffn.experts.0.w1.weight",
    "layers.0.ffn.experts.0.w2.weight",
    "layers.0.ffn.experts.0.w3.weight",
    "layers.0.ffn.gate.weight",
    "head.weight",
)


class _Factory:
    def __init__(self, quant_config: dict[str, object]) -> None:
        self._quant_config = quant_config

    def get_quant_config(self) -> dict[str, object]:
        return self._quant_config


class _WeightOnlyLinear(nn.Module):
    def __init__(self, out_features: int = 129, in_features: int = 257) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float16))


class _DeepSeekLinearSurface(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Module()])
        layer = self.layers[0]

        layer.attn = nn.Module()
        for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            setattr(layer.attn, name, _WeightOnlyLinear())
        layer.attn.compressor = nn.Module()
        layer.attn.compressor.wkv = _WeightOnlyLinear()
        layer.attn.indexer = nn.Module()
        layer.attn.indexer.wq_b = _WeightOnlyLinear()
        layer.attn.indexer.weights_proj = _WeightOnlyLinear()
        layer.attn.indexer.compressor = nn.Module()
        layer.attn.indexer.compressor.wkv = _WeightOnlyLinear()

        layer.ffn = nn.Module()
        layer.ffn.shared_experts = nn.Module()
        for name in ("w1", "w2", "w3"):
            setattr(layer.ffn.shared_experts, name, _WeightOnlyLinear())
        layer.ffn.experts = nn.ModuleList([nn.Module()])
        for name in ("w1", "w2", "w3"):
            setattr(layer.ffn.experts[0], name, _WeightOnlyLinear())
        layer.ffn.gate = _WeightOnlyLinear()

        self.head = _WeightOnlyLinear()


def _transform() -> FineGrainedFP8LinearQuantization:
    return FineGrainedFP8LinearQuantization(TransformConfig(stage=Stages.PATTERN_MATCHER))


def _deepseek_v4_checkpoint_layout() -> QuantizedCheckpointLayout:
    checkpoint_layout = QuantCheckpointLayoutRegistry.build_from_config(
        {
            "model_type": "deepseek_v4",
            "quantization_config": {
                "quant_method": "fp8",
                "scale_fmt": "ue8m0",
                "weight_block_size": [128, 128],
            },
        }
    )
    assert isinstance(checkpoint_layout, QuantizedCheckpointLayout)
    return checkpoint_layout


def _finegrained_fp8_layout() -> FineGrainedFP8CheckpointLayout:
    checkpoint_layout = _deepseek_v4_checkpoint_layout()
    assert isinstance(checkpoint_layout.finegrained_fp8, FineGrainedFP8CheckpointLayout)
    return checkpoint_layout.finegrained_fp8


def _fp8_weight(shape: tuple[int, int]) -> torch.Tensor:
    return torch.empty(shape, dtype=_FP8_DTYPE)


def _run_load_hook(
    state_dict: dict[str, torch.Tensor],
    weight_name: str,
    *,
    checkpoint_layout: object | None = None,
) -> None:
    load_hook_kwargs = {}
    if checkpoint_layout is not None:
        load_hook_kwargs["checkpoint_layout"] = checkpoint_layout
    _transform().load_hook(
        state_dict,
        "",
        weight_name=weight_name,
        **load_hook_kwargs,
    )


def _build_graph_module(weight_names: tuple[str, ...]) -> GraphModule:
    model = _DeepSeekLinearSurface()
    params = dict(model.named_parameters())
    graph = Graph()
    x = graph.placeholder("x")
    outputs = []

    for weight_name in weight_names:
        weight = graph.get_attr(weight_name)
        weight.meta["val"] = params[weight_name]
        outputs.append(graph.call_function(torch.ops.aten.linear.default, (x, weight, None)))

    graph.output(tuple(outputs))
    return GraphModule(model, graph)


def _build_grouped_einsum_graph_module(weight_name: str) -> GraphModule:
    batch_size, seq_len, num_groups, rank, group_width = 2, 3, 4, 8, 16
    model = _DeepSeekLinearSurface()
    modname, _, attrname = weight_name.rpartition(".")
    submod = model.get_submodule(modname)
    setattr(
        submod,
        attrname,
        nn.Parameter(torch.empty(num_groups * rank, group_width, dtype=torch.float16)),
    )
    params = dict(model.named_parameters())

    graph = Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty(batch_size, seq_len, num_groups, group_width, dtype=torch.float16)
    input_view = graph.call_function(
        torch.ops.auto_deploy.view.default,
        args=(x, [batch_size, seq_len, num_groups, group_width]),
        kwargs={"tp_scaled_dim": 2, "layer_type": "mla"},
    )
    weight = graph.get_attr(weight_name)
    weight.meta["val"] = params[weight_name]
    weight_view = graph.call_function(
        torch.ops.auto_deploy.view.default,
        args=(weight, [num_groups, rank, group_width]),
        kwargs={
            "tp_scaled_dim": 0,
            "layer_type": "mla",
            "tp_min_local_shape": rank,
        },
    )
    einsum = graph.call_function(
        torch.ops.aten.einsum.default,
        args=("bsgd,grd->bsgr", [input_view, weight_view]),
    )
    output = graph.call_method("flatten", args=(einsum, 2))
    graph.output(output)
    return GraphModule(model, graph)


def _linear_weight_names(gm: GraphModule, op: object) -> set[str]:
    return {node.args[1].target for node in gm.graph.nodes if is_op(node, op)}


def _linear_nodes(gm: GraphModule, op: object) -> list[Node]:
    return [node for node in gm.graph.nodes if is_op(node, op)]


def _call_function_nodes(gm: GraphModule, op: object) -> list[Node]:
    return [node for node in gm.graph.nodes if is_op(node, op)]


def test_layout_scale_alias_loads_weight_scale_inv_and_removes_alias() -> None:
    weight_name = "layers.0.attn.wq_a.weight"
    scale_name = "layers.0.attn.wq_a.scale"
    scale = torch.arange(6, dtype=torch.float16).reshape(2, 3) + 1
    state_dict = {
        weight_name: _fp8_weight((129, 257)),
        scale_name: scale,
    }

    _run_load_hook(state_dict, weight_name, checkpoint_layout=_finegrained_fp8_layout())

    assert scale_name not in state_dict
    loaded_scale = state_dict["layers.0.attn.wq_a.weight_scale_inv"]
    assert loaded_scale.dtype == torch.float32
    torch.testing.assert_close(loaded_scale, scale.to(torch.float32))


@pytest.mark.skipif(_E8M0_DTYPE is None, reason="Requires torch.float8_e8m0fnu")
def test_layout_e8m0_scale_alias_decodes_raw_exponent_bytes() -> None:
    weight_name = "layers.0.attn.wq_a.weight"
    scale_name = "layers.0.attn.wq_a.scale"
    raw_exponents = torch.tensor([[0, 1, 255], [128, 126, 130]], dtype=torch.uint8)
    try:
        scale = raw_exponents.view(_E8M0_DTYPE)
    except RuntimeError as error:
        pytest.skip(f"torch.float8_e8m0fnu view is unavailable: {error}")
    expected_bits = raw_exponents.to(torch.int32) << 23
    expected_bits[raw_exponents == 0] = 1 << 22
    expected_bits[raw_exponents == 255] = 0x7FC00000
    expected = expected_bits.view(torch.float32)
    state_dict = {
        weight_name: _fp8_weight((129, 257)),
        scale_name: scale,
    }

    _run_load_hook(state_dict, weight_name, checkpoint_layout=_finegrained_fp8_layout())

    loaded_scale = state_dict["layers.0.attn.wq_a.weight_scale_inv"]
    assert loaded_scale.dtype == torch.float32
    torch.testing.assert_close(loaded_scale, expected, equal_nan=True)
    loaded_bits = loaded_scale.contiguous().view(torch.int32)
    torch.testing.assert_close(loaded_bits, expected_bits)
    assert loaded_bits[0, 0].item() == 1 << 22
    assert torch.isnan(loaded_scale[0, 2])
    assert loaded_bits[0, 2].item() == 0x7FC00000


@pytest.mark.parametrize(
    ("weight_shape", "scale_shape"),
    (
        ((1, 1), (1, 1)),
        ((128, 128), (1, 1)),
        ((129, 257), (2, 3)),
    ),
)
def test_layout_scale_alias_accepts_ceil_shape(
    weight_shape: tuple[int, int],
    scale_shape: tuple[int, int],
) -> None:
    weight_name = "layers.0.attn.wq_a.weight"
    scale_name = "layers.0.attn.wq_a.scale"
    state_dict = {
        weight_name: _fp8_weight(weight_shape),
        scale_name: torch.ones(scale_shape, dtype=torch.float32),
    }

    _run_load_hook(state_dict, weight_name, checkpoint_layout=_finegrained_fp8_layout())

    assert state_dict["layers.0.attn.wq_a.weight_scale_inv"].shape == scale_shape


def test_layout_scale_alias_rejects_bad_ceil_shape() -> None:
    weight_name = "layers.0.attn.wq_a.weight"
    scale_name = "layers.0.attn.wq_a.scale"
    state_dict = {
        weight_name: _fp8_weight((129, 257)),
        scale_name: torch.ones((1, 3), dtype=torch.float32),
    }

    with pytest.raises(ValueError, match=r"expected \(2, 3\)"):
        _run_load_hook(state_dict, weight_name, checkpoint_layout=_finegrained_fp8_layout())

    assert scale_name in state_dict
    assert "layers.0.attn.wq_a.weight_scale_inv" not in state_dict


def test_layout_config_quantizes_only_direct_finegrained_fp8_linear_paths() -> None:
    gm = _build_graph_module(_ALLOWED_WEIGHT_NAMES + _SKIPPED_WEIGHT_NAMES)
    modules_to_not_convert_weight = "layers.0.attn.wq_b.weight"
    qcfg = {
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        "scale_fmt": "ue8m0",
        "checkpoint_layout": _deepseek_v4_checkpoint_layout(),
        "exclude_modules": [
            "embed",
            "head",
            "*.ffn.gate",
            "*.attn.compressor",
            "*.attn.indexer.compressor",
            "*.attn.indexer.weights_proj",
            "*.norm",
        ],
        "modules_to_not_convert": ["*.attn.wq_b"],
    }
    quant_op = torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear.default
    expected_quantized = set(_ALLOWED_WEIGHT_NAMES) - {modules_to_not_convert_weight}
    expected_skipped = set(_SKIPPED_WEIGHT_NAMES) | {modules_to_not_convert_weight}

    gm, info = _transform()._apply(gm, None, _Factory(qcfg), SharedConfig())

    assert info.num_matches == len(expected_quantized)
    assert _linear_weight_names(gm, quant_op) == expected_quantized
    assert _linear_weight_names(gm, torch.ops.aten.linear) == expected_skipped
    for node in _linear_nodes(gm, quant_op):
        assert node.kwargs["input_scale_fmt"] == "ue8m0"

    buffers = dict(gm.named_buffers())
    for weight_name in expected_quantized:
        scale_name = weight_name.removesuffix(".weight") + ".weight_scale_inv"
        assert buffers[scale_name].shape == (2, 3)
        assert buffers[scale_name].dtype == torch.float32

    for weight_name in expected_skipped:
        scale_name = weight_name.removesuffix(".weight") + ".weight_scale_inv"
        assert scale_name not in buffers


def test_layout_config_quantizes_targeted_grouped_einsum_with_ue8m0_scale_fmt() -> None:
    weight_name = "layers.0.attn.wo_a.weight"
    gm = _build_grouped_einsum_graph_module(weight_name)
    qcfg = {
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        "scale_fmt": "ue8m0",
        "checkpoint_layout": _deepseek_v4_checkpoint_layout(),
    }
    grouped_op = torch.ops.auto_deploy.torch_fake_quant_grouped_finegrained_fp8_linear.default

    gm, info = _transform()._apply(gm, None, _Factory(qcfg), SharedConfig())

    grouped_nodes = _call_function_nodes(gm, grouped_op)
    assert info.num_matches == 1
    assert len(grouped_nodes) == 1
    grouped_node = grouped_nodes[0]
    assert grouped_node.args[1].target == weight_name
    assert grouped_node.kwargs["input_scale_fmt"] == "ue8m0"
    assert grouped_node.kwargs["tp_mode"] == "colwise"
    assert grouped_node.kwargs["tp_min_local_shape"] == 8
    assert ShardableNode.from_node(grouped_node) is not None
    assert not _call_function_nodes(gm, torch.ops.aten.einsum)
    assert all(node.target != "flatten" for node in gm.graph.nodes if node.op == "call_method")

    buffers = dict(gm.named_buffers())
    scale_name = weight_name.removesuffix(".weight") + ".weight_scale_inv"
    assert buffers[scale_name].shape == (1, 1)
    assert buffers[scale_name].dtype == torch.float32


def test_layout_config_preserves_exported_positional_view_layer_type_on_grouped_einsum() -> None:
    class _ExportedGroupedEinsum(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].attn = nn.Module()
            self.layers[0].attn.wo_a = _WeightOnlyLinear(out_features=32, in_features=16)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.ops.auto_deploy.view(x, [2, 3, 4, 16], 2, "mla")
            weight = torch.ops.auto_deploy.view(
                self.layers[0].attn.wo_a.weight,
                [4, 8, 16],
                0,
                "mla",
                8,
            )
            return torch.einsum("bsgd,grd->bsgr", x, weight).flatten(2)

    gm = torch.export.export(_ExportedGroupedEinsum(), (torch.randn(2, 3, 4, 16),)).module()
    qcfg = {
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        "scale_fmt": "ue8m0",
        "checkpoint_layout": _deepseek_v4_checkpoint_layout(),
    }
    grouped_op = torch.ops.auto_deploy.torch_fake_quant_grouped_finegrained_fp8_linear.default

    gm, info = _transform()._apply(gm, None, _Factory(qcfg), SharedConfig())

    grouped_nodes = _call_function_nodes(gm, grouped_op)
    assert info.num_matches == 1
    assert len(grouped_nodes) == 1
    assert grouped_nodes[0].kwargs["layer_type"] == "mla"


def test_layout_config_leaves_non_targeted_grouped_einsum_untouched() -> None:
    gm = _build_grouped_einsum_graph_module("layers.0.attn.compressor.wkv.weight")
    qcfg = {
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        "scale_fmt": "ue8m0",
        "checkpoint_layout": _deepseek_v4_checkpoint_layout(),
    }
    grouped_op = torch.ops.auto_deploy.torch_fake_quant_grouped_finegrained_fp8_linear.default

    gm, info = _transform()._apply(gm, None, _Factory(qcfg), SharedConfig())

    assert info.num_matches == 0
    assert not _call_function_nodes(gm, grouped_op)
    assert len(_call_function_nodes(gm, torch.ops.aten.einsum)) == 1
    assert any(node.target == "flatten" for node in gm.graph.nodes if node.op == "call_method")
    assert "layers.0.attn.compressor.wkv.weight_scale_inv" not in dict(gm.named_buffers())


def test_layout_config_skips_linear_without_extractable_weight() -> None:
    graph = Graph()
    x = graph.placeholder("x")
    weight = graph.placeholder("weight")
    linear = graph.call_function(torch.ops.aten.linear.default, (x, weight, None))
    graph.output(linear)
    gm = GraphModule(nn.Module(), graph)
    qcfg = {
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        "checkpoint_layout": _deepseek_v4_checkpoint_layout(),
    }
    quant_op = torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear.default

    gm, info = _transform()._apply(gm, None, _Factory(qcfg), SharedConfig())

    assert info.num_matches == 0
    assert _linear_weight_names(gm, quant_op) == set()
    assert any(is_op(node, torch.ops.aten.linear) for node in gm.graph.nodes)


def test_generic_config_quantized_linear_uses_default_input_scale_fmt() -> None:
    gm = _build_graph_module(("layers.0.attn.wq_a.weight",))
    qcfg = {
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    }
    quant_op = torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear.default

    gm, info = _transform()._apply(gm, None, _Factory(qcfg), SharedConfig())

    assert info.num_matches == 1
    quant_nodes = _linear_nodes(gm, quant_op)
    assert len(quant_nodes) == 1
    assert quant_nodes[0].kwargs["input_scale_fmt"] == ""


def test_generic_fp8_uses_scale_inv_and_does_not_consume_scale_alias() -> None:
    weight_name = "layers.0.attn.wq_a.weight"
    alias_name = "layers.0.attn.wq_a.scale"
    scale_inv_name = weight_name + "_scale_inv"
    scale_inv = torch.full((2, 3), 3.0, dtype=torch.float16)
    alias = torch.full((2, 3), 7.0, dtype=torch.float16)
    state_dict = {
        weight_name: _fp8_weight((129, 257)),
        scale_inv_name: scale_inv,
        alias_name: alias,
    }

    _run_load_hook(state_dict, weight_name)

    assert alias_name in state_dict
    assert state_dict[alias_name] is alias
    assert state_dict["layers.0.attn.wq_a.weight_scale_inv"] is scale_inv
