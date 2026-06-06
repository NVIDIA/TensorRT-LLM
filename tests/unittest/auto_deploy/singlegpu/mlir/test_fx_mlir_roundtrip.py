# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Round-trip tests: FX → MLIR → FX."""

import operator

import pytest

xdsl = pytest.importorskip("xdsl")

import torch  # noqa: E402
from torch.export import Dim  # noqa: E402

import tensorrt_llm._torch.auto_deploy.custom_ops.quantization.quant  # noqa: F401, E402
from tensorrt_llm._torch.auto_deploy.custom_ops.normalization.l2norm import *  # noqa: F401, F403, E402
from tensorrt_llm._torch.auto_deploy.custom_ops.normalization.rms_norm import *  # noqa: F401, F403, E402
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm  # noqa: E402
from tensorrt_llm._torch.auto_deploy.mlir.fx_to_mlir import FXToMLIRConverter  # noqa: E402
from tensorrt_llm._torch.auto_deploy.mlir.mlir_to_fx import MLIRToFXConverter  # noqa: E402


def _single_output_fused_stub(x):
    return (x,)


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


class AddNormModel(torch.nn.Module):
    """Simple add + rmsnorm model."""

    def __init__(self, hidden_size=128, eps=1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(hidden_size, device="cuda", dtype=torch.bfloat16)
        )
        self.eps = eps

    def forward(self, x, residual):
        added = x + residual
        norm = torch.ops.auto_deploy.flashinfer_rms_norm(added, self.weight, self.eps)
        return norm, added


class MultiUserModel(torch.nn.Module):
    """Add has multiple users: rmsnorm + linear."""

    def __init__(self, hidden_size=128, eps=1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(hidden_size, device="cuda", dtype=torch.bfloat16)
        )
        self.linear = torch.nn.Linear(
            hidden_size, hidden_size, bias=False, device="cuda", dtype=torch.bfloat16
        )
        self.eps = eps

    def forward(self, residual, attn_output, moe_output):
        add_result = residual + attn_output
        norm_result = torch.ops.auto_deploy.flashinfer_rms_norm(add_result, self.weight, self.eps)
        out1 = self.linear(norm_result)
        next_residual = add_result + moe_output
        return out1, next_residual


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _export_model(model, *inputs):
    """Export model to GraphModule with dynamic batch dimension."""
    dyn = Dim.DYNAMIC
    ds = tuple({0: dyn} for _ in inputs)
    return torch_export_to_gm(model, args=inputs, dynamic_shapes=ds, clone=True)


def _build_single_output_fused_mlir_module():
    from xdsl.dialects.builtin import BFloat16Type, ModuleOp, StringAttr, TensorType
    from xdsl.ir import Block, Region

    from tensorrt_llm._torch.auto_deploy.mlir.dialect import AdGraphInput, AdGraphOutput, AdOpaque

    tensor_type = TensorType(BFloat16Type(), [2, 4])
    block = Block()

    input_op = AdGraphInput.build(
        attributes={"input_name": StringAttr("x")},
        result_types=[tensor_type],
    )
    block.add_op(input_op)

    fused_op = AdOpaque.build(
        operands=[[input_op.output]],
        attributes={
            "op_name": StringAttr("mlir_fused_single"),
            "node_key": StringAttr("mlir_fused_single"),
        },
        result_types=[[tensor_type]],
    )
    block.add_op(fused_op)

    output_op = AdGraphOutput.build(operands=[[fused_op.outputs[0]]])
    block.add_op(output_op)

    return ModuleOp(Region([block]))


def _build_precise_to_dtype_mlir_module():
    from xdsl.dialects.builtin import (
        BFloat16Type,
        Float32Type,
        IntegerAttr,
        IntegerType,
        ModuleOp,
        StringAttr,
        TensorType,
    )
    from xdsl.ir import Block, Region

    from tensorrt_llm._torch.auto_deploy.mlir.dialect import AdGraphInput, AdGraphOutput, AdToDtype

    input_type = TensorType(Float32Type(), [2, 4])
    output_type = TensorType(BFloat16Type(), [2, 4])
    block = Block()

    input_op = AdGraphInput.build(
        attributes={"input_name": StringAttr("x")},
        result_types=[input_type],
    )
    block.add_op(input_op)

    to_dtype_op = AdToDtype.build(
        operands=[input_op.output],
        attributes={"target_dtype": IntegerAttr(15, IntegerType(64))},
        result_types=[output_type],
    )
    block.add_op(to_dtype_op)

    output_op = AdGraphOutput.build(operands=[[to_dtype_op.output]])
    block.add_op(output_op)

    return ModuleOp(Region([block]))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mlir_to_fx_propagates_single_output_fused_getitem_meta():
    import operator

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    graph.output(x)
    original_gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    fake_val = torch.empty((2, 4), dtype=torch.bfloat16, device="meta")
    metadata = {
        "x": {"val": fake_val},
        "mlir_fused_single": {
            "_original_target": _single_output_fused_stub,
            "_args_template": (("__mlir_operand__", 0),),
            "_kwargs_template": {},
            "val": (fake_val,),
        },
    }

    new_gm = MLIRToFXConverter(original_gm).convert(
        _build_single_output_fused_mlir_module(), metadata
    )

    getitem_nodes = [
        n for n in new_gm.graph.nodes if n.op == "call_function" and n.target is operator.getitem
    ]

    assert len(getitem_nodes) == 1
    assert "val" in getitem_nodes[0].meta
    assert tuple(getitem_nodes[0].meta["val"].shape) == tuple(fake_val.shape)
    assert getitem_nodes[0].meta["val"].dtype == fake_val.dtype


def test_mlir_to_fx_synthesizes_precise_op_meta_from_result_type():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    graph.output(x)
    original_gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    input_val = torch.empty((2, 4), dtype=torch.float32, device="meta")
    new_gm = MLIRToFXConverter(original_gm).convert(
        _build_precise_to_dtype_mlir_module(), {"x": {"val": input_val}}
    )

    to_dtype_nodes = [
        n
        for n in new_gm.graph.nodes
        if n.op == "call_function" and n.target == torch.ops.aten.to.dtype
    ]

    assert len(to_dtype_nodes) == 1
    assert "val" in to_dtype_nodes[0].meta
    assert tuple(to_dtype_nodes[0].meta["val"].shape) == (2, 4)
    assert to_dtype_nodes[0].meta["val"].dtype == torch.bfloat16


def test_fx_to_mlir_basic():
    """FX → MLIR conversion produces expected ops."""
    model = AddNormModel()
    x = torch.randn(2, 8, 128, device="cuda", dtype=torch.bfloat16)
    res = torch.randn_like(x)
    gm = _export_model(model, x, res)

    converter = FXToMLIRConverter(gm)
    mlir_mod = converter.convert()

    # Check MLIR ops were created
    op_names = [op.name for op in mlir_mod.body.block.ops]
    assert "ad.add" in op_names, f"Expected ad.add in {op_names}"
    assert "ad.rmsnorm" in op_names, f"Expected ad.rmsnorm in {op_names}"
    assert "ad.graph_output" in op_names

    # Check metadata was stored
    assert len(converter.metadata) > 0


def test_fx_to_mlir_metadata_stored():
    """Verify FX node metadata is stored in the side-table."""
    model = AddNormModel()
    x = torch.randn(2, 8, 128, device="cuda", dtype=torch.bfloat16)
    res = torch.randn_like(x)
    gm = _export_model(model, x, res)

    converter = FXToMLIRConverter(gm)
    converter.convert()

    # Check that val metadata exists for at least some nodes
    has_val = any("val" in meta for meta in converter.metadata.values())
    assert has_val, "Expected 'val' metadata to be stored for tensor nodes"


def test_fx_to_mlir_lowers_l2norm_primitives_as_fusible_ops():
    """FX lowering covers the square -> sum -> div -> cast primitive chain."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    x.meta["val"] = torch.empty((2, 8), device="meta", dtype=torch.float32)
    y.meta["val"] = torch.empty((2, 8), device="meta", dtype=torch.float32)

    square = graph.call_function(torch.ops.aten.square.default, args=(x,))
    square.meta["val"] = torch.empty((2, 8), device="meta", dtype=torch.float32)
    summed = graph.call_function(torch.ops.aten.sum.dim_IntList, args=(square, [1], True))
    summed.meta["val"] = torch.empty((2, 1), device="meta", dtype=torch.float32)
    div = graph.call_function(torch.ops.aten.div.Tensor, args=(x, y))
    div.meta["val"] = torch.empty((2, 8), device="meta", dtype=torch.float32)
    cast = graph.call_function(torch.ops.aten.to.dtype, args=(div, torch.bfloat16))
    cast.meta["val"] = torch.empty((2, 8), device="meta", dtype=torch.bfloat16)
    graph.output((summed, cast))

    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    converter = FXToMLIRConverter(gm)
    mlir_mod = converter.convert()

    op_names = [op.name for op in mlir_mod.body.block.ops]
    assert "ad.mul" in op_names, f"Expected square to lower to ad.mul in {op_names}"
    assert "ad.reduce_sum" in op_names, f"Expected ad.reduce_sum in {op_names}"
    assert "ad.div" in op_names, f"Expected ad.div in {op_names}"
    assert "ad.to_dtype" in op_names, f"Expected ad.to_dtype in {op_names}"
    assert "ad.opaque" not in op_names

    from tensorrt_llm._torch.auto_deploy.mlir.fusion.subgraph_discovery import (
        discover_fusible_subgraphs,
    )

    subgraphs = discover_fusible_subgraphs(mlir_mod)
    assert sorted(len(sg.ops) for sg in subgraphs) == [2, 2]


def test_fx_to_mlir_lowers_torch_l2norm_to_fusible_primitives():
    """Standardized torch_l2norm custom ops lower to primitive MLIR."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((2, 8), device="meta", dtype=torch.bfloat16)

    l2norm = graph.call_function(torch.ops.auto_deploy.torch_l2norm.default, args=(x, 1e-6))
    l2norm.meta["val"] = torch.empty((2, 8), device="meta", dtype=torch.bfloat16)
    graph.output(l2norm)

    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    converter = FXToMLIRConverter(gm)
    mlir_mod = converter.convert()

    op_names = [op.name for op in mlir_mod.body.block.ops]
    assert op_names.count("ad.to_dtype") == 2
    assert "ad.reduce_sum" in op_names
    assert "ad.rsqrt" in op_names
    assert "ad.opaque" not in op_names

    from tensorrt_llm._torch.auto_deploy.mlir.fusion.subgraph_discovery import (
        discover_fusible_subgraphs,
    )

    subgraphs = discover_fusible_subgraphs(mlir_mod)
    assert len(subgraphs) == 1
    assert [op.name for op in subgraphs[0].ops] == [
        "ad.to_dtype",
        "ad.mul",
        "ad.reduce_sum",
        "ad.splat",
        "ad.add",
        "ad.rsqrt",
        "ad.mul",
        "ad.to_dtype",
    ]


def test_fx_to_mlir_lowers_narrow_silu_mul_to_single_fusible_subgraph():
    """Narrow + contiguous + silu + mul lowers to one MLIR fusible subgraph."""
    hidden = 256
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((2, hidden * 2), device="meta", dtype=torch.float16)

    gate_narrow = graph.call_function(torch.narrow, args=(x, -1, 0, hidden))
    gate_narrow.meta["val"] = torch.empty((2, hidden), device="meta", dtype=torch.float16)
    up_slice = graph.call_function(
        torch.ops.aten.slice.Tensor,
        args=(x, -1, hidden, hidden * 2, 1),
    )
    up_slice.meta["val"] = torch.empty((2, hidden), device="meta", dtype=torch.float16)
    gate = graph.call_method("contiguous", args=(gate_narrow,))
    gate.meta["val"] = torch.empty((2, hidden), device="meta", dtype=torch.float16)
    up = graph.call_method("contiguous", args=(up_slice,))
    up.meta["val"] = torch.empty((2, hidden), device="meta", dtype=torch.float16)
    silu = graph.call_function(torch.ops.aten.silu.default, args=(gate,))
    silu.meta["val"] = torch.empty((2, hidden), device="meta", dtype=torch.float16)
    mul = graph.call_function(torch.ops.aten.mul.Tensor, args=(silu, up))
    mul.meta["val"] = torch.empty((2, hidden), device="meta", dtype=torch.float16)
    graph.output(mul)

    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    converter = FXToMLIRConverter(gm)
    mlir_mod = converter.convert()

    op_names = [op.name for op in mlir_mod.body.block.ops]
    assert op_names.count("ad.slice") == 2
    assert "ad.opaque" not in op_names

    from tensorrt_llm._torch.auto_deploy.mlir.fusion.subgraph_discovery import (
        discover_fusible_subgraphs,
    )

    subgraphs = discover_fusible_subgraphs(mlir_mod)
    assert len(subgraphs) == 1
    assert [op.name for op in subgraphs[0].ops] == [
        "ad.slice",
        "ad.slice",
        "ad.silu",
        "ad.mul",
    ]


def test_fx_to_mlir_lowers_split_getitem_silu_mul_to_single_fusible_subgraph():
    """Split/getitem + silu + mul lowers to one MLIR fusible subgraph."""
    hidden = 256
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((2, hidden * 2), device="meta", dtype=torch.float16)

    split = graph.call_function(
        torch.ops.aten.split_with_sizes.default,
        args=(x, [hidden, hidden], -1),
    )
    split.meta["val"] = (
        torch.empty((2, hidden), device="meta", dtype=torch.float16),
        torch.empty((2, hidden), device="meta", dtype=torch.float16),
    )
    gate = graph.call_function(operator.getitem, args=(split, 0))
    gate.meta["val"] = torch.empty((2, hidden), device="meta", dtype=torch.float16)
    up = graph.call_function(operator.getitem, args=(split, 1))
    up.meta["val"] = torch.empty((2, hidden), device="meta", dtype=torch.float16)
    silu = graph.call_function(torch.ops.aten.silu.default, args=(gate,))
    silu.meta["val"] = torch.empty((2, hidden), device="meta", dtype=torch.float16)
    mul = graph.call_function(torch.ops.aten.mul.Tensor, args=(silu, up))
    mul.meta["val"] = torch.empty((2, hidden), device="meta", dtype=torch.float16)
    graph.output(mul)

    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    converter = FXToMLIRConverter(gm)
    mlir_mod = converter.convert()

    op_names = [op.name for op in mlir_mod.body.block.ops]
    assert op_names.count("ad.slice") == 2
    assert "ad.opaque" not in op_names

    from tensorrt_llm._torch.auto_deploy.mlir.fusion.subgraph_discovery import (
        discover_fusible_subgraphs,
    )

    subgraphs = discover_fusible_subgraphs(mlir_mod)
    assert len(subgraphs) == 1
    assert [op.name for op in subgraphs[0].ops] == [
        "ad.slice",
        "ad.slice",
        "ad.silu",
        "ad.mul",
    ]


def test_fx_to_mlir_lowers_fp8_quant_linear_to_mlir_quant_and_prequant_linear():
    """FP8 quant-linear consumers lower through ad.quant_fp8."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    weight = graph.placeholder("weight")
    input_scale = graph.placeholder("input_scale")
    weight_scale = graph.placeholder("weight_scale")
    x.meta["val"] = torch.empty((2, 128), device="meta", dtype=torch.bfloat16)
    weight.meta["val"] = torch.empty((64, 128), device="meta", dtype=torch.float8_e4m3fn)
    input_scale.meta["val"] = torch.empty((), device="meta", dtype=torch.float32)
    weight_scale.meta["val"] = torch.empty((), device="meta", dtype=torch.float32)

    relu = graph.call_function(torch.ops.aten.relu.default, args=(x,))
    relu.meta["val"] = torch.empty((2, 128), device="meta", dtype=torch.bfloat16)
    linear = graph.call_function(
        torch.ops.auto_deploy.trtllm_quant_fp8_linear.default,
        args=(relu, weight, None),
        kwargs={"input_scale": input_scale, "weight_scale": weight_scale},
    )
    linear.meta["val"] = torch.empty((2, 64), device="meta", dtype=torch.bfloat16)
    graph.output(linear)

    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    converter = FXToMLIRConverter(gm)
    mlir_mod = converter.convert()

    op_names = [op.name for op in mlir_mod.body.block.ops]
    assert "ad.quant_fp8" in op_names
    assert "ad.opaque" in op_names
    assert converter.metadata[linear.name]["_original_target"] == (
        torch.ops.auto_deploy.trtllm_fp8_prequant_linear.default
    )

    from tensorrt_llm._torch.auto_deploy.mlir.fusion.subgraph_discovery import (
        discover_fusible_subgraphs,
    )

    subgraphs = discover_fusible_subgraphs(mlir_mod)
    assert len(subgraphs) == 1
    assert [op.name for op in subgraphs[0].ops] == ["ad.relu", "ad.quant_fp8"]


def test_fx_to_mlir_lowers_relu2_nvfp4_quant_linear_to_prequant_linear():
    """Exclusive ReLU2 -> NVFP4 quant-linear lowers to MLIR-owned prequant path."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    weight = graph.placeholder("weight")
    input_scale = graph.placeholder("input_scale")
    weight_scale = graph.placeholder("weight_scale")
    alpha = graph.placeholder("alpha")
    x.meta["val"] = torch.empty((2, 128), device="meta", dtype=torch.float16)
    weight.meta["val"] = torch.empty((64, 64), device="meta", dtype=torch.uint8)
    input_scale.meta["val"] = torch.empty((), device="meta", dtype=torch.float32)
    weight_scale.meta["val"] = torch.empty((512,), device="meta", dtype=torch.uint8)
    alpha.meta["val"] = torch.empty((), device="meta", dtype=torch.float32)

    relu = graph.call_function(torch.ops.aten.relu.default, args=(x,))
    relu.meta["val"] = torch.empty((2, 128), device="meta", dtype=torch.float16)
    relu2 = graph.call_function(torch.ops.aten.mul.Tensor, args=(relu, relu))
    relu2.meta["val"] = torch.empty((2, 128), device="meta", dtype=torch.float16)
    linear = graph.call_function(
        torch.ops.auto_deploy.torch_quant_nvfp4_linear.default,
        args=(relu2, weight, None, input_scale, weight_scale, alpha),
    )
    linear.meta["val"] = torch.empty((2, 64), device="meta", dtype=torch.float16)
    graph.output(linear)

    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    converter = FXToMLIRConverter(gm)
    mlir_mod = converter.convert()

    op_names = [op.name for op in mlir_mod.body.block.ops]
    assert "ad.relu" not in op_names
    assert "ad.mul" not in op_names
    assert op_names.count("ad.opaque") == 2
    assert converter.num_direct_rewrites == 1
    assert converter.metadata[linear.name]["_original_target"] == (
        torch.ops.auto_deploy.trtllm_nvfp4_prequant_linear.default
    )
    assert any(
        meta.get("_original_target") == torch.ops.auto_deploy.trtllm_fused_relu2_quant_nvfp4.default
        for meta in converter.metadata.values()
    )


def test_fx_to_mlir_leaves_shared_relu2_nvfp4_quant_linear_opaque():
    """Shared ReLU2 users are not rewritten through the fused NVFP4 path."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    weight = graph.placeholder("weight")
    input_scale = graph.placeholder("input_scale")
    weight_scale = graph.placeholder("weight_scale")
    alpha = graph.placeholder("alpha")
    x.meta["val"] = torch.empty((2, 128), device="meta", dtype=torch.float16)
    weight.meta["val"] = torch.empty((64, 64), device="meta", dtype=torch.uint8)
    input_scale.meta["val"] = torch.empty((), device="meta", dtype=torch.float32)
    weight_scale.meta["val"] = torch.empty((512,), device="meta", dtype=torch.uint8)
    alpha.meta["val"] = torch.empty((), device="meta", dtype=torch.float32)

    relu = graph.call_function(torch.ops.aten.relu.default, args=(x,))
    relu.meta["val"] = torch.empty((2, 128), device="meta", dtype=torch.float16)
    relu2 = graph.call_function(torch.ops.aten.mul.Tensor, args=(relu, relu))
    relu2.meta["val"] = torch.empty((2, 128), device="meta", dtype=torch.float16)
    linear = graph.call_function(
        torch.ops.auto_deploy.torch_quant_nvfp4_linear.default,
        args=(relu2, weight, None, input_scale, weight_scale, alpha),
    )
    linear.meta["val"] = torch.empty((2, 64), device="meta", dtype=torch.float16)
    shared = graph.call_function(torch.ops.aten.slice.Tensor, args=(relu2, -1, 0, 64, 1))
    shared.meta["val"] = torch.empty((2, 64), device="meta", dtype=torch.float16)
    graph.output((linear, shared))

    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    converter = FXToMLIRConverter(gm)
    mlir_mod = converter.convert()

    op_names = [op.name for op in mlir_mod.body.block.ops]
    assert "ad.relu" in op_names
    assert "ad.mul" in op_names
    assert converter.num_direct_rewrites == 0
    assert converter.metadata[linear.name]["_original_target"] == (
        torch.ops.auto_deploy.torch_quant_nvfp4_linear.default
    )


def test_roundtrip_identity():
    """FX → MLIR → FX (no patterns) preserves graph structure."""
    model = AddNormModel()
    x = torch.randn(2, 8, 128, device="cuda", dtype=torch.bfloat16)
    res = torch.randn_like(x)
    gm = _export_model(model, x, res)

    # Count ops before
    add_count_before = sum(
        1 for n in gm.graph.nodes if n.op == "call_function" and "add" in str(n.target)
    )

    # Round-trip without fusion
    converter = FXToMLIRConverter(gm)
    mlir_mod = converter.convert()
    back = MLIRToFXConverter(gm)
    new_gm = back.convert(mlir_mod, converter.metadata)

    # Count ops after — should have same add ops
    add_count_after = sum(
        1 for n in new_gm.graph.nodes if n.op == "call_function" and "add" in str(n.target)
    )
    assert add_count_after == add_count_before


def test_roundtrip_with_decomposition():
    """FX → MLIR → decompose → FX round-trips correctly."""
    from tensorrt_llm._torch.auto_deploy.mlir.decompose import run_decomposition

    model = AddNormModel()
    x = torch.randn(2, 8, 128, device="cuda", dtype=torch.bfloat16)
    res = torch.randn_like(x)
    gm = _export_model(model, x, res)

    # FX → MLIR
    converter = FXToMLIRConverter(gm)
    mlir_mod = converter.convert()

    # Decompose rmsnorm into primitives
    num_decomposed = run_decomposition(mlir_mod)
    assert num_decomposed >= 1

    # MLIR → FX
    back = MLIRToFXConverter(gm)
    new_gm = back.convert(mlir_mod, converter.metadata)

    # Verify the decomposed graph has primitive ops (mul, rsqrt, etc.)
    op_names = [str(n.target) for n in new_gm.graph.nodes if n.op == "call_function"]
    op_str = " ".join(op_names)
    assert "rsqrt" in op_str or "mul" in op_str, "Expected decomposed primitives in graph"
