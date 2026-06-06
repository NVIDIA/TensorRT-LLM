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

"""End-to-end tests for the unified MLIR elementwise fusion pipeline.

Verifies the full decompose -> discover -> codegen pipeline on hand-built
MLIR IR without requiring GPU models.
"""

import operator
from pathlib import Path

import pytest
import yaml

xdsl = pytest.importorskip("xdsl")

import torch  # noqa: E402
from xdsl.dialects.builtin import (  # noqa: E402
    BFloat16Type,
    Float32Type,
    Float64Type,
    FloatAttr,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    TensorType,
)
from xdsl.ir import Block, Region  # noqa: E402

import tensorrt_llm._torch.auto_deploy.custom_ops.quantization.quant  # noqa: F401, E402
from tensorrt_llm._torch.auto_deploy.custom_ops.normalization.l2norm import *  # noqa: F401, F403, E402
from tensorrt_llm._torch.auto_deploy.custom_ops.normalization.rms_norm import *  # noqa: F401, F403, E402
from tensorrt_llm._torch.auto_deploy.mlir.codegen.kernel_cache import KernelCache  # noqa: E402
from tensorrt_llm._torch.auto_deploy.mlir.codegen.triton_emitter import (  # noqa: E402
    generate_kernel_from_subgraph,
)
from tensorrt_llm._torch.auto_deploy.mlir.decompose import run_decomposition  # noqa: E402
from tensorrt_llm._torch.auto_deploy.mlir.dialect import (  # noqa: E402
    AdAdd,
    AdDiv,
    AdEq,
    AdFloorDiv,
    AdGraphOutput,
    AdMul,
    AdQuantFP8,
    AdReduceSum,
    AdRelu,
    AdRMSNorm,
    AdSilu,
    AdSlice,
    AdToDtype,
    Float8E4M3FNType,
)
from tensorrt_llm._torch.auto_deploy.mlir.fusion.subgraph_discovery import (  # noqa: E402
    discover_fusible_subgraphs,
)
from tensorrt_llm._torch.auto_deploy.mlir.fusion.subgraph_replace import (  # noqa: E402
    replace_subgraph_with_fused_op,
)


def test_default_config_routes_migrated_elementwise_fusions_to_mlir():
    repo_root = Path(__file__).resolve().parents[5]
    config_path = repo_root / "tensorrt_llm/_torch/auto_deploy/config/default.yaml"
    with config_path.open() as f:
        transforms = yaml.safe_load(f)["transforms"]

    assert transforms["mlir_elementwise_fusion"]["enabled"] is True
    for transform_name in (
        "fuse_l2norm",
        "fuse_silu_mul",
        "fuse_relu2_quant_nvfp4",
        "fuse_rmsnorm_quant_fp8",
        "fuse_rmsnorm_quant_nvfp4",
    ):
        assert transforms[transform_name].get("enabled", True) is False


def _build_add_rmsnorm_module(hidden: int = 128):
    """Build a minimal MLIR module containing add + rmsnorm.

    Graph: (x, residual, weight) -> add(x, residual) -> rmsnorm(added, weight, eps)
    Outputs: (normed, added)
    """
    t = TensorType(BFloat16Type(), [2, hidden])
    tw = TensorType(BFloat16Type(), [hidden])
    eps = FloatAttr(1e-5, Float64Type())

    block = Block()
    x = block.insert_arg(t, 0)
    res = block.insert_arg(t, 1)
    w = block.insert_arg(tw, 2)

    add_op = AdAdd.build(operands=[x, res], result_types=[t])
    block.add_op(add_op)

    norm_op = AdRMSNorm.build(
        operands=[add_op.output, w], attributes={"eps": eps}, result_types=[t]
    )
    block.add_op(norm_op)

    out_op = AdGraphOutput.build(operands=[[norm_op.output, add_op.output]])
    block.add_op(out_op)

    return ModuleOp(Region([block]))


def _build_single_output_add_mul_module(hidden: int = 128):
    """Build a minimal MLIR module whose largest fusible subgraph has one output."""
    t = TensorType(BFloat16Type(), [2, hidden])

    block = Block()
    x = block.insert_arg(t, 0)
    y = block.insert_arg(t, 1)
    z = block.insert_arg(t, 2)

    add_op = AdAdd.build(operands=[x, y], result_types=[t])
    block.add_op(add_op)

    mul_op = AdMul.build(operands=[add_op.output, z], result_types=[t])
    block.add_op(mul_op)

    out_op = AdGraphOutput.build(operands=[[mul_op.output]])
    block.add_op(out_op)

    return ModuleOp(Region([block]))


def test_decompose_discover_codegen_pipeline():
    """Full pipeline: decompose add+rmsnorm -> discover subgraph -> generate kernel."""
    hidden = 128
    mlir_mod = _build_add_rmsnorm_module(hidden)

    # Step 1: Decompose — rmsnorm should be expanded into primitives
    num_decomposed = run_decomposition(mlir_mod)
    assert num_decomposed >= 1, "Expected at least one decomposition (rmsnorm)"

    # Step 2: Discover fusible subgraphs
    subgraphs = discover_fusible_subgraphs(mlir_mod)
    assert len(subgraphs) >= 1, "Expected at least one fusible subgraph"

    # The largest subgraph should contain the add + decomposed rmsnorm primitives
    largest_sg = max(subgraphs, key=lambda sg: len(sg.ops))
    assert len(largest_sg.ops) >= 2, f"Expected subgraph with >= 2 ops, got {len(largest_sg.ops)}"

    # Step 3: Generate kernel
    kernel_fn = generate_kernel_from_subgraph(largest_sg)
    assert callable(kernel_fn), "Expected generate_kernel_from_subgraph to return a callable"


def test_single_output_fused_metadata_uses_tuple_contract():
    """Generated fused kernels return tuples, including the one-output case."""
    hidden = 128
    mlir_mod = _build_single_output_add_mul_module(hidden)

    subgraphs = discover_fusible_subgraphs(mlir_mod)
    assert len(subgraphs) == 1

    sg = subgraphs[0]
    kernel_fn = generate_kernel_from_subgraph(sg)
    sg_hash = KernelCache.hash_subgraph(sg)
    metadata = {}

    replace_subgraph_with_fused_op(sg, kernel_fn, sg_hash, metadata)

    node_key = f"mlir_fused_{sg_hash}"
    val_meta = metadata[node_key]["val"]

    assert isinstance(val_meta, tuple)
    assert len(val_meta) == 1
    assert tuple(val_meta[0].shape) == (2, hidden)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_decompose_discover_codegen_numerical_correctness():
    """Generated kernel from add+rmsnorm decomposition matches torch reference."""
    hidden = 128
    mlir_mod = _build_add_rmsnorm_module(hidden)

    # Run full pipeline
    run_decomposition(mlir_mod)
    subgraphs = discover_fusible_subgraphs(mlir_mod)
    assert len(subgraphs) >= 1

    largest_sg = max(subgraphs, key=lambda sg: len(sg.ops))
    kernel_fn = generate_kernel_from_subgraph(largest_sg)

    # After decomposition with splat pulled into subgraph, the inputs are:
    #   input[0]: x (2D block arg)
    #   input[1]: residual (2D block arg)
    #   input[2]: weight (1D block arg)
    # And the subgraph outputs are:
    #   output[0]: added = x + residual (from ad.add)
    #   output[1]: normed = rmsnorm(added, weight) (from ad.mul)
    xt = torch.randn(2, hidden, device="cuda", dtype=torch.bfloat16)
    rest = torch.randn_like(xt)
    wt = torch.ones(hidden, device="cuda", dtype=torch.bfloat16)

    # Run generated kernel — inputs match subgraph.inputs (x, residual, weight)
    result = kernel_fn(xt, rest, wt)

    # Reference: add + rmsnorm
    added_ref = xt + rest
    x_f32 = added_ref.float()
    var = x_f32.pow(2).mean(dim=-1, keepdim=True)
    normed_ref = (x_f32 * torch.rsqrt(var + 1e-5)).to(torch.bfloat16) * wt

    # Check both outputs — find which output index corresponds to normed vs added
    # by checking shapes and values
    assert len(result) >= 2, f"Expected at least 2 outputs, got {len(result)}"
    # The added output is the simpler one (just x + residual)
    torch.testing.assert_close(result[0], added_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(result[1], normed_ref, atol=1e-2, rtol=1e-2)


def test_no_subgraphs_for_single_op():
    """A module with only one fusible op should produce no subgraphs."""
    hidden = 64
    t = TensorType(BFloat16Type(), [2, hidden])

    block = Block()
    x = block.insert_arg(t, 0)
    y = block.insert_arg(t, 1)

    add_op = AdAdd.build(operands=[x, y], result_types=[t])
    block.add_op(add_op)

    out_op = AdGraphOutput.build(operands=[[add_op.output]])
    block.add_op(out_op)

    mlir_mod = ModuleOp(Region([block]))

    # No decomposition needed (no high-level ops)
    num_decomposed = run_decomposition(mlir_mod)
    assert num_decomposed == 0

    # Single op -> no subgraph (need >= 2 ops)
    subgraphs = discover_fusible_subgraphs(mlir_mod)
    assert len(subgraphs) == 0


def test_pipeline_reports_subgraph_metadata():
    """Verify subgraph metadata (inputs, outputs, op count) is correct."""
    hidden = 128
    mlir_mod = _build_add_rmsnorm_module(hidden)

    run_decomposition(mlir_mod)
    subgraphs = discover_fusible_subgraphs(mlir_mod)
    assert len(subgraphs) >= 1

    largest_sg = max(subgraphs, key=lambda sg: len(sg.ops))

    # The subgraph should have inputs (x, residual, weight) and at least one output
    assert len(largest_sg.inputs) >= 2, f"Expected >= 2 inputs, got {len(largest_sg.inputs)}"
    assert len(largest_sg.outputs) >= 1, f"Expected >= 1 outputs, got {len(largest_sg.outputs)}"


# ---------------------------------------------------------------------------
# Novel fusion: add + rmsnorm + silu(gate) * normed
# This pattern has NO pre-existing hand-written kernel — the MLIR pipeline
# must automatically discover and generate it.
# ---------------------------------------------------------------------------


def _build_add_rmsnorm_silu_gate_module(hidden: int = 128):
    """Build MLIR module for add + rmsnorm + silu_gate (novel fusion).

    Graph: (x, residual, weight, gate)
        added  = add(x, residual)
        normed = rmsnorm(added, weight, eps=1e-5)
        gated  = normed * silu(gate)
    Outputs: (gated, added)
    """
    t = TensorType(BFloat16Type(), [2, hidden])
    tw = TensorType(BFloat16Type(), [hidden])
    eps = FloatAttr(1e-5, Float64Type())

    block = Block()
    x = block.insert_arg(t, 0)
    res = block.insert_arg(t, 1)
    w = block.insert_arg(tw, 2)
    gate = block.insert_arg(t, 3)

    add_op = AdAdd.build(operands=[x, res], result_types=[t])
    block.add_op(add_op)

    norm_op = AdRMSNorm.build(
        operands=[add_op.output, w], attributes={"eps": eps}, result_types=[t]
    )
    block.add_op(norm_op)

    silu_op = AdSilu.build(operands=[gate], result_types=[t])
    block.add_op(silu_op)

    mul_op = AdMul.build(operands=[norm_op.output, silu_op.output], result_types=[t])
    block.add_op(mul_op)

    out_op = AdGraphOutput.build(operands=[[mul_op.output, add_op.output]])
    block.add_op(out_op)

    return ModuleOp(Region([block]))


def test_novel_add_rmsnorm_silu_gate():
    """Fusion that has NO pre-existing kernel — auto-discovered and generated.

    Validates the full decompose -> discover -> codegen pipeline on a novel
    add + rmsnorm + silu(gate) pattern. After decomposition, all ops should
    be fusible primitives that land in ONE subgraph.
    """
    hidden = 128
    mlir_mod = _build_add_rmsnorm_silu_gate_module(hidden)

    # Step 1: Decompose — rmsnorm decomposes into primitives
    num_decomposed = run_decomposition(mlir_mod)
    assert num_decomposed >= 1, "Expected at least one decomposition (rmsnorm)"

    # Step 2: Discover fusible subgraphs
    subgraphs = discover_fusible_subgraphs(mlir_mod)
    assert len(subgraphs) >= 1, "Expected at least one fusible subgraph"

    # ALL primitive ops (add + rmsnorm decomp + silu + mul) should be in ONE
    # subgraph — this is the key assertion proving novel fusion discovery.
    largest_sg = max(subgraphs, key=lambda sg: len(sg.ops))
    # add(1) + rmsnorm-decomp(~7) + silu(1) + mul(1) = ~10 ops minimum
    assert len(largest_sg.ops) >= 9, (
        f"Expected >= 9 ops in the fused subgraph (add + rmsnorm decomp + "
        f"silu + mul), got {len(largest_sg.ops)}"
    )
    # Should be exactly ONE subgraph covering everything
    assert len(subgraphs) == 1, (
        f"Expected exactly 1 subgraph (all ops fused together), got {len(subgraphs)}"
    )

    # Step 3: Generate Triton kernel from the novel subgraph
    kernel_fn = generate_kernel_from_subgraph(largest_sg)
    assert callable(kernel_fn), "Expected generate_kernel_from_subgraph to return a callable"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_novel_fusion_numerical_correctness():
    """Verify the novel add+rmsnorm+silu_gate kernel matches torch reference.

    This is the key validation that the system works for novel patterns:
    no kernel was hand-written for this fused operation.
    """
    hidden = 128
    mlir_mod = _build_add_rmsnorm_silu_gate_module(hidden)

    # Run full pipeline
    run_decomposition(mlir_mod)
    subgraphs = discover_fusible_subgraphs(mlir_mod)
    assert len(subgraphs) >= 1

    largest_sg = max(subgraphs, key=lambda sg: len(sg.ops))
    kernel_fn = generate_kernel_from_subgraph(largest_sg)

    # Prepare inputs: x, residual, weight, gate
    xt = torch.randn(2, hidden, device="cuda", dtype=torch.bfloat16)
    rest = torch.randn_like(xt)
    wt = torch.ones(hidden, device="cuda", dtype=torch.bfloat16)
    gate_t = torch.randn_like(xt)

    # Run generated kernel
    result = kernel_fn(xt, rest, wt, gate_t)

    # Reference: add + rmsnorm + silu(gate) * normed
    added_ref = xt + rest
    x_f32 = added_ref.float()
    var = x_f32.pow(2).mean(dim=-1, keepdim=True)
    normed_ref = (x_f32 * torch.rsqrt(var + 1e-5)).to(torch.bfloat16) * wt
    gated_ref = normed_ref * torch.nn.functional.silu(gate_t)

    # Verify both outputs: (gated, added)
    assert len(result) >= 2, f"Expected at least 2 outputs, got {len(result)}"
    torch.testing.assert_close(result[0], added_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(result[1], gated_ref, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# L2Norm-style primitive chain: square + sum + div + dtype cast
# ---------------------------------------------------------------------------


def _build_l2norm_cast_module(hidden: int = 128):
    """Build MLIR module for a row-wise L2Norm-like primitive chain."""
    t_f32 = TensorType(Float32Type(), [2, hidden])
    t_scalar = TensorType(Float32Type(), [2, 1])
    t_bf16 = TensorType(BFloat16Type(), [2, hidden])

    block = Block()
    x = block.insert_arg(t_f32, 0)

    square = AdMul.build(operands=[x, x], result_types=[t_f32])
    block.add_op(square)

    denom = AdReduceSum.build(
        operands=[square.output],
        attributes={
            "dim": IntegerAttr(-1, IntegerType(64)),
            "keepdim": IntegerAttr(1, IntegerType(1)),
        },
        result_types=[t_scalar],
    )
    block.add_op(denom)

    normalized = AdDiv.build(operands=[x, denom.output], result_types=[t_f32])
    block.add_op(normalized)

    cast = AdToDtype.build(
        operands=[normalized.output],
        attributes={"target_dtype": IntegerAttr(15, IntegerType(64))},
        result_types=[t_bf16],
    )
    block.add_op(cast)

    out_op = AdGraphOutput.build(operands=[[cast.output]])
    block.add_op(out_op)

    return ModuleOp(Region([block]))


def test_l2norm_cast_primitives_fuse_and_generate_kernel():
    """Square + reduce_sum + div + to_dtype should fuse as one generated kernel."""
    mlir_mod = _build_l2norm_cast_module()

    subgraphs = discover_fusible_subgraphs(mlir_mod)
    assert len(subgraphs) == 1, f"Expected 1 subgraph, got {len(subgraphs)}"
    assert [op.name for op in subgraphs[0].ops] == [
        "ad.mul",
        "ad.reduce_sum",
        "ad.div",
        "ad.to_dtype",
    ]

    kernel_fn = generate_kernel_from_subgraph(subgraphs[0])
    assert callable(kernel_fn)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_l2norm_cast_primitives_numerical_correctness():
    """Generated square + sum + div + cast kernel matches torch reference."""
    hidden = 128
    mlir_mod = _build_l2norm_cast_module(hidden)
    subgraphs = discover_fusible_subgraphs(mlir_mod)
    kernel_fn = generate_kernel_from_subgraph(subgraphs[0])

    x = torch.randn(2, hidden, device="cuda", dtype=torch.float32)
    result = kernel_fn(x)

    expected = (x / (x * x).sum(dim=-1, keepdim=True)).to(torch.bfloat16)
    torch.testing.assert_close(result[0], expected, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_torch_l2norm_fx_replacement_numerical_correctness():
    """FX torch_l2norm lowers to primitives, fuses, and matches the reference op."""
    from tensorrt_llm._torch.auto_deploy.mlir.codegen.kernel_cache import KernelCache
    from tensorrt_llm._torch.auto_deploy.mlir.fusion.subgraph_replace import (
        replace_subgraph_with_fused_op,
    )
    from tensorrt_llm._torch.auto_deploy.mlir.fx_to_mlir import FXToMLIRConverter
    from tensorrt_llm._torch.auto_deploy.mlir.mlir_to_fx import MLIRToFXConverter

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((2, 128), device="meta", dtype=torch.bfloat16)
    l2norm = graph.call_function(torch.ops.auto_deploy.torch_l2norm.default, args=(x, 1e-6))
    l2norm.meta["val"] = torch.empty((2, 128), device="meta", dtype=torch.bfloat16)
    graph.output(l2norm)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    converter = FXToMLIRConverter(gm)
    mlir_module = converter.convert()
    subgraphs = discover_fusible_subgraphs(mlir_module)
    assert len(subgraphs) == 1
    sg = subgraphs[0]

    kernel_fn = generate_kernel_from_subgraph(sg)
    sg_hash = KernelCache.hash_subgraph(sg)
    replace_subgraph_with_fused_op(sg, kernel_fn, sg_hash, converter.metadata)

    new_gm = MLIRToFXConverter(gm).convert(mlir_module, converter.metadata)

    xt = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
    result = new_gm(xt)
    expected = torch.ops.auto_deploy.torch_l2norm.default(xt, 1e-6)
    torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# SiLU+Mul structural pattern: slice + silu + mul
# ---------------------------------------------------------------------------


def _build_slice_silu_mul_module(hidden: int = 256):
    """Build MLIR module for fused-gate-up style SiLU+Mul."""
    t_parent = TensorType(BFloat16Type(), [2, hidden * 2])
    t_half = TensorType(BFloat16Type(), [2, hidden])

    block = Block()
    x = block.insert_arg(t_parent, 0)

    gate = AdSlice.build(
        operands=[x],
        attributes={
            "dim": IntegerAttr(-1, IntegerType(64)),
            "start": IntegerAttr(0, IntegerType(64)),
            "length": IntegerAttr(hidden, IntegerType(64)),
        },
        result_types=[t_half],
    )
    block.add_op(gate)

    up = AdSlice.build(
        operands=[x],
        attributes={
            "dim": IntegerAttr(-1, IntegerType(64)),
            "start": IntegerAttr(hidden, IntegerType(64)),
            "length": IntegerAttr(hidden, IntegerType(64)),
        },
        result_types=[t_half],
    )
    block.add_op(up)

    silu = AdSilu.build(operands=[gate.output], result_types=[t_half])
    block.add_op(silu)

    mul = AdMul.build(operands=[silu.output, up.output], result_types=[t_half])
    block.add_op(mul)

    out_op = AdGraphOutput.build(operands=[[mul.output]])
    block.add_op(out_op)

    return ModuleOp(Region([block]))


def test_slice_silu_mul_fuses_and_generates_kernel():
    """Slice + silu + mul should fuse as one generated kernel."""
    mlir_mod = _build_slice_silu_mul_module()

    subgraphs = discover_fusible_subgraphs(mlir_mod)
    assert len(subgraphs) == 1
    assert [op.name for op in subgraphs[0].ops] == [
        "ad.slice",
        "ad.slice",
        "ad.silu",
        "ad.mul",
    ]

    kernel_fn = generate_kernel_from_subgraph(subgraphs[0])
    assert callable(kernel_fn)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_slice_silu_mul_numerical_correctness():
    """Generated slice + silu + mul kernel matches torch reference."""
    hidden = 256
    mlir_mod = _build_slice_silu_mul_module(hidden)
    subgraphs = discover_fusible_subgraphs(mlir_mod)
    kernel_fn = generate_kernel_from_subgraph(subgraphs[0])

    x = torch.randn(2, hidden * 2, device="cuda", dtype=torch.bfloat16)
    result = kernel_fn(x)

    gate = x[..., :hidden]
    up = x[..., hidden:]
    expected = torch.nn.functional.silu(gate) * up
    torch.testing.assert_close(result[0], expected, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_narrow_silu_mul_fx_replacement_numerical_correctness():
    """FX narrow + silu + mul lowers, fuses, and matches torch reference."""
    from tensorrt_llm._torch.auto_deploy.mlir.codegen.kernel_cache import KernelCache
    from tensorrt_llm._torch.auto_deploy.mlir.fusion.subgraph_replace import (
        replace_subgraph_with_fused_op,
    )
    from tensorrt_llm._torch.auto_deploy.mlir.fx_to_mlir import FXToMLIRConverter
    from tensorrt_llm._torch.auto_deploy.mlir.mlir_to_fx import MLIRToFXConverter

    hidden = 256
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((2, hidden * 2), device="meta", dtype=torch.bfloat16)
    gate = graph.call_function(torch.narrow, args=(x, -1, 0, hidden))
    gate.meta["val"] = torch.empty((2, hidden), device="meta", dtype=torch.bfloat16)
    up = graph.call_function(torch.narrow, args=(x, -1, hidden, hidden))
    up.meta["val"] = torch.empty((2, hidden), device="meta", dtype=torch.bfloat16)
    silu = graph.call_function(torch.ops.aten.silu.default, args=(gate,))
    silu.meta["val"] = torch.empty((2, hidden), device="meta", dtype=torch.bfloat16)
    mul = graph.call_function(torch.ops.aten.mul.Tensor, args=(silu, up))
    mul.meta["val"] = torch.empty((2, hidden), device="meta", dtype=torch.bfloat16)
    graph.output(mul)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    converter = FXToMLIRConverter(gm)
    mlir_module = converter.convert()
    subgraphs = discover_fusible_subgraphs(mlir_module)
    assert len(subgraphs) == 1
    sg = subgraphs[0]

    kernel_fn = generate_kernel_from_subgraph(sg)
    sg_hash = KernelCache.hash_subgraph(sg)
    replace_subgraph_with_fused_op(sg, kernel_fn, sg_hash, converter.metadata)

    new_gm = MLIRToFXConverter(gm).convert(mlir_module, converter.metadata)

    xt = torch.randn(2, hidden * 2, device="cuda", dtype=torch.bfloat16)
    result = new_gm(xt)
    expected = torch.nn.functional.silu(xt[..., :hidden]) * xt[..., hidden:]
    torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_split_getitem_silu_mul_fx_replacement_numerical_correctness():
    """FX split/getitem + silu + mul lowers, fuses, and matches torch reference."""
    from tensorrt_llm._torch.auto_deploy.mlir.codegen.kernel_cache import KernelCache
    from tensorrt_llm._torch.auto_deploy.mlir.fusion.subgraph_replace import (
        replace_subgraph_with_fused_op,
    )
    from tensorrt_llm._torch.auto_deploy.mlir.fx_to_mlir import FXToMLIRConverter
    from tensorrt_llm._torch.auto_deploy.mlir.mlir_to_fx import MLIRToFXConverter

    hidden = 256
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((2, hidden * 2), device="meta", dtype=torch.bfloat16)
    split = graph.call_function(
        torch.ops.aten.split_with_sizes.default,
        args=(x, [hidden, hidden], -1),
    )
    split.meta["val"] = (
        torch.empty((2, hidden), device="meta", dtype=torch.bfloat16),
        torch.empty((2, hidden), device="meta", dtype=torch.bfloat16),
    )
    gate = graph.call_function(operator.getitem, args=(split, 0))
    gate.meta["val"] = torch.empty((2, hidden), device="meta", dtype=torch.bfloat16)
    up = graph.call_function(operator.getitem, args=(split, 1))
    up.meta["val"] = torch.empty((2, hidden), device="meta", dtype=torch.bfloat16)
    silu = graph.call_function(torch.ops.aten.silu.default, args=(gate,))
    silu.meta["val"] = torch.empty((2, hidden), device="meta", dtype=torch.bfloat16)
    mul = graph.call_function(torch.ops.aten.mul.Tensor, args=(silu, up))
    mul.meta["val"] = torch.empty((2, hidden), device="meta", dtype=torch.bfloat16)
    graph.output(mul)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    converter = FXToMLIRConverter(gm)
    mlir_module = converter.convert()
    subgraphs = discover_fusible_subgraphs(mlir_module)
    assert len(subgraphs) == 1
    sg = subgraphs[0]

    kernel_fn = generate_kernel_from_subgraph(sg)
    sg_hash = KernelCache.hash_subgraph(sg)
    replace_subgraph_with_fused_op(sg, kernel_fn, sg_hash, converter.metadata)

    new_gm = MLIRToFXConverter(gm).convert(mlir_module, converter.metadata)

    xt = torch.randn(2, hidden * 2, device="cuda", dtype=torch.bfloat16)
    result = new_gm(xt)
    expected = torch.nn.functional.silu(xt[..., :hidden]) * xt[..., hidden:]
    torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# FP8 activation quantization for quant-linear consumers
# ---------------------------------------------------------------------------


def _build_relu_quant_fp8_module(hidden: int = 128):
    """Build MLIR module for relu + static-scale FP8 quantization."""
    t_bf16 = TensorType(BFloat16Type(), [2, hidden])
    t_scale = TensorType(Float32Type(), [])
    t_fp8 = TensorType(Float8E4M3FNType(), [2, hidden])

    block = Block()
    x = block.insert_arg(t_bf16, 0)
    scale = block.insert_arg(t_scale, 1)

    relu = AdRelu.build(operands=[x], result_types=[t_bf16])
    block.add_op(relu)

    quant = AdQuantFP8.build(operands=[relu.output, scale], result_types=[t_fp8])
    block.add_op(quant)

    out_op = AdGraphOutput.build(operands=[[quant.output]])
    block.add_op(out_op)

    return ModuleOp(Region([block]))


def test_relu_quant_fp8_fuses_and_generates_kernel():
    """ReLU + FP8 quantization should fuse as one generated kernel."""
    mlir_mod = _build_relu_quant_fp8_module()

    subgraphs = discover_fusible_subgraphs(mlir_mod)
    assert len(subgraphs) == 1
    assert [op.name for op in subgraphs[0].ops] == ["ad.relu", "ad.quant_fp8"]

    kernel_fn = generate_kernel_from_subgraph(subgraphs[0])
    assert callable(kernel_fn)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_relu_quant_fp8_numerical_correctness():
    """Generated ReLU + FP8 quantization kernel matches torch reference."""
    hidden = 128
    mlir_mod = _build_relu_quant_fp8_module(hidden)
    subgraphs = discover_fusible_subgraphs(mlir_mod)
    kernel_fn = generate_kernel_from_subgraph(subgraphs[0])

    x = torch.randn(2, hidden, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor(0.5, device="cuda", dtype=torch.float32)
    result = kernel_fn(x, scale)

    expected = (
        torch.relu(x.float())
        .div(scale)
        .clamp(
            float(torch.finfo(torch.float8_e4m3fn).min),
            float(torch.finfo(torch.float8_e4m3fn).max),
        )
        .to(torch.float8_e4m3fn)
    )
    torch.testing.assert_close(result[0].float(), expected.float(), atol=0, rtol=0)


def test_mlir_elementwise_fusion_rewrites_fp8_quant_linear_consumer():
    """The transform emits MLIR quantization and rewrites the linear to prequant."""
    from tensorrt_llm._torch.auto_deploy.transform.interface import SharedConfig
    from tensorrt_llm._torch.auto_deploy.transform.library.mlir_elementwise_fusion import (
        MLIRElementwiseFusion,
        MLIRElementwiseFusionConfig,
    )
    from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

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
        args=(relu, weight, None, input_scale, weight_scale),
    )
    linear.meta["val"] = torch.empty((2, 64), device="meta", dtype=torch.bfloat16)
    graph.output(linear)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    transform = MLIRElementwiseFusion(
        MLIRElementwiseFusionConfig(stage="post_load_fusion", enabled=True)
    )
    new_gm, info = transform._apply(
        gm,
        cm=None,
        factory=None,
        shared_config=SharedConfig(local_rank=0, world_size=1),
    )

    assert info.num_matches == 1
    assert not any(
        is_op(node, torch.ops.auto_deploy.trtllm_quant_fp8_linear) for node in new_gm.graph.nodes
    )
    assert any(
        is_op(node, torch.ops.auto_deploy.trtllm_fp8_prequant_linear) for node in new_gm.graph.nodes
    )
    assert any("mlir_fused_" in str(node.target) for node in new_gm.graph.nodes)


def test_mlir_elementwise_fusion_rewrites_rmsnorm_fp8_quant_linear_consumer():
    """RMSNorm -> FP8 quant-linear is handled by mlir_elementwise_fusion."""
    from tensorrt_llm._torch.auto_deploy.transform.interface import SharedConfig
    from tensorrt_llm._torch.auto_deploy.transform.library.mlir_elementwise_fusion import (
        MLIRElementwiseFusion,
        MLIRElementwiseFusionConfig,
    )
    from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    norm_weight = graph.placeholder("norm_weight")
    weight = graph.placeholder("weight")
    input_scale = graph.placeholder("input_scale")
    weight_scale = graph.placeholder("weight_scale")
    x.meta["val"] = torch.empty((2, 128), device="meta", dtype=torch.bfloat16)
    norm_weight.meta["val"] = torch.empty((128,), device="meta", dtype=torch.bfloat16)
    weight.meta["val"] = torch.empty((64, 128), device="meta", dtype=torch.float8_e4m3fn)
    input_scale.meta["val"] = torch.empty((), device="meta", dtype=torch.float32)
    weight_scale.meta["val"] = torch.empty((), device="meta", dtype=torch.float32)

    norm = graph.call_function(
        torch.ops.auto_deploy.flashinfer_rms_norm.default,
        args=(x, norm_weight, 1e-5),
    )
    norm.meta["val"] = torch.empty((2, 128), device="meta", dtype=torch.bfloat16)
    linear = graph.call_function(
        torch.ops.auto_deploy.trtllm_quant_fp8_linear.default,
        args=(norm, weight, None, input_scale, weight_scale),
    )
    linear.meta["val"] = torch.empty((2, 64), device="meta", dtype=torch.bfloat16)
    graph.output(linear)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    transform = MLIRElementwiseFusion(
        MLIRElementwiseFusionConfig(stage="post_load_fusion", enabled=True)
    )
    new_gm, info = transform._apply(
        gm,
        cm=None,
        factory=None,
        shared_config=SharedConfig(local_rank=0, world_size=1),
    )

    assert info.num_matches == 1
    assert not any(
        is_op(node, torch.ops.auto_deploy.trtllm_quant_fp8_linear) for node in new_gm.graph.nodes
    )
    assert any(
        is_op(node, torch.ops.auto_deploy.trtllm_fp8_prequant_linear) for node in new_gm.graph.nodes
    )
    assert any("mlir_fused_" in str(node.target) for node in new_gm.graph.nodes)


# ---------------------------------------------------------------------------
# ReLU2 + NVFP4 quant-linear consumer rewrite
# ---------------------------------------------------------------------------


def _build_relu2_nvfp4_quant_linear_graph(shared_relu2_user: bool = False):
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
    if shared_relu2_user:
        shared = graph.call_function(torch.ops.aten.slice.Tensor, args=(relu2, -1, 0, 64, 1))
        shared.meta["val"] = torch.empty((2, 64), device="meta", dtype=torch.float16)
        graph.output((linear, shared))
    else:
        graph.output(linear)
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def test_mlir_elementwise_fusion_rewrites_relu2_nvfp4_quant_linear_consumer():
    """ReLU2 -> NVFP4 quant-linear is owned by mlir_elementwise_fusion."""
    from tensorrt_llm._torch.auto_deploy.transform.interface import SharedConfig
    from tensorrt_llm._torch.auto_deploy.transform.library.mlir_elementwise_fusion import (
        MLIRElementwiseFusion,
        MLIRElementwiseFusionConfig,
    )
    from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

    gm = _build_relu2_nvfp4_quant_linear_graph()
    transform = MLIRElementwiseFusion(
        MLIRElementwiseFusionConfig(stage="post_load_fusion", enabled=True)
    )
    new_gm, info = transform._apply(
        gm,
        cm=None,
        factory=None,
        shared_config=SharedConfig(local_rank=0, world_size=1),
    )

    assert info.num_matches == 1
    assert not any(
        is_op(node, torch.ops.auto_deploy.torch_quant_nvfp4_linear) for node in new_gm.graph.nodes
    )
    assert any(
        is_op(node, torch.ops.auto_deploy.trtllm_fused_relu2_quant_nvfp4.default)
        for node in new_gm.graph.nodes
    )
    assert any(
        is_op(node, torch.ops.auto_deploy.trtllm_nvfp4_prequant_linear.default)
        for node in new_gm.graph.nodes
    )


def test_mlir_elementwise_fusion_skips_shared_relu2_nvfp4_quant_linear_consumer():
    """Shared ReLU2 users keep the original NVFP4 quant-linear path."""
    from tensorrt_llm._torch.auto_deploy.transform.interface import SharedConfig
    from tensorrt_llm._torch.auto_deploy.transform.library.mlir_elementwise_fusion import (
        MLIRElementwiseFusion,
        MLIRElementwiseFusionConfig,
    )
    from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

    gm = _build_relu2_nvfp4_quant_linear_graph(shared_relu2_user=True)
    transform = MLIRElementwiseFusion(
        MLIRElementwiseFusionConfig(stage="post_load_fusion", enabled=True)
    )
    new_gm, info = transform._apply(
        gm,
        cm=None,
        factory=None,
        shared_config=SharedConfig(local_rank=0, world_size=1),
    )

    assert info.num_matches == 0
    assert any(
        is_op(node, torch.ops.auto_deploy.torch_quant_nvfp4_linear) for node in new_gm.graph.nodes
    )
    assert not any(
        is_op(node, torch.ops.auto_deploy.trtllm_fused_relu2_quant_nvfp4.default)
        for node in new_gm.graph.nodes
    )


# ---------------------------------------------------------------------------
# RMSNorm + NVFP4 quant-linear consumer rewrites
# ---------------------------------------------------------------------------


def _run_mlir_elementwise_fusion(gm):
    from tensorrt_llm._torch.auto_deploy.transform.interface import SharedConfig
    from tensorrt_llm._torch.auto_deploy.transform.library.mlir_elementwise_fusion import (
        MLIRElementwiseFusion,
        MLIRElementwiseFusionConfig,
    )

    transform = MLIRElementwiseFusion(
        MLIRElementwiseFusionConfig(stage="post_load_fusion", enabled=True)
    )
    return transform._apply(
        gm,
        cm=None,
        factory=None,
        shared_config=SharedConfig(local_rank=0, world_size=1),
    )


def _make_nvfp4_graph_root(hidden_size=64):
    root = torch.nn.Module()
    root.register_buffer("norm_weight", torch.empty(hidden_size, dtype=torch.bfloat16))
    root.register_buffer("weight_fp4", torch.empty(32, hidden_size // 2, dtype=torch.uint8))
    root.register_buffer("input_scale", torch.empty(1, dtype=torch.float32))
    root.register_buffer("weight_scale", torch.empty(128, 4, dtype=torch.uint8))
    root.register_buffer("alpha", torch.empty(1, dtype=torch.float32))
    return root


def test_mlir_elementwise_fusion_rewrites_gated_rmsnorm_nvfp4_quant_linear_consumer():
    from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

    root = _make_nvfp4_graph_root()
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    gate = graph.placeholder("gate")
    norm_weight = graph.get_attr("norm_weight")
    weight_fp4 = graph.get_attr("weight_fp4")
    input_scale = graph.get_attr("input_scale")
    weight_scale = graph.get_attr("weight_scale")
    alpha = graph.get_attr("alpha")

    norm_out = graph.call_function(
        torch.ops.auto_deploy.triton_rmsnorm_gated.default,
        args=(x, norm_weight, gate, 1e-5, 256, False),
    )
    gemm_out = graph.call_function(
        torch.ops.auto_deploy.torch_quant_nvfp4_linear.default,
        args=(norm_out, weight_fp4, None, input_scale, weight_scale, alpha),
    )
    graph.output(gemm_out)
    x.meta["val"] = torch.empty((2, 64), device="meta", dtype=torch.bfloat16)
    gate.meta["val"] = torch.empty((2, 64), device="meta", dtype=torch.bfloat16)
    norm_out.meta["val"] = torch.empty((2, 64), device="meta", dtype=torch.bfloat16)
    gemm_out.meta["val"] = torch.empty((2, 32), device="meta", dtype=torch.bfloat16)

    new_gm, info = _run_mlir_elementwise_fusion(torch.fx.GraphModule(root, graph))

    assert info.num_matches == 1
    assert any(
        is_op(node, torch.ops.auto_deploy.trtllm_fused_gated_rmsnorm_quant_nvfp4)
        for node in new_gm.graph.nodes
    )
    assert any(
        is_op(node, torch.ops.auto_deploy.trtllm_nvfp4_prequant_linear)
        for node in new_gm.graph.nodes
    )
    assert not any(
        is_op(node, torch.ops.auto_deploy.torch_quant_nvfp4_linear) for node in new_gm.graph.nodes
    )
    assert not any(
        is_op(node, torch.ops.auto_deploy.triton_rmsnorm_gated) for node in new_gm.graph.nodes
    )


def test_mlir_elementwise_fusion_rewrites_add_rmsnorm_nvfp4_quant_linear_consumer():
    from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

    hidden_size = 2048
    root = _make_nvfp4_graph_root(hidden_size)
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    residual = graph.placeholder("residual")
    norm_weight = graph.get_attr("norm_weight")
    weight_fp4 = graph.get_attr("weight_fp4")
    input_scale = graph.get_attr("input_scale")
    weight_scale = graph.get_attr("weight_scale")
    alpha = graph.get_attr("alpha")

    add_out = graph.call_function(torch.ops.aten.add.Tensor, args=(x, residual))
    norm_out = graph.call_function(
        torch.ops.auto_deploy.flashinfer_rms_norm.default,
        args=(add_out, norm_weight, 1e-5),
    )
    gemm_out = graph.call_function(
        torch.ops.auto_deploy.torch_quant_nvfp4_linear.default,
        args=(norm_out, weight_fp4, None, input_scale, weight_scale, alpha),
    )
    graph.output((gemm_out, add_out))
    x.meta["val"] = torch.empty((2, hidden_size), device="meta", dtype=torch.bfloat16)
    residual.meta["val"] = torch.empty((2, hidden_size), device="meta", dtype=torch.bfloat16)
    add_out.meta["val"] = torch.empty((2, hidden_size), device="meta", dtype=torch.bfloat16)
    norm_out.meta["val"] = torch.empty((2, hidden_size), device="meta", dtype=torch.bfloat16)
    gemm_out.meta["val"] = torch.empty((2, 32), device="meta", dtype=torch.bfloat16)

    new_gm, info = _run_mlir_elementwise_fusion(torch.fx.GraphModule(root, graph))

    assert info.num_matches == 1
    assert any(
        is_op(node, torch.ops.auto_deploy.trtllm_fused_add_rmsnorm_quant_nvfp4)
        for node in new_gm.graph.nodes
    )
    assert any(
        is_op(node, torch.ops.auto_deploy.trtllm_nvfp4_prequant_linear)
        for node in new_gm.graph.nodes
    )
    assert not any(
        is_op(node, torch.ops.auto_deploy.torch_quant_nvfp4_linear) for node in new_gm.graph.nodes
    )
    assert not any(
        is_op(node, torch.ops.auto_deploy.flashinfer_rms_norm) for node in new_gm.graph.nodes
    )
    assert not any(is_op(node, torch.ops.aten.add.Tensor) for node in new_gm.graph.nodes)


def test_mlir_elementwise_fusion_uses_out_add_rmsnorm_nvfp4_for_mixed_norm_consumers():
    from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

    hidden_size = 2048
    root = _make_nvfp4_graph_root(hidden_size)
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    residual = graph.placeholder("residual")
    norm_weight = graph.get_attr("norm_weight")
    weight_fp4 = graph.get_attr("weight_fp4")
    input_scale = graph.get_attr("input_scale")
    weight_scale = graph.get_attr("weight_scale")
    alpha = graph.get_attr("alpha")

    add_out = graph.call_function(torch.ops.aten.add.Tensor, args=(x, residual))
    norm_out = graph.call_function(
        torch.ops.auto_deploy.flashinfer_rms_norm.default,
        args=(add_out, norm_weight, 1e-5),
    )
    extra_consumer = graph.call_function(torch.ops.aten.mul.Tensor, args=(norm_out, 2.0))
    gemm_out = graph.call_function(
        torch.ops.auto_deploy.torch_quant_nvfp4_linear.default,
        args=(norm_out, weight_fp4, None, input_scale, weight_scale, alpha),
    )
    graph.output((gemm_out, extra_consumer, add_out))
    x.meta["val"] = torch.empty((2, hidden_size), device="meta", dtype=torch.bfloat16)
    residual.meta["val"] = torch.empty((2, hidden_size), device="meta", dtype=torch.bfloat16)
    add_out.meta["val"] = torch.empty((2, hidden_size), device="meta", dtype=torch.bfloat16)
    norm_out.meta["val"] = torch.empty((2, hidden_size), device="meta", dtype=torch.bfloat16)
    extra_consumer.meta["val"] = torch.empty((2, hidden_size), device="meta", dtype=torch.bfloat16)
    gemm_out.meta["val"] = torch.empty((2, 32), device="meta", dtype=torch.bfloat16)

    new_gm, info = _run_mlir_elementwise_fusion(torch.fx.GraphModule(root, graph))

    assert info.num_matches >= 1
    assert any(
        is_op(node, torch.ops.auto_deploy.trtllm_fused_add_rmsnorm_out_quant_nvfp4)
        for node in new_gm.graph.nodes
    )
    assert any(
        is_op(node, torch.ops.auto_deploy.trtllm_nvfp4_prequant_linear)
        for node in new_gm.graph.nodes
    )
    assert not any(
        is_op(node, torch.ops.auto_deploy.torch_quant_nvfp4_linear) for node in new_gm.graph.nodes
    )
    assert not any(
        is_op(node, torch.ops.auto_deploy.flashinfer_rms_norm) for node in new_gm.graph.nodes
    )


def test_mlir_elementwise_fusion_rewrites_allreduce_rmsnorm_nvfp4_quant_linear_consumer():
    from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

    root = _make_nvfp4_graph_root()
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    residual = graph.placeholder("residual")
    norm_weight = graph.get_attr("norm_weight")
    weight_fp4 = graph.get_attr("weight_fp4")
    input_scale = graph.get_attr("input_scale")
    weight_scale = graph.get_attr("weight_scale")
    alpha = graph.get_attr("alpha")

    fused_allreduce = graph.call_function(
        torch.ops.dist.trtllm_fused_allreduce_residual_rmsnorm.default,
        args=(x, residual, norm_weight, 1e-5, "AUTO"),
    )
    norm_out = graph.call_function(operator.getitem, args=(fused_allreduce, 0))
    residual_out = graph.call_function(operator.getitem, args=(fused_allreduce, 1))
    gemm_out = graph.call_function(
        torch.ops.auto_deploy.torch_quant_nvfp4_linear.default,
        args=(norm_out, weight_fp4, None, input_scale, weight_scale, alpha),
    )
    graph.output((gemm_out, residual_out))
    x.meta["val"] = torch.empty((2, 64), device="meta", dtype=torch.bfloat16)
    residual.meta["val"] = torch.empty((2, 64), device="meta", dtype=torch.bfloat16)
    norm_out.meta["val"] = torch.empty((2, 64), device="meta", dtype=torch.bfloat16)
    residual_out.meta["val"] = torch.empty((2, 64), device="meta", dtype=torch.bfloat16)
    gemm_out.meta["val"] = torch.empty((2, 32), device="meta", dtype=torch.bfloat16)

    new_gm, info = _run_mlir_elementwise_fusion(torch.fx.GraphModule(root, graph))

    assert info.num_matches == 1
    assert any(
        is_op(node, torch.ops.dist.trtllm_fused_allreduce_residual_rmsnorm_quant_nvfp4)
        for node in new_gm.graph.nodes
    )
    assert any(
        is_op(node, torch.ops.auto_deploy.trtllm_nvfp4_prequant_linear)
        for node in new_gm.graph.nodes
    )
    assert not any(
        is_op(node, torch.ops.auto_deploy.torch_quant_nvfp4_linear) for node in new_gm.graph.nodes
    )
    assert not any(
        is_op(node, torch.ops.dist.trtllm_fused_allreduce_residual_rmsnorm)
        for node in new_gm.graph.nodes
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_full_fx_roundtrip_with_replacement():
    """Full pipeline: FX → MLIR → decompose → discover → codegen → replace → MLIR → FX.

    Verifies the generated FX graph module produces correct outputs when the
    fused kernel is actually called through the graph.
    """
    import tensorrt_llm._torch.auto_deploy.custom_ops.normalization.rms_norm  # noqa: F401

    # Import custom ops so they're registered
    import tensorrt_llm._torch.auto_deploy.mlir  # noqa: F401
    from tensorrt_llm._torch.auto_deploy.mlir.codegen.kernel_cache import KernelCache
    from tensorrt_llm._torch.auto_deploy.mlir.fusion.subgraph_replace import (
        replace_subgraph_with_fused_op,
    )
    from tensorrt_llm._torch.auto_deploy.mlir.fx_to_mlir import FXToMLIRConverter
    from tensorrt_llm._torch.auto_deploy.mlir.mlir_to_fx import MLIRToFXConverter

    hidden = 128

    class AddNormModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.ones(hidden, device="cuda", dtype=torch.bfloat16)
            )
            self.eps = 1e-5

        def forward(self, x, residual):
            added = x + residual
            norm = torch.ops.auto_deploy.torch_rmsnorm(added, self.weight, self.eps)
            return norm, added

    model = AddNormModel()
    x = torch.randn(2, 8, hidden, device="cuda", dtype=torch.bfloat16)
    res = torch.randn_like(x)

    # Export to FX graph
    from torch.export import Dim

    from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm

    gm = torch_export_to_gm(
        model,
        args=(x, res),
        dynamic_shapes=({0: Dim.DYNAMIC}, {0: Dim.DYNAMIC}),
        clone=True,
    )

    # Step 1: FX → MLIR
    converter = FXToMLIRConverter(gm)
    mlir_module = converter.convert()

    # Step 2: Decompose
    num_decomposed = run_decomposition(mlir_module)
    assert num_decomposed >= 1

    # Step 3: Discover
    subgraphs = discover_fusible_subgraphs(mlir_module)
    assert len(subgraphs) >= 1
    sg = max(subgraphs, key=lambda s: len(s.ops))

    # Step 4: Generate kernel + replace
    kernel_fn = generate_kernel_from_subgraph(sg)
    sg_hash = KernelCache.hash_subgraph(sg)
    replace_subgraph_with_fused_op(sg, kernel_fn, sg_hash, converter.metadata)

    # Step 5: MLIR → FX
    back_converter = MLIRToFXConverter(gm)
    new_gm = back_converter.convert(mlir_module, converter.metadata)

    # Step 6: Run the new graph and verify correctness
    result = new_gm(x.clone(), res.clone())
    ref = model(x.clone(), res.clone())

    assert len(result) == len(ref), f"Output count mismatch: {len(result)} vs {len(ref)}"
    torch.testing.assert_close(result[0], ref[0], atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(result[1], ref[1], atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# MoE EP sharding mask: floordiv + eq + mul  (integer + bool + bf16 mixed)
# Pattern from DeepSeek-R1 Expert Parallelism all-reduce sharding:
#   floordiv(expert_ids, experts_per_rank) → which rank owns each expert
#   eq(rank_ids, ep_rank) → boolean mask for local experts
#   mul(routing_scores, mask) → zero out non-local expert scores
# ---------------------------------------------------------------------------


def _build_moe_ep_mask_module(top_k: int = 8, experts_per_rank: int = 32, ep_rank: int = 0):
    """Build MLIR module for the MoE EP sharding mask pattern.

    Inputs:  (scores: bf16 [B, top_k], expert_ids: int32 [B, top_k])
    Ops:     floordiv → eq → mul
    Outputs: (masked_scores: bf16 [B, top_k])
    """
    from xdsl.dialects.builtin import IntegerAttr, IntegerType

    t_bf16 = TensorType(BFloat16Type(), [8, top_k])
    t_i32 = TensorType(IntegerType(32), [8, top_k])
    t_bool = TensorType(IntegerType(1), [8, top_k])

    block = Block()
    scores = block.insert_arg(t_bf16, 0)
    expert_ids = block.insert_arg(t_i32, 1)

    floordiv_op = AdFloorDiv.build(
        operands=[expert_ids],
        attributes={"divisor": IntegerAttr(experts_per_rank, IntegerType(64))},
        result_types=[t_i32],
    )
    block.add_op(floordiv_op)

    eq_op = AdEq.build(
        operands=[floordiv_op.output],
        attributes={"value": IntegerAttr(ep_rank, IntegerType(64))},
        result_types=[t_bool],
    )
    block.add_op(eq_op)

    mul_op = AdMul.build(
        operands=[scores, eq_op.output],
        result_types=[t_bf16],
    )
    block.add_op(mul_op)

    out_op = AdGraphOutput.build(operands=[[mul_op.output]])
    block.add_op(out_op)

    return ModuleOp(Region([block]))


def test_moe_ep_mask_fusion_discovery():
    """Floordiv + eq + mul should be discovered as a single fusible subgraph."""
    mlir_mod = _build_moe_ep_mask_module()

    subgraphs = discover_fusible_subgraphs(mlir_mod)
    assert len(subgraphs) == 1, f"Expected 1 subgraph, got {len(subgraphs)}"
    assert len(subgraphs[0].ops) == 3, f"Expected 3 ops, got {len(subgraphs[0].ops)}"


def test_moe_ep_mask_kernel_generation():
    """Triton kernel can be generated for the mixed-type floordiv+eq+mul pattern."""
    mlir_mod = _build_moe_ep_mask_module()
    subgraphs = discover_fusible_subgraphs(mlir_mod)
    assert len(subgraphs) == 1

    kernel_fn = generate_kernel_from_subgraph(subgraphs[0])
    assert callable(kernel_fn)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_moe_ep_mask_numerical_correctness():
    """Generated kernel matches the reference floordiv+eq+mul computation."""
    experts_per_rank = 32
    ep_rank = 0
    top_k = 8
    batch = 8

    mlir_mod = _build_moe_ep_mask_module(
        top_k=top_k,
        experts_per_rank=experts_per_rank,
        ep_rank=ep_rank,
    )
    subgraphs = discover_fusible_subgraphs(mlir_mod)
    kernel_fn = generate_kernel_from_subgraph(subgraphs[0])

    scores = torch.randn(batch, top_k, device="cuda", dtype=torch.bfloat16)
    expert_ids = torch.randint(0, 256, (batch, top_k), device="cuda", dtype=torch.int32)

    # Subgraph input order: expert_ids first (used by floordiv), scores second (used by mul)
    result = kernel_fn(expert_ids, scores)

    rank_ids = expert_ids // experts_per_rank
    mask = rank_ids == ep_rank
    expected = scores * mask

    torch.testing.assert_close(result[0], expected, atol=0, rtol=0)
