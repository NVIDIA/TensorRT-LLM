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

import pytest

xdsl = pytest.importorskip("xdsl")

import torch  # noqa: E402
from xdsl.dialects.builtin import (  # noqa: E402
    BFloat16Type,
    Float64Type,
    FloatAttr,
    ModuleOp,
    TensorType,
)
from xdsl.ir import Block, Region  # noqa: E402

from tensorrt_llm._torch.auto_deploy.mlir.codegen.triton_emitter import (  # noqa: E402
    generate_kernel_from_subgraph,
)
from tensorrt_llm._torch.auto_deploy.mlir.decompose import run_decomposition  # noqa: E402
from tensorrt_llm._torch.auto_deploy.mlir.dialect import (  # noqa: E402
    AdAdd,
    AdGraphOutput,
    AdMul,
    AdRMSNorm,
    AdSilu,
)
from tensorrt_llm._torch.auto_deploy.mlir.fusion.subgraph_discovery import (  # noqa: E402
    discover_fusible_subgraphs,
)


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
