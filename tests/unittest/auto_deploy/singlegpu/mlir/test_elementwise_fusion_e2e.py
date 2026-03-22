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
    AdRMSNorm,
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

    # After decomposition, the subgraph inputs (in order) are:
    #   input[0]: x (2D block arg)
    #   input[1]: residual (2D block arg)
    #   input[2]: eps splat output (2D scalar-like, from AdSplat outside subgraph)
    #   input[3]: weight (1D block arg)
    # And the subgraph outputs are:
    #   output[0]: added = x + residual (from ad.add)
    #   output[1]: normed = rmsnorm(added, weight) (from ad.mul)
    xt = torch.randn(2, hidden, device="cuda", dtype=torch.bfloat16)
    rest = torch.randn_like(xt)
    wt = torch.ones(hidden, device="cuda", dtype=torch.bfloat16)

    # Build input tensors matching subgraph.inputs shapes
    input_tensors = []
    block_arg_tensors = [xt, rest, wt]  # ordered by block arg index
    block_arg_idx = 0
    for inp in largest_sg.inputs:
        owner = inp.owner
        if hasattr(owner, "name") and owner.name == "ad.splat":
            # Eps constant from splat op — use the attribute value
            from xdsl.dialects.builtin import FloatAttr as FA

            eps_val = owner.attributes["value"]
            scalar = eps_val.value.data if isinstance(eps_val, FA) else float(str(eps_val))
            shape = tuple(inp.type.get_shape())
            input_tensors.append(torch.full(shape, scalar, device="cuda", dtype=torch.bfloat16))
        else:
            input_tensors.append(block_arg_tensors[block_arg_idx])
            block_arg_idx += 1

    # Run generated kernel
    result = kernel_fn(*input_tensors)

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
