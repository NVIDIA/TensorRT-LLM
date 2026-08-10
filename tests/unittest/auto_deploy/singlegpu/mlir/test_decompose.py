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

"""Tests for op decomposition into primitives."""

import pytest

xdsl = pytest.importorskip("xdsl")

from xdsl.dialects.builtin import (  # noqa: E402
    BFloat16Type,
    Float64Type,
    FloatAttr,
    ModuleOp,
    TensorType,
)
from xdsl.ir import Block, Region  # noqa: E402

from tensorrt_llm._torch.auto_deploy.mlir.decompose import run_decomposition  # noqa: E402
from tensorrt_llm._torch.auto_deploy.mlir.dialect import (  # noqa: E402
    AdAdd,
    AdGraphOutput,
    AdRMSNorm,
)


def _build_rmsnorm_ir():
    """Build IR: block_args(x, w) -> ad.rmsnorm -> ad.graph_output."""
    t = TensorType(BFloat16Type(), [2, 8, 128])
    tw = TensorType(BFloat16Type(), [128])
    eps = FloatAttr(1e-5, Float64Type())
    block = Block()
    x = block.insert_arg(t, 0)
    w = block.insert_arg(tw, 1)
    norm = AdRMSNorm.build(operands=[x, w], attributes={"eps": eps}, result_types=[t])
    block.add_op(norm)
    out = AdGraphOutput.build(operands=[[norm.output]])
    block.add_op(out)
    return ModuleOp(Region([block]))


def test_rmsnorm_decomposes_to_primitives():
    """ad.rmsnorm should be decomposed into primitive ops."""
    mod = _build_rmsnorm_ir()
    num = run_decomposition(mod)
    assert num == 1, f"Expected 1 decomposition, got {num}"
    op_names = [op.name for op in mod.body.block.ops]
    assert "ad.rmsnorm" not in op_names, "ad.rmsnorm should be decomposed"
    assert "ad.mul" in op_names
    assert "ad.reduce_mean" in op_names
    assert "ad.rsqrt" in op_names


def test_rmsnorm_decomposition_op_count():
    """Decomposed rmsnorm should produce exactly 7 primitive ops."""
    mod = _build_rmsnorm_ir()
    run_decomposition(mod)
    op_names = [op.name for op in mod.body.block.ops if op.name != "ad.graph_output"]
    # mul(x,x), reduce_mean, splat(eps), add(var,eps), rsqrt, mul(x,inv), mul(normed,w)
    assert len(op_names) == 7, f"Expected 7 primitive ops, got {len(op_names)}: {op_names}"


def test_no_decomposition_for_non_decomposable():
    """ad.add should not be decomposed (already primitive)."""
    t = TensorType(BFloat16Type(), [2, 8, 128])
    block = Block()
    x = block.insert_arg(t, 0)
    y = block.insert_arg(t, 1)
    add = AdAdd.build(operands=[x, y], result_types=[t])
    block.add_op(add)
    out = AdGraphOutput.build(operands=[[add.output]])
    block.add_op(out)
    mod = ModuleOp(Region([block]))
    num = run_decomposition(mod)
    assert num == 0


def test_rmsnorm_output_wiring():
    """After decomposition, graph_output should consume the final mul output."""
    mod = _build_rmsnorm_ir()
    run_decomposition(mod)
    # The last op should be graph_output
    ops = list(mod.body.block.ops)
    graph_out = ops[-1]
    assert graph_out.name == "ad.graph_output"
    # graph_output's input should be the output of the last mul
    last_mul = ops[-2]
    assert last_mul.name == "ad.mul"
    assert graph_out.inputs[0] is last_mul.output
