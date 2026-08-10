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

"""Tests for fusible subgraph discovery."""

import pytest

xdsl = pytest.importorskip("xdsl")

from xdsl.dialects.builtin import (  # noqa: E402
    BFloat16Type,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    TensorType,
)
from xdsl.ir import Block, Region  # noqa: E402

from tensorrt_llm._torch.auto_deploy.mlir.dialect import (  # noqa: E402
    AdAdd,
    AdGraphOutput,
    AdMul,
    AdOpaque,
    AdReduceMean,
    AdRsqrt,
)
from tensorrt_llm._torch.auto_deploy.mlir.fusion.subgraph_discovery import (  # noqa: E402
    discover_fusible_subgraphs,
)


def test_elementwise_chain_fused():
    """Chain of add -> mul -> mul should be one subgraph."""
    t = TensorType(BFloat16Type(), [2, 128])
    block = Block()
    x = block.insert_arg(t, 0)
    y = block.insert_arg(t, 1)
    z = block.insert_arg(t, 2)
    add = AdAdd.build(operands=[x, y], result_types=[t])
    block.add_op(add)
    mul1 = AdMul.build(operands=[add.output, z], result_types=[t])
    block.add_op(mul1)
    mul2 = AdMul.build(operands=[mul1.output, x], result_types=[t])
    block.add_op(mul2)
    out = AdGraphOutput.build(operands=[[mul2.output]])
    block.add_op(out)
    mod = ModuleOp(Region([block]))

    subgraphs = discover_fusible_subgraphs(mod)
    assert len(subgraphs) == 1
    assert len(subgraphs[0].ops) == 3


def test_opaque_breaks_subgraph():
    """Opaque op between elementwise ops creates separate groups."""
    t = TensorType(BFloat16Type(), [2, 128])
    block = Block()
    x = block.insert_arg(t, 0)
    y = block.insert_arg(t, 1)
    add = AdAdd.build(operands=[x, y], result_types=[t])
    block.add_op(add)
    opaque = AdOpaque.build(
        operands=[[add.output]],
        attributes={"op_name": StringAttr("some_op"), "node_key": StringAttr("k")},
        result_types=[[t]],
    )
    block.add_op(opaque)
    mul = AdMul.build(operands=[opaque.outputs[0], x], result_types=[t])
    block.add_op(mul)
    out = AdGraphOutput.build(operands=[[mul.output]])
    block.add_op(out)
    mod = ModuleOp(Region([block]))

    subgraphs = discover_fusible_subgraphs(mod)
    # Single-op subgraphs not returned, so expect 0
    assert len(subgraphs) == 0


def test_reduction_included_in_subgraph():
    """Elementwise + last-dim reduction should fuse."""
    t = TensorType(BFloat16Type(), [2, 128])
    t_r = TensorType(BFloat16Type(), [2, 1])
    block = Block()
    x = block.insert_arg(t, 0)
    sq = AdMul.build(operands=[x, x], result_types=[t])
    block.add_op(sq)
    mean = AdReduceMean.build(
        operands=[sq.output],
        attributes={
            "dim": IntegerAttr(-1, IntegerType(64)),
            "keepdim": IntegerAttr(1, IntegerType(1)),
        },
        result_types=[t_r],
    )
    block.add_op(mean)
    inv = AdRsqrt.build(operands=[mean.output], result_types=[t_r])
    block.add_op(inv)
    out = AdGraphOutput.build(operands=[[inv.output]])
    block.add_op(out)
    mod = ModuleOp(Region([block]))

    subgraphs = discover_fusible_subgraphs(mod)
    assert len(subgraphs) == 1
    assert len(subgraphs[0].ops) == 3


def test_multi_output_subgraph():
    """Subgraph where intermediate value is also used externally."""
    t = TensorType(BFloat16Type(), [2, 128])
    block = Block()
    x = block.insert_arg(t, 0)
    y = block.insert_arg(t, 1)
    add = AdAdd.build(operands=[x, y], result_types=[t])
    block.add_op(add)
    mul = AdMul.build(operands=[add.output, x], result_types=[t])
    block.add_op(mul)
    # Output uses both add (intermediate) and mul (final)
    out = AdGraphOutput.build(operands=[[mul.output, add.output]])
    block.add_op(out)
    mod = ModuleOp(Region([block]))

    subgraphs = discover_fusible_subgraphs(mod)
    assert len(subgraphs) == 1
    assert len(subgraphs[0].outputs) == 2  # both add and mul are outputs
