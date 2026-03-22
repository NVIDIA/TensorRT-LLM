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

"""Tests for the MLIR fusion patterns."""

import pytest

xdsl = pytest.importorskip("xdsl")

from xdsl.dialects.builtin import (  # noqa: E402
    BFloat16Type,
    Float64Type,
    FloatAttr,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    TensorType,
)
from xdsl.ir import Block, Region  # noqa: E402

from tensorrt_llm._torch.auto_deploy.mlir.dialect import (  # noqa: E402
    AdAdd,
    AdGraphOutput,
    AdRMSNorm,
    AdToDtype,
)
from tensorrt_llm._torch.auto_deploy.mlir.patterns import run_fusion_patterns  # noqa: E402


def _build_add_rmsnorm_ir(with_cast=False):
    """Build a minimal add → (optional cast) → rmsnorm IR."""
    t = TensorType(BFloat16Type(), [2, 8, 128])
    tw = TensorType(BFloat16Type(), [128])
    eps_attr = FloatAttr(1e-5, Float64Type())

    block = Block()
    x = block.insert_arg(t, 0)
    res = block.insert_arg(t, 1)
    w = block.insert_arg(tw, 2)

    add_op = AdAdd.build(operands=[x, res], result_types=[t])
    block.add_op(add_op)

    norm_input = add_op.output

    if with_cast:
        dtype_attr = IntegerAttr(15, IntegerType(64))  # bf16 enum
        cast_op = AdToDtype.build(
            operands=[add_op.output],
            attributes={"target_dtype": dtype_attr},
            result_types=[t],
        )
        block.add_op(cast_op)
        norm_input = cast_op.output

    norm_op = AdRMSNorm.build(
        operands=[norm_input, w],
        attributes={"eps": eps_attr},
        result_types=[t],
    )
    block.add_op(norm_op)

    out_op = AdGraphOutput.build(operands=[[norm_op.output, add_op.output]])
    block.add_op(out_op)

    return ModuleOp(Region([block]))


def test_fuse_add_rmsnorm_basic():
    """Pattern 1: ad.add → ad.rmsnorm → ad.fused_add_rmsnorm."""
    mod = _build_add_rmsnorm_ir(with_cast=False)

    # Before: add + rmsnorm
    ops_before = [op.name for op in mod.body.block.ops]
    assert "ad.add" in ops_before
    assert "ad.rmsnorm" in ops_before
    assert "ad.fused_add_rmsnorm" not in ops_before

    num_matches = run_fusion_patterns(mod)

    # After: only fused op
    ops_after = [op.name for op in mod.body.block.ops]
    assert num_matches == 1
    assert "ad.fused_add_rmsnorm" in ops_after
    assert "ad.add" not in ops_after
    assert "ad.rmsnorm" not in ops_after


def test_fuse_add_cast_rmsnorm():
    """Pattern 2: ad.add → ad.to_dtype → ad.rmsnorm → ad.fused_add_rmsnorm."""
    mod = _build_add_rmsnorm_ir(with_cast=True)

    ops_before = [op.name for op in mod.body.block.ops]
    assert "ad.to_dtype" in ops_before

    num_matches = run_fusion_patterns(mod)

    ops_after = [op.name for op in mod.body.block.ops]
    assert num_matches == 1
    assert "ad.fused_add_rmsnorm" in ops_after
    assert "ad.add" not in ops_after
    assert "ad.rmsnorm" not in ops_after
    assert "ad.to_dtype" not in ops_after


def test_fuse_multi_user_add():
    """ad.add with multiple users: one feeds rmsnorm, another feeds graph output."""
    t = TensorType(BFloat16Type(), [2, 8, 128])
    tw = TensorType(BFloat16Type(), [128])
    eps_attr = FloatAttr(1e-5, Float64Type())

    block = Block()
    x = block.insert_arg(t, 0)
    res = block.insert_arg(t, 1)
    w = block.insert_arg(tw, 2)
    moe_out = block.insert_arg(t, 3)

    # add1 has 2 users: rmsnorm + add2
    add1 = AdAdd.build(operands=[x, res], result_types=[t])
    block.add_op(add1)

    norm = AdRMSNorm.build(
        operands=[add1.output, w],
        attributes={"eps": eps_attr},
        result_types=[t],
    )
    block.add_op(norm)

    # Second add uses add1 result (multi-user)
    add2 = AdAdd.build(operands=[add1.output, moe_out], result_types=[t])
    block.add_op(add2)

    out = AdGraphOutput.build(operands=[[norm.output, add2.output]])
    block.add_op(out)

    mod = ModuleOp(Region([block]))
    num_matches = run_fusion_patterns(mod)

    assert num_matches == 1
    ops = [op.name for op in mod.body.block.ops]
    assert "ad.fused_add_rmsnorm" in ops
    assert "ad.rmsnorm" not in ops

    # add2 should still exist (it uses fused.add_result now)
    fused_ops = [op for op in mod.body.block.ops if op.name == "ad.fused_add_rmsnorm"]
    assert len(fused_ops) == 1

    # The second add should reference fused's add_result
    add2_ops = [op for op in mod.body.block.ops if op.name == "ad.add"]
    assert len(add2_ops) == 1
    assert add2_ops[0].lhs is fused_ops[0].add_result


def test_fuse_chained_add_rmsnorm():
    """Two consecutive add+rmsnorm pairs (transformer layers)."""
    t = TensorType(BFloat16Type(), [2, 8, 128])
    tw = TensorType(BFloat16Type(), [128])
    eps_attr = FloatAttr(1e-5, Float64Type())

    block = Block()
    embed = block.insert_arg(t, 0)
    attn_out = block.insert_arg(t, 1)
    mlp_out = block.insert_arg(t, 2)
    w1 = block.insert_arg(tw, 3)
    w2 = block.insert_arg(tw, 4)

    # Layer 1
    add1 = AdAdd.build(operands=[embed, attn_out], result_types=[t])
    block.add_op(add1)
    norm1 = AdRMSNorm.build(
        operands=[add1.output, w1],
        attributes={"eps": eps_attr},
        result_types=[t],
    )
    block.add_op(norm1)

    # Layer 2 (add1 feeds into add2 as residual)
    add2 = AdAdd.build(operands=[add1.output, mlp_out], result_types=[t])
    block.add_op(add2)
    norm2 = AdRMSNorm.build(
        operands=[add2.output, w2],
        attributes={"eps": eps_attr},
        result_types=[t],
    )
    block.add_op(norm2)

    out = AdGraphOutput.build(operands=[[norm1.output, norm2.output]])
    block.add_op(out)

    mod = ModuleOp(Region([block]))
    num_matches = run_fusion_patterns(mod)

    assert num_matches == 2
    ops = [op.name for op in mod.body.block.ops]
    assert ops.count("ad.fused_add_rmsnorm") == 2
    assert "ad.rmsnorm" not in ops
    assert "ad.add" not in ops


def test_no_fusion_when_no_add():
    """Rmsnorm without preceding add should not be fused."""
    t = TensorType(BFloat16Type(), [2, 8, 128])
    tw = TensorType(BFloat16Type(), [128])
    eps_attr = FloatAttr(1e-5, Float64Type())

    block = Block()
    x = block.insert_arg(t, 0)
    w = block.insert_arg(tw, 1)

    norm = AdRMSNorm.build(
        operands=[x, w],
        attributes={"eps": eps_attr},
        result_types=[t],
    )
    block.add_op(norm)
    out = AdGraphOutput.build(operands=[[norm.output]])
    block.add_op(out)

    mod = ModuleOp(Region([block]))
    num_matches = run_fusion_patterns(mod)

    assert num_matches == 0
    ops = [op.name for op in mod.body.block.ops]
    assert "ad.rmsnorm" in ops
    assert "ad.fused_add_rmsnorm" not in ops
