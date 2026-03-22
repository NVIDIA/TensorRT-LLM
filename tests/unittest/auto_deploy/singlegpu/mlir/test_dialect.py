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

"""Tests for the AutoDeploy MLIR dialect definition."""

import pytest

xdsl = pytest.importorskip("xdsl")

from xdsl.dialects.builtin import (  # noqa: E402
    BFloat16Type,
    Float64Type,
    FloatAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    TensorType,
)
from xdsl.ir import Block, Region  # noqa: E402

from tensorrt_llm._torch.auto_deploy.mlir.dialect import (  # noqa: E402
    AdAdd,
    AdFusedAddRMSNorm,
    AdGraphInput,
    AdGraphOutput,
    AdOpaque,
    AdRMSNorm,
    Float8E4M3FNType,
    Float8E5M2Type,
    register_ad_dialect,
    tensor_type_from_meta,
    torch_dtype_to_mlir,
)


def test_torch_dtype_to_mlir():
    """Verify torch dtype → MLIR element type mapping."""
    import torch

    assert isinstance(torch_dtype_to_mlir(torch.bfloat16), BFloat16Type)
    assert isinstance(torch_dtype_to_mlir(torch.float16), type(torch_dtype_to_mlir(torch.float16)))
    assert isinstance(torch_dtype_to_mlir(torch.float8_e4m3fn), Float8E4M3FNType)
    assert isinstance(torch_dtype_to_mlir(torch.float8_e5m2), Float8E5M2Type)
    assert isinstance(torch_dtype_to_mlir(torch.int32), IntegerType)
    assert isinstance(torch_dtype_to_mlir(torch.int64), IntegerType)
    assert isinstance(torch_dtype_to_mlir(torch.bool), IntegerType)
    with pytest.raises(ValueError, match="Unsupported"):
        torch_dtype_to_mlir(torch.complex64)


def test_tensor_type_from_meta():
    """Verify FakeTensor → TensorType conversion."""
    import torch

    t = torch.randn(2, 8, 128, dtype=torch.bfloat16, device="meta")
    mlir_type = tensor_type_from_meta(t)
    assert isinstance(mlir_type, TensorType)
    assert mlir_type.get_shape() == (2, 8, 128)
    assert isinstance(mlir_type.element_type, BFloat16Type)


def test_fp8_tensor_type():
    """Verify FP8 types can be used in TensorType."""
    t_e4m3 = TensorType(Float8E4M3FNType(), [2, 128])
    assert t_e4m3.get_shape() == (2, 128)
    assert isinstance(t_e4m3.element_type, Float8E4M3FNType)

    t_e5m2 = TensorType(Float8E5M2Type(), [4, 256])
    assert t_e5m2.get_shape() == (4, 256)
    assert isinstance(t_e5m2.element_type, Float8E5M2Type)


def test_fp8_round_trip_dtype():
    """Verify FP8 dtype round-trip: torch → MLIR → torch."""
    import torch

    from tensorrt_llm._torch.auto_deploy.mlir.dialect import mlir_to_torch_dtype

    for torch_dt in [torch.float8_e4m3fn, torch.float8_e5m2]:
        mlir_type = torch_dtype_to_mlir(torch_dt)
        back = mlir_to_torch_dtype(mlir_type)
        assert back == torch_dt, f"Round-trip failed for {torch_dt}: got {back}"


def test_ad_add_construction():
    """Verify ad.add op can be constructed with proper operands and result."""
    t = TensorType(BFloat16Type(), [2, 8, 128])
    block = Block()
    lhs = block.insert_arg(t, 0)
    rhs = block.insert_arg(t, 1)

    op = AdAdd.build(operands=[lhs, rhs], result_types=[t])
    block.add_op(op)

    assert op.name == "ad.add"
    assert op.lhs is lhs
    assert op.rhs is rhs
    assert op.output.type == t


def test_ad_rmsnorm_construction():
    """Verify ad.rmsnorm op with eps attribute."""
    t = TensorType(BFloat16Type(), [2, 8, 128])
    tw = TensorType(BFloat16Type(), [128])
    eps_attr = FloatAttr(1e-5, Float64Type())

    block = Block()
    inp = block.insert_arg(t, 0)
    weight = block.insert_arg(tw, 1)

    op = AdRMSNorm.build(
        operands=[inp, weight],
        attributes={"eps": eps_attr},
        result_types=[t],
    )
    block.add_op(op)

    assert op.name == "ad.rmsnorm"
    assert op.eps.value.data == pytest.approx(1e-5)


def test_ad_fused_add_rmsnorm_construction():
    """Verify ad.fused_add_rmsnorm op produces two results."""
    t = TensorType(BFloat16Type(), [2, 8, 128])
    tw = TensorType(BFloat16Type(), [128])
    eps_attr = FloatAttr(1e-5, Float64Type())

    block = Block()
    x = block.insert_arg(t, 0)
    res = block.insert_arg(t, 1)
    weight = block.insert_arg(tw, 2)

    op = AdFusedAddRMSNorm.build(
        operands=[x, res, weight],
        attributes={"eps": eps_attr},
        result_types=[t, t],
    )
    block.add_op(op)

    assert op.name == "ad.fused_add_rmsnorm"
    assert op.norm_result.type == t
    assert op.add_result.type == t


def test_ad_opaque_construction():
    """Verify ad.opaque catch-all op."""
    t = TensorType(BFloat16Type(), [2, 8, 128])
    block = Block()
    inp = block.insert_arg(t, 0)

    op = AdOpaque.build(
        operands=[[inp]],
        attributes={
            "op_name": StringAttr("aten.relu"),
            "node_key": StringAttr("relu_0"),
        },
        result_types=[[t]],
    )
    block.add_op(op)

    assert op.name == "ad.opaque"
    assert op.op_name.data == "aten.relu"
    assert op.node_key.data == "relu_0"


def test_ad_graph_input_output():
    """Verify graph boundary ops."""
    t = TensorType(BFloat16Type(), [2, 128])
    block = Block()

    inp_op = AdGraphInput.build(
        attributes={"input_name": StringAttr("x")},
        result_types=[t],
    )
    block.add_op(inp_op)

    out_op = AdGraphOutput.build(operands=[[inp_op.output]])
    block.add_op(out_op)

    assert inp_op.name == "ad.graph_input"
    assert out_op.name == "ad.graph_output"


def test_register_ad_dialect():
    """Verify dialect registration with xDSL context."""
    from xdsl.context import Context as MLContext

    ctx = MLContext()
    register_ad_dialect(ctx)
    # Should not raise — ops are now loaded
    assert ctx.get_optional_op("ad.add") is not None
    assert ctx.get_optional_op("ad.rmsnorm") is not None
    assert ctx.get_optional_op("ad.fused_add_rmsnorm") is not None


def test_build_complete_ir():
    """Build a small IR graph: input → add → rmsnorm → output."""
    t = TensorType(BFloat16Type(), [2, 8, 128])
    tw = TensorType(BFloat16Type(), [128])
    eps_attr = FloatAttr(1e-5, Float64Type())

    block = Block()
    x = block.insert_arg(t, 0)
    res = block.insert_arg(t, 1)
    w = block.insert_arg(tw, 2)

    add_op = AdAdd.build(operands=[x, res], result_types=[t])
    block.add_op(add_op)

    norm_op = AdRMSNorm.build(
        operands=[add_op.output, w],
        attributes={"eps": eps_attr},
        result_types=[t],
    )
    block.add_op(norm_op)

    out_op = AdGraphOutput.build(operands=[[norm_op.output, add_op.output]])
    block.add_op(out_op)

    mod = ModuleOp(Region([block]))
    ops = list(mod.body.block.ops)
    assert len(ops) == 3
    assert ops[0].name == "ad.add"
    assert ops[1].name == "ad.rmsnorm"
    assert ops[2].name == "ad.graph_output"
