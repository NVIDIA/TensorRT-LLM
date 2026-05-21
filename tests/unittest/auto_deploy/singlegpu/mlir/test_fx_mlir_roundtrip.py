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

import pytest

xdsl = pytest.importorskip("xdsl")

import torch  # noqa: E402
from torch.export import Dim  # noqa: E402

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
            "val": fake_val,
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
