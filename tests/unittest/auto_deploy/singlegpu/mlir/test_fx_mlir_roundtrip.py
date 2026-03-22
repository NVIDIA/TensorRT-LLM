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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


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
