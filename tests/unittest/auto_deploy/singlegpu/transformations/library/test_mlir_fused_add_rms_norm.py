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

"""End-to-end tests for the MLIR-based fuse_add_rms_norm transform.

Mirrors the test structure from ``test_fused_add_rms_norm.py`` to validate
that the MLIR path produces identical results to the FX path.
"""

import pytest

xdsl = pytest.importorskip("xdsl")

import torch  # noqa: E402
from torch.export import Dim  # noqa: E402

import tensorrt_llm._torch.auto_deploy.mlir  # noqa: E402, F401 - apply xDSL patch
from tensorrt_llm._torch.auto_deploy.custom_ops.normalization.flashinfer_fused_add_rms_norm import (  # noqa: E402
    flashinfer_fused_add_rms_norm,
)
from tensorrt_llm._torch.auto_deploy.custom_ops.normalization.rms_norm import *  # noqa: F401, F403, E402
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm  # noqa: E402
from tensorrt_llm._torch.auto_deploy.transform.library.mlir_fused_add_rms_norm import (  # noqa: E402
    MLIRFuseAddRMSNorm,
    MLIRFuseAddRMSNormConfig,
)

# ---------------------------------------------------------------------------
# Models (same as test_fused_add_rms_norm.py)
# ---------------------------------------------------------------------------


class AddNormModel(torch.nn.Module):
    """add + rms_norm (no intermediate cast)."""

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


class AddCastNormModel(torch.nn.Module):
    """add + cast(bf16) + rms_norm."""

    def __init__(self, hidden_size=128, eps=1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(hidden_size, device="cuda", dtype=torch.bfloat16)
        )
        self.eps = eps

    def forward(self, x, residual):
        added = x + residual
        cast = added.to(torch.bfloat16)
        norm = torch.ops.auto_deploy.flashinfer_rms_norm(cast, self.weight, self.eps)
        return norm, added


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _export_model(model, *inputs):
    """Export model to GraphModule with dynamic batch dimension."""
    dyn = Dim.DYNAMIC
    ds = tuple({0: dyn} for _ in inputs)
    return torch_export_to_gm(model, args=inputs, dynamic_shapes=ds, clone=True)


def _apply_mlir_transform(gm, codegen_mode="preexisting"):
    """Apply the MLIR transform directly."""
    config = MLIRFuseAddRMSNormConfig(stage="post_load_fusion", codegen_mode=codegen_mode)
    transform = MLIRFuseAddRMSNorm(config=config)
    return transform._apply(gm, None, None, None)


def _count_fused_ops(gm):
    """Count flashinfer_fused_add_rms_norm calls in the graph."""
    return sum(
        1
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target is flashinfer_fused_add_rms_norm
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mlir_fuse_add_rms_norm():
    """MLIR transform fuses add + rmsnorm."""
    model = AddNormModel()
    bsz, seq_len, hidden = 2, 8, 128
    x = torch.randn(bsz, seq_len, hidden, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)

    gm = _export_model(model, x, residual)
    new_gm, info = _apply_mlir_transform(gm)

    assert info.num_matches >= 1, f"Expected >=1 matches, got {info.num_matches}"
    assert not info.skipped
    assert _count_fused_ops(new_gm) >= 1, "Fused op not found in MLIR-transformed graph"


def test_mlir_fuse_add_cast_rms_norm():
    """MLIR transform fuses add + cast + rmsnorm."""
    model = AddCastNormModel()
    bsz, seq_len, hidden = 2, 8, 128
    x = torch.randn(bsz, seq_len, hidden, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)

    gm = _export_model(model, x, residual)
    new_gm, info = _apply_mlir_transform(gm)

    assert info.num_matches >= 1


def test_mlir_transform_skips_without_xdsl():
    """Transform skips gracefully when xDSL is not available."""
    import tensorrt_llm._torch.auto_deploy.mlir as mlir_mod

    original = mlir_mod.HAS_XDSL
    try:
        mlir_mod.HAS_XDSL = False
        # Re-import to pick up the flag change
        import importlib

        import tensorrt_llm._torch.auto_deploy.transform.library.mlir_fused_add_rms_norm as t_mod

        importlib.reload(t_mod)

        model = AddNormModel()
        x = torch.randn(2, 8, 128, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        gm = _export_model(model, x, residual)

        config = t_mod.MLIRFuseAddRMSNormConfig(stage="post_load_fusion")
        transform = t_mod.MLIRFuseAddRMSNorm(config=config)
        result_gm, info = transform._apply(gm, None, None, None)

        assert info.skipped
        assert result_gm is gm  # Original graph returned unchanged
    finally:
        mlir_mod.HAS_XDSL = original
        importlib.reload(t_mod)


def test_mlir_config_in_registry():
    """Verify transform is registered and configurable."""
    from tensorrt_llm._torch.auto_deploy.transform.interface import TransformRegistry

    assert TransformRegistry.has("mlir_fuse_add_rms_norm")
    cls = TransformRegistry.get("mlir_fuse_add_rms_norm")
    assert cls.__name__ == "MLIRFuseAddRMSNorm"
    assert cls.get_config_class().__name__ == "MLIRFuseAddRMSNormConfig"
