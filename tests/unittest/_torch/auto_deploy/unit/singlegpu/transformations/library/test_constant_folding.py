# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for the generalized constant folding transform."""

import torch
from torch.export import Dim

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer


class MambaALogModel(torch.nn.Module):
    """Model with A_log pattern: A = -exp(A_log.float())."""

    def __init__(self, d_inner=64, d_state=16):
        super().__init__()
        self.A_log = torch.nn.Parameter(
            torch.randn(d_inner, d_state, device="cuda", dtype=torch.float32)
        )

    def forward(self, x):
        # This pattern should be folded: A = -exp(A_log.float())
        A = -torch.exp(self.A_log.float())
        return x * A.sum()


class ChainedConstantModel(torch.nn.Module):
    """Model with chained constant operations."""

    def __init__(self, hidden_size=128):
        super().__init__()
        self.scale = torch.nn.Parameter(
            torch.randn(hidden_size, device="cuda", dtype=torch.float32)
        )

    def forward(self, x):
        # Chained ops: sqrt(abs(scale)) + 1
        transformed = torch.sqrt(torch.abs(self.scale)) + 1.0
        return x * transformed


class MultipleConstantModel(torch.nn.Module):
    """Model with multiple independent constant computations."""

    def __init__(self, hidden_size=64):
        super().__init__()
        self.alpha = torch.nn.Parameter(
            torch.randn(hidden_size, device="cuda", dtype=torch.float32)
        )
        self.beta = torch.nn.Parameter(torch.randn(hidden_size, device="cuda", dtype=torch.float32))

    def forward(self, x, y):
        # Two independent constant computations
        a = torch.exp(self.alpha)
        b = torch.sigmoid(self.beta)
        return x * a + y * b


class NoConstantFoldModel(torch.nn.Module):
    """Model where no constant folding should occur (all ops depend on input)."""

    def __init__(self, hidden_size=64):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(hidden_size, device="cuda", dtype=torch.float32)
        )

    def forward(self, x):
        # All operations depend on dynamic input x
        return torch.exp(x) * self.weight


def _count_get_attr_nodes(gm):
    """Count the number of get_attr nodes in the graph."""
    return sum(1 for n in gm.graph.nodes if n.op == "get_attr")


def _count_folded_const_nodes(gm):
    """Count the number of folded constant nodes (starting with _folded_const)."""
    return sum(
        1
        for n in gm.graph.nodes
        if n.op == "get_attr" and str(n.target).startswith("_folded_const")
    )


def _has_op(gm, op_name):
    """Check if the graph contains a specific operation."""
    return any(n.op == "call_function" and op_name in str(n.target) for n in gm.graph.nodes)


def _run_constant_folding(model, args, dynamic_shapes=None):
    """Export model and run constant folding transform."""
    gm = torch_export_to_gm(model, args=args, dynamic_shapes=dynamic_shapes, clone=True)

    gm_transformed = InferenceOptimizer(
        None,
        {
            "constant_folding": {
                "stage": "post_load_fusion",
            },
        },
    )(None, gm)

    return gm, gm_transformed


def test_mamba_a_log_pattern():
    """Test folding of Mamba A_log pattern: -exp(A_log.float())."""
    model = MambaALogModel(d_inner=64, d_state=16).cuda()

    x = torch.randn(2, 64, 16, device="cuda", dtype=torch.float32)
    ds = {0: Dim("batch", max=8)}

    gm_orig, gm_transformed = _run_constant_folding(model, (x,), (ds,))

    # Check that exp and neg ops are gone (folded)
    assert not _has_op(gm_transformed, "exp"), "exp op should be folded"
    assert not _has_op(gm_transformed, "neg"), "neg op should be folded"

    # Check that a folded constant was created
    assert _count_folded_const_nodes(gm_transformed) >= 1, "Should have folded constant"

    # Validate correctness
    y_orig = model(x.clone())
    y_transformed = gm_transformed(x.clone())
    torch.testing.assert_close(y_orig, y_transformed, atol=1e-5, rtol=1e-5)


def test_chained_constants():
    """Test folding of chained constant operations."""
    model = ChainedConstantModel(hidden_size=128).cuda()

    x = torch.randn(4, 128, device="cuda", dtype=torch.float32)
    ds = {0: Dim("batch", max=16)}

    gm_orig, gm_transformed = _run_constant_folding(model, (x,), (ds,))

    # Check that sqrt and abs ops are gone
    assert not _has_op(gm_transformed, "sqrt"), "sqrt op should be folded"
    assert not _has_op(gm_transformed, "abs"), "abs op should be folded"

    # Validate correctness
    y_orig = model(x.clone())
    y_transformed = gm_transformed(x.clone())
    torch.testing.assert_close(y_orig, y_transformed, atol=1e-5, rtol=1e-5)


def test_multiple_constants():
    """Test folding of multiple independent constant computations."""
    model = MultipleConstantModel(hidden_size=64).cuda()

    x = torch.randn(2, 64, device="cuda", dtype=torch.float32)
    y = torch.randn(2, 64, device="cuda", dtype=torch.float32)
    ds_x = {0: Dim("batch", max=8)}
    ds_y = {0: Dim("batch", max=8)}

    gm_orig, gm_transformed = _run_constant_folding(model, (x, y), (ds_x, ds_y))

    # Check that exp and sigmoid ops are gone
    assert not _has_op(gm_transformed, "exp"), "exp op should be folded"
    assert not _has_op(gm_transformed, "sigmoid"), "sigmoid op should be folded"

    # Should have at least 2 folded constants (one for each branch)
    assert _count_folded_const_nodes(gm_transformed) >= 2, "Should have 2+ folded constants"

    # Validate correctness
    y_orig = model(x.clone(), y.clone())
    y_transformed = gm_transformed(x.clone(), y.clone())
    torch.testing.assert_close(y_orig, y_transformed, atol=1e-5, rtol=1e-5)


def test_no_folding_when_dynamic():
    """Test that no folding occurs when all ops depend on dynamic input."""
    model = NoConstantFoldModel(hidden_size=64).cuda()

    x = torch.randn(2, 64, device="cuda", dtype=torch.float32)
    ds = {0: Dim("batch", max=8)}

    gm_orig, gm_transformed = _run_constant_folding(model, (x,), (ds,))

    # exp should NOT be folded because it operates on dynamic input
    assert _has_op(gm_transformed, "exp"), "exp op should NOT be folded (dynamic input)"

    # No folded constants should be created
    assert _count_folded_const_nodes(gm_transformed) == 0, "No constants should be folded"

    # Validate correctness
    y_orig = model(x.clone())
    y_transformed = gm_transformed(x.clone())
    torch.testing.assert_close(y_orig, y_transformed, atol=1e-5, rtol=1e-5)


def test_dtype_preservation():
    """Test that dtype is preserved through constant folding."""

    class DtypeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.randn(32, device="cuda", dtype=torch.float16))

        def forward(self, x):
            # exp of float16 parameter
            scaled = torch.exp(self.param)
            return x * scaled

    model = DtypeModel().cuda()
    x = torch.randn(4, 32, device="cuda", dtype=torch.float16)
    ds = {0: Dim("batch", max=8)}

    _, gm_transformed = _run_constant_folding(model, (x,), (ds,))

    # Find the folded constant and check its dtype
    for name, buf in gm_transformed.named_buffers():
        if name.startswith("_folded_const"):
            assert buf.dtype == torch.float16, f"Dtype should be preserved, got {buf.dtype}"

    # Validate correctness
    y_orig = model(x.clone())
    y_transformed = gm_transformed(x.clone())
    torch.testing.assert_close(y_orig, y_transformed, atol=1e-2, rtol=1e-2)
