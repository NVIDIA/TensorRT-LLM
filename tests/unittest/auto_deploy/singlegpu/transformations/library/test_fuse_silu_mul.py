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

import operator

import pytest
import torch
from torch.export import Dim

from tensorrt_llm._torch.auto_deploy.custom_ops.linear.silu_mul import HAS_FUSED_SILU_AND_MUL
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.interface import SharedConfig, TransformRegistry
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

_requires_fused_silu_mul = pytest.mark.skipif(
    not HAS_FUSED_SILU_AND_MUL, reason="requires flashinfer for fused silu_and_mul kernel"
)

_HALF_SIZE = 256


def _count_ops(gm, op):
    return sum(1 for n in gm.graph.nodes if is_op(n, op))


def _run_fuse_silu_mul(gm, enabled=True):
    """Run fuse_silu_mul transform directly."""
    shared_config = SharedConfig(local_rank=0, world_size=1)
    config_cls = TransformRegistry.get_config_class("fuse_silu_mul")
    config = config_cls(stage="post_load_fusion", enabled=enabled)
    transform = TransformRegistry.get("fuse_silu_mul")(config)
    gm, info = transform._apply(gm, cm=None, factory=None, shared_config=shared_config)
    return gm, info


def _build_narrow_silu_mul_graph(num_layers=1):
    """Build a graph with narrow+silu+mul pattern (Variant 1).

    The input placeholder represents the fused GEMM output (gate+up concatenated).
    """
    fused_size = _HALF_SIZE * 2
    graph = torch.fx.Graph()
    x = graph.placeholder("x")

    cur = x
    for _ in range(num_layers):
        gate = graph.call_function(torch.narrow, args=(cur, -1, 0, _HALF_SIZE))
        up = graph.call_function(torch.narrow, args=(cur, -1, _HALF_SIZE, _HALF_SIZE))
        silu_out = graph.call_function(torch.ops.aten.silu.default, args=(gate,))
        cur = graph.call_function(torch.ops.aten.mul.Tensor, args=(silu_out, up))

    graph.output(cur)

    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    # Set val metadata on placeholder (required by pattern matcher's shape prop)
    for n in gm.graph.nodes:
        if n.op == "placeholder":
            n.meta["val"] = torch.empty(2, fused_size, dtype=torch.float16, device="meta")

    return gm, fused_size


def _build_getitem_silu_mul_graph(num_layers=1):
    """Build a graph with getitem+silu+mul pattern (Variant 2).

    The input placeholder represents the fused GEMM output. A closure splits it
    into gate/up halves via operator.getitem (as done by quantized GEMM fusion).
    """
    fused_size = _HALF_SIZE * 2
    graph = torch.fx.Graph()
    x = graph.placeholder("x")

    cur = x
    for _ in range(num_layers):
        sizes = [_HALF_SIZE, _HALF_SIZE]

        def _split_fn(tensor, _sizes=sizes):
            return tuple(t.contiguous() for t in torch.split(tensor, _sizes, dim=-1))

        split = graph.call_function(_split_fn, args=(cur,))
        gate = graph.call_function(operator.getitem, args=(split, 0))
        up = graph.call_function(operator.getitem, args=(split, 1))
        silu_out = graph.call_function(torch.ops.aten.silu.default, args=(gate,))
        cur = graph.call_function(torch.ops.aten.mul.Tensor, args=(silu_out, up))

    graph.output(cur)

    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    # Set val metadata on mul nodes (needed for meta propagation in the replacement)
    for n in gm.graph.nodes:
        if n.op == "call_function" and n.target == torch.ops.aten.mul.Tensor:
            n.meta["val"] = torch.empty(2, _HALF_SIZE, dtype=torch.float16, device="meta")

    return gm, fused_size


@_requires_fused_silu_mul
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fuse_silu_mul_basic():
    """Test Variant 1: narrow+silu+mul pattern is fused into silu_and_mul."""
    gm, fused_size = _build_narrow_silu_mul_graph(num_layers=1)
    x = torch.randn(2, fused_size, device="cuda", dtype=torch.float16)

    ref_output = gm(x)

    gm, info = _run_fuse_silu_mul(gm)

    assert info.num_matches == 1
    assert _count_ops(gm, torch.ops.auto_deploy.silu_and_mul.default) == 1
    assert _count_ops(gm, torch.ops.aten.silu.default) == 0
    assert _count_ops(gm, torch.ops.aten.mul.Tensor) == 0

    fused_output = gm(x)
    torch.testing.assert_close(fused_output, ref_output, atol=1e-2, rtol=1e-2)


@_requires_fused_silu_mul
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fuse_silu_mul_multi_layer():
    """Test Variant 1 across multiple layers — each layer halves the size, so only first matches."""
    gm, fused_size = _build_narrow_silu_mul_graph(num_layers=1)
    x = torch.randn(2, fused_size, device="cuda", dtype=torch.float16)

    ref_output = gm(x)

    gm, info = _run_fuse_silu_mul(gm)

    assert info.num_matches == 1
    assert _count_ops(gm, torch.ops.auto_deploy.silu_and_mul.default) == 1
    assert _count_ops(gm, torch.ops.aten.silu.default) == 0
    assert _count_ops(gm, torch.ops.aten.mul.Tensor) == 0

    fused_output = gm(x)
    torch.testing.assert_close(fused_output, ref_output, atol=1e-2, rtol=1e-2)


@_requires_fused_silu_mul
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fuse_silu_mul_disabled():
    """Test that fusion is skipped when disabled."""
    gm, _ = _build_narrow_silu_mul_graph(num_layers=1)

    gm, info = _run_fuse_silu_mul(gm, enabled=False)

    assert info.skipped
    assert _count_ops(gm, torch.ops.auto_deploy.silu_and_mul.default) == 0
    assert _count_ops(gm, torch.ops.aten.silu.default) >= 1


@_requires_fused_silu_mul
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fuse_silu_mul_getitem_variant():
    """Test Variant 2: getitem+silu+mul pattern is fused into silu_and_mul."""
    gm, fused_size = _build_getitem_silu_mul_graph(num_layers=1)
    x = torch.randn(2, fused_size, device="cuda", dtype=torch.float16)

    ref_output = gm(x)

    gm, info = _run_fuse_silu_mul(gm)

    assert info.num_matches == 1
    assert _count_ops(gm, torch.ops.auto_deploy.silu_and_mul.default) == 1
    assert _count_ops(gm, torch.ops.aten.silu.default) == 0
    assert _count_ops(gm, torch.ops.aten.mul.Tensor) == 0

    fused_output = gm(x)
    torch.testing.assert_close(fused_output, ref_output, atol=1e-2, rtol=1e-2)


# ---- End-to-end test: export → GEMM fusion → silu_mul fusion -------------------------


class SwiGLUMLP(torch.nn.Module):
    """SwiGLU MLP with separate gate and up projections."""

    def __init__(self, hidden_size: int = 256, intermediate_size: int = 512):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


def _run_transform(gm, name, **config_kwargs):
    """Run a single transform, return (gm, info)."""
    shared_config = SharedConfig(local_rank=0, world_size=1)
    config_cls = TransformRegistry.get_config_class(name)
    config = config_cls(**config_kwargs)
    transform = TransformRegistry.get(name)(config)
    return transform._apply(gm, cm=None, factory=None, shared_config=shared_config)


@_requires_fused_silu_mul
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fuse_silu_mul_after_gemm_fusion():
    """End-to-end: export a SwiGLU model, run GEMM fusion, then fuse silu+mul."""
    model = SwiGLUMLP().to(device="cuda", dtype=torch.float16)
    x = torch.randn(2, 256, device="cuda", dtype=torch.float16)
    gm = torch_export_to_gm(model, (x,), dynamic_shapes={"x": {0: Dim("batch", min=1, max=16)}})

    ref_output = model(x)

    # Run GEMM fusion to merge gate+up projections
    gm, gemm_info = _run_transform(
        gm, "fuse_gemms_mixed_children", stage="post_load_fusion", enabled=True
    )
    assert gemm_info.num_matches > 0, "fuse_gemms_mixed_children found no linear ops"

    # Run silu+mul fusion
    gm, info = _run_fuse_silu_mul(gm)

    assert info.num_matches >= 1
    assert _count_ops(gm, torch.ops.auto_deploy.silu_and_mul.default) >= 1
    assert _count_ops(gm, torch.ops.aten.silu.default) == 0
    assert _count_ops(gm, torch.ops.aten.mul.Tensor) == 0

    fused_output = gm(x)
    torch.testing.assert_close(fused_output, ref_output, atol=1e-2, rtol=1e-2)
