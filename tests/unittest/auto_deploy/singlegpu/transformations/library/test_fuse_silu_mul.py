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

import torch

import tensorrt_llm._torch.auto_deploy.custom_ops.linear.silu_mul  # noqa: F401
from tensorrt_llm._torch.auto_deploy.transform.interface import SharedConfig, TransformRegistry
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

_HALF = 256


def _run_fuse_silu_mul(gm, enabled=True, backend="flashinfer"):
    """Apply just the fuse_silu_mul transform (no other passes)."""
    shared_config = SharedConfig(local_rank=0, world_size=1)
    config_cls = TransformRegistry.get_config_class("fuse_silu_mul")
    config = config_cls(stage="post_load_fusion", enabled=enabled, backend=backend)
    transform = TransformRegistry.get("fuse_silu_mul")(config)
    return transform._apply(gm, cm=None, factory=None, shared_config=shared_config)


def _build_narrow_silu_mul_graph():
    """Build a tiny FX graph with the narrow+silu+mul pattern (Variant 1)."""
    fused_size = _HALF * 2
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    gate = graph.call_function(torch.narrow, args=(x, -1, 0, _HALF))
    up = graph.call_function(torch.narrow, args=(x, -1, _HALF, _HALF))
    silu = graph.call_function(torch.ops.aten.silu.default, args=(gate,))
    mul = graph.call_function(torch.ops.aten.mul.Tensor, args=(silu, up))
    graph.output(mul)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    for n in gm.graph.nodes:
        if n.op == "placeholder":
            n.meta["val"] = torch.empty(2, fused_size, dtype=torch.float16, device="meta")
        if n.op == "call_function" and n.target == torch.ops.aten.mul.Tensor:
            n.meta["val"] = torch.empty(2, _HALF, dtype=torch.float16, device="meta")
    return gm, fused_size


def _build_getitem_silu_mul_graph():
    """Build a tiny FX graph with the split+getitem+silu+mul pattern (Variant 2)."""
    fused_size = _HALF * 2
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    sizes = [_HALF, _HALF]

    def _split_fn(tensor, _sizes=sizes):
        return tuple(t.contiguous() for t in torch.split(tensor, _sizes, dim=-1))

    split = graph.call_function(_split_fn, args=(x,))
    gate = graph.call_function(operator.getitem, args=(split, 0))
    up = graph.call_function(operator.getitem, args=(split, 1))
    silu = graph.call_function(torch.ops.aten.silu.default, args=(gate,))
    mul = graph.call_function(torch.ops.aten.mul.Tensor, args=(silu, up))
    graph.output(mul)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    for n in gm.graph.nodes:
        # Placeholder is the (..., 2*D) parent; the matcher needs its meta
        # to verify ``parent.shape[-1] == 2 * gate_size`` before fusing.
        if n.op == "placeholder":
            n.meta["val"] = torch.empty(2, fused_size, dtype=torch.float16, device="meta")
        # Set a val on each getitem so _get_narrow_info can read sizes from meta
        if n.op == "call_function" and n.target == operator.getitem:
            n.meta["val"] = torch.empty(2, _HALF, dtype=torch.float16, device="meta")
        if n.op == "call_function" and n.target == torch.ops.aten.mul.Tensor:
            n.meta["val"] = torch.empty(2, _HALF, dtype=torch.float16, device="meta")
    return gm, fused_size


def _count_ops(gm, op):
    return sum(1 for n in gm.graph.nodes if is_op(n, op))


def test_fuse_silu_mul_narrow_variant():
    """Variant 1: narrow + silu + mul → flashinfer_silu_and_mul."""
    gm, _ = _build_narrow_silu_mul_graph()
    gm, info = _run_fuse_silu_mul(gm)
    assert info.num_matches == 1
    assert _count_ops(gm, torch.ops.auto_deploy.flashinfer_silu_and_mul.default) == 1
    assert _count_ops(gm, torch.ops.aten.silu.default) == 0
    assert _count_ops(gm, torch.ops.aten.mul.Tensor) == 0


def test_fuse_silu_mul_getitem_variant():
    """Variant 2: split+getitem + silu + mul → flashinfer_silu_and_mul."""
    gm, _ = _build_getitem_silu_mul_graph()
    gm, info = _run_fuse_silu_mul(gm)
    assert info.num_matches == 1
    assert _count_ops(gm, torch.ops.auto_deploy.flashinfer_silu_and_mul.default) == 1
    assert _count_ops(gm, torch.ops.aten.silu.default) == 0
    assert _count_ops(gm, torch.ops.aten.mul.Tensor) == 0


def test_fuse_silu_mul_skipped_when_disabled():
    gm, _ = _build_narrow_silu_mul_graph()
    gm, info = _run_fuse_silu_mul(gm, enabled=False)
    assert info.skipped
    assert _count_ops(gm, torch.ops.auto_deploy.flashinfer_silu_and_mul.default) == 0
    assert _count_ops(gm, torch.ops.aten.silu.default) == 1
