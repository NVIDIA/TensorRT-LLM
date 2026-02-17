# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for piecewise_utils: is_dynamic_cached_op and split_graph_at_dynamic_ops."""

from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.fx import Graph, GraphModule

from tensorrt_llm._torch.auto_deploy.compile.piecewise_utils import (
    _CACHED_ATTENTION_OPS,
    _CACHED_CONV_OPS,
    _CACHED_DELTA_OPS,
    _CACHED_SSM_OPS,
    _LOGITS_GATHER_OPS,
    _METADATA_PREP_OPS,
    _get_all_dynamic_op_names,
    is_dynamic_cached_op,
    split_graph_at_dynamic_ops,
)

# ============================================================================
# Helpers
# ============================================================================


def _make_mock_node(op: str, target=None):
    """Create a lightweight mock FX Node for testing is_dynamic_cached_op."""
    node = SimpleNamespace(op=op, target=target)
    return node


class _FakeOpOverload:
    """Mimics torch._ops.OpOverload with a .name() method.

    Must be callable with __name__/__module__/__qualname__ because torch.fx
    validates call_function targets and generates Python code referencing them.
    """

    def __init__(self, qualified_name: str):
        self._name = qualified_name
        # Attributes required by torch.fx for codegen
        short = qualified_name.split("::")[-1]
        self.__name__ = short
        self.__qualname__ = qualified_name
        self.__module__ = "test_piecewise_utils"

    def name(self):
        return self._name

    def __call__(self, *args, **kwargs):
        # Identity pass-through for graph execution
        return args[0] if args else None


def _build_graphmodule_with_ops(dynamic_op_names=None):
    """Build a simple FX GraphModule with relu ops interspersed with fake dynamic ops.

    The graph looks like: x -> relu -> [dyn_op_0] -> relu -> [dyn_op_1] -> ... -> output
    Dynamic ops are simulated by inserting call_function nodes whose target is a
    _FakeOpOverload with a name matching one of the dynamic op registries.
    """
    if dynamic_op_names is None:
        dynamic_op_names = []

    # Build graph manually
    graph = Graph()
    x = graph.placeholder("x")

    # First static op: relu
    relu_node = graph.call_function(torch.relu, args=(x,))
    prev = relu_node

    for idx, dyn_name in enumerate(dynamic_op_names):
        # Insert a fake dynamic op. We use graph.create_node directly because
        # graph.call_function tries _target_to_str which asserts isinstance(target, str).
        fake_target = _FakeOpOverload(dyn_name)
        dyn_node = graph.create_node(
            "call_function", fake_target, args=(prev,), name=f"dyn_op_{idx}"
        )
        # Follow with another static op
        relu_after = graph.call_function(torch.relu, args=(dyn_node,))
        prev = relu_after

    graph.output(prev)

    # We need a root module -- a simple nn.Module suffices
    root = nn.Module()
    gm = GraphModule(root, graph)
    return gm


# ============================================================================
# Tests for is_dynamic_cached_op
# ============================================================================


class TestIsDynamicCachedOp:
    """Tests for is_dynamic_cached_op."""

    def test_known_attention_op_returns_true(self):
        target = _FakeOpOverload("auto_deploy::flashinfer_attention_mha_with_cache")
        node = _make_mock_node("call_function", target=target)
        assert is_dynamic_cached_op(node) is True

    def test_known_ssm_op_returns_true(self):
        target = _FakeOpOverload("auto_deploy::triton_cached_ssm")
        node = _make_mock_node("call_function", target=target)
        assert is_dynamic_cached_op(node) is True

    def test_known_conv_op_returns_true(self):
        target = _FakeOpOverload("auto_deploy::triton_cached_causal_conv1d")
        node = _make_mock_node("call_function", target=target)
        assert is_dynamic_cached_op(node) is True

    def test_known_delta_op_returns_true(self):
        target = _FakeOpOverload("auto_deploy::fla_cached_delta_rule")
        node = _make_mock_node("call_function", target=target)
        assert is_dynamic_cached_op(node) is True

    def test_known_metadata_prep_op_returns_true(self):
        target = _FakeOpOverload("auto_deploy::flashinfer_attention_prepare_metadata")
        node = _make_mock_node("call_function", target=target)
        assert is_dynamic_cached_op(node) is True

    def test_known_logits_gather_op_returns_true(self):
        target = _FakeOpOverload("auto_deploy::gather_logits_before_lm_head")
        node = _make_mock_node("call_function", target=target)
        assert is_dynamic_cached_op(node) is True

    def test_static_op_returns_false(self):
        # torch.relu is not a dynamic op
        node = _make_mock_node("call_function", target=torch.relu)
        assert is_dynamic_cached_op(node) is False

    def test_non_call_function_returns_false(self):
        target = _FakeOpOverload("auto_deploy::flashinfer_attention_mha_with_cache")
        # Even with a dynamic target, non-call_function ops return False
        for op_type in ("placeholder", "call_method", "call_module", "output", "get_attr"):
            node = _make_mock_node(op_type, target=target)
            assert is_dynamic_cached_op(node) is False, f"Should be False for op={op_type}"

    def test_op_with_default_suffix_still_matches(self):
        """Dynamic op name with .default suffix should still match (substring check)."""
        target = _FakeOpOverload("auto_deploy::triton_cached_ssm.default")
        node = _make_mock_node("call_function", target=target)
        assert is_dynamic_cached_op(node) is True

    def test_all_registry_entries_recognized(self):
        """Every op in every registry list should be recognized as dynamic."""
        all_ops = (
            _CACHED_ATTENTION_OPS
            + _CACHED_SSM_OPS
            + _CACHED_CONV_OPS
            + _CACHED_DELTA_OPS
            + _METADATA_PREP_OPS
            + _LOGITS_GATHER_OPS
        )
        for op_name in all_ops:
            target = _FakeOpOverload(op_name)
            node = _make_mock_node("call_function", target=target)
            assert is_dynamic_cached_op(node) is True, f"{op_name} should be recognized as dynamic"

    def test_get_all_dynamic_op_names_returns_full_set(self):
        all_names = _get_all_dynamic_op_names()
        assert isinstance(all_names, set)
        # Should include all registries
        for op in _CACHED_ATTENTION_OPS:
            assert op in all_names
        for op in _CACHED_SSM_OPS:
            assert op in all_names
        for op in _LOGITS_GATHER_OPS:
            assert op in all_names


# ============================================================================
# Tests for split_graph_at_dynamic_ops
# ============================================================================


class TestSplitGraphAtDynamicOps:
    """Tests for split_graph_at_dynamic_ops."""

    def test_no_dynamic_ops_returns_original(self):
        """Graph with no dynamic ops should not be split."""
        gm = _build_graphmodule_with_ops(dynamic_op_names=[])
        info = split_graph_at_dynamic_ops(gm)

        assert info.num_submodules == 1
        assert info.dynamic_submod_indices == []
        assert info.static_submod_indices == [0]
        # split_gm is the original gm
        assert info.split_gm is gm

    def test_single_dynamic_op_produces_3_submodules(self):
        """One dynamic op should produce 3 partitions: static -> dynamic -> static."""
        gm = _build_graphmodule_with_ops(
            dynamic_op_names=["auto_deploy::flashinfer_attention_mha_with_cache"]
        )
        info = split_graph_at_dynamic_ops(gm)

        # Expected: submod_0 (static: relu), submod_1 (dynamic: attn), submod_2 (static: relu)
        assert info.num_submodules == 3
        assert len(info.dynamic_submod_indices) == 1
        assert len(info.static_submod_indices) == 2

    def test_two_dynamic_ops_produces_5_submodules(self):
        """Two dynamic ops → 5 partitions: S D S D S."""
        gm = _build_graphmodule_with_ops(
            dynamic_op_names=[
                "auto_deploy::flashinfer_attention_mha_with_cache",
                "auto_deploy::triton_cached_ssm",
            ]
        )
        info = split_graph_at_dynamic_ops(gm)

        assert info.num_submodules == 5
        assert len(info.dynamic_submod_indices) == 2
        assert len(info.static_submod_indices) == 3

    def test_dynamic_and_static_indices_are_disjoint(self):
        """Dynamic and static indices should not overlap and should cover all submodules."""
        gm = _build_graphmodule_with_ops(
            dynamic_op_names=[
                "auto_deploy::flashinfer_attention_mha_with_cache",
                "auto_deploy::triton_cached_ssm",
            ]
        )
        info = split_graph_at_dynamic_ops(gm)

        all_indices = set(info.dynamic_submod_indices) | set(info.static_submod_indices)
        assert len(all_indices) == info.num_submodules
        # No overlap
        assert len(set(info.dynamic_submod_indices) & set(info.static_submod_indices)) == 0

    def test_split_submodules_are_named_correctly(self):
        """Split submodules should be named submod_0, submod_1, etc."""
        gm = _build_graphmodule_with_ops(
            dynamic_op_names=["auto_deploy::triton_cached_causal_conv1d"]
        )
        info = split_graph_at_dynamic_ops(gm)

        for i in range(info.num_submodules):
            assert hasattr(info.split_gm, f"submod_{i}"), f"Missing submod_{i}"
