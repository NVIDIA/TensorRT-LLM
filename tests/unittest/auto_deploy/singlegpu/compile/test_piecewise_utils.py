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
    _PERSISTENT_BUFFER_OPS,
    _STREAM_SWITCH_FUNCTION_NAMES,
    _get_all_dynamic_op_names,
    _submod_has_stream_switch,
    is_dynamic_cached_op,
    needs_out_buffer,
    split_graph_at_dynamic_ops,
    submod_has_cuda_ops,
)
from tensorrt_llm._torch.auto_deploy.utils.multi_stream_utils import (
    begin_aux_stream_passthrough,
    end_aux_stream_passthrough,
    wait_aux_stream_passthrough,
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


def _build_graphmodule_with_stream_switch_ops(dynamic_op_names=None):
    """Build a GraphModule with a dynamic op followed by a static region containing stream-switch ops.

    The graph simulates a piecewise-split scenario where attention (dynamic) is
    followed by a MoE layer with multi-stream passthrough functions:

        x -> relu -> [dyn_op_0] -> relu -> begin_aux -> relu -> end_aux -> relu -> output

    The region after the dynamic op contains begin_aux and end_aux, which should
    cause that static partition to be reclassified as dynamic.
    """
    if dynamic_op_names is None:
        dynamic_op_names = []

    graph = Graph()
    x = graph.placeholder("x")

    relu_node = graph.call_function(torch.relu, args=(x,))
    prev = relu_node

    # Insert dynamic ops (e.g., attention)
    for idx, dyn_name in enumerate(dynamic_op_names):
        fake_target = _FakeOpOverload(dyn_name)
        dyn_node = graph.create_node(
            "call_function", fake_target, args=(prev,), name=f"dyn_op_{idx}"
        )
        relu_after = graph.call_function(torch.relu, args=(dyn_node,))
        prev = relu_after

    # Static region with stream-switch ops (simulates multi_stream_moe transform output)
    begin_node = graph.call_function(begin_aux_stream_passthrough, args=(prev,))
    shared_expert = graph.call_function(torch.relu, args=(begin_node,))
    end_node = graph.call_function(end_aux_stream_passthrough, args=(shared_expert,))
    post = graph.call_function(torch.relu, args=(end_node,))

    graph.output(post)

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
        target = _FakeOpOverload("auto_deploy::gather_tokens")
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

    def test_stream_switch_partition_reclassified_as_dynamic(self):
        """Static partitions containing stream-switch ops should be reclassified as dynamic."""
        gm = _build_graphmodule_with_stream_switch_ops(
            dynamic_op_names=["auto_deploy::flashinfer_attention_mha_with_cache"],
        )
        info = split_graph_at_dynamic_ops(gm)

        # The split should have: static (relu) | dynamic (attn) | dynamic (reclassified:
        # relu + begin_aux + relu + end_aux + relu) | — but the exact layout depends
        # on the partition numbering.  Just check that the stream-switch partition ended
        # up in the dynamic list.
        assert len(info.dynamic_submod_indices) >= 2, (
            f"Expected at least 2 dynamic partitions (1 attention + 1 reclassified), "
            f"got {len(info.dynamic_submod_indices)}"
        )

        # Verify the reclassified partition contains a stream-switch op
        found_reclassified = False
        for idx in info.dynamic_submod_indices:
            submod = getattr(info.split_gm, f"submod_{idx}")
            if isinstance(submod, GraphModule) and _submod_has_stream_switch(submod):
                found_reclassified = True
                break
        assert found_reclassified, (
            "No reclassified stream-switch partition found in dynamic indices"
        )

    def test_no_stream_switch_ops_no_reclassification(self):
        """Partitions without stream-switch ops should remain static."""
        gm = _build_graphmodule_with_ops(
            dynamic_op_names=["auto_deploy::flashinfer_attention_mha_with_cache"]
        )
        info = split_graph_at_dynamic_ops(gm)

        # Only 1 dynamic partition (the attention op itself)
        assert len(info.dynamic_submod_indices) == 1


# ============================================================================
# Tests for stream-switch detection and needs_out_buffer
# ============================================================================


class TestStreamSwitchDetection:
    """Tests for _submod_has_stream_switch and needs_out_buffer with reclassified partitions."""

    def test_stream_switch_not_individually_dynamic(self):
        """Stream-switch functions are NOT individually dynamic ops.

        They are detected at the partition level via _submod_has_stream_switch,
        and the *entire containing partition* is reclassified as dynamic by
        split_graph_at_dynamic_ops.  This test documents the design: individual
        passthrough functions do not match is_dynamic_cached_op.
        """
        for func in (
            begin_aux_stream_passthrough,
            end_aux_stream_passthrough,
            wait_aux_stream_passthrough,
        ):
            node = _make_mock_node("call_function", target=func)
            assert is_dynamic_cached_op(node) is False, (
                f"{func.__name__} should NOT be individually dynamic — "
                "the partition-level reclassification handles it"
            )

    def test_submod_has_stream_switch_positive(self):
        """_submod_has_stream_switch returns True for submodules with passthrough ops."""
        graph = Graph()
        x = graph.placeholder("x")
        # Insert a begin_aux_stream_passthrough call
        node = graph.call_function(begin_aux_stream_passthrough, args=(x,))
        graph.output(node)
        gm = GraphModule(nn.Module(), graph)
        assert _submod_has_stream_switch(gm) is True

    def test_submod_has_stream_switch_negative(self):
        """_submod_has_stream_switch returns False for submodules without passthrough ops."""
        graph = Graph()
        x = graph.placeholder("x")
        node = graph.call_function(torch.relu, args=(x,))
        graph.output(node)
        gm = GraphModule(nn.Module(), graph)
        assert _submod_has_stream_switch(gm) is False

    def test_needs_out_buffer_false_for_stream_switch_partition(self):
        """Reclassified stream-switch partitions should not need out= buffers."""
        graph = Graph()
        x = graph.placeholder("x")
        begin = graph.call_function(begin_aux_stream_passthrough, args=(x,))
        relu = graph.call_function(torch.relu, args=(begin,))
        end = graph.call_function(end_aux_stream_passthrough, args=(relu,))
        graph.output(end)
        gm = GraphModule(nn.Module(), graph)
        assert needs_out_buffer(gm) is False

    def test_stream_switch_function_names_complete(self):
        """All exported passthrough functions should be in the detection set."""
        for func in (
            begin_aux_stream_passthrough,
            end_aux_stream_passthrough,
            wait_aux_stream_passthrough,
        ):
            assert func.__name__ in _STREAM_SWITCH_FUNCTION_NAMES, (
                f"{func.__name__} missing from _STREAM_SWITCH_FUNCTION_NAMES"
            )


# ============================================================================
# Tests for stream-switch reclassification + no preceding static runner
# ============================================================================


def _build_graphmodule_stream_switch_before_dynamic():
    """Build a GraphModule simulating multi_stream_mla_attn + trtllm attention.

    Layout:
        record_event → relu → [persistent_buf_op] → view → [attention_op] → relu → output

    After splitting:
        submod_0 (static, has record_event → reclassified as dynamic)
        submod_1 (dynamic, persistent buffer op — no out= needed)
        submod_2 (static, trivial view only — no CUDA ops, skipped)
        submod_3 (dynamic, attention — needs out=)
        submod_4 (static, relu — has CUDA ops → gets runner)

    This simulates the GLM-4.7-Flash + trtllm + multi_stream_mla_attn scenario.
    """
    from tensorrt_llm._torch.auto_deploy.utils.multi_stream_utils import record_event_passthrough

    graph = Graph()
    x = graph.placeholder("x")

    # Partition 0: static with stream switch → will be reclassified as dynamic
    rec = graph.call_function(record_event_passthrough, args=(x,))
    relu0 = graph.call_function(torch.relu, args=(rec,))

    # Partition 1: persistent buffer dynamic op
    persistent_target = _FakeOpOverload(_PERSISTENT_BUFFER_OPS[0])
    persistent_node = graph.create_node(
        "call_function", persistent_target, args=(relu0,), name="persistent_buf"
    )

    # Partition 2: trivial static (only a view, no CUDA ops)
    view_node = graph.call_method("view", args=(persistent_node, -1))

    # Partition 3: attention dynamic op (needs out= buffer)
    attn_target = _FakeOpOverload(_CACHED_ATTENTION_OPS[0])
    attn_node = graph.create_node("call_function", attn_target, args=(view_node,), name="attn_op")

    # Partition 4: static with CUDA ops
    relu_final = graph.call_function(torch.relu, args=(attn_node,))
    graph.output(relu_final)

    root = nn.Module()
    return GraphModule(root, graph)


class TestStreamSwitchBeforeDynamic:
    """Tests for stream-switch reclassification with no preceding static runner.

    Covers the scenario where stream-switch ops reclassify the first
    partition and a dynamic attention op has no preceding static runner.
    """

    def test_first_partition_reclassified_leaves_attention_without_preceding_runner(self):
        """Verify the problematic partition layout.

        First partition reclassified, trivial static between metadata and
        attention, attention needs out=.
        """
        gm = _build_graphmodule_stream_switch_before_dynamic()
        info = split_graph_at_dynamic_ops(gm)

        # The stream-switch partition should be reclassified as dynamic
        reclassified_found = False
        for idx in info.dynamic_submod_indices:
            submod = getattr(info.split_gm, f"submod_{idx}")
            if isinstance(submod, GraphModule) and _submod_has_stream_switch(submod):
                reclassified_found = True
                break
        assert reclassified_found, "Stream-switch partition should be reclassified"

        # Find the attention partition (should need out= buffer)
        attn_found = False
        for idx in info.dynamic_submod_indices:
            submod = getattr(info.split_gm, f"submod_{idx}")
            if isinstance(submod, GraphModule) and needs_out_buffer(submod):
                attn_found = True
                # Verify no preceding static partition has CUDA ops
                preceding_static_has_runner = False
                for s_idx in info.static_submod_indices:
                    if s_idx < idx:
                        s_submod = getattr(info.split_gm, f"submod_{s_idx}")
                        if submod_has_cuda_ops(s_submod):
                            preceding_static_has_runner = True
                if not preceding_static_has_runner:
                    break  # Found the problematic case
        assert attn_found, "Should have an attention partition needing out= buffer"
