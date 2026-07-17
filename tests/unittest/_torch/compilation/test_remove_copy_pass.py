# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from operator import getitem

import pytest
import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized_v2
from torch.fx import Graph

import tensorrt_llm._torch.compilation.remove_copy_pass as remove_copy_pass


def test_remove_copy_for_mutates_args_auto_functionalized_v2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = Graph()
    output = graph.placeholder("output")
    src = graph.placeholder("src")
    functionalized = graph.call_function(
        auto_functionalized_v2,
        args=(torch.ops.aten.copy_.default,),
        kwargs={
            "_all_bases": (output,),
            "_self_base_index": 0,
            "src": src,
        },
    )
    mutated_output = graph.call_function(getitem, args=(functionalized, 1))
    clone = graph.call_function(torch.ops.aten.clone.default, args=(mutated_output,))
    graph.output(clone)

    monkeypatch.setattr(
        remove_copy_pass,
        "inplace_info",
        lambda: {torch.ops.aten.copy_.default: {1: "self"}},
    )

    remove_copy_pass.remove_copy_for_mutates_args(graph)

    inplace_nodes = [node for node in graph.nodes if node.target == torch.ops.aten.copy_.default]
    assert len(inplace_nodes) == 1
    assert inplace_nodes[0].kwargs == {"src": src, "self": output}
    assert clone.args[0] is output
    assert all(node.target != auto_functionalized_v2 for node in graph.nodes)
    graph.lint()


def test_remove_copy_for_mutates_args_restores_optional_none() -> None:
    graph = Graph()
    hidden_states = graph.placeholder("hidden_states")
    output = graph.placeholder("output")
    inplace_func = torch.ops.trtllm.mla_custom_op_inplace.default
    functionalized = graph.call_function(
        auto_functionalized_v2,
        args=(inplace_func,),
        kwargs={
            "hidden_states": hidden_states,
            "position_ids": None,
            "layer_idx": "0",
            "latent_cache_gen": None,
            "enable_dsv4_epilogue_fusion": False,
            "_all_bases": (output,),
            "_output_base_index": 0,
            "_dsv4_output_base_index": None,
            "_dsv4_output_sf_base_index": None,
        },
    )
    mutated_output = graph.call_function(getitem, args=(functionalized, 1))
    clone = graph.call_function(torch.ops.aten.clone.default, args=(mutated_output,))
    graph.output(clone)

    remove_copy_pass.remove_copy_for_mutates_args(graph)

    inplace_nodes = [node for node in graph.nodes if node.target == inplace_func]
    assert len(inplace_nodes) == 1
    assert inplace_nodes[0].kwargs["output"] is output
    assert inplace_nodes[0].kwargs["dsv4_output"] is None
    assert inplace_nodes[0].kwargs["dsv4_output_sf"] is None
    assert clone.args[0] is output
    graph.lint()


def test_remove_copy_for_mutates_args_rejects_getitem_for_optional_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = Graph()
    hidden_states = graph.placeholder("hidden_states")
    output = graph.placeholder("output")
    inplace_func = torch.ops.trtllm.mla_custom_op_inplace.default
    functionalized = graph.call_function(
        auto_functionalized_v2,
        args=(inplace_func,),
        kwargs={
            "hidden_states": hidden_states,
            "position_ids": None,
            "layer_idx": "0",
            "latent_cache_gen": None,
            "enable_dsv4_epilogue_fusion": False,
            "_all_bases": (output,),
            "_output_base_index": 0,
            "_dsv4_output_base_index": None,
            "_dsv4_output_sf_base_index": None,
        },
    )
    optional_output = graph.call_function(getitem, args=(functionalized, 2))
    clone = graph.call_function(torch.ops.aten.clone.default, args=(optional_output,))
    graph.output(clone)

    monkeypatch.setattr(
        remove_copy_pass,
        "inplace_info",
        lambda: {inplace_func: {1: "output", 2: "dsv4_output"}},
    )

    with pytest.raises(
        AssertionError,
        match=(
            "getitem user for optional output 'dsv4_output' has no "
            "base tensor -- graph is malformed"
        ),
    ):
        remove_copy_pass.remove_copy_for_mutates_args(graph)
