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

from collections.abc import Callable
from operator import getitem

import pytest
import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized_v2
from torch.fx import Graph

import tensorrt_llm._torch.compilation.remove_copy_pass as remove_copy_pass

# Registers torch.ops.trtllm.mla_custom_op_inplace, used below. The op is a
# module-import side effect that the eagerly loaded model zoo used to provide
# (models/__init__ -> modeling_deepseekv3 -> modules.mla); with the zoo lazy,
# every test must import the ops it uses itself.
import tensorrt_llm._torch.modules.mla  # noqa: F401
from tensorrt_llm._torch.modules.fused_ops.fused_qk_norm_rope_gate import (
    fused_sigmoid_mul_inplace,  # noqa: F401
)


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


def test_remove_copy_for_fused_sigmoid_mul_inplace() -> None:
    graph = Graph()
    attention_output = graph.placeholder("attention_output")
    gate = graph.placeholder("gate")
    inplace_func = torch.ops.trtllm.fused_sigmoid_mul_inplace.default
    functionalized = graph.call_function(
        auto_functionalized_v2,
        args=(inplace_func,),
        kwargs={
            "_all_bases": (attention_output,),
            "_attention_output_base_index": 0,
            "gate": gate,
        },
    )
    mutated_output = graph.call_function(getitem, args=(functionalized, 1))
    clone = graph.call_function(torch.ops.aten.clone.default, args=(mutated_output,))
    graph.output(clone)

    remove_copy_pass.remove_copy_for_mutates_args(graph)

    inplace_nodes = [node for node in graph.nodes if node.target == inplace_func]
    assert len(inplace_nodes) == 1
    assert inplace_nodes[0].kwargs == {
        "attention_output": attention_output,
        "gate": gate,
    }
    assert clone.args[0] is attention_output
    assert all(node.target != auto_functionalized_v2 for node in graph.nodes)
    graph.lint()


@pytest.mark.parametrize(
    "inplace_func",
    [
        torch.ops.trtllm.pp_recv_tensors.default,
        torch.ops.trtllm.pp_send_tensors.default,
    ],
)
def test_remove_copy_for_mutates_tensor_list(
    inplace_func: Callable[..., object],
) -> None:
    graph = Graph()
    tensor_0 = graph.placeholder("tensor_0")
    tensor_1 = graph.placeholder("tensor_1")
    functionalized = graph.call_function(
        auto_functionalized_v2,
        args=(inplace_func,),
        kwargs={
            "_all_bases": (tensor_0, tensor_1),
            "_tensors_length": 2,
            "_tensors_0_base_index": 0,
            "_tensors_1_base_index": 1,
        },
    )
    mutated_0 = graph.call_function(getitem, args=(functionalized, 1))
    mutated_1 = graph.call_function(getitem, args=(functionalized, 2))
    clone_0 = graph.call_function(torch.ops.aten.clone.default, args=(mutated_0,))
    clone_1 = graph.call_function(torch.ops.aten.clone.default, args=(mutated_1,))
    graph.output((clone_0, clone_1))

    remove_copy_pass.remove_copy_for_mutates_args(graph)

    inplace_nodes = [node for node in graph.nodes if node.target == inplace_func]
    assert len(inplace_nodes) == 1
    assert inplace_nodes[0].kwargs == {"tensors": [tensor_0, tensor_1]}
    assert clone_0.args[0] is tensor_0
    assert clone_1.args[0] is tensor_1
    assert all(node.target != auto_functionalized_v2 for node in graph.nodes)
    graph.lint()


def test_remove_copy_for_mla_restores_final_output_mutation() -> None:
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
            "_all_bases": (output,),
            "_output_base_index": 0,
        },
    )
    mutated_output = graph.call_function(getitem, args=(functionalized, 1))
    clone = graph.call_function(torch.ops.aten.clone.default, args=(mutated_output,))
    graph.output(clone)

    remove_copy_pass.remove_copy_for_mutates_args(graph)

    inplace_nodes = [node for node in graph.nodes if node.target == inplace_func]
    assert len(inplace_nodes) == 1
    assert inplace_nodes[0].kwargs["output"] is output
    assert clone.args[0] is output
    graph.lint()
