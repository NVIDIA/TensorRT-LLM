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
from unittest.mock import patch

import pytest
import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized, auto_functionalized_v2
from torch.fx import Graph

import tensorrt_llm._torch.compilation.remove_copy_pass as remove_copy_pass
import tensorrt_llm._torch.custom_ops  # noqa: F401


def _fake_inplace(input, output, output_sf=None):
    pass


def _fake_inplace_map():
    return {_fake_inplace: {1: "output", 2: "output_sf"}}


def _run_remove_copy(graph):
    with patch.object(remove_copy_pass, "inplace_info", _fake_inplace_map):
        remove_copy_pass.remove_copy_for_mutates_args(graph)
    graph.lint()


def _call_function_nodes(graph):
    return [node for node in graph.nodes if node.op == "call_function"]


def _targets(graph):
    return [node.target for node in _call_function_nodes(graph)]


# Registers torch.ops.trtllm.mla_custom_op_inplace, used below. The op is a
# module-import side effect that the eagerly loaded model zoo used to provide
# (models/__init__ -> modeling_deepseekv3 -> modules.mla); with the zoo lazy,
# every test must import the ops it uses itself.
import tensorrt_llm._torch.modules.mla  # noqa: E402, F401
from tensorrt_llm._torch.modules.fused_ops.fused_qk_norm_rope_gate import (  # noqa: E402
    fused_sigmoid_mul_inplace,  # noqa: E402, F401
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


def test_remove_copy_handles_named_functionalized_outputs():
    graph = Graph()
    input_node = graph.placeholder("input")
    output = graph.placeholder("output")
    output_sf = graph.placeholder("output_sf")
    functionalized = graph.call_function(
        auto_functionalized,
        args=(_fake_inplace,),
        kwargs={
            "input": input_node,
            "output": output,
            "output_sf": output_sf,
        },
    )
    output_getitem = graph.call_function(getitem, args=(functionalized, "output"))
    output_sf_getitem = graph.call_function(getitem, args=(functionalized, "output_sf"))
    graph.output((output_getitem, output_sf_getitem))

    _run_remove_copy(graph)

    targets = _targets(graph)
    assert auto_functionalized not in targets
    assert getitem not in targets
    assert _fake_inplace in targets


def test_remove_copy_uses_registered_nvfp4_inplace_mapping():
    inplace_op = torch.ops.trtllm.nvfp4_gemm_inplace.default
    graph = Graph()
    output_tensor = graph.placeholder("output_tensor")
    functionalized = graph.call_function(
        auto_functionalized,
        args=(inplace_op,),
        kwargs={
            "act_fp4": graph.placeholder("act_fp4"),
            "weight": graph.placeholder("weight"),
            "act_sf": graph.placeholder("act_sf"),
            "weight_scale": graph.placeholder("weight_scale"),
            "alpha": graph.placeholder("alpha"),
            "output_dtype": torch.bfloat16,
            "to_userbuffers": False,
            "allowed_backends": "cutedsl",
            "output_tensor": output_tensor,
            "partition_id": 0,
        },
    )
    output_getitem = graph.call_function(getitem, args=(functionalized, 1))
    graph.output(output_getitem)

    remove_copy_pass.remove_copy_for_mutates_args(graph)
    graph.lint()

    targets = _targets(graph)
    assert auto_functionalized not in targets
    assert getitem not in targets
    inplace_node = next(node for node in _call_function_nodes(graph) if node.target is inplace_op)
    assert inplace_node.kwargs["output_tensor"] is output_tensor
    output_node = next(node for node in graph.nodes if node.op == "output")
    assert output_node.args == (output_tensor,)


def test_inplace_info_declares_every_mutated_arg():
    """Every inplace_info entry must name ALL of its op's mutates_args.

    remove_copy_for_mutates_args rebuilds the direct call from
    ``{k: v for k, v in node.kwargs.items() if not k.startswith("_")}`` plus the
    names in inplace_info. auto_functionalize hoists every mutable argument into
    underscore-prefixed metadata, so a mutable arg missing from the map is
    dropped from the rewritten call entirely — and the failure surfaces far away,
    as a bare "missing value for argument" RuntimeError at execution time rather
    than anywhere near this table.

    That is how nvbugs/6474888 recurred: DSv4 epilogue fusion added
    ``dsv4_output``/``dsv4_output_sf`` to ``mla_custom_op_inplace``'s
    mutates_args, this map still listed only ``output``, and 4xVR200
    torch_compile runs died with
    ``mla_custom_op_inplace() is missing value for argument 'dsv4_output'``.

    Asserting the invariant rather than that one op is deliberate: the same
    omission is available to every future in-place custom op, and the point fix
    would not catch the next one.
    """
    # Importing mla registers trtllm::mla_custom_op_inplace. Without it,
    # get_optional_trtllm_op() returns None, inplace_info() omits the entry
    # entirely, and this test would pass by checking nothing -- the vacuous
    # pass is indistinguishable from a real one.
    from tensorrt_llm._torch.compilation.utils import inplace_info

    table = inplace_info()
    assert table, "inplace_info() is empty; no op was registered, so this test would be vacuous"
    registered = {op._schema.name for op in table}
    assert "trtllm::mla_custom_op_inplace" in registered, (
        "trtllm::mla_custom_op_inplace is absent from inplace_info(); the "
        "regression this test exists for could not be detected. Registered: "
        f"{sorted(registered)}"
    )

    mismatches = []
    for op_overload, declared in table.items():
        schema = op_overload._schema
        actual = [
            argument.name
            for argument in schema.arguments
            if argument.alias_info is not None and argument.alias_info.is_write
        ]
        # Ordered comparison: inplace_info values follow the op's schema
        # declaration order, so order is part of the contract. The KEYS
        # are deliberately NOT asserted here. Most entries key by 1-based
        # position among the op's mutable args, but
        # verify_dynamic_tree_rejection_out_op instead keys by position in
        # the full argument list ({5, 6, 7} for the three Tensor(a!)/(b!)/
        # (c!) args in cpp/tensorrt_llm/thop/dynamicTreeOp.cpp), so
        # asserting the key convention here would go red today. That
        # divergence is split out as a separate defect: it costs a silent
        # missed copy-removal on the non-v2 path, where the getitem
        # indices are absent from output_replacements and the pass bails.
        # It is out of scope for nvbugs/6474888.
        if list(declared.values()) != actual:
            mismatches.append(
                f"{schema.name}: inplace_info declares {list(declared.values())}, "
                f"op mutates {actual}"
            )

    assert not mismatches, "inplace_info is out of sync with op schemas:\n" + "\n".join(mismatches)
