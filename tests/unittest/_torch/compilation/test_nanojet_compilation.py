# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from operator import getitem

import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import Graph, GraphModule

from tensorrt_llm._torch import nanojet_utils
from tensorrt_llm._torch.compilation import patterns as compilation_patterns
from tensorrt_llm._torch.compilation.backend import Backend
from tensorrt_llm._torch.compilation.patterns import nanojet as nanojet_patterns
from tensorrt_llm._torch.compilation.patterns.nanojet import register_nanojet_fusions
from tensorrt_llm._torch.utils import model_extra_attrs

aten = torch.ops.aten


def _nanojet_pass(pass_name: str, monkeypatch):
    monkeypatch.setattr(nanojet_patterns, "get_optional_trtllm_op", lambda _: object())
    custom_passes = []
    register_nanojet_fusions(custom_passes)
    return next(custom_pass for custom_pass in custom_passes if custom_pass.pass_name == pass_name)


def _add_norm_graph() -> GraphModule:
    graph = Graph()
    hidden_states = graph.placeholder("hidden_states")
    weight = graph.placeholder("weight")
    input_scale = graph.placeholder("input_scale")
    weight_scale = graph.placeholder("weight_scale")
    residual = graph.placeholder("residual")
    norm_weight = graph.placeholder("norm_weight")
    output_scale = graph.placeholder("output_scale")
    transposed_weight = graph.call_function(aten.permute.default, args=(weight, [1, 0]))
    mm = graph.call_function(
        torch.ops.trtllm.cublas_scaled_mm.default,
        args=(
            hidden_states,
            transposed_weight,
            input_scale,
            weight_scale,
            None,
            torch.bfloat16,
        ),
    )
    add = graph.call_function(aten.add.Tensor, args=(mm, residual))
    norm = graph.call_function(
        torch.ops.trtllm.flashinfer_rmsnorm.default,
        args=(add, norm_weight, 1e-6),
    )
    quant = graph.call_function(
        torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor.default,
        args=(norm, output_scale),
    )
    output = graph.call_function(getitem, args=(quant, 0))
    graph.output((output, add))

    with FakeTensorMode():
        hidden_value = torch.empty(4, 1024, dtype=torch.float8_e4m3fn, device="cuda")
        weight_value = torch.empty(768, 1024, dtype=torch.float8_e4m3fn, device="cuda")
        scale_value = torch.empty((), dtype=torch.float32, device="cuda")
        residual_value = torch.empty(4, 768, dtype=torch.bfloat16, device="cuda")
        norm_weight_value = torch.empty(768, dtype=torch.bfloat16, device="cuda")
        mm_value = torch.empty(4, 768, dtype=torch.bfloat16, device="cuda")
        output_value = torch.empty(4, 768, dtype=torch.float8_e4m3fn, device="cuda")
    hidden_states.meta["val"] = hidden_value
    weight.meta["val"] = weight_value
    input_scale.meta["val"] = scale_value
    weight_scale.meta["val"] = scale_value
    residual.meta["val"] = residual_value
    norm_weight.meta["val"] = norm_weight_value
    output_scale.meta["val"] = scale_value
    mm.meta["val"] = mm_value
    add.meta["val"] = residual_value
    norm.meta["val"] = residual_value
    quant.meta["val"] = (output_value, scale_value)
    output.meta["val"] = output_value
    return GraphModule({}, graph)


def _qkv_graph() -> GraphModule:
    graph = Graph()
    hidden_states = graph.placeholder("hidden_states")
    weight = graph.placeholder("weight")
    input_scale = graph.placeholder("input_scale")
    weight_scale = graph.placeholder("weight_scale")
    q_weight = graph.placeholder("q_weight")
    k_weight = graph.placeholder("k_weight")
    position_ids = graph.placeholder("position_ids")
    transposed_weight = graph.call_function(aten.permute.default, args=(weight, [1, 0]))
    projected = graph.call_function(
        torch.ops.trtllm.cublas_scaled_mm.default,
        args=(
            hidden_states,
            transposed_weight,
            input_scale,
            weight_scale,
            None,
            torch.bfloat16,
        ),
    )
    functionalized = graph.call_function(
        auto_functionalized,
        args=(torch.ops.trtllm.fused_qk_norm_rope.default,),
        kwargs={
            "qkv": projected,
            "num_heads_q": 8,
            "num_heads_k": 2,
            "num_heads_v": 2,
            "head_dim": 64,
            "rotary_dim": 64,
            "eps": 1e-6,
            "q_weight": q_weight,
            "k_weight": k_weight,
            "base": 1_000_000.0,
            "is_neox": True,
            "position_ids": position_ids,
            "factor": 1.0,
            "low": 0.0,
            "high": 0.0,
            "attention_factor": 1.0,
            "is_qk_norm": True,
            "use_gemma": False,
            "use_mrope": False,
            "mrope_section1": 0,
            "mrope_section2": 0,
        },
    )
    output = graph.call_function(getitem, args=(functionalized, 1))
    graph.output(output)

    with FakeTensorMode():
        hidden_value = torch.empty(4, 1024, dtype=torch.float8_e4m3fn, device="cuda")
        weight_value = torch.empty(768, 1024, dtype=torch.float8_e4m3fn, device="cuda")
        scale_value = torch.empty((), dtype=torch.float32, device="cuda")
        norm_weight_value = torch.empty(64, dtype=torch.bfloat16, device="cuda")
        positions_value = torch.empty(4, dtype=torch.int32, device="cuda")
        output_value = torch.empty(4, 768, dtype=torch.bfloat16, device="cuda")
    hidden_states.meta["val"] = hidden_value
    weight.meta["val"] = weight_value
    input_scale.meta["val"] = scale_value
    weight_scale.meta["val"] = scale_value
    q_weight.meta["val"] = norm_weight_value
    k_weight.meta["val"] = norm_weight_value
    position_ids.meta["val"] = positions_value
    projected.meta["val"] = output_value
    functionalized.meta["val"] = (None, output_value)
    output.meta["val"] = output_value
    return GraphModule({}, graph)


def _ensure_fused_add_norm_custom_ops() -> torch.library.Library | None:
    schemas = []
    for op, schema in (
        (
            "nanojet_gemm_fp8_add_",
            "nanojet_gemm_fp8_add_(Tensor hidden_states, Tensor weight, Tensor(a!) residual, "
            "Tensor input_scale, Tensor weight_scale) -> ()",
        ),
        (
            "nanojet_rmsnorm_fp8",
            "nanojet_rmsnorm_fp8(Tensor hidden_states, Tensor weight, float eps, "
            "Tensor output_scale) -> Tensor",
        ),
    ):
        try:
            getattr(torch.ops.trtllm, op).default
        except AttributeError:
            schemas.append((op, schema))
    if not schemas:
        return None

    library = torch.library.Library("trtllm", "FRAGMENT")
    for _, schema in schemas:
        library.define(schema)
    if any(op == "nanojet_rmsnorm_fp8" for op, _ in schemas):
        library.impl(
            "nanojet_rmsnorm_fp8",
            lambda hidden_states, weight, eps, output_scale: torch.empty_like(
                hidden_states, dtype=torch.float8_e4m3fn
            ),
            "Meta",
        )
    return library


def _ensure_qkv_custom_op() -> torch.library.Library | None:
    try:
        torch.ops.trtllm.nanojet_fused_qkv_gemm_norm_rope.default
    except AttributeError:
        library = torch.library.Library("trtllm", "FRAGMENT")
        library.define(
            "nanojet_fused_qkv_gemm_norm_rope("
            "Tensor hidden_states, Tensor qkv_weight, Tensor query_norm_weight, "
            "Tensor key_norm_weight, Tensor position_ids, Tensor input_scale, "
            "Tensor weight_scale, float eps, int num_heads_q, int num_heads_k, "
            "int num_heads_v, int head_dim) -> Tensor"
        )
        library.impl(
            "nanojet_fused_qkv_gemm_norm_rope",
            lambda hidden_states, *args: hidden_states.new_empty(
                (hidden_states.shape[0], 768), dtype=torch.bfloat16
            ),
            "Meta",
        )
        return library
    return None


def test_nanojet_uses_separate_named_aten_passes(monkeypatch) -> None:
    monkeypatch.setattr(nanojet_patterns, "get_optional_trtllm_op", lambda _: object())
    custom_passes = []
    register_nanojet_fusions(custom_passes)
    pass_names = [custom_pass.pass_name for custom_pass in custom_passes]

    assert pass_names == [
        "nanojet_rmsnorm",
        "nanojet_qkv",
        "nanojet_attention_quant",
        "nanojet_swiglu",
        "nanojet_gemm_add",
    ]


def test_nanojet_registers_passes_without_patching_backend(monkeypatch) -> None:
    monkeypatch.setattr(compilation_patterns, "_CUSTOM_PASS_REGISTRARS", {})
    monkeypatch.setattr(nanojet_patterns, "get_optional_trtllm_op", lambda _: object())
    original_build_custom_passes = Backend.build_custom_passes.__func__

    nanojet_utils._register_compilation_passes()
    nanojet_utils._register_compilation_passes()
    custom_passes = []
    compilation_patterns.append_registered_custom_passes(custom_passes)

    assert Backend.build_custom_passes.__func__ is original_build_custom_passes
    assert list(compilation_patterns._CUSTOM_PASS_REGISTRARS) == ["nanojet"]
    assert [custom_pass.pass_name for custom_pass in custom_passes] == [
        "nanojet_rmsnorm",
        "nanojet_qkv",
        "nanojet_attention_quant",
        "nanojet_swiglu",
        "nanojet_gemm_add",
    ]


def test_nanojet_passes_are_absent_without_custom_ops(monkeypatch) -> None:
    monkeypatch.setattr(nanojet_patterns, "get_optional_trtllm_op", lambda _: None)
    custom_passes = []

    register_nanojet_fusions(custom_passes)

    assert custom_passes == []


def test_add_norm_composes_nanojet_patterns(monkeypatch) -> None:
    custom_op_library = _ensure_fused_add_norm_custom_ops()
    gm = _add_norm_graph()
    monkeypatch.setattr(nanojet_patterns, "nanojet_supports", lambda *args, **kwargs: True)
    rmsnorm_pass = _nanojet_pass("nanojet_rmsnorm", monkeypatch)
    gemm_add_pass = _nanojet_pass("nanojet_gemm_add", monkeypatch)

    with model_extra_attrs({"nanojet_enabled": True}):
        assert rmsnorm_pass.apply(gm) == 1
        assert gemm_add_pass.apply(gm) == 1

    gm.graph.eliminate_dead_code()
    assert any(
        node.target == torch.ops.trtllm.nanojet_rmsnorm_fp8.default for node in gm.graph.nodes
    )
    assert any(
        node.target == auto_functionalized
        and node.args[0] == torch.ops.trtllm.nanojet_gemm_fp8_add_.default
        for node in gm.graph.nodes
    )
    assert not any(
        node.target
        in {
            aten.add.Tensor,
            torch.ops.trtllm.cublas_scaled_mm.default,
            torch.ops.trtllm.flashinfer_rmsnorm.default,
            torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor.default,
        }
        for node in gm.graph.nodes
    )
    gm.graph.lint()
    assert custom_op_library is None or isinstance(custom_op_library, torch.library.Library)


def test_qkv_pattern_replaces_projection_norm_and_rope(monkeypatch) -> None:
    custom_op_library = _ensure_qkv_custom_op()
    gm = _qkv_graph()
    monkeypatch.setattr(nanojet_patterns, "nanojet_supports", lambda *args, **kwargs: True)
    qkv_pass = _nanojet_pass("nanojet_qkv", monkeypatch)

    with model_extra_attrs(
        {
            "nanojet_enabled": True,
            "nanojet_rope_table": torch.empty(16, 64, dtype=torch.bfloat16),
        }
    ):
        assert qkv_pass.apply(gm) == 1

    gm.graph.eliminate_dead_code()
    assert any(
        node.target == torch.ops.trtllm.nanojet_fused_qkv_gemm_norm_rope.default
        for node in gm.graph.nodes
    )
    assert not any(
        node.target
        in {
            torch.ops.trtllm.cublas_scaled_mm.default,
            torch.ops.trtllm.fused_qk_norm_rope.default,
        }
        for node in gm.graph.nodes
    )
    gm.graph.lint()
    assert custom_op_library is None or isinstance(custom_op_library, torch.library.Library)
