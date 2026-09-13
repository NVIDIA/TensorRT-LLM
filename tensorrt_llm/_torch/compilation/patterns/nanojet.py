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

"""NanoJet fusion patterns for the PyTorch backend."""

from operator import getitem

import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import (
    MULTIPLE,
    CallFunction,
    KeywordArg,
    Match,
    PatternMatcherPass,
    fwd_only,
    register_replacement,
)
from torch.fx import Node

from ...nanojet_utils import nanojet_supports
from ...utils import get_model_extra_attrs
from ..utils import get_optional_trtllm_op
from . import MATCHER_SUBSYSTEM

aten = torch.ops.aten


def _is_nanojet_enabled() -> bool:
    attrs = get_model_extra_attrs()
    return bool(attrs is not None and attrs.get("nanojet_enabled", False))


def _tensor_meta(value: object) -> torch.Tensor | None:
    if not isinstance(value, Node):
        return None
    tensor = value.meta.get("val")
    return tensor if isinstance(tensor, torch.Tensor) else None


def _is_scalar_tensor(value: object) -> bool:
    tensor = _tensor_meta(value)
    return tensor is not None and tensor.numel() == 1


def _append_pass(custom_passes: list[PatternMatcherPass], pass_name: str) -> PatternMatcherPass:
    custom_pass = PatternMatcherPass(pass_name, MATCHER_SUBSYSTEM)
    custom_passes.append(custom_pass)
    return custom_pass


def _scaled_mm_pattern(
    prefix: str,
) -> tuple[CallFunction, KeywordArg, KeywordArg, KeywordArg, KeywordArg]:
    hidden_states = KeywordArg(f"{prefix}_hidden_states")
    weight = KeywordArg(f"{prefix}_weight")
    input_scale = KeywordArg(f"{prefix}_input_scale")
    weight_scale = KeywordArg(f"{prefix}_weight_scale")
    transposed_weight = CallFunction(
        aten.permute.default,
        weight,
        [1, 0],
        _users=1,
    )
    output = CallFunction(
        torch.ops.trtllm.cublas_scaled_mm.default,
        hidden_states,
        transposed_weight,
        input_scale,
        weight_scale,
        None,
        torch.bfloat16,
        _users=1,
    )
    return output, hidden_states, weight, input_scale, weight_scale


def _register_rmsnorm_fusion(custom_pass: PatternMatcherPass) -> None:
    hidden_states = KeywordArg("rmsnorm_hidden_states")
    weight = KeywordArg("rmsnorm_weight")
    eps = KeywordArg("rmsnorm_eps")
    norm = CallFunction(
        torch.ops.trtllm.flashinfer_rmsnorm.default,
        hidden_states,
        weight,
        eps,
        _users=1,
    )
    output_scale = KeywordArg("rmsnorm_output_scale")
    quant = CallFunction(
        torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor.default,
        norm,
        output_scale,
        _users=1,
    )
    output = CallFunction(getitem, quant, 0, _users=MULTIPLE)

    def empty_pattern(
        rmsnorm_hidden_states: torch.Tensor,
        rmsnorm_weight: torch.Tensor,
        rmsnorm_eps: float,
        rmsnorm_output_scale: torch.Tensor,
    ) -> None:
        return None

    def target_pattern(
        rmsnorm_hidden_states: torch.Tensor,
        rmsnorm_weight: torch.Tensor,
        rmsnorm_eps: float,
        rmsnorm_output_scale: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.trtllm.nanojet_rmsnorm_fp8(
            rmsnorm_hidden_states,
            rmsnorm_weight,
            rmsnorm_eps,
            rmsnorm_output_scale,
        )

    def extra_check(match: Match) -> bool:
        if not _is_nanojet_enabled():
            return False
        hidden = _tensor_meta(match.kwargs["rmsnorm_hidden_states"])
        weight_value = _tensor_meta(match.kwargs["rmsnorm_weight"])
        if (
            hidden is None
            or hidden.dtype != torch.bfloat16
            or hidden.ndim < 2
            or weight_value is None
            or weight_value.dtype != torch.bfloat16
            or tuple(weight_value.shape) != (hidden.shape[-1],)
            or not _is_scalar_tensor(match.kwargs["rmsnorm_output_scale"])
        ):
            return False
        return nanojet_supports(
            "unified_rmsnorm",
            hidden_size=int(hidden.shape[-1]),
            hidden_states_dtype=hidden.dtype,
            zero_centered_weight=False,
            multiply_in_fp32=False,
        )

    register_replacement(
        empty_pattern,
        target_pattern,
        [],
        fwd_only,
        custom_pass,
        search_fn_pattern=output,
        extra_check=extra_check,
    )


def _register_qkv_fusion(custom_pass: PatternMatcherPass) -> None:
    mm, hidden_states, weight, input_scale, weight_scale = _scaled_mm_pattern("qkv")
    q_weight = KeywordArg("qkv_q_weight")
    k_weight = KeywordArg("qkv_k_weight")
    position_ids = KeywordArg("qkv_position_ids")
    eps = KeywordArg("qkv_eps")
    base = KeywordArg("qkv_base")
    num_heads_q = KeywordArg("qkv_num_heads_q")
    num_heads_k = KeywordArg("qkv_num_heads_k")
    head_dim = KeywordArg("qkv_head_dim")
    functionalized = CallFunction(
        auto_functionalized,
        torch.ops.trtllm.fused_qk_norm_rope.default,
        qkv=mm,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        num_heads_v=num_heads_k,
        head_dim=head_dim,
        rotary_dim=head_dim,
        eps=eps,
        q_weight=q_weight,
        k_weight=k_weight,
        base=base,
        is_neox=True,
        position_ids=position_ids,
        factor=1.0,
        low=0.0,
        high=0.0,
        attention_factor=1.0,
        is_qk_norm=True,
        use_gemma=False,
        use_mrope=False,
        mrope_section1=0,
        mrope_section2=0,
        _users=1,
    )
    output = CallFunction(getitem, functionalized, 1, _users=MULTIPLE)

    def empty_pattern(
        qkv_hidden_states: torch.Tensor,
        qkv_weight: torch.Tensor,
        qkv_input_scale: torch.Tensor,
        qkv_weight_scale: torch.Tensor,
        qkv_q_weight: torch.Tensor,
        qkv_k_weight: torch.Tensor,
        qkv_position_ids: torch.Tensor,
        qkv_eps: float,
        qkv_base: float,
        qkv_num_heads_q: int,
        qkv_num_heads_k: int,
        qkv_head_dim: int,
    ) -> None:
        return None

    def target_pattern(
        qkv_hidden_states: torch.Tensor,
        qkv_weight: torch.Tensor,
        qkv_input_scale: torch.Tensor,
        qkv_weight_scale: torch.Tensor,
        qkv_q_weight: torch.Tensor,
        qkv_k_weight: torch.Tensor,
        qkv_position_ids: torch.Tensor,
        qkv_eps: float,
        qkv_base: float,
        qkv_num_heads_q: int,
        qkv_num_heads_k: int,
        qkv_head_dim: int,
    ) -> torch.Tensor:
        return torch.ops.trtllm.nanojet_fused_qkv_gemm_norm_rope(
            qkv_hidden_states,
            qkv_weight,
            qkv_q_weight,
            qkv_k_weight,
            qkv_position_ids,
            qkv_input_scale,
            qkv_weight_scale,
            qkv_eps,
            qkv_num_heads_q,
            qkv_num_heads_k,
            qkv_num_heads_k,
            qkv_head_dim,
        )

    def extra_check(match: Match) -> bool:
        if not _is_nanojet_enabled():
            return False
        hidden = _tensor_meta(match.kwargs["qkv_hidden_states"])
        weight_value = _tensor_meta(match.kwargs["qkv_weight"])
        q_weight_value = _tensor_meta(match.kwargs["qkv_q_weight"])
        k_weight_value = _tensor_meta(match.kwargs["qkv_k_weight"])
        positions = _tensor_meta(match.kwargs["qkv_position_ids"])
        num_q = match.kwargs["qkv_num_heads_q"]
        num_k = match.kwargs["qkv_num_heads_k"]
        dim = match.kwargs["qkv_head_dim"]
        attrs = get_model_extra_attrs()
        rope_table = None if attrs is None else attrs.get("nanojet_rope_table")
        if (
            hidden is None
            or hidden.dtype != torch.float8_e4m3fn
            or hidden.ndim != 2
            or weight_value is None
            or weight_value.dtype != torch.float8_e4m3fn
            or weight_value.ndim != 2
            or not all(isinstance(value, int) for value in (num_q, num_k, dim))
            or tuple(weight_value.shape) != ((num_q + 2 * num_k) * dim, hidden.shape[-1])
            or q_weight_value is None
            or q_weight_value.dtype != torch.bfloat16
            or tuple(q_weight_value.shape) != (dim,)
            or k_weight_value is None
            or k_weight_value.dtype != torch.bfloat16
            or tuple(k_weight_value.shape) != (dim,)
            or positions is None
            or positions.dtype != torch.int32
            or positions.numel() != hidden.shape[0]
            or not isinstance(rope_table, torch.Tensor)
            or rope_table.dtype != torch.bfloat16
            or rope_table.ndim != 2
            or rope_table.shape[1] != dim
            or not _is_scalar_tensor(match.kwargs["qkv_input_scale"])
            or not _is_scalar_tensor(match.kwargs["qkv_weight_scale"])
        ):
            return False
        return nanojet_supports(
            "fused_qkv_gemm_norm_rope",
            input_dtype=hidden.dtype,
            weight_dtype=weight_value.dtype,
            position_ids_dtype=positions.dtype,
            head_dim=dim,
        )

    register_replacement(
        empty_pattern,
        target_pattern,
        [],
        fwd_only,
        custom_pass,
        search_fn_pattern=output,
        extra_check=extra_check,
    )


def _register_attention_quant_fusion(custom_pass: PatternMatcherPass) -> None:
    attention_op = get_optional_trtllm_op("attn_custom_op_inplace")
    if attention_op is None:
        return
    q = KeywordArg("attention_q")
    k = KeywordArg("attention_k")
    v = KeywordArg("attention_v")
    attention_mask = KeywordArg("attention_mask")
    attention_window_size = KeywordArg("attention_window_size")
    attention_mask_data = KeywordArg("attention_mask_data")
    attention_sinks = KeywordArg("attention_sinks")
    relative_attention_bias = KeywordArg("relative_attention_bias")
    relative_attention_max_distance = KeywordArg("relative_attention_max_distance")
    layer_idx = KeywordArg("attention_layer_idx")
    output_buffer = KeywordArg("attention_output_buffer")
    functionalized = CallFunction(
        auto_functionalized,
        attention_op,
        q=q,
        k=k,
        v=v,
        attention_mask=attention_mask,
        mrope_rotary_cos_sin=None,
        mrope_position_deltas=None,
        attention_window_size=attention_window_size,
        attention_mask_data=attention_mask_data,
        attention_sinks=attention_sinks,
        relative_attention_bias=relative_attention_bias,
        relative_attention_max_distance=relative_attention_max_distance,
        layer_idx=layer_idx,
        output=output_buffer,
        output_sf=None,
        _users=1,
    )
    attention_output = CallFunction(getitem, functionalized, 1, _users=1)
    output_scale = KeywordArg("attention_output_scale")
    quant = CallFunction(
        torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor.default,
        attention_output,
        output_scale,
        _users=1,
    )
    output = CallFunction(getitem, quant, 0, _users=MULTIPLE)

    def empty_pattern(
        attention_q: torch.Tensor,
        attention_k: torch.Tensor | None,
        attention_v: torch.Tensor | None,
        attention_mask: str,
        attention_window_size: int | None,
        attention_mask_data: torch.Tensor | None,
        attention_sinks: torch.Tensor | None,
        relative_attention_bias: torch.Tensor | None,
        relative_attention_max_distance: int,
        attention_layer_idx: str,
        attention_output_buffer: torch.Tensor,
        attention_output_scale: torch.Tensor,
    ) -> None:
        return None

    def target_pattern(
        attention_q: torch.Tensor,
        attention_k: torch.Tensor | None,
        attention_v: torch.Tensor | None,
        attention_mask: str,
        attention_window_size: int | None,
        attention_mask_data: torch.Tensor | None,
        attention_sinks: torch.Tensor | None,
        relative_attention_bias: torch.Tensor | None,
        relative_attention_max_distance: int,
        attention_layer_idx: str,
        attention_output_buffer: torch.Tensor,
        attention_output_scale: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.trtllm.nanojet_attention_fp8(
            attention_q,
            attention_k,
            attention_v,
            attention_mask,
            attention_window_size,
            attention_mask_data,
            attention_sinks,
            relative_attention_bias,
            relative_attention_max_distance,
            attention_layer_idx,
            attention_output_scale,
        )

    def extra_check(match: Match) -> bool:
        if not _is_nanojet_enabled():
            return False
        q_value = _tensor_meta(match.kwargs["attention_q"])
        output_value = _tensor_meta(match.kwargs["attention_output_buffer"])
        return (
            q_value is not None
            and q_value.dtype == torch.bfloat16
            and q_value.ndim == 2
            and output_value is not None
            and output_value.dtype == torch.bfloat16
            and output_value.ndim == 2
            and _is_scalar_tensor(match.kwargs["attention_output_scale"])
        )

    register_replacement(
        empty_pattern,
        target_pattern,
        [],
        fwd_only,
        custom_pass,
        search_fn_pattern=output,
        extra_check=extra_check,
    )


def _register_swiglu_pattern(
    custom_pass: PatternMatcherPass, *, include_optional_args: bool
) -> None:
    mm, hidden_states, weight, input_scale, weight_scale = _scaled_mm_pattern("swiglu")
    output_scale = KeywordArg("swiglu_output_scale")
    args = [mm, output_scale, torch.float8_e4m3fn]
    if include_optional_args:
        args.extend([None, None, None])
    output = CallFunction(
        torch.ops.trtllm.silu_and_mul.default,
        *args,
        _users=MULTIPLE,
    )

    def empty_pattern(
        swiglu_hidden_states: torch.Tensor,
        swiglu_weight: torch.Tensor,
        swiglu_input_scale: torch.Tensor,
        swiglu_weight_scale: torch.Tensor,
        swiglu_output_scale: torch.Tensor,
    ) -> None:
        return None

    def target_pattern(
        swiglu_hidden_states: torch.Tensor,
        swiglu_weight: torch.Tensor,
        swiglu_input_scale: torch.Tensor,
        swiglu_weight_scale: torch.Tensor,
        swiglu_output_scale: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.trtllm.nanojet_swiglu_gemm_fp8(
            swiglu_hidden_states,
            swiglu_weight,
            swiglu_input_scale,
            swiglu_weight_scale,
            swiglu_output_scale,
        )

    def extra_check(match: Match) -> bool:
        if not _is_nanojet_enabled():
            return False
        hidden = _tensor_meta(match.kwargs["swiglu_hidden_states"])
        weight_value = _tensor_meta(match.kwargs["swiglu_weight"])
        return (
            hidden is not None
            and hidden.dtype == torch.float8_e4m3fn
            and hidden.ndim == 2
            and weight_value is not None
            and weight_value.dtype == torch.float8_e4m3fn
            and weight_value.ndim == 2
            and weight_value.shape[0] % 2 == 0
            and hidden.shape[-1] == weight_value.shape[-1]
            and _is_scalar_tensor(match.kwargs["swiglu_input_scale"])
            and _is_scalar_tensor(match.kwargs["swiglu_weight_scale"])
            and _is_scalar_tensor(match.kwargs["swiglu_output_scale"])
        )

    register_replacement(
        empty_pattern,
        target_pattern,
        [],
        fwd_only,
        custom_pass,
        search_fn_pattern=output,
        extra_check=extra_check,
    )


def _register_swiglu_fusion(custom_pass: PatternMatcherPass) -> None:
    _register_swiglu_pattern(custom_pass, include_optional_args=False)
    _register_swiglu_pattern(custom_pass, include_optional_args=True)


def _register_gemm_add_pattern(custom_pass: PatternMatcherPass, *, gemm_first: bool) -> None:
    mm, hidden_states, weight, input_scale, weight_scale = _scaled_mm_pattern("gemm_add")
    residual = KeywordArg("gemm_add_residual")
    add_args = (mm, residual) if gemm_first else (residual, mm)
    output = CallFunction(aten.add.Tensor, *add_args, _users=MULTIPLE)

    def empty_pattern(
        gemm_add_hidden_states: torch.Tensor,
        gemm_add_weight: torch.Tensor,
        gemm_add_input_scale: torch.Tensor,
        gemm_add_weight_scale: torch.Tensor,
        gemm_add_residual: torch.Tensor,
    ) -> None:
        return None

    def target_pattern(
        gemm_add_hidden_states: torch.Tensor,
        gemm_add_weight: torch.Tensor,
        gemm_add_input_scale: torch.Tensor,
        gemm_add_weight_scale: torch.Tensor,
        gemm_add_residual: torch.Tensor,
    ) -> torch.Tensor:
        functionalized = auto_functionalized(
            torch.ops.trtllm.nanojet_gemm_fp8_add_.default,
            hidden_states=gemm_add_hidden_states,
            weight=gemm_add_weight,
            residual=gemm_add_residual,
            input_scale=gemm_add_input_scale,
            weight_scale=gemm_add_weight_scale,
        )
        return functionalized[1]

    def extra_check(match: Match) -> bool:
        if not _is_nanojet_enabled():
            return False
        hidden_node = match.kwargs["gemm_add_hidden_states"]
        residual_node = match.kwargs["gemm_add_residual"]
        add_node = match.ctx.pattern_to_node[output]
        hidden = _tensor_meta(hidden_node)
        weight_value = _tensor_meta(match.kwargs["gemm_add_weight"])
        residual_value = _tensor_meta(residual_node)
        order = {node: index for index, node in enumerate(match.graph.nodes)}
        if (
            not isinstance(hidden_node, Node)
            or not isinstance(residual_node, Node)
            or not isinstance(add_node, Node)
            or hidden_node is residual_node
            or hidden is None
            or hidden.dtype != torch.float8_e4m3fn
            or hidden.ndim != 2
            or weight_value is None
            or weight_value.dtype != torch.float8_e4m3fn
            or weight_value.ndim != 2
            or residual_value is None
            or residual_value.dtype != torch.bfloat16
            or residual_value.ndim != 2
            or tuple(residual_value.shape) != (hidden.shape[0], weight_value.shape[0])
            or any(
                order[user] > order[add_node]
                for user in residual_node.users
                if user is not add_node
            )
            or not _is_scalar_tensor(match.kwargs["gemm_add_input_scale"])
            or not _is_scalar_tensor(match.kwargs["gemm_add_weight_scale"])
        ):
            return False
        return True

    register_replacement(
        empty_pattern,
        target_pattern,
        [],
        fwd_only,
        custom_pass,
        search_fn_pattern=output,
        extra_check=extra_check,
    )


def _register_gemm_add_fusion(custom_pass: PatternMatcherPass) -> None:
    _register_gemm_add_pattern(custom_pass, gemm_first=True)
    _register_gemm_add_pattern(custom_pass, gemm_first=False)


def register_nanojet_fusions(custom_passes: list[PatternMatcherPass]) -> None:
    """Register NanoJet replacements when its custom ops have been initialized."""
    if get_optional_trtllm_op("nanojet_rmsnorm_fp8") is None:
        return

    _register_rmsnorm_fusion(_append_pass(custom_passes, "nanojet_rmsnorm"))
    _register_qkv_fusion(_append_pass(custom_passes, "nanojet_qkv"))
    _register_attention_quant_fusion(_append_pass(custom_passes, "nanojet_attention_quant"))
    _register_swiglu_fusion(_append_pass(custom_passes, "nanojet_swiglu"))
    _register_gemm_add_fusion(_append_pass(custom_passes, "nanojet_gemm_add"))


__all__ = [
    "register_nanojet_fusions",
]
