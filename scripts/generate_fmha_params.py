#!/usr/bin/env python3
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
"""Generate the FMHA parameter X-macro include from one schema."""

from __future__ import annotations

import argparse
import difflib
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TypeSpec:
    cpp_type: str
    py_type: str
    cpp_default: str | None = None
    py_default: str | None = None
    is_tensor: bool = False


@dataclass(frozen=True)
class Field:
    name: str
    type: str
    init: str | None = None
    bind: bool | None = None
    enqueue: bool = False
    context: bool = False
    generation: bool = False
    data_ptr: str | None = None
    context_data_ptr: str | None = None
    generation_data_ptr: str | None = None
    ptr_type: str | None = None
    enqueue_init: str | None = None
    context_init: str | None = None
    generation_init: str | None = None


@dataclass(frozen=True)
class FieldGroup:
    comment: str
    fields: tuple[Field, ...]
    bind: bool = True
    static_config: bool = False


TYPE_SPECS: dict[str, TypeSpec] = {
    "bool": TypeSpec("bool", "bool", "false", "False"),
    "int": TypeSpec("int", "int", "0", "0"),
    "float": TypeSpec("float", "float", "0.0f", "0.0"),
    "i32": TypeSpec("int32_t", "int", "0", "0"),
    "i64": TypeSpec("int64_t", "int", "0", "0"),
    "double": TypeSpec("double", "float", "0.0", "0.0"),
    "attention_mask_type": TypeSpec("tensorrt_llm::kernels::AttentionMaskType", "int", "{}", "0"),
    "block_sparse_params": TypeSpec("tensorrt_llm::kernels::BlockSparseParams", "", "{}", None),
    "data_type": TypeSpec("tensorrt_llm::DataType", "", "{}", None),
    "mla_meta_params": TypeSpec("tensorrt_llm::kernels::MlaMetaParams", "", "{}", None),
    "optional_bool": TypeSpec("std::optional<bool>", "Optional[bool]", None, "None"),
    "optional_i64": TypeSpec("std::optional<int64_t>", "Optional[int]", None, "None"),
    "optional_double": TypeSpec("std::optional<double>", "Optional[float]", None, "None"),
    "optional_tensor": TypeSpec(
        "std::optional<torch::Tensor>",
        "Optional[torch.Tensor]",
        None,
        "None",
        True,
    ),
    "position_embedding_type": TypeSpec(
        "tensorrt_llm::kernels::PositionEmbeddingType", "int", "{}", "0"
    ),
    "quant_mode": TypeSpec("tensorrt_llm::common::QuantMode", "int", "{}", "0"),
    "rotary_scaling_type": TypeSpec("tensorrt_llm::kernels::RotaryScalingType", "int", "{}", "0"),
    "set_i32": TypeSpec("std::set<int32_t>", "", "{}", None),
    # torch::Tensor is default-constructible on the C++ side. Python uses None
    # so the generated class can also be default-constructed.
    "tensor": TypeSpec("torch::Tensor", "Optional[torch.Tensor]", None, "None", True),
}


FMHA_PARAM_GROUPS: tuple[FieldGroup, ...] = (
    FieldGroup(
        "Phase bookkeeping.",
        (
            Field("seq_offset", "i32"),
            Field("num_seqs", "i32", context=True),
            Field("token_offset", "i32", generation=True),
            Field("num_tokens", "i32", enqueue=True),
            Field("predicted_tokens_per_seq", "i32"),
            Field("input_seq_length", "i32", bind=False, enqueue=True),
            Field("max_past_kv_length", "i32", bind=False, enqueue=True),
            Field("num_requests", "i32", bind=False, generation=True),
            Field("beam_width", "i32", bind=False, generation=True, generation_init="1"),
            Field("layer_idx", "i32", bind=False, generation=True),
        ),
    ),
    FieldGroup(
        "Static AttentionOp construction and cache key.",
        (
            Field("layer_idx", "i64", init="-1"),
            Field("num_heads", "i64", init="-1"),
            Field("vision_start", "int", init="-1", bind=False),
            Field("vision_length", "int", init="-1", bind=False),
            Field("num_kv_heads", "i64", init="-1"),
            Field("head_size", "i64", init="-1"),
            Field("unidirectional", "int", init="1", bind=False),
            Field("q_scaling", "double", init="1.0"),
            Field("attn_logit_softcapping_scale", "float", bind=False),
            Field("rotary_embedding_dim", "i64"),
            Field("rotary_embedding_base", "double", init="10000.0"),
            Field("rotary_embedding_scale_type", "rotary_scaling_type"),
            Field("rotary_embedding_scale", "double", init="1.0"),
            Field("rotary_embedding_short_mscale", "double", init="1.0"),
            Field("rotary_embedding_long_mscale", "double", init="1.0"),
            Field("rotary_embedding_max_positions", "i64", init="1024"),
            Field("rotary_embedding_original_max_positions", "i64", init="1024"),
            Field("position_embedding_type", "position_embedding_type"),
            Field("use_logn_scaling", "bool", bind=False),
            Field("remove_padding", "bool", init="true", bind=False),
            Field(
                "mask_type",
                "attention_mask_type",
                init="tensorrt_llm::kernels::AttentionMaskType::CAUSAL",
            ),
            Field("block_sparse_params", "block_sparse_params", bind=False),
            Field("tokens_per_block", "i64"),
            Field("quant_mode", "quant_mode"),
            Field("tp_size", "int", init="1", bind=False),
            Field("tp_rank", "int", bind=False),
            Field("unfuse_qkv_gemm", "bool", bind=False),
            Field("type", "data_type", bind=False),
            Field("is_fp8_out", "bool", bind=False),
            Field("is_fp4_out", "bool", bind=False),
            Field("max_context_length", "i64"),
            Field("max_seq_len", "i64"),
            Field("max_num_requests", "i64"),
            Field("beam_width", "i64", init="1"),
            Field("attention_window_size", "i64"),
            Field("qkv_bias_enabled", "bool", bind=False),
            Field("cross_attention", "bool", bind=False),
            Field("max_distance", "i64"),
            Field("pos_shift_enabled", "bool", bind=False),
            Field("paged_context_fmha", "bool"),
            Field("chunk_prefill_buffer_batch_size", "i64", init="1"),
            Field("dense_context_fmha", "bool", bind=False),
            Field("has_full_attention_mask", "bool", bind=False),
            Field("is_spec_decoding_enabled", "bool"),
            Field("use_spec_decoding", "bool"),
            Field("is_spec_dec_tree", "bool", init="true"),
            Field("spec_decoding_is_generation_length_variable", "bool", bind=False),
            Field("spec_decoding_max_generation_length", "i32", init="1", bind=False),
            Field("spec_decoding_target_max_gen_len", "i32", bind=False),
            Field("force_prepare_spec_dec_tree_mask", "bool"),
            Field("is_mla_enable", "bool"),
            Field("use_sparse_attention", "bool", bind=False),
            Field("use_tllm_gen_sparse_attention_paged", "bool", bind=False),
            Field("use_tllm_gen_sparse_attention", "bool", bind=False),
            Field("mla_params", "mla_meta_params", bind=False),
            Field("cp_size", "int", init="1", bind=False),
            Field("cp_rank", "int", bind=False),
            Field("cp_group", "set_i32", bind=False),
            Field("use_kv_cache", "bool", init="true", bind=False),
            Field("skip_attn", "bool", bind=False),
            Field("fuses_dsv4_inv_rope_fp8_quant", "bool", bind=False),
            Field("attention_chunk_size", "optional_i64"),
            Field("skip_softmax_threshold_scale_factor_prefill", "double"),
            Field("skip_softmax_threshold_scale_factor_decode", "double"),
            Field("sage_attn_num_elts_per_blk_q", "i64"),
            Field("sage_attn_num_elts_per_blk_k", "i64"),
            Field("sage_attn_num_elts_per_blk_v", "i64"),
            Field("sage_attn_qk_int8", "bool"),
        ),
        static_config=True,
    ),
    FieldGroup(
        "MLA static op arguments.",
        (
            Field("q_lora_rank", "optional_i64"),
            Field("kv_lora_rank", "optional_i64"),
            Field("qk_nope_head_dim", "optional_i64"),
            Field("qk_rope_head_dim", "optional_i64"),
            Field("v_head_dim", "optional_i64"),
            Field("rope_append", "optional_bool"),
            Field("spec_decoding_target_max_draft_tokens", "optional_i64"),
        ),
    ),
    FieldGroup(
        "Phase-local input/output tensors and masks.",
        (
            Field("workspace", "tensor", data_ptr="void*"),
            Field("output", "tensor", data_ptr="void*"),
            Field("output_sf", "optional_tensor", data_ptr="void*"),
            Field("qkv_or_q", "tensor", data_ptr="T const*"),
            Field("k", "optional_tensor", context_data_ptr="T const*"),
            Field("v", "optional_tensor", context_data_ptr="T const*"),
            Field("v_stride_in_bytes", "i64", bind=False, context=True),
            Field("qkv_bias", "optional_tensor", bind=False, data_ptr="T const*"),
            Field("attention_mask", "optional_tensor", bind=False, data_ptr="bool const*"),
            Field(
                "attention_packed_mask",
                "optional_tensor",
                bind=False,
                context_data_ptr="uint32_t const*",
            ),
        ),
    ),
    FieldGroup(
        "Sequence lengths, windows, and paged-KV metadata.",
        (
            Field("sequence_length", "tensor", data_ptr="int32_t const*"),
            Field(
                "host_past_key_value_lengths",
                "tensor",
                generation_data_ptr="int32_t const*",
            ),
            Field("total_kv_len", "i32", enqueue=True),
            Field("context_lengths", "tensor", data_ptr="int32_t const*"),
            Field("host_context_lengths", "tensor", data_ptr="int32_t const*"),
            Field("max_context_q_len_override", "optional_i64"),
            Field(
                "kv_cache_block_offsets",
                "optional_tensor",
                data_ptr="kernels::KVBlockArray::DataType*",
            ),
            Field("max_blocks_per_sequence", "i32", bind=False, enqueue=True),
            Field("host_kv_cache_pool_pointers", "optional_tensor"),
            Field("host_primary_pool_pointer", "optional_tensor", bind=False, data_ptr="void*"),
            Field("host_secondary_pool_pointer", "optional_tensor", bind=False, data_ptr="void*"),
            Field(
                "host_primary_block_scale_pool_pointer",
                "optional_tensor",
                bind=False,
                data_ptr="void*",
            ),
            Field(
                "host_secondary_block_scale_pool_pointer",
                "optional_tensor",
                bind=False,
                data_ptr="void*",
            ),
            Field("host_kv_cache_pool_mapping", "optional_tensor"),
            Field(
                "cache_indirection",
                "optional_tensor",
                generation_data_ptr="int32_t const*",
            ),
            Field("max_attention_window_size", "i32", bind=False, enqueue=True),
            Field("cyclic_attention_window_size", "i32", bind=False, enqueue=True),
            Field("max_cyclic_attention_window_size", "i32", bind=False, enqueue=True),
            Field("can_use_one_more_block", "bool", bind=False, enqueue=True),
            Field("sink_token_length", "i32", bind=False, enqueue=True),
            Field("key_value_cache", "optional_tensor", bind=False, data_ptr="void*"),
        ),
    ),
    FieldGroup(
        "Quantization scales and output quantization.",
        (
            Field("kv_scale_orig_quant", "optional_tensor", data_ptr="float const*"),
            Field("kv_scale_quant_orig", "optional_tensor", data_ptr="float const*"),
            Field("out_scale", "optional_tensor", data_ptr="float const*"),
            Field("out_sf_scale", "optional_tensor", bind=False, data_ptr="float const*"),
        ),
    ),
    FieldGroup(
        "RoPE, ALiBi, and logn data.",
        (
            Field("rotary_inv_freq", "optional_tensor", data_ptr="float const*"),
            Field("rotary_cos_sin", "optional_tensor", data_ptr="float2 const*"),
            Field("alibi_slopes", "optional_tensor", bind=False, data_ptr="T const*"),
            Field("logn_scaling_ptr", "optional_tensor", bind=False, data_ptr="float const*"),
        ),
    ),
    FieldGroup(
        "MLA input/cache data.",
        (
            Field("latent_cache", "optional_tensor", ptr_type="T const*"),
            Field("q_pe", "optional_tensor", ptr_type="T*"),
            Field("block_ids_per_seq", "optional_tensor", ptr_type="int const*"),
            Field(
                "mla_param",
                "optional_tensor",
                bind=False,
                context_data_ptr="kernels::MlaParams<T>*",
            ),
        ),
    ),
    FieldGroup(
        "MRoPE and Helix position data.",
        (
            Field("mrope_rotary_cos_sin", "optional_tensor", context_data_ptr="float2 const*"),
            Field(
                "mrope_position_deltas",
                "optional_tensor",
                generation_data_ptr="int32_t const*",
            ),
            Field(
                "helix_position_offsets",
                "optional_tensor",
                context_data_ptr="int32_t const*",
                generation_data_ptr="int32_t const*",
            ),
            Field(
                "helix_is_inactive_rank",
                "optional_tensor",
                context_data_ptr="bool const*",
                generation_data_ptr="bool const*",
            ),
        ),
    ),
    FieldGroup(
        "Context chunking, Helix reduction, and softmax statistics.",
        (
            Field("softmax_stats_tensor", "optional_tensor", data_ptr="float2*"),
            Field("runtime_perf_knobs", "optional_tensor", bind=False, data_ptr="int64_t const*"),
        ),
    ),
    FieldGroup(
        "Speculative decoding masks and offsets.",
        (
            Field(
                "spec_decoding_generation_lengths",
                "optional_tensor",
                generation_data_ptr="int32_t const*",
            ),
            Field(
                "spec_decoding_position_offsets_for_cpp",
                "optional_tensor",
                generation_data_ptr="int32_t const*",
            ),
            Field(
                "spec_decoding_packed_mask",
                "optional_tensor",
                generation_data_ptr="int32_t const*",
            ),
            Field(
                "spec_decoding_bl_tree_mask_offset",
                "optional_tensor",
                generation_data_ptr="int64_t*",
            ),
            Field(
                "spec_decoding_bl_tree_mask",
                "optional_tensor",
                generation_data_ptr="uint32_t*",
            ),
            Field(
                "spec_bl_tree_first_sparse_mask_offset_kv",
                "optional_tensor",
                generation_data_ptr="int32_t*",
            ),
            Field(
                "spec_decoding_mask",
                "optional_tensor",
                bind=False,
                generation_data_ptr="bool const*",
            ),
            Field(
                "spec_decoding_is_generation_length_variable",
                "bool",
                bind=False,
                generation=True,
            ),
            Field(
                "spec_decoding_max_generation_length",
                "i32",
                bind=False,
                generation=True,
                generation_init="1",
            ),
        ),
    ),
    FieldGroup(
        "Attention sinks.",
        (Field("attention_sinks", "optional_tensor", data_ptr="float const*"),),
    ),
    FieldGroup(
        "Sparse attention, sparse MLA, and SageAttention runtime data.",
        (
            Field("sparse_kv_indices", "optional_tensor", ptr_type="int32_t*"),
            Field("sparse_kv_offsets", "optional_tensor", ptr_type="int32_t*"),
            Field("sparse_attn_indices", "optional_tensor", ptr_type="int32_t*"),
            Field("sparse_attn_offsets", "optional_tensor", ptr_type="int32_t*"),
            Field("sparse_attn_indices_block_size", "i64"),
            Field("num_sparse_topk", "i32"),
            Field("sparse_attn_kv_lens", "optional_tensor", ptr_type="int32_t*"),
            Field("sage_attn_sfs_q", "optional_tensor", bind=False, data_ptr="float const*"),
            Field("sage_attn_sfs_k", "optional_tensor", bind=False, data_ptr="float const*"),
            Field("sage_attn_sfs_v", "optional_tensor", bind=False, data_ptr="float const*"),
        ),
    ),
    FieldGroup(
        "Packed-varlen context boundaries and scheduler state.",
        (
            Field("cu_q_seqlens", "optional_tensor", context_data_ptr="int32_t const*"),
            Field("cu_kv_seqlens", "optional_tensor", context_data_ptr="int32_t const*"),
            Field("fmha_scheduler_counter", "optional_tensor", ptr_type="int32_t*"),
            Field("attention_mask_stride", "i32", bind=False, generation=True),
            Field("semaphores", "optional_tensor", bind=False, data_ptr="int32_t*"),
        ),
    ),
    FieldGroup(
        "MLA scales, quantized-Q buffers, and FlashMLA metadata.",
        (
            Field("mla_bmm1_scale", "optional_tensor", ptr_type="float*"),
            Field("mla_bmm2_scale", "optional_tensor", ptr_type="float*"),
            Field("quant_q_buffer", "optional_tensor", ptr_type="void*"),
            Field("flash_mla_tile_scheduler_metadata", "optional_tensor", ptr_type="int*"),
            Field("flash_mla_num_splits", "optional_tensor", ptr_type="int*"),
        ),
    ),
    FieldGroup(
        "TRTLLM-Gen JIT warmup.",
        (Field("trtllm_gen_jit_warmup", "bool", enqueue=True),),
    ),
    FieldGroup(
        "Auxiliary KV cache pools.",
        (Field("aux_kv_cache_pool_ptr", "optional_i64"),),
    ),
    FieldGroup(
        "Cross attention.",
        (
            Field("is_cross", "bool"),
            Field("cross_kv", "optional_tensor", context_data_ptr="T const*"),
            Field("cross_kv_length", "i32", bind=False, context=True),
            Field("num_encoder_tokens", "i32", bind=False, context=True),
            Field("relative_attention_bias", "optional_tensor", data_ptr="T const*"),
            Field("relative_attention_bias_stride", "int", bind=False, enqueue=True),
            Field(
                "encoder_input_lengths", "optional_tensor", bind=False, data_ptr="int32_t const*"
            ),
        ),
    ),
    FieldGroup(
        "DeepSeek-V4 FP8-Q/epilogue fusion.",
        (
            Field("quant_scale_qkv", "optional_tensor", ptr_type="float const*"),
            Field("dsv4_inv_rope_cos_sin_cache", "optional_tensor", ptr_type="float const*"),
            Field("enable_dsv4_epilogue_fusion", "bool"),
        ),
    ),
)


def _generated_banner(comment: str) -> str:
    return (
        f"{comment} Generated by scripts/generate_fmha_params.py. Do not edit.\n"
        f"{comment} Source table: FMHA_PARAM_GROUPS in scripts/generate_fmha_params.py.\n"
    )


def _macro_arg(value: str | None) -> str:
    return "" if value is None else value


def _macro_string(value: str | None) -> str:
    return '""' if value is None else f'"{value}"'


def _type_spec(field: Field) -> TypeSpec:
    return TYPE_SPECS[field.type]


def _is_bound(group: FieldGroup, field: Field) -> bool:
    return group.bind if field.bind is None else field.bind


def _field_cpp_default(field: Field, spec: TypeSpec) -> str | None:
    return field.init if field.init is not None else spec.cpp_default


def _tensor_ptr_type(field: Field, attr: str) -> str | None:
    ptr_type = getattr(field, attr)
    if ptr_type is None:
        return None
    if not _type_spec(field).is_tensor:
        raise ValueError(
            f"Field {field.name} uses {attr} but type={field.type!r} is not tensor-like"
        )
    return ptr_type


def _enqueue_cpp_type(field: Field, phase: str) -> str | None:
    ptr_type = _tensor_ptr_type(field, f"{phase}_data_ptr" if phase else "data_ptr")
    if ptr_type is not None:
        return ptr_type
    enabled = getattr(field, phase if phase else "enqueue")
    return _type_spec(field).cpp_type if enabled else None


def _enqueue_cpp_default(field: Field, phase: str) -> str | None:
    if _tensor_ptr_type(field, f"{phase}_data_ptr" if phase else "data_ptr") is not None:
        return "nullptr"
    init = getattr(field, f"{phase}_init" if phase else "enqueue_init")
    return init if init is not None else _type_spec(field).cpp_default


def _method_name(field: Field) -> str:
    return "get" + "".join(part.capitalize() for part in field.name.split("_"))


def _field_ptr_type(field: Field) -> str | None:
    ptr_types = {
        ptr_type
        for attr in ("data_ptr", "context_data_ptr", "generation_data_ptr", "ptr_type")
        if (ptr_type := _tensor_ptr_type(field, attr)) is not None
    }
    if not ptr_types:
        return None
    if len(ptr_types) != 1:
        raise ValueError(f"Field {field.name} has ambiguous pointer types: {sorted(ptr_types)}")
    return next(iter(ptr_types))


def _uses_template_type(ptr_type: str) -> bool:
    return ptr_type.strip() in {"T*", "T const*", "const T*"} or "<T>" in ptr_type


def _ptr_pointee_type(ptr_type: str) -> str:
    if not ptr_type.endswith("*"):
        raise ValueError(f"Pointer getter type must end with '*': {ptr_type}")
    pointee = ptr_type[:-1].strip()
    if pointee.startswith("const "):
        pointee = pointee[len("const ") :].strip()
    if pointee.endswith(" const"):
        pointee = pointee[: -len(" const")].strip()
    return pointee


def _tensor_data_ptr_expr(ptr_type: str, tensor_expr: str) -> str:
    pointee = _ptr_pointee_type(ptr_type)
    if pointee == "void":
        return f"{tensor_expr}.data_ptr()"
    if pointee in {
        "bool",
        "double",
        "float",
        "int",
        "int32_t",
        "int64_t",
        "uint32_t",
        "uint64_t",
    }:
        return f"{tensor_expr}.data_ptr<{pointee}>()"
    return f"static_cast<{ptr_type}>({tensor_expr}.data_ptr())"


def _render_fmha_param_ptr_getter_body(lines: list[str], field: Field, ptr_type: str) -> None:
    if field.type == "optional_tensor":
        lines.extend(
            [
                f"    return {field.name}.has_value()",
                f"        ? {_tensor_data_ptr_expr(ptr_type, f'{field.name}.value()')}",
                "        : nullptr;",
            ]
        )
    else:
        lines.append(f"    return {_tensor_data_ptr_expr(ptr_type, field.name)};")


def _render_special_fmha_param_ptr_getter(lines: list[str], field: Field) -> bool:
    if field.name == "qkv_or_q":
        lines.extend(
            [
                "template <typename T>",
                "T* getQkvOrQ(int64_t offset) const",
                "{",
                "    return static_cast<T*>(qkv_or_q.slice(0, offset).data_ptr());",
                "}",
            ]
        )
        return True
    if field.name == "output":
        lines.extend(
            [
                "void* getOutput(int64_t offset) const",
                "{",
                "    return output.slice(0, offset).data_ptr();",
                "}",
            ]
        )
        return True
    if field.name == "sequence_length":
        lines.extend(
            [
                "int const* getSequenceLength(int64_t offset) const",
                "{",
                "    return sequence_length.slice(0, offset).data_ptr<int>();",
                "}",
            ]
        )
        return True
    if field.name == "context_lengths":
        lines.extend(
            [
                "int const* getContextLengths(int64_t offset) const",
                "{",
                "    return context_lengths.slice(0, offset).data_ptr<int>();",
                "}",
            ]
        )
        return True
    if field.name == "kv_cache_block_offsets":
        lines.extend(
            [
                "kernels::KVBlockArray::DataType* getKvCacheBlockOffsets(int32_t poolIndex, int64_t seqOffset) const",
                "{",
                "    return kv_cache_block_offsets.has_value()",
                "        ? static_cast<kernels::KVBlockArray::DataType*>(",
                "              kv_cache_block_offsets.value().index({poolIndex, seqOffset}).data_ptr())",
                "        : nullptr;",
                "}",
            ]
        )
        return True
    if field.name == "k":
        lines.extend(
            [
                "template <typename T>",
                "T* getK(int64_t offset) const",
                "{",
                "    return k.has_value() ? static_cast<T*>(k.value().slice(0, offset).data_ptr()) : nullptr;",
                "}",
            ]
        )
        return True
    if field.name == "v":
        lines.extend(
            [
                "template <typename T>",
                "T* getV(int64_t offset) const",
                "{",
                "    return v.has_value() ? static_cast<T*>(v.value().slice(0, offset).data_ptr()) : nullptr;",
                "}",
            ]
        )
        return True
    return False


def _render_fmha_param_fields(lines: list[str]) -> None:
    lines.extend(
        [
            "#if defined(TRTLLM_FMHA_PARAM_FIELD)",
            "",
            "// Public thop FmhaParams fields.",
        ]
    )
    for group in FMHA_PARAM_GROUPS:
        fields = tuple(field for field in group.fields if _is_bound(group, field))
        if not fields:
            continue
        lines.extend(["", f"// {group.comment}"])
        for field in fields:
            spec = _type_spec(field)
            field_default = _field_cpp_default(field, spec)
            cpp_default = None if field_default is None else f"= {field_default}"
            py_default = None if spec.py_default is None else f"= {spec.py_default}"
            lines.extend(
                [
                    "TRTLLM_FMHA_PARAM_FIELD(",
                    f"    {field.name},",
                    f"    {spec.cpp_type},",
                    f"    {_macro_string(spec.py_type)},",
                    f"    {_macro_arg(cpp_default)},",
                    f"    {_macro_string(py_default)})",
                ]
            )
    lines.append("#endif")


def _render_fmha_param_getters(lines: list[str]) -> None:
    lines.extend(
        [
            "",
            "#if defined(TRTLLM_FMHA_PARAM_GETTERS)",
            "",
            "// Public thop FmhaParams getters.",
        ]
    )
    for group in FMHA_PARAM_GROUPS:
        fields = tuple(
            field
            for field in group.fields
            if _is_bound(group, field) and _field_ptr_type(field) is not None
        )
        if not fields:
            continue
        lines.extend(["", f"// {group.comment}"])
        for field in fields:
            ptr_type = _field_ptr_type(field)
            if ptr_type is None:
                raise ValueError(f"Field {field.name} has no pointer getter type")
            if _render_special_fmha_param_ptr_getter(lines, field):
                continue
            if _uses_template_type(ptr_type):
                lines.append("template <typename T>")
            lines.extend(
                [
                    f"{ptr_type} {_method_name(field)}() const",
                    "{",
                ]
            )
            _render_fmha_param_ptr_getter_body(lines, field, ptr_type)
            lines.append("}")
    lines.extend(
        r"""\
int32_t getMaxHostPastKeyValueLength(int64_t seqOffset, int64_t numSeqs) const
{
    return host_past_key_value_lengths.slice(0, seqOffset, seqOffset + numSeqs).max().item<int32_t>();
}
int32_t getMaxHostContextLength(int64_t seqOffset, int64_t numSeqs) const
{
    return host_context_lengths.slice(0, seqOffset, seqOffset + numSeqs).max().item<int32_t>();
}
int getCacheIndirectionWindowSize(int defaultValue) const
{
    return cache_indirection.has_value()
        ? static_cast<int>(cache_indirection.value().size(2))
        : defaultValue;
}
bool hasKvCache() const
{
    return kv_cache_block_offsets.has_value() && host_kv_cache_pool_pointers.has_value()
        && host_kv_cache_pool_mapping.has_value();
}
torch::Tensor const& getHostKvCachePoolPointers() const
{
    return host_kv_cache_pool_pointers.value();
}
int getMaxBlocksPerSequence() const
{
    return kv_cache_block_offsets.has_value()
        ? static_cast<int>(kv_cache_block_offsets.value().size(-1))
        : 0;
}
int32_t getKvCachePoolIndex(int64_t layerIdx) const
{
    return host_kv_cache_pool_mapping.has_value()
        ? host_kv_cache_pool_mapping.value().index({layerIdx, 0}).item<int32_t>()
        : 0;
}
int32_t getLayerIdxInCachePool(int64_t layerIdx) const
{
    return host_kv_cache_pool_mapping.has_value()
        ? host_kv_cache_pool_mapping.value().index({layerIdx, 1}).item<int32_t>()
        : 0;
}
int64_t getMlaLayerNum() const
{
    return host_kv_cache_pool_mapping.has_value() ? host_kv_cache_pool_mapping.value().size(0) : 0;
}
int64_t getSparseAttnIndicesStride() const
{
    return sparse_attn_indices.has_value() ? sparse_attn_indices.value().size(-1) : 0;
}
int32_t getCrossKvNumTokens() const
{
    return cross_kv.has_value() ? static_cast<int32_t>(cross_kv.value().size(0)) : 0;
}
char* getSparseKvCachePool(int32_t poolIndex) const
{
    return host_kv_cache_pool_pointers.has_value()
        ? reinterpret_cast<char*>(
              host_kv_cache_pool_pointers.value().index({poolIndex, 0}).item<int64_t>())
        : nullptr;
}
""".splitlines()
    )
    lines.append("#endif")


def _render_enqueue_fields(
    lines: list[str],
    macro_name: str,
    title: str,
    phase: str,
) -> None:
    lines.extend(["", f"#if defined({macro_name})", "", f"// {title}"])
    for group in FMHA_PARAM_GROUPS:
        fields = tuple(
            field for field in group.fields if _enqueue_cpp_type(field, phase) is not None
        )
        if not fields:
            continue
        lines.extend(["", f"// {group.comment}"])
        for field in fields:
            cpp_type = _enqueue_cpp_type(field, phase)
            if cpp_type is None:
                raise ValueError(f"Field {field.name} has no {macro_name} type")
            cpp_default = _enqueue_cpp_default(field, phase)
            default_arg = None if cpp_default is None else f"= {cpp_default}"
            lines.extend(
                [
                    f"{macro_name}(",
                    f"    {field.name},",
                    f"    {cpp_type},",
                    f"    {_macro_arg(default_arg)})",
                ]
            )
    lines.append("#endif")


def _render_static_config_fields(lines: list[str]) -> None:
    lines.extend(
        [
            "",
            "#if defined(TRTLLM_ATTENTION_STATIC_CONFIG_FIELD)",
            "",
            "// AttentionStaticConfig fields.",
        ]
    )
    for group in FMHA_PARAM_GROUPS:
        fields = group.fields if group.static_config else ()
        if not fields:
            continue
        lines.extend(["", f"// {group.comment}"])
        for field in fields:
            spec = _type_spec(field)
            default_arg = None if field.init is None else f"= {field.init}"
            lines.extend(
                [
                    "TRTLLM_ATTENTION_STATIC_CONFIG_FIELD(",
                    f"    {spec.cpp_type},",
                    f"    {field.name},",
                    f"    {_macro_arg(default_arg)})",
                ]
            )
    lines.append("#endif")


def render_field_list() -> str:
    lines = [
        _generated_banner("//").rstrip(),
        "//",
        "// Define one or more of the field macros before including this file:",
        "//   TRTLLM_FMHA_PARAM_FIELD",
        "//   TRTLLM_FMHA_PARAM_GETTERS",
        "//   TRTLLM_ATTENTION_STATIC_CONFIG_FIELD",
        "//   TRTLLM_FMHA_ENQUEUE_PARAM_FIELD",
        "//   TRTLLM_FMHA_ENQUEUE_CONTEXT_PARAM_FIELD",
        "//   TRTLLM_FMHA_ENQUEUE_GENERATION_PARAM_FIELD",
        "//",
        "// Example C++ struct field expansion:",
        "//   #define TRTLLM_FMHA_PARAM_FIELD(name, cpp_type, py_type, cpp_default, py_default) \\",
        "//       cpp_type name cpp_default;",
        '//   #include "fmha_params_fields.inc"',
        "//   #undef TRTLLM_FMHA_PARAM_FIELD",
        "//",
        "// Example nanobind expansion:",
        "//   #define TRTLLM_FMHA_PARAM_FIELD(name, cpp_type, py_type, cpp_default, py_default) \\",
        "//       .def_rw(#name, &torch_ext::FmhaParams::name)",
        '//   #include "fmha_params_fields.inc"',
        "//   #undef TRTLLM_FMHA_PARAM_FIELD",
        "",
        "#if !defined(TRTLLM_FMHA_PARAM_FIELD) && \\",
        "    !defined(TRTLLM_FMHA_PARAM_GETTERS) && \\",
        "    !defined(TRTLLM_ATTENTION_STATIC_CONFIG_FIELD) && \\",
        "    !defined(TRTLLM_FMHA_ENQUEUE_PARAM_FIELD) && \\",
        "    !defined(TRTLLM_FMHA_ENQUEUE_CONTEXT_PARAM_FIELD) && \\",
        "    !defined(TRTLLM_FMHA_ENQUEUE_GENERATION_PARAM_FIELD)",
        '# error "Define an FMHA parameter field macro before including fmha_params_fields.inc"',
        "#endif",
    ]
    _render_fmha_param_fields(lines)
    _render_fmha_param_getters(lines)
    _render_static_config_fields(lines)
    _render_enqueue_fields(
        lines,
        "TRTLLM_FMHA_ENQUEUE_PARAM_FIELD",
        "Shared AttentionOp::EnqueueParams fields.",
        "",
    )
    _render_enqueue_fields(
        lines,
        "TRTLLM_FMHA_ENQUEUE_CONTEXT_PARAM_FIELD",
        "AttentionOp::EnqueueContextParams fields.",
        "context",
    )
    _render_enqueue_fields(
        lines,
        "TRTLLM_FMHA_ENQUEUE_GENERATION_PARAM_FIELD",
        "AttentionOp::EnqueueGenerationParams fields.",
        "generation",
    )
    return "\n".join(lines) + "\n"


GENERATED_FILES = {
    "fmha_params_fields.inc": render_field_list,
}


def _write_if_changed(path: Path, content: str) -> bool:
    old_content = None
    if path.exists():
        old_content = path.read_text()
    if old_content == content:
        return False
    path.write_text(content)
    return True


def _check_file(path: Path, content: str) -> bool:
    try:
        old_content = path.read_text()
    except FileNotFoundError:
        print(f"{path}: missing generated file", file=sys.stderr)
        return False
    if old_content == content:
        return True
    diff = difflib.unified_diff(
        old_content.splitlines(keepends=True),
        content.splitlines(keepends=True),
        fromfile=str(path),
        tofile=f"{path} (expected)",
    )
    sys.stderr.writelines(diff)
    return False


def generate(out_dir: Path, check: bool) -> int:
    ok = True
    if not check:
        out_dir.mkdir(parents=True, exist_ok=True)
    for filename, render in GENERATED_FILES.items():
        content = render()
        path = out_dir / filename
        if check:
            ok = _check_file(path, content) and ok
        else:
            changed = _write_if_changed(path, content)
            action = "updated" if changed else "unchanged"
            print(f"{action}: {path}")
    return 0 if ok else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory that receives the generated FMHA parameter include.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare generated artifacts against --out-dir without writing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return generate(args.out_dir, args.check)


if __name__ == "__main__":
    sys.exit(main())
