# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path

_FORWARDED_ARGUMENTS = [
    "kv_norm_weight",
    "kv_norm_eps",
    "precomputed_cu_seqlens",
    "precomputed_fmha_scheduler",
    "kv_only",
    "kv_done_elsewhere",
    "quant_scale_qkv",
]


def _function(class_node: ast.ClassDef, name: str) -> ast.FunctionDef:
    return next(
        node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_mla_rope_generation_forwards_current_kernel_controls():
    """The token-major wrapper must not drop current-main MLA controls."""
    repo_root = Path(__file__).resolve().parents[5]
    source = repo_root / "tensorrt_llm/_torch/attention_backend/trtllm.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "TrtllmAttention"
    )
    public = _function(class_node, "mla_rope_generation")
    helper = _function(class_node, "_mla_rope_generation_impl")
    helper_call = next(
        node
        for node in ast.walk(public)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_mla_rope_generation_impl"
    )

    helper_args = [argument.arg for argument in helper.args.args]
    forwarded = [argument.id for argument in helper_call.args[-7:]]
    assert helper_args[-7:] == _FORWARDED_ARGUMENTS
    assert forwarded == _FORWARDED_ARGUMENTS
