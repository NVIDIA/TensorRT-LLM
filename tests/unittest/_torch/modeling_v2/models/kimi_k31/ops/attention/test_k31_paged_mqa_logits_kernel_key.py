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
"""The K3.1 FP8 paged MQA logits entry finds the kernel the shared runner compiled for it.

The entry compiles through the registered op's runner (``CuteDSLPagedMQALogitsRunner._compile``)
and then indexes the runner's ``kernel_cache`` with a key of its own. ``_compile`` stores the
kernel under the tuple of its arguments, the opt-in ``kv_evict_first`` appended only when it is
set (the default kernel keeps the key the other callers build), so the entry's key must be exactly
the default kernel's: one that differs misses the cache on every call, and the entry compiles again
and then raises ``KeyError``. Both keys are read from the sources, so this runs without CuTe DSL or
a GPU.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_REPO = next(p for p in Path(__file__).resolve().parents if (p / "tensorrt_llm").is_dir())
_RUNNER_FILE = _REPO / "tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py"
_ENTRY_FILE = (
    _REPO
    / "tensorrt_llm/_torch/_experimental/modeling_v2/models/kimi_k31/ops/attention"
    / "cute_dsl_fp8_paged_mqa_logits.py"
)


def _function(tree: ast.AST, class_name: str, name: str) -> ast.FunctionDef:
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _key_elements(function: ast.FunctionDef) -> tuple[list[str], list[str]]:
    """The source of each element ``key`` holds after the function's assignments to it, which
    are ``key = (...)``, ``key = key + (...)`` and ``key = (...) + ((x,) if x else ())``: the
    elements always there, and the ones appended only when set."""
    elements, optional = [], []
    for node in ast.walk(function):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "key"
        ):
            continue
        value = node.value
        if isinstance(value, ast.BinOp) and isinstance(value.right, ast.IfExp):
            assert isinstance(value.op, ast.Add), ast.unparse(node)
            when_set, when_not = value.right.body, value.right.orelse
            assert isinstance(when_set, ast.Tuple) and len(when_set.elts) == 1, ast.unparse(node)
            assert isinstance(when_not, ast.Tuple) and not when_not.elts, ast.unparse(node)
            assert ast.unparse(value.right.test) == ast.unparse(when_set.elts[0]), ast.unparse(node)
            optional.append(ast.unparse(when_set.elts[0]))
            value = value.left
        if isinstance(value, ast.BinOp) and isinstance(value.left, ast.Name):
            assert value.left.id == "key" and isinstance(value.op, ast.Add), ast.unparse(node)
            value = value.right
        assert isinstance(value, ast.Tuple), ast.unparse(node)
        elements += [ast.unparse(element) for element in value.elts]
    return elements, optional


def test_the_entry_key_has_every_compile_argument():
    compile_ = _function(
        ast.parse(_RUNNER_FILE.read_text()), "CuteDSLPagedMQALogitsRunner", "_compile"
    )
    parameters = [arg.arg for arg in compile_.args.args[1:]]  # without cls
    # The runner's own cache key is all of _compile's arguments, in order; kv_evict_first, the
    # last, only when set.
    required, optional = _key_elements(compile_)
    assert required + optional == parameters
    assert optional == ["kv_evict_first"]
    entry_key, entry_optional = _key_elements(
        _function(ast.parse(_ENTRY_FILE.read_text()), "CuteDslFp8PagedMqaLogits", "__call__")
    )
    # The entry runs the default kernel (no L2 evict-first hint): its key is the required part.
    assert not entry_optional
    assert len(entry_key) == len(required), dict(zip(required, entry_key))
