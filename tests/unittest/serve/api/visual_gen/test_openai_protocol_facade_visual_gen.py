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
"""The VisualGen names of the ``tensorrt_llm.serve.openai_protocol`` facade.

The facade maps the eight VisualGen request and response models to
``tensorrt_llm.serve.api.visual_gen.protocol`` and loads that module only when
one of them is read. It never imports it, not even under ``TYPE_CHECKING``, so
that CI selection can run only the VisualGen tests for VisualGen-only changes.
This file lives under a ``visual_gen/`` path so that those runs include it; the
other facade checks are in
``tests/unittest/serve/openai_protocol/test_openai_protocol_facade.py``.
"""

import ast
import importlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

_FACADE_MODULE = "tensorrt_llm.serve.openai_protocol"
_FACADE_PATH = "tensorrt_llm/serve/openai_protocol/__init__.py"
_VISUAL_GEN_PACKAGE = "tensorrt_llm.serve.api.visual_gen"
_VISUAL_GEN_MODULE = "tensorrt_llm.serve.api.visual_gen.protocol"
_VISUAL_GEN_NAMES = (
    "ImageGenerationRequest",
    "ImageEditRequest",
    "ImageObject",
    "ImageGenerationResponse",
    "MediaReferenceItem",
    "VideoGenerationRequest",
    "VideoJob",
    "VideoJobList",
)
_SUBMODULES = (
    "common",
    "sampling",
    "tools",
    "models",
    "tokenize",
    "embeddings",
    "completions",
    "chat",
    "kv_cache",
    "responses",
    "control",
    "disagg",
)


def _run_in_fresh_interpreter(code: str) -> dict:
    """Run ``code`` in a new interpreter and return the JSON it prints last."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"the probe interpreter exited with {result.returncode}; stdout tail: "
        f"{result.stdout[-1000:]!r}; stderr tail: {result.stderr[-2000:]!r}"
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_visual_gen_names_map_to_their_module():
    facade = importlib.import_module(_FACADE_MODULE)
    visual_gen = importlib.import_module(_VISUAL_GEN_MODULE)
    problems = []
    for name in _VISUAL_GEN_NAMES:
        mapped = facade._NAME_TO_MODULE.get(name)
        same_object = getattr(facade, name, None) is getattr(visual_gen, name)
        if mapped != _VISUAL_GEN_MODULE or not same_object:
            problems.append(
                f"{_FACADE_MODULE} must re-export {name} lazily from {_VISUAL_GEN_MODULE}; "
                f"map '{name}' to that module in _NAME_TO_MODULE of {_FACADE_PATH}."
            )
    assert problems == [], "\n".join(problems)


def test_protocol_submodules_do_not_load_visual_gen():
    out = _run_in_fresh_interpreter(
        f"""
        import importlib, json, sys

        target = {_VISUAL_GEN_MODULE!r}
        loaded_by = []
        import tensorrt_llm.serve.openai_protocol
        if target in sys.modules:
            loaded_by.append("import tensorrt_llm.serve.openai_protocol")
        for sub in {list(_SUBMODULES)!r}:
            importlib.import_module(f"tensorrt_llm.serve.openai_protocol.{{sub}}")
            if target in sys.modules:
                loaded_by.append(f"import tensorrt_llm.serve.openai_protocol.{{sub}}")
        from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest
        if target in sys.modules:
            loaded_by.append("from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest")
        print(json.dumps({{"loaded_by": loaded_by}}))
        """
    )
    assert out["loaded_by"] == [], (
        f"{out['loaded_by'][0]} loaded {_VISUAL_GEN_MODULE}; only a lookup of a VisualGen "
        "name through the facade may load it."
    )


def _resolve(node: ast.ImportFrom, package: str) -> str:
    """The absolute module an ``ImportFrom`` names, relative imports resolved against ``package``."""
    if node.level == 0:
        return node.module or ""
    parts = package.split(".")
    base = parts[: len(parts) - (node.level - 1)]
    if node.module:
        base.append(node.module)
    return ".".join(base)


def _under_visual_gen(module: str) -> bool:
    return module == _VISUAL_GEN_PACKAGE or module.startswith(_VISUAL_GEN_PACKAGE + ".")


def test_facade_names_visual_gen_only_as_strings():
    facade = importlib.import_module(_FACADE_MODULE)
    tree = ast.parse(Path(facade.__file__).read_text())
    problems = []
    for node in ast.walk(tree):
        modules = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve(node, _FACADE_MODULE)
            modules = [resolved] + [f"{resolved}.{alias.name}" for alias in node.names]
        for module in modules:
            if _under_visual_gen(module):
                problems.append(
                    f"{_FACADE_PATH} imports {module} at line {node.lineno}. The facade may name "
                    "VisualGen modules only as strings in _NAME_TO_MODULE; an import, even under "
                    "TYPE_CHECKING, breaks the VisualGen import isolation that lets CI run only "
                    "VisualGen tests for VisualGen-only changes."
                )
                break
    table_values = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_NAME_TO_MODULE"
            for target in node.targets
        ):
            table_values = {
                id(constant)
                for constant in ast.walk(node.value)
                if isinstance(constant, ast.Constant)
            }
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and node.value == _VISUAL_GEN_MODULE
            and id(node) not in table_values
        ):
            problems.append(
                f"{_FACADE_PATH} names {_VISUAL_GEN_MODULE} at line {node.lineno} outside "
                "_NAME_TO_MODULE; the facade refers to the VisualGen module only as the value of "
                "a row of that table."
            )
    assert problems == [], "\n".join(problems)
