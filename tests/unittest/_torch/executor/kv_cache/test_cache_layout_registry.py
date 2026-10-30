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
"""The layout adapters' registry names classes that exist, every adapter module is registered, and
no cache manager it names imports the layout package. Read off the sources, so no manager module
is imported."""

import ast
from pathlib import Path

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.cache_layout import adapter

pytestmark = pytest.mark.cpu_only

PACKAGE = Path(adapter.__file__).resolve().parent
ROOT = PACKAGE.parents[4]  # the directory holding the tensorrt_llm package


def _registry():
    """The repository's entries: manager class name -> (its module's source, adapter entry)."""
    return {
        key: entry for key, entry in adapter._ADAPTERS.items() if key.startswith("tensorrt_llm.")
    }


def _source(module: str) -> Path:
    return ROOT / (module.replace(".", "/") + ".py")


def _classes(path: Path) -> set[str]:
    return {n.name for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)}


def test_every_registered_manager_class_and_adapter_exists():
    for key, entry in _registry().items():
        module, _, name = key.rpartition(".")
        assert name in _classes(_source(module)), key
        if isinstance(entry, tuple):
            adapter_module, adapter_name = entry
            assert adapter_name in _classes(PACKAGE / "adapters" / f"{adapter_module}.py"), entry


def test_every_adapter_module_is_registered():
    registered = {entry[0] for entry in _registry().values() if isinstance(entry, tuple)}
    modules = {p.stem for p in (PACKAGE / "adapters").glob("*.py") if p.stem != "__init__"}
    assert modules == registered


def test_no_registered_cache_manager_imports_the_layout_package():
    for key in _registry():
        module = key.rpartition(".")[0]
        for node in ast.walk(ast.parse(_source(module).read_text())):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [alias.name for alias in node.names]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            assert not any("cache_layout" in name.split(".") for name in names), (
                module,
                node.lineno,
            )
