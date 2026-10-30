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
"""The layout adapters' registry names classes that exist, registers every adapter class, and no
cache manager it names imports the layout package; and how ``adapter_for`` picks an adapter. The
registry is read off the sources: the only manager code imported is ``kv_cache_manager_v2``, which
the registry module imports."""

import ast
from pathlib import Path

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.cache_layout import adapter
from tensorrt_llm._torch.pyexecutor.kv_cache.cache_layout.adapter import (
    CacheLayoutAdapter,
    adapter_for,
)

pytestmark = pytest.mark.cpu_only

PACKAGE = Path(adapter.__file__).resolve().parent
ROOT = PACKAGE.parents[4]  # the directory holding the tensorrt_llm package
# Adapter classes under ``adapters`` that no manager class takes directly, only their subclasses.
UNREGISTERED_ADAPTERS: frozenset[tuple[str, str]] = frozenset()


def _registry():
    """The repository's entries: manager class name -> its adapter entry."""
    return {
        key: entry for key, entry in adapter._ADAPTERS.items() if key.startswith("tensorrt_llm.")
    }


def _source(module: str) -> Path:
    return ROOT / (module.replace(".", "/") + ".py")


def _classes(path: Path) -> set[str]:
    return {n.name for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)}


def _qualname(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def test_every_registered_manager_class_and_adapter_exists():
    for key, entry in _registry().items():
        module, _, name = key.rpartition(".")
        assert name in _classes(_source(module)), key
        if isinstance(entry, tuple):
            adapter_module, adapter_name = entry
            assert adapter_name in _classes(PACKAGE / "adapters" / f"{adapter_module}.py"), entry


def test_every_adapter_class_is_registered():
    """An adapter class no entry names would silently leave its manager to its parent's."""
    registered = {entry for entry in _registry().values() if isinstance(entry, tuple)}
    defined = {
        (path.stem, name)
        for path in (PACKAGE / "adapters").glob("*.py")
        if path.stem != "__init__"
        for name in _classes(path)
    }
    assert defined - UNREGISTERED_ADAPTERS == registered


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


def test_an_installed_policy_yields_to_the_adapters_of_managers_deriving_from_it(monkeypatch):
    """A layout policy installed on a manager not deriving from it comes before the manager's
    class; a manager deriving from it, as every policy instance and its subclasses do, resolves
    through its MRO, so a subclass registering its own adapter gets it."""

    class Policy:
        pass

    class PolicyAdapter(CacheLayoutAdapter):
        pass

    class Registered(Policy):
        pass

    class RegisteredAdapter(PolicyAdapter):
        pass

    class Unregistered(Policy):
        pass

    class Plain:
        pass

    class PlainAdapter(CacheLayoutAdapter):
        pass

    monkeypatch.setitem(adapter._ADAPTERS, _qualname(Policy), PolicyAdapter)
    monkeypatch.setitem(adapter._ADAPTERS, _qualname(Registered), RegisteredAdapter)
    monkeypatch.setitem(adapter._ADAPTERS, _qualname(Plain), PlainAdapter)

    def installed(manager):
        manager._attention_cache_layout_policy = Policy
        return manager

    assert type(adapter_for(installed(Registered()))) is RegisteredAdapter
    assert type(adapter_for(installed(Unregistered()))) is PolicyAdapter
    assert type(adapter_for(installed(Plain()))) is PolicyAdapter
    assert type(adapter_for(Plain())) is PlainAdapter
    assert adapter_for(object()) is None
