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
"""Guards how the PEP 562 ``__getattr__`` of the lazily loaded packages reports a broken submodule.

An ``AttributeError`` raised while a submodule is being imported is a bug inside that submodule.
``hasattr()``, ``getattr(..., default)`` and ``from package import name`` all treat an
``AttributeError`` from ``__getattr__`` as "no such name", so it must leave ``__getattr__`` as an
``ImportError`` chained to the original error. Otherwise ``hasattr`` returns False,
``getattr(..., default)`` returns the default, and ``from ... import`` imports the submodule a
second time and, for a name that is not itself a submodule, reports ``cannot import name``
without the cause.

The fault is injected with a ``sys.meta_path`` finder that raises when the target submodule is
imported, so it covers both ``importlib.import_module`` and ``from . import`` inside
``__getattr__``.
"""

import importlib
import importlib.abc
import sys

import pytest

pytestmark = pytest.mark.cpu_only

_MARKER = "injected-by-test_lazy_getattr_errors"

# (package, name accessed on it, submodule that the access imports)
CASES = [
    ("tensorrt_llm", "AutoConfig", "tensorrt_llm.models.automodel"),
    ("tensorrt_llm", "profiler", "tensorrt_llm.profiler"),
    (
        "tensorrt_llm._torch.models",
        "Starcoder2ForCausalLM",
        "tensorrt_llm._torch.models.modeling_starcoder2",
    ),
    (
        "tensorrt_llm._torch.models",
        "modeling_starcoder2",
        "tensorrt_llm._torch.models.modeling_starcoder2",
    ),
    ("tensorrt_llm.visual_gen", "VisualGenParams", "tensorrt_llm.visual_gen.params"),
    ("tensorrt_llm.visual_gen", "sparse_attention", "tensorrt_llm.visual_gen.sparse_attention"),
    (
        "tensorrt_llm._torch.pyexecutor.sampler",
        "TorchSampler",
        "tensorrt_llm._torch.pyexecutor.sampler.sampler",
    ),
    (
        "tensorrt_llm._torch.pyexecutor.sampler",
        "sampler_common",
        "tensorrt_llm._torch.pyexecutor.sampler.sampler_common",
    ),
    (
        "tensorrt_llm._torch.visual_gen.modules",
        "Attention",
        "tensorrt_llm._torch.visual_gen.attention.attention",
    ),
    ("tensorrt_llm.serve", "OpenAIServer", "tensorrt_llm.serve.openai_server"),
    (
        "tensorrt_llm.serve",
        "OpenAIDisaggServer",
        "tensorrt_llm.serve.openai_disagg_server",
    ),
]
_IDS = [f"{pkg}.{name}" for pkg, name, _ in CASES]


class _RaiseOnImport(importlib.abc.MetaPathFinder):
    def __init__(self, fullname: str):
        self.fullname = fullname
        self.attempts = 0

    def find_spec(self, name, path=None, target=None):
        if name != self.fullname:
            return None
        self.attempts += 1
        raise AttributeError(f"{_MARKER}: importing {name}")


@pytest.fixture
def broken_submodule(monkeypatch):
    """Make importing ``target`` raise AttributeError until the test ends; return (package, finder)."""

    def install(pkg_name: str, name: str, target: str):
        pkg = importlib.import_module(pkg_name)
        parent_name, _, leaf = target.rpartition(".")
        parent = importlib.import_module(parent_name)
        finder = _RaiseOnImport(target)
        monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
        # Edit the namespaces directly: hasattr() (used by monkeypatch.delattr) would call the
        # package's __getattr__ and re-import the target before the test starts.
        monkeypatch.delitem(sys.modules, target, raising=False)
        monkeypatch.delitem(vars(parent), leaf, raising=False)
        monkeypatch.delitem(vars(pkg), name, raising=False)
        assert target not in sys.modules and leaf not in vars(parent) and name not in vars(pkg)
        assert finder.attempts == 0
        return pkg, finder

    return install


def _assert_chained(excinfo, finder):
    assert isinstance(excinfo.value.__cause__, AttributeError), repr(excinfo.value)
    assert _MARKER in str(excinfo.value.__cause__)
    assert finder.attempts == 1


@pytest.mark.parametrize("pkg_name,name,target", CASES, ids=_IDS)
def test_from_import_reports_cause_once(broken_submodule, pkg_name, name, target):
    _, finder = broken_submodule(pkg_name, name, target)
    with pytest.raises(ImportError) as excinfo:
        exec(f"from {pkg_name} import {name}", {})
    _assert_chained(excinfo, finder)


@pytest.mark.parametrize("pkg_name,name,target", CASES, ids=_IDS)
@pytest.mark.parametrize("probe", ["attr", "hasattr", "getattr_default"])
def test_probe_does_not_hide_broken_submodule(broken_submodule, probe, pkg_name, name, target):
    pkg, finder = broken_submodule(pkg_name, name, target)
    with pytest.raises(ImportError) as excinfo:
        if probe == "attr":
            getattr(pkg, name)
        elif probe == "hasattr":
            hasattr(pkg, name)
        else:
            getattr(pkg, name, None)
    _assert_chained(excinfo, finder)
