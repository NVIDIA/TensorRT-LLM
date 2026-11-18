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
"""Guards the lazy package-level exports of ``tensorrt_llm._torch.attention``.

The package root resolves the module-layer classes on attribute access
(PEP 562). These checks pin the contract: every exported name is the
defining submodule's own object, submodules resolve only through the import
system (unknown names, dotted ones included, raise ``AttributeError`` without
importing anything), a missing dependency inside a submodule surfaces as ``ModuleNotFoundError``, an
``AttributeError`` inside one surfaces as ``ImportError`` with its cause,
importing the package root or its ``backends`` sub-package loads no
module-layer module, and ``rotary_embedding`` (which the backends import
back) works as the first import of a process and loads none of the backends. The import-order checks run in
a fresh interpreter, because the pytest process has long since imported most
of the package.
"""

import importlib
import subprocess
import sys
import textwrap
import types

import pytest

pytestmark = pytest.mark.cpu_only

PKG = "tensorrt_llm._torch.attention"

EXPORTS = {
    "Attention": "attention",
    "CrossAttention": "cross_attention",
    "MLA": "mla",
    "QKNormRoPEAttention": "qk_norm_attention",
    "RotaryEmbedding": "rotary_embedding",
    "MRotaryEmbedding": "rotary_embedding",
}

# Module-layer modules that importing the ``backends`` sub-package must not load.
MODULE_LAYER = [
    f"{PKG}.attention",
    f"{PKG}.cross_attention",
    f"{PKG}.mla",
    f"{PKG}.qk_norm_attention",
    f"{PKG}.rotary_embedding",
]

BACKENDS = f"{PKG}.backends"

_SENTINEL = "CLOSURE-OK"


def _run_fresh(body: str, timeout: int = 600) -> None:
    script = textwrap.dedent(body) + f"\nprint({_SENTINEL!r})\n"
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0 and _SENTINEL in result.stdout, (
        f"rc={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.fixture
def pkg():
    return importlib.import_module(PKG)


@pytest.mark.parametrize("name", sorted(EXPORTS))
def test_export_is_defining_object_and_cached(pkg, name):
    defining = importlib.import_module(f"{PKG}.{EXPORTS[name]}")
    assert getattr(pkg, name) is getattr(defining, name)
    assert pkg.__dict__[name] is getattr(defining, name)


def test_all_and_dir_list_exports(pkg):
    assert sorted(pkg.__all__) == sorted(EXPORTS)
    assert sorted(pkg._LAZY_ATTRS.items()) == sorted(EXPORTS.items())
    assert set(EXPORTS) <= set(dir(pkg))


@pytest.mark.parametrize("submodule", ["backends", "kernels", "rotary_embedding", "attention"])
def test_imported_submodule_is_attribute(pkg, submodule):
    module = importlib.import_module(f"{PKG}.{submodule}")
    assert isinstance(module, types.ModuleType)
    assert getattr(pkg, submodule) is module


def test_from_import_submodule():
    from tensorrt_llm._torch.attention import attention, mla

    assert attention.__name__ == f"{PKG}.attention"
    assert mla.__name__ == f"{PKG}.mla"


def test_unexported_name_raises_attribute_error_without_import():
    """Names outside the export table, unimported submodules included, never reach the import system."""
    names = ["mla", "backends", ".models", ".", "..", "", "backends.trtllm"]
    _run_fresh(f"""
        import sys
        import tensorrt_llm._torch
        preloaded = sorted(m for m in ("{PKG}.mla", "{BACKENDS}") if m in sys.modules)
        assert not preloaded, f"already imported by tensorrt_llm._torch: {{preloaded}}"
        import {PKG} as pkg
        failures = {{}}
        for name in {names!r}:
            before = set(sys.modules)
            try:
                getattr(pkg, name)
            except AttributeError:
                pass
            else:
                failures[name] = "getattr returned a value"
                continue
            if hasattr(pkg, name):
                failures[name] = "hasattr is True"
                continue
            added = sorted(set(sys.modules) - before)
            if added:
                failures[name] = f"imported {{added}}"
        assert not failures, failures
    """)


def test_unknown_name_raises_attribute_error(pkg):
    with pytest.raises(AttributeError) as excinfo:
        pkg.NoSuchAttentionName  # noqa: B018
    assert not isinstance(excinfo.value, ModuleNotFoundError)
    assert not hasattr(pkg, "NoSuchAttentionName")


def test_missing_dependency_inside_submodule_propagates(pkg, monkeypatch):
    """A dependency missing inside an existing submodule must not turn into AttributeError."""
    real_import_module = importlib.import_module

    def fake_import_module(target, package=None):
        if package == PKG and target == ".mla":
            raise ModuleNotFoundError("No module named 'some_missing_dep'", name="some_missing_dep")
        return real_import_module(target, package)

    monkeypatch.delitem(pkg.__dict__, "MLA", raising=False)
    monkeypatch.setattr(pkg.importlib, "import_module", fake_import_module)
    with pytest.raises(ModuleNotFoundError) as excinfo:
        getattr(pkg, "MLA")
    assert excinfo.value.name == "some_missing_dep"


@pytest.mark.parametrize(
    "access",
    [
        lambda pkg: getattr(pkg, "MLA"),
        lambda pkg: hasattr(pkg, "MLA"),
        lambda pkg: getattr(pkg, "MLA", None),
        lambda pkg: exec(f"from {PKG} import MLA", {}),
    ],
    ids=["getattr", "hasattr", "getattr_default", "from_import"],
)
def test_attribute_error_inside_submodule_surfaces_as_import_error(pkg, monkeypatch, access):
    """An AttributeError raised while importing a submodule must not be swallowed or retried."""
    real_import_module = importlib.import_module
    calls = []

    def fake_import_module(target, package=None):
        if package == PKG and target == ".mla":
            calls.append(target)
            raise AttributeError("module 'torch' has no attribute 'some_missing_op'")
        return real_import_module(target, package)

    monkeypatch.delitem(pkg.__dict__, "MLA", raising=False)
    monkeypatch.setattr(pkg.importlib, "import_module", fake_import_module)
    with pytest.raises(ImportError) as excinfo:
        access(pkg)
    assert isinstance(excinfo.value.__cause__, AttributeError)
    assert "some_missing_op" in str(excinfo.value.__cause__)
    assert calls == [".mla"]


def test_package_root_import_is_lazy():
    """Importing the package root adds nothing beyond the package module itself.

    ``tensorrt_llm._torch`` may load the package root already, because the custom
    ops import their kernels from ``kernels/`` beneath it. Nothing else under the
    package may be loaded at that point.
    """
    kernels = f"{PKG}.kernels"
    _run_fresh(f"""
        import sys
        import tensorrt_llm._torch
        before = set(sys.modules)
        import {PKG}
        added = set(sys.modules) - before
        assert added <= {{{PKG!r}}}, sorted(added)
        loaded = sorted(
            m for m in sys.modules
            if m.startswith({PKG + "."!r})
            and m != {kernels!r}
            and not m.startswith({kernels + "."!r})
        )
        assert not loaded, loaded
    """)


def test_backends_import_loads_no_module_layer():
    """Backend-only callers must not pay for the module layer."""
    _run_fresh(f"""
        import sys
        import tensorrt_llm._torch
        before = set(sys.modules)
        import {PKG}.backends
        loaded = sorted(m for m in {MODULE_LAYER!r} if m in sys.modules and m not in before)
        assert not loaded, loaded
    """)


@pytest.mark.parametrize(
    "statement",
    [
        f"import {PKG}.rotary_embedding",
        f"from {PKG} import MRotaryEmbedding, RotaryEmbedding",
    ],
)
def test_rotary_embedding_first_import(statement):
    """Backends import rotary_embedding back; importing it first must not cycle."""
    _run_fresh(statement)


def test_rotary_embedding_import_loads_no_backends():
    """rotary_embedding uses the backends only in annotations, so importing it must not load them."""
    _run_fresh(f"""
        import sys
        import tensorrt_llm._torch
        before = set(sys.modules)
        backends = {BACKENDS!r}
        assert backends not in before, "tensorrt_llm._torch already imports the backends"
        import {PKG}.rotary_embedding
        loaded = sorted(m for m in set(sys.modules) - before if m == backends or m.startswith(backends + "."))
        assert not loaded, loaded
    """)
