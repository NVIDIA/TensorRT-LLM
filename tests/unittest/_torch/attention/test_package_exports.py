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
defining submodule's own object, plain submodule access still works, a
missing dependency inside a submodule surfaces as ``ModuleNotFoundError``,
importing the package root or its ``backends`` sub-package loads no
module-layer module, and ``rotary_embedding`` (which the backends import
back) works as the first import of a process. The import-order checks run in
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
def test_submodule_attribute_access(pkg, submodule):
    module = getattr(pkg, submodule)
    assert isinstance(module, types.ModuleType)
    assert module.__name__ == f"{PKG}.{submodule}"


def test_from_import_submodule():
    from tensorrt_llm._torch.attention import attention, mla

    assert attention.__name__ == f"{PKG}.attention"
    assert mla.__name__ == f"{PKG}.mla"


def test_unknown_name_raises_attribute_error(pkg):
    with pytest.raises(AttributeError) as excinfo:
        pkg.NoSuchAttentionName  # noqa: B018
    assert not isinstance(excinfo.value, ModuleNotFoundError)
    assert not hasattr(pkg, "NoSuchAttentionName")


@pytest.mark.parametrize("name", ["MLA", "mla"])
def test_missing_dependency_inside_submodule_propagates(pkg, monkeypatch, name):
    """A dependency missing inside an existing submodule must not turn into AttributeError."""
    real_import_module = importlib.import_module

    def fake_import_module(target, package=None):
        if package == PKG and target == ".mla":
            raise ModuleNotFoundError("No module named 'some_missing_dep'", name="some_missing_dep")
        return real_import_module(target, package)

    monkeypatch.delitem(pkg.__dict__, name, raising=False)
    monkeypatch.setattr(pkg.importlib, "import_module", fake_import_module)
    with pytest.raises(ModuleNotFoundError) as excinfo:
        getattr(pkg, name)
    assert excinfo.value.name == "some_missing_dep"


def test_package_root_import_is_lazy():
    """Importing the package root adds nothing beyond the package module itself."""
    _run_fresh(f"""
        import sys
        import tensorrt_llm._torch
        before = set(sys.modules)
        assert {PKG!r} not in before, "tensorrt_llm._torch already imports the package root"
        import {PKG}
        added = set(sys.modules) - before
        assert added <= {{{PKG!r}}}, sorted(added)
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
    """rotary_embedding imports the backends, which import it back; neither order may cycle."""
    _run_fresh(statement)
