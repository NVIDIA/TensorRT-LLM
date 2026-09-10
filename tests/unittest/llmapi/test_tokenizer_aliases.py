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
"""Tests for the built-in ``--custom_tokenizer`` alias table.

``llm_args`` used to keep its own copy of ``TOKENIZER_ALIASES``. The copy
drifted from the table in ``tensorrt_llm.tokenizer``, so an alias could resolve
through ``load_custom_tokenizer`` and still fail through ``LlmArgs``: the
unresolved alias reached ``rsplit('.', 1)`` and raised "not enough values to
unpack".
"""

import importlib
import sys
import types

import pytest

from tensorrt_llm import tokenizer as tokenizer_pkg
from tensorrt_llm.llmapi import llm_args as llm_args_module
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.tokenizer import TOKENIZER_ALIASES

# These are pure-Python alias-table checks with no GPU dependency. The l0_cpu
# stage runs pytest with ``-m cpu_only`` and conftest drops any file lacking
# this marker, so without it every test here is deselected and pytest exits 5.
pytestmark = pytest.mark.cpu_only


def test_llm_args_reuses_the_tokenizer_package_alias_table():
    assert llm_args_module.TOKENIZER_ALIASES is tokenizer_pkg.TOKENIZER_ALIASES


def test_every_alias_maps_to_a_module_and_a_class():
    for alias, import_path in TOKENIZER_ALIASES.items():
        module_path, _, class_name = import_path.rpartition(".")
        assert module_path, f"alias {alias!r} has no module part: {import_path!r}"
        assert class_name, f"alias {alias!r} has no class part: {import_path!r}"


@pytest.mark.parametrize("alias, import_path", sorted(TOKENIZER_ALIASES.items()))
def test_every_alias_target_is_really_importable(alias, import_path):
    """Import each alias target for real so a typo fails CI, not at runtime.

    A typo in ``TOKENIZER_ALIASES`` would otherwise only blow up inside
    ``load_custom_tokenizer``. The structural test above only checks the string
    shape of the import path;
    the ``stub_tokenizer_modules`` fixture shadows every module in
    ``sys.modules``, so a nonexistent module or class would still pass there.
    This test resolves the module and the class object without any stubbing.

    CPU-only: it never instantiates a tokenizer or loads any weights. If a
    target needs an optional third-party dependency that is not installed, the
    test skips with a clear reason; a genuine typo in the aliased module (or one
    of its parent packages) still fails.
    """
    module_path, _, class_name = import_path.rpartition(".")
    assert module_path and class_name, (
        f"alias {alias!r} has a malformed import path: {import_path!r}"
    )

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        # The aliased module itself (or one of its parent packages) is missing:
        # that is exactly the typo this test must catch, so let it fail.
        if module_path == missing or module_path.startswith(f"{missing}."):
            raise
        # Otherwise an optional dependency of the target is unavailable in this
        # CPU-only environment, which is not what this test guards.
        pytest.skip(
            f"alias {alias!r} target {import_path!r} needs optional dependency "
            f"{missing!r}, which is not importable in this environment"
        )

    assert hasattr(module, class_name), (
        f"alias {alias!r} module {module_path!r} has no attribute "
        f"{class_name!r} (declared in TOKENIZER_ALIASES as {import_path!r})"
    )
    assert isinstance(getattr(module, class_name), type), (
        f"alias {alias!r} target {import_path!r} is not a class"
    )


class _StubTokenizer:
    """Stand-in for a custom tokenizer class; records how it was loaded."""

    import_path = None

    def __init__(self, load_path, **kwargs):
        self.load_path = load_path
        self.kwargs = kwargs

    @classmethod
    def from_pretrained(cls, load_path, **kwargs):
        return cls(load_path, **kwargs)


@pytest.fixture
def stub_tokenizer_modules(monkeypatch):
    """Shadow every aliased tokenizer module with a stub in ``sys.modules``.

    ``importlib.import_module`` returns a ``sys.modules`` hit without touching
    the filesystem, so alias resolution is exercised end to end without
    importing the real tokenizers or downloading anything.
    """
    for import_path in TOKENIZER_ALIASES.values():
        module_path, _, class_name = import_path.rpartition(".")
        module = types.ModuleType(module_path)
        stub = type(class_name, (_StubTokenizer,), {"import_path": import_path})
        setattr(module, class_name, stub)
        monkeypatch.setitem(sys.modules, module_path, module)


@pytest.mark.parametrize("alias", sorted(TOKENIZER_ALIASES))
def test_alias_resolves_through_llm_args(alias, stub_tokenizer_modules):
    args = TorchLlmArgs(model="dummy", custom_tokenizer=alias)

    assert isinstance(args.tokenizer, _StubTokenizer)
    assert args.tokenizer.import_path == TOKENIZER_ALIASES[alias]
    assert args.tokenizer.load_path == "dummy"


def test_full_import_path_still_works(stub_tokenizer_modules):
    import_path = TOKENIZER_ALIASES[sorted(TOKENIZER_ALIASES)[0]]

    args = TorchLlmArgs(model="dummy", custom_tokenizer=import_path)

    assert args.tokenizer.import_path == import_path


def test_unknown_identifier_without_a_module_part_is_rejected():
    with pytest.raises(ValueError, match="Failed to load custom tokenizer"):
        TorchLlmArgs(model="dummy", custom_tokenizer="not_a_registered_alias")
