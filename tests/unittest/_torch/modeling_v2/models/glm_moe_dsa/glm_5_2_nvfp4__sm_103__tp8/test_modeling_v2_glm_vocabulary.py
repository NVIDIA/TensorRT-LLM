# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One name per fast path of the GLM-5.2 tp8 target, used the same way everywhere.

The fast-path names of ``dispatch.FAST_PATHS`` are the ``DISABLE`` tokens, the kernel package
directories (or a directory declared in ``KERNEL_DIRECTORY``), the log tags, the knob infixes,
the kernel registry keys and the kernel test directories. Read off disk: no kernel module is
imported.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (  # noqa: E501
    dispatch,
)

pytestmark = pytest.mark.cpu_only

# The family's code: its shared package and every target of it, each with its kernel packages.
_TARGET = Path(dispatch.__file__).resolve().parents[1]
_KERNEL_ROOTS = sorted(_TARGET.glob("*/kernels"))
_TESTS = Path(__file__).resolve().parents[1]
_PATHS = frozenset(dispatch.FAST_PATH_NAMES)

_LOG_TAG = "[glm_moe_dsa] "
# After the tag: a fast-path name and a separator, or the dispatcher's own lines.
_TAGGED = re.compile(r"(?P<name>[a-z0-9_]+)(?P<sep>[/ :])")
_KNOB = re.compile(re.escape(dispatch.KNOB_PREFIX) + r"(?P<infix>[A-Z0-9_]*)")
# Knobs that select among the fast paths rather than tune one.
_SELECTOR_KNOBS = ("FEATURES", "DISABLE")


def _sources(root: Path):
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _kernel_directories() -> set:
    """The kernel packages of the family, the shared ones and every target's own."""
    return {
        d.name
        for kernels in _KERNEL_ROOTS
        for d in kernels.iterdir()
        if d.is_dir() and (d / "__init__.py").is_file()
    }


def _string_constants(tree: ast.AST):
    """Every ``str`` constant, the constant parts of f-strings included."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node


def test_each_fast_path_has_one_name():
    assert len(dispatch.FAST_PATH_NAMES) == 36
    assert len(_PATHS) == 36


def test_the_kernel_directories_are_the_fast_paths_without_a_declared_directory():
    directories = _kernel_directories()
    assert len(directories) == 28
    assert directories == _PATHS - set(dispatch.KERNEL_DIRECTORY)
    assert set(dispatch.KERNEL_DIRECTORY) <= _PATHS
    assert set(dispatch.KERNEL_DIRECTORY.values()) <= directories


def test_every_log_tag_names_a_fast_path_or_the_dispatcher():
    offences = []
    for path in _sources(_TARGET):
        for node in _string_constants(ast.parse(path.read_text(encoding="utf-8"))):
            text = node.value
            start = text.find(_LOG_TAG)
            while start >= 0:
                match = _TAGGED.match(text, start + len(_LOG_TAG))
                name = match.group("name") if match else None
                if not (
                    match and (name in _PATHS or (name == "dispatch" and match.group("sep") == ":"))
                ):
                    offences.append(f"{path.relative_to(_TARGET)}:{node.lineno}: {text[start:]!r}")
                start = text.find(_LOG_TAG, start + 1)
    assert not offences, "log lines must name a fast path after the tag:\n" + "\n".join(offences)


def _knob_owner(infix: str):
    """The fast path whose name starts the knob's infix (the longest such name), if any."""
    owners = [p for p in _PATHS if infix.startswith(p.upper() + "_")]
    return max(owners, key=len) if owners else None


def test_every_knob_name_starts_with_its_fast_path():
    offences = []
    for path in _sources(_TARGET):
        text = path.read_text(encoding="utf-8")
        for match in _KNOB.finditer(text):
            infix = match.group("infix")
            if infix and infix not in _SELECTOR_KNOBS and _knob_owner(infix) is None:
                line = text.count("\n", 0, match.start()) + 1
                offences.append(f"{path.relative_to(_TARGET)}:{line}: {match.group(0)}")
    assert not offences, "knob names must start with a fast-path name:\n" + "\n".join(offences)


def _imports_register_from_dispatch(tree: ast.AST) -> bool:
    return any(
        isinstance(node, ast.ImportFrom)
        and (node.module or "").split(".")[-1] == "dispatch"
        and any(alias.name == "register" and alias.asname is None for alias in node.names)
        for node in ast.walk(tree)
    )


def _module_strings(path: Path, tree: ast.Module) -> dict:
    """Module-level ``NAME = "<str>"`` bindings, and those imported from a sibling module."""
    out = {}
    for stmt in tree.body:
        if (
            isinstance(stmt, ast.Assign)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            out.update({t.id: stmt.value.value for t in stmt.targets if isinstance(t, ast.Name)})
        elif isinstance(stmt, ast.ImportFrom) and stmt.level == 1 and stmt.module:
            sibling = path.parent / f"{stmt.module}.py"
            if sibling.is_file():
                strings = _module_strings(sibling, ast.parse(sibling.read_text(encoding="utf-8")))
                out.update(
                    {
                        alias.asname or alias.name: strings[alias.name]
                        for alias in stmt.names
                        if alias.name in strings
                    }
                )
    return out


def _registered_paths(path: Path):
    """``(line, name)`` of every ``dispatch.register(<name>, …)`` call of one kernel module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bare = _imports_register_from_dispatch(tree)
    names = _module_strings(path, tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        if not (
            (isinstance(func, ast.Name) and func.id == "register" and bare)
            or (
                isinstance(func, ast.Attribute)
                and func.attr == "register"
                and isinstance(func.value, ast.Name)
                and func.value.id == "dispatch"
            )
        ):
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            yield node.lineno, first.value
        elif isinstance(first, ast.Name) and first.id in names:
            yield node.lineno, names[first.id]
        else:
            yield node.lineno, ast.unparse(first)


def test_every_registry_entry_is_keyed_by_the_fast_path_of_its_directory():
    offences, registering = [], set()
    for kernels in _KERNEL_ROOTS:
        for path in _sources(kernels):
            relative = path.relative_to(kernels)
            if len(relative.parts) < 2:
                continue
            directory = relative.parts[0]
            for line, name in _registered_paths(path):
                if name not in _PATHS or dispatch.KERNEL_DIRECTORY.get(name, name) != directory:
                    offences.append(
                        f"{kernels.parent.name}/kernels/{relative}:{line}: register({name!r}, ...)"
                    )
                else:
                    registering.add(directory)
    assert not offences, (
        "get_impl imports kernels.<directory of the path>, so each package registers its "
        "entry points under its own path:\n" + "\n".join(offences)
    )
    assert registering == _kernel_directories()


def test_every_kernel_test_directory_is_a_kernel_directory():
    """A target's tests of how it composes a shared package live under the package's name."""
    for kernels in _KERNEL_ROOTS:
        test_root = _TESTS / kernels.parent.name / "kernels"
        if not test_root.is_dir():
            continue
        tests = {d.name for d in test_root.iterdir() if d.is_dir() and d.name != "__pycache__"}
        assert tests <= _kernel_directories(), sorted(tests - _kernel_directories())
