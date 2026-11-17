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
"""Shim identity, FutureWarning and old-path-import checks for moved serve modules.

A module move keeps the old path only when code outside this repository imports
it. Such a kept shim re-exports the public names (each the same object as at the
new path), warns once with ``FutureWarning``, and code inside the repository
must import the new path. These checks hold all three properties; the PR that
creates a shim adds its :class:`ShimRow` rows to the tables in
``tests/unittest/serve/test_shim_identity.py``.
"""

import ast
import importlib
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "SCAN_ROOTS",
    "ShimRow",
    "check_identity",
    "check_warnings",
    "find_old_path_imports",
]

SCAN_ROOTS = ("tensorrt_llm", "examples", "scripts", "triton_backend", "docs", "setup.py")


@dataclass(frozen=True)
class ShimRow:
    """One re-exported name of one kept old path."""

    old_module: str  # "tensorrt_llm.serve.openai_disagg_server"
    name: str  # "OpenAIDisaggServer"; public names only, except in interface rows
    new_module: str  # "tensorrt_llm.serve.disagg.openai_disagg_server"
    retire_in: str  # release that removes the shim; "not retired" for an interface row
    ban_old_path: bool = True  # False: repo code may keep importing name from old_module
    warns: bool = True  # False: the re-export is the module's own interface, not a shim


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def check_identity(rows: Sequence[ShimRow]) -> List[str]:
    """Each old-path name must be the same object as the new-path name.

    Identity only, never ``__module__``: some re-exported names are dicts or
    constants.
    """
    problems = []
    for row in rows:
        old_module = importlib.import_module(row.old_module)
        new_module = importlib.import_module(row.new_module)
        try:
            old_object = getattr(old_module, row.name)
        except AttributeError:
            problems.append(
                f"`{row.old_module}.{row.name}` no longer exists: the re-export was removed, "
                "probably by autoflake or ruff during pre-commit, which drop an import that "
                f"`__all__` does not list. Restore `from {row.new_module} import {row.name}` in "
                f"{row.old_module.replace('.', '/')}.py and list the name in `__all__` (a pure "
                "shim), or restore its row in `_LAZY_NAMES` (`commands/serve.py`)."
            )
            continue
        new_object = getattr(new_module, row.name)
        if old_object is not new_object:
            problems.append(
                f"`{row.old_module}.{row.name}` is not the same object as "
                f"`{row.new_module}.{row.name}`. The old module must re-export the object, "
                "not define a copy."
            )
    return problems


# ---------------------------------------------------------------------------
# FutureWarning
# ---------------------------------------------------------------------------

_IMPORT_WARNING = re.compile(
    r"^(?P<module>[A-Za-z0-9_.]+) has moved to (?P<location>[A-Za-z0-9_.]+) "
    r"and will be removed in (?P<release>\S+?)\.$"
)
_READ_WARNING = re.compile(
    r"^(?P<module>[A-Za-z0-9_.]+)\.(?P<name>[A-Za-z0-9_]+) has moved to "
    r"(?P<location>[A-Za-z0-9_.]+) and will be removed in (?P<release>\S+?)\.$"
)


def _matching_warnings(caught, prefix: str) -> List[str]:
    """Messages of the FutureWarnings that start with ``prefix``.

    Filtering by message, as ``tests/unittest/_torch/peft/test_shim_peft.py``
    does, also drops the warning of a parent shim package that a fresh import of
    one of its submodules emits.
    """
    return [
        str(record.message)
        for record in caught
        if issubclass(record.category, FutureWarning) and str(record.message).startswith(prefix)
    ]


def _no_warning_message(old_module: str, where: str, name: str = "") -> str:
    what = f"reading `{old_module}.{name}`" if where == "read" else f"importing `{old_module}`"
    return (
        f"{what} emitted no FutureWarning. A kept shim warns once with "
        "warnings.warn(..., FutureWarning, stacklevel=2): the TRT-LLM logger defaults to level "
        "error, so a logged warning would not reach callers outside the repository. Restore "
        "that call."
    )


def check_warnings(rows: Sequence[ShimRow]) -> List[str]:
    """Each kept shim warns once with the expected FutureWarning text.

    A pure shim (no ``_LAZY_NAMES``) warns exactly once at import with
    ``<old module> has moved to <location> and will be removed in <release>.``,
    where ``<location>`` is the row's new module or a package that contains it,
    and never on reads. A ``_LAZY_NAMES`` module warns never at import and
    exactly once on the first read of each lazy name; a name it imports eagerly
    never warns. Rows with ``warns=False`` (interface rows) are skipped.

    Imports each old module fresh (a module body runs once per process) and
    puts any previously imported module object back in ``sys.modules`` and on
    its parent package, so later tests keep the objects they imported.
    """
    problems: List[str] = []
    by_module: Dict[str, List[ShimRow]] = {}
    for row in rows:
        if row.warns:
            by_module.setdefault(row.old_module, []).append(row)

    for module_name, module_rows in by_module.items():
        saved = sys.modules.get(module_name)
        if saved is not None:
            del sys.modules[module_name]
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                module = importlib.import_module(module_name)
                import_messages = _matching_warnings(caught, f"{module_name} has moved to ")
                lazy_names = getattr(module, "_LAZY_NAMES", None)

                if lazy_names is None:
                    problems.extend(
                        _check_pure_shim_import(module_name, module_rows, import_messages)
                    )
                elif import_messages:
                    problems.append(
                        f"importing `{module_name}` emitted a FutureWarning; trtllm-serve "
                        "imports commands/serve.py on every start, so it may warn only when a "
                        "moved name is read."
                    )

                for row in module_rows:
                    is_lazy_name = lazy_names is not None and row.name in lazy_names
                    for read_index in (1, 2):
                        before = len(caught)
                        getattr(module, row.name)
                        read_messages = _matching_warnings(
                            caught[before:], f"{module_name}.{row.name} has moved to "
                        )
                        if is_lazy_name and read_index == 1:
                            problems.extend(_check_lazy_first_read(row, read_messages))
                        elif read_messages:
                            problems.append(
                                f"the FutureWarning of `{module_name}` says "
                                f"`{read_messages[0]}`; expected no warning on this read of "
                                f"`{row.name}`."
                            )
        finally:
            if saved is not None:
                sys.modules[module_name] = saved
                parent_name, _, tail = module_name.rpartition(".")
                if parent_name and parent_name in sys.modules:
                    setattr(sys.modules[parent_name], tail, saved)
    return problems


def _check_pure_shim_import(
    module_name: str, module_rows: Sequence[ShimRow], import_messages: List[str]
) -> List[str]:
    if not import_messages:
        return [_no_warning_message(module_name, "import")]
    problems = []
    if len(import_messages) > 1:
        problems.append(
            f"importing `{module_name}` emitted {len(import_messages)} FutureWarnings; a pure "
            "shim warns exactly once, at import."
        )
    message = import_messages[0]
    match = _IMPORT_WARNING.match(message)
    retire_in = module_rows[0].retire_in
    expected = f"{module_name} has moved to <new location> and will be removed in {retire_in}."
    if match is None or match.group("release") != retire_in:
        problems.append(
            f"the FutureWarning of `{module_name}` says `{message}`; expected `{expected}`: it "
            "names the new location and the removal release."
        )
        return problems
    location = match.group("location")
    for row in module_rows:
        if not (row.new_module == location or row.new_module.startswith(location + ".")):
            problems.append(
                f"the FutureWarning of `{module_name}` says `{message}`; expected a location "
                f"that contains `{row.new_module}`: it names the new location and the removal "
                "release."
            )
    return problems


def _check_lazy_first_read(row: ShimRow, read_messages: List[str]) -> List[str]:
    if not read_messages:
        return [_no_warning_message(row.old_module, "read", row.name)]
    expected = (
        f"{row.old_module}.{row.name} has moved to {row.new_module} "
        f"and will be removed in {row.retire_in}."
    )
    if read_messages != [expected]:
        return [
            f"the FutureWarning of `{row.old_module}` says `{read_messages[0]}`; expected "
            f"`{expected}`: it names the new location and the removal release."
        ]
    return []


# ---------------------------------------------------------------------------
# Old-path import ban
# ---------------------------------------------------------------------------


def _module_name_of(path: Path, repo_root: Path) -> str:
    relative = path.relative_to(repo_root)
    parts = list(relative.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][: -len(".py")]
    return ".".join(parts)


def _package_of(path: Path, repo_root: Path) -> str:
    module = _module_name_of(path, repo_root)
    if path.name == "__init__.py":
        return module
    return module.rpartition(".")[0]


def _resolve_relative(package: str, level: int, module: Optional[str]) -> Optional[str]:
    if level == 0:
        return module
    parts = package.split(".") if package else []
    if level - 1 > len(parts):
        return None
    base = parts[: len(parts) - (level - 1)]
    if module:
        base.append(module)
    return ".".join(base) if base else None


def find_old_path_imports(
    rows: Sequence[ShimRow], repo_root: Path, roots: Sequence[str] = SCAN_ROOTS
) -> List[str]:
    """Imports of banned names from kept old paths, anywhere under ``roots``.

    Flags ``from <old_module> import <name>`` (absolute, or relative resolved
    against the file's package) and ``<alias>.<name>`` attribute access where
    ``<alias>`` is bound to the old module in that file. Importing an old module
    that still has code is allowed; only the banned names are flagged. The old
    module's own file is exempt, and ``tests/`` is not scanned.
    """
    banned: Dict[str, set] = {}
    for row in rows:
        if row.ban_old_path:
            banned.setdefault(row.old_module, set()).add(row.name)
    if not banned:
        return []

    retire_by_module = {row.old_module: row.retire_in for row in rows}
    new_by_module_name: Dict[Tuple[str, str], str] = {
        (row.old_module, row.name): row.new_module for row in rows
    }
    problems: List[str] = []
    files: List[Path] = []
    for root in roots:
        root_path = repo_root / root
        if root_path.is_file():
            files.append(root_path)
        elif root_path.is_dir():
            files.extend(sorted(root_path.rglob("*.py")))

    for path in files:
        text = path.read_text(errors="replace")
        # Cheap prefilter: parse only files that mention an old module.
        if not any(module.rsplit(".", 1)[-1] in text for module in banned):
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        file_module = _module_name_of(path, repo_root)
        exempt_modules = {module for module in banned if file_module == module}
        package = _package_of(path, repo_root)

        aliases: Dict[str, str] = {}  # local name -> old module it is bound to
        hits: List[Tuple[int, str, str]] = []  # (line, old module, name)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in banned and alias.name not in exempt_modules and alias.asname:
                        aliases[alias.asname] = alias.name
            elif isinstance(node, ast.ImportFrom):
                resolved = _resolve_relative(package, node.level, node.module)
                if resolved is None:
                    continue
                for alias in node.names:
                    if resolved in banned and resolved not in exempt_modules:
                        if alias.name in banned[resolved]:
                            hits.append((node.lineno, resolved, alias.name))
                    # `from <old package> import <old module name>` binds the module.
                    candidate = f"{resolved}.{alias.name}"
                    if candidate in banned and candidate not in exempt_modules:
                        aliases[alias.asname or alias.name] = candidate

        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            dotted = _dotted_name(node)
            if dotted is None:
                continue
            prefix, _, attribute = dotted.rpartition(".")
            bound = aliases.get(prefix)
            if bound is None and prefix in banned and prefix not in exempt_modules:
                bound = prefix  # the dotted `import <old_module>` used as `<old_module>.<name>`
            if bound is not None and attribute in banned.get(bound, ()):
                hits.append((node.lineno, bound, attribute))

        for lineno, module, name in sorted(set(hits)):
            problems.append(
                f"{path.relative_to(repo_root)}:{lineno} imports `{name}` from the old path "
                f"`{module}`; import it from `{new_by_module_name[(module, name)]}`. The old "
                "path is kept only for callers outside the repository until "
                f"{retire_by_module[module]}."
            )
    return problems


def _dotted_name(node: ast.Attribute) -> Optional[str]:
    parts = [node.attr]
    value = node.value
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
        return ".".join(reversed(parts))
    return None
