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
"""Lint: serving code reads environment variables only through ``tensorrt_llm/serve/envs.py``.

Every environment variable that ``tensorrt_llm/serve/`` (except ``scripts/``) and
the trtllm-serve CLI read is declared once in ``tensorrt_llm/serve/envs.py`` and
read as ``envs.NAME``. This module finds the reads that bypass it, with ``ast``,
over the source tree:

- ``os.getenv(...)`` and a bare ``getenv(...)`` imported from ``os``;
- ``os.environ.get(...)``, ``os.environ[...]`` read, ``... in os.environ``;
- ``os.environ.pop(...)`` and ``os.environ.setdefault(...)`` whose value is used
  (an expression statement is a write, not a read);
- the same through ``from os import environ``, ``import os as <alias>`` and
  ``import os.<submodule>`` (which binds ``os``);
- ``from tensorrt_llm.serve.envs import <name>`` for any name but the module's API
  (``EnvVar``, ``consume``, ``environment_variables``): it freezes the value at
  import time, where ``envs.<name>`` reads it when used.

Writes (``os.environ[...] = ...``, ``del os.environ[...]``, ``update``, a ``pop``
or ``setdefault`` statement) and whole-environment copies (``copy()``,
``items()``, ``dict(os.environ)``, iteration) are not reads and pass.

Run ``python3 tests/test_common/serve/env_lint.py`` to list every finding in the
scope; ``tests/unittest/serve/test_envs.py`` fails on any.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

__all__ = ["Finding", "find_direct_env_reads", "scan_source", "scope_files"]

ENVS_MODULE = "tensorrt_llm.serve.envs"
# The module's API (its ``__all__``), the only names that may be imported from it:
# a variable imported by name would be read once, at import time.
ENVS_IMPORTABLE_NAMES = frozenset({"EnvVar", "consume", "environment_variables"})

_SERVE = Path("tensorrt_llm/serve")
_VISUAL_GEN_PACKAGE = _SERVE / "api" / "visual_gen"
# VisualGen serving code at its pre-package paths; scanned while a file exists.
_LEGACY_VISUAL_GEN_FILES = (
    _SERVE / "openai_video_routes.py",
    _SERVE / "visual_gen_utils.py",
    _SERVE / "visual_gen_metrics.py",
    _SERVE / "_visual_gen_errors.py",
)
_CLI_ENTRY = Path("tensorrt_llm/commands/serve.py")
_CLI_PACKAGE = Path("tensorrt_llm/commands/serve_cli")
_EXCLUDED_DIRS = (_SERVE / "scripts",)  # a published client package
_EXCLUDED_FILES = (_SERVE / "envs.py",)  # the module that reads os.environ for everyone


@dataclass(frozen=True)
class Finding:
    """One direct environment read (or value import) at ``path:line``."""

    path: Path
    line: int
    source: str
    kind: str = "read"  # "read" or "import"

    def message(self) -> str:
        if self.kind == "import":
            return (
                f"{self.path}:{self.line} imports a value from tensorrt_llm/serve/envs.py "
                f"({self.source}), which freezes it at import time. Import the module instead "
                "(from tensorrt_llm.serve import envs) and read envs.<NAME> where the value is "
                "used; see tensorrt_llm/serve/AGENTS.md, 'Environment variables'."
            )
        return (
            f"{self.path}:{self.line} reads the environment directly ({self.source}). Declare "
            "the variable once in tensorrt_llm/serve/envs.py (name, kind deploy/internal/external, "
            "default, description, secret) and read it as envs.<NAME>; see "
            "tensorrt_llm/serve/AGENTS.md, 'Environment variables'."
        )


def scope_files(repo_root: Path, *, visual_gen: bool) -> List[Path]:
    """The files the lint covers, as absolute paths.

    ``visual_gen=False``: every ``.py`` under ``tensorrt_llm/serve/`` except
    ``scripts/``, ``envs.py`` and the VisualGen package ``api/visual_gen/``, plus
    ``tensorrt_llm/commands/serve.py`` and ``tensorrt_llm/commands/serve_cli/``.
    ``visual_gen=True``: the VisualGen package plus the VisualGen serving files
    that still live at their older paths. The two parts are checked by separate
    tests because CI narrows VisualGen-only changes to tests under a
    ``visual_gen/`` path; the older VisualGen paths appear in both parts so that
    the VisualGen part is never empty.
    """
    repo_root = Path(repo_root)
    if visual_gen:
        files = sorted((repo_root / _VISUAL_GEN_PACKAGE).rglob("*.py"))
        files.extend(
            repo_root / path for path in _LEGACY_VISUAL_GEN_FILES if (repo_root / path).is_file()
        )
        return files
    files = []
    for path in sorted((repo_root / _SERVE).rglob("*.py")):
        relative = path.relative_to(repo_root)
        if relative in _EXCLUDED_FILES or relative.is_relative_to(_VISUAL_GEN_PACKAGE):
            continue
        if any(relative.is_relative_to(prefix) for prefix in _EXCLUDED_DIRS):
            continue
        files.append(path)
    if (repo_root / _CLI_ENTRY).is_file():
        files.append(repo_root / _CLI_ENTRY)
    files.extend(sorted((repo_root / _CLI_PACKAGE).rglob("*.py")))
    return files


def find_direct_env_reads(paths: Iterable[Path], *, root: Optional[Path] = None) -> List[Finding]:
    """Every finding in ``paths``; with ``root``, finding paths are relative to it."""
    findings: List[Finding] = []
    for path in paths:
        path = Path(path)
        shown = path
        if root is not None:
            try:
                shown = path.relative_to(root)
            except ValueError:
                shown = path
        findings.extend(scan_source(path.read_text(encoding="utf-8"), shown))
    return findings


def scan_source(source: str, path: Path) -> List[Finding]:
    """The findings of one module's source text, in line order."""
    tree = ast.parse(source, filename=str(path))
    scanner = _Scanner(source, Path(path))
    scanner.collect_names(tree)
    scanner.visit(tree)
    return sorted(scanner.findings, key=lambda finding: (finding.line, finding.source))


def _text(source: str, node: ast.AST) -> str:
    return " ".join((ast.get_source_segment(source, node) or "?").split())


class _Scanner(ast.NodeVisitor):
    def __init__(self, source: str, path: Path) -> None:
        self._source = source
        self._path = path
        self.findings: List[Finding] = []
        self._os_names: Set[str] = set()
        self._environ_names: Set[str] = set()
        self._getenv_names: Set[str] = set()
        self._statement_calls: Set[int] = set()

    def collect_names(self, tree: ast.AST) -> None:
        """Aliases of ``os``, ``os.environ`` and ``os.getenv``, wherever imported."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "os":
                        self._os_names.add(alias.asname or "os")
                    elif alias.name.startswith("os.") and alias.asname is None:
                        self._os_names.add("os")  # `import os.path` binds the name `os`
            elif isinstance(node, ast.ImportFrom) and node.module == "os" and node.level == 0:
                for alias in node.names:
                    if alias.name == "environ":
                        self._environ_names.add(alias.asname or "environ")
                    elif alias.name == "getenv":
                        self._getenv_names.add(alias.asname or "getenv")
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                self._statement_calls.add(id(node.value))

    def _is_environ(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Attribute) and node.attr == "environ":
            return isinstance(node.value, ast.Name) and node.value.id in self._os_names
        return isinstance(node, ast.Name) and node.id in self._environ_names

    def _add(self, node: ast.AST, kind: str = "read") -> None:
        self.findings.append(Finding(self._path, node.lineno, _text(self._source, node), kind))

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute):
            if (
                func.attr == "getenv"
                and isinstance(func.value, ast.Name)
                and func.value.id in self._os_names
            ):
                self._add(node)
            elif self._is_environ(func.value):
                if func.attr == "get":
                    self._add(node)
                elif func.attr in ("pop", "setdefault") and id(node) not in self._statement_calls:
                    self._add(node)
        elif isinstance(func, ast.Name) and func.id in self._getenv_names:
            self._add(node)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.ctx, ast.Load) and self._is_environ(node.value):
            self._add(node)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        if any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops) and any(
            self._is_environ(comparator) for comparator in node.comparators
        ):
            self._add(node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        is_envs = (node.level == 0 and module == ENVS_MODULE) or (
            node.level > 0 and (module == "envs" or module.endswith(".envs"))
        )
        if is_envs and any(alias.name not in ENVS_IMPORTABLE_NAMES for alias in node.names):
            self._add(node, kind="import")
        self.generic_visit(node)


def main(argv: Sequence[str]) -> int:
    repo_root = Path(__file__).resolve().parents[3]
    if len(argv) > 1:
        print(__doc__)
        return 2
    files = sorted(
        set(scope_files(repo_root, visual_gen=False)) | set(scope_files(repo_root, visual_gen=True))
    )
    findings = find_direct_env_reads(files, root=repo_root)
    for finding in findings:
        print(finding.message())
    print(f"{len(findings)} direct environment read(s) in {len(files)} files")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
