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
``tensorrt_llm/commands/serve.py`` read is declared once in ``tensorrt_llm/serve/envs.py``
and read as ``envs.NAME``. This module finds bypasses and misuses with ``ast``, over
the source tree:

- ``os.getenv(...)`` and a bare ``getenv(...)`` imported from ``os``;
- ``os.environ.get(...)``, ``os.environ[...]`` read, ``... in os.environ``;
- ``os.environ.pop(...)`` and ``os.environ.setdefault(...)`` whose value is used
  (an expression statement is a write, not a read);
- the same through ``from os import environ``, ``import os as <alias>``,
  ``import os.<submodule>`` (which binds ``os``), aliases bound to ``os.environ``,
  and reads off ``os.environ.copy()`` or ``dict(os.environ)``;
- ``from tensorrt_llm.serve.envs import <name>`` for any name but the module's API
  (``EnvVar``, ``consume``, ``environment_variables``), and module-level / class-body /
  default-argument ``envs.<name>`` reads or ``consume(...)`` calls that freeze values at
  import time;
- ``envs.<attr>`` references to names not declared in ``tensorrt_llm/serve/envs.py``.

Alias tracking is monotone per lexical scope (respecting Python lexical shadowing,
``global``, ``nonlocal``, class-scope skipping, and PEP 572 comprehension walrus
targets) so reads through aliases cannot be missed across control-flow edges.
Writes (``os.environ[...] = ...``, ``del os.environ[...]``, ``update``, a ``pop``
or ``setdefault`` statement) and whole-environment copies (``copy()``,
``items()``, ``dict(os.environ)``, iteration) without key reads pass.

Run ``python3 tests/test_common/serve/env_lint.py`` to list every finding in the
scope; ``tests/unittest/serve/test_envs.py`` fails on any.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "ENVS_IMPORTABLE_NAMES",
    "Finding",
    "declared_env_names",
    "find_direct_env_reads",
    "scan_source",
    "scope_files",
]

ENVS_MODULE = "tensorrt_llm.serve.envs"
SERVE_PACKAGE = "tensorrt_llm.serve"
# The module's API (its ``__all__``), the only names that may be imported from it:
# a variable imported by name would be read once, at import time.
ENVS_IMPORTABLE_NAMES = frozenset({"EnvVar", "consume", "environment_variables"})

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SERVE = Path("tensorrt_llm/serve")
_ENVS_FILE = _SERVE / "envs.py"
# VisualGen serving files; checked both in the main lint and under a visual_gen/
# test path because CI narrows VisualGen-only changes to tests under visual_gen/.
_VISUAL_GEN_FILES = (
    _SERVE / "openai_video_routes.py",
    _SERVE / "visual_gen_utils.py",
    _SERVE / "visual_gen_metrics.py",
    _SERVE / "_visual_gen_errors.py",
)
_CLI_ENTRY = Path("tensorrt_llm/commands/serve.py")
_EXCLUDED_DIRS = (_SERVE / "scripts",)  # a published client package
_EXCLUDED_FILES = (_ENVS_FILE,)  # the module that reads os.environ for everyone

# Monotone binding tags tracked per lexical scope.
_TAG_OS = "os"
_TAG_ENVIRON = "environ"
_TAG_GETENV = "getenv"
_TAG_ENVIRON_POP = "environ_pop"
_TAG_ENVIRON_COPY = "environ_copy"
_TAG_ENVIRON_ITEMS = "environ_items"
_TAG_ENVIRON_ITEMS_VIEW = "environ_items_view"
_TAG_ENVS = "envs"
_TAG_CONSUME = "consume"
_TAG_TLLM_ENVS_ROOT = "tllm_envs_root"
_TAG_OTHER = "other"

# Builtins that eagerly exhaust or step a generator expression passed as a positional argument.
_EAGER_ITER_CONSUMERS = frozenset(
    {
        "all",
        "any",
        "bytearray",
        "bytes",
        "dict",
        "frozenset",
        "list",
        "max",
        "min",
        "next",
        "set",
        "sorted",
        "sum",
        "tuple",
    }
)


@dataclass(frozen=True)
class Finding:
    """One direct environment read, import-time freeze, or unknown envs attribute at ``path:line``."""

    path: Path
    line: int
    source: str
    kind: str = "read"  # "read", "import", "freeze", or "unknown"

    def message(self) -> str:
        if self.kind == "import":
            return (
                f"{self.path}:{self.line} imports a value from tensorrt_llm/serve/envs.py "
                f"({self.source}), which freezes it at import time. Import the module instead "
                "(from tensorrt_llm.serve import envs) and read envs.<NAME> where the value is "
                "used; see tensorrt_llm/serve/AGENTS.md, 'Environment variables'."
            )
        if self.kind == "freeze":
            return (
                f"{self.path}:{self.line} reads tensorrt_llm/serve/envs.py at import time "
                f"({self.source}), which freezes the value. Read envs.<NAME> inside the "
                "function or method where it is used; see tensorrt_llm/serve/AGENTS.md, "
                "'Environment variables'."
            )
        if self.kind == "unknown":
            return (
                f"{self.path}:{self.line} reads an undeclared attribute from "
                f"tensorrt_llm/serve/envs.py ({self.source}). Declare the variable in "
                "tensorrt_llm/serve/envs.py before reading it; see "
                "tensorrt_llm/serve/AGENTS.md, 'Environment variables'."
            )
        return (
            f"{self.path}:{self.line} reads the environment directly ({self.source}). Declare "
            "the variable once in tensorrt_llm/serve/envs.py (name, kind deploy/internal/external, "
            "default, description, secret) and read it as envs.<NAME>; see "
            "tensorrt_llm/serve/AGENTS.md, 'Environment variables'."
        )


@dataclass
class _Scope:
    """One lexical scope frame in ``_Scanner``."""

    kind: str  # "module", "function", "class", or "comp"
    parent: _Scope | None = None
    globals_: set[str] = field(default_factory=set)
    nonlocals: set[str] = field(default_factory=set)
    local_names: set[str] = field(default_factory=set)
    bindings: dict[str, set[str]] = field(default_factory=dict)
    typing_imports: set[str] = field(default_factory=set)
    tc_imports: set[str] = field(default_factory=set)
    tc_rebound: set[str] = field(default_factory=set)


def declared_env_names(repo_root: Path = _REPO_ROOT) -> frozenset[str]:
    """Names declared in ``tensorrt_llm/serve/envs.py``'s ``TYPE_CHECKING`` block."""
    envs_path = Path(repo_root) / _ENVS_FILE
    tree = ast.parse(envs_path.read_text(encoding="utf-8"), filename=str(envs_path))
    names: set[str] = set()
    for node in tree.body:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            for statement in node.body:
                if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                    names.add(statement.target.id)
    return frozenset(names)


def scope_files(repo_root: Path, *, visual_gen: bool) -> list[Path]:
    """The files the lint covers, as absolute paths.

    ``visual_gen=False``: every ``.py`` under ``tensorrt_llm/serve/`` except
    ``scripts/`` and ``envs.py``, plus ``tensorrt_llm/commands/serve.py``.
    ``visual_gen=True``: the VisualGen serving files under ``tensorrt_llm/serve/``.
    Those files appear in both parts so VisualGen-only CI stages run the lint on
    VisualGen serving changes.
    """
    repo_root = Path(repo_root)
    if visual_gen:
        return [repo_root / path for path in _VISUAL_GEN_FILES if (repo_root / path).is_file()]
    files = []
    for path in sorted((repo_root / _SERVE).rglob("*.py")):
        relative = path.relative_to(repo_root)
        if relative in _EXCLUDED_FILES:
            continue
        if any(relative.is_relative_to(prefix) for prefix in _EXCLUDED_DIRS):
            continue
        files.append(path)
    if (repo_root / _CLI_ENTRY).is_file():
        files.append(repo_root / _CLI_ENTRY)
    return files


def find_direct_env_reads(
    paths: Iterable[Path],
    *,
    root: Path | None = None,
    declared_names: frozenset[str] | None = None,
) -> list[Finding]:
    """Every finding in ``paths``; with ``root``, finding paths are relative to it."""
    names = declared_env_names() if declared_names is None else declared_names
    findings: list[Finding] = []
    for path in paths:
        path = Path(path)
        shown = path
        if root is not None:
            try:
                shown = path.relative_to(root)
            except ValueError:
                shown = path
        findings.extend(scan_source(path.read_text(encoding="utf-8"), shown, declared_names=names))
    return findings


def scan_source(
    source: str,
    path: Path,
    *,
    declared_names: frozenset[str] | None = None,
) -> list[Finding]:
    """The findings of one module's source text, in line order."""
    names = declared_env_names() if declared_names is None else declared_names
    tree = ast.parse(source, filename=str(path))
    scanner = _Scanner(source, Path(path), names)
    scanner.scan_module(tree)
    return sorted(
        scanner.findings, key=lambda finding: (finding.line, finding.source, finding.kind)
    )


def _text(source: str, node: ast.AST) -> str:
    return " ".join((ast.get_source_segment(source, node) or "?").split())


def _has_future_annotations(tree: ast.Module) -> bool:
    for stmt in tree.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__" and stmt.level == 0:
            if any(alias.name == "annotations" for alias in stmt.names):
                return True
            continue
        break
    return False


def _package_parts(path: Path) -> tuple[str, ...] | None:
    """Return dotted package components for a file under ``tensorrt_llm/``, else ``None``.

    Uses the last ``tensorrt_llm`` path segment so absolute worktree paths like
    ``/code/tensorrt_llm/.worktrees/<wt>/tensorrt_llm/serve/render/foo.py`` resolve to
    ``('tensorrt_llm', 'serve', 'render')``.
    """
    parts = path.parts
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx] == "tensorrt_llm":
            rel = parts[idx:]
            if len(rel) < 2:
                return None
            return rel[:-1]
    return None


def _resolve_import_from_module(
    node: ast.ImportFrom, pkg_parts: tuple[str, ...] | None
) -> str | None:
    """Resolve an ``ImportFrom``'s target module name using the file's package when known."""
    module = node.module or ""
    if node.level == 0:
        return module
    if pkg_parts is None:
        return None
    drop = node.level - 1
    if drop > len(pkg_parts):
        return ""
    base = pkg_parts[: len(pkg_parts) - drop]
    if module:
        return ".".join((*base, module))
    return ".".join(base)


def _is_envs_from_import(node: ast.ImportFrom, pkg_parts: tuple[str, ...] | None) -> bool:
    resolved = _resolve_import_from_module(node, pkg_parts)
    if resolved is not None:
        return resolved == ENVS_MODULE
    module = node.module or ""
    return module == "envs" or module.endswith(".envs")


def _imports_envs_module_from(node: ast.ImportFrom, pkg_parts: tuple[str, ...] | None) -> bool:
    resolved = _resolve_import_from_module(node, pkg_parts)
    if resolved is not None:
        return resolved == SERVE_PACKAGE
    module = node.module or ""
    return module in ("", "serve") or module.endswith(".serve")


def _assigned_name_targets(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: list[str] = []
        for elt in target.elts:
            names.extend(_assigned_name_targets(elt))
        return names
    if isinstance(target, ast.Starred):
        return _assigned_name_targets(target.value)
    return []


def _pattern_bound_names(pattern: ast.pattern) -> list[str]:
    names: list[str] = []
    if isinstance(pattern, ast.MatchAs):
        if pattern.pattern is not None:
            names.extend(_pattern_bound_names(pattern.pattern))
        if pattern.name is not None:
            names.append(pattern.name)
    elif isinstance(pattern, ast.MatchOr):
        for sub in pattern.patterns:
            names.extend(_pattern_bound_names(sub))
    elif isinstance(pattern, ast.MatchSequence):
        for sub in pattern.patterns:
            names.extend(_pattern_bound_names(sub))
    elif isinstance(pattern, ast.MatchStar):
        if pattern.name is not None:
            names.append(pattern.name)
    elif isinstance(pattern, ast.MatchMapping):
        for sub in pattern.patterns:
            names.extend(_pattern_bound_names(sub))
        if pattern.rest is not None:
            names.append(pattern.rest)
    elif isinstance(pattern, ast.MatchClass):
        for sub in (*pattern.patterns, *pattern.kwd_patterns):
            names.extend(_pattern_bound_names(sub))
    return names


def _param_names(args: ast.arguments) -> list[str]:
    names = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    if args.vararg is not None:
        names.append(args.vararg.arg)
    if args.kwarg is not None:
        names.append(args.kwarg.arg)
    return names


def _walrus_scope(scope: _Scope) -> _Scope:
    cur = scope
    while cur.kind == "comp" and cur.parent is not None:
        cur = cur.parent
    return cur


class _Scanner:
    """Two-phase AST scanner with monotone per-scope lexical alias tracking.

    Phase 1 builds the lexical scope tree, collects declarations (``global``,
    ``nonlocal``, and locally bound names), and propagates monotone alias tags
    to a fixed point. Phase 2 walks the AST in lexical context to emit findings.
    """

    def __init__(self, source: str, path: Path, declared_names: frozenset[str]) -> None:
        self._source = source
        self._path = path
        self._pkg_parts = _package_parts(path)
        self._declared_names = declared_names
        self._future_annotations = False
        self.findings: list[Finding] = []
        self._statement_calls: set[int] = set()
        self._scopes_by_node: dict[int, _Scope] = {}
        self._scopes_by_id: dict[int, _Scope] = {}
        self._root_scope = _Scope("module")
        self._bindings_changed = False
        self._alias_peers: dict[tuple[int, str], set[tuple[int, str]]] = {}

    def _resolves_to_typing_guard(self, scope: _Scope, name: str, *, module_alias: bool) -> bool:
        """True iff ``name`` in ``scope`` resolves solely to an un-rebound ``typing`` guard import."""
        cur: _Scope | None = scope
        is_innermost = True
        while cur is not None:
            if not is_innermost and cur.kind == "class":
                cur = cur.parent
                continue
            if name in cur.globals_:
                cur = self._root_scope
                is_innermost = False
            if name in cur.nonlocals:
                cur = cur.parent
                is_innermost = False
                continue
            imports = cur.typing_imports if module_alias else cur.tc_imports
            other_imports = cur.tc_imports if module_alias else cur.typing_imports
            if name in cur.tc_rebound or name in imports or name in other_imports:
                if is_innermost and cur.kind == "class":
                    return (
                        name in imports
                        and name not in cur.tc_rebound
                        and name not in other_imports
                        and name not in self._root_scope.tc_rebound
                    )
                return name in imports and name not in cur.tc_rebound and name not in other_imports
            cur = cur.parent
            is_innermost = False
        return False

    def _is_type_checking_guard(self, node: ast.expr, scope: _Scope) -> bool:
        if isinstance(node, ast.Name):
            return self._resolves_to_typing_guard(scope, node.id, module_alias=False)
        return (
            isinstance(node, ast.Attribute)
            and node.attr == "TYPE_CHECKING"
            and isinstance(node.value, ast.Name)
            and self._resolves_to_typing_guard(scope, node.value.id, module_alias=True)
        )

    def scan_module(self, tree: ast.Module) -> None:
        self._future_annotations = _has_future_annotations(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                self._statement_calls.add(id(node.value))

        self._scopes_by_node[id(tree)] = self._root_scope
        self._build_scopes_and_typing(tree, self._root_scope)
        for scope in self._scopes_by_node.values():
            for name in list(scope.tc_rebound):
                if name in scope.globals_:
                    self._root_scope.tc_rebound.add(name)
                elif name in scope.nonlocals:
                    cur = scope.parent
                    while cur is not None:
                        if (
                            cur.kind == "function"
                            and (
                                name in cur.tc_imports
                                or name in cur.typing_imports
                                or name in cur.tc_rebound
                            )
                            and name not in cur.globals_
                            and name not in cur.nonlocals
                        ):
                            cur.tc_rebound.add(name)
                            break
                        cur = cur.parent
        self._collect_decls(tree, self._root_scope)
        for scope in self._scopes_by_node.values():
            for name in scope.local_names:
                if name not in scope.globals_ and name not in scope.nonlocals:
                    scope.bindings.setdefault(name, set()).add(_TAG_OTHER)

        while True:
            self._bindings_changed = False
            self._propagate(tree, self._root_scope)
            if not self._bindings_changed:
                break

        self._emit(tree, self._root_scope, runtime_depth=0, in_annotation=False)

    def _resolve_binding_scope(self, scope: _Scope, name: str) -> _Scope:
        """Resolve the lexical scope owning an existing binding of ``name`` (for in-place mutation)."""
        cur: _Scope | None = scope
        is_innermost = True
        while cur is not None:
            if not is_innermost and cur.kind == "class":
                cur = cur.parent
                continue
            if name in cur.globals_:
                return self._root_scope
            if name not in cur.nonlocals and name in cur.bindings:
                return cur
            cur = cur.parent
            is_innermost = False
        return scope

    def _add_tag_in_scope(self, target: _Scope, name: str, tag: str) -> None:
        tags = target.bindings.setdefault(name, set())
        if tag not in tags:
            tags.add(tag)
            self._bindings_changed = True

    def _add_tag(self, scope: _Scope, name: str, tag: str) -> None:
        self._add_tag_in_scope(self._target_scope(scope, name), name, tag)

    def _target_scope(self, scope: _Scope, name: str) -> _Scope:
        if name in scope.globals_:
            return self._root_scope
        if name in scope.nonlocals:
            cur = scope.parent
            while cur is not None:
                if (
                    cur.kind == "function"
                    and name in cur.local_names
                    and name not in cur.globals_
                    and name not in cur.nonlocals
                ):
                    return cur
                cur = cur.parent
        return scope

    def _lookup_tags(self, scope: _Scope, name: str) -> set[str]:
        cur: _Scope | None = scope
        is_innermost = True
        collected: set[str] = set()
        while cur is not None:
            if not is_innermost and cur.kind == "class":
                cur = cur.parent
                continue
            if name in cur.globals_:
                collected.update(self._root_scope.bindings.get(name, set()))
                return collected
            if name not in cur.nonlocals and name in cur.bindings:
                if is_innermost and cur.kind == "class":
                    # Inside a class body that also binds ``name``, Python uses LOAD_NAME,
                    # checking class locals then module globals (skipping enclosing functions).
                    collected.update(cur.bindings[name])
                    collected.update(self._root_scope.bindings.get(name, set()))
                    return collected
                collected.update(cur.bindings[name])
                return collected
            cur = cur.parent
            is_innermost = False
        return collected

    def _expr_tags(self, node: ast.AST, scope: _Scope) -> set[str]:
        if isinstance(node, ast.Name):
            return self._lookup_tags(scope, node.id) - {_TAG_OTHER}
        if isinstance(node, ast.Attribute):
            if (
                node.attr == "envs"
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "serve"
                and isinstance(node.value.value, ast.Name)
                and _TAG_TLLM_ENVS_ROOT in self._lookup_tags(scope, node.value.value.id)
            ):
                return {_TAG_ENVS}
            base = self._expr_tags(node.value, scope)
            out: set[str] = set()
            if _TAG_OS in base:
                if node.attr == "environ":
                    out.add(_TAG_ENVIRON)
                elif node.attr == "getenv":
                    out.add(_TAG_GETENV)
            if _TAG_ENVIRON in base:
                if node.attr == "get":
                    out.add(_TAG_GETENV)
                elif node.attr in ("pop", "setdefault"):
                    out.add(_TAG_ENVIRON_POP)
                elif node.attr == "copy":
                    out.add(_TAG_ENVIRON_COPY)
                elif node.attr == "items":
                    out.add(_TAG_ENVIRON_ITEMS)
            if _TAG_ENVS in base and node.attr == "consume":
                out.add(_TAG_CONSUME)
            return out
        if isinstance(node, ast.Call):
            if not node.args and not node.keywords:
                func_tags = self._expr_tags(node.func, scope)
                if _TAG_ENVIRON_COPY in func_tags:
                    return {_TAG_ENVIRON}
                if _TAG_ENVIRON_ITEMS in func_tags:
                    return {_TAG_ENVIRON_ITEMS_VIEW}
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "dict"
                and not self._lookup_tags(scope, "dict")
                and (
                    (
                        node.args
                        and bool(
                            self._expr_tags(node.args[0], scope)
                            & {_TAG_ENVIRON, _TAG_ENVIRON_ITEMS_VIEW}
                        )
                    )
                    or any(
                        kw.arg is None and _TAG_ENVIRON in self._expr_tags(kw.value, scope)
                        for kw in node.keywords
                    )
                )
            ):
                return {_TAG_ENVIRON}
            return set()
        if isinstance(node, ast.IfExp):
            return self._expr_tags(node.body, scope) | self._expr_tags(node.orelse, scope)
        if isinstance(node, ast.BoolOp):
            out = set()
            for val in node.values:
                out.update(self._expr_tags(val, scope))
            return out
        if isinstance(node, ast.NamedExpr):
            return self._expr_tags(node.value, scope)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return self._expr_tags(node.left, scope) | self._expr_tags(node.right, scope)
        if isinstance(node, ast.Dict):
            for key, val in zip(node.keys, node.values):
                if key is None and _TAG_ENVIRON in self._expr_tags(val, scope):
                    return {_TAG_ENVIRON}
            return set()
        return set()

    def _import_tags(self, node: ast.Import) -> list[tuple[str, str | None]]:
        out: list[tuple[str, str | None]] = []
        for alias in node.names:
            bound = alias.asname or alias.name.split(".", 1)[0]
            if alias.name == "os":
                out.append((bound, _TAG_OS))
            elif alias.name.startswith("os.") and alias.asname is None:
                out.append((bound, _TAG_OS))
            elif alias.name == ENVS_MODULE:
                if alias.asname is not None:
                    out.append((bound, _TAG_ENVS))
                else:
                    out.append((bound, _TAG_TLLM_ENVS_ROOT))
            else:
                out.append((bound, None))
        return out

    def _import_from_tags(self, node: ast.ImportFrom) -> list[tuple[str, str | None]]:
        out: list[tuple[str, str | None]] = []
        is_os = node.level == 0 and node.module == "os"
        is_envs_from = _is_envs_from_import(node, self._pkg_parts)
        imports_envs_mod = _imports_envs_module_from(node, self._pkg_parts)
        for alias in node.names:
            if alias.name == "*":
                continue
            bound = alias.asname or alias.name
            tag: str | None = None
            if is_os and alias.name == "environ":
                tag = _TAG_ENVIRON
            elif is_os and alias.name == "getenv":
                tag = _TAG_GETENV
            elif is_envs_from and alias.name == "consume":
                tag = _TAG_CONSUME
            elif imports_envs_mod and alias.name == "envs":
                tag = _TAG_ENVS
            out.append((bound, tag))
        return out

    # ------------------------------------------------------------------
    # Phase 1a: build lexical scope tree, typing-guard info, and bindings.
    # ------------------------------------------------------------------

    def _mark_tc_rebound(self, scope: _Scope, names: Iterable[str]) -> None:
        for name in names:
            target = self._root_scope if name in scope.globals_ else scope
            target.tc_rebound.add(name)

    def _build_scopes_and_typing(self, node: ast.AST, scope: _Scope) -> None:
        """Build every ``_Scope`` node and record per-scope ``typing`` guard imports/rebindings."""
        if isinstance(node, ast.Global):
            scope.globals_.update(node.names)
            return
        if isinstance(node, ast.Nonlocal):
            scope.nonlocals.update(node.names)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._mark_tc_rebound(scope, (node.name,))
            for child in (*node.decorator_list, *node.args.defaults):
                self._build_scopes_and_typing(child, scope)
            for child in node.args.kw_defaults:
                if child is not None:
                    self._build_scopes_and_typing(child, scope)
            for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                if arg.annotation is not None:
                    self._build_scopes_and_typing(arg.annotation, scope)
            for extra in (node.args.vararg, node.args.kwarg):
                if extra is not None and extra.annotation is not None:
                    self._build_scopes_and_typing(extra.annotation, scope)
            if node.returns is not None:
                self._build_scopes_and_typing(node.returns, scope)
            fn_scope = _Scope("function", parent=scope)
            fn_scope.tc_rebound.update(_param_names(node.args))
            self._scopes_by_node[id(node)] = fn_scope
            for stmt in node.body:
                self._build_scopes_and_typing(stmt, fn_scope)
            return
        if isinstance(node, ast.Lambda):
            for child in node.args.defaults:
                self._build_scopes_and_typing(child, scope)
            for child in node.args.kw_defaults:
                if child is not None:
                    self._build_scopes_and_typing(child, scope)
            lam_scope = _Scope("function", parent=scope)
            lam_scope.tc_rebound.update(_param_names(node.args))
            self._scopes_by_node[id(node)] = lam_scope
            self._build_scopes_and_typing(node.body, lam_scope)
            return
        if isinstance(node, ast.ClassDef):
            self._mark_tc_rebound(scope, (node.name,))
            for child in (*node.decorator_list, *node.bases):
                self._build_scopes_and_typing(child, scope)
            for kw in node.keywords:
                self._build_scopes_and_typing(kw.value, scope)
            cls_scope = _Scope("class", parent=scope)
            self._scopes_by_node[id(node)] = cls_scope
            for stmt in node.body:
                self._build_scopes_and_typing(stmt, cls_scope)
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            comp_scope = _Scope("comp", parent=scope)
            self._scopes_by_node[id(node)] = comp_scope
            self._build_scopes_and_typing(node.generators[0].iter, scope)
            for gen_idx, gen in enumerate(node.generators):
                if gen_idx > 0:
                    self._build_scopes_and_typing(gen.iter, comp_scope)
                comp_scope.tc_rebound.update(_assigned_name_targets(gen.target))
                self._build_scopes_and_typing(gen.target, comp_scope)
                for cond in gen.ifs:
                    self._build_scopes_and_typing(cond, comp_scope)
            if isinstance(node, ast.DictComp):
                self._build_scopes_and_typing(node.key, comp_scope)
                self._build_scopes_and_typing(node.value, comp_scope)
            else:
                self._build_scopes_and_typing(node.elt, comp_scope)
            return
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".", 1)[0]
                if alias.name == "typing":
                    scope.typing_imports.add(bound)
                else:
                    self._mark_tc_rebound(scope, (bound,))
            return
        if isinstance(node, ast.ImportFrom):
            is_typing = node.level == 0 and node.module == "typing"
            for alias in node.names:
                if alias.name == "*":
                    continue
                bound = alias.asname or alias.name
                if is_typing and alias.name == "TYPE_CHECKING":
                    scope.tc_imports.add(bound)
                else:
                    self._mark_tc_rebound(scope, (bound,))
            return
        if isinstance(node, ast.Assign):
            for target in node.targets:
                self._mark_tc_rebound(scope, _assigned_name_targets(target))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            if isinstance(node.target, ast.Name):
                self._mark_tc_rebound(scope, (node.target.id,))
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            self._mark_tc_rebound(scope, _assigned_name_targets(node.target))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    self._mark_tc_rebound(scope, _assigned_name_targets(item.optional_vars))
        elif isinstance(node, ast.ExceptHandler):
            if node.name is not None:
                self._mark_tc_rebound(scope, (node.name,))
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                self._mark_tc_rebound(scope, _assigned_name_targets(target))
        elif isinstance(node, ast.NamedExpr):
            self._mark_tc_rebound(_walrus_scope(scope), (node.target.id,))
        elif isinstance(node, ast.match_case):
            self._mark_tc_rebound(scope, _pattern_bound_names(node.pattern))

        for child in ast.iter_child_nodes(node):
            self._build_scopes_and_typing(child, scope)

    def _collect_decls(self, node: ast.AST, scope: _Scope) -> None:
        if isinstance(node, ast.If) and self._is_type_checking_guard(node.test, scope):
            for stmt in node.orelse:
                self._collect_decls(stmt, scope)
            return
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope.local_names.add(node.name)
            for child in (*node.decorator_list, *node.args.defaults):
                self._collect_decls(child, scope)
            for child in node.args.kw_defaults:
                if child is not None:
                    self._collect_decls(child, scope)
            for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                if arg.annotation is not None:
                    self._collect_decls(arg.annotation, scope)
            for extra in (node.args.vararg, node.args.kwarg):
                if extra is not None and extra.annotation is not None:
                    self._collect_decls(extra.annotation, scope)
            if node.returns is not None:
                self._collect_decls(node.returns, scope)
            fn_scope = self._scopes_by_node[id(node)]
            fn_scope.local_names.update(_param_names(node.args))
            for stmt in node.body:
                self._collect_decls(stmt, fn_scope)
            return
        if isinstance(node, ast.Lambda):
            for child in node.args.defaults:
                self._collect_decls(child, scope)
            for child in node.args.kw_defaults:
                if child is not None:
                    self._collect_decls(child, scope)
            lam_scope = self._scopes_by_node[id(node)]
            lam_scope.local_names.update(_param_names(node.args))
            self._collect_decls(node.body, lam_scope)
            return
        if isinstance(node, ast.ClassDef):
            scope.local_names.add(node.name)
            for child in (*node.decorator_list, *node.bases):
                self._collect_decls(child, scope)
            for kw in node.keywords:
                self._collect_decls(kw.value, scope)
            cls_scope = self._scopes_by_node[id(node)]
            for stmt in node.body:
                self._collect_decls(stmt, cls_scope)
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            comp_scope = self._scopes_by_node[id(node)]
            self._collect_decls(node.generators[0].iter, scope)
            for gen_idx, gen in enumerate(node.generators):
                if gen_idx > 0:
                    self._collect_decls(gen.iter, comp_scope)
                comp_scope.local_names.update(_assigned_name_targets(gen.target))
                self._collect_decls(gen.target, comp_scope)
                for cond in gen.ifs:
                    self._collect_decls(cond, comp_scope)
            if isinstance(node, ast.DictComp):
                self._collect_decls(node.key, comp_scope)
                self._collect_decls(node.value, comp_scope)
            else:
                self._collect_decls(node.elt, comp_scope)
            return
        if isinstance(node, ast.Import):
            for bound, _ in self._import_tags(node):
                scope.local_names.add(bound)
            return
        if isinstance(node, ast.ImportFrom):
            for bound, _ in self._import_from_tags(node):
                scope.local_names.add(bound)
            return
        if isinstance(node, ast.Assign):
            for target in node.targets:
                scope.local_names.update(_assigned_name_targets(target))
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                scope.local_names.add(node.target.id)
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name):
                scope.local_names.add(node.target.id)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            scope.local_names.update(_assigned_name_targets(node.target))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    scope.local_names.update(_assigned_name_targets(item.optional_vars))
        elif isinstance(node, ast.ExceptHandler):
            if node.name is not None:
                scope.local_names.add(node.name)
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                scope.local_names.update(_assigned_name_targets(target))
        elif isinstance(node, ast.NamedExpr):
            _walrus_scope(scope).local_names.add(node.target.id)
        elif isinstance(node, ast.match_case):
            scope.local_names.update(_pattern_bound_names(node.pattern))

        for child in ast.iter_child_nodes(node):
            self._collect_decls(child, scope)

    # ------------------------------------------------------------------
    # Phase 1b: monotone fixed-point tag propagation.
    # ------------------------------------------------------------------

    def _record_alias_pair(self, lhs_scope: _Scope, lhs: str, rhs_scope: _Scope, rhs: str) -> None:
        self._scopes_by_id[id(lhs_scope)] = lhs_scope
        self._scopes_by_id[id(rhs_scope)] = rhs_scope
        lhs_key = (id(lhs_scope), lhs)
        rhs_key = (id(rhs_scope), rhs)
        lhs_peers = self._alias_peers.setdefault(lhs_key, set())
        rhs_peers = self._alias_peers.setdefault(rhs_key, set())
        if rhs_key not in lhs_peers:
            lhs_peers.add(rhs_key)
            self._bindings_changed = True
        if lhs_key not in rhs_peers:
            rhs_peers.add(lhs_key)
            self._bindings_changed = True

    def _propagate_mutation_tag(self, owner: _Scope, name: str, tag: str) -> None:
        self._scopes_by_id[id(owner)] = owner
        stack: list[tuple[int, str]] = [(id(owner), name)]
        visited: set[tuple[int, str]] = set()
        while stack:
            key = stack.pop()
            if key in visited:
                continue
            visited.add(key)
            scope_id, cur_name = key
            self._add_tag_in_scope(self._scopes_by_id[scope_id], cur_name, tag)
            for peer_key in self._alias_peers.get(key, ()):
                if peer_key not in visited:
                    stack.append(peer_key)

    def _bind_value_to_target(self, target: ast.AST, value: ast.AST, scope: _Scope) -> None:
        if isinstance(target, ast.Name):
            if isinstance(value, ast.Name):
                lhs_owner = self._target_scope(scope, target.id)
                rhs_owner = self._resolve_binding_scope(scope, value.id)
                self._record_alias_pair(lhs_owner, target.id, rhs_owner, value.id)
            for tag in self._expr_tags(value, scope):
                self._add_tag(scope, target.id, tag)
            return
        if (
            isinstance(target, (ast.Tuple, ast.List))
            and isinstance(value, (ast.Tuple, ast.List))
            and not any(isinstance(e, ast.Starred) for e in (*target.elts, *value.elts))
            and len(target.elts) == len(value.elts)
        ):
            for sub_target, sub_value in zip(target.elts, value.elts):
                self._bind_value_to_target(sub_target, sub_value, scope)

    def _propagate_arg_defaults(
        self, args: ast.arguments, outer_scope: _Scope, fn_scope: _Scope
    ) -> None:
        pos_params = (*args.posonlyargs, *args.args)
        if args.defaults:
            for param, default in zip(
                pos_params[len(pos_params) - len(args.defaults) :], args.defaults
            ):
                self._propagate(default, outer_scope)
                for tag in self._expr_tags(default, outer_scope):
                    self._add_tag(fn_scope, param.arg, tag)
        for param, default in zip(args.kwonlyargs, args.kw_defaults):
            if default is not None:
                self._propagate(default, outer_scope)
                for tag in self._expr_tags(default, outer_scope):
                    self._add_tag(fn_scope, param.arg, tag)

    def _propagate(self, node: ast.AST, scope: _Scope) -> None:
        if isinstance(node, ast.If) and self._is_type_checking_guard(node.test, scope):
            for stmt in node.orelse:
                self._propagate(stmt, scope)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in node.decorator_list:
                self._propagate(child, scope)
            fn_scope = self._scopes_by_node[id(node)]
            self._propagate_arg_defaults(node.args, scope, fn_scope)
            for stmt in node.body:
                self._propagate(stmt, fn_scope)
            return
        if isinstance(node, ast.Lambda):
            lam_scope = self._scopes_by_node[id(node)]
            self._propagate_arg_defaults(node.args, scope, lam_scope)
            self._propagate(node.body, lam_scope)
            return
        if isinstance(node, ast.ClassDef):
            for child in (*node.decorator_list, *node.bases):
                self._propagate(child, scope)
            for kw in node.keywords:
                self._propagate(kw.value, scope)
            cls_scope = self._scopes_by_node[id(node)]
            for stmt in node.body:
                self._propagate(stmt, cls_scope)
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            comp_scope = self._scopes_by_node[id(node)]
            self._propagate(node.generators[0].iter, scope)
            for gen_idx, gen in enumerate(node.generators):
                if gen_idx > 0:
                    self._propagate(gen.iter, comp_scope)
                for cond in gen.ifs:
                    self._propagate(cond, comp_scope)
            if isinstance(node, ast.DictComp):
                self._propagate(node.key, comp_scope)
                self._propagate(node.value, comp_scope)
            else:
                self._propagate(node.elt, comp_scope)
            return
        if isinstance(node, ast.Import):
            for bound, tag in self._import_tags(node):
                if tag is not None:
                    self._add_tag(scope, bound, tag)
            return
        if isinstance(node, ast.ImportFrom):
            for bound, tag in self._import_from_tags(node):
                if tag is not None:
                    self._add_tag(scope, bound, tag)
            return
        if isinstance(node, ast.Assign):
            self._propagate(node.value, scope)
            for target in node.targets:
                self._propagate(target, scope)
                self._bind_value_to_target(target, node.value, scope)
            return
        if isinstance(node, ast.AnnAssign):
            if node.value is not None:
                self._propagate(node.value, scope)
                self._bind_value_to_target(node.target, node.value, scope)
            return
        if isinstance(node, ast.AugAssign):
            self._propagate(node.value, scope)
            if isinstance(node.op, ast.BitOr) and isinstance(node.target, ast.Name):
                for tag in self._expr_tags(node.value, scope):
                    self._add_tag(scope, node.target.id, tag)
            return
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "update"
                and isinstance(node.func.value, ast.Name)
                and (
                    (node.args and _TAG_ENVIRON in self._expr_tags(node.args[0], scope))
                    or any(
                        kw.arg is None and _TAG_ENVIRON in self._expr_tags(kw.value, scope)
                        for kw in node.keywords
                    )
                )
            ):
                owner = self._resolve_binding_scope(scope, node.func.value.id)
                self._propagate_mutation_tag(owner, node.func.value.id, _TAG_ENVIRON)
        if isinstance(node, ast.NamedExpr):
            self._propagate(node.value, scope)
            target_scope = _walrus_scope(scope)
            for tag in self._expr_tags(node.value, scope):
                self._add_tag(target_scope, node.target.id, tag)
            return

        for child in ast.iter_child_nodes(node):
            self._propagate(child, scope)

    # ------------------------------------------------------------------
    # Phase 2: emit findings in lexical context.
    # ------------------------------------------------------------------

    def _add(self, node: ast.AST, *, kind: str = "read") -> None:
        self.findings.append(Finding(self._path, node.lineno, _text(self._source, node), kind))

    def _consumes_first_arg_iter(self, func: ast.expr, node: ast.Call, scope: _Scope) -> bool:
        if not node.args:
            return False
        if (
            isinstance(func, ast.Name)
            and func.id in _EAGER_ITER_CONSUMERS
            and not self._lookup_tags(scope, func.id)
        ):
            if func.id in ("min", "max"):
                return len(node.args) == 1
            return True
        return (
            isinstance(func, ast.Attribute)
            and func.attr == "join"
            and (
                (
                    isinstance(func.value, ast.Constant)
                    and isinstance(func.value.value, (str, bytes))
                )
                or isinstance(func.value, ast.JoinedStr)
            )
        )

    def _emit_comp(
        self,
        node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
        scope: _Scope,
        *,
        runtime_depth: int,
        in_annotation: bool,
        eager_genexp: bool = False,
    ) -> None:
        comp_scope = self._scopes_by_node[id(node)]
        inner_depth = (
            runtime_depth + 1
            if isinstance(node, ast.GeneratorExp) and not eager_genexp
            else runtime_depth
        )
        first_iter = node.generators[0].iter
        if inner_depth == 0 and isinstance(first_iter, ast.GeneratorExp):
            self._emit_comp(
                first_iter,
                scope,
                runtime_depth=runtime_depth,
                in_annotation=in_annotation,
                eager_genexp=True,
            )
        else:
            self._emit(
                first_iter,
                scope,
                runtime_depth=runtime_depth,
                in_annotation=in_annotation,
            )
        for gen_idx, gen in enumerate(node.generators):
            if gen_idx > 0:
                if inner_depth == 0 and isinstance(gen.iter, ast.GeneratorExp):
                    self._emit_comp(
                        gen.iter,
                        comp_scope,
                        runtime_depth=inner_depth,
                        in_annotation=in_annotation,
                        eager_genexp=True,
                    )
                else:
                    self._emit(
                        gen.iter,
                        comp_scope,
                        runtime_depth=inner_depth,
                        in_annotation=in_annotation,
                    )
            self._emit(
                gen.target, comp_scope, runtime_depth=inner_depth, in_annotation=in_annotation
            )
            for cond in gen.ifs:
                self._emit(cond, comp_scope, runtime_depth=inner_depth, in_annotation=in_annotation)
        if isinstance(node, ast.DictComp):
            self._emit(node.key, comp_scope, runtime_depth=inner_depth, in_annotation=in_annotation)
            self._emit(
                node.value, comp_scope, runtime_depth=inner_depth, in_annotation=in_annotation
            )
        else:
            self._emit(node.elt, comp_scope, runtime_depth=inner_depth, in_annotation=in_annotation)

    def _emit(
        self,
        node: ast.AST,
        scope: _Scope,
        *,
        runtime_depth: int,
        in_annotation: bool,
    ) -> None:
        if isinstance(node, ast.If) and self._is_type_checking_guard(node.test, scope):
            for stmt in node.orelse:
                self._emit(stmt, scope, runtime_depth=runtime_depth, in_annotation=in_annotation)
            return

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                self._emit(dec, scope, runtime_depth=runtime_depth, in_annotation=False)
            for default in node.args.defaults:
                self._emit(default, scope, runtime_depth=runtime_depth, in_annotation=False)
            for default in node.args.kw_defaults:
                if default is not None:
                    self._emit(default, scope, runtime_depth=runtime_depth, in_annotation=False)
            for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                if arg.annotation is not None:
                    self._emit(
                        arg.annotation, scope, runtime_depth=runtime_depth, in_annotation=True
                    )
            for extra in (node.args.vararg, node.args.kwarg):
                if extra is not None and extra.annotation is not None:
                    self._emit(
                        extra.annotation, scope, runtime_depth=runtime_depth, in_annotation=True
                    )
            if node.returns is not None:
                self._emit(node.returns, scope, runtime_depth=runtime_depth, in_annotation=True)
            fn_scope = self._scopes_by_node[id(node)]
            for stmt in node.body:
                self._emit(stmt, fn_scope, runtime_depth=runtime_depth + 1, in_annotation=False)
            return

        if isinstance(node, ast.Lambda):
            for default in node.args.defaults:
                self._emit(default, scope, runtime_depth=runtime_depth, in_annotation=False)
            for default in node.args.kw_defaults:
                if default is not None:
                    self._emit(default, scope, runtime_depth=runtime_depth, in_annotation=False)
            lam_scope = self._scopes_by_node[id(node)]
            self._emit(node.body, lam_scope, runtime_depth=runtime_depth + 1, in_annotation=False)
            return

        if isinstance(node, ast.ClassDef):
            for dec in node.decorator_list:
                self._emit(dec, scope, runtime_depth=runtime_depth, in_annotation=False)
            for base in node.bases:
                self._emit(base, scope, runtime_depth=runtime_depth, in_annotation=False)
            for kw in node.keywords:
                self._emit(kw.value, scope, runtime_depth=runtime_depth, in_annotation=False)
            cls_scope = self._scopes_by_node[id(node)]
            for stmt in node.body:
                self._emit(stmt, cls_scope, runtime_depth=runtime_depth, in_annotation=False)
            return

        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            self._emit_comp(node, scope, runtime_depth=runtime_depth, in_annotation=in_annotation)
            return

        if (
            isinstance(node, ast.Starred)
            and runtime_depth == 0
            and isinstance(node.value, ast.GeneratorExp)
        ):
            self._emit_comp(
                node.value,
                scope,
                runtime_depth=runtime_depth,
                in_annotation=in_annotation,
                eager_genexp=True,
            )
            return

        if (
            isinstance(node, (ast.For, ast.AsyncFor))
            and runtime_depth == 0
            and isinstance(node.iter, ast.GeneratorExp)
        ):
            self._emit(node.target, scope, runtime_depth=runtime_depth, in_annotation=False)
            self._emit_comp(
                node.iter,
                scope,
                runtime_depth=runtime_depth,
                in_annotation=False,
                eager_genexp=True,
            )
            for stmt in (*node.body, *node.orelse):
                self._emit(stmt, scope, runtime_depth=runtime_depth, in_annotation=False)
            return

        if (
            isinstance(node, ast.Assign)
            and runtime_depth == 0
            and isinstance(node.value, ast.GeneratorExp)
            and any(isinstance(t, (ast.Tuple, ast.List)) for t in node.targets)
        ):
            for target in node.targets:
                self._emit(target, scope, runtime_depth=runtime_depth, in_annotation=False)
            self._emit_comp(
                node.value,
                scope,
                runtime_depth=runtime_depth,
                in_annotation=False,
                eager_genexp=True,
            )
            return

        if isinstance(node, ast.AnnAssign):
            self._emit(node.target, scope, runtime_depth=runtime_depth, in_annotation=False)
            self._emit(node.annotation, scope, runtime_depth=runtime_depth, in_annotation=True)
            if node.value is not None:
                self._emit(node.value, scope, runtime_depth=runtime_depth, in_annotation=False)
            return

        if isinstance(node, ast.ImportFrom):
            if _is_envs_from_import(node, self._pkg_parts) and any(
                alias.name not in ENVS_IMPORTABLE_NAMES for alias in node.names
            ):
                self._add(node, kind="import")
            return

        if isinstance(node, ast.Call):
            func = node.func
            func_tags = self._expr_tags(func, scope)
            if _TAG_GETENV in func_tags:
                self._add(node)
            elif _TAG_ENVIRON_POP in func_tags and id(node) not in self._statement_calls:
                self._add(node)
            elif (
                runtime_depth == 0
                and not (in_annotation and self._future_annotations)
                and _TAG_CONSUME in func_tags
            ):
                self._add(node, kind="freeze")
            if runtime_depth == 0 and self._consumes_first_arg_iter(func, node, scope):
                self._emit(func, scope, runtime_depth=runtime_depth, in_annotation=in_annotation)
                for idx, arg in enumerate(node.args):
                    if idx == 0 and isinstance(arg, ast.GeneratorExp):
                        self._emit_comp(
                            arg,
                            scope,
                            runtime_depth=runtime_depth,
                            in_annotation=in_annotation,
                            eager_genexp=True,
                        )
                    else:
                        self._emit(
                            arg, scope, runtime_depth=runtime_depth, in_annotation=in_annotation
                        )
                for kw in node.keywords:
                    self._emit(
                        kw.value, scope, runtime_depth=runtime_depth, in_annotation=in_annotation
                    )
                return
        elif isinstance(node, ast.Subscript):
            if isinstance(node.ctx, ast.Load) and _TAG_ENVIRON in self._expr_tags(
                node.value, scope
            ):
                self._add(node)
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Subscript) and _TAG_ENVIRON in self._expr_tags(
                node.target.value, scope
            ):
                self._add(node.target)
        elif isinstance(node, ast.Compare):
            if any(
                isinstance(op, (ast.In, ast.NotIn))
                and _TAG_ENVIRON in self._expr_tags(comparator, scope)
                for op, comparator in zip(node.ops, node.comparators)
            ):
                self._add(node)
            if runtime_depth == 0 and any(
                isinstance(op, (ast.In, ast.NotIn)) and isinstance(comp, ast.GeneratorExp)
                for op, comp in zip(node.ops, node.comparators)
            ):
                self._emit(
                    node.left, scope, runtime_depth=runtime_depth, in_annotation=in_annotation
                )
                for op, comp in zip(node.ops, node.comparators):
                    if isinstance(op, (ast.In, ast.NotIn)) and isinstance(comp, ast.GeneratorExp):
                        self._emit_comp(
                            comp,
                            scope,
                            runtime_depth=runtime_depth,
                            in_annotation=in_annotation,
                            eager_genexp=True,
                        )
                    else:
                        self._emit(
                            comp, scope, runtime_depth=runtime_depth, in_annotation=in_annotation
                        )
                return
        elif isinstance(node, ast.Attribute):
            if _TAG_ENVS in self._expr_tags(node.value, scope):
                if node.attr in ENVS_IMPORTABLE_NAMES or (
                    node.attr.startswith("__") and node.attr.endswith("__")
                ):
                    pass
                elif node.attr not in self._declared_names:
                    self._add(node, kind="unknown")
                elif runtime_depth == 0 and not (in_annotation and self._future_annotations):
                    self._add(node, kind="freeze")

        for child in ast.iter_child_nodes(node):
            self._emit(child, scope, runtime_depth=runtime_depth, in_annotation=in_annotation)


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
