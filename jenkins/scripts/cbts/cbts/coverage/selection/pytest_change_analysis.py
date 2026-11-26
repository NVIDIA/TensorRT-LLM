# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""AST bounds for pytest code that executes less often than its consumers."""

from __future__ import annotations

import ast
from pathlib import Path

from cbts.repository_reference import RepositoryReferenceIndex

_PER_TEST_HOOKS = {"pytest_runtest_setup", "pytest_runtest_call", "pytest_runtest_teardown"}
_CACHE_DECORATORS = {"cache", "lru_cache", "cached_property"}
_PER_CALL_DECORATORS = {
    "staticmethod",
    "classmethod",
    "property",
    "wraps",
    "contextmanager",
    "asynccontextmanager",
    "hookimpl",
}
_MUTATORS = {"append", "extend", "insert", "pop", "clear", "update", "setdefault", "add", "remove"}


def _leaf(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _references(
    node: ast.AST, class_prefix: str = "", methods: dict[str, set[str]] | None = None
) -> set[str]:
    """Include fixture injection, attribute access, and ordinary Python references."""
    refs = {
        child.id if isinstance(child, ast.Name) else child.attr
        for child in ast.walk(node)
        if isinstance(child, (ast.Name, ast.Attribute))
    } | {child.arg for child in ast.walk(node) if isinstance(child, ast.arg)}
    for child in ast.walk(node):
        if not isinstance(child, ast.Attribute):
            continue
        parts = [child.attr]
        base = child.value
        while isinstance(base, ast.Attribute):
            parts.append(base.attr)
            base = base.value
        if isinstance(base, ast.Name):
            parts.append(base.id)
            dotted = ".".join(reversed(parts))
            if base.id in {"self", "cls"} and class_prefix:
                refs.add(class_prefix + "." + ".".join(reversed(parts[:-1])))
            else:
                refs.add(dotted)
        # Aliases, super(), inherited fixtures and dynamic self/cls dispatch can
        # target another class's implementation. Keep all same-named methods as
        # possible dependencies rather than assuming an unresolved edge is absent.
        if methods:
            refs.update(methods.get(child.attr, set()))
    return refs


def _shared_reason(node: ast.FunctionDef | ast.AsyncFunctionDef, aliases: dict[str, str]) -> str:
    if node.name.startswith("pytest_") and node.name not in _PER_TEST_HOOKS:
        return "pytest session/collection hook"
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        name = aliases.get(_leaf(target), _leaf(target))
        if name in _CACHE_DECORATORS:
            return "cached helper"
        if name not in {"fixture", "yield_fixture"}:
            if name in _PER_CALL_DECORATORS or (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Attribute)
                and target.value.attr == "mark"
            ):
                continue
            return "helper with unresolved decorator"
        if not isinstance(decorator, ast.Call):
            continue  # @pytest.fixture has the default function scope.
        scope: ast.expr = ast.Constant("function")
        for keyword in decorator.keywords:
            if keyword.arg == "scope":
                scope = keyword.value
            elif keyword.arg is None:
                return "fixture with unresolved scope"
        if not isinstance(scope, ast.Constant) or scope.value != "function":
            return "shared fixture"
    # A cached helper can also keep state explicitly instead of using functools.
    local = {arg.arg for arg in ast.walk(node.args) if isinstance(arg, ast.arg)}
    local |= {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
    }
    for child in ast.walk(node):
        if isinstance(child, (ast.Global, ast.Nonlocal)):
            return "stateful helper"
        if isinstance(child, (ast.Attribute, ast.Subscript)) and isinstance(child.ctx, ast.Store):
            base = child.value
            while isinstance(base, (ast.Attribute, ast.Subscript)):
                base = base.value
            if isinstance(base, ast.Name) and (base.id not in local or base.id in {"self", "cls"}):
                return "stateful helper"
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            base = child.func.value
            if child.func.attr in _MUTATORS and isinstance(base, ast.Name) and base.id not in local:
                return "stateful helper"
    return ""


def _facts(source: str) -> tuple[dict[str, set[str]], dict[str, str]]:
    tree = ast.parse(source)
    aliases = {
        alias.asname or alias.name: alias.name.rpartition(".")[2]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    # Resolve simple decorator aliases without executing module-level code.
    for _ in range(len(tree.body)):
        previous = dict(aliases)
        for statement in tree.body:
            if isinstance(statement, ast.Assign) and isinstance(
                statement.value, (ast.Name, ast.Attribute)
            ):
                target_name = _leaf(statement.value)
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        aliases[target.id] = aliases.get(target_name, target_name)
        if previous == aliases:
            break
    nodes: dict[str, tuple[ast.AST, str]] = {}
    methods: dict[str, set[str]] = {}
    bases: dict[str, set[str]] = {}
    shared: dict[str, str] = {}

    def walk(
        node: ast.AST, prefix: str = "", class_prefix: str = "", in_function: bool = False
    ) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + child.name
                nodes[name] = (child, class_prefix)
                if class_prefix and isinstance(node, ast.ClassDef):
                    methods.setdefault(child.name, set()).add(name)
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    reason = _shared_reason(child, aliases)
                    if reason:
                        shared[name] = reason
                is_class = isinstance(child, ast.ClassDef)
                if is_class:
                    bases[name] = {ref for base in child.bases for ref in _references(base)}
                walk(
                    child,
                    name + ("." if is_class else ".<locals>."),
                    name if is_class else class_prefix,
                    in_function or not is_class,
                )
            elif isinstance(child, (ast.Assign, ast.AnnAssign)):
                targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                for target in targets:
                    for binding in ast.walk(target):
                        if isinstance(binding, ast.Name) and isinstance(binding.ctx, ast.Store):
                            name = prefix + binding.id
                            nodes[name] = (child, class_prefix)
                            if (
                                not in_function
                                and child.value is not None
                                and _references(child.value)
                            ):
                                shared[name] = (
                                    "class initialization result"
                                    if class_prefix
                                    else "module initialization result"
                                )
            else:
                walk(child, prefix, class_prefix, in_function)

    walk(tree)
    # An imported/aliased base may provide the implementation called through
    # self/cls; its body is not necessarily present in this module's graph.
    graph = {
        name: _references(node, owner, methods) | bases.get(owner, set())
        for name, (node, owner) in nodes.items()
    }
    return graph, shared


class PytestImpactIndex:
    """Decline only changes reaching shared pytest state, including imported helpers."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.references = RepositoryReferenceIndex(repo_root)
        self._sources: dict[str, str | None] = {}

    def decline_reason(
        self, path: str, source: str, changed: set[str], *, pre_source: str | None = None
    ) -> str:
        """Check both source images so removing a decorator cannot hide shared scope."""
        # Product functions already use the existing coverage policy. The added
        # repository-wide eligibility needs these extra test-infrastructure bounds.
        if path.startswith("tensorrt_llm/"):
            return ""
        for image in (source, pre_source):
            if image is None:
                continue
            reason = self._check(path, image, changed - {"<module>"}, set())
            if reason:
                return f"shared test infrastructure change: {reason}"
        return ""

    def _check(
        self, path: str, source: str, changed: set[str], visited: set[tuple[str, str]]
    ) -> str:
        fresh = {name for name in changed if (path, name) not in visited}
        if not fresh:
            return ""
        visited.update((path, name) for name in fresh)
        try:
            graph, shared = _facts(source)
        except SyntaxError:
            return f"unparsable dependency: {path}"
        affected = set(fresh)
        # Compute callers as well as fixture-parameter and module-alias dependencies.
        while True:
            wider = affected | {
                name
                for name, refs in graph.items()
                if refs & affected or any(item.startswith(name + ".") for item in affected)
            }
            if wider == affected:
                break
            affected = wider
        for name in sorted(affected & shared.keys()):
            return f"{path}::{name} ({shared[name]})"
        top = {name for name in affected if "." not in name}
        if not top or not (self.repo_root / path).is_file():
            return ""
        external = self.references.external_reference_paths(path, top)
        for consumer in sorted({p for paths in external.values() for p in paths}):
            if consumer.startswith("tensorrt_llm/"):
                continue
            if consumer not in self._sources:
                try:
                    self._sources[consumer] = (self.repo_root / consumer).read_text(
                        encoding="utf-8"
                    )
                except (OSError, UnicodeDecodeError):
                    self._sources[consumer] = None
            consumer_source = self._sources[consumer]
            if consumer_source is None:
                return f"unreadable dependency: {consumer}"
            names = {name for name, paths in external.items() if consumer in paths}
            try:
                tree = ast.parse(consumer_source)
            except SyntaxError:
                return f"unparsable dependency: {consumer}"
            # Preserve aliases and module-object imports conservatively; the
            # reference index already established this file references the owner.
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        if alias.name in names or path.rsplit("/", 1)[-1][:-3] in alias.name.split(
                            "."
                        ):
                            names.add(alias.asname or alias.name.split(".")[0])
            reason = self._check(consumer, consumer_source, names, visited)
            if reason:
                return reason
        return ""
