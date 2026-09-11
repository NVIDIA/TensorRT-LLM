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
"""Repository-wide static references to Python module bindings."""

from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass
from pathlib import Path


def _module_name(path: str) -> tuple[str, bool]:
    parts = list(Path(path).with_suffix("").parts)
    is_package = bool(parts and parts[-1] == "__init__")
    if is_package:
        parts.pop()
    return ".".join(parts), is_package


def _dotted_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else None
    return None


def _resolve_from(module: str, is_package: bool, node: ast.ImportFrom) -> str:
    if not node.level:
        return node.module or ""
    package = module.split(".") if is_package else module.split(".")[:-1]
    trim = node.level - 1
    base = package[: len(package) - trim] if trim <= len(package) else []
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


@dataclass(frozen=True)
class _ParsedModule:
    path: str
    module: str
    is_package: bool
    source: str
    tree: ast.Module | None


class RepositoryReferenceIndex:
    """Find statically visible references outside a binding's defining file."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = Path(repo_root)
        self._module_cache: dict[str, list[_ParsedModule]] = {}

    def _modules_containing(self, text: str) -> list[_ParsedModule]:
        if text in self._module_cache:
            return self._module_cache[text]
        paths: list[Path] | None = None
        try:
            result = subprocess.run(
                ["git", "grep", "-l", "-z", "-F", text, "--", "*.py"],
                cwd=self.repo_root,
                capture_output=True,
                check=False,
            )
            if result.returncode in (0, 1):
                paths = [
                    self.repo_root / path.decode() for path in result.stdout.split(b"\0") if path
                ]
        except (OSError, UnicodeDecodeError):
            pass
        if paths is None:
            paths = list(self.repo_root.rglob("*.py"))

        modules: list[_ParsedModule] = []
        for path in sorted(paths):
            try:
                relative = path.relative_to(self.repo_root).as_posix()
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError, ValueError):
                continue
            if text not in source:
                continue
            module, is_package = _module_name(relative)
            modules.append(_ParsedModule(relative, module, is_package, source, None))
        self._module_cache[text] = modules
        return modules

    def external_references(self, defining_path: str, names: set[str]) -> set[str]:
        """Return bindings referenced from another repository Python file."""
        target_module, _ = _module_name(defining_path)
        target_leaf = target_module.rpartition(".")[2]
        referenced: set[str] = set()
        for module in self._modules_containing(target_leaf):
            if module.path == defining_path or referenced == names:
                continue
            try:
                parsed = _ParsedModule(
                    module.path,
                    module.module,
                    module.is_package,
                    module.source,
                    ast.parse(module.source),
                )
            except SyntaxError:
                if target_module in module.source:
                    referenced.update(name for name in names if name in module.source)
                continue
            referenced.update(
                self._references_from_module(parsed, target_module, names - referenced)
            )
        return referenced

    @staticmethod
    def _references_from_module(
        source_module: _ParsedModule, target_module: str, names: set[str]
    ) -> set[str]:
        tree = source_module.tree
        if tree is None or not names:
            return set()

        module_aliases: set[str] = set()
        referenced: set[str] = set()
        dynamically_imported = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == target_module:
                        module_aliases.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom):
                imported_from = _resolve_from(source_module.module, source_module.is_package, node)
                if imported_from == target_module:
                    imported_names = {alias.name for alias in node.names}
                    if "*" in imported_names:
                        referenced.update(names)
                    else:
                        referenced.update(names & imported_names)
                for alias in node.names:
                    if f"{imported_from}.{alias.name}" == target_module:
                        module_aliases.add(alias.asname or alias.name)
            elif isinstance(node, ast.Call) and node.args:
                callee = _dotted_name(node.func)
                first = node.args[0]
                if (
                    callee in {"__import__", "importlib.import_module"}
                    and isinstance(first, ast.Constant)
                    and first.value == target_module
                ):
                    dynamically_imported = True

        if dynamically_imported:
            return set(names)
        if not module_aliases:
            return referenced

        # Follow direct aliases such as ``module_alias = imported_module``.
        changed = True
        while changed:
            changed = False
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                value = node.value
                if value is None or _dotted_name(value) not in module_aliases:
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and target.id not in module_aliases:
                        module_aliases.add(target.id)
                        changed = True

        parents = {
            id(child): parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                dotted = _dotted_name(node)
                for alias in module_aliases:
                    prefix = f"{alias}."
                    if dotted and dotted.startswith(prefix):
                        referenced_name = dotted.removeprefix(prefix).partition(".")[0]
                        if referenced_name in names:
                            referenced.add(referenced_name)
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in names
            ):
                # Covers getattr/module-dict access once the target module is in scope.
                referenced.add(node.value)
            if isinstance(node, (ast.Name, ast.Attribute)):
                if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
                    continue
                dotted = _dotted_name(node)
                if dotted not in module_aliases:
                    continue
                parent = parents.get(id(node))
                if isinstance(parent, ast.Attribute) and parent.value is node:
                    continue
                if isinstance(parent, (ast.Assign, ast.AnnAssign)) and parent.value is node:
                    targets = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
                    if all(isinstance(target, ast.Name) for target in targets):
                        continue
                if isinstance(parent, ast.Call) and node in parent.args:
                    position = parent.args.index(node)
                    callee = _dotted_name(parent.func)
                    if (
                        callee
                        and callee.rpartition(".")[2]
                        in {"getattr", "setattr", "delattr", "hasattr"}
                        and position + 1 < len(parent.args)
                        and isinstance(parent.args[position + 1], ast.Constant)
                        and isinstance(parent.args[position + 1].value, str)
                    ):
                        referenced_name = parent.args[position + 1].value
                        if referenced_name in names:
                            referenced.add(referenced_name)
                        continue
                # Passing or storing the module object can create aliases that
                # this index cannot follow soundly.
                return set(names)
        return referenced
