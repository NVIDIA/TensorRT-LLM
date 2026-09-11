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
"""Repository-wide static Python import and binding references."""

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


@dataclass(frozen=True)
class ImporterFacts:
    """Direct importers and whether static discovery was complete."""

    paths: tuple[str, ...]
    complete: bool
    limitation: str = ""


class RepositoryReferenceIndex:
    """Index statically visible import and binding relationships."""

    def __init__(self, repo_root: Path, *, module_prefixes: tuple[str, ...] = ()) -> None:
        self.repo_root = Path(repo_root)
        self._module_prefixes = tuple(prefix.rstrip(".") for prefix in module_prefixes)
        self._module_cache: dict[str, list[_ParsedModule]] = {}
        self._importer_cache: dict[str, ImporterFacts] = {}

    def _normalize_module(self, module: str) -> str:
        for prefix in self._module_prefixes:
            if module == prefix:
                return ""
            dotted_prefix = f"{prefix}."
            if module.startswith(dotted_prefix):
                return module.removeprefix(dotted_prefix)
        return module

    @staticmethod
    def _imported_modules(node: ast.Import | ast.ImportFrom, source: _ParsedModule) -> set[str]:
        if isinstance(node, ast.Import):
            return {alias.name for alias in node.names}
        base = _resolve_from(source.module, source.is_package, node)
        modules = {base} if base else set()
        modules.update(
            f"{base}.{alias.name}" if base else alias.name
            for alias in node.names
            if alias.name != "*"
        )
        return modules

    @staticmethod
    def _is_dynamic_import(node: ast.Call) -> bool:
        callee = _dotted_name(node.func)
        return bool(callee and callee.rpartition(".")[2] in {"__import__", "import_module"})

    def direct_importers(self, defining_path: str) -> ImporterFacts:
        """Return direct Python importers, or an incomplete result on ambiguity."""
        if defining_path in self._importer_cache:
            return self._importer_cache[defining_path]

        if not (self.repo_root / defining_path).is_file():
            result = ImporterFacts((), False, f"defining source missing: {defining_path}")
            self._importer_cache[defining_path] = result
            return result

        target_module, _ = _module_name(defining_path)
        target_parent, _, target_leaf = target_module.rpartition(".")
        importers: list[str] = []
        for source_path in sorted(self.repo_root.rglob("*.py")):
            try:
                path = source_path.relative_to(self.repo_root).as_posix()
                source = source_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError, ValueError) as error:
                result = ImporterFacts((), False, f"source read failed: {error}")
                self._importer_cache[defining_path] = result
                return result
            if path == defining_path:
                continue
            module, is_package = _module_name(path)
            try:
                parsed = _ParsedModule(path, module, is_package, source, ast.parse(source))
            except SyntaxError:
                result = ImporterFacts((), False, f"unparsable source: {path}")
                self._importer_cache[defining_path] = result
                return result

            imports_target = False
            for node in ast.walk(parsed.tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for imported in self._imported_modules(node, parsed):
                        normalized = self._normalize_module(imported)
                        if normalized == target_module:
                            imports_target = True
                        elif normalized == target_leaf and target_parent:
                            importer_parent = module.rpartition(".")[0]
                            if importer_parent == target_parent:
                                imports_target = True
                            else:
                                result = ImporterFacts(
                                    (), False, f"ambiguous short import in {path}: {imported}"
                                )
                                self._importer_cache[defining_path] = result
                                return result
                elif isinstance(node, ast.Call) and self._is_dynamic_import(node):
                    argument = node.args[0] if node.args else None
                    if (
                        not isinstance(argument, ast.Constant)
                        or not isinstance(argument.value, str)
                        or target_leaf in argument.value
                    ):
                        result = ImporterFacts(
                            (), False, f"dynamic import may target {defining_path}"
                        )
                        self._importer_cache[defining_path] = result
                        return result

            if target_leaf in source and not imports_target:
                result = ImporterFacts((), False, f"unresolved module reference in {path}")
                self._importer_cache[defining_path] = result
                return result
            if imports_target:
                importers.append(path)

        result = ImporterFacts(tuple(importers), True)
        self._importer_cache[defining_path] = result
        return result

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
                # Candidate discovery already found the target module's leaf.
                # Without an AST, no binding can be excluded soundly.
                referenced.update(names)
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
