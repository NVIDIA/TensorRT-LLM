#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Apply path, symbol, and Pydantic alias filters to Pyrefly JSON reports."""

import argparse
import ast
import json
import os
import re
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

NAME_PATTERN = re.compile(r"`([^`]+)`|\"([^\"]+)\"")
SYMBOL_PATTERN = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?:\.\*)?")


def load_exclusions(config_path: Path) -> tuple[str, ...]:
    """Load and validate custom symbol exclusions from pyproject.toml."""
    config = tomllib.loads(config_path.read_text())
    exclusions = config.get("tool", {}).get("pyrefly-exclusions")
    if not isinstance(exclusions, dict) or set(exclusions) != {"symbols"}:
        raise ValueError(f"{config_path}: expected [tool.pyrefly-exclusions] with a symbols array")
    entries = exclusions["symbols"]
    if not isinstance(entries, list) or any(
        not isinstance(entry, str) or not entry or entry != entry.strip() for entry in entries
    ):
        raise ValueError(f"{config_path}: symbols must contain nonempty strings")
    if len(entries) != len(set(entries)):
        raise ValueError(f"{config_path}: duplicate entry in symbols")
    for pattern in entries:
        if SYMBOL_PATTERN.fullmatch(pattern) is None:
            raise ValueError(f"{config_path}: invalid symbol pattern: {pattern}")
    return tuple(entries)


def load_excluded_paths(config_path: Path) -> set[Path]:
    """Expand project-excludes globs relative to the Pyrefly configuration."""
    config = tomllib.loads(config_path.read_text())
    patterns = config.get("tool", {}).get("pyrefly", {}).get("project-excludes", [])
    if not isinstance(patterns, list) or any(
        not isinstance(pattern, str) or not pattern for pattern in patterns
    ):
        raise ValueError(f"{config_path}: project-excludes must contain nonempty strings")
    return {path.resolve() for pattern in patterns for path in config_path.parent.glob(pattern)}


def relative_source_path(source_path: str, project_root: Path) -> Path | None:
    """Return a source path relative to the project root, if it is inside it."""
    path = Path(source_path)
    resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    try:
        return resolved.relative_to(project_root.resolve())
    except ValueError:
        return None


def is_excluded_path(source_path: str, project_root: Path, excluded_paths: set[Path]) -> bool:
    relative_path = relative_source_path(source_path, project_root)
    if relative_path is None:
        return False
    path = (project_root / relative_path).resolve()
    return path in excluded_paths or any(parent in excluded_paths for parent in path.parents)


def filter_input_paths(
    input_path: Path, output_path: Path, config_path: Path, project_root: Path
) -> None:
    """Keep supplied paths that are not excluded, preserving NUL separators."""
    excluded_paths = load_excluded_paths(config_path)
    paths = (os.fsdecode(path) for path in input_path.read_bytes().split(b"\0") if path)
    output_path.write_bytes(
        b"".join(
            os.fsencode(path) + b"\0"
            for path in paths
            if not is_excluded_path(path, project_root, excluded_paths)
        )
    )


class ImportScope:
    """Imports and local bindings in a Python lexical scope."""

    def __init__(self, node: ast.AST) -> None:
        self.start = getattr(node, "lineno", 0)
        self.end = getattr(node, "end_lineno", 2**31)
        self.imports: dict[str, list[tuple[int, str]]] = {}
        self.other_bindings: set[str] = set()
        self.children: list[ImportScope] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            self.other_bindings.update(
                argument.arg for argument in ast.walk(node.args) if isinstance(argument, ast.arg)
            )

        def visit(child: ast.AST) -> None:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                if hasattr(child, "name"):
                    self.other_bindings.add(child.name)
                self.children.append(ImportScope(child))
                return
            if isinstance(child, ast.Import):
                for alias in child.names:
                    local_name = alias.asname or alias.name.split(".")[0]
                    target = alias.name if alias.asname else local_name
                    self.imports.setdefault(local_name, []).append((child.lineno, target))
                return
            if isinstance(child, ast.ImportFrom):
                for alias in child.names:
                    if alias.name == "*":
                        continue
                    local_name = alias.asname or alias.name
                    if child.level == 0 and child.module:
                        target = f"{child.module}.{alias.name}"
                        self.imports.setdefault(local_name, []).append((child.lineno, target))
                    else:
                        self.other_bindings.add(local_name)
                return
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                self.other_bindings.add(child.id)
            elif isinstance(child, ast.ExceptHandler) and child.name:
                self.other_bindings.add(child.name)
            for descendant in ast.iter_child_nodes(child):
                visit(descendant)

        for child in ast.iter_child_nodes(node):
            visit(child)

    def resolve(self, name: str, line: int) -> str | None:
        scopes = [self]
        while True:
            child = next(
                (item for item in scopes[-1].children if item.start <= line <= item.end), None
            )
            if child is None:
                break
            scopes.append(child)
        for scope in reversed(scopes):
            if name in scope.other_bindings:
                return None
            imports = scope.imports.get(name)
            if imports is not None:
                targets = {target for _, target in imports}
                if len(targets) == 1 and any(import_line <= line for import_line, _ in imports):
                    return next(iter(targets))
                return None
        return None


def load_import_scope(source_path: Path) -> ImportScope | None:
    try:
        return ImportScope(ast.parse(source_path.read_text()))
    except (OSError, SyntaxError, UnicodeError):
        return None


def matches_symbol(name: str, patterns: tuple[str, ...]) -> bool:
    exact = {pattern for pattern in patterns if not pattern.endswith(".*")}
    prefixes = tuple(pattern[:-1] for pattern in patterns if pattern.endswith(".*"))
    return name in exact or name.startswith(prefixes)


def matches_symbol_reference(name: str, patterns: tuple[str, ...]) -> bool:
    """Match a symbol in a qualified name, including callable descriptions."""
    if matches_symbol(name, patterns):
        return True
    parts = name.split(".")
    for pattern in patterns:
        if pattern.endswith(".*"):
            prefix = pattern[:-2].split(".")
            if any(
                parts[index : index + len(prefix)] == prefix and len(parts) > index + len(prefix)
                for index in range(len(parts))
            ):
                return True
            continue
        target = pattern.split(".")
        if any(parts[index : index + len(target)] == target for index in range(len(parts))):
            return True
    return False


def is_excluded_symbol(
    error: dict[str, Any],
    patterns: tuple[str, ...],
    project_root: Path,
    scopes_by_path: dict[str, ImportScope | None],
) -> bool:
    description = str(error.get("concise_description") or error.get("description", ""))
    line = error.get("line")
    path = str(error.get("path", ""))
    for first, second in NAME_PATTERN.findall(description):
        name = first or second
        if matches_symbol_reference(name, patterns):
            return True
        if error.get("name") == "missing-import" and any(
            pattern.startswith(f"{name}.") for pattern in patterns
        ):
            return True
        if error.get("name") not in (
            "missing-import",
            "missing-module-attribute",
            "unknown-name",
        ):
            continue
        local_name, separator, remainder = name.partition(".")
        if path not in scopes_by_path:
            source_path = relative_source_path(path, project_root)
            scopes_by_path[path] = (
                load_import_scope(project_root / source_path) if source_path is not None else None
            )
        scope = scopes_by_path[path]
        imported_name = scope.resolve(local_name, line) if scope and isinstance(line, int) else None
        resolved_name = imported_name + separator + remainder if imported_name else None
        if resolved_name and matches_symbol_reference(resolved_name, patterns):
            return True
    return False


class PydanticAliasFilter:
    """Recognize explicitly supplied validation aliases without importing project code."""

    def __init__(self, project_root: Path) -> None:
        self.root = project_root.resolve()
        self.modules: dict[str, ast.Module | None] = {}
        self.scopes: dict[str, ImportScope] = {}

    def _module(self, name: str) -> ast.Module | None:
        if name not in self.modules:
            self.modules[name] = None
            if re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", name):
                base = self.root.joinpath(*name.split("."))
                for path in (base.with_suffix(".py"), base / "__init__.py"):
                    if relative_source_path(str(path), self.root) is None:
                        continue
                    try:
                        tree = ast.parse(path.read_text())
                    except (OSError, SyntaxError, UnicodeError):
                        continue
                    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ImportFrom) and node.level:
                            parts = package.split(".") if package else []
                            if node.level <= len(parts):
                                prefix = parts[: len(parts) - node.level + 1]
                                node.module = ".".join(
                                    prefix + ([node.module] if node.module else [])
                                )
                                node.level = 0
                    self.modules[name] = tree
                    self.scopes[name] = ImportScope(tree)
                    break
        return self.modules[name]

    def _resolve(self, node: ast.expr, module: str) -> str | None:
        if isinstance(node, ast.Attribute):
            parent = self._resolve(node.value, module)
            return f"{parent}.{node.attr}" if parent else None
        if not isinstance(node, ast.Name):
            return None
        tree = self._module(module)
        if tree is None:
            return None
        imported = self.scopes[module].resolve(node.id, node.lineno)
        if imported:
            return imported
        definitions = [
            item
            for item in tree.body
            if isinstance(item, (ast.ClassDef, ast.FunctionDef)) and item.name == node.id
        ]
        # A rebound class name is ambiguous, so leave its diagnostics untouched.
        if len(definitions) == 1 and not any(
            (isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store) and item.id == node.id)
            or (isinstance(item, ast.arg) and item.arg == node.id)
            for item in ast.walk(tree)
        ):
            return f"{module}.{node.id}"
        return None

    def _lineage(
        self, symbol: str, seen: frozenset[str] = frozenset()
    ) -> list[tuple[str, ast.ClassDef]] | None:
        if symbol in ("pydantic.BaseModel", "pydantic.main.BaseModel"):
            return []
        if symbol in seen:
            return None
        module, _, name = symbol.rpartition(".")
        tree = self._module(module)
        if tree is None:
            return None
        definitions = [
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name
        ]
        if len(definitions) != 1:
            return None
        cls = definitions[0]
        # Decorators, multiple inheritance, and custom constructors can change
        # which keywords are accepted at runtime.
        if len(cls.bases) != 1 or cls.decorator_list:
            return None
        for node in cls.body:
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "__init__"
            ):
                return None
        base = self._resolve(cls.bases[0], module)
        parents = self._lineage(base, seen | {symbol}) if base else None
        return [(module, cls), *parents] if parents is not None else None

    @staticmethod
    def _literal_mapping(node: ast.expr | None) -> dict[str | int, ast.expr] | None:
        if not isinstance(node, ast.Dict) or any(
            not isinstance(key, ast.Constant) or type(key.value) not in (str, int)
            for key in node.keys
        ):
            return None
        return {key.value: value for key, value in zip(node.keys, node.values)}

    def _aliases_enabled(self, lineage: list[tuple[str, ast.ClassDef]]) -> bool:
        config: dict[str, ast.expr] = {}
        for module, cls in reversed(lineage):
            for node in cls.body:
                if isinstance(node, ast.ClassDef) and node.name == "Config":
                    if node.bases or node.decorator_list or node.keywords:
                        return False
                    for setting in node.body:
                        if not isinstance(setting, ast.Assign) or len(setting.targets) != 1:
                            return False
                        target = setting.targets[0]
                        if not isinstance(target, ast.Name):
                            return False
                        config[target.id] = setting.value
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if not any(
                    isinstance(target, ast.Name) and target.id == "model_config"
                    for target in targets
                ):
                    continue
                value = node.value
                if isinstance(value, ast.Call) and self._resolve(value.func, module) in (
                    "pydantic.ConfigDict",
                    "pydantic.config.ConfigDict",
                ):
                    if value.args or any(keyword.arg is None for keyword in value.keywords):
                        return False
                    config.update({keyword.arg: keyword.value for keyword in value.keywords})
                else:
                    mapping = self._literal_mapping(value)
                    if mapping is None or any(not isinstance(key, str) for key in mapping):
                        return False
                    config.update(mapping)
            if any(keyword.arg is None for keyword in cls.keywords):
                return False
            config.update({keyword.arg: keyword.value for keyword in cls.keywords})
        by_alias = config.get("validate_by_alias", ast.Constant(value=True))
        generator = config.get("alias_generator", ast.Constant(value=None))
        # Config.fields and generators can replace explicit field aliases.
        return (
            isinstance(by_alias, ast.Constant)
            and by_alias.value is True
            and isinstance(generator, ast.Constant)
            and generator.value is None
            and "fields" not in config
        )

    def _alias_paths(self, node: ast.expr, module: str) -> set[tuple[str | int, ...]]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {(node.value,)}
        if not isinstance(node, ast.Call) or node.keywords:
            return set()
        constructor = self._resolve(node.func, module)
        if constructor in ("pydantic.AliasChoices", "pydantic.aliases.AliasChoices"):
            return {path for choice in node.args for path in self._alias_paths(choice, module)}
        if constructor in ("pydantic.AliasPath", "pydantic.aliases.AliasPath") and node.args:
            try:
                path = tuple(ast.literal_eval(part) for part in node.args)
            except (ValueError, TypeError, SyntaxError):
                return set()
            if isinstance(path[0], str) and all(type(part) in (str, int) for part in path):
                return {path}
        return set()

    def _aliases(self, symbol: str, field: str) -> set[tuple[str | int, ...]]:
        lineage = self._lineage(symbol)
        if lineage is None or not self._aliases_enabled(lineage):
            return set()
        for module, cls in lineage:
            for node in reversed(cls.body):
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == field for target in node.targets
                ):
                    return set()
                if not (
                    isinstance(node, ast.AnnAssign)
                    and isinstance(node.target, ast.Name)
                    and node.target.id == field
                ):
                    continue
                value = node.value
                if not isinstance(value, ast.Call) or self._resolve(value.func, module) not in (
                    "pydantic.Field",
                    "pydantic.fields.Field",
                    # This project wrapper forwards aliases to PydanticField unchanged.
                    "tensorrt_llm.llmapi.llm_args.Field",
                ):
                    return set()
                if any(keyword.arg is None for keyword in value.keywords):
                    return set()
                keywords = {keyword.arg: keyword.value for keyword in value.keywords}
                alias = keywords.get("validation_alias")
                if alias is not None and not (
                    isinstance(alias, ast.Constant) and alias.value is None
                ):
                    return self._alias_paths(alias, module)
                alias = keywords.get("alias")
                # When validation_alias is unset, Pydantic uses alias for validation too.
                return self._alias_paths(alias, module) if alias is not None else set()
        return set()

    def _has_path(self, value: ast.expr, path: tuple[str | int, ...]) -> bool:
        if not path:
            return True
        key, *rest = path
        mapping = self._literal_mapping(value)
        if mapping is not None:
            return key in mapping and self._has_path(mapping[key], tuple(rest))
        if isinstance(value, (ast.List, ast.Tuple)) and isinstance(key, int):
            if any(isinstance(element, ast.Starred) for element in value.elts):
                return False
            if -len(value.elts) <= key < len(value.elts):
                return self._has_path(value.elts[key], tuple(rest))
        return False

    def _supplied_alias(self, call: ast.Call, paths: set[tuple[str | int, ...]]) -> bool:
        supplied: dict[str, ast.expr] = {}
        for keyword in call.keywords:
            if keyword.arg is not None:
                supplied[keyword.arg] = keyword.value
            else:
                mapping = self._literal_mapping(keyword.value)
                if mapping is None or any(not isinstance(key, str) for key in mapping):
                    return False
                supplied.update(mapping)
        return any(
            path[0] in supplied and self._has_path(supplied[path[0]], path[1:]) for path in paths
        )

    def matches(self, error: dict[str, Any]) -> bool:
        """Suppress constructor-name diagnostics only for statically verified input aliases."""
        kind = error.get("name")
        if kind not in ("missing-argument", "unexpected-keyword"):
            return False
        message = str(error.get("concise_description") or error.get("description", ""))
        prefix = "Missing argument" if kind == "missing-argument" else "Unexpected keyword argument"
        match = re.fullmatch(rf"{prefix} `([\w]+)` in function `([\w.]+)\.__init__`", message)
        if match is None:
            return False
        field, symbol = match.groups()
        if kind == "missing-argument":
            aliases = self._aliases(symbol, field)
        else:
            lineage = self._lineage(symbol)
            if lineage is None:
                return False
            fields = {
                node.target.id
                for _, cls in lineage
                for node in cls.body
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
            }
            aliases = {
                path for name in fields for path in self._aliases(symbol, name) if path[0] == field
            }
        if not aliases:
            return False
        path = relative_source_path(str(error.get("path", "")), self.root)
        if path is None or path.suffix != ".py":
            return False
        module = ".".join(path.with_suffix("").parts)
        if module.endswith(".__init__"):
            module = module.removesuffix(".__init__")
        tree = self._module(module)
        if tree is None:
            return False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if kind == "missing-argument":
                # Pyrefly's missing-argument span starts at the opening parenthesis.
                if (node.func.end_lineno, node.func.end_col_offset + 1) != (
                    error.get("line"),
                    error.get("column"),
                ) or (node.end_lineno, node.end_col_offset + 1) != (
                    error.get("stop_line"),
                    error.get("stop_column"),
                ):
                    continue
            elif not any(
                keyword.arg == field
                and (keyword.lineno, keyword.col_offset + 1)
                == (error.get("line"), error.get("column"))
                for keyword in node.keywords
            ):
                continue
            if self._resolve(node.func, module) != symbol:
                return False
            return self._supplied_alias(node, aliases)
        return False


def filter_report(
    input_path: Path,
    output_path: Path,
    config_path: Path,
    project_root: Path,
    pyrefly_exit_code: int,
    stderr_path: Path | None,
) -> int:
    """Filter excluded diagnostics and save a deterministic JSON report."""
    output = input_path.read_text(errors="replace")
    stderr = stderr_path.read_text(errors="replace") if stderr_path else ""
    try:
        result = json.loads(output)
        if not isinstance(result, dict) or not isinstance(result.get("errors"), list):
            raise ValueError("Pyrefly JSON output must contain an errors array")
        symbols = load_exclusions(config_path)
        excluded_paths = load_excluded_paths(config_path)
        scopes_by_path: dict[str, ImportScope | None] = {}
        pydantic_aliases = PydanticAliasFilter(project_root)
        filtered_errors = []
        excluded_count = 0
        for error in result["errors"]:
            if not isinstance(error, dict):
                raise ValueError("Pyrefly errors must be JSON objects")
            excluded_path = is_excluded_path(
                str(error.get("path", "")), project_root, excluded_paths
            )
            if (
                excluded_path
                or is_excluded_symbol(error, symbols, project_root, scopes_by_path)
                or pydantic_aliases.matches(error)
            ):
                excluded_count += 1
            else:
                filtered_errors.append(error)
        result["errors"] = filtered_errors
        report: dict[str, Any] = {
            "tool": "pyrefly",
            "pyrefly_exit_code": pyrefly_exit_code,
            "excluded_diagnostic_count": excluded_count,
            "remaining_diagnostic_count": len(filtered_errors),
            "result": result,
        }
        if stderr:
            report["stderr"] = stderr
    except (OSError, ValueError, json.JSONDecodeError) as error:
        report = {
            "tool": "pyrefly",
            "pyrefly_exit_code": pyrefly_exit_code,
            "report_error": str(error),
            "stdout": output,
            "stderr": stderr,
        }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    matches_parser = subparsers.add_parser("matches")
    matches_parser.add_argument("source", type=str)
    matches_parser.add_argument("config", type=Path)
    matches_parser.add_argument("--project-root", type=Path, required=True)
    paths_parser = subparsers.add_parser("filter-input-paths")
    paths_parser.add_argument("input", type=Path)
    paths_parser.add_argument("output", type=Path)
    paths_parser.add_argument("config", type=Path)
    paths_parser.add_argument("--project-root", type=Path, required=True)
    report_parser = subparsers.add_parser("filter-report")
    report_parser.add_argument("input", type=Path)
    report_parser.add_argument("output", type=Path)
    report_parser.add_argument("config", type=Path)
    report_parser.add_argument("--project-root", type=Path, required=True)
    report_parser.add_argument("--pyrefly-exit-code", type=int, required=True)
    report_parser.add_argument("--stderr", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "matches":
            excluded_paths = load_excluded_paths(args.config)
            return 0 if is_excluded_path(args.source, args.project_root, excluded_paths) else 1
        if args.command == "filter-input-paths":
            filter_input_paths(args.input, args.output, args.config, args.project_root)
            return 0
        return filter_report(
            args.input,
            args.output,
            args.config,
            args.project_root,
            args.pyrefly_exit_code,
            args.stderr,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
