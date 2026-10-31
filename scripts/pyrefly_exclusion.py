# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply path and third-party symbol exclusions to Pyrefly JSON reports."""

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

NAME_PATTERN = re.compile(r"`([^`]+)`|\"([^\"]+)\"")
SYMBOL_PATTERN = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?:\.\*)?")


def load_exclusions(config_path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Load and validate path and symbol exclusion entries."""
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{config_path}: invalid JSON: {error}") from error
    if not isinstance(config, dict) or set(config) != {"paths", "symbols"}:
        raise ValueError(f"{config_path}: expected paths and symbols sections")
    sections: list[tuple[str, ...]] = []
    for section in ("paths", "symbols"):
        entries = config[section]
        if not isinstance(entries, list) or any(
            not isinstance(entry, str) or not entry or entry != entry.strip() for entry in entries
        ):
            raise ValueError(f"{config_path}: {section} must contain nonempty strings")
        if len(entries) != len(set(entries)):
            raise ValueError(f"{config_path}: duplicate entry in {section}")
        sections.append(tuple(entries))
    for pattern in sections[1]:
        if SYMBOL_PATTERN.fullmatch(pattern) is None:
            raise ValueError(f"{config_path}: invalid symbol pattern: {pattern}")
    return sections[0], sections[1]


def relative_source_path(source_path: str, project_root: Path) -> Path | None:
    """Return a source path relative to the project root, if it is inside it."""
    path = Path(source_path)
    resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    try:
        return resolved.relative_to(project_root.resolve())
    except ValueError:
        return None


def is_excluded_path(config_path: Path, source_path: str, project_root: Path) -> bool:
    paths, _ = load_exclusions(config_path)
    relative_path = relative_source_path(source_path, project_root)
    if relative_path is None:
        return False
    value = relative_path.as_posix()
    return any(value == entry.rstrip("/") or value.startswith(entry) for entry in paths)


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
        paths, symbols = load_exclusions(config_path)
        scopes_by_path: dict[str, ImportScope | None] = {}
        filtered_errors = []
        excluded_count = 0
        for error in result["errors"]:
            if not isinstance(error, dict):
                raise ValueError("Pyrefly errors must be JSON objects")
            path = relative_source_path(str(error.get("path", "")), project_root)
            excluded_path = path is not None and any(
                path.as_posix() == entry.rstrip("/") or path.as_posix().startswith(entry)
                for entry in paths
            )
            if excluded_path or is_excluded_symbol(error, symbols, project_root, scopes_by_path):
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
            return 0 if is_excluded_path(args.config, args.source, args.project_root) else 1
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
