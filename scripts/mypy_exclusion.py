# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Filter mypy diagnostics by exclusions and optional diff lines, then compare a baseline."""

import argparse
import ast
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from diff_line_filter import AddedLineFilter

ERROR_PATTERN = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+):(?P<column>\d+): error: "
    r"(?P<message>.*)  \[(?P<code>[^]]+)\]$"
)
HEADER = (
    "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & "
    "AFFILIATES. All rights reserved.\n"
    "# SPDX-License-Identifier: Apache-2.0\n"
    "# One JSON diagnostic per line: path, line, column, code, and message.\n"
)
QUOTED_NAME_PATTERN = re.compile(r'"([^"\n]+)"')
SYMBOL_PATTERN = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?:\.\*)?")


def use_explicit_sources_only(config_path: Path) -> None:
    """Remove the default file set from a temporary mypy TOML config."""
    config = config_path.read_text()
    section = re.search(r"(?m)^\[tool\.mypy\][ \t]*$", config)
    if section is None:
        raise ValueError(f"{config_path}: [tool.mypy] section not found")
    following_section = re.search(r"(?m)^\[", config[section.end() :])
    section_end = (
        section.end() + following_section.start() if following_section is not None else len(config)
    )
    default_files = list(
        re.finditer(
            r"(?ms)^files[ \t]*=[ \t]*\[[^\]]*\][ \t]*$",
            config[section.end() : section_end],
        )
    )
    if len(default_files) != 1:
        raise ValueError(f"{config_path}: expected one [tool.mypy] files setting")
    match = default_files[0]
    start = section.end() + match.start()
    end = section.end() + match.end()
    config_path.write_text(config[:start] + "files = []" + config[end:])


def load_exclusions(
    config_path: Path,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Load the exclusion sections and reject malformed entries."""
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{config_path}: invalid JSON: {error}") from error
    section_names = ("paths", "symbols")
    if not isinstance(config, dict) or set(config) != set(section_names):
        raise ValueError(f"{config_path}: expected {', '.join(section_names)} sections")
    sections: list[tuple[str, ...]] = []
    for section in section_names:
        entries = config[section]
        if not isinstance(entries, list) or any(
            not isinstance(entry, str) or not entry or entry != entry.strip() for entry in entries
        ):
            raise ValueError(f"{config_path}: {section} must contain nonempty strings")
        if len(entries) != len(set(entries)):
            raise ValueError(f"{config_path}: duplicate entry in {section}")
        sections.append(tuple(entries))
    return sections[0], sections[1]


def load_symbol_excludes(config_path: Path) -> tuple[str, ...]:
    """Return validated exact qualified names and module.* patterns."""
    _, patterns = load_exclusions(config_path)
    for pattern in patterns:
        if SYMBOL_PATTERN.fullmatch(pattern) is None:
            raise ValueError(f"{config_path}: invalid symbol pattern: {pattern}")
    return patterns


def build_exclude_regex(config_path: Path, project_root: Path | None = None) -> str:
    """Return a mypy regex for exact files and recursively excluded directories."""
    patterns: list[str] = []
    paths, _ = load_exclusions(config_path)
    for entry in paths:
        is_directory = entry.endswith("/")
        path = entry.rstrip("/")
        parts = path.split("/")
        if (
            "\\" in entry
            or any(part in ("", ".", "..") for part in parts)
            or any(character in entry for character in "*?[]")
        ):
            raise ValueError(f"{config_path}: invalid relative path: {entry}")

        target = (project_root or config_path.parent) / path
        if is_directory:
            if not target.is_dir():
                raise ValueError(f"{config_path}: directory not found: {entry}")
            patterns.append(f"^{re.escape(path)}/")
        else:
            if target.suffix not in (".py", ".pyi") or not target.is_file():
                raise ValueError(f"{config_path}: Python file not found: {entry}")
            patterns.append(f"^{re.escape(path)}$")

    return "|".join(patterns) if patterns else "(?!)"


def is_excluded_path(
    config_path: Path, source_path: Path, project_root: Path | None = None
) -> bool:
    """Check an explicitly passed source path against the same exclusion list."""
    try:
        root = project_root or config_path.parent
        relative_path = source_path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return re.search(build_exclude_regex(config_path, root), relative_path.as_posix()) is not None


def build_import_overrides(config_path: Path, project_root: Path | None = None) -> str:
    """Keep imported excluded sources available for types without reporting errors."""
    build_exclude_regex(config_path, project_root)
    modules: list[str] = []
    paths, _ = load_exclusions(config_path)
    for entry in paths:
        is_directory = entry.endswith("/")
        path = entry.rstrip("/")
        if not is_directory:
            path = str(Path(path).with_suffix(""))
            if path.endswith("/__init__"):
                path = path.removesuffix("/__init__")
        module = path.replace("/", ".")
        modules.append(module)
        if is_directory:
            modules.append(f"{module}.*")
    if not modules:
        return ""
    names = ", ".join(json.dumps(module) for module in modules)
    return (
        f"\n[[tool.mypy.overrides]]\nmodule = [{names}]\n"
        'follow_imports = "silent"\nignore_errors = true\n'
    )


@dataclass(frozen=True, order=True)
class Diagnostic:
    path: str
    line: int
    column: int
    code: str
    message: str

    def format(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: error: {self.message}  [{self.code}]"


def parse_output(output: str) -> Counter[Diagnostic]:
    """Read machine-formatted mypy errors and reject unrecognized error lines."""
    diagnostics: Counter[Diagnostic] = Counter()
    for line in output.splitlines():
        if ": error: " not in line:
            continue
        match = ERROR_PATTERN.fullmatch(line)
        if match is None:
            raise ValueError(f"Unrecognized mypy error line: {line}")
        diagnostics[
            Diagnostic(
                path=match["path"],
                line=int(match["line"]),
                column=int(match["column"]),
                code=match["code"],
                message=match["message"],
            )
        ] += 1
    return diagnostics


def load_baseline(path: Path) -> Counter[Diagnostic]:
    """Read individual baseline entries, preserving duplicate diagnostics."""
    diagnostics: Counter[Diagnostic] = Counter()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line or line.startswith("#"):
            continue
        try:
            entry = json.loads(line)
            if set(entry) != {"path", "line", "column", "code", "message"}:
                raise ValueError("unexpected fields")
            diagnostic = Diagnostic(**entry)
            if (
                not isinstance(diagnostic.path, str)
                or not isinstance(diagnostic.line, int)
                or not isinstance(diagnostic.column, int)
                or not isinstance(diagnostic.code, str)
                or not isinstance(diagnostic.message, str)
            ):
                raise ValueError("invalid field type")
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}:{line_number}: invalid baseline entry: {error}") from error
        diagnostics[diagnostic] += 1
    return diagnostics


def filter_symbol_excludes(
    diagnostics: Counter[Diagnostic], patterns: tuple[str, ...]
) -> Counter[Diagnostic]:
    """Discard diagnostics that mention an excluded quoted qualified name."""
    exact = {pattern for pattern in patterns if not pattern.endswith(".*")}
    prefixes = tuple(pattern[:-1] for pattern in patterns if pattern.endswith(".*"))
    return Counter(
        {
            diagnostic: count
            for diagnostic, count in diagnostics.items()
            if not any(
                name in exact or name.startswith(prefixes)
                for name in QUOTED_NAME_PATTERN.findall(diagnostic.message)
            )
        }
    )


@dataclass
class ImportScope:
    start: int
    end: int
    imports: dict[str, list[tuple[int, str]]]
    other_bindings: set[str]
    children: list["ImportScope"]

    def resolve(self, name: str, line: int) -> str | None:
        """Resolve a name through the lexical scopes enclosing a diagnostic."""
        scopes = [self]
        while True:
            child = next(
                (child for child in scopes[-1].children if child.start <= line <= child.end),
                None,
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
    """Collect imports and other bindings in each Python lexical scope."""
    try:
        tree = ast.parse(source_path.read_text())
    except (OSError, SyntaxError, UnicodeError):
        return None

    def build(node: ast.AST) -> ImportScope:
        scope = ImportScope(
            start=getattr(node, "lineno", 0),
            end=getattr(node, "end_lineno", 2**31),
            imports={},
            other_bindings=set(),
            children=[],
        )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            scope.other_bindings.update(
                argument.arg for argument in ast.walk(node.args) if isinstance(argument, ast.arg)
            )

        def visit(child: ast.AST) -> None:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                if hasattr(child, "name"):
                    scope.other_bindings.add(child.name)
                scope.children.append(build(child))
                return
            if isinstance(child, ast.Import):
                for alias in child.names:
                    local_name = alias.asname or alias.name.split(".")[0]
                    target = alias.name if alias.asname else local_name
                    scope.imports.setdefault(local_name, []).append((child.lineno, target))
                return
            if isinstance(child, ast.ImportFrom):
                for alias in child.names:
                    if alias.name == "*":
                        continue
                    local_name = alias.asname or alias.name
                    if child.level == 0 and child.module:
                        target = f"{child.module}.{alias.name}"
                        scope.imports.setdefault(local_name, []).append((child.lineno, target))
                    else:
                        scope.other_bindings.add(local_name)
                return
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                scope.other_bindings.add(child.id)
            elif isinstance(child, ast.ExceptHandler) and child.name:
                scope.other_bindings.add(child.name)
            for descendant in ast.iter_child_nodes(child):
                visit(descendant)

        for child in ast.iter_child_nodes(node):
            visit(child)
        return scope

    return build(tree)


def filter_imported_symbol_excludes(
    diagnostics: Counter[Diagnostic], patterns: tuple[str, ...], project_root: Path
) -> Counter[Diagnostic]:
    """Match resolved import paths while retaining the original diagnostics."""
    exact = {pattern for pattern in patterns if not pattern.endswith(".*")}
    prefixes = tuple(pattern[:-1] for pattern in patterns if pattern.endswith(".*"))
    root = project_root.resolve()
    scopes_by_path: dict[str, ImportScope | None] = {}
    filtered: Counter[Diagnostic] = Counter()
    for diagnostic, count in diagnostics.items():
        if diagnostic.code != "name-defined":
            filtered[diagnostic] = count
            continue
        match = re.fullmatch(r'Name "([^"]+)" is not defined', diagnostic.message)
        if match is None:
            filtered[diagnostic] = count
            continue
        local_name, separator, remainder = match[1].partition(".")
        if diagnostic.path not in scopes_by_path:
            source_path = (root / diagnostic.path).resolve()
            scopes_by_path[diagnostic.path] = (
                load_import_scope(source_path) if source_path.is_relative_to(root) else None
            )
        scope = scopes_by_path[diagnostic.path]
        imported_name = scope.resolve(local_name, diagnostic.line) if scope else None
        resolved_name = imported_name + separator + remainder if imported_name else None
        if resolved_name is None or not (
            resolved_name in exact or resolved_name.startswith(prefixes)
        ):
            filtered[diagnostic] = count
    return filtered


def filter_diff_line_diagnostics(
    diagnostics: Counter[Diagnostic], project_root: Path, diff_base: str
) -> tuple[Counter[Diagnostic], int]:
    """Keep diagnostics whose line falls within added lines relative to a Git ref."""
    if not diagnostics:
        return Counter(), 0
    line_filter = AddedLineFilter(project_root, diff_base)
    filtered: Counter[Diagnostic] = Counter()
    excluded_count = 0
    for diagnostic, count in diagnostics.items():
        if line_filter.contains(diagnostic.path, diagnostic.line):
            filtered[diagnostic] = count
        else:
            excluded_count += count
    return filtered, excluded_count


def write_baseline(path: Path, diagnostics: Counter[Diagnostic]) -> None:
    """Write a deterministic snapshot of individual mypy errors."""
    with path.open("w") as output:
        output.write(HEADER)
        for diagnostic in sorted(diagnostics.elements()):
            output.write(json.dumps(vars(diagnostic), ensure_ascii=False, sort_keys=True) + "\n")


def check_baseline(
    current: Counter[Diagnostic], baseline: Counter[Diagnostic], *, partial: bool = False
) -> int:
    """Report only new entries and return a failure status if any are present."""
    new = current - baseline
    for diagnostic in sorted(new.elements()):
        print(diagnostic.format())
    summary = (
        f"mypy baseline: {sum(current.values())} current, "
        f"{sum(baseline.values())} known, {sum(new.values())} new"
    )
    if partial:
        summary += " (partial scan)"
    else:
        summary += f", {sum((baseline - current).values())} no longer reported"
    print(summary)
    return 1 if new else 0


def exclusions_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Read mypy exclusions from JSON")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--matches", type=Path, metavar="SOURCE")
    parser.add_argument("--import-overrides", action="store_true")
    parser.add_argument("--explicit-sources-only", action="store_true")
    parser.add_argument("config", type=Path, help="repository-root exclusion list")
    args = parser.parse_args(argv)
    try:
        if args.explicit_sources_only:
            use_explicit_sources_only(args.config)
            return 0
        if args.import_overrides:
            print(build_import_overrides(args.config, args.project_root), end="")
            return 0
        if args.matches is not None:
            return 0 if is_excluded_path(args.config, args.matches, args.project_root) else 1
        print(build_exclude_regex(args.config, args.project_root))
        return 0
    except (OSError, ValueError) as error:
        parser.error(str(error))


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "exclusions":
        return exclusions_main(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "record"))
    parser.add_argument("output", type=Path, help="captured mypy stdout/stderr")
    parser.add_argument("baseline", type=Path)
    parser.add_argument(
        "--exclusions-config", type=Path, help="JSON path and symbol exclusion config"
    )
    parser.add_argument("--project-root", type=Path)
    parser.add_argument(
        "--diff-base",
        help="keep diagnostics only on added lines compared with this Git ref",
    )
    parser.add_argument("--mypy-exit-code", required=True, type=int)
    parser.add_argument(
        "--partial", action="store_true", help="omit resolved counts for scoped scans"
    )
    args = parser.parse_args()

    output = args.output.read_text(errors="replace")
    if args.action == "record" and args.diff_base:
        parser.error("diff-line filtering cannot be used when recording a baseline")
    if args.mypy_exit_code not in (0, 1):
        print(output, end="")
        parser.error(f"mypy failed with exit code {args.mypy_exit_code}")
    try:
        current = parse_output(output)
        if args.mypy_exit_code == 1 and not current:
            raise ValueError("mypy failed without parseable diagnostics")
        symbol_patterns = (
            load_symbol_excludes(args.exclusions_config) if args.exclusions_config else ()
        )
        project_root = args.project_root or Path.cwd()
        current = filter_imported_symbol_excludes(
            filter_symbol_excludes(current, symbol_patterns), symbol_patterns, project_root
        )
        if args.diff_base:
            current, diff_excluded_count = filter_diff_line_diagnostics(
                current, project_root, args.diff_base
            )
            print(
                f"mypy diff-line filter: {diff_excluded_count} diagnostic(s) outside added lines "
                f"for diff base {args.diff_base}"
            )
        if args.action == "record":
            write_baseline(args.baseline, current)
            print(f"Recorded {sum(current.values())} mypy errors in {args.baseline}")
            return 0
        baseline = filter_imported_symbol_excludes(
            filter_symbol_excludes(load_baseline(args.baseline), symbol_patterns),
            symbol_patterns,
            project_root,
        )
        return check_baseline(current, baseline, partial=args.partial)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
