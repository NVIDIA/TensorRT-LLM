# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report and diff helpers used by the Pyrefly shell workflow."""

import argparse
import json
import re
import subprocess
from pathlib import Path


def report_status(report_path: Path) -> int:
    """Fail strict checks on remaining diagnostics or an unsuccessful tool invocation."""
    report = json.loads(report_path.read_text())
    if report.get("report_error"):
        return 2
    status = report.get("pyrefly_exit_code", 0)
    if status not in (0, 1):
        return status
    result = report.get("result", report)
    if not isinstance(result, dict) or not isinstance(result.get("errors"), list):
        raise ValueError("Pyrefly report must contain an errors array")
    return int(bool(result["errors"]))


def print_summary(report_path: Path, *, all_diagnostics: bool = False) -> None:
    """Print diagnostics from a Pyrefly report."""
    try:
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        print(f"Could not read Pyrefly report: {error}")
    else:
        if report.get("report_error"):
            print(f"Pyrefly report error: {report['report_error']}")
        result = report.get("result", report)
        errors = result.get("errors", []) if isinstance(result, dict) else []
        if not isinstance(errors, list):
            errors = []
        if not errors:
            print("No Pyrefly diagnostics.")
        else:
            if all_diagnostics:
                print("Pyrefly type check on your changed files. Please fix the type errors below.")
                print(
                    "They are only warnings for now, but in the future they may block your commits and PR merges."
                )
            for error in errors if all_diagnostics else errors[:8]:
                path = str(error.get("path", "<unknown>"))
                line = error.get("line")
                column = error.get("column")
                location = f"{path}:{line}" if line is not None else path
                if column is not None:
                    location += f":{column}"
                kind = error.get("name", "error")
                message = (
                    error.get("description") or error.get("concise_description", "")
                    if all_diagnostics
                    else error.get("concise_description") or error.get("description", "")
                )
                if not all_diagnostics:
                    message = str(message).replace("\n", " ")
                print(f"{location}: [{kind}] {message}")
            if not all_diagnostics and len(errors) > 8:
                print(f"Showing 8 of {len(errors)} diagnostics.")
    print(f"See full Pyrefly report at: {report_path}")


def clear_path_module_exclusions(source_path: Path, destination_path: Path) -> None:
    """Write a Pyrefly config with configured path and import exclusions disabled."""
    config = source_path.read_text()
    headers = list(re.finditer(r"(?m)^\[([^\]\n]+)\][ \t]*$", config))
    if not any(header[1] == "tool.pyrefly" for header in headers):
        raise ValueError(f"{source_path} has no [tool.pyrefly] section")
    # Explicit config files use Pyrefly's standalone TOML format.
    sections = []
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(config)
        body = config[header.end() : end]
        if header[1] == "tool.pyrefly":
            sections.append(body)
        elif header[1].startswith("tool.pyrefly."):
            sections.append(f"[{header[1].removeprefix('tool.pyrefly.')}]" + body)
    section = "".join(sections)
    for key in ("project-excludes", "ignore-missing-imports", "replace-imports-with-any"):
        section, count = re.subn(
            rf"(?ms)^{re.escape(key)}\s*=\s*\[.*?^\]", f"{key} = []", section, count=1
        )
        if count != 1 and key != "replace-imports-with-any":
            raise ValueError(f"{source_path} is missing [tool.pyrefly].{key}")
    destination_path.write_text(section)


def wrap_unfiltered_report(
    source_path: Path, destination_path: Path, stderr_path: Path, status: int
) -> None:
    """Wrap raw Pyrefly JSON without applying path or symbol exclusions."""
    stderr = stderr_path.read_text()
    try:
        result = json.loads(source_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        report = {
            "tool": "pyrefly",
            "pyrefly_exit_code": status,
            "report_error": f"Pyrefly did not produce valid JSON: {error}",
            "stdout": source_path.read_text(errors="replace"),
        }
    else:
        errors = result.get("errors", []) if isinstance(result, dict) else []
        report = {
            "tool": "pyrefly",
            "pyrefly_exit_code": status,
            "excluded_diagnostic_count": 0,
            "remaining_diagnostic_count": len(errors) if isinstance(errors, list) else 0,
            "result": result,
        }
    if stderr:
        report["stderr"] = stderr
    destination_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


def filter_diff_lines(
    source_path: Path,
    destination_path: Path,
    project_root: Path,
    diff_base: str,
    files: list[str],
) -> None:
    """Filter supplied files by added and modified lines when Git identifies them."""
    project_root = project_root.resolve()
    report = json.loads(source_path.read_text())
    if isinstance(report, dict) and report.get("report_error"):
        destination_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        return
    result = report.get("result", report) if isinstance(report, dict) else None
    if not isinstance(result, dict) or not isinstance(result.get("errors"), list):
        raise ValueError("filtered Pyrefly report must contain an errors array")

    hunk_pattern = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
    changed_lines_by_path: dict[str, set[int] | None] = {}

    def relative_path(path_text: str) -> str | None:
        path = Path(path_text)
        try:
            resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
            return resolved.relative_to(project_root).as_posix()
        except (OSError, ValueError):
            return None

    requested_paths = {path for name in files if (path := relative_path(name)) is not None}

    def changed_lines(path_key: str) -> set[int] | None:
        if path_key in changed_lines_by_path:
            return changed_lines_by_path[path_key]
        try:
            tracked = subprocess.run(
                ["git", "ls-files", "--error-unmatch", "--", path_key],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=False,
            )
            if tracked.returncode == 1 and (project_root / path_key).is_file():
                source_file = project_root / path_key
                lines = set(range(1, len(source_file.read_text().splitlines()) + 1))
            elif tracked.returncode != 0:
                lines = None
            else:
                diff = subprocess.run(
                    [
                        "git",
                        "diff",
                        "--no-ext-diff",
                        "--no-renames",
                        "--unified=0",
                        diff_base,
                        "--",
                        path_key,
                    ],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                lines = set()
                if diff.returncode == 0:
                    for diff_line in diff.stdout.splitlines():
                        match = hunk_pattern.match(diff_line)
                        if match:
                            start = int(match[1])
                            count = int(match[2] or 1)
                            lines.update(range(start, start + count))
        except (OSError, UnicodeError):
            lines = None
        if not lines:
            lines = None
        changed_lines_by_path[path_key] = lines
        return lines

    original = result["errors"]
    remaining = []
    for error in original:
        if not isinstance(error, dict):
            continue
        path_key = relative_path(str(error.get("path", "")))
        if path_key is None or path_key not in requested_paths:
            continue
        lines = changed_lines(path_key)
        if lines is None or not isinstance(error.get("line"), int) or error["line"] in lines:
            remaining.append(error)
    result["errors"] = remaining
    report["diff_base"] = diff_base
    report["diff_filtered_count"] = len(original) - len(result["errors"])
    report["diff_filter_unavailable_paths"] = sorted(
        path for path, lines in changed_lines_by_path.items() if lines is None
    )
    if "remaining_diagnostic_count" in report:
        report["remaining_diagnostic_count"] = len(result["errors"])
    destination_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary_parser = subparsers.add_parser("print-summary")
    summary_parser.add_argument("report", type=Path)

    diagnostics_parser = subparsers.add_parser("print-diagnostics")
    diagnostics_parser.add_argument("report", type=Path)

    status_parser = subparsers.add_parser("report-status")
    status_parser.add_argument("report", type=Path)

    config_parser = subparsers.add_parser("clear-exclusions-config")
    config_parser.add_argument("source", type=Path)
    config_parser.add_argument("destination", type=Path)

    wrap_parser = subparsers.add_parser("wrap-unfiltered-report")
    wrap_parser.add_argument("source", type=Path)
    wrap_parser.add_argument("destination", type=Path)
    wrap_parser.add_argument("stderr", type=Path)
    wrap_parser.add_argument("status", type=int)

    diff_parser = subparsers.add_parser("filter-diff-lines")
    diff_parser.add_argument("source", type=Path)
    diff_parser.add_argument("destination", type=Path)
    diff_parser.add_argument("project_root", type=Path)
    diff_parser.add_argument("diff_base")
    diff_parser.add_argument("--files", nargs="+", required=True)

    args = parser.parse_args()
    if args.command == "print-summary":
        print_summary(args.report)
    elif args.command == "print-diagnostics":
        print_summary(args.report, all_diagnostics=True)
    elif args.command == "report-status":
        return report_status(args.report)
    elif args.command == "clear-exclusions-config":
        clear_path_module_exclusions(args.source, args.destination)
    elif args.command == "wrap-unfiltered-report":
        wrap_unfiltered_report(args.source, args.destination, args.stderr, args.status)
    elif args.command == "filter-diff-lines":
        filter_diff_lines(
            args.source, args.destination, args.project_root, args.diff_base, args.files
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
