# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report and diff helpers used by the Pyrefly shell workflows."""

import argparse
import json
import re
import subprocess
from pathlib import Path


def print_summary(report_path: Path) -> None:
    """Print a concise diagnostic summary from a Pyrefly report."""
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
            for error in errors[:8]:
                path = str(error.get("path", "<unknown>"))
                line = error.get("line")
                column = error.get("column")
                location = f"{path}:{line}" if line is not None else path
                if column is not None:
                    location += f":{column}"
                kind = error.get("name", "error")
                message = error.get("concise_description") or error.get("description", "")
                print(f"{location}: [{kind}] {str(message).replace(chr(10), ' ')}")
            if len(errors) > 8:
                print(f"Showing 8 of {len(errors)} diagnostics.")
    print(f"See full Pyrefly report at: {report_path}")


def clear_path_module_exclusions(source_path: Path, destination_path: Path) -> None:
    """Write a Pyrefly config with configured path and import exclusions disabled."""
    config = source_path.read_text()
    section_start = config.find("[tool.pyrefly]")
    if section_start < 0:
        raise ValueError(f"{source_path} has no [tool.pyrefly] section")
    next_section = re.search(r"(?m)^\[", config[section_start + 1 :])
    section_end = (
        section_start + 1 + next_section.start() if next_section is not None else len(config)
    )
    section = config[section_start:section_end]
    for key in ("project-excludes", "ignore-missing-imports", "replace-imports-with-any"):
        section, count = re.subn(
            rf"(?ms)^{re.escape(key)}\s*=\s*\[.*?^\]", f"{key} = []", section, count=1
        )
        if count != 1:
            raise ValueError(f"{source_path} is missing [tool.pyrefly].{key}")
    destination_path.write_text(config[:section_start] + section + config[section_end:])


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
    source_path: Path, destination_path: Path, project_root: Path, diff_base: str
) -> None:
    """Keep diagnostics whose source lines are added by the Git diff."""
    project_root = project_root.resolve()
    report = json.loads(source_path.read_text())
    if isinstance(report, dict) and report.get("report_error"):
        destination_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        return
    result = report.get("result", report) if isinstance(report, dict) else None
    if not isinstance(result, dict) or not isinstance(result.get("errors"), list):
        raise ValueError("filtered Pyrefly report must contain an errors array")

    hunk_pattern = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
    added_lines_by_path: dict[str, set[int]] = {}

    def added_lines(path_text: str) -> set[int]:
        path = Path(path_text)
        resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
        try:
            relative = resolved.relative_to(project_root)
        except ValueError:
            return set()
        path_key = relative.as_posix()
        if path_key in added_lines_by_path:
            return added_lines_by_path[path_key]
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", path_key],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if tracked.returncode == 1 and (project_root / relative).is_file():
            source_file = project_root / relative
            lines = set(range(1, len(source_file.read_text().splitlines()) + 1))
        elif tracked.returncode != 0:
            raise ValueError(
                f"cannot check Git tracking state for {path_key}: {tracked.stderr.strip()}"
            )
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
            if diff.returncode != 0:
                raise ValueError(f"cannot read diff for {path_key}: {diff.stderr.strip()}")
            lines = set()
            for diff_line in diff.stdout.splitlines():
                match = hunk_pattern.match(diff_line)
                if match:
                    start = int(match[1])
                    count = int(match[2] or 1)
                    lines.update(range(start, start + count))
        added_lines_by_path[path_key] = lines
        return lines

    original = result["errors"]
    result["errors"] = [
        error
        for error in original
        if isinstance(error, dict)
        and isinstance(error.get("line"), int)
        and error["line"] in added_lines(str(error.get("path", "")))
    ]
    report["diff_base"] = diff_base
    report["diff_filtered_count"] = len(original) - len(result["errors"])
    if "remaining_diagnostic_count" in report:
        report["remaining_diagnostic_count"] = len(result["errors"])
    destination_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


def annotate_rebased_report(
    report_path: Path,
    source_branch: str,
    upstream_source: str,
    base: str,
    head: str,
    worktree: str,
    diff_report: str,
    diff_filter: bool,
    path_module_filter: bool,
    checked_files: list[str],
) -> None:
    """Add rebase and filter metadata to the Pyrefly report."""
    try:
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError):
        report = {"tool": "pyrefly", "report_error": "Pyrefly report is not valid JSON"}
    report.update(
        source_branch=source_branch,
        upstream_source=upstream_source,
        upstream_main_sha=base,
        rebased_head=head,
        worktree=worktree,
        diff_report=diff_report,
        diff_line_filter=diff_filter,
        path_module_filter=path_module_filter,
        checked_files=checked_files,
    )
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


def record_rebase_failure(report_path: Path, status: int, failed_command: str) -> None:
    """Record an advisory failure that stopped the upstream-rebase workflow."""
    try:
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError):
        report = {"tool": "pyrefly"}
    report.update(
        workflow_error="rebase-upstream check failed before completion",
        workflow_exit_code=status,
        failed_command=failed_command,
    )
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary_parser = subparsers.add_parser("print-summary")
    summary_parser.add_argument("report", type=Path)

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

    annotate_parser = subparsers.add_parser("annotate-rebased-report")
    annotate_parser.add_argument("report", type=Path)
    annotate_parser.add_argument("source_branch")
    annotate_parser.add_argument("upstream_source")
    annotate_parser.add_argument("base")
    annotate_parser.add_argument("head")
    annotate_parser.add_argument("worktree")
    annotate_parser.add_argument("diff_report")
    annotate_parser.add_argument("diff_filter", type=int)
    annotate_parser.add_argument("path_module_filter", type=int)
    annotate_parser.add_argument("checked_files", nargs="*")

    failure_parser = subparsers.add_parser("record-rebase-failure")
    failure_parser.add_argument("report", type=Path)
    failure_parser.add_argument("status", type=int)
    failure_parser.add_argument("failed_command")

    args = parser.parse_args()
    if args.command == "print-summary":
        print_summary(args.report)
    elif args.command == "clear-exclusions-config":
        clear_path_module_exclusions(args.source, args.destination)
    elif args.command == "wrap-unfiltered-report":
        wrap_unfiltered_report(args.source, args.destination, args.stderr, args.status)
    elif args.command == "filter-diff-lines":
        filter_diff_lines(args.source, args.destination, args.project_root, args.diff_base)
    elif args.command == "record-rebase-failure":
        record_rebase_failure(args.report, args.status, args.failed_command)
    else:
        annotate_rebased_report(
            args.report,
            args.source_branch,
            args.upstream_source,
            args.base,
            args.head,
            args.worktree,
            args.diff_report,
            args.diff_filter == 1,
            args.path_module_filter == 1,
            args.checked_files,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
