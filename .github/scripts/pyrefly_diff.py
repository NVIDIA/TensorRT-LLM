#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare two Pyrefly JSON reports and their corresponding source trees."""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _read_report(path: Path) -> list[dict[str, Any]]:
    report = json.loads(path.read_text())
    if report.get("report_error"):
        raise ValueError(report["report_error"])
    if report.get("pyrefly_exit_code") not in (0, 1):
        raise ValueError(f"Pyrefly failed with exit code {report.get('pyrefly_exit_code')}")
    errors = report.get("result", {}).get("errors")
    if not isinstance(errors, list) or any(not isinstance(error, dict) for error in errors):
        raise ValueError("Pyrefly report must contain an errors array")
    return errors


def _fingerprint(
    error: dict[str, Any], root: Path, renames: dict[str, str]
) -> tuple[str, str, str, str]:
    path = Path(error["path"])
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected a repository-relative diagnostic path: {path}")
    lines = (root / path).read_text(errors="replace").splitlines()
    line = error["line"]
    source = lines[line - 1].strip() if isinstance(line, int) and 0 < line <= len(lines) else ""
    message = error.get("concise_description") or error.get("description", "")
    return renames.get(path.as_posix(), path.as_posix()), error["name"], message, source


def _new_errors(
    baseline: list[dict[str, Any]],
    merged: list[dict[str, Any]],
    base_root: Path,
    merged_root: Path,
    renames: dict[str, str],
) -> list[dict[str, Any]]:
    # Source text survives line shifts; counts keep newly added duplicate errors visible.
    existing = Counter(_fingerprint(error, base_root, renames) for error in baseline)
    added = []
    for error in merged:
        key = _fingerprint(error, merged_root, {})
        if existing[key]:
            existing[key] -= 1
        else:
            added.append(error)
    return added


def _renames(changes: Path) -> dict[str, str]:
    fields = iter(changes.read_text().rstrip("\0").split("\0"))
    result = {}
    for status in fields:
        if not status:
            continue
        old_path = next(fields)
        if status.startswith(("R", "C")):
            new_path = next(fields)
            if status.startswith("R"):
                result[old_path] = new_path
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", type=Path, default=os.environ.get("GITHUB_EVENT_PATH"))
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--merged-root", type=Path, required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--error", default="")
    args = parser.parse_args()
    pull = json.loads(args.event.read_text())["pull_request"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "pr_number": pull["number"],
        "base_ref": pull["base"]["ref"],
        "base_sha": args.base_sha,
        "head_sha": pull["head"]["sha"],
        "errors": [],
        "error": args.error or None,
    }
    if not args.error:
        try:
            report["errors"] = _new_errors(
                _read_report(output / "base.json"),
                _read_report(output / "merged.json"),
                args.base_root,
                args.merged_root,
                _renames(output / "changes"),
            )
        except (OSError, ValueError, KeyError) as error:
            report["error"] = str(error)[-2000:]
    if report["error"]:
        print(f"Pyrefly comparison incomplete: {report['error']}", file=sys.stderr)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
