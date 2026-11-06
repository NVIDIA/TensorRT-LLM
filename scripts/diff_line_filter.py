# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Match source locations to added lines in a Git diff."""

import re
import subprocess
from pathlib import Path

DIFF_HUNK_PATTERN = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


class AddedLineFilter:
    """Check whether source locations fall on lines added relative to a Git ref."""

    def __init__(self, project_root: Path, diff_base: str) -> None:
        self.project_root = project_root.resolve()
        self.added_lines_by_path: dict[str, set[int]] = {}

        git_root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=self.project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if git_root.returncode != 0:
            raise ValueError(f"cannot read diff lines: {git_root.stderr.strip()}")

        base = subprocess.run(
            ["git", "rev-parse", "--verify", f"{diff_base}^{{commit}}"],
            cwd=self.project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if base.returncode != 0:
            raise ValueError(f"invalid diff base {diff_base!r}: {base.stderr.strip()}")
        self.base_commit = base.stdout.strip()

    def contains(self, source_path: str, line_number: int) -> bool:
        """Return whether a path and line identify added or replaced source code."""
        path = Path(source_path)
        resolved_path = (
            path.resolve() if path.is_absolute() else (self.project_root / path).resolve()
        )
        try:
            relative_path = resolved_path.relative_to(self.project_root)
        except ValueError:
            return False

        path_key = relative_path.as_posix()
        if path_key not in self.added_lines_by_path:
            self.added_lines_by_path[path_key] = self._load_added_lines(relative_path)
        return line_number in self.added_lines_by_path[path_key]

    def _load_added_lines(self, relative_path: Path) -> set[int]:
        path = relative_path.as_posix()
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", path],
            cwd=self.project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        source_path = self.project_root / relative_path
        if tracked.returncode == 1 and source_path.is_file():
            return set(range(1, len(source_path.read_text().splitlines()) + 1))
        if tracked.returncode != 0:
            raise ValueError(
                f"cannot check Git tracking state for {path}: {tracked.stderr.strip()}"
            )

        diff = subprocess.run(
            [
                "git",
                "diff",
                "--no-ext-diff",
                "--no-renames",
                "--unified=0",
                self.base_commit,
                "--",
                path,
            ],
            cwd=self.project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if diff.returncode != 0:
            raise ValueError(f"cannot read diff for {path}: {diff.stderr.strip()}")

        lines: set[int] = set()
        for diff_line in diff.stdout.splitlines():
            match = DIFF_HUNK_PATTERN.match(diff_line)
            if match is None:
                continue
            start = int(match[1])
            count = int(match[2] or 1)
            lines.update(range(start, start + count))
        return lines
