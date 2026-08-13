#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Check (and fix) NVIDIA copyright headers in source files.

Enforces the two-part policy from CODING_GUIDELINES.md ("NVIDIA Copyright")
and AGENTS.md:

1. Every new source file must carry a copyright header. NVIDIA-authored
   code uses the NVIDIA header (the --fix default); external DCO
   contributors retain their own copyright and may use their own copyright
   line plus an Apache-2.0 SPDX tag instead, so any copyright header is
   accepted. Files that predate this check and have no header are
   grandfathered in scripts/copyright_header_baseline.txt and are skipped
   until someone backfills them (regenerate with --update-baseline).
2. When a file with an NVIDIA copyright line is modified, the year on that
   line must include the current year.

Scope rules:
- Empty files (e.g. bare __init__.py) are exempt.
- Third-party/vendored/generated code is permanently exempt via
  scripts/copyright_header_excludes.txt (pre-commit's global `exclude` also
  filters most of it before this script runs). That list is permanent —
  unlike the baseline, its entries must never be backfilled with NVIDIA
  headers.
- The year check only applies to files with uncommitted (staged or working
  tree) modifications, so `pre-commit run --all-files` on a clean checkout
  (post-merge CI) does not demand a current year on every file in the repo.

Header detection is deliberately lenient: any of the historical NVIDIA
header variants (classic "Copyright (c) NNNN, NVIDIA CORPORATION" blocks or
SPDX-style headers) is accepted. Insertion (--fix) writes the SPDX-style
Apache-2.0 header.
"""

import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = REPO_ROOT / "scripts" / "copyright_header_baseline.txt"

SOURCE_EXT_RE = re.compile(r"\.(py|c|cc|cpp|cxx|h|hpp|cu|cuh)$")

# Permanent exclusions (third-party / vendored / generated code that must
# never get an NVIDIA header). Kept in a separate committed file — distinct
# from the shrink-over-time baseline — and required to exist: failing loudly
# beats silently stamping NVIDIA headers onto vendored code.
EXCLUDES_PATH = REPO_ROOT / "scripts" / "copyright_header_excludes.txt"


def load_exclude_re():
    if not EXCLUDES_PATH.exists():
        sys.exit(
            f"error: {EXCLUDES_PATH} not found; refusing to run "
            "without the permanent third-party exclusion list"
        )
    patterns = [
        line.strip()
        for line in EXCLUDES_PATH.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return re.compile("(?:" + "|".join(patterns) + ")")


# How much of the file to scan for an existing header.
HEAD_CHARS = 2000

CURRENT_YEAR = datetime.date.today().year

APACHE_NOTICE = """SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""


def _header_lines():
    copyright_line = (
        f"SPDX-FileCopyrightText: Copyright (c) {CURRENT_YEAR} "
        "NVIDIA CORPORATION & AFFILIATES. All rights reserved."
    )
    return [copyright_line] + APACHE_NOTICE.splitlines()


def hash_comment_header():
    return "".join(f"# {line}".rstrip() + "\n" for line in _header_lines())


def c_block_header():
    lines = ["/*"]
    lines += [f" * {line}".rstrip() for line in _header_lines()]
    lines.append(" */")
    return "\n".join(lines) + "\n"


def has_copyright_header(text_head):
    # Deliberately does NOT require "NVIDIA": TensorRT-LLM contributions are
    # DCO-based (no CLA), so external contributors retain their own copyright
    # and may use their own copyright line + Apache-2.0 SPDX tag. The NVIDIA
    # header is only the --fix insertion default.
    return "opyright" in text_head or "SPDX-FileCopyrightText" in text_head


# Matches the year(s) on an NVIDIA copyright line, in the SPDX-style header,
# the classic "(c)" header, and the bare "Copyright NNNN NVIDIA ..." variant.
YEAR_RE = re.compile(r"(Copyright\s*(?:\(c\)\s*)?)(\d{4})(\s*-\s*(\d{4}))?")


def latest_nvidia_year(text_head):
    """Return (latest_year, None) or (None, reason) for NVIDIA copyright lines.

    Scans the file head for NVIDIA copyright lines and returns the latest year.
    """
    latest = None
    for line in text_head.splitlines():
        if "NVIDIA" not in line:
            continue
        m = YEAR_RE.search(line)
        if not m:
            continue
        year = int(m.group(4) or m.group(2))
        if latest is None or year > latest:
            latest = year
    if latest is None:
        return None, "no parseable year on NVIDIA copyright line"
    return latest, None


def bump_year(text):
    """Rewrite the year on NVIDIA copyright lines to end at CURRENT_YEAR."""

    def _bump_line(line):
        def _sub(m):
            start = int(m.group(2))
            if start >= CURRENT_YEAR:
                return m.group(0)
            return f"{m.group(1)}{start}-{CURRENT_YEAR}"

        return YEAR_RE.sub(_sub, line)

    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if "NVIDIA" in line and YEAR_RE.search(line):
            lines[i] = _bump_line(line)
    return "".join(lines)


def insert_header(path, text):
    """Return text with the canonical header prepended.

    For hash-comment files the header goes after any shebang / coding line.
    """
    if path.suffix == ".py":
        header = hash_comment_header()
        lines = text.splitlines(keepends=True)
        insert_at = 0
        if lines and lines[0].startswith("#!"):
            insert_at = 1
        if len(lines) > insert_at and re.match(r"#.*coding[:=]", lines[insert_at]):
            insert_at += 1
        prefix = "".join(lines[:insert_at])
        rest = "".join(lines[insert_at:])
        if rest and not rest.startswith("\n"):
            header += "\n"
        return prefix + header + rest
    header = c_block_header()
    if text and not text.startswith("\n"):
        header += "\n"
    return header + text


def locally_modified_files():
    """Files with staged or working-tree modifications, repo-relative.

    Empty on a clean checkout (e.g. CI running --all-files).
    """
    modified = set()
    for cmd in (["git", "diff", "--cached", "--name-only"], ["git", "diff", "--name-only"]):
        try:
            out = subprocess.run(
                cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True
            ).stdout
        except (subprocess.CalledProcessError, OSError):
            return set()
        modified.update(line.strip() for line in out.splitlines() if line.strip())
    return modified


def load_baseline():
    if not BASELINE_PATH.exists():
        return set()
    return {
        line.strip()
        for line in BASELINE_PATH.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }


def relpath(path):
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def check_files(files, fix, assume_modified=False):
    baseline = load_baseline()
    exclude_re = load_exclude_re()
    modified = locally_modified_files()
    problems = []
    for f in files:
        rel = relpath(f)
        path = REPO_ROOT / rel
        if not SOURCE_EXT_RE.search(rel) or exclude_re.match(rel):
            continue
        if not path.is_file():
            continue
        text = path.read_text(errors="ignore")
        if not text.strip():
            continue  # empty files are exempt
        head = text[:HEAD_CHARS]
        if not has_copyright_header(head):
            if rel in baseline:
                continue
            if fix:
                path.write_text(insert_header(path, text))
                problems.append(
                    f"{rel}: inserted missing copyright header (re-stage and "
                    "commit again; external contributors may replace the "
                    "NVIDIA copyright line with their own)"
                )
            else:
                problems.append(
                    f"{rel}: missing copyright header (run with --fix to insert the NVIDIA header)"
                )
            continue
        if not assume_modified and rel not in modified:
            continue
        year, reason = latest_nvidia_year(head)
        if year is None:
            # Header exists but the year is unparsable; leave it to humans.
            continue
        if year < CURRENT_YEAR:
            if fix:
                path.write_text(bump_year(text))
                problems.append(
                    f"{rel}: bumped copyright year to {CURRENT_YEAR} (re-stage and commit again)"
                )
            else:
                problems.append(
                    f"{rel}: copyright year {year} is stale; "
                    f"modified files must include {CURRENT_YEAR} "
                    "(run with --fix)"
                )
    return problems


def update_baseline():
    exclude_re = load_exclude_re()
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    missing = []
    for rel in out.splitlines():
        if not SOURCE_EXT_RE.search(rel) or exclude_re.match(rel):
            continue
        path = REPO_ROOT / rel
        if not path.is_file():
            continue
        text = path.read_text(errors="ignore")
        if not text.strip():
            continue
        if not has_copyright_header(text[:HEAD_CHARS]):
            missing.append(rel)
    header = (
        "# Source files that predate the copyright-header pre-commit "
        "hook and\n# have no NVIDIA copyright header. Files listed "
        "here are skipped by\n# scripts/check_copyright_header.py "
        "until they are backfilled.\n# Regenerate with: "
        "python3 scripts/check_copyright_header.py --update-baseline\n"
    )
    BASELINE_PATH.write_text(header + "".join(f"{f}\n" for f in sorted(missing)))
    print(f"wrote {len(missing)} entries to {BASELINE_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*")
    parser.add_argument(
        "--fix", action="store_true", help="insert missing headers / bump stale years in place"
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="regenerate scripts/copyright_header_baseline.txt from the current tree",
    )
    parser.add_argument(
        "--assume-modified",
        action="store_true",
        help="apply the year check to all given files, not just ones with "
        "uncommitted changes (for manual runs on already-committed trees, "
        "e.g. after a review comment)",
    )
    args = parser.parse_args()

    if args.update_baseline:
        update_baseline()
        return 0

    problems = check_files(args.files, fix=args.fix, assume_modified=args.assume_modified)
    for p in problems:
        print(p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
