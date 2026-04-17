#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QA test list maintenance for TensorRT-LLM.

Subcommands:
  sync-core  Add missing accuracy/disaggregated pytest ids from test-db GPU YAMLs
             (backend pytorch / autodeploy) into llm_function_core.txt.
             Never removes existing entries.

Run from the repository root (or pass --repo-root).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TEST_DB_REL = Path("tests/integration/test_lists/test-db")
CORE_REL = Path("tests/integration/test_lists/qa/llm_function_core.txt")
BACKENDS = frozenset({"pytorch", "autodeploy"})


def normalize_list_item(s: str) -> str | None:
    """Strip YAML list item to a pytest node id, or None if not a node id."""
    s = s.strip()
    if s.startswith("#"):
        return None
    if "#" in s:
        s = s.split("#", 1)[0].rstrip()
    s = re.sub(r"\s+TIMEOUT\s*\([^)]*\)\s*$", "", s)
    s = re.sub(r"\s+ISOLATION\s*$", "", s)
    s = s.strip()
    if not s or "::" not in s:
        return None
    return s


def extract_accuracy_disaggregated_from_yml(text: str) -> set[str]:
    """Parse test-db YAML text; return node ids from pytorch/autodeploy ``tests:`` blocks."""
    lines = text.splitlines()
    found: set[str] = set()
    i = 0
    n = len(lines)
    while i < n:
        if lines[i].strip() != "- condition:":
            i += 1
            continue
        i += 1
        backend: str | None = None
        while i < n:
            ln = lines[i]
            m = re.match(r"^\s+backend:\s*(\S+)", ln)
            if m:
                backend = m.group(1)
            if re.match(r"^\s*tests:\s*$", ln):
                break
            i += 1
        if i >= n:
            break
        i += 1
        take = backend in BACKENDS
        while i < n:
            ln = lines[i]
            if ln.strip() == "- condition:":
                break
            m = re.match(r"^(\s+)-\s+(.+)$", ln)
            if m and take:
                node = normalize_list_item(m.group(2))
                if node and (node.startswith("accuracy/") or node.startswith("disaggregated/")):
                    found.add(node)
            i += 1
    return found


def load_core_node_ids(path: Path) -> tuple[list[str], set[str]]:
    """Return ordered lines and a set of node ids (for membership tests)."""
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        line = stripped
        if "#" in line:
            line = line.split("#", 1)[0].rstrip()
        if line:
            ordered.append(line)
            seen.add(line)
    return ordered, seen


def collect_test_db_node_ids(test_db_dir: Path) -> set[str]:
    out: set[str] = set()
    for yml in sorted(test_db_dir.glob("*.yml")):
        out |= extract_accuracy_disaggregated_from_yml(yml.read_text(encoding="utf-8"))
    return out


def cmd_sync_core(repo_root: Path, dry_run: bool) -> int:
    test_db = repo_root / TEST_DB_REL
    core_path = repo_root / CORE_REL

    from_db = collect_test_db_node_ids(test_db)
    existing_lines, existing_set = load_core_node_ids(core_path)
    missing = sorted(from_db - existing_set)

    print(
        f"test-db (backend pytorch + autodeploy) accuracy + disaggregated node ids: {len(from_db)}"
    )
    print(f"{core_path.name} lines (non-comment): {len(existing_lines)}")
    print(f"missing (will add): {len(missing)}")

    if dry_run:
        if missing:
            print("  sample (new entries):")
            for line in missing[:30]:
                print(f"  {line}")
            if len(missing) > 30:
                print(f"  ... {len(missing) - 30} more")
        return 0

    all_lines = sorted(set(existing_lines) | set(missing))
    if not missing and all_lines == existing_lines:
        print("Nothing to change.")
        return 0

    new_text = "\n".join(all_lines) + "\n"
    core_path.write_text(new_text, encoding="utf-8")
    print(f"Wrote {core_path} ({len(all_lines)} lines)")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description="Sync QA test lists (llm_function_core).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_core = sub.add_parser(
        "sync-core",
        help="Add missing test-db accuracy/disaggregated ids into llm_function_core.txt (never removes)",
    )
    p_core.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (default: cwd)",
    )
    p_core.add_argument(
        "--dry-run",
        action="store_true",
        help="Print missing lines only; do not write",
    )

    args = parser.parse_args(argv)
    repo_root = (args.repo_root or Path.cwd()).resolve()

    if args.command == "sync-core":
        return cmd_sync_core(repo_root, args.dry_run)
    raise AssertionError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
