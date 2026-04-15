#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QA test list maintenance for TensorRT-LLM.

Subcommands:
  sync-core          Add missing accuracy/disaggregated pytest ids from test-db GPU YAMLs
                     (backend pytorch / autodeploy) into llm_function_core.txt.
                     Never removes existing entries.
  regenerate-sanity  Rewrite llm_function_core_sanity.txt from llm_function_core.txt using the
                     P0 filter and per-method parametrization caps.

Run from the repository root (or pass --repo-root).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

TEST_DB_REL = Path("tests/integration/test_lists/test-db")
CORE_REL = Path("tests/integration/test_lists/qa/llm_function_core.txt")
SANITY_REL = Path("tests/integration/test_lists/qa/llm_function_core_sanity.txt")
BACKENDS = frozenset({"pytorch", "autodeploy"})

HEADER_LINES = (
    "# P0 functional sanity tests (~200) — conservative subset of llm_function_core.txt",
    "# Scope: DeepSeek, Kimi, Llama 3.1 8B, Llama 3.3 70B, Qwen3, GPT-OSS, Nemotron",
)


# --- llm_function_core.txt: parse lines (sanity + core load) -----------------


def parse_test_list_line(raw: str) -> str | None:
    """Return a pytest node id, or None for blanks / whole-line comments."""
    stripped = raw.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if "#" in stripped:
        stripped = stripped.split("#", 1)[0].rstrip()
    return stripped if stripped else None


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


# --- llm_function_core_sanity.txt --------------------------------------------


def method_key(nodeid: str) -> str:
    """Pytest method id without parametrization: file::Class::test_name."""
    if "[" in nodeid:
        return nodeid[: nodeid.index("[")]
    return nodeid


def matches_p0(line: str) -> bool:
    """Return True if the node id is in the P0 model scope (see QA skill)."""
    s = line.lower()
    if "deepseek" in s:
        return True
    if "kimi" in s:
        return True
    if "gptoss" in s or "gpt-oss" in s or "gpt_oss" in s:
        return True
    if "nemotron" in s:
        return True
    if "qwen3" in s:
        return True
    if any(
        x in s
        for x in (
            "llama3_1_8b",
            "llama3.1-8b",
            "llama-3.1-8b",
            "meta-llama-3.1-8b",
        )
    ):
        return True
    if any(x in s for x in ("llama3_3_70b", "llama3.3-70b", "llama-3.3-70b")):
        return True
    return False


_P0_TOTAL_CAP = 200  # hard cap on total sanity node ids
_P0_PER_METHOD_MAX = 2  # max variants to keep per method before total cap


def collect_sanity_node_ids(core_lines: list[str]) -> list[str]:
    """Filter to accuracy/ P0 tests, cap per method, then trim to _P0_TOTAL_CAP sorted ids."""
    filtered: list[str] = []
    for raw in core_lines:
        node = parse_test_list_line(raw)
        if node is None:
            continue
        if not node.startswith("accuracy/"):
            continue
        if not matches_p0(node):
            continue
        filtered.append(node)

    by_method: dict[str, list[str]] = defaultdict(list)
    for node in filtered:
        by_method[method_key(node)].append(node)

    out: list[str] = []
    for _key in sorted(by_method.keys()):
        variants = sorted(by_method[_key])
        out.extend(variants[:_P0_PER_METHOD_MAX])
    out.sort()
    return out[:_P0_TOTAL_CAP]


def render_sanity_file(node_ids: list[str]) -> str:
    lines = list(HEADER_LINES) + node_ids
    return "\n".join(lines) + "\n"


def cmd_regenerate_sanity(repo_root: Path, dry_run: bool) -> int:
    core_path = repo_root / CORE_REL
    sanity_path = repo_root / SANITY_REL

    core_text = core_path.read_text(encoding="utf-8")
    node_ids = collect_sanity_node_ids(core_text.splitlines())
    body = render_sanity_file(node_ids)

    print(f"Core file: {core_path}")
    print(f"Sanity file: {sanity_path}")
    print(f"Selected node ids: {len(node_ids)}")

    if dry_run:
        preview = node_ids[:5]
        print("First lines (node ids only):")
        for line in preview:
            print(f"  {line}")
        if len(node_ids) > 10:
            print("  ...")
        for line in node_ids[-5:]:
            print(f"  {line}")
        return 0

    sanity_path.write_text(body, encoding="utf-8")
    print(f"Wrote {sanity_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Sync QA test lists (llm_function_core + llm_function_core_sanity)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    repo_help = "Repository root (default: cwd)"

    p_core = sub.add_parser(
        "sync-core",
        help="Add missing test-db accuracy/disaggregated ids into llm_function_core.txt (never removes)",
    )
    p_core.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help=repo_help,
    )
    p_core.add_argument(
        "--dry-run",
        action="store_true",
        help="Print missing lines only; do not write",
    )

    p_sanity = sub.add_parser(
        "regenerate-sanity",
        help="Regenerate llm_function_core_sanity.txt from llm_function_core.txt",
    )
    p_sanity.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help=repo_help,
    )
    p_sanity.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts and sample lines; do not write",
    )

    args = parser.parse_args(argv)
    repo_root = (args.repo_root or Path.cwd()).resolve()

    if args.command == "sync-core":
        return cmd_sync_core(repo_root, args.dry_run)
    if args.command == "regenerate-sanity":
        return cmd_regenerate_sanity(repo_root, args.dry_run)
    raise AssertionError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
