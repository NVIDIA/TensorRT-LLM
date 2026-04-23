#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CBTS entry point — consumed by Jenkins Groovy helper `getCbtsResult`.

Two invocation modes (see DESIGN.md for full context):

  python3 main.py --list-needed-diffs
      Print the union of all rules' `needs_diff_for` patterns, one per line.
      Groovy uses this to decide which changed files to fetch diffs for.

  python3 main.py INPUT_JSON
      Run decision logic. Groovy passes a JSON file containing only PR data:
        - changed_files: list[str]
        - diffs: {path: diff_content}
      Python self-sources everything else from the repo:
        - stage configs: parsed from jenkins/L0_Test.groovy
        - test-db YAMLs: loaded from tests/integration/test_lists/test-db/
      Output is a JSON blob on stdout with fields `scope`, `affected_cpu_arch`,
      `affected_stages`, `tests`, `reasons`. Consumed by `_cbtsParseSelectionResult`
      on the Groovy side.

Invocation assumes the current working directory is the TRT-LLM repo root,
or that --repo-root is passed explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Make sibling modules importable when invoked as `python3 <path>/main.py ...`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from blocks import Stage, YAMLIndex, parse_stages_from_groovy  # noqa: E402
from rules.base import PRInputs, Rule, RuleResult  # noqa: E402
from rules.waives_rule import WaivesRule  # noqa: E402

# --- Rule registry -----------------------------------------------------------

# Classes are used for `--list-needed-diffs` (no need to construct).
RULE_CLASSES: list[type[Rule]] = [WaivesRule]


def build_rules(yaml_index: YAMLIndex, stages: dict[str, Stage]) -> list[Rule]:
    return [WaivesRule(yaml_index, stages)]


# --- Selector ---------------------------------------------------------------


@dataclass
class SelectionResult:
    """Final aggregated decision."""

    scope: Optional[str]
    affected_stages: set[str] = field(default_factory=set)
    affected_cpu_arch: set[str] = field(default_factory=set)
    tests: set[str] = field(default_factory=set)
    reasons: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        data = {
            "scope": self.scope,
            "affected_cpu_arch": sorted(self.affected_cpu_arch),
            "affected_stages": sorted(self.affected_stages),
            "tests": sorted(self.tests),
            "reasons": list(self.reasons),
        }
        return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def _combine_scopes(scopes: list[str]) -> Optional[str]:
    """Combine scope labels from multiple rules.

    v0: single rule => passthrough.
    Multi-rule future: when all scopes agree, use that scope; otherwise return
    None (no-decision / full run) until an explicit priority table is added.
    """
    if not scopes:
        return None
    if len(set(scopes)) == 1:
        return scopes[0]
    return None


class Selector:
    def __init__(self, stages: dict[str, Stage]) -> None:
        self.stages = stages

    def run(self, pr: PRInputs, rules: list[Rule]) -> SelectionResult:
        pairs: list[tuple[Rule, RuleResult]] = []
        for rule in rules:
            result = rule.apply(pr)
            if result is not None:
                pairs.append((rule, result))

        handled: set[str] = set()
        for _, r in pairs:
            handled |= r.handled_files
        unhandled = sorted(set(pr.changed_files) - handled)
        if unhandled:
            preview = unhandled[:5]
            more = f" (+{len(unhandled) - 5} more)" if len(unhandled) > 5 else ""
            return SelectionResult(scope=None, reasons=[f"Unhandled files: {preview}{more}"])

        if not pairs:
            return SelectionResult(scope=None, reasons=["No rule contributed"])

        reasons = [f"[{rule.name}] {r.reason}" for rule, r in pairs]
        scope = _combine_scopes([r.scope for _, r in pairs])
        if scope is None:
            return SelectionResult(scope=None, reasons=reasons + ["Scopes cannot be combined"])

        affected_stages: set[str] = set()
        tests: set[str] = set()
        for _, r in pairs:
            affected_stages |= r.affected_stages
            tests |= r.tests

        affected_cpu_arch = {
            self.stages[name].cpu_arch for name in affected_stages if name in self.stages
        }

        return SelectionResult(
            scope=scope,
            affected_stages=affected_stages,
            affected_cpu_arch=affected_cpu_arch,
            tests=tests,
            reasons=reasons,
        )


# --- Input loading ----------------------------------------------------------


def _load_pr_inputs(input_json_path: Path) -> PRInputs:
    data = json.loads(input_json_path.read_text())
    return PRInputs(
        changed_files=list(data.get("changed_files", [])),
        diffs=dict(data.get("diffs", {})),
    )


# --- CLI --------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="CBTS — Change-Based Testing Selection for TRT-LLM CI",
    )
    parser.add_argument(
        "input_json",
        nargs="?",
        help="Path to the INPUT_JSON file prepared by Groovy `getCbtsResult`.",
    )
    parser.add_argument(
        "--list-needed-diffs",
        action="store_true",
        help="Print the union of all rules' needs_diff_for patterns and exit.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Path to the TRT-LLM repo root (default: current working directory).",
    )
    parser.add_argument(
        "--test-db",
        default=None,
        help="Override path to the test-db YAML directory "
        "(default: <repo-root>/tests/integration/test_lists/test-db).",
    )
    parser.add_argument(
        "--groovy-file",
        default=None,
        help="Override path to the Jenkins test Groovy file "
        "(default: <repo-root>/jenkins/L0_Test.groovy).",
    )
    args = parser.parse_args(argv)

    if args.list_needed_diffs:
        patterns: set[str] = set()
        for cls in RULE_CLASSES:
            patterns.update(cls.needs_diff_for)
        for p in sorted(patterns):
            print(p)
        return 0

    if not args.input_json:
        print(
            "error: INPUT_JSON is required (or pass --list-needed-diffs)",
            file=sys.stderr,
        )
        return 2

    input_path = Path(args.input_json)
    if not input_path.is_file():
        print(f"error: INPUT_JSON not found: {input_path}", file=sys.stderr)
        return 2

    repo_root = Path(args.repo_root).resolve()
    test_db_dir = (
        Path(args.test_db) if args.test_db else repo_root / "tests/integration/test_lists/test-db"
    )
    groovy_path = (
        Path(args.groovy_file) if args.groovy_file else repo_root / "jenkins/L0_Test.groovy"
    )

    if not test_db_dir.is_dir():
        print(f"error: test-db directory not found: {test_db_dir}", file=sys.stderr)
        return 2
    if not groovy_path.is_file():
        print(f"error: Jenkins groovy file not found: {groovy_path}", file=sys.stderr)
        return 2

    yaml_index = YAMLIndex.load(test_db_dir)
    stages = parse_stages_from_groovy(groovy_path)
    pr = _load_pr_inputs(input_path)
    rules = build_rules(yaml_index, stages)
    result = Selector(stages).run(pr, rules)
    sys.stdout.write(result.to_json())
    return 0


if __name__ == "__main__":
    sys.exit(main())
