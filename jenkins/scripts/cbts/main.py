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
      Output is a text blob on stdout, with a `# SCOPE:` header and, when a
      rule is active, `# AFFECTED_CPU_ARCH`, `# AFFECTED_STAGES`, and one
      test id per line. Parsed by `parseSelectionResult` on the Groovy side.

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
    """Instantiate rules with their dependencies.

    Add a new rule: append a new line here and a new class to RULE_CLASSES.
    """
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

    def to_text(self) -> str:
        if self.scope is None:
            reason = "; ".join(self.reasons) if self.reasons else "no decision"
            return f"# SCOPE: none\n# REASON: {reason}\n"

        lines = [f"# SCOPE: {self.scope}"]
        for r in self.reasons:
            lines.append(f"# REASON: {r}")
        lines.append(f"# AFFECTED_CPU_ARCH: {', '.join(sorted(self.affected_cpu_arch))}")
        lines.append(f"# AFFECTED_STAGES: {', '.join(sorted(self.affected_stages))}")
        lines.extend(sorted(self.tests))
        return "\n".join(lines) + "\n"


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
    def run(
        self,
        pr: PRInputs,
        rules: list[Rule],
        stages: dict[str, Stage],
    ) -> SelectionResult:
        # 1. Run all rules, keep (rule, result) pairs that apply.
        pairs: list[tuple[Rule, RuleResult]] = []
        for rule in rules:
            result = rule.apply(pr)
            if result is not None:
                pairs.append((rule, result))

        # 2. Coverage check: any changed file not handled by any rule -> no decision.
        handled: set[str] = set()
        for _, r in pairs:
            handled |= r.handled_files
        unhandled = sorted(set(pr.changed_files) - handled)
        if unhandled:
            preview = unhandled[:5]
            more = f" (+{len(unhandled) - 5} more)" if len(unhandled) > 5 else ""
            return SelectionResult(
                scope=None,
                reasons=[f"Unhandled files: {preview}{more}"],
            )

        if not pairs:
            # No changed_files and no rule applied -> no decision.
            return SelectionResult(
                scope=None,
                reasons=["No rule contributed"],
            )

        # 3. Combine scopes across rules.
        scope = _combine_scopes([r.scope for _, r in pairs])
        if scope is None:
            return SelectionResult(
                scope=None,
                reasons=[f"[{rule.name}] {r.reason}" for rule, r in pairs]
                + ["Scopes cannot be combined"],
            )

        # 4. Union stages and tests across rules; derive affected_cpu_arch from stages.
        affected_stages: set[str] = set()
        tests: set[str] = set()
        for _, r in pairs:
            affected_stages |= r.affected_stages
            tests |= r.tests

        affected_cpu_arch: set[str] = set()
        for name in affected_stages:
            stage = stages.get(name)
            if stage is not None:
                affected_cpu_arch.add(stage.cpu_arch)

        reasons = [f"[{rule.name}] {r.reason}" for rule, r in pairs]
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
    result = Selector().run(pr, rules, stages)
    sys.stdout.write(result.to_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
