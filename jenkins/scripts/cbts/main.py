#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CBTS entry point.

Modes:
  python3 main.py --list-needed-diffs
      Print the union of rules' `needs_diff_for` patterns, one per line.
  python3 main.py INPUT_JSON
      INPUT_JSON: {changed_files: [...], diffs: {path: diff}, post_merge: bool}.
      Stages are parsed from jenkins/L0_Test.groovy; YAMLs from
      tests/integration/test_lists/test-db/. Decision JSON goes to stdout
      with keys: scope, affected_stages, reasons, test_db_dir_override,
      affected_stage_test_counts.

Run from the TRT-LLM repo root or pass --repo-root.
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

from blocks import (  # noqa: E402
    Stage,
    YAMLIndex,
    compute_stage_test_counts,
    parse_stages_from_groovy,
    write_filtered_test_db,
)
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
    """Aggregated CBTS decision serialized to JSON for Groovy."""

    scope: Optional[str]
    affected_stages: set[str] = field(default_factory=set)
    reasons: list[str] = field(default_factory=list)
    block_filters: dict[tuple[str, int], dict[str, set[str]]] = field(default_factory=dict)
    test_db_dir_override: Optional[str] = None
    # Per-stage narrowed test count, used by Layer 2.5 split-collapse.
    affected_stage_test_counts: dict[str, int] = field(default_factory=dict)
    # Aggregated `any(rule.sanity_relevant)` across fired rules. Default
    # True is safe; Groovy Layer 2 keeps PackageSanityCheck only when True.
    sanity_required: bool = True
    # Aggregated `any(rule.perfsanity_relevant)`. Groovy Layer 2 keeps
    # *-PerfSanity-* stages only when True.
    perfsanity_required: bool = True

    def to_json(self) -> str:
        data = {
            "scope": self.scope,
            "affected_stages": sorted(self.affected_stages),
            "reasons": list(self.reasons),
            "test_db_dir_override": self.test_db_dir_override,
            "affected_stage_test_counts": dict(self.affected_stage_test_counts),
            "sanity_required": self.sanity_required,
            "perfsanity_required": self.perfsanity_required,
        }
        return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def _combine_scopes(scopes: list[str]) -> Optional[str]:
    """Return the common scope if all agree, else None."""
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
        for _, r in pairs:
            affected_stages |= r.affected_stages

        # If rules fired but no stages resolved, return scope=None so
        # downstream falls back to baseline. Maintains the invariant that any
        # scope!=None result has a non-empty pre-filter affected_stages set.
        if not affected_stages:
            return SelectionResult(
                scope=None,
                reasons=reasons
                + [
                    "Rules fired but no stages resolved (likely YAML/waive "
                    "granularity mismatch); falling back to baseline."
                ],
            )

        # Aggregate per-block prefix->{waive_ids} across rules.
        block_filters: dict[tuple[str, int], dict[str, set[str]]] = {}
        for _, r in pairs:
            for key, prefix_to_waives in r.block_filters.items():
                dst = block_filters.setdefault(key, {})
                for prefix, waives in prefix_to_waives.items():
                    dst.setdefault(prefix, set()).update(waives)

        sanity_required = any(r.sanity_relevant for _, r in pairs)
        perfsanity_required = any(r.perfsanity_relevant for _, r in pairs)

        return SelectionResult(
            scope=scope,
            affected_stages=affected_stages,
            reasons=reasons,
            block_filters=block_filters,
            sanity_required=sanity_required,
            perfsanity_required=perfsanity_required,
        )


# --- Input loading ----------------------------------------------------------


def _load_pr_inputs(input_json_path: Path) -> PRInputs:
    data = json.loads(input_json_path.read_text())
    return PRInputs(
        changed_files=list(data.get("changed_files", [])),
        diffs=dict(data.get("diffs", {})),
        post_merge=bool(data.get("post_merge", False)),
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
    # Include post-merge stages; the trigger-mode filter below selects the
    # subset matching the user's flag.
    stages = parse_stages_from_groovy(groovy_path, include_post_merge=True)
    pr = _load_pr_inputs(input_path)
    rules = build_rules(yaml_index, stages)
    result = Selector(stages).run(pr, rules)

    # Layer 3: write narrowed test-db when any block was filtered.
    if result.scope is not None and result.block_filters:
        out_dir_name = "cbts_test_db"
        write_filtered_test_db(
            src_dir=test_db_dir,
            output_dir=repo_root / out_dir_name,
            block_filters=result.block_filters,
        )
        result.test_db_dir_override = out_dir_name
        result.affected_stage_test_counts = compute_stage_test_counts(
            yaml_index=yaml_index,
            stages=stages,
            affected_stages=set(result.affected_stages),
            block_filters=result.block_filters,
        )

    # Filter affected_stages by trigger mode; recompute derived counts.
    pre_filter_stages = set(result.affected_stages)
    if pr.post_merge:
        result.affected_stages = {s for s in pre_filter_stages if "Post-Merge" in s}
    else:
        result.affected_stages = {s for s in pre_filter_stages if "Post-Merge" not in s}
    result.affected_stage_test_counts = {
        k: v for k, v in result.affected_stage_test_counts.items() if k in result.affected_stages
    }

    _log_decision_to_stderr(stages, result, pr, pre_filter_stages)
    sys.stdout.write(result.to_json())
    return 0


def _log_decision_to_stderr(
    stages: dict[str, Stage],
    result: SelectionResult,
    pr: PRInputs,
    pre_filter_stages: set[str],
) -> None:
    """Print the CBTS decision to stderr for Jenkins console diagnostics."""
    out = sys.stderr
    mode = "post_merge" if pr.post_merge else "pre_merge"
    dropped = sorted(pre_filter_stages - set(result.affected_stages))
    print("=" * 64, file=out)
    print(f"CBTS decision (diagnostic; stderr only) [trigger mode: {mode}]:", file=out)
    print(f"  scope: {result.scope}", file=out)
    print(f"  test_db_dir_override: {result.test_db_dir_override}", file=out)
    if dropped:
        print(
            f"  stages dropped by trigger-mode filter ({len(dropped)}, would run if mode flipped):",
            file=out,
        )
        for s in dropped:
            print(f"    - {s}", file=out)
    if result.reasons:
        print("  reasons:", file=out)
        for r in result.reasons:
            print(f"    - {r}", file=out)
    print(f"  block_filters ({len(result.block_filters)} blocks):", file=out)
    for (yaml_stem, idx), prefix_to_waives in sorted(result.block_filters.items()):
        print(f"    - {yaml_stem}#{idx}:", file=out)
        for prefix, waives in sorted(prefix_to_waives.items()):
            print(f"        {prefix} <- {sorted(waives)}", file=out)
    if result.affected_stage_test_counts:
        print(
            f"  affected_stage_test_counts ({len(result.affected_stage_test_counts)}):",
            file=out,
        )
        for name, count in sorted(result.affected_stage_test_counts.items()):
            print(f"    - {name}: {count}", file=out)
    print(f"  affected_stages ({len(result.affected_stages)}):", file=out)
    for name in sorted(result.affected_stages):
        stage = stages.get(name)
        annotation = f" [yaml_stem={stage.yaml_stem}]" if stage else ""
        print(f"    - {name}{annotation}", file=out)
    print("=" * 64, file=out)


if __name__ == "__main__":
    sys.exit(main())
