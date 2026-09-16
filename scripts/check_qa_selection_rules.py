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
"""Check tests/qa_selection/rules.json against the suite's skip decorators.

A rule matching no decorator is an error: its reason was reworded, so the rule
now matches nothing. A decorator with no rule is only reported, because
selection already keeps those tests and records them.

Decorators are read with `ast`, never imported.
"""

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_ROOT = REPO_ROOT / "tests" / "integration" / "defs"

sys.path.insert(0, str(REPO_ROOT / "tests"))
from qa_selection.rules import SkipRuleTable, default_rule_table  # noqa: E402


@dataclass(frozen=True)
class SkipDecorator:
    """One module-level `skip_* = pytest.mark.skipif(...)` assignment."""

    name: str
    reason: str
    path: Path

    @property
    def location(self) -> str:
        return f"{self.name} ({self.path})"


class SkipDecoratorScanner:
    """Finds the suite's shared skip decorators without importing anything."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def decorators(self) -> Iterator[SkipDecorator]:
        """Yield every module-level skip_* skipif assignment under the root.

        Module level only: a skipif written inline at a test is a one-off, not a
        shared rule.
        """
        for path in sorted(self.root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in tree.body:
                if not isinstance(node, ast.Assign) or not self.is_skipif_call(node.value):
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.startswith("skip_"):
                        yield SkipDecorator(target.id, self.reason_of(node.value), path)

    @staticmethod
    def is_skipif_call(node: ast.AST) -> bool:
        """True for a `pytest.mark.skipif(...)` call, however pytest is aliased."""
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "skipif"
        )

    @staticmethod
    def reason_of(call: ast.Call) -> str:
        """The literal `reason=` string, or "" when absent or computed."""
        for keyword in call.keywords:
            if keyword.arg == "reason" and isinstance(keyword.value, ast.Constant):
                if isinstance(keyword.value.value, str):
                    return keyword.value.value
        return ""


class RuleTableDriftCheck:
    """Compares the decorators found on disk against the rule table."""

    def __init__(self, root: Path, rules: SkipRuleTable) -> None:
        self.decorators = list(SkipDecoratorScanner(root).decorators())
        self.rules = rules

    def run(self) -> Tuple[int, str]:
        """Return (exit status, message); only a stale rule is an error."""
        found = {d.reason: d for d in self.decorators if d.reason}
        unreadable = [d for d in self.decorators if not d.reason]
        without_rule = sorted(set(found) - self.rules.reasons)
        stale = sorted(self.rules.reasons - set(found))

        if not (without_rule or stale or unreadable):
            return 0, (
                f"qa_selection: {len(self.rules)} rules, {len(found)} skip decorators, no drift"
            )

        report = []
        if stale:
            report += [
                "tests/qa_selection/rules.json is out of date with the skip decorators.",
                "This is drift, not a test regression: rules are keyed on reason",
                "strings, so editing one leaves its rule matching nothing.",
                "",
                "Rules matching no decorator -- the reason was reworded, or the",
                "decorator was removed. Fix the reason in rules.json, or drop the rule:",
            ]
            report += [f"  {reason!r}" for reason in stale]
        if without_rule:
            report += [
                "",
                f"{len(without_rule)} decorator(s) with no rule (reported, not an error).",
                "These are kept for every machine and reported as an unknown reason at",
                "selection time. Add a rule if the target machine can decide it:",
            ]
            report += [
                f"  {found[reason].location}\n    reason={reason!r}" for reason in without_rule
            ]
        if unreadable:
            report += ["", "Decorators whose reason could not be read statically:"]
            report += [f"  {d.location}" for d in unreadable]

        return (1 if stale else 0), "\n".join(report).strip()


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="directory scanned for skip decorators (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    if not args.source_root.is_dir():
        print(f"qa_selection: no such directory: {args.source_root}", file=sys.stderr)
        return 1

    status, message = RuleTableDriftCheck(args.source_root, default_rule_table()).run()
    print(message, file=sys.stderr if status else sys.stdout)
    return status


if __name__ == "__main__":
    sys.exit(main())
