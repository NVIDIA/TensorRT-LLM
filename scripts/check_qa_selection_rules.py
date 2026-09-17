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

Every rule is resolved through the `skip_*` decorator it names, not through the
reason string it matches on. Names survive edits to prose, so the check can say
what a reworded reason is now and hand back the replacement.

A rule whose decorator is gone, or whose reason has drifted from it, is an
error: either way the rule can never match a collected mark again, and its
tests ship to every machine. A decorator with no rule is only reported -- the
table is curated, not an inventory.

Decorators are read with `ast`, never imported.
"""

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_ROOT = REPO_ROOT / "tests" / "integration" / "defs"
RULES_FILE = "tests/qa_selection/rules.json"

sys.path.insert(0, str(REPO_ROOT / "tests"))
from qa_selection.rules import SkipRule, SkipRuleTable, default_rule_table  # noqa: E402


@dataclass(frozen=True)
class SkipDecorator:
    """One module-level `skip_* = pytest.mark.skipif(...)` assignment."""

    name: str
    reason: str
    path: Path

    @property
    def location(self) -> str:
        return f"{self.name} ({self.path})"


@dataclass(frozen=True)
class InlineSkipif:
    """One `pytest.mark.skipif(...)` written at a test rather than named."""

    reason: str
    path: Path
    lineno: int

    @property
    def location(self) -> str:
        return f"{self.path}:{self.lineno}"


class SkipifScanner:
    """Reads the suite's skipif decorators, named and inline, with `ast`.

    Only a module-level name can back a rule; the inline ones are kept so that
    a rule naming an unnamed skip is diagnosed rather than reported as stale.
    """

    def __init__(self, root: Path) -> None:
        self.named: Dict[str, SkipDecorator] = {}
        self.inline: List[InlineSkipif] = []
        for path in sorted(root.rglob("*.py")):
            self.read(path)

    def read(self, path: Path) -> None:
        """Record every skipif in one source file."""
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.Assign) and self.is_skipif_call(node.value):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.startswith("skip_"):
                        reason = self.reason_of(node.value)
                        self.named[target.id] = SkipDecorator(target.id, reason, path)

        for node in ast.walk(tree):
            decorators = getattr(node, "decorator_list", ())
            self.inline += [
                InlineSkipif(self.reason_of(call), path, call.lineno)
                for call in decorators
                if self.is_skipif_call(call)
            ]

    def inline_with_reason(self, reason: str) -> Optional[InlineSkipif]:
        """The first inline skipif carrying `reason`, or None."""
        return next((site for site in self.inline if site.reason == reason), None)

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


@dataclass
class RuleFaults:
    """Rules their declared decorator no longer backs, grouped by cause."""

    missing: List[SkipRule] = field(default_factory=list)
    inline_only: List[Tuple[SkipRule, InlineSkipif]] = field(default_factory=list)
    drifted: List[Tuple[SkipRule, SkipDecorator]] = field(default_factory=list)
    unverifiable: List[Tuple[SkipRule, SkipDecorator]] = field(default_factory=list)

    def __bool__(self) -> bool:
        """True when some rule can never match a collected mark again."""
        return bool(self.missing or self.inline_only or self.drifted)

    def report_lines(self) -> List[str]:
        """The fault sections, most actionable first."""
        report: List[str] = []
        if self:
            report += [
                f"{RULES_FILE} is out of date with the skip decorators.",
                "This is drift, not a test regression: a rule that no longer mirrors a",
                "live decorator matches nothing, so its tests ship to every machine and",
                "skip on the node.",
            ]
        report += self.drifted_lines()
        report += self.missing_lines()
        report += self.inline_only_lines()
        report += self.unverifiable_lines()
        return report

    def drifted_lines(self) -> List[str]:
        """A reworded reason, with the replacement string ready to paste."""
        if not self.drifted:
            return []
        report = [
            "",
            f"{len(self.drifted)} rule(s) whose reason has been reworded.",
            'Paste the new string over the rule\'s "reason" in the file above:',
        ]
        for rule, decorator in self.drifted:
            report += [
                f"  {decorator.location}",
                f"    rules.json has: {json.dumps(rule.reason)}",
                f"    decorator now:  {json.dumps(decorator.reason)}",
            ]
        return report

    def missing_lines(self) -> List[str]:
        """A named decorator that is gone: the rule is dead, so say so."""
        if not self.missing:
            return []
        report = [
            "",
            f"{len(self.missing)} rule(s) naming a decorator that no longer exists.",
            "The decorator was renamed or removed, so the rule is a silent no-op.",
            "Remove the rule from the file above, or restore the decorator:",
        ]
        report += [f"  {rule.decorator}\n    reason={rule.reason!r}" for rule in self.missing]
        return report

    def inline_only_lines(self) -> List[str]:
        """A rule backed only by an unnamed skipif: adopt the name, do not widen."""
        if not self.inline_only:
            return []
        report = [
            "",
            f"{len(self.inline_only)} rule(s) whose reason is carried only by an inline skipif.",
            "A curated rule must name a module-level decorator: a skip worth curating",
            "is a skip worth naming. Give the site below a module-level name matching",
            'the rule\'s "decorator", or drop the rule:',
        ]
        report += [
            f"  {rule.decorator} declared, but found at {site.location}"
            for rule, site in self.inline_only
        ]
        return report

    def unverifiable_lines(self) -> List[str]:
        """A decorator whose reason is computed, so drift cannot be checked."""
        if not self.unverifiable:
            return []
        report = [
            "",
            f"{len(self.unverifiable)} rule(s) whose decorator has no literal reason=",
            "(reported, not an error). Their reason cannot be compared statically:",
        ]
        report += [f"  {decorator.location}" for _, decorator in self.unverifiable]
        return report


class RuleTableDriftCheck:
    """Resolves every rule through the decorator it names."""

    def __init__(self, root: Path, rules: SkipRuleTable) -> None:
        self.scanner = SkipifScanner(root)
        self.rules = rules

    def run(self) -> Tuple[int, str]:
        """Return (exit status, message); only a rule that cannot match is an error."""
        faults = self.faults()
        uncurated = sorted(set(self.scanner.named) - self.rules.decorators)
        report = faults.report_lines() + self.uncurated_lines(uncurated)

        if not report:
            return 0, (
                f"qa_selection: {len(self.rules)} rules, "
                f"{len(self.scanner.named)} skip decorators, no drift"
            )
        return (1 if faults else 0), "\n".join(report).strip()

    def faults(self) -> RuleFaults:
        """Classify every rule against the decorator it declares."""
        faults = RuleFaults()
        for rule in self.rules.values():
            decorator = self.scanner.named.get(rule.decorator)
            if decorator is None:
                inline = self.scanner.inline_with_reason(rule.reason)
                if inline is None:
                    faults.missing.append(rule)
                else:
                    faults.inline_only.append((rule, inline))
            elif not decorator.reason:
                faults.unverifiable.append((rule, decorator))
            elif decorator.reason != rule.reason:
                faults.drifted.append((rule, decorator))
        return faults

    def uncurated_lines(self, uncurated: List[str]) -> List[str]:
        """Decorators with no rule: kept for every machine, by design."""
        if not uncurated:
            return []
        report = [
            "",
            f"{len(uncurated)} decorator(s) with no rule (reported, not an error).",
            "The table is curated, not an inventory: these are kept for every machine",
            "and reported as an unknown reason at selection time. Add a rule to",
            f"{RULES_FILE} if the target machine can decide one:",
        ]
        report += [
            f"  {self.scanner.named[name].location}\n    reason={self.scanner.named[name].reason!r}"
            for name in uncurated
        ]
        return report


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
