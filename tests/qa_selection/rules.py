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
"""Load the skip rules that decide whether a target machine would skip a test.

    rules.json -> SkipRuleTable.load() -> rule.blocks(profile)

A rule is keyed by the `reason=` string of a `pytest.mark.skipif` decorator in
the integration conftest. Its condition draws on a closed vocabulary of
`MachineProfile` fields and operators, both validated at load. A reason absent
from the file has no rule and is never deselected.
"""

import json
import operator
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterator, Optional, Tuple

from .machines import MachineProfile

NUMERIC_OPERATORS = ("lt", "le", "gt", "ge", "eq", "ne")
STRING_OPERATORS = ("eq", "ne", "contains", "not_contains")


class RuleConfigError(ValueError):
    """The rule file has an invalid shape or value."""

    @classmethod
    def check(cls, condition: object, message: str) -> None:
        """Raise with `message` unless `condition` holds."""
        if not condition:
            raise cls(message)


@dataclass(frozen=True)
class MachineCondition:
    """Field tests that must all hold for a machine to skip a test.

    Built from a `skip_when` object, e.g. `{"sm": {"lt": 90}}`.
    """

    # Operators each MachineProfile field accepts.
    FIELD_OPERATORS = {
        "sm": NUMERIC_OPERATORS,
        "device_memory_mib": NUMERIC_OPERATORS,
        "gpu_count": NUMERIC_OPERATORS,
        "max_gpu_per_node": NUMERIC_OPERATORS,
        "device_name": STRING_OPERATORS,
        "cpu_arch": STRING_OPERATORS,
    }

    OPERATORS = {
        "lt": operator.lt,
        "le": operator.le,
        "gt": operator.gt,
        "ge": operator.ge,
        "eq": operator.eq,
        "ne": operator.ne,
        "contains": lambda actual, expected: expected in actual,
        "not_contains": lambda actual, expected: expected not in actual,
    }

    # (field, operator, expected) per test, ANDed together.
    tests: Tuple[Tuple[str, str, Any], ...]

    @classmethod
    def from_mapping(cls, where: str, spec: object) -> "MachineCondition":
        """Validate one `skip_when` mapping, rejecting unknown fields and operators."""
        RuleConfigError.check(
            isinstance(spec, dict) and spec,
            f"{where}: skip_when must be a non-empty object",
        )

        tests = []
        for field, comparisons in spec.items():
            allowed = cls.FIELD_OPERATORS.get(field)
            RuleConfigError.check(
                allowed is not None,
                f"{where}: unknown field {field!r}; "
                f"known fields are {', '.join(sorted(cls.FIELD_OPERATORS))}",
            )
            RuleConfigError.check(
                isinstance(comparisons, dict) and comparisons,
                f"{where}.{field} must be a non-empty object of operator to value",
            )
            for name, expected in comparisons.items():
                RuleConfigError.check(
                    name in allowed,
                    f"{where}.{field}: unknown operator {name!r}; "
                    f"{field} accepts {', '.join(allowed)}",
                )
                tests.append((field, name, expected))

        return cls(tests=tuple(tests))

    def holds_for(self, profile: MachineProfile) -> bool:
        """True when every test holds for `profile`."""
        return all(
            self.OPERATORS[name](getattr(profile, field), expected)
            for field, name, expected in self.tests
        )


@dataclass(frozen=True)
class SkipRule:
    """One skipif decorator, identified by its reason string."""

    reason: str
    condition: MachineCondition

    @classmethod
    def from_mapping(cls, source: Path, index: int, entry: object) -> "SkipRule":
        """Validate one `rules` entry and build a rule from it."""
        where = f"{source}: rule #{index}"
        RuleConfigError.check(isinstance(entry, dict), f"{where} must be an object")
        unknown = entry.keys() - {"reason", "note", "skip_when"}
        RuleConfigError.check(
            not unknown, f"{where} has unknown keys: {', '.join(sorted(unknown))}"
        )

        reason = entry.get("reason")
        RuleConfigError.check(
            isinstance(reason, str) and reason,
            f"{where}: reason must be a non-empty string, got {reason!r}",
        )

        where = f"{source}: rule {reason!r}"
        RuleConfigError.check("skip_when" in entry, f"{where} has no skip_when")
        return cls(
            reason=reason,
            condition=MachineCondition.from_mapping(where, entry["skip_when"]),
        )

    def blocks(self, profile: MachineProfile) -> bool:
        """True when `profile` would skip a test carrying this rule."""
        return self.condition.holds_for(profile)


class SkipRuleTable(Mapping):
    """The skip rules selection can decide, keyed by reason string."""

    def __init__(self, rules: Mapping[str, SkipRule]) -> None:
        self._rules = dict(rules)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "SkipRuleTable":
        """Read and validate the rule file."""
        source = path or Path(__file__).with_name("rules.json")
        try:
            document = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise RuleConfigError(f"{source}:{error.lineno}: invalid JSON: {error.msg}") from error

        RuleConfigError.check(
            isinstance(document, dict) and "rules" in document,
            f"{source}: expected an object with a 'rules' key",
        )
        entries = document["rules"]
        RuleConfigError.check(
            isinstance(entries, list) and entries,
            f"{source}: 'rules' must be a non-empty list",
        )

        rules: Dict[str, SkipRule] = {}
        for index, entry in enumerate(entries, start=1):
            rule = SkipRule.from_mapping(source, index, entry)
            RuleConfigError.check(
                rule.reason not in rules, f"{source}: duplicate reason {rule.reason!r}"
            )
            rules[rule.reason] = rule
        return cls(rules)

    def __getitem__(self, reason: str) -> SkipRule:
        return self._rules[reason]

    def __iter__(self) -> Iterator[str]:
        return iter(self._rules)

    def __len__(self) -> int:
        return len(self._rules)

    @property
    def reasons(self) -> FrozenSet[str]:
        """Every reason string this file declares."""
        return frozenset(self._rules)


@lru_cache(maxsize=1)
def default_rule_table() -> SkipRuleTable:
    """The rules shipped beside this module, read and validated once."""
    return SkipRuleTable.load()
