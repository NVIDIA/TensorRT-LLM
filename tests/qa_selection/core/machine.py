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
"""`--machine`: the target's profile, and whether its card may run a test.

    MachineCheck(profile).blockers(test)     -> why the card drops `test`; () when it may run
    default_catalog().profile_for(name)      -> MachineProfile, from profiles.json
    default_rule_table()                     -> SkipRuleTable, from rules.json

A profile field holds what a run-time probe returns on that machine: `sm` for
`get_sm_version()`, and `device_name`, `device_memory_mib`, `cpu_arch` likewise.
`max_gpu_per_node` is read by no rule; it is the default ladder. A rule is keyed
by a `skipif`'s `reason=`, and a reason with no rule keeps the test.
"""

from __future__ import annotations

import json
import operator
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterator, List, Optional, Set, Tuple

from .marks import CollectedTest, ResourceMarker, ResourceMarkers, default_markers


class MachineCheck:
    """Whether one machine's card may run a test: its skip rules and per-card markers."""

    def __init__(
        self,
        profile: MachineProfile,
        rules: Optional[SkipRuleTable] = None,
        markers: Optional[ResourceMarkers] = None,
    ) -> None:
        self.profile = profile
        self.rules = rules if rules is not None else default_rule_table()
        self.markers = markers if markers is not None else default_markers()

    def blockers(self, test: CollectedTest) -> Tuple[str, ...]:
        """Every reason the card drops `test`; empty when it may run."""
        return tuple(self.skipif_blockers(test) + self.resource_blockers(test))

    def skipif_blockers(self, test: CollectedTest) -> List[str]:
        """The reason of each `skipif`, at any level, whose rule holds for the profile.

        A `skipif` whose reason has no rule keeps the test; its condition is never read.
        """
        blockers: List[str] = []
        for mark in test.iter_markers("skipif"):
            reason = mark.skipif_reason
            rule = self.rules.get(reason)
            if rule is not None and rule.blocks(self.profile):
                blockers.append(reason)
        return blockers

    def resource_blockers(self, test: CollectedTest) -> List[str]:
        """The per-card marker blockers: device name, then device memory."""
        return self.device_name_blockers(test) + self.device_memory_blockers(test)

    def device_name_blockers(self, test: CollectedTest) -> List[str]:
        """One blocker per keyword list matching nothing in the device name."""
        device_name = self.profile.device_name
        return [
            f"{marker.name}: {device_name!r} contains none of {list(keywords)}"
            for marker in self.markers.bounding(ResourceMarker.DEVICE_NAME)
            for keywords in test.requirements(marker)
            if not any(keyword in device_name for keyword in keywords)
        ]

    def device_memory_blockers(self, test: CollectedTest) -> List[str]:
        """One blocker per memory requirement above the card's."""
        available = self.profile.device_memory_mib
        return [
            f"{marker.name}: needs {int(required)} MiB, target has {available} MiB"
            for marker in self.markers.bounding(ResourceMarker.DEVICE_MEMORY_MIB)
            for required in test.requirements(marker)
            if available < int(required)
        ]


@dataclass(frozen=True)
class MachineProfile:
    """One machine's facts, as the skip rules read them."""

    name: str
    sm: int
    device_name: str
    device_memory_mib: int
    max_gpu_per_node: int
    cpu_arch: str

    @classmethod
    def from_mapping(cls, source: Path, name: str, values: object) -> MachineProfile:
        """One catalogue entry, which must hold exactly the profile's fields."""
        try:
            return cls(name=name, **values)
        except TypeError as error:
            raise ProfileConfigError(f"{source}: profile {name!r}: {error}") from error


class MachineCatalog(Mapping):
    """The machines `--machine` accepts, by name."""

    def __init__(self, profiles: Mapping[str, MachineProfile]) -> None:
        self._profiles = dict(profiles)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> MachineCatalog:
        """Read `profiles.json`, or `path`."""
        source = path or Path(__file__).with_name("profiles.json")
        try:
            document = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ProfileConfigError(
                f"{source}:{error.lineno}: invalid JSON: {error.msg}"
            ) from error

        ProfileConfigError.check(
            isinstance(document, dict) and document,
            f"{source}: expected a non-empty object",
        )
        return cls(
            {
                name: MachineProfile.from_mapping(source, name, values)
                for name, values in document.items()
            }
        )

    def __getitem__(self, name: str) -> MachineProfile:
        try:
            return self._profiles[name]
        except KeyError:
            raise KeyError(
                f"unknown machine {name!r}; known machines are {', '.join(self)}"
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._profiles))

    def __len__(self) -> int:
        return len(self._profiles)

    def profile_for(self, name: str) -> MachineProfile:
        """The named profile; the KeyError lists the known names."""
        return self[name]


class SkipRuleTable(Mapping):
    """The skip rules, keyed by `reason=` string."""

    def __init__(self, rules: Mapping[str, SkipRule]) -> None:
        self._rules = dict(rules)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> SkipRuleTable:
        """Read and validate `rules.json`, or `path`."""
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
        named: Set[str] = set()
        for index, entry in enumerate(entries, start=1):
            rule = SkipRule.from_mapping(source, index, entry)
            RuleConfigError.check(
                rule.reason not in rules, f"{source}: duplicate reason {rule.reason!r}"
            )
            RuleConfigError.check(
                rule.decorator not in named,
                f"{source}: duplicate decorator {rule.decorator}",
            )
            rules[rule.reason] = rule
            named.add(rule.decorator)
        return cls(rules)

    def __getitem__(self, reason: str) -> SkipRule:
        return self._rules[reason]

    def __iter__(self) -> Iterator[str]:
        return iter(self._rules)

    def __len__(self) -> int:
        return len(self._rules)

    @property
    def decorators(self) -> FrozenSet[str]:
        """Every `skip_*` name this file declares a rule for."""
        return frozenset(rule.decorator for rule in self._rules.values())


@dataclass(frozen=True)
class SkipRule:
    """One `skipif` decorator's rule: its name, its reason, and when it skips."""

    decorator: str
    reason: str
    condition: MachineCondition

    @classmethod
    def from_mapping(cls, source: Path, index: int, entry: object) -> SkipRule:
        """Validate one `rules` entry."""
        where = f"{source}: rule #{index}"
        RuleConfigError.check(isinstance(entry, dict), f"{where} must be an object")
        unknown = entry.keys() - {"decorator", "reason", "note", "skip_when"}
        RuleConfigError.check(
            not unknown, f"{where} has unknown keys: {', '.join(sorted(unknown))}"
        )

        decorator = entry.get("decorator")
        RuleConfigError.check(
            isinstance(decorator, str) and decorator,
            f"{where}: decorator must be a non-empty string, got {decorator!r}",
        )

        where = f"{source}: rule {decorator}"
        reason = entry.get("reason")
        RuleConfigError.check(
            isinstance(reason, str) and reason,
            f"{where}: reason must be a non-empty string, got {reason!r}",
        )

        RuleConfigError.check("skip_when" in entry, f"{where} has no skip_when")
        return cls(
            decorator=decorator,
            reason=reason,
            condition=MachineCondition.from_mapping(where, entry["skip_when"]),
        )

    def blocks(self, profile: MachineProfile) -> bool:
        """True when `profile` skips a test carrying this rule."""
        return self.condition.holds_for(profile)


@dataclass(frozen=True)
class MachineCondition:
    """Field tests that must all hold for a rule to skip, e.g. `{"sm": {"lt": 90}}`."""

    NUMERIC_OPERATORS = ("lt", "le", "gt", "ge", "eq", "ne")
    STRING_OPERATORS = ("eq", "ne", "contains", "not_contains")

    # Operators each MachineProfile field accepts.
    FIELD_OPERATORS = {
        "sm": NUMERIC_OPERATORS,
        "device_memory_mib": NUMERIC_OPERATORS,
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
    def from_mapping(cls, where: str, spec: object) -> MachineCondition:
        """Validate one `skip_when` object against `FIELD_OPERATORS`."""
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


class ProfileConfigError(ValueError):
    """`profiles.json` has an invalid shape or value."""

    @classmethod
    def check(cls, condition: object, message: str) -> None:
        """Raise with `message` unless `condition` holds."""
        if not condition:
            raise cls(message)


class RuleConfigError(ValueError):
    """`rules.json` has an invalid shape or value."""

    @classmethod
    def check(cls, condition: object, message: str) -> None:
        """Raise with `message` unless `condition` holds."""
        if not condition:
            raise cls(message)


@lru_cache(maxsize=1)
def default_catalog() -> MachineCatalog:
    """`profiles.json`, read once."""
    return MachineCatalog.load()


@lru_cache(maxsize=1)
def default_rule_table() -> SkipRuleTable:
    """`rules.json`, read and validated once."""
    return SkipRuleTable.load()
