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
"""Turn JSON machine facts into immutable allocation profiles.

    profiles.json -> MachineCatalog.load() -> profile_for(name, gpu_count) -> MachineProfile

Each field answers a run-time probe without running it: `sm` for
`get_sm_version()` (major * 10 + minor, so 10.3 is 103), `device_name` for
`get_gpu_device_list()`, `device_memory_mib` for `get_device_memory()`,
`gpu_count` for `get_device_count()`, `cpu_arch` for `platform.machine()`.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Optional


class ProfileConfigError(ValueError):
    """A machine-profile file has an invalid shape or value."""

    @classmethod
    def check(cls, condition: object, message: str) -> None:
        """Raise with `message` unless `condition` holds."""
        if not condition:
            raise cls(message)


def is_positive_int(value: object) -> bool:
    """True for a whole number above zero, as every numeric profile field must be."""
    # bool is an int subclass, so `true` in a numeric field must be rejected.
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


@dataclass(frozen=True)
class MachineProfile:
    """One machine, described the way the skip rules ask about it."""

    name: str
    sm: int
    device_name: str
    device_memory_mib: int
    max_gpu_per_node: int
    cpu_arch: str
    gpu_count: int

    @classmethod
    def from_mapping(cls, source: Path, name: str, values: object) -> "MachineProfile":
        """Validate one raw catalogue entry and build a profile from it."""
        return CatalogEntry(source, name, values).to_profile()

    def with_gpu_count(self, count: int) -> "MachineProfile":
        """Return this machine sized for one allocation, leaving self unchanged."""
        ProfileConfigError.check(
            is_positive_int(count),
            f"{self.name}: GPU count must be a positive integer, got {count!r}",
        )
        return replace(self, gpu_count=count)


class CatalogEntry:
    """One unvalidated entry from the catalogue file, with its location.

    Each check reads as a statement about this entry, and each failure names the
    profile and field that caused it.
    """

    DECLARED_FIELDS = ("sm", "device_name", "device_memory_mib", "max_gpu_per_node", "cpu_arch")
    NUMERIC_FIELDS = ("sm", "device_memory_mib", "max_gpu_per_node")
    CPU_ARCHS = ("aarch64", "x86_64")

    def __init__(self, source: Path, name: str, values: object) -> None:
        self.where = f"{source}: profile {name!r}"
        ProfileConfigError.check(isinstance(values, dict), f"{self.where} must be an object")
        self.name = name
        self.values = values

    def to_profile(self) -> MachineProfile:
        """Run every check, then build the profile."""
        self.check_fields_declared()
        for field in self.NUMERIC_FIELDS:
            self.check_positive_int(field)
        self.check_device_name()
        self.check_cpu_arch()
        # A fresh profile is sized to a full node until a caller narrows it.
        return MachineProfile(
            name=self.name, gpu_count=self.values["max_gpu_per_node"], **self.values
        )

    def check_fields_declared(self) -> None:
        """Every declared field present, and nothing else.

        Rejecting unknown fields makes a stale one fail here, not go unnoticed.
        """
        declared = set(self.DECLARED_FIELDS)
        missing = declared - self.values.keys()
        ProfileConfigError.check(
            not missing, f"{self.where} is missing: {', '.join(sorted(missing))}"
        )
        unknown = self.values.keys() - declared
        ProfileConfigError.check(
            not unknown, f"{self.where} has unknown fields: {', '.join(sorted(unknown))}"
        )

    def check_positive_int(self, field: str) -> None:
        """One numeric field holds a whole number above zero."""
        ProfileConfigError.check(
            is_positive_int(self.values[field]),
            f"{self.where}.{field} must be a positive integer, got {self.values[field]!r}",
        )

    def check_device_name(self) -> None:
        """device_name is a non-empty string; the skip rules match substrings of it."""
        device_name = self.values["device_name"]
        ProfileConfigError.check(
            isinstance(device_name, str) and device_name.strip(),
            f"{self.where}.device_name must be a non-empty string, got {device_name!r}",
        )

    def check_cpu_arch(self) -> None:
        """cpu_arch is one the skip rules recognise."""
        cpu_arch = self.values["cpu_arch"]
        ProfileConfigError.check(
            cpu_arch in self.CPU_ARCHS,
            f"{self.where}.cpu_arch must be one of {', '.join(self.CPU_ARCHS)}, got {cpu_arch!r}",
        )


class MachineCatalog(Mapping):
    """The machines selection can target, keyed by name.

    Read-only, so sizing a profile for one allocation cannot corrupt the entry
    the next caller reads.
    """

    def __init__(self, profiles: Mapping[str, MachineProfile]) -> None:
        self._profiles = dict(profiles)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "MachineCatalog":
        """Read and validate the catalogue file."""
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
            # Hardware moves faster than the catalogue: a machine we cannot
            # describe must fail loudly rather than select wrongly.
            raise KeyError(
                f"unknown machine {name!r}; known machines are {', '.join(self)}"
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._profiles))

    def __len__(self) -> int:
        return len(self._profiles)

    def profile_for(self, name: str, gpu_count: Optional[int] = None) -> MachineProfile:
        """Return the named profile, optionally sized for one allocation."""
        profile = self[name]
        return profile if gpu_count is None else profile.with_gpu_count(gpu_count)


@lru_cache(maxsize=1)
def default_catalog() -> MachineCatalog:
    """The catalogue shipped beside this module, read and validated once."""
    return MachineCatalog.load()
