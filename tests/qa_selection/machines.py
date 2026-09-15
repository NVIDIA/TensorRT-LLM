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

Selection runs on a login node with no GPU, so the facts the skip rules would
normally probe for at run time are read from committed data instead. This
module imports only the standard library.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Optional


class ProfileConfigError(ValueError):
    """A machine-profile file has an invalid shape or value."""


def _require(condition: object, message: str) -> None:
    """Reject a malformed catalogue entry, naming the profile and field."""
    if not condition:
        raise ProfileConfigError(message)


def _is_positive_int(value: object) -> bool:
    # bool is an int subclass, so `true` in a numeric field must be rejected.
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


@dataclass(frozen=True)
class MachineProfile:
    """One machine, described the way the skip rules ask about it.

    Each field answers a run-time probe without running it: `sm` for
    `get_sm_version()` (major * 10 + minor, so 10.3 is 103), `device_name` for
    `get_gpu_device_list()`, `device_memory_mib` for `get_device_memory()`,
    `gpu_count` for `get_device_count()`, `cpu_arch` for `platform.machine()`,
    `mpi_world_size` for `get_mpi_world_size()`.
    """

    name: str
    sm: int
    device_name: str
    device_memory_mib: int
    max_gpu_per_node: int
    cpu_arch: str
    gpu_count: int
    mpi_world_size: int

    @classmethod
    def from_mapping(cls, source: Path, name: str, values: object) -> "MachineProfile":
        """Validate one raw catalogue entry and build a profile from it."""
        where = f"{source}: profile {name!r}"
        _require(isinstance(values, dict), f"{where} must be an object")

        declared = {
            "sm",
            "device_name",
            "device_memory_mib",
            "max_gpu_per_node",
            "cpu_arch",
        }
        missing = declared - values.keys()
        _require(not missing, f"{where} is missing: {', '.join(sorted(missing))}")
        # Rejecting unknown fields keeps the schema self-enforcing: a stale
        # field fails here instead of being silently ignored.
        unknown = values.keys() - declared
        _require(not unknown, f"{where} has unknown fields: {', '.join(sorted(unknown))}")

        for field in ("sm", "device_memory_mib", "max_gpu_per_node"):
            _require(
                _is_positive_int(values[field]),
                f"{where}.{field} must be a positive integer, got {values[field]!r}",
            )

        device_name = values["device_name"]
        _require(
            isinstance(device_name, str) and device_name.strip(),
            f"{where}.device_name must be a non-empty string, got {device_name!r}",
        )

        cpu_arch = values["cpu_arch"]
        _require(
            cpu_arch in ("aarch64", "x86_64"),
            f"{where}.cpu_arch must be aarch64 or x86_64, got {cpu_arch!r}",
        )

        node_gpus = values["max_gpu_per_node"]
        return cls(name=name, gpu_count=node_gpus, mpi_world_size=node_gpus, **values)

    def with_gpu_count(self, count: int) -> "MachineProfile":
        """Return this machine sized for one allocation, leaving self unchanged.

        `mpi_world_size` tracks the GPU count, one rank per GPU.
        """
        _require(
            _is_positive_int(count),
            f"{self.name}: GPU count must be a positive integer, got {count!r}",
        )
        return replace(self, gpu_count=count, mpi_world_size=count)


class MachineCatalog(Mapping):
    """The machines selection can target, keyed by name.

    Read-only by construction, so sizing a profile for one allocation cannot
    corrupt the entry the next caller reads.
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

        _require(
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
