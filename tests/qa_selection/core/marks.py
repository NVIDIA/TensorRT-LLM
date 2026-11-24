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
"""What a collected test asks for, read from its marks.

    CollectedTest(nodeid, marks).requirements(marker)  -> what `marker` asks of the test
    default_markers().bounding(ResourceMarker.GPUS)    -> the markers bounding one profile fact
    default_markers().carries_requirement(name)        -> whether `args[0]` is a requirement
    default_markers().ini_lines()                      -> the `markers =` lines declaring them

A resource marker's first argument is a requirement, as `skip_less_device(8)`
asks for 8 GPUs; `markers.json` lists them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class CollectedTest:
    """One collected item: its node id and its marks, closest level first."""

    nodeid: str
    marks: Sequence[Mark] = ()

    def iter_markers(self, name: str) -> Iterator[Mark]:
        """Every mark of this name, closest level first."""
        return (mark for mark in self.marks if mark.name == name)

    def requirements(self, marker: ResourceMarker) -> Tuple[Any, ...]:
        """What `marker` asks of this test: the closest mark's, or every level's, by its `read`."""
        marks = list(self.iter_markers(marker.name))
        if not marker.every_level:
            marks = marks[:1]
        return tuple(mark.requirement for mark in marks if mark.requirement is not None)


@dataclass(frozen=True)
class Mark:
    """The part of a pytest mark `core/` reads: its name, a requirement, a `reason=`."""

    name: str
    args: Tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    @property
    def skipif_reason(self) -> Optional[str]:
        """This skipif's reason string, or None. Keyword form only."""
        reason = self.kwargs.get("reason")
        return reason if isinstance(reason, str) and reason else None

    @property
    def requirement(self) -> Optional[Any]:
        """What a resource marker asks for, as `skip_less_device(8)` asks for 8."""
        return self.args[0] if self.args else None


class ResourceMarkers(Mapping):
    """Marker name -> `ResourceMarker`, in file order."""

    FILE = Path(__file__).with_name("markers.json")

    # One `markers =` line, as `pytest.ini` and `addinivalue_line` take it.
    INI_LINE = "{name}: {description}"

    def __init__(self, markers: Dict[str, ResourceMarker]) -> None:
        self._markers = dict(markers)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> ResourceMarkers:
        """Read `markers.json`, or `path`."""
        source = cls.FILE if path is None else path
        try:
            document = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise MarkerConfigError(f"{source}: {error}") from error
        MarkerConfigError.check(
            isinstance(document, dict) and isinstance(document.get("markers"), dict),
            f"{source}: expected an object with a 'markers' object",
        )
        entries = document["markers"]
        MarkerConfigError.check(entries, f"{source} declares no markers")
        return cls(
            {
                name: ResourceMarker.from_mapping(source, name, values)
                for name, values in entries.items()
            }
        )

    def __getitem__(self, name: str) -> ResourceMarker:
        return self._markers[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._markers)

    def __len__(self) -> int:
        return len(self._markers)

    def carries_requirement(self, name: str) -> bool:
        """True when this mark's first argument is a requirement, not a condition."""
        return name in self._markers

    def bounding(self, bounds: str) -> Tuple[ResourceMarker, ...]:
        """The markers compared with one machine fact, in file order."""
        return tuple(marker for marker in self._markers.values() if marker.bounds == bounds)

    def ini_lines(self) -> List[str]:
        """Each marker as the `markers =` line that declares it."""
        return [
            self.INI_LINE.format(name=marker.name, description=marker.description)
            for marker in self._markers.values()
        ]


@dataclass(frozen=True)
class ResourceMarker:
    """One resource marker: its description, what it bounds, and how it is read.

    `bounds` is the profile fact it is compared with, or None when not evaluated;
    `read` is `CLOSEST` or `EVERY_LEVEL`.
    """

    GPUS = "gpus"
    DEVICE_MEMORY_MIB = "device_memory_mib"
    DEVICE_NAME = "device_name"
    BOUNDS = (GPUS, DEVICE_MEMORY_MIB, DEVICE_NAME)

    CLOSEST = "closest"
    EVERY_LEVEL = "every_level"
    READS = (CLOSEST, EVERY_LEVEL)

    FIELDS = ("description", "bounds", "read")

    name: str
    description: str
    bounds: Optional[str]
    read: str

    @classmethod
    def from_mapping(cls, source: Path, name: str, values: object) -> ResourceMarker:
        """Validate one entry, naming the marker and field that fail."""
        where = f"{source}: marker {name!r}"
        MarkerConfigError.check(isinstance(values, dict), f"{where} must be an object")
        missing = set(cls.FIELDS) - values.keys()
        MarkerConfigError.check(not missing, f"{where} is missing: {', '.join(sorted(missing))}")
        unknown = values.keys() - set(cls.FIELDS)
        MarkerConfigError.check(
            not unknown, f"{where} has unknown fields: {', '.join(sorted(unknown))}"
        )
        description, bounds, read = (values[field] for field in cls.FIELDS)
        MarkerConfigError.check(
            isinstance(description, str) and description.strip(),
            f"{where}.description must be a non-empty string",
        )
        MarkerConfigError.check(
            bounds is None or bounds in cls.BOUNDS,
            f"{where}.bounds must be null or one of {', '.join(cls.BOUNDS)}, got {bounds!r}",
        )
        MarkerConfigError.check(
            read in cls.READS,
            f"{where}.read must be one of {', '.join(cls.READS)}, got {read!r}",
        )
        return cls(name=name, description=description, bounds=bounds, read=read)

    @property
    def every_level(self) -> bool:
        """True when every level's mark counts, not only the nearest."""
        return self.read == self.EVERY_LEVEL


class MarkerConfigError(ValueError):
    """`markers.json` has an invalid shape or value."""

    @classmethod
    def check(cls, condition: object, message: str) -> None:
        """Raise with `message` unless `condition` holds."""
        if not condition:
            raise cls(message)


@lru_cache(maxsize=1)
def default_markers() -> ResourceMarkers:
    """`markers.json`, read and validated once."""
    return ResourceMarkers.load()
