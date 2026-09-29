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
"""Load the marks whose first argument is a requirement rather than a condition.

    markers.json -> ResourceMarkers.load() -> markers.carries_requirement(name)
                                           -> markers.ini_lines()

Membership is the adapter's allowlist: `plugin.py` copies `args[0]` for these
marks and for no others, so a `skipif`'s frozen condition never crosses into
the decision layer.

Each description is the line the integration suite's `pytest.ini` declares for
that marker, so a run outside that suite declares it the same way.
`scripts/check_qa_selection_rules.py` is what keeps the two in step.
"""

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterator, List, Tuple


class MarkerConfigError(ValueError):
    """The marker file has an invalid shape or value."""

    @classmethod
    def check(cls, condition: object, message: str) -> None:
        """Raise with `message` unless `condition` holds."""
        if not condition:
            raise cls(message)


class ResourceMarkers(Mapping):
    """Marker name -> the description `pytest.ini` declares for it."""

    FILE = Path(__file__).with_name("markers.json")

    # `pytest.ini` writes one marker per line in this shape, and
    # `config.addinivalue_line("markers", ...)` takes the same one.
    INI_LINE = "{name}: {description}"

    def __init__(self, descriptions: Dict[str, str]) -> None:
        self._descriptions = dict(descriptions)

    @classmethod
    def load(cls, path: Path = None) -> "ResourceMarkers":
        """Read the marker file, or raise `MarkerConfigError` naming the fault."""
        path = cls.FILE if path is None else path
        try:
            loaded = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            raise MarkerConfigError(f"{path}: {error}") from error
        MarkerConfigError.check(isinstance(loaded, dict), f"{path} must be an object")
        for name, description in loaded.items():
            MarkerConfigError.check(
                isinstance(description, str) and description.strip(),
                f"{path}: {name!r} must have a non-empty description",
            )
        MarkerConfigError.check(loaded, f"{path} declares no markers")
        return cls(loaded)

    def __getitem__(self, name: str) -> str:
        return self._descriptions[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._descriptions)

    def __len__(self) -> int:
        return len(self._descriptions)

    def carries_requirement(self, name: str) -> bool:
        """True when this mark's first argument is a requirement, not a condition."""
        return name in self._descriptions

    def ini_lines(self) -> List[str]:
        """Each marker as the `markers =` line that declares it."""
        return [
            self.INI_LINE.format(name=name, description=description)
            for name, description in self._descriptions.items()
        ]

    def missing_from(self, declared: Tuple[str, ...]) -> Tuple[str, ...]:
        """The names in `declared` this file does not hold, in order.

        Lets a caller state that the markers it reads are a subset of the
        markers the adapter is allowed to carry.
        """
        return tuple(name for name in declared if name not in self._descriptions)


@lru_cache(maxsize=1)
def default_markers() -> ResourceMarkers:
    """The marker file beside this module, read once."""
    return ResourceMarkers.load()
