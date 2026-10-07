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
"""What one run's output files are called.

    ArtifactNames.written_by(machine, ladder)  -> every file this run writes
    ArtifactNames.orphans_in(dir, ...)         -> this machine's stale files

`request.py` checks the output directory before collecting and `report.py`
writes the files. Both get the names here, so the check cannot disagree with
what is written.
"""

import re
from pathlib import Path
from typing import List, Tuple

from .ladder import Ladder


class ArtifactNames:
    """The filenames a run produces, and which files in a directory are its own."""

    RECORD_SUFFIX = ".json"
    IDS_SUFFIX = ".ids"
    RUNG_UNIT = "gpu"

    @classmethod
    def record(cls, machine: str) -> str:
        """The JSON record, written on every run."""
        return f"{machine}{cls.RECORD_SUFFIX}"

    @classmethod
    def rung_ids(cls, machine: str, rung: int) -> str:
        """One rung's identifier list."""
        return f"{machine}-{rung}{cls.RUNG_UNIT}{cls.IDS_SUFFIX}"

    @classmethod
    def written_by(cls, machine: str, ladder: Ladder) -> Tuple[str, ...]:
        """Every file a run for `machine` on `ladder` writes."""
        return (cls.record(machine),) + tuple(cls.rung_ids(machine, r) for r in ladder)

    @classmethod
    def shapes_of(cls, machine: str) -> Tuple[re.Pattern, ...]:
        """The filename shapes this machine's files can take in an output directory.

        The record, a rung's list, and `<machine>.ids`, which no run writes: a
        leftover one is therefore always an orphan. Anchored patterns, not a
        `<machine>*` glob: `B200*` would also match `B200X.ids`, a different
        machine's file.
        """
        name = re.escape(machine)
        return (
            re.compile(rf"^{name}{re.escape(cls.RECORD_SUFFIX)}$"),
            re.compile(rf"^{name}{re.escape(cls.IDS_SUFFIX)}$"),
            re.compile(rf"^{name}-[0-9]+{cls.RUNG_UNIT}{re.escape(cls.IDS_SUFFIX)}$"),
        )

    @classmethod
    def orphans_in(cls, out_dir: Path, machine: str, ladder: Ladder) -> List[str]:
        """This machine's files in `out_dir` that this run will not write.

        Other machines' files are ignored, and nothing is deleted: the answer
        is a list for the caller to report.
        """
        written = set(cls.written_by(machine, ladder))
        shapes = cls.shapes_of(machine)
        return sorted(
            entry.name
            for entry in out_dir.iterdir()
            if entry.is_file()
            and entry.name not in written
            and any(shape.match(entry.name) for shape in shapes)
        )
