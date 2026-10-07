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
"""What one run was asked for, resolved and checked before anything is collected.

    SelectionRequest.of(machine=..., ladder=..., rung=..., out_dir=...)

Plain values in, a request or None out, `SelectionError` when the caller must
fix something. The options are named in the messages because they are this
package's vocabulary; nothing here imports pytest, and `plugin.py` turns a
`SelectionError` into a `pytest.UsageError`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .artifacts import ArtifactNames
from .ladder import Ladder
from .machines import MachineProfile, ProfileConfigError, default_catalog


class SelectionError(ValueError):
    """A request the caller must fix, stated as the message they should read."""


@dataclass(frozen=True)
class SelectionRequest:
    """One run's target: a machine, its ladder, and which allocation this run occupies.

    `profile` answers *what the rules allow here*. `ladder` answers *how many
    GPUs, in which allocations*; its largest rung is the run's GPU count.
    """

    machine: str
    profile: MachineProfile
    ladder: Ladder
    target_rung: Optional[int]
    out_dir: Optional[Path]

    @classmethod
    def of(
        cls,
        machine: Optional[str],
        ladder: Optional[str] = None,
        rung: Optional[int] = None,
        out_dir: Optional[str] = None,
    ) -> Optional["SelectionRequest"]:
        """This run's request, or None when no machine is named.

        Returning None before any other value is read is what makes the plugin
        inert: no other option is validated while no machine is named.
        """
        if not machine:
            return None
        profile = cls.profile_for(machine)
        parsed_ladder = cls.ladder_for(ladder, profile)
        target_rung = cls.target_rung_for(rung, parsed_ladder)
        return cls(
            machine=machine,
            profile=profile,
            ladder=parsed_ladder,
            target_rung=target_rung,
            out_dir=cls.out_dir_for(out_dir, machine, parsed_ladder, target_rung),
        )

    @classmethod
    def out_dir_for(
        cls,
        text: Optional[str],
        machine: str,
        ladder: Ladder,
        target_rung: Optional[int],
    ) -> Optional[Path]:
        """Where to write, or None when nothing is written.

        Naming a rung and asking for artifacts at once is a usage error: the
        run without `--rung` writes every rung's list from the same single
        collection.

        The directory is created here rather than at write time so that a run
        which cannot produce its output says so before collecting, and says it
        as a usage error naming the option. The same reason puts the stale-file
        check here: both are answerable before a test module is imported.
        """
        if text is None:
            return None
        if target_rung is not None:
            raise SelectionError(
                "--selection-out-dir: cannot be combined with --rung; the same command "
                "without --rung writes every rung's list in one pass"
            )
        out_dir = Path(text)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise SelectionError(
                f"--selection-out-dir: cannot create {str(out_dir)!r}: {error.strerror}"
            ) from error
        cls.check_no_orphans(out_dir, machine, ladder)
        return out_dir

    @staticmethod
    def check_no_orphans(out_dir: Path, machine: str, ladder: Ladder) -> None:
        """Refuse a directory already holding this machine's stale lists.

        The directory is written into and never emptied, so a leftover list
        from a different ladder would sit beside this run's output claiming to
        be a rung that no longer exists. Another machine's files are not this
        run's business and are never examined.
        """
        orphans = ArtifactNames.orphans_in(out_dir, machine, ladder)
        if not orphans:
            return
        writes = ArtifactNames.written_by(machine, ladder)
        raise SelectionError(
            f"--selection-out-dir: {str(out_dir)!r} already holds "
            f"{', '.join(orphans)} for {machine}, which this run does not overwrite; "
            f"it writes {', '.join(writes)}. "
            f"Remove the leftover file(s), or write to a different directory"
        )

    @staticmethod
    def ladder_for(text: Optional[str], profile: MachineProfile) -> Ladder:
        """The parsed `--ladder`, or one rung of the whole node when it was not given.

        A rung larger than the machine's GPUs per node is a usage error: that
        allocation cannot be requested. A ladder *shorter* than the machine is
        not: `--ladder=4` on an 8-GPU node asks what one 4-GPU allocation of it
        can run, and what needs more is deselected.
        """
        if text is None:
            return Ladder.of([profile.max_gpu_per_node])
        try:
            ladder = Ladder.parse(text)
        except ValueError as error:
            raise SelectionError(f"--ladder: {error}")
        if ladder.largest > profile.max_gpu_per_node:
            raise SelectionError(
                f"--ladder: rung {ladder.largest} exceeds {profile.name}, which has "
                f"{profile.max_gpu_per_node} GPUs per node"
            )
        return ladder

    @staticmethod
    def target_rung_for(rung: Optional[int], ladder: Ladder) -> Optional[int]:
        """The allocation this run occupies, or None when it names none.

        A rung outside the ladder is a usage error rather than a run that
        selects nothing.
        """
        if rung is None:
            return None
        if rung not in ladder:
            raise SelectionError(f"--rung: {rung} is not a rung of --ladder={ladder}")
        return rung

    @staticmethod
    def profile_for(machine: str) -> MachineProfile:
        """The named machine, or a usage error naming the fault."""
        try:
            catalog = default_catalog()
        except (ProfileConfigError, OSError) as error:
            raise SelectionError(f"--machine: cannot read the machine catalogue: {error}")
        try:
            return catalog.profile_for(machine)
        except KeyError as error:
            raise SelectionError(f"--machine: {error.args[0]}")
