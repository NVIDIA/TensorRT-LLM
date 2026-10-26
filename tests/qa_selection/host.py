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
"""No `--machine`: this host's profile, recorded as `local`.

    LocalProfile.of(out_dir)   -> MachineProfile named `local`

The profile is `out_dir/local.json`'s `profile` when that file exists, else
this host's, from `nvidia-smi`. Raises `SavedProfileError` for a `local.json`
without a valid profile, and `ProbeError` when this host cannot be probed.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from .core.machine import MachineProfile
from .core.report import ArtifactNames


class LocalProfile:
    """The profile a run selects for when no machine is named."""

    # The record name of a run for this host; not a catalogue name.
    NAME = "local"

    @classmethod
    def of(cls, out_dir: Optional[Path]) -> MachineProfile:
        """`out_dir`'s saved `local.json` profile when the file exists, else this host's, probed.

        An existing file is never a reason to probe, even when it cannot be read.
        """
        if out_dir is not None:
            saved = out_dir / ArtifactNames.record(cls.NAME)
            if saved.exists():
                return SavedProfile.read(saved)
        return HostProbe().profile()


class SavedProfile:
    """A profile saved for this host: the `profile` object of a `local.json`.

    A catalogue entry: `MachineProfile.FIELDS`, typed, and an optional `ladder`
    that fits the node. Every other key of the file is ignored, so a previous
    run's record reads back.
    """

    COUNTS = ("sm", "device_memory_mib", "max_gpu_per_node")
    NAMES = ("device_name", "cpu_arch")
    OPTIONAL = ("ladder",)

    @classmethod
    def read(cls, path: Path) -> MachineProfile:
        """The saved profile, named `local`; raises `SavedProfileError` naming the fault."""
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise SavedProfileError(path, f"invalid JSON at line {error.lineno}") from error
        except (OSError, UnicodeDecodeError) as error:
            raise SavedProfileError(path, f"cannot be read: {error}") from error

        profile = document.get("profile") if isinstance(document, dict) else None
        SavedProfileError.check(isinstance(profile, dict), path, 'no "profile" object')
        missing = [field for field in MachineProfile.FIELDS if field not in profile]
        SavedProfileError.check(not missing, path, f"profile is missing {', '.join(missing)}")
        unexpected = sorted(set(profile) - set(MachineProfile.FIELDS) - set(cls.OPTIONAL))
        SavedProfileError.check(
            not unexpected,
            path,
            f"profile has unexpected field{'s' if len(unexpected) > 1 else ''} "
            f"{', '.join(unexpected)}",
        )
        for field in cls.COUNTS:
            value = profile[field]
            SavedProfileError.check(
                isinstance(value, int) and not isinstance(value, bool) and value > 0,
                path,
                f"profile.{field} must be a positive integer, got {json.dumps(value)}",
            )
        for field in cls.NAMES:
            value = profile[field]
            SavedProfileError.check(
                isinstance(value, str) and value,
                path,
                f"profile.{field} must be a non-empty string, got {json.dumps(value)}",
            )
        SavedProfileError.check(
            isinstance(profile.get("ladder", []), list),
            path,
            f"profile.ladder must be a list of GPU counts, got {json.dumps(profile.get('ladder'))}",
        )
        try:
            return MachineProfile(name=LocalProfile.NAME, **profile).checked_ladder()
        except ValueError as error:
            raise SavedProfileError(path, f"profile {error}") from error


class HostProbe:
    """Runs `nvidia-smi` on this host and reads its output by `MachineInfoRules`."""

    PROGRAM = "nvidia-smi"
    LISTING = ("-L",)
    QUERY = (
        "--query-gpu=index,name,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    )
    TIMEOUT_S = 30

    def profile(self) -> MachineProfile:
        """This host's profile; raises `ProbeError` when `nvidia-smi` cannot answer."""
        return MachineInfoRules.profile(
            listing=self.run(self.LISTING),
            query=self.run(self.QUERY),
            cpu_arch=platform.machine(),
        )

    def run(self, arguments: Sequence[str]) -> str:
        """`nvidia-smi`'s stdout for `arguments`."""
        try:
            done = subprocess.run(
                [self.PROGRAM, *arguments],
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT_S,
            )
        except FileNotFoundError as error:
            raise ProbeError(f"{self.PROGRAM} not found") from error
        except subprocess.TimeoutExpired as error:
            raise ProbeError(f"{self.PROGRAM} timed out after {self.TIMEOUT_S} s") from error
        except OSError as error:
            raise ProbeError(f"{self.PROGRAM} could not run: {error.strerror}") from error
        if done.returncode != 0:
            output = (done.stdout + done.stderr).strip().splitlines()
            first = output[0] if output else "no output"
            raise ProbeError(f"{self.PROGRAM} exited {done.returncode}: {first}")
        return done.stdout


class MachineInfoRules:
    """Which `nvidia-smi` devices describe the machine, and what they say.

    A device whose `-L` name contains an `IGNORED` entry is dropped before
    anything is read. The first remaining device is the one described, and
    every remaining `-L` line counts as a GPU, as the integration conftest's
    `get_device_count()` counts them.
    """

    IGNORED = ("DGX Display",)
    VENDOR_PREFIX = "NVIDIA "

    @classmethod
    def profile(cls, listing: str, query: str, cpu_arch: str) -> MachineProfile:
        """The profile `nvidia-smi -L` and `--query-gpu` describe, on `cpu_arch`."""
        devices = cls.devices(listing)
        chosen = next((device for device in devices if device.index is not None), None)
        if chosen is None:
            raise ProbeError(
                f"{HostProbe.PROGRAM} -L lists no GPU outside {', '.join(cls.IGNORED)}: "
                f"{listing.strip()!r}"
            )
        card = QueryLine.find(query, chosen.index)
        return MachineProfile(
            name=LocalProfile.NAME,
            sm=card.sm,
            device_name=cls.device_name(card.name),
            device_memory_mib=card.memory_mib,
            max_gpu_per_node=len(devices),
            cpu_arch=cpu_arch,
        )

    @classmethod
    def device_name(cls, name: str) -> str:
        """`name` without its leading `NVIDIA `, as `profiles.json` spells it."""
        return name[len(cls.VENDOR_PREFIX) :] if name.startswith(cls.VENDOR_PREFIX) else name

    @classmethod
    def devices(cls, listing: str) -> List[Device]:
        """The `-L` lines that count, in order, ignored names dropped."""
        devices = [Device.parse(line) for line in listing.splitlines() if line.strip()]
        return [
            device
            for device in devices
            if not any(ignored in device.name for ignored in cls.IGNORED)
        ]


@dataclass(frozen=True)
class Device:
    """One `nvidia-smi -L` line: a GPU with its index, or a MIG device with none."""

    GPU_LINE = re.compile(r"^GPU (\d+): (.+?) \(UUID: [^)]*\)$")
    MIG_LINE = re.compile(r"^MIG \S")

    index: Optional[int]
    name: str

    @classmethod
    def parse(cls, line: str) -> Device:
        """One line; raises `ProbeError` naming it when it is neither shape."""
        text = line.strip()
        gpu = cls.GPU_LINE.match(text)
        if gpu:
            return cls(index=int(gpu.group(1)), name=gpu.group(2))
        if cls.MIG_LINE.match(text):
            return cls(index=None, name=text)
        raise ProbeError(f"unexpected {HostProbe.PROGRAM} -L line {text!r}")


@dataclass(frozen=True)
class QueryLine:
    """One `--query-gpu=index,name,memory.total,compute_cap` line."""

    COMPUTE_CAP = re.compile(r"^(\d+)\.(\d+)$")

    index: int
    name: str
    memory_mib: int
    sm: int

    @classmethod
    def find(cls, query: str, index: int) -> QueryLine:
        """The line for GPU `index`; raises `ProbeError` naming a line it cannot read."""
        for line in query.splitlines():
            if line.strip() and line.split(",", 1)[0].strip() == str(index):
                return cls.parse(line)
        raise ProbeError(f"{HostProbe.PROGRAM} --query-gpu has no GPU {index}: {query.strip()!r}")

    @classmethod
    def parse(cls, line: str) -> QueryLine:
        """`0, NVIDIA B200, 183359, 10.0`; the name may itself hold commas."""
        fields = [field.strip() for field in line.split(",")]
        compute_cap = cls.COMPUTE_CAP.match(fields[-1])
        if len(fields) < 4 or not fields[-2].isdigit() or compute_cap is None:
            raise ProbeError(f"unexpected {HostProbe.PROGRAM} --query-gpu line {line.strip()!r}")
        major, minor = (int(part) for part in compute_cap.groups())
        return cls(
            index=int(fields[0]),
            name=", ".join(fields[1:-2]),
            memory_mib=int(fields[-2]),
            sm=major * 10 + minor,
        )


class ProbeError(RuntimeError):
    """This host could not be probed; the message says why and names `--machine`."""

    def __init__(self, why: str) -> None:
        super().__init__(
            f"cannot probe this host for its machine profile: {why}; "
            f"name a catalogue machine with --machine=NAME"
        )


class SavedProfileError(ValueError):
    """A `local.json` holds no valid profile; the message names the file, the fault and the fixes."""

    def __init__(self, path: Path, fault: str) -> None:
        super().__init__(
            f"{path}: {fault}; fix or delete it to probe this host, "
            f"or name a catalogue machine with --machine=NAME"
        )

    @classmethod
    def check(cls, condition: object, path: Path, fault: str) -> None:
        """Raise for `path` with `fault` unless `condition` holds."""
        if not condition:
            raise cls(path, fault)
