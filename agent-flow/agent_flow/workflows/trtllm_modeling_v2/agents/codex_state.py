# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where a run's Codex sessions keep their state.

Codex keeps its login, configuration and session rollouts under ``CODEX_HOME``
and its databases in SQLite. SQLite is not safe on a file system several hosts
write at once: a home on a network file system that concurrent runs share is
damaged within hours (``failed to initialize sqlite state runtime``), and from
then on every turn of every run sharing it fails. So a run gets its own home in
its workspace, which survives the orchestrator moving to another host, seeded
with the login and configuration of the launching environment's; and its
databases live on the host's local disk, where only its own processes reach
them. A process on a new host starts with empty databases and Codex rebuilds
them from the rollouts.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import shutil
from pathlib import Path
from typing import Mapping

#: The run's Codex home, under its workspace.
CODEX_HOME_DIR = "codex-home"
#: What a run's home takes from the launching environment's: the login and the configuration.
SEEDED = ("auth.json", "config.toml")
#: File systems whose locks and shared memory SQLite cannot rely on across hosts.
NETWORK_FILESYSTEMS = frozenset(
    {
        "nfs",
        "nfs4",
        "lustre",
        "cifs",
        "smb3",
        "smbfs",
        "gpfs",
        "beegfs",
        "ceph",
        "glusterfs",
        "fuse.glusterfs",
        "9p",
        "afs",
        "panfs",
        "wekafs",
        "fuse.sshfs",
    }
)


@dataclasses.dataclass(frozen=True)
class CodexState:
    home: Path
    sqlite_home: Path
    #: The file system of each, for the journal; ``""`` when unknown.
    home_fs: str
    sqlite_fs: str

    @property
    def local(self) -> bool:
        """Whether the databases are on a file system no other host writes."""
        return self.sqlite_fs not in NETWORK_FILESYSTEMS

    def describe(self) -> str:
        where = "this host's local disk" if self.local else "a shared file system"
        return (
            f"home `{self.home}` ({self.home_fs or 'unknown'}); databases `{self.sqlite_home}` "
            f"({self.sqlite_fs or 'unknown'}, {where})"
        )


def filesystem_type(path: Path, mountinfo: str = "/proc/self/mountinfo") -> str:
    """The type of the file system ``path`` is on, from the longest matching mount; ``""`` if unknown."""
    try:
        lines = Path(mountinfo).read_text().splitlines()
    except OSError:
        return ""
    target = str(path.resolve())
    best, kind = "", ""
    for line in lines:
        fields = line.split()
        if " - " not in line or len(fields) < 5:
            continue
        point = fields[4].replace("\\040", " ")
        fstype = line.split(" - ", 1)[1].split()[0]
        inside = target == point or target.startswith(point.rstrip("/") + "/")
        if inside and len(point) >= len(best):
            best, kind = point, fstype
    return kind


def _seed(home: Path, source: Path) -> None:
    """Copy the login and configuration in, when missing or older than the source's."""
    if source.resolve() == home.resolve():
        return
    for name in SEEDED:
        src, dst = source / name, home / name
        if not src.is_file():
            continue
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            continue
        tmp = dst.with_name(f".{name}.{os.getpid()}.tmp")
        shutil.copy2(src, tmp)
        tmp.chmod(0o600)
        os.replace(tmp, dst)


def prepare(
    workspace: Path,
    *,
    environ: Mapping[str, str] | None = None,
    mountinfo: str = "/proc/self/mountinfo",
) -> CodexState:
    """The run's Codex home and database directory, created and seeded."""
    env = os.environ if environ is None else environ
    home = (workspace / CODEX_HOME_DIR).absolute()
    home.mkdir(mode=0o700, parents=True, exist_ok=True)
    _seed(home, Path(env.get("CODEX_HOME") or Path.home() / ".codex"))
    key = hashlib.sha1(str(workspace.resolve()).encode()).hexdigest()[:12]
    candidates = [env.get("TMPDIR"), "/tmp", "/var/tmp", "/dev/shm"]
    sqlite_home, sqlite_fs = home, filesystem_type(home, mountinfo)
    for root in dict.fromkeys(c for c in candidates if c):
        fstype = filesystem_type(Path(root), mountinfo)
        if not fstype or fstype in NETWORK_FILESYSTEMS:
            continue
        candidate = Path(root) / f"agent-flow-codex-{os.getuid()}" / key
        try:
            candidate.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError:
            continue
        sqlite_home, sqlite_fs = candidate, fstype
        break
    return CodexState(home, sqlite_home, filesystem_type(home, mountinfo), sqlite_fs)


__all__ = [
    "CODEX_HOME_DIR",
    "NETWORK_FILESYSTEMS",
    "SEEDED",
    "CodexState",
    "filesystem_type",
    "prepare",
]
