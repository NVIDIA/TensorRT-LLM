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
"""Client page-cache verification and NFS disk-cache guards for startup QA."""

import ctypes
import hashlib
import json
import mmap
import os
import platform
import re
import stat
import subprocess
import time
from pathlib import Path

__all__ = [
    "CacheVerificationError",
    "checkpoint_manifest",
    "discover_checkpoint_files",
    "evict_and_verify",
]

_WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth"}
_SHARD_NAME = re.compile(r"(.+)-(\d+)-of-(\d+)(\.[^.]+)$")


class CacheVerificationError(RuntimeError):
    """A rejected cold-cache claim, with serializable diagnostic evidence."""

    def __init__(self, message: str, evidence: dict[str, object]) -> None:
        super().__init__(message)
        self.evidence = evidence


def _identity(path: Path, info: os.stat_result) -> dict[str, str | int]:
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
        raise ValueError(f"Checkpoint must be a nonempty regular file: {path}")
    return {
        "path": str(path),
        "device": info.st_dev,
        "inode": info.st_ino,
        "bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def checkpoint_manifest(files: list[Path]) -> list[dict[str, str | int]]:
    """Return stable stat identities, deduplicated by inode, without reading weights.

    Args:
        files: All checkpoint data files from ``discover_checkpoint_files``.
    """
    identities = {}
    for path in sorted({Path(item).resolve(strict=True) for item in files}):
        entry = _identity(path, path.stat())
        identities.setdefault((entry["device"], entry["inode"]), entry)
    if not identities:
        raise ValueError("No checkpoint data files were discovered")
    return list(identities.values())


def discover_checkpoint_files(model_dirs: list[Path]) -> list[Path]:
    """Validate shard coverage and include auxiliary weights in explicit directories.

    Args:
        model_dirs: Main, draft and auxiliary checkpoint directories. Unindexed
            nested directories are not searched; supply them explicitly.

    Returns:
        Unique data files. Index metadata is read, but weight payloads are not.
    """
    files = set()
    for directory in model_dirs:
        root = Path(directory).resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"Checkpoint directory is not a directory: {root}")
        weights = {path for path in root.iterdir() if path.suffix in _WEIGHT_SUFFIXES}
        for index in sorted(root.glob("*.index.json")):
            if not any(index.name.endswith(f"{suffix}.index.json") for suffix in _WEIGHT_SUFFIXES):
                continue
            document = json.loads(index.read_text(encoding="utf-8"))
            weight_map = document.get("weight_map") if isinstance(document, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"Missing or empty weight_map in {index}")
            for name in weight_map.values():
                if (
                    not isinstance(name, str)
                    or not name
                    or Path(name).is_absolute()
                    or ".." in Path(name).parts
                ):
                    raise ValueError(f"Invalid shard path in {index}: {name!r}")
                shard = root / name
                if not shard.is_file():
                    raise ValueError(f"Missing indexed shard: {shard}")
                weights.add(shard)
        if not weights:
            raise ValueError(f"No checkpoint data files in {root}")
        checked_groups = set()
        for path in weights:
            match = _SHARD_NAME.fullmatch(path.name)
            if match:
                prefix, number, total, suffix = match.groups()
                count = int(total)
                if not 1 <= int(number) <= count:
                    raise ValueError(f"Invalid shard numbering: {path}")
                group = (path.parent, prefix, total, suffix)
                if group in checked_groups:
                    continue
                checked_groups.add(group)
                if count > len(weights):
                    raise ValueError(f"Missing numbered shards: {path} requires {count} files")
                for rank in range(1, count + 1):
                    expected = path.with_name(f"{prefix}-{rank:0{len(number)}}-of-{total}{suffix}")
                    if expected not in weights:
                        raise ValueError(f"Missing numbered shard: {expected}")
        files.update(weights)
    return [Path(entry["path"]) for entry in checkpoint_manifest(list(files))]


def _resident_pages(fd: int, size: int, page_size: int) -> int:
    # PROT_NONE creates no readable mapping and never faults in checkpoint data.
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    libc.mmap.restype = ctypes.c_void_p
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
    libc.mincore.restype = ctypes.c_int
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.munmap.restype = ctypes.c_int
    resident = 0
    chunk_size = (1 << 30) // page_size * page_size
    for offset in range(0, size, chunk_size):
        length = min(chunk_size, size - offset)
        address = libc.mmap(None, length, 0, mmap.MAP_SHARED, fd, offset)
        if address == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(), "Checkpoint residency mmap failed")
        try:
            vector = (ctypes.c_ubyte * ((length + page_size - 1) // page_size))()
            if libc.mincore(address, length, vector) != 0:
                raise OSError(ctypes.get_errno(), "Checkpoint mincore failed")
            resident += sum(value & 1 for value in vector)
        finally:
            if libc.munmap(address, length) != 0:
                raise OSError(ctypes.get_errno(), "Checkpoint residency munmap failed")
    return resident


def _checkpoint_mount(fd: int) -> dict[str, object]:
    # Match the open file's mount ID, not a path prefix: bind mounts and stacked
    # mounts can otherwise select the wrong filesystem or mount options.
    fdinfo = Path(f"/proc/self/fdinfo/{fd}").read_text(encoding="utf-8")
    mount_ids = re.findall(r"^mnt_id:\s*([0-9]+)$", fdinfo, re.MULTILINE)
    if len(mount_ids) != 1:
        raise ValueError("Checkpoint mount ID is unavailable or ambiguous")
    mount_id = mount_ids[0]
    lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    matches = [line for line in lines if line.split(maxsplit=1)[0:1] == [mount_id]]
    if len(matches) != 1:
        raise ValueError(f"Checkpoint mount {mount_id} is unavailable or ambiguous")
    before, separator, after = matches[0].partition(" - ")
    fields, filesystem = before.split(), after.split()
    if not separator or len(fields) < 6 or len(filesystem) != 3:
        raise ValueError(f"Malformed checkpoint mount information: {matches[0]}")
    options = filesystem[2].split(",")
    if not all(options) or not {"ro", "rw"}.intersection(options):
        raise ValueError(f"Checkpoint mount {mount_id} has unverifiable superblock options")
    is_nfs = filesystem[0] in {"nfs", "nfs4"}
    # Linux nfs_show_options reports fsc (or fsc=<tag>) whenever enabled;
    # its absence in the effective superblock options means the default nofsc.
    fscache = "not_applicable"
    if is_nfs:
        fscache = (
            "enabled" if any(opt.partition("=")[0] == "fsc" for opt in options) else "disabled"
        )
    return {
        "mount_id": int(mount_id),
        "filesystem": filesystem[0],
        "source": re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), filesystem[1]),
        "mount_point": re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), fields[4]),
        "root": re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), fields[3]),
        "super_options": options,
        "nfs_fscache": fscache,
    }


def evict_and_verify(
    files: list[Path], *, exclusive: bool, reset_command: list[str] | None = None
) -> dict[str, object]:
    """Reject NFS disk caching, then evict and verify RAM pages before launch.

    Args:
        files: Complete checkpoint data-file list (including draft/auxiliary weights).
        exclusive: Explicit acknowledgement of exclusive node and checkpoint use.
        reset_command: Optional operator-approved argv helper, executed without a
            shell or implicit privilege escalation instead of per-file fadvise.

    Returns:
        RAM residency and per-file mount evidence. NFS FS-Cache must be disabled;
        other disk caches and storage-server/backend coldness are not proven.

    Raises:
        CacheVerificationError: Reset, identity or residency validation failed.
            Its ``evidence`` includes partial observations and the failure reason.
    """
    evidence: dict[str, object] = {
        "verified": False,
        "scope": "client_checkpoint_page_cache",
        "disk_cache_scope": "nfs_fscache_only",
        "mount_verification_method": "proc_fdinfo_mountinfo",
        "other_disk_caches_verified": False,
        "backend_cache_verified": False,
        "host": platform.node(),
        "exclusive_acknowledged": exclusive,
        "reset_method": "approved_command"
        if reset_command is not None
        else "posix_fadvise_dontneed",
        "verification_method": "mincore_all_pages",
        "started_unix_ns": time.time_ns(),
        "files": [],
        "errors": [],
    }
    observations = []
    evidence["files"] = observations
    try:
        if exclusive is not True:
            raise ValueError(
                "Cache reset requires explicit exclusive-node/checkpoint acknowledgement"
            )
        if platform.system() != "Linux" or ctypes.sizeof(ctypes.c_void_p) != 8:
            raise ValueError("Verified cache reset requires 64-bit Linux")
        manifest = checkpoint_manifest(files)
        evidence["manifest"] = manifest
        evidence["manifest_sha256"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest()
        page_size = os.sysconf("SC_PAGE_SIZE")
        evidence["page_size"] = page_size
        evidence["checkpoint_bytes"] = sum(entry["bytes"] for entry in manifest)
        evidence["total_pages"] = sum(
            (entry["bytes"] + page_size - 1) // page_size for entry in manifest
        )
        for entry in manifest:
            path = Path(entry["path"])
            with path.open("rb") as handle:
                if _identity(path, os.fstat(handle.fileno())) != entry:
                    raise ValueError(f"Checkpoint changed before mount verification: {path}")
                mount = _checkpoint_mount(handle.fileno())
            observations.append({**entry, "mount": mount})
            if mount["nfs_fscache"] == "enabled":
                raise ValueError(f"NFS FS-Cache is enabled for checkpoint: {path}")
        if reset_command is not None:
            if (
                not isinstance(reset_command, list)
                or not reset_command
                or not all(isinstance(arg, str) and arg for arg in reset_command)
            ):
                raise ValueError("reset_command must be a nonempty argv list")
            reset = subprocess.run(
                reset_command, shell=False, check=False, capture_output=True, text=True, timeout=300
            )
            evidence["reset_command"] = reset_command
            evidence["reset_returncode"] = reset.returncode
            evidence["reset_stdout"] = reset.stdout[-16384:]
            evidence["reset_stderr"] = reset.stderr[-16384:]
            if reset.returncode:
                raise ValueError(f"Approved cache reset failed with exit code {reset.returncode}")
        else:
            for entry in manifest:
                path = Path(entry["path"])
                with path.open("rb") as handle:
                    if _identity(path, os.fstat(handle.fileno())) != entry:
                        raise ValueError(f"Checkpoint changed before reset: {path}")
                    os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        for entry, observation in zip(manifest, observations):
            path = Path(entry["path"])
            with path.open("rb") as handle:
                if _identity(path, os.fstat(handle.fileno())) != entry:
                    raise ValueError(f"Checkpoint changed during reset: {path}")
                if _checkpoint_mount(handle.fileno()) != observation["mount"]:
                    raise ValueError(f"Checkpoint mount changed during reset: {path}")
                resident = _resident_pages(handle.fileno(), entry["bytes"], page_size)
                observation["resident_pages"] = resident
        evidence["resident_pages"] = sum(item["resident_pages"] for item in observations)
        if checkpoint_manifest(files) != manifest:
            raise ValueError("Checkpoint identity changed during verification")
        # Linux masks mincore results to all-resident for callers lacking ownership
        # or write permission. Such results must never qualify as client-cache cold.
        if evidence["resident_pages"] != 0:
            raise ValueError(
                "Checkpoint pages are resident or mincore residency is permission-masked"
            )
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError) as error:
        evidence["errors"] = [f"{type(error).__name__}: {error}"]
        evidence["finished_unix_ns"] = time.time_ns()
        raise CacheVerificationError(str(error), evidence) from error
    evidence["verified"] = True
    evidence["finished_unix_ns"] = time.time_ns()
    return evidence
