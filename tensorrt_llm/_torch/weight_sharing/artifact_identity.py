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
"""Immutable checkpoint identity for shared-weight compatibility checks."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ARTIFACT_IDENTITY_FORMAT_VERSION = 1

_HF_SNAPSHOT_SCHEME = "hf_snapshot_revision"
_CHECKPOINT_MANIFEST_SCHEME = "checkpoint_manifest_sha256"
_SUPPORTED_SCHEMES = frozenset({_HF_SNAPSHOT_SCHEME, _CHECKPOINT_MANIFEST_SCHEME})
_IGNORED_DIRECTORY_NAMES = frozenset({".cache", ".git", "__pycache__"})
_IGNORED_FILE_NAMES = frozenset({".DS_Store"})
_HASH_CHUNK_SIZE = 1024 * 1024


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_hex(value: str, lengths: tuple[int, ...]) -> bool:
    return len(value) in lengths and all(char in "0123456789abcdef" for char in value)


def _hf_snapshot_descriptor(path: Path) -> tuple[str, str] | None:
    """Return an immutable HF revision and repository-relative subpath."""
    parts = path.resolve().parts
    for index, part in enumerate(parts[:-1]):
        if part != "snapshots" or index == 0:
            continue
        if not parts[index - 1].startswith("models--"):
            continue

        revision = parts[index + 1].lower()
        if not _is_hex(revision, (40, 64)):
            continue
        subpath = "/".join(parts[index + 2 :])
        return revision, subpath
    return None


def _raise_walk_error(error: OSError) -> None:
    raise error


def _checkpoint_files(path: Path) -> tuple[Path, list[Path]]:
    if path.is_file():
        return path.parent, [path]

    files = []
    for directory, directory_names, file_names in os.walk(path, onerror=_raise_walk_error):
        retained_directories = []
        for directory_name in sorted(directory_names):
            if directory_name in _IGNORED_DIRECTORY_NAMES:
                continue
            nested_directory = Path(directory) / directory_name
            if nested_directory.is_symlink():
                raise ValueError(
                    "Checkpoint manifests do not support nested symlinked directories: "
                    f"{nested_directory}"
                )
            retained_directories.append(directory_name)
        directory_names[:] = retained_directories
        for file_name in sorted(file_names):
            if file_name in _IGNORED_FILE_NAMES:
                continue
            candidate = Path(directory) / file_name
            if candidate.is_file():
                files.append(candidate)
    files.sort(key=lambda candidate: candidate.relative_to(path).as_posix())
    if not files:
        raise ValueError(f"Checkpoint path contains no files: {path}")
    return path, files


def _sha256_file(path: Path) -> tuple[int, str]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Checkpoint file changed while being fingerprinted: {path}")
    return after.st_size, digest.hexdigest()


def _checkpoint_manifest_digest(path: Path) -> str:
    root, files = _checkpoint_files(path)
    manifest = []
    for checkpoint_file in files:
        size, digest = _sha256_file(checkpoint_file)
        manifest.append(
            {
                "path": checkpoint_file.relative_to(root).as_posix(),
                "size": size,
                "sha256": digest,
            }
        )
    return _canonical_hash(
        {
            "format_version": ARTIFACT_IDENTITY_FORMAT_VERSION,
            "files": manifest,
        }
    )


@dataclass(frozen=True)
class ArtifactIdentity:
    """Versioned identity of the immutable checkpoint artifact being loaded.

    `SourceIdentity` embeds this value as a global compatibility component.
    Hugging Face cache snapshots use their immutable commit revision; local
    checkpoints use a canonical manifest of relative paths, sizes, and file
    content digests. Absolute paths are intentionally excluded.
    """

    format_version: int
    scheme: str
    digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.format_version, int) or isinstance(self.format_version, bool):
            raise ValueError("ArtifactIdentity format version must be an integer")
        if self.format_version != ARTIFACT_IDENTITY_FORMAT_VERSION:
            raise ValueError(f"Unsupported ArtifactIdentity format version: {self.format_version}")
        if not isinstance(self.scheme, str):
            raise ValueError("ArtifactIdentity scheme must be a string")
        if self.scheme not in _SUPPORTED_SCHEMES:
            raise ValueError(f"Unsupported ArtifactIdentity scheme: {self.scheme}")
        if not isinstance(self.digest, str):
            raise ValueError("ArtifactIdentity digest must be a string")

        normalized_digest = self.digest.lower()
        if not _is_hex(normalized_digest, (64,)):
            raise ValueError("ArtifactIdentity digest must be a 64-character hex value")
        object.__setattr__(self, "digest", normalized_digest)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | os.PathLike[str]) -> "ArtifactIdentity":
        """Build an identity from an immutable snapshot or local checkpoint.

        Args:
            checkpoint_path: A model checkpoint file or directory.

        Returns:
            The path-independent checkpoint identity.

        Raises:
            FileNotFoundError: If `checkpoint_path` does not exist.
            ValueError: If a local checkpoint directory contains no files.
            RuntimeError: If a local checkpoint changes while it is hashed.

        Note:
            Local checkpoints have no authoritative immutable revision, so
            their regular files are read in full to derive a content-bound
            manifest. Hugging Face cache snapshots use the resolved immutable
            revision without rereading model shards.
        """
        path = Path(checkpoint_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

        snapshot_descriptor = _hf_snapshot_descriptor(path)
        if snapshot_descriptor is not None:
            revision, subpath = snapshot_descriptor
            digest = _canonical_hash(
                {
                    "scheme": _HF_SNAPSHOT_SCHEME,
                    "revision": revision,
                    "subpath": subpath,
                }
            )
            return cls(
                format_version=ARTIFACT_IDENTITY_FORMAT_VERSION,
                scheme=_HF_SNAPSHOT_SCHEME,
                digest=digest,
            )

        return cls(
            format_version=ARTIFACT_IDENTITY_FORMAT_VERSION,
            scheme=_CHECKPOINT_MANIFEST_SCHEME,
            digest=_checkpoint_manifest_digest(path),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "format_version": self.format_version,
            "scheme": self.scheme,
            "digest": self.digest,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactIdentity":
        """Reconstruct and validate a serialized artifact identity."""
        return cls(
            format_version=data["format_version"],
            scheme=data["scheme"],
            digest=data["digest"],
        )
