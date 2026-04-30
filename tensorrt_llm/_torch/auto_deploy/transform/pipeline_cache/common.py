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

"""Shared helpers for AutoDeploy pipeline cache identity and file IO."""

import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any


def _canonicalize_for_hash(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _canonicalize_for_hash(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize_for_hash(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _canonicalize_for_hash(value.value)
    return value


def hash_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_canonicalize_for_hash(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
    fsync_dir(path.parent)


def fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_publish_rank_dir(tmp_rank_dir: Path, rank_dir: Path) -> None:
    old_dir: Path | None = None
    if rank_dir.exists():
        old_dir = rank_dir.with_name(f"{rank_dir.name}.old.{os.getpid()}")
        rank_dir.rename(old_dir)
    try:
        tmp_rank_dir.rename(rank_dir)
        fsync_dir(rank_dir.parent)
    except OSError:
        if old_dir is not None and old_dir.exists() and not rank_dir.exists():
            old_dir.rename(rank_dir)
        raise
    if old_dir is not None:
        shutil.rmtree(old_dir, ignore_errors=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
