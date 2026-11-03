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
"""A byte store shared by the processes of one node, for cached prompt encodings.

The serving frontends of one server receive a conversation's turns in any order, so
a cache of earlier prompts helps only when every frontend process can read it. Each
value is a file in one directory, preferably on a node-local tmpfs. Writers hold a
file lock, keep a running byte total, and evict the least recently used entries
before a write would exceed the cap.
"""

import fcntl
import hashlib
import os
import stat
import struct
import threading
from typing import Optional

from ..logger import logger

_MAGIC = b"TSPS"
# magic, key digest, value length, value digest
_HEADER = struct.Struct("<4s16sQ16s")
_SUFFIX = ".entry"
_EVICT_TO = 0.9  # of the cap, so one eviction scan serves several writes


def _digest(data: bytes) -> bytes:
    return hashlib.blake2b(data, digest_size=16).digest()


def _private(info: os.stat_result) -> bool:
    """Owned by this user and not writable by group or others."""
    return info.st_uid == os.getuid() and not info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)


class SharedPromptStore:
    """Values keyed by bytes, readable and writable by every process using ``path``.

    A read returns a value only when the entry was written whole for that same key:
    files are written to a temporary name and renamed into place, and each carries
    the digests of its key and value.
    """

    def __init__(self, path: str, max_bytes: int) -> None:
        self._dir = path
        self._max_bytes = max_bytes
        self._mutex = threading.Lock()
        os.makedirs(path, mode=0o700, exist_ok=True)
        # Entries are trusted token ids, so only this user may be able to write them. Every
        # later access goes through this one open directory, so swapping the path cannot
        # redirect it.
        dir_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        if not _private(os.fstat(dir_fd)):
            os.close(dir_fd)
            raise PermissionError(
                f"Prompt token cache directory {path} must be owned by this user and not "
                "writable by group or others"
            )
        self._dir_fd = dir_fd
        # Create the lock file now, so a directory that cannot be written fails at startup.
        os.close(self._open("lock", os.O_WRONLY | os.O_CREAT))

    def __del__(self) -> None:
        dir_fd = getattr(self, "_dir_fd", None)
        if dir_fd is not None:
            os.close(dir_fd)

    def _open(self, name: str, flags: int) -> int:
        return os.open(name, flags | os.O_NOFOLLOW, 0o600, dir_fd=self._dir_fd)

    @staticmethod
    def _name(key: bytes) -> str:
        return _digest(key).hex() + _SUFFIX

    def get(self, key: bytes) -> Optional[bytes]:
        """The value stored under ``key``, or None."""
        name = self._name(key)
        try:
            with os.fdopen(self._open(name, os.O_RDONLY), "rb") as entry:
                # An entry others own or can write is not trusted, e.g. one an older version wrote.
                if not _private(os.fstat(entry.fileno())):
                    return None
                data = entry.read()
        except OSError:
            return None
        if len(data) < _HEADER.size:
            return None
        magic, key_digest, length, value_digest = _HEADER.unpack_from(data)
        value = data[_HEADER.size :]
        if (
            magic != _MAGIC
            or key_digest != _digest(key)
            or length != len(value)
            or value_digest != _digest(value)
        ):
            return None
        try:
            os.utime(name, dir_fd=self._dir_fd)  # most recently used
        except OSError:
            pass
        return value

    def demote(self, key: bytes) -> None:
        """Make ``key`` the first entry to evict, e.g. once a longer entry covers it."""
        try:
            os.utime(self._name(key), (0, 0), dir_fd=self._dir_fd)
        except OSError:
            pass

    def put(self, key: bytes, value: bytes) -> None:
        """Store ``value`` under ``key``, evicting least recently used entries to fit the cap."""
        data = _HEADER.pack(_MAGIC, _digest(key), len(value), _digest(value)) + value
        if len(data) > self._max_bytes:
            return
        name = self._name(key)
        tmp = f"{name}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            with self._mutex, os.fdopen(self._open("lock", os.O_WRONLY | os.O_CREAT), "w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    replaced = os.stat(name, dir_fd=self._dir_fd, follow_symlinks=False).st_size
                except OSError:
                    replaced = 0
                total = self._load_total() - replaced
                if total + len(data) > self._max_bytes:
                    total = self._evict(len(data), name)
                flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
                with os.fdopen(self._open(tmp, flags), "wb") as entry:
                    entry.write(data)
                os.replace(tmp, name, src_dir_fd=self._dir_fd, dst_dir_fd=self._dir_fd)
                with os.fdopen(self._open("total", flags), "w") as total_file:
                    total_file.write(str(total + len(data)))
        except OSError as error:
            logger.debug(f"Shared prompt store: write to {self._dir} failed: {error}")
            try:
                os.unlink(tmp, dir_fd=self._dir_fd)
            except OSError:
                pass

    def _load_total(self) -> int:
        """Under the lock: bytes of all entries, recounted when the running total is unreadable."""
        try:
            with os.fdopen(self._open("total", os.O_RDONLY)) as total_file:
                return int(total_file.read())
        except (OSError, ValueError):
            return sum(size for _, size, _ in self._entries())

    def _entries(self) -> list[tuple[int, int, str]]:
        entries = []
        with os.scandir(self._dir_fd) as listing:
            for item in listing:
                if item.name.endswith(_SUFFIX):
                    try:
                        info = item.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    entries.append((info.st_mtime_ns, info.st_size, item.name))
        return entries

    def _evict(self, incoming: int, replaced: str) -> int:
        """Under the lock: drop the oldest entries until ``incoming`` more bytes fit."""
        entries = sorted(entry for entry in self._entries() if entry[2] != replaced)
        total = sum(size for _, size, _ in entries)
        target = max(0, int(_EVICT_TO * self._max_bytes) - incoming)
        for _, size, name in entries:
            if total <= target:
                break
            try:
                os.unlink(name, dir_fd=self._dir_fd)
            except OSError:
                continue
            total -= size
        return total
