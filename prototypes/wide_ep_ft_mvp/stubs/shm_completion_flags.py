# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""POSIX shared-memory substitute for MNNVL fabric-memory completion flags.

The production AlltoAll watchdog reads a host-visible ``completion_flags``
table that lives in MNNVL fabric memory — *no collective is required* for the
host to read peers' flag values. That property is essential: if reading a
peer's flag required participating in a collective, the dead peer's
non-participation would block the read and the watchdog could never fire.

This single-node prototype substitutes a per-rank ``int64`` mapped from
``/dev/shm/wide_ep_ft_proto/rank_<i>.counter``. Each rank ``mmap``s its own
file read/write and advances its counter; the watchdog ``mmap``s every peer's
file read-only and reads the values directly. A SIGKILL'd rank's file simply
stops advancing; the watchdog sees no progress for ``timeout_sec`` and fires.

This is *not* what production looks like. The point of the stub is that the
read path on the survivors has the same shape (no peer participation
required), so the watchdog → mark_failed seam is exercised correctly.
"""

from __future__ import annotations

import mmap
import os
import struct
from pathlib import Path

_DEFAULT_DIR = "/dev/shm/wide_ep_ft_proto"
_COUNTER_FMT = "<q"  # little-endian int64
_COUNTER_SIZE = struct.calcsize(_COUNTER_FMT)


class ShmCompletionFlagTable:
    """Per-rank monotonic counter, shared via POSIX shm.

    All ranks must agree on ``shm_dir`` and ``run_id`` so they find each
    other's counter files. The driver creates a fresh ``run_id`` per
    invocation so repeated runs do not see each other's stale files.

    Args:
        ep_size: Number of ranks in the EP group.
        local_rank: This rank's index in ``[0, ep_size)``.
        run_id: Unique per-run identifier (e.g. ``str(os.getpid())`` of the
            driver). Used to namespace the per-rank counter files so concurrent
            runs do not collide.
        shm_dir: Base directory for the shared-memory files. Defaults to
            ``/dev/shm/wide_ep_ft_proto``.
    """

    def __init__(
        self,
        ep_size: int,
        local_rank: int,
        run_id: str,
        shm_dir: str = _DEFAULT_DIR,
    ) -> None:
        if ep_size <= 0:
            raise ValueError(f"ep_size must be > 0, got {ep_size}")
        if not 0 <= local_rank < ep_size:
            raise ValueError(f"local_rank {local_rank} not in [0, {ep_size})")
        self._ep_size = ep_size
        self._local_rank = local_rank
        base = Path(shm_dir) / run_id
        base.mkdir(parents=True, exist_ok=True)
        self._base = base

        # Create + mmap this rank's counter file (rw).
        self._own_path = base / f"rank_{local_rank}.counter"
        self._own_fd = os.open(self._own_path, os.O_CREAT | os.O_RDWR, 0o600)
        os.ftruncate(self._own_fd, _COUNTER_SIZE)
        self._own_mm = mmap.mmap(
            self._own_fd, _COUNTER_SIZE, mmap.MAP_SHARED, mmap.PROT_WRITE | mmap.PROT_READ
        )
        self._set_local_value(0)

        # mmap every peer's counter file read-only. Peers may not yet have
        # created their files when this rank opens; create the file from this
        # side too, ftruncate(0) → size, then mmap. POSIX semantics guarantee
        # the bytes are zero-initialized.
        self._peer_mms: list[mmap.mmap | None] = [None] * ep_size
        for r in range(ep_size):
            if r == local_rank:
                continue
            peer_path = base / f"rank_{r}.counter"
            fd = os.open(peer_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                os.ftruncate(fd, _COUNTER_SIZE)
                self._peer_mms[r] = mmap.mmap(fd, _COUNTER_SIZE, mmap.MAP_SHARED, mmap.PROT_READ)
            finally:
                os.close(fd)

    def _set_local_value(self, v: int) -> None:
        self._own_mm.seek(0)
        self._own_mm.write(struct.pack(_COUNTER_FMT, v))
        self._own_mm.flush()

    def _read_value(self, mm: mmap.mmap) -> int:
        mm.seek(0)
        return struct.unpack(_COUNTER_FMT, mm.read(_COUNTER_SIZE))[0]

    @property
    def local_value(self) -> int:
        return self._read_value(self._own_mm)

    def advance(self) -> None:
        """Increment this rank's counter by 1."""
        self._set_local_value(self.local_value + 1)

    def snapshot(self) -> list[int]:
        """Return [counter for each rank]. No collective, no peer participation."""
        out = [0] * self._ep_size
        out[self._local_rank] = self.local_value
        for r in range(self._ep_size):
            if r == self._local_rank:
                continue
            mm = self._peer_mms[r]
            if mm is not None:
                out[r] = self._read_value(mm)
        return out

    def close(self) -> None:
        try:
            self._own_mm.close()
        except (BufferError, ValueError):
            pass
        try:
            os.close(self._own_fd)
        except OSError:
            pass
        for mm in self._peer_mms:
            if mm is not None:
                try:
                    mm.close()
                except (BufferError, ValueError):
                    pass

    @staticmethod
    def cleanup_run(run_id: str, shm_dir: str = _DEFAULT_DIR) -> None:
        """Driver-side helper to remove a run's shared files after the test."""
        base = Path(shm_dir) / run_id
        if not base.exists():
            return
        for p in base.iterdir():
            try:
                p.unlink()
            except OSError:
                pass
        try:
            base.rmdir()
        except OSError:
            pass
