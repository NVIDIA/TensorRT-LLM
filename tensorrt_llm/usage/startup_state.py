# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local count of consecutive failed starts of one launch, for usage telemetry.

A launch is the same command of the same TensorRT-LLM version, with the same
launcher rank, visible GPUs and TensorRT-LLM process role. Each launch keeps one
small JSON file with its number of consecutive startup failures and the time of
the last one. The file name is a local hash of those values; none of them is
sent, only a bucket of the count.

The usage telemetry reporter reads and writes the file at most once per process,
and only when telemetry is enabled. Functions here never raise: file system
errors become the ``unavailable`` status. Standard library only.
"""

import hashlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Optional

STATE_DIR_ENV = "TRTLLM_USAGE_STATE_DIR"
# Reported counts are the largest of these lower bounds not above the count.
PRIOR_STARTUP_FAILURE_BUCKETS = (0, 1, 2, 3, 10, 30, 100, 300, 1000)
StartupStateAction = Literal["increment", "reset", "read"]

_STATE_VERSION = 1
_KEY_DOMAIN = b"trtllm-startup-state-v1\0"
_KEY_HEX_CHARS = 16
_FILE_PREFIX = "startup-"
_FILE_SUFFIX = ".json"
_TMP_PREFIX = ".startup-"
_TMP_SUFFIX = ".tmp"
_MAX_STATE_BYTES = 4096
_MAX_STREAK = 1_000_000
_STREAK_EXPIRY_SECONDS = 24 * 3600
_PRUNE_AGE_SECONDS = 30 * 24 * 3600
# Variables that tell apart launches of the same command: the launcher rank,
# the visible GPUs and TensorRT-LLM's process roles.
_KEY_ENV_VARS = (
    "SLURM_PROCID",
    "OMPI_COMM_WORLD_RANK",
    "PMI_RANK",
    "PMIX_RANK",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "TLLM_DISAGG_INSTANCE_IDX",
    "TRTLLM_DISAGG_WORKER_PROCESS_ID",
    "TLLM_EXECUTOR_FRONTEND_ID",
)


def launch_key(
    version: str,
    argv: Optional[Sequence[str]] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> str:
    """Return the hex key naming this launch's state file, or "" if unavailable.

    Hashes the TensorRT-LLM ``version``, the command line and ``_KEY_ENV_VARS``;
    touches no file.
    """
    try:
        environ = os.environ if environ is None else environ
        identity = {
            "version": version,  # an upgrade starts a new count
            "argv": list(sys.argv if argv is None else argv),
            "env": {name: environ.get(name, "") for name in _KEY_ENV_VARS},
        }
        encoded = json.dumps(identity, sort_keys=True).encode("utf-8")
        return hashlib.sha256(_KEY_DOMAIN + encoded).hexdigest()[:_KEY_HEX_CHARS]
    except Exception:
        return ""


def bucket(streak: int) -> int:
    """Return the reported bucket lower bound for a consecutive failure count."""
    return max(b for b in PRIOR_STARTUP_FAILURE_BUCKETS if b <= max(streak, 0))


def state_dir(environ: Optional[Mapping[str, str]] = None) -> Optional[Path]:
    """Return the state directory, or None if no home directory is available.

    ``$TRTLLM_USAGE_STATE_DIR``, else ``$XDG_CACHE_HOME/tensorrt_llm/usage``, else
    ``~/.cache/tensorrt_llm/usage``.
    """
    environ = os.environ if environ is None else environ
    override = environ.get(STATE_DIR_ENV)
    if override:
        return Path(override)
    cache_home = environ.get("XDG_CACHE_HOME")
    if cache_home and os.path.isabs(cache_home):
        return Path(cache_home) / "tensorrt_llm" / "usage"
    try:
        return Path.home() / ".cache" / "tensorrt_llm" / "usage"
    except (RuntimeError, KeyError):
        return None


def consult(
    key: str,
    action: StartupStateAction,
    *,
    now: Optional[float] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> tuple[int, str]:
    """Read this launch's count, apply ``action``, and return ``(prior, status)``.

    ``prior`` is the count of consecutive failed starts before this process; a
    count whose last failure is more than 24 hours old is 0. ``status`` is
    ``persisted`` if the file was read, ``new`` if it was missing or corrupt, and
    ``unavailable`` if it could not be read or written. ``increment`` adds this
    process's failed start, ``reset`` records a successful start, and ``read``
    leaves the file unchanged.
    """
    try:
        directory = state_dir(environ) if key else None
        if directory is None:
            return 0, "unavailable"
        now = time.time() if now is None else now
        path = directory / f"{_FILE_PREFIX}{key}{_FILE_SUFFIX}"
        prior, status = _read(path, now)
        if action == "read" or status == "unavailable":
            return prior, status
        if action == "increment":
            state = {
                "v": _STATE_VERSION,
                "streak": min(prior + 1, _MAX_STREAK),
                "last_failure": int(now),
            }
        else:
            state = {"v": _STATE_VERSION, "streak": 0, "last_failure": 0}
        if not _write(directory, path, state):
            return prior, "unavailable"
        if status == "new":
            _prune(directory, now, keep=path.name)
        return prior, status
    except Exception:
        return 0, "unavailable"


def _read(path: Path, now: float) -> tuple[int, str]:
    try:
        with open(path, "rb") as stream:
            raw = stream.read(_MAX_STATE_BYTES + 1)
    except FileNotFoundError:
        return 0, "new"
    except OSError:
        return 0, "unavailable"
    try:
        state = json.loads(raw) if len(raw) <= _MAX_STATE_BYTES else None
        streak = state["streak"]
        last_failure = state["last_failure"]
        valid = (
            state["v"] == _STATE_VERSION
            and type(streak) is int
            and streak >= 0
            and type(last_failure) in (int, float)
        )
    except (TypeError, ValueError, KeyError):
        valid = False
    if not valid:
        return 0, "new"
    if now - last_failure > _STREAK_EXPIRY_SECONDS:
        streak = 0
    return streak, "persisted"


def _write(directory: Path, path: Path, state: dict) -> bool:
    """Replace the file atomically; the directory is created private to the user."""
    tmp = None
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=_TMP_PREFIX, suffix=_TMP_SUFFIX)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(state, stream)
        os.replace(tmp, path)
        return True
    except OSError:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        return False


def _prune(directory: Path, now: float, keep: str) -> None:
    """Remove this module's files that were not updated for 30 days."""
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                name = entry.name
                ours = (name.startswith(_FILE_PREFIX) and name.endswith(_FILE_SUFFIX)) or (
                    name.startswith(_TMP_PREFIX) and name.endswith(_TMP_SUFFIX)
                )
                if not ours or name == keep:
                    continue
                try:
                    if (
                        entry.is_file(follow_symlinks=False)
                        and now - entry.stat(follow_symlinks=False).st_mtime > _PRUNE_AGE_SECONDS
                    ):
                        os.unlink(entry.path)
                except OSError:
                    pass
    except OSError:
        pass
