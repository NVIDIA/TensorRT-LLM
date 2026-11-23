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
"""Opt-in remote host KV tier: one worker lends host memory, another uses it as its last cache level.

The lender (a generation worker) allocates host DRAM next to its GPU as fabric-shareable chunks
(``cuMemCreate`` on ``HOST_NUMA`` with ``CU_MEM_HANDLE_TYPE_FABRIC``, no RDMA flag), writes a
verification word into each chunk, exports the handles and publishes them in a shared directory. It
holds the chunks for the life of the process.

The importer (a context worker) waits for the handles of the lender with the same local rank and
gives them to KVCMv2 as a last ``HOST_MEM`` cache level below its local host tier. KVCMv2 imports
the chunks, checks the verification words and maps them for its GPU only; copies into and out of
the level run on the GPU over the NVLink fabric. Pages move between levels exclusively, so the
level adds its whole size to the cache.

One handle directory serves one context instance and one generation instance of one Slurm job.
The files of local rank ``r`` and job ``J`` are:

* ``rank{r}.job{J}.lender``: the lender's exclusive lock (role, host, global rank, pid). A second
  lender of the same rank and job is refused, unless the lock names a dead process of this host.
* ``rank{r}.job{J}.handles.json`` and ``rank{r}.job{J}.ready``: the handles with the lender's
  identity, and the marker that names their digest. The marker is removed before a re-publish.
* ``rank{r}.job{J}.{nonce}.claim``: the importer's exclusive claim on one lender incarnation
  (``nonce`` is that lender's verification nonce). A second importer of the same chunks is
  refused, so two context workers never share the lent memory.

The file names carry the role, the job, the local rank and the lender incarnation; the host, the
global rank and the pid are in the file contents, because the importer cannot know them before it
reads the lender's files.

There is no fallback: when the tier is requested, any failure (no handles before the timeout, a
stale, foreign or already claimed handle file, an unexpected lent size, an import or verification
error, a cold-page codec on either side, no local host tier) raises and the worker does not start.

Knobs (all unset by default; nothing here runs unless one is set):

* lender: ``TLLM_KV_REMOTE_HOST_LEND_GIB`` (GiB to lend per rank) and
  ``TLLM_KV_REMOTE_HOST_CHUNK_GIB`` (chunk size, default 1);
* importer: ``TLLM_KV_REMOTE_HOST_TIER=1``, ``TLLM_KV_REMOTE_HOST_WAIT_S`` (default 900) and
  ``TLLM_KV_REMOTE_HOST_EXPECT_GIB`` (optional: refuse a lent size other than this many GiB);
* both: ``TLLM_KV_REMOTE_HOST_DIR``, a directory both workers can read and write, and
  ``SLURM_JOB_ID``, which must be set.

Several lenders (opt-in; unset keeps the single-lender file names and behaviour above):

* lender ``TLLM_KV_REMOTE_HOST_LENDER_INDEX=k`` (0 <= k < 16): the lender's files gain ``.l{k}``
  (``rank{r}.job{J}.l{k}.lender``, ``.handles.json``, ``.ready``, ``.{nonce}.claim``), so lenders
  with distinct indices coexist; two lenders with one index are refused as before.
* importer ``TLLM_KV_REMOTE_HOST_NUM_LENDERS=N``: waits for exactly the lenders 0..N-1 within one
  ``TLLM_KV_REMOTE_HOST_WAIT_S`` deadline (a missing lender fails the start; there is no discovery).
  ``TLLM_KV_REMOTE_HOST_EXPECT_GIB`` is then a comma list of the N per-lender sizes in index order.

The importer waits until all N handle files are ready and unchanged on a second read, validates
them, then claims all of them (all or nothing; a refused claim rolls back only this importer's own
claims, and its exit hook removes a claim only while it still owns it), then builds ONE
remote level from all chunks, the lenders interleaved in proportion to their sizes, under the
level's normal LRU. KVCMv2 checks every chunk against one nonce ``V`` (``V ^ position``), while
each lender writes its own ``nonce_k ^ i``. The importer therefore imports each chunk once itself,
checks the lender's word and, with N >= 2, writes ``V ^ position`` before KVCMv2 imports it (with
N = 1 it only checks, and ``V`` is the lender's nonce). After that re-stamp the lenders' words are
gone, so a context worker restarted inside the same job cannot import the chunks again and fails
its start.

Split lending (opt-in; unset keeps every file name and payload above), so that several context
instances share the lenders, each taking one part of every lender:

* lender ``TLLM_KV_REMOTE_HOST_LEND_PARTS=M`` (2 <= M <= 16; unset, empty or 1 means not split;
  needs ``TLLM_KV_REMOTE_HOST_LENDER_INDEX``): ``TLLM_KV_REMOTE_HOST_LEND_GIB`` stays the total per
  rank and is cut into M equal parts of whole chunks (a chunk count not divisible by M is refused).
  Every part is a lender incarnation of its own: its own nonce (chunk ``i`` of part ``j`` reads
  ``nonce_j ^ i``) and its own files ``rank{r}.job{J}.l{k}.p{j}.handles.json``, ``.p{j}.ready`` and
  claim ``.p{j}.{nonce_j}.claim``; the lock ``rank{r}.job{J}.l{k}.lender`` stays one per process.
  All parts are published once every chunk is allocated; the exit hook removes them all.
* importer ``TLLM_KV_REMOTE_HOST_PART=auto`` or ``j`` (needs ``TLLM_KV_REMOTE_HOST_NUM_LENDERS``;
  ``TLLM_KV_REMOTE_HOST_EXPECT_GIB`` then lists the per-lender PART sizes): waits for all parts of
  the lenders 0..N-1 (they must agree on M), checks every part set before claiming anything, then
  claims part ``j`` of every lender, all or nothing. ``auto`` tries the parts in order and moves on
  from a part another live process holds (its own claims of that part rolled back); importers
  that all start at part 0 of lender 0 thus end with disjoint parts. A fixed ``j`` never moves on.
  The import, the re-stamp and the level are the same as above, on the chunks of the claimed part.
"""

import atexit
import ctypes
import hashlib
import json
import math
import os
import re
import resource
import socket
import time
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from tensorrt_llm.logger import logger

LEND_GIB_ENV = "TLLM_KV_REMOTE_HOST_LEND_GIB"
CHUNK_GIB_ENV = "TLLM_KV_REMOTE_HOST_CHUNK_GIB"
TIER_ENV = "TLLM_KV_REMOTE_HOST_TIER"
DIR_ENV = "TLLM_KV_REMOTE_HOST_DIR"
WAIT_S_ENV = "TLLM_KV_REMOTE_HOST_WAIT_S"
EXPECT_GIB_ENV = "TLLM_KV_REMOTE_HOST_EXPECT_GIB"
LENDER_INDEX_ENV = "TLLM_KV_REMOTE_HOST_LENDER_INDEX"
NUM_LENDERS_ENV = "TLLM_KV_REMOTE_HOST_NUM_LENDERS"
LEND_PARTS_ENV = "TLLM_KV_REMOTE_HOST_LEND_PARTS"
PART_ENV = "TLLM_KV_REMOTE_HOST_PART"
JOB_ENV = "SLURM_JOB_ID"

MAX_LENDERS = 16
MAX_PARTS = 16
PART_AUTO = "auto"
DEFAULT_CHUNK_GIB = 1.0
DEFAULT_WAIT_S = 900.0
LOG_PREFIX = "[kv_remote_host]"
STATS_INTERVAL_S = 60.0
FORMAT_VERSION = 2
FABRIC_HANDLE_BYTES = 64

_GIB = 1 << 30
_POLL_S = 1.0
_JOB_TAG = re.compile(r"[A-Za-z0-9._-]+")


@dataclass(frozen=True)
class LendConfig:
    lend_bytes: int
    chunk_bytes: int
    directory: str
    # TLLM_KV_REMOTE_HOST_LENDER_INDEX; None (unset) keeps the single-lender file names.
    lender_index: Optional[int] = None
    # TLLM_KV_REMOTE_HOST_LEND_PARTS (2..16); None (unset, empty or 1) lends one unsplit set.
    num_parts: Optional[int] = None

    @property
    def num_chunks(self) -> int:
        return self.lend_bytes // self.chunk_bytes

    @property
    def part_chunks(self) -> int:
        """The chunks of one part (all chunks when not split)."""
        return self.num_chunks // (self.num_parts or 1)


@dataclass(frozen=True)
class TierConfig:
    directory: str
    wait_s: float
    expect_bytes: Optional[int] = None
    # TLLM_KV_REMOTE_HOST_NUM_LENDERS; None (unset) imports the one lender of the single-lender mode.
    num_lenders: Optional[int] = None
    # TLLM_KV_REMOTE_HOST_EXPECT_GIB as per-lender sizes in index order (with num_lenders only).
    expect_bytes_per_lender: Optional[Tuple[int, ...]] = None
    # TLLM_KV_REMOTE_HOST_PART: PART_AUTO or a part index (with num_lenders only); None = not split.
    part: Union[None, int, str] = None


@dataclass(frozen=True)
class Identity:
    """Who wrote a lock, handle file or claim."""

    role: str
    host: str
    rank: int
    local_rank: int
    pid: int

    @staticmethod
    def of(role: str, rank: int, local_rank: int) -> "Identity":
        return Identity(
            role=role, host=socket.gethostname(), rank=rank, local_rank=local_rank, pid=os.getpid()
        )

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _is_set(env: Mapping[str, str], name: str) -> bool:
    return env.get(name, "").strip() not in ("", "0")


def lend_requested(env: Mapping[str, str] = os.environ) -> bool:
    """Whether ``TLLM_KV_REMOTE_HOST_LEND_GIB`` asks for lending (its value is checked later)."""
    return _is_set(env, LEND_GIB_ENV)


def tier_requested(env: Mapping[str, str] = os.environ) -> bool:
    """Whether ``TLLM_KV_REMOTE_HOST_TIER`` asks for the tier (its value is checked later)."""
    return _is_set(env, TIER_ENV)


def _gib_to_bytes(name: str, text: str) -> int:
    try:
        gib = float(text)
    except ValueError as error:
        raise ValueError(f"{name}={text!r} is not a number of GiB") from error
    if not math.isfinite(gib) or gib <= 0:
        raise ValueError(f"{name}={text!r} must be a positive number of GiB")
    return int(round(gib * _GIB))


def _small_int(name: str, text: str, low: int, high: int) -> int:
    """An integer in [low, high]; anything else is refused."""
    try:
        value = int(text)
    except ValueError as error:
        raise ValueError(f"{name}={text!r} is not an integer") from error
    if not low <= value <= high:
        raise ValueError(f"{name}={text!r} must be in [{low}, {high}]")
    return value


def _lender_index(env: Mapping[str, str]) -> Optional[int]:
    """``TLLM_KV_REMOTE_HOST_LENDER_INDEX``, or None when unset or empty (0 is a valid index)."""
    text = env.get(LENDER_INDEX_ENV, "").strip()
    return _small_int(LENDER_INDEX_ENV, text, 0, MAX_LENDERS - 1) if text else None


def _directory(env: Mapping[str, str]) -> str:
    directory = env.get(DIR_ENV, "")
    if not directory:
        raise ValueError(f"{DIR_ENV} must name a directory shared by the lender and the importer")
    return directory


def parse_lend_config(env: Mapping[str, str] = os.environ) -> Optional[LendConfig]:
    """The lender config, or None when ``TLLM_KV_REMOTE_HOST_LEND_GIB`` is unset, empty or 0."""
    text = env.get(LEND_GIB_ENV, "").strip()
    if text in ("", "0"):
        return None
    lend_bytes = _gib_to_bytes(LEND_GIB_ENV, text)
    chunk_text = env.get(CHUNK_GIB_ENV, "").strip()
    chunk_bytes = (
        _gib_to_bytes(CHUNK_GIB_ENV, chunk_text) if chunk_text else int(DEFAULT_CHUNK_GIB * _GIB)
    )
    if chunk_bytes % (2 << 20) != 0:
        raise ValueError(f"{CHUNK_GIB_ENV}={chunk_text!r} must be a multiple of 2 MiB")
    if lend_bytes < chunk_bytes:
        raise ValueError(f"{LEND_GIB_ENV}={text!r} is smaller than one chunk")
    lender_index = _lender_index(env)
    num_parts = _lend_parts(env)
    if num_parts is not None:
        if lender_index is None:
            raise ValueError(f"{LEND_PARTS_ENV} needs {LENDER_INDEX_ENV} (split parts are indexed)")
        num_chunks = lend_bytes // chunk_bytes
        if num_chunks % num_parts != 0:
            raise ValueError(
                f"{LEND_PARTS_ENV}={num_parts}: {num_chunks} chunks of {chunk_bytes / _GIB:g} GiB "
                f"({LEND_GIB_ENV}={text!r}) do not split into {num_parts} equal parts"
            )
    return LendConfig(
        lend_bytes=lend_bytes // chunk_bytes * chunk_bytes,
        chunk_bytes=chunk_bytes,
        directory=_directory(env),
        lender_index=lender_index,
        num_parts=num_parts,
    )


def _lend_parts(env: Mapping[str, str]) -> Optional[int]:
    """``TLLM_KV_REMOTE_HOST_LEND_PARTS``, or None when unset, empty or 1 (not split)."""
    text = env.get(LEND_PARTS_ENV, "").strip()
    if not text:
        return None
    num_parts = _small_int(LEND_PARTS_ENV, text, 1, MAX_PARTS)
    return num_parts if num_parts > 1 else None


def _part_choice(env: Mapping[str, str]) -> Union[None, int, str]:
    """``TLLM_KV_REMOTE_HOST_PART``: PART_AUTO, a part index, or None when unset or empty."""
    text = env.get(PART_ENV, "").strip()
    if not text:
        return None
    if text == PART_AUTO:
        return PART_AUTO
    try:
        return _small_int(PART_ENV, text, 0, MAX_PARTS - 1)
    except ValueError as error:
        raise ValueError(f"{error}; use {PART_AUTO} or a part index") from None


def parse_tier_config(env: Mapping[str, str] = os.environ) -> Optional[TierConfig]:
    """The importer config, or None when ``TLLM_KV_REMOTE_HOST_TIER`` is unset, empty or 0."""
    text = env.get(TIER_ENV, "").strip()
    if text in ("", "0"):
        return None
    if text != "1":
        raise ValueError(f"{TIER_ENV}={text!r}: use 1 to enable the remote host tier, or unset it")
    wait_text = env.get(WAIT_S_ENV, "").strip()
    wait_s = DEFAULT_WAIT_S
    if wait_text:
        try:
            wait_s = float(wait_text)
        except ValueError as error:
            raise ValueError(f"{WAIT_S_ENV}={wait_text!r} is not a number of seconds") from error
        if not math.isfinite(wait_s) or wait_s <= 0:
            raise ValueError(f"{WAIT_S_ENV}={wait_text!r} must be positive")
    expect_text = env.get(EXPECT_GIB_ENV, "").strip()
    num_text = env.get(NUM_LENDERS_ENV, "").strip()
    part = _part_choice(env)
    if part is not None and not num_text:
        raise ValueError(f"{PART_ENV} needs {NUM_LENDERS_ENV} (split parts are indexed)")
    if not num_text:
        expect_bytes = _gib_to_bytes(EXPECT_GIB_ENV, expect_text) if expect_text else None
        return TierConfig(directory=_directory(env), wait_s=wait_s, expect_bytes=expect_bytes)
    num_lenders = _small_int(NUM_LENDERS_ENV, num_text, 1, MAX_LENDERS)
    per_lender: Optional[Tuple[int, ...]] = None
    if expect_text:
        parts = [part.strip() for part in expect_text.split(",")]
        if len(parts) != num_lenders:
            # A single number with N >= 2 is ambiguous (each lender's size, or the total?).
            raise ValueError(
                f"{EXPECT_GIB_ENV}={expect_text!r}: with {NUM_LENDERS_ENV}={num_lenders} give "
                f"{num_lenders} comma-separated per-lender sizes in lender index order"
            )
        per_lender = tuple(_gib_to_bytes(EXPECT_GIB_ENV, part) for part in parts)
    return TierConfig(
        directory=_directory(env),
        wait_s=wait_s,
        num_lenders=num_lenders,
        expect_bytes_per_lender=per_lender,
        part=part,
    )


def job_tag(env: Mapping[str, str] = os.environ) -> str:
    """Ties the files to one job, so a file left by an earlier job is never imported.

    Refuses an unset or empty job id: with an empty tag, files of any earlier untagged run would match.
    """
    job = env.get(JOB_ENV, "").strip()
    if not job:
        raise ValueError(f"the remote host tier needs {JOB_ENV} to tie its handle files to one job")
    if not _JOB_TAG.fullmatch(job):
        raise ValueError(f"{JOB_ENV}={job!r} is not usable in a file name")
    return job


def _prefix(
    directory: str,
    rank: int,
    job: str,
    lender_index: Optional[int] = None,
    part: Optional[int] = None,
) -> str:
    name = f"rank{rank}.job{job}"
    if lender_index is not None:
        name += f".l{lender_index}"
    if part is not None:
        name += f".p{part}"
    return os.path.join(directory, name)


def handle_path(
    directory: str,
    rank: int,
    job: str,
    lender_index: Optional[int] = None,
    part: Optional[int] = None,
) -> str:
    return f"{_prefix(directory, rank, job, lender_index, part)}.handles.json"


def ready_path(
    directory: str,
    rank: int,
    job: str,
    lender_index: Optional[int] = None,
    part: Optional[int] = None,
) -> str:
    return f"{_prefix(directory, rank, job, lender_index, part)}.ready"


def lender_lock_path(
    directory: str, rank: int, job: str, lender_index: Optional[int] = None
) -> str:
    return f"{_prefix(directory, rank, job, lender_index)}.lender"


def claim_path(
    directory: str,
    rank: int,
    job: str,
    nonce: int,
    lender_index: Optional[int] = None,
    part: Optional[int] = None,
) -> str:
    return f"{_prefix(directory, rank, job, lender_index, part)}.{nonce:016x}.claim"


def _write_atomic(path: str, data: bytes) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _read_json(path: str) -> Any:
    """The parsed file, or None when it is missing, unreadable or not JSON (for example half written)."""
    try:
        with open(path, "rb") as f:
            return json.loads(f.read())
    except (OSError, ValueError):
        return None


def _unlink_own(path: str, me: Identity) -> None:
    """Remove the claim at ``path`` only while ``me`` (this host and pid) still owns it.

    A claim rolled back by a refused all-or-nothing import may be taken by another importer
    before this process exits; the exit hook must leave that claim alone.
    """
    holder = _read_json(path)
    owner = holder.get("owner") if isinstance(holder, dict) else None
    if isinstance(owner, dict) and owner.get("host") == me.host and owner.get("pid") == me.pid:
        _unlink(path)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _is_dead_on_this_host(holder: Any, me: Identity) -> bool:
    """A lock or claim of a process of this host that no longer runs (a crash or an earlier launch)."""
    return (
        isinstance(holder, dict)
        and holder.get("host") == me.host
        and isinstance(holder.get("pid"), int)
        and holder["pid"] != me.pid
        and not _pid_alive(holder["pid"])
    )


class ClaimHeldError(RuntimeError):
    """A lock or claim file is held by another live process."""


def _create_exclusive(path: str, content: Dict[str, Any], me: Identity, what: str) -> None:
    """Create ``path`` with ``O_CREAT | O_EXCL``; refuse when another live process holds it."""
    data = json.dumps(content, sort_keys=True).encode()
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            holder = _read_json(path)
            holder_id = holder.get("owner") if isinstance(holder, dict) else None
            if _is_dead_on_this_host(holder_id, me):
                logger.warning(
                    f"{LOG_PREFIX} replacing the {what} {path} of exited process {holder_id}"
                )
                _unlink(path)
                continue
            raise ClaimHeldError(
                f"{LOG_PREFIX} {what} {path} is held by {holder_id or 'another process'}: one "
                f"context and one generation instance per {DIR_ENV} and job"
            ) from None
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        return
    raise RuntimeError(f"{LOG_PREFIX} could not take the {what} {path}")


def acquire_lender_lock(
    directory: str, rank: int, job: str, me: Identity, lender_index: Optional[int] = None
) -> str:
    """Take the lender lock of ``rank`` (and lender index) for this job; refuse a second lender."""
    os.makedirs(directory, exist_ok=True)
    path = lender_lock_path(directory, rank, job, lender_index)
    _create_exclusive(path, {"job": job, "owner": me.as_dict()}, me, "lender lock")
    return path


def publish(
    directory: str,
    rank: int,
    payload: Dict[str, Any],
    job: str,
    lender_index: Optional[int] = None,
    part: Optional[int] = None,
) -> None:
    """Write the handle file, then the ready marker that names its job, digest and lender.

    The caller holds the lender lock. The old marker is removed first, so a marker never names a
    handle file it does not match. With ``part`` (split lending) the files are the part's, and the
    marker also names the part and the number of parts.
    """
    os.makedirs(directory, exist_ok=True)
    _unlink(ready_path(directory, rank, job, lender_index, part))
    data = json.dumps(payload, sort_keys=True).encode()
    _write_atomic(handle_path(directory, rank, job, lender_index, part), data)
    marker = {
        "job": job,
        "sha256": hashlib.sha256(data).hexdigest(),
        "format": FORMAT_VERSION,
        "lender": payload.get("lender"),
    }
    if lender_index is not None:
        marker["lender_index"] = lender_index
    if part is not None:
        marker["part"] = part
        marker["num_parts"] = payload.get("num_parts")
    _write_atomic(
        ready_path(directory, rank, job, lender_index, part),
        json.dumps(marker, sort_keys=True).encode(),
    )


def _remove_published(
    directory: str,
    rank: int,
    job: str,
    lender_index: Optional[int] = None,
    num_parts: Optional[int] = None,
) -> None:
    """The lender's files, at exit: its chunks die with the process, so its handles are useless.

    With ``num_parts`` (split lending) the ready marker and handle file of every part go, then the
    lock; the unsuffixed files are never written in that mode.
    """
    if num_parts is None:
        published = [
            ready_path(directory, rank, job, lender_index),
            handle_path(directory, rank, job, lender_index),
        ]
    else:
        published = [
            path_of(directory, rank, job, lender_index, part)
            for part in range(num_parts)
            for path_of in (ready_path, handle_path)
        ]
    for path in published + [lender_lock_path(directory, rank, job, lender_index)]:
        try:
            _unlink(path)
        except OSError:
            pass


def _valid_payload(
    payload: Any,
    rank: int,
    job: str,
    lender_index: Optional[int] = None,
    part: Optional[int] = None,
) -> bool:
    if not isinstance(payload, dict):
        return False
    if lender_index is not None and payload.get("lender_index") != lender_index:
        return False
    if part is not None:
        num_parts = payload.get("num_parts")
        if (
            payload.get("part") != part
            or not isinstance(num_parts, int)
            or not 2 <= num_parts <= MAX_PARTS
            or part >= num_parts
        ):
            return False
    handles = payload.get("handles")
    chunk_bytes = payload.get("chunk_bytes")
    lender = payload.get("lender")
    if (
        payload.get("rank") != rank
        or payload.get("job") != job
        or not isinstance(handles, list)
        or not handles
        or not isinstance(chunk_bytes, int)
        or chunk_bytes <= 0
        or not isinstance(payload.get("nonce"), int)
        or not isinstance(lender, dict)
        or not isinstance(lender.get("host"), str)
        or not isinstance(lender.get("pid"), int)
    ):
        return False
    try:
        return all(
            isinstance(h, str) and len(bytes.fromhex(h)) == FABRIC_HANDLE_BYTES for h in handles
        )
    except ValueError:
        return False


def _try_read_payload(
    directory: str,
    rank: int,
    job: str,
    lender_index: Optional[int] = None,
    part: Optional[int] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """(payload, None) once the lender's files are consistent, else (None, why not yet).

    Only a file that cannot become right by waiting raises: an unknown format, or a handle file
    whose digest matches its marker but whose content is malformed.
    """
    marker = _read_json(ready_path(directory, rank, job, lender_index, part))
    if marker is None:
        return None, None
    if not isinstance(marker, dict):
        return None, "the ready marker is not a JSON object"
    if marker.get("job") != job:
        return None, f"the ready marker names job {marker.get('job')!r}"
    if marker.get("format") != FORMAT_VERSION:
        raise RuntimeError(f"{LOG_PREFIX} rank {rank}: handle format {marker.get('format')!r}")
    try:
        with open(handle_path(directory, rank, job, lender_index, part), "rb") as f:
            data = f.read()
    except OSError:
        return None, "the handle file is missing"
    if hashlib.sha256(data).hexdigest() != marker.get("sha256"):
        # A lender that re-publishes replaces the handle file before the marker.
        return None, "the handle file does not match its ready marker"
    try:
        payload = json.loads(data)
    except ValueError:
        payload = None
    if not _valid_payload(payload, rank, job, lender_index, part):
        which = "" if lender_index is None else f" lender index {lender_index}"
        if part is not None:
            which += f" part {part}"
        raise RuntimeError(f"{LOG_PREFIX} rank {rank}{which}: malformed handle file")
    return payload, None


def wait_for_payload(
    directory: str,
    rank: int,
    wait_s: float,
    job: str,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    lender_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Block until the lender of ``rank`` has published its handles for this job, then return them.

    A missing or inconsistent file (a marker and handle file that do not match yet) is retried
    until the timeout; at the timeout the last reason is reported.
    """
    deadline = clock() + wait_s
    last_reason: Optional[str] = None
    while True:
        payload, reason = _try_read_payload(directory, rank, job, lender_index)
        if payload is not None:
            return payload
        last_reason = reason or last_reason
        if clock() >= deadline:
            if last_reason is not None:
                raise RuntimeError(
                    f"{LOG_PREFIX} rank {rank}: {last_reason} in {directory} after {wait_s:g} s"
                )
            raise TimeoutError(
                f"{LOG_PREFIX} no handles from lender rank {rank} of job {job!r} in "
                f"{ready_path(directory, rank, job, lender_index)} after {wait_s:g} s"
            )
        sleep(_POLL_S)


def claim_lender(
    directory: str,
    rank: int,
    job: str,
    payload: Dict[str, Any],
    me: Identity,
    lender_index: Optional[int] = None,
    part: Optional[int] = None,
) -> str:
    """Take the importer's exclusive claim on this lender incarnation; refuse when another holds it.

    The claim is removed at exit, so a context worker restarted on this host can import the same
    chunks again (single-lender mode; see the module docstring for several lenders). An indexed
    claim (several-lender mode) is removed at exit only while this process still owns it: a
    refused import rolls its claims back, and another importer may take them before this exits.
    With ``part`` (split lending) the claim is on that part of the lender.

    Raises:
        ClaimHeldError: another live process holds the claim.
    """
    path = claim_path(directory, rank, job, int(payload["nonce"]), lender_index, part)
    _create_exclusive(
        path, {"job": job, "owner": me.as_dict(), "lender": payload["lender"]}, me, "claim"
    )
    if lender_index is None:
        atexit.register(_unlink, path)
    else:
        atexit.register(_unlink_own, path, me)
    return path


# ---- lender (generation worker) ------------------------------------------------------------------


def _cuda_driver():
    try:
        from cuda.bindings import driver as cuda
    except ImportError:
        from cuda import cuda
    return cuda


def _check(result):
    cuda = _cuda_driver()
    err, *rest = result if isinstance(result, tuple) else (result,)
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{LOG_PREFIX} CUDA driver error {err}")
    if not rest:
        return None
    return rest[0] if len(rest) == 1 else tuple(rest)


class RemoteHostLender:
    """Allocates, verifies-in and exports the chunks one rank lends; holds them until released."""

    def __init__(self, config: LendConfig, rank: int, device: int) -> None:
        self.config = config
        self.rank = rank
        self.device = device
        self.numa = -1
        # One nonce per part (one when not split); chunk i of part j reads nonces[j] ^ i.
        self.nonces = [
            int.from_bytes(os.urandom(8), "little") for _ in range(config.num_parts or 1)
        ]
        self.nonce = self.nonces[0]
        self.lend_s = 0.0
        self._handles: List[int] = []
        self.fabric_handles: List[bytes] = []

    def _host_numa(self, cu_device) -> int:
        cuda = _cuda_driver()
        try:
            numa = _check(
                cuda.cuDeviceGetAttribute(
                    cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, cu_device
                )
            )
        except (AttributeError, RuntimeError):
            numa = -1
        # Two GPUs per CPU socket: fall back to HOST_NUMA(rank // 2) when the driver does not say.
        return int(numa) if numa is not None and int(numa) >= 0 else self.rank // 2

    def _write_head(self, handle: int, value: int, cu_device) -> None:
        cuda = _cuda_driver()
        size = self.config.chunk_bytes
        # Aligned to the chunk, as the importer maps it.
        va = _check(cuda.cuMemAddressReserve(size, size, 0, 0))
        try:
            _check(cuda.cuMemMap(va, size, 0, handle, 0))
            try:
                access = cuda.CUmemAccessDesc()
                access.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
                access.location.id = int(cu_device)
                access.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                _check(cuda.cuMemSetAccess(va, size, [access], 1))
                word = ctypes.c_uint64(value)
                _check(cuda.cuMemcpyHtoD(va, ctypes.addressof(word), ctypes.sizeof(word)))
            finally:
                _check(cuda.cuMemUnmap(va, size))
        finally:
            _check(cuda.cuMemAddressFree(va, size))

    def lend(self) -> Dict[str, Any]:
        from tensorrt_llm._torch.modules.dwdp.transport import _export_handle

        cuda = _cuda_driver()
        cu_device = _check(cuda.cuDeviceGet(self.device))
        self.numa = self._host_numa(cu_device)
        prop = cuda.CUmemAllocationProp()
        prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
        prop.location.id = self.numa
        prop.requestedHandleTypes = cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        prop.allocFlags.gpuDirectRDMACapable = 0
        fabric = cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        start = time.monotonic()
        part_chunks = self.config.part_chunks
        try:
            for index in range(self.config.num_chunks):
                handle = _check(cuda.cuMemCreate(self.config.chunk_bytes, prop, 0))
                self._handles.append(handle)
                part, local = divmod(index, part_chunks)
                self._write_head(handle, self.nonces[part] ^ local, cu_device)
                exported = _export_handle(int(handle), fabric)
                if len(exported) != FABRIC_HANDLE_BYTES:
                    raise RuntimeError(f"{LOG_PREFIX} fabric handle of {len(exported)} bytes")
                self.fabric_handles.append(exported)
        except BaseException:
            self.release()
            raise
        self.lend_s = time.monotonic() - start
        return {
            "format": FORMAT_VERSION,
            "rank": self.rank,
            "device": self.device,
            "numa": self.numa,
            "chunk_bytes": self.config.chunk_bytes,
            "nonce": self.nonce,
            "handles": [h.hex() for h in self.fabric_handles],
        }

    def release(self) -> None:
        cuda = _cuda_driver()
        for handle in self._handles:
            cuda.cuMemRelease(handle)
        self._handles.clear()
        self.fabric_handles.clear()


def part_payloads(
    payload: Dict[str, Any], nonces: Sequence[int], num_parts: int
) -> List[Dict[str, Any]]:
    """The handle file of every part: ``payload`` with that part's nonce and handles only."""
    handles = payload["handles"]
    if len(nonces) != num_parts or len(handles) % num_parts != 0:
        raise ValueError(
            f"{LOG_PREFIX} {len(handles)} chunks and {len(nonces)} nonces do not form "
            f"{num_parts} parts"
        )
    size = len(handles) // num_parts
    return [
        dict(
            payload,
            nonce=nonces[part],
            handles=handles[part * size : (part + 1) * size],
            part=part,
            num_parts=num_parts,
        )
        for part in range(num_parts)
    ]


# Process-wide: a worker lends once, whatever number of KV cache managers it builds.
_LENDER: Optional[RemoteHostLender] = None


def maybe_lend(
    local_rank: int,
    device_fn: Callable[[], int],
    is_draft: bool,
    codec_configured: bool = False,
    global_rank: int = -1,
) -> Optional[RemoteHostLender]:
    """Lend this rank's chunks once per process when ``TLLM_KV_REMOTE_HOST_LEND_GIB`` is set.

    ``device_fn`` returns the CUDA device ordinal; it is called only when lending. The tier does
    not support a cold-page codec, so a lender with one is refused.
    """
    global _LENDER
    config = parse_lend_config()
    if config is None or is_draft:
        return None
    if _LENDER is not None:
        return _LENDER
    if parse_tier_config() is not None:
        raise ValueError(f"{LEND_GIB_ENV} and {TIER_ENV} are for different workers; set one")
    if codec_configured:
        raise ValueError(f"{LEND_GIB_ENV} cannot be combined with a cold-page codec")
    if os.environ.get(NUM_LENDERS_ENV, "").strip():
        raise ValueError(f"{NUM_LENDERS_ENV} is an importer knob; a lender sets {LENDER_INDEX_ENV}")
    if os.environ.get(PART_ENV, "").strip():
        raise ValueError(f"{PART_ENV} is an importer knob; a lender sets {LEND_PARTS_ENV}")
    job = job_tag()
    index = config.lender_index
    num_parts = config.num_parts
    me = Identity.of("lender", rank=global_rank, local_rank=local_rank)
    # Refuses a second lender of this rank, lender index and job before anything is allocated.
    acquire_lender_lock(config.directory, local_rank, job, me, index)
    try:
        device = int(device_fn())
        rss_before = _rss_mb()
        lender = RemoteHostLender(config, rank=local_rank, device=device)
        payload = lender.lend()  # releases its own chunks when it fails
        payload["job"] = job
        payload["lender"] = me.as_dict()
        if index is not None:
            payload["lender_index"] = index
        try:
            if num_parts is None:
                publish(config.directory, local_rank, payload, job, index)
            else:
                # Every chunk is allocated and stamped: the parts appear together.
                for part, part_payload in enumerate(
                    part_payloads(payload, lender.nonces, num_parts)
                ):
                    publish(config.directory, local_rank, part_payload, job, index, part)
        except BaseException:
            lender.release()
            raise
    except BaseException:
        _remove_published(config.directory, local_rank, job, index, num_parts)
        raise
    if num_parts is None:
        atexit.register(_remove_published, config.directory, local_rank, job, index)
    else:
        atexit.register(_remove_published, config.directory, local_rank, job, index, num_parts)
    _LENDER = lender
    split = (
        ""
        if num_parts is None
        else f"parts={num_parts} part_gib={config.part_chunks * config.chunk_bytes / _GIB:.2f} "
    )
    logger.info(
        f"{LOG_PREFIX} lent rank={local_rank} global_rank={global_rank} host={me.host} "
        f"{'' if index is None else f'lender_index={index} '}{split}"
        f"pid={me.pid} device={device} numa={lender.numa} chunks={config.num_chunks} "
        f"chunk_gib={config.chunk_bytes / _GIB:g} lent_gib={config.lend_bytes / _GIB:.2f} "
        f"lend_s={lender.lend_s:.1f} rss_mb={rss_before}->{_rss_mb()} codec=off "
        f"dir={config.directory} job={job!r}"
    )
    return lender


# ---- importer (context worker) -------------------------------------------------------------------


def build_remote_tier(
    *,
    local_rank: int,
    is_draft: bool,
    is_estimating: bool,
    local_host_quota: int,
    codec_configured: bool,
    has_disk_tier: bool,
    global_rank: int = -1,
) -> Optional[Any]:
    """The remote ``HostCacheTierConfig`` to append to the tier list, or None when the tier is off.

    Only the target manager of a rank takes the tier: the lent chunks back one cache. The draft
    manager and the throwaway estimation manager keep their local tiers.
    """
    config = parse_tier_config()
    if config is None:
        return None
    if is_draft or is_estimating:
        logger.info(
            f"{LOG_PREFIX} rank={local_rank}: no remote tier on the "
            f"{'draft' if is_draft else 'estimation'} KV cache manager"
        )
        return None
    if codec_configured:
        raise ValueError(f"{TIER_ENV}=1 cannot be combined with a cold-page codec")
    if local_host_quota <= 0:
        raise ValueError(f"{TIER_ENV}=1 needs a local host tier (host_cache_size > 0)")
    if has_disk_tier:
        raise ValueError(f"{TIER_ENV}=1 cannot be combined with a disk tier")
    if parse_lend_config() is not None:
        raise ValueError(f"{LEND_GIB_ENV} and {TIER_ENV} are for different workers; set one")
    if os.environ.get(LENDER_INDEX_ENV, "").strip():
        raise ValueError(f"{LENDER_INDEX_ENV} is a lender knob; an importer sets {NUM_LENDERS_ENV}")
    if os.environ.get(LEND_PARTS_ENV, "").strip():
        raise ValueError(f"{LEND_PARTS_ENV} is a lender knob; an importer sets {PART_ENV}")
    if config.num_lenders is not None:
        return _build_merged_tier(config, local_rank, global_rank)
    job = job_tag()
    start = time.monotonic()
    payload = wait_for_payload(config.directory, local_rank, config.wait_s, job)
    handles = [bytes.fromhex(h) for h in payload["handles"]]
    chunk_bytes = int(payload["chunk_bytes"])
    lent_bytes = len(handles) * chunk_bytes
    lender = payload["lender"]
    if config.expect_bytes is not None and lent_bytes != config.expect_bytes:
        raise ValueError(
            f"{LOG_PREFIX} rank={local_rank}: lender {lender.get('host')}:{lender.get('pid')} lent "
            f"{lent_bytes / _GIB:.2f} GiB, {EXPECT_GIB_ENV} expects {config.expect_bytes / _GIB:.2f}"
        )
    me = Identity.of("importer", rank=global_rank, local_rank=local_rank)
    claim_lender(config.directory, local_rank, job, payload, me)
    from tensorrt_llm.runtime.kv_cache_manager_v2 import HostCacheTierConfig, RemoteHostTierConfig

    remote = RemoteHostTierConfig(handles, chunk_bytes, int(payload["nonce"]))
    logger.info(
        f"{LOG_PREFIX} rank={local_rank} global_rank={global_rank}: importing {len(handles)} chunks "
        f"of {chunk_bytes / _GIB:g} GiB ({lent_bytes / _GIB:.2f} GiB) from lender "
        f"host={lender.get('host')} pid={lender.get('pid')} global_rank={lender.get('rank')} "
        f"device={payload.get('device')} numa={payload.get('numa')} after "
        f"{time.monotonic() - start:.1f} s wait"
    )
    return HostCacheTierConfig(quota=lent_bytes, remote=remote)


# ---- several lenders, one merged level (TLLM_KV_REMOTE_HOST_NUM_LENDERS) ---------------------------


@dataclass(frozen=True)
class MergedImport:
    """The chunks an importer took from several lenders, in the order KVCMv2 receives them."""

    rank: int
    global_rank: int
    chunk_bytes: int
    # KVCMv2 requires chunk ``g`` of the merged list to read ``verify_nonce ^ g``.
    verify_nonce: int
    handles: Tuple[bytes, ...]
    # The lender index of every merged position, and the lender identities by index.
    owners: Tuple[int, ...]
    lenders: Tuple[Dict[str, Any], ...]
    # The part of every lender this importer claimed (split lending), else None.
    part: Optional[int] = None

    @property
    def lent_bytes(self) -> int:
        return len(self.handles) * self.chunk_bytes


# Process-wide, like _LENDER: the import of several lenders of this process, for the spread log.
_MERGED: Optional[MergedImport] = None


def merged_import() -> Optional[MergedImport]:
    """The import of several lenders done by this process, or None (single-lender mode or no tier)."""
    return _MERGED


def interleave_order(counts: Sequence[int]) -> List[Tuple[int, int]]:
    """(lender index, chunk index) of every chunk, the lenders interleaved by their chunk counts.

    Chunk ``i`` of a lender with ``n`` chunks sits at (i + 1/2) / n of the merged list (ties go to
    the lower lender index). KVCMv2 hands the chunks out in list order and every pool takes a
    contiguous run of them, so each pool draws from all lenders in proportion to their sizes.
    """
    keyed = [(Fraction(2 * i + 1, 2 * n), k, i) for k, n in enumerate(counts) for i in range(n)]
    keyed.sort()
    return [(k, i) for _, k, i in keyed]


def wait_for_lenders(
    directory: str,
    rank: int,
    num_lenders: int,
    wait_s: float,
    job: str,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    part: Optional[int] = None,
    deadline: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Block until the lenders 0..num_lenders-1 of ``rank`` have published for this job.

    One deadline covers all of them. Once all are ready, every lender's files are read once more,
    and a lender that re-published meanwhile is waited for again, so the handles returned are the
    lenders' current ones. At the timeout the error names every missing or still changing lender
    index with its last reason; a missing lender is never left out of the level.

    With ``part`` (split lending) the files of that part of every lender are waited for.
    ``deadline`` (a ``clock`` value) replaces ``clock() + wait_s``, so several parts share one.
    """
    if deadline is None:
        deadline = clock() + wait_s

    def read(index: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if part is None:
            return _try_read_payload(directory, rank, job, index)
        return _try_read_payload(directory, rank, job, index, part)

    payloads: List[Optional[Dict[str, Any]]] = [None] * num_lenders
    reasons: Dict[int, str] = {}
    while True:
        for index in range(num_lenders):
            if payloads[index] is None:
                payload, reason = read(index)
                payloads[index] = payload
                if reason is not None:
                    reasons[index] = reason
        unsettled: List[int] = []
        if all(payload is not None for payload in payloads):
            for index in range(num_lenders):
                current, reason = read(index)
                if current != payloads[index]:
                    payloads[index] = current
                    unsettled.append(index)
                    reasons[index] = reason or "re-published during the wait"
            if not unsettled:
                return [payload for payload in payloads if payload is not None]
        if clock() >= deadline:
            missing = sorted(
                set(unsettled)
                | {index for index, payload in enumerate(payloads) if payload is None}
            )
            details = "; ".join(
                f"index {i}: {reasons.get(i) or _no_marker(directory, rank, job, i, part)}"
                for i in missing
            )
            which = "" if part is None else f" (part {part})"
            raise TimeoutError(
                f"{LOG_PREFIX} rank {rank}: lender index(es) {missing} of {num_lenders}{which} of "
                f"job {job!r} not ready in {directory} after {wait_s:g} s ({details})"
            )
        sleep(_POLL_S)


def _no_marker(directory: str, rank: int, job: str, index: int, part: Optional[int]) -> str:
    """Why a lender has no ready marker, naming the marker of the other mode when that exists."""
    if part is None:
        other = ready_path(directory, rank, job, index, 0)
        hint = f"the lender is split, set {PART_ENV}"
    else:
        other = ready_path(directory, rank, job, index)
        hint = f"the lender is not split, unset {PART_ENV} or set {LEND_PARTS_ENV} on the lenders"
    if os.path.exists(other):
        return f"no ready marker; {os.path.basename(other)} exists ({hint})"
    return "no ready marker"


def _describe(index: int, payload: Dict[str, Any]) -> str:
    lender = payload["lender"]
    part = "" if payload.get("part") is None else f" part {payload['part']}"
    return f"lender index {index}{part} host={lender.get('host')} pid={lender.get('pid')}"


def check_lenders(payloads: Sequence[Dict[str, Any]], config: TierConfig, rank: int) -> int:
    """Refuse lender sets that cannot form one level; return the common chunk size."""
    chunk_sizes = [int(payload["chunk_bytes"]) for payload in payloads]
    if len(set(chunk_sizes)) != 1:
        raise ValueError(
            f"{LOG_PREFIX} rank={rank}: the lenders lent chunks of different sizes {chunk_sizes}; "
            f"one merged level needs one {CHUNK_GIB_ENV}"
        )
    processes: Dict[Tuple[Any, Any], int] = {}
    for index, payload in enumerate(payloads):
        key = (payload["lender"].get("host"), payload["lender"].get("pid"))
        if key in processes:
            raise ValueError(
                f"{LOG_PREFIX} rank={rank}: lender indices {processes[key]} and {index} name one "
                f"process (host={key[0]} pid={key[1]})"
            )
        processes[key] = index
    seen: Dict[str, int] = {}
    for index, payload in enumerate(payloads):
        for handle in payload["handles"]:
            if handle in seen:
                raise ValueError(
                    f"{LOG_PREFIX} rank={rank}: lender indices {seen[handle]} and {index} publish "
                    f"the same chunk handle"
                )
            seen[handle] = index
    if config.expect_bytes_per_lender is not None:
        for index, (payload, expect) in enumerate(zip(payloads, config.expect_bytes_per_lender)):
            lent = len(payload["handles"]) * chunk_sizes[0]
            if lent != expect:
                raise ValueError(
                    f"{LOG_PREFIX} rank={rank}: {_describe(index, payload)} lent "
                    f"{lent / _GIB:.2f} GiB, {EXPECT_GIB_ENV} expects {expect / _GIB:.2f} for it"
                )
    return chunk_sizes[0]


def claim_lenders(
    directory: str,
    rank: int,
    job: str,
    payloads: Sequence[Dict[str, Any]],
    me: Identity,
    part: Optional[int] = None,
) -> List[str]:
    """Claim every lender in index order, all or nothing: a refusal removes the claims taken.

    With ``part`` (split lending) the claims are on that part of every lender.
    """
    taken: List[str] = []
    try:
        for index, payload in enumerate(payloads):
            taken.append(claim_lender(directory, rank, job, payload, me, index, part))
    except BaseException:
        for path in taken:
            _unlink_own(path, me)
        raise
    return taken


# ---- split lending (TLLM_KV_REMOTE_HOST_LEND_PARTS / TLLM_KV_REMOTE_HOST_PART) -------------------------


def wait_for_parts(
    directory: str,
    rank: int,
    num_lenders: int,
    wait_s: float,
    job: str,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> List[List[Dict[str, Any]]]:
    """Every part of the lenders 0..num_lenders-1 of ``rank``: ``parts[j][k]`` is part j of lender k.

    Part 0 of every lender is waited for first and names the number of parts M, on which all lenders
    must agree; then parts 1..M-1, all under one ``wait_s`` deadline.

    Raises:
        TimeoutError: a part of a lender is not ready before the deadline.
        ValueError: the lenders disagree on M.
    """
    deadline = clock() + wait_s
    first = wait_for_lenders(
        directory, rank, num_lenders, wait_s, job, clock, sleep, part=0, deadline=deadline
    )
    counts = [payload["num_parts"] for payload in first]
    if len(set(counts)) != 1:
        raise ValueError(
            f"{LOG_PREFIX} rank={rank}: the lenders lent different numbers of parts {counts} "
            f"(lender index order); one {LEND_PARTS_ENV} for all lenders"
        )
    parts = [first]
    for part in range(1, counts[0]):
        parts.append(
            wait_for_lenders(
                directory,
                rank,
                num_lenders,
                wait_s,
                job,
                clock,
                sleep,
                part=part,
                deadline=deadline,
            )
        )
    return parts


def check_parts(parts: Sequence[Sequence[Dict[str, Any]]], config: TierConfig, rank: int) -> int:
    """Refuse part sets that cannot be claimed one part at a time; return the chunk size.

    Every part set must form one level (``check_lenders``, with ``TLLM_KV_REMOTE_HOST_EXPECT_GIB`` as
    the per-lender part sizes), every part must name its lender's M and process, no chunk handle
    may appear twice, and a fixed ``TLLM_KV_REMOTE_HOST_PART`` must be below M.
    """
    num_parts = len(parts)
    if isinstance(config.part, int) and config.part >= num_parts:
        raise ValueError(
            f"{LOG_PREFIX} rank={rank}: {PART_ENV}={config.part} but the lenders lent {num_parts} parts"
        )
    chunk_bytes = [check_lenders(part_set, config, rank) for part_set in parts]
    if len(set(chunk_bytes)) != 1:
        raise ValueError(
            f"{LOG_PREFIX} rank={rank}: the parts hold chunks of different sizes {chunk_bytes}"
        )
    seen: Dict[str, Tuple[int, int]] = {}
    for part, part_set in enumerate(parts):
        for index, payload in enumerate(part_set):
            base = parts[0][index]
            if payload["num_parts"] != num_parts or payload["lender"] != base["lender"]:
                raise ValueError(
                    f"{LOG_PREFIX} rank={rank}: {_describe(index, payload)} does not match part 0 "
                    f"of that lender ({_describe(index, base)}, {base['num_parts']} parts)"
                )
            for handle in payload["handles"]:
                if handle in seen:
                    raise ValueError(
                        f"{LOG_PREFIX} rank={rank}: lender index {seen[handle][0]} part "
                        f"{seen[handle][1]} and lender index {index} part {part} publish the "
                        f"same chunk handle"
                    )
                seen[handle] = (index, part)
    return chunk_bytes[0]


def claim_part(
    directory: str,
    rank: int,
    job: str,
    parts: Sequence[Sequence[Dict[str, Any]]],
    me: Identity,
    choice: Union[int, str],
) -> int:
    """Claim one part of every lender, all or nothing, and return its index.

    A fixed ``choice`` claims that part or fails. ``PART_AUTO`` tries parts 0..M-1 in order: a part
    whose claim another live process holds is rolled back (this importer's own claims of it only)
    and the next part is tried; any other error stops the import. Importers that all start at part
    0 of lender 0 cannot meet again on a later part, so each ends with a different part.

    Raises:
        ClaimHeldError: a fixed part is held, or (``PART_AUTO``) every part is held; the error names
            the holders.
    """
    if choice != PART_AUTO:
        claim_lenders(directory, rank, job, parts[choice], me, choice)
        return int(choice)
    held: List[str] = []
    for part, part_set in enumerate(parts):
        try:
            claim_lenders(directory, rank, job, part_set, me, part)
        except ClaimHeldError as error:
            held.append(f"part {part}: {error}")
            continue
        return part
    raise ClaimHeldError(
        f"{LOG_PREFIX} rank={rank}: all {len(parts)} parts of the lenders are held by other "
        f"importers ({'; '.join(held)})"
    )


def _current_device():
    cuda = _cuda_driver()
    return _check(cuda.cuCtxGetDevice())


def _read_word(address) -> int:
    """The 8 bytes at ``address`` (a synchronous copy to this host)."""
    cuda = _cuda_driver()
    word = ctypes.c_uint64(0)
    _check(cuda.cuMemcpyDtoH(ctypes.addressof(word), address, ctypes.sizeof(word)))
    return word.value


def _verify_and_stamp_chunk(
    fabric_handle: bytes, chunk_bytes: int, cu_device, expected: int, new: Optional[int]
) -> None:
    """Import one lent chunk, require its head to read ``expected`` and, unless ``new`` is None,
    write ``new`` there and read it back. The import is unmapped and released before returning;
    KVCMv2 imports the chunk again from the same fabric handle."""
    from tensorrt_llm._torch.modules.dwdp.transport import _import_handle

    cuda = _cuda_driver()
    handle = _import_handle(fabric_handle, cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC)
    try:
        va = _check(cuda.cuMemAddressReserve(chunk_bytes, chunk_bytes, 0, 0))
        try:
            _check(cuda.cuMemMap(va, chunk_bytes, 0, handle, 0))
            try:
                access = cuda.CUmemAccessDesc()
                access.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
                access.location.id = int(cu_device)
                access.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                _check(cuda.cuMemSetAccess(va, chunk_bytes, [access], 1))
                head = _read_word(va)
                if head != expected:
                    raise RuntimeError(
                        f"reads 0x{head:016x}, expected 0x{expected:016x} (stale or foreign handles)"
                    )
                if new is not None:
                    word = ctypes.c_uint64(new)
                    _check(cuda.cuMemcpyHtoD(va, ctypes.addressof(word), ctypes.sizeof(word)))
                    # Read back synchronously, so the word has landed before the chunk is unmapped.
                    stamped = _read_word(va)
                    if stamped != new:
                        raise RuntimeError(
                            f"re-stamp reads back 0x{stamped:016x}, wrote 0x{new:016x}"
                        )
            finally:
                _check(cuda.cuMemUnmap(va, chunk_bytes))
        finally:
            _check(cuda.cuMemAddressFree(va, chunk_bytes))
    finally:
        _check(cuda.cuMemRelease(handle))


def stamp_merged_chunks(
    payloads: Sequence[Dict[str, Any]],
    order: Sequence[Tuple[int, int]],
    chunk_bytes: int,
    verify_nonce: Optional[int],
    rank: int,
) -> None:
    """Check every chunk's lender word (``nonce_k ^ i``) and, unless ``verify_nonce`` is None, write
    ``verify_nonce ^ position``. An import or read error names the lender host and index."""
    cu_device = _current_device()
    for position, (index, chunk) in enumerate(order):
        payload = payloads[index]
        new = None if verify_nonce is None else verify_nonce ^ position
        try:
            _verify_and_stamp_chunk(
                bytes.fromhex(payload["handles"][chunk]),
                chunk_bytes,
                cu_device,
                int(payload["nonce"]) ^ chunk,
                new,
            )
        except RuntimeError as error:
            raise RuntimeError(
                f"{LOG_PREFIX} rank={rank}: {_describe(index, payload)} chunk {chunk} (merged "
                f"position {position}): {error}"
            ) from error


def import_lenders(config: TierConfig, local_rank: int, global_rank: int) -> MergedImport:
    """Wait for, validate, claim and verify (re-stamp with several) the lenders 0..N-1 of a rank.

    With ``TLLM_KV_REMOTE_HOST_PART`` (split lending) the same is done on one part of every lender:
    all parts are waited for and checked, then one part is claimed (``claim_part``).
    """
    global _MERGED
    assert config.num_lenders is not None
    num_lenders = config.num_lenders
    job = job_tag()
    me = Identity.of("importer", rank=global_rank, local_rank=local_rank)
    start = time.monotonic()
    part: Optional[int] = None
    num_parts: Optional[int] = None
    if config.part is None:
        payloads = wait_for_lenders(config.directory, local_rank, num_lenders, config.wait_s, job)
        waited_s = time.monotonic() - start
        # Nothing is claimed when a check fails, so the lenders stay free for a corrected importer.
        chunk_bytes = check_lenders(payloads, config, local_rank)
        claim_lenders(config.directory, local_rank, job, payloads, me)
    else:
        parts = wait_for_parts(config.directory, local_rank, num_lenders, config.wait_s, job)
        waited_s = time.monotonic() - start
        # Every part set is checked before any claim, as above.
        chunk_bytes = check_parts(parts, config, local_rank)
        part = claim_part(config.directory, local_rank, job, parts, me, config.part)
        num_parts = len(parts)
        payloads = list(parts[part])
    order = interleave_order([len(payload["handles"]) for payload in payloads])
    # One lender keeps its own words (list order = its chunk order); several get a fresh nonce.
    restamp = num_lenders > 1
    verify_nonce = int.from_bytes(os.urandom(8), "little") if restamp else int(payloads[0]["nonce"])
    stamp_start = time.monotonic()
    stamp_merged_chunks(payloads, order, chunk_bytes, verify_nonce if restamp else None, local_rank)
    stamp_s = time.monotonic() - stamp_start
    merged = MergedImport(
        rank=local_rank,
        global_rank=global_rank,
        chunk_bytes=chunk_bytes,
        verify_nonce=verify_nonce,
        handles=tuple(bytes.fromhex(payloads[k]["handles"][i]) for k, i in order),
        owners=tuple(k for k, _ in order),
        lenders=tuple(payload["lender"] for payload in payloads),
        part=part,
    )
    _MERGED = merged
    per_lender = "; ".join(
        f"k={index} host={payload['lender'].get('host')} pid={payload['lender'].get('pid')} "
        f"global_rank={payload['lender'].get('rank')} device={payload.get('device')} "
        f"numa={payload.get('numa')} chunks={len(payload['handles'])} "
        f"gib={len(payload['handles']) * chunk_bytes / _GIB:.2f}"
        + ("" if part is None else f" nonce=0x{int(payload['nonce']):016x}")
        for index, payload in enumerate(payloads)
    )
    split = "" if part is None else f"part={part} of {num_parts} "
    logger.info(
        f"{LOG_PREFIX} rank={local_rank} global_rank={global_rank}: importing "
        f"{len(merged.handles)} chunks of {chunk_bytes / _GIB:g} GiB "
        f"({merged.lent_bytes / _GIB:.2f} GiB) from {num_lenders} lenders: [{per_lender}] {split}"
        f"order=interleaved {'restamp' if restamp else 'verify'}_s={stamp_s:.1f} after "
        f"{waited_s:.1f} s wait"
    )
    return merged


def _build_merged_tier(config: TierConfig, local_rank: int, global_rank: int) -> Any:
    merged = import_lenders(config, local_rank, global_rank)
    from tensorrt_llm.runtime.kv_cache_manager_v2 import HostCacheTierConfig, RemoteHostTierConfig

    remote = RemoteHostTierConfig(list(merged.handles), merged.chunk_bytes, merged.verify_nonce)
    return HostCacheTierConfig(quota=merged.lent_bytes, remote=remote)


def _mapped(address) -> bool:
    """Whether a chunk is mapped at ``address`` (a failed retain means not mapped)."""
    cuda = _cuda_driver()
    err, handle = cuda.cuMemRetainAllocationHandle(address)
    if err != cuda.CUresult.CUDA_SUCCESS:
        return False
    _check(cuda.cuMemRelease(handle))
    return True


def log_lender_spread(
    ranges: Sequence[Tuple[int, int]], pool_groups: Sequence[Any]
) -> Optional[str]:
    """Log once, before the level is used, how many chunks of each lender every pool group maps.

    ``ranges`` are the remote level's pool address ranges (pool groups in order, their pools in
    order) and ``pool_groups`` its storage statistics (one per pool group, ``slot_sizes`` per
    pool). Each pool maps its chunks contiguously from its base, and chunk ``g`` of the merged list
    still reads ``V ^ g``, which names its lender. Does nothing in single-lender mode. Diagnostic
    only: a failed read, or statistics or ranges of an unexpected shape, log a warning and leave
    the tier as built.
    """
    merged = _MERGED
    if merged is None:
        return None
    num_lenders = len(merged.lenders)
    try:
        pools_per_group = [len(list(group.slot_sizes)) for group in pool_groups]
    except (AttributeError, TypeError) as error:
        logger.warning(
            f"{LOG_PREFIX} rank={merged.rank}: chunks per lender per pool group not logged: "
            f"unexpected storage statistics ({error!r})"
        )
        return None
    if sum(pools_per_group) != len(ranges):
        logger.warning(
            f"{LOG_PREFIX} rank={merged.rank}: {len(ranges)} pool ranges for pools "
            f"{pools_per_group}; chunks per lender per pool group not logged"
        )
        return None
    group_of = [group for group, count in enumerate(pools_per_group) for _ in range(count)]
    counts = [[0] * num_lenders for _ in pool_groups]
    seen = set()
    unknown = 0
    try:
        for (base, length), group in zip(ranges, group_of):
            for slot in range(int(length) // merged.chunk_bytes):
                address = int(base) + slot * merged.chunk_bytes
                if not _mapped(address):
                    break
                position = _read_word(address) ^ merged.verify_nonce
                if position < len(merged.owners) and position not in seen:
                    seen.add(position)
                    counts[group][merged.owners[position]] += 1
                else:
                    unknown += 1
    except (RuntimeError, TypeError, ValueError) as error:
        # RuntimeError: a CUDA read; TypeError / ValueError: a pool range of an unexpected shape.
        logger.warning(
            f"{LOG_PREFIX} rank={merged.rank}: chunks per lender per pool group not logged: {error}"
        )
        return None
    unmapped = [merged.owners.count(k) - sum(row[k] for row in counts) for k in range(num_lenders)]
    groups = " ".join(
        f"pg{group}=[" + ",".join(f"l{k}:{c}" for k, c in enumerate(row)) + "]"
        for group, row in enumerate(counts)
    )
    hosts = ",".join(f"l{k}={lender.get('host')}" for k, lender in enumerate(merged.lenders))
    line = (
        f"{LOG_PREFIX} rank={merged.rank} chunks per lender per pool group: {groups} "
        f"unmapped=[{','.join(f'l{k}:{c}' for k, c in enumerate(unmapped))}] "
        f"unknown={unknown} lenders=[{hosts}]"
    )
    logger.info(line)
    return line


def _rss_mb() -> int:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


class RemoteHostTierStats:
    """Logs the remote tier's counters at most every ``STATS_INTERVAL_S`` seconds.

    ``rss_gib_per_m_used`` is the growth of the process RSS per million remote slots filled since
    the first line (RSS and used slots are recorded then), so the pinned local host tier and the
    rest of the process do not enter it.
    """

    def __init__(
        self,
        rank: int,
        stats_fn: Callable[[], Dict[str, int]],
        storage_fn: Callable[[int], List[Any]],
        interval_s: float = STATS_INTERVAL_S,
        rss_fn: Optional[Callable[[], int]] = None,
    ) -> None:
        self.rank = rank
        self._stats_fn = stats_fn
        self._storage_fn = storage_fn
        self._interval_s = interval_s
        self._rss_fn = rss_fn if rss_fn is not None else _rss_mb
        self._last: Optional[float] = None
        self._base: Optional[Tuple[int, int]] = None  # (RSS MiB, used slots) at the first line

    def format(self) -> str:
        s = self._stats_fn()
        level = int(s["level"])
        pools = self._storage_fn(level) if level >= 0 else []
        total = sum(int(p.total) for p in pools)
        free = sum(int(p.free) for p in pools)
        used = total - free
        fill = used / total if total else 0.0
        rss = self._rss_fn()
        if self._base is None:
            self._base = (rss, used)
        rss_base, used_base = self._base
        used_delta = used - used_base
        # RSS growth per million remote slots filled: the bound the CTX local host tier is sized against.
        rss_per_m = (rss - rss_base) / 1024 / (used_delta / 1e6) if used_delta > 0 else 0.0
        return (
            f"{LOG_PREFIX} rank={self.rank} level={level} "
            f"lent_gib={int(s['lent_bytes']) / _GIB:.2f} chunks={s['num_chunks']} "
            f"slots={total} used={used} fill={fill:.4f} hit_tokens={s['hit_tokens']} "
            f"promoted_pages={s['promoted_pages']} promoted_gib={int(s['promoted_bytes']) / _GIB:.2f} "
            f"demoted_pages={s['demoted_pages']} demoted_gib={int(s['demoted_bytes']) / _GIB:.2f} "
            f"gpu_kernel_h2h={s['gpu_kernel_host_copies']} "
            f"host_fn_h2h={s['host_function_host_copies']} rss_mb={rss} rss_base_mb={rss_base} "
            f"used_base={used_base} rss_gib_per_m_used={rss_per_m:.2f}"
        )

    def maybe_log(self, now: float, force: bool = False) -> bool:
        if not force and self._last is not None and now - self._last < self._interval_s:
            return False
        self._last = now
        logger.info(self.format())
        return True
