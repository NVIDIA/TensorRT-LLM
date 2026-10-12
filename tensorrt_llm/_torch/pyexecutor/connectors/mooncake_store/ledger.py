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
"""What each rank put into the pool, recorded where telemetry can read it.

Pool capacity is the sum of the segments its participants mounted, and no
participant can see any other's, so each writes one small file under a
directory the deployment names. Recovering the same figures from logs would tie
the report to the wording of a log line.

A record is a declaration rather than a measurement: it says what this rank
asked the master to accept. Where the blocks landed is the master's to say, and
`format_pool_report` joins the two.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from tensorrt_llm.logger import logger

from .config import SEGMENTS_DIR_NAME

__all__ = [
    "SegmentRecord",
    "format_pool_report",
    "read_segments",
    "record_segment",
    "segments_dir",
]

_GIB = 1 << 30

#: The master's own glog line for a block placement. `segment` names the client
#: process that owns the memory the block landed in, as `host:port`.
_ALLOCATION_RE = re.compile(r"allocation_succeeded size=(\d+) segment=([0-9.]+):(\d+)")


@dataclass(frozen=True)
class SegmentRecord:
    """One rank's contribution to the pool."""

    #: Host the segment is registered under, as the master knows it.
    host: str
    #: Rank within its own server, so not unique across a job. The file name
    #: carries the process identity.
    rank: int
    segment_size: int
    role: str
    model_key: str

    @property
    def gib(self) -> float:
        return self.segment_size / _GIB


def segments_dir(run_dir: str) -> str:
    """Where a server whose run directory is `run_dir` writes its records."""
    return os.path.join(run_dir, SEGMENTS_DIR_NAME)


def record_segment(
    run_dir: Optional[str],
    *,
    host: str,
    rank: int,
    segment_size: int,
    role: str,
    model_key: str,
) -> Optional[str]:
    """Write this rank's contribution, and return where it went.

    A no-op returning `None` when the deployment named no run directory, since
    a per-process temporary one is nowhere anything else could read. The file
    name is keyed by host and pid rather than by rank, because every server
    numbers its ranks from zero. A failed write is logged and swallowed.
    """
    if not run_dir:
        return None
    directory = segments_dir(run_dir)
    path = os.path.join(directory, f"segment-{host}-{os.getpid()}.json")
    record = {
        "host": host,
        "rank": rank,
        "segment_size": segment_size,
        "role": role,
        "model_key": model_key,
    }
    try:
        os.makedirs(directory, exist_ok=True)
        # Renamed into place so a reader never sees half a record.
        staging = f"{path}.partial"
        with open(staging, "w") as handle:
            json.dump(record, handle)
        os.replace(staging, path)
    except OSError as exc:
        logger.warning(
            f"mooncake-store: could not record this rank's segment in {path}: "
            f"{exc}. The pool is unaffected, but its capacity will not appear "
            "in the run's summary."
        )
        return None
    return path


def _record_paths(run_dir: str) -> List[str]:
    """Every segment record anywhere beneath `run_dir`.

    The whole tree rather than one directory in it, since each server needs a
    run directory of its own. The pool's directory is their common ancestor
    rather than their common directory.
    """
    paths: List[str] = []
    for parent, directories, names in os.walk(run_dir):
        if os.path.basename(parent) != SEGMENTS_DIR_NAME:
            continue
        # Records are files; nothing below one of these is a record.
        directories[:] = []
        paths.extend(os.path.join(parent, name) for name in names if name.endswith(".json"))
    return sorted(paths)


def read_segments(run_dir: str) -> List[SegmentRecord]:
    """Every segment recorded under `run_dir`, in host then rank order."""
    records: List[SegmentRecord] = []
    for path in _record_paths(run_dir):
        try:
            with open(path) as handle:
                raw = json.load(handle)
            records.append(
                SegmentRecord(
                    host=str(raw["host"]),
                    rank=int(raw["rank"]),
                    segment_size=int(raw["segment_size"]),
                    role=str(raw["role"]),
                    model_key=str(raw.get("model_key", "")),
                )
            )
        except (OSError, ValueError, KeyError) as exc:
            logger.warning(f"mooncake-store: skipping unreadable segment record {path}: {exc}")
    records.sort(key=lambda record: (record.host, record.rank))
    return records


def _roles_by_host(records: Sequence[SegmentRecord]) -> Dict[str, str]:
    """Which roles each host lent memory under.

    The join key for the master's placement table. A host normally runs one
    kind of server, so the value is normally a single role.
    """
    roles: Dict[str, set] = {}
    for record in records:
        roles.setdefault(record.host, set()).add(record.role)
    return {host: "+".join(sorted(found)) for host, found in roles.items()}


def _capacity_lines(records: Sequence[SegmentRecord]) -> List[str]:
    """Total capacity, how it splits by role, and whether it is uniform."""
    total = sum(record.segment_size for record in records)
    lines = [
        f"ranks         : {len(records)} across {len({r.host for r in records})} host(s)",
        f"capacity      : {total / _GIB:.1f} GiB",
    ]

    by_role: Dict[str, List[SegmentRecord]] = {}
    for record in records:
        by_role.setdefault(record.role, []).append(record)
    for role in sorted(by_role):
        held = by_role[role]
        share = 100 * sum(r.segment_size for r in held) / total if total else 0.0
        lines.append(
            f"  role {role:<9}: {len(held):>3} rank(s), "
            f"{sum(r.segment_size for r in held) / _GIB:>8.1f} GiB  {share:5.1f}%"
        )

    # Each server names its own segment_size and can see no other's, so this
    # is the only vantage point a mismatch is visible from.
    sizes = sorted({record.segment_size for record in records})
    if len(sizes) == 1:
        lines.append(f"per rank      : {sizes[0] / _GIB:.1f} GiB, uniform")
    else:
        spelled = ", ".join(f"{size / _GIB:.1f} GiB" for size in sizes)
        lines.append(
            f"per rank      : NOT UNIFORM. {len(sizes)} distinct sizes ({spelled})."
            " Contribution is meant to be uniform per rank; check that every"
            " worker config names the same segment_size."
        )
    return lines


def _placement_lines(master_log: str, records: Sequence[SegmentRecord]) -> List[str]:
    """Where blocks actually landed, by host, labelled with each host's role.

    A segment is one client process's memory, so grouping the master's
    allocations by host shows how much of the pool's contents live on a decode
    node rather than on the prefill node that computed them.
    """
    try:
        with open(master_log, errors="replace") as handle:
            text = handle.read()
    except OSError as exc:
        return [f"(master log at {master_log} could not be read: {exc})"]

    pages: Dict[str, int] = {}
    placed: Dict[str, int] = {}
    ports: Dict[str, set] = {}
    for size, host, port in _ALLOCATION_RE.findall(text):
        pages[host] = pages.get(host, 0) + 1
        placed[host] = placed.get(host, 0) + int(size)
        ports.setdefault(host, set()).add(port)

    if not pages:
        return ["(no allocations in the master log)"]

    roles = _roles_by_host(records)
    total = sum(placed.values())
    lines = []
    for host in sorted(placed, key=lambda h: -placed[h]):
        lines.append(
            f"  {host:<16} {roles.get(host, 'unknown role'):<9} "
            f"pages={pages[host]:<9} {placed[host] / _GIB:>8.2f} GiB "
            f"{100 * placed[host] / total:>5.1f}%  "
            f"segments: {','.join(sorted(ports[host]))}"
        )
    lines.append(f"  {'TOTAL':<16} {'':<9} pages={sum(pages.values()):<9} {total / _GIB:>8.2f} GiB")

    off_producer = sum(
        byte_count
        for host, byte_count in placed.items()
        if roles.get(host) and "both" not in roles[host] and "producer" not in roles[host]
    )
    if total:
        lines.append(
            f"  held off the producing side: {100 * off_producer / total:.1f}% "
            f"({off_producer / _GIB:.1f} GiB)"
        )
    return lines


def format_pool_report(run_dir: str, master_log: Optional[str] = None) -> str:
    """A human-readable account of the pool a run stood up.

    Args:
        run_dir: The pool's directory, holding `pool.json`. Segment records are
            read from the tree beneath it.
        master_log: The master's log, for the placement table. Defaults to
            `mooncake_master.log` in `run_dir` when that exists.
    """
    lines: List[str] = []

    manifest_path = os.path.join(run_dir, "pool.json")
    try:
        with open(manifest_path) as handle:
            manifest = json.load(handle)
        lines.append(f"master        : {manifest.get('master_server_address', 'unknown')}")
        lines.append(f"transport     : {manifest.get('protocol', 'unknown')}")
        lines.append(f"metadata      : {manifest.get('metadata_server', 'unknown')}")
    except (OSError, ValueError) as exc:
        lines.append(f"master        : unknown ({manifest_path}: {exc})")

    records = read_segments(run_dir)
    if not records:
        lines.append(
            "capacity      : nothing recorded. No rank mounted a segment, so "
            "the pool was empty and no lookup could hit. Check that a worker "
            "config sets kv_connector_config.mooncake_store."
        )
        return "\n".join(lines)

    lines.extend(_capacity_lines(records))

    if master_log is None:
        candidate = os.path.join(run_dir, "mooncake_master.log")
        master_log = candidate if os.path.exists(candidate) else None
    if master_log:
        lines.append("")
        lines.append("block placement by segment host:")
        lines.extend(_placement_lines(master_log, records))

    return "\n".join(lines)
