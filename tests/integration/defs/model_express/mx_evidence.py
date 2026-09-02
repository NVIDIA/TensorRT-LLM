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
"""Shared ModelExpress transfer-evidence rules (standard library only).

The MX receiver's per-rank transfer logs (`rank<N>.log` under
`MX_TRANSFER_LOG_DIR`, written by the upstream `modelexpress` client) are the
proof that weights arrived through P2P rather than through the silent disk
fallback. This module holds the single definition of what those logs must
contain so the pytest orchestrator and the worker script apply identical rules.

It is importable both as `defs.model_express.mx_evidence` from pytest and as a
bare `mx_evidence` from `mx_e2e_worker.py`, which runs as a script with its own
directory on `sys.path`. Keep it free of third-party imports.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

RECEIVER_FAILURE_MARKERS = (
    "falling back to disk",
    "partial fallback",
    "size mismatch",
    "still missing",
    "mx p2p transfer failed",
    "mx p2p unavailable",
    "source sourceidentity incompatible",
    "sourceidentity mismatch",
    "invalid sourceidentity",
)
MATCHED_PARAMS_PATTERN = re.compile(r"Matched\s+(\d+)/(\d+)\s+params", re.IGNORECASE)
RANK_LOG_PATTERN = re.compile(r"rank(\d+)\.log", re.IGNORECASE)
TRANSFERRED_PARAMS_PATTERN = re.compile(
    r"Rank\s+(\d+):\s+transferred\s+(\d+)\s+params",
    re.IGNORECASE,
)
DONOR_PROCESS_FAILURE_MARKERS = (
    b"Segfault encountered",
    b"Primary job terminated normally, but",
    b"process returned a non-zero exit code",
)
DONOR_PROCESS_FAILURE_OVERLAP = max(len(marker) for marker in DONOR_PROCESS_FAILURE_MARKERS) - 1


@dataclass(frozen=True)
class RankTransferSummary:
    """What one receiver rank log says about the transfer."""

    rank: int
    matched_summaries: tuple[tuple[int, int], ...]
    transfer_summaries: tuple[tuple[int, int], ...]
    failure_markers: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "matched_summaries": [list(item) for item in self.matched_summaries],
            "transfer_summaries": [list(item) for item in self.transfer_summaries],
            "failure_markers": list(self.failure_markers),
        }


def find_failure_markers(text: str) -> tuple[str, ...]:
    """Return the failure markers present in `text` (case-insensitive)."""
    lowered = text.lower()
    return tuple(marker for marker in RECEIVER_FAILURE_MARKERS if marker in lowered)


def transfer_logs_by_rank(transfer_log_dir: Path) -> dict[int, str]:
    """Map rank -> log text for the non-empty `rank<N>.log` files in a directory.

    Raises `ValueError` when no non-empty log exists, when a file does not
    follow the `rank<N>.log` naming, or when a rank appears twice.
    """
    log_files = tuple(path for path in Path(transfer_log_dir).rglob("*") if path.is_file())
    logs = tuple(path for path in log_files if path.stat().st_size > 0)
    if not logs:
        entries = ", ".join(
            f"{path.relative_to(transfer_log_dir)} ({path.stat().st_size} bytes)"
            for path in log_files
        )
        raise ValueError(
            "ModelExpress created no non-empty receiver transfer logs in "
            f"{transfer_log_dir}; entries: {entries or '<none>'}"
        )

    logs_by_rank: dict[int, str] = {}
    for path in sorted(logs):
        match = RANK_LOG_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected ModelExpress receiver transfer log: {path}")
        rank = int(match.group(1))
        if rank in logs_by_rank:
            raise ValueError(
                f"ModelExpress created multiple receiver transfer logs for rank {rank}"
            )
        logs_by_rank[rank] = path.read_text(encoding="utf-8", errors="replace")
    return logs_by_rank


def summarize_rank_log(rank: int, text: str) -> RankTransferSummary:
    """Extract the matched/transferred summaries and failure markers of one rank log."""
    matched = tuple(
        (int(found), int(total)) for found, total in MATCHED_PARAMS_PATTERN.findall(text)
    )
    transferred = tuple(
        (int(found_rank), int(count))
        for found_rank, count in TRANSFERRED_PARAMS_PATTERN.findall(text)
    )
    return RankTransferSummary(
        rank=rank,
        matched_summaries=matched,
        transfer_summaries=transferred,
        failure_markers=find_failure_markers(text),
    )


def summarize_transfer_logs(transfer_log_dir: Path) -> dict[int, RankTransferSummary]:
    """Summarize every rank log in `transfer_log_dir`."""
    return {
        rank: summarize_rank_log(rank, text)
        for rank, text in transfer_logs_by_rank(transfer_log_dir).items()
    }


def check_receiver_transfer_logs(
    rank_logs: Mapping[int, str], tp_size: int, extra_text: str = ""
) -> list[str]:
    """Return the list of problems with a receiver's transfer evidence (empty = pass).

    Rules: exactly ranks `0..tp_size-1` are present; neither the rank logs nor
    `extra_text` (typically the receiver's stdout) contains a failure marker;
    each rank log has exactly one `Matched m/n params` summary with `m == n > 0`
    and exactly one `Rank r: transferred k params` summary with `r == rank` and
    `k == m`.
    """
    problems: list[str] = []
    expected_ranks = set(range(tp_size))
    if set(rank_logs) != expected_ranks:
        problems.append(
            f"Expected receiver transfer logs for ranks {sorted(expected_ranks)}, "
            f"got {sorted(rank_logs)}"
        )

    combined = extra_text + "\n" + "\n".join(rank_logs[rank] for rank in sorted(rank_logs))
    for marker in find_failure_markers(combined):
        problems.append(f"MX receiver logs contain failure marker {marker!r}")

    for rank in sorted(expected_ranks & set(rank_logs)):
        summary = summarize_rank_log(rank, rank_logs[rank])
        matched: int | None = None
        if len(summary.matched_summaries) != 1:
            problems.append(
                f"Expected one matched-parameter summary for rank {rank}, "
                f"got {list(summary.matched_summaries)}"
            )
        else:
            matched, total = summary.matched_summaries[0]
            if not (matched == total > 0):
                problems.append(
                    f"MX receiver rank {rank} reported incomplete parameter match {matched}/{total}"
                )
        if len(summary.transfer_summaries) != 1:
            problems.append(
                f"Expected one transfer summary for rank {rank}, "
                f"got {list(summary.transfer_summaries)}"
            )
        else:
            transferred_rank, transferred_count = summary.transfer_summaries[0]
            if transferred_rank != rank or (matched is not None and transferred_count != matched):
                problems.append(
                    f"MX receiver rank {rank} matched {matched} params but reported transfer "
                    f"summary {list(summary.transfer_summaries[0])}"
                )
    return problems


__all__ = [
    "DONOR_PROCESS_FAILURE_MARKERS",
    "DONOR_PROCESS_FAILURE_OVERLAP",
    "MATCHED_PARAMS_PATTERN",
    "RANK_LOG_PATTERN",
    "RECEIVER_FAILURE_MARKERS",
    "TRANSFERRED_PARAMS_PATTERN",
    "RankTransferSummary",
    "check_receiver_transfer_logs",
    "find_failure_markers",
    "summarize_rank_log",
    "summarize_transfer_logs",
    "transfer_logs_by_rank",
]
