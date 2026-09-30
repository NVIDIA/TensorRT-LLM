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

The ModelExpress strategy chain logs one `[Worker <rank>] ... RDMA transfer
complete: <n> tensors, <x> GB` record per receiver rank when the shard arrived
through P2P. Those records in the receiver's log, together with the absence of
any fallback marker, are the proof that the receiver did not silently load from
disk. This module holds the single definition of that evidence so every MX
qualification test applies identical rules.

It is importable both as `defs.model_express.mx_evidence` from pytest and as a
bare `mx_evidence` from a worker script whose directory is on `sys.path`. Keep
it free of third-party imports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

RECEIVER_FAILURE_MARKERS = (
    "falling back to disk",
    "falling back to native hugging face checkpoint loading",
    "partial fallback",
    "size mismatch",
    "still missing",
    "mx p2p transfer failed",
    "mx p2p unavailable",
    "source sourceidentity incompatible",
    "sourceidentity mismatch",
    "invalid sourceidentity",
)
RDMA_TRANSFER_PATTERN = re.compile(
    r"\[Worker\s+(\d+)\].*?RDMA transfer complete:\s+(\d+)\s+tensors,\s+([0-9.]+)\s+GB",
    re.IGNORECASE,
)
DONOR_PROCESS_FAILURE_MARKERS = (
    b"Segfault encountered",
    b"Primary job terminated normally, but",
    b"process returned a non-zero exit code",
)
DONOR_PROCESS_FAILURE_OVERLAP = max(len(marker) for marker in DONOR_PROCESS_FAILURE_MARKERS) - 1


@dataclass(frozen=True)
class RdmaTransfer:
    """One `RDMA transfer complete` record of a receiver rank.

    Attributes:
        rank: Tensor-parallel rank that logged the record.
        tensor_count: Number of tensors the rank received.
        size_gb: Total bytes received, in GB as logged by ModelExpress.
    """

    rank: int
    tensor_count: int
    size_gb: float

    def to_dict(self) -> dict[str, object]:
        return {"rank": self.rank, "tensor_count": self.tensor_count, "size_gb": self.size_gb}


def find_failure_markers(text: str) -> tuple[str, ...]:
    """Return the failure markers present in `text`.

    Args:
        text: Log text to scan; matching is case-insensitive.

    Returns:
        The markers from `RECEIVER_FAILURE_MARKERS` found in `text`, in list order.
    """
    lowered = text.lower()
    return tuple(marker for marker in RECEIVER_FAILURE_MARKERS if marker in lowered)


def rdma_transfers_by_rank(text: str) -> dict[int, tuple[RdmaTransfer, ...]]:
    """Group every `RDMA transfer complete` record in `text` by rank.

    Args:
        text: The receiver's log (stdout and stderr of every rank).

    Returns:
        A dict from rank to its records in log order; ranks without a record
        are absent.
    """
    transfers: dict[int, list[RdmaTransfer]] = {}
    for rank, tensor_count, size_gb in RDMA_TRANSFER_PATTERN.findall(text):
        transfers.setdefault(int(rank), []).append(
            RdmaTransfer(rank=int(rank), tensor_count=int(tensor_count), size_gb=float(size_gb))
        )
    return {rank: tuple(records) for rank, records in transfers.items()}


def check_receiver_log(text: str, tp_size: int) -> list[str]:
    """Return the problems with a receiver's transfer evidence (empty means pass).

    Rules: exactly ranks `0..tp_size-1` logged an RDMA completion; the log
    contains no failure marker; each rank logged exactly one completion, and
    that completion moved at least one tensor and more than zero GB.

    Args:
        text: The receiver's log.
        tp_size: Expected number of ranks.

    Returns:
        Human-readable problem descriptions; an empty list means the evidence
        is complete.
    """
    problems: list[str] = []
    expected_ranks = set(range(tp_size))
    transfers = rdma_transfers_by_rank(text)
    if set(transfers) != expected_ranks:
        problems.append(
            f"Expected RDMA transfer completion for ranks {sorted(expected_ranks)}, "
            f"got {sorted(transfers)}"
        )

    for marker in find_failure_markers(text):
        problems.append(f"MX receiver log contains failure marker {marker!r}")

    for rank in sorted(expected_ranks & set(transfers)):
        records = transfers[rank]
        if len(records) != 1:
            problems.append(
                f"Expected one RDMA transfer completion for rank {rank}, "
                f"got {[record.to_dict() for record in records]}"
            )
            continue
        record = records[0]
        if record.tensor_count <= 0 or record.size_gb <= 0:
            problems.append(
                f"MX receiver rank {rank} reported an empty transfer: "
                f"{record.tensor_count} tensors, {record.size_gb} GB"
            )
    return problems


__all__ = [
    "DONOR_PROCESS_FAILURE_MARKERS",
    "DONOR_PROCESS_FAILURE_OVERLAP",
    "RDMA_TRANSFER_PATTERN",
    "RECEIVER_FAILURE_MARKERS",
    "RdmaTransfer",
    "check_receiver_log",
    "find_failure_markers",
    "rdma_transfers_by_rank",
]
