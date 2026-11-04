# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observe the checkpoint filesystem's Linux block-device counters, not file size."""

import os
import time
from pathlib import Path
from typing import Any


def sample_device(path: Path, diskstats: Path = Path("/proc/diskstats")) -> dict[str, Any]:
    """Read the device containing the checkpoint directory without flushing caches.

    Args:
        path: Existing host checkpoint directory.
        diskstats: Linux disk-statistics source; injectable for CPU tests.

    Returns:
        Cumulative counters or explicit unavailability. Network/overlay mounts
        without a matching block device do not supply physical-I/O evidence.
    """
    sampled: dict[str, Any] = {"sample_seconds": time.monotonic()}
    try:
        device = path.stat().st_dev
        identity = [os.major(device), os.minor(device)]
        sampled["device_id"] = identity
        for line in diskstats.read_text().splitlines():
            fields = line.split()
            if len(fields) < 3 or [int(fields[0]), int(fields[1])] != identity:
                continue
            if len(fields) < 14:
                raise ValueError("Device does not expose the required diskstats fields")
            counters = {
                "reads": int(fields[3]),
                "read_bytes": int(fields[5]) * 512,
                "writes": int(fields[7]),
                "write_bytes": int(fields[9]) * 512,
            }
            if any(value < 0 for value in counters.values()):
                raise ValueError("Negative cumulative diskstats counter")
            return {
                **sampled,
                "status": "AVAILABLE",
                "device_name": fields[2],
                "counters": counters,
            }
        reason = "Checkpoint filesystem has no matching Linux block-device counters"
    except (OSError, ValueError) as error:
        reason = str(error)
    return {**sampled, "status": "UNAVAILABLE", "reason": reason}


def device_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Compare same-device counters without attributing unrelated host I/O to restore.

    Args:
        before: Sample immediately before a host operation.
        after: Sample immediately after the operation, before helper cleanup.

    Returns:
        Device-wide deltas with attribution limits, or an unavailable result on
        missing counters, a device change or a counter reset. Cached reads and
        deferred writeback need not reach the device inside this interval.
    """
    result = {
        "before": before,
        "after": after,
        "scope": "checkpoint-filesystem block device; all host users",
        "attribution": "NOT_PROCESS_OR_ARTIFACT_EXCLUSIVE",
        "cache_policy": "unchanged; no sync or cache eviction",
    }
    if before["status"] != "AVAILABLE" or after["status"] != "AVAILABLE":
        return {**result, "status": "UNAVAILABLE", "reason": "Missing device sample"}
    if any(before[key] != after[key] for key in ("device_id", "device_name")):
        return {**result, "status": "UNAVAILABLE", "reason": "Device identity changed"}
    deltas = {key: after["counters"][key] - value for key, value in before["counters"].items()}
    if any(value < 0 for value in deltas.values()):
        return {**result, "status": "UNAVAILABLE", "reason": "Device counters reset"}
    return {
        **result,
        "status": "MEASURED",
        "interval_seconds": after["sample_seconds"] - before["sample_seconds"],
        "delta": deltas,
    }
