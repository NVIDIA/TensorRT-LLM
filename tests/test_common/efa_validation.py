# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Positive runtime evidence required by the EFA disaggregated CI test."""

import re


def validate_efa_worker_log(log: str, *, received_requests: int = 0) -> None:
    """Reject backend fallback, a KV cache V1 run, or missing data transfers."""
    assert "using NIXL backend: LIBFABRIC" in log, "NIXL LIBFABRIC was not initialized"
    assert "KV cache manager v2 device quota set" in log, "KV cache V2 was not initialized"
    # libfabric's log prefix includes the provider name. Require evidence from
    # the actual worker, rather than the separate fi_info prerequisite probe.
    assert re.search(r"libfabric:\d+:\d+:[^:\n]*:efa:", log), (
        "No EFA provider activity in worker log"
    )
    transfers = {
        request_id
        for request_id, size in re.findall(
            r"KVRecvTask\.print_perf_info: unique_rid=(\d+),[^\n]*?transfer_size=(\d+) byte",
            log,
        )
        if int(size) > 0
    }
    assert len(transfers) >= received_requests, (
        f"Expected {received_requests} requests with nonempty KV transfers, found {len(transfers)}"
    )
