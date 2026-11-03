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

"""Positive runtime evidence required by the EFA disaggregated CI test."""

import re


def validate_efa_worker_logs(
    context_log: str, generation_log: str, *, received_requests: int
) -> None:
    """Reject backend fallback, a KV cache V1 run, or missing data transfers."""
    for log in (context_log, generation_log):
        assert "using NIXL backend: LIBFABRIC" in log, "NIXL LIBFABRIC was not initialized"
        assert "KV cache manager v2 device quota set" in log, "KV cache V2 was not initialized"
        # Require the actual worker's provider log, not the separate fi_info probe.
        assert re.search(r"libfabric:\d+:\d+:[^:\n]*:efa:", log), (
            "No EFA provider activity in worker log"
        )
    # The Python transceiver logs byte counts on the sender and completion
    # latency on the receiver. Both records carry the same unique request ID.
    sent = {
        request_id
        for request_id, size in re.findall(
            r"KVSendTask\.print_perf_info: unique_rid=(\d+),[^\n]*?transfer_size=(\d+) byte",
            context_log,
        )
        if int(size) > 0
    }
    received = set(
        re.findall(
            r"KVRecvTask\.print_perf_info: unique_rid=(\d+),[^\n]*?task_latency=\d+(?:\.\d+)? ms",
            generation_log,
        )
    )
    transfers = sent & received
    assert len(transfers) >= received_requests, (
        f"Expected {received_requests} requests with nonempty KV transfers, found {len(transfers)}"
    )
