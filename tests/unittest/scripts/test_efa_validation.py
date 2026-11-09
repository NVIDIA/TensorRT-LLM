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

"""Keep the EFA functional test from accepting fallback or metadata-only runs."""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "test_common" / "efa_validation.py"
_SPEC = importlib.util.spec_from_file_location("efa_validation", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
validate_efa_worker_logs = _MODULE.validate_efa_worker_logs

pytestmark = pytest.mark.cpu_only

_STARTUP = """NixlTransferAgent::NixlTransferAgent using NIXL backend: LIBFABRIC
KV cache manager v2 device quota set to 8GiB
libfabric:123:1700000000::efa:domain:efa_domain_open():100<info> domain initialized
"""


def send(request_id: int, size: int) -> str:
    return (
        f"KVSendTask.print_perf_info: unique_rid={request_id}, peer_rank=0, "
        f"transfer_size={size} byte, entry_count=1, task_latency=1.0 ms\n"
    )


def receive(request_id: int) -> str:
    return (
        f"KVRecvTask.print_perf_info: unique_rid={request_id}, peer_rank=0, task_latency=1.0 ms\n"
    )


def test_accepts_efa_data_transfers() -> None:
    # Receiver records intentionally have no byte count, as in the runtime logs.
    validate_efa_worker_logs(
        _STARTUP + send(1, 4096) + send(2, 8192),
        _STARTUP + receive(1) + receive(2),
        received_requests=2,
    )


@pytest.mark.parametrize("worker", ["context", "generation"])
@pytest.mark.parametrize(
    "startup",
    [
        _STARTUP.replace("backend: LIBFABRIC", "backend: UCX"),
        _STARTUP.replace(":efa:", ":tcp:"),
        _STARTUP.replace(":efa:", ":tcp:").replace(
            "domain initialized", "another provider :efa: was advertised"
        ),
        _STARTUP.replace("KV cache manager v2 device quota set", "KV cache manager v1"),
    ],
)
def test_rejects_backend_fallback(worker: str, startup: str) -> None:
    context = startup if worker == "context" else _STARTUP
    generation = startup if worker == "generation" else _STARTUP
    with pytest.raises(AssertionError):
        validate_efa_worker_logs(
            context + send(1, 4096), generation + receive(1), received_requests=1
        )


@pytest.mark.parametrize(
    ("sent", "received"),
    [
        (send(1, 0), receive(1)),
        ("", receive(1)),
        (send(1, 4096), ""),
        (send(1, 4096), receive(2)),
    ],
)
def test_rejects_incomplete_transfers(sent: str, received: str) -> None:
    with pytest.raises(AssertionError, match="found 0"):
        validate_efa_worker_logs(_STARTUP + sent, _STARTUP + received, received_requests=1)


def test_duplicate_records_do_not_count_as_multiple_requests() -> None:
    with pytest.raises(AssertionError, match="found 1"):
        validate_efa_worker_logs(
            _STARTUP + send(1, 4096) * 2, _STARTUP + receive(1) * 2, received_requests=2
        )
