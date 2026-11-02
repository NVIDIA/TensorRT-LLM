# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
        f"transfer_size={size} byte, entry_count=71, task_latency=172.041 ms\n"
    )


def receive(request_id: int) -> str:
    return (
        f"KVRecvTask.print_perf_info: unique_rid={request_id}, peer_rank=0, "
        "task_latency=354.391 ms\n"
    )


def test_accepts_efa_data_transfers() -> None:
    # Receiver records intentionally have no byte count, as in the runtime logs.
    validate_efa_worker_logs(
        _STARTUP + send(49127173691932674, 260571136) + send(49127173691932675, 260571136),
        _STARTUP + receive(49127173691932674) + receive(49127173691932675),
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
