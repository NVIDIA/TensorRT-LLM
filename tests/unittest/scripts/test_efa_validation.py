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
validate_efa_worker_log = _MODULE.validate_efa_worker_log

pytestmark = pytest.mark.cpu_only

_STARTUP = """NixlTransferAgent::NixlTransferAgent using NIXL backend: LIBFABRIC
KV cache manager v2 device quota set to 8GiB
libfabric:123:1700000000::efa:domain:efa_domain_open():100<info> domain initialized
"""


def transfer(request_id: int, size: int) -> str:
    return (
        f"KVRecvTask.print_perf_info: unique_rid={request_id}, peer_rank=0, "
        f"transfer_size={size} byte, entry_count=1\n"
    )


def test_accepts_efa_data_transfers() -> None:
    validate_efa_worker_log(_STARTUP + transfer(1, 4096) + transfer(2, 8192), received_requests=2)


@pytest.mark.parametrize(
    "log",
    [
        _STARTUP.replace("backend: LIBFABRIC", "backend: UCX") + transfer(1, 4096),
        _STARTUP.replace(":efa:", ":tcp:") + transfer(1, 4096),
        _STARTUP.replace(":efa:", ":tcp:").replace(
            "domain initialized", "another provider :efa: was advertised"
        )
        + transfer(1, 4096),
        _STARTUP.replace("KV cache manager v2 device quota set", "KV cache manager v1")
        + transfer(1, 4096),
        _STARTUP + transfer(1, 0),
        _STARTUP,
    ],
)
def test_rejects_missing_transport_evidence(log: str) -> None:
    with pytest.raises(AssertionError):
        validate_efa_worker_log(log, received_requests=1)


def test_duplicate_records_do_not_count_as_multiple_requests() -> None:
    with pytest.raises(AssertionError, match="found 1"):
        validate_efa_worker_log(_STARTUP + transfer(1, 4096) * 2, received_requests=2)
