# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for disaggregated E2E timing assertions."""

import pytest
from test_common.perf_metrics_utils import validate_timing_metrics

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def perf_metrics_item() -> dict:
    return {
        "ctx_server": "context",
        "gen_server": "generation",
        "disagg_server_arrival_time": 1.0,
        "disagg_server_first_token_time": 2.6,
        "ctx_perf_metrics": {
            "ctx_request_id": 42,
            "perf_metrics": {
                "timing_metrics": {
                    "server_arrival_time": 1.1,
                    "server_first_token_time": 2.0,
                    "arrival_time": 1.2,
                    "first_token_time": 1.9,
                    "last_token_time": 2.0,
                }
            },
        },
        "gen_perf_metrics": {
            "ctx_request_id": 42,
            "perf_metrics": {
                "timing_metrics": {
                    "server_arrival_time": 2.1,
                    "server_first_token_time": 2.5,
                    "arrival_time": 2.1,
                    "first_scheduled_time": 2.4,
                    "first_token_time": 2.5,
                    "kv_cache_transfer_start": 2.2,
                    "kv_cache_transfer_end": 2.3,
                }
            },
        },
    }


@pytest.mark.parametrize("size", [None, 0, 8192], ids=["omitted", "zero", "positive"])
@pytest.mark.parametrize("populated_timestamps", [False, True])
def test_optional_transfer_metrics(
    perf_metrics_item: dict, size: int | None, populated_timestamps: bool
) -> None:
    timing = perf_metrics_item["gen_perf_metrics"]["perf_metrics"]["timing_metrics"]
    if size is not None:
        timing["kv_cache_size"] = size
    if not populated_timestamps:
        timing.pop("kv_cache_transfer_start")
        timing.pop("kv_cache_transfer_end")
    assert validate_timing_metrics(perf_metrics_item)


@pytest.mark.parametrize("populated_timestamps", [False, True])
def test_negative_transfer_size_is_rejected(
    perf_metrics_item: dict, populated_timestamps: bool
) -> None:
    timing = perf_metrics_item["gen_perf_metrics"]["perf_metrics"]["timing_metrics"]
    timing["kv_cache_size"] = -1
    if not populated_timestamps:
        timing.pop("kv_cache_transfer_start")
        timing.pop("kv_cache_transfer_end")
    with pytest.raises(AssertionError, match="negative kv_cache_size"):
        validate_timing_metrics(perf_metrics_item)


@pytest.mark.parametrize("size", [None, 0, 8192], ids=["omitted", "zero", "positive"])
@pytest.mark.parametrize("start,end", [(2.3, 2.2), (2.0, 2.3), (2.2, 2.5)])
def test_transfer_timing_order_is_checked_without_positive_size(
    perf_metrics_item: dict, size: int | None, start: float, end: float
) -> None:
    timing = perf_metrics_item["gen_perf_metrics"]["perf_metrics"]["timing_metrics"]
    if size is not None:
        timing["kv_cache_size"] = size
    timing.update(kv_cache_transfer_start=start, kv_cache_transfer_end=end)
    with pytest.raises(AssertionError):
        validate_timing_metrics(perf_metrics_item)
