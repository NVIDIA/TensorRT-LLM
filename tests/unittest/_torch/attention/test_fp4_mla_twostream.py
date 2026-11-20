# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side tests of the two-stream FP4 MLA decode planner (no GPU, no torch)."""

import pytest

from tensorrt_llm._torch.attention.backends.fp4_mla import fp4_mla_twostream as ts

SM_COUNT = 212  # 106 TPCs: 46 four-CTA clusters + 14 two-CTA pairs
ROWS_Q4 = 128 * 4


@pytest.fixture(autouse=True)
def _internal_kernel_calibration(monkeypatch):
    """The shapes below were measured with the internal mufu16 2x kernel (T2 scale 1.0); the main tree ships a slower
    packed path (DEFAULT_T2_SCALE 1.2), so pin the calibration."""
    monkeypatch.setenv(ts.TWOSTREAM_T2_SCALE_ENV, "1.0")
    monkeypatch.delenv(ts.TWOSTREAM_R2_ENV, raising=False)
    monkeypatch.delenv(ts.TWOSTREAM_ENV, raising=False)
    monkeypatch.delenv(ts.TWOSTREAM_MIN_GAIN_ENV, raising=False)
    monkeypatch.delenv(ts.TWOSTREAM_MAX_R2_ENV, raising=False)


def _plan(num_requests, rows, max_kv_len, **kw):
    return ts.plan_two_stream(
        num_requests=num_requests,
        rows_per_request=rows,
        max_kv_len=max_kv_len,
        sm_count=SM_COUNT,
        **kw,
    )


@pytest.mark.parametrize("max_kv_len", [30080, 16384, 8192])
def test_b128_q4_moves_thirteen_requests(max_kv_len):
    plan = _plan(128, ROWS_Q4, max_kv_len)
    # 256 4x items = 5.57 waves of 46; the smallest r2 that brings the 4x arm to 5 whole waves is 13 (230 items)
    assert plan.r2 == 13, plan
    assert plan.waves_4x_plain == 6 and plan.waves_4x == 5 and plan.rounds_2x == 4
    assert plan.num_requests_4x == 115
    assert plan.model_us < plan.model_plain_us * (1 - ts.DEFAULT_MIN_GAIN)


def test_forced_r2_and_disable(monkeypatch):
    monkeypatch.setenv(ts.TWOSTREAM_R2_ENV, "16")
    assert _plan(128, ROWS_Q4, 30080).r2 == 16
    monkeypatch.setenv(ts.TWOSTREAM_R2_ENV, "0")
    assert _plan(128, ROWS_Q4, 30080).r2 == 0
    monkeypatch.setenv(ts.TWOSTREAM_R2_ENV, "500")
    assert _plan(128, ROWS_Q4, 30080).r2 == 127  # never all requests
    monkeypatch.delenv(ts.TWOSTREAM_R2_ENV)
    monkeypatch.setenv(ts.TWOSTREAM_ENV, "0")
    plan = _plan(128, ROWS_Q4, 30080)
    assert plan.r2 == 0 and "disabled" in plan.reason


def test_min_gain_knob(monkeypatch):
    monkeypatch.setenv(ts.TWOSTREAM_MIN_GAIN_ENV, "0.5")
    assert _plan(128, ROWS_Q4, 30080).r2 == 0
    monkeypatch.setenv(ts.TWOSTREAM_MIN_GAIN_ENV, "0.05")
    assert _plan(128, ROWS_Q4, 30080).r2 == 13
