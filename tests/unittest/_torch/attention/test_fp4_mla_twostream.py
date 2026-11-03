"""Host-side tests of the two-stream FP4 MLA decode planner (no GPU, no torch)."""

import pytest

from tensorrt_llm._torch.attention.backends.fp4_mla import fp4_mla_twostream as ts

SM_COUNT = 212  # 106 TPCs: 46 four-CTA clusters + 14 two-CTA pairs
ROWS_Q4 = 128 * 4
ROWS_Q2 = 128 * 2


@pytest.fixture(autouse=True)
def _internal_kernel_calibration(monkeypatch):
    """The shapes below were measured with the internal mufu16 2x kernel (T2 scale 1.0); the main tree ships a slower
    packed path (DEFAULT_T2_SCALE 1.2), so pin the calibration and test the main one separately."""
    monkeypatch.setenv(ts.TWOSTREAM_T2_SCALE_ENV, "1.0")
    monkeypatch.delenv(ts.TWOSTREAM_R2_ENV, raising=False)
    monkeypatch.delenv(ts.TWOSTREAM_ENV, raising=False)
    monkeypatch.delenv(ts.TWOSTREAM_MIN_GAIN_ENV, raising=False)
    monkeypatch.delenv(ts.TWOSTREAM_MAX_R2_ENV, raising=False)


def test_slower_2x_kernel_calibration(monkeypatch):
    """With a 20 % slower 2x kernel (GitHub-main packed path, 2026-11-02) the B128/q4 split no longer pays: the 2x arm
    would be the critical path (measured: r2 10-13 all equal to the plain 4x time); B48/q4 still splits."""
    monkeypatch.setenv(ts.TWOSTREAM_T2_SCALE_ENV, "1.2")
    assert _plan(128, ROWS_Q4, 30080).r2 == 0
    assert _plan(96, ROWS_Q4, 30080).r2 == 0
    assert _plan(48, ROWS_Q4, 30080).r2 == 2


def _plan(num_requests, rows, max_kv_len, **kw):
    return ts.plan_two_stream(
        num_requests=num_requests, rows_per_request=rows, max_kv_len=max_kv_len, sm_count=SM_COUNT, **kw
    )


def test_lanes():
    assert ts.lanes_2x(SM_COUNT) == 14
    assert ts.lanes_2x(184) == 0


@pytest.mark.parametrize("max_kv_len", [30080, 16384, 8192])
def test_b128_q4_moves_thirteen_requests(max_kv_len, monkeypatch):
    monkeypatch.delenv(ts.TWOSTREAM_R2_ENV, raising=False)
    plan = _plan(128, ROWS_Q4, max_kv_len)
    # 256 4x items = 5.57 waves of 46; the smallest r2 that brings the 4x arm to 5 whole waves is 13 (230 items)
    assert plan.r2 == 13, plan
    assert plan.waves_4x_plain == 6 and plan.waves_4x == 5 and plan.rounds_2x == 4
    assert plan.num_requests_4x == 115
    assert plan.model_us < plan.model_plain_us * (1 - ts.DEFAULT_MIN_GAIN)


def test_b128_q4_two_stream_never_makes_2x_critical():
    plan = _plan(128, ROWS_Q4, 30080)
    t4, t2 = ts.item_costs_us(118)
    assert plan.rounds_2x * t2 <= plan.waves_4x * t4


@pytest.mark.parametrize("num_requests,rows,max_kv_len", [(64, ROWS_Q4, 30080), (128, ROWS_Q2, 16384), (23, ROWS_Q4, 30080), (46, ROWS_Q2, 30080)])
def test_integral_waves_stay_on_plain_4x(num_requests, rows, max_kv_len):
    plan = _plan(num_requests, rows, max_kv_len)
    assert plan.r2 == 0, plan
    assert not plan.enabled


def test_b96_q4_moves_four_requests():
    plan = _plan(96, ROWS_Q4, 30080)
    # 192 items = 4.17 waves -> r2 = 4 (8 fewer items) gives exactly 4 waves, 2x arm 16 m-tiles = 2 rounds
    assert plan.r2 == 4 and plan.waves_4x == 4 and plan.rounds_2x == 2, plan


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
    monkeypatch.delenv(ts.TWOSTREAM_R2_ENV, raising=False)
    monkeypatch.setenv(ts.TWOSTREAM_MIN_GAIN_ENV, "0.5")
    assert _plan(128, ROWS_Q4, 30080).r2 == 0
    monkeypatch.setenv(ts.TWOSTREAM_MIN_GAIN_ENV, "0.05")
    assert _plan(128, ROWS_Q4, 30080).r2 == 13


def test_max_r2_cap(monkeypatch):
    monkeypatch.delenv(ts.TWOSTREAM_R2_ENV, raising=False)
    monkeypatch.setenv(ts.TWOSTREAM_MAX_R2_ENV, "10")
    plan = _plan(128, ROWS_Q4, 30080)
    # with at most 10 requests the 4x arm cannot drop a wave -> plain path
    assert plan.r2 == 0


def test_degenerate_inputs():
    assert _plan(1, ROWS_Q4, 30080).r2 == 0
    assert ts.plan_two_stream(num_requests=128, rows_per_request=ROWS_Q4, max_kv_len=30080, sm_count=184).r2 == 0
    assert _plan(128, 384, 30080).r2 == 0  # rows not a multiple of 256
