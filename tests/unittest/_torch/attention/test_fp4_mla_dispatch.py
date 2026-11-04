"""Host-side tests of the FP4 MLA decode path selection (no GPU, no torch)."""

import pytest

from tensorrt_llm._torch.attention.backends.fp4_mla import fp4_mla_cutedsl_4x_host as host_4x
from tensorrt_llm._torch.attention.backends.fp4_mla import fp4_mla_dispatch as dp

SM = 212
ROWS_Q4, ROWS_Q2 = 512, 256


def _choose(num_requests, query_len, kv_capacity, *, live=None, four_x=True, **kw):
    rows = 128 * query_len
    return dp.choose_decode_strategy(
        num_requests=num_requests,
        query_len=query_len,
        num_heads=128,
        rows_per_seq=rows,
        kv_capacity=kv_capacity,
        sm_count=SM,
        envelope_reason=host_4x.envelope_reason_static,
        live_max_kv=live,
        four_x_enabled=four_x,
        **kw,
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        dp.DECODE_PATH_ENV,
        dp.MIN_KV_4X_ENV,
        dp.KV_REGIME_GRID_ENV,
        "TRTLLM_FP4_MLA_TWOSTREAM",
        "TRTLLM_FP4_MLA_TWOSTREAM_R2",
        host_4x.MIN_PAIRS_ENV,
        host_4x.SPLIT_MIN_KV_MULTIWAVE_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    # the measured table was taken with the internal 2x kernel; pin that calibration (the main tree defaults to 1.2)
    monkeypatch.setenv("TRTLLM_FP4_MLA_TWOSTREAM_T2_SCALE", "1.0")


def test_knob_off_is_2x():
    s = _choose(128, 4, 30080, four_x=False)
    assert s.path == dp.PATH_2X


def test_b128_q4_long_kv_is_two_stream():
    s = _choose(128, 4, 30080)
    assert s.path == dp.PATH_TWO_STREAM and s.r2 == 13 and s.num_requests_2x == 13
    assert s.record_tag == "ts13"
    s16 = _choose(128, 4, 16384)
    assert s16.path == dp.PATH_TWO_STREAM and s16.r2 == 13


def test_b128_q4_short_kv_stays_2x():
    # 5.6 waves: the floor is 8K, from there on the 4x paths take the call
    assert _choose(128, 4, 4096).path == dp.PATH_2X
    assert _choose(128, 4, 8192).path != dp.PATH_2X
    # live KV wins over the capacity: a 4K step inside a 160K-capacity graph bucket stays on 2x
    s = _choose(128, 4, 163840, live=4096)
    assert s.path == dp.PATH_2X and s.kv_source == "live" and "kv4096" in s.record_tag
    assert _choose(128, 4, 163840, live=8192).path != dp.PATH_2X


@pytest.mark.parametrize(
    "num_requests,query_len,waves,floor",
    [
        (16, 4, 32 / 46, 0),
        (23, 4, 1.0, 0),
        (32, 4, 64 / 46, 2048),
        (48, 4, 96 / 46, 4096),
        (64, 4, 128 / 46, 4096),
        (128, 2, 128 / 46, 4096),
        (69, 4, 3.0, 4096),
        (70, 4, 140 / 46, 8192),
        (96, 4, 192 / 46, 8192),
        (128, 4, 256 / 46, 8192),
        (256, 4, 512 / 46, 8192),
    ],
)
def test_four_x_floor_is_graded_by_waves(num_requests, query_len, waves, floor):
    rows = 128 * query_len
    assert dp.four_x_waves(num_requests, rows, SM) == pytest.approx(waves)
    assert dp.four_x_floor_kv(num_requests, rows, SM) == floor
    # the decision follows the floor: 2x right under it, a 4x path from it on (the capacity is long enough)
    if floor:
        assert _choose(num_requests, query_len, 30080, live=floor - 1).path == dp.PATH_2X
    assert _choose(num_requests, query_len, 30080, live=max(floor, 1)).path != dp.PATH_2X


def test_graded_floor_matches_the_matrix_crossovers():
    # one wave (B16) wins at every KV; 1.4 waves (B32) from 2K; 2-3 waves (B48 two-stream, B64) from 4K; more
    # (B96, B128, B256) from 8K.  Live KV inside a long-capacity bucket, as the KV-regime graph runner hands it over
    # (the multi-wave gate of the static envelope is evaluated on the capacity).
    cap = 30080
    assert _choose(16, 4, cap, live=1024).path == dp.PATH_4X
    assert _choose(32, 4, cap, live=1024).path == dp.PATH_2X
    assert _choose(32, 4, cap, live=2048).path == dp.PATH_4X
    assert _choose(48, 4, cap, live=2048).path == dp.PATH_2X
    assert _choose(48, 4, cap, live=4096).path == dp.PATH_TWO_STREAM
    assert _choose(64, 4, cap, live=2048).path == dp.PATH_2X
    assert _choose(64, 4, cap, live=4096).path != dp.PATH_2X
    for num_requests in (96, 128, 256):
        assert _choose(num_requests, 4, cap, live=4096).path == dp.PATH_2X
        assert _choose(num_requests, 4, cap, live=8192).path != dp.PATH_2X
    # a short capacity keeps multi-wave batches on 2x through the static envelope's multi-wave KV gate (8K)
    assert _choose(32, 4, 2048).path == dp.PATH_2X
    assert _choose(16, 4, 2048).path == dp.PATH_4X


def test_live_kv_long_in_big_capacity_bucket():
    s = _choose(128, 4, 163840, live=30080)
    assert s.path == dp.PATH_TWO_STREAM and s.r2 == 13 and s.kv_len_decision == 30080


@pytest.mark.parametrize(
    "num_requests,query_len,kv", [(64, 4, 30080), (32, 4, 30080), (128, 2, 16384), (46, 2, 30080)]
)
def test_integral_waves_use_plain_4x(num_requests, query_len, kv):
    s = _choose(num_requests, query_len, kv)
    assert s.path == dp.PATH_4X and s.r2 == 0, s


def test_b96_q4_two_stream_four_requests():
    s = _choose(96, 4, 30080)
    assert s.path == dp.PATH_TWO_STREAM and s.r2 == 4


def test_small_batch_follows_envelope_rules():
    # 12 requests x 4 m-tiles: the 2x two-piece plan fits one wave -> envelope keeps 2x (host rule)
    s = _choose(12, 4, 30080)
    assert s.path == dp.PATH_2X and "two" in s.reason


def test_helix_and_stats_stay_2x():
    assert _choose(128, 4, 30080, helix=True).path == dp.PATH_2X
    assert _choose(128, 4, 30080, softmax_stats=True).path == dp.PATH_2X
    assert _choose(128, 4, 30080, v_packed=False).path == dp.PATH_2X


def test_forced_paths(monkeypatch):
    monkeypatch.setenv(dp.DECODE_PATH_ENV, "2x")
    assert _choose(128, 4, 30080).path == dp.PATH_2X
    monkeypatch.setenv(dp.DECODE_PATH_ENV, "4x")
    s = _choose(128, 4, 30080)
    assert s.path == dp.PATH_4X and s.kv_source == "forced"
    monkeypatch.setenv(dp.DECODE_PATH_ENV, "twostream")
    assert _choose(128, 4, 30080).path == dp.PATH_TWO_STREAM
    # a forced two-stream falls back to 4x where no split helps, and to 2x outside the envelope
    assert _choose(64, 4, 30080).path == dp.PATH_4X
    assert _choose(128, 4, 30080, helix=True).path == dp.PATH_2X
    # forcing never bypasses the KV floor only in auto mode: forced 4x at 8K is honoured if the envelope allows it
    monkeypatch.setenv(dp.DECODE_PATH_ENV, "4x")
    assert _choose(128, 4, 8192).path == dp.PATH_4X


def test_min_kv_knob(monkeypatch):
    assert dp.min_kv_4x() is None
    # one flat floor for every shape replaces the grading
    monkeypatch.setenv(dp.MIN_KV_4X_ENV, "4096")
    assert dp.min_kv_4x() == 4096 and dp.four_x_floor_kv(16, ROWS_Q4, SM) == 4096
    assert _choose(128, 4, 30080, live=4096).path != dp.PATH_2X
    s = _choose(16, 4, 30080, live=2048)
    assert s.path == dp.PATH_2X and "flat floor" in s.reason and "waves" not in s.reason
    monkeypatch.delenv(dp.MIN_KV_4X_ENV)
    s = _choose(128, 4, 30080, live=2048)
    assert s.path == dp.PATH_2X and "5.6 4x waves" in s.reason
    monkeypatch.setenv(dp.MIN_KV_4X_ENV, "4096")
    monkeypatch.setenv(dp.MIN_KV_4X_ENV, "40000")
    assert _choose(128, 4, 30080).path == dp.PATH_2X
    assert _choose(16, 4, 30080).path == dp.PATH_2X
    monkeypatch.setenv(dp.MIN_KV_4X_ENV, "0")
    assert _choose(128, 4, 30080, live=1).path != dp.PATH_2X
    monkeypatch.setenv(dp.MIN_KV_4X_ENV, "garbage")
    assert dp.min_kv_4x() is None and dp.four_x_floor_kv(128, ROWS_Q4, SM) == 8192


def test_main_tree_2x_calibration_keeps_plain_4x_at_b128(monkeypatch):
    monkeypatch.setenv("TRTLLM_FP4_MLA_TWOSTREAM_T2_SCALE", "1.2")
    s = _choose(128, 4, 30080)
    assert s.path == dp.PATH_4X and s.r2 == 0
    assert _choose(48, 4, 30080).path == dp.PATH_TWO_STREAM


def _regimes(num_requests, query_len, capacity, num_heads=128):
    rows = num_heads * query_len
    return dp.kv_regimes_for_batch(
        num_requests=num_requests,
        query_len=query_len,
        num_heads=num_heads,
        rows_per_seq=rows,
        kv_capacity=capacity,
        sm_count=SM,
        envelope_reason=host_4x.envelope_reason_static,
    )


def test_kv_regimes_for_batch(monkeypatch):
    # one wave: 4x-eligible at every KV, a single regime; more waves: the 2x regime ends under the floor
    assert _regimes(16, 4, 30080) == (30080,)
    assert _regimes(32, 4, 30080) == (2047, 30080)
    assert _regimes(48, 4, 30080) == (4095, 30080)
    assert _regimes(64, 4, 30080) == (4095, 30080)
    assert _regimes(128, 2, 30080) == (4095, 30080)
    assert _regimes(96, 4, 30080) == (8191, 30080)
    assert _regimes(128, 4, 163840) == (8191, 163840)
    assert _regimes(256, 4, 30080) == (8191, 30080)
    # shapes the 4x paths never take have the single regime of the capacity: the small-batch two-piece rule, rows
    # that are no multiple of 256 (q = 1), the split-mode pair bound, a capacity under the multi-wave KV gate
    assert _regimes(8, 4, 30080) == (30080,)
    assert _regimes(128, 1, 30080) == (30080,)
    assert _regimes(512, 4, 30080) == (30080,)
    assert _regimes(128, 4, 4096) == (4096,)
    # the capacity at the floor keeps both regimes; under the floor the capacity alone
    assert _regimes(128, 4, 8192) == (8191, 8192)
    assert _regimes(32, 4, 2047) == (2047,)
    # the 4x-eligible span splits where the two-stream plan changes: B48/q4 keeps r2 = 2 into the tens of K and
    # runs plain 4x at 160K, so a 160K page table gets a two-stream regime for the short steps
    regimes_160k = _regimes(48, 4, 163840)
    assert regimes_160k[0] == 4095 and regimes_160k[-1] == 163840 and len(regimes_160k) == 3, (
        regimes_160k
    )
    assert 16384 <= regimes_160k[1] < 163840
    assert _choose(48, 4, 163840, live=regimes_160k[1]).path == dp.PATH_TWO_STREAM
    assert _choose(48, 4, 163840, live=163840).path == dp.PATH_4X
    # every interior boundary separates two different plans, so each regime's graph bakes the plan of its own steps
    for num_requests, query_len in ((48, 4), (96, 4), (128, 4), (255, 4), (128, 2)):
        regimes = _regimes(num_requests, query_len, 163840)
        plans = [
            _choose(num_requests, query_len, 163840, live=ceiling).r2 for ceiling in regimes[1:]
        ]
        assert all(a != b for a, b in zip(plans, plans[1:])), (
            num_requests,
            query_len,
            regimes,
            plans,
        )
    # the grid adds points between the floor and the capacity; 0 adds none
    monkeypatch.setenv(dp.KV_REGIME_GRID_ENV, "4096,12288,16384,40000")
    assert _regimes(128, 4, 30080) == (8191, 12288, 16384, 30080)
    assert _regimes(32, 4, 30080) == (2047, 4096, 12288, 16384, 30080)
    monkeypatch.setenv(dp.KV_REGIME_GRID_ENV, "0")
    assert _regimes(128, 4, 30080) == (8191, 30080)
    monkeypatch.delenv(dp.KV_REGIME_GRID_ENV)
    # a flat floor knob moves the 2x regime of every shape; 0 or a floor above the capacity leaves one regime
    monkeypatch.setenv(dp.MIN_KV_4X_ENV, "12288")
    assert _regimes(16, 4, 30080) == (12287, 30080)
    assert _regimes(128, 4, 30080) == (12287, 30080)
    monkeypatch.setenv(dp.MIN_KV_4X_ENV, "40000")
    assert _regimes(128, 4, 30080) == (30080,)
    monkeypatch.setenv(dp.MIN_KV_4X_ENV, "0")
    assert _regimes(128, 4, 30080) == (30080,)
    monkeypatch.delenv(dp.MIN_KV_4X_ENV)
    # a forced 2x or 4x path never looks at the KV length; a forced two-stream still plans on it
    monkeypatch.setenv(dp.DECODE_PATH_ENV, "2x")
    assert _regimes(128, 4, 30080) == (30080,)
    monkeypatch.setenv(dp.DECODE_PATH_ENV, "4x")
    assert _regimes(128, 4, 30080) == (30080,)
    monkeypatch.setenv(dp.DECODE_PATH_ENV, "twostream")
    assert _regimes(128, 4, 30080) == (8191, 30080)


def test_kv_regime_for_step():
    ceilings = _regimes(128, 4, 30080)
    assert dp.kv_regime_for_step(1, ceilings) == 8191
    assert dp.kv_regime_for_step(8191, ceilings) == 8191
    assert dp.kv_regime_for_step(8192, ceilings) == 30080
    assert dp.kv_regime_for_step(30080, ceilings) == 30080
    assert dp.kv_regime_for_step(40000, ceilings) == 30080
    assert dp.kv_regime_for_step(5000, (30080,)) == 30080
    # the graph of a regime decides the way the eager call would for every step in it: 2x for every KV under the
    # floor, 4x-eligible from the floor up
    for kv in (1, 5000, 8191):
        assert _choose(128, 4, 30080, live=dp.kv_regime_for_step(kv, ceilings)).path == dp.PATH_2X
        assert _choose(128, 4, 30080, live=kv).path == dp.PATH_2X
    for kv in (8192, 16384, 30080):
        assert _choose(128, 4, 30080, live=dp.kv_regime_for_step(kv, ceilings)).path != dp.PATH_2X
        assert _choose(128, 4, 30080, live=kv).path != dp.PATH_2X
    # one wave: the single regime decides 4x, as the eager call does at every KV
    for kv in (1, 1024, 30080):
        assert (
            _choose(16, 4, 30080, live=dp.kv_regime_for_step(kv, _regimes(16, 4, 30080))).path
            == dp.PATH_4X
        )
        assert _choose(16, 4, 30080, live=kv).path == dp.PATH_4X
