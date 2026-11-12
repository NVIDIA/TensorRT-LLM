"""Host-side tests of the FP4 MLA decode path selection (no GPU)."""

import math
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fp4_mla import config as fp4_mla_config
from tensorrt_llm._torch.attention.backends.fp4_mla import fp4_mla_cutedsl_4x_host as host_4x
from tensorrt_llm._torch.attention.backends.fp4_mla import fp4_mla_dispatch as dp
from tensorrt_llm._torch.attention.backends.fp4_mla import fp4_mla_twostream as ts

SM = 212
ROWS_Q4, ROWS_Q2 = 512, 256
CUDA0 = torch.device("cuda", 0)  # constructing it needs no GPU
OPT_IN_SMEM = 232448  # the 227 KiB a block may opt into without the oversized mode
HELIX = {"helix": True, "softmax_stats": True}  # the two inputs of a Helix decode call


def _choose(
    num_requests,
    query_len,
    kv_capacity,
    *,
    num_heads=128,
    live=None,
    mean=None,
    unavailable=None,
    **kw,
):
    rows = num_heads * query_len
    return dp.choose_decode_strategy(
        num_requests=num_requests,
        query_len=query_len,
        num_heads=num_heads,
        rows_per_seq=rows,
        kv_capacity=kv_capacity,
        sm_count=SM,
        envelope_reason=host_4x.envelope_reason_static,
        live_max_kv=live,
        live_mean_kv=mean,
        four_x_unavailable=unavailable,
        **kw,
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        fp4_mla_config.FP4_MLA_CUTEDSL_4X_ENV,
        fp4_mla_config.FP4_MLA_CUTEDSL_4X_H128_ENV,
        dp.DECODE_PATH_ENV,
        dp.MIN_KV_4X_ENV,
        dp.KV_REGIME_GRID_ENV,
        "TRTLLM_FP4_MLA_TWOSTREAM",
        "TRTLLM_FP4_MLA_TWOSTREAM_R2",
        host_4x.MIN_PAIRS_ENV,
        host_4x.SPLIT_MIN_KV_MULTIWAVE_ENV,
        ts.SPLIT_MIN_KV_ONEWAVE_ENV,
        host_4x.HELIX_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    # the measured table was taken with the internal 2x kernel; pin that calibration (the main tree defaults to 1.2)
    monkeypatch.setenv("TRTLLM_FP4_MLA_TWOSTREAM_T2_SCALE", "1.0")


def _set_4x_switches(monkeypatch, switch, h128_switch):
    """Set the 4x switch of every head count but 128 and the one of 128 heads; None leaves a switch unset."""
    for name, value in (
        (fp4_mla_config.FP4_MLA_CUTEDSL_4X_ENV, switch),
        (fp4_mla_config.FP4_MLA_CUTEDSL_4X_H128_ENV, h128_switch),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


@pytest.mark.parametrize(
    "switch, h128_switch, heads_128, other_heads, requested",
    [
        # a clean environment: on for 128 heads, off for every other count
        (None, None, True, False, False),
        ("", "", True, False, False),
        ("1", None, True, True, True),  # the switch of the other head counts turns them on
        ("0", None, True, False, False),  # and does not turn the 128-head decode off
        (None, "0", False, False, False),  # the 128-head switch turns the 128-head decode off,
        ("1", "0", False, True, True),  # whatever the other switch says,
        (None, "1", True, False, True),  # and turns no other head count on
        ("0", "1", True, False, True),
        ("0", "0", False, False, False),
        ("1", "1", True, True, True),
    ],
)
def test_4x_switches_go_by_the_head_count(
    monkeypatch, switch, h128_switch, heads_128, other_heads, requested
):
    _set_4x_switches(monkeypatch, switch, h128_switch)
    assert fp4_mla_config._fp4_mla_cutedsl_4x_enabled(128) is heads_128
    # 96 heads at q8 and 64 heads at q4 are inside the 4x envelope too: they follow their own switch
    for num_heads in (96, 64):
        assert fp4_mla_config._fp4_mla_cutedsl_4x_enabled(num_heads) is other_heads
    # only a switch turned on in the environment is a request for the kernel
    assert fp4_mla_config._fp4_mla_cutedsl_4x_requested() is requested


def test_unavailable_4x_is_2x(monkeypatch):
    s = _choose(128, 4, 30080, unavailable="no 4x kernel on this device")
    assert s.path == dp.PATH_2X and s.reason == "no 4x kernel on this device"
    # a forced path cannot bring the kernel back
    monkeypatch.setenv(dp.DECODE_PATH_ENV, "4x")
    assert _choose(128, 4, 30080, unavailable="no 4x kernel on this device").path == dp.PATH_2X


def test_device_unsupported_reason(monkeypatch):
    # the host names what the device lacks; a block with room for the kernel's shared memory is no reason
    monkeypatch.setattr(
        host_4x, "max_shared_memory_per_block", lambda device: host_4x.SMEM_BYTES_PER_CTA
    )
    assert host_4x.device_unsupported_reason(CUDA0) is None
    host_4x._require_device(CUDA0)
    monkeypatch.setattr(
        host_4x, "max_shared_memory_per_block", lambda device: host_4x.SMEM_BYTES_PER_CTA - 1
    )
    reason = host_4x.device_unsupported_reason(CUDA0)
    assert reason is not None and str(host_4x.SMEM_BYTES_PER_CTA) in reason
    assert _choose(128, 4, 30080, unavailable=reason).reason == reason
    # the host's own entry points refuse that device with the same reason, and whatever is not a CUDA device
    with pytest.raises(host_4x.EnvelopeError, match=str(host_4x.SMEM_BYTES_PER_CTA)):
        host_4x._require_device(CUDA0)
    assert "CUDA" in host_4x.device_unsupported_reason(torch.device("cpu"))
    with pytest.raises(host_4x.EnvelopeError, match="CUDA"):
        host_4x._require_device(torch.device("cpu"))


def _fake_cuda_driver(monkeypatch, *, known=True, get_err=0, attr_err=0, value=0):
    """Install a stand-in ``cuda.bindings.driver`` whose device reports ``value`` as its oversized limit."""
    driver = types.ModuleType("cuda.bindings.driver")
    driver.CUresult = SimpleNamespace(CUDA_SUCCESS=0)
    driver.CUdevice_attribute = SimpleNamespace(
        **({host_4x._OVERSIZED_SMEM_ATTRIBUTE: 150} if known else {})
    )
    driver.cuDeviceGet = lambda index: (get_err, index)
    driver.cuDeviceGetAttribute = lambda attribute, device: (attr_err, value)
    bindings = types.ModuleType("cuda.bindings")
    bindings.driver = driver
    package = types.ModuleType("cuda")
    package.bindings = bindings
    for name, module in (
        ("cuda", package),
        ("cuda.bindings", bindings),
        ("cuda.bindings.driver", driver),
    ):
        monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize(
    "driver, limit",
    [
        ({"value": 334848}, 334848),  # the oversized limit wins over the opt-in limit
        ({"value": 0}, OPT_IN_SMEM),  # a driver without the mode reports 0
        ({"known": False}, OPT_IN_SMEM),  # bindings that do not know the attribute
        ({"get_err": 1}, OPT_IN_SMEM),  # the device handle cannot be taken
        ({"attr_err": 1}, OPT_IN_SMEM),  # the driver refuses the attribute
    ],
)
def test_max_shared_memory_per_block(monkeypatch, driver, limit):
    _fake_cuda_driver(monkeypatch, **driver)
    queried = []

    def properties(index):
        queried.append(index)
        return SimpleNamespace(shared_memory_per_block_optin=OPT_IN_SMEM)

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    monkeypatch.setattr(host_4x, "_SMEM_PER_BLOCK_CACHE", {})
    device = torch.device("cuda", 3)  # an explicit index: the current device is never asked
    assert host_4x.max_shared_memory_per_block(device) == limit
    assert host_4x.max_shared_memory_per_block(device) == limit
    assert queried == [3]  # one device query per index, then the cache
    unsupported = host_4x.device_unsupported_reason(device) is not None
    assert unsupported == (limit < host_4x.SMEM_BYTES_PER_CTA)
    with pytest.raises(ValueError, match="CUDA"):
        host_4x.max_shared_memory_per_block(torch.device("cpu"))


@pytest.mark.parametrize(
    "switch, h128_switch, level",
    [(None, None, "info"), ("0", "0", "info"), ("1", None, "warning"), (None, "1", "warning")],
)
def test_device_without_the_4x_kernel_warns_only_when_it_was_asked_for(
    monkeypatch, switch, h128_switch, level
):
    from tensorrt_llm.logger import logger

    _set_4x_switches(monkeypatch, switch, h128_switch)
    _fake_cuda_driver(monkeypatch, known=False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(shared_memory_per_block_optin=OPT_IN_SMEM),
    )
    monkeypatch.setattr(host_4x, "_SMEM_PER_BLOCK_CACHE", {})
    logged = []
    for name in ("info", "warning"):
        monkeypatch.setattr(logger, name, lambda message, name=name: logged.append((name, message)))
    # no exception and a reason for the dispatcher; the device is asked once, so the message appears once
    assert host_4x.device_unsupported_reason(CUDA0) is not None
    assert host_4x.max_shared_memory_per_block(CUDA0) == OPT_IN_SMEM
    assert [name for name, _ in logged] == [level]
    assert "decode stays on the 2x kernel" in logged[0][1]


def test_max_shared_memory_per_block_of_an_unindexed_device(monkeypatch):
    # torch.device("cuda") names no index: the limit is the current device's, cached under that index
    _fake_cuda_driver(monkeypatch, value=334848)
    queried = []

    def properties(index):
        queried.append(index)
        return SimpleNamespace(shared_memory_per_block_optin=OPT_IN_SMEM)

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(host_4x, "_SMEM_PER_BLOCK_CACHE", {})
    assert host_4x.max_shared_memory_per_block(torch.device("cuda")) == 334848
    assert host_4x.device_unsupported_reason(torch.device("cuda")) is None
    assert queried == [2] and host_4x._SMEM_PER_BLOCK_CACHE == {2: 334848}
    # the same device named by its index is served from that entry; another index is asked on its own
    assert host_4x.max_shared_memory_per_block(torch.device("cuda", 2)) == 334848
    assert queried == [2]
    assert host_4x.max_shared_memory_per_block(torch.device("cuda", 0)) == 334848
    assert queried == [2, 0]


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


def test_unpacked_v_stays_2x():
    assert _choose(128, 4, 30080, v_packed=False).path == dp.PATH_2X


def _envelope_reason(num_requests, query_len, kv_capacity, *, num_heads, **inputs):
    return host_4x.envelope_reason_static(
        num_heads=num_heads,
        query_len=query_len,
        num_requests=num_requests,
        max_kv_len=kv_capacity,
        sm_count=SM,
        **inputs,
    )


def _envelope_refusal(*args, **kw):
    reason = _envelope_reason(*args, **kw)
    assert reason is not None, "the envelope admitted the call"
    return reason


@pytest.mark.parametrize(
    "num_heads,query_len,first,last",
    [(96, 8, 9, 15), (128, 4, 14, 23), (128, 2, 20, 46), (128, 8, 7, 11), (64, 8, 14, 23)],
)
def test_helix_envelope_is_the_one_wave_window(monkeypatch, num_heads, query_len, first, last):
    """Switch on: a Helix call takes the 4x kernel from the batch where the small-batch rule lets go up to one
    full wave of pair tasks, as the FULL-only launch whatever the page-table capacity."""
    monkeypatch.setenv(host_4x.HELIX_ENV, "1")
    for kv_capacity in (4096, 14208, 135168):
        shape = dict(query_len=query_len, kv_capacity=kv_capacity, num_heads=num_heads)
        for num_requests in range(first, last + 1):
            assert _envelope_reason(num_requests, **shape, **HELIX) is None
            assert not host_4x.use_split_mode(
                num_heads=num_heads,
                query_len=query_len,
                num_requests=num_requests,
                max_kv_len=kv_capacity,
                helix=True,
            )
        assert "2x packed plan" in _envelope_refusal(first - 1, **shape, **HELIX)
        too_many = _envelope_refusal(last + 1, **shape, **HELIX)
        assert too_many.startswith("Helix:") and "one wave" in too_many
    # what lets a graph runner key Helix graphs by no KV regime: no admitted shape has a graded KV floor
    for num_requests in range(first, last + 1):
        assert dp.four_x_floor_kv(num_requests, num_heads * query_len, SM) == 0


def test_helix_envelope_needs_the_switch_and_both_inputs(monkeypatch):
    one_wave = dict(num_requests=12, query_len=8, kv_capacity=14208, num_heads=96)  # 36 pair tasks
    two_waves = {**one_wave, "num_requests": 16}
    non_helix = [_envelope_reason(**shape) for shape in (one_wave, two_waves)]
    assert non_helix == [None, None]  # one FULL-only wave, and the split mode
    # default: the switch is off, every Helix call stays 2x and the reason names the switch
    for inputs in (HELIX, {"helix": True}, {"softmax_stats": True}):
        assert host_4x.HELIX_ENV in _envelope_refusal(**one_wave, **inputs)
    monkeypatch.setenv(host_4x.HELIX_ENV, "1")
    assert _envelope_reason(**one_wave, **HELIX) is None
    assert "together" in _envelope_refusal(**one_wave, helix=True)
    assert "together" in _envelope_refusal(**one_wave, softmax_stats=True)
    # the switch moves no non-Helix decision
    assert [_envelope_reason(**shape) for shape in (one_wave, two_waves)] == non_helix


def test_helix_one_wave_batch_runs_plain_4x(monkeypatch):
    # default: the switch is off
    assert _choose(12, 8, 14208, num_heads=96, **HELIX).path == dp.PATH_2X
    monkeypatch.setenv(host_4x.HELIX_ENV, "1")
    # 96 heads x q8 = 3 pair tasks per request: 2x under the small-batch rule, 4x up to one wave (15 requests)
    # and 2x beyond it
    for num_requests, path in (
        (1, dp.PATH_2X),
        (8, dp.PATH_2X),
        (9, dp.PATH_4X),
        (15, dp.PATH_4X),
        (16, dp.PATH_2X),
    ):
        assert _choose(num_requests, 8, 14208, num_heads=96, **HELIX).path == path, num_requests
    # one Helix input without the other stays 2x
    for one_input in ({"helix": True}, {"softmax_stats": True}):
        assert _choose(12, 8, 14208, num_heads=96, **one_input).path == dp.PATH_2X
    # a page-table capacity where a non-Helix call of this shape takes the split mode: still one FULL-only wave
    s = _choose(12, 8, 135168, num_heads=96, **HELIX)
    assert (s.path, s.r2, s.plan, s.kv_source) == (dp.PATH_4X, 0, None, "capacity")
    assert "Helix" in s.reason
    shape = dict(num_heads=96, query_len=8, num_requests=12, max_kv_len=135168)
    assert host_4x.use_split_mode(**shape) and not host_4x.use_split_mode(**shape, helix=True)
    unavailable = "no 4x kernel on this device"
    assert _choose(12, 8, 14208, num_heads=96, unavailable=unavailable, **HELIX).path == dp.PATH_2X
    # a flat KV floor above the page-table capacity keeps the call on 2x
    monkeypatch.setenv(dp.MIN_KV_4X_ENV, "16384")
    assert _choose(12, 8, 14208, num_heads=96, **HELIX).path == dp.PATH_2X


def test_helix_never_takes_the_two_stream_split(monkeypatch):
    """The two-stream arms carry neither KV bounds nor softmax statistics: where a non-Helix call splits, a Helix
    call of the same shape runs plain 4x (one wave) or stays on 2x (more than one wave)."""
    monkeypatch.setenv(host_4x.HELIX_ENV, "1")
    one_wave, many_waves = (20, 4, 14208), (48, 4, 30080)
    monkeypatch.setenv("TRTLLM_FP4_MLA_TWOSTREAM_R2", "2")
    assert _choose(*one_wave).path == dp.PATH_TWO_STREAM
    s = _choose(*one_wave, **HELIX)
    assert (s.path, s.r2, s.plan) == (dp.PATH_4X, 0, None)
    monkeypatch.delenv("TRTLLM_FP4_MLA_TWOSTREAM_R2")
    assert _choose(*many_waves).path == dp.PATH_TWO_STREAM
    assert _choose(*many_waves, **HELIX).path == dp.PATH_2X
    monkeypatch.setenv(dp.DECODE_PATH_ENV, "twostream")
    s = _choose(*one_wave, **HELIX)
    assert (s.path, s.r2, s.kv_source) == (dp.PATH_4X, 0, "forced")
    assert _choose(*many_waves, **HELIX).path == dp.PATH_2X
    monkeypatch.setenv(dp.DECODE_PATH_ENV, "4x")
    assert _choose(*one_wave, **HELIX).path == dp.PATH_4X
    assert _choose(*many_waves, **HELIX).path == dp.PATH_2X
    monkeypatch.setenv(dp.DECODE_PATH_ENV, "2x")
    assert _choose(*one_wave, **HELIX).path == dp.PATH_2X


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
    # one wave: 4x at every mean KV, behind the 2x regime of the steps too uneven for its FULL wave; more waves:
    # the 2x regime ends under the floor
    assert _regimes(16, 4, 30080) == (dp.KV_REGIME_UNEVEN, 30080)
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
    # a flat floor knob moves the 2x regime of every shape; a floor above the capacity leaves one regime, and so
    # does 0, but for the 2x regime of the steps too uneven for an uncut launch (B128/q4 splits at this calibration)
    monkeypatch.setenv(dp.MIN_KV_4X_ENV, "12288")
    assert _regimes(16, 4, 30080) == (12287, 30080)
    assert _regimes(128, 4, 30080) == (12287, 30080)
    monkeypatch.setenv(dp.MIN_KV_4X_ENV, "40000")
    assert _regimes(128, 4, 30080) == (30080,)
    monkeypatch.setenv(dp.MIN_KV_4X_ENV, "0")
    assert _regimes(128, 4, 30080) == (dp.KV_REGIME_UNEVEN, 30080)
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


# (heads, query_len, requests of one FULL 4x wave, requests of one 2x wave)
WAVE_SHAPES = [(128, 4, 23, 26), (96, 8, 15, 17), (128, 2, 46, 53), (128, 8, 11, 13)]


def test_extra_wave_floor_follows_the_wave_arithmetic():
    # the window is where ceil(pair tasks / 46) exceeds ceil(m-tiles / 106), whatever the heads and the query length
    for num_heads, query_len in ((128, 2), (128, 4), (128, 6), (128, 8), (96, 8), (64, 4), (64, 8)):
        rows = num_heads * query_len
        for num_requests in range(1, 300):
            pairs = num_requests * rows // 256
            waves_4x, waves_2x = math.ceil(pairs / 46), math.ceil(2 * pairs / (SM // 2))
            assert dp.two_x_waves(num_requests, rows, SM) == waves_2x
            floor = dp.extra_wave_floor_kv(num_requests, rows, SM)
            if waves_4x <= waves_2x:
                assert floor == 0
            else:
                # 2 against 1 waves: plain 4x from 12K; 3 against 2: from 6K; more waves: the many-wave floor;
                # never under the graded floor
                assert floor == {1: 12288, 2: 6144}.get(waves_2x, dp.FLOOR_KV_4X_MANY_WAVES)
                assert floor >= dp.four_x_floor_kv(num_requests, rows, SM)


@pytest.mark.parametrize("num_heads,query_len,full,last", WAVE_SHAPES)
def test_first_window(num_heads, query_len, full, last):
    rows = num_heads * query_len
    for cap in (30080, 163840):
        for num_requests in range(full + 1, last + 1):
            assert math.ceil(dp.four_x_waves(num_requests, rows, SM)) == 2
            assert dp.two_x_waves(num_requests, rows, SM) == 1
            assert dp.extra_wave_floor_kv(num_requests, rows, SM) == 12288
            for live in (1, 256, 768, 1024, 2047, 2048, 4096, 8192, 12287):
                s = _choose(num_requests, query_len, cap, live=live, num_heads=num_heads)
                # one FULL 4x wave on the 4x kernel (an under-filled one too: it is launched for the step's KV),
                # the other requests on the 2x kernel, up to 12K: also under the 2K graded floor of these batches
                assert s.path == dp.PATH_TWO_STREAM and s.r2 == num_requests - full, (live, s)
                assert s.plan.waves_4x == 1 and s.plan.rounds_2x == 1
                assert "one wave of the machine" in s.reason
                assert s.record_tag == f"ts{num_requests - full}|kv{live}"
            # plain 4x from the window floor
            for live in (12288, 16384, 30080):
                s = _choose(num_requests, query_len, cap, live=live, num_heads=num_heads)
                assert s.path == dp.PATH_4X and s.r2 == 0, (num_requests, live, s)
        # one 4x wave: plain 4x at every KV, no window
        assert dp.extra_wave_floor_kv(full, rows, SM) == 0
        assert _choose(full, query_len, cap, live=1024, num_heads=num_heads).path == dp.PATH_4X
        # one request past the window the 2x kernel needs its second wave too: the graded floor of 2K
        assert dp.extra_wave_floor_kv(last + 1, rows, SM) == 0
        assert _choose(last + 1, query_len, cap, live=2047, num_heads=num_heads).path == dp.PATH_2X
        assert _choose(last + 1, query_len, cap, live=2048, num_heads=num_heads).path == dp.PATH_4X


def test_one_wave_split_follows_the_host_launch(monkeypatch):
    # the planner splits where the host launches its 4x wave FULL, the host asked with the KV length of the decision,
    # whatever the page table
    for knob in (None, "0", "32768", str(1 << 20)):
        if knob is not None:
            monkeypatch.setenv(ts.SPLIT_MIN_KV_ONEWAVE_ENV, knob)
        for num_heads, query_len, full, last in WAVE_SHAPES:
            for cap in (8192, 32768, 65535, 65536, 163840):
                for num_requests in range(full + 1, last + 1):
                    s = _choose(num_requests, query_len, cap, live=2048, num_heads=num_heads)
                    full_launch = not host_4x.use_split_mode(
                        num_heads=num_heads, query_len=query_len, num_requests=full, max_kv_len=2048
                    )
                    # a split exactly where the host launches its 4x wave FULL, else 2x
                    assert (s.path == dp.PATH_TWO_STREAM) == full_launch, (
                        knob,
                        num_requests,
                        cap,
                        s,
                    )
                    assert s.path in (dp.PATH_TWO_STREAM, dp.PATH_2X)


def test_second_window_keeps_2x_up_to_its_floor():
    cap = 30080
    # 3 against 2 waves (q4 B47-53).  B47-49: the planner's split drops the extra wave with one 2x round
    for num_requests, r2 in ((47, 1), (48, 2), (49, 3)):
        assert _choose(num_requests, 4, cap, live=4095).path == dp.PATH_2X
        for live in (4096, 6143, 6144, 8192):
            s = _choose(num_requests, 4, cap, live=live)
            assert s.path == dp.PATH_TWO_STREAM and s.r2 == r2, (num_requests, live, s)
    # B50-53: the split would need two 2x rounds and the planner declines; plain 4x waits for 6K
    for num_requests in (50, 52, 53):
        assert dp.four_x_floor_kv(num_requests, ROWS_Q4, SM) == 4096
        assert dp.extra_wave_floor_kv(num_requests, ROWS_Q4, SM) == 6144
        for live in (2048, 4096, 6143):
            s = _choose(num_requests, 4, cap, live=live)
            assert s.path == dp.PATH_2X, (num_requests, live, s)
        assert "a wave more than the 2x" in _choose(num_requests, 4, cap, live=4096).reason
        assert _choose(num_requests, 4, cap, live=6144).path == dp.PATH_4X
    # B54: the 2x kernel needs three waves as well, the 4K floor again
    assert dp.extra_wave_floor_kv(54, ROWS_Q4, SM) == 0
    assert _choose(54, 4, cap, live=4096).path == dp.PATH_4X


def test_a_split_that_drops_no_wave_is_plain_4x():
    # a split must drop a whole 4x wave: batches that fill their 4x waves exactly run plain 4x
    cap = 30080
    for num_requests, query_len, num_heads in (
        (46, 4, 128),
        (69, 4, 128),
        (92, 2, 128),
        (23, 8, 128),
        (46, 8, 96),
    ):
        for live in (4096, 8192, 16384):
            s = _choose(num_requests, query_len, cap, live=live, num_heads=num_heads)
            assert s.path == dp.PATH_4X and s.r2 == 0, (num_requests, query_len, live, s)


def test_extra_wave_window_knobs(monkeypatch):
    cap = 30080
    # a flat floor replaces every per-batch floor, the window's too: plain 4x from it
    monkeypatch.setenv(dp.MIN_KV_4X_ENV, "0")
    assert dp.extra_wave_floor_kv(24, ROWS_Q4, SM) == 0
    assert _choose(24, 4, cap, live=2048).path == dp.PATH_4X
    assert _regimes(24, 4, cap) == (cap,)
    monkeypatch.setenv(dp.MIN_KV_4X_ENV, "4096")
    assert _choose(24, 4, cap, live=4095).path == dp.PATH_2X
    assert _choose(24, 4, cap, live=4096).path == dp.PATH_4X
    monkeypatch.delenv(dp.MIN_KV_4X_ENV)
    # without the two-stream the window stays on 2x up to its floor
    monkeypatch.setenv("TRTLLM_FP4_MLA_TWOSTREAM", "0")
    s = _choose(24, 4, cap, live=2048)
    assert s.path == dp.PATH_2X and "a wave more than the 2x" in s.reason
    assert _choose(24, 4, cap, live=12287).path == dp.PATH_2X
    assert _choose(24, 4, cap, live=12288).path == dp.PATH_4X
    assert _regimes(24, 4, cap) == (12287, cap)
    monkeypatch.delenv("TRTLLM_FP4_MLA_TWOSTREAM")
    # a forced r2 is the planner's; r2 = 0 means no split, and plain 4x keeps its floor
    monkeypatch.setenv("TRTLLM_FP4_MLA_TWOSTREAM_R2", "3")
    s = _choose(24, 4, cap, live=4096)
    assert s.path == dp.PATH_TWO_STREAM and s.r2 == 3
    monkeypatch.setenv("TRTLLM_FP4_MLA_TWOSTREAM_R2", "0")
    assert _choose(24, 4, cap, live=4096).path == dp.PATH_2X
    assert _choose(24, 4, cap, live=12288).path == dp.PATH_4X
    monkeypatch.delenv("TRTLLM_FP4_MLA_TWOSTREAM_R2")
    # forced paths ignore the window as they ignore the floor
    monkeypatch.setenv(dp.DECODE_PATH_ENV, "4x")
    assert _choose(24, 4, cap, live=2048).path == dp.PATH_4X
    assert _regimes(24, 4, cap) == (cap,)
    monkeypatch.setenv(dp.DECODE_PATH_ENV, "twostream")
    # a forced two-stream asks the planner alone, which has no split here: plain 4x
    assert _choose(24, 4, cap, live=2048).path == dp.PATH_4X
    monkeypatch.delenv(dp.DECODE_PATH_ENV)
    # what keeps a call on 2x everywhere keeps it here
    assert _choose(24, 4, cap, live=2048, helix=True).path == dp.PATH_2X
    assert _choose(24, 4, cap, live=2048, softmax_stats=True).path == dp.PATH_2X
    assert _choose(24, 4, cap, live=2048, unavailable="no 4x kernel").path == dp.PATH_2X
    # a page table under the multi-wave gate of the envelope (8K) keeps the whole batch on 2x
    assert _choose(24, 4, 4096).path == dp.PATH_2X


def test_kv_regimes_of_the_extra_wave_windows(monkeypatch):
    # first window: the one-wave split up to 12287 and plain 4x above, behind the 2x regime of the steps too uneven
    # for the split (three graphs)
    uneven = dp.KV_REGIME_UNEVEN
    for num_requests, query_len, num_heads in (
        (24, 4, 128),
        (26, 4, 128),
        (53, 2, 128),
        # 45 of 46 pair tasks: an under-filled wave, launched FULL for the KV of these steps
        (16, 8, 96),
        (13, 8, 128),  # 44 of 46
    ):
        assert _regimes(num_requests, query_len, 32768, num_heads) == (uneven, 12287, 32768)
        assert _regimes(num_requests, query_len, 163840, num_heads) == (uneven, 12287, 163840)
        assert _regimes(num_requests, query_len, 12288, num_heads) == (uneven, 12287, 12288)
        for cap in (32768, 163840):
            s = _choose(num_requests, query_len, cap, live=12287, num_heads=num_heads)
            assert s.path == dp.PATH_TWO_STREAM
        # a page table that ends under 12K: the split up to it
        assert _regimes(num_requests, query_len, 8192, num_heads) == (uneven, 8192)
    # second window: 2x up to 6143 where the planner has no split (the 4095 ceiling would separate 2x from 2x),
    # 2x up to 4095 and its two-stream above where it has one
    for num_requests in (50, 52, 53):
        assert _regimes(num_requests, 4, 32768) == (6143, 32768)
    for num_requests in (47, 48, 49):
        assert _regimes(num_requests, 4, 32768) == (4095, 32768)
    # further windows: the graded floor is the many-wave floor already (B96: 5 against 4 waves)
    assert dp.extra_wave_floor_kv(96, ROWS_Q4, SM) == dp.four_x_floor_kv(96, ROWS_Q4, SM) == 8192
    assert _regimes(96, 4, 30080) == (8191, 30080)
    # a batch that fills its waves exactly has no two-stream regime
    assert _regimes(46, 4, 32768) == (4095, 32768)
    # every boundary sits where the decision changes, and each graph decides like the steps of its regime
    for num_requests, query_len, num_heads, cap in (
        (24, 4, 128, 32768),
        (17, 8, 96, 32768),
        (17, 8, 96, 163840),
        (50, 2, 128, 32768),
        (12, 8, 128, 32768),
        (12, 8, 128, 163840),
        (52, 4, 128, 32768),
    ):
        regimes = _regimes(num_requests, query_len, cap, num_heads)

        def label(kv):
            s = _choose(num_requests, query_len, cap, live=kv, num_heads=num_heads)
            return s.path, s.r2

        low = 0
        for ceiling in (regime for regime in regimes if regime != dp.KV_REGIME_UNEVEN):
            assert label(low + 1) == label((low + 1 + ceiling) // 2) == label(ceiling), (
                num_requests,
                ceiling,
            )
            if ceiling != regimes[-1]:
                assert label(ceiling) != label(ceiling + 1), (num_requests, ceiling)
            assert dp.kv_regime_for_step(low + 1, regimes) == ceiling
            low = ceiling
    # the regime grid still adds its points
    monkeypatch.setenv(dp.KV_REGIME_GRID_ENV, "4096,16384")
    assert _regimes(24, 4, 32768) == (dp.KV_REGIME_UNEVEN, 4096, 12287, 16384, 32768)
    assert _regimes(52, 4, 32768) == (6143, 16384, 32768)


def test_page_table_floor(monkeypatch):
    # a page table of two KV tiles keeps every call on 2x (an eager step narrows it to its longest KV); one tile more
    # and the 4x paths take the call
    for num_requests, query_len in ((16, 4), (23, 4), (32, 2)):
        s = _choose(num_requests, query_len, dp.NARROW_KV_CAPACITY)
        assert s.path == dp.PATH_2X and "page table" in s.reason, s
        assert _choose(num_requests, query_len, dp.NARROW_KV_CAPACITY + 128).path == dp.PATH_4X
    # the flat floor replaces it, a forced path ignores it
    monkeypatch.setenv(dp.MIN_KV_4X_ENV, "0")
    assert _choose(16, 4, dp.NARROW_KV_CAPACITY).path == dp.PATH_4X
    monkeypatch.delenv(dp.MIN_KV_4X_ENV)
    monkeypatch.setenv(dp.DECODE_PATH_ENV, "4x")
    assert _choose(16, 4, dp.NARROW_KV_CAPACITY).path == dp.PATH_4X


# (requests, query_len, path at a mean KV of 8K-12K): the launches that run every request uncut -- one 4x wave
# (under-filled and full), plain 4x on waves whole requests fill exactly, the one-wave split, a planner split
UNCUT_SHAPES = [
    (16, 4, dp.PATH_4X),
    (23, 4, dp.PATH_4X),
    (46, 4, dp.PATH_4X),
    (24, 4, dp.PATH_TWO_STREAM),
    (48, 4, dp.PATH_TWO_STREAM),
]


@pytest.mark.parametrize("num_requests,query_len,path", UNCUT_SHAPES)
def test_uneven_step_leaves_an_uncut_launch_to_2x(num_requests, query_len, path):
    cap, longest = 32768, 12000
    for mean in (longest, longest - dp.SPREAD_MAX_UNCUT_KV):
        s = _choose(num_requests, query_len, cap, live=longest, mean=mean)
        assert s.path == path and s.kv_len_decision == mean, (mean, s)
    s = _choose(
        num_requests, query_len, cap, live=longest, mean=longest - dp.SPREAD_MAX_UNCUT_KV - 1
    )
    assert s.path == dp.PATH_2X and "above the mean KV" in s.reason, s


def test_uneven_step_keeps_plain_4x_that_cuts():
    # plain 4x with a last wave to fill cuts the long requests (device plan): an uneven step keeps it from the floor
    # of its mean KV, and stays on 2x under it
    s = _choose(32, 4, 30080, live=30000, mean=9000)
    assert s.path == dp.PATH_4X and s.kv_len_decision == 9000
    assert _choose(32, 4, 30080, live=30000, mean=2048).path == dp.PATH_4X
    assert _choose(32, 4, 30080, live=30000, mean=2047).path == dp.PATH_2X
    assert _choose(64, 4, 30080, live=30000, mean=4096).path == dp.PATH_4X
    assert _choose(64, 4, 30080, live=30000, mean=4095).path == dp.PATH_2X


def test_kv_regime_of_an_uneven_step():
    uneven = dp.KV_REGIME_UNEVEN
    assert dp.kv_regime_bounds(uneven) == (None, 0) and dp.kv_regime_bounds(8192) == (8192, 8192)

    def graph(num_requests, query_len, cap, regime):
        live, mean = dp.kv_regime_bounds(regime)
        return _choose(num_requests, query_len, cap, live=live, mean=mean).path

    # one wave: the regime of the mean while the step is even, the 2x regime in front once it is not
    regimes = _regimes(16, 4, 30080)
    assert regimes.uncut == {30080}
    assert dp.kv_regime_for_step(30080, regimes, 30080 - dp.SPREAD_MAX_UNCUT_KV) == 30080
    assert dp.kv_regime_for_step(30080, regimes, 1024) == uneven
    assert graph(16, 4, 30080, uneven) == dp.PATH_2X and graph(16, 4, 30080, 30080) == dp.PATH_4X
    # a planner split goes to the shape's own 2x regime; plain 4x that cuts keeps the regime of the mean
    regimes = _regimes(48, 4, 32768)
    assert regimes == (4095, 32768) and regimes.uncut == {32768}
    assert dp.kv_regime_for_step(30000, regimes, 8000) == 4095
    regimes = _regimes(32, 4, 30080)
    assert not regimes.uncut and dp.kv_regime_for_step(30000, regimes, 9000) == 30080
    # every regime's graph decides like the eager step it covers, uniform or not
    for num_requests, query_len, cap in (
        (16, 4, 30080),
        (24, 4, 32768),
        (48, 4, 32768),
        (32, 4, 30080),
    ):
        regimes = _regimes(num_requests, query_len, cap)
        for longest, mean in (
            (30000, 30000),
            (30000, 26000),
            (30000, 9000),
            (8192, 2048),
            (12000, 12000),
        ):
            eager = _choose(num_requests, query_len, cap, live=longest, mean=mean)
            regime = dp.kv_regime_for_step(longest, regimes, mean)
            assert (eager.path == dp.PATH_2X) == (
                graph(num_requests, query_len, cap, regime) == dp.PATH_2X
            ), (
                num_requests,
                longest,
                mean,
                regimes,
            )


def test_under_filled_wave_is_launched_for_the_kv_of_its_step():
    # the host launches an under-filled 4x wave in split mode from 65536 tokens of the decision's KV, a wave at most
    # half full (q2 B20: 20 pair tasks) from 16384, whatever the page table; the regime that spans the bound ends
    # under it
    for num_requests, query_len, bound in (
        (16, 4, ts.SPLIT_MIN_KV_ONEWAVE_DEFAULT),
        (20, 2, ts.SPLIT_MIN_KV_HALF_WAVE_DEFAULT),
    ):
        assert _regimes(num_requests, query_len, 163840) == (dp.KV_REGIME_UNEVEN, bound - 1, 163840)
        for live, split in ((2048, False), (bound - 1, False), (bound, True)):
            s = _choose(num_requests, query_len, 163840, live=live)
            assert s.path == dp.PATH_4X and s.kv_len_decision == live
            assert (
                host_4x.use_split_mode(
                    num_heads=128,
                    query_len=query_len,
                    num_requests=num_requests,
                    max_kv_len=s.kv_len_decision,
                )
                == split
            )
    assert _regimes(20, 2, 32768) == (dp.KV_REGIME_UNEVEN, 16383, 32768)
    # a full wave is FULL at every KV
    assert _regimes(23, 4, 163840) == (dp.KV_REGIME_UNEVEN, 163840)
