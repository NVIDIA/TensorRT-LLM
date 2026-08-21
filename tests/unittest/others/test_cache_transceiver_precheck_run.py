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
"""Unit tests for the disagg cache-transceiver precheck DRIVER.

Target: tests/scripts/perf-sanity/cache_transceiver_precheck/run_precheck.py

Two halves:

- Pure-logic tests (no torch / tensorrt_llm / MPI): wire format, rid/seed
  scheme, rendezvous + abort-flag files, StatusRecorder, bandwidth CSV
  parsing, schedule/timeout derivation.

- Internal-API contract tests (import tensorrt_llm, no GPU work): the
  precheck drives TRT-LLM internals directly (_torch.pyexecutor.*,
  bindings.internal.*, private llm_utils resolvers), which carry no stability
  promise. run_precheck.load_internal_apis() is the single owner of those
  imports; these tests exercise it plus the constructor/signature shapes the
  driver relies on, so an upstream refactor fails HERE in pre-merge CI
  instead of aborting the SLURM disagg perf pipeline at runtime.
"""

import base64
import json
import os
import sys
import types

import pytest

_PRECHECK_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "..",
    "tests",
    "scripts",
    "perf-sanity",
    "cache_transceiver_precheck",
)
sys.path.insert(0, os.path.abspath(_PRECHECK_DIR))

import run_precheck as rp  # noqa: E402  (stdlib-only at import time)

KEY = b"k" * 32


# --------------------------------------------------------------------------- #
# rid / seed scheme
# --------------------------------------------------------------------------- #
def test_make_rid_unique_and_dense_within_session():
    num_ctx, num_gen, seqs = 3, 2, 50
    rids = set()
    for gj in range(num_gen):
        for ci in range(num_ctx):
            session = [rp.make_rid(ci, gj, num_ctx, s) for s in range(seqs)]
            # Dense in-session sequence: consecutive rids -> unique low-12-bit
            # tags among any 4096 consecutive requests (tagFromRequestId).
            assert session == list(range(session[0], session[0] + seqs))
            rids.update(session)
    assert len(rids) == num_ctx * num_gen * seqs
    assert all(r >= 1 for r in rids)


def test_seed_for_deterministic_and_distinct():
    assert rp.seed_for(7, 3) == rp.seed_for(7, 3)  # rank-independent by construction
    seeds = {rp.seed_for(rid, layer) for rid in (1, 2, 3) for layer in (0, 1, 2)}
    assert len(seeds) == 9
    assert all(0 <= s <= 0x7FFFFFFF for s in seeds)


@pytest.mark.parametrize(("prompt_len", "expected_blocks"), ((1024, 8), (7408, 58)))
def test_request_block_views_excludes_untransferred_speculative_page(prompt_len, expected_blocks):
    """V2's reserved MTP tokens must not expand the verified transfer range."""
    tokens_per_block = 128
    num_allocated = (prompt_len + 2 + tokens_per_block - 1) // tokens_per_block
    allocated = [-1] + list(range(num_allocated))
    buffer = object()

    def get_batch_cache_indices(request_ids, layer_idx):
        assert request_ids == [7]
        assert layer_idx == 4
        return [allocated]

    def get_buffers(global_layer, kv_layout):
        assert global_layer == 4
        assert kv_layout == "HND"
        return buffer

    kvm = types.SimpleNamespace(
        tokens_per_block=tokens_per_block,
        pp_layers=[4],
        get_batch_cache_indices=get_batch_cache_indices,
        get_buffers=get_buffers,
    )

    views = list(rp._request_block_views(kvm, rid=7, prompt_len=prompt_len))

    assert views == [(4, buffer, list(range(expected_blocks)))]


@pytest.mark.parametrize("available_blocks", [0, 1])
def test_request_block_views_rejects_prompt_under_allocation(available_blocks):
    allocated = [-1] + list(range(available_blocks))
    kvm = types.SimpleNamespace(
        tokens_per_block=128,
        pp_layers=[4],
        get_batch_cache_indices=lambda _request_ids, layer_idx: [allocated],
        get_buffers=lambda *_args, **_kwargs: pytest.fail(
            "buffer lookup must not occur after under-allocation"
        ),
    )

    with pytest.raises(
        rp._TransferError,
        match=rf"rid=7 layer=4: required=2 available={available_blocks}",
    ):
        list(rp._request_block_views(kvm, rid=7, prompt_len=256))


# --------------------------------------------------------------------------- #
# HMAC control-channel wire format
# --------------------------------------------------------------------------- #
def test_pack_unpack_roundtrip():
    obj = ["go", {"li": 0, "rep": 1, "wave": [0, 1]}]
    assert rp.unpack_msg(rp.pack_msg(obj, KEY), KEY) == obj


def test_unpack_rejects_tampered_frame():
    raw = bytearray(rp.pack_msg(["hello", {}], KEY))
    raw[0] ^= 0xFF
    with pytest.raises(rp._TransferError, match="HMAC"):
        rp.unpack_msg(bytes(raw), KEY)


def test_unpack_rejects_wrong_key():
    raw = rp.pack_msg(["hello", {}], KEY)
    with pytest.raises(rp._TransferError, match="HMAC"):
        rp.unpack_msg(raw, b"x" * 32)


def test_unpack_rejects_short_frame():
    with pytest.raises(rp._TransferError, match="too short"):
        rp.unpack_msg(b"tiny", KEY)


def test_params_to_wire_is_json_safe():
    p = types.SimpleNamespace(
        first_gen_tokens=[1, 2],
        req_id=42,
        opaque_state=b"\x00\x01binary",
        draft_tokens=None,
        ctx_dp_rank=3,
        disagg_info_endpoint="tcp://h:1",
    )
    wire = rp.params_to_wire(p)
    decoded = json.loads(json.dumps(wire))  # must survive the ZMQ JSON hop
    assert base64.b64decode(decoded["opaque_state"]) == p.opaque_state
    assert decoded["req_id"] == 42
    assert decoded["ctx_dp_rank"] == 3
    assert decoded["ctx_info_endpoint"] == "tcp://h:1"


# --------------------------------------------------------------------------- #
# Rendezvous + abort-flag files
# --------------------------------------------------------------------------- #
def test_addr_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    path = rp.addr_path(str(tmp_path), 0, 1)
    rp.write_addr(path, {"host": "h", "port": 5, "key": KEY.hex()})
    assert os.stat(path).st_mode & 0o777 == 0o600  # carries the HMAC key
    payload = rp.wait_for_addr(path, timeout_s=5)
    assert (payload["host"], payload["port"], payload["job"]) == ("h", 5, "123")


def test_wait_for_addr_rejects_stale_job(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "old-run")
    path = rp.addr_path(str(tmp_path), 0, 0)
    rp.write_addr(path, {"host": "h", "port": 5, "key": KEY.hex()})
    monkeypatch.setenv("SLURM_JOB_ID", "new-run")  # requeued job, reused work dir
    with pytest.raises(rp._Timeout):
        rp.wait_for_addr(path, timeout_s=1.5)


def test_wait_for_addr_times_out_on_missing_file(tmp_path):
    with pytest.raises(rp._Timeout):
        rp.wait_for_addr(rp.addr_path(str(tmp_path), 0, 0), timeout_s=0)


def test_abort_flag_roundtrip_and_write_once(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    work = str(tmp_path)
    assert rp.abort_flag_reason(work) is None
    rp.raise_abort_flag(work, "first failure\nsecond line ignored")
    assert rp.abort_flag_reason(work) == "first failure"
    rp.raise_abort_flag(work, "later failure")  # write-once: first reason wins
    assert rp.abort_flag_reason(work) == "first failure"


def test_abort_flag_stale_job_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "old")
    rp.raise_abort_flag(str(tmp_path), "stale")
    monkeypatch.setenv("SLURM_JOB_ID", "new")
    assert rp.abort_flag_reason(str(tmp_path)) is None


# --------------------------------------------------------------------------- #
# StatusRecorder
# --------------------------------------------------------------------------- #
def _read_status(tmp_path, name):
    with open(os.path.join(str(tmp_path), "status", f"{name}.status")) as f:
        text = f.read()
    with open(os.path.join(str(tmp_path), "status", f"{name}.json")) as f:
        doc = json.load(f)
    return text, doc


def test_recorder_pass(tmp_path):
    rec = rp.StatusRecorder(str(tmp_path), "gen", 0, is_leader=True)
    rec.record("ctx_0", 1024, "PASS")
    text, doc = _read_status(tmp_path, "gen_0")
    assert text.startswith("RUNNING")  # not final yet: a SIGKILL must not read as PASS
    assert doc["overall"] == "RUNNING"
    rec.finalize(extra={"transceiver_runtime": "CPP"})
    text, doc = _read_status(tmp_path, "gen_0")
    assert text.startswith("PASS gen_0")
    assert doc["overall"] == "PASS"
    assert doc["transceiver_runtime"] == "CPP"


def test_recorder_failure_summary_first_line_only(tmp_path):
    rec = rp.StatusRecorder(str(tmp_path), "ctx", 1, is_leader=True)
    rec.record("gen_0", 1024, "PASS")
    rec.record("gen_1", 2048, "TRANSFER_ERROR", "boom\ntraceback line\nmore")
    rec.finalize()
    text, doc = _read_status(tmp_path, "ctx_1")
    assert text.startswith("FAIL ctx_1")
    assert "boom | traceback line" in text
    assert "more" not in text  # full reason only in the json
    assert doc["overall"] == "FAIL"
    assert doc["cases"][1]["reason"].endswith("more")


def test_recorder_skip_is_not_a_failure(tmp_path):
    rec = rp.StatusRecorder(str(tmp_path), "gen", 0, is_leader=True)
    rec.record("ctx_0", 0, "SKIP", "fail-fast")
    assert rec.failed_cases() == []
    rec.finalize()
    text, _ = _read_status(tmp_path, "gen_0")
    assert text.startswith("PASS")


def test_recorder_non_leader_writes_nothing(tmp_path):
    rec = rp.StatusRecorder(str(tmp_path), "gen", 0, is_leader=False)
    rec.record("ctx_0", 0, "TRANSFER_ERROR", "x")
    rec.finalize()
    assert not os.path.exists(os.path.join(str(tmp_path), "status"))


# --------------------------------------------------------------------------- #
# Bandwidth CSV parsing
# --------------------------------------------------------------------------- #
def test_parse_bandwidth_gbps_median(tmp_path):
    # C++ names timing CSVs "<instanceId>_<rank>_<tag>.csv" (instanceId is a
    # runtime UUID), so the parser must suffix-match, not expect "rank_*".
    path = tmp_path / "3c9f0e2a-1111-2222-3333-444455556666_2_recv.csv"
    path.write_text("RequestID,Bandwidth(Gbps),Delay(ms)\n1,80,0\n2,160,0\n3,240,0\n")
    # Gbps -> GB/s (/8); median of [10, 20, 30]
    assert rp.parse_bandwidth_gbps(str(tmp_path), 2) == 20.0


def test_parse_bandwidth_gbps_duplicate_columns_mean(tmp_path):
    # C++ repeats the Bandwidth(Gbps) column once per transmission; the parser
    # must average them per request (DictReader would keep only the last).
    (tmp_path / "uuid_0_recv.csv").write_text(
        "RequestID,Bandwidth(Gbps),Bandwidth(Gbps)\n1,80,240\n"
    )
    # mean(80, 240) = 160 Gbps -> /8 = 20 GB/s
    assert rp.parse_bandwidth_gbps(str(tmp_path), 0) == 20.0


def test_parse_bandwidth_gbps_rank_suffix_no_cross_match(tmp_path):
    # Rank 1 must not pick up rank 11's file (the suffix's leading "_").
    (tmp_path / "uuid_11_recv.csv").write_text("RequestID,Bandwidth(Gbps)\n1,80\n")
    assert rp.parse_bandwidth_gbps(str(tmp_path), 1) is None
    assert rp.parse_bandwidth_gbps(str(tmp_path), 11) == 10.0


def test_parse_bandwidth_gbps_missing_or_malformed(tmp_path):
    assert rp.parse_bandwidth_gbps(str(tmp_path), 0) is None
    (tmp_path / "uuid_0_recv.csv").write_text("RequestID,Delay(ms)\n1,0\n")
    assert rp.parse_bandwidth_gbps(str(tmp_path), 0) is None


def test_parse_python_bandwidth_gbps(tmp_path):
    # PerfLogManager gives TRTLLM_KVCACHE_TIME_OUTPUT_PATH top priority and
    # names task CSVs "<instanceUuid>_<rank>.csv" (no fixed prefix); the
    # parser identifies them by header columns, not name.
    (tmp_path / "cd93dae6-9d75-4b0e-8a89-2c9e2f0f1a2b_0.csv").write_text(
        "task_type,throughput_mbs\nKVSendTask,1024\nKVRecvTask,\n"
    )
    (tmp_path / "cd93dae6-9d75-4b0e-8a89-2c9e2f0f1a2b_1.csv").write_text(
        "task_type,throughput_mbs\nKVSendTask,3072\n"
    )
    # MiB/s -> GB/s (*1024^2/1e9); median of [1024, 3072] MiB/s = 2048 MiB/s
    expected = 2048 * 1024 * 1024 / 1e9
    assert abs(rp.parse_python_bandwidth_gbps(str(tmp_path)) - expected) < 1e-9
    assert rp.parse_python_bandwidth_gbps(str(tmp_path / "nowhere")) is None


def test_parse_python_bandwidth_gbps_ignores_cpp_csvs(tmp_path):
    # C++ send/recv and gen-summary CSVs share csv_dir; they lack the
    # task_type/throughput_mbs columns and must not contribute samples.
    (tmp_path / "uuid_0_recv.csv").write_text("RequestID,Bandwidth(Gbps)\n1,80\n")
    (tmp_path / "uuid_0_gen_transfer_summary.csv").write_text(
        "RequestID,gen_side_transfer_time(ms),kv_cache_size\n1,1.0,1024\n"
    )
    assert rp.parse_python_bandwidth_gbps(str(tmp_path)) is None


def test_parse_bandwidth_gbps_ignores_gen_summary(tmp_path):
    # "<uuid>_<rank>_gen_transfer_summary.csv" must not match the
    # "_<rank>_recv.csv" suffix.
    (tmp_path / "uuid_0_gen_transfer_summary.csv").write_text(
        "RequestID,gen_side_transfer_time(ms),kv_cache_size\n1,1.0,1024\n"
    )
    assert rp.parse_bandwidth_gbps(str(tmp_path), 0) is None


# --------------------------------------------------------------------------- #
# Schedule / timeout derivation
# --------------------------------------------------------------------------- #
def _plan(**overrides):
    plan = {
        "request_lengths": [64, 128],
        "warmup_requests": 1,
        "num_requests": 2,
        "n_pairs": 3,
        "wave_size": 2,
        "rendezvous_timeout_s": 600,
        "wireup_timeout_s": 300,
        "wave_timeout_s": 180,
        "setup_timeout_s": 600,
        "peer_progress_timeout_s": 900,
    }
    plan.update(overrides)
    return plan


def test_schedule_covers_all_cells_in_lockstep_order():
    plan = _plan()
    sched = rp._schedule(plan)
    # 2 lengths x (1 warmup + 2 measured) reps x 2 waves ([0,1] and [2])
    assert len(sched) == 2 * 3 * 2
    assert sched[0] == (0, 64, 0, [0, 1])
    assert sched[1] == (0, 64, 0, [2])
    assert sched[-1] == (1, 128, 2, [2])


def test_timeout_budgets():
    plan = _plan()
    # Peer count is deliberately absent: active peer progress refreshes this
    # bounded no-progress interval for every serialized waiter.
    assert rp.hello_timeout_s(plan) == 900
    # Only the schedule's FIRST rep pays the NIXL wire-up allowance.
    assert rp.wave_timeout_s(plan, 0, 0) == 180 + 300
    assert rp.wave_timeout_s(plan, 0, 1) == 180
    assert rp.wave_timeout_s(plan, 1, 0) == 180


@pytest.mark.parametrize("role", ["ctx", "gen"])
def test_peer_progress_marker_is_atomic_and_run_stamped(tmp_path, monkeypatch, role):
    monkeypatch.setenv("SLURM_JOB_ID", "111")
    runner = types.SimpleNamespace(
        role=role,
        is_leader=True,
        work_dir=str(tmp_path),
        server_idx=2,
    )
    rp.publish_peer_progress(runner, "peer_0 first wave")
    first = rp.read_peer_progress(str(tmp_path), role, 2)
    assert isinstance(first, int)

    rp.publish_peer_progress(runner, "peer_0 second wave")
    assert rp.read_peer_progress(str(tmp_path), role, 2) != first

    monkeypatch.setenv("SLURM_JOB_ID", "222")
    assert rp.read_peer_progress(str(tmp_path), role, 2) is None


def test_ctx_control_wait_refreshes_only_on_target_gen_progress(monkeypatch):
    class Again(Exception):
        pass

    class FakeSocket:
        def __init__(self):
            self.calls = 0

        def recv(self):
            self.calls += 1
            if self.calls <= 2:
                raise Again
            return rp.pack_msg(("done", {}), KEY)

    runner = types.SimpleNamespace(
        is_leader=True,
        work_dir="/unused",
        comm=types.SimpleNamespace(bcast=lambda value, root=0: value),
        _zmq=lambda: (types.SimpleNamespace(Again=Again), None),
    )
    progress = iter((10, 11, 11))
    monotonic = iter((0.0, 1.0, 2.0, 3.0))
    progress_reads = []

    def read_progress(work_dir, role, server_idx):
        progress_reads.append((role, server_idx))
        return next(progress)

    monkeypatch.setattr(rp, "read_peer_progress", read_progress)
    monkeypatch.setattr(rp.time, "monotonic", lambda: next(monotonic))
    arms = []

    def arm(what, seconds, python_alarm, publish_progress=True):
        arms.append((what, seconds, python_alarm, publish_progress))

    msg = rp._recv_ctx_control(
        runner,
        FakeSocket(),
        KEY,
        peer_idx=3,
        what="bye gen_3",
        timeout_s=5,
        arm=arm,
        refresh_from_gen_progress=True,
    )
    assert msg[0] == "done"
    assert arms == [
        ("bye gen_3", 5, False, True),
        ("bye gen_3", 5, False, False),
    ]
    assert progress_reads == [("gen", 3), ("gen", 3), ("gen", 3)]


def test_ctx_control_wait_times_out_without_progress(monkeypatch):
    class Again(Exception):
        pass

    class FakeSocket:
        def recv(self):
            raise Again

    runner = types.SimpleNamespace(
        is_leader=True,
        work_dir="/unused",
        comm=types.SimpleNamespace(bcast=lambda value, root=0: value),
        _zmq=lambda: (types.SimpleNamespace(Again=Again), None),
    )
    monotonic = iter((0.0, 6.0))
    monkeypatch.setattr(rp, "read_peer_progress", lambda work_dir, role, server_idx: None)
    monkeypatch.setattr(rp.time, "monotonic", lambda: next(monotonic))
    with pytest.raises(rp._Timeout, match="made no progress for 5s"):
        rp._recv_ctx_control(
            runner,
            FakeSocket(),
            KEY,
            peer_idx=3,
            what="hello gen_3",
            timeout_s=5,
            arm=lambda *args, **kwargs: None,
            refresh_from_gen_progress=True,
        )


def test_watchdog_tracks_each_phase_budget(monkeypatch):
    class FakeHangDetector:
        instance = None

        def __init__(self, timeout, on_detected):
            self.timeout = timeout
            self.on_detected = on_detected
            self.checkpoints = []
            FakeHangDetector.instance = self

        def start(self):
            pass

        def checkpoint(self):
            self.checkpoints.append(self.timeout)

        def cancel_task(self):
            pass

        def stop(self):
            pass

    api = types.SimpleNamespace(HangDetector=FakeHangDetector)
    monkeypatch.setattr(rp, "load_internal_apis", lambda: api)
    runner = types.SimpleNamespace(
        role="gen",
        is_leader=False,
        server_idx=0,
        side={"num_peers": 1},
        recorder=types.SimpleNamespace(record=lambda *args: None, finalize=lambda: None),
    )
    previous_alarm_handler = rp.signal.getsignal(rp.signal.SIGALRM)
    arm, disarm, stop, _ = rp._install_watchdog(runner, _plan(), rank=0)
    try:
        arm("setup", seconds=600)
        arm("first wave", seconds=480)
        arm("steady wave")
        assert FakeHangDetector.instance.checkpoints == [660, 540, 240]
    finally:
        disarm()
        stop()
        rp.signal.signal(rp.signal.SIGALRM, previous_alarm_handler)


# --------------------------------------------------------------------------- #
# Transfer ownership
# --------------------------------------------------------------------------- #
def _ctx_finish_runner(monkeypatch, check_status):
    events = []
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm",
        types.SimpleNamespace(logger=types.SimpleNamespace(info=lambda *_args, **_kwargs: None)),
    )
    runner = object.__new__(rp.PrecheckRunner)
    runner.xcvr = types.SimpleNamespace(check_context_transfer_status=check_status)
    runner.llm_request_state = types.SimpleNamespace(DISAGG_TRANS_ERROR="error")
    runner.server_idx = 0
    runner.rank = 0
    runner._consensus_error = lambda err: None if err is None else repr(err)
    runner._free_all = lambda reqs: events.append(("free", sorted(reqs)))
    return runner, events


def test_ctx_finish_wave_frees_only_after_every_request_completes(monkeypatch):
    runner, events = _ctx_finish_runner(monkeypatch, lambda _n: ([101, 102], []))
    reqs = {
        0: types.SimpleNamespace(py_request_id=101, state="in_progress"),
        1: types.SimpleNamespace(py_request_id=102, state="in_progress"),
    }

    runner.ctx_finish_wave(reqs)

    assert events == [("free", [0, 1])]


def test_ctx_finish_wave_retains_pages_after_transfer_failure(monkeypatch):
    runner, events = _ctx_finish_runner(monkeypatch, lambda _n: ([101], [102]))
    reqs = {
        0: types.SimpleNamespace(py_request_id=101, state="in_progress"),
        1: types.SimpleNamespace(py_request_id=102, state="in_progress"),
    }

    with pytest.raises(rp._FatalTransferError, match=r"ctx transfer failed for pairs \[1\]"):
        runner.ctx_finish_wave(reqs)

    assert events == []


def test_ctx_finish_wave_block_all_requires_terminal_status(monkeypatch):
    # A request still nonterminal after block-all returns exceeded the
    # kv_transfer_timeout deadline (Python) or violated the true block-all
    # contract (C++); both classify as a page-retaining gate failure.
    runner, events = _ctx_finish_runner(monkeypatch, lambda _n: ([101], []))
    reqs = {
        0: types.SimpleNamespace(py_request_id=101, state="in_progress"),
        1: types.SimpleNamespace(py_request_id=102, state="in_progress"),
    }

    with pytest.raises(rp._FatalTransferError, match="block-all returned before terminal"):
        runner.ctx_finish_wave(reqs)

    assert events == []


def _gen_run_wave_runner(monkeypatch, outcome):
    requests = {}
    events = []
    states = types.SimpleNamespace(
        DISAGG_GENERATION_TRANS_COMPLETE="complete",
        DISAGG_TRANS_ERROR="error",
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            cuda=types.SimpleNamespace(synchronize=lambda: events.append("cuda_sync"))
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm",
        types.SimpleNamespace(logger=types.SimpleNamespace(info=lambda *_args, **_kwargs: None)),
    )

    def make_request(_is_ctx, rid, _req_len, _runtime, ctx_params=None):
        req = types.SimpleNamespace(py_request_id=rid, state="in_progress")
        requests[rid] = req
        return req

    monkeypatch.setattr(rp, "make_request", make_request)
    monkeypatch.setattr(rp, "add_sequence", lambda *_args: None)

    def check_status(_at_least_request_num):
        completed, failed, cancelled = outcome(requests)
        for rid in completed:
            requests[rid].state = states.DISAGG_GENERATION_TRANS_COMPLETE
        for rid in failed:
            requests[rid].state = states.DISAGG_TRANS_ERROR
        return completed, failed, [requests[rid] for rid in cancelled]

    runner = object.__new__(rp.PrecheckRunner)
    runner.runtime = "PYTHON"
    runner.kvm = object()
    runner.use_v2 = True
    runner.server_idx = 0
    runner.rank = 0
    runner.llm_request_state = states
    runner.plan = {"verify_data": False, "warmup_requests": 0}
    runner.xcvr = types.SimpleNamespace(
        request_and_receive_async=lambda _req: None,
        check_gen_transfer_status=check_status,
    )
    runner.comm = types.SimpleNamespace(allgather=lambda value: [value])
    runner._owned = lambda _wave: [0, 1]
    runner._pair_rid = lambda _peer, _li, _rep, pair: 101 + pair
    runner._consensus_error = lambda err: None if err is None else repr(err)
    runner._free_all = lambda reqs: events.append(("free", sorted(reqs)))
    return runner, events


def test_gen_run_wave_frees_only_after_every_receive_completes(monkeypatch):
    runner, events = _gen_run_wave_runner(monkeypatch, lambda _reqs: ([101, 102], [], []))

    ok, detail = runner.gen_run_wave(0, 0, 64, 0, [0, 1], {0: object(), 1: object()})

    assert ok and not detail
    assert events == ["cuda_sync", ("free", [0, 1])]


@pytest.mark.parametrize(
    ("outcome", "message"),
    (
        (lambda _reqs: ([101], [102], []), r"failed=\[102\]"),
        (lambda _reqs: ([101], [], [102]), r"cancelled=\[102\]"),
        # Nonterminal after the block-all deadline: a gate failure, not a
        # keep-polling condition.
        (lambda _reqs: ([101], [], []), r"missing=\[102\]"),
    ),
)
def test_gen_run_wave_retains_pages_without_all_successes(monkeypatch, outcome, message):
    runner, events = _gen_run_wave_runner(monkeypatch, outcome)

    with pytest.raises(rp._FatalTransferError, match=message):
        runner.gen_run_wave(0, 0, 64, 0, [0, 1], {0: object(), 1: object()})

    assert events == []


@pytest.mark.parametrize(
    ("failing_resolver", "message"),
    (
        ("runtime", "refusing to validate a runtime"),
        ("manager", "refusing to assume V1"),
    ),
)
def test_model_preference_resolution_fails_closed(monkeypatch, failing_resolver, message):
    def resolve_runtime(_shim, _model_cls, _hf_view):
        if failing_resolver == "runtime":
            raise RuntimeError("runtime resolution failed")

    def resolve_manager(_args, _model_cls, _hf_view):
        if failing_resolver == "manager":
            raise RuntimeError("manager resolution failed")
        return True

    api = types.SimpleNamespace(
        resolve_transceiver_runtime_auto=resolve_runtime,
        resolve_kv_cache_manager_v2_auto=resolve_manager,
        TorchLlmArgs=lambda **kwargs: types.SimpleNamespace(**kwargs),
        MTPDecodingConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(rp, "load_internal_apis", lambda: api)
    monkeypatch.setattr(rp, "_lookup_model_cls", lambda _model_dir: (object(), object()))
    cache_cfg = types.SimpleNamespace(transceiver_runtime="auto")
    side = {
        "use_kv_cache_manager_v2": "auto",
        "parallel": {"tp": 1, "pp": 1, "cp": 1},
    }

    with pytest.raises(RuntimeError, match=message):
        rp.resolve_model_prefs("/model", side, cache_cfg)


# --------------------------------------------------------------------------- #
# Model preference resolution
# --------------------------------------------------------------------------- #
def test_resolve_model_prefs_allows_registered_class_without_preference_hook(monkeypatch):
    model_cls = type("ModelWithoutPreferenceHook", (), {})
    cache_cfg = types.SimpleNamespace(transceiver_runtime="CPP")
    calls = []

    def resolve_v2(shim, resolved_model_cls, pretrained_config):
        calls.append((shim, resolved_model_cls, pretrained_config))
        return False

    hf_view = object()
    monkeypatch.setattr(rp, "_lookup_model_cls", lambda _model_dir: (model_cls, hf_view))
    monkeypatch.setattr(
        rp,
        "load_internal_apis",
        lambda: types.SimpleNamespace(
            TorchLlmArgs=lambda **kwargs: types.SimpleNamespace(**kwargs),
            resolve_kv_cache_manager_v2_auto=resolve_v2,
        ),
    )

    use_v2 = rp.resolve_model_prefs(
        "/models/example",
        {
            "use_kv_cache_manager_v2": "auto",
            "parallel": {"tp": 1, "pp": 1, "cp": 1},
        },
        cache_cfg,
    )

    assert use_v2 is False
    assert len(calls) == 1
    assert calls[0][1:] == (model_cls, hf_view)


# --------------------------------------------------------------------------- #
# Transfer ownership
# --------------------------------------------------------------------------- #


def test_ctx_finish_wave_retains_pages_when_block_all_raises(monkeypatch):
    def check_status(_n):
        raise RuntimeError("interrupted")

    runner, events = _ctx_finish_runner(monkeypatch, check_status)
    reqs = {0: types.SimpleNamespace(py_request_id=101, state="in_progress")}

    with pytest.raises(rp._FatalTransferError, match="interrupted"):
        runner.ctx_finish_wave(reqs)

    assert events == []


def test_ctx_finish_wave_does_not_free_when_peer_rank_is_unsafe(monkeypatch):
    runner, events = _ctx_finish_runner(monkeypatch, lambda _n: ([101, 102], []))
    reqs = {
        0: types.SimpleNamespace(py_request_id=101, state="in_progress"),
        1: types.SimpleNamespace(py_request_id=102, state="in_progress"),
    }
    runner._consensus_error = lambda _err: "rank 1 did not prove completion"

    with pytest.raises(rp._FatalTransferError, match="rank 1 did not prove completion"):
        runner.ctx_finish_wave(reqs)

    assert events == []


def test_ctx_finish_wave_consensus_exception_is_fatal(monkeypatch):
    runner, events = _ctx_finish_runner(monkeypatch, lambda _n: ([101], []))
    reqs = {0: types.SimpleNamespace(py_request_id=101, state="in_progress")}

    def fail_consensus(_error):
        raise RuntimeError("MPI consensus failed")

    runner._consensus_error = fail_consensus

    with pytest.raises(rp._FatalTransferError, match="MPI consensus failed"):
        runner.ctx_finish_wave(reqs)

    assert events == []


def test_ctx_finish_wave_timeout_skips_consensus_and_free(monkeypatch):
    def check_status(_n):
        raise rp._Timeout("deadline")

    runner, events = _ctx_finish_runner(monkeypatch, check_status)
    consensus_calls = []
    runner._consensus_error = lambda err: consensus_calls.append(err)
    reqs = {0: types.SimpleNamespace(py_request_id=101, state="in_progress")}

    with pytest.raises(rp._Timeout, match="deadline"):
        runner.ctx_finish_wave(reqs)

    assert consensus_calls == []
    assert events == []


def test_ctx_run_wave_setup_error_retains_allocated_pages(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm",
        types.SimpleNamespace(logger=types.SimpleNamespace(info=lambda *_args, **_kwargs: None)),
    )
    monkeypatch.setattr(
        rp,
        "make_request",
        lambda _is_ctx, rid, _req_len, _runtime: types.SimpleNamespace(
            py_request_id=rid, context_phase_params=None
        ),
    )
    monkeypatch.setattr(rp, "add_sequence", lambda *_args: None)
    monkeypatch.setattr(rp, "fill_request", lambda *_args: None)

    calls = {"send": 0, "free": 0}

    def respond(_req):
        calls["send"] += 1
        if calls["send"] == 2:
            raise RuntimeError("injected setup failure")

    runner = object.__new__(rp.PrecheckRunner)
    runner.runtime = "PYTHON"
    runner.kvm = object()
    runner.use_v2 = True
    runner.server_idx = 0
    runner.rank = 0
    runner.is_leader = True
    runner.side = {"parallel": {"enable_attention_dp": False}}
    runner.mapping = types.SimpleNamespace(pp_rank=0)
    runner.xcvr = types.SimpleNamespace(respond_and_send_async=respond)
    runner.comm = types.SimpleNamespace(
        gather=lambda obj, root=0: [obj],
        bcast=lambda obj, root=0: obj,
    )
    runner._owned = lambda _wave: [0, 1]
    runner._pair_rid = lambda _peer, _li, _rep, pair: 101 + pair
    runner._consensus_error = lambda err: None if err is None else repr(err)
    runner._free_all = lambda _reqs: calls.__setitem__("free", calls["free"] + 1)

    with pytest.raises(rp._FatalTransferError, match="injected setup failure"):
        runner.ctx_run_wave(0, 0, 64, 0, [0, 1])

    assert calls == {"send": 2, "free": 0}


def test_ctx_run_wave_post_dispatch_collective_error_is_fatal(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm",
        types.SimpleNamespace(logger=types.SimpleNamespace(info=lambda *_args, **_kwargs: None)),
    )
    monkeypatch.setattr(
        rp,
        "make_request",
        lambda _is_ctx, rid, _req_len, _runtime: types.SimpleNamespace(
            py_request_id=rid, context_phase_params=object()
        ),
    )
    monkeypatch.setattr(rp, "add_sequence", lambda *_args: None)
    monkeypatch.setattr(rp, "fill_request", lambda *_args: None)

    runner = object.__new__(rp.PrecheckRunner)
    runner.runtime = "PYTHON"
    runner.kvm = object()
    runner.use_v2 = True
    runner.server_idx = 0
    runner.rank = 0
    runner.is_leader = True
    runner.side = {"parallel": {"enable_attention_dp": False}}
    runner.mapping = types.SimpleNamespace(pp_rank=0)
    runner.xcvr = types.SimpleNamespace(respond_and_send_async=lambda _req: None)
    runner.comm = types.SimpleNamespace(
        gather=lambda _obj, root=0: (_ for _ in ()).throw(RuntimeError("MPI gather failed")),
        bcast=lambda obj, root=0: obj,
    )
    runner._owned = lambda _wave: [0]
    runner._pair_rid = lambda *_args: 101
    runner._consensus_error = lambda _err: None

    with pytest.raises(rp._FatalTransferError, match="MPI gather failed"):
        runner.ctx_run_wave(0, 0, 64, 0, [0])


def test_gen_run_wave_checks_python_status_on_empty_owner_rank(monkeypatch):
    calls = []

    def outcome(requests):
        calls.append(dict(requests))
        return [], [], []

    runner, events = _gen_run_wave_runner(monkeypatch, outcome)
    runner._owned = lambda _wave: []

    ok, detail = runner.gen_run_wave(0, 0, 64, 0, [0, 1], {})

    assert ok and not detail
    assert calls == [{}]
    assert events == ["cuda_sync", ("free", [])]


def test_gen_run_wave_setup_error_retains_allocated_pages(monkeypatch):
    runner, events = _gen_run_wave_runner(monkeypatch, lambda _reqs: ([], [], []))
    calls = 0

    def receive(_req):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected setup failure")

    runner.xcvr.request_and_receive_async = receive

    with pytest.raises(rp._FatalTransferError, match="injected setup failure"):
        runner.gen_run_wave(0, 0, 64, 0, [0, 1], {0: object(), 1: object()})

    assert calls == 2
    assert events == []


def test_gen_run_wave_does_not_free_when_peer_rank_is_unsafe(monkeypatch):
    runner, events = _gen_run_wave_runner(monkeypatch, lambda _reqs: ([101, 102], [], []))
    consensus_calls = []

    def consensus(error):
        consensus_calls.append(error)
        # First call covers receive setup; the second is the transfer-release
        # proof and models another rank reporting a nonterminal request.
        return None if len(consensus_calls) == 1 else "rank 1 did not prove completion"

    runner._consensus_error = consensus

    with pytest.raises(rp._FatalTransferError, match="rank 1 did not prove completion"):
        runner.gen_run_wave(0, 0, 64, 0, [0, 1], {0: object(), 1: object()})

    assert consensus_calls == [None, None]
    assert events == ["cuda_sync"]


def test_gen_run_wave_transfer_consensus_exception_is_fatal(monkeypatch):
    runner, events = _gen_run_wave_runner(monkeypatch, lambda _reqs: ([101, 102], [], []))
    consensus_calls = 0

    def consensus(_error):
        nonlocal consensus_calls
        consensus_calls += 1
        if consensus_calls == 2:
            raise RuntimeError("MPI transfer consensus failed")
        return None

    runner._consensus_error = consensus

    with pytest.raises(rp._FatalTransferError, match="MPI transfer consensus failed"):
        runner.gen_run_wave(0, 0, 64, 0, [0, 1], {0: object(), 1: object()})

    assert events == ["cuda_sync"]


def test_gen_run_wave_timeout_skips_transfer_consensus_and_free(monkeypatch):
    def timeout(_reqs):
        raise rp._Timeout("deadline")

    runner, events = _gen_run_wave_runner(monkeypatch, timeout)
    consensus_calls = []

    def consensus(error):
        consensus_calls.append(error)
        return None

    runner._consensus_error = consensus

    with pytest.raises(rp._Timeout, match="deadline"):
        runner.gen_run_wave(0, 0, 64, 0, [0, 1], {0: object(), 1: object()})

    # Receive setup reaches consensus before block-all. The timeout itself
    # must propagate directly into the process-fatal path.
    assert consensus_calls == [None]
    assert events == []


def test_hard_abort_unquiesced_persists_verdict_before_abort(monkeypatch):
    class AbortSentinel(Exception):
        pass

    events = []
    recorder = types.SimpleNamespace(
        record=lambda *args: events.append(("record", *args)),
        finalize=lambda **kwargs: events.append(("finalize", kwargs)),
    )
    runner = types.SimpleNamespace(
        is_leader=True,
        recorder=recorder,
        work_dir="/tmp/precheck",
        use_v2=True,
        runtime="PYTHON",
        comm=object(),
        xcvr=types.SimpleNamespace(shutdown=lambda: events.append(("shutdown",))),
    )
    monkeypatch.setattr(rp.signal, "alarm", lambda seconds: events.append(("alarm", seconds)))
    monkeypatch.setattr(
        rp,
        "raise_abort_flag",
        lambda work_dir, reason: events.append(("abort_flag", work_dir, reason)),
    )
    monkeypatch.setattr(
        rp,
        "_coordinate_abort_after_leader_flush",
        lambda comm: events.append(("coordinate", comm)),
    )

    def abort_process(comm):
        events.append(("abort", comm))
        raise AbortSentinel

    monkeypatch.setattr(rp, "_hard_abort_process", abort_process)

    with pytest.raises(AbortSentinel):
        rp._hard_abort_unquiesced(
            runner,
            {"what": "ctx wave"},
            rp._FatalTransferError("completion unproven"),
        )

    assert events == [
        ("alarm", 0),
        ("record", "ctx wave", 0, "TRANSFER_ERROR", "completion unproven"),
        (
            "abort_flag",
            "/tmp/precheck",
            "ctx wave TRANSFER_ERROR: completion unproven",
        ),
        (
            "finalize",
            {
                "extra": {
                    "kv_cache_manager": "V2",
                    "transceiver_runtime": "PYTHON",
                }
            },
        ),
        ("coordinate", runner.comm),
        ("abort", runner.comm),
    ]


def test_hard_abort_unquiesced_still_aborts_when_status_write_fails(monkeypatch):
    class AbortSentinel(Exception):
        pass

    runner = types.SimpleNamespace(
        is_leader=True,
        recorder=types.SimpleNamespace(
            record=lambda *_args: (_ for _ in ()).throw(OSError("disk full")),
        ),
        work_dir="/tmp/precheck",
        use_v2=True,
        runtime="PYTHON",
        comm=object(),
    )
    monkeypatch.setattr(rp.signal, "alarm", lambda _seconds: None)
    monkeypatch.setattr(rp, "_coordinate_abort_after_leader_flush", lambda _comm: None)
    monkeypatch.setattr(
        rp,
        "_hard_abort_process",
        lambda _comm: (_ for _ in ()).throw(AbortSentinel()),
    )

    with pytest.raises(AbortSentinel):
        rp._hard_abort_unquiesced(
            runner,
            {"what": "ctx wave"},
            rp._FatalTransferError("completion unproven"),
        )


def test_ctx_peer_loop_reraises_fatal_transfer(monkeypatch):
    fatal = rp._FatalTransferError("not quiesced")
    runner = types.SimpleNamespace(
        side={"num_peers": 1},
        is_leader=False,
        recorder=types.SimpleNamespace(record=lambda *_args: None),
    )
    recorded = []

    def serve(*_args, **_kwargs):
        raise fatal

    monkeypatch.setattr(rp, "ctx_serve_peer", serve)

    with pytest.raises(rp._FatalTransferError, match="not quiesced"):
        rp._serve_gen_peers(
            runner,
            plan={},
            arm=lambda *_args, **_kwargs: None,
            disarm=lambda: None,
            record_peer_failure=lambda *args: recorded.append(args),
        )

    assert recorded == []


def test_gen_peer_loop_reraises_fatal_transfer(monkeypatch):
    fatal = rp._FatalTransferError("not quiesced")
    runner = types.SimpleNamespace(
        side={"num_peers": 1},
        recorder=types.SimpleNamespace(record=lambda *_args: None),
    )
    recorded = []
    monkeypatch.setattr(rp, "_consensus_abort_reason", lambda _runner: None)

    def run_peer(*_args, **_kwargs):
        raise fatal

    monkeypatch.setattr(rp, "gen_run_peer", run_peer)

    with pytest.raises(rp._FatalTransferError, match="not quiesced"):
        rp._drive_ctx_peers(
            runner,
            arm=lambda *_args, **_kwargs: None,
            disarm=lambda: None,
            record_peer_failure=lambda *args: recorded.append(args),
        )

    assert recorded == []


# --------------------------------------------------------------------------- #
# Internal-API contract (imports tensorrt_llm; no GPU work)
# --------------------------------------------------------------------------- #
class TestInternalApiContract:
    @pytest.fixture(scope="class")
    def api(self):
        pytest.importorskip("tensorrt_llm")
        return rp.load_internal_apis()

    def test_loader_caches(self, api):
        assert rp.load_internal_apis() is api

    def test_create_kv_cache_transceiver_signature(self, api):
        import inspect

        params = inspect.signature(api.create_kv_cache_transceiver).parameters
        # Exactly the positional call shape PrecheckRunner.setup uses.
        assert list(params)[:5] == [
            "mapping",
            "dist",
            "kv_cache_manager",
            "attention_type",
            "cache_transceiver_config",
        ]

    def test_transceiver_interface_methods(self, api):
        import importlib

        mod = importlib.import_module(api.create_kv_cache_transceiver.__module__)
        base = mod.KvCacheTransceiver
        for meth in (
            "respond_and_send_async",
            "request_and_receive_async",
            "check_context_transfer_status",
            "check_gen_transfer_status",
        ):
            assert hasattr(base, meth), f"KvCacheTransceiver lost {meth}"

    @pytest.mark.parametrize("manager_attr", ["KVCacheManager", "KVCacheManagerV2"])
    def test_kv_cache_manager_ctor_kwargs(self, api, manager_attr):
        import inspect

        params = inspect.signature(getattr(api, manager_attr).__init__).parameters
        needed = {
            "num_layers",
            "num_kv_heads",
            "head_dim",
            "tokens_per_block",
            "max_seq_len",
            "max_batch_size",
            "mapping",
            "dtype",
            "spec_config",
        }
        if manager_attr == "KVCacheManagerV2":
            needed |= {"vocab_size", "is_disagg"}
        missing = needed - set(params)
        assert not missing, f"{manager_attr} ctor lost kwargs: {sorted(missing)}"

    def test_serving_resolvers(self, api):
        import inspect

        # The driver calls the serving resolver with real LLM args, the model
        # class, and its pretrained config.
        v2 = inspect.signature(api.resolve_kv_cache_manager_v2_auto).parameters
        assert list(v2)[:3] == ["llm_args", "model_cls", "pretrained_config"]
        assert all(p.default is not inspect.Parameter.empty for p in list(v2.values())[1:])
        rt = inspect.signature(api.resolve_transceiver_runtime_auto).parameters
        assert list(rt)[:1] == ["llm_args"] and len(rt) >= 3

    def test_model_preference_resolver_supports_v2(self, api, monkeypatch):
        class _PreferV2:
            @classmethod
            def get_preferred_kv_cache_manager_version(cls, pretrained_config=None):
                return "V2"

        monkeypatch.setattr(rp, "load_internal_apis", lambda: api)
        monkeypatch.setattr(
            rp,
            "_lookup_model_cls",
            lambda model_dir: (_PreferV2, types.SimpleNamespace()),
        )
        cache_cfg = api.CacheTransceiverConfig(backend="NIXL", transceiver_runtime="PYTHON")

        assert rp.resolve_model_prefs(
            "/tmp/dummy_model",
            {
                "use_kv_cache_manager_v2": "auto",
                "parallel": {"tp": 1, "pp": 1, "cp": 1},
            },
            cache_cfg,
        )

    def test_deepseek_v4_auto_selects_kv_cache_manager_v2(self, api, tmp_path):
        model_dir = tmp_path / "deepseek-v4"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            json.dumps({"architectures": ["DeepseekV4ForCausalLM"]})
        )
        side = {
            "use_kv_cache_manager_v2": "auto",
            "parallel": {"tp": 1, "pp": 1, "cp": 1},
        }
        cache_cfg = api.CacheTransceiverConfig(backend="NIXL", transceiver_runtime="PYTHON")

        assert rp.resolve_model_prefs(str(model_dir), side, cache_cfg) is True

    def test_enum_members(self, api):
        for enum, members in (
            (api.DataType, ("FP8", "HALF", "BF16")),
            (api.CacheTypeCpp, ("SELF", "SELFKONLY")),
            (api.AttentionTypeCpp, ("DEFAULT", "MLA")),
            (api.LlmRequestState, ("DISAGG_GENERATION_TRANS_COMPLETE", "DISAGG_TRANS_ERROR")),
            (
                api.LlmRequestType,
                ("LLMREQUEST_TYPE_CONTEXT_ONLY", "LLMREQUEST_TYPE_GENERATION_ONLY"),
            ),
        ):
            for m in members:
                assert hasattr(enum, m), f"{enum} lost member {m}"

    def test_hang_detector_surface(self, api):
        import inspect

        params = inspect.signature(api.HangDetector.__init__).parameters
        assert {"timeout", "on_detected"} <= set(params)
        for meth in ("start", "checkpoint", "cancel_task", "stop"):
            assert hasattr(api.HangDetector, meth), f"HangDetector lost {meth}"

    def test_config_constructors(self, api):
        cache_cfg = api.CacheTransceiverConfig(backend="UCX", max_tokens_in_buffer=1024)
        assert hasattr(cache_cfg, "transceiver_runtime")
        llm_args = api.TorchLlmArgs(model="/tmp/model", tensor_parallel_size=2)
        assert llm_args.tensor_parallel_size == 2
        api.KvCacheConfigCpp(max_tokens=64, enable_block_reuse=False)
        api.MTPDecodingConfig(num_nextn_predict_layers=1)
        api.Mapping(
            world_size=1,
            rank=0,
            gpus_per_node=1,
            tp_size=1,
            pp_size=1,
            cp_size=1,
            enable_attention_dp=False,
        )
        assert hasattr(api.Distributed, "get")

    def test_params_wire_roundtrip_through_real_bindings(self, api):
        # opaque_state must DESERIALIZE in the ContextPhaseParams ctor
        # (arbitrary bytes -> std::bad_alloc), so use the serialized empty
        # state: b"" is re-encoded by the bindings into its canonical form.
        source = api.DisaggregatedParams(
            ctx_request_id=42,
            first_gen_tokens=[7, 8],
            opaque_state=b"",
            draft_tokens=[9],
            ctx_dp_rank=1,
            ctx_info_endpoint="tcp://host:1234",
        ).get_context_phase_params()
        restored = rp.params_from_wire(rp.params_to_wire(source))
        assert rp.params_to_wire(restored) == rp.params_to_wire(source)

    def test_make_request_shapes(self, api):
        ctx_req = rp.make_request(True, rid=11, req_len=8, runtime="CPP")
        assert ctx_req.py_request_id == 11
        py_ctx = rp.make_request(True, rid=12, req_len=8, runtime="PYTHON")
        assert py_ctx.py_disaggregated_params.request_type == "context_only"
        ctx_params = api.DisaggregatedParams(
            ctx_request_id=13,
            first_gen_tokens=[1],
            opaque_state=b"",
            ctx_dp_rank=0,
        ).get_context_phase_params()
        gen_req = rp.make_request(False, rid=13, req_len=8, runtime="CPP", ctx_params=ctx_params)
        assert gen_req.py_request_id == 13
        py_gen = rp.make_request(False, rid=14, req_len=8, runtime="PYTHON", ctx_params=ctx_params)
        assert py_gen.py_disaggregated_params.request_type == "generation_only"
