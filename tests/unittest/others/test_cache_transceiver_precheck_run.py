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
    path = tmp_path / "rank_2_recv.csv"
    path.write_text("RequestID,Bandwidth(Gbps),Delay(ms)\n1,80,0\n2,160,0\n3,240,0\n")
    # Gbps -> GB/s (/8); median of [10, 20, 30]
    assert rp.parse_bandwidth_gbps(str(tmp_path), 2) == 20.0


def test_parse_bandwidth_gbps_missing_or_malformed(tmp_path):
    assert rp.parse_bandwidth_gbps(str(tmp_path), 0) is None
    (tmp_path / "rank_0_recv.csv").write_text("RequestID,Delay(ms)\n1,0\n")
    assert rp.parse_bandwidth_gbps(str(tmp_path), 0) is None


def test_parse_python_bandwidth_gbps(tmp_path):
    (tmp_path / "perf_a_0.csv").write_text(
        "task_type,throughput_mbs\nKVSendTask,1024\nKVRecvTask,\n"
    )
    (tmp_path / "perf_a_1.csv").write_text("task_type,throughput_mbs\nKVSendTask,3072\n")
    # MB/s -> GB/s (/1024); median of [1, 3]
    assert rp.parse_python_bandwidth_gbps(str(tmp_path)) == 2.0
    assert rp.parse_python_bandwidth_gbps(str(tmp_path / "nowhere")) is None


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
    # Handshakes serialize across peers: rendezvous + per-peer slack.
    assert rp.hello_timeout_s(plan, 2) == 600 + 2 * (300 + 300)
    # Only the schedule's FIRST rep pays the NIXL wire-up allowance.
    assert rp.wave_timeout_s(plan, 0, 0) == 180 + 300
    assert rp.wave_timeout_s(plan, 0, 1) == 180
    assert rp.wave_timeout_s(plan, 1, 0) == 180


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

        assert len(inspect.signature(api.resolve_kv_cache_manager_v2_auto).parameters) == 2
        rt = inspect.signature(api.resolve_transceiver_runtime_auto).parameters
        assert list(rt)[:1] == ["llm_args"] and len(rt) >= 3

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
