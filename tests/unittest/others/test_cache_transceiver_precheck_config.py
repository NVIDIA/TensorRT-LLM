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
"""CPU-only tests for the disagg cache-transceiver precheck config resolution.

Target: tests/scripts/perf-sanity/cache_transceiver_precheck/precheck_config.py
"""

import json
import os
import sys

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

import precheck_config as pcfg  # noqa: E402
import run_precheck as rp  # noqa: E402  (stdlib-only at import time)


def _disagg_yaml(ctx_extra=None, gen_extra=None, **overrides):
    """Minimal disagg perf-sanity yaml shaped like the checked-in configs."""
    ctx = {
        "tensor_parallel_size": 4,
        "pipeline_parallel_size": 1,
        "context_parallel_size": 1,
        "enable_attention_dp": True,
        "kv_cache_config": {"dtype": "fp8"},
        "cache_transceiver_config": {"max_tokens_in_buffer": 16384, "backend": "NIXL"},
        "speculative_config": {"decoding_type": "MTP", "num_nextn_predict_layers": 1},
    }
    gen = {
        "tensor_parallel_size": 16,
        "pipeline_parallel_size": 1,
        "context_parallel_size": 1,
        "enable_attention_dp": True,
        "kv_cache_config": {"dtype": "fp8"},
        "cache_transceiver_config": {"max_tokens_in_buffer": 16384, "backend": "NIXL"},
        "speculative_config": {"decoding_type": "MTP", "num_nextn_predict_layers": 1},
    }
    ctx.update(ctx_extra or {})
    gen.update(gen_extra or {})
    cfg = {
        "metadata": {"model_name": "deepseek_r1_0528_fp4_v2"},
        "benchmark": {"mode": "e2e", "input_length": 8192, "output_length": 1024},
        "hardware": {"gpus_per_node": 4, "num_ctx_servers": 1, "num_gen_servers": 1},
        "worker_config": {"ctx": ctx, "gen": gen},
    }
    cfg.update(overrides)
    return cfg


def test_resolve_plan_adp_asymmetric():
    plan = pcfg.resolve_plan(_disagg_yaml())
    assert not plan["skip"]
    assert plan["ctx"] == {
        "tp": 4,
        "pp": 1,
        "cp": 1,
        "enable_attention_dp": True,
        "world_size": 4,
        "dp_size": 4,
    }
    assert plan["gen"]["world_size"] == 16 and plan["gen"]["dp_size"] == 16
    # Cover every gen dp rank.
    assert plan["n_pairs"] == 16
    assert plan["chunk_size"] == 8
    assert plan["request_lengths"] == [1024, 8192]
    assert plan["ctx_num_nextn_predict_layers"] == 1
    assert plan["ctx_cache_transceiver_config"]["backend"] == "NIXL"


def test_request_lengths_clamped_by_buffer_and_cap():
    cfg = _disagg_yaml(benchmark={"mode": "e2e", "input_length": 131072})
    cfg["worker_config"]["ctx"]["cache_transceiver_config"]["max_tokens_in_buffer"] = 131104
    cfg["worker_config"]["gen"]["cache_transceiver_config"]["max_tokens_in_buffer"] = 131104
    plan = pcfg.resolve_plan(cfg)
    # Derived ISL is capped by max_request_length (default 32768).
    assert plan["request_lengths"] == [1024, 32768]

    cfg["worker_config"]["ctx"]["cache_transceiver_config"]["max_tokens_in_buffer"] = 4096
    cfg["worker_config"]["gen"]["cache_transceiver_config"]["max_tokens_in_buffer"] = 4096
    plan = pcfg.resolve_plan(cfg)
    assert plan["request_lengths"] == [1024, 4096]

    # Explicit yaml override is used as-is (not capped).
    cfg["cache_transceiver_precheck"] = {"request_lengths": [64000]}
    cfg["worker_config"]["ctx"]["cache_transceiver_config"]["max_tokens_in_buffer"] = 131104
    cfg["worker_config"]["gen"]["cache_transceiver_config"]["max_tokens_in_buffer"] = 131104
    plan = pcfg.resolve_plan(cfg)
    assert plan["request_lengths"] == [64000]


def test_gen_only_no_context_skips():
    cfg = _disagg_yaml(benchmark={"mode": "gen_only_no_context", "input_length": 1024})
    plan = pcfg.resolve_plan(cfg, benchmark_mode="gen_only")
    assert plan["skip"]
    # e2e over the same yaml still runs (ctx servers are launched there).
    assert not pcfg.resolve_plan(cfg, benchmark_mode="e2e")["skip"]


def test_backend_mismatch_raises():
    cfg = _disagg_yaml(
        gen_extra={"cache_transceiver_config": {"backend": "UCX", "max_tokens_in_buffer": 16384}}
    )
    with pytest.raises(ValueError, match="backend mismatch"):
        pcfg.resolve_plan(cfg)


def test_pair_participation_and_chunks():
    plan = pcfg.resolve_plan(_disagg_yaml())
    # ADP ctx (dp4): pair k belongs to tp_rank k % 4.
    assert pcfg.pair_participates(plan, "ctx", 1, 5)
    assert not pcfg.pair_participates(plan, "ctx", 0, 5)
    # ADP gen (dp16): 1:1.
    assert pcfg.pair_participates(plan, "gen", 5, 5)
    assert not pcfg.pair_participates(plan, "gen", 4, 5)
    assert pcfg.chunks(plan) == [list(range(8)), list(range(8, 16))]
    # ctx rank owns 2 pairs per chunk of 8; gen rank owns at most 1.
    assert pcfg.max_owned_per_chunk(plan, "ctx") == 2
    assert pcfg.max_owned_per_chunk(plan, "gen") == 1

    # Non-ADP side participates everywhere and owns the whole chunk.
    plan_pp = pcfg.resolve_plan(
        _disagg_yaml(
            ctx_extra={
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 8,
                "enable_attention_dp": False,
            }
        )
    )
    assert plan_pp["ctx"]["dp_size"] == 1 and plan_pp["n_pairs"] == 16
    assert pcfg.pair_participates(plan_pp, "ctx", 0, 11)
    assert pcfg.max_owned_per_chunk(plan_pp, "ctx") == plan_pp["chunk_size"]


def test_fingerprint_role_agnostic():
    plan_a = pcfg.resolve_plan(_disagg_yaml())
    plan_b = pcfg.resolve_plan(_disagg_yaml())
    assert plan_a["fingerprint"] == plan_b["fingerprint"]
    changed = _disagg_yaml()
    changed["worker_config"]["gen"]["tensor_parallel_size"] = 8
    assert pcfg.resolve_plan(changed)["fingerprint"] != plan_a["fingerprint"]


def test_model_kv_shape_mla_and_gqa(tmp_path):
    mla = tmp_path / "mla"
    mla.mkdir()
    (mla / "config.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 61,
                "kv_lora_rank": 512,
                "qk_rope_head_dim": 64,
                "num_attention_heads": 128,
            }
        )
    )
    shape = pcfg.model_kv_shape(str(mla))
    assert shape == {
        "num_layers": 61,
        "num_kv_heads": 1,
        "head_dim": 576,
        "is_mla": True,
        "vocab_size": None,
        "source": "config.json (MLA)",
    }

    gqa = tmp_path / "gqa"
    gqa.mkdir()
    (gqa / "config.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "hidden_size": 4096,
            }
        )
    )
    shape = pcfg.model_kv_shape(str(gqa))
    assert shape["num_kv_heads"] == 8 and shape["head_dim"] == 128 and not shape["is_mla"]

    # Unresolvable model dir -> synthetic fallback (precheck still runs).
    assert pcfg.model_kv_shape(None)["source"] == "fallback"
    assert pcfg.model_kv_shape(str(tmp_path / "missing"))["source"] == "fallback"


def test_side_plan_views():
    plan = pcfg.resolve_plan(_disagg_yaml())
    ctx_view = pcfg.side_plan(plan, "ctx")
    gen_view = pcfg.side_plan(plan, "gen")
    assert ctx_view["parallel"]["world_size"] == 4
    assert ctx_view["num_peers"] == 1 and gen_view["num_peers"] == 1
    assert gen_view["cache_transceiver_config"]["max_tokens_in_buffer"] == 16384


class TestControlWireFormat:
    """run_precheck's HMAC-JSON control frames (importable without torch)."""

    def test_roundtrip(self):
        key = b"\x01" * 32
        msg = ["go", {"li": 0, "rep": 1, "chunk": 2}]
        assert rp.unpack_msg(rp.pack_msg(msg, key), key) == msg

    def test_tampered_frame_rejected(self):
        key = b"\x01" * 32
        raw = rp.pack_msg(["hello", {}], key)
        bad = raw[:-1] + bytes([raw[-1] ^ 0xFF])
        with pytest.raises(rp._TransferError):
            rp.unpack_msg(bad, key)

    def test_wrong_key_rejected(self):
        raw = rp.pack_msg(["hello", {}], b"\x01" * 32)
        with pytest.raises(rp._TransferError):
            rp.unpack_msg(raw, b"\x02" * 32)

    def test_short_frame_rejected(self):
        with pytest.raises(rp._TransferError):
            rp.unpack_msg(b"tiny", b"\x01" * 32)

    def test_addr_file_owner_only(self, tmp_path):
        path = str(tmp_path / "rendezvous" / "ctx0_gen0.addr")
        rp.write_addr(path, {"host": "h", "port": 1, "key": "aa"})
        assert (os.stat(path).st_mode & 0o777) == 0o600
        with open(path) as f:
            assert json.load(f)["key"] == "aa"


def test_use_kv_cache_manager_v2_flags():
    # Absent -> "auto" (the driver resolves it against the model's
    # get_model_defaults at runtime, like serving).
    plan = pcfg.resolve_plan(_disagg_yaml())
    assert plan["ctx_use_kv_cache_manager_v2"] == "auto"
    assert plan["gen_use_kv_cache_manager_v2"] == "auto"
    assert pcfg.side_plan(plan, "ctx")["use_kv_cache_manager_v2"] == "auto"

    # Explicit yaml values win, per side.
    plan = pcfg.resolve_plan(
        _disagg_yaml(
            ctx_extra={"kv_cache_config": {"dtype": "fp8", "use_kv_cache_manager_v2": False}},
            gen_extra={"kv_cache_config": {"dtype": "fp8", "use_kv_cache_manager_v2": True}},
        )
    )
    assert plan["ctx_use_kv_cache_manager_v2"] is False
    assert plan["gen_use_kv_cache_manager_v2"] is True
    assert pcfg.side_plan(plan, "gen")["use_kv_cache_manager_v2"] is True


def test_model_kv_shape_vocab_size(tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 129280,
            }
        )
    )
    assert pcfg.model_kv_shape(str(model_dir))["vocab_size"] == 129280


class TestRendezvousStaleness:
    """wait_for_addr must skip addr files stamped by a previous run."""

    def test_same_job_accepted(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        p = str(tmp_path / "rendezvous" / "ctx0_gen0.addr")
        rp.write_addr(p, {"host": "h", "port": 1, "key": "aa"})
        got = rp.wait_for_addr(p, timeout_s=2)
        assert got["job"] == "12345" and got["port"] == 1

    def test_stale_job_skipped_until_timeout(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "11111")
        p = str(tmp_path / "rendezvous" / "ctx0_gen0.addr")
        rp.write_addr(p, {"host": "h", "port": 1, "key": "aa"})  # stamped 11111
        monkeypatch.setenv("SLURM_JOB_ID", "22222")  # new run
        with pytest.raises(rp._Timeout):
            rp.wait_for_addr(p, timeout_s=2)

    def test_no_job_id_accepts_any(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "11111")
        p = str(tmp_path / "rendezvous" / "ctx0_gen0.addr")
        rp.write_addr(p, {"host": "h", "port": 1, "key": "aa"})
        monkeypatch.delenv("SLURM_JOB_ID")  # manual non-slurm run
        assert rp.wait_for_addr(p, timeout_s=2)["port"] == 1

    def test_write_addr_replaces_stale_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "11111")
        p = str(tmp_path / "rendezvous" / "ctx0_gen0.addr")
        rp.write_addr(p, {"host": "old", "port": 1, "key": "aa"})
        monkeypatch.setenv("SLURM_JOB_ID", "22222")
        rp.write_addr(p, {"host": "new", "port": 2, "key": "bb"})
        got = rp.wait_for_addr(p, timeout_s=2)
        assert got["host"] == "new" and got["job"] == "22222"


def test_wireup_timeout_derivation():
    plan = pcfg.resolve_plan(_disagg_yaml())  # ctx dep4 -> gen dep16
    assert plan["wireup_timeout_s"] == min(1800, 150 * 16)
    plan = pcfg.resolve_plan(
        _disagg_yaml(gen_extra={"tensor_parallel_size": 4})
    )
    assert plan["wireup_timeout_s"] == 600
    plan = pcfg.resolve_plan(_disagg_yaml(cache_transceiver_precheck={"wireup_timeout_s": 42}))
    assert plan["wireup_timeout_s"] == 42


def test_rid_tags_dense_within_session():
    """The C++ notification tag is rid & 0xFFF: rids must be dense within a
    (ctx, gen) session so tags cannot alias across reps/lengths."""
    plan = pcfg.resolve_plan(_disagg_yaml())  # n_pairs=16
    total_reps = plan["warmup_requests"] + plan["num_requests"]
    n_pairs = plan["n_pairs"]

    def session_rids(ctx_idx, gen_idx):
        out = []
        for li in range(2):
            for rep in range(total_reps):
                for pair in range(n_pairs):
                    seq = (li * total_reps + rep) * n_pairs + pair
                    out.append(rp.make_rid(ctx_idx, gen_idx, 2, seq))
        return out

    a = session_rids(0, 0)
    b = session_rids(1, 0)
    assert len(set(a)) == len(a) and len(set(b)) == len(b)
    assert not (set(a) & set(b))  # globally unique across sessions
    tags = [r & 0xFFF for r in a]
    assert len(set(tags)) == len(tags)  # no tag aliasing within a session


class TestMultiPeerOrchestration:
    """CPU-only end-to-end run of the 2-ctx x 1-gen session protocol.

    Exercises the exact multi-instance logic of the hardware "B" topology:
    real ZMQ sockets + HMAC frames + StatusRecorder + rendezvous files via
    the real PrecheckRunner/_serve_gen_peers/_drive_ctx_peers, with only the
    GPU transfer methods stubbed out.
    """

    class _FakeComm:
        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

        def bcast(self, obj, root=0):
            return obj

        def gather(self, obj, root=0):
            return [obj]

        def allgather(self, obj):
            return [obj]

    class _FakeParams:
        first_gen_tokens = [0]
        req_id = 1
        opaque_state = b"op"
        draft_tokens = None
        ctx_dp_rank = 0
        disagg_info_endpoint = None

    def _mk_runner(self, role, server_idx, plan, work_dir, monkeypatch, fail_ctx=False):
        import sys
        import types

        # PrecheckRunner.__init__ imports mpi4py only to ensure MPI init.
        monkeypatch.setitem(sys.modules, "mpi4py", types.SimpleNamespace(MPI=None))
        # The gen side converts wire params through tensorrt_llm bindings;
        # identity is fine here (params_to_wire is covered separately).
        monkeypatch.setattr(rp, "params_from_wire", lambda d: d)

        args = types.SimpleNamespace(server_idx=server_idx, work_dir=work_dir)
        side = pcfg.side_plan(plan, role)
        runner = rp.PrecheckRunner(args, plan, side, self._FakeComm())

        calls = {"chunks": 0}

        def ctx_run_chunk(peer_idx, li, req_len, rep, chunk):
            if fail_ctx:
                raise rp._TransferError("injected ctx failure")
            calls["chunks"] += 1
            return {p: self._FakeParams() for p in chunk}, {}

        runner.ctx_run_chunk = ctx_run_chunk
        runner.ctx_finish_chunk = lambda reqs: None
        runner.gen_run_chunk = (
            lambda peer_idx, li, req_len, rep, chunk, params: (True, "")
        )
        runner._calls = calls
        return runner

    def _run(self, tmp_path, monkeypatch, fail_ctx1=False):
        import threading

        monkeypatch.setenv("SLURM_JOB_ID", "777")
        # Publish loopback in the addr files: the real node hostname may not
        # resolve in sandboxed/CI environments, and everything is one process.
        monkeypatch.setenv("SLURMD_NODENAME", "127.0.0.1")
        cfg = _disagg_yaml(
            hardware={"gpus_per_node": 4, "num_ctx_servers": 2, "num_gen_servers": 1},
            cache_transceiver_precheck={
                "request_lengths": [32],
                "num_requests": 1,
                "warmup_requests": 1,
                "rendezvous_timeout_s": 30,
                "chunk_timeout_s": 30,
                "wireup_timeout_s": 0,
            },
        )
        plan = pcfg.resolve_plan(cfg)
        work = str(tmp_path)
        noop = lambda *a, **k: None  # noqa: E731 - signal.alarm needs main thread

        gen = self._mk_runner("gen", 0, plan, work, monkeypatch)
        ctxs = [
            self._mk_runner("ctx", i, plan, work, monkeypatch, fail_ctx=(fail_ctx1 and i == 1))
            for i in range(2)
        ]

        failures = []

        def rec(peer, exc):
            failures.append((peer, type(exc).__name__))

        threads = [
            threading.Thread(
                target=rp._serve_gen_peers, args=(c, plan, noop, noop, rec), daemon=True
            )
            for c in ctxs
        ]
        for t in threads:
            t.start()
        rp._drive_ctx_peers(
            gen, noop, noop, rp._make_peer_failure_recorder(gen, noop, {"what": "test"})
        )
        for t in threads:
            t.join(timeout=60)
            assert not t.is_alive(), "ctx serve thread wedged"
        return plan, gen, ctxs, failures

    def test_two_ctx_full_pass(self, tmp_path, monkeypatch):
        plan, gen, ctxs, failures = self._run(tmp_path, monkeypatch)
        assert not failures
        # gen recorded a PASS per (peer, req_len)
        assert {(c["peer"], c["status"]) for c in gen.recorder.cases} == {
            ("ctx_0", "PASS"),
            ("ctx_1", "PASS"),
        }
        # every ctx served the full schedule (reps x chunks) and got its
        # deferred done (PASS recorded only after done/bye completes)
        total_chunks = len(pcfg.chunks(plan)) * (
            plan["warmup_requests"] + plan["num_requests"]
        )
        for c in ctxs:
            assert c._calls["chunks"] == total_chunks
            assert [x["status"] for x in c.recorder.cases] == ["PASS"]

    def test_ctx_failure_isolated(self, tmp_path, monkeypatch):
        plan, gen, ctxs, failures = self._run(tmp_path, monkeypatch, fail_ctx1=True)
        # gen side: healthy peer unaffected, failing peer gets a clear verdict
        by_peer = {c["peer"]: c["status"] for c in gen.recorder.cases}
        assert by_peer == {"ctx_0": "PASS", "ctx_1": "TRANSFER_ERROR"}
        # ctx_1's own serve loop surfaced the failure (its peer is gen_0)
        assert ("gen_0", "_TransferError") in failures
        # ctx_0 served its full schedule and got the deferred done
        assert [c["status"] for c in ctxs[0].recorder.cases] == ["PASS"]


def test_status_env_snapshot_excludes_nixl(tmp_path, monkeypatch):
    """NIXL_* is not captured: the only such variable seen in practice is
    NIXL_VERSION, a stale NGC-base-image marker that misstates the version of
    the actually-linked library."""
    monkeypatch.setenv("NIXL_VERSION", "1.0.0")
    monkeypatch.setenv("NIXL_PLUGIN_DIR", "/opt/x")
    monkeypatch.setenv("UCX_TLS", "rc,cuda_copy")
    rec = rp.StatusRecorder(str(tmp_path), "gen", 0, is_leader=True)
    assert not any(k.startswith("NIXL_") for k in rec.env)
    assert rec.env["UCX_TLS"] == "rc,cuda_copy"  # behavioral vars still captured
