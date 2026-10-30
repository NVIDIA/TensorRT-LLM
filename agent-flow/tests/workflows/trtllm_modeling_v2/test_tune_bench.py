# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""bench.py: the warmup, a server of several ranks, and how a point's requests are summarised."""

import json
import signal
import subprocess
import sys

import pytest

from agent_flow.workflows.trtllm_modeling_v2.tune import bench
from agent_flow.workflows.trtllm_modeling_v2.tune.bench import (
    PER_REQUEST_BASIS,
    USER_RATE_BASIS,
    rank_launch,
    reap,
    summarize,
    warmup_shape,
)
from agent_flow.workflows.trtllm_modeling_v2.tune.records import (
    Measurement,
    NoiseFloor,
    PerfLedger,
    compare,
    relative,
)


def test_the_warmup_defaults_to_the_largest_point():
    assert warmup_shape([1, 128]) == (128, 256)


def test_explicit_warmup_values_win():
    assert warmup_shape([1, 128], concurrency=16, prompts=16) == (16, 16)
    assert warmup_shape([1, 128], concurrency=32) == (32, 64)


def test_one_rank_serves_without_a_launcher():
    assert rank_launch(1) == []


def test_several_ranks_exist_before_the_server_starts():
    argv = rank_launch(4, "/venv/bin/trtllm-llmapi-launch")
    mpirun = argv.index("mpirun")
    assert argv[mpirun : mpirun + 3] == ["mpirun", "-n", "4"]
    assert argv[-1] == "/venv/bin/trtllm-llmapi-launch"
    # The step's task variables are cleared, or every rank reads itself as rank 0 of 1.
    assert argv[0] == "env" and ["-u", "SLURM_PROCID"] == argv[1:3]
    assert "SLURM_NTASKS" in argv[:mpirun]


def test_a_rank_that_ignores_sigterm_is_killed_after_the_grace():
    ignore_term = (
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print(1, flush=True); time.sleep(60)"
    )
    stubborn = subprocess.Popen([sys.executable, "-c", ignore_term], stdout=subprocess.PIPE)
    stubborn.stdout.readline()  # the handler is installed
    polite = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    signalled = reap({stubborn.pid, polite.pid}, grace_s=1.0)
    assert stubborn.wait(timeout=10) == -signal.SIGKILL
    assert polite.wait(timeout=10) == -signal.SIGTERM
    # SIGTERM to both, then SIGKILL to the one still alive.
    assert sorted(signalled) == sorted([stubborn.pid, polite.pid, stubborn.pid])


def test_nothing_alive_is_nothing_to_reap():
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait()
    assert reap({done.pid}, grace_s=0.1) == []


def test_every_device_a_server_will_use_is_checked_clean(monkeypatch):
    asked = []

    def fake_run(argv, **kwargs):
        asked.append(argv[argv.index("-i") + 1])
        return subprocess.CompletedProcess(argv, 0, stdout="12\n263451\n", stderr="")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    # A leftover on the second device is as dirty as one on the first.
    assert bench.device_memory_used_mib(2) == 263451
    assert asked == ["0,1"]


def test_only_the_jobs_own_devices_are_asked_for_holders(monkeypatch):
    asked = []

    def fake_run(argv, **kwargs):
        asked.append(argv[argv.index("-i") + 1])
        return subprocess.CompletedProcess(argv, 0, stdout="4242\n", stderr="")

    # A job given devices 2 and 3 of a shared node: a holder on 0 or 1 is another job's.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    assert bench.device_holders(2) == {4242}
    assert asked == ["2,3"]


# ------------------------------------------------------------------------- tok/s/user
# benchmark_serving's records of dep4.con128 (512 requests of 8192 + 1024 tokens) on one node,
# three trees whose dep4 forward is the same (land-allreduce-oneshot, perf-r2): A and A' admitted
# the waves staggered, B admitted each wave as a block.
STAGGERED = {"duration": 55.38251158403, "user_throughput": 76.26265758790811,
             "mean_ttft_ms": 1691.3264435632982, "median_ttft_ms": 1026.0670560237486,
             "mean_tpot_ms": 11.852110082477765, "mean_e2el_ms": 13816.035057938052,
             "std_e2el_ms": 2194.217347309817}  # fmt: skip
STAGGERED_AGAIN = {"duration": 55.37650179199409, "user_throughput": 76.24971123304607,
                   "mean_ttft_ms": 1690.4002388757817, "median_ttft_ms": 1025.0757600006182,
                   "mean_tpot_ms": 11.85119365915167, "mean_e2el_ms": 13814.171352187941,
                   "std_e2el_ms": 2190.7803149266956}  # fmt: skip
BLOCKED = {"duration": 55.48618304001866, "user_throughput": 74.44669489238754,
           "mean_ttft_ms": 3328.2823285002223, "median_ttft_ms": 3248.881040024571,
           "mean_tpot_ms": 10.281713920760131, "mean_e2el_ms": 13846.475669437837,
           "std_e2el_ms": 1109.5208393461435}  # fmt: skip


def _record(con: int, prompts: int, osl: int, **fields) -> dict:
    """A ``--save-result`` record of benchmark_serving (no ``--save-detailed`` lists)."""
    record = {"completed": prompts, "num_prompts": prompts, "max_concurrency": con,
              "total_input_tokens": prompts * 8192, "total_output_tokens": prompts * osl,
              "p99_ttft_ms": 6600.0, "p99_tpot_ms": 13.0, "mean_itl_ms": 11.0, **fields}  # fmt: skip
    record["output_throughput"] = record["total_output_tokens"] / record["duration"]
    record["total_token_throughput"] = (
        record["total_input_tokens"] + record["total_output_tokens"]
    ) / record["duration"]
    record["request_throughput"] = prompts / record["duration"]
    return record


def _label(root, name: str, points: dict[int, dict], world: int = 4) -> Measurement:
    """A label dir as bench.py finalizes it: the point records, summary.json and meta.json."""
    out = root / name
    out.mkdir(parents=True)
    for con, record in points.items():
        (out / bench.point_file(con)).write_text(json.dumps(record))
    rows = summarize(out, sorted(points), world)
    (out / "summary.json").write_text(json.dumps(rows))
    (out / "meta.json").write_text(json.dumps({"label": name, "hostname": "node0"}))
    return Measurement.load(out)


def test_tok_s_user_is_the_tokens_over_the_summed_latency_and_says_so(tmp_path):
    out = tmp_path / "dep4"
    out.mkdir()
    (out / "con128.json").write_text(json.dumps(_record(128, 512, 1024, **BLOCKED)))
    (row,) = summarize(out, [128], 4)
    assert row["tok_s_user"] == pytest.approx(524288 / (512 * 13.846475669437837))
    assert row["tok_s_user_basis"] == USER_RATE_BASIS
    assert row["tok_s_user_per_request"] == BLOCKED["user_throughput"]
    assert row["output_tok_s_gpu"] == pytest.approx(9448.982994953254 / 4)


def test_how_the_waves_were_admitted_moves_the_per_request_mean_not_the_users_rate(tmp_path):
    """Same throughput, same mean e2e latency: the per-request mean read B 2.4% low, a LOSS."""
    a = _label(tmp_path, "A", {128: _record(128, 512, 1024, **STAGGERED)})
    a2 = _label(tmp_path, "A2", {128: _record(128, 512, 1024, **STAGGERED_AGAIN)})
    b = _label(tmp_path, "B", {128: _record(128, 512, 1024, **BLOCKED)})
    per_request = [m.points[128]["tok_s_user_per_request"] for m in (a, b, a2)]
    assert relative(per_request[1], per_request[0]) < -0.023
    assert abs(relative(b.points[128]["tok_s_user"], a.points[128]["tok_s_user"])) < 0.003
    noise = NoiseFloor.from_measurements([a, a2])
    cmp = compare(b, Measurement.mean([a, a2]), noise, min_gain=0.02, noise_factor=3.0)
    assert cmp.verdict == "NEUTRAL", cmp.reason


def test_at_concurrency_one_the_users_rate_is_the_requests_own(tmp_path):
    """tp1.con1 of the same round: 16 requests one after another."""
    one = _record(1, 16, 1024, duration=57.52999222400831, user_throughput=284.81107071351016,
                  mean_ttft_ms=132.09138399906806, mean_tpot_ms=3.38543655132042,
                  mean_e2el_ms=3595.3929759998573)  # fmt: skip
    m = _label(tmp_path, "tp1", {1: one}, world=1)
    assert abs(relative(m.points[1]["tok_s_user"], one["user_throughput"])) < 1e-4


def test_a_point_without_e2e_latency_keeps_the_per_request_mean_and_says_so(tmp_path):
    record = _record(16, 64, 1024, **BLOCKED)
    del record["mean_e2el_ms"]
    out = tmp_path / "tp1"
    out.mkdir()
    (out / "con016.json").write_text(json.dumps(record))
    (row,) = summarize(out, [16], 1)
    assert row["tok_s_user"] == BLOCKED["user_throughput"]
    assert row["tok_s_user_basis"] == PER_REQUEST_BASIS


def test_a_label_summarised_before_the_basis_was_recorded_is_restated(tmp_path):
    """Old controls and new candidates of one campaign are compared on one basis."""
    new = _label(tmp_path, "new", {128: _record(128, 512, 1024, **BLOCKED)})
    old = tmp_path / "old"
    old.mkdir()
    (row,) = summarize(new.path, [128], 4)
    for key in ("tok_s_user_basis", "tok_s_user_per_request"):
        del row[key]
    row["tok_s_user"] = BLOCKED["user_throughput"]  # what bench.py wrote there before
    (old / "summary.json").write_text(json.dumps([row]))
    (old / "meta.json").write_text(json.dumps({"label": "old"}))
    restated = Measurement.load(old).points[128]
    assert restated["tok_s_user"] == pytest.approx(new.points[128]["tok_s_user"])
    assert restated["tok_s_user_per_request"] == BLOCKED["user_throughput"]
    assert restated["tok_s_user_basis"] == USER_RATE_BASIS
    # A row without the fields to restate it is read as written.
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "summary.json").write_text(json.dumps([{"concurrency": 1, "tok_s_user": 99.0}]))
    (bare / "meta.json").write_text("{}")
    assert Measurement.load(bare).points[1]["tok_s_user"] == 99.0
    entry = PerfLedger(tmp_path / "ledger.jsonl").measurement(Measurement.load(old))
    assert entry["points"]["128"]["tok_s_user_per_request"] == BLOCKED["user_throughput"]
