# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tune campaign regression tests."""

from __future__ import annotations

import hashlib
import json
import math
import shlex
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.tune import sol
from agent_flow.workflows.trtllm_modeling_v2.tune.inert import inert_members
from agent_flow.workflows.trtllm_modeling_v2.tune.records import (
    PARTS_FILE,
    Measurement,
    NoiseFloor,
    compare,
)
from agent_flow.workflows.trtllm_modeling_v2.tune.spec import TuneSpec
from agent_flow.workflows.trtllm_modeling_v2.tune.split import classify
from agent_flow.workflows.trtllm_modeling_v2.tune.stage import TuneStage

from .test_tune_stage import _microbench


def test_a_constraint_is_identified_by_site_and_key_not_measured_value():
    def run(label: str, tpot: float, user: float = 100.0) -> Measurement:
        row = {
            "tok_s_user": user,
            "output_tok_s_gpu": 10.0,
            "mean_tpot_ms": tpot,
            "mean_ttft_ms": 50.0,
        }
        part = Measurement(label="tp4", path=Path(label), points={1: row}, meta={})
        return Measurement(
            label=label,
            path=Path(label),
            points={},
            meta={},
            parts={"tp4": part},
            constraints={"tp4": {"tpot_ms_max": 10.0}},
        )

    noise, bands = (NoiseFloor(default=0.01), dict(min_gain=0.03, noise_factor=2.0))
    faster = run("b", 12.0, user=110.0)
    assert compare(faster, run("a", 11.0), noise, **bands).verdict == "WIN"
    assert classify(faster, run("a", 11.0), noise, **bands).broken == ()
    assert compare(faster, run("a", 9.0), noise, **bands).verdict == "LOSS"
    assert classify(faster, run("a", 9.0), noise, **bands).broken


def test_the_perf_block_runs_without_a_stock_reference_or_repeats_and_says_so():
    from types import SimpleNamespace

    from agent_flow.workflows.trtllm_modeling_v2.spec.lint import perf_rules
    from agent_flow.workflows.trtllm_modeling_v2.tune.schema import validate_perf_block

    perf = {"reference": "stock", "noise_repeats": 2, "goal": {"ratio_to_reference": 1.1}}
    (error,) = validate_perf_block({"perf": perf})
    assert "ratio_to_baseline and/or pct_of_sol" in error
    warned = [f.message for f in perf_rules(SimpleNamespace(task={"perf": perf}))]
    assert warned == [
        "`perf.noise_repeats` is ignored: the baseline is measured once",
        "`perf.reference` is ignored: no stock reference is measured; a goal is a ratio to "
        "the baseline",
    ]
    assert validate_perf_block({"perf": {"goal": {"ratio_to_baseline": 1.1}}}) == []
    campaign = {"checkpoint_keeps": 0, "micro_noise": -1, "logits": {"min_top1": 2, "k": 1}}
    assert validate_perf_block({"perf": {"campaign": campaign}}) == [
        "perf.campaign.micro_noise: a fraction >= 0 (0.05 is 5%)",
        "perf.campaign.checkpoint_keeps: a positive number of keeps",
        "perf.campaign.logits.k: unknown key",
        "perf.campaign.logits.min_top1: a number in [0, 1]",
    ]


def test_a_perf_config_whose_cuda_graphs_stop_short_of_its_load_is_a_warning(tmp_path):
    """128 requests over 4 attention-DP ranks: 32 per rank on average, up to 48 as they finish."""
    from agent_flow.workflows.trtllm_modeling_v2.spec.lint import WARN, perf_rules

    from .test_tune_stage import FakeRunner, _effect, _repo

    adp = {"modeling_v2": "require", "tensor_parallel_size": 4, "enable_attention_dp": True}
    configs = {
        "short": {**adp, "cuda_graph_config": {"enable_padding": True, "max_batch_size": 32}},
        "covered": {**adp, "cuda_graph_config": {"enable_padding": True, "max_batch_size": 48}},
        "eager": {**adp, "cuda_graph_config": None},
        "bounded": {**adp, "max_batch_size": 16, "cuda_graph_config": {"max_batch_size": 16}},
    }
    entries = []
    for name, options in configs.items():
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(options))
        entries.append(
            {
                "name": name,
                "workload": {"isl": 1024, "osl": 128, "concurrency": [1, 128]},
                "identity_configs": [str(tmp_path / f"{name}.yaml")],
            }
        )
    perf = {"configs": entries, "spools": ["s1", "s2", "s3", "s4"], "env_script": "/env.sh"}
    task = {"trtllm_repo_path": str(_repo(tmp_path)), "checkpoint_path": "/ckpt", "perf": perf}
    found = list(perf_rules(SimpleNamespace(task=task)))
    assert [(f.rule, f.severity) for f in found] == [("perf-cuda-graph-coverage", WARN)] * 2
    short, eager = (f.message for f in found)
    assert short.startswith("perf config `short` captures CUDA graphs up to batch 32")
    assert "32 live requests per rank on average, and a rank holds up to 48" in short
    # Capture what one rank holds, not the whole concurrency.
    assert "max_batch_size: 48}`, with the engine's `max_batch_size`" in found[0].fix
    assert eager.startswith("perf config `eager` turns CUDA graphs off")
    # The campaign journals it when it starts, and measures as the spec says.
    perf["campaign"] = {"enabled": False}
    spec = TuneSpec.from_task(task, tmp_path / "ws", target="t/")
    logs = []
    stage = TuneStage(spec, {}, dispatch=lambda *a: "", checkpoint=lambda: None,
                      runner_factory=lambda s, p: FakeRunner(s, p, _effect), poll_s=0,
                      log=logs.append)  # fmt: skip
    assert stage.run() == "tuned"
    warned = [m for m in logs if m.startswith("[tune] warning: ")]
    assert [m.removeprefix("[tune] warning: ").split(". Fix: ")[0] for m in warned] == [
        short,
        eager,
    ]


AGENTX = {"client": "agentx", "isl": 8192, "osl": 1024, "concurrency": [1], "duration_s": 600}


def test_an_agentx_config_is_validated_resolved_and_replayed_by_aiperf(tmp_path, monkeypatch):
    from dataclasses import replace

    from agent_flow.workflows.trtllm_modeling_v2.tune.harness import bench
    from agent_flow.workflows.trtllm_modeling_v2.tune.measure import _differs, evidence_key
    from agent_flow.workflows.trtllm_modeling_v2.tune.schema import validate_perf_block

    agentx = {**AGENTX, "extra_args": ["--random-seed", "42"]}
    configs = [
        {"name": "tp4", "workload": agentx, "constraint": {"kv_hit_pct_min": 95}},
        {
            "name": "dep4",
            "workload": {"isl": 8, "osl": 8, "concurrency": [128], "duration_s": 60},
            "constraint": {"kv_hit_pct_min": 90},
        },
    ]
    assert validate_perf_block({"perf": {"configs": configs}}) == [
        "perf.configs[1].workload.duration_s: only with client: agentx",
        "perf.configs[1].constraint.kv_hit_pct_min: only with client: agentx, whose points "
        "report it",
    ]
    bad = [{"name": "a", "workload": {**AGENTX, "client": "trace"}},
           {"name": "b", "workload": {**AGENTX, "extra_args": "--x"}}]  # fmt: skip
    assert validate_perf_block({"perf": {"configs": bad, "workload": {"client": "agentx"}}}) == [
        "perf.workload.client: another client runs only in a perf.configs entry",
        "perf.configs[0].workload.client: one of ['random', 'agentx']",
        "perf.configs[1].workload.extra_args: a list of aiperf arguments, each a string (quote "
        "a number: '42')",
    ]

    task = {"checkpoint_path": str(tmp_path), "perf": {"configs": configs[:1]}}
    spec = TuneSpec.from_task(task, tmp_path / "ws")
    (part,) = spec.parts
    hf_home = str((tmp_path / "ws" / "tune" / "agentx-cache" / "hf").resolve())
    assert (part.client, part.agentx) == (
        "agentx",
        {"dataset": bench.AGENTX_DATASET, "duration_s": 600, "extra_args": ["--random-seed", "42"],
         "hf_home": hf_home},
    )  # fmt: skip
    assert "AgentX trace replay" in spec.describe_workload()
    argv = shlex.split(spec.bench().command("t-baseline", tmp_path / "data", []).splitlines()[1])
    assert argv[argv.index("--client") + 1] == "agentx"
    assert json.loads(argv[argv.index("--agentx") + 1]) == part.agentx
    # Where the corpus is cached is not what is measured; the client is.
    key = evidence_key(spec.bench(), [])
    moved = replace(part, agentx={**part.agentx, "hf_home": str(tmp_path / "elsewhere")})
    assert evidence_key(replace(spec.bench(), parts=(moved,)), []) == key
    random = replace(part, client="random", agentx={})
    assert evidence_key(replace(spec.bench(), parts=(random,)), []) != key
    # A measurement on disk is reused only if it ran the same client.
    meta = tmp_path / "old" / "tp4" / "meta.json"
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({"isl": 8192, "osl": 1024, "concurrencies": [1]}))
    assert _differs(meta.parents[1], spec.bench(), [], None).startswith(
        "was measured with another client than agentx"
    )
    # One scored before AgentX's own scoring is measured again, not compared.
    meta.write_text(json.dumps({"client": "agentx", "agentx": moved.agentx}))
    assert _differs(meta.parents[1], spec.bench(), [], None).startswith(
        "was measured with another client than agentx"
    )
    meta.write_text(
        json.dumps({"client": "agentx", "agentx": moved.agentx, "agentx_score": bench.AGENTX_SCORE})
    )
    assert _differs(meta.parents[1], spec.bench(), [], None) == ""

    seen = {}
    monkeypatch.setattr(
        bench, "supervise_client", lambda argv, *_, **kw: seen.update(argv=argv, **kw)
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    bench.run_agentx_client(
        aiperf=Path("aiperf"), model="m", port=8000, concurrency=4, settings=dict(part.agentx),
        artifact_dir=tmp_path / "agentx_con004", server=None, log_path=tmp_path / "server.log",
        client_log=tmp_path / "client.log",
    )  # fmt: skip
    argv = seen["argv"]
    assert argv[:4] == ["aiperf", "profile", "--scenario", "agentx"]
    assert argv[argv.index("--concurrency") + 1] == "4"
    # Shorter than aiperf's safe duration, so overridden; the spec's own flags come last.
    assert argv[argv.index("--benchmark-duration") + 1] == "600" and "--unsafe-override" in argv
    assert argv[-2:] == ["--random-seed", "42"]
    assert seen["env"]["HF_HOME"] == hf_home and "HF_HUB_OFFLINE" not in seen["env"]


def test_an_aiperf_export_becomes_a_point_and_its_kv_floor_is_judged(tmp_path):
    """Aiperf's csv in benchmark_serving's keys; a hit rate under the floor, or none, breaks it."""
    from agent_flow.workflows.trtllm_modeling_v2.tune.harness import bench

    artifact = tmp_path / "agentx_con001"
    artifact.mkdir()
    (artifact / "profile_export_aiperf.csv").write_text(
        "Metric,avg,min,max,sum,p1,p5,p10,p25,p50,p75,p90,p95,p99,std\n"
        "Time to First Token (ms),120.5,,,,,,,,100.0,,180.0,,250.0,\n"
        "Inter Token Latency (ms),6.25,,,,,,,,6.0,,7.0,,9.5,\n"
        "Request Latency (ms),9000.0,,,,,,,,8800.0,,9900.0,,12000.0,\n"
        "Input Sequence Length (tokens),200000.0,,,4000000,,,,,,,,,,\n"
        "Output Sequence Length (tokens),1400.0,,,28000,,,,,,,,,,\n"
        "\n"
        "Metric,Value\n"
        "Request Count,20\n"
        "Benchmark Duration (sec),600.0\n"
        "Request Throughput (requests/sec),0.0333\n"
        "Output Token Throughput (tokens/sec),46.7\n"
        'Total Usage Prompt Cache Read Tokens (tokens),"3,880,000"\n'
    )
    point = bench.agentx_point(artifact, {"label": "t", "concurrency": 1})
    assert point["completed"] == 20 and point["total_output_tokens"] == 28000
    assert (point["mean_ttft_ms"], point["p99_tpot_ms"], point["median_e2el_ms"]) == (
        120.5,
        9.5,
        8800.0,
    )
    assert round(point["kv_hit_pct"], 1) == 97.0

    (tmp_path / bench.point_file(1)).write_text(json.dumps(point))
    [row] = bench.summarize(tmp_path, [1], world_size=4)
    assert (row["kv_hit_pct"], row["isl_avg"], row["mean_tpot_ms"]) == (
        point["kv_hit_pct"],
        200000.0,
        6.25,
    )
    assert row["tok_s_user"] == 28000 / (20 * 9.0)

    def run(hit):
        part = Measurement(
            label="tp4", path=tmp_path, points={1: {**row, "kv_hit_pct": hit}}, meta={}
        )
        return Measurement(
            label="t", path=tmp_path, points={}, meta={}, parts={"tp4": part},
            constraints={"tp4": {"kv_hit_pct_min": 95.0}},
        )  # fmt: skip

    assert run(row["kv_hit_pct"]).violations() == []
    assert run(90.0).violations() == ["tp4@con1: kv_hit_pct 90.0 < 95.0"]
    # Nothing shows the floor held.
    assert run(None).violations() == ["tp4@con1: no kv_hit_pct (>= 95.0)"]


def test_an_agentx_point_ranks_as_inferencex_scores_it_and_its_client_is_pinned(tmp_path):
    """P90 full-response ITL over the profiling requests; tokens over their window; one client."""
    from agent_flow.workflows.trtllm_modeling_v2.tune.harness import bench

    artifact = tmp_path / "agentx_con002"
    (artifact / "export").mkdir(parents=True)
    (artifact / "profile_export_aiperf.csv").write_text(
        "Metric,avg,min,max,sum,p1,p5,p10,p25,p50,p75,p90,p95,p99,std\n"
        "Request Latency (ms),9000.0,,,,,,,,8800.0,,9900.0,,12000.0,\n"
        "Output Sequence Length (tokens),100.0,,,400,,,,,,,,,,\n"
        "\nMetric,Value\nRequest Count,4\nOutput Token Throughput (tokens/sec),99.0\n"
    )

    def request(start_s, end_s, full, legacy, isl=1000, osl=100, **extra):
        metrics = {
            "full_response_inter_token_latency": {"value": full},
            "inter_token_latency": {"value": legacy},
            "input_sequence_length": {"value": isl},
            "output_sequence_length": {"value": osl},
        }
        metadata = {"request_start_ns": int(start_s * 1e9), "request_end_ns": int(end_s * 1e9)}
        return json.dumps({"metrics": metrics, "metadata": {**metadata, **extra}})

    lines = [
        request(0, 10, 10.0, 8.0, benchmark_phase="warmup"),  # dropped: warmup
        request(10, 20, 20.0, 9.0, benchmark_phase="profiling"),
        request(12, 30, 30.0, 9.0, benchmark_phase="profiling"),
        request(15, 40, 40.0, 9.0, benchmark_phase="profiling"),
        request(20, 50, 50.0, 9.0, benchmark_phase="profiling"),
        json.dumps({"error": "timeout", "metadata": {"benchmark_phase": "profiling"}}),
    ]
    (artifact / "export" / "profile_export.jsonl").write_text("\n".join(lines) + "\n")
    point = bench.agentx_point(artifact, {"label": "t", "concurrency": 2})
    # P90 of 20, 30, 40, 50 ms, interpolated: 47 ms; the window runs from 10 s to 50 s.
    assert point["agentx_records"] == 4 and point["agentx_itl_source"] == "full-response"
    assert math.isclose(point["agentx_p90_itl_ms"], 47.0)
    assert math.isclose(point["agentx_window_s"], 40.0)
    (tmp_path / bench.point_file(2)).write_text(json.dumps(point))
    [row] = bench.summarize(tmp_path, [2], world_size=4)
    assert math.isclose(row["tok_s_user"], 1000.0 / 47.0)
    assert row["tok_s_user_basis"] == "AgentX: 1 / P90 full-response inter-token latency"
    assert math.isclose(row["output_tok_s_gpu"], 400 / 40.0 / 4)
    assert math.isclose(row["total_tok_s_gpu"], (4000 + 400) / 40.0 / 4)
    assert row["throughput_basis"] == bench.AGENTX_THROUGHPUT_BASIS
    # Without the per-request export the point keeps the csv's numbers.
    (artifact / "export" / "profile_export.jsonl").unlink()
    assert "agentx_p90_itl_ms" not in bench.agentx_point(artifact, {})

    # Every install of one campaign resolves the client's dependencies as of its first.
    shared = tmp_path / "agentx_client_cache"
    first = bench.resolved_as_of(shared)
    (shared / bench.AGENTX_RESOLVED_FILE).write_text("2026-11-01T00:00:00Z\n")
    assert first and bench.resolved_as_of(shared) == "2026-11-01T00:00:00Z"


def test_an_agentx_install_step_that_stalls_fails_before_the_jobs_wall(tmp_path, monkeypatch):
    """Each install step is bounded: a stalled uv fails both attempts with the diagnostic."""
    from agent_flow.workflows.trtllm_modeling_v2.tune.harness import bench

    uv = tmp_path / "uv"
    uv.write_text("#!/bin/sh\nsleep 30\n")
    uv.chmod(0o755)
    monkeypatch.setattr(bench.shutil, "which", lambda name: str(uv))
    monkeypatch.setattr(bench, "AGENTX_INSTALL_STEP_S", 0.2)
    settings = {"hf_home": str(tmp_path / "cache" / "hf")}
    log = tmp_path / "client.log"
    started = time.monotonic()
    with pytest.raises(SystemExit, match="did not install"):
        bench.agentx_client(settings, log, tmp_path / "freeze.txt")
    assert time.monotonic() - started < 10
    assert log.read_text().count("stopped after 0.2 s") == 2


_NOISE = NoiseFloor(spread={}, samples=0, default=0.01)


def _curve(label, *, user=100.0, gpu=1000.0, ttft=50.0, ttft_hi=900.0):
    return Measurement(
        label=label,
        path=Path(label),
        points={
            1: {"tok_s_user": user, "output_tok_s_gpu": gpu / 10, "mean_ttft_ms": ttft},
            128: {"tok_s_user": user / 4, "output_tok_s_gpu": gpu, "mean_ttft_ms": ttft_hi},
        },
        meta={},
    )


def test_a_faster_curve_that_slows_the_first_token_loses():
    cmp = compare(
        _curve("b", user=110.0, ttft=60.0), _curve("a"), _NOISE, min_gain=0.03, noise_factor=2.0
    )
    assert cmp.verdict == "LOSS"
    assert cmp.losses == ("con1.mean_ttft_ms",)
    assert "con1.mean_ttft_ms +20.0% (> +10%)" in cmp.reason
    assert cmp.ttft == {"con1.mean_ttft_ms": pytest.approx(0.2)}
    assert "con1.mean_ttft_ms" not in cmp.deltas


RULE = {"min_gain": 0.02, "noise_factor": 3.0}
MEMBERS = {"tp1": {"tensor_parallel_size": 1}, "tp2": {"tensor_parallel_size": 2}}
CONS = (1, 16)
MARKER = "partial sums reduce through a workspace"


def _bind(slots: int, ranks: int) -> str:
    return f"modeling_v2 x: bound Pool #2: {slots} recurrent slots, 32 states per slot\n" * ranks


def _write(root: Path, label: str, scale: dict, logs: dict, host: str = "h1") -> Measurement:
    """A measurement over :data:`MEMBERS`: ``scale[site]`` multiplies both metrics at that site."""
    for name, config in MEMBERS.items():
        part = root / label / name
        part.mkdir(parents=True, exist_ok=True)
        rows = []
        for con in CONS:
            g = scale.get(f"{name}.con{con}", 1.0)
            rows.append(
                {
                    "concurrency": con,
                    "tok_s_user": 100.0 * g,
                    "output_tok_s_gpu": 1000.0 * g,
                    "mean_tpot_ms": 10.0,
                    "mean_ttft_ms": 100.0,
                }
            )
        (part / "summary.json").write_text(json.dumps(rows))
        meta = {"label": name, "config": {"modeling_v2": "require", **config}, "hostname": host}
        (part / "meta.json").write_text(json.dumps(meta))
        (part / "server.log").write_text(logs.get(name, ""))
    (root / label / PARTS_FILE).write_text(json.dumps({"parts": list(MEMBERS), "constraints": {}}))
    (root / label / "meta.json").write_text(json.dumps({"label": label, "hostname": host}))
    (root / label / "summary.json").write_text("[]")
    return Measurement.load(root / label)


def test_a_member_without_the_marker_and_with_the_controls_pools_is_inert(tmp_path):
    ctrl = _write(tmp_path, "a", {}, {"tp1": _bind(1334, 1), "tp2": _bind(3461, 2)})
    cand = _write(
        tmp_path,
        "b",
        {"tp1.con16": 0.988, "tp2.con1": 1.15},
        {"tp1": _bind(1334, 1), "tp2": f"{MARKER}\n" * 2 + _bind(3450, 2)},
    )
    inert = inert_members([cand], [ctrl], marker=MARKER)
    assert list(inert) == ["tp1"] and "recurrent slots [1334] as the control's" in inert["tp1"]
    noise = NoiseFloor(spread={k: 0.0001 for k in ctrl.headline()})
    cmp = compare(cand, ctrl, noise, inert=inert, **RULE)
    assert cmp.verdict == "WIN" and cmp.inert_losses == ("tp1.con16",)
    assert "tp1.con16" not in str(cmp.judged())


def _inputs():
    work = {"flops": 4, "bytes": 1, "assumptions": "synthetic same-scope work"}
    return {
        "hardware": {
            "compute_flops_s": {"value": 2000, "source": "synthetic arithmetic fixture"},
            "memory_bytes_s": {"value": 1000, "source": "synthetic arithmetic fixture"},
            "assumptions": "one synthetic processor, dense arithmetic, explicit memory boundary",
        },
        "regions": [
            {
                "id": "layer/decode/shape/rank0",
                **work,
                "measured_ms": 8,
                "measurement_source": "synthetic timing",
            }
        ],
        "headline": {
            "con1.tok_s_user": {**work, "output_tokens": 1},
            "con1.output_tok_s_gpu": {**work, "output_tokens": 1, "gpu_count": 1},
        },
    }


def _baseline(value=400):
    return Measurement("base", Path("."), {1: {"tok_s_user": value, "output_tok_s_gpu": value}}, {})


def test_request_command_produces_valid_output_and_goal(tmp_path):
    spec = TuneSpec.from_task({"perf": {}}, tmp_path, target="target")
    stage = TuneStage(spec, {}, dispatch=lambda *args: "", checkpoint=lambda: None)
    request = stage.requests.sol_request(_baseline()).read_text()
    command = request.split("```sh\n", 1)[1].split("\n```", 1)[0]
    args = shlex.split(command)
    Path(args[args.index("--input") + 1]).write_text(yaml.safe_dump(_inputs()))
    subprocess.run([sys.executable, *args[1:]], check=True)
    output = Path(args[args.index("--output") + 1])
    ceilings, errors = sol.load_sol(output, _baseline())
    assert not errors and ceilings.reached(_baseline(), 0.8)
    assert ceilings.regions[0]["headroom_ms"] == 6
    assert not ceilings.reached(_baseline(501), 0.8)
    assert "internal-perf-sol-analysis" not in request + stage.requests.sol_ask("roadmap")
    region_output = tmp_path / "roadmap-sol.json"
    subprocess.run(
        [sys.executable, args[1], "--input", args[3], "--output", str(region_output)], check=True
    )
    assert json.loads(region_output.read_text())["regions"] == ceilings.regions
    raw = yaml.safe_load(output.read_text())
    raw["ceilings"]["con1.tok_s_user"]["value"] *= 2
    output.write_text(yaml.safe_dump(raw))
    assert sol.load_sol(output, _baseline())[0] is None


def test_missing_inputs_never_fabricate_bounds_or_stop(tmp_path):
    output = tmp_path / "sol.yaml"

    def never_reached(result):
        output.write_text(yaml.safe_dump(result))
        ceilings, errors = sol.load_sol(output, _baseline())
        assert not errors and (not ceilings.reached(_baseline(), 0.01))
        return ceilings

    inputs = _inputs()
    inputs["hardware"].pop("compute_flops_s")
    result = sol.calculate(inputs)
    assert not result["ceilings"] and result["regions"][0]["headroom_ms"] is None
    assert "unknown" in sol.render_table(never_reached(result), [_baseline()])
    inputs = _inputs()
    inputs["headline"].pop("con1.output_tok_s_gpu")
    never_reached(sol.calculate(inputs))
    inputs = _inputs()
    inputs["headline"]["con1.tok_s_user"]["overhead"] = {
        "ms": 1,
        "source": "synthetic trace",
        "assumption": "exposed and unchanged",
    }
    ceilings = never_reached(sol.calculate(inputs))
    assert "conditional" in ceilings.unknown["con1.tok_s_user"]


def _analysis(user=4.0, gpu=4.0, memory=1000, region_bytes=1, counted=None):
    """``_inputs`` with each headline's FLOPs, the memory capacity and the region's bytes set."""
    inputs = deepcopy(_inputs())
    inputs["headline"]["con1.tok_s_user"]["flops"] = user
    inputs["headline"]["con1.output_tok_s_gpu"]["flops"] = gpu
    inputs["hardware"]["memory_bytes_s"]["value"] = memory
    inputs["regions"][0]["bytes"] = region_bytes
    if counted is not None:
        inputs["regions"][0]["measured_bytes"] = counted
    return sol.calculate(inputs)


def test_analyses_that_agree_give_the_highest_ceiling_and_the_rest_is_ruled_or_unknown():
    from agent_flow.workflows.trtllm_modeling_v2.tune.sol_consensus import read_rulings, reconcile

    base = _baseline().headline()
    analyses = {"analytic": _analysis(4), "code": _analysis(4.4, 8), "profile": _analysis(7, 8)}
    c = reconcile(analyses, base)
    user = c.report["headline.con1.tok_s_user"]
    assert user["status"] == "agreed" and user["chosen"] == "analytic"
    assert user["below_baseline"] == ["profile"]
    assert c.inputs["headline"]["con1.tok_s_user"]["assumptions"].startswith("analytic: ")
    assert c.disputes == [
        {
            "item": "headline.con1.output_tok_s_gpu",
            "values": {"analytic": 500.0},
            "why": "unconfirmed: only analytic models it",
        }
    ]
    assert c.report["regions"]["agreed"] == 1 and len(c.inputs["regions"]) == 1
    rulings, problems = read_rulings(
        {
            "rulings": [
                {
                    "item": "headline.con1.output_tok_s_gpu",
                    "analysis": "analytic",
                    "evidence": "recounted",
                },
                {"item": "headline.con1.tok_s_user", "analysis": "code", "evidence": "x"},
                {"item": "headline.con1.output_tok_s_gpu", "analysis": "code", "evidence": "x"},
            ]
        },
        c.disputes,
    )
    assert rulings == {"headline.con1.output_tok_s_gpu": "analytic"} and len(problems) == 2
    ruled = reconcile(analyses, base, rulings=rulings)
    assert sol.calculate(ruled.inputs)["ceilings"]["con1.output_tok_s_gpu"]["value"] == 500
    apart = reconcile({"analytic": _analysis(4), "code": _analysis(3)}, base)
    assert [d["item"] for d in apart.disputes] == ["headline.con1.tok_s_user"]
    assert "differ by 33%" in apart.unknown["con1.tok_s_user"]
    assert "con1.tok_s_user" not in apart.inputs["headline"]


def test_the_stage_asks_each_method_once_and_one_adjudication_for_what_they_dispute(tmp_path):
    from dataclasses import replace

    from .test_tune_stage import FakeRunner, _budget_spec, _effect, _write_sol

    spec = replace(_budget_spec(tmp_path), sol={"enabled": True})
    turns = []

    def dispatch(role, what, prompt):
        turns.append(what)
        if what == "sol-adjudicate":
            rulings = [
                {"item": "headline.con1.tok_s_user", "analysis": "code", "evidence": "re-counted"},
                {"item": "headline.con128.tok_s_user", "analysis": None, "evidence": "none holds"},
            ]
            (spec.workdir / "sol" / "rulings.yaml").write_text(yaml.safe_dump({"rulings": rulings}))
        else:
            method = what.removeprefix("sol-")
            _write_sol(spec.workdir / "sol" / method, work=20 if method == "profile" else 10)
        return "done"

    stage = TuneStage(
        spec,
        {},
        dispatch=dispatch,
        checkpoint=lambda: None,
        poll_s=0,
        runner_factory=lambda spools, prefix: FakeRunner(spools, prefix, _effect),
    )
    assert stage.ground()
    stage._sol()
    assert sorted(turns) == ["sol-adjudicate", "sol-analytic", "sol-code", "sol-profile"]
    ceilings = stage._ceilings()
    assert set(ceilings.ceilings) == {"con1.tok_s_user"}
    assert "no analysis's work model holds" in ceilings.unknown["con128.tok_s_user"]
    assert "differ by 100%" in ceilings.unknown["con128.output_tok_s_gpu"]
    report = json.loads((spec.workdir / "sol" / "consensus.json").read_text())
    assert report["headline.con1.tok_s_user"]["status"] == "ruled"
    stage.t.pop("sol")
    stage._sol()
    assert len(turns) == 4
    for what in turns:
        (spec.workdir / f"{what}-decision.yaml").write_text("decision: CANDIDATE\n")
    stage._invalidate_profile("source changed")
    stage._sol()
    assert len(turns) == 8 and stage._ceilings().ceilings
    assert list(spec.workdir.glob("sol-analytic-decision.yaml.stale-*"))


def test_wrapper_times_callables_not_claimed_numbers(tmp_path, monkeypatch):
    from agent_flow.workflows.trtllm_modeling_v2.tune.harness import microbench

    script = Path(_microbench(tmp_path)["script"])
    digest = hashlib.sha256(script.read_bytes()).hexdigest()
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "microbench.py",
            "--script",
            str(script),
            "--output",
            str(output),
            "--tree",
            str(tmp_path),
            "--commit",
            "revision",
            "--harness-sha256",
            digest,
        ],
    )
    identity = Mock()
    monkeypatch.setattr(microbench, "_identity", identity)
    monkeypatch.setattr(
        microbench.importlib,
        "import_module",
        lambda name: SimpleNamespace(__file__=str(tmp_path / "tensorrt_llm/__init__.py")),
    )
    event = Mock(elapsed_time=Mock(return_value=0.125))
    cuda = SimpleNamespace(Event=Mock(return_value=event), synchronize=Mock())
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    operation = Mock()
    monkeypatch.setattr(
        microbench.runpy, "run_path", lambda path: {"benchmark": lambda: {"case": operation}}
    )
    microbench.main()
    assert operation.call_count == 110 and cuda.synchronize.call_count == 2
    assert identity.call_count == 2
    assert microbench.load_result(output, commit="revision", harness=digest, cases=["case"])[
        "latency_us"
    ] == {"case": 125.0}
    monkeypatch.setattr(
        microbench.runpy, "run_path", lambda path: {"benchmark": lambda: {"case": 0.001}}
    )
    with pytest.raises(ValueError, match="not claimed timings"):
        microbench.main()


class _Tensor:
    def __init__(self, values, dtype="float32", shape=None):
        self.values = tuple(values)
        self.dtype = dtype
        self.shape = (len(self.values),) if shape is None else shape

    def is_floating_point(self):
        return self.dtype.startswith("float")

    def is_complex(self):
        return False

    def numel(self):
        return len(self.values)

    def float(self):
        return self

    def abs(self):
        return _Tensor([abs(v) for v in self.values])

    def max(self):
        return max(self.values)

    def __sub__(self, other):
        return _Tensor([a - b for a, b in zip(self.values, other.values)])


@pytest.mark.parametrize(
    "candidate,control,problem",
    [
        ([_Tensor([1.0], shape=(1, 1))], [_Tensor([1.0])], "shape or dtype"),
        ([], [], "no tensor"),
        ([_Tensor([2**60 + 1], "int64")], [_Tensor([2**60], "int64")], "discrete"),
        ([_Tensor([math.nan])], [_Tensor([1.0])], "non-finite"),
        ([_Tensor([1.01])], [_Tensor([1.0])], ""),
    ],
    ids=["shape", "missing", "integer", "nan", "rounding"],
)
def test_numerical_evidence_rejects_incompatible_or_invalid_outputs(
    monkeypatch, candidate, control, problem
):
    from agent_flow.workflows.trtllm_modeling_v2.tune.harness import microbench

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            isfinite=lambda tensor: SimpleNamespace(
                all=lambda: all((math.isfinite(value) for value in tensor.values))
            ),
            equal=lambda a, b: a.values == b.values,
        ),
    )
    numerics = microbench._compare(candidate, control)
    failure = microbench.diverged({"case": numerics})
    if problem:
        assert problem in failure
    else:
        assert failure == ""
        exact = candidate[0].values == control[0].values
        assert numerics["exact"] is exact
        assert numerics["error"] == pytest.approx(0.0 if exact else 0.01)
