# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A campaign over several perf configs: one measurement covers all, constraints and a goal."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

from agent_flow.workflows.trtllm_modeling_v2.tune.measure import BenchPart, BenchSpec
from agent_flow.workflows.trtllm_modeling_v2.tune.records import (
    PARTS_FILE,
    Measurement,
    NoiseFloor,
    compare,
    goal_reached,
)
from agent_flow.workflows.trtllm_modeling_v2.tune.schema import validate_perf_block
from agent_flow.workflows.trtllm_modeling_v2.tune.stage import TuneSpec


def _curve(path: Path, label: str, lat: float, thr: float, tpot: float = 20.0) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rows = [
        {"concurrency": 1, "tok_s_user": lat, "output_tok_s_gpu": lat, "mean_tpot_ms": tpot},
        {"concurrency": 64, "tok_s_user": thr / 64, "output_tok_s_gpu": thr, "mean_tpot_ms": tpot},
    ]
    (path / "summary.json").write_text(json.dumps(rows))
    (path / "meta.json").write_text(json.dumps({"label": label, "hostname": "h1"}))


def _composite(root: Path, label: str, parts: dict, constraints=None) -> Measurement:
    for name, (lat, thr, tpot) in parts.items():
        _curve(root / label / name, name, lat, thr, tpot)
    (root / label / PARTS_FILE).write_text(
        json.dumps({"parts": list(parts), "constraints": constraints or {}})
    )
    (root / label / "meta.json").write_text(json.dumps({"label": label, "hostname": "h1"}))
    return Measurement.load(root / label)


def test_a_composite_headline_prefixes_each_config(tmp_path):
    m = _composite(tmp_path, "c", {"chat": (100, 6400, 20), "batch": (50, 9000, 40)})
    headline = m.headline()
    assert headline["chat.con1.tok_s_user"] == 100
    assert headline["batch.con64.output_tok_s_gpu"] == 9000


def test_a_win_on_one_config_and_a_loss_on_another_is_a_loss(tmp_path):
    ctrl = _composite(tmp_path, "ctrl", {"chat": (100, 6400, 20), "batch": (50, 9000, 40)})
    cand = _composite(tmp_path, "cand", {"chat": (120, 7000, 20), "batch": (40, 8000, 40)})
    assert compare(cand, ctrl, NoiseFloor(), min_gain=0.03).verdict == "LOSS"


def test_breaking_a_constraint_the_control_kept_is_a_loss(tmp_path):
    bound = {"chat": {"tpot_ms_max": 30.0}}
    ctrl = _composite(tmp_path, "ctrl", {"chat": (100, 6400, 25)}, bound)
    cand = _composite(tmp_path, "cand", {"chat": (130, 9000, 35)}, bound)
    result = compare(cand, ctrl, NoiseFloor(), min_gain=0.03)
    assert result.verdict == "LOSS"
    assert "breaks a constraint" in result.reason and "chat@con1" in result.reason


def test_the_goal_is_every_headline_against_the_reference(tmp_path):
    ref = _composite(tmp_path, "ref", {"chat": (100, 6400, 20)})
    near = _composite(tmp_path, "near", {"chat": (112, 6900, 20)})
    there = _composite(tmp_path, "there", {"chat": (112, 7100, 20)})
    assert not goal_reached(near, ref, 1.1)
    assert goal_reached(there, ref, 1.1)


def test_one_job_measures_every_config_and_indexes_them(tmp_path):
    bench = BenchSpec(
        bench_py="/pkg/bench.py",
        model="/ckpt",
        env_script="/env.sh",
        isl=0,
        osl=0,
        concurrencies=[1],
        tree="/repo",
        parts=(
            BenchPart("chat", 1024, 256, [1, 16], [Path("/c/chat.yaml")], {"tpot_ms_max": 30.0}),
            BenchPart("batch", 8192, 1024, [64], [Path("/c/batch.yaml")]),
        ),
    )
    command = bench.command("e2e-x", tmp_path, [Path("/c/overlay.yaml")])
    assert command.count("bench.py") == 2
    assert "--label chat" in command and "--isl 1024" in command
    assert "--config /c/chat.yaml --config /c/overlay.yaml" in command
    # The last step indexes the parts; run it over parts a bench run would have left.
    for name in ("chat", "batch"):
        _curve(tmp_path / "e2e-x" / name, name, 10, 100)
    merge = command.split(" && \\\n")[-1]
    subprocess.run(["bash", "-c", merge], check=True)
    m = Measurement.load(tmp_path / "e2e-x")
    assert list(m.parts) == ["chat", "batch"]
    assert m.meta["label"] == "e2e-x" and m.meta["hostname"] == "h1"
    assert m.constraints["chat"] == {"tpot_ms_max": 30.0}


def test_each_config_gets_its_own_derived_serving_config(tmp_path):
    task = {
        "trtllm_repo_path": str(tmp_path),
        "checkpoint_path": "/ckpt",
        "perf": {
            "configs": [
                {"name": "chat", "workload": {"isl": 1024, "osl": 256, "concurrency": [1, 16]}},
                {
                    "name": "batch",
                    "workload": {"isl": 8192, "osl": 1024, "concurrency": [64]},
                    "llm_api_options": {"enable_chunked_prefill": True},
                    "constraint": {"tpot_ms_max": 40},
                },
            ],
            "goal": {"ratio_to_reference": 1.1},
        },
    }
    spec = TuneSpec.from_task(task, tmp_path / "ws")
    assert spec.identity_configs == []
    assert [p.name for p in spec.parts] == ["chat", "batch"]
    batch = yaml.safe_load(spec.parts[1].configs[0].read_text())
    assert batch["max_batch_size"] == 64 and batch["enable_chunked_prefill"] is True
    assert spec.parts[1].constraint == {"tpot_ms_max": 40.0}
    assert spec.goal_ratio == 1.1
    assert "`chat`" in spec.describe_workload() and "constraint" in spec.describe_workload()
    assert spec.bench().parts == tuple(spec.parts)


def test_the_configs_and_goal_are_validated():
    errors = validate_perf_block(
        {
            "perf": {
                "configs": [
                    {"name": "a", "workload": {"isl": 1, "osl": 1, "concurrency": [1]}},
                    {"name": "a", "workload": {"isl": 0}, "constraint": {"p50": 1}, "x": 1},
                ],
                "goal": {"ratio_to_reference": 1.1},
                "reference": "none",
            }
        }
    )
    joined = "\n".join(errors)
    assert "names must be unique" in joined
    assert "perf.configs[1].x: unknown key" in joined
    assert "perf.configs[1].workload.isl" in joined
    assert "perf.configs[1].constraint.p50" in joined
    assert "needs reference: stock" in joined
