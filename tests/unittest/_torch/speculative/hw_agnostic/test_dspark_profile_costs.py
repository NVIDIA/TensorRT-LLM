# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the DSpark cost profiler schedule and the profiling tool's grid."""

import json

import pytest

from tensorrt_llm._torch.speculative.dspark_planner import ExactSpsCostRow, ExactSpsCostTable
from tensorrt_llm._torch.speculative.dspark_profile import DSparkCostProfiler
from tensorrt_llm.tools.dspark_profile_costs import (
    _replace_k,
    compact_budgets,
    main,
    plan_grid,
    short_cells,
    with_graph_ladder,
    without_adp_balance,
)

pytestmark = pytest.mark.cpu_only


def test_default_tiers_reproduce_the_measured_k5_grid():
    fractions = (0.5, 0.6, 0.7, 0.8)
    assert compact_budgets(128, 5, fractions) == [448, 512, 576, 640]
    assert compact_budgets(64, 5, fractions) == [224, 256, 288, 320]
    assert compact_budgets(16, 3, fractions) == [40, 45, 50, 54]
    # Rounding can collapse tiers at tiny G; they are deduplicated.
    assert compact_budgets(1, 5, fractions) == [4, 5]
    # V=28 at G16 would leave some request of a full batch with no draft token.
    assert compact_budgets(16, 3, (0.25, 0.5)) == [40]


def test_tier_fractions_below_one_draft_token_fail_before_profiling(tmp_path, monkeypatch):
    config = tmp_path / "runtime.yaml"
    config.write_text(
        json.dumps(
            {
                "speculative_config": {"decoding_type": "DSpark", "max_draft_len": 3},
                "cuda_graph_config": {"batch_sizes": [16, 32]},
            }
        )
    )

    def run(*args, **kwargs):
        raise AssertionError("profiling must not start")

    monkeypatch.setattr("subprocess.run", run)
    argv = ["--model", "m", "--config", str(config), "--output", str(tmp_path / "costs.json")]
    with pytest.raises(SystemExit, match="V=28 for K=3 at G=16"):
        main(argv + ["--tier-fractions", "0.25", "0.5"])


def test_rerun_in_the_same_workdir_starts_from_fresh_samples(tmp_path, monkeypatch):
    config = tmp_path / "runtime.yaml"
    config.write_text(
        json.dumps(
            {
                "speculative_config": {"decoding_type": "DSpark", "max_draft_len": 3},
                "cuda_graph_config": {"batch_sizes": [16]},
            }
        )
    )
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "samples-k3.json").write_text(json.dumps({"samples": []}))
    (workdir / "costs-k3.json").write_text("{}")
    payload = {"schema_version": 3, "setup": {}, "costs": {"3": {}}}

    def run(command, env, check):
        assert not (workdir / "samples-k3.json").exists()
        assert env["TLLM_DSPARK_PROFILE_COSTS"] == str(workdir / "samples-k3.json")
        worker_output = command[command.index("--worker-output") + 1]
        assert not (workdir / "costs-k3.json").exists()
        with open(worker_output, "w") as file:
            json.dump(payload, file)

    monkeypatch.setattr("subprocess.run", run)
    output = tmp_path / "costs.json"
    argv = ["--model", "m", "--config", str(config), "--output", str(output)]
    assert main(argv + ["--workdir", str(workdir)]) == 0
    assert json.loads(output.read_text()) == payload


def test_grid_gives_large_graphs_priority_within_the_runtime_cell_limit():
    ladder = [1, 2, 4] + [8 * i for i in range(1, 17)]
    grid = plan_grid(ladder, 5, (0.5, 0.6, 0.7, 0.8))
    assert sorted(grid) == ladder
    assert sum(len(budgets) for budgets in grid.values()) == 32
    assert all(len(grid[g]) == 4 for g in (72, 80, 88, 96, 104, 112, 120, 128))
    assert grid[8] == []
    # The planned grid is a table the runtime accepts.
    ExactSpsCostTable(
        tables={
            g: ExactSpsCostRow(token_counts=[0] + budgets, step_time_ms=[1.0] * (1 + len(budgets)))
            for g, budgets in grid.items()
        },
        max_draft_len=5,
    )


def test_worker_command_profiles_one_k():
    command = ["python", "-m", "tool", "--model", "m", "--k", "3", "5", "--output", "o.json"]
    assert _replace_k(command, 5) == [
        "python",
        "-m",
        "tool",
        "--model",
        "m",
        "--output",
        "o.json",
        "--k",
        "5",
    ]


def test_profiler_round_robins_budgets_and_times_full_steps(tmp_path, monkeypatch):
    clock = iter(float(step) * 0.05 for step in range(1000))
    monkeypatch.setattr("time.perf_counter", lambda: next(clock))
    output = tmp_path / "samples.json"
    profiler = DSparkCostProfiler(str(output), write_samples=True, hold_steps=4, warmup_steps=1)

    chosen = [
        profiler.choose(iteration=i, graph_batch_size=8, budgets=(0, 20, 30), all_ranks_full=True)
        for i in range(24)
    ]
    profiler.flush()

    assert chosen == [0] * 4 + [20] * 4 + [30] * 4 + [0] * 4 + [20] * 4 + [30] * 4
    samples = {
        (entry["graph_batch_size"], entry["verifier_budget"]): entry["step_time_ms"]
        for entry in json.loads(output.read_text())["samples"]
    }
    # Two holds per budget, each dropping its first step; the last step of the
    # final hold has no successor yet.
    assert len(samples[(8, 0)]) == 6
    assert len(samples[(8, 20)]) == 6
    assert len(samples[(8, 30)]) == 5
    assert all(value == pytest.approx(50.0) for values in samples.values() for value in values)


def test_profiler_skips_partial_steps_and_iteration_gaps(tmp_path, monkeypatch):
    clock = iter(float(step) for step in range(1000))
    monkeypatch.setattr("time.perf_counter", lambda: next(clock))
    profiler = DSparkCostProfiler(
        str(tmp_path / "samples.json"), write_samples=False, hold_steps=4, warmup_steps=0
    )

    assert (
        profiler.choose(iteration=0, graph_batch_size=8, budgets=(0, 20), all_ranks_full=True) == 0
    )
    # A rank without G real requests runs native and is not timed.
    assert (
        profiler.choose(iteration=1, graph_batch_size=8, budgets=(0, 20), all_ranks_full=False) == 0
    )
    assert (
        profiler.choose(iteration=2, graph_batch_size=8, budgets=(0, 20), all_ranks_full=True) == 0
    )
    # Iteration 3 had no decode decision (for example a mixed step).
    assert (
        profiler.choose(iteration=4, graph_batch_size=8, budgets=(0, 20), all_ranks_full=True) == 0
    )

    assert dict(profiler._samples) == {(8, 0): [1000.0]}


def test_short_cells_reports_budgets_below_the_sample_minimum(tmp_path):
    samples = tmp_path / "samples.json"
    assert short_cells(samples, 64, [0, 224], 20) == {0: 0, 224: 0}
    samples.write_text(
        json.dumps(
            {
                "samples": [
                    {"graph_batch_size": 64, "verifier_budget": 0, "step_time_ms": [1.0] * 26},
                    {"graph_batch_size": 64, "verifier_budget": 224, "step_time_ms": [1.0] * 17},
                    {"graph_batch_size": 128, "verifier_budget": 448, "step_time_ms": [1.0]},
                ]
            }
        )
    )
    assert short_cells(samples, 64, [0, 224, 256], 20) == {224: 17, 256: 0}
    assert short_cells(samples, 64, [0], 20) == {}


def test_profiling_turns_off_attention_dp_balancing_only():
    balanced = {
        "max_batch_size": 64,
        "attention_dp_config": {"enable_balance": True, "batching_wait_iters": 10},
    }
    profiled = without_adp_balance(balanced)
    assert profiled["attention_dp_config"] == {"enable_balance": False, "batching_wait_iters": 10}
    assert profiled["max_batch_size"] == 64
    assert balanced["attention_dp_config"]["enable_balance"] is True
    assert without_adp_balance({"max_batch_size": 64}) == {"max_batch_size": 64}


def test_graph_batch_size_override_captures_only_the_profiled_graphs():
    config = {
        "max_batch_size": 64,
        "cuda_graph_config": {"batch_sizes": [16, 32, 64], "enable_padding": True},
    }
    profiled = with_graph_ladder(config, [64])
    assert profiled["cuda_graph_config"] == {
        "batch_sizes": [64],
        "max_batch_size": 64,
        "enable_padding": True,
    }
    assert profiled["max_batch_size"] == 64
    assert config["cuda_graph_config"]["batch_sizes"] == [16, 32, 64]
    assert with_graph_ladder({}, [16, 32])["cuda_graph_config"] == {
        "batch_sizes": [16, 32],
        "max_batch_size": 32,
    }
