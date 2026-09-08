"""Tests for reading the benchmark suite's configs and results.

This module runs nothing, so there is nothing to stub: every test writes
the file the harness would write and checks what is made of it. That is
also the only way the shape can be caught drifting — a stub agrees with
whatever it was taught, and the sweep row order and the CSV column names
are exactly the things that move upstream.
"""

from __future__ import annotations

import pytest

from agent_flow.workflows.perf_optimize import bench_cli

#: A gen sweep row, in the harness' order:
#: [ctx, gen, tp, batch, max_num_tokens, attention_dp, gpu_mem_frac, mtp,
#:  eplb, concurrency_list]
SWEEP = {
    "model_id": "deepseek-ai/DeepSeek-V4-Pro",
    "model_path": "/models/DeepSeek-V4-Pro",
    "dataset_file": "/data/DeepSeek-V4-8192-1024-200000-ratio-08_for_serve.json",
    "precision": "fp4",
    "benchmark_client": "trtllm",
    "isl": 8192,
    "osl": 1024,
    "gen_configs": [[1, 1, 4, 64, 256, False, "0.9", 3, 0, "1,32"]],
}


def test_a_row_expands_over_its_concurrency_list():
    cases = bench_cli.gen_cases(SWEEP)
    assert [c["config"]["concurrency"] for c in cases] == [1, 32]
    assert all(c["stage"] == "gen" for c in cases)


def test_the_shape_name_matches_the_column_the_postprocessor_writes():
    """`{dep|tep}_{tp}_eplb{N}_mtp{M}`.

    A scored row is addressed by that name plus its concurrency, so the
    name this module derives has to be the one the CSV carries or an
    attempt cannot be aligned with its baseline.
    """
    assert bench_cli.gen_cases(SWEEP)[0]["name"] == "tep_4_eplb0_mtp3"
    dep = {**SWEEP, "gen_configs": [[1, 1, 32, 128, 512, True, "0.7", 1, 384, "512"]]}
    assert bench_cli.gen_cases(dep)[0]["name"] == "dep_32_eplb384_mtp1"


def test_the_dict_row_form_is_accepted_too():
    """The harness takes both; a sweep written either way must expand."""
    rows = {
        **SWEEP,
        "gen_configs": [
            {
                "ctx_num": 1,
                "gen_num": 1,
                "gen_tp_size": 8,
                "gen_batch_size": 128,
                "gen_max_num_tokens": 512,
                "gen_enable_attention_dp": False,
                "gen_gpu_memory_fraction": "0.9",
                "gen_mtp_size": 3,
                "gen_eplb_num_slots": 0,
                "gen_concurrency_list": "4,8",
            }
        ],
    }
    cases = bench_cli.gen_cases(rows)
    assert [c["name"] for c in cases] == ["tep_8_eplb0_mtp3"] * 2
    assert [c["config"]["concurrency"] for c in cases] == [4, 8]


def test_the_operating_point_is_the_product_not_the_listed_value():
    """A row's concurrency is per generation server.

    The client is driven at `concurrency * gen_num` and the harness names
    the result directory after that product, so that is the point.
    """
    assert bench_cli.operating_point({"concurrency": 64, "gen_num": 2}) == 128
    assert bench_cli.operating_point({"concurrency": 64}) == 64


def test_a_ctx_case_is_addressed_by_max_batch():
    """It has no concurrency; `max_batch` is the in-flight count."""
    assert bench_cli.operating_point({"isl": 1024, "max_batch": 16, "tp_size": 4}) == 16
    assert bench_cli.operating_point({"isl": 1024}) is None


def test_a_ctx_block_expands_over_batch_tp_and_mtp():
    cases = bench_cli.ctx_cases(
        {
            "benchmarks": [
                {
                    "isl": 1024,
                    "osl": 1,
                    "max_batch": [8, 16],
                    "tp_size": [4],
                    "ratio": [1],
                    "mtp_range": [0, 3],
                },
            ]
        }
    )
    assert len(cases) == 4
    assert {c["config"]["max_batch"] for c in cases} == {8, 16}
    assert all(c["stage"] == "ctx" for c in cases)


def test_isl_is_reported_as_a_bound_beside_the_corpus_not_instead_of_it():
    """The checked-in 8k sweep pairs isl 8192 with a 200000-sample corpus.

    Both numbers are right: one bounds the KV allocation and the client's
    request, the other is what actually gets served.
    """
    load = bench_cli.workload(SWEEP)
    assert load["isl"] == 8192
    assert load["dataset"].endswith("_for_serve.json")


#: A ctx sweep as the config repo really writes one. It is not a gen
#: sweep with fewer keys: the model moves under `model:` and the lengths
#: move into the benchmark entries, because sweeping the input length is
#: the normal thing to do on a prefill-only run.
CTX_SWEEP = {
    "model": {
        "model_card": "deepseek-ai/DeepSeek-V4-Pro",
        "model_path": {"aga-gb300": "/models/dsv4-pro"},
        "dataset_file": "${home_dir}/dataset/DeepSeek-V4-8192-1-20000-ratio-08_for_bench.json",
    },
    "benchmarks": [
        {"isl": 8192, "osl": 1, "max_batch": [2], "tp_size": [4], "ratio": [0.8], "mtp_range": [3]}
    ],
}


def test_a_ctx_sweep_states_its_workload_somewhere_else_and_is_read_there():
    """Reading only the gen shape resolves a ctx campaign to all `None`s.

    Which does not fail -- it reconciles nothing, and `task.yaml` keeps
    the defaults block's `random_input_len: 1024` as this campaign's
    stated input length however long the requests really are.
    """
    load = bench_cli.workload(CTX_SWEEP)
    assert load["model"] == "deepseek-ai/DeepSeek-V4-Pro"
    assert load["isl"] == 8192
    assert load["osl"] == 1
    assert load["dataset"].endswith("_for_bench.json")
    assert load["model_path"] == {"aga-gb300": "/models/dsv4-pro"}


def test_a_ctx_sweep_spanning_two_input_lengths_states_no_single_one():
    """Two rows at two lengths are two workloads.

    Resolving to the first, or to the largest, would let a campaign quote
    one of them and describe half its own cases wrongly, with nothing
    raising.
    """
    spanning = {
        **CTX_SWEEP,
        "benchmarks": [
            {"isl": 1024, "osl": 1, "max_batch": [16], "tp_size": [4]},
            {"isl": 8192, "osl": 1, "max_batch": [2], "tp_size": [4]},
        ],
    }
    load = bench_cli.workload(spanning)
    assert load["isl"] is None
    assert load["osl"] == 1  # they do agree on this one, so it is stated


def test_a_top_level_length_still_wins_over_the_benchmark_entries():
    """The gen shape is not overridden by a sweep that carries both."""
    both = {**CTX_SWEEP, "isl": 4096, "osl": 512}
    load = bench_cli.workload(both)
    assert (load["isl"], load["osl"]) == (4096, 512)


FRONTIER = (
    "name,concurrency,throughput_per_user,output_tput_per_gpu,"
    "ctx_gen_inst_ratio_round_float,ctx_request_rate,ctx_gpus_round,"
    "gen_num_round,total_gpus_round\n"
    "tep_4_eplb0_mtp3,1,165.3594,39.809,0.016735,10.67,8,4,12\n"
    "tep_4_eplb0_mtp3,32,66.1748,368.7572,0.21532,10.67,8,4,12\n"
)


def test_the_csv_column_is_already_the_name_this_workflow_scores(tmp_path):
    (tmp_path / "sol_frontier_mtp.csv").write_text(FRONTIER)
    points = bench_cli.frontier_points(tmp_path)
    assert [p["concurrency"] for p in points] == [1, 32]
    assert points[0]["metrics"]["throughput_per_user"] == 165.3594
    # And the frontier the gate's metric is not, carried alongside.
    assert points[0]["metrics"]["output_tput_per_gpu"] == 39.809
    assert points[0]["gpus"] == {"ctx": 8.0, "gen": 4.0, "total": 12.0}


def test_a_row_without_a_usable_metric_is_dropped_not_defaulted(tmp_path):
    (tmp_path / "sol_frontier_mtp.csv").write_text(
        FRONTIER + "tep_4_eplb0_mtp3,64,,10.0,0.3,10.67,8,4,12\n"
    )
    assert [p["concurrency"] for p in bench_cli.frontier_points(tmp_path)] == [1, 32]


def test_a_nan_is_not_a_measurement(tmp_path):
    """The postprocessor writes NaN into columns it could not compute."""
    (tmp_path / "sol_frontier_mtp.csv").write_text(
        "name,concurrency,throughput_per_user\ntep_4_eplb0_mtp3,1,NaN\n"
    )
    with pytest.raises(bench_cli.BenchCliError, match="no row with"):
        bench_cli.frontier_points(tmp_path)


def test_a_run_dir_with_no_curve_names_the_command_that_says_why(tmp_path):
    with pytest.raises(bench_cli.BenchCliError, match="process frontier"):
        bench_cli.frontier_points(tmp_path)


def test_the_later_csv_wins_for_a_repeated_point(tmp_path):
    """A run dir can hold several CSVs -- one per mtp tag and variant.

    Sorting makes which one wins deterministic rather than filesystem
    order, so two reads of one directory agree.
    """
    (tmp_path / "a_frontier_mtp0.csv").write_text(FRONTIER)
    (tmp_path / "b_frontier_mtp.csv").write_text(
        "name,concurrency,throughput_per_user\ntep_4_eplb0_mtp3,1,200.0\n"
    )
    points = {p["concurrency"]: p for p in bench_cli.frontier_points(tmp_path)}
    assert points[1]["metrics"]["throughput_per_user"] == 200.0
