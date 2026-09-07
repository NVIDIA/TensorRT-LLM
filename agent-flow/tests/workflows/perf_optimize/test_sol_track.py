"""Tests for the ``sol_track`` block.

The authority rule is the one :mod:`.disagg` states — the sweep owns the
measurement conditions, ``task.yaml`` owns only the campaign knobs.

Nothing here is stubbed. The harness communicates through files: a sweep
YAML going in, a frontier CSV and a ``run_*.json`` coming out. So the
tests write those files and read what this module makes of them, which is
also the only way to catch the shape drifting — a stub agrees with
whatever it was taught.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.perf_optimize import bench_cli, sol_track, task_schema
from agent_flow.workflows.perf_optimize.sol_track import SOL_TRACK_FIELD

#: A gen sweep as the config repo writes one. The row order is the
#: harness': [ctx, gen, tp, batch, max_num_tokens, attention_dp,
#: gpu_mem_frac, mtp, eplb, concurrency_list].
GEN_SWEEP = {
    "benchmark_mode": "gen_only",
    "model_id": "deepseek-ai/DeepSeek-V4-Pro",
    "model_path": "/models/DeepSeek-V4-Pro",
    "precision": "fp4",
    "benchmark_client": "trtllm",
    # isl is a sizing bound, not the corpus' length: the checked-in 8k
    # sweep pairs isl 8192 with a ...-8192-1024-200000-... dataset.
    "isl": 8192,
    "osl": 1024,
    "dataset_file": "/data/DeepSeek-V4-8192-1024-200000-ratio-08_for_serve.json",
    "accept_rate": {"rate": "1:1.93,2:2.54,3:2.82", "source": "upstream"},
    "gen_configs": [[1, 1, 4, 64, 256, False, "0.9", 3, 0, "1,32"]],
}

CTX_SWEEP = {
    "model": {"model_card": "deepseek-ai/DeepSeek-V4-Pro"},
    "benchmarks": [
        {"isl": 1024, "osl": 1, "max_batch": [16], "tp_size": [4], "ratio": [1], "mtp_range": [3]},
    ],
}


def _write(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _sweep_dir(tmp_path, sweep=None, track="gen") -> Path:
    """A sweep as it really is: a directory whose files reference siblings."""
    root = tmp_path / "campaign"
    root.mkdir(exist_ok=True)
    (root / "gen_worker_config.py").write_text("# imported from beside the sweep\n")
    return _write(
        root / "sweep.yaml",
        sweep if sweep is not None else (GEN_SWEEP if track == "gen" else CTX_SWEEP),
    )


def _write_task(tmp_path, extra: dict | None = None) -> Path:
    for sub in ("ckpt", "repo"):
        (tmp_path / sub).mkdir(exist_ok=True)
    data = {
        "checkpoint_path": str(tmp_path / "ckpt"),
        "trtllm_repo_path": str(tmp_path / "repo"),
        # An upstream sweep carries no build source, so `config` is what a
        # campaign against one can do. `code` needs a rung of the ladder
        # and has its own tests.
        "optimize": {"approaches": ["config"]},
    }
    data.update(extra or {})
    return _write(tmp_path / "task.yaml", data)


def _block(tmp_path, track="gen", sweep=None, **extra) -> dict:
    return {
        "track": track,
        "sweep": str(_sweep_dir(tmp_path, sweep, track)),
        "workspace": str(tmp_path / "work"),
        **extra,
    }


def _gen(tmp_path, **extra) -> dict:
    """A gen block with an anchor -- required, so most tests want it."""
    anchor = tmp_path / "anchor.json"
    anchor.write_text(json.dumps([{"isl": 8192, "avg_request_throughput_req_s": 91.2}]))
    return _block(tmp_path, ctx_json=str(anchor), **extra)


# ------------------------------------------------------------- the expansion


def test_the_sweep_is_the_authority_for_points_and_corpus(tmp_path):
    """Nothing in task.yaml states the operating points; the sweep does."""
    data = task_schema.load_and_validate_task_yaml(
        _write_task(tmp_path, {SOL_TRACK_FIELD: _gen(tmp_path)})
    )
    assert data["benchmark"]["concurrency"] == [1, 32]
    assert data["benchmark"]["dataset_name"].endswith("_for_serve.json")
    assert data["optimize"]["target_metric"] == "throughput_per_user"
    assert data["profile"]["methods"] == ["nsys"]
    assert task_schema.is_curve_mode(data)


def test_the_operating_point_is_the_product_of_concurrency_and_gen_num(tmp_path):
    """The one trap the harness keeps: a row's concurrency is PER GEN SERVER.

    The client is driven at `concurrency * gen_num` and the result
    directory is named for that product, so that is what `task.yaml`
    means by concurrency.
    """
    sweep = {**GEN_SWEEP, "gen_configs": [[1, 2, 4, 64, 256, False, "0.9", 3, 0, "64"]]}
    data = task_schema.load_and_validate_task_yaml(
        _write_task(tmp_path, {SOL_TRACK_FIELD: _gen(tmp_path, sweep=sweep)})
    )
    assert data["benchmark"]["concurrency"] == 128  # 64 x 2 -> a single point


def test_a_ctx_case_is_addressed_by_max_batch(tmp_path):
    """A ctx entry has no concurrency at all.

    `max_batch` is the in-flight request count for a prefill-only run,
    which is what this workflow means by concurrency everywhere else.
    """
    data = task_schema.load_and_validate_task_yaml(
        _write_task(tmp_path, {SOL_TRACK_FIELD: _block(tmp_path, track="ctx")})
    )
    assert data["benchmark"]["concurrency"] == 16
    assert data["optimize"]["target_metric"] == "avg_request_throughput_req_s"


def test_the_case_name_mirrors_the_column_the_postprocessor_writes(tmp_path):
    """`{dep|tep}_{tp}_eplb{N}_mtp{M}` -- so a scored row addresses its row."""
    cases = bench_cli.gen_cases(GEN_SWEEP)
    assert [c["name"] for c in cases] == ["tep_4_eplb0_mtp3"] * 2
    assert [c["config"]["concurrency"] for c in cases] == [1, 32]
    dep = bench_cli.gen_cases(
        {**GEN_SWEEP, "gen_configs": [[1, 1, 8, 64, 256, True, "0.8", 0, 384, "16"]]}
    )
    assert dep[0]["name"] == "dep_8_eplb384_mtp0"


def test_a_track_the_sweep_does_not_measure_is_refused(tmp_path):
    """A ctx campaign against a gen-only sweep would measure nothing."""
    with pytest.raises(task_schema.TaskSchemaError, match="plans no 'ctx' cases"):
        task_schema.load_and_validate_task_yaml(
            _write_task(tmp_path, {SOL_TRACK_FIELD: _block(tmp_path, track="ctx", sweep=GEN_SWEEP)})
        )


# ------------------------------------------------------------------ the block


def test_a_campaign_needs_no_work_dir_of_its_own(tmp_path):
    """Submission and collection are one path, so there is nothing to state.

    A stage submits into `<its result dir>/run` and collects from the same
    place. `sol_track.workspace` is kept because task.yaml files carry it,
    but a spec without one validates: there is no second path left for it
    to disagree with.
    """
    anchor = tmp_path / "anchor.json"
    anchor.write_text(json.dumps([{"isl": 8192}]))
    data = task_schema.load_and_validate_task_yaml(
        _write_task(
            tmp_path,
            {
                SOL_TRACK_FIELD: {
                    "track": "gen",
                    "sweep": str(_sweep_dir(tmp_path)),
                    "ctx_json": str(anchor),
                }
            },
        )
    )
    assert data["benchmark"]["concurrency"] == [1, 32]


def test_a_missing_sweep_file_is_refused(tmp_path):
    with pytest.raises(task_schema.TaskSchemaError, match="is not a file"):
        task_schema.load_and_validate_task_yaml(
            _write_task(
                tmp_path,
                {SOL_TRACK_FIELD: {"track": "gen", "sweep": "/nope.yaml", "workspace": "/w"}},
            )
        )


def test_it_cannot_be_combined_with_a_disagg_campaign(tmp_path):
    """One campaign measures one thing, and each reconciles from its own file."""
    harness = _write(tmp_path / "disagg.yaml", {"worker_config": {"ctx": {}, "gen": {}}})
    with pytest.raises(task_schema.TaskSchemaError, match="cannot be combined"):
        task_schema.load_and_validate_task_yaml(
            _write_task(
                tmp_path, {SOL_TRACK_FIELD: _gen(tmp_path), "disagg": {"config": str(harness)}}
            )
        )


def test_extra_llm_api_options_beside_a_track_is_refused(tmp_path):
    """Two seeds for one tuning file: the named one would be discarded."""
    other = _write(tmp_path / "extra.yaml", {"knob": 1})
    with pytest.raises(task_schema.TaskSchemaError, match="cannot be combined"):
        task_schema.load_and_validate_task_yaml(
            _write_task(
                tmp_path, {SOL_TRACK_FIELD: _gen(tmp_path), "extra_llm_api_options": str(other)}
            )
        )


def test_a_condition_the_user_wrote_that_disagrees_is_an_error(tmp_path):
    """Stated on `concurrency`: it is what each point is scored at."""
    with pytest.raises(task_schema.TaskSchemaError, match="contradicts the sweep"):
        task_schema.load_and_validate_task_yaml(
            _write_task(
                tmp_path, {SOL_TRACK_FIELD: _gen(tmp_path), "benchmark": {"concurrency": [1, 999]}}
            )
        )


def test_the_corpus_is_the_workload_not_the_sizing_bound(tmp_path):
    """`isl` bounds the KV allocation; the dataset is what gets served.

    The checked-in sweep pairs `isl: 8192` with a `...-200000-...` corpus
    and both numbers are right. Copying the bound into `random_input_len`
    asserted a synthetic dataset of uniformly 8192-token requests.
    """
    data = task_schema.load_and_validate_task_yaml(
        _write_task(tmp_path, {SOL_TRACK_FIELD: _gen(tmp_path)})
    )
    bench = data["benchmark"]
    assert bench["dataset_path"] == GEN_SWEEP["dataset_file"]
    assert "random_input_len" not in bench
    assert "random_output_len" not in bench
    notes = " ".join(data[SOL_TRACK_FIELD]["filled_from_sweep_plan"])
    assert "sequence-length bounds" in notes


def test_an_owner_who_names_another_metric_still_wins(tmp_path):
    data = task_schema.load_and_validate_task_yaml(
        _write_task(
            tmp_path,
            {
                SOL_TRACK_FIELD: _gen(tmp_path),
                "optimize": {"target_metric": "mine", "approaches": ["config"]},
            },
        )
    )
    assert data["optimize"]["target_metric"] == "mine"


# ------------------------------------------------------------------ anchors


def _anchor(tmp_path, rows) -> Path:
    path = tmp_path / "ctx_anchor.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_a_gen_track_without_a_ctx_anchor_runs_and_says_what_it_cannot_see(tmp_path):
    """The anchor buys the end-to-end view, not the right to be scored.

    The gate's metric is `accept_rate / avg_iteration_time` -- decode
    iterations only, no context term -- so no ctx measurement can move it.
    Requiring an anchor here made a decode campaign wait on somebody's
    prefill run to score a change prefill cannot affect. What must not
    happen is that the missing end-to-end half goes unmentioned.
    """
    data = task_schema.load_and_validate_task_yaml(
        _write_task(tmp_path, {SOL_TRACK_FIELD: _block(tmp_path)})
    )
    assert data["optimize"]["target_metric"] == "throughput_per_user"
    notes = " ".join(data[SOL_TRACK_FIELD]["filled_from_sweep_plan"])
    assert "output_tput_per_gpu" in notes
    assert "ABSENT" in notes


def test_an_anchor_measured_on_other_work_is_refused(tmp_path):
    """A frontier is a decode rate over a prefill rate.

    Feed it halves measured at different input lengths and it still
    returns a number, on a curve that still looks like a frontier.
    """
    anchor = _anchor(tmp_path, [{"isl": 1024, "avg_request_throughput_req_s": 91.2}])
    with pytest.raises(sol_track.SolTrackError, match="measured at isl 1024"):
        sol_track.require_matching_anchor(anchor, bench_cli.plan(GEN_SWEEP))


def test_a_matching_anchor_passes_and_an_unstated_one_says_so(tmp_path):
    """'not checked' and 'checked and matched' must not read alike."""
    good = _anchor(tmp_path, [{"isl": 8192}])
    assert sol_track.require_matching_anchor(good, bench_cli.plan(GEN_SWEEP)) is None
    silent = _anchor(tmp_path, [{"avg_request_throughput_req_s": 10.7}])
    note = sol_track.require_matching_anchor(silent, bench_cli.plan(GEN_SWEEP))
    assert note and "no 'isl'" in note


# ------------------------------------------------------------------ the score


FRONTIER_HEADER = (
    "name,concurrency,throughput_per_user,output_tput_per_gpu,"
    "ctx_gen_inst_ratio_round_float,ctx_request_rate,ctx_gpus_round,"
    "gen_num_round,total_gpus_round\n"
)


def _run_dir(into, rows=None, name="bm_deepseek-v4-pro-sol-8192-1024-20260907-GB300") -> Path:
    """A harness run directory under the result dir a stage collects into."""
    run = Path(into) / sol_track.RUN_SUBDIR / name
    run.mkdir(parents=True, exist_ok=True)
    body = (
        rows
        if rows is not None
        else (
            "tep_4_eplb0_mtp3,1,165.3594,39.809,0.016735,10.67,8,4,12\n"
            "tep_4_eplb0_mtp3,32,66.1748,368.7572,0.21532,10.67,8,4,12\n"
        )
    )
    (run / "sol_frontier_mtp.csv").write_text(FRONTIER_HEADER + body, encoding="utf-8")
    return run


GEN_ONLY_HEADER = (
    "config,concurrency,mtp,tp,adp,eplb,tps_per_user,"
    "output_tps_per_gen_gpu,output_tput,avg_itertime_ms,num_iters\n"
)


def _gen_only_dir(into, rows=None, name="bm_deepseek-v4-pro-sol-8192-1024-20260907-GB300") -> Path:
    """A run dir scored by the anchor-free extractor instead of a frontier."""
    run = Path(into) / sol_track.RUN_SUBDIR / name
    run.mkdir(parents=True, exist_ok=True)
    body = (
        rows
        if rows is not None
        else (
            "ctx1_gen1_tep4_c1_eplb0_mtp3,1,3,4,False,0,165.36,41.34,165.36,11.67,412\n"
            "ctx1_gen1_tep4_c32_eplb0_mtp3,32,3,4,False,0,66.17,529.39,2117.57,29.16,388\n"
        )
    )
    (run / bench_cli.GEN_ONLY_CSV).write_text(GEN_ONLY_HEADER + body, encoding="utf-8")
    return run


def _task(track="gen", **extra) -> dict:
    """A resolved sol_track block.

    A gen campaign carries an anchor by default, because that is the
    end-to-end shape and it is the one the frontier reader applies to.
    Pass ``ctx_json=None`` for the anchor-free half; the reader dispatches
    on what the campaign DECLARES, never on which file is on disk.
    """
    block = {"track": track, **extra}
    if track == sol_track.GEN_TRACK:
        block.setdefault("ctx_json", "/anchors/ctx_anchor.json")
    return {SOL_TRACK_FIELD: block}


def test_the_metric_lands_under_the_name_the_campaign_scores(tmp_path):
    """The CSV column is already `throughput_per_user`; no rename happens.

    What matters is that it lands under `optimize.target_metric`, because
    the baseline gate looks that key up and reports its absence as a stage
    that measured nothing.
    """
    _run_dir(tmp_path / "baseline")
    written = sol_track.collect(_task(), tmp_path / "baseline")
    assert [p.parent.name for p in written] == ["concurrency_1", "concurrency_32"]
    payload = json.loads(written[0].read_text())
    assert payload["throughput_per_user"] == 165.3594
    assert payload["shape"] == "tep_4_eplb0_mtp3"
    assert payload["source_csv"].endswith("sol_frontier_mtp.csv")


def test_it_writes_wherever_the_stage_was_told_to(tmp_path):
    """Not just the baseline: every attempt has its own result directory."""
    _run_dir(tmp_path / "baseline")
    _run_dir(tmp_path / "r1/attempt_1")
    written = sol_track.collect(_task(), tmp_path / "r1/attempt_1")
    assert written[0] == tmp_path / "r1/attempt_1/concurrency_1" / sol_track.SOL_RESULT_NAME


def test_what_a_gen_gain_is_worth_at_the_frontier_travels_with_it(tmp_path):
    """The gate's metric is not the deployment's objective.

    `throughput_per_user` is anchor-free; `output_tput_per_gpu` divides by
    a denominator the context side owns, so the same measured +1 % is
    worth different amounts at two ends of one curve.
    """
    _run_dir(tmp_path / "baseline")
    sol_track.collect(_task(), tmp_path / "baseline")
    at_1 = json.loads((tmp_path / "baseline/concurrency_1" / sol_track.SOL_RESULT_NAME).read_text())
    at_32 = json.loads(
        (tmp_path / "baseline/concurrency_32" / sol_track.SOL_RESULT_NAME).read_text()
    )
    # gen_gpus / (ctx_gpus * ctx_per_gen + gen_gpus), all three read off the row.
    assert at_1["frontier_elasticity"] == pytest.approx(4 / (8 * 0.016735 + 4), rel=1e-6)
    assert at_1["frontier_elasticity"] == pytest.approx(0.968, abs=0.005)
    assert at_32["frontier_elasticity"] < at_1["frontier_elasticity"]


def test_two_points_at_one_concurrency_are_refused_rather_than_overwritten(tmp_path):
    """Keeping the last writer scores the campaign on whichever row came second."""
    _run_dir(
        tmp_path / "baseline",
        rows=(
            "tep_4_eplb0_mtp3,8,165.0,39.8,0.0167,10.67,8,4,12\n"
            "dep_8_eplb0_mtp3,8,120.0,50.0,0.0200,10.67,8,4,12\n"
        ),
    )
    with pytest.raises(sol_track.SolTrackError, match="both measure 8 requests in flight"):
        sol_track.collect(_task(), tmp_path / "baseline")


def test_a_run_dir_with_no_scored_curve_names_the_command_that_says_why(tmp_path):
    (tmp_path / "baseline" / sol_track.RUN_SUBDIR / "bm_x").mkdir(parents=True)
    with pytest.raises(sol_track.SolTrackError, match="process frontier"):
        sol_track.collect(_task(), tmp_path / "baseline")


def test_two_run_dirs_under_one_result_dir_are_refused(tmp_path):
    """One work dir per campaign per workload.

    Two of them means the score cannot say which run it came from -- and
    the run directory name carries no code identity, so a second attempt
    submitted into the same place would silently join the first.
    """
    _run_dir(tmp_path / "baseline", name="bm_a")
    _run_dir(tmp_path / "baseline", name="bm_b")
    with pytest.raises(sol_track.SolTrackError, match="2 run directories"):
        sol_track.collect(_task(), tmp_path / "baseline")


def test_a_row_missing_its_metric_is_skipped_not_defaulted(tmp_path):
    _run_dir(
        tmp_path / "baseline",
        rows=(
            "tep_4_eplb0_mtp3,1,165.3594,39.809,0.016735,10.67,8,4,12\n"
            "tep_4_eplb0_mtp3,32,,368.7572,0.21532,10.67,8,4,12\n"
        ),
    )
    written = sol_track.collect(_task(), tmp_path / "baseline")
    assert [p.parent.name for p in written] == ["concurrency_1"]


# ------------------------------------------------- the anchor-free gen reader


def test_a_gen_campaign_with_no_anchor_is_scored_from_the_iteration_logs(tmp_path):
    """Same column, same formula, no prefill measurement anywhere in it.

    `tps_per_user` is what `get_gen_only_perf` calls the quantity the
    frontier CSV calls `throughput_per_user`; the rename is the only
    difference in the number.
    """
    _gen_only_dir(tmp_path / "baseline")
    written = sol_track.collect(_task(ctx_json=None), tmp_path / "baseline")
    assert [p.parent.name for p in written] == ["concurrency_1", "concurrency_32"]
    payload = json.loads(written[0].read_text())
    assert payload["throughput_per_user"] == 165.36
    assert payload["source_csv"].endswith(bench_cli.GEN_ONLY_CSV)


def test_the_two_gen_readers_name_a_point_the_same_way(tmp_path):
    """A point must address the planned row through either reader.

    The extractor writes its own `config` string; the shape is rebuilt
    from the columns instead, so it matches both the frontier CSV's `name`
    and the sweep expansion's.
    """
    _gen_only_dir(tmp_path / "unanchored")
    _run_dir(tmp_path / "anchored")
    unanchored = json.loads(
        sol_track.collect(_task(ctx_json=None), tmp_path / "unanchored")[0].read_text()
    )
    anchored = json.loads(sol_track.collect(_task(), tmp_path / "anchored")[0].read_text())
    assert unanchored["shape"] == anchored["shape"] == "tep_4_eplb0_mtp3"
    assert {case["name"] for case in bench_cli.gen_cases(GEN_SWEEP)} == {unanchored["shape"]}


def test_the_missing_end_to_end_half_is_stated_rather_than_left_blank(tmp_path):
    """Absent, not zero and not unchanged.

    A reader who opens one of these files must not be able to mistake
    "this campaign could not see the frontier" for "the frontier did not
    move" -- so the reason is written into the result, not merely implied
    by a missing key.
    """
    _gen_only_dir(tmp_path / "baseline")
    payload = json.loads(
        sol_track.collect(_task(ctx_json=None), tmp_path / "baseline")[0].read_text()
    )
    assert payload["e2e_view"] is None
    assert "ABSENT" in payload["e2e_view_absent"]
    assert "frontier_elasticity" in payload["e2e_view_absent"]
    assert "frontier_elasticity" not in payload
    assert "frontier_metrics" not in payload


def test_the_flattering_per_gen_gpu_number_is_not_renamed_to_the_frontier_s(tmp_path):
    """`output_tps_per_gen_gpu` divides by the generation GPUs alone.

    The frontier's `output_tput_per_gpu` divides by the whole rate-matched
    deployment, so the two are never the same number and this one is
    always the larger. Carried under its own name, or a reader quotes an
    end-to-end result the campaign never measured.
    """
    _gen_only_dir(tmp_path / "baseline")
    payload = json.loads(
        sol_track.collect(_task(ctx_json=None), tmp_path / "baseline")[0].read_text()
    )
    assert payload["gen_only_metrics"]["output_tps_per_gen_gpu"] == 41.34
    assert "output_tput_per_gpu" not in json.dumps(payload["gen_only_metrics"])


def test_a_frontier_the_campaign_never_declared_an_anchor_for_is_not_read(tmp_path):
    """Dispatch is on the declaration, never on what is on disk.

    The only thing checked about an anchor is that its input length is
    this sweep's. An anchor nobody declared was checked against nothing,
    and the frontier it produces still plots a clean curve.
    """
    _run_dir(tmp_path / "baseline")  # a frontier CSV, and only that
    with pytest.raises(sol_track.SolTrackError, match="get_gen_only_perf"):
        sol_track.collect(_task(ctx_json=None), tmp_path / "baseline")


def test_a_case_that_never_reached_steady_state_is_skipped_not_defaulted(tmp_path):
    """The extractor drops such a case, so its row arrives without a score."""
    _gen_only_dir(
        tmp_path / "baseline",
        rows=(
            "ctx1_gen1_tep4_c1_eplb0_mtp3,1,3,4,False,0,165.36,41.34,165.36,11.67,412\n"
            "ctx1_gen1_tep4_c32_eplb0_mtp3,32,3,4,False,0,,529.39,2117.57,29.16,388\n"
        ),
    )
    written = sol_track.collect(_task(ctx_json=None), tmp_path / "baseline")
    assert [p.parent.name for p in written] == ["concurrency_1"]


def test_an_attention_dp_row_is_named_dep_rather_than_tep(tmp_path):
    """`adp` arrives as the literal `True`/`False` pandas writes."""
    _gen_only_dir(
        tmp_path / "baseline",
        rows="ctx1_gen1_dep8_c32_eplb256_mtp3,32,3,8,True,256,66.17,529.39,2117.57,29.16,388\n",
    )
    payload = json.loads(
        sol_track.collect(_task(ctx_json=None), tmp_path / "baseline")[0].read_text()
    )
    assert payload["shape"] == "dep_8_eplb256_mtp3"


def test_a_ctx_campaign_is_scored_from_the_field_the_harness_validates_on(tmp_path):
    """There is no frontier on this track and running one is an error.

    `process frontier` rate-matches the GEN curve and takes ctx only as
    its anchor, so a ctx campaign reads the `run_*.json` its own case
    left -- whose `performance.request_throughput_req_s` is the field the
    harness itself requires before calling the case successful.
    """
    case = (
        tmp_path
        / "baseline"
        / sol_track.RUN_SUBDIR
        / "bm_ctx"
        / "ctx_1024_1_ratio1_16_16640_dep4_MTP3_test2"
    )
    case.mkdir(parents=True)
    (case / "run_dep4_MTP3.json").write_text(
        json.dumps({"performance": {"request_throughput_req_s": 65.405}})
    )
    written = sol_track.collect(_task("ctx"), tmp_path / "baseline")
    assert [p.parent.name for p in written] == ["concurrency_16"]
    payload = json.loads(written[0].read_text())
    assert payload["avg_request_throughput_req_s"] == 65.405
    assert payload["source_field"] == "performance.request_throughput_req_s"


def test_a_ctx_run_with_nothing_validated_says_which_command_reports_why(tmp_path):
    (tmp_path / "baseline" / sol_track.RUN_SUBDIR / "bm_ctx").mkdir(parents=True)
    with pytest.raises(sol_track.SolTrackError, match="jobs check"):
        sol_track.collect(_task("ctx"), tmp_path / "baseline")


# ------------------------------------------------------------------ guards


def _overlay(tmp_path, body: dict):
    stage = _sweep_dir(tmp_path)
    tuning = tmp_path / "tuning.yaml"
    tuning.write_text(yaml.safe_dump(body), encoding="utf-8")
    return {SOL_TRACK_FIELD: {"track": "gen", "workspace": "/w", "sweep": str(stage)}}, tuning


def test_an_overlay_may_not_move_the_operating_point(tmp_path):
    """The sweep row's knobs ARE the point; changing one voids the comparison.

    The run succeeds, the number is plausible, and the only trace is a
    node count in a log line.
    """
    task, tuning = _overlay(tmp_path, {"tensor_parallel_size": 8, "moe_config": {"backend": "X"}})
    with pytest.raises(sol_track.SolTrackError, match="tensor_parallel_size"):
        sol_track.apply_overlay(task, tuning)


def test_a_knob_that_is_not_the_operating_point_is_still_tunable(tmp_path):
    task, tuning = _overlay(tmp_path, {"moe_config": {"backend": "TRTLLM"}})
    written = sol_track.apply_overlay(task, tuning)
    assert yaml.safe_load(written.read_text())["gen_extra_llm_api"] == {
        "moe_config": {"backend": "TRTLLM"}
    }


def test_the_campaign_measures_a_copy_and_never_writes_the_original(tmp_path):
    """Nothing restores an edited sweep, so the next campaign inherits it."""
    original = _sweep_dir(tmp_path)
    task = {SOL_TRACK_FIELD: {"track": "gen", "workspace": "/w", "sweep": str(original)}}
    ws = tmp_path / "ws"
    ws.mkdir()

    adopted = sol_track.adopt_sweep(task, ws)
    assert adopted == ws / "sweep" / "sweep.yaml"
    assert task[SOL_TRACK_FIELD]["adopted_from"] == str(original)
    # The generator beside the sweep travels with it: the harness imports
    # the model plugin from that directory.
    assert (ws / "sweep" / "gen_worker_config.py").is_file()

    tuning = tmp_path / "t.yaml"
    tuning.write_text(yaml.safe_dump({"moe_config": {"backend": "TRTLLM"}}), encoding="utf-8")
    sol_track.apply_overlay(task, tuning)
    assert "gen_extra_llm_api" not in yaml.safe_load(original.read_text())
    assert yaml.safe_load((ws / "sweep" / "sweep.yaml").read_text())["gen_extra_llm_api"]


def test_a_resumed_run_keeps_measuring_what_it_started_with(tmp_path):
    task = {
        SOL_TRACK_FIELD: {"track": "gen", "workspace": "/w", "sweep": str(_sweep_dir(tmp_path))}
    }
    ws = tmp_path / "ws"
    ws.mkdir()
    sol_track.adopt_sweep(task, ws)
    (ws / "sweep" / "sweep.yaml").write_text(
        yaml.safe_dump({**GEN_SWEEP, "gen_extra_llm_api": {"mid": "flight"}}), encoding="utf-8"
    )
    sol_track.adopt_sweep(dict(task), ws)
    assert yaml.safe_load((ws / "sweep" / "sweep.yaml").read_text())["gen_extra_llm_api"] == {
        "mid": "flight"
    }


# ------------------------------------------------------------------ the prompt


def _sections():
    from agent_flow.workflows.perf_optimize.prompts._common import SOL_TRACK_CTX, SOL_TRACK_GEN

    return SOL_TRACK_CTX, SOL_TRACK_GEN


def test_neither_track_is_told_to_hand_write_its_result():
    """The code that lands a score and the prompt that asks for it must agree.

    `collect` grew a ctx path once while the ctx prompt still said "write
    it down yourself", and the first real ctx campaign duly hand-wrote the
    JSON -- correctly, as it happened, by the route the gen track had
    already stopped using.
    """
    for section in _sections():
        assert "--collect" in section
        assert "Do not hand-write that JSON" in section


def test_only_the_track_that_has_a_frontier_is_told_to_build_one():
    """`process frontier` is GEN-only and answers with an error otherwise.

    A ctx agent told to run it would spend a step learning that, on a run
    that measured perfectly.
    """
    ctx, gen = _sections()
    assert "process frontier" in gen
    assert "ctx_json" in gen
    assert "There is no `process frontier` on this track" in ctx


def test_both_gen_scorers_are_given_with_the_rule_for_choosing():
    """One command per declaration, and neither is the other's fallback.

    The gen track can be scored with or without an anchor. Naming only the
    frontier stranded an unanchored campaign; naming both without the rule
    would invite an agent to reach for a frontier by supplying an anchor
    nobody declared, which is the one mistake whose output looks right.
    """
    _, gen = _sections()
    assert "ibc-bench process frontier" in gen
    assert bench_cli.GEN_ONLY_MODULE in gen
    assert "Never run (a) with an anchor `task.yaml` did not declare" in gen


def test_a_missing_gen_only_csv_names_the_command_that_writes_it():
    """The step the agent skipped, not the file the reader wanted.

    The extractor has no `ibc-bench` subcommand, so an agent that only
    knows the CLI has no way to guess it -- the error has to carry it.
    """
    with pytest.raises(bench_cli.BenchCliError, match=bench_cli.GEN_ONLY_MODULE):
        bench_cli.gen_only_points(Path("/nonexistent-run-dir"))


def test_the_gate_is_stated_to_be_the_same_number_either_way():
    """Otherwise the two paths read as two different metrics.

    They are one formula -- `accept_rate / avg_iteration_time` -- and what
    the anchor adds is only the end-to-end view on top.
    """
    _, gen = _sections()
    assert "accept_rate / avg_iteration_time" in gen
    assert "ABSENT" in gen
    assert "flattering" in gen


def test_the_work_dir_is_derived_and_the_prompt_says_why():
    """`-w` is not a free choice, and the reason is not obvious."""
    for section in _sections():
        assert "carries no code identity" in section
        assert "the second overwrites the first" in section


def test_attribution_is_stated_as_the_reader_s_job():
    """Nothing in this stack fingerprints a configuration any more.

    So the obligation the tool used to discharge is named, with the
    artifact that discharges it -- rather than left implied and lost.
    """
    for section in _sections():
        assert 'Never infer "it took effect" from "the number moved."' in section
        assert "gen_config.yaml" in section


def test_the_skills_the_config_repo_ships_are_named_where_they_apply():
    """They carry knowledge that is in no `--help`.

    `sol-postprocess`'s first step is not an action but a check: that the
    run's archived sweep carries a measured `accept_rate`, because the
    built-in table it replaced was 15 % high on one model and multiplies
    the metric linearly with no symptom.
    """
    ctx, gen = _sections()
    assert "sol-postprocess" in gen
    assert "15 %" in gen
    for section in (ctx, gen):
        assert "check-job" in section
