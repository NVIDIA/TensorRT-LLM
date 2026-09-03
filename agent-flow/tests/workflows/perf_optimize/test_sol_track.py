"""Tests for the ``sol_track`` block.

The authority rule is the one :mod:`.disagg` states — the sweep owns the
measurement conditions, ``task.yaml`` owns only the campaign knobs — but
what the sweep *expands to* is no longer computed here. It is asked of
``bench-disagg sweep plan``, so these tests stub that boundary and pin
what this module does with the answer.

Stubbing rather than invoking is deliberate: the CLI is a real
dependency of a real campaign, but a unit test that needed it installed
would be an integration test wearing a disguise, and would stop running
on any machine that has not pip-installed the benchmark wheel.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.perf_optimize import bench_cli, sol_track, task_schema
from agent_flow.workflows.perf_optimize.sol_track import SOL_TRACK_FIELD

SWEEP = {
    "name": "probe",
    "options": {"accept_rate": "3:2.48"},
    "server_config": "server-aga-gb300.config",
    "stages": {
        "ctx": {"enabled": False, "config": "ctx_stage.yaml"},
        "gen": {"enabled": True, "config": "gen_stage.yaml"},
    },
}

#: The shape `sweep plan --cases` returns, taken from the installed 0.2.0
#: CLI against a real two-point sweep rather than paraphrased.
PLAN = {
    "workspace": "probe",
    "workload": {
        "model": "deepseek-ai/DeepSeek-V4-Pro",
        # isl is a sizing bound, not the corpus' length -- the real
        # reference sweep pairs isl 200000 with a c190000 dataset.
        "isl": 200000,
        "osl": 1024,
        "dataset": "/data/DeepSeek-V4-Pro-coding-c190000-n128-o1024-8192_for_serve.json",
    },
    "environment": {"cluster": "aga", "gpu": "GB300"},
    "code_id": "code-15e1380839e3",
    "code": {"image": "/img.sqsh", "build": None, "code_change": None, "worker_overrides": None},
    "summary": {"fresh": 2, "inflight": 0, "measured": 0, "total": 2},
    "cases": [
        {
            "case": "gen-ctx1_gen1_tep4_b32_mnt128_gmf0.9_eplb0_mtp3_conc1",
            "stage": "gen",
            "status": "fresh",
            "config": {"gen_num": 1, "concurrency": 1, "tp_size": 4, "mtp_size": 3},
        },
        {
            "case": "gen-ctx1_gen1_dep32_b64_mnt256_gmf0.7_eplb0_mtp3_conc128",
            "stage": "gen",
            "status": "fresh",
            "config": {"gen_num": 1, "concurrency": 128, "tp_size": 32, "mtp_size": 3},
        },
    ],
}


@pytest.fixture
def planned(monkeypatch):
    """Stub `sweep plan`, and record what it was asked."""
    calls: list = []

    def _plan(sweep, workspace, **kwargs):
        calls.append({"sweep": str(sweep), "workspace": workspace, **kwargs})
        return _plan.result

    _plan.result = PLAN
    monkeypatch.setattr(task_schema, "sweep_plan", _plan)
    return _plan, calls


def _write(tmp_path, name: str, payload: dict):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_task(tmp_path, extra: dict | None = None):
    for sub in ("ckpt", "repo"):
        (tmp_path / sub).mkdir(exist_ok=True)
    data = {
        "checkpoint_path": str(tmp_path / "ckpt"),
        "trtllm_repo_path": str(tmp_path / "repo"),
    }
    data.update(extra or {})
    return _write(tmp_path, "task.yaml", data)


def _block(tmp_path, track: str = "gen", sweep: dict | None = None, **extra):
    path = _write(tmp_path, "sweep.yaml", sweep if sweep is not None else SWEEP)
    return {"track": track, "sweep": str(path), "workspace": "probe", **extra}


def _gen(tmp_path, **extra):
    """A gen block with an anchor -- required, so most tests want it."""
    anchor = tmp_path / "anchor.json"
    anchor.write_text("[]", encoding="utf-8")
    return _block(tmp_path, ctx_json=str(anchor), **extra)


# ------------------------------------------------------------------ the plan


def test_the_plan_is_the_authority_for_points_and_lengths(tmp_path, planned):
    """Nothing here re-expands the sweep; it reads what will actually run."""
    _, calls = planned
    data = task_schema.load_and_validate_task_yaml(
        _write_task(tmp_path, {SOL_TRACK_FIELD: _gen(tmp_path)})
    )
    assert data["benchmark"]["concurrency"] == [1, 128]
    assert data["optimize"]["target_metric"] == "throughput_per_user"
    assert data["profile"]["methods"] == ["nsys"]
    assert task_schema.is_curve_mode(data)
    # The workspace is passed through, so plan and submit address the same one.
    assert calls[0]["workspace"] == "probe"
    # And the code the points were planned against is recorded.
    assert "code-15e1380839e3" in " ".join(data[SOL_TRACK_FIELD]["filled_from_sweep_plan"])


def test_the_operating_point_is_the_product_of_concurrency_and_gen_num(tmp_path, planned):
    """The one trap that survives driving the CLI.

    A row's concurrency is per generation server. The case name keeps the
    listed value because that is its address; `task.yaml` must carry the
    product, because that is what the client is driven at.
    """
    plan, _ = planned
    plan.result = {
        **PLAN,
        "cases": [
            {"case": "gen-a", "stage": "gen", "config": {"gen_num": 2, "concurrency": 64}},
        ],
    }
    data = task_schema.load_and_validate_task_yaml(
        _write_task(tmp_path, {SOL_TRACK_FIELD: _gen(tmp_path)})
    )
    assert data["benchmark"]["concurrency"] == 128  # 64 x 2, single point -> scalar


def test_a_track_the_sweep_does_not_measure_is_refused(tmp_path, planned):
    """A ctx campaign against a gen-only sweep would measure nothing.

    The stage is disabled in the sweep, so `plan` returns no ctx cases and
    the campaign would run to the first submit before noticing.
    """
    with pytest.raises(task_schema.TaskSchemaError, match="plans no 'ctx' cases"):
        task_schema.load_and_validate_task_yaml(
            _write_task(tmp_path, {SOL_TRACK_FIELD: _block(tmp_path, track="ctx")})
        )


def test_a_failing_cli_is_surfaced_with_its_error_code(tmp_path, monkeypatch):
    """The taxonomy is what a caller branches on, so it must not be swallowed."""

    def _boom(*a, **k):
        raise bench_cli.BenchCliError("workload changed", code="CONTEXT_CHANGED")

    monkeypatch.setattr(task_schema, "sweep_plan", _boom)
    with pytest.raises(task_schema.TaskSchemaError, match=r"\[CONTEXT_CHANGED\]"):
        task_schema.load_and_validate_task_yaml(
            _write_task(tmp_path, {SOL_TRACK_FIELD: _gen(tmp_path)})
        )


# ------------------------------------------------------------------ the block


def test_a_sweep_without_an_accept_rate_is_refused(tmp_path, planned):
    """Every `frontier build` requires it and none is inferred.

    Without it a campaign measures perfectly well and then cannot be
    turned into a curve — after the cluster time is spent. Cheaper here.
    """
    sweep = {**SWEEP, "options": {}}
    with pytest.raises(task_schema.TaskSchemaError, match="options.accept_rate"):
        task_schema.load_and_validate_task_yaml(
            _write_task(tmp_path, {SOL_TRACK_FIELD: _block(tmp_path, sweep=sweep)})
        )


def test_the_workspace_is_required(tmp_path, planned):
    """It is what makes an attempt comparable to the baseline it must beat."""
    path = _write(tmp_path, "sweep.yaml", SWEEP)
    with pytest.raises(task_schema.TaskSchemaError, match="sol_track.workspace"):
        task_schema.load_and_validate_task_yaml(
            _write_task(tmp_path, {SOL_TRACK_FIELD: {"track": "gen", "sweep": str(path)}})
        )


def test_a_missing_sweep_file_is_refused(tmp_path, planned):
    with pytest.raises(task_schema.TaskSchemaError, match="is not a file"):
        task_schema.load_and_validate_task_yaml(
            _write_task(
                tmp_path,
                {SOL_TRACK_FIELD: {"track": "gen", "workspace": "w", "sweep": "/nope.yaml"}},
            )
        )


def test_it_cannot_be_combined_with_a_disagg_campaign(tmp_path, planned):
    """Both reconcile `benchmark` from a different file; one would lose."""
    harness = _write(tmp_path, "harness.yaml", {})
    with pytest.raises(task_schema.TaskSchemaError, match="cannot be combined"):
        task_schema.load_and_validate_task_yaml(
            _write_task(
                tmp_path,
                {
                    SOL_TRACK_FIELD: _block(tmp_path),
                    "disagg": {"config": str(harness)},
                },
            )
        )


# ------------------------------------------------------------------ authority


def test_a_condition_the_user_wrote_that_disagrees_is_an_error(tmp_path, planned):
    """Not a silent overwrite — the same rule the disagg block states.

    Stated on `concurrency`, which is the condition that matters: it is
    what the campaign is scored per point at, so a task.yaml naming a
    point the sweep never plans would have every stage quoting a
    measurement that does not exist.
    """
    with pytest.raises(task_schema.TaskSchemaError, match="contradicts the sweep"):
        task_schema.load_and_validate_task_yaml(
            _write_task(
                tmp_path,
                {
                    SOL_TRACK_FIELD: _gen(tmp_path),
                    "benchmark": {"concurrency": [1, 999]},
                },
            )
        )


def test_the_corpus_is_reconciled_too(tmp_path, planned):
    """A task.yaml naming another dataset is the same class of error."""
    with pytest.raises(task_schema.TaskSchemaError, match="contradicts the sweep"):
        task_schema.load_and_validate_task_yaml(
            _write_task(
                tmp_path,
                {
                    SOL_TRACK_FIELD: _gen(tmp_path),
                    "benchmark": {"dataset_path": "/data/some-other-corpus.json"},
                },
            )
        )


def test_an_owner_who_names_another_metric_still_wins(tmp_path, planned):
    data = task_schema.load_and_validate_task_yaml(
        _write_task(
            tmp_path,
            {
                SOL_TRACK_FIELD: _gen(tmp_path),
                "optimize": {"target_metric": "tps_per_gpu"},
            },
        )
    )
    assert data["optimize"]["target_metric"] == "tps_per_gpu"
    filled = " ".join(data[SOL_TRACK_FIELD]["filled_from_sweep_plan"])
    assert "random_input_len" in filled and "target_metric" not in filled


# ------------------------------------------------------------------ the overlay


def test_the_tuning_seed_is_a_partial_overlay_not_a_role_config(tmp_path):
    """`bench-disagg` deep-merges it onto the config the sweep row generated.

    So the seed carries only what the sweep already overrode — usually
    nothing — and the topology the row owns stays outside the one file
    the optimizer edits.
    """
    from agent_flow.workflows.perf_optimize.sol_track import tuning_seed_yaml

    _write(tmp_path, "gen_stage.yaml", {"gen_configs": [], "gen_extra_llm_api": {"a": 1}})
    seeded = tuning_seed_yaml({SOL_TRACK_FIELD: _block(tmp_path)})
    assert yaml.safe_load(seeded) == {"a": 1}

    _write(tmp_path, "gen_stage.yaml", {"gen_configs": []})
    empty = tuning_seed_yaml({SOL_TRACK_FIELD: _block(tmp_path)})
    assert yaml.safe_load(empty) == {}


# ------------------------------------------------------------------ the prompts


def test_only_the_running_track_s_section_is_composed():
    """Selective composition, the same way disagg / slurm / sol work."""
    from agent_flow.workflows.perf_optimize.prompts import build_perf_optimize_prompts
    from agent_flow.workflows.perf_optimize.prompts._common import SOL_TRACK_CTX, SOL_TRACK_GEN

    aggregate = build_perf_optimize_prompts()
    gen = build_perf_optimize_prompts(sol_track="gen")
    ctx = build_perf_optimize_prompts(sol_track="ctx")
    for role in ("benchmarker", "analyzer", "optimizer", "evaluator", "qa"):
        assert SOL_TRACK_GEN not in getattr(aggregate, role)
        assert SOL_TRACK_GEN in getattr(gen, role)
        assert SOL_TRACK_CTX in getattr(ctx, role)
        assert SOL_TRACK_CTX not in getattr(gen, role)
        # Composed last, or the guidance it supersedes wins on position.
        assert getattr(gen, role).rstrip().endswith(SOL_TRACK_GEN.rstrip())
    assert SOL_TRACK_GEN not in gen.reporter and SOL_TRACK_GEN not in gen.projector


# ------------------------------------------------------------------ the anchor


def test_a_gen_track_without_a_ctx_anchor_is_refused(tmp_path, planned):
    """`frontier build` rate-matches the whole curve, so it needs the ctx rate.

    The gate's metric is purely generation-side, which is what makes this
    easy to miss: everything measures fine, and the build then raises
    ANCHOR_MISSING with the cluster time already spent.
    """
    with pytest.raises(task_schema.TaskSchemaError, match="a gen track needs a CTX anchor"):
        task_schema.load_and_validate_task_yaml(
            _write_task(tmp_path, {SOL_TRACK_FIELD: _block(tmp_path)})
        )


def test_an_enabled_ctx_stage_is_one_way_to_have_one(tmp_path, planned):
    sweep = {**SWEEP, "stages": {**SWEEP["stages"], "ctx": {"enabled": True, "config": "c.yaml"}}}
    data = task_schema.load_and_validate_task_yaml(
        _write_task(tmp_path, {SOL_TRACK_FIELD: _block(tmp_path, sweep=sweep)})
    )
    assert data["benchmark"]["concurrency"] == [1, 128]


def test_an_existing_ctx_json_is_the_other(tmp_path, planned):
    anchor = tmp_path / "ctx_anchor.json"
    anchor.write_text("[]", encoding="utf-8")
    data = task_schema.load_and_validate_task_yaml(
        _write_task(tmp_path, {SOL_TRACK_FIELD: _block(tmp_path, ctx_json=str(anchor))})
    )
    assert data["benchmark"]["concurrency"] == [1, 128]


def test_a_ctx_json_that_is_not_a_file_is_refused(tmp_path, planned):
    with pytest.raises(task_schema.TaskSchemaError, match="ctx_json' is not a file"):
        task_schema.load_and_validate_task_yaml(
            _write_task(tmp_path, {SOL_TRACK_FIELD: _block(tmp_path, ctx_json="/nope.json")})
        )


# --------------------------------------------------------------- the score

#: What `frontier show` answers, shaped like the snapshot `frontier.py`
#: writes: every scored point, `frontier: true` marking the edge, and the
#: gen metric spelled `tps_per_user`.
SNAPSHOT = {
    "snapshot_id": "snap-0002-latest",
    "workspace": "probe",
    "select": "latest",
    "code": {"mixed": False, "code_ids": ["code-15e1380839e3"]},
    "points": [
        {
            "case": PLAN["cases"][0]["case"],
            "config": {"gen_num": 1, "concurrency": 1, "tp_size": 4},
            "metrics": {
                "tps_per_user": 164.5556,
                "tps_per_gpu": 39.809,
                "ctx_per_gen": 0.016735,
            },
            "frontier": True,
            "samples": 2,
            "config_id": "be0ba392",
            "code_id": "code-15e1380839e3",
        },
        {
            "case": PLAN["cases"][1]["case"],
            "config": {"gen_num": 2, "concurrency": 64, "tp_size": 4},
            "metrics": {
                "tps_per_user": 65.8497,
                "tps_per_gpu": 368.7572,
                "ctx_per_gen": 0.21532,
            },
            "frontier": False,
            "samples": 1,
            "config_id": "887735c9",
            "code_id": "code-15e1380839e3",
        },
    ],
}


@pytest.fixture
def snapshot(monkeypatch):
    """Stub `frontier show`, and record what it was asked."""
    calls: list = []

    def _show(workspace, snapshot="latest"):
        calls.append({"workspace": workspace, "snapshot": snapshot})
        return _show.result

    _show.result = SNAPSHOT
    monkeypatch.setattr(sol_track, "frontier_show", _show)
    return _show, calls


def _task(track: str = "gen", **extra):
    return {SOL_TRACK_FIELD: {"track": track, "workspace": "probe", **extra}}


def _collected(tmp_path, into, **extra):
    return sol_track.collect(_task(**extra), into)


def test_the_snapshot_metric_is_written_under_the_name_the_campaign_scores(tmp_path, snapshot):
    """`tps_per_user` in, `throughput_per_user` out.

    The rename is the whole reason this step exists: the baseline gate
    looks for `optimize.target_metric`, so a file carrying the CLI's
    spelling reads as a stage that measured nothing.
    """
    _collected(tmp_path, tmp_path / "baseline")
    payload = json.loads((tmp_path / "baseline/concurrency_1/sol_result.json").read_text())
    assert payload["throughput_per_user"] == 164.5556
    assert "tps_per_user" not in payload
    # Kept, but only as provenance -- reading it back is not the contract.
    assert payload["snapshot_metrics"]["tps_per_user"] == 164.5556


def test_the_directory_is_named_for_the_total_in_flight_not_the_listed_value(tmp_path, snapshot):
    """The per-generation-server trap, on the way back out.

    The second point lists 64 at `gen_num: 2`, so the client drove 128 and
    that is the point `task.yaml` names. Writing `concurrency_64` would
    file a measurement against an operating point nobody ran.
    """
    written = _collected(tmp_path, tmp_path / "baseline")
    assert [path.parent.name for path in written] == ["concurrency_1", "concurrency_128"]


def test_it_writes_wherever_the_stage_was_told_to(tmp_path, snapshot):
    """Not just the baseline: every attempt has its own result directory."""
    written = _collected(tmp_path, tmp_path / "rounds/r1/attempt_1")
    assert written[0] == tmp_path / "rounds/r1/attempt_1/concurrency_1/sol_result.json"


def test_which_selection_produced_the_number_travels_with_it(tmp_path, snapshot):
    """`best` and `latest` are different questions; a bare number hides which."""
    _collected(tmp_path, tmp_path / "baseline")
    payload = json.loads((tmp_path / "baseline/concurrency_1/sol_result.json").read_text())
    assert payload["select"] == "latest"
    assert payload["snapshot_id"] == "snap-0002-latest"
    assert payload["config_id"] == "be0ba392"


def test_two_points_at_one_concurrency_are_refused_rather_than_overwritten(tmp_path, snapshot):
    """One directory per operating point, so a collision loses a measurement.

    Silently keeping the last one would score the campaign on whichever
    case the snapshot happened to list second.
    """
    snapshot[0].result = {
        **SNAPSHOT,
        "points": [
            {**SNAPSHOT["points"][0], "case": "a", "config": {"gen_num": 1, "concurrency": 8}},
            {**SNAPSHOT["points"][1], "case": "b", "config": {"gen_num": 2, "concurrency": 4}},
        ],
    }
    with pytest.raises(sol_track.SolTrackError, match="both measure 8 requests in flight"):
        _collected(tmp_path, tmp_path / "baseline")


def test_a_snapshot_that_scored_nothing_names_the_command_that_says_why(tmp_path, snapshot):
    snapshot[0].result = {**SNAPSHOT, "points": []}
    with pytest.raises(sol_track.SolTrackError, match="sweep status"):
        _collected(tmp_path, tmp_path / "baseline")


def test_a_point_missing_its_metric_is_skipped_not_defaulted(tmp_path, snapshot):
    snapshot[0].result = {
        **SNAPSHOT,
        "points": [SNAPSHOT["points"][0], {**SNAPSHOT["points"][1], "metrics": {}}],
    }
    written = _collected(tmp_path, tmp_path / "baseline")
    assert [path.parent.name for path in written] == ["concurrency_1"]


def test_a_ctx_campaign_is_scored_from_the_artifact_the_cli_validated_it_on(tmp_path, monkeypatch):
    """No snapshot exists to read, and none is needed.

    `frontier build` selects stage == GEN and refuses a workspace holding
    none, taking the ctx side as its anchor -- so a ctx campaign has no
    curve. `sweep status` hands over the `run_*.json` the backend
    validated the case on, and the key it validated is the number.
    """
    run_json = tmp_path / "run_dep8_MTP3.json"
    run_json.write_text(json.dumps({"performance": {"request_throughput_req_s": 10.67}}))
    monkeypatch.setattr(
        sol_track,
        "status",
        lambda ws: {
            "cases": [
                {
                    "case": "ctx-isl200000_osl1_b16_tp4_mtp3_ratio0.8",
                    "stage": "ctx",
                    "state": "success",
                    # A ctx case has no `concurrency`; `max_batch` is the
                    # in-flight request count for a prefill-only run.
                    "config": {"isl": 200000, "osl": 1, "max_batch": 16, "tp_size": 4},
                    "result": str(run_json),
                    "workdir": "/w/m-0001",
                },
                {"case": "gen-x", "stage": "gen", "state": "success", "config": {}},
            ]
        },
    )
    written = sol_track.collect(_task("ctx"), tmp_path / "baseline")
    assert [p.parent.name for p in written] == ["concurrency_16"]
    payload = json.loads(written[0].read_text())
    assert payload["avg_request_throughput_req_s"] == 10.67
    assert payload["result"] == str(run_json)


def test_a_ctx_case_that_did_not_validate_is_skipped_with_its_state(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sol_track,
        "status",
        lambda ws: {
            "cases": [
                {
                    "case": "ctx-a",
                    "stage": "ctx",
                    "state": "failed",
                    "config": {"max_batch": 16},
                    "result": None,
                }
            ]
        },
    )
    with pytest.raises(sol_track.SolTrackError, match=r"ctx-a \(failed\)"):
        sol_track.collect(_task("ctx"), tmp_path / "baseline")


def test_the_owner_s_metric_is_what_gets_written(tmp_path, snapshot):
    """`apply_plan` only defaults `target_metric`; an owner may name another.

    Writing the track's own spelling instead produces a file the baseline
    gate cannot see, and the gate then says the stage measured nothing.
    """
    task = _task()
    task["optimize"] = {"target_metric": "my_own_metric"}
    sol_track.collect(task, tmp_path / "baseline")
    payload = json.loads((tmp_path / "baseline/concurrency_1/sol_result.json").read_text())
    assert payload["my_own_metric"] == 164.5556
    assert "throughput_per_user" not in payload


def test_a_snapshot_built_before_this_attempt_s_overlay_is_refused(tmp_path, snapshot):
    """`--snapshot latest` says nothing about WHICH attempt built it.

    Skip the build (or have it fail) and this step still succeeds, writing
    the previous attempt's numbers into this attempt's directory. The
    evaluator then measures zero and rejects a change nobody ran.
    """
    stage = tmp_path / "gen_stage.yaml"
    stage.write_text(yaml.safe_dump({"gen_extra_llm_api": {"knob": "new"}}), encoding="utf-8")
    sweep = _write(
        tmp_path,
        "sweep.yaml",
        {**SWEEP, "stages": {"gen": {"enabled": True, "config": "gen_stage.yaml"}}},
    )
    task = _task(sweep=str(sweep))
    snapshot[0].result = {
        **SNAPSHOT,
        "code": {"mixed": False, "detail": {"code-old": {"worker_overrides": {}}}},
    }
    with pytest.raises(sol_track.SolTrackError, match="predates the overlay"):
        sol_track.collect(task, tmp_path / "attempt_1")

    # ... and passes once the snapshot carries the same overlay.
    snapshot[0].result = {
        **SNAPSHOT,
        "code": {
            "mixed": False,
            "detail": {"code-new": {"worker_overrides": {"gen_extra_llm_api": {"knob": "new"}}}},
        },
    }
    assert sol_track.collect(task, tmp_path / "attempt_1")


def test_a_mixed_code_curve_is_recorded_not_refused(tmp_path, snapshot):
    """The mixing is a property of the CURVE; this point's code is definite.

    The prompt tells the evaluator a `code.mixed` curve is not evidence.
    That judgement is the reader's -- Python's job is to make sure the
    reader can see it.
    """
    snapshot[0].result = {**SNAPSHOT, "code": {"mixed": True, "ids": ["code-a", "code-b"]}}
    sol_track.collect(_task(), tmp_path / "baseline")
    payload = json.loads((tmp_path / "baseline/concurrency_1/sol_result.json").read_text())
    assert payload["code_mixed"] is True
    assert payload["code_ids"] == ["code-a", "code-b"]


def test_what_a_gen_gain_is_worth_at_the_frontier_travels_with_it(tmp_path, snapshot):
    """The gate's metric is not the deployment's objective.

    `tps_per_user` is anchor-free; `tps_per_gpu` divides by a denominator
    the context side owns. So the same measured +1 % is worth different
    amounts at different points of one curve, and nothing said so.
    """
    sol_track.collect(_task(), tmp_path / "baseline")
    at_1 = json.loads((tmp_path / "baseline/concurrency_1/sol_result.json").read_text())
    at_128 = json.loads((tmp_path / "baseline/concurrency_128/sol_result.json").read_text())
    # denominator = tps_per_user * concurrency / tps_per_gpu; gen gpus = tp_size * gen_num
    assert at_1["frontier_elasticity"] == pytest.approx(4 / (164.5556 * 1 / 39.809), rel=1e-4)
    assert at_128["frontier_elasticity"] == pytest.approx(8 / (65.8497 * 128 / 368.7572), rel=1e-4)
    # The measured campaign: ~0.97 of a gen gain survives at the low end.
    assert at_1["frontier_elasticity"] == pytest.approx(0.968, abs=0.005)
    # And the high-concurrency end keeps materially less of it.
    assert at_128["frontier_elasticity"] < at_1["frontier_elasticity"]


def test_the_snapshot_can_be_named_rather_than_taken_as_latest(tmp_path, snapshot):
    sol_track.collect(_task(), tmp_path / "baseline", snapshot="baseline")
    assert snapshot[1] == [{"workspace": "probe", "snapshot": "baseline"}]


# ------------------------------------------------- ported from the script driver

# The two guards below were found by real runs of the script-driven
# implementation, not by reading this code. Neither failure raises
# anything on its own: each produces a plausible number against a
# comparison that has silently become void.


def _overlay(tmp_path, body: dict) -> tuple[dict, Path]:
    stage = tmp_path / "gen_stage.yaml"
    stage.write_text(yaml.safe_dump({"isl": 200000}), encoding="utf-8")
    sweep = _write(
        tmp_path,
        "sweep.yaml",
        {**SWEEP, "stages": {"gen": {"enabled": True, "config": "gen_stage.yaml"}}},
    )
    tuning = tmp_path / "tuning.yaml"
    tuning.write_text(yaml.safe_dump(body), encoding="utf-8")
    return {SOL_TRACK_FIELD: {"track": "gen", "workspace": "probe", "sweep": str(sweep)}}, tuning


def test_an_overlay_may_not_move_the_operating_point(tmp_path):
    """The sweep row's knobs ARE the point; changing one voids the comparison.

    Caught by a dry run against a workspace holding a stale full-config
    seed: merged as an override it took tensor_parallel_size from 4 to 8
    and the job from two nodes to three. The run succeeds, the number is
    plausible, and the only trace is a node count in a log line.
    """
    task, tuning = _overlay(tmp_path, {"tensor_parallel_size": 8, "moe_config": {"backend": "X"}})
    with pytest.raises(sol_track.SolTrackError, match="tensor_parallel_size"):
        sol_track.apply_overlay(task, tuning)
    # And nothing was written: the refusal is before the stage config is touched.
    assert "tensor_parallel_size" not in (tmp_path / "gen_stage.yaml").read_text()


def test_a_knob_that_is_not_the_operating_point_is_still_tunable(tmp_path):
    task, tuning = _overlay(tmp_path, {"moe_config": {"backend": "TRTLLM"}})
    written = sol_track.apply_overlay(task, tuning)
    assert yaml.safe_load(written.read_text())["gen_extra_llm_api"] == {
        "moe_config": {"backend": "TRTLLM"}
    }


def _anchor(tmp_path, rows) -> Path:
    path = tmp_path / "ctx_anchor.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_an_anchor_measured_on_other_work_is_refused(tmp_path):
    """A frontier is a decode rate over a prefill rate.

    Feed it halves measured at different input lengths and it still
    returns a number, on a curve that still looks like a frontier,
    describing a deployment whose two roles never served the same traffic.
    """
    anchor = _anchor(tmp_path, [{"isl": 1024, "avg_request_throughput_req_s": 91.2}])
    with pytest.raises(sol_track.SolTrackError, match="measured at isl 1024"):
        sol_track.require_matching_anchor(anchor, PLAN)


def test_a_matching_anchor_passes_and_says_so(tmp_path):
    anchor = _anchor(tmp_path, [{"isl": 200000, "avg_request_throughput_req_s": 10.67}])
    assert sol_track.require_matching_anchor(anchor, PLAN) is None


def test_an_anchor_that_states_no_isl_is_reported_as_unverified(tmp_path):
    """'not checked' and 'checked and matched' must not read alike."""
    anchor = _anchor(tmp_path, [{"avg_request_throughput_req_s": 10.67}])
    note = sol_track.require_matching_anchor(anchor, PLAN)
    assert note and "no 'isl'" in note


def test_the_anchor_check_runs_during_validation(tmp_path, planned):
    bad = _anchor(tmp_path, [{"isl": 1024}])
    with pytest.raises(task_schema.TaskSchemaError, match="measured at isl 1024"):
        task_schema.load_and_validate_task_yaml(
            _write_task(tmp_path, {SOL_TRACK_FIELD: _block(tmp_path, ctx_json=str(bad))})
        )


def test_a_matching_anchor_is_recorded_in_the_backfill_notes(tmp_path, planned):
    good = _anchor(tmp_path, [{"isl": 200000}])
    data = task_schema.load_and_validate_task_yaml(
        _write_task(tmp_path, {SOL_TRACK_FIELD: _block(tmp_path, ctx_json=str(good))})
    )
    assert any("anchor" in note for note in data[SOL_TRACK_FIELD]["filled_from_sweep_plan"])


# ------------------------------------------------------------- the workload


def test_the_corpus_is_the_workload_not_the_sizing_bound(tmp_path, planned):
    """`isl` bounds the KV allocation; the dataset is what gets served.

    The checked-in reference sweep pairs `isl: 200000` with a
    `...-c190000-...` corpus and both numbers are right. Copying the
    bound into `random_input_len` asserted a synthetic dataset of
    uniformly 200000-token requests -- neither of which is true -- and
    every prompt and the report then quoted it.
    """
    data = task_schema.load_and_validate_task_yaml(
        _write_task(tmp_path, {SOL_TRACK_FIELD: _gen(tmp_path)})
    )
    bench = data["benchmark"]
    assert bench["dataset_name"] == "DeepSeek-V4-Pro-coding-c190000-n128-o1024-8192_for_serve.json"
    assert bench["dataset_path"].endswith("_for_serve.json")
    # Not merely unset by the sweep: actively removed, because the schema
    # defaults seed them for every campaign and 1024 is no truer than 200000.
    assert "random_input_len" not in bench
    assert "random_output_len" not in bench
    notes = " ".join(data[SOL_TRACK_FIELD]["filled_from_sweep_plan"])
    assert "sequence-length bounds" in notes and "NOT the corpus" in notes


def test_a_sweep_naming_no_corpus_still_gets_its_lengths(tmp_path, planned):
    """The fallback: with no dataset, the lengths really are the shape."""
    plan, _ = planned
    plan.result = {**PLAN, "workload": {"isl": 8192, "osl": 1024}}
    data = task_schema.load_and_validate_task_yaml(
        _write_task(tmp_path, {SOL_TRACK_FIELD: _gen(tmp_path)})
    )
    assert data["benchmark"]["random_input_len"] == 8192
    assert data["benchmark"]["random_output_len"] == 1024
    assert "dataset_path" not in data["benchmark"]


def test_a_user_who_states_a_length_anyway_keeps_it(tmp_path, planned):
    """Dropping a default is not the same as overruling the author."""
    block = {SOL_TRACK_FIELD: _gen(tmp_path)}
    task = _write_task(tmp_path, {**block, "benchmark": {"random_input_len": 190000}})
    data = task_schema.load_and_validate_task_yaml(task)
    assert data["benchmark"]["random_input_len"] == 190000


# ----------------------------------------------------------------- the copy


def _sweep_dir(tmp_path) -> Path:
    """A sweep as it really is: a directory whose files reference siblings."""
    root = tmp_path / "campaign"
    root.mkdir()
    (root / "gen_stage.yaml").write_text(
        yaml.safe_dump({"isl": 200000, "gen_extra_llm_api": {"user_knob": 1}}), encoding="utf-8"
    )
    (root / "server.config").write_text("PARTITION=batch\n", encoding="utf-8")
    (root / "gen_worker_config.py").write_text("# imported from beside the stage config\n")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "gen_worker_config.pyc").write_text("stale")
    (root / "sweep.yaml").write_text(
        yaml.safe_dump({**SWEEP, "stages": {"gen": {"enabled": True, "config": "gen_stage.yaml"}}}),
        encoding="utf-8",
    )
    return root / "sweep.yaml"


def test_the_campaign_measures_a_copy_and_never_writes_the_original(tmp_path):
    """Nothing restores an edited sweep, so the next campaign inherits it.

    That campaign then seeds its tuning file from the previous one's
    accepted overlay and calls the result a baseline -- measured with an
    optimization already applied, reported as the sweep's own.
    """
    original = _sweep_dir(tmp_path)
    before = original.parent / "gen_stage.yaml"
    task = {SOL_TRACK_FIELD: {"track": "gen", "workspace": "probe", "sweep": str(original)}}
    ws = tmp_path / "ws"
    ws.mkdir()

    adopted = sol_track.adopt_sweep(task, ws)
    assert adopted == ws / "sweep" / "sweep.yaml"
    # The spec now names the copy, so every agent and apply_overlay follow it.
    assert task[SOL_TRACK_FIELD]["sweep"] == str(adopted)
    assert task[SOL_TRACK_FIELD]["adopted_from"] == str(original)

    tuning = tmp_path / "tuning.yaml"
    tuning.write_text(yaml.safe_dump({"moe_config": {"backend": "TRTLLM"}}), encoding="utf-8")
    sol_track.apply_overlay(task, tuning)

    # The user's file is byte-identical; the copy carries the campaign's edit.
    assert yaml.safe_load(before.read_text())["gen_extra_llm_api"] == {"user_knob": 1}
    written = yaml.safe_load((ws / "sweep" / "gen_stage.yaml").read_text())
    assert written["gen_extra_llm_api"] == {"moe_config": {"backend": "TRTLLM"}}


def test_the_whole_directory_travels_because_a_stage_config_has_siblings(tmp_path):
    """The unit of copy is the directory, not the file.

    A stage config resolves its neighbours relative to itself, and the
    harness imports the model's config generator from beside it.
    """
    task = {
        SOL_TRACK_FIELD: {"track": "gen", "workspace": "probe", "sweep": str(_sweep_dir(tmp_path))}
    }
    ws = tmp_path / "ws"
    ws.mkdir()
    sol_track.adopt_sweep(task, ws)
    copied = {p.name for p in (ws / "sweep").iterdir()}
    assert {"sweep.yaml", "gen_stage.yaml", "server.config", "gen_worker_config.py"} <= copied
    # ... but not a stale import cache of that generator.
    assert "__pycache__" not in copied


def test_a_resumed_run_keeps_measuring_what_it_started_with(tmp_path):
    """Idempotent: re-adopting must not discard an in-flight campaign's edits."""
    task = {
        SOL_TRACK_FIELD: {"track": "gen", "workspace": "probe", "sweep": str(_sweep_dir(tmp_path))}
    }
    ws = tmp_path / "ws"
    ws.mkdir()
    sol_track.adopt_sweep(task, ws)
    (ws / "sweep" / "gen_stage.yaml").write_text(
        yaml.safe_dump({"isl": 200000, "gen_extra_llm_api": {"mid": "flight"}}), encoding="utf-8"
    )
    sol_track.adopt_sweep(dict(task), ws)
    assert yaml.safe_load((ws / "sweep" / "gen_stage.yaml").read_text())["gen_extra_llm_api"] == {
        "mid": "flight"
    }


# ---------------------------------------------------------------- the prompt


def test_only_the_running_track_gets_a_scoring_path_it_actually_has():
    """A prompt that names an artifact the track cannot produce is a bug.

    `frontier build` is GEN-only and the elasticity is recovered from a
    snapshot's two metrics, so neither belongs in the ctx section -- and a
    ctx agent told to run `frontier compare` gets NO_DATA on a run that
    measured perfectly.
    """
    from agent_flow.workflows.perf_optimize.prompts._common import SOL_TRACK_CTX, SOL_TRACK_GEN

    assert "frontier compare" in SOL_TRACK_GEN
    assert "frontier_elasticity" in SOL_TRACK_GEN
    assert "frontier compare" not in SOL_TRACK_CTX
    assert "frontier_elasticity" not in SOL_TRACK_CTX
    # What both tracks do share: where the artifacts are, and what the
    # workers were actually handed.
    for section in (SOL_TRACK_CTX, SOL_TRACK_GEN):
        assert "worker_overrides" in section
        assert "start_logs" in section
        assert "trtllm_wheel_path" in section


def test_the_evidence_order_puts_the_noise_floor_before_the_delta():
    """Listing the quality fields without a rule is what shipped before.

    The campaign that exposed it compared a 1-sample point against a
    3-sample point and read -1.31 % as a result, with the workspace's own
    repeatability at roughly a third of that at the other operating point.
    """
    from agent_flow.workflows.perf_optimize.prompts._common import SOL_TRACK_GEN

    order = [
        SOL_TRACK_GEN.index(token)
        for token in ("comparable: false", "from_samples", "resampled: true", "delta_pct")
    ]
    assert order == sorted(order)
    assert "smaller than the repeatability" in SOL_TRACK_GEN


def test_neither_track_is_told_to_hand_write_its_result():
    """The code that lands a score and the prompt that asks for it must agree.

    `collect` grew a ctx path while the ctx prompt still said "write it
    down yourself", so the first real ctx campaign hand-wrote the JSON --
    correctly, as it happens, but by the route the gen track stopped using
    precisely because hand-writing is where a number lands under the wrong
    key or the wrong concurrency.
    """
    from agent_flow.workflows.perf_optimize.prompts._common import SOL_TRACK_CTX, SOL_TRACK_GEN

    for section in (SOL_TRACK_CTX, SOL_TRACK_GEN):
        assert "--collect" in section
        assert "sol_result.json" not in section or "Do not hand-write" in section


# ------------------------------------------------------- a source change lands


def _code_task(tmp_path, install=None):
    stage = {"isl": 1024}
    if install is not None:
        stage["trtllm_install"] = install
    (tmp_path / "gen_stage.yaml").write_text(yaml.safe_dump(stage), encoding="utf-8")
    sweep = {**SWEEP, "stages": {"gen": {"enabled": True, "config": "gen_stage.yaml"}}}
    return _write_task(
        tmp_path,
        {
            SOL_TRACK_FIELD: _gen(tmp_path, sweep=sweep),
            "optimize": {"approaches": ["code"]},
        },
    )


def test_a_source_campaign_whose_sweep_installs_nothing_is_refused(tmp_path, planned):
    """The harness runs the image unless told otherwise.

    With no `trtllm_install`, the optimizer edits the checkout, the job
    measures the image, the number returns at the baseline, and the
    evaluator rejects an untested change as "no gain" -- a full allocation
    spent learning nothing about it, with no error anywhere.
    """
    with pytest.raises(task_schema.TaskSchemaError, match="names no 'trtllm_install'"):
        task_schema.load_and_validate_task_yaml(_code_task(tmp_path))


def test_naming_a_repo_to_build_is_enough(tmp_path, planned):
    task_schema.load_and_validate_task_yaml(
        _code_task(tmp_path, {"trtllm_repo": "/repo", "build_wheel": True})
    )


def test_naming_a_prebuilt_wheel_is_also_enough(tmp_path, planned):
    task_schema.load_and_validate_task_yaml(
        _code_task(tmp_path, {"trtllm_wheel_path": "/w/trtllm.whl"})
    )


def test_an_empty_install_block_does_not_count(tmp_path, planned):
    """`trtllm_install: {}` names nothing to install, so it installs nothing."""
    with pytest.raises(task_schema.TaskSchemaError, match="names no 'trtllm_install'"):
        task_schema.load_and_validate_task_yaml(_code_task(tmp_path, {"build_wheel": True}))


def test_a_config_only_campaign_is_unaffected(tmp_path, planned):
    """Nothing to install is the normal case when no source changes."""
    (tmp_path / "gen_stage.yaml").write_text(yaml.safe_dump({"isl": 1024}), encoding="utf-8")
    sweep = {**SWEEP, "stages": {"gen": {"enabled": True, "config": "gen_stage.yaml"}}}
    task_schema.load_and_validate_task_yaml(
        _write_task(
            tmp_path,
            {
                SOL_TRACK_FIELD: _gen(tmp_path, sweep=sweep),
                "optimize": {"approaches": ["config"]},
            },
        )
    )


def test_a_finished_campaign_hands_its_checkout_back(tmp_path):
    """A claim nothing releases is indistinguishable from a live one.

    The branch is the claim, which is what lets the guard work without a
    lock file over ssh. But the first campaign to run after that guard
    landed was refused by its own predecessor -- a campaign that had
    finished hours earlier and simply never let go.
    """
    from agent_flow.workflows.perf_optimize.workflow import PerfOptimizeWorkflow

    calls: list = []

    class _Git:
        GitOpsError = RuntimeError

        @staticmethod
        def rev_parse_head(repo):
            return _Git.head

        @staticmethod
        def checkout(repo, ref):
            calls.append(ref)

    wf = PerfOptimizeWorkflow.__new__(PerfOptimizeWorkflow)
    wf._trtllm_repo_path = lambda: "/repo"

    class _State:
        git_base_commit = "base123"

    import agent_flow.workflows.perf_optimize.workflow as mod

    original, mod.gitops = mod.gitops, _Git
    try:
        # Nothing committed: the campaign has nothing to show, so it lets go.
        _Git.head = "base123"
        wf._release_repo(_State())
        assert calls == ["base123"]

        # Something committed: the branch IS the result, so it stays put --
        # detaching would hide what was accepted from whoever looks next.
        calls.clear()
        _Git.head = "newcommit"
        wf._release_repo(_State())
        assert calls == []
    finally:
        mod.gitops = original
