"""The layer above a SOL campaign: fixing the operating point before optimizing at it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_flow.workflows.perf_optimize import bench_cli, disagg_sol

FIELD = disagg_sol.DISAGG_SOL_FIELD

HEADER = (
    "config,concurrency,mtp,tp,adp,eplb,tps_per_user,"
    "output_tps_per_gen_gpu,output_tput,avg_itertime_ms,num_iters\n"
)


def _shape_run(root: Path, name: str, rows: list[tuple]) -> Path:
    """One shape's run dir, scored by the anchor-free extractor.

    Each row is (concurrency, tp, adp, eplb, mtp, tps_per_user, per_gen_gpu).
    """
    run = Path(root) / name
    run.mkdir(parents=True, exist_ok=True)
    body = "".join(
        f"cfg_{c},{c},{mtp},{tp},{adp},{eplb},{tpu},{ppg},{tpu * c},10.0,400\n"
        for c, tp, adp, eplb, mtp, tpu, ppg in rows
    )
    (run / bench_cli.GEN_ONLY_CSV).write_text(HEADER + body, encoding="utf-8")
    return run


def _design(tmp_path: Path, *, state: dict | None = None) -> Path:
    d = tmp_path / "sweep_design"
    d.mkdir(parents=True, exist_ok=True)
    payload = state if state is not None else {"phase": "PHASE3_DONE", "model_dir": "ds-v4"}
    (d / disagg_sol.DESIGN_STATE).write_text(json.dumps(payload), encoding="utf-8")
    return d


def _task(**design) -> dict:
    return {FIELD: {"tracks": ["ctx", "gen"], "design": design}}


# ------------------------------------------------------------------ the block


def test_both_halves_are_the_default_and_the_order_is_stable():
    assert disagg_sol.tracks({FIELD: {}}) == ["ctx", "gen"]
    assert disagg_sol.tracks({FIELD: {"tracks": "gen"}}) == ["gen"]
    assert disagg_sol.tracks({FIELD: {"tracks": ["gen", "ctx", "gen"]}}) == ["gen", "ctx"]


def test_an_unknown_half_is_refused_by_name():
    with pytest.raises(disagg_sol.DisaggSolError, match=r"\['e2e'\]"):
        disagg_sol.tracks({FIELD: {"tracks": ["ctx", "e2e"]}})


def test_a_campaign_that_cannot_name_a_design_is_inheriting_its_point(tmp_path):
    """The whole point of this layer is that the point was established.

    A spec with no design dir is a spec that took its operating point from
    somewhere it cannot cite -- which is exactly the failure this module
    exists to make impossible.
    """
    with pytest.raises(disagg_sol.DisaggSolError, match="inheriting its operating point"):
        disagg_sol.design_dir({FIELD: {"design": {}}})


def test_which_end_of_the_frontier_matters_is_never_inferred():
    """The curve states the trade-off; it cannot state what it is bought for."""
    for bad in (None, "fastest", "balanced", ""):
        with pytest.raises(disagg_sol.DisaggSolError, match="cannot state which end"):
            disagg_sol.preference({FIELD: {"design": {"prefer": bad}}})
    assert disagg_sol.preference(_task(prefer="interactive")) == "interactive"
    assert disagg_sol.preference(_task(prefer="throughput")) == "throughput"


# ----------------------------------------------------------------- the design


def test_a_design_that_never_ran_is_named_as_such(tmp_path):
    with pytest.raises(disagg_sol.DisaggSolError, match="never run"):
        disagg_sol.design_state(tmp_path / "nowhere")


def test_the_design_state_is_read_not_reimplemented(tmp_path):
    d = _design(tmp_path, state={"phase": "PHASE2_PROBES_SUBMITTED", "gpu_name": "GB300"})
    assert disagg_sol.design_state(d)["phase"] == "PHASE2_PROBES_SUBMITTED"


def test_every_shape_contributes_its_own_measured_ladder(tmp_path):
    """One run dir per shape, each at ITS measured batch -- not a common one."""
    d = _design(tmp_path)
    _shape_run(
        d, "bm_tep4", [(1, 4, "False", 0, 0, 200.0, 50.0), (64, 4, "False", 0, 0, 60.0, 960.0)]
    )
    _shape_run(d, "bm_dep8", [(8, 8, "True", 0, 0, 150.0, 150.0)])
    points = disagg_sol.sweep_points(d)
    assert {p["shape"] for p in points} == {"tep_4_eplb0_mtp0", "dep_8_eplb0_mtp0"}
    assert {p["concurrency"] for p in points} == {1, 64, 8}


def test_an_unscored_design_names_the_command_that_scores_it(tmp_path):
    with pytest.raises(disagg_sol.DisaggSolError, match="get_gen_only_perf"):
        disagg_sol.sweep_points(_design(tmp_path))


def test_a_case_that_never_settled_is_dropped_not_defaulted(tmp_path):
    d = _design(tmp_path)
    run = _shape_run(d, "bm_tep4", [(1, 4, "False", 0, 0, 200.0, 50.0)])
    (run / bench_cli.GEN_ONLY_CSV).write_text(
        HEADER + "cfg_32,32,0,4,False,0,,,,10.0,400\n"
        "cfg_1,1,0,4,False,0,200.0,50.0,200.0,10.0,400\n",
        encoding="utf-8",
    )
    points = disagg_sol.sweep_points(d)
    assert [p["concurrency"] for p in points] == [1]


# ----------------------------------------------------------------- the front


def test_a_point_beaten_on_both_axes_is_dropped():
    points = [
        {
            "shape": "a",
            "concurrency": 1,
            "throughput_per_user": 200.0,
            "output_tps_per_gen_gpu": 50.0,
        },
        {
            "shape": "b",
            "concurrency": 1,
            "throughput_per_user": 190.0,
            "output_tps_per_gen_gpu": 40.0,
        },
        {
            "shape": "c",
            "concurrency": 64,
            "throughput_per_user": 60.0,
            "output_tps_per_gen_gpu": 960.0,
        },
    ]
    front = disagg_sol.pareto_front(points)
    assert {p["shape"] for p in front} == {"a", "c"}


def test_a_tie_is_kept_because_choosing_between_them_is_not_arithmetic():
    points = [
        {
            "shape": "a",
            "concurrency": 1,
            "throughput_per_user": 200.0,
            "output_tps_per_gen_gpu": 50.0,
        },
        {
            "shape": "b",
            "concurrency": 2,
            "throughput_per_user": 200.0,
            "output_tps_per_gen_gpu": 50.0,
        },
    ]
    assert len(disagg_sol.pareto_front(points)) == 2


# -------------------------------------------------------------- the selection

FRONT = [
    {
        "shape": "tep_4",
        "concurrency": 1,
        "throughput_per_user": 214.0,
        "output_tps_per_gen_gpu": 53.0,
    },
    {
        "shape": "tep_4",
        "concurrency": 32,
        "throughput_per_user": 97.0,
        "output_tps_per_gen_gpu": 782.0,
    },
    {
        "shape": "dep_32",
        "concurrency": 256,
        "throughput_per_user": 40.0,
        "output_tps_per_gen_gpu": 1200.0,
    },
]


def test_the_two_ends_of_one_curve_select_different_points():
    interactive = disagg_sol.select_point(FRONT, prefer="interactive")
    throughput = disagg_sol.select_point(FRONT, prefer="throughput")
    assert (interactive["shape"], interactive["concurrency"]) == ("tep_4", 1)
    assert (throughput["shape"], throughput["concurrency"]) == ("dep_32", 256)


def test_the_missing_deployment_view_travels_with_the_chosen_number():
    """A shape that drags more context GPUs ranks better here than it deploys.

    `output_tps_per_gen_gpu` has no context term, so the ranking this
    selection uses is not the deployment ranking -- and the reason has to be
    attached to the result, not left in a docstring, because the result is
    what a report quotes.
    """
    chosen = disagg_sol.select_point(FRONT, prefer="throughput")
    assert "output_tput_per_gpu" in chosen["e2e_view_absent"]
    assert "GENERATION" in chosen["e2e_view_absent"]
    assert "Absent, not flat" in chosen["e2e_view_absent"]


def test_the_incumbent_is_reported_against_but_never_ranked_with():
    """The incumbent is reported against, never ranked with.

    "The selection agrees with what we ran" and "it moved us" are the two
    answers this staging exists to distinguish, and only one of them leaves
    the previous campaigns' measurements meaningful.
    """
    incumbent = {"shape": "tep_4", "concurrency": 32}
    chosen = disagg_sol.select_point(FRONT, prefer="interactive", incumbent=incumbent)
    assert chosen["moved"] is True
    assert chosen["incumbent_on_pareto"] is True
    assert chosen["incumbent"] == incumbent
    # ...and the incumbent did not change what was picked.
    assert disagg_sol.select_point(FRONT, prefer="interactive")["concurrency"] == 1


def test_an_incumbent_that_is_dominated_is_said_to_be():
    dominated = {"shape": "tep_4", "concurrency": 999}
    chosen = disagg_sol.select_point(FRONT, prefer="throughput", incumbent=dominated)
    assert chosen["incumbent_on_pareto"] is False
    assert chosen["moved"] is True


def test_selecting_from_nothing_is_refused_rather_than_defaulted():
    with pytest.raises(disagg_sol.DisaggSolError, match="no measured point"):
        disagg_sol.select_point([], prefer="interactive")


def test_an_unknown_preference_is_refused_at_selection_too():
    with pytest.raises(disagg_sol.DisaggSolError, match="unknown preference"):
        disagg_sol.select_point(FRONT, prefer="latency")


# --------------------------------------------------------------- the campaigns


def test_a_design_is_reusable_once_its_sweep_is_scored(tmp_path):
    """Fixing the point costs ~10x the campaigns it enables.

    Paying that per campaign would be absurd, so "is it already established"
    has to be answerable -- and answerable from the artefacts, not from what
    `state.json` claims, because an interrupted design has the claim without
    the CSVs.
    """
    d = _design(tmp_path, state={"phase": "PHASE3_DONE"})
    assert disagg_sol.established(d) is False
    _shape_run(d, "bm_tep4", [(1, 4, "False", 0, 0, 200.0, 50.0)])
    assert disagg_sol.established(d) is True


BASE = {
    "checkpoint_path": "/ckpt",
    "optimize": {"max_rounds": 3, "approaches": ["code"]},
    "disagg_sol": {"tracks": ["ctx", "gen"], "design": {"prefer": "interactive"}},
}
CTX_POINT = {"ctx_gpus": 4, "max_batch": 2, "adp": True}
POINT = {
    "shape": "tep_4_eplb0_mtp0",
    "concurrency": 1,
    "prefer": "interactive",
    "ranked_on": "throughput_per_user",
    "e2e_view_absent": disagg_sol.NO_JOIN,
}


def test_each_campaign_is_an_ordinary_single_track_spec(tmp_path):
    """This layer is additive: nothing below it changes shape."""
    spec = disagg_sol.campaign_spec(
        BASE,
        track="gen",
        sweep=tmp_path / "s.yaml",
        repo=tmp_path / "r",
        point=POINT,
        design=tmp_path / "d",
    )
    assert spec["sol_track"]["track"] == "gen"
    assert spec["optimize"]["max_rounds"] == 3
    assert "disagg_sol" not in spec


def test_the_campaign_can_say_where_its_operating_point_came_from(tmp_path):
    """A path instead of a shrug -- the whole reason this layer exists."""
    spec = disagg_sol.campaign_spec(
        BASE,
        track="gen",
        sweep=tmp_path / "s.yaml",
        repo=tmp_path / "r",
        point=POINT,
        design=tmp_path / "design",
    )
    prov = spec["sol_track"]["point_provenance"]
    assert prov["design_dir"].endswith("design")
    assert prov["selected"]["shape"] == "tep_4_eplb0_mtp0"
    assert "Absent, not flat" in prov["e2e_view_absent"]


def test_no_anchor_is_handed_to_the_gen_half(tmp_path):
    """There is no join in this scope, so there is nothing to anchor against."""
    spec = disagg_sol.campaign_spec(
        BASE,
        track="gen",
        sweep=tmp_path / "s.yaml",
        repo=tmp_path / "r",
        point=POINT,
        design=tmp_path / "d",
    )
    assert "ctx_json" not in spec["sol_track"]


def _sweeps(tmp_path, *, gen_shape="tep", gen_tp=4, gen_conc="1", ctx_tp=4, ctx_batch=2):
    """Real sweep files that contain (or deliberately miss) the selected point."""
    import yaml as _yaml

    g = tmp_path / "gen.yaml"
    g.write_text(
        _yaml.safe_dump(
            {"gen_configs": [[1, 1, gen_tp, 64, 64, gen_shape == "dep", "0.9", 0, 0, gen_conc]]}
        )
    )
    c = tmp_path / "ctx.yaml"
    c.write_text(
        _yaml.safe_dump(
            {"benchmarks": [{"isl": 8192, "osl": 1, "max_batch": [ctx_batch], "tp_size": [ctx_tp]}]}
        )
    )
    return {"ctx": c, "gen": g}


def _plan(tmp_path, repos=None):
    return disagg_sol.launch_plan(
        BASE,
        sweeps=_sweeps(tmp_path, gen_shape="tep", gen_tp=4, gen_conc="1"),
        repos=repos or {"ctx": tmp_path / "trtllm-ctx", "gen": tmp_path / "trtllm-gen"},
        workspace_root=tmp_path,
        label="run1",
        points={"ctx": CTX_POINT, "gen": POINT},
        design=tmp_path / "d",
    )


def test_the_plan_is_complete_before_anything_starts(tmp_path):
    launches = _plan(tmp_path)
    assert [run.track for run in launches] == ["ctx", "gen"]
    assert launches[0].workspace.name == "ws-run1-ctx"
    assert launches[1].workspace.name == "ws-run1-gen"
    assert launches[0].argv[:2] == ["perf-optimize", "--task"]


def test_two_campaigns_may_not_share_one_checkout(tmp_path):
    """A campaign resets the checkout it is given.

    Two sharing one would each revert the other's work mid-flight, and the
    measurement that followed would be of neither's code -- while looking
    entirely normal.
    """
    one = tmp_path / "shared"
    with pytest.raises(disagg_sol.DisaggSolError, match="each revert the other"):
        _plan(tmp_path, repos={"ctx": one, "gen": one})


def test_a_track_with_no_sweep_or_no_checkout_is_refused_by_name(tmp_path):
    with pytest.raises(disagg_sol.DisaggSolError, match=r"neither a sweep of their own"):
        disagg_sol.launch_plan(
            BASE,
            sweeps={"ctx": _sweeps(tmp_path)["ctx"]},
            repos={"ctx": tmp_path / "a", "gen": tmp_path / "b"},
            workspace_root=tmp_path,
            label="x",
            points={"ctx": CTX_POINT, "gen": POINT},
            design=tmp_path,
        )
    with pytest.raises(disagg_sol.DisaggSolError, match=r"trtllm_repo_path.*\['gen'\]"):
        disagg_sol.launch_plan(
            BASE,
            sweeps=_sweeps(tmp_path, gen_tp=4, gen_conc="1"),
            repos={"ctx": tmp_path / "a"},
            workspace_root=tmp_path,
            label="x",
            points={"ctx": CTX_POINT, "gen": POINT},
            design=tmp_path,
        )


# ------------------------------------------------------------- the ctx half


def _ctx_case(root: Path, name: str, req_s: float) -> Path:
    d = Path(root) / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "run_dep4_MTP3.json").write_text(
        json.dumps({"performance": {"request_throughput_req_s": req_s}}), encoding="utf-8"
    )
    return d


def test_the_ctx_half_is_measured_and_selected_not_computed(tmp_path):
    """Ranking prefill needs no rate match, so it is possible without a join.

    That is why this half can be measured even under a no-join scope -- the
    objective lives entirely inside the ctx measurement.
    """
    d = _design(tmp_path)
    _ctx_case(d, "ctx_8192_1_ratio08_2_16416_dep4_MTP3_test1", 9.09)
    _ctx_case(d, "ctx_8192_1_ratio08_4_16416_dep8_MTP3_test1", 14.0)
    points = disagg_sol.ctx_points(d)
    assert {p["ctx_gpus"] for p in points} == {4, 8}
    # 9.09/4 = 2.27 beats 14.0/8 = 1.75 -- more total throughput, less per GPU.
    chosen = disagg_sol.select_ctx_point(points)
    assert chosen["ctx_gpus"] == 4
    assert disagg_sol.CTX_RANK in chosen["ranked_on"]


def test_a_tie_on_efficiency_breaks_toward_the_smaller_worker(tmp_path):
    """Two configurations at one efficiency are not equal.

    The smaller one leaves the rest of the node to the generation side.
    """
    d = _design(tmp_path)
    _ctx_case(d, "ctx_8192_1_ratio08_2_16416_dep4_MTP3_test1", 8.0)
    _ctx_case(d, "ctx_8192_1_ratio08_4_16416_dep8_MTP3_test1", 16.0)
    chosen = disagg_sol.select_ctx_point(disagg_sol.ctx_points(d))
    assert chosen["ctx_gpus"] == 4


def test_an_unparseable_ctx_case_name_is_skipped_not_guessed(tmp_path):
    d = _design(tmp_path)
    _ctx_case(d, "ctx_8192_1_ratio08_2_16416_dep4_MTP3_test1", 9.09)
    _ctx_case(d, "something_else_entirely", 99.0)
    assert [p["ctx_gpus"] for p in disagg_sol.ctx_points(d)] == [4]


def test_a_design_with_no_scored_ctx_case_says_the_half_would_be_computed(tmp_path):
    with pytest.raises(disagg_sol.DisaggSolError, match="fall back to a computed point"):
        disagg_sol.ctx_points(_design(tmp_path))


# ------------------------------------------------- what one frozen point hides


def test_freezing_one_point_records_what_it_cannot_see():
    """opt-006 measured +1.52 % at one point and -1.85 % at another.

    With both frozen it was rejected; with one it is an accept. The gate is
    the cheapest honest one, but the blind spot has to travel with it.
    """
    chosen = disagg_sol.select_point(FRONT, prefer="interactive")
    assert "Unobserved, not unchanged" in chosen["off_point_effects_unobserved"]
    assert "indistinguishable" in chosen["off_point_effects_unobserved"]


def test_repeats_of_one_configuration_are_averaged_not_competed(tmp_path):
    """Twelve directories, one configuration -- this is the real shape.

    A ctx sweep repeats each case and expands over mtp_range, and mtp is not
    part of a ctx operating point: the generation sweep's `ctx_config` block
    has no mtp field. Ranking the repeats as candidates picks the luckiest
    sample, and the measured spread across repeats (4.0 %) is wider than the
    difference between configurations this selection has to resolve.
    """
    d = _design(tmp_path)
    for mtp in (0, 3):
        for test, req_s in ((1, 8.697), (2, 8.701), (3, 8.788)):
            _ctx_case(d, f"ctx_8192_1_ratio08_2_16416_dep4_MTP{mtp}_test{test}", req_s)
    chosen = disagg_sol.select_ctx_point(disagg_sol.ctx_points(d))
    assert chosen["candidates"] == 1  # one configuration...
    assert chosen["measurements"] == 6  # ...measured six times
    assert chosen["repeats"] == 6
    # the mean, not the 8.788 maximum
    assert chosen[disagg_sol.CTX_METRIC] == pytest.approx(8.7286667, rel=1e-6)


def test_a_different_repeat_of_the_incumbent_is_not_a_move(tmp_path):
    """`moved` is the answer this whole staging exists to give.

    Reporting True because a different directory won would say the operating
    point should change when the selection in fact confirmed it.
    """
    d = _design(tmp_path)
    _ctx_case(d, "ctx_8192_1_ratio08_2_16416_dep4_MTP0_test3", 8.788)
    _ctx_case(d, "ctx_8192_1_ratio08_2_16416_dep4_MTP3_test1", 8.484)
    chosen = disagg_sol.select_ctx_point(
        disagg_sol.ctx_points(d),
        incumbent={"case": "ctx_8192_1_ratio08_2_16416_dep4_MTP3_test1"},
    )
    assert chosen["moved"] is False


def test_a_genuinely_different_configuration_is_a_move(tmp_path):
    d = _design(tmp_path)
    _ctx_case(d, "ctx_8192_1_ratio08_2_16416_dep4_MTP3_test1", 12.0)
    _ctx_case(d, "ctx_8192_1_ratio08_4_16416_dep8_MTP3_test1", 8.0)
    chosen = disagg_sol.select_ctx_point(
        disagg_sol.ctx_points(d),
        incumbent={"ctx_gpus": 8, "max_batch": 4, "adp": True},
    )
    assert chosen["ctx_gpus"] == 4
    assert chosen["moved"] is True


def test_the_configuration_with_the_better_mean_wins_a_lucky_repeat(tmp_path):
    """The inversion the old rule produced, at the measured spread."""
    d = _design(tmp_path)
    for test, req_s in ((1, 10.0), (2, 10.0), (3, 10.4)):  # mean 10.13, max 10.4
        _ctx_case(d, f"ctx_8192_1_ratio08_2_16416_dep4_MTP0_test{test}", req_s)
    for test, req_s in ((1, 10.2), (2, 10.3), (3, 10.2)):  # mean 10.23, max 10.3
        _ctx_case(d, f"ctx_8192_1_ratio08_4_16416_dep4_MTP0_test{test}", req_s)
    chosen = disagg_sol.select_ctx_point(disagg_sol.ctx_points(d))
    assert chosen["max_batch"] == 4  # the better mean, not the luckier max
    assert chosen["spread_pct"] < 2.0


# ---------------------------------------------------------------- the run


def _supervisable(tmp_path):
    d = _design(tmp_path)
    _ctx_case(d, "ctx_8192_1_ratio08_2_16416_dep4_MTP0_test1", 8.7)
    _shape_run(d, "bm_tep4", [(1, 4, "False", 0, 0, 214.0, 53.0)])
    spec = {
        "checkpoint_path": "/ckpt",
        "optimize": {"approaches": ["code"]},
        FIELD: {
            "tracks": ["ctx", "gen"],
            "design": {"design_dir": str(d), "prefer": "interactive"},
        },
    }
    return d, spec


def test_a_design_that_was_never_scored_is_refused_not_fallen_back_from(tmp_path):
    """Falling back is exactly the behaviour this layer exists to remove."""
    d = _design(tmp_path)
    spec = {FIELD: {"tracks": ["gen"], "design": {"design_dir": str(d), "prefer": "interactive"}}}
    with pytest.raises(disagg_sol.DisaggSolError, match="no measured space to choose|no scored"):
        disagg_sol.supervise(
            spec,
            sweeps={"gen": tmp_path / "g.yaml"},
            repos={"gen": tmp_path / "r"},
            workspace_root=tmp_path / "ws",
            label="t",
            dry_run=True,
        )


def test_a_dry_run_selects_and_writes_every_spec_without_starting_anything(tmp_path):
    """The same code path minus the processes."""
    d, spec = _supervisable(tmp_path)
    record = disagg_sol.supervise(
        spec,
        sweeps=_sweeps(tmp_path, gen_tp=4, gen_conc="1"),
        repos={"ctx": tmp_path / "rc", "gen": tmp_path / "rg"},
        workspace_root=tmp_path / "ws",
        label="t",
        dry_run=True,
    )
    assert record["started"] is False
    assert record["ctx_point"]["ctx_gpus"] == 4
    assert record["gen_point"]["concurrency"] == 1
    assert [c["track"] for c in record["campaigns"]] == ["ctx", "gen"]
    assert (tmp_path / "ws" / disagg_sol.RUN_RECORD).is_file()


def test_the_record_says_what_was_selected_and_against_what(tmp_path):
    """Answerable afterwards from a file, not from a shrug."""
    d, spec = _supervisable(tmp_path)
    record = disagg_sol.supervise(
        spec,
        sweeps=_sweeps(tmp_path, gen_tp=4, gen_conc="1"),
        repos={"ctx": tmp_path / "rc", "gen": tmp_path / "rg"},
        workspace_root=tmp_path / "ws",
        label="t",
        dry_run=True,
        incumbent={"shape": "tep_4_eplb0_mtp0", "concurrency": 1},
    )
    assert record["gen_point"]["moved"] is False
    assert record["design_dir"] == str(d)
    assert "e2e_view_absent" in record["gen_point"]


def test_a_half_that_is_ready_does_not_wait_on_one_that_is_not(tmp_path):
    """The two halves are established by different artefacts.

    Under a no-join scope neither waits on the other, so requiring both would
    idle a ready half on a dependency the scope does not have. Found by
    launching the flow: a ctx-only spec was refused for a generation sweep it
    had not asked for.
    """
    d = _design(tmp_path)
    _ctx_case(d, "ctx_8192_1_ratio08_2_16416_dep4_MTP0_test1", 8.7)
    assert disagg_sol.established(d, disagg_sol.CTX_TRACK) is True
    assert disagg_sol.established(d, disagg_sol.GEN_TRACK) is False

    spec = {
        "checkpoint_path": "/ckpt",
        "optimize": {"approaches": ["code"]},
        FIELD: {"tracks": ["ctx"], "design": {"design_dir": str(d), "prefer": "interactive"}},
    }
    record = disagg_sol.supervise(
        spec,
        sweeps={"ctx": _sweeps(tmp_path)["ctx"]},
        repos={"ctx": tmp_path / "rc"},
        workspace_root=tmp_path / "ws",
        label="t",
        dry_run=True,
    )
    assert record["ctx_point"]["ctx_gpus"] == 4
    assert "gen_point" not in record


def test_the_refusal_names_which_half_is_missing_what(tmp_path):
    d = _design(tmp_path)
    _ctx_case(d, "ctx_8192_1_ratio08_2_16416_dep4_MTP0_test1", 8.7)
    spec = {
        FIELD: {"tracks": ["ctx", "gen"], "design": {"design_dir": str(d), "prefer": "interactive"}}
    }
    with pytest.raises(disagg_sol.DisaggSolError, match=r"concurrency sweep.*for track 'gen'"):
        disagg_sol.supervise(
            spec,
            sweeps=_sweeps(tmp_path),
            repos={"ctx": tmp_path / "a", "gen": tmp_path / "b"},
            workspace_root=tmp_path / "ws",
            label="t",
            dry_run=True,
        )


def test_a_sweep_that_does_not_contain_the_selected_point_is_refused(tmp_path):
    """Without this the selection is a report, not a decision.

    The chosen point goes into every campaign's `point_provenance` while the
    campaign runs whatever its sweep says -- so a mismatched pair produces a
    record claiming an operating point the run never used. That is worse than
    not selecting at all: an untraceable point is at least honest about being
    untraceable.
    """
    d, spec = _supervisable(tmp_path)
    # the design measured tep_4 @ c=1; hand it a sweep for dep_16 @ c=64
    with pytest.raises(disagg_sol.DisaggSolError, match="does not contain the selected point"):
        disagg_sol.supervise(
            spec,
            sweeps=_sweeps(tmp_path, gen_shape="dep", gen_tp=16, gen_conc="64"),
            repos={"ctx": tmp_path / "rc", "gen": tmp_path / "rg"},
            workspace_root=tmp_path / "ws2",
            label="t",
            dry_run=True,
        )


def test_the_refusal_lists_what_the_sweep_does_contain(tmp_path):
    """So the reader can see whether the sweep or the selection is wrong."""
    d, spec = _supervisable(tmp_path)
    with pytest.raises(disagg_sol.DisaggSolError, match=r"It expands to \[\('dep_16"):
        disagg_sol.supervise(
            spec,
            sweeps=_sweeps(tmp_path, gen_shape="dep", gen_tp=16, gen_conc="64"),
            repos={"ctx": tmp_path / "rc", "gen": tmp_path / "rg"},
            workspace_root=tmp_path / "ws3",
            label="t",
            dry_run=True,
        )


def test_a_ctx_sweep_at_the_wrong_worker_shape_is_refused(tmp_path):
    """The ctx point is (tp_size, max_batch); a sweep at another is a mismatch."""
    d, spec = _supervisable(tmp_path)
    spec = {**spec, FIELD: {**spec[FIELD], "tracks": ["ctx"]}}
    with pytest.raises(disagg_sol.DisaggSolError, match="does not contain the selected point"):
        disagg_sol.supervise(
            spec,
            sweeps={"ctx": _sweeps(tmp_path, ctx_tp=8, ctx_batch=16)["ctx"]},
            repos={"ctx": tmp_path / "rc"},
            workspace_root=tmp_path / "ws4",
            label="t",
            dry_run=True,
        )


def test_an_unestablished_design_is_run_rather_than_refused_when_one_can_be(tmp_path):
    """An unestablished design is established rather than refused.

    Treating it purely as an input degenerated into nobody running it once
    already -- that is how the two campaigns this layer replaces came to
    inherit an unmeasured row. When a designer is available the run
    establishes what it needs and re-checks: not a fallback, a first step.
    """
    d = _design(tmp_path)
    _ctx_case(d, "ctx_8192_1_ratio08_2_16416_dep4_MTP0_test1", 8.7)
    spec = {
        "checkpoint_path": "/ckpt",
        "optimize": {"approaches": ["code"]},
        FIELD: {"tracks": ["gen"], "design": {"design_dir": str(d), "prefer": "interactive"}},
    }
    seen: list[str] = []

    def designer(instruction: str) -> None:
        seen.append(instruction)
        _shape_run(d, "bm_tep4", [(1, 4, "False", 0, 0, 214.0, 53.0)])  # now established

    record = disagg_sol.supervise(
        spec,
        sweeps={"gen": _sweeps(tmp_path)["gen"]},
        repos={"gen": tmp_path / "rg"},
        workspace_root=tmp_path / "wsd",
        label="t",
        dry_run=True,
        designer=designer,
    )
    assert len(seen) == 1
    assert "create-sweep" in seen[0]
    assert record["gen_point"]["shape"] == "tep_4_eplb0_mtp0"


def test_a_designer_that_did_not_establish_the_design_still_refuses(tmp_path):
    """Running it is not the same as it having worked."""
    d = _design(tmp_path)
    spec = {FIELD: {"tracks": ["gen"], "design": {"design_dir": str(d), "prefer": "interactive"}}}
    with pytest.raises(disagg_sol.DisaggSolError, match="concurrency sweep"):
        disagg_sol.supervise(
            spec,
            sweeps={"gen": _sweeps(tmp_path)["gen"]},
            repos={"gen": tmp_path / "rg"},
            workspace_root=tmp_path / "wse",
            label="t",
            dry_run=True,
            designer=lambda instruction: None,
        )


def test_only_the_unready_halves_are_named_to_the_designer(tmp_path):
    """No point re-establishing a half that is already measured."""
    d = _design(tmp_path)
    _ctx_case(d, "ctx_8192_1_ratio08_2_16416_dep4_MTP0_test1", 8.7)
    spec = {
        "checkpoint_path": "/ckpt",
        FIELD: {
            "tracks": ["ctx", "gen"],
            "design": {"design_dir": str(d), "prefer": "interactive"},
        },
    }
    seen: list[str] = []

    def designer(instruction: str) -> None:
        seen.append(instruction)
        _shape_run(d, "bm_tep4", [(1, 4, "False", 0, 0, 214.0, 53.0)])

    disagg_sol.supervise(
        spec,
        sweeps=_sweeps(tmp_path),
        repos={"ctx": tmp_path / "a", "gen": tmp_path / "b"},
        workspace_root=tmp_path / "wsf",
        label="t",
        dry_run=True,
        designer=designer,
    )
    assert "['gen']" in seen[0]
    assert "'ctx'" not in seen[0]


# ------------------------------------------------------- deriving the sweep


DESIGN_SWEEP = {
    "model_id": "deepseek-ai/DeepSeek-V4-Pro",
    "isl": 8192,
    "osl": 1024,
    "gen_configs": [
        [1, 1, 4, 64, 64, False, "0.9", 0, 0, "1,2,4,8,16,32,64"],
        [1, 1, 8, 64, 64, False, "0.9", 0, 0, "1,2,4,8,16,32,64"],
        [1, 1, 16, 32, 32, True, "0.9", 0, 0, "16,64,512,1024,2048"],
    ],
    "benchmarks": [
        {"isl": 8192, "osl": 1, "max_batch": [1, 2, 4], "tp_size": [4, 8], "ratio": [0.8]}
    ],
    "gpu_overrides": {"GB300": {"benchmarks": [{"max_batch": [2]}]}},
}


def _design_sweep(tmp_path):
    import yaml as _yaml

    p = tmp_path / "sweep_design" / "8k1k_sol_mtp0.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_yaml.safe_dump(DESIGN_SWEEP))
    return p


def test_a_campaign_cannot_run_the_whole_measured_space(tmp_path):
    """Two shapes at one concurrency both land in `concurrency_<c>`.

    That is what `_place` refuses, so the design sweep -- which is the whole
    space -- is not something a campaign can freeze on. Hence the cut.
    """
    import yaml as _yaml

    from agent_flow.workflows.perf_optimize import bench_cli

    cases = bench_cli.gen_cases(_yaml.safe_load(_design_sweep(tmp_path).read_text()))
    at_64 = [c for c in cases if (c["config"] or {}).get("concurrency") == 64]
    assert len({c["name"] for c in at_64}) > 1  # tep_4 and tep_8 collide


def test_the_derived_sweep_keeps_one_row_at_the_selected_point(tmp_path):
    import yaml as _yaml

    out = disagg_sol.derive_sweep_at_point(
        _design_sweep(tmp_path),
        "gen",
        {"shape": "tep_8_eplb0_mtp0", "concurrency": 64},
        into=tmp_path / "ws",
    )
    got = _yaml.safe_load(out.read_text())
    assert len(got["gen_configs"]) == 1
    assert got["gen_configs"][0][2] == 8  # tp_size
    assert got["gen_configs"][0][9] == "64"  # this point only, not the ladder


def test_everything_but_the_row_filter_comes_from_the_design_sweep(tmp_path):
    """So the campaign measures what the selection was made on.

    The two campaigns this layer replaces differed from their own recorded
    point in three fields at once, and nothing noticed.
    """
    import yaml as _yaml

    out = disagg_sol.derive_sweep_at_point(
        _design_sweep(tmp_path),
        "gen",
        {"shape": "tep_4_eplb0_mtp0", "concurrency": 32},
        into=tmp_path / "ws",
    )
    got = _yaml.safe_load(out.read_text())
    assert got["model_id"] == DESIGN_SWEEP["model_id"]
    assert (got["isl"], got["osl"]) == (8192, 1024)
    assert got["_derived_from"]["design_sweep"].endswith("8k1k_sol_mtp0.yaml")


def test_the_ctx_cut_narrows_both_axes_and_drops_the_override_table(tmp_path):
    import yaml as _yaml

    out = disagg_sol.derive_sweep_at_point(
        _design_sweep(tmp_path), "ctx", {"ctx_gpus": 8, "max_batch": 4}, into=tmp_path / "ws"
    )
    got = _yaml.safe_load(out.read_text())
    assert got["benchmarks"] == [
        {"isl": 8192, "osl": 1, "max_batch": [4], "tp_size": [8], "ratio": [0.8]}
    ]
    assert "gpu_overrides" not in got


def test_a_derived_sweep_is_written_only_inside_the_campaign_workspace(tmp_path):
    """The blunt rule an earlier module in this package lacked.

    It took an output path as a free parameter and would have overwritten a
    curated config other people maintain -- unrecoverable in a way a wrong
    measurement is not.
    """
    ws = tmp_path / "ws"
    out = disagg_sol.derive_sweep_at_point(
        _design_sweep(tmp_path), "gen", {"shape": "tep_8_eplb0_mtp0", "concurrency": 1}, into=ws
    )
    assert out.parent == ws.resolve()
    assert out.name == disagg_sol.DERIVED_SWEEP_NAME
    # the design's own sweep is untouched
    assert "1,2,4,8,16,32,64" in _design_sweep(tmp_path).read_text()


def test_a_point_the_design_sweep_cannot_produce_is_refused(tmp_path):
    with pytest.raises(disagg_sol.DisaggSolError, match="0 rows matching"):
        disagg_sol.derive_sweep_at_point(
            _design_sweep(tmp_path),
            "gen",
            {"shape": "dep_32_eplb0_mtp0", "concurrency": 4096},
            into=tmp_path / "ws",
        )


# ------------------------------------------------- where the design landed


def _established_at(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / disagg_sol.DESIGN_STATE).write_text(json.dumps({"phase": "PHASE3_DONE"}))
    _ctx_case(d, "ctx_8192_1_ratio08_2_16416_dep4_MTP0_test1", 8.7)
    _shape_run(d, "bm_tep4", [(1, 4, "False", 0, 0, 214.0, 53.0)])
    return d


def test_the_requested_design_is_used_when_it_is_the_established_one(tmp_path):
    d = _established_at(tmp_path / "model", "sweep_design2")
    got, note = disagg_sol.resolve_design_dir(d, ["ctx", "gen"])
    assert got == d
    assert note is None


def test_a_design_established_next_door_is_followed_and_recorded(tmp_path):
    """Resuming an existing design instead of re-probing is the right call.

    It saved roughly twenty node-hours on the run this was written after.
    Refusing it would forbid a correct decision; following it silently would
    leave every later artefact citing a directory the spec never named.
    """
    model = tmp_path / "model"
    actual = _established_at(model, "sweep_design")
    requested = model / "sweep_design2"
    requested.mkdir()

    got, note = disagg_sol.resolve_design_dir(requested, ["ctx", "gen"])
    assert got == actual
    assert "sweep_design2" in note and "sweep_design" in note
    assert "Point the spec at" in note


def test_two_established_designs_side_by_side_are_refused(tmp_path):
    """Which measurement to freeze on is not this layer's question."""
    model = tmp_path / "model"
    _established_at(model, "sweep_design")
    _established_at(model, "sweep_design_b")
    with pytest.raises(disagg_sol.DisaggSolError, match="is not something this layer may pick"):
        disagg_sol.resolve_design_dir(model / "sweep_design2", ["ctx", "gen"])


def test_an_unestablished_neighbour_is_not_followed(tmp_path):
    """The search admits only directories that measured what is wanted.

    A supervisor that hunts the filesystem for something that looks like a
    design will eventually find one that is not.
    """
    model = tmp_path / "model"
    empty = model / "sweep_design"
    empty.mkdir(parents=True)
    (empty / disagg_sol.DESIGN_STATE).write_text(json.dumps({"phase": "PHASE2"}))
    requested = model / "sweep_design2"
    got, note = disagg_sol.resolve_design_dir(requested, ["gen"])
    assert got == requested and note is None


def test_the_redirection_travels_in_the_run_record(tmp_path):
    """A reader must not have to already know."""
    model = tmp_path / "model"
    _established_at(model, "sweep_design")
    requested = model / "sweep_design2"
    requested.mkdir()
    spec = {
        "checkpoint_path": "/ckpt",
        FIELD: {
            "tracks": ["gen"],
            "design": {"design_dir": str(requested), "prefer": "interactive"},
        },
    }
    record = disagg_sol.supervise(
        spec,
        sweeps={"gen": _sweeps(tmp_path)["gen"]},
        repos={"gen": tmp_path / "rg"},
        workspace_root=tmp_path / "wsr",
        label="t",
        dry_run=True,
        designer=lambda instruction: None,  # the design is already next door
    )
    assert record["design_dir"].endswith("sweep_design")
    assert "design_dir_redirected" in record


def test_a_design_that_already_exists_is_not_paid_for_again(tmp_path):
    """The resolution happens before the designer is considered.

    A design costs about an order of magnitude more than the campaigns it
    enables, so running one that already exists next door is the expensive
    half of this mistake -- and avoiding exactly that is why the agent
    resumed a neighbour to begin with.
    """
    model = tmp_path / "model"
    _established_at(model, "sweep_design")
    requested = model / "sweep_design2"
    requested.mkdir()
    spec = {
        "checkpoint_path": "/ckpt",
        FIELD: {
            "tracks": ["gen"],
            "design": {"design_dir": str(requested), "prefer": "interactive"},
        },
    }
    ran: list[str] = []
    record = disagg_sol.supervise(
        spec,
        sweeps={"gen": _sweeps(tmp_path)["gen"]},
        repos={"gen": tmp_path / "rg"},
        workspace_root=tmp_path / "wsn",
        label="t",
        dry_run=True,
        designer=lambda instruction: ran.append(instruction),
    )
    assert ran == []  # the designer was never started
    assert record["design_dir"].endswith("sweep_design")
    assert "design_dir_redirected" in record


def test_a_redirected_design_carries_its_sweep_with_it(tmp_path):
    """`design_dir` and `design_sweep` are two fields naming one thing.

    A sweep path still rooted at the requested directory names a file that
    was never written, and the failure arrives after the points have been
    read and the point chosen -- the most expensive moment to find a path
    problem.
    """
    requested = tmp_path / "model" / "sweep_design2"
    actual = tmp_path / "model" / "sweep_design"
    actual.mkdir(parents=True)
    (actual / "8k1k_sol_mtp0.yaml").write_text("gen_configs: []\n")
    got = disagg_sol.rebase_design_sweep(requested / "8k1k_sol_mtp0.yaml", requested, actual)
    assert got == actual / "8k1k_sol_mtp0.yaml"


def test_a_sweep_that_exists_where_stated_is_left_alone(tmp_path):
    here = tmp_path / "elsewhere.yaml"
    here.write_text("gen_configs: []\n")
    assert disagg_sol.rebase_design_sweep(here, tmp_path / "a", tmp_path / "b") == here


def test_a_sweep_pointing_outside_the_design_is_a_deliberate_choice(tmp_path):
    """Not under the requested directory, so not this redirection's business."""
    outside = tmp_path / "curated" / "sweep.yaml"
    got = disagg_sol.rebase_design_sweep(outside, tmp_path / "a", tmp_path / "b")
    assert got == outside


def test_each_half_is_given_its_own_point_not_the_other_s(tmp_path):
    """A ctx point is (tp_size, max_batch); a gen point is (shape, concurrency).

    Handing one to both asks the ctx sweep for a row described in the
    generation half's vocabulary -- observed as "0 benchmark entries matching
    the selected ctx point tp_size None @ max_batch None".
    """
    launches = disagg_sol.launch_plan(
        BASE,
        sweeps=_sweeps(tmp_path, gen_tp=4, gen_conc="1"),
        repos={"ctx": tmp_path / "rc", "gen": tmp_path / "rg"},
        workspace_root=tmp_path / "wsp",
        label="t",
        points={"ctx": CTX_POINT, "gen": POINT},
        design=tmp_path / "d",
    )
    by_track = {run.track: run.spec["sol_track"]["point_provenance"] for run in launches}
    assert by_track["ctx"]["selected"]["ctx_gpus"] == 4
    assert by_track["ctx"]["selected"]["max_batch"] == 2
    assert by_track["gen"]["selected"]["shape"] == "tep_4_eplb0_mtp0"
    assert by_track["gen"]["selected"]["concurrency"] == 1


def test_the_two_halves_are_cut_from_different_design_sweeps(tmp_path):
    """They are measured by different sweeps, so `design_sweep` is per track."""
    d, spec = _supervisable(tmp_path)
    spec = {**spec, FIELD: {**spec[FIELD], "tracks": ["ctx", "gen"]}}
    with pytest.raises(disagg_sol.DisaggSolError, match=r"neither a sweep of their own"):
        disagg_sol.supervise(
            spec,
            sweeps={"gen": _sweeps(tmp_path)["gen"]},  # ctx has neither
            repos={"ctx": tmp_path / "rc", "gen": tmp_path / "rg"},
            workspace_root=tmp_path / "wsq",
            label="t",
            dry_run=True,
        )


def test_a_derived_sweep_is_the_one_the_campaign_is_given(tmp_path):
    """The resolution and the spec must name the same file.

    Reading the caller's `sweeps` again after deriving one silently hands the
    campaign a path that was never derived -- and for a track with no sweep of
    its own, no path at all.
    """
    d = _design(tmp_path)
    _shape_run(d, "bm_tep4", [(1, 4, "False", 0, 0, 214.0, 53.0)])
    design_sweep = tmp_path / "8k1k_sol_mtp0.yaml"
    design_sweep.write_text("gen_configs:\n- [1, 1, 4, 64, 64, false, '0.9', 0, 0, '1,32']\n")
    launches = disagg_sol.launch_plan(
        {**BASE, FIELD: {**BASE[FIELD], "tracks": ["gen"]}},
        sweeps={},
        repos={"gen": tmp_path / "rg"},
        workspace_root=tmp_path / "wsd",
        label="t",
        points={"gen": POINT},
        design=d,
        design_sweeps={"gen": design_sweep},
    )
    named = Path(launches[0].spec["sol_track"]["sweep"])
    assert named.name == disagg_sol.DERIVED_SWEEP_NAME
    assert named.is_file()
    assert named.parent == launches[0].workspace


def test_the_derived_sweep_carries_the_campaign_s_own_build_source(tmp_path):
    """The design measures the image; a campaign measures its own edits.

    `sweep_design` refuses a build source in a design sweep -- choosing an
    operating point on code no campaign starts from picks it for a different
    program. A campaign is the opposite case: without a rung it would run the
    image and report every change as no-gain. The rung is added at the
    boundary where the purpose changes.
    """
    import yaml as _yaml

    out = disagg_sol.derive_sweep_at_point(
        _design_sweep(tmp_path),
        "gen",
        {"shape": "tep_8_eplb0_mtp0", "concurrency": 64},
        into=tmp_path / "ws",
        repo=tmp_path / "trtllm-gen",
    )
    got = _yaml.safe_load(out.read_text())
    assert got["trtllm_install"]["trtllm_repo"].endswith("trtllm-gen")
    # ...and the design's own sweep still has none
    assert "trtllm_install" not in _yaml.safe_load(_design_sweep(tmp_path).read_text())


def test_a_derivation_without_a_checkout_adds_no_rung(tmp_path):
    import yaml as _yaml

    out = disagg_sol.derive_sweep_at_point(
        _design_sweep(tmp_path),
        "gen",
        {"shape": "tep_8_eplb0_mtp0", "concurrency": 64},
        into=tmp_path / "ws2",
    )
    assert "trtllm_install" not in _yaml.safe_load(out.read_text())
