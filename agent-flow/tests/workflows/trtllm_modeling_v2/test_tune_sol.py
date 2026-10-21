# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SOL ceilings: what makes ``sol.yaml`` usable, and what it says about a measurement."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from agent_flow.workflows.trtllm_modeling_v2.tune import sol as S
from agent_flow.workflows.trtllm_modeling_v2.tune.records import Measurement
from agent_flow.workflows.trtllm_modeling_v2.tune.schema import validate_perf_block

from .test_tune_stage import FakeRunner, _contract, _effect, _repo


def _measurement(tmp_path: Path, label: str, user: float, gpu: float) -> Measurement:
    path = tmp_path / label
    path.mkdir()
    (path / "meta.json").write_text(json.dumps({"label": label}))
    rows = [
        {"concurrency": 1, "tok_s_user": user, "output_tok_s_gpu": user},
        {"concurrency": 64, "tok_s_user": gpu / 64, "output_tok_s_gpu": gpu},
    ]
    (path / "summary.json").write_text(json.dumps(rows))
    return Measurement.load(path)


def _sol_file(tmp_path: Path, ceilings: dict, method="calculator", peaks=True) -> Path:
    doc = {"method": method, "ceilings": ceilings}
    if peaks:
        (tmp_path / "peaks.json").write_text("{}")
        doc["peaks"] = "peaks.json"
    path = tmp_path / "sol.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def _c(value, bound="memory"):
    return {"value": value, "bound": bound, "derivation": "31 GB / 8 TB/s"}


_ALL = {"con1.tok_s_user": _c(400), "con64.output_tok_s_gpu": _c(9000, "compute"),
        "con64.tok_s_user": _c(200)}  # fmt: skip


def test_a_complete_file_bounds_the_baseline(tmp_path):
    base = _measurement(tmp_path, "t-baseline", 200, 4500)
    sol, errors = S.load_sol(_sol_file(tmp_path, _ALL), base, tmp_path)
    assert errors == [] and sol.calculated
    assert sol.pct_of_sol(base) == {
        "con1.tok_s_user": 50.0,
        "con64.output_tok_s_gpu": 50.0,
        "con64.tok_s_user": 35.2,
    }
    assert not sol.reached(base, 0.5) and sol.reached(base, 0.35)


def test_every_problem_is_named(tmp_path):
    base = _measurement(tmp_path, "t-baseline", 200, 4500)
    ceilings = {
        "con1.tok_s_user": _c(100),  # below what was measured
        "con64.output_tok_s_gpu": {"value": 9000, "bound": "vibes", "derivation": ""},
        "con9.made_up": _c(1),
    }
    _, errors = S.load_sol(_sol_file(tmp_path, ceilings, peaks=False), base, tmp_path)
    text = "\n".join(errors)
    assert "peaks: the calculator's peaks file" in text
    assert "ceilings.con64.tok_s_user: missing" in text
    assert "measured 200.0, above this ceiling of 100.0" in text
    assert "bound must be one of" in text and "derivation must show" in text
    assert "con9.made_up: not a headline metric" in text


def test_hand_derived_ceilings_need_no_peaks_file_and_say_so(tmp_path):
    base = _measurement(tmp_path, "t-baseline", 200, 4500)
    sol, errors = S.load_sol(_sol_file(tmp_path, _ALL, "reduced", peaks=False), base, tmp_path)
    assert errors == [] and not sol.calculated
    assert "reduced: derived by hand" in S.render_table(sol, [base])


def test_the_table_shows_each_measurement_against_each_ceiling(tmp_path):
    base = _measurement(tmp_path, "t-baseline", 200, 4500)
    final = _measurement(tmp_path, "t-final", 300, 6300)
    sol, _ = S.load_sol(_sol_file(tmp_path, _ALL), base, tmp_path)
    table = S.render_table(sol, [base, final])
    assert "| con1.tok_s_user | 400.0 | memory | 50.0 | 75.0 |" in table
    assert S.from_state(S.as_state(sol)) == sol


def test_a_missing_or_broken_file(tmp_path):
    base = _measurement(tmp_path, "t-baseline", 200, 4500)
    assert "was not written" in S.load_sol(tmp_path / "sol.yaml", base, tmp_path)[1][0]
    (tmp_path / "sol.yaml").write_text("[1, 2]")
    assert "a mapping" in S.load_sol(tmp_path / "sol.yaml", base, tmp_path)[1][0]


def test_the_perf_block_takes_sol_and_a_sol_goal():
    def errors(**perf):
        return validate_perf_block({"perf": {"workload": {"concurrency": [1]}, **perf}})

    assert errors(sol={"enabled": True, "gpu": "gb200"}, goal={"pct_of_sol": 0.7}) == []
    assert errors(goal={"ratio_to_reference": 1.1, "pct_of_sol": 0.5}) == []
    assert errors(goal={"pct_of_sol": 1.5}) == [
        "perf.goal.pct_of_sol: a fraction in (0, 1], e.g. 0.7"
    ]
    assert "needs perf.sol enabled" in errors(sol={"enabled": False}, goal={"pct_of_sol": 0.5})[0]
    assert errors(sol={"part": "gb200"}) == ["perf.sol.part: unknown key"]
    assert "ratio_to_reference and/or pct_of_sol" in errors(goal={"speed": 2})[0]


def test_reduced_ceilings_stop_a_campaign_at_their_practical_peaks(tmp_path):
    """Hand-derived ceilings judge a SOL goal too: scaled by measured / published peak when recorded."""
    from types import SimpleNamespace

    from agent_flow.workflows.trtllm_modeling_v2.tune.stage import TuneSpec, TuneStage

    task = {
        "trtllm_repo_path": str(tmp_path),
        "checkpoint_path": "/ckpt",
        **_contract(tmp_path),
        "perf": {
            "identity_configs": [str(tmp_path / "id.yaml")],
            "workload": {"concurrency": [1, 64]},
            "spools": ["s1"],
            "env_script": "/env.sh",
            "reference": "none",
            "goal": {"pct_of_sol": 0.5},
        },
    }
    spec = TuneSpec.from_task(task, tmp_path / "ws")
    state: dict = {}
    stage = TuneStage(spec, state, dispatch=lambda *a: "", checkpoint=lambda: None,
                      runner_factory=lambda spools, prefix: None, log=lambda m: None)  # fmt: skip
    base = _measurement(stage.data, "t-baseline", 200, 4500)
    camp = SimpleNamespace(state=SimpleNamespace(kept=[], candidates={}))
    assert not stage._goal_met(camp)  # no ceilings yet
    sol, _ = S.load_sol(_sol_file(tmp_path, _ALL, "reduced", peaks=False), base, tmp_path)
    state["sol"] = S.as_state(sol)
    assert not stage._goal_met(camp)  # con64.tok_s_user is at 35% of its published ceiling
    spec.goal_pct_of_sol = 0.35
    assert stage._goal_met(camp)
    assert stage._goal_ceilings()[1] == "reduced ceilings at published peaks"
    # HBM measured at 3/4 of its published peak, dense FLOP/s at 4/5: 200 / (200 x 0.75) = 47%.
    spec.goal_pct_of_sol = 0.45
    assert not stage._goal_met(camp)
    state["sol"]["practical"] = {"memory": 0.75, "compute": 0.8}
    assert stage._goal_met(camp)
    assert stage._goal_ceilings()[1] == (
        "reduced ceilings scaled by measured / published peak (compute x0.800, memory x0.750)"
    )
    sol, _ = S.load_sol(_sol_file(tmp_path, _ALL), base, tmp_path)
    state["sol"] = {**S.as_state(sol), "practical": {"memory": 0.5}}
    assert not stage._goal_met(camp)  # calculator ceilings are never scaled
    spec.goal_pct_of_sol = 0.35
    assert stage._goal_met(camp) and stage._goal_ceilings()[1] == "calculator ceilings"


def _reduced_campaign(tmp_path, peaks: dict):
    """A campaign whose goal is 50% of a reduced SOL; the baseline sits at 39-50% of the published one."""
    from agent_flow.workflows.trtllm_modeling_v2.tune.stage import TuneSpec, TuneStage

    identity = tmp_path / "identity.yaml"
    identity.write_text(yaml.safe_dump({"modeling_v2": "require"}))
    task = {
        "trtllm_repo_path": str(_repo(tmp_path)),
        "checkpoint_path": "/ckpt",
        **_contract(tmp_path),
        "perf": {
            "identity_configs": [str(identity)],
            "workload": {"isl": 8192, "osl": 1024, "concurrency": [1, 128]},
            "spools": ["s1"],
            "env_script": "/env.sh",
            "reference": "none",
            "noise_repeats": 0,
            "goal": {"pct_of_sol": 0.5},
            "campaign": {"max_roadmaps": 1, "kernel_ledger": "off"},
        },
    }  # fmt: skip
    spec = TuneSpec.from_task(task, tmp_path / "ws", target="t/")
    turns, logs, asks = [], [], {}

    def dispatch(role, what, prompt):
        turns.append(what)
        if what == "sol":
            asks["sol"] = (spec.workdir / "sol-request.md").read_text()
            # The fake baseline: con1.tok_s_user 100, con128.output_tok_s_gpu 2000, con128.tok_s_user 15.6.
            ceilings = {"con1.tok_s_user": _c(250), "con128.output_tok_s_gpu": _c(4000, "compute"),
                        "con128.tok_s_user": _c(40)}  # fmt: skip
            (spec.workdir / "sol.yaml").write_text(
                yaml.safe_dump({"method": "reduced", "ceilings": ceilings, **peaks})
            )
        elif what == "roadmap":
            (spec.workdir / "roadmap.yaml").write_text(yaml.safe_dump([{"slug": "no-op"}]))
        else:
            (spec.workdir / "candidates" / what / "decision.yaml").write_text(
                yaml.safe_dump({"decision": "DROPPED", "worktree": ""})
            )
        return ""

    state: dict = {}
    stage = TuneStage(spec, state, dispatch=dispatch, checkpoint=lambda: None,
                      runner_factory=lambda spools, prefix: FakeRunner(spools, prefix, _effect),
                      poll_s=0, log=logs.append)  # fmt: skip
    assert stage.run() == "tuned"
    return spec, state, turns, logs, asks


def test_a_reduced_sol_goal_is_reached_and_the_journal_names_its_basis(tmp_path):
    """The live failure: a reduced SOL never met the goal. Measured peaks make the ceilings practical."""
    peaks = {"published_peaks": {"memory": 8.0, "compute": 2500.0, "comm": 900.0},
             "measured_peaks": {"memory": 6.0, "compute": 2000.0}}  # fmt: skip
    spec, state, turns, logs, asks = _reduced_campaign(tmp_path, peaks)
    assert "published_peaks" in asks["sol"] and "measured_peaks" in asks["sol"]
    assert state["sol"]["practical"] == {"memory": 0.75, "compute": 0.8}
    assert state["goal_reached"] and turns == [
        "sol"
    ]  # the baseline meets it: no roadmap, no candidate
    basis = "reduced ceilings scaled by measured / published peak (compute x0.800, memory x0.750)"
    assert any(
        m.startswith(f"[tune] goal reached: 50% of SOL on {basis}, by t-baseline") for m in logs
    )
    records = [
        json.loads(line) for line in (spec.workdir / "perf_ledger.jsonl").read_text().splitlines()
    ]
    assert [r["basis"] for r in records if r["kind"] == "goal_reached"] == [basis]
    assert f"Goal: 50% of SOL on {basis}, reached" in (spec.workdir / "summary.md").read_text()


def test_a_reduced_sol_without_measured_peaks_is_judged_at_the_published_ones(tmp_path):
    spec, state, turns, logs, _ = _reduced_campaign(tmp_path, {})
    assert "practical" not in state["sol"] and not state["goal_reached"]
    assert "no-op" in turns  # 39-50% of the published ceilings: the campaign goes on
    assert (
        "Goal: 50% of SOL on reduced ceilings at published peaks, not reached"
        in (spec.workdir / "summary.md").read_text()
    )
