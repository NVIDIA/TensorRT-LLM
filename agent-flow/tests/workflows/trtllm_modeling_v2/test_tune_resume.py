# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tune campaign regression tests."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.tune.campaign import Campaign
from agent_flow.workflows.trtllm_modeling_v2.tune.measure import (
    RC_NODE_LOST,
    MeasurementUnavailable,
)
from agent_flow.workflows.trtllm_modeling_v2.tune.stage import TuneStage
from agent_flow.workflows.trtllm_modeling_v2.tune.worktree import measured_tree

from .test_tune_stage import (
    FakeRunner,
    _budget_effect,
    _campaign,
    _effect,
    _git,
    _microbench,
    _repo,
)
from .test_tune_stage import _budget_spec as _spec


def test_commit_snapshots_never_change_when_best_or_another_snapshot_moves(tmp_path):
    repo = _repo(tmp_path)
    first = _git(repo, "rev-parse", "HEAD")
    old = measured_tree(repo, tmp_path / "trees", first)
    (repo / "modeling.py").write_text("FAST = True\n")
    _git(repo, "commit", "-qam", "feature")
    second = _git(repo, "rev-parse", "HEAD")
    new = measured_tree(repo, tmp_path / "trees", second)
    assert old != new and old != repo and (new != repo)
    assert _git(old, "rev-parse", "HEAD") == first
    assert measured_tree(repo, tmp_path / "trees", first) == old
    assert (old / "modeling.py").read_text() == "FAST = False\n"


def test_source_sync_preserves_keeps_and_does_not_change_delivery_or_snapshots(tmp_path):
    camp = _campaign(tmp_path, FakeRunner(["g1"], "gate", _effect), [])
    source = camp.state.best_commit
    while not camp.state.kept:
        camp.step()
    kept = camp.state.best_commit
    snapshot = camp._tree_at(kept)
    camp.state.calibration = 0.3
    (camp.repo / "feature.py").write_text("MTP = True\n")
    _git(camp.repo, "add", "feature.py")
    _git(camp.repo, "commit", "-qm", "feature")
    delivered = _git(camp.repo, "rev-parse", "HEAD")
    synced = camp.sync_source(source, delivered)
    assert synced != kept
    assert Path(camp.state.best_tree, "feature.py").exists()
    assert Path(camp.state.best_tree, "modeling.py").read_text() == "FAST = True\n"
    assert (camp.repo / "modeling.py").read_text() == "FAST = False\n"
    assert _git(snapshot, "rev-parse", "HEAD") == kept
    assert camp.state.checkpoint["good"] == delivered
    assert camp._checkpoint_due()
    assert camp.state.calibration == 1.0 and (not camp.state.controls)
    assert not camp.state.candidates["fast-path"].projected
    assert "share" not in camp.state.candidates["fast-path"].microbench


def test_source_conflict_or_running_job_leaves_best_untouched(tmp_path):
    camp = _campaign(tmp_path, FakeRunner(["g1"], "gate", _effect), [])
    source = camp.state.best_commit
    while not camp.state.kept:
        camp.step()
    kept = camp.state.best_commit
    (camp.repo / "modeling.py").write_text("FEATURE = True\n")
    _git(camp.repo, "commit", "-qam", "conflict")
    delivered = _git(camp.repo, "rev-parse", "HEAD")
    camp.gates.inflight["held"] = ("g1", "job")
    with pytest.raises(MeasurementUnavailable, match="waits for measurement"):
        camp.sync_source(source, delivered)
    camp.gates.inflight.clear()
    with pytest.raises(MeasurementUnavailable, match="conflicts"):
        camp.sync_source(source, delivered)
    assert _git(Path(camp.state.best_tree), "rev-parse", "HEAD") == kept


class LostRunner(FakeRunner):
    def poll(self):
        done = [(label, spool, RC_NODE_LOST) for spool, label in self.inflight.values()]
        self.inflight.clear()
        return done


def test_expired_baseline_has_no_verdict_and_resumes_on_a_replacement_session(tmp_path):
    spec = _spec(tmp_path, enabled=False)
    state, logs = ({}, [])
    stage = TuneStage(
        spec,
        state,
        dispatch=lambda *a: "",
        checkpoint=lambda: None,
        runner_factory=lambda s, p: LostRunner(s, p, _effect),
        poll_s=0,
        log=logs.append,
    )
    assert stage.run() is None
    assert state["phase"] == "ground" and "failure" not in state
    assert "rc=218" in state["deferred"]
    stage = TuneStage(
        spec,
        state,
        dispatch=lambda *a: "",
        checkpoint=lambda: None,
        runner_factory=lambda s, p: FakeRunner(s, p, _effect),
        poll_s=0,
        log=logs.append,
    )
    assert stage.run() == "tuned"


def test_tuner_future_logs_actual_exception_and_does_not_count_a_miss(tmp_path):
    spec = _spec(tmp_path, max_roadmaps=1)
    state, logs = ({}, [])

    def dispatch(role, what, prompt):
        if what == "roadmap":
            (spec.workdir / "roadmap.yaml").write_text(yaml.safe_dump([{"slug": "x"}]))
            return ""
        raise RuntimeError("backend session expired")

    stage = TuneStage(
        spec,
        state,
        dispatch=dispatch,
        checkpoint=lambda: None,
        runner_factory=lambda s, p: FakeRunner(s, p, _effect),
        poll_s=0,
        log=logs.append,
    )
    assert stage.run() is None
    assert any(("tuner raised RuntimeError: backend session expired" in line for line in logs))
    assert not state.get("misses") and state["phase"] == "campaign"
    ledger = [
        json.loads(line) for line in (spec.workdir / "perf_ledger.jsonl").read_text().splitlines()
    ]
    assert not any((row.get("verdict") == "LOSS" for row in ledger))


def test_checkpoint_expiry_preserves_keeps_and_is_resumable(tmp_path):
    camp = _campaign(tmp_path, FakeRunner(["g1"], "gate", _effect), [])
    while not camp.state.kept:
        camp.step()
    camp.perf = LostRunner(["s1"], "camp", _effect)
    with pytest.raises(MeasurementUnavailable, match="no verdict"):
        for _ in range(10):
            camp.step()
    assert camp.state.kept == ["fast-path"]
    assert camp.state.candidates["fast-path"].status == "kept"
    assert camp.state.checkpoint["label"]
    camp.perf = FakeRunner(["s1"], "camp", _effect)
    for _ in range(15):
        if not camp.step():
            break
    assert camp.state.kept == ["fast-path"]
    assert camp.state.checkpoint == {"good": camp.state.best_commit}


def test_stage_syncs_a_later_feature_source_and_remeasures_its_baseline(tmp_path):
    spec = _spec(tmp_path, max_roadmaps=1)
    initial = _git(spec.repo, "rev-parse", "HEAD")
    spec.base_commit = initial
    current = [initial]
    runners = []

    def factory(spools, prefix):
        runner = FakeRunner(spools, prefix, _effect)
        runners.append(runner)
        return runner

    turns = []

    def dispatch(role, what, prompt):
        assert what == "roadmap"
        turns.append(what)
        if len(turns) == 1:
            (spec.repo / "feature.py").write_text("MTP = True\n")
            _git(spec.repo, "add", "feature.py")
            _git(spec.repo, "commit", "-qm", "feature")
            current[0] = _git(spec.repo, "rev-parse", "HEAD")
        (spec.workdir / "roadmap.yaml").write_text("[]\n")
        return ""

    state = {}
    stage = TuneStage(
        spec,
        state,
        dispatch=dispatch,
        checkpoint=lambda: None,
        runner_factory=factory,
        poll_s=0,
        log=lambda m: None,
        tested_source=lambda: current[0],
    )
    assert stage.run() == "tuned"
    assert state["source_commit"] == current[0]
    assert Path(state["best_tree"], "feature.py").exists()
    assert sum((r.submitted.count("t-baseline") for r in runners)) == 2
    meta = json.loads((spec.workdir / "data/t-baseline/meta.json").read_text())
    assert meta["tree"]["commit"] == current[0]
    assert turns == ["roadmap", "roadmap"]
    assert list(spec.workdir.glob("roadmap.yaml.stale-*"))


def test_changed_inputs_revalidate_keeps_and_archive_old_checkpoint(tmp_path):
    camp = _campaign(tmp_path, FakeRunner(["g1"], "gate", _effect), [])
    while not camp.state.kept:
        camp.step()
    for _ in range(4):
        camp.step()
    old = camp.state.best_commit
    camp.state.save(camp.state_path)
    camp.base_configs[0].write_text("modeling_v2: require\nmax_batch_size: 128\n")
    resumed = Campaign(
        workdir=camp.workdir,
        repo=camp.repo,
        bench=camp.bench,
        base_configs=camp.base_configs,
        perf_runner=FakeRunner(["s1"], "camp", _effect),
        gate_runner=FakeRunner(["g1"], "gate", _effect),
        keep_gate=camp.keep_gate,
        noise=camp.noise,
        log=lambda m: None,
    )
    assert resumed.evidence_changed and resumed.state.generation == 1
    assert resumed.state.best_commit == old and resumed._checkpoint_due()
    assert not resumed.state.controls and (not resumed.state.candidates["fast-path"].projected)
    label = resumed._label(f"ckpt-{old[:10]}")
    stale = resumed.data / label
    stale.mkdir(parents=True)
    (stale / "meta.json").write_text('{"label":"old","tree":{"commit":"wrong"}}')
    (stale / "summary.json").write_text("[]")
    for _ in range(25):
        if not resumed.step():
            break
    assert resumed.state.kept == ["fast-path"]
    assert resumed.state.checkpoint == {"good": old}
    assert list(resumed.data.glob(f"{label}.stale-*"))
    # A config change can coincide with new validated source while best already carries keeps.
    from agent_flow.workflows.trtllm_modeling_v2.tune.measure import evidence_key
    from agent_flow.workflows.trtllm_modeling_v2.tune.spec import TuneSpec

    source = _git(camp.repo, "rev-parse", "HEAD")
    (camp.repo / "feature.py").write_text("MTP = True\n")
    _git(camp.repo, "add", "feature.py")
    _git(camp.repo, "commit", "-qm", "feature")
    latest = _git(camp.repo, "rev-parse", "HEAD")
    spec = TuneSpec(
        workdir=camp.workdir,
        repo=camp.repo,
        model=camp.bench.model,
        target="t/",
        identity_configs=camp.base_configs,
        isl=8192,
        osl=1024,
        concurrency=[1, 128],
        spools=["s1"],
        dev_spools=["g1"],
        env_script=Path(camp.bench.env_script),
        campaign={"max_hours": 0.001},
        sol={"enabled": False},
        base_commit=source,
        task=dict(camp.keep_gate.spec),
    )
    state = {
        "phase": "campaign",
        "source_commit": source,
        "baseline": "t-baseline",
        "measurement_inputs": "old",
        "started_at": 0,
    }
    stage = TuneStage(
        spec,
        state,
        dispatch=lambda *a: "",
        checkpoint=lambda: None,
        runner_factory=lambda s, p: FakeRunner(s, p, _effect),
        poll_s=0,
        tested_source=lambda: latest,
        log=lambda m: None,
    )
    assert stage.run() == "tuned"
    assert state["source_commit"] == latest and state["kept"] == ["fast-path"]
    assert state["measurement_inputs"] == evidence_key(spec.bench(), spec.identity_configs)


def test_changed_constraints_are_not_reused_as_current_measurements(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.tune.measure import (
        BenchPart,
        evidence_key,
        validate_measurement,
    )
    from agent_flow.workflows.trtllm_modeling_v2.tune.records import Measurement

    spec = _spec(tmp_path)
    bench = spec.bench()
    config = spec.identity_configs
    bench.parts = [BenchPart("tp1", 8192, 1024, [1, 128], [], {"ttft_ms_max": 1000})]
    state = {}
    stage = TuneStage(
        spec,
        state,
        dispatch=lambda *a: "",
        checkpoint=lambda: None,
        runner_factory=lambda s, p: FakeRunner(s, p, _effect),
        poll_s=0,
    )
    tree = stage._best_tree()
    m = stage._measure({"old": config}, tree=str(tree), bench=bench)["old"]
    key = evidence_key(bench, config)
    bench.parts = [replace(bench.parts[0], constraint={"ttft_ms_max": 2000})]
    assert evidence_key(bench, config) != key
    with pytest.raises(ValueError, match="constraints"):
        validate_measurement(m, bench, config, _git(tree, "rev-parse", "HEAD"))
    fresh = stage._measure({"old": config}, tree=str(tree), bench=bench)["old"]
    assert fresh.constraints == {"tp1": {"ttft_ms_max": 2000}}
    assert Measurement.load(m.path).constraints == fresh.constraints
    assert list(stage.data.glob("old.stale-*"))


def test_final_validation_pairs_the_original_baseline_on_the_new_host(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from agent_flow.workflows.trtllm_modeling_v2.tune.measure import (
        load_measurement,
        measurement_hosts,
    )

    spec = _spec(tmp_path, enabled=False)
    _watch(spec.spools[0], "h1")
    runners = []

    class FinalRunner(FakeRunner):
        def poll(self):
            done = super().poll()
            if any((label == "t-final" for label, _, _ in done)):
                _watch(spec.spools[0], "h3")
            return done

    def factory(spools, prefix):
        runner = FinalRunner(spools, prefix, _effect)
        runners.append(runner)
        return runner

    state = {}
    stage = TuneStage(
        spec,
        state,
        dispatch=lambda *a: "",
        checkpoint=lambda: None,
        runner_factory=factory,
        poll_s=0,
        log=lambda m: None,
    )
    assert stage.ground()
    tree = stage._best_tree()
    (tree / "modeling.py").write_text("FAST = True\n")
    _git(tree, "commit", "-qam", "fast")
    state.update(phase="wrap", kept=["fast"], best_tree=str(tree))
    _watch(spec.spools[0], "h2")
    assert stage.run() == "tuned"
    control = load_measurement(stage.data, state["final_control"])
    final = load_measurement(stage.data, "t-final")
    assert measurement_hosts(final) == measurement_hosts(control) == {"": "h3"}
    assert list(stage.data.glob("t-final.stale-*"))
    assert final.meta["tree"]["commit"] != control.meta["tree"]["commit"]
    assert state["final_validation"]["status"].startswith("WIN")
    submitted = [label for r in runners for label in r.submitted]
    assert submitted.count("t-baseline") == 1
    assert len([label for label in submitted if label.startswith("t-final-control-")]) == 1


def test_shared_gate_discovery_keeps_the_required_target_path_when_missing(tmp_path):
    spec = _spec(tmp_path)
    shared = spec.repo / "tests/unittest/_torch/modeling_v2"
    shared.mkdir(parents=True)
    (shared / "test_modeling_v2_claims.py").write_text("def test_claim(): pass\n")
    _git(spec.repo, "add", ".")
    _git(spec.repo, "commit", "-qm", "shared gate")
    stage = TuneStage(
        spec,
        {},
        dispatch=lambda *a: "",
        checkpoint=lambda: None,
        runner_factory=lambda s, p: FakeRunner(s, p, _effect),
    )
    assert stage._gate_tests() == [
        "t/",
        "tests/unittest/_torch/modeling_v2/test_modeling_v2_claims.py",
    ]


def _watch(spool: str, host: str) -> None:
    """The session on ``spool`` now runs on ``host``."""
    Path(spool).mkdir(parents=True, exist_ok=True)
    record = {"watcher": host, "host": host, "alive_epoch": time.time(), "slurm_job": "1"}
    Path(spool, "status.json").write_text(json.dumps(record))


class MovingRunner(FakeRunner):
    """The session moves from h1 to h2 just before the first job ``moves(label)`` picks runs."""

    def __init__(self, spools, prefix, moves, moved: list):
        super().__init__(spools, prefix, _budget_effect)
        self.moves, self.moved = (moves, moved)

    def poll(self):
        for spool, label in list(self.inflight.values()):
            if not self.moved and self.moves(label):
                self.moved.append(label)
                _watch(spool, "h2")
        return super().poll()


def _run_moving(tmp_path, monkeypatch, moves, scope: str):
    monkeypatch.chdir(tmp_path)
    spec = _spec(tmp_path, max_roadmaps=1)
    _watch(spec.spools[0], "h1")

    def dispatch(role, what, prompt):
        ws = spec.workdir
        if what == "roadmap":
            (ws / "roadmap.yaml").write_text(yaml.safe_dump([{"slug": "fast-path"}]))
            return ""
        wt = ws / "worktrees" / what
        (wt / "modeling.py").write_text("FAST = True\n")
        _git(wt, "commit", "-q", "-am", what)
        decision = {
            "decision": "CANDIDATE",
            "worktree": str(wt),
            "commits": [_git(wt, "rev-parse", "HEAD")],
        }
        if scope == "micro":
            decision["microbench"] = _microbench(ws / "candidates" / what)
        else:
            decision["perf_scope"] = "e2e"
        (ws / "candidates" / what / "decision.yaml").write_text(yaml.safe_dump(decision))
        return ""

    runners: list[MovingRunner] = []
    moved: list[str] = []

    def runner_factory(spools, prefix):
        runners.append(MovingRunner(spools, prefix, moves, moved))
        return runners[-1]

    state: dict = {"phase": "campaign"}
    stage = TuneStage(
        spec,
        state,
        dispatch=dispatch,
        checkpoint=lambda: None,
        runner_factory=runner_factory,
        poll_s=0,
        log=lambda m: None,
    )
    assert stage.ground()
    assert stage.run() == "tuned"
    camp = json.loads((spec.workdir / "campaign_state.json").read_text())["candidates"]
    return (state, camp["fast-path"], [label for r in runners for label in r.submitted])


def test_a_micro_ab_split_by_a_session_move_measures_both_sides_again(tmp_path, monkeypatch):
    state, fast, submitted = _run_moving(
        tmp_path, monkeypatch, lambda label: label.endswith("-candidate"), "micro"
    )
    micro = [label for label in submitted if label.startswith("micro-")]
    assert sorted(micro) == sorted([micro[0], micro[1]] * 2) and micro[0].endswith("-control")
    assert fast["microbench"]["host_moves"] == 1
    assert state["kept"] == ["fast-path"]


def test_a_serving_ab_split_by_a_session_move_takes_the_new_hosts_control(tmp_path, monkeypatch):
    state, fast, submitted = _run_moving(
        tmp_path, monkeypatch, lambda label: label.startswith("e2e-"), "e2e"
    )
    controls = [label for label in submitted if label.startswith("ctl-")]
    assert len([label for label in submitted if label.startswith("e2e-")]) == 1
    assert controls and controls[-1].endswith("-h2")
    assert fast["microbench"]["host_moves"] == 1
    assert state["kept"] == ["fast-path"]


def test_a_checkpoint_split_by_a_session_move_measures_its_control_on_the_new_node(
    tmp_path, monkeypatch
):
    state, fast, submitted = _run_moving(
        tmp_path, monkeypatch, lambda label: label.startswith("ckpt-"), "micro"
    )
    ckpt = next((i for i, label in enumerate(submitted) if label.startswith("ckpt-")))
    assert [label for label in submitted[ckpt:] if label.startswith("ctl-")][-1].endswith("-h2")
    assert state["kept"] == ["fast-path"]


def test_a_control_registered_for_a_host_it_was_not_measured_on_is_not_reused(tmp_path):
    """A hand-over can register a predecessor's measurement under the successor's host."""
    from agent_flow.workflows.trtllm_modeling_v2.tune.campaign import Campaign, CampaignState

    from .test_tune_configs import _write

    _write(tmp_path, "ckpt-old", {}, {}, host="h1")
    camp = object.__new__(Campaign)
    camp.data, camp.state = (tmp_path, CampaignState())
    camp.state.controls = {"good|h2": "ckpt-old", "good|h1": "ckpt-old"}
    assert camp._host_control("good", "h1") == "ckpt-old"
    assert camp._host_control("good", "h2") is None
    assert "good|h2" not in camp.state.controls
    camp.state.controls["good|h3"] = "ctl-good-h3"
    assert camp._host_control("good", "h3") == "ctl-good-h3"
