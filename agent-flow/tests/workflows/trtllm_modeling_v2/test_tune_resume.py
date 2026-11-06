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
from agent_flow.workflows.trtllm_modeling_v2.tune.spec import TuneSpec
from agent_flow.workflows.trtllm_modeling_v2.tune.stage import TuneStage
from agent_flow.workflows.trtllm_modeling_v2.tune.worktree import measured_tree

from .test_tune_stage import (
    FakeRunner,
    _budget_effect,
    _campaign,
    _contract,
    _effect,
    _git,
    _ledger,
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


class LosesFirstRunner(FakeRunner):
    """Loses its first job to a node that went away, then runs every job."""

    lost = False

    def poll(self):
        if self.lost or not self.inflight:
            return super().poll()
        self.lost = True
        return LostRunner.poll(self)


class FailingRunner(FakeRunner):
    """Every job runs and fails: a verdict about the tree, not a lost node."""

    def poll(self):
        done = [(label, spool, 1) for spool, label in self.inflight.values()]
        self.inflight.clear()
        return done


def _deferred_stage(root: Path, runner, **campaign) -> tuple[TuneStage, dict, list[str]]:
    root.mkdir()
    spec = _spec(root, enabled=False, **campaign)
    state, logs = {}, []
    stage = TuneStage(spec, state, dispatch=lambda *a: "", checkpoint=lambda: None,
                      runner_factory=lambda s, p: runner(s, p, _effect), poll_s=0,
                      log=logs.append)  # fmt: skip
    return stage, state, logs


def test_a_lost_baseline_is_waited_out_in_process_and_a_failed_one_ends_the_run(tmp_path):
    # Without waits a lost job ends the process without a verdict; the next one resumes.
    stage, state, _ = _deferred_stage(tmp_path / "once", LostRunner)
    assert stage.run() is None
    assert state["phase"] == "ground" and "failure" not in state
    assert "rc=218" in state["deferred"]
    # With waits the same process measures it again once the session is back.
    stage, state, logs = _deferred_stage(
        tmp_path / "waits", LosesFirstRunner, max_retries=2, retry_wait_s=0
    )
    assert stage.run() == "tuned"
    waits = [r for r in _ledger(stage.spec) if r["kind"] == "deferred"]
    assert len(waits) == 1 and "rc=218" in waits[0]["reason"] and waits[0]["phase"] == "ground"
    assert any("waiting 0 s, then resuming from ground in this process" in m for m in logs)
    # A baseline that ran and failed is no wait's to fix: the process ends at once.
    stage, state, logs = _deferred_stage(
        tmp_path / "failed", FailingRunner, max_retries=2, retry_wait_s=0
    )
    assert stage.run() is None and "rc=1" in state["deferred"]
    assert not any("waiting" in m for m in logs)
    # A cause no one has judged a wait ends the process too: only those raised as transient wait.
    assert not MeasurementUnavailable("a new kind of deferral").transient


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


def test_a_standalone_tune_takes_its_tested_source_from_the_onboarding_it_names(tmp_path):
    """``perf.source_workspace``: the core's verdicts there, not the tune's own, name the source."""
    from types import SimpleNamespace

    from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract as contract_mod
    from agent_flow.workflows.trtllm_modeling_v2.tune import stage as stage_mod
    from agent_flow.workflows.trtllm_modeling_v2.tune.schema import validate_perf_block

    task = _spec(tmp_path).task
    head = _git(Path(task["trtllm_repo_path"]), "rev-parse", "HEAD")
    onboarding, standalone = tmp_path / "ws", tmp_path / "ws-tune"
    onboarding.mkdir()
    (onboarding / "task.yaml").write_text("{}\n")
    contract_mod.ContractStatus.load(onboarding).record(
        "core",
        SimpleNamespace(name="boot", member="tp1", passed=True, detail=""),
        marker=f"m-{head}",
    )
    said = []
    wf = SimpleNamespace(
        workspace=standalone,
        journal=SimpleNamespace(event=lambda what, line: said.append(line)),
        _contract=contract_mod.Contract(
            (contract_mod.Check("core", "boot", "tp1", contract_mod.TIER_INNER),)
        ),
        _head=lambda: head,
        _commit_marker=lambda sha: f"m-{sha}",
        _contract_status=lambda: contract_mod.ContractStatus.load(standalone),
        _checkpoint=lambda state: None,
    )
    wf.task_data = task
    assert stage_mod._stage_for(wf, SimpleNamespace(tune={})).tested_source() == ""
    wf.task_data = {**task, "perf": {**task["perf"], "source_workspace": str(onboarding)}}
    assert not validate_perf_block(wf.task_data)
    assert stage_mod._stage_for(wf, SimpleNamespace(tune={})).tested_source() == head
    assert any(str(onboarding) in line for line in said)
    assert validate_perf_block({"perf": {"source_workspace": 3}})
    # A path that holds no run (a relative one read from elsewhere) stops the stage at once.
    missing = tmp_path / "elsewhere" / "ws"
    wf.task_data = {**task, "perf": {**task["perf"], "source_workspace": str(missing)}}
    assert stage_mod._stage_for(wf, SimpleNamespace(tune={})) == "tune-failed:spec"
    assert f"perf.source_workspace `{missing}` holds no run" in said[-1]


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


def test_a_changed_perf_config_is_measured_again_alone_and_the_roadmap_stays(tmp_path):
    repo = _repo(tmp_path)
    files = {name: tmp_path / f"{name}.yaml" for name in ("lat", "thr")}
    for path in files.values():
        path.write_text(yaml.safe_dump({"modeling_v2": "require"}))
    workload = {"isl": 1024, "osl": 128, "concurrency": [1, 128]}
    task = {
        "trtllm_repo_path": str(repo),
        "checkpoint_path": "/ckpt",
        **_contract(tmp_path),
        "perf": {
            "configs": [
                {"name": name, "workload": workload, "identity_configs": [str(path)]}
                for name, path in files.items()
            ],
            "spools": ["s1", "s2"],
            "env_script": "/env.sh",
            "campaign": {"enabled": False},
        },
    }
    state: dict = {}

    def run() -> tuple[TuneSpec, list[str]]:
        spec = TuneSpec.from_task(task, tmp_path / "ws", target="t/")
        runners = []

        def factory(spools, prefix):
            runners.append(FakeRunner(spools, prefix, _effect))
            return runners[-1]

        stage = TuneStage(spec, state, dispatch=lambda *a: "", checkpoint=lambda: None,
                          runner_factory=factory, poll_s=0, log=lambda m: None)  # fmt: skip
        assert stage.run() == "tuned"
        return spec, [label for runner in runners for label in runner.submitted]

    spec, submitted = run()
    assert sorted(submitted) == ["t-baseline.lat", "t-baseline.thr"]
    baseline = spec.workdir / "data" / "t-baseline"
    lat = (baseline / "lat" / "meta.json").read_text()
    roadmap = [{"slug": "fuse", "share": {"lat.con1": 0.3, "thr.con128": 0.4}}]
    (spec.workdir / "roadmap.yaml").write_text(yaml.safe_dump(roadmap))
    state["sol"] = {"method": "roofline", "ceilings": {}}
    # Only the high-throughput config's CUDA graphs change.
    files["thr"].write_text(
        yaml.safe_dump({"modeling_v2": "require", "cuda_graph_config": {"max_batch_size": 128}})
    )
    spec, submitted = run()
    assert submitted == ["t-baseline.thr"]
    assert (baseline / "lat" / "meta.json").read_text() == lat
    assert list(baseline.glob("thr.stale-*")) and (baseline / "meta.json").exists()
    assert yaml.safe_load((spec.workdir / "roadmap.yaml").read_text()) == [
        {"slug": "fuse", "share": {"lat.con1": 0.3}}
    ]
    assert "sol" not in state
    (invalidated,) = [r for r in _ledger(spec) if r["kind"] == "profile_invalidated"]
    assert invalidated["configs"] == ["thr"] and invalidated["kept"] == ["roadmap"]


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
