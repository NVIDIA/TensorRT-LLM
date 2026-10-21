# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a restart or a new attempt of the ``tune`` task takes over, and the perf smoke before it."""

from __future__ import annotations

import json
import shlex
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.tune import stage as stage_mod
from agent_flow.workflows.trtllm_modeling_v2.tune.bench import resolve_config
from agent_flow.workflows.trtllm_modeling_v2.tune.measure import SpoolRunner
from agent_flow.workflows.trtllm_modeling_v2.tune.stage import TuneSpec, TuneStage

from .test_tune_stage import FakeRunner, _contract, _effect, _git, _repo, _write_sol


def _task(repo: Path, identity: Path, **perf) -> dict:
    return {
        "trtllm_repo_path": str(repo),
        "checkpoint_path": "/ckpt",
        **_contract(repo.parent),
        "perf": {
            "identity_configs": [str(identity)],
            "workload": {"isl": 8192, "osl": 1024, "concurrency": [1, 128]},
            "spools": ["s1"],
            "env_script": "/env.sh",
            "reference": "none",
            "noise_repeats": 0,
            **perf,
        },
    }


def _identity(tmp_path: Path, **config) -> Path:
    path = tmp_path / "identity.yaml"
    path.write_text(yaml.safe_dump({"modeling_v2": "require", **config}))
    return path


def _argv(command: str) -> list[str]:
    return shlex.split(command.strip().splitlines()[-1])


def _result(out: Path, argv: list[str], tree: str | None) -> None:
    """What bench.py leaves for one run: the curve, and a meta.json naming what it measured."""
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        {"concurrency": 1, "tok_s_user": 100.0, "output_tok_s_gpu": 100.0,
         "mean_tpot_ms": 10, "mean_ttft_ms": 200},
        {"concurrency": 128, "tok_s_user": 15.6, "output_tok_s_gpu": 2000.0,
         "mean_tpot_ms": 50, "mean_ttft_ms": 4000},
    ]  # fmt: skip
    (out / "summary.json").write_text(json.dumps(rows))
    configs = [Path(argv[i + 1]) for i, a in enumerate(argv) if a == "--config"]
    config = resolve_config(configs, [])
    meta = {
        "label": out.name,
        "modeling_v2": config.get("modeling_v2", "off"),
        "config": json.loads(json.dumps(config, default=str)),
        "isl": int(argv[argv.index("--isl") + 1]),
        "osl": int(argv[argv.index("--osl") + 1]),
        "concurrencies": [int(c) for c in argv[argv.index("--concurrency") + 1].split(",")],
        "tree": {"commit": _git(Path(tree), "rev-parse", "HEAD") if tree else None},
        "hostname": "h1",
    }
    (out / "meta.json").write_text(json.dumps(meta))


class Recorded(FakeRunner):
    """A runner whose results record what bench.py records, and that counts every job."""

    def __init__(self, spools, prefix, effect, submitted: list[str] | None = None):
        super().__init__(spools, prefix, effect)
        self.submitted = submitted if submitted is not None else []

    def submit(self, spool, label, command, *, wall="2h", tree=None, gpus="all"):
        self.submitted.append(label)
        return super().submit(spool, label, command, wall=wall, tree=tree, gpus=gpus)

    def poll(self):
        done = []
        for name, (spool, label) in list(self.inflight.items()):
            command, tree = self.jobs[label]
            argv = _argv(command)
            _result(Path(argv[argv.index("--out-root") + 1]) / label, argv, tree)
            del self.inflight[name]
            done.append((label, spool, 0))
        return done


# ----------------------------------------------------------------------------- receipts


def _spool(path: Path) -> str:
    for sub in ("pending", "running", "done", "logs", "history"):
        (path / sub).mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (path / "watcher.heartbeat").write_text(f"alive={stamp}\n")
    return str(path)


class Session(SpoolRunner):
    """A real spool client over a session run in the caller's thread.

    Every poll first runs the pending jobs; ``rcs[label]`` lists the return codes of
    that label's next runs (0 when it runs out).
    """

    def __init__(self, spools, prefix, rcs):
        super().__init__(spools, prefix, settle_s=0)
        self.rcs, self.runs = rcs, []

    def poll(self):
        for spool in self.spools:
            for script in sorted(Path(spool, "pending").glob("*.sh")):
                label = script.stem.split("-", 1)[1]
                self.runs.append(label)
                rc = self.rcs[label].pop(0) if self.rcs.get(label) else 0
                if rc == 0:
                    argv = _argv(script.read_text())
                    out = Path(argv[argv.index("--out-root") + 1]) / label
                    tree = next(
                        (f.split("=", 1)[1] for f in script.read_text().split() if f.startswith("tree=")),
                        None,
                    )  # fmt: skip
                    _result(out, argv, tree)
                Path(spool, "done", f"{script.stem}.rc").write_text(f"{rc}\n")
                script.unlink()
        return super().poll()


def test_a_receipt_is_adopted_only_while_it_can_still_be_this_runs_result(tmp_path):
    """Pending, running and succeeded jobs are adopted; a failed receipt is set aside on the next submit."""
    spool = _spool(tmp_path / "spool")
    runner = SpoolRunner([spool], prefix="tuneab12cd")
    done = Path(spool, "done")
    for label, where, text in (
        ("queued", "pending", "#!/bin/bash\n"),
        ("running", "running", "#!/bin/bash\n"),
        ("passed", "done", "0\n"),
        ("failed", "done", "1\n"),
        ("blank", "done", ""),
    ):
        suffix = ".rc" if where == "done" else ".sh"
        Path(spool, where, f"tuneab12cd-{label}{suffix}").write_text(text)
    assert [runner.adopt(label) for label in ("queued", "running", "passed")] == [True] * 3
    assert not runner.adopt("failed") and not runner.adopt("blank") and not runner.adopt("never")
    runner.submit(spool, "failed", "true")
    assert not (done / "tuneab12cd-failed.rc").exists()
    assert list(done.glob("tuneab12cd-failed.rc.superseded-*"))
    assert Path(spool, "pending", "tuneab12cd-failed.sh").exists()


def test_a_retry_adopts_the_last_attempts_jobs_in_flight_never_its_receipts(tmp_path):
    spool = _spool(tmp_path / "spool")
    Path(spool, "running", "tuneab12cd-t-baseline.sh").write_text("#!/bin/bash\n")
    Path(spool, "done", "tuneab12cd-t-reference.rc").write_text("0\n")
    Path(spool, "done", "tuneab12cd-t-baseline-r1.rc").write_text("1\n")
    retry = SpoolRunner([spool], prefix="tuneab12cda2", earlier=["tuneab12cd"])
    assert retry.adopt("t-baseline") and retry.queued("t-baseline")  # never measured twice at once
    assert not retry.adopt("t-reference") and not retry.adopt("t-baseline-r1")
    assert retry.cancel("t-baseline") and not retry.queued("t-baseline")
    assert Path(spool, "cancel", "tuneab12cd-t-baseline").exists()


@pytest.mark.parametrize("attempt", [1, 2], ids=["planner-clone", "retry-in-place"])
def test_a_new_attempt_gets_its_own_retry_not_the_last_attempts_failure(tmp_path, attempt):
    """Tune failed its baseline twice; the next attempt (the flow fixed) must not adopt that failure.

    Taking the old failed receipt as its own first result would spend the attempt's only
    retry on it: its one real run fails once more and it fails like tune did. A planner's
    clone starts again at attempt 1 (the same job names); a retry in place is attempt 2
    (its own). What tune left that is good -- the smoke measured at the same config and
    commit -- is reused either way.
    """
    repo = _repo(tmp_path)
    task = _task(repo, _identity(tmp_path), spools=[_spool(tmp_path / "spool")],
                 campaign={"enabled": False})  # fmt: skip
    spec = TuneSpec.from_task(task, tmp_path / "ws")
    spec.base_commit = _git(repo, "rev-parse", "HEAD")
    state: dict = {}
    sessions = []

    def run(rcs, attempt=1):
        def factory(spools, prefix):
            sessions.append(Session(spools, prefix, rcs))
            return sessions[-1]

        return TuneStage(spec, state, dispatch=lambda *a: "", checkpoint=lambda: None,
                         runner_factory=factory, poll_s=0, log=lambda m: None, attempt=attempt).run()  # fmt: skip

    assert run({"t-baseline": [1, 1]}) == "tune-failed:baseline"
    assert sessions[0].runs == ["t-smoke-w1", "t-baseline", "t-baseline"]
    log = Path(spec.spools[0], "logs", f"{sessions[0].prefix}-t-baseline.log")
    assert state["failure"] == f"baseline: rc=1, see `{log}`"
    # The next attempt's first run fails too (rc 1); its own retry measures.
    assert run({"t-baseline": [1, 0]}, attempt) == "tuned"
    assert sessions[1].runs == ["t-baseline", "t-baseline", "t-final"]
    assert sessions[1].prefix == sessions[0].prefix + ("a2" if attempt == 2 else "")
    assert "failure" not in state


def test_a_measurement_of_another_config_or_commit_is_measured_again(tmp_path):
    """A label on disk is reused only at the same workload, config and commit; anything else is set aside."""
    repo = _repo(tmp_path)
    identity = _identity(tmp_path)
    spec = TuneSpec.from_task(_task(repo, identity, campaign={"smoke": False}), tmp_path / "ws")
    spec.base_commit = _git(repo, "rev-parse", "HEAD")
    submitted: list[str] = []
    logs: list[str] = []

    def stage(state):
        return TuneStage(spec, state, dispatch=lambda *a: "", checkpoint=lambda: None,
                         runner_factory=lambda spools, prefix: Recorded(spools, prefix, _effect, submitted),
                         poll_s=0, log=logs.append)  # fmt: skip

    assert stage({}).ground()
    assert stage({}).ground() and submitted == ["t-baseline"]  # same config, same commit: reused
    identity.write_text(yaml.safe_dump({"modeling_v2": "require", "max_batch_size": 64}))
    assert stage({}).ground() and submitted == ["t-baseline", "t-baseline"]
    assert list((spec.workdir / "data").glob("t-baseline.stale-*"))
    assert any("t-baseline on disk was measured at another config" in m for m in logs)
    # The wrap-up measures the best tree as it is: a keep since the last t-final re-measures it.
    best = spec.workdir / "worktrees" / "best"
    tuned = stage({"phase": "wrap", "best_tree": str(best)})
    tuned.wrap()
    tuned.wrap()
    assert submitted.count("t-final") == 1
    (best / "modeling.py").write_text("FAST = True\n")
    _git(best, "commit", "-qam", "a keep")
    tuned.wrap()
    assert submitted.count("t-final") == 2
    assert any(f"t-final on disk was measured at {spec.base_commit[:10]}" in m for m in logs)


# ------------------------------------------------------------------------------- resume


class Killed(Exception):
    """The orchestrator died here."""


def test_a_restarted_campaign_skips_what_is_on_disk(tmp_path):
    """An orchestrator restart re-runs the tune task from its last checkpoint.

    The smoke, the baseline, its noise repeats, the SOL turn and the roadmap turn already
    answered are neither measured nor asked again; the campaign goes on where it stopped.
    """
    repo = _repo(tmp_path)
    task = _task(
        repo,
        _identity(tmp_path),
        noise_repeats=1,
        campaign={"max_roadmaps": 1, "kernel_ledger": "off"},
    )
    spec = TuneSpec.from_task(task, tmp_path / "ws", target="t/")
    spec.base_commit = _git(repo, "rev-parse", "HEAD")
    turns: list[str] = []
    submitted: list[str] = []
    crash = {"sol", "roadmap"}  # each turn writes its answer, then the orchestrator dies

    def dispatch(role, what, prompt):
        turns.append(what)
        if what == "sol":
            _write_sol(spec.workdir)
        elif what == "roadmap":
            (spec.workdir / "roadmap.yaml").write_text(
                yaml.safe_dump([{"slug": "slow-path", "hypothesis": "h"}])
            )
        else:
            (spec.workdir / "candidates" / what / "decision.yaml").write_text(
                yaml.safe_dump({"decision": "DROPPED", "worktree": ""})
            )
        (spec.workdir / f"{what}-decision.yaml").write_text("decision: CANDIDATE\n")
        if what in crash:
            crash.discard(what)
            raise Killed(what)
        return ""

    disk: dict = {}  # state.json: what the last checkpoint wrote

    def process():
        state = json.loads(json.dumps(disk))

        def checkpoint():
            disk.clear()
            disk.update(json.loads(json.dumps(state)))

        return TuneStage(spec, state, dispatch=dispatch, checkpoint=checkpoint,
                         runner_factory=lambda spools, prefix: Recorded(spools, prefix, _effect, submitted),
                         poll_s=0, log=lambda m: None)  # fmt: skip

    with pytest.raises(Killed):
        process().run()
    assert submitted == ["t-smoke-w1", "t-baseline", "t-baseline-r1"] and turns == ["sol"]
    assert disk["phase"] == "campaign" and "sol" not in disk  # died before the SOL checkpoint
    with pytest.raises(Killed):
        process().run()
    assert turns == ["sol", "roadmap"] and disk["sol"]["method"] == "calculator"
    assert process().run() == "tuned"
    assert turns == ["sol", "roadmap", "slow-path"]
    assert submitted == ["t-smoke-w1", "t-baseline", "t-baseline-r1", "t-final"]


def test_a_rebase_cut_off_by_a_restart_is_asked_again_and_a_finished_one_is_taken(tmp_path):
    repo = _repo(tmp_path)
    spec = TuneSpec.from_task(_task(repo, _identity(tmp_path)), tmp_path / "ws")
    ref = "0123456789abcdef0123456789abcdef01234567"
    state = {f"rebase-{slug}-{ref[:10]}": True for slug in ("finished", "cut-off")}
    stage = TuneStage(spec, state, dispatch=lambda *a: "", checkpoint=lambda: None,
                      runner_factory=lambda spools, prefix: FakeRunner(spools, prefix, _effect),
                      poll_s=0, log=lambda m: None)  # fmt: skip
    for slug, extra in (("finished", {"rebased_onto": ref[:12]}), ("cut-off", {})):
        out = spec.workdir / "candidates" / slug
        out.mkdir(parents=True)
        (out / "decision.yaml").write_text(
            yaml.safe_dump({"decision": "CANDIDATE", "worktree": "", **extra})
        )
    submitted = []
    candidates = {
        slug: SimpleNamespace(slug=slug, status="conflict", rebase_ref=ref)
        for slug in ("finished", "cut-off")
    }
    camp = SimpleNamespace(state=SimpleNamespace(candidates=candidates),
                           submit_candidate=lambda slug, path: submitted.append(slug))  # fmt: skip
    stage._resume_rebases(camp)
    assert submitted == ["finished"]
    assert state == {f"rebase-finished-{ref[:10]}": True, "phase": "ground"}


# -------------------------------------------------------------------------------- smoke


def _configs_task(repo: Path, spools: list[str], **perf) -> dict:
    return {
        "trtllm_repo_path": str(repo),
        "checkpoint_path": "/ckpt",
        **_contract(repo.parent),
        "perf": {
            "configs": [
                {"name": "tp1", "workload": {"isl": 1024, "osl": 256, "concurrency": [1, 16]}},
                {"name": "tp2", "workload": {"isl": 2048, "osl": 256, "concurrency": [1, 32]},
                 "llm_api_options": {"tensor_parallel_size": 2}},
                {"name": "tep2", "workload": {"isl": 2048, "osl": 256, "concurrency": [8]},
                 "llm_api_options": {"tensor_parallel_size": 2, "moe_expert_parallel_size": 2}},
            ],
            "spools": spools,
            "env_script": "/env.sh",
            "reference": "none",
            **perf,
        },
    }  # fmt: skip


def test_the_smoke_serves_each_world_size_once_briefly(tmp_path):
    spec = TuneSpec.from_task(_configs_task(_repo(tmp_path), ["s1", "s2"]), tmp_path / "ws")
    runners = []

    def factory(spools, prefix):
        runners.append(Recorded(spools, prefix, _effect))
        return runners[-1]

    stage = TuneStage(spec, {}, dispatch=lambda *a: "", checkpoint=lambda: None,
                      runner_factory=factory, poll_s=0, log=lambda m: None)  # fmt: skip
    assert stage.smoke()
    jobs = runners[0].jobs
    assert sorted(jobs) == ["t-smoke-w1", "t-smoke-w2"]  # tep2 shares tp2's world size
    argv = _argv(jobs["t-smoke-w2"][0])
    assert argv[argv.index("--config") + 1].endswith("serving-tp2.yaml")
    assert [argv[argv.index(f) + 1] for f in ("--isl", "--osl", "--concurrency", "--num-prompts")] == [
        "1024", "32", "1", "1:4",
    ]  # fmt: skip
    records = [
        json.loads(line) for line in (spec.workdir / "perf_ledger.jsonl").read_text().splitlines()
    ]
    assert [r["worlds"] for r in records if r["kind"] == "smoke"] == [[1, 2]]


def test_a_failed_smoke_stops_the_task_before_its_baseline(tmp_path, monkeypatch):
    """A multi-rank launch that fails fails the task in minutes, with the job log the planner reads."""

    class FailsWorld2(Recorded):
        def poll(self):
            failed = []
            for name, (spool, label) in list(self.inflight.items()):
                if label == "t-smoke-w2":
                    del self.inflight[name]
                    failed.append((label, spool, 1))
            return failed + super().poll()

    submitted: list[str] = []
    runners = []

    def runner(spools, prefix, earlier):
        runners.append(FailsWorld2(spools, prefix, _effect, submitted))
        runners[-1].earlier = earlier
        return runners[-1]

    monkeypatch.setattr(stage_mod, "SpoolRunner", runner)
    monkeypatch.setattr(stage_mod.time, "sleep", lambda s: None)
    repo = _repo(tmp_path)
    wf = SimpleNamespace(task_data=_configs_task(repo, [str(tmp_path / "perf")]),
                         workspace=tmp_path / "ws", journal=SimpleNamespace(event=lambda *a: None),
                         _run_turn=lambda *a, **k: "", _checkpoint=lambda state: None)  # fmt: skip
    state = SimpleNamespace(tune={})
    decision, note = stage_mod.run_tune_task(wf, state, _git(repo, "rev-parse", "HEAD"), attempt=3)
    assert decision == "TUNE_FAILED (tune-failed:smoke)"
    assert sorted(submitted) == ["t-smoke-w1", "t-smoke-w2"]  # no retry, no baseline
    perf = runners[0]
    assert perf.prefix.endswith("a3") and perf.earlier == [
        perf.prefix[:-2],
        perf.prefix[:-2] + "a2",
    ]
    log = Path(perf.spools[0], "logs", f"{perf.prefix}-t-smoke-w2.log")
    assert note == f"smoke: world size 2 (`tp2`): rc=1, see `{log}`"
    assert state.tune["phase"] == "ground"


def test_a_smoke_that_serves_the_stock_model_fails(tmp_path):
    """A perf config that names no ``modeling_v2`` serves the stock model: that is not the target."""
    spec = TuneSpec.from_task(_task(_repo(tmp_path), _identity(tmp_path)), tmp_path / "ws")
    spec.identity_configs[0].write_text(yaml.safe_dump({"max_batch_size": 8}))
    state: dict = {}
    stage = TuneStage(spec, state, dispatch=lambda *a: "", checkpoint=lambda: None,
                      runner_factory=lambda spools, prefix: Recorded(spools, prefix, _effect),
                      poll_s=0, log=lambda m: None)  # fmt: skip
    assert stage.run() == "tune-failed:smoke"
    assert "served `modeling_v2: off`" in state["failure"]


def test_the_smoke_is_skipped_once_the_baseline_is_measured(tmp_path):
    spec = TuneSpec.from_task(_task(_repo(tmp_path), _identity(tmp_path), noise_repeats=1,
                                    campaign={"enabled": False}), tmp_path / "ws")  # fmt: skip
    submitted: list[str] = []
    factory = lambda spools, prefix: Recorded(spools, prefix, _effect, submitted)  # noqa: E731
    first = TuneStage(spec, {}, dispatch=lambda *a: "", checkpoint=lambda: None,
                      runner_factory=factory, poll_s=0, log=lambda m: None)  # fmt: skip
    first._measure({"t-baseline": spec.identity_configs})
    submitted.clear()
    stage = TuneStage(spec, {}, dispatch=lambda *a: "", checkpoint=lambda: None,
                      runner_factory=factory, poll_s=0, log=lambda m: None)  # fmt: skip
    assert stage.run() == "tuned"
    assert submitted == ["t-baseline-r1", "t-final"]
