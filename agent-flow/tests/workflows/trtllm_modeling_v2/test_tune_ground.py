# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ground truth over every perf node: members spread, repeats on other hosts, same-host floors.

Fake nodes throughout (no GPU): each job's runs are measured on the host its session
reports, at a speed of that host's, and leave what ``bench.py`` leaves for a run.
"""

from __future__ import annotations

import json
import shlex
import time
from pathlib import Path

import pytest

from agent_flow.workflows.trtllm_modeling_v2.tune.campaign import Campaign
from agent_flow.workflows.trtllm_modeling_v2.tune.measure import (
    BenchPart,
    BenchSpec,
    Run,
    SpoolRunner,
    host_controls,
    write_index,
)
from agent_flow.workflows.trtllm_modeling_v2.tune.records import (
    Measurement,
    NoiseFloor,
    ground_floors,
)
from agent_flow.workflows.trtllm_modeling_v2.tune.stage import TuneSpec, TuneStage
from agent_flow.workflows.trtllm_modeling_v2.tune.worktree import open_candidate

from .test_tune_stage import FakeRunner, _contract, _effect, _git, _keep_gate, _repo

MEMBERS = ("tp1", "tp2", "tep2")
GROUND = ("t-baseline", "t-baseline-r1", "t-baseline-r2")


class Nodes(FakeRunner):
    """Measures each run of a job in turn on the host its session reports.

    ``speed[host]`` scales what that host measures. ``fail[run]`` lists the return codes
    the job gets when it reaches that run, one per time (the runs after it never start).
    ``busy[spool]`` is how many reads of the idle nodes still find that node busy with
    another job. ``jobs_run`` records each job: its session and its runs.
    """

    def __init__(self, spools, prefix, data, speed=None, fail=None, busy=None):
        super().__init__(spools, prefix, _effect)
        self.data = Path(data)
        self.speed = dict(speed or {})
        self.fail = {run: list(rcs) for run, rcs in (fail or {}).items()}
        self.busy = dict(busy or {})
        self.jobs_run: list[tuple[str, str, list[str]]] = []

    def free_spools(self):
        free = []
        for spool in super().free_spools():
            if self.busy.get(spool, 0) > 0:
                self.busy[spool] -= 1
                continue
            free.append(spool)
        return free

    def submit(self, spool, label, command, **kw):
        self.jobs_run.append((spool, label, [_run_of(step, self.data) for step in _steps(command)]))
        return super().submit(spool, label, command, **kw)

    def poll(self):
        done = []
        for name, (spool, job) in list(self.inflight.items()):
            command, tree = self.jobs[job]
            host = json.loads(Path(spool, "status.json").read_text())["host"]
            rc = 0
            for step in _steps(command):
                run = _run_of(step, self.data)
                if self.fail.get(run):
                    rc = self.fail[run].pop(0)
                    break
                _bench(step, host, self.speed.get(host, 1.0), tree)
            del self.inflight[name]
            done.append((job, spool, rc))
        return done


def _steps(command: str) -> list[list[str]]:
    """The ``bench.py`` argv of each run of a job's command."""
    body = command.split("\n", 1)[1]
    return [shlex.split(step) for step in body.split(" && \\\n") if "--label" in step]


def _run_of(argv: list[str], data: Path) -> str:
    label, root = argv[argv.index("--label") + 1], Path(argv[argv.index("--out-root") + 1])
    return label if root == data else f"{root.name}.{label}"


def _bench(argv: list[str], host: str, speed: float, tree: str | None) -> None:
    """What ``bench.py`` leaves for one run: the curve and a meta naming its host and commit."""
    out = Path(argv[argv.index("--out-root") + 1]) / argv[argv.index("--label") + 1]
    out.mkdir(parents=True, exist_ok=True)
    cons = [int(c) for c in argv[argv.index("--concurrency") + 1].split(",")]
    rows = [{"concurrency": c, "tok_s_user": 100.0 * speed / c, "output_tok_s_gpu": 100.0 * speed,
             "mean_tpot_ms": 10.0, "mean_ttft_ms": 100.0} for c in cons]  # fmt: skip
    (out / "summary.json").write_text(json.dumps(rows))
    commit = _git(Path(tree), "rev-parse", "HEAD") if tree else None
    meta = {
        "label": out.name,
        "modeling_v2": "require",
        "hostname": host,
        "tree": {"commit": commit},
    }
    (out / "meta.json").write_text(json.dumps(meta))


def _nodes(tmp_path: Path, hosts: dict[str, str]) -> list[str]:
    """One perf session per ``{name: host}``, each reporting its host."""
    spools = []
    for name, host in hosts.items():
        path = tmp_path / name
        path.mkdir(parents=True, exist_ok=True)
        (path / "status.json").write_text(json.dumps({"host": host}))
        spools.append(str(path))
    return spools


def _task(repo: Path, spools: list[str], **perf) -> dict:
    return {
        "trtllm_repo_path": str(repo),
        "checkpoint_path": "/ckpt",
        **_contract(repo.parent),
        "perf": {
            "configs": [
                {"name": name, "workload": {"isl": 1024, "osl": 128, "concurrency": [1, 16]},
                 "llm_api_options": {"tensor_parallel_size": 1 + (name != "tp1")}}
                for name in MEMBERS
            ],
            "spools": spools,
            "env_script": "/env.sh",
            "reference": "none",
            "noise_repeats": 2,
            **perf,
        },
    }  # fmt: skip


def _stage(tmp_path, spools, **runner):
    repo = _repo(tmp_path)
    perf = {k: runner.pop(k) for k in ("reference", "noise_repeats") if k in runner}
    spec = TuneSpec.from_task(_task(repo, spools, **perf), tmp_path / "ws", target="t/")
    spec.base_commit = _git(repo, "rev-parse", "HEAD")
    runners: list[Nodes] = []

    def factory(spools, prefix):
        runners.append(Nodes(spools, prefix, spec.workdir / "data", **runner))
        return runners[-1]

    logs: list[str] = []
    stage = TuneStage(spec, {}, dispatch=lambda *a: "", checkpoint=lambda: None,
                      runner_factory=factory, poll_s=0, log=logs.append)  # fmt: skip
    return spec, stage, runners[0], logs


def _host(data: Path, label: str, member: str) -> str:
    return json.loads((data / label / member / "meta.json").read_text())["hostname"]


# ---------------------------------------------------------------------- spread over the nodes


def test_each_node_measures_its_share_and_a_members_repeats_run_on_other_hosts(tmp_path):
    """Three idle perf nodes take the baseline and both repeats at once, a member per host each.

    Run serially on one node, ground truth took three whole measurements of wall time while
    the other perf nodes stood idle.
    """
    spools = _nodes(tmp_path, {"perf": "h1", "perf-2": "h2", "perf-3": "h3"})
    spec, stage, runner, logs = _stage(tmp_path, spools, speed={"h2": 0.97})
    assert stage.ground()
    data = spec.workdir / "data"
    assert sorted(sp for sp, _, _ in runner.jobs_run) == sorted(spools)  # one job per node, at once
    assert all(len(runs) <= len(MEMBERS) for _, _, runs in runner.jobs_run)
    for member in MEMBERS:
        assert {_host(data, label, member) for label in GROUND} == {"h1", "h2", "h3"}
    base = Measurement.load(data / "t-baseline")
    assert set(base.parts) == set(MEMBERS) and base.meta["label"] == "t-baseline"
    # The repeats ran across hosts: no member ran twice on one node, so the floor a keep is
    # judged under is left to each node's controls; the spread across hosts judges nothing.
    assert stage.t["noise_spread"] == {}
    assert stage.t["cross_host_spread"]["tp1.con1.tok_s_user"] == pytest.approx(2 / 99)
    ledger = [
        json.loads(line) for line in (spec.workdir / "perf_ledger.jsonl").read_text().splitlines()
    ]
    floor = next(r for r in ledger if r["kind"] == "noise_floor")
    assert floor["spread"] == {} and floor["cross_host"]["tep2.con16.output_tok_s_gpu"] > 0.02
    assert any("across hosts up to 2.02%" in m and "not a floor" in m for m in logs)


def test_a_busy_node_is_not_waited_for_and_a_member_run_twice_on_one_node_sets_the_floor(tmp_path):
    """Another job holds the third node: the idle two take the runs; it takes what is left once free."""
    spools = _nodes(tmp_path, {"perf": "h1", "perf-2": "h2", "perf-3": "h3"})
    spec, stage, runner, _ = _stage(tmp_path, spools, speed={"h1": 1.004},
                                    busy={spools[2]: 2})  # fmt: skip
    assert stage.ground()
    data = spec.workdir / "data"
    first = runner.jobs_run[:2]
    assert [sp for sp, _, _ in first] == spools[:2]
    assert [runs for _, _, runs in first] == [
        [f"t-baseline.{m}" for m in MEMBERS],
        [f"t-baseline-r1.{m}" for m in MEMBERS],
    ]
    # Each node then takes one more run, of a member it ran once; the third node, once idle,
    # the last one.
    assert [(sp, runs) for sp, _, runs in runner.jobs_run[2:]] == [
        (spools[0], ["t-baseline-r2.tp1"]),
        (spools[1], ["t-baseline-r2.tp2"]),
        (spools[2], ["t-baseline-r2.tep2"]),
    ]
    assert _host(data, "t-baseline-r2", "tp1") == "h1"
    # tp1 ran twice on h1 and tp2 twice on h2: their same-host spread is the keep floor.
    assert set(stage.t["noise_spread"]) == {
        f"{m}.con{c}.{metric}"
        for m in ("tp1", "tp2")
        for c, metric in ((1, "tok_s_user"), (16, "output_tok_s_gpu"), (16, "tok_s_user"))
    }
    assert stage.t["noise_spread"]["tp1.con1.tok_s_user"] == pytest.approx(0.0)
    assert stage.t["noise_samples"] == 2


def test_with_more_nodes_than_samples_each_node_takes_a_whole_set_of_members(tmp_path):
    """Four nodes, three samples: three nodes each run every member once; the fourth stays free.

    Split thinner, no node would have run every member, and the campaign would measure a
    control on each before judging a candidate there.
    """
    spools = _nodes(tmp_path, {"perf": "h1", "perf-2": "h2", "perf-3": "h3", "perf-4": "h4"})
    spec, stage, runner, _ = _stage(tmp_path, spools)
    assert stage.ground()
    assert [(sp, runs) for sp, _, runs in runner.jobs_run] == [
        (spools[n], [f"{label}.{m}" for m in MEMBERS]) for n, label in enumerate(GROUND)
    ]
    data = spec.workdir / "data"
    assert host_controls(data, list(GROUND), list(MEMBERS)) == [data / label for label in GROUND]


def test_on_one_node_the_runs_go_label_by_label_and_the_floor_is_the_same_host_one(tmp_path):
    """With one perf node, ground truth is what it always was: the baseline, then each repeat."""
    spools = _nodes(tmp_path, {"perf": "h1"})
    spec, stage, runner, _ = _stage(tmp_path, spools)
    assert stage.ground()
    assert [runs for _, _, runs in runner.jobs_run] == [
        [f"{label}.{m}" for m in MEMBERS] for label in GROUND
    ]
    data = spec.workdir / "data"
    old = NoiseFloor.from_measurements([Measurement.load(data / label) for label in GROUND])
    assert stage.t["noise_spread"] == old.spread and stage.t["noise_samples"] == 3
    assert stage.t["cross_host_spread"] == {}
    assert host_controls(data, list(GROUND), list(MEMBERS)) == [data / label for label in GROUND]


def test_the_reference_fills_the_nodes_once_the_repeats_are_placed(tmp_path):
    spools = _nodes(tmp_path, {"perf": "h1", "perf-2": "h2", "perf-3": "h3"})
    spec, stage, runner, _ = _stage(tmp_path, spools, reference="stock")
    assert stage.ground()
    labels = [runs for _, _, runs in runner.jobs_run]
    assert labels[:3] == [[f"{label}.{m}" for m in MEMBERS] for label in GROUND]
    assert sorted(labels[3:]) == [[f"t-reference.{m}"] for m in sorted(MEMBERS)]
    reference = Measurement.load(spec.workdir / "data" / "t-reference")
    assert set(reference.meta["hosts"].values()) == {"h1", "h2", "h3"}
    assert reference.meta["hostname"] == ""  # measured across nodes: no host's control


# ------------------------------------------------------------------------- failures, restarts


def test_a_failed_run_is_measured_again_and_the_runs_after_it_carry_no_penalty(tmp_path):
    spools = _nodes(tmp_path, {"perf": "h1"})
    spec, stage, runner, logs = _stage(tmp_path, spools, noise_repeats=0,
                                       fail={"t-baseline.tp2": [212, 212]})  # fmt: skip
    assert stage.ground()
    assert [runs for _, _, runs in runner.jobs_run] == [
        [f"t-baseline.{m}" for m in MEMBERS],
        ["t-baseline.tp2", "t-baseline.tep2"],  # tp1 is on disk; tep2 never started
        ["t-baseline.tp2", "t-baseline.tep2"],
    ]
    assert logs.count("[tune] t-baseline.tp2 failed rc=212; re-queued") == 2


def test_a_run_that_keeps_failing_gives_up_its_label_and_names_its_jobs_log(tmp_path):
    spools = _nodes(tmp_path, {"perf": "h1"})
    spec, stage, runner, _ = _stage(tmp_path, spools, noise_repeats=0,
                                    fail={"t-baseline.tp2": [1, 1]})  # fmt: skip
    assert not stage.ground()
    log = Path(spools[0], "logs", f"{runner.prefix}-t-baseline.tp2.log")
    assert stage.t["failure"] == f"baseline: rc=1, see `{log}`"
    assert len(runner.jobs_run) == 2  # its first run, and one retry from where it failed


def test_a_run_measured_at_another_commit_is_set_aside_alone(tmp_path):
    """Of a label half measured before a restart, only the run of another commit is measured again."""
    spools = _nodes(tmp_path, {"perf": "h1"})
    spec, stage, runner, logs = _stage(tmp_path, spools, noise_repeats=0)
    root = spec.workdir / "data" / "t-baseline"
    best = str(stage._best_tree())
    for member, tree in (("tp1", best), ("tp2", None)):
        _bench(
            ["--label", member, "--out-root", str(root), "--concurrency", "1,16"], "h1", 1.0, tree
        )
    meta = json.loads((root / "tp2" / "meta.json").read_text())
    (root / "tp2" / "meta.json").write_text(json.dumps({**meta, "tree": {"commit": "0" * 40}}))
    assert stage.ground()
    assert [runs for _, _, runs in runner.jobs_run] == [["t-baseline.tp2", "t-baseline.tep2"]]
    assert list(root.glob("tp2.stale-*"))
    assert any("t-baseline.tp2 on disk was measured at 0000000000" in m for m in logs)
    assert Measurement.load(root).meta["hosts"] == {m: "h1" for m in MEMBERS}


def _spool(path: Path) -> str:
    for sub in ("pending", "running", "done", "logs", "history"):
        (path / sub).mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (path / "watcher.heartbeat").write_text(f"alive={stamp}\n")
    (path / "status.json").write_text(json.dumps({"host": "h1"}))
    return str(path)


class Session(SpoolRunner):
    """A real spool client; every poll first runs the jobs pending in its sessions."""

    def __init__(self, spools, prefix, data):
        super().__init__(spools, prefix, settle_s=0)
        self.data, self.queued_jobs, self.ran = Path(data), [], []

    def submit(self, spool, label, command, **kw):
        self.queued_jobs.append(label)
        return super().submit(spool, label, command, **kw)

    def poll(self):
        for spool in self.spools:
            for script in sorted(Path(spool, "pending").glob("*.sh")):
                # Past the shebang, ``set -u`` and the ``#spool:`` header: the job's command.
                for argv in _steps(script.read_text().split("\n", 3)[3]):
                    self.ran.append(_run_of(argv, self.data))
                    _bench(argv, "h1", 1.0, None)
                Path(spool, "done", f"{script.stem}.rc").write_text("0\n")
                script.unlink()
        return super().poll()


def test_a_restart_adopts_the_job_its_runs_are_in_and_queues_the_rest(tmp_path):
    """The jobs a measurement has in flight are checkpointed with their runs: a restart waits for them."""
    spool = _spool(tmp_path / "perf")
    repo = _repo(tmp_path)
    spec = TuneSpec.from_task(_task(repo, [spool], noise_repeats=1), tmp_path / "ws", target="t/")
    data = spec.workdir / "data"
    sessions: list[Session] = []

    def factory(spools, prefix):
        sessions.append(Session(spools, prefix, data))
        return sessions[-1]

    state: dict = {}
    stage = TuneStage(spec, state, dispatch=lambda *a: "", checkpoint=lambda: None,
                      runner_factory=factory, poll_s=0, log=lambda m: None)  # fmt: skip
    # What the last process left: one job of the baseline's runs, pending in the session.
    runs = [Run("t-baseline", m) for m in MEMBERS]
    labels = {label: [] for label in ("t-baseline", "t-baseline-r1")}
    left = spec.bench().runs_command(runs, data, labels)
    Path(spool, "pending", f"{sessions[0].prefix}-t-baseline.tp1.sh").write_text(
        "#!/bin/bash\nset -u\n#spool: gpus=all wall=2h\n" + left + "\n"
    )
    state["queued_runs"] = {"t-baseline.tp1": [[r.label, r.part] for r in runs]}
    assert stage.ground()
    assert sessions[0].queued_jobs == ["t-baseline-r1.tp1"]  # the baseline's job was adopted
    assert sessions[0].ran == [r.name for r in runs] + [f"t-baseline-r1.{m}" for m in MEMBERS]
    assert state["queued_runs"] == {}


# ------------------------------------------------------------------------ per-host controls


def _measured(data: Path, label: str, hosts: dict[str, str], speed: float = 1.0) -> None:
    for member, host in hosts.items():
        _bench(["--label", member, "--out-root", str(data / label), "--concurrency", "1,16"],
               host, speed, None)  # fmt: skip
    parts = [BenchPart(m, 1024, 128, [1, 16], []) for m in hosts]
    write_index(data, label, parts)


def test_a_hosts_runs_of_the_baseline_and_its_repeats_join_into_its_controls(tmp_path):
    """Each time a node ran every member is a control of its host, from whichever labels ran them."""
    data = tmp_path / "data"
    _measured(data, "t-baseline", {"tp1": "h1", "tp2": "h2"})
    _measured(data, "t-baseline-r1", {"tp1": "h2", "tp2": "h1"}, speed=1.01)
    _measured(data, "t-baseline-r2", {"tp1": "h3", "tp2": "h3"})
    base = Measurement.load(data / "t-baseline")
    assert base.meta["hosts"] == {"tp1": "h1", "tp2": "h2"} and base.meta["hostname"] == ""
    controls = host_controls(data, list(GROUND), ["tp1", "tp2"])
    assert [c.name for c in controls] == ["t-ground-h1-1", "t-ground-h2-1", "t-baseline-r2"]
    h1 = Measurement.load(controls[0])
    assert h1.meta["hostname"] == "h1" and h1.meta["runs"] == {
        "tp1": "t-baseline/tp1", "tp2": "t-baseline-r1/tp2",
    }  # fmt: skip
    assert h1.parts["tp2"].points[1]["tok_s_user"] == pytest.approx(101.0)
    assert host_controls(data, list(GROUND), ["tp1", "tp2"]) == controls  # made once, read again

    repo = _repo(tmp_path)
    parts = [BenchPart(m, 1024, 128, [1, 16], []) for m in ("tp1", "tp2")]
    bench = BenchSpec(bench_py="b", model="m", env_script="e", isl=1024, osl=128,
                      concurrencies=[1, 16], tree=str(repo), parts=tuple(parts))  # fmt: skip
    work = tmp_path / "camp"
    work.mkdir()
    (work / "data").symlink_to(data, target_is_directory=True)
    camp = Campaign(workdir=work, repo=repo, bench=bench, base_configs=[],
                    perf_runner=FakeRunner(["p"], "c", _effect), gate_runner=FakeRunner(["g"], "g", _effect),
                    keep_gate=_keep_gate(tmp_path), noise=NoiseFloor(), log=lambda m: None)  # fmt: skip
    camp.state.best_tree = str(open_candidate(repo, work / "wt", "best", "HEAD"))
    camp.state.best_commit = _git(Path(camp.state.best_tree), "rev-parse", "HEAD")
    for path in controls:
        camp.set_control(path)
    assert [m.label for m in camp._controls("h1")] == ["t-ground-h1-1"]
    assert [m.label for m in camp._controls("h3")] == ["t-baseline-r2"]
    assert camp.state.control_label == "t-ground-h1-1"


def test_the_floor_a_keep_is_judged_under_is_never_the_spread_across_hosts(tmp_path):
    data = tmp_path / "data"
    _measured(data, "t-baseline", {"tp1": "h1", "tp2": "h1"})
    _measured(data, "t-baseline-r1", {"tp1": "h1", "tp2": "h2"}, speed=1.002)
    _measured(data, "t-baseline-r2", {"tp1": "h2", "tp2": "h2"}, speed=0.96)
    same, cross = ground_floors([Measurement.load(data / label) for label in GROUND])
    # tp1 ran twice on h1 (1.0, 1.002), tp2 twice on h2 (1.002, 0.96): each host's own spread.
    assert same.spread["tp1.con1.tok_s_user"] == pytest.approx(0.1 / 100.1)
    assert same.spread["tp2.con1.tok_s_user"] == pytest.approx(2.1 / 98.1)
    assert same.samples == 2
    # Across hosts tp1 read 100, 100.2 and 96: the hosts' difference, far wider than h1's own.
    mean = (100 + 100.2 + 96) / 3
    assert cross.spread["tp1.con1.tok_s_user"] == pytest.approx((mean - 96) / mean)
    one = Measurement.load(data / "t-baseline").parts["tp1"]
    assert ground_floors([one, one])[0].spread["con1.tok_s_user"] == 0.0  # one config: no prefix
