# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A LOSS judged against how much the tree under test varies, never at a member it does not change.

Fake measurements throughout (no GPU): the floor of each best tree from its own same-host
controls, A B A' on one idle node, the re-measure of a loss one disturbed run explains, and
members the candidate provably does not change.
"""

from __future__ import annotations

import json
import os
import shlex
import time
from pathlib import Path

import yaml

from agent_flow.workflows.trtllm_modeling_v2.tune.bench import resolve_config
from agent_flow.workflows.trtllm_modeling_v2.tune.campaign import Campaign, CampaignState
from agent_flow.workflows.trtllm_modeling_v2.tune.cli import judge_ab
from agent_flow.workflows.trtllm_modeling_v2.tune.inert import inert_members
from agent_flow.workflows.trtllm_modeling_v2.tune.measure import (
    BenchPart,
    BenchSpec,
    SpoolRunner,
    node_busy,
)
from agent_flow.workflows.trtllm_modeling_v2.tune.records import (
    PARTS_FILE,
    Measurement,
    NoiseFloor,
    compare,
    remeasure_reason,
)
from agent_flow.workflows.trtllm_modeling_v2.tune.stage import TuneSpec, TuneStage
from agent_flow.workflows.trtllm_modeling_v2.tune.worktree import open_candidate

from .test_tune_stage import FakeRunner, _contract, _effect, _git, _keep_gate, _repo

RULE = {"min_gain": 0.02, "noise_factor": 3.0}
#: Two members: one rank, and two ranks (where a collective change runs).
MEMBERS = {"tp1": {"tensor_parallel_size": 1}, "tp2": {"tensor_parallel_size": 2}}
CONS = (1, 16)
MARKER = "partial sums reduce through a workspace"


def _bind(slots: int, ranks: int) -> str:
    return f"modeling_v2 x: bound Pool #2: {slots} recurrent slots, 32 states per slot\n" * ranks


def _write(root: Path, label: str, scale: dict, logs: dict, host: str = "h1") -> Measurement:
    """A measurement over :data:`MEMBERS`: ``scale[site]`` multiplies both metrics at that site."""
    for name, config in MEMBERS.items():
        part = root / label / name
        part.mkdir(parents=True, exist_ok=True)
        rows = []
        for con in CONS:
            g = scale.get(f"{name}.con{con}", 1.0)
            rows.append({"concurrency": con, "tok_s_user": 100.0 * g, "output_tok_s_gpu": 1000.0 * g,
                         "mean_tpot_ms": 10.0, "mean_ttft_ms": 100.0})  # fmt: skip
        (part / "summary.json").write_text(json.dumps(rows))
        meta = {"label": name, "config": {"modeling_v2": "require", **config}, "hostname": host}
        (part / "meta.json").write_text(json.dumps(meta))
        (part / "server.log").write_text(logs.get(name, ""))
    (root / label / PARTS_FILE).write_text(json.dumps({"parts": list(MEMBERS), "constraints": {}}))
    (root / label / "meta.json").write_text(json.dumps({"label": label, "hostname": host}))
    (root / label / "summary.json").write_text("[]")
    return Measurement.load(root / label)


# ------------------------------------------------------------------------------ the floor


def test_both_metrics_of_a_site_share_its_wider_spread():
    """dep4.con256 read 4.2% (tok/s/user) and 0.24% (tok/s/GPU) from the same two runs."""
    noise = NoiseFloor(
        spread={
            "dep4.con256.tok_s_user": 0.0417,
            "dep4.con256.output_tok_s_gpu": 0.0024,
            "tp1.con1.tok_s_user": 0.001,
        }
    )
    assert noise.of("dep4.con256.output_tok_s_gpu") == noise.of("dep4.con256.tok_s_user") == 0.0417
    assert noise.of("tp1.con1.tok_s_user") == 0.001
    wider = NoiseFloor.widest([noise, NoiseFloor(spread={"tp1.con1.tok_s_user": 0.01})])
    assert wider.of("tp1.con1.tok_s_user") == 0.01 and wider.of("dep4.con256.tok_s_user") == 0.0417


def test_a_pair_of_one_tree_judges_its_own_spread_as_noise(tmp_path):
    """e9a72cd9e9 on one host read dep4.con256 2425.6 then 2580.5: no verdict either way."""
    a = _write(tmp_path, "a", {}, {})
    a2 = _write(tmp_path, "a2", {"tp2.con16": 1.0638}, {})
    old = NoiseFloor(spread={k: 0.0008 for k in a.headline()})
    assert compare(a2, a, old, **RULE).verdict == "WIN"  # the floor of an older tree
    assert compare(a, a2, old, **RULE).verdict == "LOSS"
    own = NoiseFloor.from_measurements([a, a2])
    assert compare(a2, a, own, **RULE).verdict == compare(a, a2, own, **RULE).verdict == "NEUTRAL"


def test_the_mean_of_controls_averages_every_point_of_every_member(tmp_path):
    a = _write(tmp_path, "a", {"tp1.con1": 1.02}, {})
    a2 = _write(tmp_path, "a2", {"tp1.con1": 0.98}, {})
    mean = Measurement.mean([a, a2])
    assert mean.label == "a+a2" and mean.meta["hostname"] == "h1"
    assert mean.headline()["tp1.con1.tok_s_user"] == 100.0
    assert mean.parts["tp2"].points[16]["output_tok_s_gpu"] == 1000.0


def test_a_loss_one_disturbed_run_explains_is_measured_again_and_a_far_one_stands(tmp_path):
    a = _write(tmp_path, "a", {}, {})
    noise = NoiseFloor(spread={k: 0.0001 for k in a.headline()})
    near = compare(_write(tmp_path, "near", {"tp1.con16": 0.988, "tp2.con1": 1.1}, {}), a, noise,
                   **RULE)  # fmt: skip
    assert near.verdict == "LOSS" and near.losses == (
        "tp1.con16.output_tok_s_gpu",
        "tp1.con16.tok_s_user",
    )
    assert remeasure_reason(near, noise, **RULE).startswith("every loss is within 2x its floor")
    far = compare(_write(tmp_path, "far", {"tp1.con16": 0.93}, {}), a, noise, **RULE)
    assert far.verdict == "LOSS" and remeasure_reason(far, noise, **RULE) == ""


# ------------------------------------------------------------------------------ inert members


def test_a_member_without_the_marker_and_with_the_controls_pools_is_inert(tmp_path):
    """land-allreduce-oneshot: its workspace runs at tp2; tp1 builds nothing and binds as before."""
    ctrl = _write(tmp_path, "a", {}, {"tp1": _bind(1334, 1), "tp2": _bind(3461, 2)})
    cand = _write(
        tmp_path, "b", {"tp1.con16": 0.988, "tp2.con1": 1.15},
        {"tp1": _bind(1334, 1), "tp2": f"{MARKER}\n" * 2 + _bind(3450, 2)},
    )  # fmt: skip
    inert = inert_members([cand], [ctrl], marker=MARKER)
    assert list(inert) == ["tp1"] and "recurrent slots [1334] as the control's" in inert["tp1"]
    noise = NoiseFloor(spread={k: 0.0001 for k in ctrl.headline()})
    cmp = compare(cand, ctrl, noise, inert=inert, **RULE)
    assert cmp.verdict == "WIN" and cmp.inert_losses == ("tp1.con16",)
    assert "tp1.con16" not in str(cmp.judged())
    assert remeasure_reason(cmp, noise, **RULE) == "it lost where it changes nothing: tp1.con16"


def test_nothing_is_inert_without_a_marker_that_printed_or_where_the_engine_differs(tmp_path):
    ctrl = _write(tmp_path, "a", {}, {"tp1": _bind(1334, 1), "tp2": _bind(3461, 2)})
    silent = _write(tmp_path, "b", {}, {"tp1": _bind(1334, 1), "tp2": _bind(3461, 2)})
    assert inert_members([silent], [ctrl], marker=MARKER) == {}
    assert inert_members([silent], [ctrl], changed=["tp2"]) == {}  # no marker: no reading
    moved = _write(tmp_path, "c", {}, {"tp1": _bind(1300, 1), "tp2": MARKER + "\n"})
    assert inert_members([moved], [ctrl], marker=MARKER) == {}  # tp1's pool moved
    named = _write(tmp_path, "d", {}, {"tp1": _bind(1334, 1), "tp2": MARKER + "\n"})
    assert inert_members([named], [ctrl], changed=["tp1", "tp2"], marker=MARKER) == {}
    bare = _write(tmp_path, "e", {}, {"tp2": MARKER + "\n"})
    bare_ctrl = _write(tmp_path, "f", {}, {})
    assert inert_members([bare], [bare_ctrl], marker=MARKER) == {}  # no pools, no member list
    assert (
        "not among the members"
        in inert_members([bare], [bare_ctrl], changed=["tp2"], marker=MARKER)["tp1"]
    )


def test_a_round_of_members_outside_the_change_is_judged_nowhere(tmp_path):
    """land-allreduce-oneshot's re-measure: tp1 and dep4 only, its workspace runs at tp2/tp4.

    The marker has nowhere to print; the decision's members and the engines that read as the
    control's make both members inert, so a host's loss at one of them decides nothing.
    """
    logs = {"tp1": _bind(1334, 1), "tp2": _bind(2953, 2)}
    a, a2 = _write(tmp_path, "a", {}, logs), _write(tmp_path, "a2", {}, logs)
    b = _write(tmp_path, "b", {"tp2.con1": 0.976}, logs)
    ctrl, noise = Measurement.mean([a, a2]), NoiseFloor.from_measurements([a, a2])
    assert compare(b, ctrl, noise, **RULE).verdict == "LOSS"
    inert = inert_members([b], [a, a2], changed=["tep4", "tp4"], marker=MARKER)
    assert sorted(inert) == ["tp1", "tp2"]
    assert inert["tp2"] == (
        "not among the members the change runs at (tep4, tp4), none of which this measurement "
        f"holds; `{MARKER}` printed nowhere; recurrent slots [2953, 2953] as the control's"
    )
    cmp = compare(b, ctrl, noise, inert=inert, **RULE)
    assert cmp.verdict == "NEUTRAL" and cmp.inert_losses == ("tp2.con1",)
    # Without a marker the members the decision names say the same.
    assert sorted(inert_members([b], [a, a2], changed=["tp4"])) == ["tp1", "tp2"]


def test_the_members_named_alone_make_no_member_inert_without_a_reading(tmp_path):
    ctrl = _write(tmp_path, "a", {}, {"tp1": _bind(1334, 1), "tp2": _bind(2953, 2)})
    # tp1 bound another pool: the change reached it, whatever the decision says.
    moved = _write(tmp_path, "b", {}, {"tp1": _bind(1300, 1), "tp2": _bind(2953, 2)})
    assert list(inert_members([moved], [ctrl], changed=["tp4"], marker=MARKER)) == ["tp2"]
    # Logs without engine records or pool lines are no reading.
    bare, bare_ctrl = _write(tmp_path, "c", {}, {"tp1": "up\n"}), _write(tmp_path, "d", {}, {})
    assert inert_members([bare], [bare_ctrl], changed=["tp4"], marker=MARKER) == {}
    # A measurement that holds a member the decision names had the marker's chance to print.
    silent = _write(tmp_path, "e", {}, {"tp1": _bind(1334, 1), "tp2": _bind(2953, 2)})
    assert inert_members([silent], [ctrl], changed=["tp2", "tp4"], marker=MARKER) == {}


def test_a_marker_that_printed_in_another_log_counts_as_printed(tmp_path):
    logs = {"tp1": _bind(1334, 1), "tp2": _bind(2953, 2)}
    ctrl, cand = _write(tmp_path, "a", {}, logs), _write(tmp_path, "b", {}, logs)
    gate = {"gate-boot-tp4.log": f"{MARKER} over 4 ranks\n"}
    inert = inert_members([cand], [ctrl], marker=MARKER, marker_logs=gate)
    assert sorted(inert) == ["tp1", "tp2"]
    assert inert["tp1"].startswith(f"`{MARKER}` absent (printed at gate-boot-tp4.log)")
    # tp2 is named: never inert, even with the marker absent from its log.
    assert list(inert_members([cand], [ctrl], changed=["tp2"], marker=MARKER,
                              marker_logs=gate)) == ["tp1"]  # fmt: skip
    # A log without the marker shows nothing.
    assert inert_members([cand], [ctrl], marker=MARKER, marker_logs={"x.log": "up\n"}) == {}


def test_perf_ab_takes_the_members_named_and_the_logs_the_marker_printed_in(tmp_path, capsys):
    from agent_flow.workflows.trtllm_modeling_v2.tune.cli import main, read_marker_logs

    logs = {"tp1": _bind(1334, 1), "tp2": _bind(2953, 2)}
    for label, scale in (("A", {}), ("B", {"tp1.con16": 0.976}), ("A2", {})):
        _write(tmp_path, label, scale, logs)
    ab = ["ab", "--a", str(tmp_path / "A"), "--b", str(tmp_path / "B"), "--a", str(tmp_path / "A2")]
    assert main(ab) == 0
    assert "verdict: LOSS" in capsys.readouterr().out
    assert main([*ab, "--changed-members", "tp4, tep4", "--path-marker", MARKER]) == 0
    out = capsys.readouterr().out
    assert "inert tp1: not among the members the change runs at (tep4, tp4)" in out
    assert "verdict: NEUTRAL" in out and "not judged at tp1, tp2" in out
    gate = tmp_path / "gate.log"
    gate.write_text(f"{MARKER} over 4 ranks\n")
    assert main([*ab, "--path-marker", MARKER, "--marker-log", str(gate)]) == 0
    assert f"inert tp1: `{MARKER}` absent (printed at {gate})" in capsys.readouterr().out
    other = tmp_path / "other.log"
    other.write_text("up\n")
    for paths, marker, why in (
        ([gate], "", "--marker-log needs --path-marker"),
        ([other], MARKER, "is not in it"),
        ([tmp_path / "missing.log"], MARKER, "missing.log"),
    ):
        try:
            read_marker_logs(paths, marker)
        except SystemExit as exc:
            assert why in str(exc)
        else:
            raise AssertionError(f"{paths} was taken")


# ------------------------------------------------------------------------------ an idle node


def _session(path: Path, *, running=(), records=(), receipt_age: float | None = None) -> str:
    for sub in ("pending", "running", "done", "logs", "watchers"):
        (path / sub).mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (path / "watcher.heartbeat").write_text(f"alive={stamp}\n")
    for name in running:
        (path / "running" / f"{name}.sh").write_text("#!/bin/bash\n")
    for i, (age, jobs) in enumerate(records):
        record = {"watcher": f"w{i}", "alive_epoch": time.time() - age,
                  "running": [{"name": n} for n in jobs], "devices": ["0"], "free": []}  # fmt: skip
        (path / "watchers" / f"w{i}.json").write_text(json.dumps(record))
    if receipt_age is not None:
        rc = path / "done" / "other-job.rc"
        rc.write_text("0\n")
        then = time.time() - receipt_age
        os.utime(rc, (then, then))
    return str(path)


def test_a_measurement_starts_only_on_an_idle_node(tmp_path):
    claimed = _session(tmp_path / "claimed", running=["r205-B-x-perf"])
    reported = _session(tmp_path / "reported", records=[(5, ["r205-A-y-perf-3"])])
    stale = _session(tmp_path / "stale", records=[(3600, ["gone"])], receipt_age=600)
    settling = _session(tmp_path / "settling", receipt_age=30)
    idle = _session(tmp_path / "idle", receipt_age=600)
    assert node_busy(claimed) == "running r205-B-x-perf"
    assert node_busy(reported) == "running r205-A-y-perf-3"
    assert node_busy(stale) == "" and node_busy(idle) == ""
    assert node_busy(settling).startswith("a job ended ")
    assert node_busy(settling, settle_s=0) == ""
    runner = SpoolRunner([claimed, reported, stale, settling, idle], prefix="tune")
    assert runner.free_spools() == [stale, idle]
    assert runner.not_idle[settling].startswith("a job ended")
    assert runner.alive(claimed) and not runner.alive(str(tmp_path / "none"))


def test_noise_repeats_go_to_an_idle_node_while_another_job_holds_the_baselines(tmp_path):
    """Busy is not waited for: the repeats run beside the baseline, on whichever node is idle.

    The second repeat finds the baseline's node held by another job and runs on the first
    repeat's: that pair is a same-host floor, the baseline beside them only a cross-host one.
    """
    repo = _repo(tmp_path)
    identity = tmp_path / "identity.yaml"
    identity.write_text(yaml.safe_dump({"modeling_v2": "require"}))
    spools = []
    for name in ("s1", "s2"):
        Path(tmp_path, name).mkdir()
        Path(tmp_path, name, "status.json").write_text(json.dumps({"host": f"host-{name}"}))
        spools.append(str(tmp_path / name))
    task = {"trtllm_repo_path": str(repo), "checkpoint_path": "/ckpt", **_contract(tmp_path),
            "perf": {"identity_configs": [str(identity)], "spools": spools, "env_script": "/e",
                     "workload": {"concurrency": [1, 128]}, "reference": "none"}}  # fmt: skip
    spec = TuneSpec.from_task(task, tmp_path / "ws", target="t/")

    class Shared(FakeRunner):
        """Another job holds the baseline's session for a few polls after the baseline."""

        busy = 0

        def free_spools(self):
            free = super().free_spools()
            if self.busy:
                self.busy -= 1
                return [s for s in free if s != self.spools[0]]
            return free

        def poll(self):
            done = super().poll()
            if any(label == "t-baseline" for label, _, _ in done):
                self.busy = 3
            return done

    runners = []

    def factory(spools, prefix):
        runners.append(Shared(spools, prefix, _effect))
        return runners[-1]

    state: dict = {}
    stage = TuneStage(spec, state, dispatch=lambda *a: "", checkpoint=lambda: None,
                      runner_factory=factory, poll_s=0, log=lambda m: None)  # fmt: skip
    assert stage.ground()
    where = runners[0].where
    assert where["t-baseline"] == spools[0]
    assert where["t-baseline-r1"] == where["t-baseline-r2"] == spools[1]
    assert set(state["noise_spread"]) == {"con1.tok_s_user", "con128.output_tok_s_gpu",
                                          "con128.tok_s_user"}  # fmt: skip
    assert state["noise_samples"] == 2 and set(state["cross_host_spread"]) == set(
        state["noise_spread"]
    )


# ------------------------------------------------------------------------------ the campaign


class PartsRunner(FakeRunner):
    """Measures every member of a job, with the server log each member's engine would print.

    A tree whose ``modeling.py`` says ``RUNS_AT = [...]`` runs its change at those members:
    +15% there, its marker in their logs, and one recurrent slot fewer (its workspace);
    ``SLOW_AT = [...]`` makes it 10% slower at those members' ``con16``. A label starting
    with a key of ``disturbed`` reads 1.5% low at ``tp1.con16`` the first ``disturbed[key]``
    times it is measured (a node still busy with the job before).
    """

    def __init__(self, spools, prefix, disturbed=None):
        super().__init__(spools, prefix, _effect)
        self.disturbed = dict(disturbed or {})
        self.order: list[tuple[str, str]] = []

    def submit(self, spool, label, command, **kw):
        assert not any(sp == spool for sp, _ in self.inflight.values()), "the node is busy"
        self.order.append((Path(spool).name, label))
        return super().submit(spool, label, command, **kw)

    def poll(self):
        done = []
        for name, (spool, label) in list(self.inflight.items()):
            command, tree = self.jobs[label]
            text = Path(tree, "modeling.py").read_text() if tree else ""
            runs_at, slow_at = (
                next((yaml.safe_load(line.split("=", 1)[1]) for line in text.splitlines()
                      if line.startswith(name)), [])
                for name in ("RUNS_AT", "SLOW_AT")
            )  # fmt: skip
            key = next((k for k in self.disturbed if label.startswith(k)), None)
            shaken = key is not None and self.disturbed[key] > 0
            if shaken:
                self.disturbed[key] -= 1
            host = json.loads(Path(spool, "status.json").read_text())["host"]
            for step in command.split("\n", 1)[1].split(" && \\\n"):
                argv = shlex.split(step)
                if "--label" not in argv:
                    continue
                part = argv[argv.index("--label") + 1]
                root = Path(argv[argv.index("--out-root") + 1])
                config = resolve_config(
                    [Path(argv[i + 1]) for i, a in enumerate(argv) if a == "--config"], []
                )
                ranks = int(config.get("tensor_parallel_size", 1))
                gain = 1.15 if part in runs_at else 1.0
                rows = []
                for con in CONS:
                    g = gain * (0.985 if shaken and (part, con) == ("tp1", 16) else 1.0)
                    g *= 0.9 if part in slow_at and con == 16 else 1.0
                    rows.append({"concurrency": con, "tok_s_user": 100.0 * g,
                                 "output_tok_s_gpu": 1000.0 * g, "mean_tpot_ms": 10.0})  # fmt: skip
                d = root / part
                d.mkdir(parents=True, exist_ok=True)
                (d / "summary.json").write_text(json.dumps(rows))
                (d / "meta.json").write_text(json.dumps(
                    {"label": part, "config": config, "hostname": host, "tree": {}}))  # fmt: skip
                log = (MARKER + "\n") * ranks if part in runs_at else ""
                (d / "server.log").write_text(log + _bind(3000 - (part in runs_at), ranks))
            out = root
            (out / PARTS_FILE).write_text(json.dumps({"parts": list(MEMBERS), "constraints": {}}))
            (out / "meta.json").write_text(json.dumps({"label": label, "hostname": host}))
            (out / "summary.json").write_text("[]")
            del self.inflight[name]
            done.append((label, spool, 0))
        return done


def _parts_campaign(tmp_path: Path, perf: FakeRunner, hosts=("node0",)) -> Campaign:
    repo = _repo(tmp_path)
    for name, host in zip(perf.spools, hosts):
        Path(name).mkdir(parents=True, exist_ok=True)
        Path(name, "status.json").write_text(json.dumps({"host": host}))
    parts = []
    for name, config in MEMBERS.items():
        path = tmp_path / f"serving-{name}.yaml"
        path.write_text(yaml.safe_dump({"modeling_v2": "require", **config}))
        parts.append(BenchPart(name=name, isl=128, osl=16, concurrencies=CONS, configs=[path]))
    bench = BenchSpec(bench_py="b", model="m", env_script="e", isl=128, osl=16,
                      concurrencies=CONS, tree=str(repo), parts=tuple(parts))  # fmt: skip
    work = tmp_path / "camp"
    camp = Campaign(workdir=work, repo=repo, bench=bench, base_configs=[], perf_runner=perf,
                    gate_runner=FakeRunner(["g1"], "campg", _effect), keep_gate=_keep_gate(tmp_path),
                    noise=NoiseFloor(), log=perf.log.append, **RULE)  # fmt: skip
    camp.state.best_tree = str(open_candidate(repo, work / "wt", "best", "HEAD"))
    camp.state.best_commit = _git(Path(camp.state.best_tree), "rev-parse", "HEAD")
    return camp


def _candidate(
    camp: Campaign, tmp_path: Path, slug: str, runs_at: list, slow_at: tuple = (), **decision
) -> None:
    wt = open_candidate(camp.repo, camp.workdir / "wt", slug, "HEAD")
    (wt / "modeling.py").write_text(f"RUNS_AT = {runs_at}\nSLOW_AT = {list(slow_at)}\n")
    _git(wt, "commit", "-q", "-am", slug)
    path = tmp_path / f"{slug}.yaml"
    path.write_text(yaml.safe_dump({"decision": "CANDIDATE", "worktree": str(wt),
                                    "commits": [_git(wt, "rev-parse", "HEAD")], **decision}))  # fmt: skip
    camp.submit_candidate(slug, path)


def _run(camp: Campaign) -> list[dict]:
    for _ in range(40):
        if not camp.step():
            break
    return [json.loads(line) for line in camp.ledger.path.read_text().splitlines()]


def _runner(tmp_path, n=1, **kw) -> PartsRunner:
    runner = PartsRunner([str(tmp_path / f"s{i}") for i in range(n)], "camp", **kw)
    runner.log = []
    return runner


def test_a_near_loss_is_measured_again_interleaved_on_its_node_before_it_counts(tmp_path):
    """tp1.con16 read 1.5% low once (the node was still busy): B' and A'' clear it; it is kept."""
    perf = _runner(tmp_path, disturbed={"e2e-collective": 1})
    camp = _parts_campaign(tmp_path, perf)
    _candidate(camp, tmp_path, "collective", ["tp2"])
    records = _run(camp)
    st = camp.state.candidates["collective"]
    assert st.status == "kept", camp.summary()
    kinds = [(r["kind"], r.get("slug")) for r in records]
    assert ("remeasure", "collective") in kinds and ("candidate_e2e", "collective") in kinds
    assert {spool for spool, _ in perf.order} == {"s0"}
    assert [label.split("-")[0] for _, label in perf.order] == ["ctl", "e2e", "ctl", "e2e", "ctl"]
    assert perf.order[3][1].endswith("-r2") and perf.order[4][1].endswith("-s3")
    e2e = next(r for r in records if r["kind"] == "candidate_e2e")
    assert e2e["verdict"] == "WIN" and len(e2e["controls"]) == 3 and len(e2e["samples"]) == 2
    assert e2e["remeasured"].startswith("every loss is within 2x its floor")
    # The keep's two runs are the new best tree's first controls on that node.
    key = f"{camp.state.best_commit[:10]}||node0"
    assert camp.state.samples[key] == [st.e2e_label, *st.samples]


def test_a_near_loss_that_repeats_stands(tmp_path):
    perf = _runner(tmp_path, disturbed={"e2e-collective": 2})
    camp = _parts_campaign(tmp_path, perf)
    _candidate(camp, tmp_path, "collective", ["tp2"])
    _run(camp)
    st = camp.state.candidates["collective"]
    assert st.verdict == "LOSS" and st.status != "kept" and st.remeasure["why"]


def test_a_loss_past_twice_its_floor_is_not_measured_again(tmp_path):
    perf = _runner(tmp_path)
    camp = _parts_campaign(tmp_path, perf)
    _candidate(camp, tmp_path, "slower", ["tp2"], slow_at=["tp1"])
    records = _run(camp)
    assert camp.state.candidates["slower"].verdict == "LOSS"
    assert "remeasure" not in [r["kind"] for r in records]
    assert [label.split("-")[0] for _, label in perf.order] == ["ctl", "e2e", "ctl"]


def test_a_loss_where_the_candidate_changes_nothing_never_reverts_it(tmp_path):
    """The change runs at tp2 only; tp1 reads low twice: reported as the host's noise, kept."""
    perf = _runner(tmp_path, disturbed={"e2e-workspace": 2})
    camp = _parts_campaign(tmp_path, perf)
    _candidate(camp, tmp_path, "workspace", ["tp2"], path_marker=MARKER, changed_members=["tp2"])
    records = _run(camp)
    st = camp.state.candidates["workspace"]
    assert st.status == "kept", camp.summary()
    assert list(st.inert) == ["tp1"] and st.changed_members == ["tp2"]
    noise = [r for r in records if r["kind"] == "host_noise"]
    assert noise and noise[0]["sites"] == ["tp1.con16"] and noise[0]["host"] == "node0"
    e2e = next(r for r in records if r["kind"] == "candidate_e2e")
    assert e2e["verdict"] == "WIN" and e2e["inert"] == ["tp1"]
    assert all(not k.startswith("tp1.") for k in st.deltas)
    assert any("the noise of node0, not held against it" in m for m in perf.log)


def test_the_floor_follows_the_best_tree(tmp_path):
    """Each tree's floor is its own controls' spread, the widest host's per site.

    The floor the campaign was opened with is its first tree's: once the best moves, it is stale.
    """
    perf = _runner(tmp_path, n=2)
    camp = _parts_campaign(tmp_path, perf, hosts=("node0", "node1"))
    ground = NoiseFloor(spread={"tp1.con1.tok_s_user": 0.2})
    camp.noise, camp.noise_commit = ground, camp.state.best_commit
    old = camp.state.best_commit[:10]
    data = camp.data
    for host, labels in (("node0", ("a", "a2")), ("node1", ("b", "b2"))):
        camp.state.samples[f"{old}||{host}"] = list(labels)
    _write(data, "a", {}, {}, "node0")
    _write(data, "a2", {"tp2.con16": 1.02}, {}, "node0")
    _write(data, "b", {}, {}, "node1")
    _write(data, "b2", {"tp1.con16": 1.04}, {}, "node1")
    floor = camp._floor()
    assert floor.of("tp1.con1.tok_s_user") == 0.2  # the ground floor, its tree still the best
    assert round(floor.of("tp1.con16.output_tok_s_gpu"), 4) == round(0.04 / 2.04, 4)
    assert round(floor.of("tp2.con16.tok_s_user"), 4) == round(0.02 / 2.02, 4)
    camp.state.best_commit = "f" * 40  # a keep moved the best tree
    assert camp._floor().spread == {}  # nothing measured on it yet; nothing inherited
    camp.state.samples[f"{'f' * 10}||node0"] = ["c", "c2"]
    _write(data, "c", {}, {}, "node0")
    _write(data, "c2", {"tp1.con1": 1.01}, {}, "node0")
    new = camp._floor()
    assert new.of("tp1.con1.tok_s_user") < 0.01 and new.of("tp1.con16.tok_s_user") == 0.0


def test_a_resumed_campaign_measures_a_second_control_before_it_judges(tmp_path):
    """A state with one control per host (written before pairs) is A; A' comes next."""
    perf = _runner(tmp_path)
    camp = _parts_campaign(tmp_path, perf)
    best = camp.state.best_commit[:10]
    _write(camp.data, "ctl-old", {}, {}, "node0")
    camp.state.controls = {f"{best}||node0": "ctl-old"}
    camp.state.samples = {}
    camp.state.save(camp.state_path)
    state = CampaignState.load(camp.state_path)
    assert state.samples == {f"{best}||node0": ["ctl-old"]}
    camp.state = state
    _candidate(camp, tmp_path, "collective", ["tp1", "tp2"])
    _run(camp)
    assert [label.split("-")[0] for _, label in perf.order] == ["e2e", "ctl"]
    assert perf.order[1][1].endswith("-s2")
    assert camp.state.candidates["collective"].controls == ["ctl-old", perf.order[1][1]]


def test_a_candidate_whose_node_left_before_its_pair_is_measured_again_elsewhere(tmp_path):
    """The session ends after B and before A': B is set aside and measured A B A' elsewhere."""

    class Leaves(PartsRunner):
        left = False

        def poll(self):
            done = super().poll()
            if not self.left and any(label.startswith("e2e-") for label, _, _ in done):
                self.left = True  # a new session took the spool, on another node
                Path(self.spools[0], "status.json").write_text(json.dumps({"host": "node9"}))
            return done

    perf = Leaves([str(tmp_path / "s0"), str(tmp_path / "s1")], "camp")
    perf.log = []
    camp = _parts_campaign(tmp_path, perf, hosts=("node0", "node1"))
    _candidate(camp, tmp_path, "collective", ["tp2"])
    records = _run(camp)
    st = camp.state.candidates["collective"]
    assert "unpaired" in [r["kind"] for r in records]
    assert list(camp.data.glob(f"{st.e2e_label}.unpaired-*"))
    assert st.status == "kept", camp.summary()
    assert [label.split("-")[0] for _, label in perf.order] == ["ctl", "e2e", "ctl", "e2e", "ctl"]


def test_ab_judges_b_against_both_controls_and_says_when_to_measure_again(tmp_path):
    logs = {"tp1": _bind(1334, 1), "tp2": _bind(3461, 2)}
    a = _write(tmp_path, "A", {}, logs)
    a2 = _write(tmp_path, "A2", {"tp2.con1": 1.001}, logs)
    b = _write(tmp_path, "B", {"tp1.con16": 0.985, "tp2.con1": 1.15},
               {"tp1": _bind(1334, 1), "tp2": MARKER + "\n" + _bind(3450, 2)})  # fmt: skip
    out = judge_ab([a, a2], [b], marker=MARKER, **RULE)
    assert "inert tp1: `" in out and "verdict: WIN" in out
    assert "re-measure before it counts (it lost where it changes nothing: tp1.con16)" in out
    plain = judge_ab([a, a2], [b], **RULE)
    assert "verdict: LOSS" in plain and "re-measure before it counts (every loss" in plain
    assert "one control measures no floor" in judge_ab([a], [b], **RULE)


def test_the_settle_is_the_campaigns_and_the_request_asks_for_what_inertness_reads(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.tune.measure import IDLE_SETTLE_S
    from agent_flow.workflows.trtllm_modeling_v2.tune.requests import (
        CampaignContext,
        render_request,
    )

    repo = _repo(tmp_path)
    task = {"trtllm_repo_path": str(repo), "checkpoint_path": "/ckpt", **_contract(tmp_path),
            "perf": {"workload": {"concurrency": [1, 128]}, "spools": ["s1"], "env_script": "/e"}}  # fmt: skip
    spec = TuneSpec.from_task(task, tmp_path / "ws", target="t/")
    stage = TuneStage(spec, {}, dispatch=lambda *a: "", checkpoint=lambda: None, log=lambda m: None)
    assert stage.perf.settle_s == IDLE_SETTLE_S
    spec.campaign["idle_settle_s"] = 30
    stage = TuneStage(spec, {}, dispatch=lambda *a: "", checkpoint=lambda: None, log=lambda m: None)
    assert stage.perf.settle_s == 30.0
    ctx = CampaignContext(repo=repo, package=tmp_path, campaign_dir=tmp_path / "c", perf_root=tmp_path,
                          target_dir="t/", target_tests="tests/t/", model="m", dev_spools=[],
                          env_script=Path("/e"), embargo=[], workload="w", standings_table="")  # fmt: skip
    text = render_request(ctx, slug="x", worktree=tmp_path / "wt", base="HEAD", hypothesis="h",
                          deliverables="d", job_prefix="x").read_text()  # fmt: skip
    assert "changed_members: [<perf config>, ...]" in text and 'path_marker: "<text>"' in text
