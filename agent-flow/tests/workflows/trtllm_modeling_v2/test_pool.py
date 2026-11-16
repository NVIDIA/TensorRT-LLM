# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The pool's controller and its Slurm launcher, against a fake Slurm on ``PATH``.

``sbatch``, ``squeue``, ``sacct`` and ``scancel`` are small scripts over one JSON file, so every
case runs the real command lines and reads the real answers: a renewal, a reply that was lost,
a refusal, a lost holder, a holder no work needs, and an end that touches nothing the flow did
not record. Every Slurm value here is a placeholder.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.infra.compute import pool as pool_mod
from agent_flow.workflows.trtllm_modeling_v2.infra.compute import sessions as sessions_mod
from agent_flow.workflows.trtllm_modeling_v2.infra.compute import slurm as slurm_mod
from agent_flow.workflows.trtllm_modeling_v2.infra.compute import spoolctl
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.scheduler import SpoolPaths
from agent_flow.workflows.trtllm_modeling_v2.infra.files import read_json, write_json_atomic
from agent_flow.workflows.trtllm_modeling_v2.infra.monitor import status
from agent_flow.workflows.trtllm_modeling_v2.spec import task_schema

_FAKE = r'''#!{python}
"""A fake Slurm command over $FAKE_SLURM/jobs.json: sbatch, squeue, sacct or scancel by name."""
import json, os, re, sys, time
from pathlib import Path

root = Path(os.environ["FAKE_SLURM"])
tool, args = Path(sys.argv[0]).name, sys.argv[1:]
store = root / "jobs.json"
jobs = json.loads(store.read_text()) if store.exists() else []
with (root / "calls.log").open("a") as log:
    log.write(tool + " " + " ".join(args) + "\n")


def save():
    store.write_text(json.dumps(jobs))


def flag(name):
    path = root / name
    if path.exists():
        text = path.read_text().strip()
        return text or "yes"
    return ""


def option(prefix):
    for index, arg in enumerate(args):
        if arg.startswith(prefix + "="):
            return arg.split("=", 1)[1]
        if arg == prefix and index + 1 < len(args):
            return args[index + 1]
    return ""


def row(job):
    start, end = job.get("start") or "N/A", job.get("end") or "N/A"
    fields = (job["id"], job["name"], job["state"], job.get("reason", ""), start, end)
    return "|".join(str(v) for v in fields)


if tool == "sbatch":
    if flag("refuse"):
        sys.exit("sbatch: error: " + flag("refuse"))
    text = Path(args[-1]).read_text()
    job = {{
        "id": str(1000 + len(jobs) + 1),
        "name": re.search(r"--job-name=(\S+)", text).group(1),
        "nodes": int(re.search(r"--nodes=(\d+)", text).group(1)),
        "state": "PENDING",
        "reason": "Priority",
        "script": args[-1],
    }}
    lost = flag("lose_reply")
    if lost != "dropped":
        jobs.append(job)
        save()
    if lost:
        (root / "lose_reply").unlink()
        sys.exit("sbatch: error: Batch job submission failed: Socket timed out on send/recv operation")
    print(job["id"])
elif tool == "squeue":
    if flag("squeue_fail"):
        sys.exit("squeue: error: Unable to contact slurm controller (connect failure)")
    wanted = option("-j")
    if wanted:
        known = [j for j in jobs if j["id"] == wanted and not j.get("forgotten")]
        if not known:
            sys.exit("slurm_load_jobs error: Invalid job id specified")
        print(known[0]["id"])
    else:
        names = option("--name").split(",")
        for job in jobs:
            if job["name"] in names and not (job.get("forgotten") or job.get("unlisted")):
                print(row(job))
elif tool == "sacct":
    if flag("sacct_fail"):
        sys.exit("sacct: error: accounting is not configured")
    if not flag("sacct_empty"):
        wanted, name = option("-j"), option("--name")
        for job in jobs:
            if (wanted and job["id"] == wanted) or (name and job["name"] == name):
                times = [str(job.get("end") or "Unknown"), str(job.get("start") or "None")]
                print("|".join([job["id"], job["name"], job["state"], *times]))
elif tool == "scancel":
    pending_only = "--state=PENDING" in args
    for job in jobs:
        if job["id"] in args and job["state"] in ("PENDING", "RUNNING"):
            if pending_only and job["state"] != "PENDING":
                continue
            job["state"] = "CANCELLED by 1000"
    save()
'''

SLURM = {
    "account": "ACCOUNT",
    "partition": "PARTITION",
    "qos": "QOS",
    "time": "00:30:00",
    "gpus_per_node": 4,
    "container_image": "/images/IMAGE.sqsh",
    "container_mounts": ["/shared:/shared"],
    "sbatch_args": ["--exclusive", "--mem=0"],
    "lead": "5m",
}


class Clock:
    def __init__(self) -> None:
        self.now = time.time()

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeSlurm:
    """The cluster the fake commands answer for, and the hand that changes it."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def jobs(self) -> list[dict]:
        path = self.root / "jobs.json"
        return json.loads(path.read_text()) if path.exists() else []

    def job(self, job_id: str) -> dict:
        return next(job for job in self.jobs if job["id"] == str(job_id))

    def change(self, job_id: str, **fields) -> None:
        jobs = self.jobs
        for job in jobs:
            if job["id"] == str(job_id):
                job.update(fields)
        (self.root / "jobs.json").write_text(json.dumps(jobs))

    def add(self, name: str, state: str = "RUNNING", nodes: int = 1) -> str:
        jobs = self.jobs
        job_id = str(1000 + len(jobs) + 1)
        jobs.append({"id": job_id, "name": name, "state": state, "nodes": nodes})
        (self.root / "jobs.json").write_text(json.dumps(jobs))
        return job_id

    def set(self, name: str, text: str = "") -> None:
        (self.root / name).write_text(text)

    def clear(self, name: str) -> None:
        (self.root / name).unlink(missing_ok=True)

    def calls(self, tool: str) -> list[str]:
        log = self.root / "calls.log"
        lines = log.read_text().splitlines() if log.exists() else []
        return [line[len(tool) + 1 :] for line in lines if line.split(" ", 1)[0] == tool]


@pytest.fixture
def fake(tmp_path, monkeypatch) -> FakeSlurm:
    bindir, root = tmp_path / "bin", tmp_path / "fake-slurm"
    bindir.mkdir()
    root.mkdir()
    for tool in ("sbatch", "squeue", "sacct", "scancel"):
        script = bindir / tool
        script.write_text(_FAKE.format(python=sys.executable))
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SLURM", str(root))
    return FakeSlurm(root)


def _pool(tmp_path, clock, **slurm) -> pool_mod.Pool:
    spool = tmp_path / "spool"
    cfg = {**SLURM, **slurm}
    environment = slurm_mod.SlurmEnvironment(
        cfg,
        spool=spool,
        run="RUN",
        watcher_entry=spool / pool_mod.WATCHER_ENTRY,
        lead_entry=spool / pool_mod.LEAD_ENTRY,
    )
    return pool_mod.Pool(spool, environment, clock=clock)


def _beat(pool, clock, holder: str, *, job: str, ends_in: float, nodes: int = 1, running=(), n=1):
    """Heart-beats of ``n`` watchers of ``holder``, as its nodes would write them."""
    for index in range(n):
        write_json_atomic(
            pool.paths.watchers / f"{holder}-node{index}.json",
            {
                "watcher": f"{holder}-node{index}",
                "host": f"{holder}-node{index}",
                "holder": holder,
                "nodes": nodes,
                "slurm_job": job,
                "alive_epoch": clock.now,
                "ends_at_epoch": clock.now + ends_in,
                "running": [{"name": name} for name in running],
            },
        )


def _run(fake, pool, clock, record: dict, *, ends_in: float = 1800.0, running=()) -> None:
    """The holder of ``record`` leaves the queue: it runs, and its watchers heart-beat."""
    fake.change(
        record["job_id"], state="RUNNING", start=int(clock.now), end=int(clock.now + ends_in)
    )
    _beat(
        pool, clock, record["name"], job=record["job_id"], ends_in=ends_in,
        nodes=record["nodes"], running=running, n=record["nodes"],
    )  # fmt: skip


def _mounts(tmp_path) -> list[str]:
    """Mounts that cover a test's own paths and the flow's checkout, the same inside and out."""
    flow = str(Path(pool_mod.__file__).resolve().parents[5])
    return [f"{tmp_path}:{tmp_path}", f"{flow}:{flow}"]


def _kinds(events) -> list[str]:
    return [event.kind for event in events]


def _open(pool) -> list[dict]:
    return [record for record in pool.records() if not record.get("closed")]


# ------------------------------------------------------------------ slurm.py


def test_a_missing_slurm_key_prints_one_line_that_says_how_to_find_its_value():
    errors: list[str] = []
    slurm_mod.validate({}, errors)
    assert all("\n" not in line and " is missing: " in line for line in errors)
    assert [line.split(" ", 1)[0] for line in errors] == [
        f"environment.slurm.{key}" for key in ("account", "partition", "time", "gpus_per_node")
    ]
    # A complete block is quiet; a container needs its mounts.
    errors = []
    slurm_mod.validate(SLURM, errors)
    assert errors == []
    slurm_mod.validate({**SLURM, "container_mounts": None, "time": "soon"}, errors)
    assert [line.split(" ", 1)[0].strip("'") for line in errors] == [
        "environment.slurm.time",
        "environment.slurm.container_mounts",
    ]
    # The cap on holders is gone: a spec that still states it is told what limits a run.
    errors = []
    slurm_mod.validate({**SLURM, "max_holders": 2}, errors)
    assert errors == [
        "'environment.slurm.max_holders' is no key any more: what limits a run's holders is "
        "the node-hours it may ask for (`environment.slurm.max_node_hours`)"
    ]
    # What the outer session may change is not a key: any Slurm value changes through `resolve`.
    errors = []
    slurm_mod.validate({**SLURM, "alternatives": [{"qos": "OTHER"}]}, errors)
    assert errors and "alternatives" in errors[0]
    for text, seconds in (("30", 1800), ("01:30", 90), ("02:00:00", 7200), ("1-12", 129600)):
        assert slurm_mod.slurm_time_seconds(text) == seconds


def test_a_flag_the_flow_owns_is_refused_in_sbatch_args_and_srun_args():
    block = {
        **SLURM,
        "sbatch_args": ["--exclusive", "--time=08:00:00", "-N2", "--begin=now+1hour", "--requeue"],
        "srun_args": ["--container-writable", "--mpi=pmi2", "--overlap", "--container-image=/x"],
    }
    errors: list[str] = []
    slurm_mod.validate(block, errors)
    assert errors == [
        f"environment.slurm.{key} sets {flag}, which the flow sets itself"
        for key, flag in (
            ("sbatch_args", "--time"),
            ("sbatch_args", "-N"),
            ("sbatch_args", "--begin"),
            # A requeued holder would hold its nodes again under a record that counts once.
            ("sbatch_args", "--requeue"),
            ("srun_args", "--mpi"),
            ("srun_args", "--overlap"),
            ("srun_args", "--container-image"),
        )
    ]
    # Site arguments that are nobody's pass through: exclusivity, memory, a comment.
    assert slurm_mod.owned_flag_errors(
        {"sbatch_args": ["--exclusive", "--mem=0", "--comment=x"], "srun_args": ["--no-container-mount-home"]},
        "environment.slurm",
    ) == []  # fmt: skip


def test_a_batch_host_python_too_old_for_the_lead_is_said_in_one_line(tmp_path):
    # An interpreter that answers as an old one does: the version test fails, -V names it.
    old = tmp_path / "python-old"
    old.write_text('#!/bin/sh\n[ "$1" = -V ] && { echo "Python 3.6.8"; exit 0; }\nexit 1\n')
    old.chmod(0o755)
    entry = tmp_path / "lead.sh"
    entry.write_text(
        sessions_mod.render_entry(
            {"environment": {}},
            spool=str(tmp_path / "s"),
            session="pool",
            lead=True,
            host_python=str(old),
        )
    )
    done = subprocess.run(["bash", str(entry)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 127
    (line,) = done.stderr.strip().splitlines()
    assert line.startswith("error: the holder's lead needs Python 3.8 or newer on ")
    assert line.endswith(f"{old} is Python 3.6.8: name a newer one (environment.slurm.host_python)")
    # This interpreter passes the same test, so its lead would start.
    check = "import sys; sys.exit(sys.version_info < (3, 8))"
    assert f"if ! python3 -c '{check}' 2>/dev/null; then" in sessions_mod.render_entry(
        {"environment": {}}, spool="/s", session="pool", lead=True
    )
    assert subprocess.run([sys.executable, "-c", check]).returncode == 0


def test_the_holder_script_carries_every_mount_on_the_watcher_step_and_the_lead_on_the_batch_host():
    script = slurm_mod.render_holder(
        SLURM,
        run="RUN",
        name="mv2flow-RUN-pool-3",
        nodes=2,
        spool="/shared/run/spool",
        watcher_entry="/shared/run/spool/watcher_entry.sh",
        lead_entry="/shared/run/spool/lead_entry.sh",
    )
    lines = script.splitlines()
    assert lines[0] == "#!/bin/bash"
    assert [line[8:] for line in lines if line.startswith("#SBATCH ")] == [
        "--job-name=mv2flow-RUN-pool-3",
        "--account=ACCOUNT",
        "--partition=PARTITION",
        "--qos=QOS",
        "--time=00:30:00",
        "--nodes=2",
        "--gpus-per-node=4",
        "--no-requeue",
        "--chdir=/shared/run/spool",
        "--output=/shared/run/spool/slurm-%j.out",
        "--exclusive",
        "--mem=0",
    ]
    assert "--begin" not in script and "--dependency" not in script
    assert lines[-3] == "export MV2FLOW_HOLDER=mv2flow-RUN-pool-3"
    # One watcher per node, in the named container, which keeps the mounts of this step.
    assert lines[-2] == (
        "srun --mpi=none --nodes=2 --ntasks=2 --ntasks-per-node=1 "
        "--container-image=/images/IMAGE.sqsh --container-mounts=/shared:/shared "
        "--container-name=mv2flow-RUN bash /shared/run/spool/watcher_entry.sh &"
    )
    assert lines[-1] == "exec bash /shared/run/spool/lead_entry.sh"


def test_a_step_overlaps_the_watcher_step_and_sets_everything_inside_the_launched_shell():
    hosts = '"$MV2FLOW_JOB_HOSTS"'
    world = slurm_mod.step_line(
        SLURM, run="RUN", ranks=8, nodes=2, hosts=hosts, command="bash /shared/job/server.sh"
    )
    assert world == (
        "srun --overlap --mpi=pmix --kill-on-bad-exit=1 --nodes=2 --ntasks=8 --ntasks-per-node=4 "
        '--nodelist="$MV2FLOW_JOB_HOSTS" --container-image=/images/IMAGE.sqsh '
        "--container-name=mv2flow-RUN bash /shared/job/server.sh"
    )
    assert "--container-mounts" not in world, "attaching to the named container ignores mounts"
    one = slurm_mod.step_line(SLURM, run="RUN", ranks=1, nodes=1, hosts="nodeA", command="true")
    assert "--mpi=none" in one and "--mpi=pmix" not in one and "--overlap" in one
    with pytest.raises(ValueError, match="6 ranks do not fill 4 nodes evenly"):
        slurm_mod.step_line(SLURM, run="RUN", ranks=6, nodes=4, hosts="h", command="true")
    bare = slurm_mod.step_line(
        {"srun_args": ["--cpu-bind=none"]}, run="RUN", ranks=2, nodes=1, hosts="h", command="x"
    )
    assert bare.endswith("--nodelist=h --cpu-bind=none x") and "container" not in bare
    inner = slurm_mod.inner_script(
        "trtllm-llmapi-launch trtllm-serve /ckpt",
        python="/opt/venv/bin/python",
        tree="/shared/tree",
        pythonpath=["/shared/hook"],
        env={"HF_HOME": "/shared/hf"},
        role="ctx0",
        devices=[0, 1],
    )
    assert inner.splitlines()[2:] == [
        'export PATH=/opt/venv/bin:"$PATH"',
        'export PYTHONPATH=/shared/tree:/shared/hook"${PYTHONPATH:+:$PYTHONPATH}"',
        "export HF_HOME=/shared/hf",
        "export MV2FLOW_ROLE=ctx0",
        "export CUDA_VISIBLE_DEVICES=0,1",
        "cd /shared/tree || exit 1",
        "exec trtllm-llmapi-launch trtllm-serve /ckpt",
    ]


# ------------------------------------------------------------------- pool.py


def test_a_holder_is_renewed_late_and_overlaps_the_one_it_follows(tmp_path, fake):
    """One record per submission, written before sbatch; a successor only when it is needed."""
    clock = Clock()
    pool = _pool(tmp_path, clock, time="01:00:00")
    demand = {1: 600.0}
    assert pool.start(demand) == []
    (first,) = pool.records()
    assert first["name"] == "mv2flow-RUN-pool-1" and first["job_id"] == fake.jobs[0]["id"]
    assert fake.jobs[0]["name"] == first["name"], "the record's name is the job's"
    script = pool.paths.holders / "mv2flow-RUN-pool-1.sbatch"
    # What the flow owns of the request is on the command line, where it wins over the
    # caller's environment; the script carries the same lines for whoever reads it.
    (call,) = fake.calls("sbatch")
    assert script.is_file() and call.endswith(f" {script}")
    assert call.startswith("--parsable --job-name=mv2flow-RUN-pool-1 --account=ACCOUNT ")
    assert "--time=01:00:00 --nodes=1 --gpus-per-node=4 --no-requeue" in call
    # Pending: nothing more is asked for, however often the pass runs.
    for _ in range(3):
        assert pool.reconcile(demand) == []
    assert len(fake.jobs) == 1
    _run(fake, pool, clock, first)
    assert pool.reconcile(demand) == [] and len(fake.jobs) == 1
    # It fits the longest wall, the margin and the lead (600 s + 120 s + 300 s = 17 minutes):
    # no successor yet. Once less than that is left, one is asked for.
    clock.advance(1800 - 1030)
    _beat(pool, clock, first["name"], job=first["job_id"], ends_in=1030)
    assert pool.reconcile(demand) == [] and len(fake.jobs) == 1
    clock.advance(20)
    _beat(pool, clock, first["name"], job=first["job_id"], ends_in=1010)
    assert pool.reconcile(demand) == []
    second = pool.records()[1]
    assert second["name"] == "mv2flow-RUN-pool-2" and len(fake.jobs) == 2
    assert "--begin" not in " ".join(fake.calls("sbatch"))
    # Both stand on the one spool; the first is not cancelled, it retires by itself.
    _run(fake, pool, clock, second)
    assert pool.reconcile(demand) == [] and fake.calls("scancel") == []
    fake.change(first["job_id"], state="COMPLETED")
    assert _kinds(pool.reconcile(demand)) == [], "a holder that retired is not a lost one"
    assert [r["name"] for r in _open(pool)] == [second["name"]]
    snapshot = read_json(pool.paths.pool)
    assert [h["name"] for h in snapshot["holders"]] == [second["name"]]
    # No work left of any shape: nothing is renewed.
    clock.advance(1700)
    _beat(pool, clock, second["name"], job=second["job_id"], ends_in=100)
    assert pool.reconcile({}) == [] and len(fake.jobs) == 2


def test_a_wall_near_a_holders_whole_time_asks_for_no_new_holder_every_few_minutes(tmp_path, fake):
    """A run's last twenty minutes on a cluster: three one-node holders in a row for one job."""
    clock = Clock()
    pool = _pool(tmp_path, clock, time="04:00:00", lead="30m")
    demand = {1: 13380.0}  # final tests: a wall a holder fits only in its first ten minutes
    assert pool.start(demand) == []
    (first,) = pool.records()
    _run(fake, pool, clock, first, ends_in=14400)
    for left in (13800, 10800, 7200):
        clock.advance(14400 - left - (clock.now - first["written_at"]))
        _beat(pool, clock, first["name"], job=first["job_id"], ends_in=left)
        assert pool.reconcile(demand) == [] and len(fake.jobs) == 1, "not in its first half"
    clock.advance(300)
    _beat(pool, clock, first["name"], job=first["job_id"], ends_in=6900)
    assert pool.reconcile(demand) == [] and len(fake.jobs) == 2, "its successor, once"


def test_work_counts_on_a_holder_of_its_own_width(tmp_path, fake):
    clock = Clock()
    pool = _pool(tmp_path, clock)
    pool.start({2: 600.0})
    (wide,) = pool.records()
    _run(fake, pool, clock, wide)
    # One-node work beside the two-node job gets a holder of its own: the wide one would take
    # a one-node job while it is there, and it is given back when its own work is over.
    assert pool.reconcile({1: 600.0, 2: 600.0}) == []
    assert [record["nodes"] for record in pool.records()] == [2, 1]


def test_only_slurms_own_refusal_closes_a_submission_and_the_callers_environment_owns_nothing():
    """An answer that does not say whether the job exists is unknown, whatever its exit code."""
    import subprocess

    def answer(code, err="", out=""):
        return lambda argv, **kwargs: subprocess.CompletedProcess(argv, code, out, err)

    refusals = (
        "sbatch: error: Batch job submission failed: Invalid account or account/partition "
        "combination specified",
        "sbatch: error: Batch job submission failed: Invalid qos specification",
        "sbatch: error: QOSMaxSubmitJobPerUserLimit",
        "sbatch: error: Batch job submission failed: Job violates accounting/QOS policy "
        "(job submit limit, user's size and/or time limits)",
        "sbatch: error: Batch job submission failed: Requested node configuration is not available",
        "sbatch: error: Unable to open file /no/such/script",
        "sbatch: error: Batch job submission failed: Unable to contact slurm controller (connect failure)",
    )
    for text in refusals:
        reply = slurm_mod.sbatch("x.sbatch", run=answer(1, err=text))
        assert (reply.outcome, reply.text) == ("refused", text)
    unknowns = (
        "sbatch: error: Batch job submission failed: Socket timed out on send/recv operation",
        "sbatch: error: Unexpected response from controller",
        "sbatch: error: Batch job submission failed: Zero Bytes were transmitted or received",
        "",
    )
    for text in unknowns:
        assert slurm_mod.sbatch("x.sbatch", run=answer(1, err=text)).outcome == "unknown"
    assert (
        slurm_mod.sbatch("x.sbatch", run=answer(0, out="Submitted batch job 7")).outcome
        == "unknown"
    )
    accepted = slurm_mod.sbatch("x.sbatch", run=answer(0, out="1234;cluster\n"))
    assert (accepted.outcome, accepted.job_id) == ("accepted", "1234")
    # The caller's environment sets nothing the flow owns, in a request or in a question.
    env = slurm_mod.command_environment(
        {"SBATCH_JOB_NAME": "x", "SBATCH_TIMELIMIT": "9", "SBATCH_ARRAY_INX": "0-9", "SBATCH_WAIT": "1",
         "SQUEUE_FORMAT": "%i", "SACCT_FORMAT": "JobID", "SBATCH_RESERVATION": "theirs", "PATH": "/bin"}
    )  # fmt: skip
    assert env == {"SBATCH_RESERVATION": "theirs", "PATH": "/bin", "SLURM_TIME_FORMAT": "%s"}


def test_a_lost_reply_never_causes_a_second_allocation(tmp_path, fake):
    """Accepted, and the answer never arrived: the record stays, the listing resolves it."""
    clock = Clock()
    pool = _pool(tmp_path, clock)
    fake.set("lose_reply")
    (event,) = pool.start({1: 600.0})
    assert event.kind == "POOL_SUBMIT_FAILED" and "Socket timed out" in event.detail
    (record,) = pool.records()
    assert "job_id" not in record and record["reply"]["outcome"] == "unknown"
    # The next pass finds the job by the record's name: its id is the authority from then on.
    assert pool.reconcile({1: 600.0}) == []
    (record,) = pool.records()
    assert record["job_id"] == fake.jobs[0]["id"] and len(fake.jobs) == 1
    assert len(fake.calls("sbatch")) == 1


def test_an_unknown_submission_is_not_asked_for_again_until_somebody_looked(tmp_path, fake):
    """Nothing lists it and nothing refused it: never resolved by its age, never retried."""
    clock = Clock()
    pool = _pool(tmp_path, clock)
    fake.set("lose_reply", "dropped")
    assert _kinds(pool.start({1: 600.0})) == ["POOL_SUBMIT_FAILED"]
    for _ in range(3):
        clock.advance(6 * 3600)
        (event,) = pool.reconcile({1: 600.0})
        assert event.kind == "POOL_SUBMIT_FAILED"
        assert (
            "no other holder of its shape is asked for until" in event.detail
            and "`resolve pool-retry pool --close-unknown 1`" in event.detail
        )
    assert len(fake.calls("sbatch")) == 1 and fake.jobs == []
    # An empty accounting answer proves nothing either.
    fake.set("sacct_empty")
    assert _kinds(pool.reconcile({1: 600.0})) == ["POOL_SUBMIT_FAILED"]
    fake.clear("sacct_empty")
    # Closed after a look, by whoever looked: that pass asks for the shape again.
    pool.close(1, "squeue and sacct show no job of that name")
    assert pool.reconcile({1: 600.0}) == []
    assert len(fake.calls("sbatch")) == 2 and len(fake.jobs) == 1
    # A record whose job is listed is not unknown, and is not closed by hand.
    with pytest.raises(pool_mod.PoolError, match="is listed as PENDING"):
        pool.close(2, "impatience")


def test_a_refusal_stops_the_start_with_slurms_line_and_is_tried_again_every_two_minutes(
    tmp_path, fake
):
    clock = Clock()
    pool = _pool(tmp_path, clock)
    refusal = "QOSMaxSubmitJobPerUserLimit"
    fake.set("refuse", refusal)
    with pytest.raises(pool_mod.PoolError, match=refusal):
        pool.start({1: 600.0})
    (record,) = pool.records()
    assert record["closed"]["state"] == "refused" and fake.jobs == []
    # Later in a run it is an event, with Slurm's own text; the retry waits two minutes.
    clock.advance(60)
    (event,) = pool.reconcile({1: 600.0})
    assert event.kind == "POOL_SUBMIT_FAILED" and refusal in event.detail
    assert len(fake.calls("sbatch")) == 1
    clock.advance(61)
    assert _kinds(pool.reconcile({1: 600.0})) == ["POOL_SUBMIT_FAILED"]
    assert len(fake.calls("sbatch")) == 2
    fake.clear("refuse")
    clock.advance(121)
    assert pool.reconcile({1: 600.0}) == []
    assert len(fake.jobs) == 1 and [r["seq"] for r in _open(pool)] == [3]
    assert "POOL_SUBMIT_FAILED" in (pool.paths.root / "pool.log").read_text()


def test_a_holder_that_left_before_its_end_is_said_and_replaced(tmp_path, fake):
    clock = Clock()
    pool = _pool(tmp_path, clock)
    demand = {1: 600.0}
    pool.start(demand)
    (first,) = pool.records()
    _run(fake, pool, clock, first)
    assert pool.reconcile(demand) == []
    log = pool.paths.root / "pool.log"

    def said_lost(name: str) -> list[str]:
        return [line for line in log.read_text().splitlines() if f"holder {name} " in line
                and "left before its end" in line]  # fmt: skip

    # Cancelled by hand, or lost with its node: not the flow's own end. It asks nothing of
    # whoever watches: the same pass asked for its replacement, and the log says it once.
    fake.change(first["job_id"], state="CANCELLED by 4242")
    assert pool.reconcile(demand) == []
    assert len(fake.jobs) == 2 and [r["seq"] for r in _open(pool)] == [2]
    (line,) = said_lost(first["name"])
    assert f"(job {first['job_id']}, 1 node(s)) left before its end: CANCELLED by 4242" in line
    assert line.endswith("its jobs are re-run once or closed by the scheduler")
    assert pool.records()[0]["lost"]["state"] == "CANCELLED by 4242"
    second = pool.records()[1]
    assert pool.reconcile(demand) == [] and len(said_lost(first["name"])) == 1
    _run(fake, pool, clock, second)
    assert pool.reconcile(demand) == []
    # It runs and none of its watchers ever heart-beat: it holds nodes and serves nothing.
    for path in pool.paths.watchers.glob(f"{second['name']}-*.json"):
        path.unlink()
    assert pool.reconcile(demand) == [], "a watcher step takes a while to come up"
    clock.advance(pool_mod.START_SLACK_S + 1)
    (event,) = pool.reconcile(demand)
    assert event.kind == "POOL_DEVICE_HELD" and event.key == second["name"]
    assert "it holds its nodes and serves no job" in event.detail
    assert f"{pool.paths.root}/slurm-{second['job_id']}.out" in event.detail, "where the cause is"
    _beat(pool, clock, second["name"], job=second["job_id"], ends_in=800)
    assert pool.reconcile(demand) == []
    # Requeued under its jobs: it ran, and is pending again. Said once, in the log.
    fake.change(second["job_id"], state="PENDING", reason="BeginTime")
    assert pool.reconcile(demand) == [] and pool.reconcile(demand) == []
    (line,) = said_lost(second["name"])
    assert "left before its end: PENDING (BeginTime)" in line
    assert len(fake.jobs) == 2, "a requeued holder is the same job: no other is asked for"


def test_a_holder_that_keeps_leaving_early_is_asked_for_no_faster_than_a_refused_one(
    tmp_path, fake
):
    clock = Clock()
    pool = _pool(tmp_path, clock)
    demand = {1: 600.0}
    pool.start(demand)
    # The first early end is replaced in the same pass: a node that went down.
    fake.change(pool.records()[0]["job_id"], state="FAILED")
    assert pool.reconcile(demand) == [] and len(fake.jobs) == 2
    # The second in a row is a request that keeps failing, in a second each time: it is said,
    # and asked for again only after the pause a refusal gets.
    second = pool.records()[1]
    fake.change(second["job_id"], state="FAILED")
    (event,) = pool.reconcile(demand)
    assert event.kind == "POOL_SUBMIT_FAILED" and len(fake.jobs) == 2
    assert f"holder {second['name']} (1 node(s)) left before its end (FAILED), as the one" in (
        event.detail
    )
    assert (
        f"job {second['job_id']}, its output is in {pool.paths.root}; tried again" in event.detail
    )
    assert pool.reconcile(demand) == [event] and len(fake.jobs) == 2
    clock.advance(pool_mod.REFUSAL_RETRY_S + 1)
    assert pool.reconcile(demand) == [] and len(fake.jobs) == 3


def test_a_holder_that_no_work_of_its_width_needs_is_given_back(tmp_path, fake):
    """Whole nodes for a session, then one-node work only: they are not kept for it."""
    clock = Clock()
    pool = _pool(tmp_path, clock)
    narrow, both = {1: 600.0}, {1: 600.0, 2: 900.0}
    pool.start(narrow)
    (one,) = pool.records()
    _run(fake, pool, clock, one)
    # A session of two nodes is in the spool: its holder is asked for, and comes in 40 s.
    assert pool.reconcile(both) == [] and len(fake.jobs) == 2
    two = pool.records()[1]
    clock.advance(40)
    _run(fake, pool, clock, two)
    assert two["nodes"] == 2 and pool.reconcile(both) == []

    def later(seconds: float) -> list:
        clock.advance(seconds)
        for record in (one, two):
            _beat(pool, clock, record["name"], job=record["job_id"], ends_in=1000,
                  nodes=record["nodes"], n=record["nodes"])  # fmt: skip
        return pool.reconcile(narrow)

    # The session is over and one-node work goes on. The two nodes are kept as long as
    # getting them again would take (their 40 s in the queue and a start), and no longer.
    assert later(15) == [] and later(pool_mod.START_SLACK_S + 39) == []
    assert not any(record.get("stop_requested") for record in pool.records())
    assert later(2) == []
    one, two = pool.records()
    assert two.get("stop_requested") and not one.get("stop_requested"), "what work needs stays"
    said = (pool.paths.root / "pool.log").read_text()
    assert (
        f"holder {two['name']} asked to end: no work of 2 node(s) has been ready or running "
        "for 5m; it was kept as long as it took to get (40s) and a start"
    ) in said
    # It runs nothing, so it is cancelled in that pass; a session queued again asks again.
    assert fake.job(two["job_id"])["state"].startswith("CANCELLED")
    assert pool.reconcile(both) == [] and len(fake.jobs) == 3 and pool.records()[2]["nodes"] == 2


def test_a_holder_whose_watchers_left_for_its_successor_is_not_said_to_be_held(tmp_path, fake):
    """The hand-over: the old holder's watchers leave first, its job moments later."""
    clock = Clock()
    pool = _pool(tmp_path, clock)
    demand = {1: 600.0}
    pool.start(demand)
    (first,) = pool.records()
    _run(fake, pool, clock, first)
    assert pool.reconcile(demand) == []
    clock.advance(1200)
    _beat(pool, clock, first["name"], job=first["job_id"], ends_in=600)
    assert pool.reconcile(demand) == [] and len(pool.records()) == 2, "renewed"
    second = pool.records()[1]
    _run(fake, pool, clock, second)
    # The first one's watchers retire for the second and are gone; Slurm still lists its job.
    for path in pool.paths.watchers.glob(f"{first['name']}-*.json"):
        path.unlink()
    assert pool.reconcile(demand) == [], "running for twenty minutes, alone for none"
    clock.advance(20)
    _beat(pool, clock, second["name"], job=second["job_id"], ends_in=1780)
    fake.change(first["job_id"], state="COMPLETED")
    assert pool.reconcile(demand) == [] and "lost" not in pool.records()[0]
    # Nor is one the pool itself asked to end, however long its job takes to go; one that
    # nobody asked and nobody watches for fifteen minutes is.
    alone = {"seq": 9, "name": "h", "nodes": 1, "unwatched_since": clock.now - 1000}
    held = pool_mod.Holder(dict(alone), "running")
    asked = pool_mod.Holder({**alone, "stop_requested": clock.now - 1000}, "running")
    assert _kinds(pool._standing([held, asked], clock.now)) == ["POOL_DEVICE_HELD"]


def _ready(pool, name: str, *, wall: int, gpus: int = 1) -> None:
    """A job in the spool that no watcher has started yet."""
    header = {"gpus": gpus, "wall": wall, "owner": "test"}
    script = spoolctl.render_job("true", header=header)
    spoolctl.queue_job(pool.paths.root, name, script, owner="test", require_watcher=False)


def test_a_wall_no_holder_can_ever_fit_is_said_with_both_numbers(tmp_path, fake):
    del fake
    clock = Clock()
    pool = _pool(tmp_path, clock, time="00:10:00")
    (event,) = pool.start({1: 3600.0})
    assert event.kind == "POOL_CAPACITY" and event.key == "wall:nodes=1"
    assert "states a wall of 1h (3600s); one holder of this spec lasts 10m (600s)" in event.detail
    assert "it can never start" in event.detail
    # The shape is still held for its shorter work, and a wall that fits says nothing.
    assert len(pool.records()) == 1
    assert pool.reconcile({1: 120.0}) == []


def test_a_failed_query_decides_nothing_and_a_forgotten_job_is_closed_by_the_controllers_word(
    tmp_path, fake
):
    clock = Clock()
    pool = _pool(tmp_path, clock)
    demand = {1: 600.0}
    pool.start(demand)
    (first,) = pool.records()
    # squeue fails: not an empty queue. Nothing is submitted.
    fake.set("squeue_fail")
    clock.advance(4000)
    assert pool.reconcile(demand) == [] and len(fake.jobs) == 1
    fake.clear("squeue_fail")
    # The controller forgot the job and accounting is silent: still not gone...
    fake.change(first["job_id"], forgotten=True, state="COMPLETED")
    fake.set("sacct_fail")
    # ...until the controller itself says the id is no job of its: then it is closed. It
    # never ran, so it left before its end, and its replacement is asked for.
    assert pool.reconcile(demand) == []
    assert "the controller no longer knows the job" in (pool.paths.root / "pool.log").read_text()
    assert pool.records()[0]["closed"]["state"].startswith("GONE") and len(fake.jobs) == 2


def test_ending_cancels_what_is_pending_then_what_is_idle_and_nothing_foreign(tmp_path, fake):
    clock = Clock()
    pool = _pool(tmp_path, clock)
    pool.start({1: 600.0})
    busy = pool.records()[0]
    _run(fake, pool, clock, busy, ends_in=400)
    pool.reconcile({1: 600.0})
    idle = pool.records()[1]
    _run(fake, pool, clock, idle, ends_in=500)
    pool.reconcile({1: 600.0})
    pending = pool.records()[2]
    _beat(pool, clock, busy["name"], job=busy["job_id"], ends_in=400, running=["gate-1"])
    # A job that waits for the idle holder's node (it never started) keeps no holder.
    beat = pool.paths.watchers / f"{idle['name']}-node0.json"
    write_json_atomic(beat, {**read_json(beat), "reserved_for": "session-2"})
    # Jobs the flow did not record: another run's holder under a like name, and anybody's.
    other_run = fake.add("mv2flow-RUN-pool-9")
    foreign = fake.add("training")
    late: list[dict] = []

    def meanwhile(seconds: float) -> None:
        # A pass that was under way when the marker was written still got its holder.
        pool.paths.stop.unlink()
        pool.reconcile({2: 600.0})
        pool.paths.stop.touch()
        late.append(pool.records()[3])
        assert seconds == 60.0

    report = pool.end(wait_s=60.0, sleep=meanwhile)
    assert pool.paths.stop.is_file()
    assert fake.calls("scancel") == [
        f"--state=PENDING {pending['job_id']}",
        idle["job_id"],
        f"--state=PENDING {late[0]['job_id']}",
    ]
    assert [c["name"] for c in report["cancelled"]] == [
        pending["name"],
        idle["name"],
        late[0]["name"],
    ]
    # A holder with a running job is left: it sees the stop marker and ends once idle.
    assert report["left"] == [{"name": busy["name"], "job_id": busy["job_id"], "state": "running"}]
    assert fake.job(busy["job_id"])["state"] == "RUNNING"
    assert fake.job(other_run)["state"] == "RUNNING" and fake.job(foreign)["state"] == "RUNNING"
    # The marker forbids any submission from then on, and tells the spool's clients.
    before = len(fake.calls("sbatch"))
    assert pool.reconcile({1: 600.0, 2: 600.0}) == [] and len(fake.calls("sbatch")) == before
    assert read_json(pool.paths.pool)["stopping"] is True
    # A new run on the same spool has work: its start removes the marker.
    pool.start({1: 600.0})
    assert not pool.paths.stop.exists()


def test_an_end_that_cannot_ask_what_is_left_says_unknown_and_never_nothing(tmp_path, fake):
    clock = Clock()
    pool = _pool(tmp_path, clock)
    pool.start({1: 600.0})
    (record,) = pool.records()
    fake.set("squeue_fail")
    report = pool.end(wait_s=0)
    assert report["cancelled"] == [] and report["left"] == []
    assert [(row["name"], row["state"]) for row in report["unknown"]] == [
        (record["name"], "unknown")
    ]
    assert report["error"].startswith("the environment could not be asked what is left")


def test_a_device_held_by_a_foreign_process_is_reported(tmp_path, fake):
    clock = Clock()
    pool = _pool(tmp_path, clock)
    demand = {1: 600.0}
    pool.start(demand)
    (holder,) = pool.records()
    _run(fake, pool, clock, holder, ends_in=7200)
    path = pool.paths.watchers / f"{holder['name']}-node0.json"
    record = read_json(path)
    record["held"] = {"2": {"used_mib": 30000, "pids": [4242], "since": clock.now - 30}}
    write_json_atomic(path, record)
    assert pool.reconcile(demand) == [], "a job's own memory goes within moments"
    clock.advance(200)
    record["alive_epoch"] = clock.now
    write_json_atomic(path, record)
    (event,) = pool.reconcile(demand)
    assert event.kind == "POOL_DEVICE_HELD" and event.key == f"{holder['name']}-node0:2"
    assert "30000 MiB" in event.detail and "pids 4242" in event.detail
    assert "nothing is killed" in event.detail


def test_a_holder_ready_work_waits_on_for_memory_that_came_with_its_node_is_given_back(
    tmp_path, fake, monkeypatch
):
    """Such memory does not go by waiting: the next holder is asked for without the node."""
    clock = Clock()
    pool = _pool(tmp_path, clock, sbatch_args=["--exclude=site-node"])
    pool.start({1: 600.0})
    (first,) = pool.records()
    _run(fake, pool, clock, first)
    held = {"3": {"used_mib": 24832, "pids": [], "since": clock.now - 200}}

    def holds(record: dict) -> str:
        """The node reports the memory, under the long name it calls itself by."""
        beat, host = pool.paths.watchers / f"{record['name']}-node0.json", f"{record['name']}-node0"
        write_json_atomic(beat, {**read_json(beat), "held": held, "host": f"{host}.site.example"})
        return host

    host = holds(first)
    # Nothing waits on it: the memory is said, and the holder's other devices serve.
    assert _kinds(pool.reconcile({1: 600.0})) == ["POOL_DEVICE_HELD"]
    assert not pool.records()[0].get("stop_requested")
    # A job that needs the node whole is ready and cannot start there. One that has only
    # just arrived may still start on a free device: the holder goes once it has waited.
    _ready(pool, "test-session", wall=300, gpus=4)
    demand = pool_mod.spool_demand(pool.paths.root, {1: 600.0})
    assert _kinds(pool.reconcile(demand)) == ["POOL_DEVICE_HELD"] and len(pool.records()) == 1
    clock.advance(150)
    _run(fake, pool, clock, first, ends_in=1650)
    holds(first)
    assert pool.reconcile(demand) == []
    first, second = pool.records()
    assert first["avoid"] == [host] and fake.calls("scancel") == [first["job_id"]]
    assert f"--exclude=site-node,{host} " in fake.calls("sbatch")[-1]
    assert (
        f"#SBATCH --exclude=site-node,{host}\n"
        in (pool.paths.holders / f"{second['name']}.sbatch").read_text()
    )
    # A run gives back so many and no more: past that the event stands for the session.
    monkeypatch.setattr(pool_mod, "GIVE_BACK_MOST", 1)
    _run(fake, pool, clock, second)
    holds(second)
    (event,) = pool.reconcile(demand)
    assert event.kind == "POOL_DEVICE_HELD" and second["name"] in event.key
    assert not pool.records()[1].get("stop_requested")
    # Slurm refuses a request without those nodes: the next one names none.
    fake.change(second["job_id"], state="COMPLETED")
    fake.set("refuse", "Requested node configuration is not available")
    clock.advance(200)
    assert _kinds(pool.reconcile(demand)) == ["POOL_SUBMIT_FAILED"]
    fake.clear("refuse")
    clock.advance(200)
    assert pool.reconcile(demand) == [] and "--exclude" not in fake.calls("sbatch")[-1]


class _OneHost:
    """One host as the pool sees it, with no process: a holder is a name that is alive."""

    kind, renews, lead_s, time_s = "local", True, 0.0, None

    def __init__(self) -> None:
        self.alive: list[str] = []
        self.ended: list[str] = []

    def holder_name(self, seq: int) -> str:
        return f"local-{seq}"

    def submit(self, name: str, nodes: int, avoid=()) -> pool_mod.Reply:
        self.alive.append(name)
        return pool_mod.Reply("accepted", job_id=name)

    def listing(self, records) -> dict:
        state = lambda r: "running" if r["name"] in self.alive else "ended"  # noqa: E731
        return {r["name"]: [pool_mod.Row(state(r), job_id=r["name"])] for r in records}

    def end(self, record, *, pending_only: bool = False) -> str:
        self.alive.remove(record["name"])
        self.ended.append(record["name"])
        return ""


def test_on_one_host_no_holder_is_ended_for_another_and_work_that_cannot_fit_is_continued(
    tmp_path,
):
    """Inside a job the host's watcher ends with it: a new holder there is the same place."""
    clock, host = Clock(), _OneHost()
    pool = pool_mod.Pool(tmp_path / "spool", host, clock=clock)
    pool.start({1: 600.0})
    # The final tests ask for two hours, and 6,770 s of the job this host runs in are left.
    _beat(pool, clock, "local-1", job="local-1", ends_in=6770)
    _ready(pool, "test-final", wall=7200)
    told: list[str] = []
    for _ in range(4):
        clock.advance(15)
        _beat(pool, clock, "local-1", job="local-1", ends_in=6770 - (clock.now - pool.clock()))
        assert pool.reconcile(pool_mod.spool_demand(pool.paths.root, {})) == []
    assert host.ended == [] and [r["name"] for r in pool.records()] == ["local-1"]
    assert not pool.records()[0].get("stop_requested")
    # Nothing that is ready fits what is left and nothing runs: said once, and the keeper
    # tells the controller until it takes it.
    assert "job test-final states a wall of 2h and this host's watcher ends in 1h5" in pool.late
    assert (pool.paths.root / "pool.log").read_text().count("no job that is ready fits") == 1
    keeper = pool_mod.Keeper(
        pool,
        lambda: pool_mod.spool_demand(pool.paths.root, {}),
        interval_s=0.05,
        on_late=lambda why: told.append(why) or len(told) > 1,
    ).start()
    deadline = time.time() + 20
    while len(told) < 2 and time.time() < deadline:
        time.sleep(0.05)
    time.sleep(0.3)
    keeper.stop()
    assert len(told) == 2 and told[0] == pool.late, "asked again until taken, then no more"
    # A job that fits is ready too: there is work for this host, and nothing is late.
    _ready(pool, "test-short", wall=600)
    assert pool.reconcile(pool_mod.spool_demand(pool.paths.root, {})) == [] and pool.late == ""


# ------------------------------------------------------- which environment


def test_a_spec_that_names_no_environment_is_slurm_where_sbatch_is_and_this_host_where_not(
    tmp_path, monkeypatch
):
    kind = pool_mod.environment_kind
    assert kind({}) == kind(None) == "slurm"
    assert kind({"kind": "local", "slurm": SLURM}) == "local"
    assert kind({"slurm": SLURM, "spool": "/s"}) == "slurm"
    assert kind({"spool": "/s"}) == "slurm"
    # Where sbatch is, the schema prints what is missing, one line a key, and what else may
    # be stated.
    slurm_host = tmp_path / "bin"
    slurm_host.mkdir()
    (slurm_host / "sbatch").write_text("#!/bin/sh\n")
    (slurm_host / "sbatch").chmod(0o755)
    monkeypatch.setenv("PATH", str(slurm_host))
    errors: list[str] = []
    task_schema._validate_environment({"python": "python3"}, errors)
    assert errors[0].startswith("'environment' names no environment, so it is Slurm")
    assert "`environment.kind: local`" in errors[0]
    assert [line.split(" ", 1)[0] for line in errors[1:]] == [
        f"environment.slurm.{key}" for key in ("account", "partition", "time", "gpus_per_node")
    ]
    for block in ({"kind": "local"}, {"slurm": dict(SLURM)}):
        errors = []
        task_schema._validate_environment({"python": "python3", **block}, errors)
        assert errors == []
    # Watchers kept by hand are not served any more: each way a spec named them is refused
    # with one line that says what the flow has in their place.
    for block in ({"kind": "session_spool", "spool": "/s"}, {"spool": "/s", "sessions": []}):
        errors = []
        task_schema._validate_environment({"python": "python3", **block}, errors)
        assert len(errors) == 1 and "serves no watcher" in errors[0]
        assert "environment.slurm" in errors[0] and "`environment.kind: local`" in errors[0]
    # The resolved spec states the kind and the run's one spool.
    task = {"environment": {"python": "python3", "slurm": dict(SLURM)}}
    pool_mod.resolve_spool(task, tmp_path / "ws")
    assert task["environment"]["kind"] == "slurm"
    assert task["environment"]["spool"] == str((tmp_path / "ws").resolve() / "spool")
    # Without sbatch a block that names nothing is this one host: nothing to state, and the
    # kind is written into the spec the run reads.
    monkeypatch.setenv("PATH", str(tmp_path))
    errors, alone = [], {"python": "python3"}
    task_schema._validate_environment(alone, errors)
    assert errors == [] and alone["kind"] == "local"
    # A spec that states Slurm stops there, with what the flow can launch and what it needs.
    task["environment"]["slurm"]["container_mounts"] = _mounts(tmp_path)
    with pytest.raises(pool_mod.PoolError) as refused:
        pool_mod.build(task, workspace=tmp_path / "ws")
    assert str(refused.value) == pool_mod.NO_SBATCH


def test_an_environment_that_is_neither_slurm_nor_one_host_gets_one_message(tmp_path, monkeypatch):
    """Wherever it is found: the spec's load, the pool, a step across nodes, `intake derive`."""
    from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import (
        EnvironmentError as StepError,
    )
    from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import RunEnvironment
    from agent_flow.workflows.trtllm_modeling_v2.spec import derive

    said = pool_mod.OTHER_ENVIRONMENT
    # A spec that names another kind, or the strings of a launcher the flow no longer reads.
    for block, named in (
        ({"python": "python3", "kind": "recipe"}, "`environment.kind: recipe`"),
        ({"python": "python3", "pool": {"holder": "get {nodes}"}}, "`environment.pool`"),
    ):
        errors: list[str] = []
        task_schema._validate_environment(dict(block), errors)
        assert errors == [
            f"'environment' names an environment the flow has no launcher for ({named}): {said}"
        ]
    # One host: a step across nodes is refused with it, and `intake derive` lists a deployment
    # wider than the node before any start.
    with pytest.raises(StepError) as no_step:
        RunEnvironment("local", "python3").step(ranks=4, nodes=2, hosts="a,b", command="true")
    assert str(no_step.value).endswith(f"needs environment.slurm: {said}")
    monkeypatch.setenv("PATH", str(tmp_path))
    task = {
        "environment": {"python": "python3", "gpus_per_node": 2, "kind": "local"},
        "deployments": [{"name": "tp4", "options": {"tensor_parallel_size": 4}}],
    }
    listed: list[tuple] = []
    derive._no_launcher(task, task["deployments"], lambda *row: listed.append(row))
    ((key, what, why),) = listed
    assert (key, what) == ("deployments.tp4", "refused before any load") and why.endswith(said)
    # `intake derive` writes the kind it detected and says where it comes from.
    environment, told = {"python": "python3"}, []
    derive._state_environment(environment, told)
    assert environment["kind"] == "local"
    assert told == [
        ("environment.kind", "local", "detected: no sbatch on this host, so this one host")
    ]


def test_a_preflight_that_fails_says_which_environment_it_failed_in(tmp_path, monkeypatch, capsys):
    """The kind is in the report of a preflight that passed; one that failed says it as well."""
    from types import SimpleNamespace

    from agent_flow.workflows.trtllm_modeling_v2.run.startup import bootstrap, preflight

    def fails(task_data, **more):
        raise preflight.PreflightError("nvidia-smi is not on PATH")

    monkeypatch.setattr(bootstrap, "run_preflight", fails)
    journal = SimpleNamespace(open=lambda **kw: None, section=lambda *a: None)
    # As the schema leaves a spec that states no kind on a host with no `sbatch`.
    task = {"environment": {"python": sys.executable, "kind": "local"}}
    workflow = SimpleNamespace(task_data=task, _command_runner=None, journal=journal)
    assert bootstrap._run_preflight(workflow, SimpleNamespace(), None) is False
    out = " ".join(capsys.readouterr().out.split())
    assert "✘ preflight: nvidia-smi is not on PATH" in out
    assert "the environment is `local` (environment.kind). A block that names none" in out


def test_a_slurm_pool_renders_its_entries_and_says_what_no_mount_covers(tmp_path, fake):
    del fake
    task = {
        "checkpoint_path": "/models/ckpt",
        "environment": {"python": "/opt/venv/bin/python", "slurm": dict(SLURM)},
    }
    pool_mod.resolve_spool(task, tmp_path / "ws")
    with pytest.raises(pool_mod.PoolError) as refused:
        pool_mod.build(task, workspace=tmp_path / "ws")
    lines = str(refused.value).splitlines()
    assert lines[0] == f"{tmp_path / 'ws' / 'spool'} (the spool) is under no container mount"
    assert lines[1].endswith("scheduler.py (the flow's own scheduler) is under no container mount")
    assert lines[2:] == [
        f"{tmp_path / 'ws'} (the workspace) is under no container mount",
        "/models/ckpt (the checkpoint) is under no container mount",
    ]
    task["environment"]["slurm"]["container_mounts"] = [*_mounts(tmp_path), "/models:/models"]
    task["environment"]["slurm"]["host_python"] = "/usr/bin/python3.11"
    pool = pool_mod.build(task, workspace=tmp_path / "ws")
    assert pool.env.kind == "slurm" and pool.env.lead_s == 300.0
    watcher = (pool.paths.root / "watcher_entry.sh").read_text()
    lead = (pool.paths.root / "lead_entry.sh").read_text()
    assert "--session pool --gpus 4" in watcher and "import tensorrt_llm" in watcher
    # Every job of a watcher finds the interpreter's directory first; the lead, on the batch
    # host, has nothing of the container's. A spec that states its own PATH keeps its own.
    assert 'export PATH=/opt/venv/bin:"$PATH"\n' in watcher and "export PATH" not in lead
    task["environment"]["env"] = {"PATH": "/site/bin"}
    own = sessions_mod.render_entry(task, spool="/s", session="pool")
    assert 'export PATH=/site/bin:"$PATH"\n' in own and "/opt/venv/bin:" not in own
    assert lead.rstrip().endswith("--session pool --gpus 0 --lead")
    assert "exec /usr/bin/python3.11 " in lead and "tensorrt_llm" not in lead


def test_one_host_gets_one_local_watcher_that_the_flow_starts_and_stops(tmp_path, monkeypatch):
    """The spool is the device ledger there too: a job runs under the watcher the flow started."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    # The engine's install: its interpreter, and a tool that is only beside it (as `pip` is).
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "python").symlink_to(sys.executable)
    (venv / "engine-tool").write_text("#!/bin/sh\necho from-the-engines-install\n")
    (venv / "engine-tool").chmod(0o755)
    task = {"environment": {"kind": "local", "python": str(venv / "python")}}
    pool_mod.resolve_spool(task, tmp_path)
    pool = pool_mod.build(task, workspace=tmp_path)
    assert pool.env.kind == "local"
    assert pool.start({1: 60.0}) == []
    (record,) = pool.records()
    try:
        deadline = time.time() + 30
        while not pool.paths.live_watchers(60) and time.time() < deadline:
            time.sleep(0.1)
        (watcher,) = pool.paths.live_watchers(60)
        assert watcher["holder"] == "local-1" and watcher["ends_at_epoch"] is None
        assert pool.reconcile({1: 60.0}) == [] and len(pool.records()) == 1, "one watcher, no end"
        code = spoolctl.main(
            ["--spool", str(pool.paths.root), "submit", "--gpus", "0", "--wall", "60", "hello"]
            + ["echo", "ran-under-the-watcher"]
        )
        assert code == 0
        assert "ran-under-the-watcher" in SpoolPaths(pool.paths.root).log("hello").read_text()
        # A job finds the engine's own tools first, as with its environment activated: the
        # spec states the interpreter, and nobody states a PATH.
        code = spoolctl.main(
            ["--spool", str(pool.paths.root), "submit", "--gpus", "0", "--wall", "60", "tool"]
            + ["engine-tool"]
        )
        assert code == 0, SpoolPaths(pool.paths.root).log("tool").read_text()
        assert "from-the-engines-install" in SpoolPaths(pool.paths.root).log("tool").read_text()
        # A job of two nodes cannot be held here: that is said, and the watcher is not ended.
        (event,) = pool.reconcile({1: 60.0, 2: 60.0})
        assert event.kind == "POOL_CAPACITY" and "this one host is one node" in event.detail
        assert len(pool.records()) == 1 and pool.env._alive(record)
    finally:
        pool.end(wait_s=0)
        pool.env.end(record)
    deadline = time.time() + 20
    while pool.env._alive(record) and time.time() < deadline:
        time.sleep(0.1)
    assert not pool.env._alive(record), "the watcher never outlives the flow"


def _workflow(tmp_path, environment: dict, **settings):
    """What the start and the end of a run read of a workflow, with nothing else of one."""
    from types import SimpleNamespace

    said: list[tuple[str, str]] = []
    task = {
        "environment": {"python": sys.executable, **environment},
        # One session of four hours with its teardown: the longest wall the run states.
        "deployments": [{"name": "tp1", "options": {}, "timeout": 13800}],
    }
    pool_mod.resolve_spool(task, tmp_path / "ws")
    workflow = SimpleNamespace(
        task_data=task,
        workspace=tmp_path / "ws",
        journal=SimpleNamespace(event=lambda kind, text: said.append((kind, text))),
        _shell_runner_override=None,
        dry_run=False,
        **settings,
    )
    return workflow, said


def test_a_run_asks_its_scheduler_for_no_more_node_hours_than_its_spec_states(
    tmp_path, fake, capsys
):
    """The bound: derived from the holders' records, said with the verb, and the run stops at it."""
    from agent_flow.workflows.trtllm_modeling_v2.run import escalation as escalation_mod

    slurm = {**SLURM, "container_mounts": _mounts(tmp_path), "time": "04:00:00", "lead": "30m"}
    workflow, _ = _workflow(tmp_path, {"slurm": {**slurm, "max_node_hours": 10}})
    task, workspace = workflow.task_data, workflow.workspace
    # Never "none" on Slurm: a spec that states no bound has the default.
    unstated = pool_mod.build({"environment": {**task["environment"], "slurm": slurm}})
    assert unstated.env.max_node_hours == pool_mod.DEFAULT_MAX_NODE_HOURS == 48.0
    errors: list[str] = []
    slurm_mod.validate({**slurm, "max_node_hours": 0}, errors)
    assert errors == ["'environment.slurm.max_node_hours' must be a number of node-hours above 0"]
    clock = Clock()
    pool = pool_mod.build(task, workspace=workspace, clock=clock)
    demand = {1: 600.0}
    pool.start(demand)
    first = pool.records()[0]
    # A record says what was asked for, and counts from the moment it is written.
    assert (first["nodes"], first["time_s"]) == (1, 14400.0)
    assert pool.asked().render() == ("node-hours asked: 4.0 of 10 (1 holder job(s))")
    _run(fake, pool, clock, first, ends_in=14400)
    clock.advance(12000)
    _beat(pool, clock, first["name"], job=first["job_id"], ends_in=2400)
    assert pool.reconcile(demand) == [] and len(fake.jobs) == 2
    second = pool.records()[1]
    _run(fake, pool, clock, second, ends_in=14400)
    clock.advance(2500)
    fake.change(first["job_id"], state="COMPLETED")
    assert pool.reconcile(demand) == [] and pool.asked().holders == 8.0
    # The next renewal would make twelve: it is not asked for. One event, which names the
    # verb; what runs goes on.
    clock.advance(9600)
    _beat(pool, clock, second["name"], job=second["job_id"], ends_in=2300)
    (event,) = pool.reconcile(demand)
    assert event.kind == "POOL_CAPACITY" and event.key == "budget" and len(fake.jobs) == 2
    assert "8.0 of 10 (2 holder job(s)); one more holder of 1 node(s) for 4h" in (event.detail)
    assert "`resolve pool-adjust pool --set max_node_hours=<more>` raises the bound" in event.detail
    assert pool.reconcile(demand) == [event] and not pool.exhausted
    # `monitor wait` returns it from the pool's last pass, and nothing else is kept of it.
    (workspace / "task.yaml").write_text(yaml.safe_dump(task))
    said = status.poll_once(status.load_config(workspace))
    assert (said.kind, said.detail) == ("POOL_CAPACITY", event.detail)
    assert not (workspace / "escalations").exists()
    # The last holder ends: nothing of the run can run any more, and its keeper says so once.
    clock.advance(2400)
    fake.change(second["job_id"], state="COMPLETED")
    assert _kinds(pool.reconcile(demand)) == ["POOL_CAPACITY"] and pool.exhausted
    told: list[str] = []
    keeper = pool_mod.Keeper(
        pool, lambda: demand, interval_s=0.05, on_exhausted=lambda why: told.append(why) or True
    ).start()
    deadline = time.time() + 20
    while not told and time.time() < deadline:
        time.sleep(0.05)
    time.sleep(0.3)
    keeper.stop()
    assert len(told) == 1 and "would make 12.0" in told[0] and len(fake.jobs) == 2
    # `pool status` says what was asked of the bound.
    assert pool_mod.main(["--workspace", str(workspace), "status"]) == 0
    assert "  node-hours asked: 8.0 of 10 (2 holder job(s))" in (capsys.readouterr().out)
    # A spec that names no spool (a check by hand in a directory of its own) has the workspace's.
    named = task["environment"]
    unnamed = {k: v for k, v in named.items() if k not in ("spool", "kind")}
    (workspace / "task.yaml").write_text(yaml.safe_dump({**task, "environment": unnamed}))
    assert pool_mod.main(["--workspace", str(workspace), "status"]) == 0
    assert f"pool {named['spool']}: slurm" in capsys.readouterr().out
    # Raised by the verb, the pool asks again in the same pass, and its end says the sum.
    escalation_mod.set_spec(task, "environment.slurm.max_node_hours", 20)
    pool.env = pool_mod.build(task, workspace=workspace).env
    assert pool.reconcile(demand, retry_now=True) == [] and len(fake.jobs) == 3
    assert not pool.exhausted
    assert pool.end(wait_s=0)["asked"] == ("node-hours asked: 12.0 of 20 (3 holder job(s))")


def test_a_workspace_named_relative_to_the_start_is_under_a_mount_when_its_directory_is(
    tmp_path, fake, monkeypatch
):
    """`--workspace ws`, as a person types it: the mounts are checked against where it is."""
    del fake
    monkeypatch.chdir(tmp_path)
    slurm = {**SLURM, "container_mounts": _mounts(tmp_path)}
    task = {"environment": {"python": sys.executable, "slurm": slurm}}
    pool_mod.resolve_spool(task, "ws")
    assert pool_mod.build(task, workspace="ws").env.kind == "slurm"
    # One that is under no mount is still said, by its whole path.
    elsewhere = {
        **slurm,
        "container_mounts": [f"{tmp_path}/other:{tmp_path}/other", *_mounts(tmp_path)[1:]],
    }
    task = {"environment": {"python": sys.executable, "slurm": elsewhere}}
    pool_mod.resolve_spool(task, "ws")
    with pytest.raises(pool_mod.PoolError) as refused:
        pool_mod.build(task, workspace="ws")
    assert f"{tmp_path}/ws (the workspace) is under no container mount" in str(refused.value)


def test_a_job_that_is_over_counts_what_it_held_and_an_open_one_what_was_asked(tmp_path, fake):
    """A start that failed in a minute does not cost the run a holder's whole time."""
    slurm = {**SLURM, "container_mounts": _mounts(tmp_path), "time": "04:00:00"}
    workflow, _ = _workflow(tmp_path, {"slurm": slurm})
    clock = Clock()
    pool = pool_mod.build(workflow.task_data, workspace=workflow.workspace, clock=clock)
    pool.start({1: 600.0})
    first = pool.records()[0]
    _run(fake, pool, clock, first, ends_in=14400)
    assert pool.reconcile({1: 600.0}) == [] and pool.asked().holders == 4.0
    # Cancelled by hand after ten minutes: the scheduler's own start and end say what it held.
    clock.advance(600)
    fake.change(first["job_id"], state="CANCELLED by 1000", end=int(clock.now))
    pool.reconcile({})
    closed = pool.records()[0]["closed"]
    assert closed["end"] - closed["start"] == 600 and round(pool.asked().holders, 2) == 0.17
    # One that was cancelled while it waited never started: it held nothing.
    pool.reconcile({1: 600.0}, retry_now=True)
    second = pool.records()[1]
    assert pool.asked().holders == pytest.approx(4.0 + 600 / 3600)
    fake.change(second["job_id"], state="CANCELLED by 1000", end=int(clock.now))
    pool.reconcile({})
    assert pool.asked().holder_jobs == 2 and round(pool.asked().holders, 2) == 0.17
    # One the scheduler has forgotten says nothing of its times: what was asked stands.
    pool.reconcile({1: 600.0}, retry_now=True)
    third = pool.records()[2]
    fake.set("sacct_empty")
    fake.change(third["job_id"], forgotten=True)
    pool.reconcile({})
    assert pool.records()[2]["closed"]["state"].startswith("GONE")
    assert pool.asked().holders == pytest.approx(4.0 + 600 / 3600)


def test_a_start_says_the_bound_and_stops_at_once_when_it_is_already_reached(
    tmp_path, fake, capsys
):
    from agent_flow.workflows.trtllm_modeling_v2.run.startup import bootstrap

    slurm = {**SLURM, "container_mounts": _mounts(tmp_path)}
    # One holder of this spec is half an hour of one node: over a bound of 0.4 node-hours.
    workflow, said = _workflow(tmp_path, {"slurm": {**slurm, "max_node_hours": 0.4}})
    assert bootstrap._start_pool(workflow, None) is False
    out = " ".join(capsys.readouterr().out.split())
    assert "node-hours asked: 0.0 of 0.4 (0 holder job(s))" in out
    assert "POOL_CAPACITY the run's node-hours asked: 0.0 of 0.4 (0 holder job(s))" in out
    assert workflow._budget_stop is True and fake.jobs == [], "to be continued, nothing asked for"
    assert said[-1][0] == "pool cannot start"
    # Under the bound the start says it and goes on.
    workflow, said = _workflow(tmp_path, {"slurm": {**slurm, "max_node_hours": 10}})
    assert bootstrap._start_pool(workflow, None) is True
    workflow._pool_keeper.stop()
    out = " ".join(capsys.readouterr().out.split())
    assert "node-hours asked: 0.5 of 10 (1 holder job(s))" in out
    assert not getattr(workflow, "_budget_stop", False) and len(fake.jobs) == 1
    workflow._pool.end(wait_s=0)


def test_a_run_asks_for_its_first_holder_before_any_turn_and_ends_its_pool_only_when_closed(
    tmp_path, fake, capsys, monkeypatch
):
    from types import SimpleNamespace

    from agent_flow.workflows.trtllm_modeling_v2.run.startup import bootstrap

    slurm = {**SLURM, "container_mounts": _mounts(tmp_path)}
    # A refusal of the very first submission stops the start with Slurm's own line.
    workflow, said = _workflow(tmp_path, {"slurm": slurm})
    fake.set("refuse", "Invalid account or account/partition combination specified")
    assert bootstrap._start_pool(workflow, None) is False
    out = " ".join(capsys.readouterr().out.split())
    assert "was refused: sbatch: error: Invalid account or account/partition combination" in out
    assert "tried again" not in out, "the start stops; nothing is retried behind it"
    assert workflow._pool is None and said[-1][0] == "pool cannot start"
    fake.clear("refuse")
    # A dry run asks only whether the environment resolves: nothing is submitted.
    workflow.dry_run = True
    assert bootstrap._start_pool(workflow, None) is True and fake.jobs == []
    workflow.dry_run = False
    # Nor is anything asked for when an injected runner runs the commands.
    workflow._shell_runner_override = lambda *a, **k: None
    assert bootstrap._start_pool(workflow, None) is True and fake.jobs == []
    workflow._shell_runner_override = None
    assert bootstrap._start_pool(workflow, None) is True
    assert [job["name"] for job in fake.jobs] == [workflow._pool.env.holder_name(2)]
    session = {"deployments": [{"name": "tp1", "options": {}, "timeout": 300}]}
    assert bootstrap.pool_demand(session) == {1: 900.0}  # its timeout and the teardown
    assert bootstrap.pool_demand(workflow.task_data) == {1: 14400.0}
    # Work that cannot start before this host's job ends is continued at once only when no
    # task of the run is running or ready: until then the keeper asks again.
    keeper, continued = workflow._pool_keeper, []
    monkeypatch.setattr(
        bootstrap.controller_mod, "continue_now", lambda ws, why: continued.append(why) or True
    )
    workflow._task_demand = lambda: {1: 60.0}
    assert keeper.on_late("nothing fits") is False and continued == []
    workflow._task_demand = lambda: {}
    assert keeper.on_late("nothing fits") is True and continued == ["nothing fits"]
    del workflow._task_demand
    # A run that stops short of its close leaves its holders to the command that resumes it.
    bootstrap.end_pool(workflow, SimpleNamespace(done=False), None)
    assert "its holder(s) stay" in " ".join(capsys.readouterr().out.split())
    assert fake.calls("scancel") == []
    assert workflow._pool is None and workflow._pool_keeper is None
    # A closed run has no work left: the pool ends.
    assert bootstrap._start_pool(workflow, None) is True and len(fake.jobs) == 1
    pool = workflow._pool
    pool_mod.END_WAIT_S, kept = 0.0, pool_mod.END_WAIT_S
    try:
        bootstrap.end_pool(workflow, SimpleNamespace(done=True), None)
    finally:
        pool_mod.END_WAIT_S = kept
    assert fake.calls("scancel") == [f"--state=PENDING {fake.jobs[0]['id']}"]
    assert pool.paths.stop.is_file()


def test_one_host_runs_through_the_same_start_and_never_leaves_its_watcher(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from agent_flow.workflows.trtllm_modeling_v2.run.startup import bootstrap

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    workflow, _ = _workflow(tmp_path, {"kind": "local"})
    assert bootstrap._start_pool(workflow, None) is True
    pool = workflow._pool
    (record,) = pool.records()
    assert pool.env._alive(record)
    # Not closed, and still the watcher goes with the process that started it.
    bootstrap.end_pool(workflow, SimpleNamespace(done=False), None)
    deadline = time.time() + 20
    while pool.env._alive(record) and time.time() < deadline:
        time.sleep(0.1)
    assert not pool.env._alive(record)


# ------------------------------------------------------- the allocation probe


def _serving(spool, *, devices=("0", "1"), host="", **config):
    """A watcher on ``spool`` in a thread, with devices that read clean; returns its stop."""
    import threading

    from agent_flow.workflows.trtllm_modeling_v2.infra.compute.scheduler import (
        Scheduler,
        SchedulerConfig,
    )

    environ = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "PMI"))}
    sched = Scheduler(
        spool,
        SchedulerConfig(devices=devices, leftover_grace_s=0.0, **config),
        environ=environ,
        watcher_id=f"{host}-lead" if config.get("lead") else (f"{host}-w" if host else None),
    )
    sched.host = host or sched.host
    sched.device_memory_mib = lambda: {device: 0 for device in devices}
    sched.device_holders = lambda: {}
    sched.announce()
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            sched.tick()
            time.sleep(0.05)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return stop


def test_the_allocation_probe_answers_with_no_model_loaded(tmp_path):
    """Launch, import, a collective, the cache directory, the dataset: seconds, not a load."""
    import subprocess

    from agent_flow.workflows.trtllm_modeling_v2.infra.compute import probe as probe_mod
    from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import RunEnvironment

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "test.jsonl").write_text("{}\n")
    cache = tmp_path / "cache" / "nested"
    # The harness on one rank, as a job would run it.
    argv = probe_mod.probe_arguments(
        tmp_path / "out", 1, modules=["json"], cache_dirs=[str(cache)], datasets=[str(dataset)]
    )
    done = subprocess.run([sys.executable, *argv], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0 and "[alloc-probe] probe rank 0/1" in done.stdout
    (written,) = (tmp_path / "out").glob("probe.rank0.*.json")
    record = json.loads(written.read_text())
    assert record["ok"] and record["modules"]["json"]["file"] and not list(cache.iterdir())
    assert probe_mod.judge(tmp_path / "out", 1) == ([], [record["host"]])
    # Each thing that is wrong is one line: a module, a directory, a dataset, the world's size.
    bad = probe_mod.probe_arguments(
        tmp_path / "bad",
        2,
        modules=["no_such_module_here"],
        cache_dirs=["/proc/nowhere"],
        datasets=[str(tmp_path / "absent")],
    )
    done = subprocess.run([sys.executable, *bad], capture_output=True, text=True, timeout=60)
    assert done.returncode == 1
    problems, _ = probe_mod.judge(tmp_path / "bad", 2)
    assert problems[0].startswith("no record from rank(s) 1 of 2: the launch did not start one")
    said = " | ".join(problems[1:])
    assert (
        "no_such_module_here does not import" in said
        and "/proc/nowhere does not take a file" in said
    )
    assert f"{tmp_path / 'absent'} cannot be read" in said
    assert "2 were asked for" in said or "mpi4py does not import" in said
    # Every rank exactly once, each in the world that was asked for: nothing less passes.
    twice = tmp_path / "twice"
    twice.mkdir()
    for host in ("nodeA", "nodeB"):
        row = {"rank": 0, "size": 1, "host": host, "problems": []}
        (twice / f"probe.rank0.{host}.1.json").write_text(json.dumps(row))
    assert probe_mod.judge(twice, 1)[0] == [
        "world rank 0 was reported 2 times (on nodeA, nodeB): the launch made more than one world"
    ]

    # Through the spool, under a watcher: one rank on one device.
    spool = tmp_path / "spool"
    stop = _serving(spool)
    environment = RunEnvironment("local", sys.executable, spool=str(spool))
    try:
        found = probe_mod.run_allocation_probe(
            environment,
            ranks=1,
            out=tmp_path / "probe",
            modules=["json"],
            cache_dirs=[str(cache)],
            datasets=[str(dataset)],
            wall_s=60,
        )
        assert found.ok, found.render()
        assert found.render().startswith("allocation probe: 1 rank(s) on 1 node(s) ok (")
        job = next((spool / "done").glob("alloc-probe-*.sh")).read_text()
        assert "gpus=1 wall=60" in job and "--expect-ranks 1" in job and "mpirun" not in job
        # A watcher that never read its devices can never call one free: said in one line.
        blind = {"watcher": "blind", "host": "nodeB", "devices": ["0"], "alive_epoch": time.time()}
        write_json_atomic(SpoolPaths(spool).watchers / "blind.json", blind)
        (line,) = probe_mod.watchers_without_reading(spool)
        assert line.startswith("watcher blind on nodeB has no device reading")
    finally:
        stop.set()

    # Several ranks on one node are started as an engine's are: one process per rank.
    asked = []

    class Executor:
        def execute(self, spec):
            asked.append(spec)
            return SimpleNamespace(stdout="", infrastructure="", returncode=0)

    from types import SimpleNamespace

    probe_mod_select = probe_mod.select_command_executor
    probe_mod.select_command_executor = lambda *a, **k: (asked.append(k), Executor())[1]
    try:
        four = probe_mod.run_allocation_probe(environment, ranks=4, out=tmp_path / "four")
    finally:
        probe_mod.select_command_executor = probe_mod_select
    assert asked[0]["gpus"] == 4
    command = asked[1].argv
    assert f"mpirun -n 4 --oversubscribe --allow-run-as-root {sys.executable} " in command
    assert "trtllm-llmapi-launch" not in command, "every rank runs the probe itself"
    assert four.problems[0].startswith("no record from rank(s) 0, 1, 2, 3 of 4")


def test_a_wide_probe_is_one_step_across_the_reserved_nodes_run_by_the_lead(tmp_path, monkeypatch):
    """The lead forms the job from its holder's watchers and runs the step line once."""
    from agent_flow.workflows.trtllm_modeling_v2.infra.compute import probe as probe_mod
    from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import RunEnvironment

    # A stand-in for `srun`, which the step line names: it runs the command once, so the
    # world it makes has one rank where two were asked for.
    launcher = tmp_path / "launcher"
    launcher.mkdir()
    (launcher / "srun").write_text(
        '#!/bin/bash\nwhile [[ "${1:-}" == --* ]]; do shift; done\nexec "$@"\n'
    )
    (launcher / "srun").chmod(0o755)
    monkeypatch.setenv("PATH", f"{launcher}{os.pathsep}{os.environ['PATH']}")
    spool = tmp_path / "spool"
    stops = [
        _serving(spool, host="nodeA", holder="h1", nodes=2),
        _serving(spool, host="nodeB", holder="h1", nodes=2),
        _serving(spool, host="batch", devices=(), holder="h1", nodes=2, lead=True),
    ]
    environment = RunEnvironment(
        "slurm",
        sys.executable,
        spool=str(spool),
        slurm={"gpus_per_node": 4},
        env={"PROBE_ENV": "1"},
    )
    try:
        found = probe_mod.run_allocation_probe(
            environment, ranks=2, nodes=2, out=tmp_path / "probe", modules=["json"], wall_s=60
        )
    finally:
        for stop in stops:
            stop.set()
    script = next((spool / "done").glob("alloc-probe-*.sh")).read_text()
    assert "nodes=2 wall=60" in script and "gpus=" not in script
    (step,) = [line for line in script.splitlines() if line.startswith("srun --overlap ")]
    assert '--nodes=2 --ntasks=2 --ntasks-per-node=1 --nodelist="$MV2FLOW_JOB_HOSTS"' in step
    assert "--mpi=pmix" in step and step.endswith("probe.sh")
    inner = next((spool / "jobs").glob("alloc-probe-*/probe.sh")).read_text()
    assert f'export PATH={os.path.dirname(sys.executable)}:"$PATH"' in inner
    assert "export PROBE_ENV=1" in inner and "export MV2FLOW_ROLE=probe" in inner
    # It ran once, where the lead runs: the probe says the launch did not make the world.
    record = json.loads(next((spool / "done").glob("alloc-probe-*.json")).read_text())
    assert record["nodes"] == 2 and record["hosts"] == ["nodeA", "nodeB"]
    assert record["watcher"] == "batch-lead"
    assert not found.ok
    assert found.problems[0] == "the probe's job exited 1", "a launch that fails proves nothing"
    assert any(p.startswith("no record from rank(s) 1 of 2") for p in found.problems)
    assert any("2 were asked for" in p or "mpi4py does not import" in p for p in found.problems)
