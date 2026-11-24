# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A member wider than a node and a stated disaggregated deployment: what a gate writes and judges, and the lead."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import disagg, serve
from agent_flow.workflows.trtllm_modeling_v2.infra.compute import placement, provenance, sessions
from agent_flow.workflows.trtllm_modeling_v2.infra.compute import scheduler as sched
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import resolve_environment
from agent_flow.workflows.trtllm_modeling_v2.run.startup import datasets, preflight
from agent_flow.workflows.trtllm_modeling_v2.run.validation.check_runner import CheckRunner
from agent_flow.workflows.trtllm_modeling_v2.spec import task_schema

ALLOCATION = "4242"
A, B = "nodeA", "nodeB"
#: What a deployment's client says once its servers and their proxy are up.
UP = disagg.disagg_driver.SESSION_MARKER + ' {"started": true}\n'
#: A step of one task on a job's first node: a deployment's client, a wide member's driver.
ONE = '--mpi=none --kill-on-bad-exit=1 --nodes=1 --ntasks=1 --ntasks-per-node=1 --nodelist="${HOSTS[0]}"'


def _deployment(directory, ctx=(2, 2), gen=(4, 1), **said):
    """``accuracy.disagg`` as the tree's three files. A side: (a server's ranks, its servers)."""
    fleets = zip(("context_servers", "generation_servers"), (ctx[1], gen[1]))
    proxy = {"max_retries": 2, **{side: {"num_instances": count} for side, count in fleets}}
    own = {"tensor_parallel_size": gen[0], "moe_expert_parallel_size": gen[0]}
    directory.mkdir(parents=True, exist_ok=True)
    paths = {key: directory / f"{key}.yaml" for key in task_schema.DISAGG_FILES}
    for path, content in zip(paths.values(), (proxy, {"tensor_parallel_size": ctx[0]}, own)):
        path.write_text(yaml.safe_dump(content))
    return {**{key: str(path) for key, path in paths.items()}, **said}


@pytest.fixture
def site(tmp_path, monkeypatch):
    """A tree, a checkpoint and its anchor, and the task of a session of two nodes of four."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    (tmp_path / "tree").mkdir()
    (tmp_path / "org" / "model").mkdir(parents=True)
    (tmp_path / "reference.yaml").write_text(yaml.safe_dump({"org/model": [{"accuracy": 41.77}]}))
    spool = str(tmp_path / "spool")
    gates = {"name": "gates", "spool": spool, "nodes": 2, "setup": ["export FROM_SESSION=entry"]}
    return {
        "checkpoint_path": str(tmp_path / "org" / "model"),
        "trtllm_repo_path": str(tmp_path / "tree"),
        "parallel": "unified",
        "members": ["tep4"],
        "features": ["disagg"],
        "environment": {
            "kind": "session_spool",
            "python": sys.executable,
            "spool": spool,
            "gpus_per_node": 4,
            "env": {"TMPDIR": str(tmp_path / "node-tmp")},
            "step_args": ["--container-name=mv2flow"],
            "sessions": [gates],
        },
        "accuracy": {
            "benchmark": "gsm8k",
            "reference_yaml": str(tmp_path / "reference.yaml"),
            "disagg": _deployment(tmp_path / "stated", env={"UCX_TLS": "^cuda_ipc"}, timeout=120),
        },
    }


def _scheduler(spool, host, *, lead=False, allocation=ALLOCATION) -> sched.Scheduler:
    """A watcher (or the lead) of ``spool`` on ``host``, in one allocation, its devices clean."""
    environ = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "PMI"))}
    environ.update(SLURM_JOB_ID=allocation, SLURM_PROCID="0", SLURM_STEP_ID="0")
    devices = () if lead else ("0", "1", "2", "3")
    made = sched.Scheduler(
        spool,
        sched.SchedulerConfig(devices=devices, leftover_grace_s=0.0, lead=lead),
        environ=environ,
        watcher_id=f"{host}-lead" if lead else f"{host}-w",
    )
    made.host = host
    made.device_memory_mib = lambda: {device: 0 for device in devices}
    made.heartbeat()
    return made


def _answers(stdout="", returncode=0):
    """A gate's runner that runs nothing: the job answers ``stdout`` and ends with ``returncode``."""
    return lambda command, env=None, timeout=0: CommandResult(returncode=returncode, stdout=stdout)


def _lines(directory):
    """The line of each server in the script a gate wrote, then its client's."""
    script = (directory / "servers.sh").read_text().splitlines()
    return [*(line for line in script if line.startswith("{ ")), script[script.index("rc=$?") - 1]]


def test_a_spec_states_a_deployment_and_its_layout(site):
    stated = site["accuracy"]["disagg"]
    # A side's servers are the proxy's config's, a server's ranks its own options'.
    assert disagg.DisaggSpec.from_task(site).servers() == [("ctx0", 2), ("ctx1", 2), ("gen0", 4)]
    test = "tests/x/test_y.py::TestZ::test_hop"
    kept = disagg.DisaggSpec.from_task({"accuracy": {"disagg": {"test": test, "nodes": 2}}})
    assert not kept.served and kept.job_nodes() == 2
    found: list[str] = []
    task_schema._validate_disagg({**stated, "test": test}, found)
    assert "state one of the two" in found[0]

    def layout(*servers, per_node=4):
        nodes, slots = disagg.plan_layout(servers, per_node)
        return nodes, {name: (s.node, s.nodes, s.devices) for name, s in slots.items()}

    whole = (0, 1, 2, 3)
    shared = {"ctx0": (0, 1, (0, 1)), "ctx1": (0, 1, (2, 3)), "gen0": (1, 1, whole)}
    assert layout(("ctx0", 2), ("ctx1", 2), ("gen0", 4)) == (2, shared)
    assert layout(("ctx0", 16), ("gen0", 16)) == (8, {"ctx0": (0, 4, whole), "gen0": (4, 4, whole)})
    assert layout(("ctx0", 4), ("gen0", 4), per_node=None)[0] == 1
    with pytest.raises(ValueError, match="a world wider than a node takes whole nodes"):
        layout(("ctx0", 6), ("gen0", 2))


@pytest.mark.parametrize(
    ("members", "ctx", "gen", "nodes", "lead"),
    [
        (["dep16"], (16, 1), (16, 1), 8, True),  # a server a side, each across four nodes
        (["tep4"], (2, 1), (2, 1), 1, False),  # both sides on one node
    ],
)
def test_the_specs_of_the_real_cases_load_and_size_their_checks_jobs(
    task_file, tmp_path, monkeypatch, members, ctx, gen, nodes, lead
):
    spec = yaml.safe_load(task_file.read_text())
    spool = tmp_path / "spool"
    spool.mkdir()
    spec.update(parallel="unified", members=members, features=["disagg"])
    spec["environment"].update(
        kind="session_spool",
        spool=str(spool),
        gpus_per_node=4,
        step_args=["--container-image=/images/trtllm.sqsh", "--container-name=mv2flow"],
        sessions=[{"name": "gates", "spool": str(spool), "role": "gates", "nodes": max(nodes, 4)}],
    )
    _deployment(task_file.parent / "stated", ctx, gen)  # named relative to the spec
    spec["accuracy"]["disagg"] = {key: f"stated/{key}.yaml" for key in task_schema.DISAGG_FILES}
    task_file.write_text(yaml.safe_dump(spec))
    task = task_schema.load_and_validate_task_yaml(task_file)
    runner = CheckRunner.__new__(CheckRunner)
    runner.workflow = SimpleNamespace(_gate_env=lambda: resolve_environment(task, world_size=4))
    wide = runner._member_env(members[0]).nodes
    assert disagg.job_span(task, "disagg", wide) == (nodes, lead)
    assert disagg.job_span(task, "boot", wide) == ((4, True) if wide > 1 else (1, False))
    assert preflight.check_disagg_placement(task, 4) == []
    # A stated deployment reads the staged data of its own benchmark.
    staged = {"gsm8k": "/staged/gsm8k", "mmlu": "/staged/mmlu"}
    monkeypatch.setattr(datasets, "status", lambda workspace: staged)
    monkeypatch.setattr(datasets, "wait", lambda workspace, name: staged[name])
    runner.workspace = tmp_path
    own = {**task["accuracy"], "benchmark": {"default": "gsm8k", "disagg": "mmlu"}}
    given = runner._with_dataset({**task, "accuracy": own}, "disagg", "disagg")
    assert given["accuracy"]["dataset_path"] == "/staged/mmlu"


def test_what_is_rendered_for_a_session_of_several_nodes_and_for_a_step(site, tmp_path, capsys):
    environment = replace(
        resolve_environment(site, world_size=16),
        python="/venv/bin/python",
        env={"TMPDIR": "/scratch/tmp", "PATH": "/opt/bin", "TRITON_CACHE_DIR": "/shared/cache"},
        step_args=("--container-image=/images/IMAGE.sqsh", "--container-name=mv2flow"),
    ).at_width(8)
    line = environment.python_command(["driver.py", "--spec", "x.json"])
    assert line.startswith(
        "srun --overlap --mpi=pmix --kill-on-bad-exit=1 --nodes=2 --ntasks=8 --ntasks-per-node=4 "
        '--nodelist="$MV2FLOW_JOB_HOSTS" --container-image=/images/IMAGE.sqsh '
        "--container-name=mv2flow bash -c "
    )
    assert (
        """'[ -r "$1/session_env.sh" ] && . "$1/session_env.sh"; cd "$2" && shift 2 && """
        """export PATH=/venv/bin:/opt/bin:"$PATH" && exec env "$@"' step "${MV2FLOW_SPOOL:-}" """
        """"$PWD" PYTHONPATH="${PYTHONPATH:-}" """
    ) in line
    # Each node's own Triton cache.
    assert line.endswith(
        " TMPDIR=/scratch/tmp TRITON_CACHE_DIR=/scratch/tmp/triton-cache "
        "CUDA_VISIBLE_DEVICES=0,1,2,3 trtllm-llmapi-launch "
        "/venv/bin/python driver.py --spec x.json"
    )
    one = environment.python_command(["probe.py"], single_process=True)
    assert "--mpi=none --kill-on-bad-exit=1 --nodes=1 --ntasks=1 --ntasks-per-node=1" in one
    assert one.endswith("CUDA_VISIBLE_DEVICES= /venv/bin/python probe.py")
    narrow = replace(environment, nodes=1, world_size=4).command("x")
    assert "mpirun -n 4" in narrow and "TRITON_CACHE_DIR=/shared/cache" in narrow
    told = replace(environment, world_size=16, spool="/s").describe()
    assert "16 ranks spans 4 nodes here: only `trtllm-modeling-v2 gate ...` starts it" in told
    assert "mpirun -n 16" not in told
    # The lead keeps the allocation's identity and drops a task's; it may stop its own children.
    kept = sched.lead_environment(
        {"SLURM_JOB_ID": "7", "SLURM_JOB_NODELIST": "a,b", "SLURM_PROCID": "0", "PMIX_RANK": "0",
         "SLURM_STEP_ID": "1", "OMPI_MCA_x": "y", "HOME": "/h"}
    )  # fmt: skip
    assert kept == {"SLURM_JOB_ID": "7", "SLURM_JOB_NODELIST": "a,b", "HOME": "/h"}
    assert not sched.is_mpi_attached(kept, job_identity=True) and sched.is_mpi_attached(kept)
    assert sched.is_mpi_attached({**kept, "SLURM_PROCID": "3"}, job_identity=True)
    task = {"environment": {**site["environment"], "env": {"TMPDIR": "/scratch/t"}}}
    task["environment"]["sessions"][0].update(time_limit="4h", gpus_per_node=4)
    spec_file = tmp_path / "render.yaml"
    spec_file.write_text(yaml.safe_dump(task))
    assert sessions.main(["render", "--task", str(spec_file), "--name", "gates"]) == 0
    entry, lead, given = (Path(line) for line in capsys.readouterr().out.split())
    assert [p.name for p in (entry, lead, given)] == [
        "session_entry.sh", "session_lead.sh", "session_env.sh",
    ]  # fmt: skip
    said = lead.read_text().rstrip()
    assert said.endswith("--lead --time-limit 14400 --handover-lead 3600 --session gates")
    assert "TMPDIR" not in said and "--gpus" not in said
    assert "export TMPDIR=/scratch/t\n" in given.read_text()
    assert given.read_text().endswith("export FROM_SESSION=entry\n")
    given.unlink()  # written elsewhere, the entry leaves what a step reads in the spool
    moved = sessions.install_session_entry(task, "gates", out=tmp_path / "out" / "session_entry.sh")
    assert moved.with_name("session_lead.sh").exists() and given.exists()
    assert not moved.with_name("session_env.sh").exists()
    task["environment"]["sessions"][0]["nodes"] = 1
    other = sessions.install_session_entry(task, "gates", out=tmp_path / "one" / "session_entry.sh")
    assert not other.with_name("session_lead.sh").exists()


def test_a_job_across_nodes_is_run_once_by_the_lead_after_its_nodes_drained(site, tmp_path):
    spool = Path(site["environment"]["spool"])
    paths = sched.SpoolPaths(spool)
    watchers = [_scheduler(spool, "batch", lead=True), *(_scheduler(spool, h) for h in (A, B))]
    said = tmp_path / "said"

    def run(name, header, body):
        """Queue a job and tick every watcher in turn until one has recorded it."""
        (paths.pending / f"{name}.sh").write_text(
            f"#!/bin/bash\n#spool: {header} wall=60\n{body}\n"
        )
        for _ in range(500):
            for watcher in watchers:
                watcher.tick()
            if (paths.done / f"{name}.json").exists():
                return json.loads((paths.done / f"{name}.json").read_text())
            time.sleep(0.01)
        raise AssertionError([watcher.pending_reasons for watcher in watchers])

    body = (
        f'echo "$MV2FLOW_JOB_NODES $MV2FLOW_JOB_HOSTS ${{SLURM_JOB_ID:-}} ${{SLURM_PROCID:-none}}" '
        f">> {said}; ls {paths.gangs}/* > /dev/null"
    )
    record = run("gate-wide", "gpus=all nodes=2 lead=1", body)
    assert (record["rc"], record["watcher"], record["hosts"]) == (0, "batch-lead", [A, B])
    assert said.read_text() == f"2 {A},{B} {ALLOCATION} none\n"
    assert not any(paths.gangs.iterdir())
    # Without a lead a job of several nodes runs on each; the lead takes no node's job.
    each = run("gate-each", "gpus=all nodes=2", f'echo "part $MV2FLOW_JOB_NODE_RANK" >> {said}')
    assert each["rc"] == 0 and sorted(said.read_text().splitlines()[1:]) == ["part 0", "part 1"]
    assert run("gate-one", "gpus=1", "true")["watcher"] in (f"{A}-w", f"{B}-w")
    # A lead holds no device: the gate pool does not take it for a free node.
    job = placement.JobNeeds.from_script("x", "#spool: gpus=all nodes=2 lead=1\n")
    assert placement.assess(spool, job, stale_s=180, now=time.time()).devices == 4
    _scheduler(tmp_path / "alone", "batch", lead=True)
    alone = placement.assess(tmp_path / "alone", job, stale_s=180, now=time.time())
    assert alone.reasons == ["no watcher has ever heart-beat there"]
    # Nor for the nodes of one allocation beside the lead of another, a node short.
    for host, allocation, lead in (
        (A, "1", False),
        (B, "1", False),
        (A, "2", False),
        ("b", "2", True),
    ):
        _scheduler(tmp_path / "narrow", host, lead=lead, allocation=allocation)
    narrow = placement.assess(tmp_path / "narrow", job, stale_s=180, now=time.time())
    assert narrow.reasons == ["the job spans 2 nodes under a lead: no allocation here has them"]


def test_a_node_outlives_a_record_its_lead_removes_under_it(site):
    spool = Path(site["environment"]["spool"])
    lead, node = _scheduler(spool, "batch", lead=True), _scheduler(spool, "nodeB")
    root = sched.SpoolPaths(spool).gangs / "gate-x"
    root.mkdir(parents=True)
    record = {"leader": lead.watcher_id, "hosts": ["batch", "nodeA", "nodeB"], "once": True}
    record["members"] = [lead.watcher_id, "nodeA-w", node.watcher_id]
    (root / sched.GANG_RECORD).write_text(json.dumps(record))
    placed = node._place

    def gone_meanwhile(spec, now):
        shutil.rmtree(root)
        return placed(spec, now)

    node._place = gone_meanwhile
    node._serve_gangs(time.time())
    assert not root.exists() and node._reserved_for == ""


def test_a_node_says_it_is_ready_only_while_this_pass_reads_its_devices_clean(site):
    spool = Path(site["environment"]["spool"])
    paths = sched.SpoolPaths(spool)
    lead, node = _scheduler(spool, "batch", lead=True), _scheduler(spool, B)
    (paths.pending / "one.sh").write_text("#!/bin/bash\n#spool: gpus=1 wall=60\ntrue\n")
    node.tick()  # reads the devices clean and starts it
    node.jobs["one"].proc.wait(timeout=30)
    root = paths.gangs / "gate-x"
    root.mkdir(parents=True)
    record = {"leader": lead.watcher_id, "hosts": ["batch", A, B], "once": True}
    record["members"] = [lead.watcher_id, f"{A}-w", node.watcher_id]
    (root / sched.GANG_RECORD).write_text(json.dumps(record))
    ready = root / sched.GANG_READY / node.watcher_id
    left = {device: 9999 for device in node.config.devices}
    for reading, said in ((left, False), ({}, True), (left, False)):
        node.device_memory_mib = lambda reading=reading: {d: reading.get(d, 0) for d in "0123"}
        node.tick()
        assert ready.exists() is said, reading


# A job of some devices ranks before a job of whole nodes only by its priority.
@pytest.mark.parametrize("ask", ["gpus=all", "gpus=2 priority=1"])
def test_a_job_across_nodes_waits_its_turn_and_a_dead_leaders_record_holds_nothing(site, ask):
    spool = Path(site["environment"]["spool"])
    paths = sched.SpoolPaths(spool)
    lead = _scheduler(spool, "batch", lead=True)
    nodes = [_scheduler(spool, host) for host in (A, B)]
    # A record a session that is gone left behind, claimed and never ended.
    stale = paths.gangs / "gate-old"
    stale.mkdir(parents=True)
    (stale / sched.GANG_RECORD).write_text(json.dumps({"leader": "gone-lead", "members": []}))
    (stale / sched.GANG_GO).write_text("{}")
    for at, (name, header) in enumerate(
        (("gate-one", ask), ("gate-wide", "gpus=all nodes=2 lead=1"))
    ):
        job = paths.pending / f"{name}.sh"
        job.write_text(f"#!/bin/bash\n#spool: {header} wall=60\nsleep 0.2\n")
        os.utime(job, (1000 + at, 1000 + at))
    lead.tick()
    # The nodes' own job was queued first: forming now would empty both nodes ahead of it.
    assert lead.pending_reasons == {"gate-wide": "queued behind gate-one"}
    assert [p.name for p in paths.gangs.iterdir()] == ["gate-old"]
    nodes[0].tick()
    assert not (paths.pending / "gate-one.sh").exists()
    lead.tick()
    assert sorted(p.name for p in paths.gangs.iterdir()) == ["gate-wide"]
    formed = json.loads((paths.gangs / "gate-wide" / sched.GANG_RECORD).read_text())
    assert formed["leader"] == lead.watcher_id and formed["hosts"] == ["batch", A, B]


def test_a_job_the_lead_cannot_start_leaves_no_record_to_hold_its_nodes(site, monkeypatch):
    spool = Path(site["environment"]["spool"])
    paths = sched.SpoolPaths(spool)
    lead = _scheduler(spool, "batch", lead=True)
    nodes = [_scheduler(spool, host) for host in (A, B)]
    (paths.pending / "gate-wide.sh").write_text(
        "#!/bin/bash\n#spool: gpus=all nodes=2 lead=1 wall=60\ntrue\n"
    )
    lead.tick()
    for node in nodes:
        node.tick()

    def no_process(*args, **kwargs):
        raise OSError("Resource temporarily unavailable")

    with monkeypatch.context() as patched:
        patched.setattr(sched.subprocess, "Popen", no_process)
        lead.tick()
    assert json.loads((paths.done / "gate-wide.json").read_text())["reason"] == "launch_failed"
    assert not any(paths.gangs.iterdir())
    (paths.pending / "gate-one.sh").write_text("#!/bin/bash\n#spool: gpus=all wall=60\ntrue\n")
    nodes[0].tick()
    assert not (paths.pending / "gate-one.sh").exists() and not nodes[0].pending_reasons


def test_two_jobs_across_nodes_formed_in_one_moment_give_way(site, monkeypatch):
    spool = Path(site["environment"]["spool"])
    paths = sched.SpoolPaths(spool)
    lead = _scheduler(spool, "batch", lead=True)
    for host in (A, B):
        _scheduler(spool, host)
    (paths.pending / "gate-wide.sh").write_text(
        "#!/bin/bash\n#spool: gpus=all nodes=2 lead=1 wall=60\ntrue\n"
    )
    written = sched.write_json_atomic

    def beside_another(path, data):
        if path.name == sched.GANG_RECORD:  # a node forms a job of its own meanwhile
            (paths.gangs / "gate-test").mkdir()
        written(path, data)

    monkeypatch.setattr(sched, "write_json_atomic", beside_another)
    lead.tick()
    assert [p.name for p in paths.gangs.iterdir()] == ["gate-test"]
    assert lead.pending_reasons == {"gate-wide": "gate-test was formed in the same moment"}


def test_a_member_across_two_nodes_is_one_world_and_serve_is_judged_as_on_one(site, tmp_path):
    task = {**site, "features": [], "members": ["tep8"]}
    served = serve.run_serve_gate(
        resolve_environment(site, world_size=4).at_width(8), task_data=task, runner=_answers("", 1),
        features=[], unit="core", member="tep8", workdir=tmp_path / "gates",
    )  # fmt: skip
    # The engine is one step of eight ranks over both nodes; its driver another, of one.
    server, driver = _lines(tmp_path / "gates" / "serve.core.tep8.steps")
    across = '--nodes=2 --ntasks=8 --ntasks-per-node=4 --nodelist="${HOSTS[0]},${HOSTS[1]}"'
    assert "--mpi=pmix " in server and across in server
    assert " trtllm-llmapi-launch trtllm-serve " in server
    assert "TRTLLM_DISAGG_ROLE" not in server and " CUDA_VISIBLE_DEVICES=0,1,2,3 " in server
    assert ONE in driver
    # The driver is told where the script's server reports, and its word is judged as on one node.
    asked = json.loads((tmp_path / "gates" / "serve.core.tep8.json").read_text())
    assert asked["attach"]["address"].endswith("serve.core.tep8.steps/address.server")
    assert not served.passed and "the serve driver exited 1" in served.detail
    # Import records one file a node, read as the job's.
    for node in (A, B):
        log = tmp_path / f"job.resolve.jsonl.{node}"
        log.write_text(json.dumps({"module": "tensorrt_llm", "rank": node}) + "\n")
    assert [r["rank"] for r in provenance.read_records(tmp_path / "job.resolve.jsonl")] == [A, B]


def test_a_deployment_on_one_node_is_served_and_scored_with_no_lead_and_no_step(site, tmp_path):
    stated = _deployment(tmp_path / "one", (1, 1), (2, 1), timeout=120)
    task = {**site, "accuracy": {**site["accuracy"], "disagg": stated}}
    assert disagg.DisaggSpec.from_task(task).device_count("tep4") == 3
    environment = replace(resolve_environment(task, world_size=4), kind="local")

    def gate(said, code=0, task=task):
        return disagg.run_disagg_gate(
            environment, task_data=task, workdir=tmp_path / "gates",
            runner=_answers(said, code), unit="disagg", member="tep4",
        )  # fmt: skip

    result = gate(UP + "lm-eval gsm8k average accuracy: 41.70\n")
    assert result.passed, result.detail
    assert result.detail.startswith("2 server(s) on 1 node(s) and their proxy came up; ")
    assert (result.measured, result.reference, result.samples) == (41.70, 41.77, 1319)
    # Each server on devices of the job's own with its role, an engine's launch above one rank.
    directory = tmp_path / "gates" / "disagg.disagg.tep4"
    ctx, gen, client = _lines(directory)
    assert "srun --" not in ctx + gen + client
    assert ctx.startswith("{ CUDA_VISIBLE_DEVICES=${DEVICES[0]} ") and " mpirun " not in ctx
    assert " TRTLLM_DISAGG_ROLE=context " in ctx and ctx.count(" --server_role context ") == 1
    assert gen.startswith("{ CUDA_VISIBLE_DEVICES=${DEVICES[1]},${DEVICES[2]} ")
    assert " TRTLLM_DISAGG_ROLE=generation " in gen and " --server_role generation " in gen
    assert (
        " mpirun -n 2 --oversubscribe --allow-run-as-root trtllm-llmapi-launch trtllm-serve " in gen
    )
    # Its side's file with the gate's mode.
    config = yaml.safe_load((directory / "options.gen.yaml").read_text())
    assert config == dict(tensor_parallel_size=2, moe_expert_parallel_size=2, modeling_v2="require")
    assert "--servers ctx0,gen0 --timeout 120 gsm8k'" in client
    # The unit's protocol reaches the evaluator, and its score pattern picks the row.
    turns = {
        **task["accuracy"],
        "benchmark_args": ["--apply_chat_template", "--fewshot_as_multiturn"],
        "score_pattern": r"lm-eval gsm8k flexible-extract accuracy:\s*([0-9.]+)",
    }
    rows = "lm-eval gsm8k flexible-extract accuracy: 41.60\nlm-eval gsm8k average accuracy: 41.70\n"
    row = gate(UP + rows, task={**task, "accuracy": turns})
    assert row.passed and row.measured == 41.60, row.detail
    assert _lines(directory)[-1].endswith(" gsm8k --apply_chat_template --fewshot_as_multiturn'")
    # What the client says of servers that did not come up, or of a benchmark that failed, is the verdict.
    why = "ctx0 ended with 3 before every server was healthy; its log says: Failed to create NIXL backend"
    down = gate(
        disagg.disagg_driver.SESSION_MARKER + " " + json.dumps({"started": False, "error": why}), 1
    )
    assert not down.passed and down.detail == f"the deployment did not start: {why}"
    failed = gate(UP, 1)
    assert not failed.passed and failed.detail.endswith("; gsm8k through it: the client exited 1")
    held = dict(max_tokens=256, n=1, best_of=None, temperature=None, top_p=None, stop=None, seed=7)
    sent = disagg.disagg_driver.request_body("m", "p", SimpleNamespace(**held))
    assert sent == {**held, "model": "m", "prompt": "p", "stream": False, "temperature": 0}
    # Greedy to the API only where the engine is: with nothing of its sampling set.
    for knob, value, sends in (("top_k", 40, None), ("temperature", 0.6, 0.6)):
        asked = disagg.disagg_driver.request_body(
            "m", "p", SimpleNamespace(**{**held, knob: value})
        )
        assert (asked[knob], asked["temperature"]) == (value, sends)


def test_a_deployment_across_nodes_is_one_step_a_server_run_once_by_the_lead(site, tmp_path):
    assert disagg.DisaggSpec.from_task(site).device_count("tep4") is None
    result = disagg.run_disagg_gate(
        replace(resolve_environment(site, world_size=4), nodes=1),
        task_data=site,
        workdir=tmp_path / "gates",
        runner=_answers(UP + "lm-eval gsm8k average accuracy: 41.70\n"),
        unit="disagg",
        member="tep4",
    )
    assert result.passed, result.detail
    assert result.detail.startswith("3 server(s) on 2 node(s) and their proxy came up; ")
    directory = tmp_path / "gates" / "disagg.disagg.tep4"
    assert "run once by the session's lead" in (directory / "servers.sh").read_text()
    *servers, client = _lines(directory)
    # A step a server: its ranks, its node, its devices there, and its role in every rank.
    asked = [
        tuple(re.search(pattern, line)[1] for pattern in (
            r"--ntasks=(\d+) ", r'--nodelist="\$\{HOSTS\[(\d)\]\}" ', r" CUDA_VISIBLE_DEVICES=(\S+) ",
            r" TRTLLM_DISAGG_ROLE=(\w+) ", r" --server_role (\w+) ",
        ))
        for line in servers
    ]  # fmt: skip
    assert asked == [
        ("2", "0", "0,1", "context", "context"),
        ("2", "0", "2,3", "context", "context"),
        ("4", "1", "0,1,2,3", "generation", "generation"),
    ]
    for line in servers:
        assert "--mpi=pmix " in line and " UCX_TLS='^cuda_ipc' " in line
        assert " trtllm-llmapi-launch trtllm-serve " in line
    assert ONE in client
    assert "TRTLLM_DISAGG_ROLE" not in client and "--servers ctx0,ctx1,gen0 " in client
    # The proxy gets the deployment's own config with the servers' addresses.
    stated = {"max_retries": 2, "context_servers": {"num_instances": 2}}
    proxy = disagg.disagg_driver.proxy_config(stated, "/m", ["a:1", "a:2"], ["b:3"])
    assert (proxy["max_retries"], proxy["port"], proxy["model"]) == (2, 0, "/m")
    assert proxy["context_servers"] == {"num_instances": 2, "urls": ["a:1", "a:2"]}
    assert proxy["generation_servers"] == {"urls": ["b:3"]}


def test_preflight_refuses_what_no_gates_session_holds(site):
    held = site["environment"]["sessions"][0]
    wide = {**site, "features": [], "members": ["tep4", "dep16"]}
    wide["environment"] = {**site["environment"], "sessions": []}
    with pytest.raises(preflight.PreflightError, match="needs 4 nodes in one session_spool"):
        preflight.check_disagg_placement(wide, 16)
    wide["environment"]["sessions"] = [{**held, "nodes": 4}]
    assert preflight.check_disagg_placement(wide, 16) == []
    with pytest.raises(preflight.PreflightError, match="`tep6` is one world of 6 ranks"):
        preflight.check_disagg_placement({**wide, "members": ["tep6"]}, 6)
    variant = {**site, "features": [], "variants": {"wide": {"members": ["tep16"]}}}
    with pytest.raises(preflight.PreflightError, match="needs 4 nodes in one session_spool"):
        preflight.check_disagg_placement(variant, 4)
    short = {**site, "environment": {**site["environment"], "sessions": []}}
    with pytest.raises(preflight.PreflightError, match="needs 2 nodes in one session_spool"):
        preflight.check_disagg_placement(short, 4)
