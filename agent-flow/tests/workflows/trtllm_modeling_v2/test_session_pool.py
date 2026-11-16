# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A deployment's session on the pool: the allocation probe before it, what its job states."""

from __future__ import annotations

import json
import sys
import threading
import time
from types import SimpleNamespace

from agent_flow.workflows.trtllm_modeling_v2.checks import session
from agent_flow.workflows.trtllm_modeling_v2.infra.compute import probe as probe_mod
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.run.startup import bootstrap
from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract as contract_mod
from agent_flow.workflows.trtllm_modeling_v2.run.validation.check_runner import CheckRunner
from agent_flow.workflows.trtllm_modeling_v2.spec import deployments

MOVES = {"cache_transceiver_config": {"backend": "DEFAULT"}}
WIDE = {
    "deployments": [
        {"name": "tp2", "options": {"tensor_parallel_size": 2}},
        {
            "name": "pd",
            "ctx": {"options": MOVES},
            "gen": {"options": {"tensor_parallel_size": 4, **MOVES}},
        },
    ]
}


def _start(tmp_path, per_node=4):
    """What the start reads of a workflow and of preflight's result, with nothing else of them."""
    said: list[tuple[str, str]] = []
    environment = RunEnvironment(
        "slurm",
        sys.executable,
        spool=str(tmp_path / "spool"),
        gpus_per_node=per_node,
        env={"TMPDIR": str(tmp_path / "tmp")},
    )
    workflow = SimpleNamespace(
        task_data={**WIDE, "trtllm_repo_path": str(tmp_path / "tree")},
        workspace=tmp_path / "ws",
        journal=SimpleNamespace(
            event=lambda kind, text: said.append((kind, text)),
            section=lambda title, text: said.append((title, text)),
        ),
        _pool=object(),
    )
    return workflow, SimpleNamespace(environment=environment), said


def test_the_start_probes_the_widest_deployment_with_no_model_and_only_an_unread_watcher_stops_it(
    tmp_path, monkeypatch
):
    # The widest deployment holds five devices: on nodes of four, two whole nodes.
    assert bootstrap.allocation_shape(WIDE, 4) == (8, 2)
    assert bootstrap.allocation_shape(WIDE, 8) == (5, 1)
    assert bootstrap.allocation_shape(WIDE, None) == (5, 1)
    assert bootstrap.allocation_shape({"deployments": WIDE["deployments"][:1]}, 4) == (2, 1)
    asked: list[dict] = []
    unread: list[str] = []

    def probe(environment, **shape):
        asked.append(shape)
        problems = ["mpi4py does not import on rank 3"]
        return probe_mod.ProbeResult(ranks=shape["ranks"], nodes=shape["nodes"], problems=problems)

    monkeypatch.setattr(probe_mod, "run_allocation_probe", probe)
    monkeypatch.setattr(probe_mod, "watchers_without_reading", lambda spool: list(unread))
    workflow, result, said = _start(tmp_path)
    # What the probe finds is advice: it is said and journaled, and the start goes on.
    assert bootstrap._prove_allocation(workflow, result, None) is True
    assert (asked[0]["ranks"], asked[0]["nodes"]) == (8, 2)
    assert asked[0]["tree"] == str(tmp_path / "tree") and asked[0]["cache_dirs"] == [
        str(tmp_path / "tmp")
    ]
    # A directory nothing can pin is probed as it is.
    assert asked[0]["pin"] is None
    assert said[-1] == (
        "allocation probe",
        "allocation probe: 8 rank(s) on 2 node(s) failed: mpi4py does not import on rank 3",
    )
    # The one thing it stops for: a watcher that cannot read its devices starts no GPU job.
    unread.append("watcher w1 on nodeA has no device reading")
    assert bootstrap._prove_allocation(workflow, result, None) is False
    # With an injected runner no spool job is run, and nothing is probed.
    workflow._pool = None
    assert bootstrap._prove_allocation(workflow, result, None) is True and len(asked) == 2


class _Runner(CheckRunner):
    """A check runner with what `_gate` reads, whose jobs and sessions are recorded, not run."""

    def __init__(self, workflow, task_data, order):
        self.workflow, self.task_data, self.order = workflow, task_data, order
        self.workspace = workflow.workspace
        self.pin = None
        self.gate_log = SimpleNamespace(record=lambda result: None)
        self.journal = workflow.journal
        self._session = threading.local()

    def _job_runner(self, gpus, *, nodes=1, ranks=""):
        self.order.append(("job", gpus, nodes, ranks))
        return lambda *args, **kwargs: None

    def _relative_scores(self, deployment, marker):
        return {}

    def _dataset_paths(self, deployment):
        return {}

    def _gate_until_verdict(self, run, log):
        return run()


def test_the_first_session_waits_for_the_probe_and_its_job_states_each_instances_ranks(
    tmp_path, monkeypatch
):
    order: list = []
    release = threading.Event()

    def probe(environment, **shape):
        release.wait(10)
        order.append("probe answered")
        return probe_mod.ProbeResult(ranks=shape["ranks"], nodes=shape["nodes"])

    monkeypatch.setattr(probe_mod, "run_allocation_probe", probe)
    monkeypatch.setattr(probe_mod, "watchers_without_reading", lambda spool: [])
    workflow, result, _ = _start(tmp_path, per_node=8)
    workflow._gate_env = lambda: result.environment
    workflow._check_key = lambda check, commit="": "marker"
    stops: list[str] = []
    # The probe runs beside the run's first turns.
    bootstrap._prove_allocation_beside(workflow, result, None, stops.append)

    def ran(deployment, environment, **kwargs):
        order.append(("session", deployment.name))
        line = SimpleNamespace(passed=True, summary=lambda: "ok", stdout_tail="")
        return session.SessionResult((), line)

    monkeypatch.setattr(session, "run_deployment", ran)
    runner = _Runner(workflow, workflow.task_data, order)
    check = contract_mod.Check("pd", session.GATE_NAME, "pd5")
    gate = threading.Thread(target=lambda: runner._gate(check, None))
    gate.start()
    time.sleep(0.2)
    assert order == [], "no session job is asked for while the probe has not answered"
    release.set()
    gate.join(10)
    # The job states every device of the session and the ranks of each role instance.
    assert order == ["probe answered", ("job", 5, 1, "ctx0:1,gen0:4"), ("session", "pd")]
    assert stops == []
    pd = deployments.named(workflow.task_data, "pd")
    assert session.stated_ranks(pd) == "ctx0:1,gen0:4"
    assert session.stated_ranks(deployments.named(workflow.task_data, "tp2")) == "server:2"


def _checkout(tmp_path):
    """A git checkout whose run keeps trees: it is pinned, as every job's tree is."""
    import subprocess

    from agent_flow.workflows.trtllm_modeling_v2.infra.compute import pintree

    repo = tmp_path / "checkout"
    (repo / "tensorrt_llm").mkdir(parents=True)
    (repo / "tensorrt_llm" / "__init__.py").write_text("WHERE = 'here'\n")
    git = ["git", "-C", str(repo), "-c", "user.name=T", "-c", "user.email=t@example.com"]
    for words in (["init", "-q", "-b", "main"], ["add", "."], ["commit", "-q", "-m", "base"]):
        subprocess.run([*git, *words], check=True, capture_output=True)
    config = {"repo": str(repo), "root": str(tmp_path / "trees")}
    pintree.write_config(repo, config)
    return repo


def test_the_probe_and_every_session_job_name_the_pin_they_run(tmp_path, monkeypatch):
    from agent_flow.workflows.trtllm_modeling_v2.infra.compute import pintree
    from agent_flow.workflows.trtllm_modeling_v2.infra.compute.scheduler import SpoolPaths
    from agent_flow.workflows.trtllm_modeling_v2.run.validation import check_runner

    repo = _checkout(tmp_path)
    # The start's probe: the checkout as it stands now, pinned by the run's trees.
    asked: list[dict] = []

    def probe(environment, **shape):
        asked.append(shape)
        return probe_mod.ProbeResult(ranks=shape["ranks"], nodes=shape["nodes"])

    monkeypatch.setattr(probe_mod, "run_allocation_probe", probe)
    monkeypatch.setattr(probe_mod, "watchers_without_reading", lambda spool: [])
    workflow, result, _ = _start(tmp_path)
    workflow.task_data["trtllm_repo_path"] = str(repo)
    assert bootstrap._prove_allocation(workflow, result, None) is True
    # Once a workspace and a shape: a start of the same workspace asks nothing again,
    # and a spec whose widest deployment changed asks for the new shape. So does one
    # whose queue changed: another QoS may refuse what this one gave.
    assert bootstrap._prove_allocation(workflow, result, None) is True and len(asked) == 1
    kept = json.loads((workflow.workspace / "alloc_probe" / "pass.json").read_text())
    (workflow.workspace / "alloc_probe" / "pass.json").write_text(
        json.dumps({**kept, "ranks": kept["ranks"] + 1})
    )
    assert bootstrap._prove_allocation(workflow, result, None) is True and len(asked) == 2
    workflow.task_data.setdefault("environment", {}).setdefault("slurm", {})["qos"] = "other"
    assert bootstrap._prove_allocation(workflow, result, None) is True and len(asked) == 3
    assert bootstrap._prove_allocation(workflow, result, None) is True and len(asked) == 3
    pin_file = asked[0]["pin"]
    assert pin_file and pintree.read_pin(pin_file)["commit"]
    monkeypatch.undo()
    # Across nodes the probe's job names that pin, and each rank's step runs the pin's
    # prelude instead of entering a tree by its path.
    spool = tmp_path / "spool"
    environment = RunEnvironment(
        "slurm", "/venv/bin/python", spool=str(spool), slurm={"gpus_per_node": 4}
    )
    threading.Thread(
        target=probe_mod._run_across_nodes,
        args=(environment, ["probe.py"], 8, 2, str(repo), 5, None),
        kwargs={"pin": pin_file},
        daemon=True,
    ).start()
    deadline = time.time() + 10
    queued: list = []
    while not queued and time.time() < deadline:
        time.sleep(0.05)
        queued = list((spool / "pending").glob("alloc-probe-*.sh"))
    script, name = queued[0].read_text(), queued[0].stem
    (header,) = [line for line in script.splitlines() if line.startswith("#spool:")]
    assert f"pin={pin_file}" in header and "nodes=2" in header
    (step,) = [line for line in script.splitlines() if line.startswith("srun --overlap ")]
    record = SpoolPaths(spool).tree_log(name)
    assert (
        f'--nodes=2 --ntasks=8 --ntasks-per-node=4 --nodelist="$MV2FLOW_JOB_HOSTS" '
        f"/venv/bin/python {pintree.__file__} run --pin {pin_file} --job {name} "
        f"--record {record} -- bash "
    ) in step
    inner = next((spool / "jobs").glob("alloc-probe-*/probe.sh")).read_text()
    assert "cd " not in inner, "a pinned job never enters a tree by its path"
    queued[0].unlink()
    # A check's jobs name the pin the check measures.
    seen: list[dict] = []

    def executor(env, **asked_for):
        seen.append(asked_for)
        return SimpleNamespace()

    monkeypatch.setattr(check_runner, "select_command_executor", executor)
    monkeypatch.setattr(check_runner, "shell_runner_for", lambda found: found)
    runner = _Runner.__new__(_Runner)
    runner.workflow = SimpleNamespace(
        _gate_env=lambda: environment, _shell_runner_override=None, _shell_runner=None
    )
    runner.task_data = {"trtllm_repo_path": str(repo)}
    runner.pin, runner._member_runners = None, {}
    runner.journal = SimpleNamespace(event=lambda *said: None)
    runner._noting_session_loss = lambda found: found
    view = CheckRunner._on_snapshot.__get__(runner)
    pinned = SimpleNamespace(
        spec=lambda data, tracked: dict(data),
        snapshots=SimpleNamespace(tracked=lambda commit: ()),
        commit="c0ffee",
        runner=lambda found, executor: found,
    )
    on_pin = view(pinned, pin_file)
    CheckRunner._job_runner(on_pin, 2, ranks="server:2")
    assert seen[0]["pin"] == pin_file and seen[0]["ranks"] == "server:2"
    CheckRunner._job_runner(runner, 2)
    assert seen[1]["pin"] is None


def test_a_reserved_nodes_memory_is_its_jobs_while_the_lead_runs_it_and_held_again_after(tmp_path):
    """On GPUs a session's own engines were reported as `POOL_DEVICE_HELD` on both of its nodes."""
    from agent_flow.workflows.trtllm_modeling_v2.infra.compute import pool as pool_mod
    from agent_flow.workflows.trtllm_modeling_v2.infra.compute.scheduler import SpoolPaths

    from .test_spool_scheduler import (
        _lead,
        _node,
        _pump,
        _spool_scheduler_blocking,
        _spool_scheduler_rc,
        _spool_scheduler_release,
        _spool_scheduler_submit,
    )

    spool = tmp_path / "spool"
    SpoolPaths(spool).ensure()
    at = [time.time()]
    clock = lambda: at[0]  # noqa: E731
    used = {"mib": 0}
    nodes = [_node(spool, clock, "nodeA"), _node(spool, clock, "nodeB")]
    for node in nodes:
        # Devices 0 and 1 of each node carry what the job's steps load there.
        node.device_memory_mib = lambda: {
            device: used["mib"] if device in ("0", "1") else 0 for device in ("0", "1", "2", "3")
        }
        node.device_holders = lambda: {}
    everyone = [*nodes, _lead(spool, clock)]
    lead = everyone[-1]
    body = _spool_scheduler_blocking(spool, "session")
    _spool_scheduler_submit(spool, "session", body, header="nodes=2 wall=600")
    _pump(everyone, lambda: "session" in lead.jobs)
    # The lead runs the job; its steps load engines on the reserved nodes' devices.
    used["mib"] = 3655

    def later(seconds: int) -> None:
        # In steps short enough that every watcher's heartbeat stays fresh to the others.
        for _ in range(seconds // 10):
            at[0] += 10
            _pump(everyone, lambda: True)

    later(270)  # past a device reading's age and the time after which held memory is reported
    for node in nodes:
        said = node.snapshot(clock())
        assert said["reserved_for"] == "session" and said["held"] == {}
    watchers = [node.snapshot(clock()) for node in nodes]
    assert pool_mod.device_events(watchers, clock()) == []
    # The job ends and its receipt releases the nodes: memory still there is then nobody's,
    # and is reported as before.
    _spool_scheduler_release(spool, "session")
    _pump(everyone, lambda: _spool_scheduler_rc(spool, "session") is not None)
    later(270)
    for node in nodes:
        said = node.snapshot(clock())
        assert not said["reserved_for"] and sorted(said["held"]) == ["0", "1"]
    events = pool_mod.device_events([node.snapshot(clock()) for node in nodes], clock())
    assert len(events) == 4 and "holds 3655 MiB that no job of its watcher accounts for" in str(
        events[0]
    )
