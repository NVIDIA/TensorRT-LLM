# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A member wider than one node: one engine across the nodes of a multi-node spool job."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_flow.workflows.trtllm_modeling_v2.checks.verdict import GateResult
from agent_flow.workflows.trtllm_modeling_v2.infra.compute import GANG_PATH, gang, placement
from agent_flow.workflows.trtllm_modeling_v2.infra.compute import executor as executor_mod
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import (
    RunEnvironment,
    resolve_environment,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.scheduler import (
    RC_MIXED_CHECKOUT,
    Scheduler,
    SchedulerConfig,
    SpoolPaths,
    part_name,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.server_runtime import (
    GANG_MPI_HOSTS_ENV,
    gang_mpi_hosts,
    rank_launch,
)
from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract as contract_mod
from agent_flow.workflows.trtllm_modeling_v2.run.validation.check_runner import CheckRunner
from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow
from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import (
    member_nodes,
    nodes_for_width,
    widest_spanning_width,
)

_DEVICES = ("0", "1", "2", "3")
_ALLOCATION = "5151"


# ------------------------------------------------------------------ geometry


def test_an_engine_spans_as_many_nodes_as_its_width_needs():
    assert nodes_for_width(8, 4) == 2
    assert nodes_for_width(4, 4) == 1
    assert nodes_for_width(6, 4) == 2
    assert nodes_for_width(16, 4) == 4
    # Undeclared devices per node keep every engine on one node, as before.
    assert nodes_for_width(8, None) == 1
    spec = {"environment": {"kind": "session_spool", "gpus_per_node": 4}}
    assert member_nodes(spec, "tep8") == 2
    assert member_nodes(spec, "dep4") == 1
    assert member_nodes({"environment": {"sessions": [{"gpus_per_node": 8}]}}, "tep8") == 1
    assert member_nodes(spec, None) == 1
    # The widest member that spans is what a CI stage has to hold; none when all fit a node.
    unified = {**spec, "parallel": "unified"}
    assert widest_spanning_width({**unified, "members": ["tp4", "tep8", "dep16"]}) == 16
    assert (
        widest_spanning_width({**unified, "members": {"default": ["tp4"], "mtp": ["tep8"]}}) is None
    )
    wide_feature = {
        **unified,
        "members": {"default": ["tp4"], "mtp": ["tep8"]},
        "features": ["mtp"],
    }
    assert widest_spanning_width(wide_feature) == 8
    assert widest_spanning_width({**unified, "members": ["tp4", "tep4"]}) is None
    assert widest_spanning_width({"parallel": "unified", "members": ["tep8"]}) is None


# ------------------------------------------------------------------- launch


def test_rank_launch_spreads_the_ranks_over_the_gang_hosts_only_when_given():
    single = rank_launch(4, "trtllm-llmapi-launch")
    assert single[single.index("mpirun") :] == [
        "mpirun", "-n", "4", "--oversubscribe", "--allow-run-as-root", "trtllm-llmapi-launch",
    ]  # fmt: skip
    spread = rank_launch(8, "trtllm-llmapi-launch", hosts="a:4,b:4")
    assert spread[spread.index("mpirun") :] == [
        "mpirun", "-n", "8", "--host", "a:4,b:4", "--map-by", "slot",
        "--allow-run-as-root", "trtllm-llmapi-launch",
    ]  # fmt: skip
    # The Slurm task variables are cleared in both, ahead of mpirun.
    assert spread[:3] == ["env", "-u", "SLURM_PROCID"]
    assert rank_launch(1, "x", hosts="a:4,b:4") == []
    assert gang_mpi_hosts({GANG_MPI_HOSTS_ENV: "a:4,b:4"}) == "a:4,b:4"
    assert gang_mpi_hosts({}) is None


def _spool_env(tmp_path, **changes) -> RunEnvironment:
    task = {
        "environment": {
            "kind": "session_spool",
            "python": "/venv/bin/python",
            "spool": str(tmp_path / "spool"),
            "gpus_per_node": 4,
            "env": {"TMPDIR": "/scratch/t"},
        }
    }
    task["environment"].update(changes)
    return resolve_environment(task, world_size=8)


def test_a_member_wider_than_a_node_is_launched_through_the_gang_on_every_node(tmp_path):
    run = _spool_env(tmp_path)
    assert run.gpus_per_node == 4 and run.nodes == 1
    wide = run.at_width(8)
    assert wide.nodes == 2 and wide.world_size == 8
    command = wide.python_command(["engine_driver.py", "--spec", "s.json"])
    lead = f"/venv/bin/python {GANG_PATH} lead --nodes 2 --ranks 8 --"
    assert command == (
        f'TMPDIR=/scratch/t PATH=/venv/bin:"$PATH" {lead} trtllm-llmapi-launch '
        "/venv/bin/python engine_driver.py --spec s.json"
    )
    # No single-node launch is left inside it: the gang chooses the hosts.
    assert "--oversubscribe" not in command
    # Work that starts its own ranks (a server) runs on node 0 alone, MPI-scrubbed, at one task.
    served = wide.python_command(["serve_driver.py", "--spec", "s.json"], single_process=True)
    assert served.startswith(
        f"TLLM_WORKER_USE_SINGLE_PROCESS=1 TMPDIR=/scratch/t /venv/bin/python {GANG_PATH} "
        "lead --nodes 2 -- bash -c "
    )
    assert "exec /venv/bin/python serve_driver.py --spec s.json" in served
    assert "--ranks" not in served
    # Its scrub of the MPI environment keeps the launch agent node 0 set for its mpirun.
    assert '[ "$v" != OMPI_MCA_plm_rsh_agent ]' in served
    # A member that fits a node launches exactly as before.
    narrow = run.at_width(4)
    assert narrow.nodes == 1
    assert "mpirun -n 4 --oversubscribe" in narrow.command("x")
    assert str(GANG_PATH) not in narrow.command("x")


def test_the_prompt_says_how_to_run_an_engine_across_nodes(tmp_path):
    run = _spool_env(tmp_path)
    text = run.describe()
    assert "wider than one node** (4 GPUs): 8 ranks span 2 nodes" in text
    assert "`--nodes 2` and no `--gpus`" in text
    assert f"{GANG_PATH} lead --nodes 2 --ranks 8 -- trtllm-llmapi-launch <your command>" in text
    assert "mpirun -n 4 --oversubscribe" in text
    # The single-node multi-rank block would name an 8-rank launch on one node: not said.
    assert "mpirun -n 8 --oversubscribe" not in text
    one_node = _spool_env(tmp_path, gpus_per_node=8).describe()
    assert "wider than one node" not in one_node
    assert "mpirun -n 8 --oversubscribe" in one_node
    # Of several sessions, the prompt names the ones that hold the engine's nodes.
    sessions = [
        {"name": "wide", "spool": "/s/wide", "role": "gates", "nodes": 2, "gpus_per_node": 4},
        {"name": "narrow", "spool": "/s/narrow", "role": "gates", "nodes": 1},
        {"name": "bench", "spool": "/s/bench", "role": "work", "nodes": 2},
    ]
    several = _spool_env(tmp_path, sessions=sessions).describe()
    assert "to a session of that many nodes (`wide`: `SPOOL=/s/wide`); on any other" in several


# --------------------------------------------------------------- gang.lead()


def _gang_environ(tmp_path, rank: int, **changes) -> dict[str, str]:
    root = tmp_path / "spool" / "gangs" / "wide"
    root.mkdir(parents=True, exist_ok=True)
    environ = {
        "PATH": os.environ.get("PATH", ""),
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        gang.NODES_ENV: "2",
        gang.NODE_RANK_ENV: str(rank),
        gang.HOSTS_ENV: "nodeA,nodeB",
        gang.GANG_DIR_ENV: str(root),
    }
    environ.update(changes)
    return environ


def test_node_zero_leads_with_the_hosts_and_the_launch_agent(tmp_path):
    environ = _gang_environ(tmp_path, 0)
    argv, env = gang.lead(
        ["trtllm-llmapi-launch", "python", "x.py"], nodes=2, ranks=8, environ=environ
    )
    mpirun = argv[argv.index("mpirun") :]
    assert mpirun == [
        "mpirun", "-n", "8", "--host", "nodeA:4,nodeB:4", "--map-by", "slot",
        "--allow-run-as-root", "trtllm-llmapi-launch", "python", "x.py",
    ]  # fmt: skip
    agent = Path(environ[gang.GANG_DIR_ENV]) / gang.RELAY_DIRNAME / gang.AGENT_NAME
    assert env["PRTE_MCA_plm_ssh_agent"] == env["OMPI_MCA_plm_rsh_agent"] == str(agent)
    assert env["PRTE_MCA_plm_ssh_no_tree_spawn"] == "1"
    assert env[GANG_MPI_HOSTS_ENV] == "nodeA:4,nodeB:4"
    assert os.access(agent, os.X_OK)
    assert f"{GANG_PATH} agent" in agent.read_text()
    # A process that starts its own ranks runs as given, the hosts in its environment.
    argv, env = gang.lead(["python", "serve_driver.py"], nodes=2, environ=environ)
    assert argv == ["python", "serve_driver.py"]
    assert env[GANG_MPI_HOSTS_ENV] == "nodeA:4,nodeB:4"


def test_every_other_node_serves_and_a_job_of_another_size_is_refused(tmp_path):
    assert gang.lead(["x"], nodes=2, ranks=8, environ=_gang_environ(tmp_path, 1)) is None
    with pytest.raises(gang.GangError, match="spans 2 node\\(s\\) and its job holds 1"):
        gang.lead(["x"], nodes=2, ranks=8, environ={"PATH": "/bin"})
    with pytest.raises(gang.GangError, match="spans 2 node\\(s\\) and its job holds 3"):
        gang.lead(
            ["x"], nodes=2, ranks=8, environ=_gang_environ(tmp_path, 0, **{gang.NODES_ENV: "3"})
        )
    # Every node but the last is filled, and the last holds a rank.
    with pytest.raises(gang.GangError, match="4 ranks do not span 2 nodes of 4 devices"):
        gang.lead(["x"], nodes=2, ranks=4, environ=_gang_environ(tmp_path, 0))
    with pytest.raises(gang.GangError, match="9 ranks do not span 2 nodes"):
        gang.lead(["x"], nodes=2, ranks=9, environ=_gang_environ(tmp_path, 0))
    # A scheduler that does not say where the job's directory is predates this launcher.
    environ = _gang_environ(tmp_path, 0)
    del environ[gang.GANG_DIR_ENV]
    with pytest.raises(gang.GangError, match="older than this launcher"):
        gang.lead(["x"], nodes=2, ranks=8, environ=environ)
    # Outside a multi-node job, one node launches as it always has.
    argv, _ = gang.lead(["trtllm-llmapi-launch", "a"], nodes=1, ranks=4, environ={"PATH": "/bin"})
    assert argv[argv.index("mpirun") :][:4] == ["mpirun", "-n", "4", "--oversubscribe"]


# ----------------------------------------------------------------- the relay


def _run_gang(*args: str, environ: dict[str, str]) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, str(GANG_PATH), *args],
        env=environ,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def test_a_withdrawn_command_is_stopped_and_the_node_ends_with_node_zero(tmp_path):
    member = _gang_environ(tmp_path, 1)
    root = Path(member[gang.GANG_DIR_ENV])
    serving = _run_gang("lead", "--nodes", "2", "--", "true", environ=member)
    agent = _run_gang("agent", "nodeB", "sleep", "30", environ=_gang_environ(tmp_path, 0))
    inbox = root / gang.RELAY_DIRNAME / "1"
    deadline = time.monotonic() + 15
    while not list(inbox.glob("*.json")) and time.monotonic() < deadline:
        time.sleep(0.05)
    (request,) = inbox.glob("*.json")
    assert json.loads(request.read_text())["command"] == "sleep 30"
    time.sleep(0.5)
    agent.send_signal(signal.SIGTERM)
    assert agent.wait(timeout=10) == 128 + signal.SIGTERM
    rc_file = request.with_suffix(".rc")
    while not rc_file.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert int(rc_file.read_text()) == -signal.SIGTERM
    # Node 0's part ends: the serving node ends with it, a success.
    (root / "parts").mkdir()
    (root / "parts" / "0.json").write_text("{}")
    assert serving.wait(timeout=10) == 0
    assert "running" in serving.stdout.read()


def test_the_agent_gives_up_when_the_node_it_relays_to_has_ended(tmp_path):
    environ = _gang_environ(tmp_path, 0)
    root = Path(environ[gang.GANG_DIR_ENV])
    (root / "parts").mkdir()
    (root / "parts" / "1.json").write_text("{}")
    assert gang.agent("nodeB", ["true"], environ) == gang.RC_NODE_GONE
    assert gang.agent("nodeZ", ["true"], environ) == gang.RC_NODE_GONE
    # A job whose directory the scheduler has removed is over: nothing makes it again.
    gone = tmp_path / "spool" / "gangs" / "ended"
    environ[gang.GANG_DIR_ENV] = str(gone)
    assert gang.agent("nodeB", ["true"], environ) == gang.RC_NODE_GONE
    assert gang.serve(gone, 1) == 0
    assert not gone.exists()
    # Open MPI may name a host by its short name.
    assert gang._rank_of("nodeB", ["nodeA.cluster", "nodeB.cluster"]) == 1
    assert gang._rank_of("nodeB.cluster", ["nodeA", "nodeB"]) == 1


# ------------------------------------------------- a multi-node job, end to end

_FAKE_MPIRUN = r"""#!/bin/bash
# Stands in for Open MPI's mpirun: records its arguments, starts every other host's share
# through the launch agent the environment names (as PRRTE's ssh launcher does), then node 0's.
printf '%s\n' "$*" > "$OUT/mpirun.args"
hosts=""
while [ $# -gt 0 ]; do
  case "$1" in
    -n|--map-by) shift 2 ;;
    --host) hosts=$2; shift 2 ;;
    --allow-run-as-root|--oversubscribe) shift ;;
    *) break ;;
  esac
done
[ "$PRTE_MCA_plm_ssh_agent" = "$OMPI_MCA_plm_rsh_agent" ] || exit 90
[ "$PRTE_MCA_plm_ssh_no_tree_spawn" = 1 ] || exit 91
rc=0
IFS=, read -ra entries <<< "$hosts"
for entry in "${entries[@]:1}"; do
  "$PRTE_MCA_plm_ssh_agent" "${entry%%:*}" "$@" || rc=$?
done
"$@" || rc=$?
exit $rc
"""

_FAKE_RANKS = r"""#!/bin/bash
# One node's ranks: say where they ran and what the command line gave them.
echo "$MV2FLOW_JOB_NODE_RANK $MARK $CUDA_VISIBLE_DEVICES" > "$OUT/node$MV2FLOW_JOB_NODE_RANK.txt"
if [ "$MV2FLOW_JOB_NODE_RANK" = 1 ] && [ -n "${FAIL_REMOTE:-}" ]; then exit 3; fi
"""


@pytest.fixture
def gang_spool(tmp_path):
    root = tmp_path / "spool"
    SpoolPaths(root).ensure()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, text in (
        ("mpirun", _FAKE_MPIRUN),
        ("trtllm-llmapi-launch", '#!/bin/bash\nexec "$@"\n'),
        ("ranks", _FAKE_RANKS),
    ):
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    out = tmp_path / "out"
    out.mkdir()
    return SimpleNamespace(root=root, bin=bin_dir, out=out)


def _node(spool, host: str) -> Scheduler:
    environ = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "MV2FLOW_"))}
    environ.update({"SLURM_JOB_ID": _ALLOCATION, "PATH": f"{spool.bin}:{environ.get('PATH', '')}"})
    environ["OUT"] = str(spool.out)
    config = SchedulerConfig(devices=_DEVICES, leftover_grace_s=0.0)
    node = Scheduler(spool.root, config, environ=environ, watcher_id=f"{host}-w")
    node.host = host
    node.device_memory_mib = lambda: {d: 0 for d in _DEVICES}
    node.heartbeat()
    return node


def _pump(nodes, until, timeout: float = 60.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        for node in nodes:
            node.tick()
        if until():
            return
        time.sleep(0.02)
    raise AssertionError(f"not reached; status: {[n.snapshot(0) for n in nodes]}")


def _submit(spool, name: str, command: str) -> None:
    text = f"#!/bin/bash\nset -u\n#spool: gpus=all nodes=2 wall=120s\n{command}\n"
    staging = spool.root / "pending" / f".{name}.tmp"
    staging.write_text(text)
    os.replace(staging, spool.root / "pending" / f"{name}.sh")


def _receipt(spool, name: str) -> dict | None:
    paths = SpoolPaths(spool.root)
    if not paths.receipt(name).exists():
        return None
    return json.loads(paths.record(name).read_text())


def _wide_command(**env: str) -> str:
    environment = RunEnvironment(
        kind="session_spool", python=sys.executable, env=env, gpus_per_node=4
    ).at_width(8)
    # The interpreter's own directory goes first on PATH: keep the stand-ins ahead of it.
    return environment.command("ranks").replace(f"{Path(sys.executable).parent}:", "", 1)


def test_one_engine_spans_the_nodes_of_a_multi_node_job(gang_spool):
    a, b = _node(gang_spool, "nodeA"), _node(gang_spool, "nodeB")
    _submit(gang_spool, "wide", _wide_command(MARK="from-the-command-line"))
    _pump([a, b], lambda: _receipt(gang_spool, "wide") is not None)
    receipt = _receipt(gang_spool, "wide")
    assert receipt["rc"] == 0, (gang_spool.root / "logs" / "wide.log").read_text()
    assert {p["rank"]: p["rc"] for p in receipt["parts"]} == {0: 0, 1: 0}
    args = (gang_spool.out / "mpirun.args").read_text().split()
    assert args[:6] == ["-n", "8", "--host", "nodeA:4,nodeB:4", "--map-by", "slot"]
    # Node 1's share ran in node 1's part: its rank, its devices, the job's own command line.
    assert (gang_spool.out / "node1.txt").read_text().split() == [
        "1",
        "from-the-command-line",
        "0,1,2,3",
    ]
    assert (gang_spool.out / "node0.txt").read_text().split()[0] == "0"
    member_log = (gang_spool.root / "logs" / "wide.node1.log").read_text()
    assert "node 1: running" in member_log
    # The job's directory, relay included, is gone with its receipt.
    assert not (gang_spool.root / "gangs" / "wide").exists()


def test_a_failure_on_another_node_is_the_jobs_failure(gang_spool):
    a, b = _node(gang_spool, "nodeA"), _node(gang_spool, "nodeB")
    _submit(gang_spool, "wide", _wide_command(MARK="m", FAIL_REMOTE="1"))
    _pump([a, b], lambda: _receipt(gang_spool, "wide") is not None)
    receipt = _receipt(gang_spool, "wide")
    # mpirun on node 0 reports what its remote share returned, and the receipt names node 0.
    assert receipt["rc"] == 3 and receipt["detail"].startswith("node 0 (nodeA)")
    assert {p["rank"]: p["rc"] for p in receipt["parts"]}[0] == 3


def test_what_another_nodes_ranks_imported_is_judged_with_node_zeros(gang_spool, tmp_path):
    """Node 1's ranks import the package from another checkout: the job describes no one tree."""
    trees = {}
    for name in ("tree", "other"):
        package = tmp_path / name / "tensorrt_llm"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(f"WHERE = {name!r}\n", encoding="utf-8")
        trees[name] = tmp_path / name
    # Each node's ranks import from their working directory: the job's tree, except on node 1.
    (gang_spool.bin / "ranks").write_text(
        "#!/bin/bash\n"
        f'[ "$MV2FLOW_JOB_NODE_RANK" = 1 ] && cd {trees["other"]}\n'
        f"exec {sys.executable} -c 'import tensorrt_llm'\n"
    )
    a, b = _node(gang_spool, "nodeA"), _node(gang_spool, "nodeB")
    name = "wide"
    text = (
        f"#!/bin/bash\nset -u\n#spool: gpus=all nodes=2 wall=120s tree={trees['tree']}\n"
        f"{_wide_command()}\n"
    )
    staging = gang_spool.root / "pending" / f".{name}.tmp"
    staging.write_text(text)
    os.replace(staging, gang_spool.root / "pending" / f"{name}.sh")
    _pump([a, b], lambda: _receipt(gang_spool, name) is not None)
    receipt = _receipt(gang_spool, name)
    # Every part exited 0; what node 1's ranks imported is in its own part's log of imports.
    assert {p["rank"]: p["rc"] for p in receipt["parts"]} == {0: 0, 1: 0}
    paths = SpoolPaths(gang_spool.root)
    assert str(trees["other"]) in paths.resolve_log(part_name(name, 1)).read_text()
    assert str(trees["other"]) not in paths.resolve_log(name).read_text()
    assert receipt["rc"] == RC_MIXED_CHECKOUT and receipt["reason"] == "mixed_checkout"
    assert str(trees["other"]) in receipt["detail"] and str(trees["tree"]) in receipt["detail"]


# ------------------------------------------------------------ placement


def test_a_multi_node_job_is_not_sent_to_a_spool_with_too_few_nodes(gang_spool):
    job = placement.JobNeeds.from_script("wide", "#spool: gpus=all nodes=2\ntrue\n")
    assert job.nodes == 2
    now = time.time()
    _node(gang_spool, "nodeA")
    view = placement.assess(gang_spool.root, job, stale_s=180, now=now)
    assert not view.can_take
    assert "asks for 2 nodes of one allocation and this spool has 1 live" in view.reasons[0]
    _node(gang_spool, "nodeB")
    assert placement.assess(gang_spool.root, job, stale_s=180, now=now).can_take
    one = placement.JobNeeds.from_script("one", "#spool: gpus=4\ntrue\n")
    assert one.nodes == 1


# ----------------------------------------------- check job construction


def _runner(tmp_path, *, gpus_per_node: int | None = 4) -> CheckRunner:
    workflow = TrtllmModelingV2Workflow(workspace=tmp_path / "ws")
    workflow.environment = RunEnvironment(
        kind="session_spool",
        python="python3",
        world_size=8,
        spool=str(tmp_path / "spool"),
        gpus_per_node=gpus_per_node,
    )
    workflow.task_data = {
        "trtllm_repo_path": str(tmp_path),
        "parallel": "unified",
        "members": ["tep4", "tep8"],
        "accuracy": {"disagg": {"test": "tests/t.py::T::test_d", "nodes": 2}},
    }
    return CheckRunner(workflow)


def test_a_member_check_wider_than_a_node_is_a_multi_node_job(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    assert runner._member_env("tep8").nodes == 2
    assert runner._member_env("tep4").nodes == 1
    boot = contract_mod.Check("core", "boot", "tep8", contract_mod.TIER_INNER)
    accuracy = contract_mod.Check("core", "accuracy[gsm8k]", "tep8", contract_mod.TIER_INNER)
    narrow = contract_mod.Check("core", "boot", "tep4", contract_mod.TIER_INNER)
    disagg = contract_mod.Check("disagg", "disagg", "tep4", contract_mod.TIER_INNER)
    assert [runner._gate_nodes(c) for c in (boot, accuracy, narrow, disagg)] == [2, 2, 1, 2]

    # The job the check queues: the gang's whole nodes, the member's ranks across them.
    queued = {}

    def queue_job(root, name, script, **_):
        queued[name] = script

    monkeypatch.setattr(executor_mod.spoolctl, "queue_job", queue_job)
    monkeypatch.setattr(
        executor_mod.spoolctl, "wait_for_job", lambda *a, **k: {"rc": 0, "reason": "exit"}
    )
    command = runner._member_env("tep8").python_command(["engine_driver.py"])
    result = runner._member_runner("tep8", nodes=runner._gate_nodes(boot))(command, None, 600)
    assert result.returncode == 0
    (script,) = queued.values()
    header = next(line for line in script.splitlines() if line.startswith("#spool:"))
    assert "gpus=all" in header and "nodes=2" in header
    assert f"{GANG_PATH} lead --nodes 2 --ranks 8 -- trtllm-llmapi-launch" in script

    # _gate hands the gate's runner that many nodes.
    asked = []
    monkeypatch.setattr(
        runner, "_member_runner", lambda member, gpus=None, *, nodes=1: asked.append(nodes)
    )
    monkeypatch.setattr(
        runner,
        "_gate_until_verdict",
        lambda run, log: GateResult(name="boot", member="tep8", passed=True, detail="ok"),
    )
    runner._gate(boot, None)
    runner._gate(narrow, None)
    assert asked == [2, 1]


def test_without_devices_per_node_every_member_stays_on_one_node(tmp_path):
    runner = _runner(tmp_path, gpus_per_node=None)
    assert runner._member_env("tep8").nodes == 1
    assert "mpirun -n 8 --oversubscribe" in runner._member_env("tep8").command("x")


def test_a_recorded_substrate_pass_stands_whatever_the_new_environment_fields_say(tmp_path):
    from dataclasses import replace

    from agent_flow.workflows.trtllm_modeling_v2.run.startup.bootstrap import substrate_inputs

    task = {
        "checkpoint_path": str(tmp_path),
        "parallel": "tep8",
        "environment": {"kind": "session_spool", "python": "python3", "gpus_per_node": 4},
        "accuracy": {},
    }
    environment = resolve_environment(task, world_size=8)
    recorded = substrate_inputs(task, replace(environment, gpus_per_node=None))
    assert substrate_inputs(task, environment) == recorded
    assert substrate_inputs(task, environment.at_width(8)) == recorded
