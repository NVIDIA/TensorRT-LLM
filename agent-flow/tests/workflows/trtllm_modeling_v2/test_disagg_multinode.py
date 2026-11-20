# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A member wider than a node and a stated disaggregated deployment, with stand-ins for `srun` and the servers."""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import check_runtime as runtime
from agent_flow.workflows.trtllm_modeling_v2.checks import disagg, serve
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import serve_driver
from agent_flow.workflows.trtllm_modeling_v2.infra.compute import executor as executors
from agent_flow.workflows.trtllm_modeling_v2.infra.compute import placement, provenance, sessions
from agent_flow.workflows.trtllm_modeling_v2.infra.compute import scheduler as sched
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import resolve_environment
from agent_flow.workflows.trtllm_modeling_v2.run.startup import preflight
from agent_flow.workflows.trtllm_modeling_v2.run.validation.check_runner import CheckRunner
from agent_flow.workflows.trtllm_modeling_v2.spec import task_schema

# `trtllm-serve`: rank 0 of a world serves (the proxy for `disaggregated`) at its node's name
# and reports it; every rank writes what it was started with beside the address.
_FAKE_SERVE = """#!{python}
import http.server, json, os, signal, sys
argv = sys.argv[1:]
proxy = argv[:1] == ["disaggregated"]
report = argv[argv.index("--report_addr") + 1]
rank = os.environ.get("SLURM_PROCID", "0")
if os.environ.get("FAKE_SERVE_EXIT") and not proxy:
    print("RuntimeError: Failed to create NIXL backend")
    sys.exit(3)
if not proxy:
    with open(report + ".rank" + rank + ".json", "w") as out:
        json.dump({{"argv": argv, "env": dict(os.environ)}}, out)
    import tensorrt_llm  # the rank's import record
if not proxy and rank != "0":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    while True:
        signal.pause()

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        asked = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or "{{}}")
        text = " Paris, after: " + str(asked.get("prompt"))
        body = json.dumps({{"choices": [{{"text": text, "finish_reason": "stop"}}]}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST

server = http.server.ThreadingHTTPServer((os.environ.get("SLURMD_NODENAME", "127.0.0.1"), 0), Handler)
with open(report + ".tmp", "w") as out:
    out.write("%s:%d\\n" % (os.environ.get("SLURMD_NODENAME", "127.0.0.1"), server.server_address[1]))
os.replace(report + ".tmp", report)
signal.signal(signal.SIGTERM, lambda *a: os._exit(0))
server.serve_forever()
"""
# `mpirun`: one process, the command after the LLM API's launcher.
_FAKE_MPIRUN = (
    '#!/bin/bash\nwhile [ $# -gt 0 ] && [ "${1##*/}" != trtllm-llmapi-launch ]; do shift; done\n'
    'shift\nexec "$@"\n'
)
# `srun`: the command once per rank, each told its rank and node, in a fresh environment.
_FAKE_STEP = """#!{python}
import json, os, signal, subprocess, sys

command, asked = sys.argv[1:], {{}}
while command and command[0].startswith("--"):
    key, _, value = command.pop(0)[2:].partition("=")
    asked[key] = value
ranks, nodes, hosts = int(asked["ntasks"]), int(asked["nodes"]), asked["nodelist"].split(",")
assert "overlap" in asked and int(asked["ntasks-per-node"]) * nodes == ranks, asked
order = [host for host in hosts for _ in range(ranks // nodes)]
with open(os.environ["FAKE_STEP_LOG"], "a") as log:
    log.write(json.dumps({{"asked": asked, "hosts": hosts, "command": command}}) + "\\n")
fresh = {{k: v for k, v in os.environ.items() if k != "PYTHONPATH" and not k.startswith("MV2FLOW_")}}
started = []
for rank in range(ranks):
    env = dict(fresh, SLURM_PROCID=str(rank), SLURM_NTASKS=str(ranks), SLURMD_NODENAME=order[rank])
    started.append(subprocess.Popen(command, env=env))
signal.signal(signal.SIGTERM, lambda *_: [process.terminate() for process in started])
codes = [process.wait() for process in started]
sys.exit(next((code for code in codes if code), 0))
"""
# What the client uses of the tree: `trtllm-eval`'s benchmark command, a tokenizer, the logger.
_FAKE_TREE = {
    "__init__.py": "",
    "logger.py": "import types\nlogger = types.SimpleNamespace(set_level=lambda level: None)\n",
    "llmapi/tokenizer.py": "def load_hf_tokenizer(model):\n    return 'tokenizer'\n",
    "executor/result.py": "class GenerationResultBase:\n    def __init__(self, *args):\n        pass\n",
    "llmapi/__init__.py": """
class CompletionOutput:
    def __init__(self, text, index):
        self.text = text
class RequestOutput:
    @classmethod
    def _from_generation_result(cls, result, prompt=None):
        return cls()
""",
    "commands/eval.py": """
import json, os, types
class Benchmark:
    def main(self, args, obj, standalone_mode, **_):
        held = dict(max_tokens=4, n=1, best_of=None, temperature=None, top_p=None, stop=None, seed=1)
        answer = obj.generate_async("2 + 2 =", types.SimpleNamespace(**held)).result().result()
        print("benchmark: " + json.dumps([obj.tokenizer, answer.outputs[0].text, *args]))
        if os.environ.get("FAKE_ROW"):
            print("lm-eval gsm8k flexible-extract accuracy: " + os.environ["FAKE_ROW"])
        print("lm-eval gsm8k average accuracy: 41.70")
        obj.shutdown()
main = types.SimpleNamespace(commands={"gsm8k": Benchmark()})
""",
}
ALLOCATION = "4242"
# Two nodes, by names that resolve here.
A, B = "127.0.0.1", "127.0.0.2"


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
    """Stand-ins on ``PATH``, a tree the ranks import from, a checkpoint and its anchor; the task."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, text in (
        ("trtllm-serve", _FAKE_SERVE.format(python=sys.executable)),
        ("nvidia-smi", "#!/bin/bash\nexit 0\n"),
        ("mpirun", _FAKE_MPIRUN),
        ("trtllm-llmapi-launch", '#!/bin/bash\nexec "$@"\n'),
        ("srun", _FAKE_STEP.format(python=sys.executable)),
    ):
        path = bin_dir / name
        path.write_text(text)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_STEP_LOG", str(tmp_path / "steps.log"))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    for name, text in _FAKE_TREE.items():
        path = tmp_path / "tree" / "tensorrt_llm" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
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


def _scheduler(spool, host, *, lead=False) -> sched.Scheduler:
    """A watcher (or the lead) of ``spool`` on ``host``, in one allocation, its devices clean."""
    environ = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "PMI"))}
    environ.update(SLURM_JOB_ID=ALLOCATION, SLURM_PROCID="0", SLURM_STEP_ID="0")
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


class TwoNodes:
    """A session of two nodes of four devices and its lead, on one spool."""

    def __init__(self, task, tmp_path):
        self.task, self.tmp_path = task, tmp_path
        self.spool = Path(task["environment"]["spool"])
        self.environment = resolve_environment(task, world_size=4)

    def runner(self, gpus="all", **job):
        tree = self.task["trtllm_repo_path"]
        made = executors.SpoolCommandExecutor(self.spool, poll_s=0.1, gpus=gpus, tree=tree, **job)
        return executors.shell_runner_for(made)

    def record(self):
        path = sorted((self.spool / "done").glob("gate-*.json"))[-1]
        return json.loads(path.read_text()), path.with_suffix(".sh").read_text()

    def steps(self):
        return [json.loads(line) for line in (self.tmp_path / "steps.log").read_text().splitlines()]


def _rank(directory, name, rank=0):
    """What rank ``rank`` of the server ``name`` was started with."""
    return json.loads((directory / f"address.{name}.rank{rank}.json").read_text())


@pytest.fixture
def two_nodes(site, tmp_path):
    run = TwoNodes(site, tmp_path)
    sessions.install_session_entry(site, "gates")
    stop = threading.Event()

    def loop(watcher):
        while not stop.is_set():
            watcher.tick()
            time.sleep(0.05)

    for host in (A, B, "batch"):
        watcher = _scheduler(run.spool, host, lead=host == "batch")
        threading.Thread(target=loop, args=(watcher,), daemon=True).start()
    yield run
    stop.set()


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
        (["dep16"], (16, 1), (16, 1), 8, True),  # Kimi K3 on GB300
        (["tep4"], (2, 1), (2, 1), 1, False),  # both sides on one node
    ],
)
def test_the_specs_of_the_real_cases_load_and_size_their_checks_jobs(
    task_file, tmp_path, members, ctx, gen, nodes, lead
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
    task["environment"]["sessions"][0]["nodes"] = 1
    other = sessions.install_session_entry(task, "gates", out=tmp_path / "one" / "session_entry.sh")
    assert not other.with_name("session_lead.sh").exists()


def test_a_job_across_nodes_is_run_once_by_the_lead_after_its_nodes_drained(two_nodes):
    paths = sched.SpoolPaths(two_nodes.spool)
    said = two_nodes.tmp_path / "said"
    body = (
        f'echo "$MV2FLOW_JOB_NODES $MV2FLOW_JOB_HOSTS ${{SLURM_JOB_ID:-}} ${{SLURM_PROCID:-none}}" '
        f">> {said}; ls {paths.gangs}/* > /dev/null"
    )
    result = two_nodes.runner(nodes=2, lead=True)(body, None, 60)
    assert result.returncode == 0, result.stdout
    assert said.read_text() == f"2 {A},{B} {ALLOCATION} none\n"
    record, script = two_nodes.record()
    assert record["watcher"] == "batch-lead" and record["hosts"] == [A, B]
    assert "nodes=2 lead=1" in script and not any(paths.gangs.iterdir())
    # Without a lead a job of several nodes runs on each; the lead takes no node's job.
    result = two_nodes.runner(nodes=2)(f'echo "part $MV2FLOW_JOB_NODE_RANK" >> {said}', None, 60)
    assert result.returncode == 0, result.stdout
    assert sorted(said.read_text().splitlines()[1:]) == ["part 0", "part 1"]
    assert two_nodes.runner(gpus="1")("echo one", None, 60).returncode == 0
    assert two_nodes.record()[0]["watcher"] in (f"{A}-w", f"{B}-w")
    # A lead holds no device: the gate pool does not take it for a free node.
    job = placement.JobNeeds.from_script("x", "#spool: gpus=all nodes=2 lead=1\n")
    assert placement.assess(two_nodes.spool, job, stale_s=180, now=time.time()).devices == 4
    _scheduler(two_nodes.tmp_path / "alone", "batch", lead=True)
    alone = placement.assess(two_nodes.tmp_path / "alone", job, stale_s=180, now=time.time())
    assert alone.reasons == ["no watcher has ever heart-beat there"]


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


def test_a_member_across_two_nodes_is_one_world_and_serve_is_judged_as_on_one(two_nodes):
    environment = runtime.with_mode(two_nodes.environment.at_width(8), "require")
    probe = (
        "import json, os, tensorrt_llm; print('RANK ' + json.dumps({k: os.environ.get(k) for k in "
        "('SLURM_PROCID', 'SLURMD_NODENAME', 'CUDA_VISIBLE_DEVICES', 'TRTLLM_MODELING_V2', "
        "'MV2FLOW_TREE', 'FROM_SESSION', 'TRITON_CACHE_DIR')} | {'cwd': os.getcwd()}))"
    )
    run = two_nodes.runner(nodes=2, lead=True)
    result = run(environment.python_command(["-c", probe]), None, 60)
    assert result.returncode == 0, result.stdout
    ranks = sorted(
        (json.loads(line[5:]) for line in result.stdout.splitlines() if line.startswith("RANK ")),
        key=lambda rank: int(rank["SLURM_PROCID"]),
    )
    tree, tmp = two_nodes.task["trtllm_repo_path"], two_nodes.task["environment"]["env"]["TMPDIR"]
    assert [r["SLURMD_NODENAME"] for r in ranks] == [A] * 4 + [B] * 4
    assert {r["CUDA_VISIBLE_DEVICES"] for r in ranks} == {"0,1,2,3"}
    # A step's fresh shell got the tree, the check's variables and the session's setup.
    keys = ("cwd", "MV2FLOW_TREE", "TRTLLM_MODELING_V2", "FROM_SESSION", "TRITON_CACHE_DIR")
    handed = {tuple(r[k] for k in keys) for r in ranks}
    assert handed == {(tree, tree, "require", "entry", f"{tmp}/triton-cache")}
    # Import records one file a node, read as the job's.
    logs = sorted((two_nodes.spool / "logs").glob("gate-*.resolve.jsonl.*"))
    assert [log.name.partition(".jsonl.")[2] for log in logs] == [A, B]
    origins = [json.loads(line) for log in logs for line in log.read_text().splitlines()]
    assert sorted(int(o["rank"]) for o in origins if o["module"] == "tensorrt_llm") == [*range(8)]
    job_log = logs[0].with_name(logs[0].name.partition(f".{A}")[0])
    assert len(provenance.read_records(job_log)) == len(origins)
    (step,) = two_nodes.steps()
    assert step["asked"]["mpi"] == "pmix" and step["asked"]["container-name"] == "mv2flow"
    task = {**two_nodes.task, "features": [], "members": ["tep8"]}
    served = serve.run_serve_gate(
        two_nodes.environment.at_width(8), task_data=task, runner=run, features=[], unit="core",
        member="tep8", workdir=two_nodes.tmp_path / "gates",
    )  # fmt: skip
    assert served.passed, served.detail + served.stdout_tail
    server, driver = sorted(two_nodes.steps()[1:], key=lambda step: -int(step["asked"]["ntasks"]))
    assert (server["asked"]["ntasks"], server["hosts"]) == ("8", [A, B])
    assert "trtllm-llmapi-launch trtllm-serve" in " ".join(server["command"])
    assert (driver["asked"]["ntasks"], driver["asked"]["mpi"]) == ("1", "none")
    started = _rank(two_nodes.tmp_path / "gates" / "serve.core.tep8.steps", "server", 5)
    assert "TRTLLM_DISAGG_ROLE" not in started["env"]


def test_a_deployment_on_one_node_is_served_and_scored_with_no_lead_and_no_step(
    site, tmp_path, monkeypatch
):
    stated = _deployment(tmp_path / "one", (1, 1), (2, 1), timeout=120)
    task = {**site, "accuracy": {**site["accuracy"], "disagg": stated}}
    assert disagg.DisaggSpec.from_task(task).device_count("tep4") == 3
    environment = replace(resolve_environment(task, world_size=4), kind="local")

    def gate(task=task, **said):
        with monkeypatch.context() as patch:
            for name, value in said.items():
                patch.setenv(name, value)
            return disagg.run_disagg_gate(
                environment, task_data=task, workdir=tmp_path / "gates",
                runner=runtime.in_tree(runtime.default_shell_runner, task["trtllm_repo_path"]),
                unit="disagg", member="tep4",
            )  # fmt: skip

    result = gate(CUDA_VISIBLE_DEVICES="4,5,6")
    assert result.passed, result.detail + result.stdout_tail
    directory = tmp_path / "gates" / "disagg.disagg.tep4"
    ran = (directory / "servers.sh").read_text()
    assert "srun" not in ran and " mpirun -n 2 " in ran and not (tmp_path / "steps.log").exists()
    assert result.detail.startswith("2 server(s) on 1 node(s) and their proxy came up; ")
    assert (result.measured, result.reference, result.samples) == (41.70, 41.77, 1319)
    # Each server its own devices of the job's and its role; its side's file with the gate's mode.
    ctx, gen = _rank(directory, "ctx0"), _rank(directory, "gen0")
    own = [(s["env"]["CUDA_VISIBLE_DEVICES"], s["env"]["TRTLLM_DISAGG_ROLE"]) for s in (ctx, gen)]
    assert own == [("4", "context"), ("5,6", "generation")]
    assert gen["argv"][gen["argv"].index("--server_role") + 1] == "generation"
    config = yaml.safe_load((directory / "options.gen.yaml").read_text())
    assert config == dict(tensor_parallel_size=2, moe_expert_parallel_size=2, modeling_v2="require")
    # The script ended its servers.
    for name in ("ctx0", "gen0", "proxy"):
        address = (directory / f"address.{name}").read_text().strip()
        with pytest.raises(OSError):
            for _ in range(100):
                serve_driver._DIRECT.open(f"http://{address}", timeout=5).close()
                time.sleep(0.1)
    # The unit's protocol reaches the evaluator, and its score pattern picks the row.
    turns = {
        **task["accuracy"],
        "benchmark_args": ["--apply_chat_template", "--fewshot_as_multiturn"],
        "score_pattern": r"lm-eval gsm8k flexible-extract accuracy:\s*([0-9.]+)",
    }
    row = gate({**task, "accuracy": turns}, FAKE_ROW="41.60")
    assert row.passed and row.measured == 41.60, row.detail
    asked = (
        '"tokenizer", " Paris, after: 2 + 2 =", "--apply_chat_template", "--fewshot_as_multiturn"'
    )
    assert f"benchmark: [{asked}]" in row.stdout_tail
    held = dict(max_tokens=256, n=1, best_of=None, temperature=None, top_p=None, stop=None, seed=7)
    sent = disagg.disagg_driver.request_body("m", "p", SimpleNamespace(**held))
    assert sent == {**held, "model": "m", "prompt": "p", "stream": False, "temperature": 0}
    down = gate(FAKE_SERVE_EXIT="1")
    assert not down.passed and "Failed to create NIXL backend" in down.detail
    other = gate({**task, "accuracy": {**task["accuracy"], "benchmark": "mmlu"}})
    assert not other.passed and "through it: the client exited 1" in other.detail
    assert "KeyError: 'mmlu'" in other.stdout_tail


def test_a_deployment_across_nodes_is_one_step_a_server_run_once_by_the_lead(two_nodes):
    assert disagg.DisaggSpec.from_task(two_nodes.task).device_count("tep4") is None
    result = disagg.run_disagg_gate(
        replace(two_nodes.environment, nodes=1),
        task_data=two_nodes.task,
        workdir=two_nodes.tmp_path / "gates",
        runner=two_nodes.runner(nodes=2, lead=True),
        unit="disagg",
        member="tep4",
    )
    assert result.passed, result.detail + result.stdout_tail
    assert result.detail.startswith("3 server(s) on 2 node(s) and their proxy came up; ")
    assert two_nodes.record()[0]["watcher"] == "batch-lead"
    steps = two_nodes.steps()
    assert sorted((s["asked"]["ntasks"], s["hosts"]) for s in steps) == [
        ("1", [A]), ("2", [A]), ("2", [A]), ("4", [B]),
    ]  # fmt: skip
    directory = two_nodes.tmp_path / "gates" / "disagg.disagg.tep4"
    devices = [_rank(directory, n)["env"]["CUDA_VISIBLE_DEVICES"] for n in ("ctx0", "ctx1", "gen0")]
    assert devices == ["0,1", "2,3", "0,1,2,3"]
    # The proxy got the deployment's own config with the servers' addresses.
    proxy = yaml.safe_load((directory / "proxy.yaml").read_text())
    said = (proxy["max_retries"], proxy["context_servers"]["num_instances"], proxy["port"])
    assert said == (2, 2, 0) and proxy["model"] == two_nodes.task["checkpoint_path"]
    assert [url.rpartition(":")[0] for url in proxy["generation_servers"]["urls"]] == [B]
    for rank in range(4):
        started = _rank(directory, "gen0", rank)["env"]
        assert started["TRTLLM_DISAGG_ROLE"] == "generation" and started["UCX_TLS"] == "^cuda_ipc"
    assert not list(directory.glob("exit.*"))  # no server ended before the script ended it


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
    short = {**site, "environment": {**site["environment"], "sessions": []}}
    with pytest.raises(preflight.PreflightError, match="needs 2 nodes in one session_spool"):
        preflight.check_disagg_placement(short, 4)
