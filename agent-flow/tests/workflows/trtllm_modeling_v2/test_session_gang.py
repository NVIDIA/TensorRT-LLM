# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A session wider than one node: its layout, its job's script, and a run of it under a lead."""

from __future__ import annotations

import json
import os
import sys

import pytest

from agent_flow.workflows.trtllm_modeling_v2.checks import gang
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import session_driver
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import (
    EnvironmentError as NoStepLine,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.spec import deployments

from .conftest import engine_definitions

MOVES = {"cache_transceiver_config": {"backend": "DEFAULT"}}
BOOT = {"prompts": [{"prompt": "The capital of France is", "keywords_any": ["Paris"]}]}
SCORED = [{"name": "gsm8k", "anchor": 41.77, "source": "measured on the stock path"}]
#: Two context servers of one rank and a generation server of four: unequal sides, two
#: instances of one role that share a node, and a world on a node of its own.
UNEQUAL = {
    "name": "pd",
    "ctx": {"options": MOVES, "instances": 2},
    "gen": {"options": {"tensor_parallel_size": 4, **MOVES}},
    "benchmarks": SCORED,
    "timeout": 120,
}
WIDE = {"name": "tp8", "options": {"tensor_parallel_size": 8}, "benchmarks": SCORED, "timeout": 120}


def _slots(layout, name):
    return [(slot.node, slot.devices) for slot in layout[name]]


def test_instances_are_packed_onto_whole_nodes_one_device_per_rank():
    pd = deployments.named({"deployments": [UNEQUAL]})
    nodes, layout = gang.plan_layout(pd, 4)
    assert nodes == 2
    assert _slots(layout, "ctx0") == [(0, (0,))] and _slots(layout, "ctx1") == [(0, (1,))]
    # The generation world does not fit in what is left of the first node: a fresh one.
    assert _slots(layout, "gen0") == [(1, (0, 1, 2, 3))]
    assert gang.describe({"gen0": layout["gen0"]}) == "gen0: node 1 device(s) 0,1,2,3"
    # Roles of four ranks each on nodes of four: each on a node of its own.
    sides = {
        "ctx": {"options": {"tensor_parallel_size": 4}},
        "gen": {"options": {"tensor_parallel_size": 4}},
    }
    nodes, layout = gang.plan_layout(
        deployments.named({"deployments": [{"name": "pd", **sides}]}), 4
    )
    assert nodes == 2 and _slots(layout, "ctx0") == [(0, (0, 1, 2, 3))]
    assert _slots(layout, "gen0") == [(1, (0, 1, 2, 3))]
    # One world over both nodes; one that would end in the middle of a node is refused.
    nodes, layout = gang.plan_layout(deployments.named({"deployments": [WIDE]}), 4)
    assert nodes == 2 and _slots(layout, "server") == [(0, (0, 1, 2, 3)), (1, (0, 1, 2, 3))]
    six = deployments.named(
        {"deployments": [{"name": "tp6", "options": {"tensor_parallel_size": 6}}]}
    )
    with pytest.raises(ValueError, match="a world wider than a node takes whole nodes"):
        gang.plan_layout(six, 4)


def test_the_job_script_is_one_step_per_role_instance_built_by_the_pools_step_line(tmp_path):
    slurm = {
        "account": "ACCOUNT",
        "partition": "PARTITION",
        "time": "04:00:00",
        "gpus_per_node": 4,
        "container_image": "/images/IMAGE.sqsh",
        "container_mounts": ["/shared:/shared"],
    }
    environment = RunEnvironment(
        "slurm", "/venv/bin/python", spool="/shared/spool", slurm=slurm, gpus_per_node=4
    )
    pd = deployments.named({"deployments": [UNEQUAL]})
    nodes, layout = gang.plan_layout(pd, 4)
    files = gang.render(
        environment,
        pd,
        layout,
        nodes,
        directory=tmp_path,
        model="/ckpt/model",
        options={"ctx": "/w/options.ctx.yaml", "gen": "/w/options.gen.yaml"},
        tree="/shared/tree",
        driver=["/venv/bin/python", "/flow/session_driver.py", "--spec", "/w/session.json"],
        env={"TRTLLM_MODELING_V2": "require"},
        pythonpath=["/flow/rankhook"],
    )
    # What only the script's text shows; the two-node run below executes it.
    script = files[gang.SCRIPT]
    # No step mounts anything: the container of the holder's watcher step carries every mount.
    assert "--container-mounts" not in script and "--container-name=" in script
    steps = [line for line in script.splitlines() if "srun " in line]
    assert len(steps) == 8 and all(' ${PINNED[@]+"${PINNED[@]}"} bash ' in line for line in steps)
    # The server each rank becomes: the launcher past one rank, every interface, its role.
    assert (
        files["serve.gen0.sh"]
        .rstrip()
        .endswith(
            "exec trtllm-llmapi-launch trtllm-serve /ckpt/model --config /w/options.gen.yaml "
            f"--host 0.0.0.0 --port 0 --report_addr {tmp_path}/address.gen0 "
            "--num_serve_frontends=1 --server_role generation"
        )
    )
    assert "exec trtllm-serve /ckpt/model" in files["serve.ctx0.sh"]
    inner = files["rank.gen0.sh"]
    assert "export MV2FLOW_ROLE=gen0" in inner and "export TRTLLM_MODELING_V2=require" in inner
    assert 'export PYTHONPATH=/shared/tree:/flow/rankhook"${PYTHONPATH:+:$PYTHONPATH}"' in inner
    # An environment with no way to run a step says what it needs.
    with pytest.raises(NoStepLine, match="needs environment.slurm: the flow starts engines"):
        gang.render(
            RunEnvironment("local", sys.executable), pd, layout, nodes, directory=tmp_path,
            model="/m", options={"ctx": "c", "gen": "g"}, tree=None, driver=["d"], env={},
        )  # fmt: skip


def test_a_hop_over_two_nodes_is_run_once_by_the_lead_one_step_per_role_instance(two_nodes):
    placement, result = two_nodes.session(UNEQUAL)
    assert placement.nodes == 2 and placement.gpus == 6
    assert result.result.passed, result.render()
    found = {line.name: line for line in result.lines}
    assert list(found) == ["devices", "attest", "boot", "api", "hop", "accuracy[gsm8k]"]
    # Before any load: every server's devices, on each host it runs on, none given twice.
    assert found["devices"].detail == (
        "before any load, 3 server(s) on devices of their own across 2 hosts: ctx0 on nodeA: 0; "
        "ctx1 on nodeA: 1; gen0 on nodeB: 0,1,2,3"
    )
    # Every rank of every role instance ran its options, on its instance's devices of its host.
    assert "6 rank(s) of 3 server(s)" in found["attest"].detail
    by_instance = {r["instance"]: r for r in found["attest"].engine if "status" in r}
    assert set(by_instance["gen0"]["hosts"]) == {"nodeB"}
    assert "generation received the context's state for 1 of 1 request(s)" in found["hop"].detail
    assert "through the hop" in found["accuracy[gsm8k]"].detail
    assert "3 model load(s) for 3 role instance(s)" in result.result.detail
    # One job of two whole nodes, run once by the lead, that states each instance's ranks.
    record, script = two_nodes.record()
    assert record["nodes"] == 2 and record["hosts"] == ["nodeA", "nodeB"]
    assert record["watcher"] == "batch-lead" and record["rc"] == 0
    assert "nodes=2" in script and "ranks=ctx0:1,ctx1:1,gen0:4" in script and "gpus=" not in script
    # Every rank a step started reported where it imported the package from: its shell was
    # given the pool's origin hook after the tree, and where the job's origin records go.
    directory = two_nodes.tmp_path / "sessions" / "pd"
    (path_line,) = [
        line
        for line in (directory / "rank.gen0.sh").read_text().splitlines()
        if line.startswith("export PYTHONPATH=")
    ]
    entries = path_line.split("=", 1)[1].split('"')[0].split(":")
    assert entries[0] == two_nodes.task["trtllm_repo_path"]
    assert entries[1].endswith("infra/compute/resolvehook") and entries[2].endswith("rankhook")
    told = (directory / "origin.env").read_text()
    assert f"export MV2FLOW_RESOLVE_LOG={two_nodes.spool}/logs/session-" in told
    assert f"export MV2FLOW_TREE={two_nodes.task['trtllm_repo_path']}" in told
    origins = [
        json.loads(line)
        for line in next((two_nodes.spool / "logs").glob("session-*.resolve.jsonl"))
        .read_text()
        .splitlines()
    ]
    heard = sorted((o["role"], int(o["rank"])) for o in origins if o.get("origin") and o["role"])
    assert heard == [("ctx0", 0), ("ctx1", 0), *(("gen0", rank) for rank in range(4))]
    # Each rank of each world kept its engine's record in a file of its own: four processes
    # of `gen0` share one log, and none of their records depends on it.
    assert sorted(path.name for path in directory.glob("rank-engine.*")) == [
        "rank-engine.ctx0.0.jsonl",
        "rank-engine.ctx1.0.jsonl",
        *(f"rank-engine.gen0.{rank}.jsonl" for rank in range(4)),
    ]
    # The generation world is one step of four ranks on the second node.
    steps = [eval(line) for line in two_nodes.steps.read_text().splitlines()]  # noqa: S307
    loads = [said for said, command in steps if command[-1].endswith("rank.gen0.sh")]
    assert loads == [{"ranks": 4, "nodes": 1, "hosts": ["nodeB"], "order": ["nodeB"] * 4}]
    # Each instance was placed before anything was loaded, and loaded once.
    names = [
        command[-1].rsplit("/", 1)[-1] if command[-1] != "place" else "place"
        for _, command in steps
    ]
    assert names[:4] == ["place"] * 3 + ["check.sh"]
    # The servers' steps and the clients' task run side by side.
    assert sorted(names[4:]) == ["clients.sh", "rank.ctx0.sh", "rank.ctx1.sh", "rank.gen0.sh"]


def test_across_nodes_the_evaluators_data_is_asked_for_before_the_servers_steps(two_nodes):
    why = "Couldn't reach 'openai/gsm8k' on the Hub (OfflineModeIsEnabled)"
    scored = [{**SCORED[0], "env": {"FAKE_NO_DATA": why}}]
    _, result = two_nodes.session({**UNEQUAL, "benchmarks": scored})
    found = {line.name: line for line in result.lines}
    # The lead asked with the device check, before it started a server's step; the clients
    # that attach afterwards skip the benchmark, and every other check has run.
    assert found["accuracy[gsm8k]"].blocked.endswith(why)
    assert found["hop"].passed and found["attest"].passed
    assert "1 check(s) blocked before the load: accuracy[gsm8k]" in result.result.detail


def test_every_rank_of_a_role_has_the_engines_own_role_variable_and_one_engine_has_none(
    two_nodes, monkeypatch
):
    def said(block, name):
        directory = two_nodes.tmp_path / "sessions" / block["name"]
        started = json.loads((directory / f"address.{name}.started.json").read_text())
        return started["env"].get("TRTLLM_DISAGG_ROLE"), (
            directory / f"serve.{name}.sh"
        ).read_text()

    # Across nodes: each rank runs its instance's script, which exports the engine's value
    # for its role before the server's command (the engine sets it in rank 0 alone).
    _, result = two_nodes.session(UNEQUAL)
    assert result.result.passed, result.render()
    assert [said(UNEQUAL, n)[0] for n in ("ctx0", "ctx1", "gen0")] == ["context"] * 2 + [
        "generation"
    ]
    assert "export TRTLLM_DISAGG_ROLE=generation\n" in said(UNEQUAL, "gen0")[1]
    # One engine across nodes has no role, and borrows none from its job's environment.
    _, result = two_nodes.session(WIDE)
    assert (
        said(WIDE, "server")[0] is None and "unset TRTLLM_DISAGG_ROLE\n" in said(WIDE, "server")[1]
    )
    # On one node the server's whole world is started with it, or without it.
    monkeypatch.setenv("TRTLLM_DISAGG_ROLE", "borrowed")
    env = session_driver.server_env(
        dict(os.environ), {"name": "ctx0", "role": "ctx", "index": 0, "ranks": 4}, ["0"]
    )
    assert env["TRTLLM_DISAGG_ROLE"] == "context"
    one = {"name": "server", "role": "server", "index": 0, "ranks": 4}
    assert "TRTLLM_DISAGG_ROLE" not in session_driver.server_env(dict(os.environ), one, ["0"])


def test_one_world_over_both_nodes_and_a_rank_that_reports_no_origin_leaves_no_result(
    two_nodes,
):
    placement, result = two_nodes.session(WIDE)
    assert placement.nodes == 2 and result.result.passed, result.render()
    found = {line.name: line for line in result.lines}
    assert found["devices"].detail.endswith("server on nodeA: 0,1,2,3; server on nodeB: 0,1,2,3")
    assert "8 rank(s) of 1 server(s)" in found["attest"].detail
    assert "1 model load(s) for 1 role instance(s)" in result.result.detail
    record, script = two_nodes.record()
    assert record["hosts"] == ["nodeA", "nodeB"] and "ranks=server:8" in script
    # A rank that never imported the package reported no origin: the job has no result,
    # whatever its checks said, and the session's line says so.
    _, result = two_nodes.session({**WIDE, "env": {"FAKE_SILENT_RANK": "5"}})
    assert not result.result.passed and result.lines == ()
    assert "no result: no origin record from server[5]" in result.result.infrastructure


def test_a_rank_that_said_nothing_or_a_device_given_twice_ends_the_job_before_any_load(
    tmp_path, capsys
):
    from agent_flow.workflows.trtllm_modeling_v2.checks.harness import session_driver

    instances = [
        {"name": "ctx0", "role": "ctx", "index": 0, "ranks": 1},
        {"name": "gen0", "role": "gen", "index": 0, "ranks": 3},
    ]
    spec = tmp_path / "session.json"
    spec.write_text(json.dumps({"directory": str(tmp_path), "instances": instances}))

    def place(name, rank, host, devices, held=""):
        (tmp_path / f"placed.{name}.{rank}").write_text(f"{host} {rank} {devices} {held}\n")

    place("ctx0", 0, "nodeA", "0")
    place("gen0", 0, "nodeA", "1,2")
    place("gen0", 1, "nodeA", "1,2")
    # One rank of the generation world never said where it is.
    records, problem = session_driver.placed(tmp_path, instances)
    assert problem == "gen0 rank 2 said nothing of its place before the load"
    assert session_driver.main(["--spec", str(spec), "--placed"]) == 3
    assert '"error": "nothing was loaded: gen0 rank 2 said nothing' in capsys.readouterr().out
    # Every rank placed, each instance's devices its own: one record per instance and host.
    place("gen0", 2, "nodeB", "0", held="4242")
    records, problem = session_driver.placed(tmp_path, instances)
    assert problem == ""
    assert [(r["instance"], r["host"], r["devices"], r["ranks"]) for r in records] == [
        ("ctx0", "nodeA", ["0"], 1),
        ("gen0", "nodeA", ["1", "2"], 2),
        ("gen0", "nodeB", ["0"], 1),
    ]
    assert records[2]["held_before"] == [4242] and records[0]["held_before"] is None
    assert session_driver.main(["--spec", str(spec), "--placed"]) == 0
    # The same device of one host in two instances; the same index on another host is no clash.
    place("ctx0", 0, "nodeA", "1")
    assert session_driver.placed(tmp_path, instances)[1] == (
        "gen0 and ctx0 were both given device 1 of nodeA"
    )
    # More ranks on a host than devices there.
    place("ctx0", 0, "nodeA", "0")
    place("gen0", 1, "nodeB", "0")
    assert session_driver.placed(tmp_path, instances)[1] == (
        "gen0 has 2 rank(s) on nodeB and 1 device(s) there"
    )
    # A server bound to every interface is reached at its first rank's host.
    assert session_driver.routable("0.0.0.0:8000", "nodeB") == "nodeB:8000"
    assert session_driver.routable("nodeA:8000", "nodeB") == "nodeA:8000"
    assert session_driver.routable("0.0.0.0:8000", "") == "0.0.0.0:8000"


def test_a_pinned_session_across_nodes_makes_its_tree_on_every_node_before_any_rank_imports(
    two_nodes, tmp_path
):
    """As a run's checks are: the measured tree is a pin's, and every job names the pin."""
    import subprocess

    from agent_flow.workflows.trtllm_modeling_v2.infra.compute import pintree

    # A checkout whose run keeps trees, pinned as it stands: no native library in it.
    repo = tmp_path / "checkout"
    (repo / "tensorrt_llm").mkdir(parents=True)
    (repo / "tensorrt_llm" / "__init__.py").write_text("")
    engine_definitions(repo)
    git = ["git", "-C", str(repo), "-c", "user.name=T", "-c", "user.email=t@example.com"]
    for words in (["init", "-q", "-b", "main"], ["add", "."], ["commit", "-q", "-m", "base"]):
        subprocess.run([*git, *words], check=True, capture_output=True)
    place = tmp_path / "trees"
    pintree.write_config(repo, {"repo": str(repo), "root": str(place)})
    pin_file = pintree.pin_tree(str(repo), "session")
    pin = {**pintree.read_pin(pin_file), "file": pin_file}
    # On the controller's side the check reads the pin's tree; the job is given the pin.
    tree = pintree.at(pin)
    assert tree == str(pintree.pinned_tree(place, pin["id"]))
    two_nodes.task["trtllm_repo_path"] = tree
    two_nodes.pin = pin_file
    placement, result = two_nodes.session(UNEQUAL)
    assert placement.nodes == 2 and result.result.passed, result.render()
    record, script = two_nodes.record()
    assert f"pin={pin_file}" in script and "tree=" not in script.splitlines()[1]
    assert record["rc"] == 0 and record["hosts"] == ["nodeA", "nodeB"]
    # Every step ran the pin's prelude before its command: the job's tree record says,
    # for each time a step made or found the tree, that its libraries were checked there.
    made = pintree.read_tree_records(two_nodes.spool / "logs" / f"{record['name']}.tree.jsonl")
    assert made, "no step ran the pin's prelude"
    # Each rank imported the package from the pin's tree, and the origin judge heard them all.
    origins = [
        json.loads(line)
        for line in (two_nodes.spool / "logs" / f"{record['name']}.resolve.jsonl")
        .read_text()
        .splitlines()
    ]
    heard = sorted((o["role"], int(o["rank"])) for o in origins if o.get("origin") and o["role"])
    assert heard == [("ctx0", 0), ("ctx1", 0), *(("gen0", rank) for rank in range(4))]
    assert all(str(o["origin"]).startswith(tree) for o in origins if o.get("origin"))
