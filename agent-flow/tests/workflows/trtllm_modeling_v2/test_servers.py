# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A deployment's servers: the launch, the address, what the run owns, the stop, the lock."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import socket
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import servers
from agent_flow.workflows.trtllm_modeling_v2.spec.deployments import Deployment, Server

HOST = socket.gethostname()


def _started(workspace: Path, name: str, role: str = "") -> dict:
    suffix = f".{role}" if role else ""
    path = workspace / "deployments" / name / f"address{suffix}.started.json"
    return json.loads(path.read_text())


def _one(name="tp1", **server) -> Deployment:
    return Deployment(name=name, server=Server(options={"max_seq_len": 5120}, **server))


def test_a_server_runs_this_tree_is_asked_and_is_stopped(site, monkeypatch):
    workspace, tree = site / "ws", site / "tree"
    tree.mkdir()
    monkeypatch.setenv("SITE_VALUE", "declared")
    running = servers.start(
        _one(),
        model="/ckpt/model",
        workspace=workspace,
        timeout=30,
        tree=tree,
        python="/opt/v/bin/py",
        env={"SITE_VALUE": "declared"},
    )
    try:
        assert running.address.startswith("127.0.0.1:")
        started = _started(workspace, "tp1")
        argv, env = started["argv"], started["env"]
        options = workspace / "deployments" / "tp1" / "tp1.yaml"
        assert argv[:3] == ["/ckpt/model", "--config", str(options)]
        # What the deployment states, and the switch every checked server runs under.
        assert yaml.safe_load(options.read_text()) == {
            "max_seq_len": 5120,
            "modeling_v2": "require",
        }
        # The rank hook, then the measured tree, lead PYTHONPATH; the run's interpreter leads PATH.
        hook, first = env["PYTHONPATH"].split(":")[:2]
        assert hook.endswith("harness/rankhook") and first == str(tree.resolve())
        assert env["PATH"].startswith("/opt/v/bin:") and env["TRTLLM_MODELING_V2"] == "require"
        assert env["MV2FLOW_RANK_FACTS"] == "1"
        # The kernel picks the port and the server says which; one frontend.
        assert argv[argv.index("--port") + 1] == "0" and "--num_serve_frontends=1" in argv
        [owned] = servers.load_owned(workspace)
        assert (owned.name, owned.pgid, owned.host) == ("tp1", owned.pid, HOST) and owned.started
        # A declared value is recorded though it equals this process's own: whoever ends
        # the server later, under another, still runs the site's commands with it.
        monkeypatch.setenv("SITE_VALUE", "another process's")
        assert servers._site_env(owned)["SITE_VALUE"] == "declared"
        assert "Paris" in servers.ask(running, "The capital of France is", max_tokens=8)
    finally:
        assert servers.stop(running) == ""
    assert servers.load_owned(workspace) == []
    with pytest.raises(ProcessLookupError):
        os.killpg(owned.pgid, 0)


def test_the_flow_adds_the_ranks_launcher_and_the_prefix_stays_the_sites(site, monkeypatch):
    for name in ("SLURM_JOB_ID", "SLURM_PROCID", "PMIX_RANK", "OMPI_COMM_WORLD_SIZE"):
        monkeypatch.setenv(name, "7")
    workspace = site / "ws"
    prefixed = Deployment(
        name="dep16",
        server=Server(
            options={"tensor_parallel_size": 16}, launch="env SITE_PREFIX=1 mpirun -n 16"
        ),
        env={"UCX_TLS": "tcp,self", "OMPI_MCA_btl": "self"},
    )
    running = servers.start(prefixed, model="/m", workspace=workspace, env={"HF_HOME": "/hf"})
    servers.stop(running)
    # Behind the prefix, which made the ranks: the LLM API's launcher, then the program.
    line = (workspace / "deployments/dep16/dep16.launch").read_text()
    assert line.startswith("env SITE_PREFIX=1 mpirun -n 16 trtllm-llmapi-launch trtllm-serve /m ")
    env = _started(workspace, "dep16")["env"]
    # Nothing of the orchestrator's scheduler or MPI identity; what the spec declares is passed.
    assert not [k for k in env if k.startswith(("SLURM_", "PMIX_")) or k == "OMPI_COMM_WORLD_SIZE"]
    assert (env["SITE_PREFIX"], env["UCX_TLS"], env["OMPI_MCA_btl"], env["HF_HOME"]) == (
        "1", "tcp,self", "self", "/hf",
    )  # fmt: skip
    assert env[servers.SERVER_NAME_ENV] == "dep16" and "CUDA_VISIBLE_DEVICES" not in env
    # A prefix that starts one process cannot place two ranks: said before anything is loaded.
    lone = Deployment(
        name="tp2", server=Server(options={"tensor_parallel_size": 2}, launch="env SITE_PREFIX=1")
    )
    with pytest.raises(servers.ServerError) as refused:
        servers.start(lone, model="/m", workspace=workspace)
    assert str(refused.value).startswith(
        "the launch prefix of 'tp2' starts 1 process(es); its options need 2 ranks: "
    )
    assert not (workspace / "deployments/tp2/address.started.json").exists()
    assert servers.load_owned(workspace) == []
    # With no prefix: this host's launch. One rank runs its worker in its own process.
    running = servers.start(_one("local"), model="/m", workspace=workspace, devices=(4, 5))
    servers.stop(running)
    env = _started(workspace, "local")["env"]
    assert env["CUDA_VISIBLE_DEVICES"] == "4" and env["TLLM_WORKER_USE_SINGLE_PROCESS"] == "1"
    assert "SLURM_JOB_ID" not in env and "PMIX_RANK" not in env and env["SLURM_PROCID"] == "7"
    # A server takes its device because it is one, whatever its options say: `{}` too.
    bare = Deployment(name="bare", server=Server(options={}))
    servers.stop(servers.start(bare, model="/m", workspace=workspace, devices=(6,)))
    assert _started(workspace, "bare")["env"]["CUDA_VISIBLE_DEVICES"] == "6"
    # Several ranks on this host: `mpirun` makes their MPI world, and nothing of the one this
    # process may be a rank of reaches it. The scheduler's variables stay: its slots are there.
    two = Deployment(
        name="two", server=Server(options={"tensor_parallel_size": 2}), env={"OMPI_MCA_btl": "self"}
    )
    servers.stop(servers.start(two, model="/m", workspace=workspace, devices=(4, 5)))
    env = _started(workspace, "two")["env"]
    assert "PMIX_RANK" not in env and "OMPI_COMM_WORLD_SIZE" not in env
    assert (env["OMPI_MCA_btl"], env["SLURM_JOB_ID"], env["CUDA_VISIBLE_DEVICES"]) == (
        "self", "7", "4,5",
    )  # fmt: skip
    assert "TLLM_WORKER_USE_SINGLE_PROCESS" not in env


def test_a_disaggregated_deployment_is_two_servers_and_a_proxy_on_their_own_devices(site):
    workspace = site / "ws"
    pd = Deployment(
        name="pd",
        ctx=Server(options={"tensor_parallel_size": 2, "enable_attention_dp": True}),
        gen=Server(options={"tensor_parallel_size": 2}),
        env={"UCX_TLS": "tcp"},
    )
    running = servers.start(
        pd, model="/ckpt/k3", workspace=workspace, timeout=30, devices=range(4, 9)
    )
    try:
        assert [o.name for o in servers.load_owned(workspace)] == ["pd.ctx", "pd.gen", "pd.proxy"]
        ctx, gen = _started(workspace, "pd", "ctx"), _started(workspace, "pd", "gen")
        assert ctx["argv"][-2:] == ["--server_role", "context"]
        assert gen["argv"][-2:] == ["--server_role", "generation"]
        # Each under mpirun and the launcher, on its own share of the devices; the proxy on none.
        directory = workspace / "deployments" / "pd"
        launched = (directory / "pd.ctx.launch").read_text()
        assert "mpirun -n 2 --oversubscribe --allow-run-as-root trtllm-llmapi-launch" in launched
        assert (ctx["env"]["CUDA_VISIBLE_DEVICES"], gen["env"]["CUDA_VISIBLE_DEVICES"]) == (
            "4,5",
            "6,7",
        )
        assert "CUDA_VISIBLE_DEVICES" not in _started(workspace, "pd", "proxy")["env"]
        proxy = yaml.safe_load((directory / "proxy.yaml").read_text())
        assert proxy["context_servers"]["urls"] == [(directory / "address.ctx").read_text().strip()]
        assert proxy["generation_servers"]["urls"] == [
            (directory / "address.gen").read_text().strip()
        ]
        assert proxy["model"] == "/ckpt/k3" and proxy["port"] == 0
        started = _started(workspace, "pd", "proxy")["argv"]
        assert started[:3] == ["disaggregated", "-c", str(directory / "proxy.yaml")]
        # The address asked is the proxy's: a request crosses the hop.
        assert running.address == (directory / "address.proxy").read_text().strip()
        assert "Paris" in servers.ask(running, "The capital of France is", max_tokens=8)
    finally:
        servers.stop(running)
    assert servers.load_owned(workspace) == []


def test_a_start_that_fails_names_the_server_and_its_log_and_leaves_nothing(site, monkeypatch):
    workspace = site / "ws"
    monkeypatch.setenv("FAKE_SERVE_EXIT", "1")
    with pytest.raises(servers.ServerError) as raised:
        servers.start(_one(), model="/m", workspace=workspace, timeout=30)
    said = str(raised.value)
    assert said.startswith("tp1 exited before tp1 was bound to an address")
    assert f"({workspace / 'deployments/tp1/tp1.log'})" in said
    # The lines that state an error come first, then the log's end: the cause of a dead
    # server is often far above the bottom of its backtrace.
    cause = "E1111 09:00:00.000000 77 nixl.cpp:9] Invalid UCX_TLS=tcp for NIXL UCX backend"
    lines = said.splitlines()[1:]
    assert lines[:3] == [cause, "RuntimeError: Failed to create NIXL backend", "  ..."]
    assert lines[3:] == [f"#{n} 0x0000ffff in frame" for n in range(32, 40)]
    assert "fake: refused to start" not in said
    assert servers.load_owned(workspace) == []
    monkeypatch.delenv("FAKE_SERVE_EXIT")
    # A worker that dies while the proxy is waited for does not hide behind it.
    monkeypatch.setenv("FAKE_DIE_AFTER_BIND", "gen")
    pd = Deployment(name="pd", ctx=Server(options={"a": 1}), gen=Server(options={"a": 1}))
    began = time.monotonic()
    with pytest.raises(servers.ServerError) as raised:
        servers.start(pd, model="/m", workspace=workspace, timeout=60)
    said = str(raised.value)
    assert said.startswith("pd.gen exited before pd.proxy was") and "pd.gen.log" in said
    assert "fake: generation worker died" in said and time.monotonic() - began < 30
    assert servers.load_owned(workspace) == []


def test_every_wait_is_bounded_by_the_deployments_timeout(site, monkeypatch):
    workspace = site / "ws"
    looks: list[float] = []
    real = servers._holders

    def holders(launch, env, timeout):
        looks.append(timeout)
        return real(launch, env, timeout)

    monkeypatch.setattr(servers, "_holders", holders)
    monkeypatch.setenv("FAKE_SERVE_HANG", "1")  # it never reports an address
    began = time.monotonic()
    with pytest.raises(servers.ServerError, match="tp1 reported no address within its timeout"):
        servers.start(_one(), model="/m", workspace=workspace, timeout=2)
    assert time.monotonic() - began < 15 and looks and max(looks) <= 2
    assert servers.load_owned(workspace) == []


def test_the_waits_of_a_stop_share_its_budget(monkeypatch):
    owned = servers.Owned("tp1", 10001, 10001, before=[], stop="sleep 999")
    clock, looks, budgets = [0.0], [], []

    def holders(launch, env, budget):
        looks.append(budget)
        clock[0] += 0.1 if len(looks) == 1 else budget  # a later look takes all it is given
        return ["node-1 4242"]

    def stalled(argv, **said):
        budgets.append(said["timeout"])
        raise subprocess.TimeoutExpired(argv, said["timeout"])

    with monkeypatch.context() as patched:
        patched.setattr(servers, "_holders", holders)
        patched.setattr(servers.time, "monotonic", lambda: clock[0])
        patched.setattr(servers.time, "sleep", lambda n: clock.__setitem__(0, clock[0] + n))
        # One second to see the devices free is one second, however many looks it takes.
        assert servers._released(owned, 1.0) == "node-1 4242"
        assert len(looks) > 1 and clock[0] <= 1.0 + 1e-9
    # The site's `stop` gets what remains of the grace, never a multiple of it.
    monkeypatch.setattr(servers.subprocess, "run", stalled)
    servers._end(owned, 1.0)
    assert budgets and max(budgets) <= 1.0


def test_ranks_left_after_a_stop_are_no_verdict_and_stay_owned_until_the_devices_are_free(site):
    workspace = site / "ws"
    holders = site / "holders"
    holders.write_text("111\n")  # another job's process, there before the start
    marker = site / "stopped"
    deployment = Deployment(
        name="tp1",
        server=Server(
            options={"x": 1},
            launch="env SITE=1",
            stop=f'echo "$MV2FLOW_SERVER $SITE_KEY" >> {marker}',
        ),
        env={"SITE_KEY": "first"},
    )
    running = servers.start(deployment, model="/m", workspace=workspace, timeout=30)
    assert servers.load_owned(workspace)[0].before == [f"{HOST} 111"]
    holders.write_text("111\n4242\n")  # a rank the launch left behind
    with pytest.raises(servers.Leftover) as raised:
        servers.stop(running)
    assert f"ranks were left holding a device after the stop (tp1: {HOST} 4242)" in str(
        raised.value
    )
    assert "no verdict" in str(raised.value)
    # The site's stop ran under the server's own name; its record stays for the next start.
    assert marker.read_text().split() == ["tp1", "first"]
    assert [o.name for o in servers.load_owned(workspace)] == ["tp1"]
    # The next start cleans up first and starts nothing while the devices are held. What it
    # runs is the recorded stop in the recorded environment, never the next deployment's.
    other = Deployment(name="tep4", server=Server(options={"x": 1}), env={"SITE_KEY": "next"})
    with pytest.raises(servers.Leftover, match="ranks an earlier check left still hold a device"):
        servers.start(other, model="/m", workspace=workspace, timeout=30)
    assert marker.read_text().split() == ["tp1", "first", "tp1", "first"]
    assert not (workspace / "deployments" / "tep4" / "tep4.launch").exists()
    holders.write_text("111\n")  # they are gone
    running = servers.start(other, model="/m", workspace=workspace, timeout=30)
    assert [o.name for o in servers.load_owned(workspace)] == ["tep4"]
    servers.stop(running)


def test_a_look_reads_its_markers_wherever_the_prefix_puts_them(site):
    """`srun --label` prints `0: ` before each line, and a prefix may print lines of its own."""
    label = site / "bin" / "label"
    label.write_text('#!/bin/bash\necho "site banner: node-1 777"\n"$@" | sed "s/^/0: /"\n')
    label.chmod(0o755)
    (site / "holders").write_text("4242\n111\n")
    holders, processes = servers._look("label", dict(os.environ), 30)
    # The banner is no holder, the labelled lines are read, and the one process is counted.
    assert (holders, processes) == ([f"{HOST} 111", f"{HOST} 4242"], 1)
    assert servers._look("", dict(os.environ), 30) == (holders, 1)
    # A look that fails is unknown, never "no holder".
    assert servers._look("false", dict(os.environ), 30) == (None, 0)


def test_every_server_is_ended_before_the_devices_are_looked_at(tmp_path, monkeypatch):
    """A deployment's other servers, still up while one stops, are not ranks left behind."""
    roles = (("ctx", 1001), ("gen", 1002), ("proxy", 1003))
    owned = [servers.Owned(f"pd.{role}", pid, pid, before=[]) for role, pid in roles]
    holding = {"1001", "1002"}  # the two engines hold their devices until they are ended
    monkeypatch.setattr(servers, "_end", lambda item, *a: holding.discard(str(item.pid)))
    monkeypatch.setattr(servers, "_holders", lambda *a: sorted(holding))
    monkeypatch.setattr(servers, "RELEASE_TIMEOUT_S", 0)
    pd = Deployment(name="pd", ctx=Server(options={}), gen=Server(options={}))
    assert servers.stop(servers.Running(pd, "", owned, tmp_path)) == "" and not holding


def test_devices_that_cannot_be_looked_at_are_said_never_taken_for_clean(site, monkeypatch):
    workspace = site / "ws"
    running = servers.start(_one(), model="/m", workspace=workspace, timeout=30)
    monkeypatch.setattr(servers, "_holders", lambda *a, **k: None)
    assert servers.stop(running) == servers.UNVERIFIED == "devices not verified free"
    assert servers.load_owned(workspace) == []


def test_a_restart_signals_only_a_group_that_is_still_the_server_it_started(site, monkeypatch):
    workspace = site / "ws"
    servers.start(_one(), model="/m", workspace=workspace, timeout=30)
    [left] = servers.load_owned(workspace)  # the process "died" here: nothing was stopped
    signalled: list[int] = []
    real = os.killpg

    def killpg(pgid, sig):
        if sig:
            signalled.append(pgid)
        return real(pgid, sig)

    monkeypatch.setattr(servers.os, "killpg", killpg)
    marker = site / "stopped"
    stop = f"echo stop >> {marker}"
    # Recorded on another host, or with a leader that is not the process started: the group
    # is not signalled; only the recorded stop runs.
    for changed in ({"host": "another-node"}, {"started": "1"}):
        row = {**asdict(left), "stop": stop, **changed}
        (workspace / servers.OWNED_FILENAME).write_text(json.dumps([row]))
        servers.clean_owned(workspace)
        assert signalled == [] and servers.load_owned(workspace) == []
    assert marker.read_text().split() == ["stop", "stop"]
    real(left.pgid, 0)  # still running: it was never ours to signal
    # The record as it was written: the same host and the same leader. It is ended.
    (workspace / servers.OWNED_FILENAME).write_text(json.dumps([asdict(left)]))
    servers.clean_owned(workspace)
    assert signalled and set(signalled) == {left.pgid}
    with pytest.raises(ProcessLookupError):
        real(left.pgid, 0)


def test_one_engine_lock_per_workspace_wherever_tmpdir_points(site, monkeypatch):
    workspace = site / "ws"
    monkeypatch.setenv("TMPDIR", str(site / "one"))
    with servers.engine_lock(workspace):
        monkeypatch.setenv("TMPDIR", str(site / "two"))  # a worker's own temporary directory
        directory = next(d for d in servers.LOCK_DIRS if os.access(d, os.W_OK))
        digest = hashlib.sha256(str(workspace.resolve()).encode()).hexdigest()[:16]
        # This workspace's lock, found without the first caller's TMPDIR, and held.
        with (Path(directory) / f"mv2flow-engine-{os.getuid()}-{digest}.lock").open("w") as other:
            with pytest.raises(OSError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    with servers.engine_lock(workspace), servers.engine_lock(None):
        pass
    assert time.monotonic() - started < 5


@pytest.mark.parametrize("finish", ["stop", "length", "timeout", "cancelled", "None", "", None])
def test_an_answer_counts_only_once_it_completed(site, monkeypatch, finish):
    """The executor ends a request it timed out or cancelled with that word, and text may come with it."""

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def read(self):
            return json.dumps({"choices": [{"text": "Paris", "finish_reason": finish}]}).encode()

    monkeypatch.setattr(servers, "_DIRECT", SimpleNamespace(open=lambda *a, **kw: Reply()))
    running = servers.Running(_one(), "127.0.0.1:1", [], site)
    if finish in ("stop", "length"):
        assert servers.ask(running, "capital", max_tokens=8) == "Paris"
        return
    with pytest.raises(servers.ServerError) as refused:
        servers.ask(running, "capital", max_tokens=8)
    said = f"without a completed choice (finish reason {finish!r}, not stop or length)"
    assert said in str(refused.value)
