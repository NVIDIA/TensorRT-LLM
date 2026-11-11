# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A deployment's servers: the launch, the address, what the run owns, the stop, the lock."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import time
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import servers
from agent_flow.workflows.trtllm_modeling_v2.spec.deployments import Deployment, Server


def _started(workspace: Path, name: str, role: str = "") -> dict:
    suffix = f".{role}" if role else ""
    path = workspace / "deployments" / name / f"address{suffix}.started.json"
    return json.loads(path.read_text())


def _one(name="tp1", **server) -> Deployment:
    return Deployment(name=name, server=Server(options={"max_seq_len": 5120}, **server))


def test_a_server_reports_its_address_is_asked_and_is_stopped(site):
    workspace = site / "ws"
    running = servers.start(_one(), model="/ckpt/model", workspace=workspace, timeout=30)
    try:
        assert running.address.startswith("127.0.0.1:")
        argv = _started(workspace, "tp1")["argv"]
        options = workspace / "deployments" / "tp1" / "tp1.yaml"
        assert argv[:3] == ["/ckpt/model", "--config", str(options)]
        assert yaml.safe_load(options.read_text()) == {"max_seq_len": 5120}
        # The kernel picks the port and the server says which; one frontend.
        assert argv[argv.index("--port") + 1] == "0" and "--num_serve_frontends=1" in argv
        assert "--server_role" not in argv
        [owned] = servers.load_owned(workspace)
        assert (owned.name, owned.pgid) == ("tp1", owned.pid) and os.getpgid(owned.pid) == owned.pid
        assert "Paris" in servers.ask(running, "The capital of France is", max_tokens=8)
    finally:
        servers.stop(running)
    assert servers.load_owned(workspace) == []
    with pytest.raises(ProcessLookupError):
        os.killpg(owned.pgid, 0)


def test_a_launch_behind_a_prefix_inherits_no_scheduler_or_mpi_identity(site, monkeypatch):
    for name in ("SLURM_JOB_ID", "SLURM_PROCID", "PMIX_RANK", "OMPI_COMM_WORLD_SIZE"):
        monkeypatch.setenv(name, "7")
    workspace = site / "ws"
    prefixed = Deployment(
        name="dep16",
        server=Server(options={}, launch="env SITE_PREFIX=1"),
        env={"UCX_TLS": "tcp,self", "OMPI_MCA_btl": "self"},
    )
    running = servers.start(prefixed, model="/m", workspace=workspace, env={"HF_HOME": "/hf"})
    servers.stop(running)
    env = _started(workspace, "dep16")["env"]
    assert not [k for k in env if k.startswith(("SLURM_", "PMIX_")) or k == "OMPI_COMM_WORLD_SIZE"]
    # What the spec and the deployment declare is passed, an MPI setting included.
    assert (env["SITE_PREFIX"], env["UCX_TLS"], env["OMPI_MCA_btl"], env["HF_HOME"]) == (
        "1", "tcp,self", "self", "/hf",
    )  # fmt: skip
    assert env[servers.SERVER_NAME_ENV] == "dep16"
    # On the orchestrator's own host nothing is taken away: the local launch rule applies.
    running = servers.start(_one("local"), model="/m", workspace=workspace)
    servers.stop(running)
    assert _started(workspace, "local")["env"]["SLURM_JOB_ID"] == "7"


def test_a_disaggregated_deployment_is_two_servers_and_a_proxy_written_from_their_addresses(site):
    workspace = site / "ws"
    pd = Deployment(
        name="pd",
        ctx=Server(options={"enable_attention_dp": True}, launch="env SIDE=ctx"),
        gen=Server(options={"enable_attention_dp": False}, launch="env SIDE=gen"),
        env={"UCX_TLS": "tcp"},
    )
    running = servers.start(pd, model="/ckpt/k3", workspace=workspace, timeout=30)
    try:
        assert [o.name for o in servers.load_owned(workspace)] == ["pd.ctx", "pd.gen", "pd.proxy"]
        ctx, gen = _started(workspace, "pd", "ctx"), _started(workspace, "pd", "gen")
        assert ctx["argv"][-2:] == ["--server_role", "context"] and ctx["env"]["SIDE"] == "ctx"
        assert gen["argv"][-2:] == ["--server_role", "generation"] and gen["env"]["SIDE"] == "gen"
        directory = workspace / "deployments" / "pd"
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


def test_a_server_that_ends_before_it_is_bound_is_an_error_and_leaves_nothing(site, monkeypatch):
    monkeypatch.setenv("FAKE_SERVE_EXIT", "1")
    workspace = site / "ws"
    with pytest.raises(servers.ServerError) as raised:
        servers.start(_one(), model="/m", workspace=workspace, timeout=30)
    assert "tp1 exited before it was bound" in str(raised.value)
    assert "fake: refused to start" in str(raised.value)
    assert servers.load_owned(workspace) == []


def test_a_new_start_first_ends_what_an_earlier_process_left(site):
    workspace = site / "ws"
    first = servers.start(_one(), model="/m", workspace=workspace, timeout=30)
    [left] = servers.load_owned(workspace)  # the process "died" here: nothing was stopped
    second = servers.start(_one("tep4"), model="/m", workspace=workspace, timeout=30)
    try:
        with pytest.raises(ProcessLookupError):
            os.killpg(left.pgid, 0)
        assert [o.name for o in servers.load_owned(workspace)] == ["tep4"]
    finally:
        servers.stop(second)
    assert first.address != second.address
    assert servers.clean_owned(workspace) == []


def test_ranks_left_on_the_devices_after_a_stop_mean_no_verdict(site):
    workspace = site / "ws"
    holders = site / "holders"
    holders.write_text("111\n")  # another job's process, there before the start
    marker = site / "stopped"
    deployment = _one(launch="env SITE=1", stop=f'echo "$MV2FLOW_SERVER" > {marker}')
    running = servers.start(deployment, model="/m", workspace=workspace, timeout=30)
    assert servers.load_owned(workspace)[0].before == ["111"]
    holders.write_text("111\n4242\n")  # a rank the launch left behind
    with pytest.raises(servers.ServerError) as raised:
        servers.stop(running)
    assert "ranks were left holding a device after the stop (tp1: 4242)" in str(raised.value)
    assert "no verdict" in str(raised.value)
    # The site's stop ran, under the name the server was started with.
    assert marker.read_text().strip() == "tp1"
    assert servers.load_owned(workspace) == []


def test_one_engine_lock_per_workspace(site):
    workspace = site / "ws"
    with servers.engine_lock(workspace):
        digest = [p for p in Path(tempfile.gettempdir()).glob("mv2flow-engine-*.lock")]
        assert digest
        held = 0
        for path in digest:
            with path.open("w") as other:
                try:
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(other, fcntl.LOCK_UN)
                except OSError:
                    held += 1
        assert held == 1  # this workspace's, and no other's
    started = time.monotonic()
    with servers.engine_lock(workspace), servers.engine_lock(None):
        pass
    assert time.monotonic() - started < 5
