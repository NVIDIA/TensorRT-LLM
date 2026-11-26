# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for multinode stress coordination; no model or TRT-LLM imports."""

import socket
from concurrent.futures import ThreadPoolExecutor

import pytest
from multinode_utils import check_peer, peer_connection, slurm_topology, wait_for_signal
from test_disaggregated_multinode_stress import _worker_config

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def step_env() -> dict[str, str]:
    return {
        "SLURM_STEP_NUM_NODES": "2",
        "SLURM_STEP_NUM_TASKS": "2",
        "SLURM_LOCALID": "0",
        "SLURM_NODEID": "1",
        "SLURM_PROCID": "1",
        "SLURM_STEP_NODELIST": "node[01-02]",
    }


def test_step_topology_uses_active_step(step_env: dict) -> None:
    step_env["SLURM_NODELIST"] = "node[01-08]"

    assert slurm_topology(step_env) == (1, ["node01", "node02"])


@pytest.mark.parametrize("nodelist", ["node01,node02", "node[01,02]", "node[01],node02"])
def test_step_topology_expands_host_lists(step_env: dict, nodelist: str) -> None:
    step_env["SLURM_STEP_NODELIST"] = nodelist
    assert slurm_topology(step_env) == (1, ["node01", "node02"])


@pytest.mark.parametrize("nodelist", ["node[01-03]", "node01,node01", "node[02-01]", "node["])
def test_step_topology_rejects_invalid_host_lists(step_env: dict, nodelist: str) -> None:
    step_env["SLURM_STEP_NODELIST"] = nodelist
    with pytest.raises(ValueError):
        slurm_topology(step_env)


@pytest.mark.parametrize("name,value", [("SLURM_LOCALID", "1"), ("SLURM_STEP_NUM_TASKS", "8")])
def test_step_topology_rejects_multiple_tasks(step_env: dict, name: str, value: str) -> None:
    step_env[name] = value
    with pytest.raises(ValueError, match="one task per node"):
        slurm_topology(step_env)


def test_peer_reports_unexpected_exit() -> None:
    coordinator, generation = socket.socketpair()
    with coordinator, generation:
        generation.close()
        with pytest.raises(RuntimeError, match="Peer task stopped"):
            check_peer(coordinator)
        with pytest.raises(RuntimeError, match="Peer exited"):
            wait_for_signal(coordinator, b"D", 1, lambda: None)


def test_signal_wait_checks_local_processes() -> None:
    def dead_worker() -> None:
        raise RuntimeError("generation worker exited")

    coordinator, generation = socket.socketpair()
    with coordinator, generation:
        with pytest.raises(RuntimeError, match="generation worker exited"):
            wait_for_signal(generation, b"D", 10, dead_worker)
        with pytest.raises(TimeoutError, match="Peer did not send"):
            wait_for_signal(coordinator, b"R", 0, lambda: None)


@pytest.mark.parametrize("same_test", [True, False])
def test_peer_handshake_and_completion(same_test: bool) -> None:
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]

    def coordinator() -> None:
        with peer_connection(0, "127.0.0.1", port, "test-a", timeout=5) as connection:
            wait_for_signal(connection, b"R", 5, lambda: None)
            connection.sendall(b"D")
            wait_for_signal(connection, b"P", 5, lambda: None)

    def generation() -> None:
        test_id = "test-a" if same_test else "test-b"
        with peer_connection(1, "127.0.0.1", port, test_id, timeout=5) as connection:
            connection.sendall(b"R")
            wait_for_signal(connection, b"D", 5, lambda: None)
            connection.sendall(b"P")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(coordinator), pool.submit(generation)]
        for future in futures:
            if same_test:
                future.result(timeout=10)
            else:
                with pytest.raises(RuntimeError, match="Expected peer message"):
                    future.result(timeout=10)


def test_worker_configs_allow_stress_lengths_and_match_cache_layout() -> None:
    context, generation = _worker_config(False), _worker_config(True)
    max_seq_len = context["max_seq_len"]
    assert isinstance(max_seq_len, int) and max_seq_len >= 8192 + 1024
    assert context["tensor_parallel_size"] == generation["tensor_parallel_size"] == 4
    assert context["kv_cache_config"] == generation["kv_cache_config"]
    assert context["cache_transceiver_config"] == generation["cache_transceiver_config"]
    assert context["speculative_config"] is None
    speculative_config = generation["speculative_config"]
    assert isinstance(speculative_config, dict)
    assert speculative_config["decoding_type"] == "DSpark"
