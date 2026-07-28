# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_llm._torch.disaggregation.diagnostics import get_diagnostic_host_identity


def test_diagnostic_host_prefers_slurm_execution_node(monkeypatch):
    monkeypatch.setenv("SLURMD_NODENAME", "node 11")
    monkeypatch.setenv("HOSTNAME", "stale-launch-node")
    monkeypatch.setattr("socket.gethostname", lambda: "container-node")

    assert get_diagnostic_host_identity() == ("node_11", "slurm")


def test_diagnostic_host_falls_back_to_kernel_hostname(monkeypatch):
    monkeypatch.delenv("SLURMD_NODENAME", raising=False)
    monkeypatch.setenv("HOSTNAME", "stale-launch-node")
    monkeypatch.setattr("socket.gethostname", lambda: "execution-node")

    assert get_diagnostic_host_identity() == ("execution-node", "socket")


def test_diagnostic_host_falls_back_to_environment(monkeypatch):
    monkeypatch.delenv("SLURMD_NODENAME", raising=False)
    monkeypatch.setenv("HOSTNAME", "environment node")

    def fail_to_read_hostname():
        raise OSError("hostname unavailable")

    monkeypatch.setattr("socket.gethostname", fail_to_read_hostname)

    assert get_diagnostic_host_identity() == ("environment_node", "environment")
