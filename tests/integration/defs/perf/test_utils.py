#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for perf ``utils`` helpers that don't require a GPU.

Covers ``resolve_node_hostname`` precedence (SLURM_JOB_NODELIST ->
SLURM_NODELIST -> socket.gethostname()), the per-case node attribution recorded
into the perf CSV/YAML ``hostname`` field.
"""

from . import utils


def test_hostname_prefers_slurm_job_nodelist(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_NODELIST", "node[001-004]")
    monkeypatch.setenv("SLURM_NODELIST", "node999")
    monkeypatch.setattr(utils.socket, "gethostname", lambda: "local-box")
    assert utils.resolve_node_hostname() == "node[001-004]"


def test_hostname_falls_back_to_slurm_nodelist(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_NODELIST", raising=False)
    monkeypatch.setenv("SLURM_NODELIST", "node042")
    monkeypatch.setattr(utils.socket, "gethostname", lambda: "local-box")
    assert utils.resolve_node_hostname() == "node042"


def test_hostname_falls_back_to_socket_when_not_slurm(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_NODELIST", raising=False)
    monkeypatch.delenv("SLURM_NODELIST", raising=False)
    monkeypatch.setattr(utils.socket, "gethostname", lambda: "bare-metal-host")
    assert utils.resolve_node_hostname() == "bare-metal-host"


def test_hostname_empty_slurm_var_is_skipped(monkeypatch):
    # An empty string is falsy -> precedence falls through, not recorded as "".
    monkeypatch.setenv("SLURM_JOB_NODELIST", "")
    monkeypatch.setenv("SLURM_NODELIST", "")
    monkeypatch.setattr(utils.socket, "gethostname", lambda: "bare-metal-host")
    assert utils.resolve_node_hostname() == "bare-metal-host"
