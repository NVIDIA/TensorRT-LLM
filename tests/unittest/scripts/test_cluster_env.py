#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess

import pytest
from test_common import cluster_env


@pytest.mark.parametrize(
    ("gpu_name", "expected_gpu"),
    (
        ("NVIDIA GB300", "GB300"),
        ("NVIDIA H100 80GB HBM3", "H100"),
        ("NVIDIA L40S", ""),
    ),
)
def test_gpu_type_from_product_name(gpu_name: str, expected_gpu: str) -> None:
    assert cluster_env.gpu_type_from_stage_name(gpu_name) == expected_gpu


def test_apply_ucx_env_uses_slurm_cluster_name() -> None:
    environ = {
        "SLURM_CLUSTER_NAME": "oci-jhb-slurm-1",
        "UCX_TLS": "tcp",
        "UCX_NET_DEVICES": "eth0",
        "UCX_CUDA_IPC_ENABLE_MNNVL": "n",
        "OTHER": "kept",
    }

    applied = cluster_env.apply_ucx_env(environ, gpu_type="GB300")

    expected = {"UCX_TLS": "cuda_ipc,cuda_copy,sm,self,tcp", "UCX_TCP_AF_PRIO": "inet"}
    assert applied == expected
    assert environ == {"SLURM_CLUSTER_NAME": "oci-jhb-slurm-1", "OTHER": "kept", **expected}


def test_apply_ucx_env_is_noop_on_unlisted_cluster() -> None:
    environ = {"UCX_TLS": "tcp", "UCX_NET_DEVICES": "eth0", "UCX_CUDA_IPC_ENABLE_MNNVL": "n"}

    applied = cluster_env.apply_ucx_env(environ, cluster_name="dlcluster", gpu_type="H100")

    assert applied is None
    assert environ == {
        "UCX_TLS": "tcp",
        "UCX_NET_DEVICES": "eth0",
        "UCX_CUDA_IPC_ENABLE_MNNVL": "n",
    }


def test_apply_ucx_env_is_noop_outside_slurm() -> None:
    environ = {"UCX_TLS": "tcp", "UCX_NET_DEVICES": "eth0"}

    assert cluster_env.apply_ucx_env(environ, gpu_type="H100") is None
    assert environ == {"UCX_TLS": "tcp", "UCX_NET_DEVICES": "eth0"}


@pytest.fixture
def fresh_detect_gpu_type():
    cluster_env.detect_gpu_type.cache_clear()
    yield cluster_env.detect_gpu_type
    cluster_env.detect_gpu_type.cache_clear()


def test_detect_gpu_type_parses_nvidia_smi_once(
    monkeypatch: pytest.MonkeyPatch, fresh_detect_gpu_type
) -> None:
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="NVIDIA GB300\n", stderr="")

    monkeypatch.setattr(cluster_env.subprocess, "run", fake_run)

    assert fresh_detect_gpu_type() == "GB300"
    assert fresh_detect_gpu_type() == "GB300"
    assert len(calls) == 1


@pytest.mark.parametrize(
    "error",
    (FileNotFoundError("nvidia-smi"), subprocess.TimeoutExpired("nvidia-smi", 30)),
)
def test_detect_gpu_type_returns_empty_on_failure(
    monkeypatch: pytest.MonkeyPatch, fresh_detect_gpu_type, error: Exception
) -> None:
    def fake_run(cmd, **kwargs):
        raise error

    monkeypatch.setattr(cluster_env.subprocess, "run", fake_run)

    assert fresh_detect_gpu_type() == ""
