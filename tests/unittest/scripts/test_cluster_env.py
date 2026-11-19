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

import importlib.util
import socket
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from test_common import cluster_env

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PERF_CLUSTER_ENV_PATH = REPO_ROOT / "jenkins" / "scripts" / "perf" / "cluster_env.py"


@pytest.fixture(scope="module")
def cluster_env_module() -> ModuleType:
    return cluster_env


@pytest.fixture(scope="module")
def perf_cluster_env_module() -> ModuleType:
    """Load the PerfSanity re-export without requiring jenkins to be a Python package."""
    spec = importlib.util.spec_from_file_location("perf_cluster_env", PERF_CLUSTER_ENV_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("stage_name", "expected_gpu"),
    (
        ("DGX_B200-8_GPUs-PyTorch-PerfSanity-1", "B200"),
        ("DGX_GB200-4_GPUs-PyTorch-PerfSanity-1", "GB200"),
        ("dgx_gb300-4_gpus-pytorch-perfsanity-1", "GB300"),
        ("VR200-HECATE-4_GPUs-PyTorch-PerfSanity-1", "VR200"),
        ("unknown-stage", ""),
        ("", ""),
    ),
)
def test_gpu_type_from_stage_name(
    cluster_env_module: ModuleType, stage_name: str, expected_gpu: str
) -> None:
    assert cluster_env_module.gpu_type_from_stage_name(stage_name) == expected_gpu


@pytest.mark.parametrize(
    ("supported_gpus", "expected_gpu"),
    (
        (["B200"], "B200"),
        (["b200", "gb200"], "GB200"),
        (["GB300", "B300"], "GB300"),
        (["VR200"], "VR200"),
        (["L40S"], ""),
        ([], ""),
    ),
)
def test_gpu_type_from_supported_gpus(
    cluster_env_module: ModuleType, supported_gpus: list[str], expected_gpu: str
) -> None:
    assert cluster_env_module.gpu_type_from_supported_gpus(supported_gpus) == expected_gpu


@pytest.mark.parametrize(
    ("cluster_name", "expected_export"),
    (
        (
            "GCP-NRT-CS-001",
            "export UCX_NET_DEVICES=rocep145s0:1,rocep146s0:1,rocep152s0:1,"
            "rocep153s0:1,rocep198s0:1,rocep199s0:1,rocep205s0:1,rocep206s0:1 "
            "UCX_IB_GID_INDEX=auto UCX_IB_TRAFFIC_CLASS=52 UCX_IB_SL=0",
        ),
        (
            "nsc-svg-slurm-1",
            "export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,"
            "mlx5_4:1,mlx5_5:1,mlx5_10:1,mlx5_11:1",
        ),
        (
            "oci-hsg-cs-001",
            "export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_3:1,mlx5_4:1,eth0",
        ),
        (
            "oci-aga-cs-001",
            "export UCX_TLS=cuda_ipc,cuda_copy,sm,self,tcp UCX_TCP_AF_PRIO=inet",
        ),
        (
            "oci-jhb-slurm-1",
            "export UCX_TLS=cuda_ipc,cuda_copy,sm,self,tcp UCX_TCP_AF_PRIO=inet",
        ),
        (
            "aws-cmh",
            "export UCX_TLS=cuda_ipc,cuda_copy,sm,self,tcp "
            "UCX_NET_DEVICES=eth0,mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,"
            "mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1",
        ),
        ("aws-dfw-prod", "export UCX_TLS=^gdr_copy"),
    ),
)
def test_get_ucx_tls_cmd_selects_cluster_rule(
    cluster_env_module: ModuleType, cluster_name: str, expected_export: str
) -> None:
    command = cluster_env_module.get_ucx_tls_cmd(cluster_name, "B200")

    assert command == f"{cluster_env_module.BASE_UCX_UNSET} && {expected_export} &&"


@pytest.mark.parametrize("cluster_name", ("", "unknown-cluster"))
def test_get_ucx_tls_cmd_uses_base_unset_for_unknown_cluster(
    cluster_env_module: ModuleType, cluster_name: str
) -> None:
    command = cluster_env_module.get_ucx_tls_cmd(cluster_name, "GB300")

    assert command == f"{cluster_env_module.BASE_UCX_UNSET} &&"


def test_perf_scripts_share_the_rule_table(perf_cluster_env_module: ModuleType) -> None:
    assert perf_cluster_env_module.UCX_ENV_RULES == cluster_env.UCX_ENV_RULES
    assert perf_cluster_env_module.get_ucx_tls_cmd(
        "oci-jhb", "GB300"
    ) == cluster_env.get_ucx_tls_cmd("oci-jhb", "GB300")


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

    expected = {
        "UCX_TLS": "cuda_ipc,cuda_copy,sm,self,tcp",
        "UCX_TCP_AF_PRIO": "inet",
        "UCX_NET_DEVICES": "eth0",
    }
    assert applied == expected
    assert environ == {"SLURM_CLUSTER_NAME": "oci-jhb-slurm-1", "OTHER": "kept", **expected}


def test_apply_ucx_env_detection_opt_out_on_unlisted_cluster() -> None:
    environ = {"UCX_TLS": "tcp", "UCX_NET_DEVICES": "eth0", "UCX_CUDA_IPC_ENABLE_MNNVL": "n"}

    applied = cluster_env.apply_ucx_env(
        environ, cluster_name="dlcluster", gpu_type="H100", detect_devices=False
    )

    assert applied is None
    assert environ == {
        "UCX_TLS": "tcp",
        "UCX_NET_DEVICES": "eth0",
        "UCX_CUDA_IPC_ENABLE_MNNVL": "n",
    }


def test_apply_ucx_env_detection_opt_out_outside_slurm() -> None:
    environ = {"UCX_TLS": "tcp", "UCX_NET_DEVICES": "eth0"}

    assert cluster_env.apply_ucx_env(environ, gpu_type="H100", detect_devices=False) is None
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


@pytest.mark.parametrize("cluster_name", ("", "aws-pdx-slurm-1", "oci-jhb"))
def test_apply_ucx_env_detects_local_device(monkeypatch, cluster_name):
    calls = []

    def local_device():
        calls.append(True)
        return "bond0"

    monkeypatch.setattr(cluster_env, "_local_net_device", local_device)
    environ = {"UCX_TLS": "old", "OTHER": "kept"}
    applied = cluster_env.apply_ucx_env(environ, cluster_name=cluster_name, gpu_type="B300")
    assert calls == [True]
    assert applied["UCX_NET_DEVICES"] == "bond0"
    assert environ == {"OTHER": "kept", **applied}
    if cluster_name == "oci-jhb":
        assert applied["UCX_TCP_AF_PRIO"] == "inet"
        assert applied["UCX_TLS"] == "cuda_ipc,cuda_copy,sm,self,tcp"
    else:
        assert applied["UCX_TLS"] == "tcp,self,sm,cuda_copy,cuda_ipc"


@pytest.mark.parametrize("cluster_name", ("aws-pdx", "nsc-svg", "oci-jhb"))
def test_apply_ucx_env_detection_preserves_explicit_devices(monkeypatch, cluster_name):
    def unexpected_detection():
        pytest.fail("Explicit device selection must bypass detection")

    monkeypatch.setattr(cluster_env, "_local_net_device", unexpected_detection)
    environ = {"UCX_NET_DEVICES": "eth-custom"}
    applied = cluster_env.apply_ucx_env(environ, cluster_name=cluster_name, gpu_type="B300")
    assert applied["UCX_NET_DEVICES"] == "eth-custom"
    assert environ["UCX_NET_DEVICES"] == "eth-custom"


def test_apply_ucx_env_cluster_device_bypasses_detection(monkeypatch):
    def unexpected_detection():
        pytest.fail("Cluster-configured devices must bypass detection")

    monkeypatch.setattr(cluster_env, "_local_net_device", unexpected_detection)
    environ = {}
    applied = cluster_env.apply_ucx_env(environ, cluster_name="nsc-svg", gpu_type="B300")
    assert applied["UCX_NET_DEVICES"].startswith("mlx5_0:1,")


@pytest.mark.parametrize(
    ("scenario", "expected"),
    (
        ("default", "bond0"),
        ("lowest_metric", "bond0"),
        ("single", "bond0"),
        ("no_route_file", "bond0"),
        ("ambiguous", None),
        ("equal_metrics", None),
        ("no_devices", None),
        ("down_default", "eth0"),
        ("rejected_default", "eth0"),
    ),
)
def test_local_net_device(monkeypatch, tmp_path, scenario, expected):
    addresses = {
        "lo": [SimpleNamespace(family=socket.AF_INET, address="127.0.0.1")],
        "bond0": [SimpleNamespace(family=socket.AF_INET, address="10.0.0.1")],
        "eth0": [SimpleNamespace(family=socket.AF_INET, address="10.1.0.1")],
        "eth1": [SimpleNamespace(family=socket.AF_INET6, address="fe80::1")],
        "eth2": [SimpleNamespace(family=socket.AF_INET, address="169.254.1.1")],
    }
    if scenario in ("single", "no_route_file"):
        del addresses["eth0"]
    if scenario == "no_devices":
        addresses = {}
    stats = {name: SimpleNamespace(isup=True) for name in addresses}
    if scenario == "down_default":
        stats["bond0"].isup = False
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(net_if_addrs=lambda: addresses, net_if_stats=lambda: stats),
    )
    route_file = tmp_path / "route"
    monkeypatch.setattr(cluster_env, "Path", lambda path: route_file)
    routes = "Iface Destination Gateway Flags RefCnt Use Metric Mask\n"
    if scenario not in ("single", "ambiguous", "no_devices", "no_route_file"):
        flags = "0203" if scenario == "rejected_default" else "0003"
        routes += f"bond0 00000000 0100000A {flags} 0 0 100 00000000\n"
    if scenario in ("lowest_metric", "equal_metrics", "rejected_default"):
        metric = 100 if scenario == "equal_metrics" else 200
        routes += f"eth0 00000000 0100010A 0003 0 0 {metric} 00000000\n"
    if scenario != "no_route_file":
        route_file.write_text(routes)
    if expected is not None:
        assert cluster_env._local_net_device() == expected
    else:
        with pytest.raises(RuntimeError, match="set UCX_NET_DEVICES explicitly"):
            cluster_env._local_net_device()
