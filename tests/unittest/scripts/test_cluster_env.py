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
from pathlib import Path
from types import ModuleType

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

    expected = {"UCX_TLS": "cuda_ipc,cuda_copy,sm,self,tcp", "UCX_TCP_AF_PRIO": "inet"}
    assert applied == expected
    assert environ == {"SLURM_CLUSTER_NAME": "oci-jhb-slurm-1", "OTHER": "kept", **expected}


def test_apply_ucx_env_clears_leaked_vars_on_unlisted_cluster() -> None:
    environ = {"UCX_TLS": "tcp", "UCX_NET_DEVICES": "eth0"}

    applied = cluster_env.apply_ucx_env(environ, cluster_name="dlcluster", gpu_type="H100")

    assert applied == {}
    assert environ == {}


def test_apply_ucx_env_is_noop_outside_slurm() -> None:
    environ = {"UCX_TLS": "tcp", "UCX_NET_DEVICES": "eth0"}

    assert cluster_env.apply_ucx_env(environ, gpu_type="H100") is None
    assert environ == {"UCX_TLS": "tcp", "UCX_NET_DEVICES": "eth0"}
