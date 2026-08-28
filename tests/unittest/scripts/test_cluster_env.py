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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CLUSTER_ENV_PATH = REPO_ROOT / "jenkins" / "scripts" / "perf" / "cluster_env.py"


@pytest.fixture(scope="module")
def cluster_env_module() -> ModuleType:
    """Load cluster_env.py without requiring jenkins to be a Python package."""
    spec = importlib.util.spec_from_file_location("cluster_env", CLUSTER_ENV_PATH)
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
            "export UCX_TLS=cuda_ipc,cuda_copy,sm,self,tcp "
            "UCX_TCP_AF_PRIO=inet UCX_NET_DEVICES=eth0",
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

    assert command == (
        f"{cluster_env_module.BASE_UCX_UNSET} && {expected_export} && "
        f"{cluster_env_module.PRESERVE_UCX_ENV} &&"
    )


@pytest.mark.parametrize("cluster_name", ("", "unknown-cluster"))
def test_get_ucx_tls_cmd_uses_base_unset_for_unknown_cluster(
    cluster_env_module: ModuleType, cluster_name: str
) -> None:
    command = cluster_env_module.get_ucx_tls_cmd(cluster_name, "GB300")

    assert command == (
        f"{cluster_env_module.BASE_UCX_UNSET} && {cluster_env_module.PRESERVE_UCX_ENV} &&"
    )
