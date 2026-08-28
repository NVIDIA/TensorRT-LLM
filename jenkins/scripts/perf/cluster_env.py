#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cluster+GPU-aware environment settings for PerfSanity SLURM jobs.

UCX transport selection depends primarily on the cluster's network fabric
(IB vs RoCE vs TCP-only), not just the GPU model. The same CI stage can land
on different clusters (frontend "auto:*" platforms are load-balanced across
backend clusters in bloom's SlurmConfig), so the rules below are keyed on
(cluster name, GPU type) instead of GPU type alone.

Cluster names follow SlurmPartition.clusterName in the bloom Jenkins shared
library (src/com/nvidia/bloom/SlurmConfig.groovy), e.g. "gcp-nrt", "aws-cmh",
"aws-dfw", "oci-hsg", "nsc-svg", "dlcluster", "computelabSC01". In CI,
L0_Test.groovy passes the resolved cluster via --cluster-name; for local
submission it can be given explicitly or is best-effort detected from the
Slurm frontend. Slurm's own ClusterName carries a deployment suffix (e.g.
bloom "oci-nrt" -> slurm.conf "oci-nrt-cs-001"), so rules use prefix
wildcards to match both forms; pass --cluster-name explicitly if a cluster
breaks this naming convention.
"""

from fnmatch import fnmatch

# Applied on every cluster before any cluster-specific export: clear settings
# that may leak in from the outer environment and break UCX transport
# auto-selection.
BASE_UCX_UNSET = (
    "unset UCX_CUDA_IPC_ENABLE_MNNVL UCX_TLS UCX_NET_DEVICES "
    "UCX_TCP_AF_PRIO UCX_IB_GID_INDEX UCX_IB_TRAFFIC_CLASS UCX_IB_SL "
    "UCX_IB_MLX5_DEVX"
)
PRESERVE_UCX_ENV = "export TRTLLM_PRESERVE_UCX_ENV=1"

# (cluster_pattern, gpu_pattern, extra_export) — evaluated in order, first
# match wins; the matched export is appended after BASE_UCX_UNSET (empty
# string = base unset only). Patterns are shell-style wildcards, matched
# case-insensitively. Cluster patterns are prefix wildcards so they match
# both the bloom name (CI, e.g. "aws-cmh") and the cluster's own slurm.conf
# ClusterName (local detection, e.g. "nsc-svg" -> "nsc-svg-slurm-1").
UCX_ENV_RULES = [
    # gcp-nrt: RoCE fabric; pin the usable rocep ports and set the RoCE GID /
    # QoS parameters required on this fabric.
    (
        "gcp-nrt*",
        "*",
        "export UCX_NET_DEVICES="
        "rocep145s0:1,rocep146s0:1,rocep152s0:1,rocep153s0:1,"
        "rocep198s0:1,rocep199s0:1,rocep205s0:1,rocep206s0:1"
        " UCX_IB_GID_INDEX=auto UCX_IB_TRAFFIC_CLASS=52 UCX_IB_SL=0",
    ),
    # oci-aga: use TCP over IPv4 alongside the local CUDA/shared-memory
    # transports, and keep TCP off the RDMA VF interfaces that have no usable
    # container-visible IP address.
    (
        "oci-aga*",
        "*",
        "export UCX_TLS=cuda_ipc,cuda_copy,sm,self,tcp UCX_TCP_AF_PRIO=inet UCX_NET_DEVICES=eth0",
    ),
    # oci-hsg: UCX picks wrong RDMA devices; pin the usable mlx5 ports and
    # keep eth0 as the TCP fallback device.
    (
        "oci-hsg*",
        "*",
        "export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_3:1,mlx5_4:1,eth0",
    ),
    # nsc-svg: UCX picks wrong RDMA devices; pin the usable mlx5 ports.
    (
        "nsc-svg*",
        "*",
        "export UCX_NET_DEVICES="
        "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_10:1,mlx5_11:1",
    ),
    # aws-cmh: UCX transport/device auto-selection hangs on this fabric; pin
    # the working transport set and Ethernet/RDMA devices explicitly.
    (
        "aws-cmh*",
        "*",
        "export UCX_TLS=cuda_ipc,cuda_copy,sm,self,tcp "
        "UCX_NET_DEVICES=eth0,mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,"
        "mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1",
    ),
    # aws-dfw: gdr_copy is broken on this cluster; exclude it.
    ("aws-dfw*", "*", "export UCX_TLS=^gdr_copy"),
    # Default: base unset only.
    ("*", "*", ""),
]

# Ordered so composite names win over their substrings (GB200 before B200,
# GB300 before B300).
KNOWN_GPU_TYPES = ("GB300", "GB200", "GB10X", "B300", "B200", "H200", "H100", "A100")


def gpu_type_from_stage_name(stage_name):
    """Extract the GPU type token from a CI stage name.

    Scans for the first match in KNOWN_GPU_TYPES (ordered longest-first to
    avoid substring collisions, e.g. GB200 before B200).

    Args:
        stage_name: CI stage name string, e.g.
            "DGX_B200-8_GPUs-PyTorch-PerfSanity-1". None or empty string
            is accepted and returns "".

    Returns:
        A GPU type token such as "B200" or "GB300", or "" if no known GPU
        type is found in the stage name.
    """
    upper = (stage_name or "").upper()
    for gpu in KNOWN_GPU_TYPES:
        if gpu in upper:
            return gpu
    return ""


def gpu_type_from_supported_gpus(supported_gpus):
    """Pick the GPU type from a config yaml's metadata.supported_gpus list.

    Args:
        supported_gpus: List of GPU type strings from the config yaml
            ``metadata.supported_gpus`` field. None or empty list returns "".

    Returns:
        The first matching GPU type token from KNOWN_GPU_TYPES, or "" if
        none of the known types appear in the list.
    """
    gpus = {str(gpu).upper() for gpu in supported_gpus or []}
    for gpu in KNOWN_GPU_TYPES:
        if gpu in gpus:
            return gpu
    return ""


def get_ucx_tls_cmd(cluster_name, gpu_type):
    """Return the shell prefix that sets UCX env vars for (cluster, GPU).

    Evaluates UCX_ENV_RULES in order, matching cluster_name and gpu_type
    against shell-style wildcard patterns (case-insensitive). Cluster
    patterns are prefix wildcards that match both the bloom CI name (e.g.
    "aws-cmh") and the cluster's own slurm.conf ClusterName (e.g.
    "aws-cmh-cs-001"). The first matching rule wins.

    Args:
        cluster_name: Cluster name string (bloom CI name or detected via
            scontrol). None or empty string matches only the catch-all "*"
            rule.
        gpu_type: GPU type token such as "B200" or "GB300". None or empty
            string matches only the catch-all "*" rule.

    Returns:
        A shell command prefix string that unsets leaking UCX env vars and
        optionally exports cluster-specific overrides, ending with "&&" so
        it can be prepended directly to the worker command. Example:
        ``"unset UCX_CUDA_IPC_ENABLE_MNNVL UCX_TLS UCX_NET_DEVICES &&
        export UCX_TLS=cuda_ipc,cuda_copy,sm,self,tcp &&"``.
    """
    cluster = (cluster_name or "").lower()
    gpu = (gpu_type or "").upper()
    extra = ""
    for cluster_pat, gpu_pat, cmd in UCX_ENV_RULES:
        if fnmatch(cluster, cluster_pat.lower()) and fnmatch(gpu, gpu_pat.upper()):
            extra = cmd
            break
    if extra:
        return f"{BASE_UCX_UNSET} && {extra} && {PRESERVE_UCX_ENV} &&"
    return f"{BASE_UCX_UNSET} && {PRESERVE_UCX_ENV} &&"


def get_ucx_env_cmd(
    runtime_mode: str,
    hardware_config: dict[str, int],
    cluster_name: str,
    gpu_type: str,
) -> str:
    """Return cluster-specific UCX setup for launches that can initialize UCX.

    Disaggregated launches can initialize UCX even in single-rank roles through
    the cache transceiver. Aggregated launches initialize it through MPI only
    when the model spans multiple ranks.
    """
    if runtime_mode == "aggregated" and hardware_config.get("gpus_per_server", 1) <= 1:
        return ""
    return get_ucx_tls_cmd(cluster_name, gpu_type)
