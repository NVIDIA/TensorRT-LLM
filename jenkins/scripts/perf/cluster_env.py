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
BASE_UCX_UNSET = "unset UCX_CUDA_IPC_ENABLE_MNNVL UCX_TLS UCX_NET_DEVICES"

# (cluster_pattern, gpu_pattern, extra_export) — evaluated in order, first
# match wins; the matched export is appended after BASE_UCX_UNSET (empty
# string = base unset only). Patterns are shell-style wildcards, matched
# case-insensitively. Cluster patterns are prefix wildcards so they match
# both the bloom name (CI, e.g. "aws-cmh") and the cluster's own slurm.conf
# ClusterName (local detection, e.g. "nsc-svg" -> "nsc-svg-slurm-1").
UCX_ENV_RULES = [
    # nsc-svg: UCX picks wrong RDMA devices; pin the usable mlx5 ports.
    (
        "nsc-svg*",
        "*",
        "export UCX_NET_DEVICES="
        "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_10:1,mlx5_11:1",
    ),
    # aws-cmh: UCX transport auto-selection hangs on this fabric; pin the
    # working transport set explicitly.
    ("aws-cmh*", "*", "export UCX_TLS=cuda_ipc,cuda_copy,sm,self,tcp"),
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

    E.g. "GB300-4_GPUs-PyTorch-PerfSanity-Post-Merge-1" -> "GB300",
    "DGX_B200-8_GPUs-PyTorch-PerfSanity-1" -> "B200".
    """
    upper = (stage_name or "").upper()
    for gpu in KNOWN_GPU_TYPES:
        if gpu in upper:
            return gpu
    return ""


def gpu_type_from_supported_gpus(supported_gpus):
    """Pick the GPU type from a config yaml's metadata.supported_gpus list."""
    gpus = {str(gpu).upper() for gpu in supported_gpus or []}
    for gpu in KNOWN_GPU_TYPES:
        if gpu in gpus:
            return gpu
    return ""


def get_ucx_tls_cmd(cluster_name, gpu_type):
    """Return the shell prefix that sets UCX env vars for (cluster, GPU)."""
    cluster = (cluster_name or "").lower()
    gpu = (gpu_type or "").upper()
    extra = ""
    for cluster_pat, gpu_pat, cmd in UCX_ENV_RULES:
        if fnmatch(cluster, cluster_pat.lower()) and fnmatch(gpu, gpu_pat.upper()):
            extra = cmd
            break
    if extra:
        return f"{BASE_UCX_UNSET} && {extra} &&"
    return f"{BASE_UCX_UNSET} &&"
