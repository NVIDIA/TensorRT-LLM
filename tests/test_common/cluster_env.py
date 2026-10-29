#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cluster+GPU-aware UCX environment settings for test jobs.

UCX transport selection depends primarily on the cluster's network fabric
(IB vs RoCE vs TCP-only), not just the GPU model. The same CI stage can land
on different clusters (frontend "auto:*" platforms are load-balanced across
backend clusters in bloom's SlurmConfig), so the rules below are keyed on
(cluster name, GPU type) instead of GPU type alone.

There are two consumers:

* PerfSanity submission (jenkins/scripts/perf/submit.py and
  jenkins/scripts/perf/local/submit.py) turns the rules into a shell prefix
  with get_ucx_tls_cmd(). L0_Test.groovy passes the resolved cluster via
  --cluster-name.
* Multi-node tests call apply_ucx_env() on the environment they hand to the
  servers they spawn, with the cluster taken from SLURM_CLUSTER_NAME. Only
  cluster-specific rules are applied there; the catch-all rule's clear-only
  behavior is limited to PerfSanity.

Cluster names follow SlurmPartition.clusterName in the bloom Jenkins shared
library (src/com/nvidia/bloom/SlurmConfig.groovy), e.g. "gcp-nrt", "aws-cmh",
"aws-dfw", "oci-hsg", "oci-aga", "oci-jhb", "nsc-svg", "dlcluster",
"computelabSC01". Slurm's own ClusterName carries a deployment suffix (e.g.
bloom "oci-jhb" -> slurm.conf "oci-jhb-slurm-1"), so rules use prefix
wildcards to match both forms.
"""

import functools
import os
import subprocess
from fnmatch import fnmatch

# Cleared on every cluster before any cluster-specific setting: these may leak
# in from the outer environment and break UCX transport auto-selection.
BASE_UCX_UNSET_VARS = ("UCX_CUDA_IPC_ENABLE_MNNVL", "UCX_TLS", "UCX_NET_DEVICES")
BASE_UCX_UNSET = "unset " + " ".join(BASE_UCX_UNSET_VARS)

# (cluster_pattern, gpu_pattern, env) — evaluated in order, first match wins;
# the matched env is applied after BASE_UCX_UNSET_VARS are cleared (empty
# dict = clear only). Patterns are shell-style wildcards, matched
# case-insensitively. Cluster patterns are prefix wildcards so they match
# both the bloom name (CI, e.g. "aws-cmh") and the cluster's own slurm.conf
# ClusterName (e.g. "nsc-svg" -> "nsc-svg-slurm-1").
UCX_ENV_RULES = [
    # gcp-nrt: RoCE fabric; pin the usable rocep ports and set the RoCE GID /
    # QoS parameters required on this fabric.
    (
        "gcp-nrt*",
        "*",
        {
            "UCX_NET_DEVICES": "rocep145s0:1,rocep146s0:1,rocep152s0:1,rocep153s0:1,"
            "rocep198s0:1,rocep199s0:1,rocep205s0:1,rocep206s0:1",
            "UCX_IB_GID_INDEX": "auto",
            "UCX_IB_TRAFFIC_CLASS": "52",
            "UCX_IB_SL": "0",
        },
    ),
    # oci-aga / oci-jhb: UCX auto-selects tcp/rdma_vf_rail0 and binds a
    # unique-local IPv6 that is not assignable in the container, so NIXL
    # fails to create the UCX worker. Pin CUDA/shm/tcp and only use NICs
    # with an IPv4 address.
    (
        "oci-aga*",
        "*",
        {"UCX_TLS": "cuda_ipc,cuda_copy,sm,self,tcp", "UCX_TCP_AF_PRIO": "inet"},
    ),
    (
        "oci-jhb*",
        "*",
        {"UCX_TLS": "cuda_ipc,cuda_copy,sm,self,tcp", "UCX_TCP_AF_PRIO": "inet"},
    ),
    # oci-hsg: UCX picks wrong RDMA devices; pin the usable mlx5 ports and
    # keep eth0 as the TCP fallback device.
    (
        "oci-hsg*",
        "*",
        {"UCX_NET_DEVICES": "mlx5_0:1,mlx5_1:1,mlx5_3:1,mlx5_4:1,eth0"},
    ),
    # nsc-svg: UCX picks wrong RDMA devices; pin the usable mlx5 ports.
    (
        "nsc-svg*",
        "*",
        {
            "UCX_NET_DEVICES": "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,"
            "mlx5_4:1,mlx5_5:1,mlx5_10:1,mlx5_11:1",
        },
    ),
    # aws-cmh: UCX transport/device auto-selection hangs on this fabric; pin
    # the working transport set and Ethernet/RDMA devices explicitly.
    (
        "aws-cmh*",
        "*",
        {
            "UCX_TLS": "cuda_ipc,cuda_copy,sm,self,tcp",
            "UCX_NET_DEVICES": "eth0,mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,"
            "mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1",
        },
    ),
    # aws-dfw: gdr_copy is broken on this cluster; exclude it.
    ("aws-dfw*", "*", {"UCX_TLS": "^gdr_copy"}),
    # Default: clear only.
    ("*", "*", {}),
]

# Ordered so composite names win over their substrings (GB200 before B200,
# GB300 before B300).
KNOWN_GPU_TYPES = (
    "GB300",
    "GB200",
    "GB10X",
    "VR200",
    "B300",
    "B200",
    "H200",
    "H100",
    "A100",
)


def gpu_type_from_stage_name(stage_name):
    """Extract the GPU type token from a CI stage name or GPU product name.

    Scans for the first match in KNOWN_GPU_TYPES (ordered longest-first to
    avoid substring collisions, e.g. GB200 before B200).

    Args:
        stage_name: CI stage name string, e.g.
            "DGX_B200-8_GPUs-PyTorch-PerfSanity-1", or a GPU product name
            such as "NVIDIA GB300". None or empty string is accepted and
            returns "".

    Returns:
        A GPU type token such as "B200" or "GB300", or "" if no known GPU
        type is found in the name.
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


def get_ucx_env(cluster_name, gpu_type):
    """Return the UCX env vars to set for (cluster, GPU).

    Evaluates UCX_ENV_RULES in order, matching cluster_name and gpu_type
    against shell-style wildcard patterns (case-insensitive). The first
    matching rule wins.

    Args:
        cluster_name: Cluster name string (bloom CI name or slurm.conf
            ClusterName). None or empty string matches only the catch-all
            "*" rule.
        gpu_type: GPU type token such as "B200" or "GB300". None or empty
            string matches only the catch-all "*" rule.

    Returns:
        A new dict of env var name to value; empty when the matched rule
        only clears BASE_UCX_UNSET_VARS.
    """
    cluster = (cluster_name or "").lower()
    gpu = (gpu_type or "").upper()
    for cluster_pat, gpu_pat, env in UCX_ENV_RULES:
        if fnmatch(cluster, cluster_pat.lower()) and fnmatch(gpu, gpu_pat.upper()):
            return dict(env)
    return {}


def get_ucx_tls_cmd(cluster_name, gpu_type):
    """Return the shell prefix that sets UCX env vars for (cluster, GPU).

    Args:
        cluster_name: See get_ucx_env().
        gpu_type: See get_ucx_env().

    Returns:
        A shell command prefix string that unsets leaking UCX env vars and
        optionally exports cluster-specific overrides, ending with "&&" so
        it can be prepended directly to the worker command. Example:
        ``"unset UCX_CUDA_IPC_ENABLE_MNNVL UCX_TLS UCX_NET_DEVICES &&
        export UCX_TLS=cuda_ipc,cuda_copy,sm,self,tcp &&"``.
    """
    env = get_ucx_env(cluster_name, gpu_type)
    if env:
        exports = " ".join(f"{key}={value}" for key, value in env.items())
        return f"{BASE_UCX_UNSET} && export {exports} &&"
    return f"{BASE_UCX_UNSET} &&"


@functools.lru_cache(maxsize=None)
def detect_gpu_type():
    """Return the GPU type token of GPU 0 from nvidia-smi, or "" if unknown.

    The result is cached for the life of the process.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "--id=0"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    return gpu_type_from_stage_name(out)


def apply_ucx_env(environ=None, cluster_name=None, gpu_type=None):
    """Apply the cluster's UCX rule to an environment mapping in place.

    When a cluster-specific rule matches, clears BASE_UCX_UNSET_VARS and sets
    the rule's vars. When the cluster is unset or only the catch-all rule
    matches, the mapping is left unchanged.

    Args:
        environ: Mapping to update; defaults to os.environ.
        cluster_name: Cluster name; defaults to SLURM_CLUSTER_NAME.
        gpu_type: GPU type token; defaults to detect_gpu_type().

    Returns:
        The vars that were set, or None when no cluster-specific rule applies.
    """
    if environ is None:
        environ = os.environ
    if cluster_name is None:
        cluster_name = environ.get("SLURM_CLUSTER_NAME", "")
    if not cluster_name:
        return None
    if gpu_type is None:
        gpu_type = detect_gpu_type()
    env = get_ucx_env(cluster_name, gpu_type)
    if not env:
        return None
    for key in BASE_UCX_UNSET_VARS:
        environ.pop(key, None)
    environ.update(env)
    print(
        f"UCX env for cluster={cluster_name!r} gpu={gpu_type!r}: "
        f"cleared {list(BASE_UCX_UNSET_VARS)}, set {env}"
    )
    return env
