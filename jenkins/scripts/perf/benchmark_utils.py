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

import os
from typing import Iterator, Optional

import yaml

GEN_ONLY_NO_CONTEXT_MODE = "gen_only_no_context"

DISAGG_BENCHMARK_MODES = ("e2e", "gen_only")

AGGREGATED_DISAGG_YAML_MODES = ("ctx_only", GEN_ONLY_NO_CONTEXT_MODE)

DISAGG_CONFIG_MODES = DISAGG_BENCHMARK_MODES + AGGREGATED_DISAGG_YAML_MODES

GEN_ONLY_MODES = ("gen_only", GEN_ONLY_NO_CONTEXT_MODE)


def dwdp_ctx_size(config):
    """DWDP group size of the context side, or 0 when the case does not use DWDP.

    A DWDP context side is ``num_ctx_servers`` single-GPU instances that share
    expert weights. They must be ranks of one MPI world, which a separate srun
    per context server cannot provide, so launch scripts start them as one MPI
    step instead.

    Args:
        config: The parsed disaggregated config YAML.

    Returns:
        int: ``worker_config.ctx.dwdp_config.dwdp_size`` if it is > 1, else 0.

    Raises:
        ValueError: If the DWDP context side is not a set of single-GPU
            instances that exactly fill ``num_groups`` DWDP groups.
    """
    worker_config = (config or {}).get("worker_config", {}) or {}
    ctx_config = worker_config.get("ctx", {}) or {}
    dwdp_config = ctx_config.get("dwdp_config") or {}
    dwdp_size = int(dwdp_config.get("dwdp_size", 1) or 1)
    if dwdp_size <= 1:
        return 0

    ctx_world_size = (
        ctx_config.get("tensor_parallel_size", 1)
        * ctx_config.get("pipeline_parallel_size", 1)
        * ctx_config.get("context_parallel_size", 1)
    )
    if ctx_world_size != 1:
        raise ValueError(
            "DWDP context servers must each run on one GPU "
            f"(tp * pp * cp == 1), got {ctx_world_size}"
        )
    num_groups = int(dwdp_config.get("num_groups", 1) or 1)
    hardware = config.get("hardware", {}) or {}
    num_ctx_servers = hardware.get("num_ctx_servers")
    if num_ctx_servers != dwdp_size * num_groups:
        raise ValueError(
            "hardware.num_ctx_servers must equal dwdp_size * num_groups "
            f"({dwdp_size} * {num_groups}), got {num_ctx_servers}"
        )
    return dwdp_size


def ctx_only_world_size(config):
    """GPU count of the single aggregated server a ctx_only case runs.

    A plain ctx side is one server of ``tp * pp * cp`` GPUs. A DWDP ctx side is
    ``num_ctx_servers`` one-GPU servers; ctx_only folds them into one aggregated
    server with attention DP over all of them, the layout in which DWDP shares
    experts outside disaggregated serving (each rank stays a full replica).

    Args:
        config: The parsed disaggregated config YAML.

    Returns:
        int: The ctx_only server's world size.
    """
    if dwdp_ctx_size(config):
        return config["hardware"]["num_ctx_servers"]
    ctx_config = ((config or {}).get("worker_config", {}) or {}).get("ctx", {}) or {}
    return (
        ctx_config.get("tensor_parallel_size", 1)
        * ctx_config.get("pipeline_parallel_size", 1)
        * ctx_config.get("context_parallel_size", 1)
    )


def dwdp_ctx_layout(config):
    """Node layout of a DWDP context side: one world rank per context server.

    Args:
        config: The parsed disaggregated config YAML.

    Returns:
        tuple: ``(num_ctx_ranks, ctx_nodes, ranks_per_node)``.

    Raises:
        ValueError: If the context ranks do not fill whole nodes, or the case
            does not use DWDP.
    """
    if not dwdp_ctx_size(config):
        raise ValueError("dwdp_ctx_layout() requires a DWDP context side")
    num_ctx_ranks = config["hardware"]["num_ctx_servers"]
    gpus_per_node = config["hardware"]["gpus_per_node"]
    if num_ctx_ranks % gpus_per_node:
        raise ValueError(
            "DWDP context ranks must fill whole nodes: num_ctx_servers "
            f"({num_ctx_ranks}) is not a multiple of gpus_per_node ({gpus_per_node})"
        )
    return num_ctx_ranks, num_ctx_ranks // gpus_per_node, gpus_per_node


def is_gen_only_no_context(benchmark_mode, config):
    """Return True when this case must run with zero context workers.

    Args:
        benchmark_mode: The mode parsed out of the test id, or None.
        config: The parsed config YAML, or None.

    Returns:
        bool
    """
    if benchmark_mode == GEN_ONLY_NO_CONTEXT_MODE:
        return True
    if benchmark_mode != "gen_only":
        return False
    benchmark = (config or {}).get("benchmark") or {}
    return GEN_ONLY_NO_CONTEXT_MODE in str(benchmark.get("mode", ""))


def gen_only_no_context_world_size(config):
    """GPU count for the single gen worker of a gen_only_no_context case.

    Args:
        config: The parsed disaggregated config YAML.

    Returns:
        int: worker_config.gen's world size, i.e. tp * pp * cp.

    Raises:
        ValueError: If worker_config.gen is missing.
    """
    gen_config = (config.get("worker_config", {}) or {}).get("gen", {}) or {}
    if not gen_config:
        raise ValueError("worker_config.gen is required for %s mode" % (GEN_ONLY_NO_CONTEXT_MODE,))
    return gen_config.get("world_size") or (
        gen_config.get("tensor_parallel_size", 1)
        * gen_config.get("pipeline_parallel_size", 1)
        * gen_config.get("context_parallel_size", 1)
    )


def gen_only_no_context_server_counts():
    """The (num_ctx_servers, num_gen_servers) this mode launches, always (0, 1).

    Returns:
        tuple: (0, 1) -- no ctx fleet, exactly one gen worker.
    """
    return 0, 1


def parse_positive_concurrency(value: object) -> int:
    """Parse a positive benchmark concurrency from YAML configuration.

    Args:
        value: Integer or numeric string from ``benchmark.concurrency_list``.

    Returns:
        The parsed positive integer.

    Raises:
        ValueError: If ``value`` is not an integer or is not positive.
    """
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"benchmark.concurrency_list must be a positive integer, got {value!r}")

    try:
        concurrency = int(value)
    except ValueError as error:
        raise ValueError(
            f"benchmark.concurrency_list must be a positive integer, got {value!r}"
        ) from error

    if concurrency <= 0:
        raise ValueError(f"benchmark.concurrency_list must be a positive integer, got {value!r}")
    return concurrency


# Online EPLB stages a host copy of every MoE layer's expert weights in POSIX
# shared memory on each node, so the container's /dev/shm has to hold all of
# them: ~532 GiB for Kimi-K2.5 NVFP4, ~380 GiB for GLM-5 NVFP4. /dev/shm
# defaults to half of host RAM (480 GiB on a GB300 node).
EPLB_SHM_TMPFS_MOUNT = "tmpfs:/dev/shm:size=640G"


def _iter_worker_llm_configs(config: Optional[dict]) -> Iterator[dict]:
    """Yield every per-server LLM API config dict in a perf-sanity YAML.

    Disaggregated YAMLs keep them under ``worker_config.{ctx,gen}``; aggregated
    YAMLs keep one per ``server_configs`` entry.
    """
    for role_config in ((config or {}).get("worker_config") or {}).values():
        if isinstance(role_config, dict):
            yield role_config
    for server_config in (config or {}).get("server_configs") or []:
        if isinstance(server_config, dict):
            yield server_config


def _load_yaml(path: str, llm_src: str) -> object:
    """Load a YAML file; a relative ``path`` is resolved against ``llm_src``."""
    if not os.path.isabs(path):
        path = os.path.join(llm_src, path)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _merged_llm_api_config(server_config: dict, llm_src: str) -> dict:
    """Return the LLM API config one server is launched with.

    The file at ``extra_llm_api_config_path`` is the base and the inline fields
    replace its top-level keys, matching ``ServerConfig.merged_llm_api_config_data``
    in ``tests/integration/defs/perf/test_perf_sanity.py``.
    """
    extra_config_path = server_config.get("extra_llm_api_config_path")
    if not extra_config_path:
        return server_config
    extra_config = _load_yaml(extra_config_path, llm_src) or {}
    if not isinstance(extra_config, dict):
        raise ValueError(
            f"extra_llm_api_config_path {extra_config_path} does not contain a YAML mapping"
        )
    return {**extra_config, **server_config}


def uses_online_eplb(config: Optional[dict], llm_src: str = "") -> bool:
    """Return True when any server in the YAML enables online EPLB.

    Online EPLB (``moe_config.load_balancer.layer_updates_per_iter > 0``) is the
    only EPLB mode that keeps expert weights in host shared memory.
    ``moe_config`` is read from the config each server is launched with, so a
    load balancer set only in its ``extra_llm_api_config_path`` file counts.
    ``load_balancer`` is either an inline dict or a path to an EPLB YAML.
    Relative paths are resolved against ``llm_src``.

    Args:
        config: The parsed perf-sanity config YAML.
        llm_src: TensorRT-LLM source root used to resolve relative paths.

    Returns:
        bool
    """
    for server_config in _iter_worker_llm_configs(config):
        llm_config = _merged_llm_api_config(server_config, llm_src)
        load_balancer = (llm_config.get("moe_config") or {}).get("load_balancer")
        if isinstance(load_balancer, str) and load_balancer:
            load_balancer = _load_yaml(load_balancer, llm_src)
        if (
            isinstance(load_balancer, dict)
            and (load_balancer.get("layer_updates_per_iter") or 0) > 0
        ):
            return True
    return False


def _mount_target(entry: str) -> str:
    """Return the container path of one ``--container-mounts`` entry.

    pyxis mounts a source-only entry such as ``/dev/shm`` on the same path.
    """
    parts = entry.split(":")
    target = parts[1] if len(parts) > 1 else parts[0]
    return target.rstrip("/") or "/"


def add_eplb_shm_mount(mounts: str, config: Optional[dict], llm_src: str = "") -> str:
    """Append the large /dev/shm tmpfs mount when the YAML uses online EPLB.

    Args:
        mounts: Comma-separated ``--container-mounts`` value (may be empty).
        config: The parsed perf-sanity config YAML.
        llm_src: TensorRT-LLM source root used to resolve relative paths.

    Returns:
        str: ``mounts`` unchanged, or with ``EPLB_SHM_TMPFS_MOUNT`` appended. A
        caller-provided mount on ``/dev/shm`` is kept as is.
    """
    if not uses_online_eplb(config, llm_src):
        return mounts
    entries = [entry for entry in mounts.split(",") if entry]
    if any(_mount_target(entry) == "/dev/shm" for entry in entries):
        return mounts
    return ",".join(entries + [EPLB_SHM_TMPFS_MOUNT])
