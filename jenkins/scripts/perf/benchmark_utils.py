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

import yaml

GEN_ONLY_NO_CONTEXT_MODE = "gen_only_no_context"

DISAGG_BENCHMARK_MODES = ("e2e", "gen_only")

AGGREGATED_DISAGG_YAML_MODES = ("ctx_only", GEN_ONLY_NO_CONTEXT_MODE)

DISAGG_CONFIG_MODES = DISAGG_BENCHMARK_MODES + AGGREGATED_DISAGG_YAML_MODES

GEN_ONLY_MODES = ("gen_only", GEN_ONLY_NO_CONTEXT_MODE)


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


def _iter_worker_llm_configs(config):
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


def uses_online_eplb(config, llm_src=""):
    """Return True when any server in the YAML enables online EPLB.

    Online EPLB (``moe_config.load_balancer.layer_updates_per_iter > 0``) is the
    only EPLB mode that keeps expert weights in host shared memory.
    ``load_balancer`` is either an inline dict or a path to an EPLB YAML; a
    relative path is resolved against ``llm_src``.

    Args:
        config: The parsed perf-sanity config YAML.
        llm_src: TensorRT-LLM source root used to resolve relative paths.

    Returns:
        bool
    """
    for llm_config in _iter_worker_llm_configs(config):
        load_balancer = (llm_config.get("moe_config") or {}).get("load_balancer")
        if isinstance(load_balancer, str) and load_balancer:
            path = load_balancer
            if not os.path.isabs(path):
                path = os.path.join(llm_src, path)
            with open(path, "r") as f:
                load_balancer = yaml.safe_load(f)
        if (
            isinstance(load_balancer, dict)
            and (load_balancer.get("layer_updates_per_iter") or 0) > 0
        ):
            return True
    return False


def add_eplb_shm_mount(mounts, config, llm_src=""):
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
    if any(entry.split(":")[1:2] == ["/dev/shm"] for entry in entries):
        return mounts
    return ",".join(entries + [EPLB_SHM_TMPFS_MOUNT])
