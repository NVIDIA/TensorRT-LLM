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
