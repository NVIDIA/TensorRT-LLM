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

# The disaggregated benchmark mode that runs generation with no context workers at
# all: the gen worker fabricates its own KV blocks (the product side keys this off
# TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1, see
# tensorrt_llm/_torch/disaggregation/orchestration/coordinator.py) instead of
# receiving them over the cache transceiver. Still a *disaggregated* topology --
# disagg proxy plus N gen workers plus zero ctx workers -- because the fake-KV
# shortcut is only reachable from the disagg gen path. It is not an aggregated
# case and must not be wired behind an `aggr-` prefix.
GEN_ONLY_NO_CONTEXT_MODE = "gen_only_no_context"

# Every benchmark mode a disaggregated test id may name.
DISAGG_BENCHMARK_MODES = ("e2e", "gen_only", GEN_ONLY_NO_CONTEXT_MODE)

# Every benchmark mode whose config YAML lives in the disaggregated config folder.
# ctx_only belongs here even though it *runs* on the aggregated path: it reads a
# disagg yaml and synthesises an aggregated case out of the ctx worker section.
DISAGG_CONFIG_MODES = DISAGG_BENCHMARK_MODES + ("ctx_only",)

# The modes whose gen worker does pure decode with no context phase of its own,
# i.e. every mode that a `gen_only` special case has to cover. Compare against
# this tuple rather than spelling `== "gen_only"`: the two modes must stay
# treated identically everywhere except the ctx fleet and the injected env vars,
# or an A/B between them measures the harness instead of the mode.
GEN_ONLY_MODES = ("gen_only", GEN_ONLY_NO_CONTEXT_MODE)


def is_gen_only_no_context(benchmark_mode, config):
    """Return True when this case must run with zero context workers.

    Two ways in, and they must agree everywhere, because the node arithmetic and
    the env injection are separate call sites: get one and not the other and the
    job is sized for no ctx fleet while the gen worker still waits for KV that
    never arrives (or vice versa).

    1. The test id names the mode outright (``disagg-gen_only_no_context-<stem>``).
       This is the primary mechanism, and the only one that lets a single config
       YAML be benchmarked both ways.
    2. Legacy opt-in: the id says ``gen_only`` and the config YAML's
       ``benchmark.mode`` contains ``gen_only_no_context``. Kept for back-compat
       with the pre-existing behaviour (no checked-in config uses it today).

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
