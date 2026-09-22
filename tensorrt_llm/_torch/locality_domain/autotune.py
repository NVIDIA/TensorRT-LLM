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
"""Autotuning helpers for kernels that execute on multiple locality domain partitions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import torch

from tensorrt_llm._torch.autotuner import (
    AutoTuner,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from tensorrt_llm._torch.locality_domain.runtime import LocalityDomainRuntime


class LocalityDomainConcurrentTunableRunner(TunableRunner):
    """Profile one tactic while every locality domain compute partition is active."""

    def __init__(
        self,
        op_runner: TunableRunner,
        runtime: LocalityDomainRuntime,
        num_partitions: int,
        launch_fn: Callable[[int, list[torch.Tensor], object], None],
    ) -> None:
        super().__init__()
        if num_partitions <= 0:
            raise ValueError(f"num_partitions must be positive, got {num_partitions}")
        runtime_num_partitions = getattr(runtime, "num_partitions", num_partitions)
        if runtime_num_partitions != num_partitions:
            raise ValueError(
                "num_partitions does not match the locality domain runtime: "
                f"{num_partitions} != {runtime_num_partitions}"
            )
        self._op_runner = op_runner
        self._runtime = runtime
        self._num_partitions = num_partitions
        self._launch_fn = launch_fn

    @property
    def op_runner(self) -> TunableRunner:
        """Return the underlying single-partition runner."""
        return self._op_runner

    def unique_id(self) -> tuple[Any, int, tuple[tuple[int, int], ...]]:
        return (
            self._op_runner.unique_id(),
            self._num_partitions,
            self._runtime.topology_identity(),
        )

    def get_valid_tactics(
        self,
        inputs: list[torch.Tensor],
        profile: OptimizationProfile,
        **kwargs: Any,
    ) -> list[Any]:
        return self._op_runner.get_valid_tactics(inputs, profile, **kwargs)

    def should_profile_tactic_in_subprocess(
        self,
        custom_op: str,
        inputs: list[torch.Tensor],
        tactic: Any,
        tuning_config: TuningConfig,
        **kwargs: Any,
    ) -> bool:
        # A subprocess cannot reconstruct the process-local CUDA partition
        # topology, streams, and green contexts used by the launch callback.
        return False

    def forward(
        self,
        inputs: list[torch.Tensor],
        tactic: Any = -1,
        **kwargs: Any,
    ) -> None:
        self._runtime.fork()
        try:
            for partition_id in range(self._num_partitions):
                with self._runtime.partition_context(partition_id):
                    self._launch_fn(partition_id, inputs, tactic)
        finally:
            self._runtime.join()


def tune_locality_domain_concurrent(
    op_name: str,
    op_runner: TunableRunner,
    runtime: LocalityDomainRuntime,
    num_partitions: int,
    launch_fn: Callable[[int, list[torch.Tensor], object], None],
    inputs: list[torch.Tensor],
    tuning_config: TuningConfig,
    **choose_one_kwargs: Any,
) -> tuple[LocalityDomainConcurrentTunableRunner, Any]:
    """Choose one tactic by profiling all locality domain partitions concurrently."""
    runner = LocalityDomainConcurrentTunableRunner(
        op_runner,
        runtime,
        num_partitions,
        launch_fn,
    )
    # AutoTuner implements cold-L2 profiling by cloning every tensor into
    # ordinary CUDA allocations. That loses the VMM node locality of locality domain
    # weights and would profile a different memory topology. Keep the original
    # localized tensors, accepting warm-L2 profiling for concurrent locality domain ops.
    if tuning_config.use_cold_l2_cache:
        tuning_config = replace(tuning_config, use_cold_l2_cache=False)
    _, tactic = AutoTuner.get().choose_one(
        f"{op_name}::locality_domain_concurrent",
        [runner],
        tuning_config,
        inputs,
        **choose_one_kwargs,
    )
    return runner, tactic
