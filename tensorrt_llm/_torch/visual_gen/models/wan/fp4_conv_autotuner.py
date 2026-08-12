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

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from tensorrt_llm._torch.autotuner import (
    AutoTuner,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from tensorrt_llm.logger import logger


@dataclass(frozen=True)
class FP4ConvTactic:
    mma_tiler: tuple[int, int]
    preferred_cluster: tuple[int, int]
    fallback_cluster: tuple[int, int]
    use_2cta: bool


# Keep the no-cache fallback deliberately distinct from the provider-recommended
# maximum tile. This makes an acceptance run prove that profiling discovers the
# fast tactic instead of silently inheriting the existing fixed configuration.
FP4_CONV_FALLBACK_TACTIC = FP4ConvTactic((128, 128), (1, 1), (1, 1), False)
FP4_CONV_FIXED_TACTIC = FP4ConvTactic((256, 256), (2, 1), (2, 1), True)

# A small, valid shmoo over N tile size and 1CTA/2CTA scheduling. A 2CTA
# instruction requires cluster-M to be divisible by two for both preferred and
# fallback shapes, so all 2CTA candidates use the provider-recommended 2x1 CGA.
FP4_CONV_TACTICS: tuple[FP4ConvTactic, ...] = (
    FP4ConvTactic((128, 64), (1, 1), (1, 1), False),
    FP4_CONV_FALLBACK_TACTIC,
    FP4ConvTactic((128, 192), (1, 1), (1, 1), False),
    FP4ConvTactic((128, 256), (1, 1), (1, 1), False),
    FP4ConvTactic((256, 64), (2, 1), (2, 1), True),
    FP4ConvTactic((256, 128), (2, 1), (2, 1), True),
    FP4ConvTactic((256, 192), (2, 1), (2, 1), True),
    FP4_CONV_FIXED_TACTIC,
)
_FP4_CONV_TACTIC_SET_VERSION = 2
_selected_tactics: dict[tuple[object, ...], FP4ConvTactic] = {}


class FP4ConvTunableRunner(TunableRunner):
    """Tune precompiled CuTe Conv3d launch tactics for one runtime shape."""

    tuning_config = TuningConfig(use_cuda_graph=False)

    def __init__(
        self,
        *,
        signature: tuple[object, ...],
        compile_tactic: Callable[[FP4ConvTactic], object],
        launch: Callable[[object], None],
        output: torch.Tensor,
    ) -> None:
        self.signature = signature
        self.compile_tactic = compile_tactic
        self.launch = launch
        self.output = output

    def unique_id(self) -> tuple[object, ...]:
        # Tactic IDs are persisted by the shared autotuner. Version the ordered
        # candidate set so a future reorder cannot reinterpret a cached ID.
        return (_FP4_CONV_TACTIC_SET_VERSION, *self.signature)

    def get_valid_tactics(
        self,
        inputs: list[torch.Tensor],
        profile: OptimizationProfile,
        **kwargs,
    ) -> list[int]:
        del inputs, profile, kwargs
        return list(range(len(FP4_CONV_TACTICS)))

    @staticmethod
    def resolve_tactic(tactic: int) -> FP4ConvTactic:
        if tactic == -1:
            return FP4_CONV_FALLBACK_TACTIC
        return FP4_CONV_TACTICS[tactic]

    def forward(
        self,
        inputs: list[torch.Tensor],
        *,
        tactic: int = -1,
        do_preparation: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        del inputs, kwargs
        if do_preparation:
            # CuTe compilation can be orders of magnitude slower than a launch;
            # compile every candidate before the autotuner starts timing.
            for candidate in FP4_CONV_TACTICS:
                self.compile_tactic(candidate)
            return self.output

        compiled = self.compile_tactic(self.resolve_tactic(tactic))
        self.launch(compiled)
        return self.output


def run_tuned_fp4_conv(
    *,
    signature: tuple[object, ...],
    problem_shape: tuple[int, ...],
    tuning_inputs: Sequence[torch.Tensor | None],
    compile_tactic: Callable[[FP4ConvTactic], object],
    launch: Callable[[object], None],
    output: torch.Tensor,
) -> tuple[torch.Tensor, FP4ConvTactic]:
    """Choose and launch a Conv3d tactic; return output and chosen config."""
    selection_key = (_FP4_CONV_TACTIC_SET_VERSION, *signature, problem_shape)
    if (tactic := _selected_tactics.get(selection_key)) is not None:
        launch(compile_tactic(tactic))
        return output, tactic

    tuner = AutoTuner.get()
    runner = FP4ConvTunableRunner(
        signature=signature,
        compile_tactic=compile_tactic,
        launch=launch,
        output=output,
    )
    inputs = list(tuning_inputs)
    selected_runner, tactic_id = tuner.choose_one(
        "wan_nvfp4_conv3d",
        [runner],
        FP4ConvTunableRunner.tuning_config,
        inputs,
    )
    selected_runner(inputs, tactic=tactic_id)
    tactic = runner.resolve_tactic(tactic_id)
    # Do not memoize the eager fallback (-1): a later explicit tuning pass in
    # the same process must still be able to profile the real candidates.
    if tactic_id != -1:
        _selected_tactics[selection_key] = tactic
    logger.debug_once(
        f"Wan NVFP4 Conv3d selected tactic {tactic} for {signature}",
        key=("wan_nvfp4_conv3d", signature, tactic),
    )
    return output, tactic
