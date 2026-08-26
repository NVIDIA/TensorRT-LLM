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

from dataclasses import astuple, dataclass
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


FP4_CONV_FIXED_TACTIC = FP4ConvTactic((256, 256), (2, 1), (2, 1), True)
# Use the validated fixed configuration when no tuned selection is available.
FP4_CONV_FALLBACK_TACTIC = FP4_CONV_FIXED_TACTIC

# NVFP4 alignment fixes the M tile at 128 for 1CTA and 256 for 2CTA. Sweep N
# tiles from 64 through 256. A 2CTA instruction requires cluster-M to be
# divisible by two for both preferred and fallback shapes.
FP4_CONV_TACTICS: tuple[FP4ConvTactic, ...] = (
    FP4ConvTactic((128, 64), (1, 1), (1, 1), False),
    FP4ConvTactic((128, 128), (1, 1), (1, 1), False),
    FP4ConvTactic((128, 192), (1, 1), (1, 1), False),
    FP4ConvTactic((128, 256), (1, 1), (1, 1), False),
    FP4ConvTactic((256, 64), (2, 1), (2, 1), True),
    FP4ConvTactic((256, 128), (2, 1), (2, 1), True),
    FP4ConvTactic((256, 192), (2, 1), (2, 1), True),
    FP4_CONV_FIXED_TACTIC,
)
_selected_tactics: dict[tuple[object, ...], tuple[int, FP4ConvTactic]] = {}
_failed_tactics: dict[tuple[object, ...], set[int]] = {}


def _tactic_set_key() -> tuple[tuple[object, ...], ...]:
    """Persist the ordered tactic definitions."""
    return tuple(astuple(tactic) for tactic in FP4_CONV_TACTICS)


def _clear_fp4_conv_tactic_cache() -> None:
    """Clear in-process selections and failed-candidate records."""
    _selected_tactics.clear()
    _failed_tactics.clear()


def _launch_fallback_tactic(
    *,
    signature: tuple[object, ...],
    problem_shape: tuple[int, ...],
    failed_tactic: FP4ConvTactic,
    error: Exception,
    compile_tactic: Callable[[FP4ConvTactic], object],
    launch: Callable[[object], None],
) -> FP4ConvTactic:
    try:
        torch.cuda.synchronize()
    except RuntimeError as sync_error:
        logger.debug(
            f"Wan NVFP4 Conv3d CUDA synchronize failed after tactic {failed_tactic}: {sync_error}"
        )
    logger.warning_once(
        f"Wan NVFP4 Conv3d tactic {failed_tactic} failed for {signature}, "
        f"{problem_shape}; using the fallback tactic: {error}",
        key=("wan_nvfp4_conv3d", "runtime_fallback", signature, problem_shape, failed_tactic),
    )
    launch(compile_tactic(FP4_CONV_FALLBACK_TACTIC))
    return FP4_CONV_FALLBACK_TACTIC


class FP4ConvTunableRunner(TunableRunner):
    """Tune precompiled CuTe Conv3d launch tactics for one runtime shape."""

    # The launch closures bind live tensor and layout objects.
    tuning_config = TuningConfig(use_cuda_graph=False)

    def __init__(
        self,
        *,
        signature: tuple[object, ...],
        problem_shape: tuple[int, ...],
        compile_tactic: Callable[[FP4ConvTactic], object],
        launch: Callable[[object], None],
        output: torch.Tensor,
    ) -> None:
        self.signature = signature
        self.problem_shape = problem_shape
        self.compile_tactic = compile_tactic
        self.launch = launch
        self.output = output
        self._runner_key = (_tactic_set_key(), *signature, problem_shape)
        # Runners are rebuilt for each invocation. Share failures for an exact
        # signature and problem shape so capture/replay does not re-enumerate a
        # tactic that compiled successfully but failed during profiling launch.
        self._failed_tactics = _failed_tactics.setdefault(self._runner_key, set())

    def unique_id(self) -> tuple[object, ...]:
        # Tactic IDs are persisted by the shared autotuner. Include the ordered
        # definitions so a reorder or field change cannot reinterpret an ID.
        return (_tactic_set_key(), *self.signature)

    def get_valid_tactics(
        self,
        inputs: list[torch.Tensor],
        profile: OptimizationProfile,
        **kwargs,
    ) -> list[int]:
        del inputs, profile, kwargs
        return [
            tactic_id
            for tactic_id in range(len(FP4_CONV_TACTICS))
            if tactic_id not in self._failed_tactics
        ]

    @staticmethod
    def resolve_tactic(tactic: object) -> FP4ConvTactic:
        if tactic == -1:
            return FP4_CONV_FALLBACK_TACTIC
        if not isinstance(tactic, int) or not 0 <= tactic < len(FP4_CONV_TACTICS):
            logger.warning_once(
                f"Wan NVFP4 Conv3d received invalid tactic ID {tactic}; using fallback",
                key=("wan_nvfp4_conv3d", "invalid_tactic", repr(tactic)),
            )
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
            for tactic_id, candidate in enumerate(FP4_CONV_TACTICS):
                try:
                    self.compile_tactic(candidate)
                except Exception as error:
                    # CuTe compilation may touch CUDA before rejecting a
                    # configuration; clear any pending error before continuing.
                    try:
                        torch.cuda.synchronize()
                    except RuntimeError as sync_error:
                        logger.debug(
                            "Wan NVFP4 Conv3d CUDA synchronize failed after "
                            f"tactic {tactic_id} compilation: {sync_error}"
                        )
                    self._failed_tactics.add(tactic_id)
                    logger.warning_once(
                        "Wan NVFP4 Conv3d could not compile "
                        f"tactic {tactic_id} for {self.signature}, {self.problem_shape}: {error}",
                        key=(
                            "wan_nvfp4_conv3d",
                            "compile_failure",
                            self.signature,
                            self.problem_shape,
                            tactic_id,
                        ),
                    )
            return self.output

        if tactic in self._failed_tactics:
            raise RuntimeError(f"Wan NVFP4 Conv3d tactic {tactic} failed during preparation")
        compiled = self.compile_tactic(self.resolve_tactic(tactic))
        try:
            self.launch(compiled)
        except Exception as error:
            if isinstance(tactic, int) and tactic >= 0:
                self._failed_tactics.add(tactic)
            raise RuntimeError(f"Wan NVFP4 Conv3d tactic {tactic} failed during launch") from error
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
    tuner = AutoTuner.get()
    selection_key = (_tactic_set_key(), *signature, problem_shape)
    bypass_fast_cache = tuner.is_tuning_mode or tuner.is_capturing_tactics
    if bypass_fast_cache:
        # Tuning may replace a local winner during distributed post-merge, and
        # capture/replay must reach choose_one() to force each requested tactic.
        _selected_tactics.pop(selection_key, None)
    elif (cached_selection := _selected_tactics.get(selection_key)) is not None:
        generation, tactic = cached_selection
        if generation == tuner.profiling_cache.generation:
            try:
                launch(compile_tactic(tactic))
            except Exception as error:
                if tactic == FP4_CONV_FALLBACK_TACTIC:
                    raise
                tactic = _launch_fallback_tactic(
                    signature=signature,
                    problem_shape=problem_shape,
                    failed_tactic=tactic,
                    error=error,
                    compile_tactic=compile_tactic,
                    launch=launch,
                )
                _selected_tactics[selection_key] = (generation, tactic)
            return output, tactic
        _selected_tactics.pop(selection_key, None)

    runner = FP4ConvTunableRunner(
        signature=signature,
        problem_shape=problem_shape,
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
    tactic = runner.resolve_tactic(tactic_id)
    try:
        selected_runner(inputs, tactic=tactic_id)
    except Exception as error:
        if tactic == FP4_CONV_FALLBACK_TACTIC:
            raise
        tactic = _launch_fallback_tactic(
            signature=signature,
            problem_shape=problem_shape,
            failed_tactic=tactic,
            error=error,
            compile_tactic=compile_tactic,
            launch=launch,
        )
    # Cache only selections tied to the current profiling-cache generation.
    # Tuning and capture/replay must continue through choose_one().
    if not bypass_fast_cache:
        _selected_tactics[selection_key] = (tuner.profiling_cache.generation, tactic)
    logger.debug_once(
        f"Wan NVFP4 Conv3d selected tactic {tactic} for {signature}",
        key=("wan_nvfp4_conv3d", signature, tactic),
    )
    return output, tactic
