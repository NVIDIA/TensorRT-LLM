# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SeaCache step-skipping logic (https://arxiv.org/abs/2602.18993).

SeaCache decides per denoising step whether a transformer stack must run. Its
indicator is the noisy input latent passed through a spectral (Wiener-style)
filter that keeps the content expected to survive denoising at the current
noise level. When the accumulated relative-L1 change of that indicator since
the last computed step stays below a threshold, the stack is skipped and its
residual (output minus input) is predicted from the residuals of the last two
computed steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

# Exponent of the clean-signal power prior P(f) ~ 1 / |f|^power used for video latents.
SEA_POWER_EXP = 3.0


def apply_sea_filter(latent: torch.Tensor, sigma: float, power_exp: float = SEA_POWER_EXP):
    """Filter a ``[T, H, W, C]`` latent with the separable SEA gain over ``T, H, W``.

    The per-axis gain ``s * P / (s^2 * P + n^2)`` uses signal scale ``s = 1 - sigma``
    and noise scale ``n = sigma`` of a flow-matching latent. The combined gain is
    normalized to unit mean so distances stay comparable across noise levels.
    Returns a tensor of the input's shape and dtype.
    """
    dtype = latent.dtype
    latent = latent.contiguous().float()
    dims = (0, 1, 2)
    spectrum = torch.fft.fftn(latent, dim=dims)

    sigma = min(max(float(sigma), 1e-6), 1.0 - 1e-6)
    signal_scale = 1.0 - sigma
    noise_scale = sigma
    gain = None
    for axis in dims:
        freqs = torch.fft.fftfreq(latent.shape[axis], device=latent.device, dtype=torch.float32)
        clean_power = 1.0 / (freqs.abs().pow(power_exp) + 1e-16)
        axis_gain = (
            signal_scale * clean_power / (signal_scale**2 * clean_power + noise_scale**2 + 1e-16)
        )
        shape = [1] * latent.ndim
        shape[axis] = axis_gain.shape[0]
        axis_gain = axis_gain.reshape(shape)
        gain = axis_gain if gain is None else gain * axis_gain
    gain = gain / gain.mean()
    return torch.fft.ifftn(spectrum * gain, dim=dims).real.to(dtype)


def build_indicator(latents: torch.Tensor, sigma: float) -> torch.Tensor:
    """SEA-filter each sample of ``[B, C, T, H, W]`` latents; returns ``[B, T, H, W, C]``."""
    return torch.stack([apply_sea_filter(sample.movedim(0, -1), sigma) for sample in latents])


def indicator_distance(current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    """Mean over samples of the relative L1 distance between two indicators (0-dim tensor)."""
    dims = tuple(range(1, current.ndim))
    numerator = (current.float() - previous.float()).abs().mean(dim=dims)
    denominator = previous.float().abs().mean(dim=dims) + 1e-16
    return (numerator / denominator).mean()


def predict_stack_output(
    history: List[Tuple[int, torch.Tensor]], step: int, stack_input: torch.Tensor
) -> torch.Tensor:
    """Return ``stack_input`` plus the residual predicted for ``step``.

    The residual is extrapolated linearly from the last two computed steps; with
    a single computed step its residual is reused unchanged.
    """
    if len(history) < 2:
        return stack_input + history[-1][1]
    (step_0, residual_0), (step_1, residual_1) = history[-2:]
    scale = (step - step_1) / (step_1 - step_0)
    # residual_1 + scale * (residual_1 - residual_0) with a single full-size allocation.
    return torch.lerp(residual_1, residual_0, -scale).add_(stack_input)


@dataclass
class _TrajectoryState:
    last_step: Optional[int] = None
    previous_indicator: Optional[torch.Tensor] = None
    accumulated_distance: float = 0.0
    consecutive_cached: int = 0
    history: List[Tuple[int, torch.Tensor]] = field(default_factory=list)


class SeaCacheController:
    """Per-generation SeaCache state and the compute-or-reuse decision for one stack.

    One instance serves one transformer stack. Call :meth:`reset` before each
    generation and :meth:`step` once per denoising step, in step order.
    """

    def __init__(self, threshold: float, max_consecutive_cached_steps: int):
        self.threshold = threshold
        self.max_consecutive_cached_steps = max_consecutive_cached_steps
        self.num_inference_steps = 0
        self._state = _TrajectoryState()
        self.cached_steps: List[int] = []
        self.total_steps = 0
        self._warned_non_adjacent = False

    def reset(self, num_inference_steps: int) -> None:
        self.num_inference_steps = num_inference_steps
        self._state = _TrajectoryState()
        self.cached_steps = []
        self.total_steps = 0

    def _should_compute(self, step: int, indicator: torch.Tensor) -> bool:
        state = self._state
        if state.last_step is not None and step != state.last_step + 1:
            if not self._warned_non_adjacent:
                self._warned_non_adjacent = True
                logger.warning(
                    f"SeaCache: step {step} does not follow step {state.last_step}; "
                    "restarting the cache trajectory."
                )
            self._state = state = _TrajectoryState()
        state.last_step = step

        previous = state.previous_indicator
        state.previous_indicator = indicator
        forced = (
            step < 1
            or step >= self.num_inference_steps - 1
            or (
                self.max_consecutive_cached_steps > 0
                and state.consecutive_cached >= self.max_consecutive_cached_steps
            )
            or not state.history
            or previous is None
            or previous.shape != indicator.shape
        )
        if forced:
            state.accumulated_distance = 0.0
            return True

        state.accumulated_distance += float(indicator_distance(indicator, previous))
        if state.accumulated_distance < self.threshold:
            return False
        state.accumulated_distance = 0.0
        return True

    def step(
        self,
        step: int,
        sigma: float,
        latents: torch.Tensor,
        stack_input: torch.Tensor,
        run_stack: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Return the stack output for this step, computed or predicted.

        Args:
            step: Denoising step index, starting at 0.
            sigma: Flow-matching noise level of this step's input, in ``[0, 1]``.
            latents: Raw noisy latents ``[B, C, T, H, W]`` fed to the transformer.
            stack_input: Input of the cached stack.
            run_stack: Runs the stack on ``stack_input``; must return the same shape.
        """
        self.total_steps += 1
        should_compute = self._should_compute(step, build_indicator(latents, sigma))
        # Read after the decision: a non-adjacent step replaces the state.
        state = self._state
        if not should_compute:
            if all(residual.shape == stack_input.shape for _, residual in state.history):
                state.consecutive_cached += 1
                self.cached_steps.append(step)
                return predict_stack_output(state.history, step, stack_input)
            logger.warning("SeaCache: stack input shape changed; running the full stack.")
            state.history.clear()
            state.accumulated_distance = 0.0

        output = run_stack(stack_input)
        if output.shape == stack_input.shape:
            # Drop the oldest residual first so at most two are alive at once.
            del state.history[:-1]
            state.history.append((step, output - stack_input))
        else:
            state.history.clear()
        state.consecutive_cached = 0
        return output

    def get_stats(self):
        total = self.total_steps
        cached = len(self.cached_steps)
        return {
            "hit_rate": cached / max(total, 1),
            "total": total,
            "cached": cached,
            "cached_step_indices": list(self.cached_steps),
        }
