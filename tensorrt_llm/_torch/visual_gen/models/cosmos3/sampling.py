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
"""Per-checkpoint sampling policy for Cosmos3.

Exactly two recipes are supported, read from the checkpoint's
``scheduler/scheduler_config.json``:

* ``UniPCMultistepScheduler`` without fixed sigmas — base checkpoints:
  request tables drive steps/guidance, and the flow shift comes from the mode
  table unless the request overrides it.
* ``FlowMatchEulerDiscreteScheduler`` with ``stochastic_sampling`` enabled
  and a nonempty ``fixed_step_sampler_config.t_list`` — distilled
  checkpoints: the step count is locked to the schedule, classifier-free
  guidance is baked into the weights (scale 1.0), and every step draws
  seeded SDE noise.

The pipeline owns the scheduler instances; :class:`Cosmos3SamplingPolicy` is
an immutable value object of config-derived facts whose methods take
schedulers as arguments.

A request may select the sigma grid of a base checkpoint with
``extra_params["sigma_grid"]`` (:data:`SIGMA_GRIDS`); :func:`resolve_sampling_schedule`
turns that selection, the request's ``flow_shift`` and the mode's own defaults
into a :class:`SamplingSchedule`. Omitting ``sigma_grid`` selects checkpoint- and
mode-specific defaults.
"""

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Optional, get_args

import numpy as np
from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler

from tensorrt_llm.logger import logger

# Distilled checkpoints bake classifier-free guidance into the weights; the
# only valid scale is 1.0 ("off": a single conditional forward per step).
DISTILLED_GUIDANCE_SCALE = 1.0

# Grids a request can select through ``extra_params["sigma_grid"]``:
#   karras    -- the checkpoint's Karras levels converted to flow sigmas; no flow shift.
#   native    -- the Cosmos framework's shifted-linear flow grid.
#   diffusers -- the pinned Diffusers UniPC shifted-linear flow grid.
SigmaGrid = Literal["karras", "native", "diffusers"]
SIGMA_GRIDS: tuple[str, ...] = get_args(SigmaGrid)

# Every grid a resolved schedule can name: the selectable grids, plus
#   checkpoint_native_flow -- descending linear flow sigmas starting at 1 - 1/T, excluding
#                             the terminal zero, for checkpoints that declare the native
#                             flow schedule (1 - 1/T is not rounded to float32, unlike
#                             ``native``);
#   scheduler_config       -- the non-Karras grid of a scheduler without flow sigmas;
#   fixed                  -- a distilled checkpoint's fixed sigmas.
ResolvedGrid = Literal[
    "karras", "native", "diffusers", "checkpoint_native_flow", "scheduler_config", "fixed"
]
RESOLVED_GRIDS: tuple[str, ...] = get_args(ResolvedGrid)

# Diffusers UniPC lowers a first shifted flow sigma within this distance of 1
# (its log-SNR is singular at sigma 1). The native grid never applies it.
_DIFFUSERS_FLOW_SIGMA_EPS = 1e-6


def native_flow_upper_sigma(num_train_timesteps: int) -> float:
    """The first unshifted level of the native grid: ``1 - 1/num_train_timesteps`` rounded to float32."""
    # The Cosmos framework stores its training sigmas 1 - linspace(1, 1/T, T)[::-1] as
    # float32 and spaces inference levels between the stored endpoints, which it reads
    # back as Python floats.
    return float(np.float32(1.0 - 1.0 / num_train_timesteps))


def native_flow_base_sigmas(num_inference_steps: int, num_train_timesteps: int) -> np.ndarray:
    """Return ``num_inference_steps`` unshifted, evenly spaced sigma levels as a float64 NumPy array.

    The first level is :func:`native_flow_upper_sigma`, ``1 - 1/num_train_timesteps``
    rounded to float32; the terminal zero is excluded.
    """
    upper = native_flow_upper_sigma(num_train_timesteps)
    return np.linspace(upper, 0.0, num_inference_steps + 1)[:-1]


def _shift_flow_sigma(sigma: float, flow_shift: float) -> float:
    return flow_shift * sigma / (1.0 + (flow_shift - 1.0) * sigma)


@dataclass(frozen=True)
class SamplingSchedule:
    """The concrete sigma grid and flow shift one request samples with.

    ``grid`` is the grid the request's scheduler is programmed with (one of
    :data:`RESOLVED_GRIDS`). ``flow_shift`` is the shift its UniPC scheduler is built
    with; the ``karras`` and ``fixed`` grids do not use it. ``flow_shift_explicit``
    records whether the request supplied ``flow_shift`` itself rather than taking the
    mode's.
    """

    grid: ResolvedGrid
    flow_shift: float
    flow_shift_explicit: bool = False

    def __post_init__(self) -> None:
        if self.grid not in RESOLVED_GRIDS:
            raise ValueError(
                f"Unknown Cosmos3 sigma grid {self.grid!r}; expected one of {RESOLVED_GRIDS}."
            )

    @property
    def use_karras_sigmas(self) -> bool:
        """Whether the request's UniPC scheduler is built with Karras sigmas."""
        return self.grid == "karras"

    @property
    def inactive_flow_shift(self) -> bool:
        """Whether the request supplied a ``flow_shift`` while sampling the Karras grid, which ignores it."""
        return self.flow_shift_explicit and self.grid == "karras"


def resolve_sampling_schedule(
    sampling: "Cosmos3SamplingPolicy",
    *,
    sigma_grid: Optional[SigmaGrid],
    flow_shift: Optional[float],
    mode_flow_shift: float,
    mode_use_karras_sigmas: Optional[bool],
) -> SamplingSchedule:
    """Resolve one request's concrete sampling schedule.

    Each setting is taken from the first source that sets it: the request
    (``sigma_grid``, ``flow_shift``), then the mode (``mode_use_karras_sigmas``,
    ``mode_flow_shift``), then the checkpoint. Without ``sigma_grid`` the grid is
    chosen by :func:`_default_schedule`.

    Args:
        sampling: The checkpoint's sampling policy.
        sigma_grid: ``extra_params["sigma_grid"]``; ``None`` when omitted.
        flow_shift: ``extra_params["flow_shift"]``; ``None`` when omitted.
        mode_flow_shift: The shift the mode uses when the request names none.
        mode_use_karras_sigmas: ``False`` when the mode needs a non-Karras grid
            (V2V and transfer); ``None`` to follow the checkpoint.

    Raises:
        ValueError: The selection cannot be honored for this checkpoint or the
            explicit values conflict.
    """
    if sigma_grid is None:
        return _default_schedule(
            sampling,
            flow_shift=flow_shift,
            mode_flow_shift=mode_flow_shift,
            mode_use_karras_sigmas=mode_use_karras_sigmas,
        )
    if sampling.is_distilled:
        raise ValueError(
            f"sigma_grid={sigma_grid!r} is not supported for a distilled Cosmos3 "
            "checkpoint: its sampling schedule is fixed by the checkpoint. Omit "
            "sigma_grid."
        )
    if sigma_grid not in SIGMA_GRIDS:
        raise ValueError(f"sigma_grid must be one of {list(SIGMA_GRIDS)}, got {sigma_grid!r}.")
    base_config = sampling.unipc_base_config
    if base_config is not None and not _config_get(base_config, "use_flow_sigmas", False):
        raise ValueError(
            f"sigma_grid={sigma_grid!r} requires a flow-sigma UniPC scheduler; this "
            "checkpoint's scheduler config does not set use_flow_sigmas."
        )

    if sigma_grid == "karras":
        if flow_shift is not None:
            raise ValueError(
                "flow_shift has no effect with sigma_grid='karras': the Karras grid is "
                "not flow-shifted. Omit flow_shift, or select sigma_grid='native' or "
                "'diffusers'."
            )
        return SamplingSchedule(grid="karras", flow_shift=mode_flow_shift)

    shift = mode_flow_shift if flow_shift is None else float(flow_shift)
    if not math.isfinite(shift) or shift <= 0.0:
        raise ValueError(f"flow_shift must be a positive finite number, got {shift!r}.")
    if sigma_grid == "native":
        _validate_native_grid(base_config, shift)
    return SamplingSchedule(
        grid=sigma_grid, flow_shift=shift, flow_shift_explicit=flow_shift is not None
    )


def _default_schedule(
    sampling: "Cosmos3SamplingPolicy",
    *,
    flow_shift: Optional[float],
    mode_flow_shift: float,
    mode_use_karras_sigmas: Optional[bool],
) -> SamplingSchedule:
    """The schedule of a request that selects no ``sigma_grid``.

    The grid is the fixed sigmas of a distilled checkpoint; the checkpoint's linear
    flow grid when it declares the native flow schedule; otherwise Karras when the
    mode or, failing that, the scheduler config enables it, and else the scheduler
    config's flow grid (``diffusers``) or non-flow grid (``scheduler_config``). The
    shift is the request's ``flow_shift``, else the mode's.
    """
    if sampling.is_distilled:
        grid = "fixed"
    elif sampling.native_flow_schedule:
        grid = "checkpoint_native_flow"
    else:
        base_config = sampling.unipc_base_config
        karras = mode_use_karras_sigmas
        if karras is None:
            karras = bool(_config_get(base_config, "use_karras_sigmas", False))
        if karras:
            grid = "karras"
        elif _config_get(base_config, "use_flow_sigmas", False):
            grid = "diffusers"
        else:
            grid = "scheduler_config"
    explicit = flow_shift is not None
    return SamplingSchedule(
        grid=grid,
        flow_shift=float(flow_shift) if explicit else mode_flow_shift,
        flow_shift_explicit=explicit,
    )


def _validate_native_grid(scheduler_config: Any, flow_shift: float) -> None:
    """Raise ``ValueError`` when ``native`` cannot reproduce the Cosmos framework grid.

    The framework grid needs the scheduler's default flow settings, and a first
    shifted level far enough from 1 that UniPC does not adjust it.
    """
    unsupported = [
        key
        for key, value in (
            ("use_dynamic_shifting", _config_get(scheduler_config, "use_dynamic_shifting", False)),
            ("shift_terminal", _config_get(scheduler_config, "shift_terminal", None)),
        )
        if value
    ]
    if _config_get(scheduler_config, "final_sigmas_type", "zero") != "zero":
        unsupported.append("final_sigmas_type")
    if unsupported:
        raise ValueError(
            "sigma_grid='native' reproduces the Cosmos framework grid, which needs the "
            "scheduler's default flow settings; this checkpoint sets "
            f"{', '.join(unsupported)}."
        )
    num_train = int(_config_get(scheduler_config, "num_train_timesteps", 1000))
    upper = _shift_flow_sigma(native_flow_upper_sigma(num_train), flow_shift)
    if abs(upper - 1.0) < _DIFFUSERS_FLOW_SIGMA_EPS:
        raise ValueError(
            f"flow_shift={flow_shift} moves the first native sigma within "
            f"{_DIFFUSERS_FLOW_SIGMA_EPS} of 1, where the UniPC solver alters it; the "
            "native grid cannot be reproduced at this shift."
        )


def _validate_unipc_sigmas(scheduler: Any, schedule: SamplingSchedule) -> None:
    """Raise ``ValueError`` when UniPC cannot step through ``scheduler``'s programmed sigmas.

    UniPC steps between consecutive levels in log-SNR. Every level must be finite, the
    levels it steps from must be positive and strictly decreasing, and the terminal
    level must lie between 0 and the last of them. Flow sigmas must also stay below 1,
    where the log-SNR is singular. An extreme ``flow_shift`` breaks these through
    rounding: leading levels reach 1 or more, or stop decreasing.
    """
    sigmas = scheduler.sigmas.numpy()
    levels, terminal = sigmas[:-1], sigmas[-1]
    bad = ~np.isfinite(levels) | (levels <= 0.0)
    if _config_get(scheduler.config, "use_flow_sigmas", False):
        bad |= levels >= 1.0
    bad[1:] |= levels[1:] >= levels[:-1]
    if bad.any():
        index = int(np.argmax(bad))
        problem = f"level {index} is {float(levels[index])}"
        if index > 0:
            problem += f" after {float(levels[index - 1])}"
    elif not (np.isfinite(terminal) and 0.0 <= terminal <= levels[-1]):
        problem = f"the terminal level is {float(terminal)}"
    else:
        return
    raise ValueError(
        f"flow_shift={schedule.flow_shift} on the {schedule.grid!r} grid with "
        f"{len(levels)} steps gives sigmas the UniPC solver cannot step through. Levels "
        "must be finite, positive, strictly decreasing and, for flow sigmas, below 1; "
        f"{problem}. Use a less extreme flow_shift."
    )


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    """Fetch a key from a plain dict, a diffusers FrozenDict, or a config object."""
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _resolve_distilled_sigmas(scheduler_config: Any) -> "tuple[float, ...] | None":
    """``fixed_step_sampler_config.t_list`` as floats, or None (base checkpoints)."""
    fixed_step_cfg = _config_get(scheduler_config, "fixed_step_sampler_config")
    t_list = _config_get(fixed_step_cfg, "t_list") if fixed_step_cfg else None
    if not t_list:
        return None
    return tuple(float(sigma) for sigma in t_list)


def load_scheduler(checkpoint_dir: str, subfolder: str = "scheduler") -> Any:
    """Instantiate the scheduler class the checkpoint declares.

    Base checkpoints declare ``UniPCMultistepScheduler`` (a missing declaration
    also resolves to UniPC, preserving pre-declaration Cosmos3 behavior);
    distilled ones declare ``FlowMatchEulerDiscreteScheduler``. An explicitly
    unknown declaration is a load-time error — silently substituting UniPC
    would sample the checkpoint with the wrong integrator.
    """
    config_path = os.path.join(checkpoint_dir, subfolder, "scheduler_config.json")
    class_name = ""
    if os.path.exists(config_path):
        with open(config_path) as f:
            class_name = json.load(f).get("_class_name", "")
    if class_name == "FlowMatchEulerDiscreteScheduler":
        scheduler_cls = FlowMatchEulerDiscreteScheduler
    elif class_name in ("", None, "UniPCMultistepScheduler"):
        scheduler_cls = UniPCMultistepScheduler
    else:
        raise ValueError(
            f"Unsupported Cosmos3 scheduler class {class_name!r}; supported: "
            "UniPCMultistepScheduler (base), FlowMatchEulerDiscreteScheduler (distilled)."
        )
    return scheduler_cls.from_pretrained(checkpoint_dir, subfolder=subfolder)


@dataclass(frozen=True)
class Cosmos3SamplingPolicy:
    """Immutable sampling facts of a loaded Cosmos3 checkpoint.

    Construct via :meth:`from_scheduler`, which validates the recipe at load
    time. A default-constructed policy (all fields ``None``) is the explicit
    pre-load placeholder the pipeline holds before its scheduler exists: not
    distilled, no flow-shift rebuild capability (``set_flow_shift`` is a
    no-op), and replaced by ``from_scheduler`` when components load.

    Methods take scheduler instances as arguments; the current flow shift is
    read from the supplied scheduler's config rather than tracked here.
    """

    # Fixed distilled schedule (t_list); None for base checkpoints.
    fixed_sigmas: "tuple[float, ...] | None" = None
    # Checkpoint scheduler config, kept for flow-shift rebuilds (UniPC only).
    unipc_base_config: Optional[Any] = None
    # Checkpoint-declared (model_index): UniPC runs on explicit linear flow
    # sigmas with a runtime shift instead of the config's karras grid.
    native_flow_schedule: bool = False

    @classmethod
    def from_scheduler(
        cls, scheduler: Any, native_flow_schedule: bool = False
    ) -> "Cosmos3SamplingPolicy":
        """Derive the policy from a loaded scheduler's config.

        Valid recipes: UniPC without fixed sigmas (base) and stochastic
        FlowMatchEuler with a nonempty ``fixed_step_sampler_config.t_list``
        (distilled); anything else fails here, at load time.
        """
        fixed_sigmas = _resolve_distilled_sigmas(scheduler.config)
        is_unipc = isinstance(scheduler, UniPCMultistepScheduler)
        is_flow_match = isinstance(scheduler, FlowMatchEulerDiscreteScheduler)

        if (
            _config_get(scheduler.config, "fixed_step_requires_explicit_sigmas", False)
            and fixed_sigmas is None
        ):
            raise ValueError(
                "Malformed distilled checkpoint: the scheduler config declares "
                "fixed_step_requires_explicit_sigmas but carries no usable "
                "fixed_step_sampler_config.t_list."
            )

        if is_unipc and fixed_sigmas is None:
            return cls(
                fixed_sigmas=None,
                unipc_base_config=scheduler.config,
                native_flow_schedule=native_flow_schedule,
            )

        if is_flow_match and fixed_sigmas is not None:
            if not _config_get(scheduler.config, "stochastic_sampling", False):
                raise ValueError(
                    "Unsupported Cosmos3 sampling recipe: FlowMatchEulerDiscreteScheduler "
                    "declares a fixed step schedule without stochastic_sampling. The "
                    "distilled recipe draws SDE noise at every step; running the schedule "
                    "as an ODE would sample the checkpoint incorrectly."
                )
            fixed_step_cfg = _config_get(scheduler.config, "fixed_step_sampler_config")
            sample_type = _config_get(fixed_step_cfg, "sample_type")
            if sample_type is not None and sample_type != "sde":
                raise ValueError(
                    "Unsupported Cosmos3 sampling recipe: fixed_step_sampler_config "
                    f"declares sample_type={sample_type!r}; only 'sde' is supported."
                )
            logger.info(
                f"Distilled Cosmos3 checkpoint: fixed {len(fixed_sigmas)}-step schedule "
                f"{list(fixed_sigmas)}, classifier-free guidance baked in."
            )
            return cls(fixed_sigmas=fixed_sigmas, unipc_base_config=None)

        raise ValueError(
            f"Unsupported Cosmos3 sampling recipe: {type(scheduler).__name__} with "
            f"fixed sigmas {'present' if fixed_sigmas is not None else 'absent'}. "
            "Supported: UniPCMultistepScheduler without fixed sigmas (base), "
            "stochastic FlowMatchEulerDiscreteScheduler with "
            "fixed_step_sampler_config.t_list (distilled)."
        )

    @property
    def is_distilled(self) -> bool:
        return self.fixed_sigmas is not None

    def generation_default_overrides(self) -> dict:
        """Checkpoint-mandated overrides of the table generation defaults.

        Merged over the checkpoint family's video table by the pipeline's
        ``default_generation_params``, so executor-merged requests arrive
        carrying the checkpoint's true defaults.
        """
        if not self.is_distilled:
            return {}
        return {
            "num_inference_steps": len(self.fixed_sigmas),
            "guidance_scale": DISTILLED_GUIDANCE_SCALE,
        }

    def num_steps(self, default: int) -> int:
        """The only step count this policy can run: fixed for distilled, else ``default``."""
        return len(self.fixed_sigmas) if self.is_distilled else default

    def validate_request(
        self, num_inference_steps: Optional[int], guidance_scale: Optional[float]
    ) -> None:
        """Reject sampling parameters incompatible with a distilled checkpoint."""
        if not self.is_distilled:
            return
        distilled_steps = len(self.fixed_sigmas)
        if num_inference_steps is not None and num_inference_steps != distilled_steps:
            raise ValueError(
                "This is a distilled Cosmos3 checkpoint; the step count is fixed by the "
                f"scheduler's fixed_step_sampler_config.t_list ({distilled_steps} steps). "
                f"num_inference_steps must be {distilled_steps} or left unset "
                f"(got {num_inference_steps})."
            )
        if guidance_scale is not None and float(guidance_scale) != DISTILLED_GUIDANCE_SCALE:
            raise ValueError(
                "This is a distilled Cosmos3 checkpoint; classifier-free guidance is baked "
                f"into the weights. guidance_scale must be {DISTILLED_GUIDANCE_SCALE} or "
                f"left unset (got {guidance_scale})."
            )

    def set_timesteps(
        self,
        scheduler: Any,
        num_inference_steps: int,
        device: Any,
        schedule: SamplingSchedule,
    ) -> None:
        """Program ``scheduler`` with ``schedule``'s timesteps for one generation.

        ``scheduler`` must be built for ``schedule`` (see :meth:`set_flow_shift`).
        A distilled checkpoint's fixed schedule ignores ``num_inference_steps``; every
        other grid runs ``num_inference_steps`` steps. Programming resets the
        scheduler's solver state and starts the next generation at step 0.

        Raises:
            RuntimeError: ``schedule`` does not match this checkpoint or ``scheduler``.
            ValueError: The programmed sigmas cannot be stepped through (see
                :func:`_validate_unipc_sigmas`), which an extreme ``flow_shift`` causes.
        """
        is_fixed = schedule.grid == "fixed"
        if is_fixed != self.is_distilled:
            raise RuntimeError(
                f"Cosmos3 sigma grid {schedule.grid!r} does not match a "
                f"{'distilled' if self.is_distilled else 'base'} checkpoint."
            )
        if is_fixed:
            scheduler.set_timesteps(sigmas=list(self.fixed_sigmas), device=device)
            return
        scheduler_karras = bool(_config_get(scheduler.config, "use_karras_sigmas", False))
        if schedule.use_karras_sigmas != scheduler_karras:
            raise RuntimeError(
                f"Cosmos3 sigma grid {schedule.grid!r} does not match a scheduler built with "
                f"use_karras_sigmas={scheduler_karras}."
            )
        num_train = int(_config_get(scheduler.config, "num_train_timesteps", 1000))
        # The native and checkpoint_native_flow levels are passed to UniPC unshifted, so
        # its flow_shift shifts them exactly once; the karras, diffusers and
        # scheduler_config grids are generated by the scheduler from its config.
        if schedule.grid == "native":
            sigmas = native_flow_base_sigmas(num_inference_steps, num_train)
        elif schedule.grid == "checkpoint_native_flow":
            # A numpy array is required (a list breaks diffusers 0.39).
            sigmas = np.linspace(1.0 - 1.0 / num_train, 0.0, num_inference_steps + 1)[:-1]
        else:
            sigmas = None
        if sigmas is None:
            scheduler.set_timesteps(num_inference_steps, device=device)
        else:
            scheduler.set_timesteps(num_inference_steps, device=device, sigmas=sigmas)
        _validate_unipc_sigmas(scheduler, schedule)
        # A strongly shifted grid can repeat its leading integer timesteps (999, 999, ...).
        # UniPC looks up its first step by timestep value and, on repeats, starts at the
        # second match; every generation runs the whole schedule, so start at step 0.
        scheduler.set_begin_index(0)

    def scheduler_step_kwargs(self, generator: Any) -> dict:
        """Extra kwargs each ``scheduler.step()`` call requires.

        The distilled FlowMatchEuler scheduler is stochastic: every step draws
        SDE noise, which must come from the request-seeded ``generator`` —
        otherwise it comes from the process-global RNG, breaking seed
        reproducibility and diverging the replicated latents across ranks
        (each rank's global RNG state is independent). UniPC steps are
        deterministic and accept no ``generator`` argument, so base
        checkpoints pass nothing.
        """
        if self.is_distilled:
            return {"generator": generator}
        return {}

    @property
    def checkpoint_flow_shift(self) -> float:
        """The flow shift the checkpoint shipped with (UniPC only; 1.0 otherwise)."""
        if self.unipc_base_config is None:
            return 1.0
        return float(_config_get(self.unipc_base_config, "flow_shift", 1.0) or 1.0)

    def set_flow_shift(self, scheduler: Any, schedule: SamplingSchedule) -> Any:
        """Return a UniPC scheduler built for ``schedule``'s flow shift and Karras setting.

        ``scheduler`` itself is returned when its config already matches; otherwise a
        new scheduler is built from the checkpoint config, which is never modified.
        Distilled checkpoints return ``scheduler`` unchanged: their fixed schedule takes
        no flow shift.
        """
        if self.unipc_base_config is None:
            return scheduler
        target_shift = float(schedule.flow_shift)
        target_karras = schedule.use_karras_sigmas
        current_shift = float(_config_get(scheduler.config, "flow_shift", 1.0) or 1.0)
        current_karras = bool(_config_get(scheduler.config, "use_karras_sigmas", False))
        if current_shift == target_shift and current_karras == target_karras:
            return scheduler
        return UniPCMultistepScheduler.from_config(
            self.unipc_base_config,
            flow_shift=target_shift,
            use_karras_sigmas=target_karras,
        )
