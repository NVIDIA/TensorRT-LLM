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
  request tables drive steps/guidance; per-mode flow shifts are table facts,
  so the pipeline prebuilds one scheduler per mode at load time.
* ``FlowMatchEulerDiscreteScheduler`` with a nonempty
  ``fixed_step_sampler_config.t_list`` — distilled checkpoints: the step
  count is locked to the schedule, classifier-free guidance is baked into
  the weights (scale 1.0), and the stochastic steps draw seeded noise.

The pipeline owns the scheduler instances; :class:`Cosmos3SamplingPolicy` is
an immutable value object of config-derived facts whose methods take
schedulers as arguments.
"""

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler

from tensorrt_llm.logger import logger

# Distilled checkpoints bake classifier-free guidance into the weights; the
# only valid scale is 1.0 ("off": a single conditional forward per step).
DISTILLED_GUIDANCE_SCALE = 1.0


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

        Valid recipes: UniPC without fixed sigmas (base) and FlowMatchEuler
        with a nonempty ``fixed_step_sampler_config.t_list`` (distilled);
        anything else fails here, at load time.
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
            logger.info(
                f"Distilled Cosmos3 checkpoint: fixed {len(fixed_sigmas)}-step schedule "
                f"{list(fixed_sigmas)}, classifier-free guidance baked in."
            )
            return cls(fixed_sigmas=fixed_sigmas, unipc_base_config=None)

        raise ValueError(
            f"Unsupported Cosmos3 sampling recipe: {type(scheduler).__name__} with "
            f"fixed sigmas {'present' if fixed_sigmas is not None else 'absent'}. "
            "Supported: UniPCMultistepScheduler without fixed sigmas (base), "
            "FlowMatchEulerDiscreteScheduler with fixed_step_sampler_config.t_list "
            "(distilled)."
        )

    @property
    def is_distilled(self) -> bool:
        return self.fixed_sigmas is not None

    def generation_default_overrides(self) -> dict:
        """Checkpoint-mandated overrides of the table generation defaults.

        Merged over ``COSMOS3_720P_PARAMS`` by the pipeline's
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

    def set_timesteps(self, scheduler: Any, num_inference_steps: int, device: Any) -> None:
        """Program a scheduler for one generation: fixed sigmas or a step count."""
        if self.is_distilled:
            scheduler.set_timesteps(sigmas=list(self.fixed_sigmas), device=device)
        elif self.native_flow_schedule:
            # The PyTorch-backend base grid: linear flow sigmas over
            # (1 - 1/T, 0]. UniPC applies its flow_shift to provided sigmas;
            # a numpy array is required (a list breaks diffusers 0.39).
            num_train = int(_config_get(scheduler.config, "num_train_timesteps", 1000))
            sigmas = np.linspace(1.0 - 1.0 / num_train, 0.0, num_inference_steps + 1)[:-1]
            scheduler.set_timesteps(num_inference_steps, device=device, sigmas=sigmas)
        else:
            scheduler.set_timesteps(num_inference_steps, device=device)

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

    def with_flow_shift(self, scheduler: Any, target_shift: Optional[float]) -> Any:
        """A scheduler configured with ``flow_shift=target_shift``.

        Reuses ``scheduler`` when its config already matches, otherwise builds
        a new instance from the checkpoint config (diffusers scheduler configs
        are frozen). Called once per mode at component-load time. Structural
        no-op for distilled checkpoints (no UniPC base config) and for
        ``target_shift=None``.
        """
        if target_shift is None or self.unipc_base_config is None:
            return scheduler
        target_shift = float(target_shift)
        current_shift = float(_config_get(scheduler.config, "flow_shift", 1.0) or 1.0)
        karras_mismatch = self.native_flow_schedule and bool(
            _config_get(scheduler.config, "use_karras_sigmas", False)
        )
        if current_shift == target_shift and not karras_mismatch:
            return scheduler
        overrides = {"flow_shift": target_shift}
        if self.native_flow_schedule:
            overrides["use_karras_sigmas"] = False
        return UniPCMultistepScheduler.from_config(self.unipc_base_config, **overrides)
