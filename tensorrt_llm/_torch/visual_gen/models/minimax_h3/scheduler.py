# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.

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

"""Native MiniMax-H3 FlowMatch scheduler boundary.

The MiniMax-H3 denoising loop is TRTLLM-owned.  This module provides the native
scheduler API used by ``pipeline_minimax_h3.py`` and avoids importing upstream
Diffusers runtime code.  The scheduler math is covered by MiniMax-H3 component
contract tests and by the run's module parity report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import torch


@dataclass
class MinimaxH3SchedulerConfig:
    """Serializable FlowMatch/Euler scheduler configuration.

    Defaults match the MiniMax-H3 two-scheduler contract used by the existing
    tests: audio uses shift=3.0 and video overrides to shift=12.0 in
    ``MiniMaxH3Pipeline.post_load_weights``.
    """

    num_train_timesteps: int = 1000
    shift: float = 3.0
    use_dynamic_shifting: bool = True
    base_shift: float = 0.5
    max_shift: float = 1.15
    base_image_seq_len: int = 256
    max_image_seq_len: int = 4096
    invert_sigmas: bool = False
    shift_terminal: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "MinimaxH3SchedulerConfig":
        if data is None:
            return cls()
        known = {name for name in cls.__dataclass_fields__ if name != "extra"}  # type: ignore[attr-defined]
        values = {key: data[key] for key in data if key in known}
        values["extra"] = {
            key: value
            for key, value in data.items()
            if key not in known and not str(key).startswith("_")
        }
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "num_train_timesteps": self.num_train_timesteps,
            "shift": self.shift,
            "use_dynamic_shifting": self.use_dynamic_shifting,
            "base_shift": self.base_shift,
            "max_shift": self.max_shift,
            "base_image_seq_len": self.base_image_seq_len,
            "max_image_seq_len": self.max_image_seq_len,
            "invert_sigmas": self.invert_sigmas,
            "shift_terminal": self.shift_terminal,
        }
        data.update(self.extra)
        return data


@dataclass
class MiniMaxH3SchedulerOutput:
    prev_sample: torch.Tensor


class MiniMaxH3Scheduler:
    """FlowMatch Euler scheduler with the API needed by MiniMax-H3.

    The public surface mirrors the scheduler methods exercised by the native
    MiniMax-H3 pipeline: ``from_pretrained``, ``register_to_config``,
    ``set_timesteps``, ``scale_noise``, ``scale_model_input``, and ``step``.
    """

    config: MinimaxH3SchedulerConfig
    order = 1

    def __init__(
        self,
        config: MinimaxH3SchedulerConfig | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(config, MinimaxH3SchedulerConfig):
            data = config.to_dict()
        elif isinstance(config, Mapping):
            data = dict(config)
        elif config is None:
            data = {}
        else:
            raise TypeError(f"Unsupported MiniMax-H3 scheduler config: {type(config)!r}")
        data.update(kwargs)
        data.pop("use_diffusers", None)
        self.config = MinimaxH3SchedulerConfig.from_dict(data)
        self.timesteps = torch.empty(0)
        self.sigmas = torch.empty(0)
        self.num_inference_steps: Optional[int] = None
        self._step_index: Optional[int] = None
        self._begin_index: Optional[int] = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        subfolder: str | None = None,
        **kwargs: Any,
    ) -> "MiniMaxH3Scheduler":
        path = Path(pretrained_model_name_or_path).expanduser()
        if subfolder:
            path = path / str(subfolder)
        config_path = path / "scheduler_config.json"
        if not config_path.is_file():
            config_path = path / "config.json"
        payload: dict[str, Any] = {}
        if config_path.is_file():
            try:
                payload = json.loads(config_path.read_text())
            except (OSError, json.JSONDecodeError):
                payload = {}
        payload.update(kwargs)
        return cls(payload)

    def register_to_config(self, **kwargs: Any) -> None:
        data = self.config.to_dict()
        data.update(kwargs)
        data.pop("use_diffusers", None)
        self.config = MinimaxH3SchedulerConfig.from_dict(data)

    @property
    def uses_diffusers(self) -> bool:
        return False

    @property
    def shift(self) -> float:
        return float(self.config.shift)

    @property
    def step_index(self) -> Optional[int]:
        return self._step_index

    @property
    def begin_index(self) -> Optional[int]:
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = begin_index

    def set_shift(self, shift: float) -> None:
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}.")
        self.register_to_config(shift=float(shift))

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: torch.device | str | None = None,
        sigmas: list[float] | torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        if sigmas is None:
            if num_inference_steps is None or num_inference_steps < 2:
                raise ValueError(
                    "`set_timesteps` requires either an explicit `sigmas` schedule or "
                    f"`num_inference_steps` >= 2, got {num_inference_steps}."
                )
            base_sigmas = torch.linspace(
                1.0,
                0.0,
                int(num_inference_steps),
                dtype=torch.float32,
            )
            schedule = torch.unique_consecutive(self._apply_shift(base_sigmas))
        else:
            schedule = torch.as_tensor(sigmas, dtype=torch.float32).flatten().cpu()
            if (
                schedule.numel() < 2
                or not bool((schedule[1:] < schedule[:-1]).all())
                or schedule[-1].item() != 0.0
            ):
                raise ValueError(
                    "`sigmas` must hold at least two strictly decreasing values ending at 0.0."
                )

        self.sigmas = schedule.to(device=device)
        self.timesteps = (1.0 - self.sigmas[:-1]).to(device=device)
        self.num_inference_steps = int(self.timesteps.numel())
        self._step_index = None
        self._begin_index = None
        return self.timesteps

    def _apply_shift(self, sigmas: torch.Tensor) -> torch.Tensor:
        shift = float(self.config.shift)
        if shift == 1.0:
            return sigmas
        return shift * sigmas / (1 + (shift - 1) * sigmas)

    def index_for_timestep(self, timestep: float | torch.Tensor) -> int:
        if self.timesteps.numel() == 0:
            raise RuntimeError("Call `set_timesteps` before looking up a timestep.")
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        indices = (self.timesteps == timestep).nonzero()
        if len(indices) == 0:
            raise ValueError(
                "Passed `timestep` is not in `self.timesteps`. Use a value from the active schedule."
            )
        return indices[0].item()

    def scale_noise(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        timestep = torch.as_tensor(timestep, device=sample.device, dtype=sample.dtype)
        while timestep.ndim < sample.ndim:
            timestep = timestep.unsqueeze(-1)
        return timestep * sample + (1.0 - timestep) * noise

    def scale_model_input(
        self, sample: torch.Tensor, timestep: torch.Tensor | float | int
    ) -> torch.Tensor:
        del timestep
        return sample

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor | float | int,
        sample: torch.Tensor,
        return_dict: bool = True,
        **_: Any,
    ) -> MiniMaxH3SchedulerOutput | tuple[torch.Tensor]:
        if isinstance(timestep, int) or (
            isinstance(timestep, torch.Tensor) and not timestep.is_floating_point()
        ):
            raise ValueError(
                "Integer schedule indices are not valid timesteps. Pass a value from `scheduler.timesteps`."
            )
        if self.sigmas.numel() == 0:
            raise RuntimeError("Call `set_timesteps` before `step`.")
        if self._step_index is None:
            self._step_index = (
                self.index_for_timestep(timestep)
                if self._begin_index is None
                else self._begin_index
            )
        if self._step_index + 1 >= self.sigmas.numel():
            raise IndexError("The active MiniMax-H3 schedule has no remaining Euler step.")

        timestep = torch.as_tensor(timestep, device=sample.device, dtype=sample.dtype)
        sigma_from_timestep = 1.0 - timestep
        while sigma_from_timestep.ndim < sample.ndim:
            sigma_from_timestep = sigma_from_timestep.unsqueeze(-1)
        denoised = sample + sigma_from_timestep * model_output

        compute_dtype = (
            torch.float32 if sample.dtype in (torch.float16, torch.bfloat16) else sample.dtype
        )
        sigma = self.sigmas[self._step_index].to(device=sample.device, dtype=compute_dtype)
        sigma_next = self.sigmas[self._step_index + 1].to(device=sample.device, dtype=compute_dtype)
        ratio = sigma_next / sigma
        prev_sample = ratio * sample.to(dtype=compute_dtype) + (1.0 - ratio) * denoised.to(
            dtype=compute_dtype
        )
        prev_sample = prev_sample.to(dtype=sample.dtype)
        self._step_index += 1
        if return_dict:
            return MiniMaxH3SchedulerOutput(prev_sample=prev_sample)
        return (prev_sample,)
