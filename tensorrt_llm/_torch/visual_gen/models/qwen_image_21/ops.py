# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the TensorRT-LLM project
"""Native TRTLLM helper math for Qwen-Image 2.1 VisualGen.

This module contains small stateless tensor/shape recipes from Diffusers'
Qwen-Image 2.1 pipeline plus the FlowMatch Euler scheduler subset required by
the TRTLLM-owned denoise loop.  Keeping these pieces in TRTLLM source lets the
runtime avoid importing the upstream Diffusers pipeline/transformer/attention
components while sharing observable scheduler and latent-packing semantics with
the reference implementation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch


class _SchedulerConfig(dict):
    """Dictionary with attribute access for Diffusers-style scheduler config."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - mirrors normal attr errors.
            raise AttributeError(name) from exc


class QwenImage21FlowMatchEulerScheduler:
    """TRTLLM-owned FlowMatch Euler scheduler used by Qwen-Image 2.1.

    The implementation is the narrow inference subset exercised by
    ``QwenImage21Pipeline`` and matches the public Diffusers
    ``FlowMatchEulerDiscreteScheduler`` equations for ``set_timesteps`` and
    ``step`` under the Qwen-Image 2.1 scheduler configuration
    (dynamic shifting, terminal shift, deterministic sampling).
    """

    config_name = "scheduler_config.json"
    order = 1

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = True,
        base_shift: float | None = 0.5,
        max_shift: float | None = 0.9,
        base_image_seq_len: int = 256,
        max_image_seq_len: int = 8192,
        invert_sigmas: bool = False,
        shift_terminal: float | None = 0.02,
        use_karras_sigmas: bool = False,
        use_exponential_sigmas: bool = False,
        use_beta_sigmas: bool = False,
        time_shift_type: str = "exponential",
        stochastic_sampling: bool = False,
        **kwargs: Any,
    ) -> None:
        del kwargs
        if time_shift_type not in {"exponential", "linear"}:
            raise ValueError("time_shift_type must be 'exponential' or 'linear'.")
        if sum(bool(x) for x in (use_karras_sigmas, use_exponential_sigmas, use_beta_sigmas)) > 1:
            raise ValueError("Only one sigma conversion mode can be enabled.")
        if use_karras_sigmas or use_exponential_sigmas or use_beta_sigmas:
            raise NotImplementedError(
                "QwenImage21FlowMatchEulerScheduler implements the Qwen-Image 2.1 deterministic "
                "configuration; alternate sigma conversions should be added with parity tests before use."
            )
        if stochastic_sampling:
            raise NotImplementedError(
                "Qwen-Image 2.1 scheduler_config.json sets stochastic_sampling=false; stochastic sampling "
                "is intentionally not enabled without dedicated parity coverage."
            )

        self.config = _SchedulerConfig(
            num_train_timesteps=int(num_train_timesteps),
            shift=float(shift),
            use_dynamic_shifting=bool(use_dynamic_shifting),
            base_shift=base_shift,
            max_shift=max_shift,
            base_image_seq_len=int(base_image_seq_len),
            max_image_seq_len=int(max_image_seq_len),
            invert_sigmas=bool(invert_sigmas),
            shift_terminal=shift_terminal,
            use_karras_sigmas=bool(use_karras_sigmas),
            use_exponential_sigmas=bool(use_exponential_sigmas),
            use_beta_sigmas=bool(use_beta_sigmas),
            time_shift_type=time_shift_type,
            stochastic_sampling=bool(stochastic_sampling),
        )
        self.num_inference_steps: Optional[int] = None
        self._step_index: Optional[int] = None
        self._begin_index: Optional[int] = None
        self._shift = float(shift)
        self._init_training_schedule()

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str, subfolder: str = "scheduler") -> "QwenImage21FlowMatchEulerScheduler":
        """Load ``scheduler_config.json`` from a Diffusers-format checkpoint."""

        path = Path(checkpoint_dir)
        if subfolder:
            path = path / subfolder
        candidates = (path / cls.config_name, path / "config.json")
        for config_path in candidates:
            if config_path.is_file():
                payload = json.loads(config_path.read_text())
                payload.pop("_class_name", None)
                payload.pop("_diffusers_version", None)
                return cls(**payload)
        raise FileNotFoundError(f"No scheduler_config.json or config.json found under {path}")

    @property
    def shift(self) -> float:
        return self._shift

    @property
    def step_index(self) -> Optional[int]:
        return self._step_index

    @property
    def begin_index(self) -> Optional[int]:
        return self._begin_index

    def __len__(self) -> int:
        return int(self.config.num_train_timesteps)

    def _init_training_schedule(self) -> None:
        timesteps = np.linspace(1, self.config.num_train_timesteps, self.config.num_train_timesteps, dtype=np.float32)[
            ::-1
        ].copy()
        timesteps_t = torch.from_numpy(timesteps).to(dtype=torch.float32)
        sigmas = timesteps_t / self.config.num_train_timesteps
        if not self.config.use_dynamic_shifting:
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        self.timesteps = sigmas * self.config.num_train_timesteps
        self.sigmas = sigmas.to("cpu")
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = int(begin_index)

    def set_shift(self, shift: float) -> None:
        self._shift = float(shift)

    def _sigma_to_t(self, sigma: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return sigma * self.config.num_train_timesteps

    def _time_shift_exponential(self, mu: float, sigma: float, t: np.ndarray) -> np.ndarray:
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def _time_shift_linear(self, mu: float, sigma: float, t: np.ndarray) -> np.ndarray:
        return mu / (mu + (1 / t - 1) ** sigma)

    def time_shift(self, mu: float, sigma: float, t: np.ndarray) -> np.ndarray:
        if self.config.time_shift_type == "exponential":
            return self._time_shift_exponential(mu, sigma, t)
        return self._time_shift_linear(mu, sigma, t)

    def stretch_shift_to_terminal(self, t: np.ndarray) -> np.ndarray:
        one_minus_z = 1 - t
        scale_factor = one_minus_z[-1] / (1 - self.config.shift_terminal)
        return 1 - (one_minus_z / scale_factor)

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        sigmas: list[float] | np.ndarray | None = None,
        mu: float | None = None,
        timesteps: list[float] | np.ndarray | None = None,
    ) -> None:
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError("mu must be passed when use_dynamic_shifting is True.")
        if sigmas is not None and timesteps is not None and len(sigmas) != len(timesteps):
            raise ValueError("sigmas and timesteps should have the same length.")
        if num_inference_steps is not None:
            if sigmas is not None and len(sigmas) != num_inference_steps:
                raise ValueError("sigmas should have the same length as num_inference_steps.")
            if timesteps is not None and len(timesteps) != num_inference_steps:
                raise ValueError("timesteps should have the same length as num_inference_steps.")
        else:
            if sigmas is None and timesteps is None:
                raise ValueError("num_inference_steps, sigmas, or timesteps must be provided.")
            num_inference_steps = len(sigmas) if sigmas is not None else len(timesteps)  # type: ignore[arg-type]

        self.num_inference_steps = int(num_inference_steps)
        if timesteps is not None:
            timesteps = np.array(timesteps).astype(np.float32)
        if sigmas is None:
            if timesteps is None:
                timesteps = np.linspace(
                    self._sigma_to_t(self.sigma_max),
                    self._sigma_to_t(self.sigma_min),
                    self.num_inference_steps,
                    dtype=np.float32,
                )
            sigmas = np.asarray(timesteps, dtype=np.float32) / self.config.num_train_timesteps
        else:
            sigmas = np.asarray(sigmas, dtype=np.float32)

        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(float(mu), 1.0, sigmas)
        else:
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        if self.config.shift_terminal:
            sigmas = self.stretch_shift_to_terminal(sigmas)

        sigmas_t = torch.from_numpy(np.asarray(sigmas, dtype=np.float32)).to(dtype=torch.float32, device=device)
        timesteps_t = sigmas_t * self.config.num_train_timesteps
        if self.config.invert_sigmas:
            sigmas_t = 1.0 - sigmas_t
            timesteps_t = sigmas_t * self.config.num_train_timesteps
            sigmas_t = torch.cat([sigmas_t, torch.ones(1, device=sigmas_t.device)])
        else:
            sigmas_t = torch.cat([sigmas_t, torch.zeros(1, device=sigmas_t.device)])

        self.timesteps = timesteps_t
        self.sigmas = sigmas_t
        self._step_index = None
        self._begin_index = None

    def index_for_timestep(
        self,
        timestep: Union[float, torch.Tensor],
        schedule_timesteps: Optional[torch.Tensor] = None,
    ) -> int:
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(schedule_timesteps.device)
        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def _init_step_index(self, timestep: Union[float, torch.Tensor]) -> None:
        if self.begin_index is None:
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> tuple[torch.Tensor] | dict[str, torch.Tensor]:
        del kwargs
        if isinstance(timestep, (int, torch.IntTensor, torch.LongTensor)):
            raise ValueError("Pass an actual scheduler timestep, not an integer loop index.")
        if self.step_index is None:
            self._init_step_index(timestep)

        latents_dtype = sample.dtype
        sample = sample.to(torch.float32)
        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
        dt = sigma_next - sigma
        prev_sample = (sample + dt * model_output).to(latents_dtype)
        self._step_index += 1
        if return_dict:
            return {"prev_sample": prev_sample}
        return (prev_sample,)


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Return the FlowMatch dynamic shift used by Qwen-Image 2.1.

    Source parity target:
    ``diffusers.pipelines.qwenimage21.pipeline_qwenimage21.calculate_shift``.
    """

    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def calculate_dimensions(target_area: int | float, ratio: int | float) -> tuple[int, int, None]:
    """Round a target-area/aspect-ratio pair to Qwen's 32-pixel grid."""

    width = math.sqrt(float(target_area) * float(ratio))
    height = width / float(ratio)
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return int(width), int(height), None


def latent_grid_size(pixel_size: int, vae_scale_factor: int = 16) -> int:
    """Return Qwen-Image 2.1's even latent grid dimension for one image side."""

    return 2 * (int(pixel_size) // (int(vae_scale_factor) * 2))


def pack_latents(latents: Any, batch_size: int, num_channels_latents: int, height: int, width: int) -> Any:
    """Flatten Qwen-Image 2.1 5-D latents into transformer token order.

    Qwen-Image 2.1 consumes latents unpatched, unlike earlier Qwen-Image
    variants that group 2x2 patches before the transformer.  ``latents`` is a
    torch-like tensor with shape ``[B, 1, C, H, W]`` and returns ``[B, H*W, C]``.
    """

    return latents.view(batch_size, num_channels_latents, height * width).transpose(1, 2)


def unpack_latents(latents: Any, height: int, width: int, vae_scale_factor: int = 16) -> Any:
    """Undo :func:`pack_latents` for VAE decode."""

    batch_size, _, channels = latents.shape
    latent_height = latent_grid_size(height, vae_scale_factor)
    latent_width = latent_grid_size(width, vae_scale_factor)
    return latents.transpose(1, 2).reshape(batch_size, channels, 1, latent_height, latent_width)


def append_target_image_mask_slots(mask: Any, target_latent_token_count: int) -> Any:
    """Append target-image mask slots for Qwen-Image 2.1 block-causal attention.

    The 2.1 transformer expands every VLM image-pad slot to four latent tokens.
    Therefore the pipeline appends one image-pad slot per 2x2 target-latent
    group before the transformer repeats slots into actual latent-token masks.
    """

    return mask.new_ones(mask.shape[0], int(target_latent_token_count) // 4)


def append_target_slots(mask: Any, target_latent_token_count: int) -> Any:
    """Return ``mask`` with Qwen-Image 2.1 target-image slots appended."""

    slots = append_target_image_mask_slots(mask, target_latent_token_count)
    return torch.cat([mask, slots], dim=1)
