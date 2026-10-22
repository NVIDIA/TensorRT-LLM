# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the TensorRT-LLM project
"""Qwen-Image 2.1 latent helpers and Diffusers scheduler compatibility."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from diffusers import FlowMatchEulerDiscreteScheduler


class QwenImage21FlowMatchEulerScheduler(FlowMatchEulerDiscreteScheduler):
    """Diffusers scheduler with Qwen defaults and local checkpoint loading."""

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
                "QwenImage21FlowMatchEulerScheduler supports the Qwen-Image 2.1 deterministic "
                "configuration; alternate sigma conversions should be added with parity tests before use."
            )
        if stochastic_sampling:
            raise NotImplementedError(
                "Qwen-Image 2.1 scheduler_config.json sets stochastic_sampling=false; stochastic sampling "
                "is intentionally not enabled without dedicated parity coverage."
            )

        super().__init__(
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

    @classmethod
    def from_pretrained(
        cls, checkpoint_dir: str, subfolder: str = "scheduler"
    ) -> "QwenImage21FlowMatchEulerScheduler":
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


def pack_latents(
    latents: Any, batch_size: int, num_channels_latents: int, height: int, width: int
) -> Any:
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
