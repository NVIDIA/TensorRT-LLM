# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

from typing import Protocol, Tuple

import torch

from .types import AudioLatentShape, VideoLatentShape


class Patchifier(Protocol):
    """Protocol for patchify/unpatchify latent tensors."""

    def patchify(self, latents: torch.Tensor) -> torch.Tensor: ...
    def unpatchify(
        self, latents: torch.Tensor, output_shape: AudioLatentShape | VideoLatentShape
    ) -> torch.Tensor: ...

    @property
    def patch_size(self) -> Tuple[int, int, int]: ...

    def get_patch_grid_bounds(
        self,
        output_shape: AudioLatentShape | VideoLatentShape,
        device: torch.device | None = None,
    ) -> torch.Tensor: ...


class SchedulerProtocol(Protocol):
    """Protocol for sigma schedule."""

    def execute(self, steps: int, **kwargs) -> torch.FloatTensor: ...


class DiffusionStepProtocol(Protocol):
    """Protocol for one diffusion step (e.g. Euler)."""

    def step(
        self,
        sample: torch.Tensor,
        denoised_sample: torch.Tensor,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor: ...
