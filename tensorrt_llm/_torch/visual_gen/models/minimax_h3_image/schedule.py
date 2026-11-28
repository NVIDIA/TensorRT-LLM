# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The fixed PDD schedule used by the released MiniMax-H3-Image checkpoint."""

import torch

SAMPLE_SIGMAS = (
    1.0,
    0.9675236940383911,
    0.8875485062599182,
    0.75,
    0.5327739119529724,
    0.23200887441635132,
    0.0,
)
NUM_PDD_STEPS = len(SAMPLE_SIGMAS) - 1


class ImagePDDSchedule:
    def __init__(self, device: torch.device) -> None:
        # The released scheduler rounds through the data-ward time in FP32 and
        # back. Directly casting the sigma knots changes the penultimate sigma.
        times = torch.tensor([1 - s for s in SAMPLE_SIGMAS], dtype=torch.float32)
        self.sigmas = (1 - times).to(device)
        self.timesteps = 1 - self.sigmas[:-1]

    def step(self, index: int, latents: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        if not 0 <= index < NUM_PDD_STEPS:
            raise ValueError(f"PDD head index must be in [0, {NUM_PDD_STEPS}), got {index}")
        if latents.shape != velocity.shape:
            raise ValueError("One velocity row is required for every generated latent row.")
        delta = self.sigmas[index] - self.sigmas[index + 1]
        return latents.float() + delta * velocity.float()
