# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

from dataclasses import dataclass

import torch


@dataclass
class LTX23TextConditioning:
    """Connector-processed text conditioning without static per-block K/V."""

    video_context: torch.Tensor | None = None
    video_mask: torch.Tensor | None = None
    video_pe: tuple[torch.Tensor, torch.Tensor] | None = None
    video_cross_pe: tuple[torch.Tensor, torch.Tensor] | None = None

    audio_context: torch.Tensor | None = None
    audio_mask: torch.Tensor | None = None
    audio_pe: tuple[torch.Tensor, torch.Tensor] | None = None
    audio_cross_pe: tuple[torch.Tensor, torch.Tensor] | None = None
