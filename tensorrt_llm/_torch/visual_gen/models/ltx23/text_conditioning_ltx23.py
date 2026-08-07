# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2
"""LTX-2.3 text conditioning cache.

Deliberately distinct from LTX-2's ``TextCache``: it caches only the
**sigma-independent** connector outputs (context, mask, positional embeddings).

It intentionally has NO ``video_kv`` / ``audio_kv`` fields. LTX-2 pre-projects a
static per-block text K/V once (because its text cross-attention K/V never
changes across the denoise loop). LTX-2.3's text K/V is modulated by
``prompt_scale_shift_table`` using a sigma-derived ``prompt_timestep`` and is
therefore step-varying, so it must be (re)projected inside each denoise step.
Omitting the K/V fields here makes it structurally impossible to accidentally
reuse a stale LTX-2-style static K/V.
"""

from dataclasses import dataclass

import torch


@dataclass
class LTX23TextConditioning:
    """Step-invariant, connector-processed text conditioning."""

    video_context: torch.Tensor | None = None
    video_mask: torch.Tensor | None = None
    video_pe: tuple[torch.Tensor, torch.Tensor] | None = None
    video_cross_pe: tuple[torch.Tensor, torch.Tensor] | None = None

    audio_context: torch.Tensor | None = None
    audio_mask: torch.Tensor | None = None
    audio_pe: tuple[torch.Tensor, torch.Tensor] | None = None
    audio_cross_pe: tuple[torch.Tensor, torch.Tensor] | None = None
