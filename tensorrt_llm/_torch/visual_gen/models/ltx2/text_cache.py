# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Step-invariant preprocessor outputs for the LTX2 denoise loop.

Computed once before the denoise loop and passed into each ``LTXModel.forward()``
call.  Keeps ``forward()`` free of data-dependent branches, making it safe
for CUDA graph capture/replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TextCache:
    """Pre-computed text-derived tensors that are constant across denoise steps.

    Attributes:
        video_context: Projected text embedding for video cross-attention.
        video_mask: Attention mask for video text cross-attention.
        video_pe: RoPE (cos, sin) for video.
        audio_context: Projected text embedding for audio cross-attention.
        audio_mask: Attention mask for audio text cross-attention.
        audio_pe: RoPE (cos, sin) for audio.
        video_cross_pe: Cross-modal RoPE for video (audio-video model only).
        audio_cross_pe: Cross-modal RoPE for audio (audio-video model only).
        video_kv: Per-layer pre-projected text K/V for video cross-attention.
        audio_kv: Per-layer pre-projected text K/V for audio cross-attention.
    """

    video_context: Optional[torch.Tensor] = None
    video_mask: Optional[torch.Tensor] = None
    video_pe: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    video_cross_pe: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    video_kv: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None
    audio_context: Optional[torch.Tensor] = None
    audio_mask: Optional[torch.Tensor] = None
    audio_pe: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    audio_cross_pe: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    audio_kv: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None
