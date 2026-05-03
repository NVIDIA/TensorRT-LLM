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
        video_pe: RoPE (cos, sin) for video self-attention. Full (un-sharded) layout.
        video_pe_local: Sharded contiguous self-attn RoPE for video. Pre-sharded
            once here so neither the per-step ``_shard_pe`` slice nor the
            downstream non-contiguous reshape copy in the fuse-op prologue runs.
        video_cross_pe: Cross-modal RoPE for video (audio-video model only).
            Full layout, used for K-rope after all-gather.
        video_cross_pe_local: Sharded contiguous Cross-modal RoPE for video.
        audio_context: Projected text embedding for audio cross-attention.
        audio_mask: Attention mask for audio text cross-attention.
        audio_pe: RoPE (cos, sin) for audio self-attention. Full layout.
        audio_pe_local: Sharded contiguous self-attn RoPE for audio.
        audio_cross_pe: Cross-modal RoPE for audio. Full layout.
        audio_cross_pe_local: Sharded contiguous Cross-modal RoPE for audio.
        video_kv: Per-layer pre-projected text K/V for video cross-attention.
        audio_kv: Per-layer pre-projected text K/V for audio cross-attention.
    """

    video_context: Optional[torch.Tensor] = None
    video_mask: Optional[torch.Tensor] = None
    video_pe: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    video_pe_local: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    video_cross_pe: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    video_cross_pe_local: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    video_kv: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None
    audio_context: Optional[torch.Tensor] = None
    audio_mask: Optional[torch.Tensor] = None
    audio_pe: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    audio_pe_local: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    audio_cross_pe: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    audio_cross_pe_local: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    audio_kv: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None
