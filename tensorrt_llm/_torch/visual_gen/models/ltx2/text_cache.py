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

    The ``*_pe`` fields hold sharded-local positional embeddings in the form
    the consumer wants:

      - ``fuse_qk_norm_rope=True`` (LTX-2 default): 2D ``[T_local, H*D]``
        contiguous, fed directly to the fused norm+rope kernel.
      - ``fuse_qk_norm_rope=False``: 4D ``[B, T_local, H, D]`` sharded but
        otherwise unchanged, for the naive ``apply_rotary_emb`` path.

    Form is decided at cache-build time (``LTXModel.prepare_text_cache``); no
    per-step reshape, ``.contiguous()``, or shard slicing.

    Attributes:
        video_context: Projected text embedding for video cross-attention.
        video_mask: Attention mask for video text cross-attention.
        video_pe: Sharded-local RoPE (cos, sin) for video self-attn.
        video_cross_pe: Sharded-local RoPE for video AV cross-attn (audio-video model only).
        audio_context: Projected text embedding for audio cross-attention.
        audio_mask: Attention mask for audio text cross-attention.
        audio_pe: Sharded-local RoPE (cos, sin) for audio self-attn.
        audio_cross_pe: Sharded-local RoPE for audio AV cross-attn (audio-video model only).
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
