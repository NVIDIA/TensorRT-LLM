# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""LTX2 text context cache — caches constant text-derived computations across denoise steps.

Text context (prompt embeddings) is constant throughout the denoising loop.
This module caches two levels of derived computation:

1. **Preprocessor outputs** (per modality): ``caption_projection(context)``,
   attention mask, RoPE positional embeddings, and cross-PE.
2. **Per-block KV projections** (per layer per modality): ``to_k(context)``,
   ``to_v(context)``, and ``norm_k(k)`` for text cross-attention.

Supports 2 CFG slots (conditional + unconditional) so that single-GPU CFG
does not pollute the cache.

Lifecycle:
- Created once by the pipeline (survives across requests).
- ``invalidate()`` marks all slots dirty before each denoising loop.
  Buffers are retained for reuse via ``copy_()``.
- ``fill_kv()`` is called from ``LTXModel.forward()`` to fill per-layer KV.
- Compiled blocks read via ``get_kv()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .transformer_ltx2 import BasicAVTransformerBlock

_COND_SLOT = 0
_UNCOND_SLOT = 1


def _slot(is_unconditional: bool) -> int:
    return _UNCOND_SLOT if is_unconditional else _COND_SLOT


@dataclass
class _PreprocEntry:
    """Cached preprocessor outputs for one modality in one CFG slot."""

    context: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    pe: tuple[torch.Tensor, torch.Tensor] | None = None
    cross_pe: tuple[torch.Tensor, torch.Tensor] | None = None


class LTX2TextContextCache:
    """Caches text-derived computations across denoise steps.

    Args:
        num_layers: Number of transformer blocks.
        max_batch_size: Maximum batch size for pre-allocated KV buffers.
    """

    _NUM_SLOTS = 2  # conditional + unconditional

    def __init__(self, num_layers: int, max_batch_size: int = 1) -> None:
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size

        # Preprocessor cache: per slot, per modality.
        self._preproc_video: list[_PreprocEntry] = [_PreprocEntry() for _ in range(self._NUM_SLOTS)]
        self._preproc_audio: list[_PreprocEntry] = [_PreprocEntry() for _ in range(self._NUM_SLOTS)]

        # Per-block KV cache: per slot, per layer.
        # Buffers are allocated on first fill and retained across invalidate().
        self._video_kv: list[list[tuple[torch.Tensor, torch.Tensor] | None]] = [
            [None] * num_layers for _ in range(self._NUM_SLOTS)
        ]
        self._audio_kv: list[list[tuple[torch.Tensor, torch.Tensor] | None]] = [
            [None] * num_layers for _ in range(self._NUM_SLOTS)
        ]
        # Dirty flag per slot — True means needs refill, False means cache valid.
        self._kv_dirty: list[bool] = [True] * self._NUM_SLOTS

    def invalidate(self) -> None:
        """Mark all slots dirty.  Buffers are retained for ``copy_()`` reuse."""
        for i in range(self._NUM_SLOTS):
            self._preproc_video[i] = _PreprocEntry()
            self._preproc_audio[i] = _PreprocEntry()
            self._kv_dirty[i] = True

    # -- Preprocessor cache ------------------------------------------------

    def get_preproc(self, is_unconditional: bool, modality: str) -> _PreprocEntry:
        """Return the preprocessor cache entry for reading/writing."""
        entries = self._preproc_video if modality == "video" else self._preproc_audio
        return entries[_slot(is_unconditional)]

    # -- KV cache ----------------------------------------------------------

    def fill_kv(
        self,
        is_unconditional: bool,
        blocks: list[BasicAVTransformerBlock],
        video_context: torch.Tensor | None,
        audio_context: torch.Tensor | None,
    ) -> None:
        """Fill per-block KV cache if dirty.  Runs outside torch.compile."""
        s = _slot(is_unconditional)
        if not self._kv_dirty[s]:
            return

        for layer_idx, block in enumerate(blocks):
            if video_context is not None:
                k, v = block.attn2.project_kv(video_context)
                self._store(self._video_kv, s, layer_idx, k, v)
            if audio_context is not None:
                k, v = block.audio_attn2.project_kv(audio_context)
                self._store(self._audio_kv, s, layer_idx, k, v)

        self._kv_dirty[s] = False

    def get_kv(
        self, modality: str, is_unconditional: bool, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached KV for a compiled block."""
        cache = self._video_kv if modality == "video" else self._audio_kv
        return cache[_slot(is_unconditional)][layer_idx]

    # -- Internal ----------------------------------------------------------

    @staticmethod
    def _store(
        cache: list[list[tuple[torch.Tensor, torch.Tensor] | None]],
        slot: int,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Copy into existing buffers, or allocate on first fill."""
        existing = cache[slot][layer_idx]
        if existing is not None:
            # Reuse pre-allocated buffers.
            existing[0].copy_(k)
            existing[1].copy_(v)
        else:
            # First fill — allocate.
            cache[slot][layer_idx] = (k.clone(), v.clone())
