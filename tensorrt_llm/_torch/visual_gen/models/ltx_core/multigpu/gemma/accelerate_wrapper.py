"""Accelerate-based Gemma text encoder wrapper for multi-GPU inference.
One rank (``src_rank``) holds the real ``LTXGemmaTextEncoder`` loaded with
``device_map="auto"``; other ranks hold a lightweight stub. Every public
method runs on the source rank and broadcasts results to all ranks via the
provided NCCL process group.
The ``broadcast_group`` should cover **all ranks that need the
embeddings** (typically the transformer group or world group).
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.visual_gen.models.ltx_core.multigpu.gemma.broadcast_wrapper import BroadcastGemmaWrapper


class AccelerateGemmaWrapper(BroadcastGemmaWrapper):
    """Source-rank encode + NCCL broadcast around a sharded ``LTXGemmaTextEncoder``."""

    def encode(self, prompts: list[str]) -> list[tuple[tuple[torch.Tensor, ...], torch.Tensor]]:
        """Fuse all prompts into one Gemma call on the source rank, broadcast each output."""
        local_outputs = self._encoder.encode(prompts) if self._rank == self._src_rank else [(None, None)] * len(prompts)
        return [self._broadcast_encoder_output(hs, mask, self._src_rank) for hs, mask in local_outputs]
