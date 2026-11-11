"""Batch-parallel Gemma text encoder wrapper for multi-GPU inference.
Each rank holds a full :class:`LTXGemmaTextEncoder` replica resident on its
own GPU.  ``encode`` partitions the prompt list across ranks (each rank
encodes a disjoint slice) and broadcasts every prompt's outputs from its
encoding rank to all other ranks, so all ranks end up with the full list
in the original order.
``enhance_t2v`` / ``enhance_i2v`` (inherited) involve sampling, so they
execute on ``src_rank`` only and the generated string is broadcast.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.multigpu.gemma.broadcast_wrapper import BroadcastGemmaWrapper
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.text_encoders.gemma.encoders.base_encoder import LTXGemmaTextEncoder


def _partition(total: int, world_size: int) -> list[int]:
    """Spread ``total`` items across ``world_size`` ranks; remainder lands on the first ranks."""
    base, rem = divmod(total, world_size)
    return [base + (1 if i < rem else 0) for i in range(world_size)]


class BatchParallelGemmaWrapper(BroadcastGemmaWrapper):
    """Per-rank Gemma replica; ``encode`` parallelises a batch across ranks."""

    _encoder: LTXGemmaTextEncoder  # always resident on every rank, unlike the base's optional encoder

    def __init__(
        self,
        encoder: LTXGemmaTextEncoder,
        broadcast_group: dist.ProcessGroup | None,
        src_rank: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | None = None,
    ) -> None:
        """Wrap a per-rank Gemma replica for batch-parallel encoding.
        Args:
            encoder: Full Gemma replica; required and resident on every rank (unlike
                the base, where it is optional and real only on ``src_rank``).
            broadcast_group: NCCL group spanning the ranks that share the encode work.
            src_rank: Rank within ``broadcast_group`` that runs the inherited sampling
                methods (``enhance_t2v`` / ``enhance_i2v``); ``encode`` uses every rank.
            dtype: Target dtype for output tensors.
            device: Target device for output tensors; defaults to the current CUDA device.
        """
        super().__init__(encoder, broadcast_group, src_rank, dtype, device)
        self._world_size = dist.get_world_size(broadcast_group)

    def encode(self, prompts: list[str]) -> list[tuple[tuple[torch.Tensor, ...], torch.Tensor]]:
        """Partition prompts across ranks, encode in parallel, broadcast per-prompt outputs.
        With B prompts on W ranks, each rank gets ``ceil(B/W)`` or ``floor(B/W)``
        prompts; the typical pos+neg case (B=2, W=2) gives one prompt per rank,
        running both Gemma forwards concurrently on different GPUs.
        """
        n = len(prompts)
        if n == 0:
            return []
        counts = _partition(n, self._world_size)
        start = sum(counts[: self._rank])
        local_prompts = prompts[start : start + counts[self._rank]]
        local_outputs = self._encoder.encode(local_prompts) if local_prompts else []

        all_outputs: list[tuple[tuple[torch.Tensor, ...], torch.Tensor]] = []
        for owner_rank, owner_count in enumerate(counts):
            for slot in range(owner_count):
                hs, mask = local_outputs[slot] if owner_rank == self._rank else (None, None)
                all_outputs.append(self._broadcast_encoder_output(hs, mask, owner_rank))
        return all_outputs
