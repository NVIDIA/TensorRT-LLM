"""Batch-splitting adapter for the transformer.
Wraps an ``X0Model`` (including one backed by ``StreamingModel``) and splits batched inputs
into smaller chunks before forwarding, then concatenates the results. This
controls peak activation memory at the cost of more forward passes.
The adapter is transparent — it has the same ``forward`` signature as
``X0Model`` and proxies attribute access to the wrapped model.
Example
-------
>>> from ltx_core.batch_split import BatchSplitAdapter
>>> adapter = BatchSplitAdapter(model, max_batch_size=1)
>>> # Receives B=4, runs 4xB=1 internally, returns B=4
>>> denoised_video, denoised_audio = adapter(video=v_b4, audio=a_b4, perturbations=ptb)
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from tensorrt_llm._torch.visual_gen.models.ltx_core.guidance.perturbations import BatchedPerturbationConfig
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.transformer.modality import Modality


def _split_perturbations(config: BatchedPerturbationConfig, sizes: list[int]) -> list[BatchedPerturbationConfig]:
    """Split a ``BatchedPerturbationConfig`` along the batch dimension."""
    chunks = []
    offset = 0
    for size in sizes:
        chunks.append(config.batch_slice(offset, offset + size))
        offset += size
    return chunks


def _merge_tensors(tensors: list[torch.Tensor | None]) -> torch.Tensor | None:
    """Concatenate tensors along batch dim, or return None if all are None."""
    non_none = [t for t in tensors if t is not None]
    if not non_none:
        return None
    return torch.cat(non_none, dim=0)


class BatchSplitAdapter(nn.Module):
    """Wraps a model and splits batched forward calls into smaller chunks.
    Has the same ``forward`` signature as ``X0Model``:
    ``(video, audio, perturbations) -> (denoised_video, denoised_audio)``.
    Args:
        model: The model to wrap (for example, ``X0Model``).
        max_batch_size: Maximum batch size per forward pass. Input batches
            larger than this are split into sequential chunks.
    """

    def __init__(self, model: nn.Module, max_batch_size: int) -> None:
        if max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
        super().__init__()
        self._model = model
        self._max_batch_size = max_batch_size

    @property
    def num_blocks(self) -> int:
        return self._model.num_blocks

    def _get_chunk_sizes(self, batch_size: int) -> list[int]:
        full, remainder = divmod(batch_size, self._max_batch_size)
        sizes = [self._max_batch_size] * full
        if remainder:
            sizes.append(remainder)
        return sizes

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        batch_size = (video or audio).latent.shape[0]

        if batch_size <= self._max_batch_size:
            return self._model(video=video, audio=audio, perturbations=perturbations)

        sizes = self._get_chunk_sizes(batch_size)
        n = len(sizes)

        v_chunks = video.split(sizes) if video is not None else [None] * n
        a_chunks = audio.split(sizes) if audio is not None else [None] * n
        # A None config means "perturb nothing"; forward it per chunk so the inner model
        # builds a per-chunk all-keep mask (splitting None has nothing to slice).
        p_chunks = _split_perturbations(perturbations, sizes) if perturbations is not None else [None] * n

        chunk_results = [
            self._model(video=vc, audio=ac, perturbations=pc)
            for vc, ac, pc in zip(v_chunks, a_chunks, p_chunks, strict=True)
        ]

        results_v, results_a = zip(*chunk_results, strict=True)
        return _merge_tensors(list(results_v)), _merge_tensors(list(results_a))

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Proxy attribute access to the wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)
