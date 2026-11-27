# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Final-latent numerical fidelity, not a perceptual quality metric."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors.torch import load_file, save_file

if TYPE_CHECKING:
    from tensorrt_llm.visual_gen.output import VisualGenOutput


def save_latents(output: VisualGenOutput, path: Path) -> None:
    """Save an output requested with ``extra_params.output_type=latent``.

    Explicit latent keys distinguish these artifacts from decoded media saved
    with VisualGenOutput.save(). The caller must request latent generation.
    """
    tensors = {
        f"{name}_latent": tensor.detach().cpu().contiguous()
        for name in ("video", "audio")
        if (tensor := getattr(output, name)) is not None
    }
    _validate_latents(tensors, path)
    save_file(tensors, str(path))


def _validate_latents(tensors: dict[str, torch.Tensor], path: Path) -> None:
    if "video_latent" not in tensors or set(tensors) - {"video_latent", "audio_latent"}:
        raise ValueError(f"{path}: expected video_latent and optional audio_latent tensors")
    for name, tensor in tensors.items():
        if not tensor.is_floating_point() or tensor.numel() == 0:
            raise ValueError(f"{path}: {name} must be a non-empty floating-point tensor")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{path}: {name} contains NaN or infinity")


def score_latent_files(candidate: Path, reference: Path) -> dict[str, float]:
    """Compute sqrt(mean((candidate-reference)**2)) separately per modality.

    Use CPU float64 arithmetic to avoid reduced-precision subtraction/overflow.
    Require identical shapes and modalities; never broadcast, resize, or truncate.
    """
    actual = load_file(str(candidate), device="cpu")
    expected = load_file(str(reference), device="cpu")
    _validate_latents(actual, candidate)
    _validate_latents(expected, reference)
    if actual.keys() != expected.keys():
        raise ValueError(f"Latent modalities differ: {candidate} versus {reference}")
    scores = {}
    for name, baseline in expected.items():
        tensor = actual[name]
        if tensor.shape != baseline.shape:
            raise ValueError(
                f"{name} shape mismatch: {candidate} {tuple(tensor.shape)} versus "
                f"{reference} {tuple(baseline.shape)}"
            )
        scores[f"{name}_rmse"] = (tensor.double() - baseline.double()).square().mean().sqrt().item()
    return scores


class LatentRMSEEvaluator:
    def __init__(self, references: dict[str, Path]) -> None:
        self.references = references

    def evaluate(self, prompt_id: str, prompt: str, video: Path) -> dict[str, float]:
        return score_latent_files(video, self.references[prompt_id])

    def close(self) -> None:
        pass
