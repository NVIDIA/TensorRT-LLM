# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Adapt current upstream perturbation masks to TRTLLM's cached graph interface."""

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.guidance.perturbations import (
    BatchedPerturbationConfig as UpstreamBatchedPerturbationConfig,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.guidance.perturbations import (
    Perturbation as Perturbation,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.guidance.perturbations import (
    PerturbationConfig as PerturbationConfig,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.guidance.perturbations import (
    PerturbationType as PerturbationType,
)


class BatchedPerturbationConfig:
    def __init__(self, perturbations):
        self.perturbations = perturbations
        self._blocks = {}
        self._mask_cache = {}

    def _for_block(self, block):
        # The existing pipeline does not pass model depth. Build only the requested
        # block using current upstream's mask constructor; cache for graph replay.
        if block not in self._blocks:
            projected = [
                PerturbationConfig(
                    [
                        Perturbation(p.type, None)
                        for p in (pc.perturbations or [])
                        if p.is_perturbed(p.type, block)
                    ]
                )
                for pc in self.perturbations
            ]
            self._blocks[block] = UpstreamBatchedPerturbationConfig(projected, 1)
        return self._blocks[block]

    def all_in_batch(self, perturbation_type, block):
        return self._for_block(block).all_in_batch(perturbation_type, 0)

    def any_in_batch(self, perturbation_type, block):
        return self._for_block(block).any_in_batch(perturbation_type, 0)

    def mask(self, perturbation_type, block, device, dtype):
        key = (perturbation_type, block, device, dtype)
        if key not in self._mask_cache:
            # The graph runner warms this same configuration before capture.
            self._mask_cache[key] = (
                self._for_block(block)
                .mask(perturbation_type, 0)
                .flatten()
                .to(device=device, dtype=dtype)
            )
        return self._mask_cache[key]

    def mask_like(self, perturbation_type, block, values):
        return self.mask(perturbation_type, block, values.device, values.dtype).reshape(
            -1, *([1] * (values.ndim - 1))
        )


def build_stg_perturbation_config(stg_blocks):
    return PerturbationConfig(
        [
            Perturbation(PerturbationType.SKIP_VIDEO_SELF_ATTN, stg_blocks),
            Perturbation(PerturbationType.SKIP_AUDIO_SELF_ATTN, stg_blocks),
        ]
    )
