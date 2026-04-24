# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image pipeline (Phase 0 scaffolding).

The transformer backbone is currently a stub that raises NotImplementedError
on ``load_weights`` and ``forward``. The pipeline registers detection with
``AutoPipeline`` and loads the non-transformer sidecars so that a clear error
is emitted from the denoise step rather than a registry miss.

See PHASE1_PLAN.md in the repository root notes for the native port plan.
"""

from .pipeline_qwen_image import QwenImagePipeline
from .transformer_qwen_image import (
    AdaLayerNormContinuous,
    FeedForward,
    GELU,
    QwenEmbedRope,
    QwenImageTransformer2DModel,
    QwenImageTransformerBlock,
    QwenJointAttention,
    QwenTimestepProjEmbeddings,
    RMSNorm,
    Timesteps,
    TimestepEmbedding,
    apply_rotary_emb_qwen,
    get_timestep_embedding,
)

__all__ = [
    "AdaLayerNormContinuous",
    "FeedForward",
    "GELU",
    "QwenEmbedRope",
    "QwenImagePipeline",
    "QwenImageTransformer2DModel",
    "QwenImageTransformerBlock",
    "QwenJointAttention",
    "QwenTimestepProjEmbeddings",
    "RMSNorm",
    "Timesteps",
    "TimestepEmbedding",
    "apply_rotary_emb_qwen",
    "get_timestep_embedding",
]
