# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2
"""LTX-2 core components ported from the official LTX-2 repository.

Major modifications:
- attention.py: Replaced nn.Linear with TRT-LLM Linear; removed standalone
  attention computation (delegated to visual_gen attention backends).
- transformer_args.py: Adapted mask preparation for TRT-LLM attention backends.
- connector.py: Replaced nn.Linear with TRT-LLM Linear for quantization support.
"""

from .adaln import AdaLayerNormSingle
from .attention import Attention, FeedForward, GELUApprox
from .connector import (
    Embeddings1DConnector,
    Embeddings1DConnectorConfigurator,
    GemmaFeaturesExtractorProjLinear,
)
from .diffusion_steps import EulerDiffusionStep
from .modality import Modality
from .normalization import NormType, PixelNorm, build_normalization_layer
from .patchifier import AudioPatchifier, VideoLatentPatchifier, get_pixel_coords
from .protocols import DiffusionStepProtocol, Patchifier, SchedulerProtocol
from .rope import LTXRopeType, apply_rotary_emb, precompute_freqs_cis
from .scheduler_adapter import NativeSchedulerAdapter
from .schedulers import LTX2Scheduler
from .text_projection import PixArtAlphaTextProjection
from .timestep_embedding import (
    PixArtAlphaCombinedTimestepSizeEmbeddings,
    TimestepEmbedding,
    Timesteps,
)
from .transformer_args import (
    MultiModalTransformerArgsPreprocessor,
    TransformerArgs,
    TransformerArgsPreprocessor,
)
from .types import (
    VIDEO_SCALE_FACTORS,
    AudioLatentShape,
    SpatioTemporalScaleFactors,
    VideoLatentShape,
    VideoPixelShape,
)
from .utils_ltx2 import rms_norm, to_velocity

__all__ = [
    "AdaLayerNormSingle",
    "Attention",
    "AudioLatentShape",
    "AudioPatchifier",
    "DiffusionStepProtocol",
    "Embeddings1DConnector",
    "Embeddings1DConnectorConfigurator",
    "EulerDiffusionStep",
    "FeedForward",
    "GELUApprox",
    "GemmaFeaturesExtractorProjLinear",
    "LTX2Scheduler",
    "LTXRopeType",
    "Modality",
    "MultiModalTransformerArgsPreprocessor",
    "NativeSchedulerAdapter",
    "NormType",
    "Patchifier",
    "PixArtAlphaCombinedTimestepSizeEmbeddings",
    "PixArtAlphaTextProjection",
    "PixelNorm",
    "SchedulerProtocol",
    "SpatioTemporalScaleFactors",
    "TimestepEmbedding",
    "Timesteps",
    "TransformerArgs",
    "TransformerArgsPreprocessor",
    "VIDEO_SCALE_FACTORS",
    "VideoLatentPatchifier",
    "VideoLatentShape",
    "VideoPixelShape",
    "apply_rotary_emb",
    "build_normalization_layer",
    "get_pixel_coords",
    "precompute_freqs_cis",
    "rms_norm",
    "to_velocity",
]
