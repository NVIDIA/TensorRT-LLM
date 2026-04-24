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
from .transformer_qwen_image import QwenImageTransformer2DModel

__all__ = [
    "QwenImagePipeline",
    "QwenImageTransformer2DModel",
]
