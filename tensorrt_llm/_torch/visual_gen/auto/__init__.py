# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VisGen-Auto: torch.export + FX-rewrite path for diffusers transformers.

Coexists with the handwritten visual-gen pipelines as a fallback. Dispatch
between the two is in `tensorrt_llm._torch.visual_gen.pipeline_registry.AutoPipeline`,
keyed on `DiffusionModelConfig.pipeline_mode` (`auto` | `fallback` | `strict`).
Public surface: `AutoTransformerPipeline`, `VisGenFamilyAdapter`, `RewritePolicy`.
"""

from .adapter import VisGenFamilyAdapter
from .pipeline import AutoTransformerPipeline
from .policy import RewritePolicy

__all__ = [
    "AutoTransformerPipeline",
    "RewritePolicy",
    "VisGenFamilyAdapter",
]
