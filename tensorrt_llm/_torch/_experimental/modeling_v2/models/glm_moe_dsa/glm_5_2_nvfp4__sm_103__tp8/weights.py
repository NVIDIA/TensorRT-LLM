# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The weights this target expects: the GLM-5.2 NVFP4 checkpoint as the built-in model loads it.

The target's module tree is the built-in model's, so the built-in loader maps
every checkpoint tensor, including the bf16 MTP layer and the NVFP4 scales.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tensorrt_llm._torch.models.modeling_deepseekv3 import GlmMoeDsaForCausalLM

if TYPE_CHECKING:
    from .modeling import ModelingV2Glm52Nvfp4Sm103Tp8


def load(model: "ModelingV2Glm52Nvfp4Sm103Tp8", weights: Any) -> None:
    """Load the checkpoint ``weights`` into ``model`` with the built-in loader."""
    GlmMoeDsaForCausalLM.load_weights(model, weights)
