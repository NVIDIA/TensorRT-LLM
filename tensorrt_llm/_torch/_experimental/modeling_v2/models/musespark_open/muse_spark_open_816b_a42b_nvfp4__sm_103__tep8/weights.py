# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The weights this target expects: the Muse Spark Open NVFP4 checkpoint as the built-in model loads it.

The target's module tree is the built-in model's, so the built-in loader maps
every checkpoint tensor -- the open-format renames, the stacked NVFP4 experts and
their scales, the head-interleaved query and output-gate projection, and the
perception towers when they are built. The drafter's weights come from the
``speculative_model`` checkpoint through the built-in draft loader, not from here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from tensorrt_llm._torch.models.modeling_musespark_open_omni import (
    MuseSparkOpenOmniForConditionalGeneration,
)

if TYPE_CHECKING:
    from .modeling import ModelingV2MuseSparkOpen816bA42bNvfp4Sm103Tep8


def load(
    model: "ModelingV2MuseSparkOpen816bA42bNvfp4Sm103Tep8",
    weights: Any,
    weight_mapper: Optional[Any] = None,
) -> Dict[str, int]:
    """Load the checkpoint ``weights`` into ``model`` with the built-in loader.

    Returns the built-in loader's key-accounting summary.
    """
    return MuseSparkOpenOmniForConditionalGeneration.load_weights(model, weights, weight_mapper)
