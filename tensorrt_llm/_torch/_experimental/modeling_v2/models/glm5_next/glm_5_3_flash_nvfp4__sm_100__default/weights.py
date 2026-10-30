# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight loading: GLM-5.3-Flash (NVFP4) / sm_100 / default.

The checkpoint is the vision-language wrapper's. ``model.language_model.*`` holds the language model (its
``layers.45.*`` are the MTP layer), ``lm_head.*`` the output head and ``model.visual.*`` the vision tower. The
registered ``Glm5NextHfWeightMapper`` audits every key against the target's destinations: text and MTP keys
resolve to an owner (a decoder layer, the embedding, the final norm or the head), the vision namespace is a
predicted non-load, and any key that resolves nowhere fails the load, naming it. Each owner is then filled through
its modules' own loaders, which keep this rank's shard and the NVFP4 block-scale layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper


def load(model: Any, weights: Any, weight_mapper: Optional[BaseWeightMapper] = None) -> None:
    """Load the language model's and the MTP layer's weights into ``model``, the target."""
    # The owner-by-owner loader is the copied causal LM's; imported here because modeling.py imports this module.
    from .modeling import Glm5NextForCausalLM

    Glm5NextForCausalLM.load_weights(model, weights, weight_mapper)
