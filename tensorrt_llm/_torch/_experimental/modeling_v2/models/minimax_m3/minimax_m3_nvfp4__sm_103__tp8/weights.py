# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight loading: MiniMax-M3 NVFP4 / sm_103 / tp8.

The checkpoint is the vision-language wrapper's. ``language_model.*`` holds the
language model; ``vision_tower.*``, ``multi_modal_projector.*`` and
``patch_merge_mlp.*`` hold the vision tower and its projector. This target is
text only, so the keys split three ways:

* ``language_model.*`` goes to the built-in MiniMax-M3 text loader with the
  prefix stripped (its multi-token-prediction head, ``model.mtp.*``, is not
  loaded, as in the built-in model). That loader packs the five-way
  Q/K/V/index-Q/index-K projection, folds the Gemma boundary norms and shards
  every tensor for tp8; the modules it fills are this target's subclasses of
  the built-in ones, under the built-in names.
* The vision tower and the projector are a predicted non-load: listed here,
  never read.
* Any other key fails the load, naming it, rather than being dropped.
"""

from typing import Any, Dict, Optional

from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper
from tensorrt_llm._torch.models.modeling_minimaxm3 import (
    MiniMaxM3ForCausalLM,
    _strip_language_model_prefix,
)

_LANG_PREFIX = "language_model."

# The vision tower's and the projector's key families: in the checkpoint, not loaded by this text-only target.
PREDICTED_NON_LOAD = ("vision_tower.", "multi_modal_projector.", "patch_merge_mlp.")


def load(
    model,
    weights: Dict[str, Any],
    weight_mapper: Optional[BaseWeightMapper] = None,
    params_map: Optional[Dict[str, str]] = None,
    allow_partial_loading: bool = False,
) -> None:
    """Load the language model's weights into ``model``, the target shell, and check every other key is predicted."""
    unknown = sorted(
        k
        for k in weights.keys()
        if not k.startswith(_LANG_PREFIX) and not k.startswith(PREDICTED_NON_LOAD)
    )
    assert not unknown, (
        f"{len(unknown)} checkpoint key(s) are neither language-model weights nor a predicted non-load, "
        f"first {unknown[:5]}"
    )
    text_weights, _ = _strip_language_model_prefix(
        {k: v for k, v in weights.items() if k.startswith(_LANG_PREFIX)}
    )
    assert text_weights, f"the checkpoint has no {_LANG_PREFIX}* keys"
    MiniMaxM3ForCausalLM.load_weights(
        model,
        text_weights,
        weight_mapper=weight_mapper,
        params_map=params_map,
        allow_partial_loading=allow_partial_loading,
    )
