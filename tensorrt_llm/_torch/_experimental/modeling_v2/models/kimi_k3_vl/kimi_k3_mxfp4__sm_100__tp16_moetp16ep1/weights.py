# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# >>> route B: this target's expert slice
"""Weight loading: Kimi K3 (MXFP4) / sm_100 / tp16_moetp16ep1.

The checkpoint is the vision-language wrapper's. `language_model.*` holds the language model; `vision_tower.*` and
`mm_projector.*` hold the vision tower and its projector. This target is text only, so the keys split three ways:

* `language_model.*` goes to the built-in text model's loader with the prefix stripped. That loader streams the
  routed experts one at a time, keeps this rank's slice (a sixteenth of the width of every expert under moe_tp 16 x
  moe_ep 1: 192 of 3072, zero-padded to 256 for the MoE kernels' tiles), and checks its own key coverage.
* The vision tower and the projector are a predicted non-load: listed here, never read.
* Any other key fails the load, naming it, rather than being dropped.
"""
# <<< route B

from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict
from tensorrt_llm._torch.models.modeling_kimi_linear import KimiLinearForCausalLM
from tensorrt_llm._torch.models.modeling_utils import filter_weights

_LANG_PREFIX = "language_model."

# The vision tower's and the projector's key families: in the checkpoint, not loaded by this text-only target.
PREDICTED_NON_LOAD = ("vision_tower.", "mm_projector.")


def load(model, weights) -> None:
    """Load the language model's weights into `model`, the target shell, and check every other key is predicted."""
    unknown = sorted(
        k
        for k in weights.keys()
        if not k.startswith(_LANG_PREFIX) and not k.startswith(PREDICTED_NON_LOAD)
    )
    assert not unknown, (
        f"{len(unknown)} checkpoint key(s) are neither language-model weights nor a predicted non-load, "
        f"first {unknown[:5]}"
    )
    lm_weights = ConsumableWeightsDict(filter_weights(_LANG_PREFIX[:-1], weights))
    assert len(lm_weights), f"the checkpoint has no {_LANG_PREFIX}* keys"
    checkpoint_dir = getattr(weights, "checkpoint_dir", None)
    if checkpoint_dir is not None:
        lm_weights.checkpoint_dir = checkpoint_dir
    lm_weights.checkpoint_prefix = _LANG_PREFIX
    KimiLinearForCausalLM.load_weights(model, lm_weights)
