# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLUX.3 text encoder helpers: Qwen3-VL trimming and prompt padding windows."""

import math

from torch import nn

# Hidden states stacked along channels; 8 x 2560 is the transformer's joint_attention_dim.
HIDDEN_STATE_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
# Prompts are right-padded to a multiple of TEXT_PAD_MULTIPLE tokens, at most TEXT_MAX_TOKENS.
# The DiT attends to the padding, so the padded length is part of the conditioning.
TEXT_PAD_MULTIPLE = 80
TEXT_MAX_TOKENS = 8192
# Padding token position id; matches the transformers version FLUX.3 was trained with.
PADDING_POSITION_ID = 1


def padded_text_length(
    num_tokens: int, pad_multiple: int = TEXT_PAD_MULTIPLE, max_tokens: int = TEXT_MAX_TOKENS
) -> int:
    """Window of a prompt with ``num_tokens`` tokens: rounded up to ``pad_multiple``, capped."""
    return min(math.ceil(num_tokens / pad_multiple) * pad_multiple, max_tokens)


def trim_for_text(model: nn.Module, last_layer: int) -> None:
    """Keep a Qwen3-VL model's first ``last_layer`` decoder layers and drop its vision tower.

    A cut model's final norm becomes Identity, so hidden_states[last_layer] stays un-normed.
    """
    language_model = model.model.language_model
    num_layers = len(language_model.layers)
    if not 1 <= last_layer <= num_layers:
        raise ValueError(f"FLUX.3 text encoder layer {last_layer} is not in 1..{num_layers}")
    if last_layer < num_layers:
        language_model.layers = language_model.layers[:last_layer]
        language_model.norm = nn.Identity()
    # FLUX.3 encodes text only; the vision tower is never called.
    model.model.visual = None
