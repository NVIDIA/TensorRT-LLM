# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# DSpark draft I/O logic is ported from DeepSeek's DeepSeek-V4-Pro-DSpark
# reference (`inference/model.py`, DSparkBlock.forward_embed / forward_head).
"""DSpark draft I/O: block input and proposal stages.

This module holds the *framework-agnostic* (pure-torch) input/output stages of
the DSpark draft block, separated from the heavy V4 backbone (MLA + MoE + mHC) so
they can be unit-tested in isolation:

  - ``build_draft_input_ids``: ``[bonus_token, noise, noise, ...]`` block input.
  - ``dspark_propose``: given the per-position backbone ``base_logits`` and the
    Markov / confidence heads, run the autoregressive Markov refinement to sample
    the block tokens and score them with the confidence head.

The backbone (3 V4 blocks producing ``block_hidden``) lives in the model module;
this file is the part fully specified by the reference and validated against it.
"""

from typing import Optional

import torch
from torch import nn


def build_draft_input_ids(
    bonus_token_ids: torch.Tensor, *, block_size: int, noise_token_id: int
) -> torch.Tensor:
    """``[batch] -> [batch, block_size]`` = ``[bonus, noise, noise, ...]``.

    The first position is the verified bonus token (the target's last accepted
    token); the rest are the DSpark noise/mask token (id 128799 for V4-Pro).
    """
    batch = bonus_token_ids.shape[0]
    out = bonus_token_ids.new_full((batch, block_size), int(noise_token_id))
    out[:, 0] = bonus_token_ids
    return out


def dspark_propose(
    base_logits: torch.Tensor,
    *,
    bonus_token_ids: torch.Tensor,
    block_hidden: torch.Tensor,
    markov_head: Optional[nn.Module],
    confidence_head: Optional[nn.Module],
    block_size: int,
    temperature: float = 0.0,
    return_confidence: bool = False,
    return_logits: bool = False,
) -> tuple:
    """Produce DSpark draft tokens for one block (functional-first, static length).

    Always proposes the *full* block: the confidence head does not shorten the
    draft (the block is produced by a single parallel backbone pass, so there is
    nothing to save there). It scores the block so the verification scheduler can
    decide how many of those tokens are worth sending to the target.

    Args:
        base_logits: ``[batch, block_size, vocab]`` from the backbone + lm_head.
        bonus_token_ids: ``[batch]`` the token preceding the first draft position.
        block_hidden: ``[batch, block_size, hidden]`` backbone hidden (feeds the
            confidence head, and the RNN-head variant).
        markov_head / confidence_head: the validated DSpark heads (may be None).
        return_confidence: compute the per-position confidence logits. This is a
            run-constant flag (read once from the decoding config), never a
            per-step decision, so it cannot make the captured graph diverge.
    Returns:
        draft_tokens: ``[batch, block_size]`` sampled tokens (full block; callers
            keep the tensor fixed-width for CUDA-graph safety).
        confidence: ``[batch, block_size]`` fp32 *raw* confidence logits, or None
            when disabled / no head. Calibrate with ``confidence_head.apply_sts``
            before taking the cumulative product.
    """
    # ``draft_logits`` are the per-position distributions the draft token is drawn
    # from (markov-corrected when a head is present, else the raw base logits).
    # Surfaced under ``return_logits`` to feed the rejection-sampling verifier;
    # the normal path ignores them.
    draft_logits = base_logits
    if markov_head is not None:
        draft_tokens, corrected = markov_head.sample_block_tokens(
            base_logits,
            first_prev_token_ids=bonus_token_ids,
            hidden_states=block_hidden,
            temperature=temperature,
        )
        draft_logits = corrected
    else:
        from .heads import greedy_or_sample

        draft_tokens = greedy_or_sample(base_logits, temperature)

    # Confidence scoring: branch-free and fixed-shape (no ``.item()``, no
    # data-dependent shapes), so the whole block stays CUDA-graph capturable.
    confidence = None
    if return_confidence and confidence_head is not None:
        # prev token at position k is [bonus, draft_0, ..., draft_{k-1}]
        prev_ids = torch.cat([bonus_token_ids.unsqueeze(1), draft_tokens[:, :-1]], dim=1)
        prev_emb = (
            markov_head.get_prev_embeddings(prev_ids)
            if (markov_head is not None and confidence_head.with_markov)
            else None
        )
        confidence = (
            confidence_head(block_hidden, prev_embeddings=prev_emb)
            if prev_emb is not None
            else confidence_head(block_hidden)
        )
    if return_logits:
        return draft_tokens, confidence, draft_logits
    return draft_tokens, confidence
