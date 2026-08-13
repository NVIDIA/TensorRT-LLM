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
    the block tokens and apply the static confidence-threshold truncation.

The backbone (3 V4 blocks producing ``block_hidden``) lives in the model module;
this file is the part fully specified by the reference and validated against it.
"""

from typing import Optional

import torch
from torch import nn

from .heads import confident_prefix_length


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
    confidence_threshold: float = 0.0,
    return_logits: bool = False,
) -> tuple:
    """Produce DSpark draft tokens for one block (functional-first, static length).

    Args:
        base_logits: ``[batch, block_size, vocab]`` from the backbone + lm_head.
        bonus_token_ids: ``[batch]`` the token preceding the first draft position.
        block_hidden: ``[batch, block_size, hidden]`` backbone hidden (feeds the
            confidence head, and the RNN-head variant).
        markov_head / confidence_head: the validated DSpark heads (may be None).
    Returns:
        draft_tokens: ``[batch, block_size]`` sampled tokens (full block; callers
            keep the tensor fixed-width for CUDA-graph safety).
        num_proposed: ``[batch]`` int32 — how many leading tokens survive the
            static confidence-threshold truncation (== block_size when no head /
            threshold<=0).
    """
    batch = base_logits.shape[0]
    # ``draft_logits`` are the per-position distributions the draft token is drawn
    # from (markov-corrected when a head is present, else the raw base logits).
    # Surfaced under ``return_logits`` for the §7.9 probabilistic-acceptance
    # (1-TV) measurement; the normal path ignores them.
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

    # Scaffolding: confidence-based dynamic drafting is NOT enabled in this PR.
    # The worker always calls with confidence_threshold=0.0, so the block below is
    # inert and num_proposed stays == block_size (the full block is proposed). The
    # returned num_proposed is intentionally not yet consumed by the speculative
    # scheduler/verifier; wiring it through is a follow-up (see PR description).
    num_proposed = torch.full(
        (batch,), int(block_size), dtype=torch.int32, device=base_logits.device
    )
    if confidence_head is not None and confidence_threshold > 0.0:
        # prev token at position k is [bonus, draft_0, ..., draft_{k-1}]
        prev_ids = torch.cat([bonus_token_ids.unsqueeze(1), draft_tokens[:, :-1]], dim=1)
        prev_emb = (
            markov_head.get_prev_embeddings(prev_ids)
            if (markov_head is not None and getattr(confidence_head, "with_markov", False))
            else None
        )
        conf_logits = (
            confidence_head(block_hidden, prev_embeddings=prev_emb)
            if prev_emb is not None
            else confidence_head(block_hidden)
        )
        # Per-request prefix truncation (batch handled row-wise to stay simple;
        # functional-first scope typically runs batch=1 for the draft).
        for b in range(batch):
            num_proposed[b] = confident_prefix_length(
                conf_logits[b : b + 1], block_size=block_size, threshold=confidence_threshold
            )
    if return_logits:
        return draft_tokens, num_proposed, draft_logits
    return draft_tokens, num_proposed
