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
# The DSpark Markov/RNN/confidence-head math is ported from DeepSeek's DeepSpec
# reference implementation (https://github.com/deepseek-ai/DeepSpec, MIT License).
"""DSpark draft-network heads (pure-torch, framework-agnostic).

These modules implement the *sequential refinement* and *acceptance-confidence*
parts of DeepSeek's DSpark speculative-decoding draft network:

  - Markov head: a low-rank token-bigram logit bias ``logits_k += W2(W1[t_{k-1}])``
    applied autoregressively across the ``block_size`` draft positions (the cheap
    "sequential" half of DSpark's "semi-parallel" drafting). RNN variant carries
    a GRU-style recurrent state across positions.
  - Confidence head: predicts a per-position acceptance probability; the cumulative
    product over positions estimates prefix-acceptance and is used only to
    *truncate* the proposed draft length (NOT to decide acceptance).

This file deliberately depends on ``torch`` only so it can be unit-tested in
isolation (token-for-token) against the DeepSpec reference.
"""

from typing import Optional

import torch
from torch import nn


def greedy_or_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Argmax for temperature<=0, else temperature-scaled multinomial.

    Args:
        logits: ``[..., vocab]``.
    Returns:
        token ids with the trailing vocab dim reduced.
    """
    if temperature <= 0.0:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    flat = probs.reshape(-1, probs.shape[-1])
    sampled = torch.multinomial(flat, num_samples=1).squeeze(-1)
    return sampled.view(probs.shape[:-1])


class VanillaMarkov(nn.Module):
    """Low-rank token-bigram logit bias: ``bias = W2(W1[token])``."""

    markov_head_type = "vanilla"

    def __init__(self, *, vocab_size: int, markov_rank: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        assert self.markov_rank > 0, (
            f"VanillaMarkov requires markov_rank > 0, got {self.markov_rank}."
        )
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids.long())

    def project_bias(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(latent_states)

    def compute_step_bias(
        self, token_ids: torch.Tensor, hidden_states: Optional[torch.Tensor]
    ) -> torch.Tensor:
        del hidden_states
        return self.project_bias(self.get_prev_embeddings(token_ids))

    def apply_step_logits(
        self,
        logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return logits + self.compute_step_bias(token_ids, hidden_states)

    def sample_block_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive block sampling with the (memoryless) Markov bias.

        Args:
            base_logits: ``[batch, block_size, vocab]`` from the backbone+lm_head.
            first_prev_token_ids: ``[batch]`` token preceding the first position.
            hidden_states: ``[batch, block_size, d]`` (unused by vanilla/gated).
        Returns:
            sampled_tokens ``[batch, block_size]``, corrected_logits ``[batch, block_size, vocab]``.
        """
        batch_size, block_size = base_logits.shape[:2]
        if block_size == 0:
            empty = torch.empty(batch_size, 0, dtype=torch.long, device=base_logits.device)
            return empty, base_logits
        sampled, corrected = [], []
        prev = first_prev_token_ids.long()
        for k in range(block_size):
            step_hidden = None if hidden_states is None else hidden_states[:, k]
            step_logits = self.apply_step_logits(
                base_logits[:, k], token_ids=prev, hidden_states=step_hidden
            )
            corrected.append(step_logits.unsqueeze(1))
            prev = greedy_or_sample(step_logits, temperature)
            sampled.append(prev)
        return torch.stack(sampled, dim=1), torch.cat(corrected, dim=1)


class GatedMarkovHead(VanillaMarkov):
    """Markov bias gated by a sigmoid of [hidden, prev_embedding]."""

    markov_head_type = "gated"

    def __init__(self, *, vocab_size: int, markov_rank: int, hidden_size: int):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.gate_proj = nn.Linear(hidden_size + markov_rank, markov_rank)

    def compute_step_bias(
        self, token_ids: torch.Tensor, hidden_states: Optional[torch.Tensor]
    ) -> torch.Tensor:
        assert hidden_states is not None
        prev_emb = self.get_prev_embeddings(token_ids)
        gate = torch.sigmoid(self.gate_proj(torch.cat([hidden_states, prev_emb], dim=-1))).to(
            dtype=prev_emb.dtype
        )
        return self.project_bias(gate * prev_emb)


class RNNHead(VanillaMarkov):
    """GRU-style head carrying recurrent state across block positions."""

    markov_head_type = "rnn"

    def __init__(self, *, vocab_size: int, markov_rank: int, hidden_size: int):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.hidden_size = int(hidden_size)
        # [s_{k-1}; W1[x_{k-1}]; h_k] -> [gate; candidate; output]
        self.joint_proj = nn.Linear(2 * markov_rank + hidden_size, 3 * markov_rank)

    def _rnn_step(self, state, prev_embeddings, hidden_states):
        z = torch.cat([state, prev_embeddings, hidden_states], dim=-1)
        gate_raw, cand_raw, out_raw = self.joint_proj(z).chunk(3, dim=-1)
        gate = torch.sigmoid(gate_raw)
        candidate = torch.tanh(cand_raw)
        new_state = gate * state + (1.0 - gate) * candidate
        bias = self.project_bias(torch.tanh(out_raw))
        return new_state, bias

    def sample_block_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert hidden_states is not None
        batch_size, block_size = base_logits.shape[:2]
        if block_size == 0:
            empty = torch.empty(batch_size, 0, dtype=torch.long, device=base_logits.device)
            return empty, base_logits
        state = torch.zeros(
            batch_size, self.markov_rank, device=base_logits.device, dtype=hidden_states.dtype
        )
        sampled, corrected = [], []
        prev = first_prev_token_ids.long()
        for k in range(block_size):
            prev_emb = self.get_prev_embeddings(prev)
            state, bias = self._rnn_step(state, prev_emb, hidden_states[:, k])
            step_logits = base_logits[:, k] + bias
            corrected.append(step_logits.unsqueeze(1))
            prev = greedy_or_sample(step_logits, temperature)
            sampled.append(prev)
        return torch.stack(sampled, dim=1), torch.cat(corrected, dim=1)


def build_markov_head(
    *, markov_head_type: str, vocab_size: int, markov_rank: int, hidden_size: int
) -> Optional[nn.Module]:
    """Factory mirroring DeepSpec ``build_markov_head``; returns None if rank==0."""
    if int(markov_rank) <= 0:
        return None
    kind = str(markov_head_type).lower()
    if kind == "vanilla":
        return VanillaMarkov(vocab_size=vocab_size, markov_rank=markov_rank)
    if kind == "gated":
        return GatedMarkovHead(
            vocab_size=vocab_size, markov_rank=markov_rank, hidden_size=hidden_size
        )
    if kind == "rnn":
        return RNNHead(vocab_size=vocab_size, markov_rank=markov_rank, hidden_size=hidden_size)
    raise ValueError(f"Unsupported markov_head_type: {markov_head_type!r}")


class DSparkConfidenceHead(nn.Module):
    """Per-position acceptance-confidence predictor (DeepSpec AcceptRatePredictor).

    Input features are the backbone hidden state, optionally concatenated with the
    Markov head's previous-token embedding. Output is a single logit per position.
    """

    def __init__(self, *, hidden_size: int, markov_rank: int = 0, with_markov: bool = False):
        super().__init__()
        self.with_markov = bool(with_markov)
        input_dim = int(hidden_size) + (int(markov_rank) if with_markov else 0)
        # The checkpoint stores ``proj`` as a bias-free bf16 weight, but the
        # confidence score is computed in fp32 (mirrors the DeepSpec reference
        # ``Linear(input_dim, 1, dtype=torch.float32)`` with the fp32 matmul).
        self.proj = nn.Linear(input_dim, 1, bias=False, dtype=torch.float32)

    def forward(
        self, hidden_states: torch.Tensor, prev_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.with_markov:
            assert prev_embeddings is not None
            features = torch.cat([hidden_states, prev_embeddings.to(hidden_states.dtype)], dim=-1)
        else:
            features = hidden_states
        # fp32 matmul for a stable confidence score (mirrors the reference).
        return self.proj(features.float()).squeeze(-1)


def confident_prefix_length(
    confidence_logits: torch.Tensor, *, block_size: int, threshold: float
) -> int:
    """First position k where ``sigmoid(confidence_k) < threshold``.

    Returns ``block_size`` when threshold<=0 (no truncation) or all positions
    are confident. Assumes batch size 1 (functional-first scope).
    """
    if threshold <= 0.0:
        return int(block_size)
    below = confidence_logits.sigmoid() < threshold
    if not bool(below[0].any().item()):
        return int(block_size)
    return int(torch.nonzero(below[0], as_tuple=False)[0].item())
