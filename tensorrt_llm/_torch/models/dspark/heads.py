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
"""DSpark draft-network heads (pure-torch so they unit-test against DeepSpec).

Markov/RNN heads apply an autoregressive per-position logit bias across the
draft block. The confidence head predicts per-position *conditional* acceptance
probabilities (``P(accept_k | accept_1..k-1)``) whose cumulative product the
verification scheduler budgets against; it decides *how many* tokens are
verified, never *whether* one is accepted, so scheduling stays lossless.
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

    Emits one raw logit per position; :meth:`apply_sts` turns it into a
    probability, folding in the sequential-temperature-scaling (STS)
    calibration (``sts_temperatures`` defaults to all-ones: plain sigmoid).
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        markov_rank: int = 0,
        with_markov: bool = False,
        bias: bool = False,
        block_size: int = 0,
    ):
        super().__init__()
        self.with_markov = bool(with_markov)
        input_dim = int(hidden_size) + (int(markov_rank) if with_markov else 0)
        # Checkpoint weight is bf16, but the score matmul is fp32 (DeepSpec parity).
        self.proj = nn.Linear(input_dim, 1, bias=bool(bias), dtype=torch.float32)
        # Per-position temperatures, broadcast against ``[batch, block]``.
        # Update in place only: a captured CUDA graph holds this storage;
        # rebinding the attribute would be invisible to the graph.
        self.register_buffer(
            "sts_temperatures",
            torch.ones(max(int(block_size), 1), dtype=torch.float32),
            persistent=False,
        )
        # CPU mirror of ``sts_temperatures``; rebuilt lazily by
        # ``_host_sts_temperatures`` and dropped by ``load_sts_temperatures``.
        self._sts_temperatures_host: Optional[torch.Tensor] = None

    def forward(
        self, hidden_states: torch.Tensor, prev_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """``[*, block, hidden] -> [*, block]`` raw (uncalibrated) logits."""
        if self.with_markov:
            assert prev_embeddings is not None
            features = torch.cat([hidden_states, prev_embeddings.to(hidden_states.dtype)], dim=-1)
        else:
            features = hidden_states
        # fp32 matmul for a stable confidence score (mirrors the reference).
        return self.proj(features.float()).squeeze(-1)

    def apply_sts(self, confidence_logits: torch.Tensor) -> torch.Tensor:
        """Raw logits -> calibrated per-position acceptance probabilities in [0, 1].

        Must accept CPU logits: the verification planner calibrates on pinned
        host memory while the head lives on the device. The device-mismatch
        branch is host-side and devices match on the in-graph path, so nothing
        extra is captured.
        """
        temperatures = self.sts_temperatures
        if temperatures.device != confidence_logits.device:
            if confidence_logits.device.type == "cpu":
                temperatures = self._host_sts_temperatures()
            else:
                temperatures = temperatures.to(confidence_logits.device)
        return torch.sigmoid(confidence_logits.float() / temperatures)

    def _host_sts_temperatures(self) -> torch.Tensor:
        """CPU mirror of ``sts_temperatures``, materialized at most once."""
        cached = self._sts_temperatures_host
        if cached is None:
            cached = self.sts_temperatures.detach().to("cpu")
            self._sts_temperatures_host = cached
        return cached

    @torch.no_grad()
    def load_sts_temperatures(self, temperatures: torch.Tensor) -> None:
        """In-place update of the STS table (never rebind: see ``__init__``)."""
        flat = temperatures.reshape(-1).to(device=self.sts_temperatures.device, dtype=torch.float32)
        if flat.numel() != self.sts_temperatures.numel():
            raise ValueError(
                f"STS temperature table has {flat.numel()} entries but the confidence "
                f"head expects {self.sts_temperatures.numel()} (one per block position)"
            )
        if not bool(torch.all(flat > 0.0)):
            raise ValueError("STS temperatures must be strictly positive")
        self.sts_temperatures.copy_(flat)
        # Drop the CPU mirror; apply_sts rebuilds it on next use.
        self._sts_temperatures_host = None

    def load_weights(self, weights: list) -> None:
        """Strict loader: fail loudly on checkpoint keys the module lacks (a
        silently dropped ``proj.bias`` would shift every confidence score)."""
        (module_weights,) = weights
        expected = dict(self.named_parameters())
        unexpected = [k for k in module_weights if k not in expected]
        if unexpected:
            raise ValueError(
                f"DSpark confidence head checkpoint has unsupported key(s) {sorted(unexpected)}; "
                f"the module exposes {sorted(expected)}. A checkpoint bias requires constructing "
                f"DSparkConfidenceHead(bias=True) -- silently dropping it would bias every "
                f"per-position confidence score."
            )
        for name, param in expected.items():
            param.data.copy_(module_weights[name][:])
