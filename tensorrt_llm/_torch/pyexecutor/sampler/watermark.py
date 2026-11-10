# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GPU-authoritative context and post-filter Gumbel watermark selection."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from tensorrt_llm.watermarking.config import WatermarkConfig
from tensorrt_llm.watermarking.prf import (
    CONTEXT_HASH_SEEDS,
    PHILOX_M0,
    PHILOX_M1,
    PHILOX_W0,
    PHILOX_W1,
    UINT32_MASK,
)

if TYPE_CHECKING:
    from ..llm_request import LlmRequest


def _rotate_left(value: torch.Tensor, amount: int) -> torch.Tensor:
    return ((value << amount) | (value >> (32 - amount))) & UINT32_MASK


def _context_hash(context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Hash int64 contexts of shape [batch, context_width] into uint32 words."""
    hashes = []
    for seed in CONTEXT_HASH_SEEDS:
        value = torch.full_like(context[:, 0], seed)
        for index in range(context.shape[1]):
            word = ((context[:, index] & UINT32_MASK) * 0xCC9E2D51) & UINT32_MASK
            word = (_rotate_left(word, 15) * 0x1B873593) & UINT32_MASK
            value = _rotate_left(value ^ word, 13)
            value = (value * 5 + 0xE6546B64) & UINT32_MASK
        value = value ^ (4 * context.shape[1])
        value = value ^ (value >> 16)
        value = (value * 0x85EBCA6B) & UINT32_MASK
        value = value ^ (value >> 13)
        value = (value * 0xC2B2AE35) & UINT32_MASK
        value = value ^ (value >> 16)
        hashes.append(value)
    return hashes[0], hashes[1]


def _multiply_high_low(multiplier: int, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Splitting at 16 bits keeps every intermediate inside signed int64,
    # including the unsigned high word of a 32x32-bit product.
    product_low = multiplier * (value & 0xFFFF)
    product_high = multiplier * (value >> 16)
    carry = product_low + ((product_high & 0xFFFF) << 16)
    return (product_high >> 16) + (carry >> 32), carry & UINT32_MASK


def philox_uniforms(
    key: int,
    context: torch.Tensor,
    tokens: torch.Tensor,
    domain: int = 0,
    *,
    paired: bool = False,
) -> torch.Tensor:
    """Version-1 keyed uniforms for int64 [batch, width] contexts and [vocab] ids.

    Returns float64 [batch, vocab] on the input device, or [batch, 1] when
    ``paired=True`` supplies one token id per context. Float64 preserves the
    open interval even at the largest uint32 output; rounding to float32 before
    the Gumbel transform could produce infinity.
    """
    hash0, hash1 = _context_hash(context)
    if paired:
        if tokens.ndim != 1 or tokens.shape[0] != context.shape[0]:
            raise ValueError("Paired Philox input requires one token per context")
        c0 = tokens[:, None] & UINT32_MASK
    else:
        c0 = tokens[None, :].expand(context.shape[0], -1) & UINT32_MASK
    c1 = hash0[:, None].expand_as(c0)
    c2 = hash1[:, None].expand_as(c0)
    c3 = torch.full_like(c0, domain)
    k0, k1 = key & UINT32_MASK, key >> 32
    for _ in range(10):
        high0, low0 = _multiply_high_low(PHILOX_M0, c0)
        high1, low1 = _multiply_high_low(PHILOX_M1, c2)
        c0, c1, c2, c3 = (
            (high1 ^ c1 ^ k0) & UINT32_MASK,
            low1,
            (high0 ^ c3 ^ k1) & UINT32_MASK,
            low0,
        )
        k0 = (k0 + PHILOX_W0) & UINT32_MASK
        k1 = (k1 + PHILOX_W1) & UINT32_MASK
    return (c0.to(torch.float64) + 0.5) / (1 << 32)


@dataclass
class _SlotWatermarkState:
    request_id: int
    context: torch.Tensor
    history: torch.Tensor
    num_tokens: int
    position: int = 0
    history_count: int = 0


class WatermarkHandler:
    """Maintain per-slot device history and select from native filtered probabilities.

    Each state is initialized once, from the full logical prompt at final
    prefill, and updated immediately with the selected device token. No decode
    step reads the request's lagging host output list. Only active slots allocate
    context-deduplication storage, and release or a new owner discards the state.
    """

    def __init__(self, config: WatermarkConfig | None, device: str = "cuda"):
        self.config = config
        self._device = device
        self._states: dict[int, _SlotWatermarkState] = {}
        # Each Philox intermediate has at most this many candidate elements,
        # or one full vocabulary row when that row alone exceeds the budget.
        self._max_candidate_elements = 1 << 20

    def enabled(self, request: "LlmRequest") -> bool:
        """Whether the engine and request enable watermarking, before greedy bypass."""
        return (
            self.config is not None
            and not request.is_dummy
            and getattr(request, "py_watermarking", None) is not False
        )

    def setup_request(self, request: "LlmRequest") -> None:
        """Reset one slot from its new owner's full prompt, or remove stale state."""
        slot = request.py_seq_slot
        assert slot is not None
        existing = self._states.get(slot)
        if existing is not None and existing.request_id == request.py_request_id:
            return
        for previous_slot, previous_state in list(self._states.items()):
            if previous_state.request_id == request.py_request_id:
                self._states[slot] = self._states.pop(previous_slot)
                return
        self._states.pop(slot, None)
        if not self.enabled(request):
            return
        assert self.config is not None
        width = self.config.context_width
        tokens = request.get_tokens(0)
        suffix = list(tokens[-width:])
        history_capacity = self.config.deduplicate_contexts_max_history
        first_scored_position = max(request.py_orig_prompt_len, width)
        prior_count = max(0, len(tokens) - first_scored_position)
        state = _SlotWatermarkState(
            request_id=request.py_request_id,
            context=torch.tensor(
                [0] * (width - len(suffix)) + suffix, dtype=torch.int64, device=self._device
            ),
            history=torch.empty(
                (history_capacity, width),
                dtype=torch.int64,
                device=self._device,
            ),
            num_tokens=len(tokens),
            position=prior_count,
            history_count=min(prior_count, history_capacity),
        )
        # A recomputed request may already contain accepted completion tokens.
        # Reconstruct the bounded history from that committed prefix when its
        # previous device state was released with the old execution resources.
        if prior_count:
            prior_positions = range(max(0, prior_count - history_capacity), prior_count)
            history_contexts = [
                tokens[first_scored_position + position - width : first_scored_position + position]
                for position in prior_positions
            ]
            history_indices = torch.tensor(
                [position % history_capacity for position in prior_positions],
                dtype=torch.int64,
                device=self._device,
            )
            state.history.index_copy_(
                0,
                history_indices,
                torch.tensor(history_contexts, dtype=torch.int64, device=self._device),
            )
        self._states[slot] = state

    def release_request(self, request: "LlmRequest") -> None:
        """Discard a departing request's device history without touching a newer owner."""
        slot = request.py_seq_slot
        state = self._states.get(slot)
        if state is not None and state.request_id == request.py_request_id:
            del self._states[slot]

    def sample(
        self, probabilities: torch.Tensor, stock_tokens: torch.Tensor, slots: list[int]
    ) -> torch.Tensor:
        """Select watermarked tokens for [batch, vocab] float32 native distributions.

        The caller admits only ordinary stochastic rows with one row per slot.
        Insufficient context and repeated contexts return the native sampled
        token. The original probabilities also remain the reported logprobs.
        """
        assert self.config is not None
        if len(slots) != len(set(slots)):
            raise ValueError("Watermarking requires one ordinary decode row per sequence")
        states = [self._states[slot] for slot in slots]
        contexts = torch.stack([state.context for state in states])
        vocabulary = torch.arange(probabilities.shape[1], device=probabilities.device)
        positions = torch.tensor(
            [state.position for state in states], dtype=torch.int64, device=contexts.device
        )
        rows_per_chunk = max(1, self._max_candidate_elements // probabilities.shape[1])
        selected = torch.cat(
            [
                self._select_chunk(
                    probabilities[start : start + rows_per_chunk],
                    contexts[start : start + rows_per_chunk],
                    positions[start : start + rows_per_chunk],
                    vocabulary,
                )
                for start in range(0, len(slots), rows_per_chunk)
            ]
        )
        result = []
        for row, state in enumerate(states):
            chosen = stock_tokens[row]
            if state.num_tokens >= self.config.context_width:
                repeated = (state.history[: state.history_count] == state.context).all(dim=-1).any()
                chosen = torch.where(repeated, chosen, selected[row].to(chosen.dtype))
                capacity = state.history.shape[0]
                if capacity:
                    state.history[state.position % capacity].copy_(state.context)
                    state.history_count = min(state.history_count + 1, capacity)
                state.position += 1
            state.context.copy_(torch.cat((state.context[1:], chosen.reshape(1))))
            state.num_tokens += 1
            result.append(chosen)
        return torch.stack(result)

    def _select_chunk(
        self,
        probabilities: torch.Tensor,
        contexts: torch.Tensor,
        positions: torch.Tensor,
        vocabulary: torch.Tensor,
    ) -> torch.Tensor:
        """Bound Philox workspace independently of the overall batch size."""
        assert self.config is not None
        uniforms = philox_uniforms(self.config.key, contexts, vocabulary)
        if self.config.algorithm == "dual_key_gumbel":
            secondary = philox_uniforms(self.config.key, contexts, vocabulary, domain=1)
            # Context and position domain-separate this choice from candidate
            # noise. Paired inputs avoid allocating a batch-by-batch matrix.
            choices = philox_uniforms(
                self.config.key, contexts, positions, domain=2, paired=True
            ).squeeze(-1)
            uniforms = torch.where((choices < self.config.alpha)[:, None], secondary, uniforms)
        gumbels = -torch.log(-torch.log(uniforms))
        return (torch.log(probabilities.to(torch.float64)) + gumbels).argmax(dim=-1)
