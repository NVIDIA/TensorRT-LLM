# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Per-slot state for text watermarking, shared by ``TorchSampler`` and one-model speculation.

A watermarked token is a keyed Gumbel-max draw whose randomness depends on the secret key
and the preceding ``context_width`` tokens (see ``tensorrt_llm.watermark``). The sampler
therefore needs, for every sequence slot, those preceding tokens on the device: under the
overlap scheduler the host-side token lists lag the device, and CUDA graphs cannot read
host state.

``WatermarkHandler`` keeps, per sequence slot:

* ``tail``: the last ``context_width`` tokens, initialised from the request's tokens when
  the slot gets a new occupant and advanced on the device from the tokens that were
  actually committed;
* a ring of the seeds of recent contexts, used to send a repeated context back to ordinary
  sampling (``deduplicate_contexts="single_turn"``).

A sampling step covers a *window* of positions per request: one position for plain
sampling, ``1 + k`` positions when ``k`` draft tokens are verified together. Position ``j``
of a window sees the context ``tail ++ drafts[:j]``. Verification accepts a draft only when
the target's own sample equals it, so the context of every position that is committed equals
the committed prefix and the output matches what non-speculative decoding would produce.

:meth:`WatermarkHandler.plan` and :meth:`WatermarkHandler.commit` use static shapes and no
host synchronization, so they can be captured in a CUDA graph.
"""

import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import torch

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.watermark.prf import PAD_TOKEN, context_seeds_torch, secondary_key
from tensorrt_llm.watermark.state_kernels import commit_windows, plan_windows
from tensorrt_llm.watermark.step_kernel import StepScratch, watermark_step

from ..llm_request import LlmRequest

if TYPE_CHECKING:
    from tensorrt_llm.watermark.config import WatermarkConfig

__all__ = ["WatermarkHandler", "WatermarkRows", "request_wants_watermark"]


def request_wants_watermark(config: "Optional[WatermarkConfig]", request: LlmRequest) -> bool:
    """Whether ``request`` asks for (or is forced into) a watermark, ignoring its sampling mode.

    ``SamplingParams.watermarking`` is None or True to request one and False to opt out; an
    opt-out is ignored when the engine sets ``allow_request_opt_out=False``.
    """
    if config is None:
        return False
    if getattr(request, "py_watermarking", None) is False:
        return not config.allow_request_opt_out
    return True


def _signed64(value: int) -> int:
    """``value`` in ``[0, 2**64)`` as the int64 with the same bit pattern."""
    return value - (1 << 64) if value >= (1 << 63) else value


def _owner_ref(request: LlmRequest) -> Callable[[], Optional[LlmRequest]]:
    """A reference to ``request`` that does not keep it alive when it can be weak."""
    try:
        return weakref.ref(request)
    except TypeError:
        return lambda: request


class PinnedStager:
    """Copies small host lists to a device buffer without allocating pinned memory per step.

    Allocating pinned host memory synchronizes with the device, which can deadlock a tensor
    parallel step whose ranks are inside a collective. The staging buffers are allocated once
    and reused round-robin; a buffer is rewritten only after its previous copy has finished.
    """

    def __init__(self, capacity: int, dtype: torch.dtype, depth: int = 4):
        self._dtype = dtype
        self._buffers = [
            torch.empty(capacity, dtype=dtype, pin_memory=prefer_pinned()) for _ in range(depth)
        ]
        self._events: list[Optional[torch.cuda.Event]] = [None] * depth
        self._next = 0

    def copy_to(self, values: list, dest: torch.Tensor) -> None:
        """Write ``values`` to the leading entries of ``dest`` (device) and zero-fill nothing else."""
        index = self._next
        self._next = (index + 1) % len(self._buffers)
        event = self._events[index]
        if event is not None:
            event.synchronize()
        count = len(values)
        buffer = self._buffers[index]
        buffer[:count] = torch.tensor(values, dtype=self._dtype)
        dest[:count].copy_(buffer[:count], non_blocking=True)
        event = self._events[index] = torch.cuda.Event()
        event.record()


@dataclass
class WatermarkRows:
    """Seeds and masks for the windows of one sampling call (``B`` requests, ``K1`` positions)."""

    slots: torch.Tensor  # [B] int64 sequence slots
    enabled: torch.Tensor  # [B] bool, requests whose window carries a watermark
    seeds: torch.Tensor  # [B, K1] int64
    active: torch.Tensor  # [B, K1] bool, positions that take the keyed draw
    # Snapshot of the state the window started from; only the reference implementation
    # (``plan_reference`` / ``commit_reference``) needs it.
    tail: Optional[torch.Tensor] = None  # [B, context_width] int64
    count: Optional[torch.Tensor] = None  # [B] int64
    # Seeds of the second key, for the tokens that follow a rejection (dual-key watermarking
    # with rejection sampling); equal to ``seeds`` when there is a single key.
    seeds_secondary: Optional[torch.Tensor] = None


class WatermarkHandler:
    def __init__(
        self,
        config: "WatermarkConfig",
        *,
        max_num_sequences: int,
        max_seq_len: int,
        device: str | torch.device = "cuda",
    ):
        self.config = config
        self._device = torch.device(device)
        self._secondary_key = (
            secondary_key(config.key) if config.algorithm == "gumbel_dual_key" else config.key
        )
        num_rows = max_num_sequences + 1  # the extra row is the scratch slot of padding requests
        width = config.context_width
        self._dual = self._secondary_key != config.key
        self._keys = torch.tensor(
            [_signed64(config.key), _signed64(self._secondary_key)],
            dtype=torch.int64,
            device=device,
        )
        self._ones = torch.ones(max_num_sequences + 1, dtype=torch.int64, device=device)
        self._step_scratch = StepScratch(num_rows, self._device)
        self._begin_cache: Optional[tuple] = None
        self._tail = torch.full((num_rows, width), PAD_TOKEN, dtype=torch.int64, device=device)
        self._dedupe = config.deduplicate_contexts == "single_turn"
        history = config.deduplicate_contexts_max_history
        self._ring_size = max_seq_len if history is None else min(history, max_seq_len)
        if self._dedupe:
            # One trash column past the ring takes the writes of positions that were not committed.
            self._ring = torch.zeros(
                (num_rows, self._ring_size + 1), dtype=torch.int64, device=device
            )
            self._ring_arange = torch.arange(self._ring_size, device=device)
        self._count = torch.zeros(num_rows, dtype=torch.int64, device=device)
        self._width_arange = torch.arange(width, device=device)
        # The request object that last took each slot. Compared by identity, not by request id:
        # warmup dummy requests reuse the ids that real requests are given later.
        self._slot_owner: list[Optional[Callable[[], Optional[LlmRequest]]]] = [None] * num_rows
        self._slot_enabled: list[bool] = [False] * num_rows
        # Whether the slot's sampling params filter nothing but the temperature (no top-k, top-p
        # or min-p), so its keyed draw can run on the logits without forming probabilities.
        self._slot_neutral: list[bool] = [False] * num_rows

    def requested(self, request: LlmRequest) -> bool:
        return request_wants_watermark(self.config, request)

    def observe(
        self,
        requests: list[LlmRequest],
        is_sampling: Callable[[LlmRequest], bool],
        is_neutral: Optional[Callable[[LlmRequest], bool]] = None,
    ) -> None:
        """Reset the state of every slot whose occupant changed.

        Called on the host before the sampling step, with the requests that will sample in it.
        A new occupant's context window starts as the last ``context_width`` tokens it holds,
        which is the prompt (plus any first token a disaggregated context worker already
        produced). ``is_sampling`` says whether the request's sampling mode can carry a
        watermark at all (greedy and beam search cannot).
        """
        width = self.config.context_width
        new_slots: list[int] = []
        new_tails: list[list[int]] = []
        for request in requests:
            slot = request.py_seq_slot
            if slot is None:
                continue
            owner = self._slot_owner[slot]
            if owner is not None and owner() is request:
                continue
            self._slot_owner[slot] = _owner_ref(request)
            enabled = self.requested(request) and is_sampling(request)
            self._slot_enabled[slot] = enabled
            self._slot_neutral[slot] = enabled and is_neutral is not None and is_neutral(request)
            if not enabled:
                continue
            tokens = request.get_tokens(0)[-width:]
            new_slots.append(slot)
            new_tails.append([PAD_TOKEN] * (width - len(tokens)) + list(tokens))
        if not new_slots:
            return
        pin = prefer_pinned()
        slots = torch.tensor(new_slots, dtype=torch.int64, pin_memory=pin).to(
            self._device, non_blocking=True
        )
        tails = torch.tensor(new_tails, dtype=torch.int64, pin_memory=pin).to(
            self._device, non_blocking=True
        )
        self._tail.index_copy_(0, slots, tails)
        self._count.index_fill_(0, slots, 0)

    def is_enabled(self, slot: int) -> bool:
        return self._slot_enabled[slot]

    def plan(
        self,
        slots: torch.Tensor,
        enabled: torch.Tensor,
        drafts: Optional[torch.Tensor] = None,
    ) -> WatermarkRows:
        """Seeds for the windows of ``B`` requests. Device-only; safe to capture in a graph.

        Args:
            slots: ``[B]`` int64 sequence slots (padding requests use the scratch slot).
            enabled: ``[B]`` bool, whether each request carries a watermark in this step.
            drafts: optional ``[B, k]`` draft tokens being verified; the window then has
                ``k + 1`` positions, otherwise one. Position ``j`` sees the last
                ``context_width`` tokens of ``tail ++ drafts[:j]``.
        """
        seeds, seeds_secondary, active = plan_windows(
            slots,
            enabled,
            self._tail,
            self._count,
            self._ring if self._dedupe else None,
            self._ring_size,
            self._keys,
            self._dual,
            drafts,
        )
        return WatermarkRows(
            slots=slots,
            enabled=enabled,
            seeds=seeds,
            active=active,
            seeds_secondary=seeds if seeds_secondary is None else seeds_secondary,
        )

    def plan_reference(
        self,
        slots: torch.Tensor,
        enabled: torch.Tensor,
        drafts: Optional[torch.Tensor] = None,
    ) -> WatermarkRows:
        """Plain-torch ``plan``: a slow reference the fused kernel is tested against.

        Args:
            slots: ``[B]`` int64 sequence slots (padding requests use the scratch slot).
            enabled: ``[B]`` bool, whether each request carries a watermark in this step.
            drafts: optional ``[B, k]`` draft tokens being verified; the window then has
                ``k + 1`` positions, otherwise one.
        """
        width = self.config.context_width
        batch = slots.shape[0]
        tail = self._tail.index_select(0, slots)
        extended = tail if drafts is None else torch.cat([tail, drafts.to(torch.int64)], dim=1)
        # Position j sees the last `width` tokens of tail ++ drafts[:j], i.e. extended[j : j+width].
        contexts = extended.unfold(1, width, 1)
        num_positions = contexts.shape[1]
        seeds = context_seeds_torch(self.config.key, contexts.reshape(-1, width)).reshape(
            batch, num_positions
        )
        active = enabled.unsqueeze(1).expand(batch, num_positions)
        count = self._count.index_select(0, slots)
        if self._dedupe:
            filled = torch.clamp(count, max=self._ring_size)
            ring = self._ring.index_select(0, slots)[:, : self._ring_size]
            seen = ring.unsqueeze(1) == seeds.unsqueeze(2)
            seen &= self._ring_arange.view(1, 1, -1) < filled.view(-1, 1, 1)
            same = seeds.unsqueeze(2) == seeds.unsqueeze(1)
            earlier = torch.tril(same, diagonal=-1).any(dim=2)
            active = active & ~(seen.any(dim=2) | earlier)
        if self._secondary_key == self.config.key:
            seeds_secondary = seeds
        else:
            seeds_secondary = context_seeds_torch(
                self._secondary_key, contexts.reshape(-1, width)
            ).reshape(batch, num_positions)
        return WatermarkRows(
            slots=slots,
            enabled=enabled,
            tail=tail,
            count=count,
            seeds=seeds,
            active=active,
            seeds_secondary=seeds_secondary,
        )

    def plan_draft_step(
        self,
        slots: torch.Tensor,
        enabled: torch.Tensor,
        prior_drafts: torch.Tensor,
        prior_seeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Seeds and activity of the next draft token of each request. Device-only.

        The draft continues from the state :meth:`commit` left behind, so its context is the
        committed tail followed by the drafts already produced in this step.

        Args:
            prior_drafts: ``[B, i]`` target-vocabulary ids of the ``i`` drafts produced so far.
            prior_seeds: ``[B, i]`` the seeds those drafts were drawn with.

        Returns:
            ``(seeds [B], active [B])``.
        """
        width = self.config.context_width
        tail = self._tail.index_select(0, slots)
        extended = torch.cat([tail, prior_drafts.to(torch.int64)], dim=1)
        seeds = context_seeds_torch(self.config.key, extended[:, -width:])
        active = enabled
        if self._dedupe:
            filled = torch.clamp(self._count.index_select(0, slots), max=self._ring_size)
            ring = self._ring.index_select(0, slots)[:, : self._ring_size]
            seen = ring == seeds.unsqueeze(1)
            seen &= self._ring_arange.unsqueeze(0) < filled.unsqueeze(1)
            earlier = (prior_seeds == seeds.unsqueeze(1)).any(dim=1)
            active = active & ~(seen.any(dim=1) | earlier)
        return seeds, active

    def commit(
        self,
        rows: WatermarkRows,
        accepted_tokens: torch.Tensor,
        num_accepted: Optional[torch.Tensor] = None,
    ) -> None:
        """Advance each window's state by the tokens that were committed.

        Args:
            accepted_tokens: ``[B, K1]`` (or ``[B]`` for one position) tokens of the window;
                only the first ``num_accepted`` of each row count.
            num_accepted: ``[B]`` committed tokens per request; defaults to one.
        """
        if accepted_tokens.dim() == 1:
            accepted_tokens = accepted_tokens.unsqueeze(1)
        batch = rows.seeds.shape[0]
        if num_accepted is None:
            num_accepted = self._ones[:batch]
        commit_windows(
            rows.slots,
            rows.enabled,
            accepted_tokens,
            num_accepted,
            rows.seeds,
            self._tail,
            self._count,
            self._ring if self._dedupe else None,
            self._ring_size,
        )

    def commit_reference(
        self,
        rows: WatermarkRows,
        accepted_tokens: torch.Tensor,
        num_accepted: Optional[torch.Tensor] = None,
    ) -> None:
        """Plain-torch ``commit``: a slow reference the fused kernel is tested against.

        Args:
            accepted_tokens: ``[B, K1]`` (or ``[B]`` for one position) tokens of the window;
                only the first ``num_accepted`` of each row count.
            num_accepted: ``[B]`` committed tokens per request; defaults to one.
        """
        if accepted_tokens.dim() == 1:
            accepted_tokens = accepted_tokens.unsqueeze(1)
        batch, num_positions = rows.seeds.shape
        if num_accepted is None:
            num_accepted = torch.ones(batch, dtype=torch.int64, device=accepted_tokens.device)
        # A request without a watermark leaves its slot untouched.
        committed = num_accepted.to(torch.int64) * rows.enabled.to(torch.int64)
        positions = torch.arange(num_positions, device=committed.device)
        if self._dedupe:
            target = (rows.count.unsqueeze(1) + positions) % self._ring_size
            target = torch.where(
                positions.unsqueeze(0) < committed.unsqueeze(1), target, self._ring_size
            )
            self._ring.index_put_(
                (rows.slots.unsqueeze(1).expand(batch, num_positions), target), rows.seeds
            )
        self._count.index_put_((rows.slots,), rows.count + committed)
        # New tail = the last `width` tokens of (old tail ++ the committed tokens).
        history = torch.cat([rows.tail, accepted_tokens.to(torch.int64)], dim=1)
        advanced = history.gather(1, committed.unsqueeze(1) + self._width_arange)
        self._tail.index_copy_(0, rows.slots, advanced)

    def begin(self, slots_per_row: list[int]) -> Optional[WatermarkRows]:
        """Host entry for plain sampling: one position per row, rows on distinct slots.

        Returns None when no row carries a watermark.
        """
        flags = tuple(self._slot_enabled[slot] for slot in slots_per_row)
        if not any(flags):
            return None
        # A steady decode batch repeats the same rows step after step: reuse its device tensors.
        key = (tuple(slots_per_row), flags)
        if self._begin_cache is not None and self._begin_cache[0] == key:
            slots, enabled = self._begin_cache[1], self._begin_cache[2]
        else:
            pin = prefer_pinned()
            slots = torch.tensor(slots_per_row, dtype=torch.int64, pin_memory=pin).to(
                self._device, non_blocking=True
            )
            enabled = torch.tensor(flags, dtype=torch.bool, pin_memory=pin).to(
                self._device, non_blocking=True
            )
            self._begin_cache = (key, slots, enabled)
        return self.plan(slots, enabled)

    def step_inputs(self, slots_per_row: list[int]) -> Optional[tuple]:
        """Host entry for the fused decode step: ``(slots, enabled, all_enabled, all_neutral)``.

        ``slots`` and ``enabled`` are device tensors (reused while the batch repeats);
        ``all_enabled`` and ``all_neutral`` say whether every row carries a watermark and
        whether every such row filters nothing but its temperature. None when no row carries
        a watermark.
        """
        flags = tuple(self._slot_enabled[slot] for slot in slots_per_row)
        if not any(flags):
            return None
        key = (tuple(slots_per_row), flags)
        if self._begin_cache is not None and self._begin_cache[0] == key:
            slots, enabled = self._begin_cache[1], self._begin_cache[2]
        else:
            pin = prefer_pinned()
            slots = torch.tensor(slots_per_row, dtype=torch.int64, pin_memory=pin).to(
                self._device, non_blocking=True
            )
            enabled = torch.tensor(flags, dtype=torch.bool, pin_memory=pin).to(
                self._device, non_blocking=True
            )
            self._begin_cache = (key, slots, enabled)
        all_neutral = all(
            self._slot_neutral[slot] for slot, flag in zip(slots_per_row, flags) if flag
        )
        return slots, enabled, all(flags), all_neutral

    @property
    def needs_fallback_token(self) -> bool:
        """Whether a watermarked row can fall back to the ordinary sampler's token."""
        return self._dedupe

    def sample_step(
        self,
        values: torch.Tensor,
        temperature: Optional[torch.Tensor],
        slots: torch.Tensor,
        enabled: torch.Tensor,
        fallback: Optional[torch.Tensor],
        *,
        from_probs: bool,
    ) -> torch.Tensor:
        """Plan, draw and commit one plain decode step in a single launch. Device-only."""
        return watermark_step(
            values,
            temperature,
            slots,
            enabled,
            fallback,
            self._tail,
            self._count,
            self._ring if self._dedupe else None,
            self._ring_size,
            self._keys,
            self._step_scratch,
            from_probs=from_probs,
        )
