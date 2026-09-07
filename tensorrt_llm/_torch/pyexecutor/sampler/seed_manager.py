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

"""Per-slot RNG state for ``SamplingParams.seed``.

``_SeedManager`` owns the Philox ``(seed, offset)`` pair per sequence slot;
``TorchSampler`` holds one instance and drives it per step. See the class
docstring for why a single batch-wide ``torch.Generator`` is not sufficient.
"""

from typing import Optional

import torch

from tensorrt_llm.logger import logger

from ..llm_request import LlmRequest
from .sampler_common import RequestSeeds, request_random_seed

__all__ = ["_SeedManager"]


class _SeedManager:
    """Per-sequence-slot RNG state implementing ``SamplingParams.seed``.

    A seeded request must produce the same tokens regardless of which other
    requests share its batch, so its RNG stream cannot come from a single
    batch-wide ``torch.Generator`` whose state advances by the batch's total
    draw count. Instead each row is sampled with an explicit Philox
    ``(seed, offset)`` pair: ``seed`` is the user's seed and ``offset`` is that
    request's own running draw count, both indexed by sequence slot.

    Requests without a seed fall back to ``global_seed`` and share the same
    per-slot offset counter. They are therefore deterministic for a given
    sequence slot, but their token stream is not stable across runs that batch
    them differently or assign them a different slot.

    .. warning::
       This plumbing is complete but **not yet effective for batched requests**.
       The pinned ``flashinfer-python`` (0.6.15) reads only ``seed[0]`` and
       ``offset[0]`` from the tensors it is handed and separates rows by
       ``blockIdx.x`` instead, so every row of a grouped sampling call draws
       from the first row's seed. A seeded request is reproducible only when it
       is the first row of its strategy group. ``observe`` emits a one-time
       warning when a seeded request is seen. The per-row state is kept here so
       that honoring ``SamplingParams.seed`` becomes a FlashInfer version bump
       rather than a redesign.

       Upstream fix in progress: https://github.com/flashinfer-ai/flashinfer/pull/2345
       ("add per-request generator support for sampling kernels"), which also
       switches ``curand_init`` to ``subsequence=0`` for per-request RNG. Note
       that it exposes the feature as ``generator=(seed_arr, offset_arr)``
       rather than the current ``seed=``/``offset=`` arguments, and advances
       ``offset_arr`` in-kernel -- so adopting it means passing the tuple and
       dropping ``advance`` below, not just bumping the pin.
    """

    # Philox offset units reserved per row per sampling call.
    #
    # An offset must advance by at least the number of random values the kernel
    # consumed, or the next call replays part of the same stream. That count is
    # not uniform: flashinfer sizes its offset allocation as ``batch_size`` for
    # categorical/min-p sampling but ``batch_size * 32`` for the top-k/top-p
    # rejection samplers, whose per-row consumption is data-dependent and only
    # bounded by that reserve.
    #
    # Rather than mirror flashinfer's per-op arithmetic here -- a detail that
    # can change upstream without any signal to this code -- every row advances
    # by the largest reserve. Offsets are int64, so spending 32 units where 1
    # would do costs nothing observable, and the stream stays non-overlapping
    # for every strategy.
    OFFSET_STRIDE = 32

    def __init__(self, *, max_num_sequences: int, global_seed: int):
        self._global_seed = global_seed
        # Indexed by py_seq_slot. Host-side int64; copied to device per step.
        self._seeds = torch.full((max_num_sequences,), global_seed, dtype=torch.int64)
        self._offsets = torch.zeros((max_num_sequences,), dtype=torch.int64)
        # Request id currently owning each slot, so that slot reuse by a new
        # request re-seeds instead of inheriting the previous occupant's stream.
        self._slot_owner: list[Optional[int]] = [None] * max_num_sequences
        # Per-slot flag: does the request currently occupying this slot carry a
        # user seed? Recomputed per step from the scheduled requests rather than
        # accumulated, so a finished seeded request stops forcing the per-row
        # path (and its cost) onto later unseeded requests.
        self._slot_seeded: list[bool] = [False] * max_num_sequences
        # Whether the batch observed in the current step contains a seeded
        # request. While false, the sampler keeps using the plain generator
        # path and pays nothing.
        self._any_seeded = False
        # Whether the batch observed in the current step is a draft batch.
        self._batch_is_draft = False

    @property
    def any_seeded(self) -> bool:
        """Whether per-row seeds apply to the batch observed this step.

        False for draft batches: their slot numbers come from a different
        ``SeqSlotManager`` and would index the wrong requests' RNG state, so
        draft sampling stays on the shared generator.
        """
        return self._any_seeded and not self._batch_is_draft

    def observe(self, requests: list[LlmRequest]) -> None:
        """Seed any slot whose occupant changed, and recompute ``any_seeded``.

        Called at the top of each sampling step rather than at slot-allocation
        time, which keeps this state owned entirely by the sampler. Resetting
        the offset on ownership change is what makes a seeded request start at
        the beginning of its stream instead of wherever the slot's previous
        occupant left off.

        ``any_seeded`` reflects only the requests passed in, so it falls back to
        False once no scheduled request carries a seed.

        Draft batches are ignored. ``ModelDrafter`` allocates draft slots from
        its own ``SeqSlotManager`` over the same numeric range, so a draft
        request can occupy a slot number that a live target request owns here.
        Observing it would look like a change of occupant and reset that
        target's offset, making it replay a stretch of its Philox stream. Draft
        sampling keeps using the shared generator.
        """
        # Batches are homogeneous (see TorchSampler._is_draft_batch), so the
        # first request decides for the whole batch.
        self._batch_is_draft = bool(requests) and requests[0].py_is_draft
        if self._batch_is_draft:
            return

        any_seeded = False
        for request in requests:
            seq_slot = request.py_seq_slot
            if seq_slot is None:
                continue
            request_id = request.py_request_id
            if self._slot_owner[seq_slot] != request_id:
                self._slot_owner[seq_slot] = request_id
                seed = request_random_seed(request)
                self._seeds[seq_slot] = self._global_seed if seed is None else seed
                self._offsets[seq_slot] = 0
                self._slot_seeded[seq_slot] = seed is not None
                if seed is not None:
                    logger.warning_once(
                        "SamplingParams.seed is only partially effective with the "
                        "TorchSampler: the pinned FlashInfer reads a single "
                        "seed/offset per sampling call and distinguishes rows "
                        "internally, so when several requests are sampled together "
                        "only the first row's seed applies. Seeded requests are "
                        "therefore not yet reproducible unless sampled alone. "
                        "Full per-request seeding will land once FlashInfer "
                        "honors per-row seeds (tracked in "
                        "https://github.com/flashinfer-ai/flashinfer/pull/2345).",
                        key="torch_sampler_per_request_seed_unsupported",
                    )
            any_seeded = any_seeded or self._slot_seeded[seq_slot]
        # Derived from this step's requests only; a seeded request that has
        # since finished no longer keeps the per-row path enabled.
        self._any_seeded = any_seeded

    def make_row_seeds(
        self,
        slots_per_row: list[int],
        *,
        device: torch.device,
    ) -> RequestSeeds:
        """Build the per-row ``(seed, offset)`` for one grouped sampling call.

        ``slots_per_row`` gives the sequence slot of each logits row, already
        expanded per step (a request drawing N tokens this iteration occupies N
        consecutive rows). Rows of the same request are spaced ``OFFSET_STRIDE``
        apart so each draw consumes a distinct stretch of the request's stream.
        """
        slots = torch.tensor(slots_per_row, dtype=torch.int64)
        seeds = self._seeds[slots]
        # Within-call, per-request draw index, scaled by the stride so rows of
        # the same request occupy disjoint stretches: 0, 32, 64, ...
        seen: dict[int, int] = {}
        within_list: list[int] = []
        for slot in slots_per_row:
            draw = seen.get(slot, 0)
            within_list.append(draw * self.OFFSET_STRIDE)
            seen[slot] = draw + 1
        within = torch.tensor(within_list, dtype=torch.int64)
        offsets = self._offsets[slots] + within
        return RequestSeeds(
            seed=seeds.to(device=device, non_blocking=True),
            offset=offsets.to(device=device, non_blocking=True),
        )

    def advance(self, slots_per_row: list[int]) -> None:
        """Advance each slot past the stream its rows consumed this step."""
        for slot in slots_per_row:
            self._offsets[slot] += self.OFFSET_STRIDE
