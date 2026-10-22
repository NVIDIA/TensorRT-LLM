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

"""Per-row RNG state for ``TorchSampler``.

``_SeedManager`` hands each sampled row its Philox ``(seed, offset)`` pair;
``TorchSampler`` holds one instance and drives it per step. See the class
docstring for why a single batch-wide ``torch.Generator`` is not sufficient.
"""

from typing import Optional

import torch

from tensorrt_llm._utils import prefer_pinned

from ..llm_request import LlmRequest
from .ops.custom import UNSEEDED_OFFSET_BASE
from .sampler_common import RequestSeeds, request_random_seed

__all__ = ["_SeedManager"]


class _SeedManager:
    """Per-row Philox ``(seed, offset)`` for the fused sampling kernel.

    A seeded request's RNG stream must not depend on which other requests
    share its batch, so it cannot come from a single batch-wide
    ``torch.Generator`` whose state advances by the batch's total draw count.
    Instead every row is sampled with its own Philox ``(seed, offset)``, and
    the kernel keeps the row index out of per-row streams, so a row's draws
    depend on that pair alone.

    - A seeded request uses its own seed, and an offset counted per sequence
      slot. The counter restarts when a new request takes the slot, so the
      stream depends only on how far the request has decoded.
    - An unseeded request uses ``global_seed`` and an offset from a counter
      shared by all unseeded rows, which starts at ``UNSEEDED_OFFSET_BASE``
      and only ever grows. Unseeded requests therefore never replay each
      other's streams or their own earlier steps, nor the stream of a request
      whose seed equals ``global_seed``.
    """

    def __init__(self, *, max_num_sequences: int, global_seed: int):
        self._global_seed = global_seed
        # Indexed by py_seq_slot. Host-side int64; copied to device per step.
        self._seeds = torch.full((max_num_sequences,), global_seed, dtype=torch.int64)
        self._offsets = torch.zeros((max_num_sequences,), dtype=torch.int64)
        # Request id currently owning each slot, so that slot reuse by a new
        # request re-seeds instead of inheriting the previous occupant's stream.
        self._slot_owner: list[Optional[int]] = [None] * max_num_sequences
        # Per-slot flag: does the request currently occupying this slot carry a
        # user seed?
        self._slot_seeded: list[bool] = [False] * max_num_sequences
        # Offset of the next unseeded row.
        self._unseeded_offset = UNSEEDED_OFFSET_BASE
        # Whether the batch observed in the current step is a draft batch.
        self._batch_is_draft = False

    def observe(self, requests: list[LlmRequest]) -> None:
        """Seed any slot whose occupant changed.

        Called at the top of each sampling step rather than at slot-allocation
        time, which keeps this state owned entirely by the sampler. Resetting
        the offset on ownership change is what makes a seeded request start at
        the beginning of its stream instead of wherever the slot's previous
        occupant left off.

        Draft batches are not recorded per slot. A drafter allocates draft slots
        from its own ``SeqSlotManager`` over the same numeric range, so a draft
        request can occupy a slot number that a live target request owns here.
        Observing it would look like a change of occupant and reset that
        target's offset, making it replay a stretch of its Philox stream. Draft
        rows draw as unseeded rows instead.
        """
        # Batches are homogeneous (see TorchSampler._is_draft_batch), so the
        # first request decides for the whole batch.
        self._batch_is_draft = bool(requests) and requests[0].py_is_draft
        if self._batch_is_draft:
            return

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

    def take_row_seeds(
        self,
        slots_per_row: list[int],
        *,
        device: torch.device,
    ) -> RequestSeeds:
        """Assign each row of one sampling call its ``(seed, offset)``, and advance.

        ``slots_per_row`` gives the sequence slot of each logits row, already
        expanded per step (a request drawing N tokens this iteration occupies N
        consecutive rows). Every row is given an offset that no earlier row of
        the same stream has used.
        """
        num_rows = len(slots_per_row)
        # Calls without a seeded row are the common case; build them on the
        # device, without a per-row loop or a host-to-device copy.
        if self._batch_is_draft or not any(map(self._slot_seeded.__getitem__, slots_per_row)):
            start = self._unseeded_offset
            self._unseeded_offset += num_rows
            return RequestSeeds(
                seed=torch.full((num_rows,), self._global_seed, dtype=torch.int64, device=device),
                offset=torch.arange(start, self._unseeded_offset, dtype=torch.int64, device=device),
            )
        seeds: list[int] = []
        offsets: list[int] = []
        for slot in slots_per_row:
            if self._batch_is_draft or not self._slot_seeded[slot]:
                seeds.append(self._global_seed)
                offsets.append(self._unseeded_offset)
                self._unseeded_offset += 1
            else:
                seeds.append(int(self._seeds[slot]))
                offsets.append(int(self._offsets[slot]))
                self._offsets[slot] += 1
        pin = prefer_pinned()
        return RequestSeeds(
            seed=torch.tensor(seeds, dtype=torch.int64, pin_memory=pin).to(
                device=device, non_blocking=True
            ),
            offset=torch.tensor(offsets, dtype=torch.int64, pin_memory=pin).to(
                device=device, non_blocking=True
            ),
        )
