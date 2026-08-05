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

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest


@dataclass
class AuxBufferMeta:
    ptrs: np.ndarray  # dtype=np.int64
    size: np.ndarray  # dtype=np.int64
    item_sizes: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    device: str = "cpu"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ptrs": self.ptrs.tolist(),
            "size": self.size.tolist(),
            "item_sizes": self.item_sizes.tolist(),
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuxBufferMeta":
        return cls(
            ptrs=np.array(data["ptrs"], dtype=np.int64),
            size=np.array(data["size"], dtype=np.int64),
            item_sizes=np.array(data.get("item_sizes", []), dtype=np.int64),
            device=data.get("device", "cpu"),
        )


@dataclass(frozen=True, slots=True)
class AuxAllocationIdentity:
    """One allocator-issued generation of an auxiliary-buffer slot."""

    allocator_domain_id: UUID
    request_id: int | None
    allocation_generation: int


@dataclass(frozen=True, slots=True)
class AuxSlot:
    """An allocated auxiliary slot and its immutable allocation identity."""

    id: int
    buffer: "AuxBufferBase"
    identity: AuxAllocationIdentity


class AuxBufferBase(ABC):
    """
    Abstract base class defining the interface for auxiliary buffer management.
    """

    @abstractmethod
    def alloc_slot(self, request_id: int | None = None) -> AuxSlot:
        """
        Allocate a free slot and return its index.
        """
        ...

    @abstractmethod
    def free_slot(
        self,
        slot: int,
        identity: AuxAllocationIdentity | None = None,
    ) -> None:
        """
        Release the specified slot.

        When ``identity`` is supplied, the release applies only to that exact
        allocation generation. This prevents delayed cleanup from freeing a
        slot that has already been recycled for a newer request.
        """
        ...

    @abstractmethod
    def allocation_identity(self, slot: int) -> AuxAllocationIdentity:
        """Return the immutable identity of the live slot allocation."""
        ...

    @property
    @abstractmethod
    def meta(self) -> AuxBufferMeta:
        """
        Retrieve meta-information about the underlying buffer(s).
        Returns buffer info (e.g., pointers, sizes, device).
        """
        ...

    @abstractmethod
    def fill_slot(
        self,
        slot: int,
        request: LlmRequest,
        identity: AuxAllocationIdentity | None = None,
    ) -> None:
        """
        Fill/overwrite the contents of the given slot with data from the request.

        When ``identity`` is supplied, the write applies only to that exact
        allocation generation.
        """
        ...

    @abstractmethod
    def get_slot_tokens(self, slot: int) -> tuple[list[int], list[int]]:
        """
        Get the token data (e.g., first/draft tokens) from the specified slot.
        """
        ...

    @abstractmethod
    def get_slot_data(
        self,
        slot: int,
        identity: AuxAllocationIdentity | None = None,
    ) -> tuple[list[int], list[int], tuple[int, int]]:
        """
        Get the token data and prompt token counts from the specified slot.

        When ``identity`` is supplied, the read applies only to that exact
        allocation generation.

        Returns:
            (first_gen_tokens, draft_tokens, (prompt_tokens, cached_tokens))
        """
        ...


class AuxBuffer(AuxBufferBase):
    def __init__(self, max_slot_num: int, beam_width: int, max_draft_len: int, device: str = "cpu"):
        # public constructor args remain the same, internals are private
        self._max_slot_num = int(max_slot_num)
        self._beam_width = int(beam_width)
        self._max_draft_len = int(max_draft_len)
        self._device = device

        self._free_slots = deque(list(range(self._max_slot_num)))
        self._occupied_slots: set[int] = set()
        self._allocator_domain_id = uuid4()
        self._allocation_generation = 0
        self._slot_identities: dict[int, AuxAllocationIdentity] = {}
        self._lock = RLock()
        self._slot_token_counts: dict[
            int, tuple[int, int]
        ] = {}  # slot -> (first_tokens_len, draft_tokens_len)

        data_type = torch.int32
        self._first_tokens_buffer = torch.empty(
            self._max_slot_num, self._beam_width, dtype=data_type, device=self._device
        )

        self._draft_tokens_buffer = torch.empty(
            self._max_slot_num, self._max_draft_len, dtype=data_type, device=self._device
        )

        # Stores (first_tokens_len, draft_tokens_len) per slot as a tensor so it
        # gets transferred via RDMA alongside the token data.
        self._token_counts_buffer = torch.zeros(
            self._max_slot_num, 2, dtype=data_type, device=self._device
        )
        self._prompt_token_counts_buffer = torch.zeros(
            self._max_slot_num, 2, dtype=data_type, device=self._device
        )

        self._meta = AuxBufferMeta(
            ptrs=np.array(
                [
                    self._first_tokens_buffer.data_ptr(),
                    self._draft_tokens_buffer.data_ptr(),
                    self._token_counts_buffer.data_ptr(),
                    self._prompt_token_counts_buffer.data_ptr(),
                ],
                dtype=np.int64,
            ),
            size=np.array(
                [
                    self._first_tokens_buffer.numel() * self._first_tokens_buffer.element_size(),
                    self._draft_tokens_buffer.numel() * self._draft_tokens_buffer.element_size(),
                    self._token_counts_buffer.numel() * self._token_counts_buffer.element_size(),
                    self._prompt_token_counts_buffer.numel()
                    * self._prompt_token_counts_buffer.element_size(),
                ],
                dtype=np.int64,
            ),
            item_sizes=np.array(
                [
                    self._first_tokens_buffer[0].numel() * self._first_tokens_buffer.element_size(),
                    self._draft_tokens_buffer[0].numel() * self._draft_tokens_buffer.element_size(),
                    self._token_counts_buffer[0].numel() * self._token_counts_buffer.element_size(),
                    self._prompt_token_counts_buffer[0].numel()
                    * self._prompt_token_counts_buffer.element_size(),
                ],
                dtype=np.int64,
            ),
            device=self._device,
        )

    def alloc_slot(self, request_id: int | None = None) -> AuxSlot:
        if request_id is not None and (
            isinstance(request_id, bool) or not isinstance(request_id, int) or request_id < 0
        ):
            raise ValueError("request_id must be a non-negative integer or None")
        with self._lock:
            if not self._free_slots:
                raise ValueError(
                    f"No free auxiliary buffer slots available "
                    f"(max slots = {self._max_slot_num}). "
                    "All slots are currently occupied."
                )
            slot_id = self._free_slots.popleft()
            if slot_id in self._occupied_slots:
                # This should not happen — defensive check.
                raise RuntimeError(
                    f"Invariant error: selected slot {slot_id} is already marked as occupied. "
                    "This indicates a bug in slot management."
                )
            self._allocation_generation += 1
            identity = AuxAllocationIdentity(
                allocator_domain_id=self._allocator_domain_id,
                request_id=request_id,
                allocation_generation=self._allocation_generation,
            )
            self._occupied_slots.add(slot_id)
            self._slot_identities[slot_id] = identity
            self._slot_token_counts[slot_id] = (0, 0)
            return AuxSlot(slot_id, self, identity)

    def free_slot(
        self,
        slot: int,
        identity: AuxAllocationIdentity | None = None,
    ) -> None:
        if slot < 0 or slot >= self._max_slot_num:
            raise ValueError(
                f"Invalid slot id {slot}. Valid slot indices are in the range 0..{self._max_slot_num - 1}."
            )
        with self._lock:
            if slot not in self._occupied_slots:
                raise ValueError(
                    f"Attempted to free slot {slot}, but that slot is not currently allocated. "
                    "Ensure `alloc_slot` was called and the slot wasn't freed already."
                )
            current_identity = self._slot_identities[slot]
            if identity is not None and identity != current_identity:
                raise ValueError(
                    f"Attempted to free stale auxiliary slot generation for slot {slot}"
                )
            self._occupied_slots.remove(slot)
            self._slot_identities.pop(slot)
            self._slot_token_counts.pop(slot, None)
            self._free_slots.append(slot)

    def allocation_identity(self, slot: int) -> AuxAllocationIdentity:
        """Return the immutable identity of the currently allocated slot."""
        with self._lock:
            try:
                return self._slot_identities[slot]
            except KeyError as error:
                raise ValueError(
                    f"Cannot identify slot {slot}: slot is not currently allocated."
                ) from error

    @property
    def meta(self) -> AuxBufferMeta:
        return self._meta

    def _validate_live_identity_locked(
        self,
        slot: int,
        identity: AuxAllocationIdentity | None,
    ) -> None:
        if slot not in self._occupied_slots:
            raise ValueError(f"Slot {slot} is not currently allocated.")
        if identity is not None and self._slot_identities[slot] != identity:
            raise ValueError(f"Slot {slot} has a different allocation generation.")

    def fill_slot(
        self,
        slot: int,
        request: LlmRequest,
        identity: AuxAllocationIdentity | None = None,
    ) -> None:
        with self._lock:
            try:
                self._validate_live_identity_locked(slot, identity)
            except ValueError as error:
                raise ValueError(
                    f"Cannot fill slot {slot}: {error} Call `alloc_slot` first."
                ) from error
            first_gen_tokens = request.get_last_tokens()
            draft_tokens = request.py_draft_tokens

            if len(first_gen_tokens) > self._beam_width:
                raise ValueError(
                    f"`first_gen_tokens` length ({len(first_gen_tokens)}) exceeds "
                    f"`beam_width` ({self._beam_width}). Consider truncating the token "
                    "list or increasing the beam_width when creating the `AuxBuffer`."
                )
            if len(draft_tokens) > self._max_draft_len:
                raise ValueError(
                    f"`draft_tokens` length ({len(draft_tokens)}) exceeds "
                    f"`max_draft_len` ({self._max_draft_len}). Consider truncating the "
                    "draft tokens or increasing max_draft_len when creating the `AuxBuffer`."
                )

            self._first_tokens_buffer[slot][: len(first_gen_tokens)].copy_(
                torch.tensor(first_gen_tokens, dtype=torch.int32, device=self._device)
            )
            self._draft_tokens_buffer[slot][: len(draft_tokens)].copy_(
                torch.tensor(draft_tokens, dtype=torch.int32, device=self._device)
            )
            self._slot_token_counts[slot] = (len(first_gen_tokens), len(draft_tokens))
            self._token_counts_buffer[slot].copy_(
                torch.tensor(
                    [len(first_gen_tokens), len(draft_tokens)],
                    dtype=torch.int32,
                    device=self._device,
                )
            )
            prompt_tokens, cached_tokens = self._resolve_prompt_token_counts(request)
            self._prompt_token_counts_buffer[slot].copy_(
                torch.tensor(
                    [prompt_tokens, cached_tokens],
                    dtype=torch.int32,
                    device=self._device,
                )
            )

    @staticmethod
    def _resolve_prompt_token_counts(request: LlmRequest) -> tuple[int, int]:
        ctx_usage = (
            request.py_disaggregated_params.ctx_usage
            if request.py_disaggregated_params is not None
            else None
        )
        if ctx_usage is not None:
            prompt_tokens = ctx_usage.get("prompt_tokens", 0)
            details = ctx_usage.get("prompt_tokens_details") or {}
            cached_tokens = details.get("cached_tokens", 0)
        else:
            prompt_tokens = request.prompt_len
            cached_tokens = request.cached_tokens
        return int(prompt_tokens or 0), int(cached_tokens or 0)

    def get_slot_tokens(self, slot: int) -> tuple[list[int], list[int]]:
        with self._lock:
            if slot not in self._occupied_slots:
                raise ValueError(f"Cannot read slot {slot}: slot is not currently allocated.")
            first_len, draft_len = self._token_counts_buffer[slot].tolist()
            first_gen_tokens = self._first_tokens_buffer[slot][:first_len].tolist()
            draft_tokens = self._draft_tokens_buffer[slot][:draft_len].tolist()

            return first_gen_tokens, draft_tokens

    def get_slot_data(
        self,
        slot: int,
        identity: AuxAllocationIdentity | None = None,
    ) -> tuple[list[int], list[int], tuple[int, int]]:
        with self._lock:
            self._validate_live_identity_locked(slot, identity)
            first_gen_tokens, draft_tokens = self.get_slot_tokens(slot)
            prompt_tokens, cached_tokens = self._prompt_token_counts_buffer[slot].tolist()
            return first_gen_tokens, draft_tokens, (int(prompt_tokens), int(cached_tokens))
