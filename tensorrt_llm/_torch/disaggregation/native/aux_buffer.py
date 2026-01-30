from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest


@dataclass
class AuxBufferMeta:
    ptrs: list[int]
    size: list[int]
    item_sizes: list[int] = field(default_factory=list)
    device: str = "cpu"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ptrs": self.ptrs,
            "size": self.size,
            "item_sizes": self.item_sizes,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuxBufferMeta":
        return cls(
            ptrs=data["ptrs"],
            size=data["size"],
            item_sizes=data.get("item_sizes", []),
            device=data.get("device", "cpu"),
        )


class AuxBufferBase(ABC):
    """
    Abstract base class defining the interface for auxiliary buffer management.
    """

    @abstractmethod
    def alloc_slot(self) -> int:
        """
        Allocate a free slot and return its index.
        """
        ...

    @abstractmethod
    def free_slot(self, slot: int) -> None:
        """
        Release the specified slot.
        """
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
    def fill_slot(self, slot: int, request: LlmRequest) -> None:
        """
        Fill/overwrite the contents of the given slot with data from the request.
        """
        ...

    @abstractmethod
    def get_slot_tokens(self, slot: int) -> tuple[list[int], list[int]]:
        """
        Get the token data (e.g., first/draft tokens) from the specified slot.
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

        data_type = torch.int32
        self._first_tokens_buffer = torch.empty(
            self._max_slot_num, self._beam_width, dtype=data_type, device=self._device
        )

        self._draft_tokens_buffer = torch.empty(
            self._max_slot_num, self._max_draft_len, dtype=data_type, device=self._device
        )

        self._meta = AuxBufferMeta(
            ptrs=[self._first_tokens_buffer.data_ptr(), self._draft_tokens_buffer.data_ptr()],
            size=[
                self._first_tokens_buffer.numel() * self._first_tokens_buffer.element_size(),
                self._draft_tokens_buffer.numel() * self._draft_tokens_buffer.element_size(),
            ],
            item_sizes=[
                self._first_tokens_buffer[0].numel() * self._first_tokens_buffer.element_size(),
                self._draft_tokens_buffer[0].numel() * self._draft_tokens_buffer.element_size(),
            ],
            device=self._device,
        )

    def alloc_slot(self) -> int:
        if not self._free_slots:
            raise ValueError(
                f"No free auxiliary buffer slots available (max slots = {self._max_slot_num}). "
                "All slots are currently occupied."
            )
        slot_id = self._free_slots.popleft()
        if slot_id in self._occupied_slots:
            # This should not happen — defensive check.
            raise RuntimeError(
                f"Invariant error: selected slot {slot_id} is already marked as occupied. "
                "This indicates a bug in slot management."
            )
        self._occupied_slots.add(slot_id)
        return slot_id

    def free_slot(self, slot: int) -> None:
        if slot not in self._occupied_slots:
            raise ValueError(
                f"Attempted to free slot {slot}, but that slot is not currently allocated. "
                "Ensure `alloc_slot` was called and the slot wasn't freed already."
            )
        if slot < 0 or slot >= self._max_slot_num:
            raise ValueError(
                f"Invalid slot id {slot}. Valid slot indices are in the range 0..{self._max_slot_num - 1}."
            )
        self._occupied_slots.remove(slot)
        self._free_slots.append(slot)

    @property
    def meta(self) -> AuxBufferMeta:
        return self._meta

    def fill_slot(self, slot: int, request: LlmRequest) -> None:
        first_gen_tokens = request.get_last_tokens()
        draft_tokens = request.py_draft_tokens

        if len(first_gen_tokens) > self._beam_width:
            raise ValueError(
                f"`first_gen_tokens` length ({len(first_gen_tokens)}) exceeds `beam_width` ({self._beam_width}). "
                "Consider truncating the token list or increasing the beam_width when creating the `AuxBuffer`."
            )
        if len(draft_tokens) > self._max_draft_len:
            raise ValueError(
                f"`draft_tokens` length ({len(draft_tokens)}) exceeds `max_draft_len` ({self._max_draft_len}). "
                "Consider truncating draft tokens or increasing `max_draft_len` when creating the `AuxBuffer`."
            )

        self._first_tokens_buffer[slot][: len(first_gen_tokens)].copy_(
            torch.tensor(first_gen_tokens, dtype=torch.int32, device=self._device)
        )
        self._draft_tokens_buffer[slot][: len(draft_tokens)].copy_(
            torch.tensor(draft_tokens, dtype=torch.int32, device=self._device)
        )

    def get_slot_tokens(self, slot: int) -> tuple[list[int], list[int]]:
        first_gen_tokens = self._first_tokens_buffer[slot].tolist()
        draft_tokens = self._draft_tokens_buffer[slot].tolist()

        return first_gen_tokens, draft_tokens
