# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections import deque, namedtuple
from dataclasses import dataclass, field
from typing import Any

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


@dataclass(frozen=True)
class AuxTransferLayout:
    """Static auxiliary memory layout shared by all transfers to one peer."""

    src_base_ptrs: np.ndarray
    dst_base_ptrs: np.ndarray
    src_item_sizes: np.ndarray
    dst_item_sizes: np.ndarray


def _readonly(array: np.ndarray) -> np.ndarray:
    """Return an array protected from accidental in-place updates."""
    array.flags.writeable = False
    return array


def get_non_empty_aux_indices(ptrs: np.ndarray, sizes: np.ndarray, context: str) -> np.ndarray:
    """Validate auxiliary memory descriptors and return their non-empty indices."""
    if ptrs.shape != sizes.shape:
        raise ValueError(f"{context}: pointer/size count mismatch: {ptrs.shape=} != {sizes.shape=}")

    negative_sizes = sizes < 0
    if negative_sizes.any():
        indices = np.flatnonzero(negative_sizes).tolist()
        raise ValueError(f"{context}: negative sizes at indices {indices}")

    null_non_empty = (ptrs == 0) & (sizes > 0)
    if null_non_empty.any():
        indices = np.flatnonzero(null_non_empty).tolist()
        raise ValueError(f"{context}: null pointers with non-zero sizes at indices {indices}")

    return np.flatnonzero(sizes > 0)


def build_aux_transfer_layout(
    src_meta: AuxBufferMeta, dst_meta: AuxBufferMeta
) -> AuxTransferLayout:
    """Validate and build the static auxiliary transfer layout for one peer."""
    src_item_sizes = src_meta.item_sizes.astype(np.int64, copy=False)
    dst_item_sizes = dst_meta.item_sizes.astype(np.int64, copy=False)
    src_indices = get_non_empty_aux_indices(
        src_meta.ptrs, src_item_sizes, "source auxiliary transfer"
    )
    dst_indices = get_non_empty_aux_indices(
        dst_meta.ptrs, dst_item_sizes, "destination auxiliary transfer"
    )
    if src_meta.ptrs.shape != dst_meta.ptrs.shape:
        raise ValueError(
            "Source and destination auxiliary layouts do not match: "
            f"{src_meta.ptrs.shape=} != {dst_meta.ptrs.shape=}"
        )

    dst_non_empty = np.zeros(dst_meta.ptrs.shape, dtype=bool)
    dst_non_empty[dst_indices] = True
    missing_dst = src_indices[~dst_non_empty[src_indices]]
    if missing_dst.size > 0:
        raise ValueError(
            "Destination auxiliary buffers are empty for non-empty source "
            f"indices {missing_dst.tolist()}"
        )

    too_small = src_indices[dst_item_sizes[src_indices] < src_item_sizes[src_indices]]
    if too_small.size > 0:
        raise ValueError(
            f"Destination auxiliary buffers are too small at indices {too_small.tolist()}"
        )

    return AuxTransferLayout(
        src_base_ptrs=_readonly(src_meta.ptrs[src_indices]),
        dst_base_ptrs=_readonly(dst_meta.ptrs[src_indices]),
        src_item_sizes=_readonly(src_item_sizes[src_indices]),
        dst_item_sizes=_readonly(dst_item_sizes[src_indices]),
    )


AuxSlot = namedtuple("AuxSlot", ["id", "buffer"])

_DRAFT_HISTORY_VERSION = 1
_DRAFT_HISTORY_FIELDS = 8
_DRAFT_DTYPE_CODES = {"torch.float16": 1, "torch.bfloat16": 2}
_DRAFT_BACKEND_CODES = {"VANILLA": 1, "TRTLLM": 2}


def _encode_draft_history(history: dict[str, Any]) -> list[int]:
    """Encode committed history and rank-local storage identity for the wire."""
    if not isinstance(history, dict):
        raise ValueError("Standalone draft transfer requires draft history metadata")
    layout = history.get("layout")
    if not isinstance(layout, dict):
        raise ValueError("Standalone draft transfer requires a storage layout")
    integer_values = [
        history.get("valid_length"),
        history.get("position"),
        layout.get("num_layers"),
        layout.get("num_kv_heads"),
        layout.get("head_dim"),
    ]
    if any(type(value) is not int for value in integer_values):
        raise ValueError(
            "Standalone draft transfer metadata requires integer lengths and dimensions"
        )
    valid_length, position, num_layers, num_kv_heads, head_dim = integer_values
    if valid_length < 0 or position < valid_length:
        raise ValueError("Invalid standalone draft history length or position")
    if min(num_layers, num_kv_heads, head_dim) <= 0:
        raise ValueError("Standalone draft transfer dimensions must be positive")
    dtype_code = _DRAFT_DTYPE_CODES.get(layout.get("dtype"))
    backend_code = _DRAFT_BACKEND_CODES.get(layout.get("attention_backend"))
    if dtype_code is None or backend_code is None:
        raise ValueError("Unsupported standalone draft transfer dtype or attention backend")
    return [_DRAFT_HISTORY_VERSION, *integer_values, dtype_code, backend_code]


def _decode_draft_history(values: list[int]) -> dict[str, Any]:
    if len(values) != _DRAFT_HISTORY_FIELDS or values[0] != _DRAFT_HISTORY_VERSION:
        raise ValueError("Missing or unsupported standalone draft history metadata version")
    _, valid_length, position, num_layers, num_kv_heads, head_dim, dtype_code, backend_code = values
    dtypes = {code: name for name, code in _DRAFT_DTYPE_CODES.items()}
    backends = {code: name for name, code in _DRAFT_BACKEND_CODES.items()}
    history = {
        "valid_length": valid_length,
        "position": position,
        "layout": {
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "dtype": dtypes.get(dtype_code),
            "attention_backend": backends.get(backend_code),
        },
    }
    _encode_draft_history(history)
    return history


class AuxBufferBase(ABC):
    """
    Abstract base class defining the interface for auxiliary buffer management.
    """

    @abstractmethod
    def alloc_slot(self) -> AuxSlot:
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

    @abstractmethod
    def get_slot_data(self, slot: int) -> tuple[list[int], list[int], tuple[int, int]]:
        """
        Get the token data and prompt token counts from the specified slot.

        Returns:
            (first_gen_tokens, draft_tokens, (prompt_tokens, cached_tokens))
        """
        ...


class AuxBuffer(AuxBufferBase):
    def __init__(
        self,
        max_slot_num: int,
        beam_width: int,
        max_draft_len: int,
        device: str = "cpu",
        *,
        draft_history: bool = False,
    ) -> None:
        self._max_slot_num = int(max_slot_num)
        self._beam_width = int(beam_width)
        self._max_draft_len = int(max_draft_len)
        self._device = device

        self._free_slots = deque(list(range(self._max_slot_num)))
        self._occupied_slots: set[int] = set()
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
        # This participates in the existing auxiliary memory registration and
        # transfer. Version zero denotes an unfilled or newly allocated slot.
        self._draft_history_buffer = (
            torch.zeros(
                self._max_slot_num, _DRAFT_HISTORY_FIELDS, dtype=torch.int64, device=self._device
            )
            if draft_history
            else None
        )

        buffers = [
            self._first_tokens_buffer,
            self._draft_tokens_buffer,
            self._token_counts_buffer,
            self._prompt_token_counts_buffer,
        ]
        if self._draft_history_buffer is not None:
            buffers.append(self._draft_history_buffer)

        self._meta = AuxBufferMeta(
            ptrs=np.array([buffer.data_ptr() for buffer in buffers], dtype=np.int64),
            size=np.array(
                [buffer.numel() * buffer.element_size() for buffer in buffers],
                dtype=np.int64,
            ),
            item_sizes=np.array(
                [buffer[0].numel() * buffer.element_size() for buffer in buffers],
                dtype=np.int64,
            ),
            device=self._device,
        )

    def alloc_slot(self) -> AuxSlot:
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
        self._slot_token_counts[slot_id] = (0, 0)
        if self._draft_history_buffer is not None:
            self._draft_history_buffer[slot_id].zero_()
        return AuxSlot(slot_id, self)

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
        self._slot_token_counts.pop(slot, None)
        self._free_slots.append(slot)

    @property
    def meta(self) -> AuxBufferMeta:
        return self._meta

    @property
    def has_draft_history(self) -> bool:
        """Whether this buffer transfers standalone drafter history metadata."""
        return self._draft_history_buffer is not None

    def fill_slot(self, slot: int, request: LlmRequest) -> None:
        if slot not in self._occupied_slots:
            raise ValueError(
                f"Cannot fill slot {slot}: slot is not currently allocated. "
                "Call `alloc_slot` first."
            )
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
        self._slot_token_counts[slot] = (len(first_gen_tokens), len(draft_tokens))
        self._token_counts_buffer[slot].copy_(
            torch.tensor(
                [len(first_gen_tokens), len(draft_tokens)], dtype=torch.int32, device=self._device
            )
        )
        prompt_tokens, cached_tokens = self._resolve_prompt_token_counts(request)
        self._prompt_token_counts_buffer[slot].copy_(
            torch.tensor([prompt_tokens, cached_tokens], dtype=torch.int32, device=self._device)
        )
        if self._draft_history_buffer is not None:
            values = _encode_draft_history(request.py_draft_transfer_history)
            self._draft_history_buffer[slot].copy_(
                torch.tensor(values, dtype=torch.int64, device=self._device)
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
        if slot not in self._occupied_slots:
            raise ValueError(f"Cannot read slot {slot}: slot is not currently allocated.")
        first_len, draft_len = self._token_counts_buffer[slot].tolist()
        first_gen_tokens = self._first_tokens_buffer[slot][:first_len].tolist()
        draft_tokens = self._draft_tokens_buffer[slot][:draft_len].tolist()

        return first_gen_tokens, draft_tokens

    def get_slot_data(self, slot: int) -> tuple[list[int], list[int], tuple[int, int]]:
        first_gen_tokens, draft_tokens = self.get_slot_tokens(slot)
        prompt_tokens, cached_tokens = self._prompt_token_counts_buffer[slot].tolist()
        return first_gen_tokens, draft_tokens, (int(prompt_tokens), int(cached_tokens))

    def get_slot_draft_history(self, slot: int) -> dict[str, Any]:
        """Read transferred history, rejecting missing or unsupported metadata."""
        if slot not in self._occupied_slots:
            raise ValueError(f"Cannot read slot {slot}: slot is not currently allocated.")
        if self._draft_history_buffer is None:
            raise ValueError("Standalone draft history transfer is not enabled for this buffer")
        return _decode_draft_history(self._draft_history_buffer[slot].tolist())
