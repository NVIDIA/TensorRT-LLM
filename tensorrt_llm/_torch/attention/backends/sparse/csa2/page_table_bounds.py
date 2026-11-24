# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from weakref import ref

import numpy as np
import torch


class PageTableBounds:
    """Track the columns past which each destination row holds BAD_PAGE_INDEX.

    The metadata bank owns disjoint buffers of fixed shape. Manager destinations
    may be overlapping views and must be claimed before writing.
    """

    def __init__(self, *, track_overlaps: bool = False) -> None:
        self._track_overlaps = track_overlaps
        self._entries = {}

    @staticmethod
    def _address_range(tensor: torch.Tensor) -> tuple[int, int]:
        extent = 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))
        start = tensor.data_ptr()
        return start, start + (extent if tensor.numel() else 0) * tensor.element_size()

    def get(
        self, target: torch.Tensor, address_range: tuple[int, int] | None = None
    ) -> np.ndarray | None:
        entry = self._entries.get(id(target))
        if entry is not None and entry[0]() is target and entry[1] == address_range:
            return entry[2]
        return None

    def claim(self, target: torch.Tensor) -> np.ndarray | None:
        """Take the previous bound and invalidate entries that this write may affect."""
        address_range = self._address_range(target)
        live = self.get(target, address_range)
        first, last = address_range
        for key, (tensor_ref, extent, _) in list(self._entries.items()):
            if tensor_ref() is None or (
                extent is not None and extent[0] < last and first < extent[1]
            ):
                del self._entries[key]
        return live

    def record(self, target: torch.Tensor, live: np.ndarray | None) -> None:
        if live is None:
            self.forget(target)
            return
        extent = self._address_range(target) if self._track_overlaps else None
        self._entries[id(target)] = (ref(target), extent, np.array(live, dtype=np.int64))

    def forget(self, target: torch.Tensor) -> None:
        self._entries.pop(id(target), None)

    def prune(self) -> None:
        for key in [
            key for key, (tensor_ref, _, _) in self._entries.items() if tensor_ref() is None
        ]:
            del self._entries[key]
