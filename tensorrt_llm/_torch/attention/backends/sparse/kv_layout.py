# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model entry formats; physical placement belongs to KVCM and the cold-page codec."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import BufferId


@dataclass(frozen=True)
class EntryComponent:
    """A contiguous tensor within one model buffer, such as KV or quantization scales.

    Components appear in storage order. shape and entry_axis describe the model's
    tensor format (for example NHD or HND), independent of pool placement.
    """

    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    entry_axis: int

    def __post_init__(self) -> None:
        if not self.name or not self.shape or min(self.shape) <= 0:
            raise ValueError("Components need a name and positive tensor dimensions")
        if not isinstance(self.dtype, torch.dtype) or not 0 <= self.entry_axis < len(self.shape):
            raise ValueError("Components need a dtype and a valid entry axis")


@dataclass(frozen=True)
class EntryFormat:
    """Model-only format for one BufferId (layer_id, DataRole).

    tokens_per_entry describes model compression; 1 means ordinary token entries.
    Components describe one native model buffer. KVCM supplies any expansion into
    several native buffers per logical page, and the codec supplies their location.
    There are no pool IDs, lifecycle IDs, slot sizes, or coalesced-buffer offsets here.
    """

    buffer_id: BufferId
    components: tuple[EntryComponent, ...]
    tokens_per_entry: int = 1

    def __post_init__(self) -> None:
        layer_id, role = self.buffer_id
        if layer_id < 0 or not role or self.tokens_per_entry <= 0:
            raise ValueError("Format needs a valid BufferId and positive tokens_per_entry")
        if not self.components or len({c.name for c in self.components}) != len(self.components):
            raise ValueError("Format needs components with unique names")
        if len({c.shape[c.entry_axis] for c in self.components}) != 1:
            raise ValueError("Components must describe the same number of entries")
