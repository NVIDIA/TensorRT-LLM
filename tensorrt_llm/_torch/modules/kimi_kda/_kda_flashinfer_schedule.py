# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Switch and per-step schedule of the opt-in FlashInfer KDA prefill.

Kept apart from ``_kda_flashinfer_prefill.py`` so the K3 Mamba metadata can read the switch
without importing the prefill path. ``TLLM_KDA_FLASHINFER_PREFILL`` is ``0`` (default, the
in-tree prefill) or ``1`` (FlashInfer prefill); any other value is an error.
"""

from __future__ import annotations

import os
from itertools import accumulate

import torch

from tensorrt_llm._utils import prefer_pinned

FLASHINFER_PREFILL_ENV = "TLLM_KDA_FLASHINFER_PREFILL"


def flashinfer_prefill_requested() -> bool:
    """Whether ``TLLM_KDA_FLASHINFER_PREFILL`` asks for the FlashInfer prefill (default ``0``)."""
    value = os.environ.get(FLASHINFER_PREFILL_ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{FLASHINFER_PREFILL_ENV} must be 0 or 1, got {value!r}")
    return value == "1"


class KDAFlashInferSchedule:
    """Host-prepared scheduling metadata for packed FlashInfer KDA prefill.

    One transfer per step refreshes what all KDA layers share: ``kda_max_seqlen``, the
    stable descending-length order ``kda_seq_order`` and ``kda_cu_chunks`` (prefix sums
    of ``ceil(length / 16)`` in request order). The device storage stays fixed for CUDA
    graph replay.
    """

    def __init__(self, max_batch_size: int) -> None:
        # Align both rows to 16 bytes.
        width = ((max_batch_size + 1 + 3) // 4) * 4
        self._schedule = torch.zeros((2, width), dtype=torch.int32, device="cuda")

    def prepare(self, metadata, context_lengths: list[int]) -> None:
        """Set the three ``kda_*`` attributes of ``metadata`` for these context lengths."""
        n = len(context_lengths)
        metadata.kda_max_seqlen = max(context_lengths, default=0)
        metadata.kda_seq_order = self._schedule[0, :n]
        metadata.kda_cu_chunks = self._schedule[1, : n + 1]
        if not n:
            return
        order = sorted(range(n), key=lambda i: (-context_lengths[i], i))
        chunks = list(accumulate(((length + 15) // 16 for length in context_lengths), initial=0))
        width = self._schedule.shape[1]
        # Use a fresh pinned source so another prepare cannot overwrite an
        # in-flight transfer. The pinned allocator retains it until DMA ends.
        host_schedule = torch.tensor(
            [order + [0] * (width - n), chunks + [0] * (width - n - 1)],
            dtype=torch.int32,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        self._schedule.copy_(host_schedule, non_blocking=True)
