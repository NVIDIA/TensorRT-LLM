# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storage and history contracts for manager-owned DSpark/DFlash drafters."""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .kv_cache_manager_v2 import KVCacheManagerV2


@dataclass(frozen=True)
class StandaloneDraftHistory:
    """Committed length and next absolute position, independent of target/scratch state."""

    valid_length: int
    position: int

    def __post_init__(self) -> None:
        if type(self.valid_length) is not int or type(self.position) is not int:
            raise ValueError("Standalone draft history requires integer length and position")
        if self.valid_length < 0 or self.position < self.valid_length:
            raise ValueError("Invalid standalone draft history length or position")


#: Opt-in (default unset), with ``TLLM_DSPARK_DRAFT_CONTEXT_LIMIT``: a prompt above the limit is
#: drafted from the most recent tokens of its own transferred context instead of an empty one
#: (see :func:`draft_context_window_from_env`).
DRAFT_CONTEXT_WINDOW_ENV = "TLLM_DSPARK_DRAFT_CONTEXT_WINDOW"
#: Opt-in (default 0), with ``TLLM_DSPARK_DRAFT_CONTEXT_WINDOW``: the leading prompt tokens the
#: window keeps in front of the most recent ones.
DRAFT_CONTEXT_SINK_ENV = "TLLM_DSPARK_DRAFT_CONTEXT_SINK"


@dataclass(frozen=True)
class DraftContextWindow:
    """The part of a long prompt's transferred drafter context a generation worker drafts from."""

    #: Leading prompt tokens kept at their own positions.
    sink: int
    #: Most recent prompt tokens kept, placed right after the sink.
    recent: int

    @property
    def length(self) -> int:
        return self.sink + self.recent


def _env_token_count(name: str, raw: str, minimum: int) -> int:
    try:
        value = int(raw)
    except ValueError:
        value = minimum - 1
    if value < minimum:
        kind = "a positive" if minimum > 0 else "a non-negative"
        raise ValueError(f"{name} must be {kind} integer, got {raw!r}")
    return value


def draft_context_window_from_env() -> DraftContextWindow | None:
    """The drafter context of a prompt above ``TLLM_DSPARK_DRAFT_CONTEXT_LIMIT``, if not empty.

    Without it a generation worker drafts a prompt above the limit from an empty context. With
    ``TLLM_DSPARK_DRAFT_CONTEXT_WINDOW=W`` it drafts from the last ``W`` tokens of the prompt's own
    transferred context, preceded by its first ``TLLM_DSPARK_DRAFT_CONTEXT_SINK`` tokens (default
    0). The kept latents are moved to the start of the request's draft pages and their RoPE
    positions are rebased to their new places, so the drafter sees them as one context of
    ``sink + W`` tokens and every distance it attends over stays within the range it drafts well
    over. Only the generation worker acts on these variables, and only for prompts above the limit.

    Returns:
        The window, or None when ``TLLM_DSPARK_DRAFT_CONTEXT_WINDOW`` is unset, which keeps the
        empty context of the limit alone.
    """
    raw = os.environ.get(DRAFT_CONTEXT_WINDOW_ENV, "").strip()
    raw_sink = os.environ.get(DRAFT_CONTEXT_SINK_ENV, "").strip()
    if not raw:
        if raw_sink:
            raise ValueError(f"{DRAFT_CONTEXT_SINK_ENV} needs {DRAFT_CONTEXT_WINDOW_ENV}")
        return None
    recent = _env_token_count(DRAFT_CONTEXT_WINDOW_ENV, raw, 1)
    sink = _env_token_count(DRAFT_CONTEXT_SINK_ENV, raw_sink, 0) if raw_sink else 0
    return DraftContextWindow(sink=sink, recent=recent)


@dataclass(frozen=True)
class DraftContextWindowMove:
    """Draft context positions a generation worker relocates before it first drafts a request.

    The worker copies the request's own context latents ``[source_start, source_start + length)``
    to ``[dest_start, dest_start + length)`` of the same request's draft pages and rotates their
    RoPE slice back by ``source_start - dest_start`` positions, so the drafter attends to them at
    the start of its context (see ``TLLM_DSPARK_DRAFT_CONTEXT_WINDOW``).
    """

    source_start: int
    dest_start: int
    length: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int for value in (self.source_start, self.dest_start, self.length)
        ):
            raise ValueError("A draft context window move requires integer positions")
        if self.length <= 0 or self.dest_start < 0 or self.source_start < self.dest_start:
            raise ValueError("Invalid draft context window move")

    @property
    def shift(self) -> int:
        """Positions the moved latents travel towards the start of the context."""
        return self.source_start - self.dest_start


@dataclass(frozen=True)
class DraftHistoryUpdate:
    """One execution's draft history, published after its sampling event completes."""

    manager: "KVCacheManagerV2"
    request_ids: tuple[int, ...]
    cache_instances: tuple[object, ...]
    values_host: torch.Tensor

    @classmethod
    def capture(
        cls,
        manager: "KVCacheManagerV2",
        request_ids: tuple[int, ...],
        values: torch.Tensor,
    ) -> "DraftHistoryUpdate":
        """Queue an independent length/position readback on the execution stream."""
        if values.shape != (len(request_ids), 2):
            raise ValueError("Draft history update requires one length/position pair per request")
        values_host = torch.empty(
            values.shape, dtype=values.dtype, device="cpu", pin_memory=values.is_cuda
        )
        values_host.copy_(values, non_blocking=True)
        return cls(
            manager,
            tuple(request_ids),
            tuple(manager.kv_cache_map[request_id] for request_id in request_ids),
            values_host,
        )

    def publish(self) -> None:
        """Publish completed writes without reviving released or replaced requests."""
        for request_id, cache, (length, position) in zip(
            self.request_ids, self.cache_instances, self.values_host.tolist()
        ):
            current = self.manager.kv_cache_map.get(request_id)
            if current is cache and current.is_active:
                self.manager.set_draft_history(request_id, length, position)
