# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph-safe minimum-length termination bans for standalone DSpark verification."""

from __future__ import annotations

import os
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig

    from ..pyexecutor.llm_request import LlmRequest
    from .interface import SpecMetadata

# Bound the per-row save/restore workspace independently of the vocabulary size.
MAX_TERMINATION_IDS = 64


def supports_config(config: DecodingBaseConfig | None) -> bool:
    """Whether this deployment uses standalone DSpark's linear worker verification."""
    return bool(
        config is not None
        and config.spec_dec_mode.is_dspark()
        and not config.draft_is_embedded_in_target
        and float(os.environ.get("TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS", "0")) == 0
    )


def enabled(metadata: SpecMetadata) -> bool:
    """Whether this metadata owns the supported minimum-length implementation."""
    return bool(
        getattr(metadata, "supports_min_tokens", False) and metadata.spec_dec_mode.is_dspark()
    )


def termination_ids(request: LlmRequest) -> tuple[int, ...]:
    """Original EOS plus single-token stop words, matching TorchSampler's bans.

    Multi-token stop words retain the ordinary finish-handler semantics.
    """
    eos = getattr(request, "py_original_end_id", request.py_end_id)
    ids = {int(eos)} if eos is not None and int(eos) >= 0 else set()
    ids.update(int(word[0]) for word in request.py_stop_words_list or () if len(word) == 1)
    if any(token < 0 for token in ids):
        raise ValueError("DSpark minimum length requires nonnegative stop token IDs")
    if len(ids) > MAX_TERMINATION_IDS:
        raise ValueError(
            f"DSpark minimum length supports at most {MAX_TERMINATION_IDS} unique EOS and "
            "single-token stop IDs per request"
        )
    return tuple(sorted(ids))


def minimum_remaining(minimum: int, generated: int) -> int:
    """Remaining EOS-banned positions, counting generated tokens rather than the prompt."""
    if minimum < 0:
        raise ValueError("minimum length must be nonnegative")
    return max(int(minimum) - max(int(generated), 0), 0)


def request_slot(request: LlmRequest, scratch: int) -> int:
    """Route anonymous and recognized graph dummies to the neutral scratch slot."""
    if request.py_seq_slot is None or getattr(request, "is_dummy", False):
        return scratch
    return request.py_seq_slot


@dataclass
class MinimumLengthState:
    """Device progress shared across eager steps and graph buckets.

    ``remaining``/``commit`` have shape ``[slots + 1]`` and ``stop_ids`` has
    shape ``[slots + 1, MAX_TERMINATION_IDS]``. The last slot is neutral scratch.
    Weak host ownership prevents stale progress after slot or numeric-ID reuse.
    """

    remaining: torch.Tensor
    stop_ids: torch.Tensor
    commit: torch.Tensor
    owners: list[weakref.ReferenceType[LlmRequest] | None] = field(default_factory=list)
    parameters: list[tuple[tuple[int | None, int], int, tuple[int, ...]] | None] = field(
        default_factory=list
    )
    commit_values: list[bool] = field(default_factory=list)

    @property
    def slots(self) -> int:
        return self.remaining.shape[0] - 1

    @classmethod
    def create(cls, slots: int, device: torch.device | str = "cuda") -> MinimumLengthState:
        """Allocate neutral state once before graph views copy metadata."""
        if slots <= 0:
            raise ValueError("minimum-length storage needs positive slot capacity")
        return cls(
            torch.zeros(slots + 1, dtype=torch.int64, device=device),
            torch.full((slots + 1, MAX_TERMINATION_IDS), -1, dtype=torch.int64, device=device),
            torch.zeros(slots + 1, dtype=torch.bool, device=device),
            [None] * slots,
            [None] * slots,
            [False] * slots,
        )


def prepare(metadata: SpecMetadata) -> None:
    """Prepare shared progress and the existing request-to-slot map."""
    if not enabled(metadata):
        return
    capacity = metadata.num_seq_slots or metadata.max_num_requests
    if capacity <= 0:
        raise RuntimeError("DSpark minimum-length state has no sequence slots")
    if metadata.minimum_length_state is None:
        metadata.minimum_length_state = MinimumLengthState.create(capacity)
    state = metadata.minimum_length_state
    if capacity > state.slots:
        raise RuntimeError("minimum-length storage cannot grow after allocation")
    metadata.dummy_slot_row = state.slots
    if metadata.batch_slot_ids is None:
        metadata.batch_slot_ids = torch.empty(
            metadata.max_num_requests, dtype=torch.long, device=state.remaining.device
        )
    if metadata.batch_slot_ids.numel() < metadata.max_num_requests:
        raise RuntimeError("minimum-length batch slot map is undersized")


def _device(
    values: list[int] | list[bool] | list[list[int]], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    if device.type == "cuda":
        from tensorrt_llm._utils import prefer_pinned

        return torch.tensor(values, dtype=dtype, pin_memory=prefer_pinned()).to(
            device, non_blocking=True
        )
    return torch.tensor(values, dtype=dtype, device=device)


def populate(metadata: SpecMetadata, requests: list[LlmRequest]) -> None:
    """Upload new owners' parameters without overwriting accepted device progress."""
    if not enabled(metadata):
        return
    state = metadata.minimum_length_state
    if state is None:
        raise RuntimeError("DSpark minimum-length buffers were not prepared")
    slots, commit_slots, commits, resets, remaining, stops, owners = [], [], [], [], [], [], []
    for request in requests:
        if request.py_seq_slot is None or getattr(request, "is_dummy", False):
            continue
        slot = request.py_seq_slot
        if not 0 <= slot < state.slots or slot in slots:
            raise ValueError("invalid or duplicate minimum-length sequence slot")
        owner = (request.py_client_id, request.py_request_id)
        if owner[1] is None:
            raise ValueError("minimum-length state needs a unique request identity")
        minimum = int(request.py_min_length or 0)
        ids = termination_ids(request) if minimum else ()
        if any(token >= metadata.vocab_size for token in ids):
            raise ValueError("minimum-length stop token is outside the target vocabulary")
        if minimum and len(ids) == metadata.vocab_size:
            raise ValueError("minimum-length stop tokens cannot cover the target vocabulary")
        generated = int(request.get_num_tokens(0)) - int(request.py_orig_prompt_len)
        left = minimum_remaining(minimum, generated)
        commit = not (request.is_context_init_state and not request.is_last_context_chunk)
        slots.append(slot)
        if state.commit_values[slot] != commit:
            commit_slots.append(slot)
            commits.append(commit)
        previous = state.owners[slot]
        parameters = (owner, minimum, ids)
        if previous is None or previous() is not request or state.parameters[slot][0] != owner:
            resets.append(slot)
            remaining.append(left)
            stops.append([*ids, *([-1] * (MAX_TERMINATION_IDS - len(ids)))])
            owners.append((slot, weakref.ref(request), parameters))
        elif state.parameters[slot] != parameters:
            raise ValueError("minimum-length parameters changed within a request")
    device = state.remaining.device
    if resets:
        indices = _device(resets, torch.long, device)
        state.remaining.index_copy_(0, indices, _device(remaining, torch.int64, device))
        state.stop_ids.index_copy_(0, indices, _device(stops, torch.int64, device))
    if commit_slots:
        state.commit.index_copy_(
            0, _device(commit_slots, torch.long, device), _device(commits, torch.bool, device)
        )
        for slot, value in zip(commit_slots, commits):
            state.commit_values[slot] = value
    for slot, owner, parameters in owners:
        state.owners[slot] = owner
        state.parameters[slot] = parameters


def apply(
    logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    num_contexts: int,
    batch_size: int,
    metadata: SpecMetadata,
    *,
    vocab_offset: int = 0,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    """Mask termination scores before target consumers, saving bounded restoration data.

    ``logits`` is ``[contexts + verification rows, local_vocab]``; proposals are
    ``[generations, draft_len]``. ``vocab_offset`` identifies a vocabulary shard.
    All data-dependent decisions stay on device and use static output shapes.
    """
    if not enabled(metadata):
        return logits, None
    state = metadata.minimum_length_state
    if state is None or metadata.batch_slot_ids is None:
        raise RuntimeError("DSpark minimum-length metadata is missing")
    if logits.dim() != 2 or not logits.is_floating_point() or logits.shape[1] <= 0:
        raise RuntimeError("DSpark minimum length requires floating-point target scores")
    num_gens = batch_size - num_contexts
    if (
        not 0 <= num_contexts <= batch_size
        or draft_tokens.dim() != 2
        or draft_tokens.shape[0] != num_gens
        or vocab_offset < 0
    ):
        raise RuntimeError("minimum-length request layout differs")
    slots = metadata.batch_slot_ids[:batch_size]
    if metadata.is_ragged_verify:
        lens, starts, total = metadata.verify_lens, metadata.qo_indptr, metadata.total_verify_tokens
        if lens is None or starts is None or total is None:
            raise RuntimeError("minimum-length ragged metadata is incomplete")
        if lens.numel() != num_gens or starts.numel() != num_gens + 1:
            raise RuntimeError("minimum-length ragged metadata shape differs")
        request_ids = torch.repeat_interleave(
            torch.arange(num_gens, device=logits.device),
            lens.to(torch.long),
            output_size=int(total),
        )
        positions = torch.arange(int(total), device=logits.device) - starts[request_ids]
    else:
        width = draft_tokens.shape[1] + 1
        flat = torch.arange(num_gens * width, device=logits.device)
        request_ids, positions = flat // width, flat % width
    row_slots = torch.cat((slots[:num_contexts], slots[num_contexts:][request_ids]))
    positions = torch.cat(
        (torch.zeros(num_contexts, dtype=torch.long, device=logits.device), positions)
    )
    if row_slots.numel() != logits.shape[0]:
        raise RuntimeError("minimum-length target row count differs")
    columns = state.stop_ids[row_slots] - vocab_offset
    local = (columns >= 0) & (columns < logits.shape[1])
    # Invalid/padded IDs share an owned column, so duplicate writes always store
    # identical values and cannot undo a real ban on local column zero.
    first = torch.where(local, columns, logits.shape[1]).amin(dim=1)
    safe = torch.where(local, columns, first.clamp(max=logits.shape[1] - 1).unsqueeze(1))
    banned = (positions < state.remaining[row_slots]) & local.any(dim=1)
    rows = torch.arange(logits.shape[0], device=logits.device).unsqueeze(1)
    original = logits[rows, safe].clone()
    logits[rows, safe] = torch.where(banned.unsqueeze(1), -float("inf"), original)
    return logits, (rows, safe, original)


def restore(
    logits: torch.Tensor, saved: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
) -> None:
    """Restore raw scores after all target sampling/probability reads are enqueued."""
    if saved is not None:
        rows, columns, original = saved
        logits[rows, columns] = original


def commit(
    accepted: tuple[torch.Tensor, torch.Tensor], batch_size: int, metadata: SpecMetadata
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance only the accepted prefix, including a recovery/bonus target token."""
    if not enabled(metadata):
        return accepted
    state = metadata.minimum_length_state
    slots = metadata.batch_slot_ids[:batch_size]
    tokens, counts = accepted
    if counts.shape != (batch_size,) or tokens.shape[0] != batch_size:
        raise RuntimeError("minimum-length accepted-prefix shape differs")
    advance = counts.to(torch.int64) * state.commit[slots].to(torch.int64)
    state.remaining.index_add_(0, slots, -advance)
    state.remaining.clamp_(min=0)
    return accepted
