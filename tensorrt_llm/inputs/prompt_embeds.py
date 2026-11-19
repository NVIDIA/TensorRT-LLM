# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owned snapshots of precomputed prompt embeddings.

The request contract is documented once, on ``TokensPrompt.prompt_embeds``.
"""

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union

import torch

if TYPE_CHECKING:
    from .multimodal import MultimodalInput

# Bump when the bytes hashed by ``_digest`` change; the version is part of every
# digest so old and new identities never compare equal.
PROMPT_EMBEDS_IDENTITY_VERSION = 1

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

_SALT_PREFIX = f"prompt_embeds.v{PROMPT_EMBEDS_IDENTITY_VERSION}:"


@dataclass(frozen=True)
class PromptEmbeds:
    """Owned prompt-embedding rows and the prompt positions they replace.

    Attributes:
        embeds: ``[K, H]`` contiguous CPU tensor, or None in a finished
            request's client copy (``identity_only``).
        positions: ``[K]`` int64 CPU tensor of strictly increasing positions.
        hidden_size: ``H``.
        row_digests: ``K`` concatenated 32-byte SHA-256 digests, one per row.
        digest: Hex SHA-256 over dtype, shape, positions and row digests.
    """

    embeds: Optional[torch.Tensor]
    positions: torch.Tensor
    hidden_size: int
    row_digests: bytes
    digest: str

    @property
    def num_rows(self) -> int:
        return int(self.positions.numel())

    def chunk_rows(self, begin: int, end: int) -> Tuple[int, int]:
        """Return the row range ``(lo, hi)`` whose positions fall in ``[begin, end)``."""
        lo = int(torch.searchsorted(self.positions, begin, right=False))
        hi = int(torch.searchsorted(self.positions, end, right=False))
        return lo, hi

    def covers(self, position: int) -> bool:
        lo, hi = self.chunk_rows(position, position + 1)
        return hi > lo

    def row_digest(self, row: int) -> bytes:
        return self.row_digests[row * _ROW_DIGEST_LEN : (row + 1) * _ROW_DIGEST_LEN]

    def identity_only(self) -> "PromptEmbeds":
        return PromptEmbeds(
            embeds=None,
            positions=self.positions,
            hidden_size=self.hidden_size,
            row_digests=self.row_digests,
            digest=self.digest,
        )


_ROW_DIGEST_LEN = 32


def _row_digests(embeds: torch.Tensor) -> bytes:
    prefix = f"prompt_embeds.v{PROMPT_EMBEDS_IDENTITY_VERSION}|{embeds.dtype}|{embeds.shape[1]}|"
    # Reinterpret as bytes so bfloat16 (no numpy dtype) hashes the same way.
    rows = embeds.view(torch.uint8).numpy()
    out = bytearray()
    for row in rows:
        h = hashlib.sha256(prefix.encode())
        h.update(row.tobytes())
        out += h.digest()
    return bytes(out)


def _digest(embeds: torch.Tensor, positions: torch.Tensor, row_digests: bytes) -> str:
    h = hashlib.sha256()
    h.update(f"v{PROMPT_EMBEDS_IDENTITY_VERSION}|{embeds.dtype}|{tuple(embeds.shape)}|".encode())
    h.update(positions.numpy().tobytes())
    h.update(row_digests)
    return h.hexdigest()


def _as_mask(mask: Union[Sequence[bool], torch.Tensor], length: int) -> torch.Tensor:
    if isinstance(mask, torch.Tensor):
        if mask.dim() != 1:
            raise ValueError(f"prompt_embeds_mask must be 1-D, got shape {tuple(mask.shape)}")
        if mask.dtype != torch.bool:
            raise TypeError(f"prompt_embeds_mask tensor must be torch.bool, got {mask.dtype}")
        mask_t = mask.detach().to("cpu")
    else:
        values = list(mask)
        if not all(isinstance(v, bool) for v in values):
            raise TypeError("prompt_embeds_mask must contain only bools")
        mask_t = torch.tensor(values, dtype=torch.bool)
    if mask_t.numel() != length:
        raise ValueError(
            f"prompt_embeds_mask length ({mask_t.numel()}) must equal the "
            f"number of prompt tokens ({length})"
        )
    return mask_t


def build_prompt_embeds(
    prompt_token_ids: Optional[List[int]],
    prompt_embeds: Any,
    prompt_embeds_mask: Optional[Union[Sequence[bool], torch.Tensor]] = None,
    placeholder_token_id: int = 0,
) -> Tuple[List[int], PromptEmbeds]:
    """Validate a ``TokensPrompt``/``EmbedsPrompt`` request and snapshot its rows.

    The contract is on ``TokensPrompt.prompt_embeds``.

    Args:
        prompt_token_ids: Full-length ids, or None when every position is an embedding.
        prompt_embeds: ``[K, H]`` float16, bfloat16 or float32 rows, in mask order.
        prompt_embeds_mask: Length-``L`` bools selecting the ``K`` positions;
            optional when ``K == L``.
        placeholder_token_id: Id scheduled everywhere when ``prompt_token_ids`` is None.

    Returns:
        The prompt token ids to schedule and an owned CPU snapshot; later writes
        to ``prompt_embeds`` do not affect it.
    """
    if not isinstance(prompt_embeds, torch.Tensor):
        raise TypeError(f"prompt_embeds must be a torch.Tensor, got {type(prompt_embeds).__name__}")
    if prompt_embeds.dim() != 2:
        raise ValueError(
            f"prompt_embeds must have shape [num_rows, hidden_size], got "
            f"{tuple(prompt_embeds.shape)}"
        )
    if prompt_embeds.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"prompt_embeds dtype must be one of "
            f"{[str(d) for d in _SUPPORTED_DTYPES]}, got "
            f"{prompt_embeds.dtype}"
        )
    if prompt_embeds.is_sparse or prompt_embeds.layout != torch.strided:
        raise TypeError("prompt_embeds must be a dense strided tensor")
    num_rows, hidden_size = prompt_embeds.shape
    if num_rows == 0 or hidden_size == 0:
        raise ValueError("prompt_embeds must have at least one row and column")

    if prompt_token_ids is None:
        if prompt_embeds_mask is not None:
            raise ValueError(
                "prompt_embeds_mask requires prompt_token_ids; without ids every "
                "prompt position is an embedding"
            )
        prompt_token_ids = [placeholder_token_id] * num_rows
        positions = torch.arange(num_rows, dtype=torch.int64)
    else:
        prompt_token_ids = list(prompt_token_ids)
        if prompt_embeds_mask is None:
            if num_rows != len(prompt_token_ids):
                raise ValueError(
                    f"prompt_embeds has {num_rows} rows for "
                    f"{len(prompt_token_ids)} prompt tokens; pass "
                    "prompt_embeds_mask to place a subset of positions"
                )
            positions = torch.arange(num_rows, dtype=torch.int64)
        else:
            mask_t = _as_mask(prompt_embeds_mask, len(prompt_token_ids))
            positions = torch.nonzero(mask_t, as_tuple=False).flatten().to(torch.int64)
            if positions.numel() != num_rows:
                raise ValueError(
                    f"prompt_embeds has {num_rows} rows but prompt_embeds_mask "
                    f"selects {positions.numel()} positions"
                )

    # Later caller writes must not change what is hashed or what the engine reads.
    embeds = prompt_embeds.detach().to("cpu", copy=True).contiguous()
    if not bool(torch.isfinite(embeds).all()):
        raise ValueError("prompt_embeds contains non-finite values")
    row_digests = _row_digests(embeds)
    snapshot = PromptEmbeds(
        embeds=embeds,
        positions=positions.contiguous(),
        hidden_size=int(hidden_size),
        row_digests=row_digests,
        digest=_digest(embeds, positions, row_digests),
    )
    return prompt_token_ids, snapshot


def prompt_embeds_cache_salt(user_salt: Optional[str], digest: str) -> str:
    """Return the cache salt that scopes reuse to identical embeddings.

    Used where blocks are keyed by token ids alone (the V1 manager, KV connectors).
    """
    h = hashlib.sha256()
    h.update(digest.encode())
    if user_salt is not None:
        h.update(b"|salt|")
        h.update(user_salt.encode("utf-8"))
    return _SALT_PREFIX + h.hexdigest()


def remap_prompt_embeds(
    prompt_embeds: PromptEmbeds,
    original_ids: Sequence[int],
    expanded_ids: Sequence[int],
    item_spans: Sequence[tuple[int, int]],
    run_spans: Optional[Sequence[tuple[int, int]]] = None,
) -> PromptEmbeds:
    """Move prompt-embedding positions through multimodal placeholder expansion.

    Input processors replace each multimodal placeholder with that item's rows,
    possibly framed by extra tokens (for example Qwen3-VL per-frame timestamps
    and vision markers). The ``k``-th placeholder in ``original_ids`` (the token
    at item ``k``'s first row) anchors to that row. The original tokens between
    two placeholders are matched, in order, only within the expanded tokens
    between the two items' spans; unmatched tokens there may only be framing
    inserted right before the next item. No embedding position may be a
    placeholder.

    Args:
        prompt_embeds: Snapshot whose positions index ``original_ids``.
        original_ids: Prompt ids before expansion.
        expanded_ids: Prompt ids the engine schedules.
        item_spans: ``[start, end)`` expanded rows of each multimodal item, in
            prompt order, from its first multimodal row to its last.
        run_spans: ``[start, end)`` of each contiguous multimodal run (a video
            frame); defaults to ``item_spans``.

    Returns:
        A snapshot with the same rows whose positions index ``expanded_ids``.
    """
    if prompt_embeds.embeds is None:
        raise ValueError("Remapping prompt_embeds needs the rows.")
    if list(original_ids) == list(expanded_ids):
        # Already expanded (e.g. a disaggregated encoder handoff): positions
        # stand, and none may be a multimodal row (text between frames may).
        runs = item_spans if run_spans is None else run_spans
        for p in prompt_embeds.positions.tolist():
            if any(start <= p < end for start, end in runs):
                raise ValueError("prompt_embeds_mask selects a multimodal placeholder position.")
        return prompt_embeds
    mapping: list[int] = []
    placeholders = set()
    gap_start = 0
    for start, end in list(item_spans) + [(len(expanded_ids), len(expanded_ids))]:
        is_item = start < len(expanded_ids)
        token = expanded_ids[start] if is_item else None
        segment_end = len(original_ids)
        if is_item:
            segment_end = len(mapping)
            while segment_end < len(original_ids) and original_ids[segment_end] != token:
                segment_end += 1
            if segment_end == len(original_ids):
                _raise_unaligned()
        segment = original_ids[len(mapping) : segment_end]
        mapping.extend(_align_gap(segment, expanded_ids, gap_start, start, framed=is_item))
        if is_item:
            placeholders.add(len(mapping))
            mapping.append(start)
            gap_start = end
    rows = prompt_embeds.positions.tolist()
    if any(p in placeholders for p in rows):
        raise ValueError("prompt_embeds_mask selects a multimodal placeholder position.")
    positions = torch.tensor([mapping[p] for p in rows], dtype=torch.int64)
    embeds = prompt_embeds.embeds
    return PromptEmbeds(
        embeds=embeds,
        positions=positions,
        hidden_size=prompt_embeds.hidden_size,
        row_digests=prompt_embeds.row_digests,
        digest=_digest(embeds, positions, prompt_embeds.row_digests),
    )


def _raise_unaligned() -> None:
    raise ValueError(
        "prompt_token_ids did not survive multimodal processing unchanged "
        "outside the multimodal placeholders; use placeholder ids that "
        "detokenize and retokenize to themselves (for example the ids of the "
        "replaced tokens)."
    )


def _align_gap(
    segment: Sequence[int], expanded_ids: Sequence[int], begin: int, end: int, framed: bool
) -> list[int]:
    """Return the expanded index of each ``segment`` token, matched in order in ``[begin, end)``.

    ``framed``: an item starts at ``end`` and its framing may precede it.
    """
    mapping = []
    j = begin
    for token in segment:
        while j < end and expanded_ids[j] != token:
            j += 1
        if j == end:
            _raise_unaligned()
        mapping.append(j)
        j += 1
    # Unmatched tokens may only be the next item's framing: after the
    # second-to-last matched token, i.e. around the item's start marker.
    if not framed:
        framing_from = end
    elif len(mapping) >= 2:
        framing_from = mapping[-2] + 1
    else:
        framing_from = begin
    matched = set(mapping)
    if any(k not in matched for k in range(begin, framing_from)):
        _raise_unaligned()
    return mapping


def multimodal_item_spans(multimodal_input: "MultimodalInput") -> list[tuple[int, int]]:
    """Return each multimodal item's ``[first row, last row + 1)`` in prompt order."""
    offsets = multimodal_input.multimodal_item_run_cu_offsets
    if multimodal_input.multimodal_run_positions is not None and offsets is not None:
        starts = multimodal_input.multimodal_run_positions
        lengths = multimodal_input.multimodal_run_lengths
        spans = [
            (
                int(starts[offsets[k]]),
                int(starts[offsets[k + 1] - 1]) + int(lengths[offsets[k + 1] - 1]),
            )
            for k in range(len(offsets) - 1)
            if offsets[k + 1] > offsets[k]
        ]
    else:
        spans = [
            (int(start), int(start) + int(length))
            for start, length in zip(
                multimodal_input.multimodal_positions, multimodal_input.multimodal_lengths
            )
        ]
    return sorted(spans)


def multimodal_run_spans(multimodal_input: "MultimodalInput") -> list[tuple[int, int]]:
    """Return each contiguous multimodal run's ``[start, end)`` in prompt order."""
    if multimodal_input.multimodal_run_positions is None:
        return multimodal_item_spans(multimodal_input)
    return sorted(
        (int(start), int(start) + int(length))
        for start, length in zip(
            multimodal_input.multimodal_run_positions, multimodal_input.multimodal_run_lengths
        )
    )


def prompt_embeds_block_key(row_digest: bytes, placeholder_id: int) -> bytes:
    """Return the 32-byte KV block key of an embedding position (row and placeholder id)."""
    h = hashlib.sha256(b"prompt_embeds.key|")
    h.update(row_digest)
    h.update(int(placeholder_id).to_bytes(8, "little", signed=True))
    return h.digest()
