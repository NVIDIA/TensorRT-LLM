# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owned snapshots of precomputed prompt embeddings.

The request contract is documented once, on ``TokensPrompt.prompt_embeds``.
"""

import dataclasses
import hashlib
import math
import pickle  # nosec B403
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch

if TYPE_CHECKING:
    from .multimodal import MultimodalInput

# Bump when the bytes hashed by ``_digest`` change; the version is part of every
# digest so old and new identities never compare equal.
PROMPT_EMBEDS_IDENTITY_VERSION = 1

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# Derived salts live under this prefix; caller salts that start with it are hashed into a
# separate namespace (``plain_cache_salt``), so no plain request can reproduce a derived salt.
_DERIVED_SALT_NAMESPACE = "prompt_embeds."
_SALT_PREFIX = f"{_DERIVED_SALT_NAMESPACE}v{PROMPT_EMBEDS_IDENTITY_VERSION}:"
_ESCAPED_SALT_PREFIX = f"{_DERIVED_SALT_NAMESPACE}user:"


@dataclass(frozen=True)
class PromptEmbeds:
    """Owned prompt-embedding rows and the prompt positions they replace.

    Attributes:
        embeds: ``[K, H]`` contiguous CPU tensor, or None in a copy without rows
            (``identity_only``, ``for_peer_ranks``).
        positions: ``[K]`` int64 CPU tensor of strictly increasing positions.
        hidden_size: ``H``.
        dtype: dtype of ``embeds``.
        row_digests: ``K`` concatenated 32-byte SHA-256 digests, one per row.
        digest: Hex SHA-256 over dtype, shape, positions and row digests.
        prefillable: Whether the prefill can run, from ``embeds`` or from the
            rows TP rank 0 sends each chunk; False once the rows are gone.
    """

    embeds: torch.Tensor | None
    positions: torch.Tensor
    hidden_size: int
    dtype: torch.dtype
    row_digests: bytes
    digest: str
    prefillable: bool = True

    @property
    def num_rows(self) -> int:
        return int(self.positions.numel())

    def chunk_rows(self, begin: int, end: int) -> tuple[int, int]:
        """Return the row range ``(lo, hi)`` whose positions fall in ``[begin, end)``."""
        lo = int(torch.searchsorted(self.positions, begin, right=False))
        hi = int(torch.searchsorted(self.positions, end, right=False))
        return lo, hi

    def covers(self, position: int) -> bool:
        """Return whether prompt ``position`` takes a supplied row."""
        lo, hi = self.chunk_rows(position, position + 1)
        return hi > lo

    def row_digest(self, row: int) -> bytes:
        """Return the 32-byte SHA-256 digest of row index ``row`` (not a prompt position)."""
        return self.row_digests[row * _ROW_DIGEST_LEN : (row + 1) * _ROW_DIGEST_LEN]

    def identity_only(self) -> "PromptEmbeds":
        """Return a copy without the rows, keeping positions and digests."""
        return dataclasses.replace(self, embeds=None, prefillable=False)

    def for_peer_ranks(self) -> "PromptEmbeds":
        """Return the copy other ranks receive; TP rank 0 sends them each chunk's rows."""
        return dataclasses.replace(self, embeds=None)

    def __reduce_ex__(self, protocol: int) -> tuple:
        # Ship the rows as one raw buffer: a tensor pickles through torch.save
        # and copies the rows twice on each side.
        fields = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        embeds = fields.pop("embeds")
        raw = None
        if embeds is not None:
            raw = embeds.view(torch.uint8).numpy()
            # Protocol 5 writes the buffer in place and loads it as a bytearray.
            raw = pickle.PickleBuffer(raw) if protocol >= 5 else raw.tobytes()
        return _unpickle_prompt_embeds, (raw, fields)


def _unpickle_prompt_embeds(raw: bytearray | bytes | None, fields: dict) -> PromptEmbeds:
    embeds = None
    if raw is not None:
        buffer = raw if isinstance(raw, bytearray) else bytearray(raw)
        embeds = torch.frombuffer(buffer, dtype=fields["dtype"]).view(-1, fields["hidden_size"])
    return PromptEmbeds(embeds=embeds, **fields)


_ROW_DIGEST_LEN = 32


def _row_digests(embeds: torch.Tensor) -> bytes:
    prefix = f"prompt_embeds.v{PROMPT_EMBEDS_IDENTITY_VERSION}|{embeds.dtype}|{embeds.shape[1]}|"
    base = hashlib.sha256(prefix.encode())
    # Reinterpret as bytes so bfloat16 (no numpy dtype) hashes the same way; each
    # row is hashed from the buffer without a copy.
    out = bytearray()
    for row in embeds.view(torch.uint8).numpy():
        h = base.copy()
        h.update(row)
        out += h.digest()
    return bytes(out)


def _digest(embeds: torch.Tensor, positions: torch.Tensor, row_digests: bytes) -> str:
    h = hashlib.sha256()
    h.update(f"v{PROMPT_EMBEDS_IDENTITY_VERSION}|{embeds.dtype}|{tuple(embeds.shape)}|".encode())
    h.update(positions.numpy().tobytes())
    h.update(row_digests)
    return h.hexdigest()


def value_range(embeds: torch.Tensor) -> tuple[float, float]:
    """Return the minimum and maximum of ``embeds``; NaN if any value is NaN."""
    lo, hi = torch.aminmax(embeds)
    return lo.item(), hi.item()


def _all_finite(embeds: torch.Tensor) -> bool:
    # One reduction with no [K, H] mask: NaN propagates to both extremes and an
    # infinity is an extreme.
    lo, hi = value_range(embeds)
    return math.isfinite(lo) and math.isfinite(hi)


def _as_mask(mask: Sequence[bool] | torch.Tensor, length: int) -> torch.Tensor:
    if isinstance(mask, torch.Tensor):
        if mask.dim() != 1:
            raise ValueError(f"prompt_embeds_mask must be 1-D, got shape {tuple(mask.shape)}")
        if mask.dtype != torch.bool:
            raise TypeError(f"prompt_embeds_mask tensor must be torch.bool, got {mask.dtype}")
        _check_mask_length(mask.numel(), length)
        return mask.detach().to("cpu").to_dense()
    values = list(mask)
    if not all(isinstance(v, bool) for v in values):
        raise TypeError("prompt_embeds_mask must contain only bools")
    _check_mask_length(len(values), length)
    return torch.tensor(values, dtype=torch.bool, device="cpu")


def _check_mask_length(mask_length: int, length: int) -> None:
    if mask_length != length:
        raise ValueError(
            f"prompt_embeds_mask length ({mask_length}) must equal the "
            f"number of prompt tokens ({length})"
        )


def build_prompt_embeds(
    prompt_token_ids: list[int] | None,
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: Sequence[bool] | torch.Tensor | None = None,
    placeholder_token_id: int = 0,
) -> tuple[list[int], PromptEmbeds]:
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
        positions = torch.arange(num_rows, dtype=torch.int64, device="cpu")
    else:
        prompt_token_ids = list(prompt_token_ids)
        if prompt_embeds_mask is None:
            if num_rows != len(prompt_token_ids):
                raise ValueError(
                    f"prompt_embeds has {num_rows} rows for "
                    f"{len(prompt_token_ids)} prompt tokens; pass "
                    "prompt_embeds_mask to place a subset of positions"
                )
            positions = torch.arange(num_rows, dtype=torch.int64, device="cpu")
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
    if not _all_finite(embeds):
        raise ValueError("prompt_embeds contains non-finite values")
    row_digests = _row_digests(embeds)
    snapshot = PromptEmbeds(
        embeds=embeds,
        positions=positions.contiguous(),
        hidden_size=int(hidden_size),
        dtype=embeds.dtype,
        row_digests=row_digests,
        digest=_digest(embeds, positions, row_digests),
    )
    return prompt_token_ids, snapshot


@dataclass(frozen=True)
class PrebuiltPromptEmbeds(PromptEmbeds):
    """A snapshot handed to ``LLM.generate_async`` by the code that built it.

    For trtllm-serve, which builds it off its event loop: ``generate_async``
    takes it without copying or hashing the rows again, so the builder must
    not keep or change them. Callers of the LLM API pass a tensor instead.
    """


def prebuild_prompt_embeds(
    prompt_token_ids: list[int] | None,
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: Sequence[bool] | torch.Tensor | None = None,
) -> tuple[list[int], PrebuiltPromptEmbeds]:
    """Run ``build_prompt_embeds`` for a snapshot to hand to ``LLM.generate_async``."""
    token_ids, snapshot = build_prompt_embeds(prompt_token_ids, prompt_embeds, prompt_embeds_mask)
    fields = {f.name: getattr(snapshot, f.name) for f in dataclasses.fields(snapshot)}
    return token_ids, PrebuiltPromptEmbeds(**fields)


def prompt_embeds_cache_salt(user_salt: str | None, digest: str) -> str:
    """Return the cache salt that scopes reuse to identical embeddings.

    Used where blocks are keyed by token ids alone (the V1 manager, KV connectors).
    """
    h = hashlib.sha256()
    h.update(digest.encode())
    if user_salt is not None:
        h.update(b"|salt|")
        h.update(user_salt.encode("utf-8"))
    return _SALT_PREFIX + h.hexdigest()


def plain_cache_salt(user_salt: str | None) -> str | None:
    """Return the cache salt of a request without prompt embeddings.

    Salts outside the derived namespace pass through unchanged.
    """
    if user_salt is None or not user_salt.startswith(_DERIVED_SALT_NAMESPACE):
        return user_salt
    return _ESCAPED_SALT_PREFIX + hashlib.sha256(user_salt.encode("utf-8")).hexdigest()


def remap_prompt_embeds(
    prompt_embeds: PromptEmbeds,
    original_ids: Sequence[int],
    expanded_ids: Sequence[int],
    item_spans: Sequence[tuple[int, int]],
    run_spans: Sequence[tuple[int, int]] | None = None,
) -> PromptEmbeds:
    """Move prompt-embedding positions through multimodal placeholder expansion.

    Input processors replace each multimodal placeholder with that item's rows,
    possibly framed by extra tokens (for example Qwen3-VL per-frame timestamps
    and vision markers). The ``k``-th placeholder in ``original_ids`` (the token
    at item ``k``'s first row) anchors to that row. The original tokens between
    two items must re-tokenize to themselves, except where framing is inserted
    right before the next item: the tokens around it match from both ends, and
    a text token merged with the framing is left unplaced. No embedding
    position may be a placeholder or an unplaced token.

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
                if token not in original_ids:
                    raise ValueError(
                        "This model expands its multimodal placeholder into different "
                        "token ids, so prompt_embeds cannot be combined with "
                        "multimodal data for it."
                    )
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
    if any(mapping[p] == _AMBIGUOUS for p in rows):
        raise ValueError(
            "prompt_embeds_mask selects a token after the framing multimodal processing "
            "inserts before an item; its place is ambiguous because the framing can hold "
            "the same id. Keep that position a token."
        )
    if any(mapping[p] < 0 for p in rows):
        raise ValueError(
            "prompt_embeds_mask selects a token that multimodal processing merged "
            "with the framing it inserts before an item (for example a Qwen3-VL "
            "frame timestamp); keep that position a token."
        )
    positions = torch.tensor([mapping[p] for p in rows], dtype=torch.int64, device="cpu")
    embeds = prompt_embeds.embeds
    return dataclasses.replace(
        prompt_embeds,
        positions=positions,
        digest=_digest(embeds, positions, prompt_embeds.row_digests),
    )


def _raise_unaligned() -> None:
    raise ValueError(
        "prompt_token_ids did not survive multimodal processing unchanged "
        "outside the multimodal placeholders; use placeholder ids that "
        "detokenize and retokenize to themselves (for example the ids of the "
        "replaced tokens)."
    )


# ``_align_gap`` value of a token matched from the end of a gap after inserted framing.
_AMBIGUOUS = -2


def _align_gap(
    segment: Sequence[int], expanded_ids: Sequence[int], begin: int, end: int, framed: bool
) -> list[int]:
    """Return the expanded index of each ``segment`` token in ``[begin, end)``, or -1 if unplaced.

    ``framed``: an item starts at ``end`` and framing inserted before it may
    merge with the last text tokens, so the segment matches from both ends.
    Once the head stops matching, framing changed the gap, so tokens matched from
    the end are ``_AMBIGUOUS``: the framing can contain their id.
    """
    n = len(segment)
    head = 0
    while head < n and begin + head < end and expanded_ids[begin + head] == segment[head]:
        head += 1
    if not framed:
        if head != n or begin + n != end:
            _raise_unaligned()
        return list(range(begin, end))
    tail = 0
    while (
        tail < n - head
        and end - 1 - tail >= begin + head
        and expanded_ids[end - 1 - tail] == segment[n - 1 - tail]
    ):
        tail += 1
    tail_positions = list(range(end - tail, end))
    if head < n:
        tail_positions = [_AMBIGUOUS] * tail
    return list(range(begin, begin + head)) + [-1] * (n - head - tail) + tail_positions


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
