# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Precomputed prompt embeddings that replace the token embedding at chosen positions.

A request carries ``prompt_token_ids`` of full length ``L`` plus ``K`` embedding
rows and the ``K`` prompt positions they occupy. The engine looks up the token
embedding of every position as usual and then overwrites the masked positions
with the supplied rows, so the rows must equal the output of the model's input
embedding module (after any scaling the module applies).
"""

import hashlib
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch

# Bump when the bytes hashed by ``_digest`` change; the version is part of every
# digest so old and new identities never compare equal.
PROMPT_EMBEDS_IDENTITY_VERSION = 1

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# Matches GenerationRequest.MAX_CACHE_SALT_LEN; the derived salt is far shorter.
_SALT_PREFIX = f"prompt_embeds.v{PROMPT_EMBEDS_IDENTITY_VERSION}:"


@dataclass(frozen=True)
class PromptEmbeds:
    """Owned prompt-embedding rows and the prompt positions they replace.

    Attributes:
        embeds: ``[K, H]`` contiguous CPU tensor owned by this object, or None
            when only the identity travels (generation-only disaggregated
            workers never need the values).
        positions: ``[K]`` int64 CPU tensor of strictly increasing prompt
            positions.
        hidden_size: ``H``.
        digest: Hex SHA-256 over the identity version, dtype, shape, positions
            and row bytes.
    """

    embeds: Optional[torch.Tensor]
    positions: torch.Tensor
    hidden_size: int
    digest: str

    @property
    def num_rows(self) -> int:
        """Number of prompt positions this request replaces."""
        return int(self.positions.numel())

    def chunk_rows(self, begin: int, end: int) -> Tuple[int, int]:
        """Return the half-open row range whose positions fall in ``[begin, end)``.

        Args:
            begin: First prompt position of the chunk.
            end: One past the last prompt position of the chunk.

        Returns:
            ``(lo, hi)`` indices into ``embeds`` and ``positions``.
        """
        lo = int(torch.searchsorted(self.positions, begin, right=False))
        hi = int(torch.searchsorted(self.positions, end, right=False))
        return lo, hi

    def covers(self, position: int) -> bool:
        """Return whether ``position`` takes a supplied embedding row."""
        lo, hi = self.chunk_rows(position, position + 1)
        return hi > lo

    def identity_only(self) -> "PromptEmbeds":
        """Return a copy that keeps the identity but drops the row values."""
        return PromptEmbeds(
            embeds=None, positions=self.positions, hidden_size=self.hidden_size, digest=self.digest
        )

    def to_dtype(self, dtype: torch.dtype) -> "PromptEmbeds":
        """Return a copy whose rows are cast to ``dtype``; the identity is unchanged."""
        if self.embeds is None or self.embeds.dtype == dtype:
            return self
        return PromptEmbeds(
            embeds=self.embeds.to(dtype).contiguous(),
            positions=self.positions,
            hidden_size=self.hidden_size,
            digest=self.digest,
        )


def _digest(embeds: torch.Tensor, positions: torch.Tensor) -> str:
    """Hash the canonical identity of an owned embedding snapshot."""
    h = hashlib.sha256()
    h.update(f"v{PROMPT_EMBEDS_IDENTITY_VERSION}|{embeds.dtype}|{tuple(embeds.shape)}|".encode())
    h.update(positions.numpy().tobytes())
    # Reinterpret as bytes so bfloat16 (no numpy dtype) hashes the same way.
    h.update(embeds.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _as_mask(mask: Union[Sequence[bool], torch.Tensor], length: int) -> torch.Tensor:
    """Validate a per-position mask and return it as a bool CPU tensor."""
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
    """Validate a prompt-embedding request and take an owned canonical snapshot.

    Args:
        prompt_token_ids: Full-length prompt ids, or None when every position is
            an embedding (the prompt length is then ``K``).
        prompt_embeds: ``[K, H]`` floating tensor of rows for the masked
            positions, in mask order.
        prompt_embeds_mask: Length-``L`` bools, True where a row replaces the
            token embedding. Optional when ``K == L``.
        placeholder_token_id: Token id used for every position when
            ``prompt_token_ids`` is None.

    Returns:
        The prompt token ids to schedule and the snapshot.
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

    # Owned contiguous CPU copy: later caller writes to its tensor must not
    # change what is hashed or what the engine reads.
    embeds = prompt_embeds.detach().to("cpu", copy=True).contiguous()
    if not bool(torch.isfinite(embeds).all()):
        raise ValueError("prompt_embeds contains non-finite values")
    snapshot = PromptEmbeds(
        embeds=embeds,
        positions=positions.contiguous(),
        hidden_size=int(hidden_size),
        digest=_digest(embeds, positions),
    )
    return prompt_token_ids, snapshot


def prompt_embeds_cache_salt(user_salt: Optional[str], digest: str) -> str:
    """Return the KV cache salt that scopes reuse to identical embeddings.

    Requests with the same ids, embeddings and user salt share KV blocks; any
    other request, including a plain-token request with the same placeholder
    ids, never matches.
    """
    h = hashlib.sha256()
    h.update(digest.encode())
    if user_salt is not None:
        h.update(b"|salt|")
        h.update(user_salt.encode("utf-8"))
    return _SALT_PREFIX + h.hexdigest()
