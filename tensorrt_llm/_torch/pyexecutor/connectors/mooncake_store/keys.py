# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Block identity and store key naming for the Mooncake store connector.

`KVCacheManagerV2` reports `RequestData.block_hashes` empty, so content identity
is derived here. A block's hash covers its own tokens and every token before it,
so a key can only be reused by a request whose prefix is byte-identical. The
chain is seeded with a `ReuseScope` covering what the tokens do not record.

Under one-model speculative decoding the draft layers write the KV at a position
from the prompt tokens after it, so a block's hash then also covers the prompt
lookahead the draft reads past the block's end, and the chain is seeded apart
from the one a server without that lookahead uses. Pages a speculative server
writes are therefore never served to a server whose draft reads differently,
and the keys of a server without a draft are unchanged.

A key is `<namespace>/<block hash>`. The namespace pins down everything else
that decides what the stored bytes mean: the model, the attention shard that
produced them, the layer group inside that shard, the tokens each page holds and
how many bytes a page is. A change to any of those reads as a miss rather than
as garbage.
"""

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "BlockHashChain",
    "KeyNamespace",
    "ReuseScope",
    "HASH_DIGEST_BYTES",
]

#: 128 bits. Collisions decide whether one request reads another's KV, so the
#: digest is sized to make that negligible over any realistic cache lifetime,
#: while staying half the width of a full blake2b digest in every key.
HASH_DIGEST_BYTES = 16


def _digest(*parts: bytes) -> bytes:
    hasher = hashlib.blake2b(digest_size=HASH_DIGEST_BYTES)
    for part in parts:
        hasher.update(part)
    return hasher.digest()


def _framed(*parts: bytes) -> bytes:
    """Length-prefix each part, so no two different part lists encode alike."""
    return b"".join(len(part).to_bytes(4, "little") + part for part in parts)


@dataclass(frozen=True)
class ReuseScope:
    """What, besides the tokens, decides whose KV a block hash names.

    A multimodal placeholder token carries the same id whichever image stands
    behind it, so two requests differing only in their media would otherwise
    hash alike and read each other's pages. Everything here is a property of
    the whole request, so it seeds the chain rather than being mixed into each
    block.
    """

    #: `LlmRequest.cache_salt`: the caller's own partition of the cache.
    cache_salt: Optional[str] = None
    #: Content digest per multimodal item, in prompt order. Seeding with the
    #: whole set means a request whose media differ diverges from its first
    #: block, so sharing only a media prefix is a miss rather than a wrong page.
    multimodal_digests: Tuple[bytes, ...] = field(default_factory=tuple)

    def seed(self) -> bytes:
        """The digest a request in this scope starts its hash chain from."""
        return _digest(_framed(self._salt_bytes(), *self.multimodal_digests))

    def _salt_bytes(self) -> bytes:
        """Presence-tagged, so an absent salt and an empty one name different scopes."""
        if self.cache_salt is None:
            return b"\x00"
        return b"\x01" + self.cache_salt.encode("utf-8")


class BlockHashChain:
    """Rolling hashes of a request's full blocks, one entry per block ordinal.

    Extended in place as the token list grows, so a generation step costs one
    digest per newly completed block rather than a rehash of the prompt.

    With a prompt lookahead of `L` tokens, block `b` is hashed over its own
    tokens and the `L` that follow it, `tokens[b * T : (b + 1) * T + L]` for `T`
    tokens per block, so a block exists only once the `L` tokens after it do.
    """

    def __init__(
        self,
        tokens_per_block: int,
        scope: Optional[ReuseScope] = None,
        *,
        prompt_lookahead: int = 0,
    ):
        """Start an empty chain.

        Args:
            tokens_per_block: Tokens each block, and so each hash, covers.
            scope: What besides the tokens decides whose KV a hash names.
            prompt_lookahead: Prompt tokens past a block's end that its KV
                depends on: the `draft_prompt_lookahead` of a one-model draft
                sharing the KV cache, 0 without one. A non-zero value seeds the
                chain apart from the lookahead-0 one, whose keys it leaves as
                they are.

        Raises:
            ValueError: When `tokens_per_block` is not positive or
                `prompt_lookahead` is negative.
        """
        if tokens_per_block <= 0:
            raise ValueError(f"tokens_per_block must be > 0, got {tokens_per_block}")
        if prompt_lookahead < 0:
            raise ValueError(f"prompt_lookahead must be >= 0, got {prompt_lookahead}")
        self._tokens_per_block = int(tokens_per_block)
        self._prompt_lookahead = int(prompt_lookahead)
        self._seed = (scope or ReuseScope()).seed()
        if self._prompt_lookahead:
            # Mixed in only when non-zero, so every lookahead-0 key stays the one
            # pools already hold.
            self._seed = _digest(
                self._seed, b"prompt-lookahead", self._prompt_lookahead.to_bytes(8, "little")
            )
        self._hashes: List[bytes] = []

    @property
    def tokens_per_block(self) -> int:
        """Tokens covered by each entry in the chain."""
        return self._tokens_per_block

    @property
    def prompt_lookahead(self) -> int:
        """Tokens past each block's end that its hash also covers."""
        return self._prompt_lookahead

    @property
    def hashes(self) -> Sequence[bytes]:
        """Hashes computed so far, indexed by block ordinal."""
        return self._hashes

    def extend(self, tokens: Sequence[int]) -> Sequence[bytes]:
        """Grow the chain to cover every full block of `tokens`.

        A block counts as full only once its prompt lookahead is there too.

        Args:
            tokens: The request's complete token list, prompt first. Must
                extend what was passed previously; a shorter list means the
                caller mixed up requests.

        Returns:
            The full chain, indexed by block ordinal.
        """
        tokens_per_block = self._tokens_per_block
        lookahead = self._prompt_lookahead
        num_full_blocks = max(0, len(tokens) - lookahead) // tokens_per_block
        first_new = len(self._hashes)
        if num_full_blocks < first_new:
            raise ValueError(
                f"token list shrank from {first_new} to {num_full_blocks} "
                "full blocks; a hash chain belongs to exactly one request"
            )
        if num_full_blocks == first_new:
            return self._hashes
        # Fixed-width little-endian token ids: a delimiter-free encoding would
        # let two different token sequences serialize identically. '<i8' is the
        # signed 8-byte little-endian encoding `int.to_bytes` produces, so the
        # keys are unchanged; it is done in one call for every new block because
        # per-token encoding dominated the lookup of a long prompt. A token
        # outside int64 raises OverflowError before any block is appended.
        # Windows of a block plus its lookahead start one block apart, so the
        # payload runs `lookahead` tokens past the last new block.
        payload = memoryview(
            np.asarray(
                tokens[
                    first_new * tokens_per_block : num_full_blocks * tokens_per_block + lookahead
                ],
                dtype="<i8",
            ).tobytes()
        )
        block_bytes = 8 * tokens_per_block
        window_bytes = block_bytes + 8 * lookahead
        parent = self._hashes[-1] if self._hashes else self._seed
        for offset in range(0, (num_full_blocks - first_new) * block_bytes, block_bytes):
            parent = _digest(parent, payload[offset : offset + window_bytes])
            self._hashes.append(parent)
        return self._hashes


@dataclass(frozen=True)
class KeyNamespace:
    """The part of a store key that is fixed for one shard and layer group."""

    namespace: str
    model_key: str
    #: The attention shard whose KV these bytes are, and the shard count it was
    #: produced under: the global rank and world size under tensor parallelism,
    #: 0 of 1 under attention DP, where every owner holds all heads. Both are
    #: needed: rank 3 of 8 holds different heads than rank 3 of 4.
    rank: int
    world_size: int
    layer_group_id: int
    tokens_per_block: int
    bytes_per_page: int

    @property
    def prefix(self) -> str:
        """The literal string every key in this namespace starts with."""
        return (
            f"{self.namespace}/{self.model_key}"
            f"/w{self.world_size}r{self.rank}"
            f"/lg{self.layer_group_id}"
            f"/t{self.tokens_per_block}b{self.bytes_per_page}"
        )

    def key(self, block_hash: bytes) -> str:
        """The store key holding one page of this namespace."""
        return f"{self.prefix}/{block_hash.hex()}"
