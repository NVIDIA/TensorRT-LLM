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

``KVCacheManagerV2`` exposes no block hashes to a connector -- ``RequestData``
reports them empty -- so content identity is derived here instead. The chain is
the standard one: a block's hash covers its own tokens *and* every token before
it, so a key can only be reused by a request whose prefix is byte-identical.

A key is ``<namespace>/<block hash>``. The namespace pins down everything that
would make the stored bytes mean something different: the model, the shard that
produced them, the layer group inside that shard, the tokens each page holds and
how many bytes a page is. Anything that changes those reads as a cache miss
rather than as garbage.
"""

import hashlib
from dataclasses import dataclass
from typing import List, Optional, Sequence

__all__ = [
    "BlockHashChain",
    "KeyNamespace",
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


class BlockHashChain:
    """Rolling hashes of a request's full blocks, one entry per block ordinal.

    Extended in place as a request's token list grows, so generation steps cost
    one digest per newly completed block rather than a rehash of the prompt.
    """

    def __init__(self, tokens_per_block: int, cache_salt: Optional[str] = None):
        if tokens_per_block <= 0:
            raise ValueError(f"tokens_per_block must be > 0, got {tokens_per_block}")
        self._tokens_per_block = int(tokens_per_block)
        # The salt seeds the chain rather than being mixed into every block, so
        # a request carrying a different salt diverges from the first block on.
        salt_bytes = b"" if cache_salt is None else str(cache_salt).encode()
        self._seed = _digest(b"salt", salt_bytes)
        self._hashes: List[bytes] = []

    @property
    def tokens_per_block(self) -> int:
        """Tokens covered by each entry in the chain."""
        return self._tokens_per_block

    @property
    def hashes(self) -> Sequence[bytes]:
        """Hashes computed so far, indexed by block ordinal."""
        return self._hashes

    def extend(self, tokens: Sequence[int]) -> Sequence[bytes]:
        """Grow the chain to cover every full block of ``tokens``.

        Args:
            tokens: The request's complete token list, prompt first. Must be an
                extension of what was passed previously; a request's tokens only
                ever grow, so a shorter list means the caller mixed up requests.

        Returns:
            The full chain, indexed by block ordinal.
        """
        num_full_blocks = len(tokens) // self._tokens_per_block
        if num_full_blocks < len(self._hashes):
            raise ValueError(
                f"token list shrank from {len(self._hashes)} to {num_full_blocks} "
                "full blocks; a hash chain belongs to exactly one request"
            )
        for ordinal in range(len(self._hashes), num_full_blocks):
            start = ordinal * self._tokens_per_block
            block = tokens[start : start + self._tokens_per_block]
            parent = self._hashes[-1] if self._hashes else self._seed
            # Fixed-width little-endian token ids: a delimiter-free encoding
            # would let two different token sequences serialize identically.
            payload = b"".join(int(token).to_bytes(8, "little", signed=True) for token in block)
            self._hashes.append(_digest(parent, payload))
        return self._hashes


@dataclass(frozen=True)
class KeyNamespace:
    """The part of a store key that is fixed for one shard and layer group."""

    cache_prefix: str
    model_key: str
    #: Global rank of the shard whose KV these bytes are, and the world size it
    #: was produced under. Both are needed: rank 3 of 8 holds different heads
    #: than rank 3 of 4.
    rank: int
    world_size: int
    layer_group_id: int
    tokens_per_block: int
    bytes_per_page: int

    @property
    def prefix(self) -> str:
        """The literal string every key in this namespace starts with."""
        return (
            f"{self.cache_prefix}/{self.model_key}"
            f"/w{self.world_size}r{self.rank}"
            f"/lg{self.layer_group_id}"
            f"/t{self.tokens_per_block}b{self.bytes_per_page}"
        )

    def key(self, block_hash: bytes) -> str:
        """The store key holding one page of this namespace."""
        return f"{self.prefix}/{block_hash.hex()}"
