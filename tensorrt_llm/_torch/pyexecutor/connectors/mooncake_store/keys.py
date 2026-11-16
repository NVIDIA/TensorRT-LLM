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

A key is `<namespace>/<block hash>`. The namespace pins down everything else
that decides what the stored bytes mean: the model, the shard that produced
them, the layer group inside that shard, the tokens each page holds and how many
bytes a page is. A change to any of those reads as a miss rather than as garbage.
"""

import array
import functools
import hashlib
import os
import struct
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

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


#: M3.1 packed block hash port (peihengh keys.py; his kill switch TRTLLM_M31_DISABLE_PACKED_BLOCK_HASH): with
#: TRTLLM_MOONCAKE_STORE_PACKED_BLOCK_HASH=1 (default off) the new blocks' token ids are packed with one
#: `array('q', ...)` pass, with no per-call argument tuple, and sliced per block through a memoryview, with no copy.
#: The payload bytes, and so every hash and key, are identical to the `struct.pack` path.
PACKED_BLOCK_HASH_ENV = "TRTLLM_MOONCAKE_STORE_PACKED_BLOCK_HASH"
_PACK_TOKENS = os.environ.get(PACKED_BLOCK_HASH_ENV, "0") == "1"


#: Receipt that the packed path ran: printed once per process, on stderr, the first time it hashes.
PACKED_HASH_RECEIPT = "M31_PACKED_HASH_ON"
_packed_receipt_sent = False


def _packed_receipt() -> None:
    global _packed_receipt_sent
    if _packed_receipt_sent:
        return
    _packed_receipt_sent = True
    try:
        print(f"{PACKED_HASH_RECEIPT} pid={os.getpid()}", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001 - a receipt must never take the server down
        pass


def _pack_tokens_array(tokens: Sequence[int]) -> memoryview:
    """Token ids as fixed-width little-endian int64, back to back (the packed path)."""
    packed = array.array("q", tokens)
    if sys.byteorder != "little":
        packed.byteswap()
    return memoryview(packed).cast("B")


def _digest(*parts: bytes) -> bytes:
    hasher = hashlib.blake2b(digest_size=HASH_DIGEST_BYTES)
    for part in parts:
        hasher.update(part)
    return hasher.digest()


#: Cross-request hash-chain reuse (lanes/mc/ovh, default off). A new request's chain is computed over its whole
#: prompt, ~3.7k blake2b digests for a 120k-token agentic turn, on every TP rank under rank-local lookups, although
#: the session's previous turn hashed the same prefix moments earlier. With TRTLLM_MOONCAKE_STORE_CHAIN_MEMO_MB=<MiB>
#: a process-wide LRU (bounded by that many MiB of packed tokens) keeps recent prompts' packed token bytes and chains;
#: a fresh chain reuses the hashes of the longest byte-identical block prefix it finds (same seed, same block size)
#: and digests only the rest. Candidates are found by the bytes of block 2^k - 1 (k = 0, 1, ...) and every reused
#: prefix is verified by comparing the packed bytes, so the chain is identical to computing it from scratch.
CHAIN_MEMO_ENV = "TRTLLM_MOONCAKE_STORE_CHAIN_MEMO_MB"
#: Prompts shorter than this many full blocks are hashed directly.
CHAIN_MEMO_MIN_BLOCKS = 64
#: Candidates verified per level before trying a shorter prefix.
_CHAIN_MEMO_PROBE = 4
#: Receipt printed (stderr) the first time a chain is served through the memo.
CHAIN_MEMO_RECEIPT = "CHAIN_MEMO_ARMED"


class _ChainMemo:
    """LRU of recent prompts: (seed, block bytes) -> packed token bytes and the chain over them."""

    def __init__(self, budget_bytes: int):
        self._budget = int(budget_bytes)
        self._bytes = 0
        self._entries: "OrderedDict[int, Tuple[bytes, int, bytes, Tuple[bytes, ...]]]" = OrderedDict()
        self._index: Dict[tuple, List[int]] = {}
        self._next = 0
        self._lock = threading.Lock()
        self._armed = False
        #: Fresh chains served, of them with a reused prefix, blocks reused, blocks digested.
        self.chains = self.hits = self.reused = self.hashed = 0

    @staticmethod
    def _levels(num_blocks: int):
        blocks = 1
        while blocks <= num_blocks:
            yield blocks
            blocks <<= 1

    @staticmethod
    def _key(seed: bytes, step: int, packed: bytes, blocks: int) -> tuple:
        return (seed, step, blocks, packed[(blocks - 1) * step : blocks * step])

    def reuse(self, seed: bytes, step: int, packed: bytes) -> Tuple[bytes, ...]:
        """Hashes of the longest stored block prefix byte-identical to `packed` (empty when there is none)."""
        n = len(packed) // step
        with self._lock:
            for blocks in reversed(list(self._levels(n))):
                ids = self._index.get(self._key(seed, step, packed, blocks))
                if not ids:
                    continue
                for entry_id in reversed(ids[-_CHAIN_MEMO_PROBE:]):
                    entry = self._entries.get(entry_id)
                    if entry is None:
                        continue
                    _, _, stored, hashes = entry
                    if stored[: blocks * step] != packed[: blocks * step]:
                        continue
                    # Largest m in [blocks, hi] whose m-block prefixes are equal (equality is monotone in m). The
                    # first lo blocks are known equal, so each step compares only blocks [lo, mid).
                    lo, hi = blocks, min(n, len(stored) // step, len(hashes))
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        if stored[lo * step : mid * step] == packed[lo * step : mid * step]:
                            lo = mid
                        else:
                            hi = mid - 1
                    self._entries.move_to_end(entry_id)
                    return hashes[:lo]
        return ()

    def add(self, seed: bytes, step: int, packed: bytes, hashes: Sequence[bytes]) -> None:
        n = min(len(packed) // step, len(hashes))
        if n == 0 or len(packed) > self._budget:
            return
        packed = bytes(packed[: n * step])
        with self._lock:
            entry_id = self._next
            self._next += 1
            self._entries[entry_id] = (seed, step, packed, tuple(hashes[:n]))
            self._bytes += len(packed)
            for blocks in self._levels(n):
                self._index.setdefault(self._key(seed, step, packed, blocks), []).append(entry_id)
            while self._bytes > self._budget and self._entries:
                old_id, (old_seed, old_step, old_packed, _) = self._entries.popitem(last=False)
                self._bytes -= len(old_packed)
                for blocks in self._levels(len(old_packed) // old_step):
                    key = self._key(old_seed, old_step, old_packed, blocks)
                    ids = self._index.get(key)
                    if ids is None:
                        continue
                    try:
                        ids.remove(old_id)
                    except ValueError:
                        pass
                    if not ids:
                        del self._index[key]

    def receipt(self) -> None:
        if self._armed:
            return
        self._armed = True
        try:
            print(
                f"{CHAIN_MEMO_RECEIPT} pid={os.getpid()} budget_mib={self._budget >> 20}", file=sys.stderr, flush=True
            )
        except Exception:  # noqa: BLE001 - a receipt must never take the server down
            pass

    def stats(self) -> str:
        return (
            f"chain memo: {self.chains} fresh chains, {self.hits} reused a prefix, blocks reused {self.reused} "
            f"digested {self.hashed}, {len(self._entries)} prompts {self._bytes / 2**20:.0f} MiB"
        )


def _chain_memo_from_env() -> Optional[_ChainMemo]:
    try:
        mib = int(os.environ.get(CHAIN_MEMO_ENV, "0") or 0)
    except ValueError:
        mib = 0
    return _ChainMemo(mib << 20) if mib > 0 else None


_CHAIN_MEMO: Optional[_ChainMemo] = _chain_memo_from_env()


def chain_memo_stats() -> Optional[str]:
    """The memo's cumulative stats line, or None when it is off."""
    memo = _CHAIN_MEMO
    return memo.stats() if memo is not None else None


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
        return _digest(
            _framed(
                b"" if self.cache_salt is None else str(self.cache_salt).encode(),
                *self.multimodal_digests,
            )
        )


class BlockHashChain:
    """Rolling hashes of a request's full blocks, one entry per block ordinal.

    Extended in place as the token list grows, so a generation step costs one
    digest per newly completed block rather than a rehash of the prompt.
    """

    def __init__(self, tokens_per_block: int, scope: Optional[ReuseScope] = None):
        if tokens_per_block <= 0:
            raise ValueError(f"tokens_per_block must be > 0, got {tokens_per_block}")
        self._tokens_per_block = int(tokens_per_block)
        self._seed = (scope or ReuseScope()).seed()
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
        """Grow the chain to cover every full block of `tokens`.

        Args:
            tokens: The request's complete token list, prompt first. Must
                extend what was passed previously; a shorter list means the
                caller mixed up requests.

        Returns:
            The full chain, indexed by block ordinal.
        """
        num_full_blocks = len(tokens) // self._tokens_per_block
        if num_full_blocks < len(self._hashes):
            raise ValueError(
                f"token list shrank from {len(self._hashes)} to {num_full_blocks} "
                "full blocks; a hash chain belongs to exactly one request"
            )
        first = len(self._hashes)
        if first == num_full_blocks:
            return self._hashes
        # Fixed-width little-endian token ids: a delimiter-free encoding would
        # let two different token sequences serialize identically. The new
        # blocks are packed in one call and sliced per block, which gives the
        # same payloads as packing each token, at a fraction of the cost on the
        # scheduler's path.
        start, end = first * self._tokens_per_block, num_full_blocks * self._tokens_per_block
        memo = _CHAIN_MEMO
        if memo is not None and first == 0 and num_full_blocks >= CHAIN_MEMO_MIN_BLOCKS:
            return self._extend_through_memo(memo, tokens, end)
        if _PACK_TOKENS:
            if not _packed_receipt_sent:
                _packed_receipt()
            packed = _pack_tokens_array(tokens[start:end])
        else:
            packed = struct.pack(f"<{end - start}q", *tokens[start:end])
        step = 8 * self._tokens_per_block
        parent = self._hashes[-1] if self._hashes else self._seed
        for offset in range(0, len(packed), step):
            parent = _digest(parent, packed[offset : offset + step])
            self._hashes.append(parent)
        return self._hashes

    def _extend_through_memo(self, memo: "_ChainMemo", tokens: Sequence[int], end: int) -> Sequence[bytes]:
        """A fresh chain over tokens[:end]: the longest stored identical prefix's hashes, then digests for the rest."""
        if _PACK_TOKENS:
            # Same payload bytes as struct.pack; the packed path keeps its receipt when the memo serves the chain.
            if not _packed_receipt_sent:
                _packed_receipt()
            packed = bytes(_pack_tokens_array(tokens[:end]))
        else:
            packed = struct.pack(f"<{end}q", *tokens[:end])
        step = 8 * self._tokens_per_block
        reused = memo.reuse(self._seed, step, packed)
        self._hashes.extend(reused)
        parent = reused[-1] if reused else self._seed
        for offset in range(len(reused) * step, len(packed), step):
            parent = _digest(parent, packed[offset : offset + step])
            self._hashes.append(parent)
        memo.chains += 1
        memo.hits += 1 if reused else 0
        memo.reused += len(reused)
        memo.hashed += len(self._hashes) - len(reused)
        memo.add(self._seed, step, packed, self._hashes)
        memo.receipt()
        return self._hashes


@dataclass(frozen=True)
class KeyNamespace:
    """The part of a store key that is fixed for one shard and layer group."""

    namespace: str
    model_key: str
    #: Global rank of the shard whose KV these bytes are, and the world size it
    #: was produced under. Both are needed: rank 3 of 8 holds different heads
    #: than rank 3 of 4.
    rank: int
    world_size: int
    layer_group_id: int
    tokens_per_block: int
    bytes_per_page: int

    @functools.cached_property
    def prefix(self) -> str:
        """The literal string every key in this namespace starts with (computed once: a lookup builds thousands)."""
        return (
            f"{self.namespace}/{self.model_key}"
            f"/w{self.world_size}r{self.rank}"
            f"/lg{self.layer_group_id}"
            f"/t{self.tokens_per_block}b{self.bytes_per_page}"
        )

    def key(self, block_hash: bytes) -> str:
        """The store key holding one page of this namespace."""
        return f"{self.prefix}/{block_hash.hex()}"
