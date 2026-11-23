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
"""Reuse guard for draft-layer pages that hold only a window of the drafter context.

A context server whose drafter reads a sliding window of ``read_tokens`` context rows can write
the drafter context (the draft layers' K/V rows) only for the last ``keep_tokens`` prompt
positions of each request: a generation server reads rows ``[prompt - read_tokens, prompt)``
and nothing below. The draft rows below ``prompt - keep_tokens`` of a request's pages are then
never written, while its target rows are, and both are committed for reuse in the same pages.
A later request that reuses such a prefix inherits rows it did not compute, and its own drafter
window may reach into rows nobody wrote.

``DraftContextWindowGuard`` caps the reuse claim of every context request so that the inherited
part of its drafter window holds only written rows:

* **Marks.** When a request claims a prefix of ``d`` tokens and computes ``[d, prompt)``, the
  draft rows of ``[d, prompt - keep_tokens)`` are not written. Every block that holds such a
  row is marked by the token prefix *before* the block (the chain key of the blocks ahead of
  it, ``_PrefixHasher``), not by its own content: a reuse match that ends inside a block reuses
  a page of some block under the same prefix, whose own tokens differ from the request's, so
  only the prefix ahead of it is common to every page the match can land on.
* **Claim limit.** A request of ``prompt`` tokens reads the draft rows ``[prompt -
  read_tokens, prompt)``. The limit is the start of the first block in that range whose prefix
  is marked (no limit when there is none), so a claim ending at or below it inherits only
  blocks under prefixes no request ever left unwritten, and recomputes everything from there.
* **Never forgotten.** A page stays in the radix tree when another request commits the same
  block again (the existing page wins), so a mark cannot be cleared by a later full write.
  Marks live in a Bloom filter: a false positive only caps a claim that was safe (a shorter
  match, more recompute), and the filter never forgets a mark.

Everything is decided from the request's own tokens, its reuse scope and the claims in their
order, so every pipeline stage that keeps a guard caps alike. ``keep_tokens`` does not enter the
claim limit, only the marks: a larger rewind or block size is never unsafe, it only marks less.

**Per-request keys.** The prefix keys are chain keys, so hashing a prompt costs O(prompt) however
few blocks a call reads. A caller that claims for one request more than once (a store prefetch
sized by the claim, a pipeline stage agreement's reservation, then the resume) passes the request
object as ``owner`` (the KV cache manager does so with ``TLLM_KV_DRAFT_CONTEXT_HELD_KEYS=1``,
``draft_context_held_keys_enabled``): the guard hashes
the prompt once, keeps the chain keys on that object (``RequestKeys``), and serves every later
claim limit and note of the same request from them. The keys of a block prefix are the same
however many blocks are hashed (the chain is a running sum), so a served call sees the same keys
as one that hashes. Only keys are kept: every limit is still read from the live marks. The owner
also records the lowest block its notes marked, so a later note of the same request adds only the
blocks below it; marks are a union of bits, so the filter ends up with the same bits either way.
"""

import os
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from tensorrt_llm.logger import logger

from .swa_stale_endpoint import _PrefixHasher

LOG_PREFIX = "[draft-context-window]"
#: Bits of the mark filter (32 MiB). At three hashes it keeps the false-positive rate under
#: 0.2 % up to about 3e7 marked blocks.
DEFAULT_FILTER_BITS = 1 << 28
DEFAULT_NUM_HASHES = 3
DEFAULT_LOG_EVERY = 2000
#: The attribute of a claim's owner (the request object) that holds its ``RequestKeys``.
REQUEST_KEYS_ATTR = "py_draft_context_keys"
#: Opt-in: the KV cache manager passes each request as the owner of its claims and notes.
HELD_KEYS_ENV = "TLLM_KV_DRAFT_CONTEXT_HELD_KEYS"


def draft_context_held_keys_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Whether ``TLLM_KV_DRAFT_CONTEXT_HELD_KEYS=1`` is set (default off).

    Args:
        environ: environment to read (default ``os.environ``).
    """
    environ = os.environ if environ is None else environ
    return environ.get(HELD_KEYS_ENV, "0").strip() == "1"


_MASK64 = np.uint64((1 << 64) - 1)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_SALT = np.uint64(0xD6E8FEB86659FD93)
_POPCOUNT = np.array([bin(value).count("1") for value in range(256)], dtype=np.uint8)


def _mix(values: np.ndarray) -> np.ndarray:
    """SplitMix64 finalizer of a uint64 array (wrapping arithmetic)."""
    z = values + _GOLDEN
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


class MarkFilter:
    """A Bloom filter of uint64 keys: no false negatives, never forgets a key."""

    def __init__(self, num_bits: int = DEFAULT_FILTER_BITS, num_hashes: int = DEFAULT_NUM_HASHES):
        if num_bits < 64 or num_bits & (num_bits - 1):
            raise ValueError(f"num_bits must be a power of two >= 64, got {num_bits}")
        if num_hashes < 1:
            raise ValueError(f"num_hashes must be >= 1, got {num_hashes}")
        self.num_bits = num_bits
        self.num_hashes = num_hashes
        self._mask = np.uint64(num_bits - 1)
        self._bits = np.zeros(num_bits // 8, dtype=np.uint8)
        self.num_added = 0

    def _positions(self, keys: np.ndarray) -> np.ndarray:
        """``[num_hashes, len(keys)]`` bit positions (double hashing)."""
        keys = np.asarray(keys, dtype=np.uint64)
        h1 = _mix(keys)
        h2 = _mix(keys ^ _SALT) | np.uint64(1)
        steps = np.arange(self.num_hashes, dtype=np.uint64)[:, None]
        return (h1[None, :] + steps * h2[None, :]) & self._mask

    def add(self, keys: np.ndarray) -> None:
        if len(keys) == 0:
            return
        positions = self._positions(keys).ravel()
        np.bitwise_or.at(
            self._bits,
            (positions >> np.uint64(3)).astype(np.int64),
            (np.uint8(1) << (positions & np.uint64(7)).astype(np.uint8)),
        )
        self.num_added += len(keys)

    def contains(self, keys: np.ndarray) -> np.ndarray:
        """Whether each key may have been added (bool array, ``len(keys)``)."""
        if len(keys) == 0:
            return np.zeros(0, dtype=bool)
        positions = self._positions(keys)
        bits = self._bits[(positions >> np.uint64(3)).astype(np.int64)]
        hit = (bits >> (positions & np.uint64(7)).astype(np.uint8)) & np.uint8(1)
        return hit.all(axis=0).astype(bool)

    def fill(self) -> float:
        """The fraction of bits set: a scan of the whole filter, so not for a hot path."""
        return float(_POPCOUNT[self._bits].sum(dtype=np.int64)) / self.num_bits

    def clear(self) -> None:
        self._bits[:] = 0
        self.num_added = 0


class RequestKeys:
    """One request's prefix chain keys for one guard, and the lowest block its notes marked.

    ``keys`` are the chain keys of the prompt's first ``(prompt_len - 1) // tokens_per_block``
    blocks: every claim limit and note of a ``prompt_len``-token prompt reads within them.
    ``marked_from`` is the lowest block a note of this request marked while the guard's marks
    were at ``epoch`` (None before the first note); a ``clear`` of the marks starts a new epoch.
    ``token`` is the guard's identity token, not the guard: the keys must not keep the guard and
    its mark filter alive, and a copied request (``copy.deepcopy``, pickle) holds a new token
    object that matches no guard, so it hashes its prompt again.
    """

    __slots__ = ("token", "prompt_len", "seed", "keys", "marked_from", "epoch")

    def __init__(self, token: object, prompt_len: int, seed: int, keys):
        self.token = token
        self.prompt_len = prompt_len
        self.seed = seed
        self.keys = keys
        self.marked_from: Optional[int] = None
        self.epoch = -1


class DraftContextWindowGuard:
    """Caps reuse claims so that a request's drafter window inherits only written draft rows.

    Args:
        tokens_per_block: the KV cache block size.
        read_tokens: the draft rows a reader reads below a request's prompt end (the drafter's
            attention window).
        keep_tokens: the draft rows the context server writes below a request's prompt end.
        filter_bits / num_hashes: the mark filter's geometry (``MarkFilter``).
    """

    def __init__(
        self,
        tokens_per_block: int,
        read_tokens: int,
        keep_tokens: int,
        filter_bits: int = DEFAULT_FILTER_BITS,
        num_hashes: int = DEFAULT_NUM_HASHES,
        log_every: int = DEFAULT_LOG_EVERY,
        name: str = "",
    ) -> None:
        if tokens_per_block < 1 or read_tokens < 1 or keep_tokens < read_tokens:
            raise ValueError(
                "invalid drafter context window: "
                f"tokens_per_block={tokens_per_block}, read_tokens={read_tokens}, "
                f"keep_tokens={keep_tokens} (keep_tokens must cover read_tokens)"
            )
        self.tokens_per_block = tokens_per_block
        self.read_tokens = read_tokens
        self.keep_tokens = keep_tokens
        self._hasher = _PrefixHasher(tokens_per_block)
        self._marks = MarkFilter(filter_bits, num_hashes)
        #: Bumped by ``clear``: a ``RequestKeys.marked_from`` of an older epoch names marks
        #: the filter no longer holds.
        self._epoch = 0
        #: Identifies this guard to the ``RequestKeys`` it hands out (compared with ``is``).
        self._keys_token = object()
        self._log_every = log_every
        self._name = name
        self.stats: Dict[str, int] = {}
        self._reset_stats()

    def _reset_stats(self) -> None:
        self.stats = {
            "claims": 0,
            "capped": 0,
            "capped_window_tokens": 0,
            "notes": 0,
            "marked_blocks": 0,
            "prefix_hashes": 0,
            "skipped_marks": 0,
        }

    def _chain(self, tokens: Sequence[int], num_blocks: int, seed: int) -> np.ndarray:
        self.stats["prefix_hashes"] += 1
        return self._hasher.block_keys(tokens, num_blocks, seed)

    def _request_keys(
        self, owner: Any, tokens: Sequence[int], prompt_len: int, seed: int
    ) -> RequestKeys:
        """``owner``'s keys, hashed now unless it holds this guard's keys of the same prompt."""
        held = getattr(owner, REQUEST_KEYS_ATTR, None)
        if (
            isinstance(held, RequestKeys)
            and held.token is self._keys_token
            and held.prompt_len == prompt_len
            and held.seed == seed
        ):
            return held
        num_blocks = max(0, (prompt_len - 1) // self.tokens_per_block)
        held = RequestKeys(
            self._keys_token, prompt_len, seed, self._chain(tokens, num_blocks, seed)
        )
        setattr(owner, REQUEST_KEYS_ATTR, held)
        return held

    def _parent_keys(
        self,
        tokens: Sequence[int],
        first: int,
        end: int,
        seed: int,
        held: Optional[RequestKeys] = None,
    ) -> np.ndarray:
        """The prefix keys of blocks ``[first, end)``: block ``j``'s is the chain key of the
        blocks ``[0, j)`` (the seed for ``j == 0``). From ``held`` when given."""
        if end <= first:
            return np.zeros(0, dtype=np.uint64)
        chain = held.keys[: end - 1] if held is not None else self._chain(tokens, end - 1, seed)
        if first > 0:
            return chain[first - 1 : end - 1]
        return np.concatenate([np.array([seed], dtype=np.uint64), chain[: end - 1]])

    def release(self, owner: Any) -> None:
        """Drop the keys ``owner`` holds (after its last claim of the prompt)."""
        if getattr(owner, REQUEST_KEYS_ATTR, None) is not None:
            setattr(owner, REQUEST_KEYS_ATTR, None)

    def claim_limit(
        self,
        tokens: Sequence[int],
        prompt_len: int,
        seed: int,
        count: bool = True,
        owner: Any = None,
    ) -> Optional[int]:
        """The largest reuse claim whose inherited drafter window holds only written rows.

        None when no block of the window ``[prompt_len - read_tokens, prompt_len)`` lies under a
        marked prefix; else the start of the first such block (a multiple of the block size).
        ``count`` False asks without counting a claim in ``stats`` (a prefetch that sizes itself
        by the claim the request will make later).
        With ``owner`` (the request object), its keys are hashed once and kept on it.
        """
        if count:
            self.stats["claims"] += 1
        if prompt_len <= 0:
            return None
        tpb = self.tokens_per_block
        first = max(0, prompt_len - self.read_tokens) // tpb
        end = (prompt_len - 1) // tpb + 1
        held = None if owner is None else self._request_keys(owner, tokens, prompt_len, seed)
        hits = self._marks.contains(self._parent_keys(tokens, first, end, seed, held))
        if not hits.any():
            if count:
                self._maybe_log()
            return None
        limit = (first + int(np.argmax(hits))) * tpb
        if count:
            self.stats["capped"] += 1
            self.stats["capped_window_tokens"] += prompt_len - max(
                limit, prompt_len - self.read_tokens
            )
            self._maybe_log()
        return limit

    def note_claim(
        self, tokens: Sequence[int], prompt_len: int, seed: int, start: int, owner: Any = None
    ) -> None:
        """Mark the blocks a request computing ``[start, prompt_len)`` leaves partly unwritten.

        With ``owner`` (the request object), its kept keys serve the note, and blocks an
        earlier note of the same request marked (since the last ``clear``) are not added again.
        """
        self.stats["notes"] += 1
        stop = prompt_len - self.keep_tokens
        if stop <= start:
            return
        tpb = self.tokens_per_block
        first = max(0, start) // tpb
        end = (stop - 1) // tpb + 1
        if owner is None:
            keys = self._parent_keys(tokens, first, end, seed)
            self._marks.add(keys)
            self.stats["marked_blocks"] += len(keys)
            return
        if end <= first:
            # An empty range: like the call without an owner, count and hash nothing.
            return
        held = self._request_keys(owner, tokens, prompt_len, seed)
        self.stats["marked_blocks"] += end - first
        add_end = end
        if held.epoch == self._epoch and held.marked_from is not None:
            add_end = min(end, held.marked_from)
            self.stats["skipped_marks"] += end - max(first, add_end)
            held.marked_from = min(held.marked_from, first)
        else:
            held.marked_from = first
            held.epoch = self._epoch
        if add_end > first:
            self._marks.add(self._parent_keys(tokens, first, add_end, seed, held))

    def _maybe_log(self) -> None:
        if self._log_every > 0 and self.stats["claims"] % self._log_every == 0:
            self.log_stats()

    def clear(self) -> None:
        """Forget every mark: call only once no committed block is left for reuse."""
        self._marks.clear()
        self._epoch += 1

    def log_stats(self, with_fill: bool = False) -> None:
        """Log the counts; ``with_fill`` adds the filter's fill, which scans the whole filter
        (the periodic line on the scheduling path leaves it out, the shutdown line has it)."""
        s = self.stats
        fill = f" filter_fill={self._marks.fill():.4f}" if with_fill else ""
        logger.info(
            f"{LOG_PREFIX}{self._name} read={self.read_tokens} keep={self.keep_tokens} "
            f"claims={s['claims']} capped={s['capped']} "
            f"capped_window_tokens={s['capped_window_tokens']} notes={s['notes']} "
            f"marked_blocks={s['marked_blocks']} skipped_marks={s['skipped_marks']} "
            f"prefix_hashes={s['prefix_hashes']}{fill}"
        )
