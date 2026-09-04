# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Prefix-tokenization cache for ``DefaultInputProcessor``.

In multi-turn serving each prompt is usually the previous prompt plus a small
delta, so re-tokenizing the whole prompt every turn is wasted work. This cache
remembers the token ids of recent prompts and, when a new prompt extends a
cached one, tokenizes only the tail and splices it onto the cached ids.

Tokenizing a tail in isolation is not generally identical to tokenizing the
whole prompt, because BPE merges can straddle the split. The cache therefore
backs off ``overlap`` tokens from the end of the cached prefix, re-tokenizes
from there, and accepts the splice only if the first ``resync`` re-tokenized ids
equal the cached ids over the same span. Otherwise the prompt is tokenized in
full.

Off by default; enable with ``TLLM_PREFIX_TOKEN_CACHE=1``. Each
``DefaultInputProcessor`` owns its own cache, so cached ids are never shared
across tokenizers.
"""

from __future__ import annotations

import os
import threading
from array import array
from collections import OrderedDict
from dataclasses import dataclass
from typing import ClassVar, Mapping, Protocol, Sequence

from ..logger import logger

__all__ = [
    "PrefixTokenCache",
    "PrefixTokenCacheConfig",
    "create_prefix_token_cache",
    "prefix_cache_enabled",
]

ENABLE_ENV_VAR = "TLLM_PREFIX_TOKEN_CACHE"


def prefix_cache_enabled() -> bool:
    return os.environ.get(ENABLE_ENV_VAR, "0") == "1"


class OffsetTokenizer(Protocol):
    """The subset of the HF fast-tokenizer call interface the cache relies on."""

    def __call__(
        self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool
    ) -> Mapping[str, Sequence]: ...


@dataclass(frozen=True)
class PrefixTokenCacheConfig:
    """Tunables for :class:`PrefixTokenCache`.

    The sizing fields have env overrides (see ``ENV_VARS``); ``overlap`` and
    ``resync`` are correctness internals and do not.
    """

    max_entries: int = 512
    # Total characters of cached prompt text. The cached ids are int32, about
    # one byte per character of English text, so this bounds host memory to
    # roughly twice this many bytes.
    max_total_chars: int = 64 * 1024 * 1024
    # Prompts shorter than this are tokenized normally and never cached. Also
    # the number of leading characters entries are bucketed by, so a lookup does
    # not scan every entry.
    min_chars: int = 4096
    # Tokens to back off from the end of the cached prefix before re-tokenizing.
    overlap: int = 64
    # Re-tokenized ids that must equal the cached ids for the splice to be used.
    resync: int = 32

    ENV_VARS: ClassVar[dict[str, str]] = {
        "max_entries": "TLLM_PREFIX_TOKEN_CACHE_ENTRIES",
        "max_total_chars": "TLLM_PREFIX_TOKEN_CACHE_MAX_CHARS",
        "min_chars": "TLLM_PREFIX_TOKEN_CACHE_MIN_CHARS",
    }

    @classmethod
    def from_env(cls) -> PrefixTokenCacheConfig:
        """Build a config from the environment.

        Raises:
            ValueError: if an override is not a positive integer.
        """
        overrides: dict[str, int] = {}
        for field_name, env_var in cls.ENV_VARS.items():
            raw = os.environ.get(env_var)
            if raw is None:
                continue
            if not raw.isdigit() or int(raw) == 0:
                raise ValueError(f"{env_var} must be a positive integer, got {raw!r}")
            overrides[field_name] = int(raw)
        return cls(**overrides)


@dataclass(slots=True)
class _Entry:
    text: str
    ids: array  # int32
    # Every prompt that extends ``text`` is re-tokenized from the same place:
    # token ``split_token``, which starts at character ``split_char``.
    split_token: int
    split_char: int


class PrefixTokenCache:
    """Splice a cached prefix's token ids with a freshly tokenized tail.

    Thread-safe. Eviction is least-recently-used, bounded by both entry count
    and total cached characters. When a prompt extends a cached entry, the new
    prompt replaces that entry rather than sitting alongside it, so a
    conversation costs one entry however many turns it has.

    An unexpected error disables the cache, with a warning, and every prompt is
    tokenized in full from then on: a cache bug must never fail a request.
    """

    def __init__(self, config: PrefixTokenCacheConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._entries: OrderedDict[int, _Entry] = OrderedDict()  # LRU order
        self._buckets: dict[int, set[int]] = {}
        self._total_chars = 0
        self._next_id = 0
        self.disabled = False
        # Counters for tests.
        self.hits = 0
        self.misses = 0
        self.resync_failures = 0

    def encode(self, tokenizer: OffsetTokenizer, text: str) -> list[int]:
        """Return the token ids of ``text`` without special tokens."""
        if self.disabled or len(text) < self._config.min_chars:
            return self._tokenize(tokenizer, text)[0]
        try:
            return self._encode(tokenizer, text)
        except Exception as e:
            self.disabled = True
            logger.warning(f"Disabling the prefix token cache after an error: {e!r}")
            return self._tokenize(tokenizer, text)[0]

    def _encode(self, tokenizer: OffsetTokenizer, text: str) -> list[int]:
        found = self._lookup(text)
        ids = None
        if found is not None:
            eid, entry = found
            tail_ids, tail_offsets = self._tokenize(tokenizer, text[entry.split_char :])
            if self._resynced(entry, tail_ids):
                ids = entry.ids[: entry.split_token].tolist() + list(tail_ids)
                base_token, base_char = entry.split_token, entry.split_char
        if ids is None:
            tail_ids, tail_offsets = self._tokenize(tokenizer, text)
            ids, base_token, base_char = tail_ids, 0, 0

        # Where a prompt extending this one will be re-tokenized from.
        split_token = len(ids) - self._config.overlap
        tail_index = split_token - base_token

        with self._lock:
            if found is None:
                self.misses += 1
            elif base_token == 0:
                self.misses += 1
                self.resync_failures += 1
            else:
                self.hits += 1
            if found is not None:
                self._remove(found[0])
            if split_token > 0 and tail_index >= 0:
                self._insert(text, ids, split_token, base_char + tail_offsets[tail_index][0])
        return ids

    @staticmethod
    def _tokenize(
        tokenizer: OffsetTokenizer, text: str
    ) -> tuple[list[int], Sequence[tuple[int, int]]]:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return enc["input_ids"], enc["offset_mapping"]

    def _resynced(self, entry: _Entry, tail_ids: Sequence[int]) -> bool:
        """Whether the tokenizer re-synchronized with the cached ids at the seam."""
        cached_tail = entry.ids[entry.split_token :]
        span = min(self._config.resync, len(tail_ids), len(cached_tail))
        return span > 0 and list(tail_ids[:span]) == cached_tail[:span].tolist()

    def _bucket_key(self, text: str) -> int:
        return hash(text[: self._config.min_chars])

    def _lookup(self, text: str) -> tuple[int, _Entry] | None:
        """Longest cached entry that is a prefix of ``text``, promoted to MRU."""
        with self._lock:
            candidates = self._buckets.get(self._bucket_key(text), ())
            for eid in sorted(candidates, key=lambda e: len(self._entries[e].text), reverse=True):
                entry = self._entries[eid]
                if text.startswith(entry.text):
                    self._entries.move_to_end(eid)
                    return eid, entry
            return None

    def _insert(self, text: str, ids: Sequence[int], split_token: int, split_char: int) -> None:
        """Caller holds the lock."""
        key = self._bucket_key(text)
        bucket = self._buckets.setdefault(key, set())
        if any(self._entries[eid].text == text for eid in bucket):
            return  # concurrent misses on the same prompt
        eid, self._next_id = self._next_id, self._next_id + 1
        self._entries[eid] = _Entry(text, array("i", ids), split_token, split_char)
        bucket.add(eid)
        self._total_chars += len(text)
        while self._entries and (
            len(self._entries) > self._config.max_entries
            or self._total_chars > self._config.max_total_chars
        ):
            self._remove(next(iter(self._entries)))

    def _remove(self, eid: int) -> None:
        """Caller holds the lock. A no-op if another thread already removed it."""
        entry = self._entries.pop(eid, None)
        if entry is None:
            return
        self._total_chars -= len(entry.text)
        key = self._bucket_key(entry.text)
        bucket = self._buckets[key]
        bucket.discard(eid)
        if not bucket:
            del self._buckets[key]


def create_prefix_token_cache(tokenizer: object) -> PrefixTokenCache | None:
    """Return a cache for ``tokenizer`` if the feature is enabled and usable.

    Returns None when the feature is off or the tokenizer cannot report
    offsets. Raises ``ValueError`` for a malformed env override.
    """
    if not prefix_cache_enabled() or tokenizer is None:
        return None
    if not getattr(tokenizer, "is_fast", False):
        logger.warning(
            f"{ENABLE_ENV_VAR} is set but the tokenizer is not a fast tokenizer, "
            "which the prefix token cache needs for offset mappings; disabling it."
        )
        return None
    return PrefixTokenCache(PrefixTokenCacheConfig.from_env())
