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
    """Tunables for :class:`PrefixTokenCache`; every field has an env override."""

    max_entries: int = 512
    # Total characters of cached prompt text; a proxy for host memory, since the
    # cached ids are proportional to it.
    max_total_chars: int = 64 * 1024 * 1024
    # Tokens to back off from the end of the cached prefix before re-tokenizing.
    overlap: int = 64
    # Re-tokenized ids that must equal the cached ids for the splice to be used.
    resync: int = 32
    # Prompts shorter than this are not worth caching.
    min_chars: int = 4096
    # Entries are bucketed by a hash of this many leading characters so a lookup
    # does not scan every entry. Prompts shorter than this are never cached.
    bucket_chars: int = 2048

    ENV_VARS: ClassVar[dict[str, str]] = {
        "max_entries": "TLLM_PREFIX_TOKEN_CACHE_ENTRIES",
        "max_total_chars": "TLLM_PREFIX_TOKEN_CACHE_MAX_CHARS",
        "overlap": "TLLM_PREFIX_TOKEN_CACHE_OVERLAP",
        "resync": "TLLM_PREFIX_TOKEN_CACHE_RESYNC",
        "min_chars": "TLLM_PREFIX_TOKEN_CACHE_MIN_CHARS",
    }

    @classmethod
    def from_env(cls) -> PrefixTokenCacheConfig:
        """Build a config from the environment.

        Raises:
            ValueError: if an override is not a non-negative integer, or is zero
                for a field other than ``min_chars``.
        """
        overrides: dict[str, int] = {}
        for field_name, env_var in cls.ENV_VARS.items():
            raw = os.environ.get(env_var)
            if raw is None:
                continue
            try:
                value = int(raw)
            except ValueError:
                raise ValueError(f"{env_var} must be an integer, got {raw!r}") from None
            if value < 0 or (value == 0 and field_name != "min_chars"):
                raise ValueError(f"{env_var} must be positive, got {value}")
            overrides[field_name] = value
        return cls(**overrides)


@dataclass(slots=True)
class _Entry:
    text: str
    ids: list[int]
    # Every prompt that extends ``text`` is re-tokenized from the same place:
    # token ``split_token``, which starts at character ``split_char``.
    split_token: int
    split_char: int
    bucket: int


class PrefixTokenCache:
    """Splice a cached prefix's token ids with a freshly tokenized tail.

    Thread-safe. Eviction is least-recently-used, bounded by both entry count
    and total cached characters.
    """

    def __init__(self, config: PrefixTokenCacheConfig | None = None) -> None:
        self._config = config or PrefixTokenCacheConfig()
        self._lock = threading.Lock()
        self._entries: OrderedDict[int, _Entry] = OrderedDict()  # LRU order
        self._buckets: dict[int, set[int]] = {}
        self._total_chars = 0
        self._next_id = 0
        # Counters for logging and tests.
        self.hits = 0
        self.misses = 0
        self.resync_failures = 0

    @property
    def min_chars(self) -> int:
        return self._config.min_chars

    def encode(self, tokenizer: OffsetTokenizer, text: str) -> list[int]:
        """Return the token ids of ``text`` without special tokens."""
        entry = self._lookup(text)
        spliced = self._extend(tokenizer, text, entry) if entry is not None else None
        if spliced is not None:
            ids, split = spliced
        else:
            ids, offsets = self._tokenize(tokenizer, text)
            split = self._split_point(ids, offsets, base_token=0, base_char=0)

        with self._lock:
            if spliced is not None:
                self.hits += 1
            else:
                self.misses += 1
                if entry is not None:
                    self.resync_failures += 1
            # An exact repeat of a cached prompt has nothing new to store.
            if split is not None and (entry is None or len(entry.text) != len(text)):
                self._insert(text, ids, *split)
        return ids

    @staticmethod
    def _tokenize(
        tokenizer: OffsetTokenizer, text: str
    ) -> tuple[list[int], Sequence[tuple[int, int]]]:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return list(enc["input_ids"]), enc["offset_mapping"]

    def _split_point(
        self,
        ids: Sequence[int],
        offsets: Sequence[tuple[int, int]],
        base_token: int,
        base_char: int,
    ) -> tuple[int, int] | None:
        """Where a prompt extending this one is re-tokenized from, or None.

        ``ids``/``offsets`` describe the tail starting at ``base_token``, whose
        offsets are relative to ``base_char``. The split point is ``overlap``
        tokens before the end of the whole prompt.
        """
        split_token = base_token + len(ids) - self._config.overlap
        tail_index = split_token - base_token
        if split_token <= 0 or tail_index < 0:
            return None
        return split_token, base_char + offsets[tail_index][0]

    def _extend(
        self, tokenizer: OffsetTokenizer, text: str, entry: _Entry
    ) -> tuple[list[int], tuple[int, int] | None] | None:
        """Splice ``entry`` with the re-tokenized tail of ``text``.

        Returns None if the tokenizer did not re-synchronize at the seam.
        """
        tail_ids, tail_offsets = self._tokenize(tokenizer, text[entry.split_char :])
        cached_tail = entry.ids[entry.split_token :]
        span = min(self._config.resync, len(tail_ids), len(cached_tail))
        if span <= 0 or tail_ids[:span] != cached_tail[:span]:
            return None
        ids = entry.ids[: entry.split_token] + tail_ids
        split = self._split_point(tail_ids, tail_offsets, entry.split_token, entry.split_char)
        return ids, split

    def _bucket_key(self, text: str) -> int:
        return hash(text[: self._config.bucket_chars])

    def _lookup(self, text: str) -> _Entry | None:
        """Longest cached entry that is a prefix of ``text``, promoted to MRU."""
        with self._lock:
            best_id, best = None, None
            for eid in self._buckets.get(self._bucket_key(text), ()):
                entry = self._entries[eid]
                if (best is None or len(entry.text) > len(best.text)) and text.startswith(
                    entry.text
                ):
                    best_id, best = eid, entry
            if best_id is not None:
                self._entries.move_to_end(best_id)
            return best

    def _insert(self, text: str, ids: list[int], split_token: int, split_char: int) -> None:
        """Caller holds the lock."""
        if len(text) < self._config.bucket_chars:
            return  # a lookup could never find it
        eid, self._next_id = self._next_id, self._next_id + 1
        key = self._bucket_key(text)
        self._entries[eid] = _Entry(text, ids, split_token, split_char, key)
        self._buckets.setdefault(key, set()).add(eid)
        self._total_chars += len(text)
        while self._entries and (
            len(self._entries) > self._config.max_entries
            or self._total_chars > self._config.max_total_chars
        ):
            self._evict_oldest()

    def _evict_oldest(self) -> None:
        """Caller holds the lock."""
        eid, entry = self._entries.popitem(last=False)
        self._total_chars -= len(entry.text)
        bucket = self._buckets[entry.bucket]
        bucket.discard(eid)
        if not bucket:
            del self._buckets[entry.bucket]


def create_prefix_token_cache(tokenizer: object) -> PrefixTokenCache | None:
    """Return a cache for ``tokenizer`` if the feature is enabled and usable.

    Returns None, logging why, when the cache is disabled by configuration,
    the tokenizer cannot report offsets, or an env override is malformed.
    """
    if not prefix_cache_enabled() or tokenizer is None:
        return None
    if not getattr(tokenizer, "is_fast", False):
        logger.warning(
            f"{ENABLE_ENV_VAR} is set but the tokenizer is not a fast tokenizer, "
            "which the prefix token cache needs for offset mappings; disabling it."
        )
        return None
    try:
        config = PrefixTokenCacheConfig.from_env()
    except ValueError as e:
        logger.warning(f"Disabling the prefix token cache: {e}")
        return None
    return PrefixTokenCache(config)
