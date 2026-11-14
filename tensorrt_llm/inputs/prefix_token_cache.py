# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Prefix-tokenization cache for ``DefaultInputProcessor``.

In multi-turn serving each prompt is usually the previous prompt plus a small
delta, so re-tokenizing the whole prompt every turn is wasted work. This cache
remembers the token ids of recent prompts and, when a new prompt keeps a cached
one up to its re-tokenization point, tokenizes only the tail and splices it onto
the cached ids. The re-tokenization point lies ``overlap`` tokens before the end
of the cached prompt, so the new prompt may also rewrite that end, as a chat
template does when the previous turn's generation prefix is replaced by the
response.
Agentic clients also rewrite the end of the history between turns (a trimmed
tool result, a replaced last message), so a prompt that diverges from a cached
one before its re-tokenization point is spliced at a checkpoint of the cached
prompt: the last one that has another checkpoint after it inside their longest
common prefix.

Tokenizing a tail in isolation is not generally identical to tokenizing the
whole prompt, because BPE merges can straddle the split. The cache therefore
backs off ``overlap`` tokens from the end of the cached prefix (or, for a
diverging prompt, at least ``checkpoint_tokens`` tokens from the divergence),
re-tokenizes from there, and accepts the splice only if the first ``resync``
re-tokenized ids equal the cached ids over the same span (for a diverging
prompt, over the whole span up to the next checkpoint, which must also start a
re-tokenized token). Otherwise the prompt is tokenized in full. Tokenizers that alter
every text they are called on, such as those whose normalizer prepends a
``▁``, are probed for once when the cache is created and get no cache.

Off by default; enable with the ``enable_tokenization_cache`` LLM argument.
Each ``DefaultInputProcessor`` owns its own cache, so cached ids are never
shared across tokenizers. Sizing knobs, eligibility rules, and measured effect
are documented in ``docs/source/features/prefix-tokenization-cache.md``.
"""

from __future__ import annotations

import bisect
import contextlib
import os
import threading
from array import array
from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, ClassVar, Iterator, Mapping, NamedTuple, Protocol, Sequence

from ..logger import logger

__all__ = [
    "OffsetTokenizer",
    "PrefixTokenCache",
    "TokenIdList",
    "int32_bytes",
    "PrefixTokenCacheConfig",
    "RouteFailed",
    "create_prefix_token_cache",
    "tokenization_routing_key",
]

# Log the hit rate every this many cache-eligible requests, so an operator can
# see whether the cache is doing anything without reading its counters.
LOG_EVERY_REQUESTS = 1000
# Agentic prompts share a system/tool preamble that can run to tens of thousands
# of characters, so entries in one bucket cannot be told apart by their start.
# Each entry keeps the PROBE_CHARS characters before its re-tokenization point; a
# candidate is only compared in full when the incoming text has that probe at the
# same position.
PROBE_CHARS = 256
# A hit splices into an int32 array (the cached prefix is copied as one block, only
# the tail is converted) and that array is cached as is, instead of building the
# full id list and converting it back to an array. The ids are the same either way.
_ARRAY_SPLICE = os.environ.get("TRTLLM_PREFIX_TOKEN_CACHE_DISABLE_ARRAY_SPLICE", "0") != "1"
# A spliced hit returns a TokenIdList that keeps its int32 array, so a context-only
# reply can serialize the prompt ids without converting the list back.
_INT32_IDS = os.environ.get("TRTLLM_PREFIX_TOKEN_CACHE_DISABLE_INT32_IDS", "0") != "1"


class TokenIdList(list):
    """A list of token ids that also holds the same ids as an int32 array.

    ``int32`` is the array while the list is unchanged: every list mutation sets it to
    None first, so a reader that finds it can use it as the ids' int32 form (see
    ``int32_bytes``). Pickles, copies and deep-copies as a plain list.
    """

    __slots__ = ("int32",)

    def __reduce_ex__(self, protocol):
        return (list, (), None, iter(self))


def _dropping_int32(name):
    method = getattr(list, name)

    def mutate(self, *args, **kwargs):
        self.int32 = None
        return method(self, *args, **kwargs)

    mutate.__name__ = mutate.__qualname__ = name
    return mutate


for _name in (
    "__setitem__",
    "__delitem__",
    "__iadd__",
    "__imul__",
    "append",
    "extend",
    "insert",
    "pop",
    "remove",
    "reverse",
    "sort",
    "clear",
):
    setattr(TokenIdList, _name, _dropping_int32(_name))


def int32_bytes(ids: Sequence[int]) -> bytes:
    """``ids`` as int32 bytes: a ``TokenIdList``'s own array when it still holds one, else converted."""
    buf = getattr(ids, "int32", None)
    if buf is not None and len(buf) == len(ids):
        return buf.tobytes()
    return array("i", ids).tobytes()


# Block size, in characters, of the coarse scan for the first differing
# character of two prompts.
_COMPARE_BLOCK_CHARS = 4096

# The conversation the prompt being tokenized belongs to, if the caller knows it.
_ROUTING_KEY: ContextVar[str | None] = ContextVar("tokenization_routing_key", default=None)


@contextlib.contextmanager
def tokenization_routing_key(key: str | None) -> Iterator[None]:
    """Tag the prompts tokenized in this block with conversation ``key``.

    A cache with a ``route`` hands such prompts to the process that caches that
    conversation.
    """
    token = _ROUTING_KEY.set(key)
    try:
        yield
    finally:
        _ROUTING_KEY.reset(token)


class RouteFailed(Exception):
    """``PrefixTokenCache.route`` could not get the ids from the owning process."""


class OffsetTokenizer(Protocol):
    """The subset of the HF fast-tokenizer call interface the cache relies on."""

    def __call__(
        self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool
    ) -> Mapping[str, Sequence]: ...


@dataclass(frozen=True)
class PrefixTokenCacheConfig:
    """Tunables for :class:`PrefixTokenCache`.

    The sizing fields have env overrides (see ``ENV_VARS``); ``overlap``,
    ``resync`` and ``checkpoint_tokens`` are correctness internals and do not.
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
    # Spacing, in tokens, of the split points a diverging prompt can be spliced
    # at. A split is used only when the next checkpoint still lies inside the
    # common prefix, so it must exceed ``resync``.
    checkpoint_tokens: int = 256

    ENV_VARS: ClassVar[dict[str, str]] = {
        "max_entries": "TLLM_PREFIX_TOKEN_CACHE_ENTRIES",
        "max_total_chars": "TLLM_PREFIX_TOKEN_CACHE_MAX_CHARS",
        "min_chars": "TLLM_PREFIX_TOKEN_CACHE_MIN_CHARS",
    }

    def __post_init__(self) -> None:
        if self.checkpoint_tokens <= self.resync:
            raise ValueError(
                f"checkpoint_tokens ({self.checkpoint_tokens}) must exceed resync ({self.resync})"
            )

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


@dataclass(slots=True, frozen=True)
class _Entry:
    text: str
    ids: array  # int32
    # Every prompt that keeps ``text[:split_char]`` is re-tokenized from the same
    # place: token ``split_token``, the first token of character ``split_char``.
    split_token: int
    split_char: int
    probe: str  # the PROBE_CHARS characters of ``text`` before ``split_char``
    # Where a prompt that diverges from ``text`` can be re-tokenized from: the
    # first token of the character holding every ``checkpoint_tokens``-th token
    # (0, C, 2C, ...), and the start character of that token.
    checkpoint_token_indices: array  # int32
    checkpoint_chars: array  # int32


class _Match(NamedTuple):
    eid: int
    entry: _Entry
    split_token: int
    split_char: int
    extends: bool  # the prompt keeps ``entry.text[: entry.split_char]``
    # For a diverging prompt, the token and character of the checkpoint after
    # the split: the re-tokenized tail must match the cached ids up to there.
    next_checkpoint: tuple[int, int] | None = None


class PrefixTokenCache:
    """Splice a cached prefix's token ids with a freshly tokenized tail.

    Thread-safe. Eviction is least-recently-used, bounded by both entry count
    and total cached characters. A prompt that extends a cached entry replaces
    that entry rather than sitting alongside it, so a conversation costs one
    entry however many turns it has. A prompt that diverges from a cached entry
    is added next to it, since the divergence may be a branch of a history that
    continues.

    An unexpected error disables the cache, with a warning, and every prompt is
    tokenized in full from then on: a cache bug must never fail a request. A
    prompt the tokenizer itself rejects fails as it would without the cache and
    leaves the cache enabled.
    """

    def __init__(self, config: PrefixTokenCacheConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._entries: OrderedDict[int, _Entry] = OrderedDict()  # LRU order
        self._buckets: dict[int, dict[int, _Entry]] = {}
        self._total_chars = 0
        self._next_id = 0
        # Read and written without the lock: a bool flip is atomic, and the
        # worst case of a race is two threads each logging the disable warning.
        self.disabled = False
        # Counters, reported by periodic log lines and read by tests.
        self.hits = 0
        self.misses = 0
        self.resync_failures = 0
        # Hits on a cached prompt the new one diverges from (also in ``hits``).
        self.divergent_hits = 0
        # Cache-eligible prompts, routed or not; paces the periodic log line.
        self.requests = 0
        # Prompts another process tokenized through the route.
        self.routed = 0
        # Prompts tokenized here because the route failed.
        self.route_failures = 0
        self._route: Callable[[str, str], list[int] | None] | None = None

    def set_route(self, route: Callable[[str, str], list[int] | None] | None) -> None:
        """Hand prompts tagged with a routing key to ``route``, or stop if None.

        ``route(key, text)`` returns the ids of ``text`` as tokenized by the
        process that caches conversation ``key``, or None to tokenize it here,
        and raises :class:`RouteFailed` if that process did not answer; the
        prompt is then tokenized here too.
        """
        self._route = route

    def encode(self, tokenizer: OffsetTokenizer, text: str) -> list[int]:
        """Return the token ids of ``text`` without special tokens.

        Raises whatever the tokenizer raises for a prompt it cannot tokenize.
        """
        if self.disabled or len(text) < self._config.min_chars:
            return self._tokenize(tokenizer, text)[0]
        try:
            ids = self._routed(text)
            if ids is None:
                ids = self._encode_or_disable(tokenizer, text)
            return ids
        finally:
            self._count_request()

    def _routed(self, text: str) -> list[int] | None:
        """Ids from the process that owns ``text``'s conversation, if any."""
        route, key = self._route, _ROUTING_KEY.get()
        if route is None or key is None:
            return None
        try:
            ids = route(key, text)
        except RouteFailed as e:
            logger.debug(f"Tokenizing a prompt locally: {e}")
            with self._lock:
                self.route_failures += 1
            return None
        if ids is not None:
            with self._lock:
                self.routed += 1
        return ids

    def _encode_or_disable(self, tokenizer: OffsetTokenizer, text: str) -> list[int]:
        try:
            return self._encode(tokenizer, text)
        except Exception as e:
            # Raises, leaving the cache enabled, if the tokenizer rejects the
            # prompt: that is the request's error, not the cache's.
            ids = self._tokenize(tokenizer, text)[0]
            self.disabled = True
            logger.warning(
                f"Disabling the prefix token cache after an error: {e!r} ({self.summary()})"
            )
            return ids

    def _count_request(self) -> None:
        with self._lock:
            self.requests += 1
            report = self.requests % LOG_EVERY_REQUESTS == 0
            summary = self.summary() if report else None
        if summary is not None:
            logger.info(f"Prefix token cache: {summary}")

    def summary(self) -> str:
        """One-line hit-rate and occupancy summary. Caller need not hold the lock."""
        total = self.hits + self.misses
        rate = self.hits / total if total else 0.0
        return (
            f"{self.hits}/{total} hits ({rate:.1%}, {self.divergent_hits} divergent), "
            f"{self.resync_failures} resync fallbacks, {self.routed} routed, "
            f"{self.route_failures} route failures, "
            f"{len(self._entries)} entries, {self._total_chars} cached chars"
        )

    def _encode(self, tokenizer: OffsetTokenizer, text: str) -> list[int]:
        found = self._lookup(text)
        ids = None
        stored = None
        if found is not None:
            entry = found.entry
            tail_ids, tail_offsets = self._tokenize(tokenizer, text[found.split_char :])
            if self._resynced(entry, found.split_token, tail_ids) and self._aligned(
                found, tail_ids, tail_offsets
            ):
                if _ARRAY_SPLICE:
                    stored = entry.ids[: found.split_token]
                    stored.extend(tail_ids)
                    ids = stored.tolist()
                    if _INT32_IDS:
                        ids = TokenIdList(ids)
                        ids.int32 = stored
                else:
                    ids = entry.ids[: found.split_token].tolist() + list(tail_ids)
                base_token, base_char = found.split_token, found.split_char
        if ids is None:
            tail_ids, tail_offsets = self._tokenize(tokenizer, text)
            ids, base_token, base_char = tail_ids, 0, 0

        # Where a prompt extending this one will be re-tokenized from.
        split_token = len(ids) - self._config.overlap
        tail_index = split_token - base_token
        if tail_index >= 0:
            tail_index = _first_of_char(tail_offsets, tail_index)
            split_token = base_token + tail_index

        with self._lock:
            if found is None:
                self.misses += 1
            elif base_token == 0:
                self.misses += 1
                self.resync_failures += 1
            else:
                self.hits += 1
                self.divergent_hits += not found.extends
            # A prompt that reuses an entry supersedes it. A diverging prompt
            # may be a branch (a subagent off a shared history), so the entry
            # it was spliced from stays for the other branch.
            if found is not None and found.extends:
                self._remove(found.eid)
            if split_token > 0 and tail_index >= 0:
                checkpoint_token_indices, checkpoint_chars = self._checkpoints(
                    found, base_token, base_char, tail_offsets, len(ids)
                )
                self._insert(
                    text,
                    ids,
                    split_token,
                    base_char + tail_offsets[tail_index][0],
                    checkpoint_token_indices,
                    checkpoint_chars,
                    stored,
                )
        return ids

    @staticmethod
    def _tokenize(
        tokenizer: OffsetTokenizer, text: str
    ) -> tuple[list[int], Sequence[tuple[int, int]]]:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return enc["input_ids"], enc["offset_mapping"]

    def _checkpoints(
        self,
        found: _Match | None,
        base_token: int,
        base_char: int,
        tail_offsets: Sequence[tuple[int, int]],
        num_ids: int,
    ) -> tuple[array, array]:
        """Checkpoint tokens and chars of ``base_token`` cached ids plus a tail."""
        step = self._config.checkpoint_tokens
        # Spliced ids before base_token are the cached entry's, at the same chars.
        num_kept = -(-base_token // step) if base_token else 0
        tokens = array("i", found.entry.checkpoint_token_indices[:num_kept] if num_kept else ())
        chars = array("i", found.entry.checkpoint_chars[:num_kept] if num_kept else ())
        for index in range(num_kept * step, num_ids, step):
            tail_index = _first_of_char(tail_offsets, index - base_token)
            tokens.append(base_token + tail_index)
            chars.append(base_char + tail_offsets[tail_index][0])
        return tokens, chars

    def _resynced(self, entry: _Entry, split_token: int, tail_ids: Sequence[int]) -> bool:
        """Whether the tokenizer re-synchronized with the cached ids at the seam."""
        cached_tail = entry.ids[split_token:]
        span = min(self._config.resync, len(tail_ids), len(cached_tail))
        return span > 0 and list(tail_ids[:span]) == cached_tail[:span].tolist()

    @staticmethod
    def _aligned(
        found: _Match, tail_ids: Sequence[int], tail_offsets: Sequence[tuple[int, int]]
    ) -> bool:
        """Whether a diverging prompt's tail matches the cached ids up to the next checkpoint.

        The resync span can lie inside a run of equal tokens, such as a long
        run of spaces, where a tail shifted by a character still agrees with
        the cached ids; the shift shows where the run ends, or as tokens that
        start one character off. The next checkpoint starts inside the common
        prefix, so an unshifted tail has the cached ids up to it and starts a
        token there.
        """
        if found.next_checkpoint is None:
            return True
        token, char = found.next_checkpoint
        index = token - found.split_token
        return (
            index < len(tail_ids)
            and tail_offsets[index][0] == char - found.split_char
            and list(tail_ids[:index]) == found.entry.ids[found.split_token : token].tolist()
        )

    def _bucket_key(self, text: str) -> int:
        return hash(text[: self._config.min_chars])

    def _lookup(self, text: str) -> _Match | None:
        """Best cached entry to splice ``text`` from, promoted to MRU if ``text`` extends it.

        The entry with the longest re-tokenization prefix that ``text`` keeps
        wins. Without one, or when it leaves more than two checkpoint intervals
        to re-tokenize, the entries ``text`` diverges from are also tried, each
        split at the checkpoint one interval before its last checkpoint inside
        the common prefix, and the latest split wins.
        """
        with self._lock:
            candidates = list(self._buckets.get(self._bucket_key(text), {}).items())
        # Entries are immutable, so the comparisons run outside the lock. The
        # probe check rejects other conversations in O(PROBE_CHARS); only a true
        # prefix pays the full compare over the shared preamble.
        best: _Match | None = None
        for eid, entry in candidates:
            n = entry.split_char
            if (
                n <= len(text)
                and (best is None or n > best.split_char)
                and text[n - len(entry.probe) : n] == entry.probe
                and text.startswith(entry.text[:n])
            ):
                best = _Match(eid, entry, entry.split_token, entry.split_char, True)
        if best is None or self._leaves_long_tail(best, text):
            for eid, entry in candidates:
                match = self._divergent_match(eid, entry, text)
                if match is not None and (best is None or match.split_token > best.split_token):
                    best = match
        # A diverging prompt is not promoted: when the history was rewritten
        # rather than branched, the entry it diverged from is dead.
        if best is not None and best.extends:
            with self._lock:
                if best.eid in self._entries:
                    self._entries.move_to_end(best.eid)
            # If it was evicted meanwhile the entry is still a valid prefix to
            # splice from; _remove() on it is a no-op.
        return best

    def _leaves_long_tail(self, match: _Match, text: str) -> bool:
        """Whether ``text`` extends ``match`` by more than two checkpoint intervals."""
        entry = match.entry
        chars_per_token = entry.split_char / max(entry.split_token, 1)
        tail_tokens = (len(text) - match.split_char) / max(chars_per_token, 1e-6)
        return tail_tokens > 2 * self._config.checkpoint_tokens

    def _divergent_match(self, eid: int, entry: _Entry, text: str) -> _Match | None:
        common = _common_prefix_len(entry.text, text)
        # The checkpoint after the split must start inside the common prefix.
        index = bisect.bisect_left(entry.checkpoint_chars, common) - 2
        if index < 1:
            return None
        return _Match(
            eid,
            entry,
            entry.checkpoint_token_indices[index],
            entry.checkpoint_chars[index],
            False,
            (entry.checkpoint_token_indices[index + 1], entry.checkpoint_chars[index + 1]),
        )

    def _insert(
        self,
        text: str,
        ids: Sequence[int],
        split_token: int,
        split_char: int,
        checkpoint_token_indices: array,
        checkpoint_chars: array,
        stored: array | None = None,
    ) -> None:
        """Caller holds the lock. ``stored``, when given, is ``ids`` as an int32 array already."""
        key = self._bucket_key(text)
        bucket = self._buckets.setdefault(key, {})
        if any(entry.text == text for entry in bucket.values()):
            return  # concurrent misses on the same prompt
        eid, self._next_id = self._next_id, self._next_id + 1
        probe = text[max(split_char - PROBE_CHARS, 0) : split_char]
        entry = _Entry(
            text,
            stored if stored is not None else array("i", ids),
            split_token,
            split_char,
            probe,
            checkpoint_token_indices,
            checkpoint_chars,
        )
        self._entries[eid] = entry
        bucket[eid] = entry
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
        bucket.pop(eid, None)
        if not bucket:
            del self._buckets[key]


def _common_prefix_len(a: str, b: str) -> int:
    """Length of the longest common prefix, compared in blocks."""
    n, i, block = min(len(a), len(b)), 0, _COMPARE_BLOCK_CHARS
    while i < n and a[i : i + block] == b[i : i + block]:
        i += block
    # The first mismatch, if any, lies in the block at i; halve it to find it.
    lo, hi = min(i, n), min(i + block, n)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[lo:mid] == b[lo:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _first_of_char(offsets: Sequence[tuple[int, int]], index: int) -> int:
    """First token of the character token ``index`` starts in.

    A character tokenized into several tokens (a byte-fallback character, an
    expanded ligature) reports the same start for each; a split must not
    separate them, or the tail would repeat the tokens of that character.
    """
    while index > 0 and offsets[index - 1][0] == offsets[index][0]:
        index -= 1
    return index


def _splices_wrongly(tokenizer: OffsetTokenizer, config: PrefixTokenCacheConfig) -> bool:
    """Whether a tail the cache would accept at the seam can differ from full tokenization.

    Tokenizers that alter every text they are called on, such as one whose
    normalizer prepends ``▁``, shift a tail by a character. Inside a run of
    equal tokens the shift passes the resync check and only shows at the end
    of the run, so the probe splits inside a long run of spaces.
    """
    text = "x" + " " * 4096 + "y"
    ids, offsets = PrefixTokenCache._tokenize(tokenizer, text)
    split = next((i for i, (start, _) in enumerate(offsets) if start >= 1024), None)
    if split is None:
        return False
    split = _first_of_char(offsets, split)
    tail_ids = PrefixTokenCache._tokenize(tokenizer, text[offsets[split][0] :])[0]
    span = min(config.resync, len(tail_ids), len(ids) - split)
    resynced = list(tail_ids[:span]) == list(ids[split : split + span])
    return resynced and list(ids[:split]) + list(tail_ids) != list(ids)


def create_prefix_token_cache(tokenizer: object) -> PrefixTokenCache | None:
    """Return a cache for ``tokenizer`` if the tokenizer can support one.

    Returns None when there is no tokenizer, it cannot report offsets, or a
    spliced tail can differ from tokenizing the whole prompt.
    Raises ``ValueError`` for a malformed env override.
    """
    if tokenizer is None:
        return None
    if not getattr(tokenizer, "is_fast", False):
        logger.warning(
            "enable_tokenization_cache is set but the tokenizer is not a fast "
            "tokenizer, which the prefix token cache needs for offset mappings; "
            "disabling it."
        )
        return None
    config = PrefixTokenCacheConfig.from_env()
    try:
        unsafe = _splices_wrongly(tokenizer, config)
    except Exception as e:
        logger.warning(
            f"Could not probe the tokenizer for the prefix token cache ({e!r}); disabling it."
        )
        return None
    if unsafe:
        logger.warning(
            "enable_tokenization_cache is set but the tokenizer changes the start "
            "of every text it tokenizes (for example, its normalizer prepends a "
            "space marker), so spliced token ids could differ from tokenizing "
            "the whole prompt; disabling the prefix token cache."
        )
        return None
    logger.info(f"Prefix token cache enabled: {config}")
    return PrefixTokenCache(config)
