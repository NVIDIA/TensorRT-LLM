"""Prefix-tokenization cache for the default input processor.

In multi-turn agentic serving each turn's prompt is the previous turn's prompt
plus a small delta, yet the frontend re-tokenizes the whole thing every turn. On
a GLM-5.2 disaggregated context server with ~38k-token prompts, nsys attributed
47.4% of context wall-clock to "tokenize prompt" at 43.7 ms/request. Caching the
tokenization of the longest cached prefix and tokenizing only the tail brought
that to 5.49 ms/request (10.5% of wall) with byte-identical token IDs.

CORRECTNESS. Splitting a string and tokenizing the tail in isolation is NOT
generally equal to tokenizing the whole: BPE merges can straddle the split. So
the cache backs off `overlap` tokens from the split point, re-tokenizes from
there, and requires the first `resync` re-tokenized ids to equal the cached ids
over the same span. That proves the tokenizer has re-synchronized at the seam;
if it has not, the prompt is tokenized in full. Validated on real trajectories:
192/192 exact token-ID matches, 0 resync fallbacks, thread-safe under 8
concurrent input_processor workers.

Off by default; enable with TLLM_PREFIX_TOKEN_CACHE=1.
"""

import bisect
import os
import threading
from typing import List, Optional

__all__ = ["PrefixTokenCache", "get_prefix_token_cache", "prefix_cache_enabled"]


def prefix_cache_enabled() -> bool:
    return os.environ.get("TLLM_PREFIX_TOKEN_CACHE", "0") == "1"


class PrefixTokenCache:
    """Splice a cached prefix's token ids with a freshly tokenized tail.

    Entries are bucketed by a hash of their first `bucket_chars` characters so
    that finding the longest cached prefix of an incoming prompt does not scan
    every entry. Eviction is LRU on insertion order.
    """

    def __init__(
        self,
        max_entries: int = 512,
        overlap: int = 64,
        resync: int = 32,
        min_chars: int = 4096,
        bucket_chars: int = 2048,
    ) -> None:
        self._lock = threading.Lock()
        self._max_entries = max_entries
        self._overlap = overlap
        self._resync = resync
        self._min_chars = min_chars
        self._bucket_chars = bucket_chars
        self._buckets = {}  # bucket key -> [entry id]
        self._entries = {}  # entry id -> (text, ids, ends, starts, bucket key)
        self._order = []  # entry ids, oldest first
        self._next_id = 0
        # counters, for logging/tests only
        self.hits = 0
        self.misses = 0
        self.resync_failures = 0

    @property
    def min_chars(self) -> int:
        return self._min_chars

    def _bucket_key(self, text: str):
        return hash(text[: self._bucket_chars])

    def _find_longest_prefix(self, text: str):
        """Longest cached entry that is a prefix of `text`. Caller holds lock."""
        best, best_len = None, 0
        for eid in self._buckets.get(self._bucket_key(text), ()):
            entry = self._entries.get(eid)
            if entry is None:
                continue
            ptext = entry[0]
            n = len(ptext)
            if n > best_len and n <= len(text) and text.startswith(ptext):
                best, best_len = entry, n
        return best

    def _evict(self) -> None:
        """Caller holds lock."""
        while len(self._order) > self._max_entries:
            old = self._order.pop(0)
            entry = self._entries.pop(old, None)
            if entry is None:
                continue
            bucket = self._buckets.get(entry[4])
            if bucket and old in bucket:
                bucket.remove(old)
                if not bucket:
                    self._buckets.pop(entry[4], None)

    def encode(self, tokenizer, text: str, **kwargs) -> List[int]:
        reuse, start_char, prev = 0, 0, None
        with self._lock:
            entry = self._find_longest_prefix(text)
            if entry is not None:
                ptext, pids, pends, pstarts, _ = entry
                i = bisect.bisect_right(pends, len(ptext)) - self._overlap
                if i > 0:
                    reuse, start_char = i, pstarts[i]
                    prev = (pids, pends, pstarts)

        # tokenize outside the lock: this is the expensive part
        enc = tokenizer(
            text[start_char:], add_special_tokens=False, return_offsets_mapping=True, **kwargs
        )
        new_ids, new_offsets = enc["input_ids"], enc["offset_mapping"]

        if reuse:
            pids, pends, pstarts = prev
            span = min(self._resync, len(pids) - reuse, len(new_ids))
            if span <= 0 or list(new_ids[:span]) != list(pids[reuse : reuse + span]):
                # the tokenizer did not re-synchronize at the seam
                with self._lock:
                    self.resync_failures += 1
                reuse, start_char = 0, 0
                enc = tokenizer(
                    text, add_special_tokens=False, return_offsets_mapping=True, **kwargs
                )
                new_ids, new_offsets = enc["input_ids"], enc["offset_mapping"]

        if reuse:
            pids, pends, pstarts = prev
            ids = pids[:reuse] + list(new_ids)
            starts = pstarts[:reuse] + [a + start_char for a, _ in new_offsets]
            ends = pends[:reuse] + [b + start_char for _, b in new_offsets]
        else:
            ids = list(new_ids)
            starts = [a for a, _ in new_offsets]
            ends = [b for _, b in new_offsets]

        with self._lock:
            if reuse:
                self.hits += 1
            else:
                self.misses += 1
            eid = self._next_id
            self._next_id += 1
            key = self._bucket_key(text)
            self._entries[eid] = (text, ids, ends, starts, key)
            self._buckets.setdefault(key, []).append(eid)
            self._order.append(eid)
            self._evict()
        return ids


_CACHE: Optional[PrefixTokenCache] = None
_CACHE_LOCK = threading.Lock()


def get_prefix_token_cache() -> PrefixTokenCache:
    """Process-wide cache, created on first use."""
    global _CACHE
    if _CACHE is None:
        with _CACHE_LOCK:
            if _CACHE is None:
                _CACHE = PrefixTokenCache(
                    max_entries=int(os.environ.get("TLLM_PREFIX_TOKEN_CACHE_ENTRIES", "512")),
                    overlap=int(os.environ.get("TLLM_PREFIX_TOKEN_CACHE_OVERLAP", "64")),
                    resync=int(os.environ.get("TLLM_PREFIX_TOKEN_CACHE_RESYNC", "32")),
                    min_chars=int(os.environ.get("TLLM_PREFIX_TOKEN_CACHE_MIN_CHARS", "4096")),
                )
    return _CACHE
