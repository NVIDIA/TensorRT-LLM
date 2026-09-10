r"""Prefix cache primitives for KV-cache hit analysis.

Mirrors TRT-LLM's KV cache accounting so the offline upper bound
:func:`compute_cache_hit_upper_bound` always dominates the engine-measured
``num_reused_blocks / (num_reused + num_missed)`` per request.

Key invariants:
* Every committed sequence contributes ``ceil(L / block)`` blocks (the
  partial trailing block is hashed by its actual token content, not
  padding).
* A future request's reused-block count is
  ``ceil(matched_tokens / block)``, with ``+1`` on a clean-aligned match
  (the engine occasionally counts the next block as reused when cache
  state lets the partial-overlap hash hit). Capped at the request's
  total block count.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence


class _TokenNode:
    """Compressed-edge radix-tree node, keyed by first token of each child."""

    __slots__ = ("edge_tail", "children")

    def __init__(self, edge_tail: List[int]) -> None:
        # ``edge_tail`` is the token sequence on the incoming edge AFTER the
        # first token (which is the key in the parent's children dict).
        self.edge_tail: List[int] = edge_tail
        self.children: Dict[int, "_TokenNode"] = {}


class TokenPrefixCache:
    """Infinite-capacity token-level prefix cache.

    Stores arbitrary token sequences and answers ``longest matching prefix``
    queries in O(L) time, where L = query length. Each ``insert`` adds the
    full token list; a later ``match`` returns the longest prefix length
    (in tokens) shared with any previously inserted sequence.
    """

    def __init__(self) -> None:
        self._root = _TokenNode([])

    def insert(self, tokens: Sequence[int]) -> None:
        """Insert *tokens* as a complete sequence."""
        toks = list(tokens)
        n = len(toks)
        if n == 0:
            return
        node = self._root
        i = 0
        while i < n:
            first = toks[i]
            child = node.children.get(first)
            if child is None:
                node.children[first] = _TokenNode(toks[i + 1 : n])
                return
            tail = child.edge_tail
            j = 0
            while j < len(tail) and i + 1 + j < n and tail[j] == toks[i + 1 + j]:
                j += 1
            if j == len(tail):
                node = child
                i += 1 + j
                continue
            # Partial edge match: split the edge into two.
            new_lower = _TokenNode(tail[j + 1 :])
            new_lower.children = child.children
            child.children = {tail[j]: new_lower}
            child.edge_tail = tail[:j]
            i += 1 + j
            if i < n:
                child.children[toks[i]] = _TokenNode(toks[i + 1 :])
            return

    def match(self, query: Sequence[int]) -> int:
        """Return the longest prefix of *query* (in tokens) seen before."""
        q = list(query)
        n = len(q)
        if n == 0:
            return 0
        node = self._root
        matched = 0
        while matched < n:
            child = node.children.get(q[matched])
            if child is None:
                break
            tail = child.edge_tail
            matched += 1
            j = 0
            while j < len(tail) and matched + j < n and tail[j] == q[matched + j]:
                j += 1
            matched += j
            if j < len(tail):
                break
            node = child
        return matched


def engine_aligned_block_count(token_len: int, tokens_per_block: int) -> int:
    """Total blocks an engine allocates for a request of *token_len* tokens.

    TRT-LLM counts ``ceil(L / block_size)`` blocks (the trailing partial
    block is allocated and hashed by its content).
    """
    if token_len <= 0:
        return 0
    return math.ceil(token_len / tokens_per_block)


def engine_aligned_hit_blocks(
    matched_tokens: int,
    n_total_blocks: int,
    tokens_per_block: int,
) -> int:
    """Strict upper bound on the engine-reported ``num_reused_blocks``.

    A matched prefix of length M tokens covers ``ceil(M / block)`` blocks.
    When M lands exactly on a block boundary the engine sometimes also
    counts the next block (random partial-overlap hash hit, ~0.04% of
    measurements with KV-cache-aware routing); we always include that
    extra boundary block so UB >= measured. Capped at ``n_total_blocks``.
    """
    if matched_tokens <= 0 or n_total_blocks <= 0:
        return 0
    hit = math.ceil(matched_tokens / tokens_per_block)
    if matched_tokens % tokens_per_block == 0:
        hit += 1
    return min(hit, n_total_blocks)


def validate_tokens_per_block(tokens_per_block: int) -> None:
    if tokens_per_block <= 0:
        raise ValueError(f"tokens_per_block must be positive, got {tokens_per_block}")
