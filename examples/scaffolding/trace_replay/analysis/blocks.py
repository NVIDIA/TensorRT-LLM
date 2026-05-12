r"""Block-level prefix cache primitives for KV-cache hit analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple


@dataclass
class _TrieNode:
    children: Dict[Tuple[int, ...], "_TrieNode"] = field(default_factory=dict)


class BlockPrefixCache:
    """Infinite-capacity prefix cache backed by a block radix tree.

    Models an idealized TRT-LLM C++ KV cache manager where every previously
    inserted block remains resident forever. Each block is a tuple of
    ``tokens_per_block`` token IDs; the radix tree uses tuples as edge keys.
    """

    def __init__(self) -> None:
        self._root = _TrieNode()
        self.cached_blocks = 0

    def insert_and_count_hit(self, blocks: Sequence[Tuple[int, ...]]) -> Tuple[int, int]:
        """Insert *blocks* and return ``(hit_blocks, new_cached_blocks)``."""
        node = self._root
        hit_blocks = 0
        for block in blocks:
            child = node.children.get(block)
            if child is None:
                break
            node = child
            hit_blocks += 1

        new_blocks = 0
        for block in blocks[hit_blocks:]:
            child = _TrieNode()
            node.children[block] = child
            node = child
            new_blocks += 1

        self.cached_blocks += new_blocks
        return hit_blocks, new_blocks

    def insert_only(self, blocks: Sequence[Tuple[int, ...]]) -> int:
        """Pre-warm the cache with *blocks*; returns count of newly inserted blocks."""
        _, new_blocks = self.insert_and_count_hit(blocks)
        return new_blocks


def full_blocks(tokens: Sequence[int], tokens_per_block: int) -> List[Tuple[int, ...]]:
    """Cut *tokens* into back-to-back complete blocks of ``tokens_per_block``.

    Trailing tokens that would not fill a complete block are discarded — they
    cannot participate in block-level prefix reuse.
    """
    block_count = len(tokens) // tokens_per_block
    return [
        tuple(tokens[i * tokens_per_block : (i + 1) * tokens_per_block]) for i in range(block_count)
    ]


def reusable_token_len(token_len: int, exclude_last_token_from_blocks: bool) -> int:
    """Tokens eligible for block formation under the chosen rule."""
    if exclude_last_token_from_blocks and token_len:
        return token_len - 1
    return max(token_len, 0)


def validate_tokens_per_block(tokens_per_block: int) -> None:
    if tokens_per_block <= 0:
        raise ValueError(f"tokens_per_block must be positive, got {tokens_per_block}")
