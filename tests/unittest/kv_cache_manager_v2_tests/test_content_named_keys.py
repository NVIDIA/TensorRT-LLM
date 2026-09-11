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
"""Pure unit tests for naming a chain of blocks by content instead of tokens.

``BlockRadixTree.match_keys`` is reachable from Python only in the pure-Python
implementation; the two are compared in ``test_content_named_lookup.py``.
"""

from collections.abc import Iterator
from typing import cast

import pytest

from tensorrt_llm.runtime.kv_cache_manager_v2 import TokenId, TokenIdExt
from tensorrt_llm.runtime.kv_cache_manager_v2._block_radix_tree import (
    Block,
    BlockKey,
    BlockRadixTree,
    ReuseScope,
    sequence_to_blockchain_keys,
)
from tensorrt_llm.runtime.kv_cache_manager_v2._life_cycle_registry import LifeCycleRegistry

pytestmark = pytest.mark.cpu_only

TOKENS_PER_BLOCK = 4
SCOPE = ReuseScope(lora_id=7, salt=11)
# A well-formed key of the right width that names nothing in any tree here.
ABSENT_KEY: BlockKey = bytes(range(32))


class _EmptyLifeCycles:
    """No life cycle at all, so pruning cannot shorten a match.

    Page residency is the manager's concern; what is under test here is the walk.
    """

    size = 0

    @property
    def ssm_life_cycle_id(self) -> None:
        return None

    def attention_life_cycles(self) -> Iterator[tuple[object, object]]:
        return iter(())


def _new_tree() -> BlockRadixTree:
    return BlockRadixTree(
        cast(LifeCycleRegistry, _EmptyLifeCycles()), tokens_per_block=TOKENS_PER_BLOCK
    )


def _tokens(count: int, first: int = 0) -> list[TokenIdExt]:
    return [TokenId(first + i) for i in range(count)]


def _chain(tokens: list[TokenIdExt], scope: ReuseScope = SCOPE) -> list[BlockKey]:
    """The root key followed by one key per block, as a holder receives it."""
    return [key for _, key in sequence_to_blockchain_keys(TOKENS_PER_BLOCK, scope, tokens)]


def _populate(tree: BlockRadixTree, tokens: list[TokenIdExt], scope: ReuseScope = SCOPE) -> None:
    """Insert ``tokens`` block by block; the tree owns what it inserts."""
    node = tree.add_or_get_existing(scope)
    for beg in range(0, len(tokens), TOKENS_PER_BLOCK):
        node = Block(tokens[beg : beg + TOKENS_PER_BLOCK], node)


def test_a_key_chain_and_its_tokens_match_the_same_blocks() -> None:
    tokens = _tokens(3 * TOKENS_PER_BLOCK)
    tree = _new_tree()
    _populate(tree, tokens)

    by_keys = tree.match_keys(_chain(tokens))
    by_tokens = tree.match(SCOPE, tokens)

    # Block-aligned, so even num_lookup_tokens -- the one field derived
    # differently (chain length there, token count here) -- must coincide.
    assert by_keys == by_tokens
    assert by_keys.num_tokens == 3 * TOKENS_PER_BLOCK


def test_an_empty_chain_names_nothing() -> None:
    # The capability probe at assembly asks exactly this; it must answer, not raise.
    match = _new_tree().match_keys([])
    assert match.blocks == []
    assert match.num_tokens == 0
    assert match.num_lookup_tokens == 0
    assert match.num_reusable_tokens_before_pruning == 0
    assert match.num_reusable_tokens_before_hybrid_pruning == 0


def test_the_root_alone_names_no_block() -> None:
    tokens = _tokens(2 * TOKENS_PER_BLOCK)
    tree = _new_tree()
    _populate(tree, tokens)

    match = tree.match_keys(_chain(tokens)[:1])
    assert match.blocks == []
    assert match.num_tokens == 0
    # One key is one namespace and zero blocks, so nothing was even looked up.
    assert match.num_lookup_tokens == 0


def test_a_chain_rooted_outside_this_scope_matches_nothing() -> None:
    tokens = _tokens(2 * TOKENS_PER_BLOCK)
    tree = _new_tree()
    _populate(tree, tokens)

    other_scope = ReuseScope(lora_id=7, salt=12)
    match = tree.match_keys(_chain(tokens, other_scope))
    assert match.blocks == []
    assert match.num_tokens == 0
    # What was asked for is still what was asked for.
    assert match.num_lookup_tokens == 2 * TOKENS_PER_BLOCK


def test_a_chain_stops_at_the_first_link_the_tree_does_not_hold() -> None:
    tokens = _tokens(3 * TOKENS_PER_BLOCK)
    tree = _new_tree()
    _populate(tree, tokens)

    forked = _chain(tokens)[:-1] + [ABSENT_KEY]
    match = tree.match_keys(forked)
    assert match.num_tokens == 2 * TOKENS_PER_BLOCK
    assert len(match.blocks) == 2
    assert match.num_lookup_tokens == 3 * TOKENS_PER_BLOCK
    # Below the lookup length: this is where the chain forks from the tree.
    assert match.num_reusable_tokens_before_pruning == 2 * TOKENS_PER_BLOCK


def test_a_chain_shorter_than_the_tree_ends_on_a_block_boundary() -> None:
    tokens = _tokens(3 * TOKENS_PER_BLOCK)
    tree = _new_tree()
    _populate(tree, tokens)

    match = tree.match_keys(_chain(tokens)[:3])
    assert len(match.blocks) == 2
    assert match.num_tokens == 2 * TOKENS_PER_BLOCK
    assert match.num_lookup_tokens == 2 * TOKENS_PER_BLOCK
    # Whole blocks only: nothing may end part-way in, which is what lets a
    # receiver anchor its block list at the start of the extent.
    assert match.num_tokens % TOKENS_PER_BLOCK == 0


def test_a_chain_never_partial_matches_a_sibling() -> None:
    """The walk carries placeholder tokens; a sibling starting with them must not match.

    Partial match scans tokens, so leaving it defaulted would have the
    placeholders match whatever block happens to begin with them.
    """
    tree = _new_tree()
    root = tree.add_or_get_existing(SCOPE)
    # Leads with the placeholder ``match_keys`` walks with, and diverges at the end.
    Block([TokenId(0), TokenId(0), TokenId(0), TokenId(9)], root)

    # By tokens, three of the placeholders do match this block.
    by_tokens = tree.match(SCOPE, [TokenId(0)] * 3, enable_partial_match=True)
    assert by_tokens.num_tokens == 3

    by_keys = tree.match_keys([root.key, ABSENT_KEY])
    assert by_keys.blocks == []
    assert by_keys.num_tokens == 0


def test_a_named_partial_block_is_counted_as_a_whole_one() -> None:
    """A key for a partial block over-reports; only whole blocks are ever named.

    The walk has no tokens to measure with, so every link counts a full block.
    The caller strips the partial tail; this pins what happens if it stops.
    """
    tokens = _tokens(TOKENS_PER_BLOCK + 2)
    tree = _new_tree()
    root = tree.add_or_get_existing(SCOPE)
    full = Block(tokens[:TOKENS_PER_BLOCK], root)
    partial = Block(tokens[TOKENS_PER_BLOCK:], full)
    assert len(partial.tokens) < TOKENS_PER_BLOCK

    match = tree.match_keys(_chain(tokens))
    assert match.blocks == [full, partial]
    assert match.num_tokens == 2 * TOKENS_PER_BLOCK
