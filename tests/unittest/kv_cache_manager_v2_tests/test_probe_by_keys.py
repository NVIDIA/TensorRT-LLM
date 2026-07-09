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
"""Pure unit tests for BlockRadixTree.match_block_keys (hash-free probing)."""

import unittest
from collections.abc import Iterator
from importlib.util import find_spec
from typing import TYPE_CHECKING, cast

if not TYPE_CHECKING and find_spec("kv_cache_manager_v2") is not None:
    from kv_cache_manager_v2 import ReuseScope, TokenId
    from kv_cache_manager_v2._block_radix_tree import (
        Block,
        BlockRadixTree,
        remove_subtree,
        sequence_to_blockchain_keys,
    )
    from kv_cache_manager_v2._life_cycle_registry import LifeCycleRegistry
else:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import ReuseScope, TokenId
    from tensorrt_llm.runtime.kv_cache_manager_v2._block_radix_tree import (
        Block,
        BlockRadixTree,
        remove_subtree,
        sequence_to_blockchain_keys,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._life_cycle_registry import LifeCycleRegistry


class _EmptyLifeCycles:
    size = 0

    @property
    def ssm_life_cycle_id(self) -> None:
        return None

    def attention_life_cycles(self) -> Iterator[tuple[object, object]]:
        return iter(())


TPB = 2


class TestMatchBlockKeys(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = BlockRadixTree(cast(LifeCycleRegistry, _EmptyLifeCycles()), tokens_per_block=TPB)
        self.scope = ReuseScope(lora_id=7, salt=11)
        self.tokens = [TokenId(t) for t in (1, 2, 3, 4, 5, 6)]
        root = self.tree.add_or_get_existing(self.scope)
        self.blocks: list[Block] = []
        prev: object = root
        for beg in range(0, len(self.tokens), TPB):
            block = Block(self.tokens[beg : beg + TPB], prev)
            self.blocks.append(block)
            prev = block

    def _committed_keys(self) -> list[bytes]:
        return [block.key for block in self.blocks]

    def test_matches_full_chain_like_match(self) -> None:
        expected = self.tree.match(self.scope, self.tokens)
        result = self.tree.match_block_keys(self.scope, self._committed_keys())
        self.assertEqual(result.blocks, expected.blocks)
        self.assertEqual(result.num_tokens, expected.num_tokens)
        self.assertEqual(result.num_tokens, len(self.tokens))

    def test_key_prefix_matches_shorter_chain(self) -> None:
        result = self.tree.match_block_keys(self.scope, self._committed_keys()[:2])
        self.assertEqual(result.blocks, self.blocks[:2])
        self.assertEqual(result.num_tokens, 2 * TPB)

    def test_wrong_scope_matches_nothing(self) -> None:
        other_scope = ReuseScope(lora_id=7, salt=12)
        result = self.tree.match_block_keys(other_scope, self._committed_keys())
        self.assertEqual(result.blocks, [])
        self.assertEqual(result.num_tokens, 0)

    def test_unknown_key_stops_the_walk(self) -> None:
        keys = self._committed_keys()
        keys[1] = b"\x00" * len(keys[1])
        result = self.tree.match_block_keys(self.scope, keys)
        self.assertEqual(result.blocks, self.blocks[:1])
        self.assertEqual(result.num_tokens, TPB)

    def test_removed_subtree_degrades_to_deepest_survivor(self) -> None:
        # Simulate eviction dropping block 1 (and thus its descendants): the
        # walk must stop at the deepest surviving block instead of reporting
        # a match for detached nodes.
        remove_subtree(self.blocks[1])
        result = self.tree.match_block_keys(self.scope, self._committed_keys())
        self.assertEqual(result.blocks, self.blocks[:1])
        self.assertEqual(result.num_tokens, TPB)

    def test_partial_tail_block_counts_partial_tokens(self) -> None:
        partial_tokens = [TokenId(9)]
        partial = Block(partial_tokens, self.blocks[-1])
        keys = self._committed_keys() + [partial.key]
        result = self.tree.match_block_keys(self.scope, keys)
        self.assertEqual(result.blocks, self.blocks + [partial])
        self.assertEqual(result.num_tokens, len(self.tokens) + len(partial_tokens))

    def test_keys_agree_with_sequence_to_blockchain_keys(self) -> None:
        # committed_block_keys() feeds keys produced during commit; verify the
        # tree-node keys equal what hashing the token stream would produce, so
        # the by-keys walk is exactly the hash walk minus the hashing.
        hashed = [key for token_block, key in sequence_to_blockchain_keys(TPB, self.scope, self.tokens) if token_block]
        self.assertEqual(self._committed_keys(), hashed)

    def test_empty_keys_matches_nothing(self) -> None:
        result = self.tree.match_block_keys(self.scope, [])
        self.assertEqual(result.blocks, [])
        self.assertEqual(result.num_tokens, 0)


if __name__ == "__main__":
    unittest.main()
