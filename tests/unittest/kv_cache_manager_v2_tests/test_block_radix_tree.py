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
"""Pure unit tests for BlockRadixTree lifecycle (clear / prune) semantics."""

import unittest
from collections.abc import Iterator
from importlib.util import find_spec
from typing import TYPE_CHECKING, cast

import pytest

pytestmark = pytest.mark.cpu_only


if not TYPE_CHECKING and find_spec("kv_cache_manager_v2") is not None:
    from kv_cache_manager_v2 import TokenId
    from kv_cache_manager_v2._block_radix_tree import Block, BlockRadixTree, ReuseScope
    from kv_cache_manager_v2._life_cycle_registry import LifeCycleRegistry
else:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import TokenId
    from tensorrt_llm.runtime.kv_cache_manager_v2._block_radix_tree import (
        Block,
        BlockRadixTree,
        ReuseScope,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._life_cycle_registry import LifeCycleRegistry


class _EmptyLifeCycles:
    size = 0

    @property
    def ssm_life_cycle_id(self) -> None:
        return None

    def attention_life_cycles(self) -> Iterator[tuple[object, object]]:
        return iter(())


class TestBlockRadixTreeClear(unittest.TestCase):
    def _make_tree(self, tokens_per_block: int = 2) -> BlockRadixTree:
        return BlockRadixTree(
            cast(LifeCycleRegistry, _EmptyLifeCycles()), tokens_per_block=tokens_per_block
        )

    @staticmethod
    def _add_chain(
        tree: BlockRadixTree, scope: ReuseScope, num_full_blocks: int
    ) -> tuple[object, list[Block]]:
        """Attach a chain of `num_full_blocks` full blocks under a fresh root."""
        root = tree.add_or_get_existing(scope)
        blocks: list[Block] = []
        prev = root
        for i in range(num_full_blocks):
            block = Block([TokenId(2 * i + 1), TokenId(2 * i + 2)], prev)
            blocks.append(block)
            prev = block
        return root, blocks

    def test_clear_empty_tree_returns(self) -> None:
        tree = self._make_tree()
        tree.clear()
        self.assertEqual(tree.next, {})

    def test_clear_with_childless_root_returns(self) -> None:
        # Regression: a RootBlock is published by add_or_get_existing before any
        # child is attached. A childless root has no child whose detachment could
        # auto-prune it, so clear() used to re-select it forever and never return.
        tree = self._make_tree()
        tree.add_or_get_existing(ReuseScope(lora_id=0, salt=None))
        self.assertEqual(len(tree.next), 1)
        tree.clear()
        self.assertEqual(tree.next, {})

    def test_clear_removes_single_chain(self) -> None:
        tree = self._make_tree()
        scope = ReuseScope(lora_id=0, salt=None)
        _, blocks = self._add_chain(tree, scope, 3)
        self.assertEqual(len(tree.next), 1)
        self.assertFalse(blocks[0].is_orphan)

        tree.clear()
        self.assertEqual(tree.next, {})
        # Detached leaves/blocks must no longer be attached to the tree.
        self.assertTrue(all(block.is_orphan for block in blocks))

    def test_clear_mixed_childless_and_populated_roots(self) -> None:
        tree = self._make_tree()
        self._add_chain(tree, ReuseScope(lora_id=1, salt=None), 2)
        tree.add_or_get_existing(ReuseScope(lora_id=2, salt=None))  # childless
        self._add_chain(tree, ReuseScope(lora_id=3, salt=None), 4)
        tree.add_or_get_existing(ReuseScope(lora_id=4, salt=None))  # childless
        self.assertEqual(len(tree.next), 4)

        tree.clear()
        self.assertEqual(tree.next, {})

    def test_clear_removes_multi_branch_subtrees(self) -> None:
        tree = self._make_tree()
        scope_a = ReuseScope(lora_id=1, salt=None)
        root_a = tree.add_or_get_existing(scope_a)
        b1 = Block([TokenId(1), TokenId(2)], root_a)
        Block([TokenId(3), TokenId(4)], b1)
        b1_sibling = Block([TokenId(5), TokenId(6)], root_a)  # second branch
        root_b = tree.add_or_get_existing(ReuseScope(lora_id=2, salt=None))
        Block([TokenId(9), TokenId(10)], root_b)
        self.assertEqual(len(root_a.next), 2)
        self.assertEqual(len(tree.next), 2)

        tree.clear()
        self.assertEqual(tree.next, {})
        self.assertTrue(b1.is_orphan)
        self.assertTrue(b1_sibling.is_orphan)

    def test_clear_is_idempotent_and_tree_reusable(self) -> None:
        tree = self._make_tree()
        scope = ReuseScope(lora_id=5, salt=7)
        root1, _ = self._add_chain(tree, scope, 2)
        tree.clear()
        tree.clear()
        self.assertEqual(tree.next, {})

        # The same reuse scope must seed a fresh root after a clear.
        root2 = tree.add_or_get_existing(scope)
        self.assertIsNot(root2, root1)
        block = Block([TokenId(1), TokenId(2)], root2)
        match = tree.match(scope, [TokenId(1), TokenId(2)])
        self.assertEqual(match.blocks, [block])
        self.assertEqual(match.num_tokens, 2)
        tree.clear()
        self.assertEqual(tree.next, {})


if __name__ == "__main__":
    unittest.main()
