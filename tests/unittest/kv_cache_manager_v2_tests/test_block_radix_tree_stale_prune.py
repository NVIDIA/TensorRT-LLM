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
"""Pure unit tests for the stale-block tail prune in ``clear_stale_blocks_after_page_unlink``.

Regression guard for a multi-life-cycle tree: unlinking one life cycle's page
from a childless tip must not detach the block while another life cycle still
holds a live page there. Hybrid models (Kimi K3: MLA attention + KDA/SSM) are
the ones that hit this, because they are the ones with more than one life cycle.

The C++ mirror was fixed upstream in PR #17323
(``Block::clearStaleBlocksAfterPageUnlink`` in
``cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/blockRadixTree.cpp``);
this covers the Python implementation, which is the one selected by
``TLLM_KV_CACHE_MANAGER_V2_BACKEND=python``.
"""

import unittest
from collections.abc import Iterator
from importlib.util import find_spec
from typing import TYPE_CHECKING, cast

if not TYPE_CHECKING and find_spec("kv_cache_manager_v2") is not None:
    from kv_cache_manager_v2 import TokenId
    from kv_cache_manager_v2._block_radix_tree import Block, BlockRadixTree, ReuseScope
    from kv_cache_manager_v2._life_cycle_registry import (
        AttnLifeCycle,
        LifeCycleId,
        LifeCycleRegistry,
    )
else:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import TokenId
    from tensorrt_llm.runtime.kv_cache_manager_v2._block_radix_tree import (
        Block,
        BlockRadixTree,
        ReuseScope,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._life_cycle_registry import (
        AttnLifeCycle,
        LifeCycleId,
        LifeCycleRegistry,
    )


class _TwoLifeCycles:
    """Minimal ``LifeCycleRegistry`` stand-in: only ``size`` is reached here.

    ``Block.storage`` is sized from ``num_life_cycles``, which is all the prune
    predicate needs. Two life cycles is the smallest tree that can express
    "this block lost life cycle 0's page but still holds life cycle 1's".
    """

    size = LifeCycleId(2)

    @property
    def ssm_life_cycle_id(self) -> None:
        return None

    def attention_life_cycles(self) -> Iterator[tuple[object, object]]:
        return iter(())


class _DeadPageRef:
    """Stand-in for ``rawref.ref[CommittedPage]``: occupied slot, dead referent.

    The predicate under test only compares the slot against ``None``. Returning
    ``None`` from ``__call__`` keeps ``Block._release_pages`` on its
    already-collected path, so teardown never dereferences a fake page.
    """

    def __call__(self) -> None:
        return None


# Windowed, no sink blocks: keeps ``clear_stale_blocks_after_page_unlink`` off
# the ``remove_subtree`` branch (that branch fires for full attention or sink
# blocks and would drop the subtree regardless of the tail-prune predicate),
# so the test isolates the prune loop.
_WINDOWED_ATTN = AttnLifeCycle(window_size=64, num_sink_blocks=0)

_LC_UNLINKED = LifeCycleId(0)
_LC_OTHER = LifeCycleId(1)


class TestStaleTailPrune(unittest.TestCase):
    def _build_chain(self) -> "tuple[BlockRadixTree, object, Block, Block]":
        """Root -> first -> tip, two life cycles, tokens_per_block=2."""
        tree = BlockRadixTree(cast(LifeCycleRegistry, _TwoLifeCycles()), tokens_per_block=2)
        root = tree.add_or_get_existing(ReuseScope())
        first = Block([TokenId(1), TokenId(2)], root)
        tip = Block([TokenId(3), TokenId(4)], first)
        self.assertEqual(len(tip.storage), 2)
        return tree, root, first, tip

    def test_tip_with_live_page_in_another_life_cycle_is_kept(self) -> None:
        tree, root, first, tip = self._build_chain()
        # Life cycle 0's page was just unlinked; life cycle 1 still holds one.
        tip.storage[_LC_UNLINKED] = None
        tip.storage[_LC_OTHER] = cast(object, _DeadPageRef())

        Block.clear_stale_blocks_after_page_unlink(tip, _LC_UNLINKED, _WINDOWED_ATTN)

        # Detaching the tip here would orphan the committed chain of an
        # in-flight sequence that is still using life cycle 1.
        self.assertIn(tip.key, first.next)
        self.assertIs(first.next[tip.key], tip)
        self.assertIsNotNone(tip._prev())
        self.assertIn(first.key, root.next)

    def test_tip_with_no_live_page_anywhere_is_pruned(self) -> None:
        # Negative control: the fix must not stop the prune it is supposed to
        # allow, otherwise dead tails accumulate forever.
        tree, root, first, tip = self._build_chain()
        tip.storage[_LC_UNLINKED] = None
        tip.storage[_LC_OTHER] = None

        Block.clear_stale_blocks_after_page_unlink(tip, _LC_UNLINKED, _WINDOWED_ATTN)

        # tip is detached, and the walk continues up through `first`, which is
        # now itself a childless tip with no pages; the emptied root is then
        # dropped from the tree.
        self.assertNotIn(tip.key, first.next)
        self.assertNotIn(first.key, root.next)
        self.assertEqual(tree.next, {})

    def test_tip_keeps_when_only_the_unlinked_life_cycle_is_empty(self) -> None:
        # Same as the first case but with the roles of the two life cycles
        # swapped, so the test cannot pass by hard-coding an index.
        tree, root, first, tip = self._build_chain()
        tip.storage[_LC_OTHER] = None
        tip.storage[_LC_UNLINKED] = cast(object, _DeadPageRef())

        Block.clear_stale_blocks_after_page_unlink(tip, _LC_OTHER, _WINDOWED_ATTN)

        self.assertIn(tip.key, first.next)
        self.assertIs(first.next[tip.key], tip)


if __name__ == "__main__":
    unittest.main()
