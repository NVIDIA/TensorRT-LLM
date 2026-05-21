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
"""Pure unit tests for KV cache reuse scopes."""

import unittest
from importlib.util import find_spec
from typing import TYPE_CHECKING, cast

if not TYPE_CHECKING and find_spec("kv_cache_manager_v2") is not None:
    from kv_cache_manager_v2 import ReuseScope, TokenId
    from kv_cache_manager_v2._block_radix_tree import (
        Block,
        BlockRadixTree,
        sequence_to_blockchain_keys,
    )
    from kv_cache_manager_v2._life_cycle_registry import LifeCycleRegistry
else:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import ReuseScope, TokenId
    from tensorrt_llm.runtime.kv_cache_manager_v2._block_radix_tree import (
        Block,
        BlockRadixTree,
        sequence_to_blockchain_keys,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._life_cycle_registry import LifeCycleRegistry


class _EmptyLifeCycles:
    size = 0

    @property
    def ssm_life_cycle_id(self) -> None:
        return None

    def attention_life_cycles(self):
        return iter(())


class TestReuseScope(unittest.TestCase):
    def test_to_bytes_distinguishes_scope_fields(self) -> None:
        scopes = [
            ReuseScope(),
            ReuseScope(lora_id=0),
            ReuseScope(salt=0),
            ReuseScope(lora_id=0, salt=0),
            ReuseScope(lora_id=7, salt=11),
        ]

        serialized = [scope.to_bytes() for scope in scopes]
        self.assertEqual(len(set(serialized)), len(scopes))
        self.assertEqual(serialized[0], b"\x00")
        self.assertEqual(serialized, [scope.to_bytes() for scope in scopes])

    def test_blockchain_keys_are_seeded_by_reuse_scope(self) -> None:
        tokens = [TokenId(1), TokenId(2), TokenId(3), TokenId(4)]
        scope = ReuseScope(lora_id=7, salt=11)

        keys = list(sequence_to_blockchain_keys(2, scope, tokens))
        same_scope_keys = list(sequence_to_blockchain_keys(2, ReuseScope(7, 11), tokens))
        different_scope_keys = list(sequence_to_blockchain_keys(2, ReuseScope(7, 12), tokens))

        self.assertEqual(keys, same_scope_keys)
        self.assertNotEqual([key for _, key in keys], [key for _, key in different_scope_keys])

    def test_radix_tree_match_is_scoped(self) -> None:
        tree = BlockRadixTree(cast(LifeCycleRegistry, _EmptyLifeCycles()), tokens_per_block=2)
        scope = ReuseScope(lora_id=7, salt=11)
        other_scope = ReuseScope(lora_id=7, salt=12)
        tokens = [TokenId(1), TokenId(2)]

        root = tree.add_or_get_existing(scope)
        block = Block(tokens, root)

        self.assertEqual(list(tree.match(scope, tokens)), [(block, len(tokens))])
        self.assertEqual(list(tree.match(other_scope, tokens)), [])
        self.assertEqual(list(tree.match(other_scope, tokens[:1], enable_partial_match=True)), [])


if __name__ == "__main__":
    unittest.main()
