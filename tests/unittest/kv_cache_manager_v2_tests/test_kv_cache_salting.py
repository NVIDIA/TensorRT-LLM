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
from typing import TYPE_CHECKING

import pytest

pytestmark = pytest.mark.cpu_only


if not TYPE_CHECKING and find_spec("kv_cache_manager_v2") is not None:
    from kv_cache_manager_v2 import ReuseScope, TokenId, sequence_to_blockchain_keys
else:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import (
        ReuseScope,
        TokenId,
        sequence_to_blockchain_keys,
    )


def _root_key(scope):
    """The reuse-scope digest a blockchain starts from, which is its first key."""
    return next(iter(sequence_to_blockchain_keys(1, scope, [])))[1]


class TestReuseScope(unittest.TestCase):
    def test_reuse_scope_seeds_distinct_keys(self) -> None:
        # Distinct reuse scopes -- including the None-vs-0 cases for each field --
        # must seed distinct radix-tree keys.
        scopes = [
            ReuseScope(),
            ReuseScope(lora_id=0),
            ReuseScope(salt=0),
            ReuseScope(lora_id=0, salt=0),
            ReuseScope(lora_id=7, salt=11),
        ]

        roots = [_root_key(scope) for scope in scopes]
        self.assertEqual(len(set(roots)), len(scopes))
        # Deterministic across repeated derivation.
        self.assertEqual(roots, [_root_key(scope) for scope in scopes])

    def test_blockchain_keys_are_seeded_by_reuse_scope(self) -> None:
        tokens = [TokenId(1), TokenId(2), TokenId(3), TokenId(4)]
        scope = ReuseScope(lora_id=7, salt=11)

        keys = list(sequence_to_blockchain_keys(2, scope, tokens))
        same_scope_keys = list(sequence_to_blockchain_keys(2, ReuseScope(7, 11), tokens))
        different_scope_keys = list(sequence_to_blockchain_keys(2, ReuseScope(7, 12), tokens))

        self.assertEqual(keys, same_scope_keys)
        self.assertNotEqual([key for _, key in keys], [key for _, key in different_scope_keys])

    def test_blockchain_keys_chain_from_the_scope_root(self) -> None:
        # The first blockchain key is the reuse-scope digest, and each subsequent key
        # is that chain extended by one block of tokens.
        tokens = [TokenId(1), TokenId(2), TokenId(3), TokenId(4)]
        scope = ReuseScope(lora_id=7, salt=11)

        keys = list(sequence_to_blockchain_keys(2, scope, tokens))
        self.assertEqual([chunk for chunk, _ in keys], [[], [1, 2], [3, 4]])
        self.assertEqual(keys[0][1], _root_key(scope))

        # Each key covers a prefix of the sequence, so extending the sequence leaves the
        # earlier keys untouched.
        prefix = list(sequence_to_blockchain_keys(2, scope, tokens[:2]))
        self.assertEqual([key for _, key in prefix], [key for _, key in keys[:2]])

    def test_block_keys_are_scoped_through_their_root(self) -> None:
        tokens = [TokenId(1), TokenId(2)]

        def block_keys(scope):
            return [key for _, key in sequence_to_blockchain_keys(2, scope, tokens)][1:]

        self.assertNotEqual(
            block_keys(ReuseScope(lora_id=7, salt=11)),
            block_keys(ReuseScope(lora_id=7, salt=12)),
        )
        self.assertEqual(
            block_keys(ReuseScope(lora_id=7, salt=11)),
            block_keys(ReuseScope(7, 11)),
        )


if __name__ == "__main__":
    unittest.main()
