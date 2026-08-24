# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for ``KVCacheManagerV2.probe_first_new_block_key``.

The probe feeds the scheduler's prefix-aware skip. Its one silent failure mode
is keying a *different* block than the request goes on to commit -- the skip
then simply never fires, with no error anywhere. These tests pin the two halves
of that contract:

* the probe marshals tokens and the reuse scope exactly like
  ``_prepare_context_impl`` does, and
* the key it selects is the one for the first block past the reusable prefix.

No GPU: the KV cache manager is stubbed down to the attributes both paths touch.
"""

from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import (
    KVCacheManagerV2,
    _first_new_block_key,
)
from tensorrt_llm.runtime.kv_cache_manager_v2 import ReuseScope, sequence_to_blockchain_keys

pytestmark = pytest.mark.cpu_only

TOKENS_PER_BLOCK = 4


def make_stub_manager(tokens_per_block=TOKENS_PER_BLOCK, enable_block_reuse=True, num_reusable=0):
    """A KVCacheManagerV2 reduced to what the two token paths need."""
    mgr = object.__new__(KVCacheManagerV2)
    mgr.tokens_per_block = tokens_per_block
    mgr.enable_block_reuse = enable_block_reuse
    mgr.vocab_size = 32000
    mgr.conversation_manager = None
    mgr.kv_cache_map = {}
    mgr.index_mapper = Mock()
    mgr.index_mapper.num_free_slots.return_value = 1
    mgr.index_mapper.add_new_sequence.return_value = 0
    mgr.max_beam_width = 1
    mgr.num_pools = 0  # no pool buffers wired in this stub
    mgr._has_cp_helix = False
    # _create_kv_cache consults these before it reaches impl.create_kv_cache;
    # per-request stats are opt-in and off in this stub's manager.
    mgr.is_draft = False
    mgr.enable_stats = False
    mgr._request_stats_enabled_ids = set()
    mgr._stream = Mock()
    mgr.impl = Mock()
    mgr.impl.probe_reuse.return_value = num_reusable
    mgr.impl.create_kv_cache.return_value = Mock(num_committed_tokens=0)
    # Resume touches real CUDA state; the token marshalling is already done.
    mgr._resume_and_restore = lambda req_id, kv_cache: True
    return mgr


class StubRequest:
    """The request fields the two token paths read, and nothing else.

    Deliberately not a ``Mock``: an attribute the stub forgets must raise here
    rather than autocreate a truthy stand-in that silently steers the code
    under test down a branch the stub does not model.
    """

    def __init__(
        self,
        tokens,
        request_id=0,
        lora_task_id=None,
        cache_salt=None,
        return_perf_metrics=False,
    ):
        self._tokens = list(tokens)
        self.py_request_id = request_id
        self.is_first_context_chunk = True
        self.is_disagg_generation_init_state = False
        self.is_dummy = False
        self.lora_task_id = lora_task_id
        self.cache_salt = cache_salt
        # LlmRequest's own default; a truthy value here turns on the
        # per-request stats branch of _create_kv_cache.
        self.return_perf_metrics = return_perf_metrics
        self.prompt_len = len(self._tokens)
        self.total_input_len_cp = len(self._tokens)
        self.context_current_position = 0
        self.multimodal_hashes = None
        self.multimodal_positions = None
        self.multimodal_lengths = None

    # Both backends' token sources return the same sequence here.
    def get_tokens(self, beam=0):
        return list(self._tokens)

    def get_tokens_view(self, beam=0):
        return list(self._tokens)

    def set_prepopulated_prompt_len(self, num_tokens, tokens_per_block):
        pass


def make_request(tokens, request_id=0, lora_task_id=None, cache_salt=None, **kwargs):
    return StubRequest(tokens, request_id, lora_task_id, cache_salt, **kwargs)


def prepared_tokens_and_scope(mgr, req):
    """Tokens and reuse scope that ``prepare_context`` hands to the impl."""
    mgr.impl.create_kv_cache.reset_mock()
    mgr._prepare_context_impl(req)
    args, _ = mgr.impl.create_kv_cache.call_args
    scope, tokens = args[0], args[1]
    return list(tokens), scope


def probed_tokens_and_scope(mgr, req):
    """Tokens and reuse scope that the probe hands to the impl."""
    mgr.impl.probe_reuse.reset_mock()
    mgr.probe_first_new_block_key(req)
    args, _ = mgr.impl.probe_reuse.call_args
    scope, tokens = args[0], args[1]
    return list(tokens), scope


class TestTokenParity:
    """The probe must key the same block the request will commit."""

    @pytest.mark.parametrize("return_perf_metrics", [False, True])
    def test_tokens_match_prepare_context(self, return_perf_metrics):
        # Both settings, because return_perf_metrics selects which manager
        # attributes _create_kv_cache reads on the way to the impl.
        mgr = make_stub_manager()
        req = make_request(range(20), return_perf_metrics=return_perf_metrics)
        assert probed_tokens_and_scope(mgr, req)[0] == prepared_tokens_and_scope(mgr, req)[0]

    def test_last_token_is_excluded(self):
        """prepare_context drops the final token (it cannot be recovered)."""
        mgr = make_stub_manager()
        tokens = list(range(20))
        req = make_request(tokens)
        assert probed_tokens_and_scope(mgr, req)[0] == tokens[:-1]

    def test_reuse_scope_matches_prepare_context(self):
        mgr = make_stub_manager()
        req = make_request(range(20), lora_task_id=7, cache_salt="tenant-a")
        _, probe_scope = probed_tokens_and_scope(mgr, req)
        _, create_scope = prepared_tokens_and_scope(mgr, req)
        # Iterate rather than read attributes: both the NamedTuple and the C++
        # binding are field-iterable (see reuse_scope_to_bytes), lora_id first.
        assert tuple(probe_scope) == tuple(create_scope)
        assert tuple(probe_scope)[0] == 7
        assert tuple(probe_scope)[1] is not None

    def test_cache_salt_changes_the_key(self):
        """Different reuse namespaces must not dedup against each other."""
        mgr = make_stub_manager()
        plain = mgr.probe_first_new_block_key(make_request(range(20)))
        salted = mgr.probe_first_new_block_key(make_request(range(20), cache_salt="tenant-a"))
        assert plain is not None and salted is not None
        assert plain != salted

    def test_lora_id_changes_the_key(self):
        mgr = make_stub_manager()
        base = mgr.probe_first_new_block_key(make_request(range(20)))
        lora = mgr.probe_first_new_block_key(make_request(range(20), lora_task_id=3))
        assert base != lora

    def test_augmentation_call_matches_prepare_context(self):
        """Multimodal requests key on content digests, so the probe has to use
        the same augmentation call -- same bounds, same request."""
        mgr = make_stub_manager()
        calls = []
        original = mgr._augment_tokens_for_block_reuse

        def recording(tokens, req, start=0, end=None):
            calls.append((list(tokens), id(req), start, end))
            return original(tokens, req, start, end)

        mgr._augment_tokens_for_block_reuse = recording
        req = make_request(range(20))
        mgr._prepare_context_impl(req)
        mgr.probe_first_new_block_key(req)
        assert len(calls) == 2
        assert calls[0] == calls[1]


class TestKeySelection:
    def test_key_is_the_block_after_the_reusable_prefix(self):
        tokens = list(range(16))
        scope = ReuseScope()
        keys = [key for _, key in sequence_to_blockchain_keys(TOKENS_PER_BLOCK, scope, tokens)]
        # keys[0] is the root digest; keys[i + 1] belongs to block i.
        for num_reusable, expected_block in ((0, 0), (4, 1), (8, 2), (12, 3)):
            assert (
                _first_new_block_key(tokens, TOKENS_PER_BLOCK, scope, num_reusable)
                == keys[expected_block + 1]
            )

    def test_partial_match_selects_the_block_being_completed(self):
        tokens = list(range(16))
        scope = ReuseScope()
        keys = [key for _, key in sequence_to_blockchain_keys(TOKENS_PER_BLOCK, scope, tokens)]
        # A mid-block match still leaves block 1 as the first one completed.
        assert _first_new_block_key(tokens, TOKENS_PER_BLOCK, scope, 5) == keys[2]
        assert _first_new_block_key(tokens, TOKENS_PER_BLOCK, scope, 7) == keys[2]

    def test_none_when_next_block_is_partial(self):
        scope = ReuseScope()
        # 10 tokens, 4 per block: block 2 would hold only 2 tokens.
        assert _first_new_block_key(list(range(10)), TOKENS_PER_BLOCK, scope, 8) is None

    def test_none_when_everything_is_reusable(self):
        scope = ReuseScope()
        tokens = list(range(16))
        assert _first_new_block_key(tokens, TOKENS_PER_BLOCK, scope, len(tokens)) is None

    def test_truncation_is_exact(self):
        """The key is hashed from a truncated prefix; that must equal the key
        computed over the whole sequence."""
        scope = ReuseScope(lora_id=2, salt=99)
        short, long = list(range(8)), list(range(64))
        assert _first_new_block_key(short, TOKENS_PER_BLOCK, scope, 0) == _first_new_block_key(
            long, TOKENS_PER_BLOCK, scope, 0
        )

    def test_shared_prefix_yields_the_same_key(self):
        """The property the scheduler relies on: two requests whose prompts
        share the uncached prefix collide on the same key."""
        scope = ReuseScope()
        a = list(range(16)) + [100, 101]
        b = list(range(16)) + [200, 201]
        assert _first_new_block_key(a, TOKENS_PER_BLOCK, scope, 0) == _first_new_block_key(
            b, TOKENS_PER_BLOCK, scope, 0
        )

    def test_diverging_prefix_yields_different_keys(self):
        scope = ReuseScope()
        a = [1, 2, 3, 4, 5, 6, 7, 8]
        b = [1, 2, 3, 9, 5, 6, 7, 8]
        assert _first_new_block_key(a, TOKENS_PER_BLOCK, scope, 0) != _first_new_block_key(
            b, TOKENS_PER_BLOCK, scope, 0
        )


class TestGuards:
    def test_none_without_block_reuse(self):
        mgr = make_stub_manager(enable_block_reuse=False)
        assert mgr.probe_first_new_block_key(make_request(range(20))) is None
        mgr.impl.probe_reuse.assert_not_called()

    @pytest.mark.parametrize("num_tokens", [0, 1])
    def test_none_for_degenerate_prompts(self, num_tokens):
        mgr = make_stub_manager()
        assert mgr.probe_first_new_block_key(make_request(range(num_tokens))) is None

    def test_none_when_prompt_shorter_than_one_block(self):
        mgr = make_stub_manager()
        assert mgr.probe_first_new_block_key(make_request(range(TOKENS_PER_BLOCK))) is None

    def test_probe_does_not_create_a_kv_cache(self):
        """Read-only: the probe must not touch the tree or take a slot."""
        mgr = make_stub_manager()
        mgr.probe_first_new_block_key(make_request(range(20)))
        mgr.impl.create_kv_cache.assert_not_called()
        assert mgr.kv_cache_map == {}
