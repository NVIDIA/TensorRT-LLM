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
"""``KVCacheManagerV2``'s window-bounded drafter context: the keep count it derives and the
reuse guard on its claims, on a bare manager whose cache claim is a fake.

* ``enable_draft_context_window`` derives the keep count from the window, the reuse rewind and
  the block size, builds the guard with block reuse only, and refuses a KV connector;
* without it, ``prepare_context`` passes the pipeline stage agreement's cap alone as the claim
  limit and notes nothing;
* with it, the guard's limit joins the limit chain (with the pipeline stage agreement's cap),
  first chunks only, and every claim, a stage-agreement reservation included, is noted from
  where the request computes;
* with ``TLLM_KV_DRAFT_CONTEXT_HELD_KEYS=1``, the request object holds its prefix keys from its
  first claim (the store prefetch's uncounted one, or the reservation) to its resume: a
  prefetched, reserved and resumed first chunk hashes its prompt once, a failed resume keeps the
  keys for the retry, a resumed first chunk drops them, and every limit and mark equals the
  path without held keys; without the knob, no keys are held and the calls hash as before;
* with the context-path counters' guard timing (``PPCtxDiag.guard_timing``), every guard
  call is counted per kind with the guard's prefix hash count; without it, none is.
"""

import numpy as np
import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.draft_context_window import (
    HELD_KEYS_ENV,
    REQUEST_KEYS_ATTR,
    DraftContextWindowGuard,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.swa_stale_endpoint import scope_seed
from tensorrt_llm._torch.pyexecutor.pp_ctx_diag import PPCtxDiag

TPB = 32


def _manager(*, reuse=True, rewind=512, connector=None):
    manager = object.__new__(KVCacheManagerV2)
    manager.tokens_per_block = TPB
    manager.enable_block_reuse = reuse
    manager.is_draft = False
    manager._swa_endpoint_rewind = rewind
    manager.kv_connector_manager = connector
    manager.claims = []
    manager.claim_result = 0

    def prepare_context_cache(req, limit=None, resume=True):
        manager.claims.append((limit, resume))
        return manager.claim_result

    manager.prepare_context_cache = prepare_context_cache
    manager._prepare_connector_prefix_reservation = lambda req: None
    manager._reuse_token_source = lambda req: req.tokens
    return manager


class _Req:
    def __init__(self, tokens, first=True, pp_cap=None):
        self.tokens = np.asarray(tokens, dtype=np.int32)
        self.prompt_len = len(tokens)
        self.is_first_context_chunk = first
        self.is_disagg_generation_init_state = False
        self.is_dummy = False
        self.lora_task_id = None
        self.cache_salt = None
        self.multimodal_hashes = None
        self.py_pp_reuse_cap = pp_cap
        self.py_connector_served_position = 0
        self.context_current_position = 0
        self.context_chunk_size = 0
        self.prepopulated = None

    @property
    def context_remaining_length(self):
        return self.prompt_len - self.context_current_position

    def set_prepopulated_prompt_len(self, reuse, tokens_per_block):
        self.prepopulated = reuse
        self.context_current_position = max(self.context_current_position, reuse)


class _Recorder(DraftContextWindowGuard):
    def __init__(self, limit):
        super().__init__(TPB, 2048, 2048 + 512 + TPB, filter_bits=1 << 12, log_every=0)
        self.limit = limit
        self.notes = []

    def claim_limit(self, tokens, prompt_len, seed, count=True, owner=None):
        return self.limit

    def note_claim(self, tokens, prompt_len, seed, start, owner=None):
        self.notes.append((prompt_len, start))


def test_enable_derives_the_keep_count_and_builds_the_guard():
    manager = _manager()
    assert manager.enable_draft_context_window(2048) == 2048 + 512 + TPB
    assert manager.draft_context_keep_tokens == 2592
    guard = manager._draft_context_guard
    assert isinstance(guard, DraftContextWindowGuard)
    assert (guard.read_tokens, guard.keep_tokens) == (2048, 2592)


def test_enable_follows_the_configured_rewind():
    manager = _manager(rewind=128)
    assert manager.enable_draft_context_window(2048) == 2048 + 128 + TPB


def test_enable_without_block_reuse_keeps_the_window_and_builds_no_guard():
    manager = _manager(reuse=False, rewind=0)
    assert manager.enable_draft_context_window(2048) == 2048 + TPB
    assert manager._draft_context_guard is None


@pytest.mark.parametrize("window", [0, -1])
def test_enable_refuses_a_bad_window(window):
    with pytest.raises(ValueError, match="positive"):
        _manager().enable_draft_context_window(window)


def test_enable_refuses_a_kv_connector():
    manager = _manager(connector=object())
    with pytest.raises(ValueError, match="KV connector"):
        manager.enable_draft_context_window(2048)
    assert manager.draft_context_keep_tokens is None


@pytest.mark.parametrize("resume", [True, False])
@pytest.mark.parametrize("pp_cap", [None, 640])
def test_without_a_window_the_limit_chain_is_unchanged(resume, pp_cap):
    manager = _manager()
    req = _Req(range(5000), pp_cap=pp_cap)
    assert manager.prepare_context(req, resume=resume)
    assert manager.claims == [(pp_cap, resume)]
    assert manager.draft_context_keep_tokens is None


@pytest.mark.parametrize(
    "guard_limit, pp_cap, expected",
    [(None, None, None), (1024, None, 1024), (1024, 640, 640), (512, 640, 512)],
)
def test_the_guard_joins_the_limit_chain(guard_limit, pp_cap, expected):
    manager = _manager()
    manager._draft_context_guard = _Recorder(guard_limit)
    manager.claim_result = 448
    req = _Req(range(5000), pp_cap=pp_cap)
    assert manager.prepare_context(req)
    assert manager.claims == [(expected, True)]
    # Noted from where the request computes: the settled cursor at the claim.
    assert manager._draft_context_guard.notes == [(5000, 448)]
    assert req.context_current_position == 448


def test_a_reservation_is_noted_too():
    manager = _manager()
    manager._draft_context_guard = _Recorder(1024)
    manager.claim_result = 960
    req = _Req(range(5000))
    assert manager.prepare_context(req, resume=False)
    assert manager.claims == [(1024, False)]
    assert manager._draft_context_guard.notes == [(5000, 960)]


def test_a_failed_reservation_notes_nothing():
    manager = _manager()
    manager._draft_context_guard = _Recorder(None)
    manager.claim_result = None
    assert not manager.prepare_context(_Req(range(100)), resume=False)
    assert manager._draft_context_guard.notes == []


def test_a_later_chunk_is_neither_capped_nor_noted():
    manager = _manager()
    manager._draft_context_guard = _Recorder(0)
    manager.claim_result = 4096
    assert manager.prepare_context(_Req(range(5000), first=False))
    assert manager.claims == [(None, True)]
    assert manager._draft_context_guard.notes == []


def test_a_multimodal_prompt_reuses_nothing_and_is_noted():
    manager = _manager()
    manager._draft_context_guard = _Recorder(None)
    req = _Req(range(100))
    req.multimodal_hashes = [[1, 2, 3, 4, 5, 6, 7, 8]]
    assert manager.prepare_context(req)
    assert manager.claims == [(0, True)]
    assert manager._draft_context_guard.notes == [(100, 0)]


def test_a_text_prompt_behind_a_multimodal_prompts_text_prefix_is_capped():
    """A multimodal prompt commits the blocks of its text prefix under the keys a text prompt
    with that prefix matches; its unwritten drafter rows there must cap that text prompt."""
    manager = _manager()
    manager.enable_draft_context_window(2048)
    system = list(range(10_000, 18_000))
    image = _Req(system + [7] * 64 + list(range(100, 3100)))
    image.multimodal_hashes = [[1, 2, 3, 4, 5, 6, 7, 8]]
    manager.claim_result = 0
    assert manager.prepare_context(image)
    assert manager.claims[-1] == (0, True)
    other = _Req(system + [1, 2, 3])
    assert manager.prepare_context(other)
    limit, _ = manager.claims[-1]
    assert limit is not None and limit <= len(system) + 3 - 2048


def test_the_real_guard_caps_a_short_suffix_behind_a_prefix_left_unwritten():
    manager = _manager()
    manager.enable_draft_context_window(2048)
    system = list(range(10_000, 18_000))
    manager.claim_result = 0
    # A long first turn leaves the whole system prompt's drafter rows unwritten.
    assert manager.prepare_context(_Req(system + list(range(100, 3100))))
    other = _Req(system + [1, 2, 3])
    manager.claim_result = 128
    assert manager.prepare_context(other)
    limit, _ = manager.claims[-1]
    assert limit is not None and limit % TPB == 0
    assert len(system) + 3 - 2048 - TPB < limit <= len(system) + 3 - 2048


@pytest.fixture
def held_keys(monkeypatch):
    monkeypatch.setenv(HELD_KEYS_ENV, "1")


@pytest.fixture
def no_held_keys(monkeypatch):
    monkeypatch.delenv(HELD_KEYS_ENV, raising=False)


def _windowed_manager():
    manager = _manager()
    manager.enable_draft_context_window(2048)
    return manager, manager._draft_context_guard


def test_without_the_knob_no_keys_are_held_and_every_call_hashes(no_held_keys):
    manager, guard = _windowed_manager()
    assert manager._draft_context_held_keys is False
    req = _Req(range(9000))
    assert manager._prefetch_claim_limit(req) is None
    assert manager.prepare_context(req, resume=False)
    assert manager.prepare_context(req)
    assert getattr(req, REQUEST_KEYS_ATTR, None) is None
    assert guard.stats["claims"] == 2 and guard.stats["notes"] == 2
    assert guard.stats["prefix_hashes"] == 5
    assert guard.stats["skipped_marks"] == 0


def test_a_prefetched_reserved_and_resumed_first_chunk_hashes_its_prompt_once(held_keys):
    """The store prefetch's uncounted claim limit makes the keys; the reservation and the
    resume (each a limit and a note) are served from them."""
    manager, guard = _windowed_manager()
    assert manager._draft_context_held_keys is True
    req = _Req(range(9000))
    manager._prefetch_claim_limit(req)
    assert getattr(req, REQUEST_KEYS_ATTR) is not None
    assert guard.stats["claims"] == 0 and guard.stats["prefix_hashes"] == 1
    assert manager.prepare_context(req, resume=False)
    assert manager.prepare_context(req)
    assert guard.stats["claims"] == 2 and guard.stats["notes"] == 2
    assert guard.stats["prefix_hashes"] == 1
    assert getattr(req, REQUEST_KEYS_ATTR) is None


@pytest.mark.parametrize("timing", [True, False])
def test_guard_calls_are_timed_only_with_the_guard_timing(held_keys, timing):
    manager, guard = _windowed_manager()
    manager.ctx_diag = PPCtxDiag(0, 4, guard_timing=timing)
    req = _Req(range(9000))
    manager._prefetch_claim_limit(req)
    assert manager.prepare_context(req, resume=False)
    assert manager.prepare_context(req)
    diag = manager.ctx_diag
    if not timing:
        assert diag.guard_calls == {"prefetch": 0, "claim": 0, "note": 0}
        return
    assert diag.guard_calls == {"prefetch": 1, "claim": 2, "note": 2}
    assert all(seconds >= 0.0 for seconds in diag.guard_host_s.values())
    assert diag.guard_prefix_hashes == guard.stats["prefix_hashes"] == 1


def test_a_reserved_and_resumed_first_chunk_hashes_its_prompt_once(held_keys):
    manager = _manager()
    manager.enable_draft_context_window(2048)
    guard = manager._draft_context_guard
    req = _Req(range(9000))
    manager.claim_result = 0
    assert manager.prepare_context(req, resume=False)
    assert getattr(req, REQUEST_KEYS_ATTR) is not None
    assert manager.prepare_context(req)
    assert guard.stats["claims"] == 2 and guard.stats["notes"] == 2
    assert guard.stats["prefix_hashes"] == 1
    # The resume noted the same blocks as the reservation: nothing added twice.
    assert guard.stats["skipped_marks"] == guard.stats["marked_blocks"] // 2
    assert getattr(req, REQUEST_KEYS_ATTR) is None


def test_a_failed_resume_keeps_the_keys_for_the_retry(held_keys):
    manager = _manager()
    manager.enable_draft_context_window(2048)
    guard = manager._draft_context_guard
    req = _Req(range(9000))
    manager.claim_result = 0
    assert manager.prepare_context(req, resume=False)
    manager.claim_result = None
    assert not manager.prepare_context(req)
    assert getattr(req, REQUEST_KEYS_ATTR) is not None
    manager.claim_result = 0
    assert manager.prepare_context(req)
    assert guard.stats["prefix_hashes"] == 1
    assert getattr(req, REQUEST_KEYS_ATTR) is None


def test_held_keys_leave_the_limits_and_marks_unchanged(held_keys):
    """The same requests through a manager and through a bare guard without held keys."""
    manager = _manager()
    manager.enable_draft_context_window(2048)
    guard = manager._draft_context_guard
    plain = DraftContextWindowGuard(TPB, 2048, 2048 + 512 + TPB)
    seed = scope_seed(None, None)
    system = list(range(10_000, 18_000))
    prompts = [
        system + list(range(100, 3100)),
        system + [1, 2, 3],
        system + list(range(100, 1100)) + [5] * 40,
    ]
    for tokens in prompts:
        req = _Req(tokens)
        manager._prefetch_claim_limit(req)
        manager.claim_result = 0
        assert manager.prepare_context(req, resume=False)
        manager.claim_result = 128
        assert manager.prepare_context(req)
        array = np.asarray(tokens, dtype=np.int32)
        reserved = plain.claim_limit(array, len(tokens), seed)
        plain.note_claim(array, len(tokens), seed, 0)
        resumed = plain.claim_limit(array, len(tokens), seed)
        plain.note_claim(array, len(tokens), seed, min(128, req.context_current_position))
        assert manager.claims[-2:] == [(reserved, False), (resumed, True)]
    assert np.array_equal(guard._marks._bits, plain._marks._bits)
