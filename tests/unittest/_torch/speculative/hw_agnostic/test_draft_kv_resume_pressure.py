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
"""A draft KV cache that cannot be resumed must fail loudly, not silently.

``resume_request`` documents a False from ``_resume_and_restore`` as a REFUSAL
under GPU pressure rather than corruption, so deferring the request looks
attractive -- and the neighbouring mirror-shortage branch does exactly that.

It is not the same case, and the difference is why these tests pin a raise. A
request with no mirrored cache stays unmapped, and ``copy_batch_block_offsets``
asserts in C++ before anything reads it. A cache that merely failed to RESUME is
still in ``kv_cache_map``: the offsets copy succeeds, the forward proceeds, and
the drafter writes context K/V through the inactive cache's stale page table.
Skipping here therefore trades a loud crash for silent corruption. That was
tried, caught in review, and reverted.

Deferring safely means not forwarding the request at all this iteration, which
is an admission decision owned by the scheduler -- by the time draft preparation
runs, the target manager has already prepared the same request. The scheduler
now makes that decision through ``admit_mirror`` (see
``TestUnpairedDraftAdmission`` in the V2 scheduler tests), so a mirror should no
longer arrive here suspended. The raise stays as the invariant of last resort
and these tests hold it in place.
"""

from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

pytestmark = pytest.mark.cpu_only


class _Req:
    def __init__(self, req_id):
        self.py_request_id = req_id
        self.lora_task_id = None
        self.cache_salt = None
        self.is_dummy = False
        self.is_last_context_chunk = True
        self.context_current_position = 0
        self.context_chunk_size = 8
        self.py_draft_tokens = None


class _Batch:
    def __init__(self, context_requests=(), generation_requests=()):
        self.context_requests = list(context_requests)
        self.generation_requests = list(generation_requests)

    def all_requests(self):
        """Consumed by the real ``request_context``, which is left unpatched so
        the draft-model scope the method runs under is the production one."""
        return self.context_requests + self.generation_requests


def _manager(resume_ok):
    """A draft manager whose resume answers ``resume_ok``, with nothing else real.

    Built without ``__init__``: the method under test needs only the mirror
    lookup, the resume and a request-context manager, and building a real
    manager would need a pool, a stream and a device.
    """
    manager = object.__new__(KVCacheManagerV2)
    manager.is_draft = True
    manager.enable_joint_kv_cache_reuse = False
    manager.num_extra_kv_tokens = 0
    manager._allocated_draft_lens = set()
    cache = MagicMock(name="kv_cache")
    cache.resize.return_value = True
    cache.capacity = 0
    manager._mirror_draft_kv_cache = MagicMock(return_value=cache)
    manager._resume_and_restore = MagicMock(return_value=resume_ok)
    manager._required_gen_capacity = MagicMock(return_value=1)
    return manager, cache


def test_context_request_raises_on_a_resume_refusal():
    """The safe behaviour: fail loudly rather than forward through a stale table."""
    manager, cache = _manager(resume_ok=False)

    with pytest.raises(RuntimeError, match="Failed to resume draft KV cache"):
        manager._prepare_draft_resources(_Batch(context_requests=[_Req(341)]))

    # And nothing was sized on the strength of a cache that never resumed.
    cache.resize.assert_not_called()


def test_a_refused_context_request_is_never_silently_skipped():
    """Regression pin for the reverted fix: the batch must not simply continue.

    The reverted version logged and skipped, leaving the request in the batch
    with a mapped but inactive cache -- which is the corruption path.
    """
    manager, _ = _manager(resume_ok=False)
    batch = _Batch(context_requests=[_Req(341), _Req(342), _Req(343)])

    with pytest.raises(RuntimeError):
        manager._prepare_draft_resources(batch)

    # It stopped at the first refusal rather than walking the rest of the batch.
    assert manager._resume_and_restore.call_count == 1


def test_context_request_is_prepared_normally_when_resume_succeeds():
    """The ordinary path is unchanged: resume, then size the cache."""
    manager, cache = _manager(resume_ok=True)

    manager._prepare_draft_resources(_Batch(context_requests=[_Req(7)]))

    manager._resume_and_restore.assert_called_once()
    cache.resize.assert_called_once()


def test_generation_request_still_raises_on_a_resume_refusal():
    """Skipping a generation request asserts later in C++, so it must raise here."""
    manager, _ = _manager(resume_ok=False)

    with pytest.raises(RuntimeError, match="Failed to resume draft KV cache"):
        manager._prepare_draft_resources(_Batch(generation_requests=[_Req(341)]))


def _mirror_manager(resume_ok, cache, create=None):
    """A draft manager carrying only what ``admit_mirror`` reads."""
    manager = object.__new__(KVCacheManagerV2)
    manager.is_draft = True
    manager.kv_cache_map = {341: cache} if cache is not None else {}
    manager._mirror_draft_kv_cache = MagicMock(return_value=cache if cache is not None else create)
    manager._resume_and_restore = MagicMock(return_value=resume_ok)
    return manager


def test_admit_mirror_creates_the_mirror_a_first_sight_request_has_not_got():
    """The hole this replaced, and the run that found it.

    An earlier version answered "nothing to resume" for a request with no
    mirror. That is wrong twice over: ``_prepare_draft_resources`` will create
    one moments later, and a fresh ``_KVCache`` is born SUSPENDED -- so its
    FIRST resume goes through the same pressure gate as any other and can be
    refused. A c=256 run died on exactly that request, the first whose mirror
    was born while the draft pool was above the gate.
    """
    made = MagicMock(name="fresh_kv_cache")
    manager = _mirror_manager(resume_ok=True, cache=None, create=made)

    assert manager.admit_mirror(_Req(341)) is True
    manager._mirror_draft_kv_cache.assert_called_once()
    manager._resume_and_restore.assert_called_once()


def test_admit_mirror_refuses_when_a_fresh_mirror_cannot_be_resumed():
    made = MagicMock(name="fresh_kv_cache")
    manager = _mirror_manager(resume_ok=False, cache=None, create=made)

    assert manager.admit_mirror(_Req(341)) is False


def test_admit_mirror_refuses_when_no_mirror_can_be_created():
    """IndexMapper saturation defers the request rather than forwarding it."""
    manager = _mirror_manager(resume_ok=True, cache=None, create=None)

    assert manager.admit_mirror(_Req(341)) is False
    manager._resume_and_restore.assert_not_called()


def test_admit_mirror_passes_a_real_refusal_through():
    manager = _mirror_manager(resume_ok=False, cache=MagicMock(name="kv_cache"))

    assert manager.admit_mirror(_Req(341)) is False
    manager._resume_and_restore.assert_called_once()


def test_admit_mirror_reports_a_resumed_mirror_as_admissible():
    manager = _mirror_manager(resume_ok=True, cache=MagicMock(name="kv_cache"))

    assert manager.admit_mirror(_Req(341)) is True
