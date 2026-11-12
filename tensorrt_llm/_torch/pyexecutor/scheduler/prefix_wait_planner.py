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
"""Keep a new context request queued while another request prefills a prefix it shares."""

import hashlib
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..llm_request import LlmRequest

# Prompt prefixes are compared in units of this many cache blocks.
PREFIX_UNIT_BLOCKS = 8
# A request waits only when that makes at least this many more blocks reusable.
MIN_GAIN_BLOCKS = 64
# Scheduler passes a request may wait in total.
MAX_WAIT_PASSES = 128


class PrefixWaitPlanner:
    """Decide which new context requests wait for a prefix another request prefills.

    Agent fan-outs send sibling requests that share a long prefix at about the
    same time; admitted together, each of them computes that prefix. A context
    request without a cache claim waits while a request ahead of it shares a
    prefix at least ``MIN_GAIN_BLOCKS`` blocks deeper than the reuse tree
    offers: an owner, i.e. a request admitted in an earlier pass that is still
    prefilling, or a request that is new in this pass and does not wait itself.
    The owner is asked to make the shared depth reusable
    (``KVCacheManagerV2.request_prefix_snapshot``); once it has, the waiting
    request's reuse probe finds the prefix and it is admitted. A request waits
    at most ``MAX_WAIT_PASSES`` passes in total.

    Prefixes are compared by chained digests of ``PREFIX_UNIT_BLOCKS``-block
    units of the prompt, seeded with the request's reuse scope. Multimodal
    prompts are left out: their placeholder tokens do not identify the content.
    Decisions depend only on the requests and the cache manager's reuse tree,
    and each rank schedules once per executor iteration, so ranks that schedule
    the same requests take the same decisions and spend the same wait budget.

    Owned by :class:`KVCacheV2Scheduler`, which calls :meth:`plan` once per
    pass and :meth:`add_owner` for each context request it admits.
    """

    def __init__(self, kv_cache_manager) -> None:
        self._kv_cache_manager = kv_cache_manager
        tokens_per_block = kv_cache_manager.tokens_per_block
        self._unit_tokens = PREFIX_UNIT_BLOCKS * tokens_per_block
        self._min_gain_tokens = MIN_GAIN_BLOCKS * tokens_per_block
        # Prefix digests of this pass's candidates and of the owners.
        self._fingerprints: Dict[int, List[bytes]] = {}
        self._owners: Dict[int, LlmRequest] = {}
        self._owners_by_digest: Dict[bytes, set[int]] = defaultdict(set)
        # Passes each candidate of the last pass has waited.
        self._wait_passes: Dict[int, int] = {}

    def plan(self, requests: Sequence[LlmRequest]) -> Dict[int, Optional[int]]:
        """Context requests to keep queued in this pass.

        ``requests`` are the pass's pending context requests in scheduling
        order. Returns the waiting request ids, each mapped to the id of the
        request new in this pass it waits for, or None when it waits for an
        owner. A request that waits for a new request only waits if the
        scheduler admits that request ahead of it.
        """
        self._drop_finished_owners()
        waiting: Dict[int, Optional[int]] = {}
        wait_passes: Dict[int, int] = {}
        # Candidates of this pass that do not wait, by each of their digests.
        pass_owners: Dict[bytes, LlmRequest] = {}
        for req in requests:
            if not self._is_candidate(req):
                continue
            request_id = req.py_request_id
            fingerprints = self._fingerprints_of(req)
            waited = self._wait_passes.get(request_id, 0)
            owner = (
                self._owner_to_wait_for(req, fingerprints, pass_owners)
                if waited < MAX_WAIT_PASSES
                else None
            )
            if owner is None:
                wait_passes[request_id] = waited
                for digest in fingerprints:
                    pass_owners.setdefault(digest, req)
                continue
            wait_passes[request_id] = waited + 1
            owner_id = owner.py_request_id
            waiting[request_id] = None if owner_id in self._owners else owner_id
        self._wait_passes = wait_passes
        for request_id in [
            request_id
            for request_id in self._fingerprints
            if request_id not in wait_passes and request_id not in self._owners
        ]:
            del self._fingerprints[request_id]
        return waiting

    def add_owner(self, req: LlmRequest) -> None:
        """Let later requests wait for the prefix of ``req``, admitted with a fresh claim."""
        request_id = req.py_request_id
        fingerprints = self._fingerprints.get(request_id)
        if fingerprints is None or request_id in self._owners:
            return
        self._owners[request_id] = req
        for digest in fingerprints:
            self._owners_by_digest[digest].add(request_id)

    def _is_candidate(self, req: LlmRequest) -> bool:
        return (
            req.is_first_context_chunk
            and req.py_request_id not in self._kv_cache_manager.kv_cache_map
            and not req.is_dummy_request
            and req.multimodal_hashes is None
        )

    def _drop_finished_owners(self) -> None:
        """Forget owners that finished their context or gave up their claim."""
        kv_cache_map = self._kv_cache_manager.kv_cache_map
        finished = [
            request_id
            for request_id, owner in self._owners.items()
            if owner.context_remaining_length == 0 or request_id not in kv_cache_map
        ]
        for request_id in finished:
            del self._owners[request_id]
            for digest in self._fingerprints.pop(request_id, ()):
                owner_ids = self._owners_by_digest[digest]
                owner_ids.discard(request_id)
                if not owner_ids:
                    del self._owners_by_digest[digest]

    def _fingerprints_of(self, req: LlmRequest) -> List[bytes]:
        """Chained digests of the prompt's full units, seeded with its reuse scope.

        Equal digests at depth ``k`` mean equal first ``k`` units in the same scope.
        """
        request_id = req.py_request_id
        cached = self._fingerprints.get(request_id)
        if cached is not None:
            return cached
        # Beam 0, which the reuse tree is keyed on.
        tokens = req.get_tokens_view(0)
        unit = self._unit_tokens
        # The last prompt token is recomputed, so it never counts as reusable.
        num_units = max(len(tokens) - 1, 0) // unit
        array = np.asarray(tokens[: num_units * unit], dtype=np.int32)
        scope = repr((req.lora_task_id, req.cache_salt)).encode()
        digest = hashlib.blake2b(scope, digest_size=16).digest()
        fingerprints = []
        for start in range(0, num_units * unit, unit):
            digest = hashlib.blake2b(
                digest + array[start : start + unit].tobytes(), digest_size=16
            ).digest()
            fingerprints.append(digest)
        self._fingerprints[request_id] = fingerprints
        return fingerprints

    def _deepest_owner(self, fingerprints: List[bytes]) -> Tuple[Optional[LlmRequest], int]:
        """The owner that shares the most units with ``fingerprints``, and that count."""
        for depth in range(len(fingerprints), 0, -1):
            owner_ids = self._owners_by_digest.get(fingerprints[depth - 1])
            if owner_ids:
                return self._owners[min(owner_ids)], depth
        return None, 0

    def _owner_to_wait_for(
        self,
        req: LlmRequest,
        fingerprints: List[bytes],
        pass_owners: Dict[bytes, LlmRequest],
    ) -> Optional[LlmRequest]:
        owner, units = self._deepest_owner(fingerprints)
        for depth in range(len(fingerprints), units, -1):
            pass_owner = pass_owners.get(fingerprints[depth - 1])
            if pass_owner is not None:
                owner, units = pass_owner, depth
                break
        if owner is None:
            return None
        shared = units * self._unit_tokens
        reusable = self._kv_cache_manager.probe_context_reuse(req) or 0
        if shared - reusable < self._min_gain_tokens:
            return None
        if not self._kv_cache_manager.request_prefix_snapshot(owner, shared):
            return None
        return owner
