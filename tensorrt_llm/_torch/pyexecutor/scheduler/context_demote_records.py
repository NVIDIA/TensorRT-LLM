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
"""A lower bound of a waiting request's reusable prefix from its conversation's earlier prompts.

With long-prefill demotion and its reuse probe (scheduler_config.context_demote_probe) the
scheduler classifies a waiting first chunk by a radix walk over its whole prompt, one hash a
block: 1.2-1.5 ms a request at agentic prompt lengths. A continuing turn of a conversation
usually starts with the previous turn's prompt, whose prefix that turn's prefill committed. Here
the scheduler keeps, per conversation, the prompts of the requests it scheduled, cut at the
depth their prefill leaves reusable (``committed_prompt_depth``). A new prompt that starts with
such a prefix reuses at least it, so when what is left is within the demotion threshold the
request is short and no walk is needed (scheduler_config.context_demote_no_walk).

That is wrong only when those blocks are no longer in the tree (dropped by the cache tiers, or
not committed): the request is then not demoted and prefills in arrival order. It never demotes
a request: a record that leaves the request long is only a lower bound, and the walk decides.
The class affects the context order only.
"""

import time
from collections import Counter, OrderedDict
from typing import Callable, Hashable, Optional, Sequence

import numpy as np

from tensorrt_llm.logger import logger

from ..llm_request import LlmRequest


def _conversation_key(req: LlmRequest) -> Optional[Hashable]:
    """The conversation and reuse scope a record is kept under, or None."""
    if req.is_dummy_request:
        return None
    params = req.py_conversation_params
    if params is None:
        return None
    conversation_id = params.conversation_id.strip()
    if not conversation_id:
        return None
    return (conversation_id, req.lora_task_id, req.cache_salt)


class _Record:
    """One prompt of a conversation: its first ``depth`` tokens in ``tokens`` (with room to
    extend in place), the request that wrote it and when."""

    __slots__ = ("request_id", "tokens", "depth", "time")

    def __init__(self) -> None:
        self.request_id = -1
        self.tokens: Optional[np.ndarray] = None
        self.depth = 0
        self.time = 0.0


class ConversationPromptRecords:
    """Per conversation (and reuse scope), the committed prefixes of up to ``SLOTS`` recent
    prompts: an agentic client can replay one conversation id in several sessions at different
    depths. A scheduled request extends the record it was proved against (its own session's
    last turn) and otherwise takes a free slot or the oldest one. Executor thread only."""

    #: Conversations kept (least recently used dropped first).
    MAX_CONVERSATIONS = 1024
    SLOTS = 3
    #: A record older than this is not trusted, and is dropped as others are written: the
    #: cache tiers drop idle conversations first.
    MAX_AGE_S = 300.0
    #: Spare tokens in a record's buffer, so the next turns append in place.
    HEADROOM = 49152
    #: Requests proved and not scheduled yet that are remembered.
    MAX_PROOFS = 4096
    #: Probes between two counter lines.
    LOG_EVERY = 8192

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._records: "OrderedDict[Hashable, list[_Record]]" = OrderedDict()
        # request id -> (the record it was proved against, that record's writer then)
        self._proofs: dict[int, tuple[_Record, int]] = {}
        self.stats: Counter = Counter()

    def __len__(self) -> int:
        return len(self._records)

    def proven_depth(
        self, req: LlmRequest, tokens: Sequence[int], holds_cache: Callable[[int], bool]
    ) -> Optional[int]:
        """The depth of the deepest record of the request's conversation that its prompt
        ``tokens`` starts with, or None: no record, too old, written by this request or by one
        that still holds its cache (it may not have committed yet), or the prompt leaves every
        record before its end."""
        stats = self.stats
        stats["calls"] += 1
        if stats["calls"] % self.LOG_EVERY == 0:
            logger.info(
                "KVCacheV2Scheduler: demotion probe from conversation records "
                + " ".join(f"{name}={value}" for name, value in sorted(stats.items()))
                + f" conversations={len(self._records)}"
            )
        key = _conversation_key(req)
        slots = self._records.get(key) if key is not None else None
        if not slots:
            stats["no_record"] += 1
            return None
        self._records.move_to_end(key)
        prompt = np.asarray(tokens, dtype=np.int32)
        # The last prompt token is always recomputed.
        limit = len(prompt) - 1
        now = self._clock()
        why = "left_record"
        for record in sorted(slots, key=lambda r: -r.depth):
            if record.request_id == req.py_request_id or holds_cache(record.request_id):
                why = "inflight"
            elif now - record.time > self.MAX_AGE_S:
                if why != "inflight":
                    why = "stale"
            elif record.depth <= limit and np.array_equal(
                prompt[: record.depth], record.tokens[: record.depth]
            ):
                if len(self._proofs) >= self.MAX_PROOFS:
                    self._proofs.clear()
                self._proofs[req.py_request_id] = (record, record.request_id)
                return record.depth
        stats[why] += 1
        return None

    def record(self, req: LlmRequest, tokens: Sequence[int], depth: int) -> None:
        """Remember the first ``depth`` tokens of a request whose first chunk was scheduled."""
        rid = req.py_request_id
        proof = self._proofs.pop(rid, None)
        key = _conversation_key(req)
        if key is None or depth <= 0:
            return
        slots = self._records.get(key)
        if slots is None:
            slots = self._records[key] = []
        elif any(record.request_id == rid for record in slots):
            # Scheduled before (preempted and rewound): its record stands.
            return
        record = None
        if (
            proof is not None
            and proof[0].request_id == proof[1]
            and any(r is proof[0] for r in slots)
        ):
            # Unchanged since this request was proved against it: its own session goes on.
            record = proof[0]
        extend = record is not None and record.depth <= depth
        if record is None and len(slots) >= self.SLOTS:
            record = min(slots, key=lambda r: r.time)
        if record is None:
            record = _Record()
            slots.append(record)
        start = record.depth if extend else 0
        prompt = np.asarray(tokens, dtype=np.int32)
        if record.tokens is None or record.tokens.size < depth:
            buffer = np.empty(depth + self.HEADROOM, dtype=np.int32)
            if start:
                buffer[:start] = record.tokens[:start]
            record.tokens = buffer
        # A proved continuation adds its tail only; anything else is copied whole.
        record.tokens[start:depth] = prompt[start:depth]
        now = self._clock()
        record.request_id, record.depth, record.time = rid, depth, now
        self._records.move_to_end(key)
        self.stats["records_written"] += 1
        # Records too old to answer only hold memory: this conversation's other slots, then
        # whole conversations from the least recently used end.
        if len(slots) > 1:
            slots[:] = [r for r in slots if now - r.time <= self.MAX_AGE_S]
        while len(self._records) > 1:
            oldest = next(iter(self._records.values()))
            if len(self._records) <= self.MAX_CONVERSATIONS and any(
                now - r.time <= self.MAX_AGE_S for r in oldest
            ):
                break
            self._records.popitem(last=False)
