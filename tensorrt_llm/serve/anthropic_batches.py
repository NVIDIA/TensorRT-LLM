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
"""In-memory store and executor for the Anthropic Message Batches API.

A batch is a set of ordinary Messages requests submitted together and run
asynchronously. This module owns their lifecycle - admission, execution,
cancellation, expiry, and result rendering - while the HTTP routes and the
per-request work stay in openai_server.

DURABILITY: batches live only in this process. A server restart loses them,
and their ids then resolve to 404. Anthropic's contract is 24h durability, so
this is a deliberate deviation: persisting would mean a storage directory, a
file layout, and cleanup of its own, which is a larger change than the feature
warrants here. Callers that need results across a restart must re-submit.

SCHEDULING: batch work shares the engine with interactive traffic, so it is
capped by a semaphore rather than dispatched all at once. An uncapped batch of
100k requests would flood the scheduler and starve live clients, which is
exactly the failure a batch API is supposed to avoid.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.anthropic_protocol import (
    AnthropicBatchRequestCounts,
    AnthropicBatchRequestItem,
    AnthropicMessageBatch,
    AnthropicMessagesRequest,
)

# How many batched requests may be in flight at once, across all batches.
# Deliberately small: batches are throughput work with no waiting user, and the
# engine is shared with interactive traffic that does have one.
BATCH_CONCURRENCY_ENV = "TRTLLM_ANTHROPIC_BATCH_CONCURRENCY"
DEFAULT_BATCH_CONCURRENCY = 4

# Anthropic expires batches 24h after creation.
BATCH_TTL = timedelta(hours=24)

# Bound on retained batches. Results are held in memory, so without a cap a
# long-lived server accumulates them until it dies.
MAX_RETAINED_BATCHES_ENV = "TRTLLM_ANTHROPIC_BATCH_MAX_RETAINED"
DEFAULT_MAX_RETAINED_BATCHES = 100


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rfc3339(value: datetime) -> str:
    """RFC 3339 with a trailing Z, which is the shape the API documents."""
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default
    if value < 1:
        logger.warning("%s=%d must be >= 1; using %d", name, value, default)
        return default
    return value


# One request's outcome. `payload` is the message body for a success, or the
# error envelope otherwise.
@dataclass
class _ItemResult:
    custom_id: str
    result_type: str
    payload: Dict[str, Any]

    def as_jsonl_obj(self) -> Dict[str, Any]:
        if self.result_type == "succeeded":
            result: Dict[str, Any] = {"type": "succeeded", "message": self.payload}
        elif self.result_type == "errored":
            # MessageBatchErroredResult.error is an ErrorResponse, which wraps
            # the ErrorObject in ANOTHER layer: {type: "error", error: {...},
            # request_id}. Emitting the bare ErrorObject here breaks SDK
            # deserialization of result.error, so keep both levels.
            result = {
                "type": "errored",
                "error": {
                    "type": "error",
                    "error": self.payload,
                    "request_id": None,
                },
            }
        else:
            # canceled and expired carry no body in the documented schema.
            result = {"type": self.result_type}
        return {"custom_id": self.custom_id, "result": result}


@dataclass
class _BatchRecord:
    batch: AnthropicMessageBatch
    items: List[AnthropicBatchRequestItem]
    results: List[_ItemResult] = field(default_factory=list)
    task: Optional[asyncio.Task] = None
    # Set by cancel(); the worker checks it between requests. In-flight requests
    # are allowed to finish rather than being torn down mid-generation, which
    # matches "cancellation is best-effort" and keeps the engine consistent.
    cancel_requested: bool = False

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        now = now or _now()
        return now >= datetime.fromisoformat(self.batch.expires_at.replace("Z", "+00:00"))


# Runs one Messages request and returns (result_type, payload). Supplied by the
# server so this module stays free of engine and HTTP concerns.
BatchItemRunner = Callable[[AnthropicMessagesRequest], Awaitable[Tuple[str, Dict[str, Any]]]]


class AnthropicBatchStore:
    """Owns every batch this process knows about."""

    def __init__(
        self,
        runner: BatchItemRunner,
        concurrency: Optional[int] = None,
        max_retained: Optional[int] = None,
    ) -> None:
        self._runner = runner
        self._batches: Dict[str, _BatchRecord] = {}
        self._concurrency = concurrency or _int_env(
            BATCH_CONCURRENCY_ENV, DEFAULT_BATCH_CONCURRENCY
        )
        self._max_retained = max_retained or _int_env(
            MAX_RETAINED_BATCHES_ENV, DEFAULT_MAX_RETAINED_BATCHES
        )
        self._semaphore = asyncio.Semaphore(self._concurrency)

    # -- lifecycle ---------------------------------------------------------

    def create(self, items: List[AnthropicBatchRequestItem]) -> AnthropicMessageBatch:
        """Admit a batch and start it. Raises ValueError on duplicate custom_id."""
        seen = set()
        for item in items:
            if item.custom_id in seen:
                # Results are matched by custom_id, so duplicates would make the
                # response ambiguous. Reject at admission rather than produce a
                # result set the client cannot interpret.
                raise ValueError(
                    f"custom_id {item.custom_id!r} is repeated; must be unique within a batch"
                )
            seen.add(item.custom_id)

        created = _now()
        batch = AnthropicMessageBatch(
            created_at=_rfc3339(created),
            expires_at=_rfc3339(created + BATCH_TTL),
            request_counts=AnthropicBatchRequestCounts(processing=len(items)),
        )
        record = _BatchRecord(batch=batch, items=list(items))
        self._evict_if_needed()
        self._batches[batch.id] = record
        record.task = asyncio.create_task(self._run(record))
        logger.info("Anthropic batch %s created with %d request(s)", batch.id, len(items))
        return batch

    def get(self, batch_id: str) -> Optional[AnthropicMessageBatch]:
        record = self._batches.get(batch_id)
        if record is None:
            return None
        self._expire_if_due(record)
        return record.batch

    def list(
        self, limit: int = 20, after_id: Optional[str] = None, before_id: Optional[str] = None
    ) -> Tuple[List[AnthropicMessageBatch], bool]:
        """One page of batches, newest first, plus whether more remain.

        Returns (page, has_more). The caller needs has_more because a bare
        truncated list is indistinguishable from a complete one: a client that
        asked for 20 and got 20 has no way to tell whether it has seen
        everything, so it silently misses batches.
        """
        # Clamp before slicing: a negative limit would turn batches[:limit]
        # into "all but the last N", silently hiding the newest batches from a
        # client that asked for fewer.
        limit = max(1, min(int(limit), self._max_retained))
        records = list(self._batches.values())
        for record in records:
            self._expire_if_due(record)
        # Newest first, which is the useful order for polling clients. Ties on
        # created_at are broken by id so the order is total - an unstable sort
        # key would let a cursor skip or repeat entries between pages.
        batches = sorted(
            (r.batch for r in records), key=lambda b: (b.created_at, b.id), reverse=True
        )

        # Cursors are ids, not offsets: batches are created and evicted while a
        # client pages, and an offset would shift under it. An unknown cursor
        # raises rather than returning an empty page, because an empty page is
        # byte-identical to a finished walk - the client would stop early
        # believing it had seen everything. The batch the client cursored on is
        # routinely gone: fetch a page, process and DELETE each batch, then ask
        # for the next page with after_id=<a batch just deleted>.
        if after_id is not None:
            index = next((i for i, b in enumerate(batches) if b.id == after_id), None)
            if index is None:
                raise LookupError(after_id)
            batches = batches[index + 1 :]
        if before_id is not None:
            index = next((i for i, b in enumerate(batches) if b.id == before_id), None)
            if index is None:
                raise LookupError(before_id)
            # Everything before the cursor, then the LAST `limit` of them: with
            # newest-first ordering the page immediately before the cursor is
            # the tail of that prefix. Taking the head returns the newest
            # batches instead, and a client walking backwards from there
            # immediately dead-ends on its own first page.
            older = batches[:index]
            page = older[-limit:]
            return page, len(older) > len(page)

        page = batches[:limit]
        return page, len(batches) > len(page)

    def cancel(self, batch_id: str) -> Optional[AnthropicMessageBatch]:
        record = self._batches.get(batch_id)
        if record is None:
            return None
        # Expiry is evaluated on read, so it has to run on every entry point.
        # Skipping it here let an already-expired batch be marked "canceling",
        # reporting its requests as canceled rather than expired.
        self._expire_if_due(record)
        if record.batch.processing_status == "in_progress":
            record.cancel_requested = True
            record.batch.processing_status = "canceling"
            record.batch.cancel_initiated_at = _rfc3339(_now())
        return record.batch

    def delete(self, batch_id: str) -> bool:
        """Archive and drop a batch. Only meaningful once it has ended."""
        record = self._batches.get(batch_id)
        if record is None:
            return False
        # Without this an expired batch never transitions to "ended" unless
        # someone happens to GET it, so DELETE would refuse it indefinitely.
        self._expire_if_due(record)
        if record.batch.processing_status != "ended":
            raise ValueError(
                "Message Batch cannot be deleted while it is still processing; "
                "cancel it first and wait for processing to end"
            )
        del self._batches[batch_id]
        return True

    def results(self, batch_id: str) -> Optional[List[Dict[str, Any]]]:
        """The .jsonl payload, or None if the batch is unknown/unfinished."""
        record = self._batches.get(batch_id)
        if record is None:
            return None
        self._expire_if_due(record)
        if record.batch.processing_status != "ended":
            return None
        return [item.as_jsonl_obj() for item in record.results]

    def has(self, batch_id: str) -> bool:
        return batch_id in self._batches

    def status_of(self, batch_id: str) -> Optional[str]:
        record = self._batches.get(batch_id)
        if record is None:
            return None
        self._expire_if_due(record)
        return record.batch.processing_status

    # -- internals ---------------------------------------------------------

    def _evict_if_needed(self) -> None:
        if len(self._batches) < self._max_retained:
            return
        # Drop the oldest ended batch. Never evict one that is still running:
        # losing a live batch would strand the client polling it.
        ended = [
            (r.batch.created_at, bid)
            for bid, r in self._batches.items()
            if r.batch.processing_status == "ended"
        ]
        if not ended:
            logger.warning(
                "Anthropic batch store holds %d unfinished batches (limit %d); "
                "not evicting a running batch",
                len(self._batches),
                self._max_retained,
            )
            return
        ended.sort()
        evicted = ended[0][1]
        del self._batches[evicted]
        logger.info("Evicted ended Anthropic batch %s to stay within the retention limit", evicted)

    def _expire_if_due(self, record: _BatchRecord) -> None:
        """Expire lazily, on read.

        A background sweeper would need its own task and shutdown handling for
        a deadline that is 24h away; checking when someone actually looks is
        equivalent from the client's point of view and has no lifecycle.
        """
        if record.batch.processing_status == "ended" or not record.is_expired():
            return
        remaining = len(record.items) - len(record.results)
        if remaining > 0:
            record.results.extend(
                _ItemResult(item.custom_id, "expired", {})
                for item in record.items[len(record.results) :]
            )
        if record.task is not None and not record.task.done():
            record.task.cancel()
        self._finalize(record)

    async def _run(self, record: _BatchRecord) -> None:
        """Run every request in the batch, self._concurrency at a time.

        Items are dispatched together and gated by the semaphore, NOT awaited
        one after another. An earlier version held the semaphore inside a
        sequential loop, which pinned a single batch to one in-flight request
        no matter how the limit was configured - a 1000-request batch became
        1000 serial round-trips, which is precisely the latency a batch API is
        meant to amortise. The semaphore still bounds the total in flight, so
        interactive traffic is protected exactly as before.
        """

        async def run_one(item: AnthropicBatchRequestItem) -> None:
            if record.cancel_requested or record.is_expired():
                state = "canceled" if record.cancel_requested else "expired"
                record.results.append(_ItemResult(item.custom_id, state, {}))
                return
            async with self._semaphore:
                # Re-check after queueing: cancellation or expiry may have
                # arrived while this item waited for a slot, and starting
                # generation then would waste engine time on a dead batch.
                if record.cancel_requested or record.is_expired():
                    state = "canceled" if record.cancel_requested else "expired"
                    record.results.append(_ItemResult(item.custom_id, state, {}))
                    return
                try:
                    result_type, payload = await self._runner(item.params)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - one bad request must not kill the batch
                    logger.error(
                        "Anthropic batch %s request %s failed: %s",
                        record.batch.id,
                        item.custom_id,
                        exc,
                    )
                    result_type, payload = (
                        "errored",
                        {
                            "type": "api_error",
                            "message": "Internal server error",
                        },
                    )
            # Appended as each item lands rather than collected at the end, so
            # a cancelled task keeps the results it already earned; _finalize
            # pads only what never ran.
            record.results.append(_ItemResult(item.custom_id, result_type, payload))

        try:
            await asyncio.gather(*(run_one(item) for item in record.items))
        except asyncio.CancelledError:
            # Expiry cancels the task; whatever finished is already recorded.
            raise
        finally:
            self._finalize(record)

    def _finalize(self, record: _BatchRecord) -> None:
        if record.batch.processing_status == "ended":
            return
        # Pad anything the worker never reached. _run finalizes from a finally
        # block, so a cancelled task (loop teardown, expiry) would otherwise end
        # the batch with fewer results than requests - request_counts would sum
        # to less than the batch size, breaking the one invariant clients divide
        # by, and results_url would serve a truncated .jsonl.
        if len(record.results) < len(record.items):
            done = {result.custom_id for result in record.results}
            record.results.extend(
                _ItemResult(item.custom_id, "canceled", {})
                for item in record.items
                if item.custom_id not in done
            )
        counts = AnthropicBatchRequestCounts()
        for result in record.results:
            setattr(counts, result.result_type, getattr(counts, result.result_type) + 1)
        record.batch.request_counts = counts
        record.batch.processing_status = "ended"
        record.batch.ended_at = _rfc3339(_now())
        # Relative rather than absolute: the server does not reliably know the
        # scheme/host a client reached it by, and a wrong absolute URL is worse
        # than a relative one the client can resolve itself.
        record.batch.results_url = f"/v1/messages/batches/{record.batch.id}/results"
        logger.info(
            "Anthropic batch %s ended: %s",
            record.batch.id,
            record.batch.request_counts.model_dump(),
        )


def results_to_jsonl(objs: List[Dict[str, Any]]) -> str:
    """Render result objects as newline-delimited JSON."""
    return "".join(json.dumps(obj) + "\n" for obj in objs)
