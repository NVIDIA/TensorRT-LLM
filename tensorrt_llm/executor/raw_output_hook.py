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
"""A built-in post-processor hook that records the raw model output.

``RawOutputDump`` records what the model generated for each request as the
post-processor hook sees it: after detokenization, and before the reasoning and
tool-call parsers and the per-endpoint response formatter. Comparing a record
with the response a client received tells an error in the generated output from
one introduced by a parser or by the response formatting.

It records only when both are set: ``--post_processor_hook
tensorrt_llm.executor.raw_output_hook.RawOutputDump`` and the
``TRTLLM_RAW_OUTPUT_DIR`` environment variable, naming the output directory.
Without the variable the hook logs a warning when it is built, then passes every
chunk through and writes nothing.

Each process that runs the hook (the server, or each post-processing worker)
appends JSON lines for finished outputs and partial capture segments to
``$TRTLLM_RAW_OUTPUT_DIR/<UTC hour>/raw-<pid>.jsonl``, for example
``2026-11-04T05/raw-1234.jsonl``.
"""

import json
import os
import threading
from collections import OrderedDict
from datetime import datetime, timezone

from ..logger import logger
from .postprocessor_hook import PostProcessorHookChunk, PostProcessorHookVerdict, emit

__all__ = ["RAW_OUTPUT_DIR_ENV", "RawOutputDump"]

RAW_OUTPUT_DIR_ENV = "TRTLLM_RAW_OUTPUT_DIR"

# Length of the "YYYY-MM-DDTHH" prefix of an ISO-8601 time, which names the hour directory.
_HOUR_PREFIX_LEN = 13

# Dropped records are logged on the first drop and then on every this-many-th one.
_DROP_LOG_INTERVAL = 1000


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _Pending:
    """The retained history segment of one in-flight output."""

    __slots__ = ("frames", "token_ids", "started_at", "last", "history_reason")

    def __init__(self, first: PostProcessorHookChunk) -> None:
        self.frames: list[str] = []
        self.token_ids: list[int] = []
        self.started_at = _utc_now()
        self.history_reason: str | None = None
        # The latest chunk seen for the output; its ``text`` is the full text so far.
        self.last = first


class RawOutputDump:
    """Record each output's generated text and token ids, without changing them.

    ``load_post_processor_hook`` builds one instance per process that runs the
    hook, and it is called once per output per chunk. Every verdict is
    ``emit(chunk.text_diff)``, which leaves the served output unchanged.

    When an output's final chunk arrives, one JSON record is appended with:

    - ``request_id``: ``chunk.request_id``, the engine's id for the request. A
      streaming chat response, or a streaming completions response for a
      single prompt, carries it in its id (``chatcmpl-<id>`` / ``cmpl-<id>``).
    - ``disagg_request_id``: ``chunk.disagg_request_id``, or null when serving is
      not disaggregated. Both ids are written as strings: a disaggregated id can
      be larger than 2**53, past which a JSON reader that parses numbers as
      doubles loses precision.
    - ``output_index``: the output's index within the request.
    - ``text``: the output's full detokenized text.
    - ``frames``: the text each chunk added (``chunk.text_diff``), in order;
      chunks that added none are left out. They show where the stream was
      split, which is where incremental parsers go wrong.
    - ``token_ids``, ``num_output_tokens``: the captured history of generated
      token ids from ``chunk.token_ids_diff``, and its count, not necessarily
      the final output's total. They also show tokens that detokenization
      leaves out of ``text``, such as skipped special tokens.
    - ``frames_consistent``: whether ``frames`` add up to ``text``.
    - ``streaming``, ``aborted``: from the last observed chunk.
    - ``started_at``, ``finished_at``: UTC ISO-8601 times of the current retained
      segment's first chunk and of the record.
    - ``partial``, ``end_reason``: whether capture ended early or its history is
      incomplete or uncertain, and why. They can be true and set even after a
      normal final chunk; they are not the model's stop reason.
    - ``event``: always ``"raw_output"``.

    ``frames`` and ``token_ids`` preserve the history handed to the hook, not a
    reconstruction of the final output: for example, a stop-word correction
    can remove text or token ids already seen in earlier chunks. When a request
    generates several sequences (``n`` or ``best_of`` above 1, which includes
    beam search), these channels are what the hook was handed for that output
    index, which need not be one sequence's own history: ``token_ids`` can repeat
    ids, and when sampling with ``best_of`` above ``n`` the index is a rank that can change
    between chunks, so one record can join chunks of different sequences.
    ``text`` is always the text at the output's last chunk, and
    ``frames_consistent`` is false when ``frames`` do not add up to it.

    When :meth:`on_request_end` reports an error, retained segments that produced
    text or token ids are written with ``partial`` true and ``end_reason`` set;
    already evicted segments are not rewritten. An output that gets neither a
    final chunk nor that callback (without the worker pool, a
    stream whose client disconnects first) stays in flight. At most
    :attr:`max_pending` outputs are kept in flight: past that, the least
    recently updated one is written with ``partial`` true and ``end_reason``
    ``"evicted"``. If it later resumes, its new segment is also marked
    ``"evicted"``, even if that segment receives a final chunk. At most
    ``max_pending`` evicted keys are remembered. If that budget overflows,
    future new segments without a known eviction marker are conservatively
    marked ``"history_unavailable"`` until the hook instance restarts, including
    newly started outputs. Already retained segments keep their history status;
    capture continues in either case. ``partial`` false means no known history
    gap, not that the recorded history equals the final output snapshot.

    The hook seam fails a request whose hook raises. A debug dump must not, so
    this hook never raises: a record that cannot be written is dropped and
    counted, and the count is logged as a warning on the first drop and on
    every 1000th after it.
    """

    #: The most outputs kept in flight. A subclass can set another limit and be
    #: passed to ``--post_processor_hook`` instead.
    max_pending = 4096

    def __init__(self) -> None:
        self._dir = os.environ.get(RAW_OUTPUT_DIR_ENV) or None
        self._file_name = f"raw-{os.getpid()}.jsonl"
        # Ordered from the least to the most recently updated output.
        self._pending: OrderedDict[tuple[int, int], _Pending] = OrderedDict()
        self._evicted_keys: set[tuple[int, int]] = set()
        # Once an eviction cannot be remembered, an unknown key may be a continuation.
        self._history_uncertain = False
        # One instance can be called from more than one thread of its process (for
        # example, threads each waiting on an ``LLM`` result): ``_lock`` guards the
        # pending/history state and the drop count, ``_write_lock`` keeps each line whole.
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._dropped = 0
        if self._dir is None:
            logger.warning("%s is not set, so no raw model output is recorded", RAW_OUTPUT_DIR_ENV)
        else:
            logger.info("Recording raw model output to %s", self._dir)

    def __call__(self, chunk: PostProcessorHookChunk) -> PostProcessorHookVerdict:
        verdict = emit(chunk.text_diff)
        if self._dir is None:
            return verdict
        try:
            ended = self._observe(chunk)
        except Exception as error:  # noqa: BLE001 - a debug dump must not fail a request
            self._note_drop(error)
            return verdict
        for pending, end_reason in ended:
            self._write_or_drop(pending, end_reason)
        return verdict

    def on_request_end(self, request_id: int, reason: str) -> None:
        """Release a request that ended without a final chunk, recording what it generated.

        Called through ``end_post_processor_hook`` when a request ends in an
        error. Each of the request's outputs that produced text or token ids is
        written with ``partial`` true and ``end_reason`` set to ``reason``.

        Args:
            request_id: The request's ``chunk.request_id``.
            reason: Why the request ended, as the caller reports it.
        """
        with self._lock:
            keys = [key for key in self._pending if key[0] == request_id]
            ended = [self._pending.pop(key) for key in keys]
            self._evicted_keys = {key for key in self._evicted_keys if key[0] != request_id}
        for pending in ended:
            self._write_or_drop(pending, reason)

    def _observe(self, chunk: PostProcessorHookChunk) -> list[tuple[_Pending, str | None]]:
        """Add ``chunk`` to its output's state.

        Returns the outputs to write, each with its end reason: the chunk's
        output once it has finished (reason None), and any output evicted to
        keep at most ``max_pending`` in flight (reason ``"evicted"``).
        """
        key = (chunk.request_id, chunk.output_index)
        ended = []
        warn_uncertain = False
        with self._lock:
            pending = self._pending.pop(key, None)
            if pending is None:
                pending = _Pending(chunk)
                if key in self._evicted_keys:
                    self._evicted_keys.remove(key)
                    pending.history_reason = "evicted"
                elif self._history_uncertain:
                    pending.history_reason = "history_unavailable"
            pending.last = chunk
            # A chunk that added no text (the final call often adds none) is not a frame.
            if chunk.text_diff:
                pending.frames.append(chunk.text_diff)
            pending.token_ids.extend(chunk.token_ids_diff)
            if chunk.is_final:
                # ``is_final`` is request-level: every output the result returns gets its
                # final chunk in the same call, each under its own key. When sampling with
                # best_of > n, the result returns the top n and each key holds a rank, so a
                # sequence that drops out of the top n gets no final chunk under the key it
                # had: the sequence now at that rank continues its entry (see the class
                # docstring).
                ended.append((pending, None))
            else:
                self._pending[key] = pending
            while len(self._pending) > self.max_pending:
                evicted_key, evicted = self._pending.popitem(last=False)
                ended.append((evicted, "evicted"))
                if len(self._evicted_keys) < self.max_pending:
                    self._evicted_keys.add(evicted_key)
                elif not self._history_uncertain:
                    self._history_uncertain = True
                    warn_uncertain = True
        if warn_uncertain:
            logger.warning(
                "Raw output eviction history reached its key limit; future new captures "
                "without a known eviction marker are marked history_unavailable until "
                "the hook restarts. Recording continues."
            )
        return ended

    def _write_or_drop(self, pending: _Pending, end_reason: str | None) -> None:
        """Write the record of ``pending``; a record that cannot be written is counted as dropped.

        An output that ended early (the actual ``end_reason`` set) without
        producing text or token ids in its retained segment is not written.
        Capture-history reasons are applied only in ``_write``: a normal final
        chunk remains eligible even when its retained segment is empty.
        """
        if end_reason is not None and not (pending.frames or pending.token_ids):
            return
        try:
            self._write(pending, end_reason)
        except Exception as error:  # noqa: BLE001 - a debug dump must not fail a request
            self._note_drop(error)

    def _write(self, pending: _Pending, end_reason: str | None = None) -> None:
        """Append the record of one output to this process's file for the current hour."""
        if end_reason is None:
            end_reason = pending.history_reason
        chunk = pending.last
        finished_at = _utc_now()
        record = {
            "event": "raw_output",
            "request_id": str(chunk.request_id),
            "disagg_request_id": (
                None if chunk.disagg_request_id is None else str(chunk.disagg_request_id)
            ),
            "output_index": chunk.output_index,
            "started_at": pending.started_at,
            "finished_at": finished_at,
            "streaming": chunk.streaming,
            "aborted": chunk.aborted,
            "partial": end_reason is not None,
            "end_reason": end_reason,
            "text": chunk.text,
            "frames": pending.frames,
            "frames_consistent": "".join(pending.frames) == chunk.text,
            "token_ids": pending.token_ids,
            "num_output_tokens": len(pending.token_ids),
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        directory = os.path.join(self._dir, finished_at[:_HOUR_PREFIX_LEN])
        # One open per record, when a retained segment finishes rather than per chunk;
        # the hook has no shutdown call that could close a kept-open file.
        with self._write_lock:
            os.makedirs(directory, exist_ok=True)
            with open(os.path.join(directory, self._file_name), "a", encoding="utf-8") as file:
                file.write(line)

    def _note_drop(self, error: Exception) -> None:
        with self._lock:
            self._dropped += 1
            dropped = self._dropped
        if dropped == 1 or dropped % _DROP_LOG_INTERVAL == 0:
            logger.warning("Dropped %d raw output record(s); latest error: %s", dropped, error)
