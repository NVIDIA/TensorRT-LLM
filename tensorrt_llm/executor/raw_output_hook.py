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
appends one JSON line per finished output to
``$TRTLLM_RAW_OUTPUT_DIR/<UTC hour>/raw-<pid>.jsonl``, for example
``2026-11-04T05/raw-1234.jsonl``.
"""

import json
import os
import threading
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
    """What one in-flight output has produced so far."""

    __slots__ = ("frames", "token_ids", "started_at", "last")

    def __init__(self, first: PostProcessorHookChunk) -> None:
        self.frames: list[str] = []
        self.token_ids: list[int] = []
        self.started_at = _utc_now()
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
    - ``token_ids``, ``num_output_tokens``: the generated token ids, collected
      from ``chunk.token_ids_diff``, and their count. They also show tokens that
      detokenization leaves out of ``text``, such as skipped special tokens.
    - ``streaming``, ``aborted``: from the final chunk.
    - ``started_at``, ``finished_at``: UTC ISO-8601 times of the output's first
      chunk and of the record.
    - ``partial``, ``end_reason``: false and null for an output that finished.
    - ``event``: always ``"raw_output"``.

    A request that a post-processing worker drops because of an error gets no
    final chunk; :meth:`on_request_end` writes what each of its outputs had
    generated, with ``partial`` true and ``end_reason`` set. A request that
    gets neither (without the worker pool: an executor error, or a stream whose
    client disconnects first) is not recorded, and its state is kept until the
    process exits.

    The hook seam fails a request whose hook raises. A debug dump must not, so
    this hook never raises: a record that cannot be written is dropped and
    counted, and the count is logged as a warning on the first drop and on
    every 1000th after it.
    """

    def __init__(self) -> None:
        self._dir = os.environ.get(RAW_OUTPUT_DIR_ENV) or None
        self._file_name = f"raw-{os.getpid()}.jsonl"
        self._pending: dict[tuple[int, int], _Pending] = {}
        # One instance can be called from more than one thread of its process (for
        # example, threads each waiting on an ``LLM`` result): ``_lock`` guards the
        # pending state and the drop count, ``_write_lock`` keeps each line whole.
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
            finished = self._observe(chunk)
            if finished is not None:
                self._write(finished)
        except Exception as error:  # noqa: BLE001 - a debug dump must not fail a request
            self._note_drop(error)
        return verdict

    def on_request_end(self, request_id: int, reason: str) -> None:
        """Release a request that ended without a final chunk, recording what it generated.

        Called through ``end_post_processor_hook`` by the post-processing worker
        when it drops a request after an error. Each of the request's outputs
        that produced text or token ids is written with ``partial`` true and
        ``end_reason`` set to ``reason``.

        Args:
            request_id: The request's ``chunk.request_id``.
            reason: Why the request ended, as the worker reports it.
        """
        with self._lock:
            keys = [key for key in self._pending if key[0] == request_id]
            ended = [self._pending.pop(key) for key in keys]
        for pending in ended:
            if not (pending.frames or pending.token_ids):
                continue
            try:
                self._write(pending, end_reason=reason)
            except Exception as error:  # noqa: BLE001 - a debug dump must not fail a request
                self._note_drop(error)

    def _observe(self, chunk: PostProcessorHookChunk) -> _Pending | None:
        """Add ``chunk`` to its output's state; return the state once the output has finished."""
        key = (chunk.request_id, chunk.output_index)
        with self._lock:
            pending = self._pending.get(key)
            if pending is None:
                pending = self._pending[key] = _Pending(chunk)
            pending.last = chunk
            # A chunk that added no text (the final call often adds none) is not a frame.
            if chunk.text_diff:
                pending.frames.append(chunk.text_diff)
            pending.token_ids.extend(chunk.token_ids_diff)
            if not chunk.is_final:
                return None
            # ``is_final`` is request-level: with n > 1 every output of the request gets
            # its final chunk in the same call, each under its own key.
            del self._pending[key]
            return pending

    def _write(self, pending: _Pending, end_reason: str | None = None) -> None:
        """Append the record of one output to this process's file for the current hour."""
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
            "token_ids": pending.token_ids,
            "num_output_tokens": len(pending.token_ids),
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        directory = os.path.join(self._dir, finished_at[:_HOUR_PREFIX_LEN])
        # One open per record: records are written once per finished output, not per
        # chunk, and the hook has no shutdown call that could close a kept-open file.
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
