# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-request streaming state: response formatting, registration, stall watchdog."""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from openengine.v1 import error_pb2, generation_pb2

from .disagg import prefill_ready_response
from .formatting import (
    _MAX_PROMPT_LOGPROB_ENTRIES,
    _TERMINAL_FINISH_REASONS,
    _delta_logprobs,
    _engine_error_response,
    _finish_event,
    _logprob_shortfall,
    _prompt_output,
    _StopPrefixTracker,
    _token_holdback,
    _token_infos,
    _usage,
)

RESPONSE_STALL_TIMEOUT_SECONDS = 30.0


@dataclass(kw_only=True)
class _OpenEngineFormatState:
    """Per-request state for ``_format_result``."""

    tokenizer: Any = None
    request_id: str = ""
    sampling_params: Any = None
    is_context_only: bool = False
    kv_transfer_backend: str = ""
    # None until first use: _token_holdback needs sampling_params._stop_word_ids,
    # which generate_async only populates after the request is submitted, so it is
    # computed lazily on the first _format_result call (inline or in a worker).
    token_holdback: Optional[int] = None
    stop_texts: list = field(default_factory=list)
    exclude_stop: bool = True
    sent_token_counts: dict = field(default_factory=dict)
    sent_text_lengths: dict = field(default_factory=dict)
    observed_text_lengths: dict = field(default_factory=dict)
    stop_prefix_trackers: dict = field(default_factory=dict)
    # Per output index, how many leading tokens carry no logprob. Fixed for the
    # stream once the engine has produced any, so it is resolved once instead of
    # being re-derived from the growing cumulative list on every step.
    logprob_shortfalls: dict = field(default_factory=dict)
    finished_indices: set = field(default_factory=set)
    prompt_sent: bool = False
    prefill_ready_sent: set = field(default_factory=set)


def _format_result(
    result: Any, args: _OpenEngineFormatState
) -> list[generation_pb2.GenerateResponse]:
    """Format one engine result into OpenEngine responses.

    Produces prompt/token/finished/prefill/error events. Runs on the event loop
    when postprocessing workers are off, or inside a worker process when they are
    on — the servicer only relays these responses, keeping the CPU-bound detok +
    protobuf build off its event loop under load.
    """
    request_id = args.request_id
    sampling_params = args.sampling_params
    tokenizer = args.tokenizer
    if args.token_holdback is None:
        args.token_holdback = _token_holdback(sampling_params)
    responses: list[generation_pb2.GenerateResponse] = []

    if result.error:
        responses.append(_engine_error_response(request_id, str(result.error), result))
        return responses

    if sampling_params.prompt_logprobs is not None and not args.prompt_sent and result.outputs:
        # Sized by prompt_length * candidates, both client-controlled, and built
        # in one non-preemptible slice: refuse rather than stall every other
        # in-flight stream for the seconds-to-minutes this would take.
        entries = len(result.prompt_token_ids or ()) * max(sampling_params.prompt_logprobs, 1)
        if entries > _MAX_PROMPT_LOGPROB_ENTRIES:
            args.prompt_sent = True
            responses.append(
                _engine_error_response(
                    request_id,
                    f"prompt logprobs would produce {entries} entries, above the "
                    f"{_MAX_PROMPT_LOGPROB_ENTRIES} supported; lower "
                    "response.prompt_candidates or shorten the prompt",
                    result,
                    code=error_pb2.ERROR_CODE_INVALID_ARGUMENT,
                )
            )
            return responses
        responses.append(
            generation_pb2.GenerateResponse(
                request_id=request_id,
                prompt=_prompt_output(tokenizer, result, sampling_params.prompt_logprobs),
            )
        )
        args.prompt_sent = True

    newly_finished = []
    for output in result.outputs:
        token_ids = output.token_ids or []
        sent_token_count = min(args.sent_token_counts.get(output.index, 0), len(token_ids))
        safe_token_count = len(token_ids)
        if args.exclude_stop and not output.finish_reason:
            safe_token_count = max(0, safe_token_count - args.token_holdback)
        delta_token_ids = token_ids[sent_token_count:safe_token_count]
        all_text = output.text or ""
        sent_text_length = min(args.sent_text_lengths.get(output.index, 0), len(all_text))
        safe_text_length = len(all_text)
        if (
            args.exclude_stop
            and not output.finish_reason
            and (args.stop_texts or args.token_holdback)
        ):
            tracker = args.stop_prefix_trackers.setdefault(
                output.index, _StopPrefixTracker(args.stop_texts)
            )
            safe_text_length = tracker.safe_length(all_text)
            if args.token_holdback:
                safe_text_length = min(
                    safe_text_length, args.observed_text_lengths.get(output.index, 0)
                )
        delta_text = all_text[sent_text_length:safe_text_length]
        args.observed_text_lengths[output.index] = len(all_text)

        if delta_token_ids or delta_text:
            logprob_shortfall = args.logprob_shortfalls.get(output.index)
            if logprob_shortfall is None:
                logprob_shortfall = _logprob_shortfall(output)
                if output.logprobs:
                    args.logprob_shortfalls[output.index] = logprob_shortfall
            delta_logprobs = _delta_logprobs(output, sent_token_count, logprob_shortfall)
            token_infos = _token_infos(
                tokenizer, delta_token_ids, delta_logprobs, sampling_params.logprobs or 0
            )
            responses.append(
                generation_pb2.GenerateResponse(
                    request_id=request_id,
                    token=generation_pb2.TokenOutput(
                        output_index=output.index, tokens=token_infos, text=delta_text
                    ),
                )
            )
        args.sent_token_counts[output.index] = safe_token_count
        args.sent_text_lengths[output.index] = safe_text_length

        if args.is_context_only and output.index not in args.prefill_ready_sent:
            # `disaggregated_params` is seeded from the *request*, so it is never
            # None here; only a context phase that actually transmitted fills in
            # ctx_request_id. Emitting without it fabricates a handoff whose
            # session the generation worker cannot resolve.
            out_disagg = getattr(output, "disaggregated_params", None)
            if out_disagg is not None and out_disagg.ctx_request_id is not None:
                args.prefill_ready_sent.add(output.index)
                responses.append(
                    prefill_ready_response(
                        request_id,
                        out_disagg,
                        args.kv_transfer_backend,
                        usage=_usage(result),
                    )
                )

        if output.finish_reason and output.index not in args.finished_indices:
            args.finished_indices.add(output.index)
            newly_finished.append(output)

    for index, output in enumerate(newly_finished):
        # PrefillReady is the terminal signal for a context request that produced
        # a handoff. Without one -- cancelled or aborted before transmission --
        # the client must still get the real terminal event rather than a clean
        # stream with no ending. A context request that simply has nothing to
        # report ends as not-finished, which would map to a spurious UNSPECIFIED.
        if args.is_context_only and (
            output.index in args.prefill_ready_sent
            or output.finish_reason not in _TERMINAL_FINISH_REASONS
        ):
            continue
        is_final = (
            len(args.finished_indices) == sampling_params.n and index == len(newly_finished) - 1
        )
        response_kwargs: dict[str, Any] = {
            "request_id": request_id,
            "finished": _finish_event(output, sampling_params.end_id),
        }
        if is_final:
            response_kwargs["usage"] = _usage(result)
        responses.append(generation_pb2.GenerateResponse(**response_kwargs))

    return responses


class StallWatchdog:
    """Aborts a request whose client has stopped reading the response stream.

    Owns the two facts the old closure soup spread across four ``nonlocal``s:
    whether a response is currently awaiting delivery, and the single
    self-rescheduling timer that measures how long it has been waiting. Only
    consumer-pull latency is measured, so a slow engine -- nothing to send yet --
    is never misread as a stalled consumer.

    ``pending()`` must be called before *every* yield, terminal ones included: a
    yield that skips it leaves the watchdog blind, and the generator suspended
    there never resumes to run its own cleanup.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, timeout: float, on_stall: Callable) -> None:
        self._loop = loop
        self._timeout = timeout
        self._on_stall = on_stall
        self._pending_since: Optional[float] = None
        self._timer: Optional[asyncio.TimerHandle] = None
        self.stalled = False

    def start(self) -> None:
        self._timer = self._loop.call_later(self._timeout, self._check)

    def pending(self) -> None:
        self._pending_since = self._loop.time()

    def delivered(self) -> None:
        self._pending_since = None

    def close(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _check(self) -> None:
        # One self-rescheduling timer rather than a per-message alloc/cancel.
        if self._pending_since is None:
            self._timer = self._loop.call_later(self._timeout, self._check)
            return
        idle = self._loop.time() - self._pending_since
        if idle < self._timeout:
            self._timer = self._loop.call_later(self._timeout - idle, self._check)
            return
        self.stalled = True
        self._timer = None
        self._on_stall()


class ActiveRequest:
    """Owns one request's entry in the servicer's in-flight table.

    Registration has two removal sites -- the generator's ``finally`` and the
    stall watchdog, which must release the id even though the generator may
    never resume -- so ownership is checked in one place here rather than
    duplicated at each. Once an id is dropped a resubmission is admitted, and an
    unguarded removal would untrack the newer request's handle.
    """

    def __init__(self, table: dict, request_id: str, handle: Any) -> None:
        self._table = table
        self._request_id = request_id
        self._handle = handle

    def register(self) -> bool:
        if self._request_id in self._table:
            return False
        self._table[self._request_id] = self._handle
        return True

    def release(self) -> None:
        if self._table.get(self._request_id) is self._handle:
            del self._table[self._request_id]
