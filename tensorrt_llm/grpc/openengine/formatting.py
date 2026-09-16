# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn engine results into OpenEngine response messages."""

from collections.abc import Sequence
from typing import Any

from openengine.v1 import error_pb2, generation_pb2

from tensorrt_llm.sampling_params import SamplingParams

# Floor for logprob values so a -inf (masked token) is JSON/proto-safe, matching
# the HTTP /v1/completions clamp in create_logprobs.
_MIN_LOGPROB = -9999.0


def clamp_logprob(value: Any) -> float:
    return max(float(value), _MIN_LOGPROB)


# Ceiling on prompt_length * prompt_candidates. Building a PromptOutput is one
# synchronous, non-preemptible slice of work sized by both, and both are
# client-controlled: a 128k prompt at top_n=100 is minutes of detokenization and
# protobuf construction on the shared event loop, stalling every other in-flight
# stream, before producing a message too large to send. Rejecting up front costs
# nothing; the limit is generous enough that no realistic request reaches it.
_MAX_PROMPT_LOGPROB_ENTRIES = 1_000_000


# Finish reasons that carry real terminal information. Anything else (notably
# "not_finished") would map to FINISH_REASON_UNSPECIFIED.
_TERMINAL_FINISH_REASONS = frozenset({"stop", "length", "cancelled", "timeout"})


def _token_strings(tokenizer: Any, token_ids: Sequence[int]) -> list[str]:
    if not token_ids or tokenizer is None:
        return [""] * len(token_ids)
    # Decode each id to human-readable text, matching the HTTP /v1/completions
    # logprobs token representation (`tokenizer.decode(token_id)`), rather than
    # raw tokenizer pieces (e.g. "▁Paris", "<0x0A>") from convert_ids_to_tokens.
    return [tokenizer.decode([token_id]) or "" for token_id in token_ids]


def _token_infos(
    tokenizer: Any,
    token_ids: Sequence[int],
    logprobs: Sequence[Any] = (),
    candidate_limit: int = 0,
) -> list[generation_pb2.TokenInfo]:
    # Only decode per-token strings when logprobs are attached: the `token` /
    # candidate strings exist for logprob display (mirroring the HTTP path, which
    # only surfaces token strings inside logprobs), so the common token-ids/text
    # streaming path skips per-token detokenization entirely.
    if not logprobs:
        # The common streaming path: no logprobs means no token or candidate
        # strings, so skip detokenization and the candidate bookkeeping rather
        # than building empty scaffolding for every token of every request.
        return [generation_pb2.TokenInfo(token_id=token_id) for token_id in token_ids]

    token_strings = _token_strings(tokenizer, token_ids)
    candidate_items: list[list[tuple[int, Any]]] = []
    candidate_ids: dict[int, None] = {}
    for index in range(len(token_ids)):
        token_logprobs = logprobs[index] if index < len(logprobs) else None
        candidates = []
        if isinstance(token_logprobs, dict) and candidate_limit > 0:
            candidates = sorted(
                (
                    (candidate_id, candidate)
                    for candidate_id, candidate in token_logprobs.items()
                    if candidate.rank is not None and candidate.rank <= candidate_limit
                ),
                key=lambda item: item[1].rank,
            )
            candidate_ids.update((candidate_id, None) for candidate_id, _ in candidates)
        candidate_items.append(candidates)

    candidate_strings = dict(
        zip(candidate_ids, _token_strings(tokenizer, list(candidate_ids)), strict=True)
    )
    token_infos = []
    for index, (token_id, token) in enumerate(zip(token_ids, token_strings, strict=True)):
        token_logprobs = logprobs[index] if index < len(logprobs) else None
        kwargs: dict[str, Any] = {"token_id": token_id, "token": token}
        if isinstance(token_logprobs, (int, float)):
            kwargs["logprob"] = clamp_logprob(token_logprobs)
        elif isinstance(token_logprobs, dict):
            sampled = token_logprobs.get(token_id)
            if sampled is not None:
                kwargs["logprob"] = clamp_logprob(sampled.logprob)
                if sampled.rank is not None:
                    kwargs["rank"] = sampled.rank
            candidates = []
            for candidate_id, candidate in candidate_items[index]:
                logprob = generation_pb2.LogProb(
                    token_id=candidate_id,
                    logprob=clamp_logprob(candidate.logprob),
                    token=candidate_strings[candidate_id],
                )
                if candidate.rank is not None:
                    logprob.rank = candidate.rank
                candidates.append(logprob)
            kwargs["candidates"] = candidates
        token_infos.append(generation_pb2.TokenInfo(**kwargs))
    return token_infos


def _stop_texts(sampling_params: SamplingParams) -> list[str]:
    stop = sampling_params.stop
    if isinstance(stop, str):
        return [stop]
    return list(stop or [])


def _token_holdback(sampling_params: SamplingParams) -> int:
    """Number of trailing tokens to withhold so a stop token id is not streamed.

    NOTE: this depends on the private ``SamplingParams._stop_word_ids``, which the
    LLM engine populates with the tokenized stop words. It is read after
    ``generate_async`` returns, when it is expected to be set. If the engine ever
    stops populating it (or populates it later), this degrades to 0 (no holdback)
    rather than failing — ``test_token_holdback_*`` guards the current behavior.
    """
    if sampling_params.include_stop_str_in_output:
        return 0
    stop_word_ids = getattr(sampling_params, "_stop_word_ids", None) or []
    if not stop_word_ids:
        return 0
    return max(1, max(len(token_ids) for token_ids in stop_word_ids) - 1)


def _prefix_table(pattern: str) -> list[int]:
    table = [0] * len(pattern)
    matched = 0
    for index in range(1, len(pattern)):
        while matched and pattern[index] != pattern[matched]:
            matched = table[matched - 1]
        if pattern[index] == pattern[matched]:
            matched += 1
            table[index] = matched
    return table


class _StopPrefixTracker:
    def __init__(self, stop_texts: Sequence[str]) -> None:
        self._patterns = [pattern for pattern in stop_texts if pattern]
        self._tables = [_prefix_table(pattern) for pattern in self._patterns]
        self._states = [0] * len(self._patterns)
        self._observed_length = 0

    def safe_length(self, text: str) -> int:
        if len(text) < self._observed_length:
            self._states = [0] * len(self._patterns)
            self._observed_length = 0

        delta = text[self._observed_length :]
        for pattern_index, (pattern, table) in enumerate(zip(self._patterns, self._tables)):
            matched = self._states[pattern_index]
            for character in delta:
                while matched and (matched == len(pattern) or character != pattern[matched]):
                    matched = table[matched - 1]
                if matched < len(pattern) and character == pattern[matched]:
                    matched += 1
            self._states[pattern_index] = matched
        self._observed_length = len(text)
        return len(text) - max(self._states, default=0)


def _prompt_output(
    tokenizer: Any, result: Any, candidate_limit: int
) -> generation_pb2.PromptOutput:
    token_ids = list(result.prompt_token_ids)
    prompt_logprobs = result.outputs[0].prompt_logprobs if result.outputs else None
    prompt_logprobs = prompt_logprobs or []
    aligned_logprobs = [None, *prompt_logprobs[: max(0, len(token_ids) - 1)]]
    return generation_pb2.PromptOutput(
        tokens=_token_infos(tokenizer, token_ids, aligned_logprobs, candidate_limit)
    )


def _logprob_shortfall(output: Any) -> int:
    """How many leading tokens carry no logprob.

    On a generation_only request the engine tolerates being one logprob short --
    the context worker did not transfer the first token's -- and only logs a
    warning. Slicing positionally against that would attribute every logprob,
    rank and candidate set to the wrong token.
    """
    logprobs = output.logprobs or []
    if not logprobs:
        return 0
    return max(0, len(output.token_ids or []) - len(logprobs))


def _delta_logprobs(output: Any, sent_token_count: int, shortfall: int) -> Sequence[Any]:
    """Logprobs for ``token_ids[sent_token_count:]``, head-padded if short.

    Indexes into the engine's cumulative list rather than rebuilding a padded
    copy of it. ``output.logprobs`` grows for the whole stream, so materializing
    ``[None] * shortfall + list(logprobs)`` on every streaming step made a
    logprobs-enabled generation_only request O(n^2) in output length -- and the
    shortfall branch is not an edge case there, it holds for the entire stream.
    """
    logprobs = output.logprobs or []
    if not shortfall:
        return logprobs[sent_token_count:]
    if sent_token_count >= shortfall:
        return logprobs[sent_token_count - shortfall :]
    return [None] * (shortfall - sent_token_count) + list(logprobs)


def _finish_event(output: Any, end_id: int | None) -> generation_pb2.GenerationFinished:
    reason_map = {
        "stop": generation_pb2.FINISH_REASON_STOP,
        "length": generation_pb2.FINISH_REASON_LENGTH,
        "cancelled": generation_pb2.FINISH_REASON_CANCELLED,
        "timeout": generation_pb2.FINISH_REASON_CANCELLED,
    }
    kwargs: dict[str, Any] = {
        "output_index": output.index,
        "reason": reason_map.get(output.finish_reason, generation_pb2.FINISH_REASON_UNSPECIFIED),
    }
    if output.finish_reason == "timeout":
        kwargs["message"] = "generation timed out"
    if output.finish_reason == "stop":
        if isinstance(output.stop_reason, int):
            kwargs["stop_match"] = generation_pb2.StopMatch(stop_token_id=output.stop_reason)
        elif isinstance(output.stop_reason, str):
            kwargs["stop_match"] = generation_pb2.StopMatch(stop_text=output.stop_reason)
        elif end_id is not None:
            kwargs["stop_match"] = generation_pb2.StopMatch(eos_token_id=end_id)
    return generation_pb2.GenerationFinished(**kwargs)


def _usage(result: Any) -> generation_pb2.Usage:
    prompt_tokens = len(result.prompt_token_ids or ())
    completion_tokens = sum(len(output.token_ids or []) for output in (result.outputs or ()))
    return generation_pb2.Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cached_prompt_tokens=result.cached_tokens,
    )


def _engine_error_response(
    request_id: str,
    message: str,
    result: Any,
    *,
    code: int = error_pb2.ERROR_CODE_INTERNAL,
    retryable: bool = False,
) -> generation_pb2.GenerateResponse:
    return generation_pb2.GenerateResponse(
        request_id=request_id,
        error=error_pb2.EngineError(code=code, message=message, retryable=retryable),
        usage=_usage(result),
    )


__all__ = [
    "clamp_logprob",
]
