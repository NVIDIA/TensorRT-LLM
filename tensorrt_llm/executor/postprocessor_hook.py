# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""User-pluggable post-processing hook for ``trtllm-serve`` (TRTLLM-12622).

A user supplies a picklable, importable callable class via the
``--post_processor`` import path. One instance is built per owner (the ``LLM``
for the in-proxy detok path, and each post-processing worker process when
enabled) and invoked once per output, per streaming chunk (plus a final call),
*after* detokenization and *before* the per-endpoint response formatter. The
hook owns its per-request state, keyed by ``chunk.request_id``.

Stdlib-only so it can be loaded in the post-processing worker process.
"""

import dataclasses
import enum
import importlib
import logging
from typing import List, Optional, Protocol, runtime_checkable

__all__ = [
    "PostProcAction",
    "PostProcChunk",
    "PostProcVerdict",
    "PostProcessorHook",
    "emit",
    "suppress",
    "terminate",
    "apply_post_processor_hook",
    "load_post_processor_hook",
]

logger = logging.getLogger(__name__)


def load_post_processor_hook(import_path: str) -> "PostProcessorHook":
    """Build a post-processor hook instance from a dotted import path.

    Mirrors ``tensorrt_llm.tokenizer.load_custom_tokenizer``: resolve
    ``module.path.ClassName``, import the module, instantiate it with no
    arguments. Only ``import_path`` crosses a process boundary (never the
    instance), so the class must be importable and picklable.

    Raises:
        ValueError: If the path cannot be resolved, imported, or instantiated.
    """
    try:
        module_path, class_name = import_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        hook_class = getattr(module, class_name)
        return hook_class()
    except (ValueError, ImportError, AttributeError, TypeError) as e:
        raise ValueError(
            f"Failed to load post-processor hook '{import_path}': {e}. "
            "Expected format: 'module.path.ClassName' resolving to a "
            "no-arg-constructible callable class."
        ) from e


@dataclasses.dataclass
class PostProcChunk:
    """The payload handed to the post-processing hook for one output chunk.

    Attributes:
        request_id: Stable identifier for the request; the same value is passed
            for every chunk of a given response, so the hook can key its own
            per-request state on it.
        output_index: Index of the output/beam within the request.
        text_diff: Newly detokenized text produced by this chunk (streaming).
            For non-streaming requests this equals ``text``.
        text: Full accumulated detokenized text so far for this output.
        token_ids_diff: Newly generated token ids for this chunk.
        is_final: True on the terminating call for this output.
        aborted: True if the request has been marked aborted in this process
            (e.g. a prior ``terminate`` verdict, or an abort observed by the
            detok process). Output-side observation only; do not rely on it to
            detect every upstream client cancellation.
        streaming: True for streaming requests.
    """

    request_id: int
    output_index: int
    text_diff: str
    text: str
    token_ids_diff: List[int]
    is_final: bool
    aborted: bool
    streaming: bool


class PostProcAction(str, enum.Enum):
    """The kind of decision a hook returns for one chunk."""

    EMIT = "emit"
    SUPPRESS = "suppress"
    TERMINATE = "terminate"


@dataclasses.dataclass
class PostProcVerdict:
    """The hook's decision for one chunk.

    Use the :func:`emit`, :func:`suppress`, and :func:`terminate` helpers rather
    than constructing this directly.
    """

    action: PostProcAction
    text: str = ""
    reason: Optional[str] = None

    def __post_init__(self):
        # Coerce/validate so a hook can never smuggle an unknown action.
        self.action = PostProcAction(self.action)


def emit(text: str) -> PostProcVerdict:
    """Emit ``text`` for this chunk (use to rewrite/redact, or pass through)."""
    return PostProcVerdict(action=PostProcAction.EMIT, text=text)


def suppress() -> PostProcVerdict:
    """Withhold this chunk entirely (no client-visible output)."""
    return PostProcVerdict(action=PostProcAction.SUPPRESS)


def terminate(reason: str) -> PostProcVerdict:
    """Stop the stream for this request. ``reason`` is surfaced as stop_reason."""
    return PostProcVerdict(action=PostProcAction.TERMINATE, reason=reason)


@runtime_checkable
class PostProcessorHook(Protocol):
    """The interface a user post-processor implements.

    The instance is built once per owner (its ``__init__`` is the one-time
    setup) and called once per output, per chunk. It owns any per-request state
    and is responsible for releasing it on ``chunk.is_final``.
    """

    def __call__(self, chunk: PostProcChunk) -> PostProcVerdict: ...


def _withhold_token_channel(output, streaming: bool) -> None:
    """Withhold the raw token-id / logprob channels alongside the blanked text.

    Otherwise a suppressed/terminated output leaks via them (``token_ids`` on
    ``/v1/completions`` with ``detokenize=False``, ``logprobs`` on both
    endpoints). Streaming emits per-chunk diffs, so advancing the diff watermark
    empties this chunk; non-streaming emits the full lists, so truncate them back
    to the already-emitted prefix, mirroring how the text is blanked.
    """
    if streaming:
        output._last_token_ids_len = len(output.token_ids)
        if getattr(output, "logprobs", None) is not None:
            output._last_logprobs_len = len(output.logprobs)
    else:
        output.token_ids = output.token_ids[: output._last_token_ids_len]
        if getattr(output, "logprobs", None) is not None:
            output.logprobs = output.logprobs[: output._last_logprobs_len]


def apply_post_processor_hook(hook: PostProcessorHook, result, streaming: bool) -> None:
    """Run ``hook`` over ``result.outputs`` in place at the detok chokepoint.

    Applies each verdict by rewriting the chunk's text diff on the output
    (preserving the already-emitted prefix), suppressing it, or terminating the
    stream via the existing abort machinery.

    A hook exception fails the request closed (re-raised), never serving the
    un-vetted chunk: the in-proxy path surfaces it to the serving handler, the
    worker path converts it to an ``ErrorResponse``. Both keep the server and
    other requests alive (mirrors Triton's per-request fail-closed model).
    """
    # ``is_final`` is request-level (``result._done``): for n>1 / beam it fires
    # once when the whole request finishes. emit/suppress act per output, but a
    # terminate cancels the whole request (all outputs) because the engine
    # request is the unit of cancellation. Hooks needing per-sequence state
    # should key on (request_id, output_index).
    is_final = result._done
    for output in result.outputs:
        chunk = PostProcChunk(
            request_id=result.id,
            output_index=output.index,
            text_diff=output.text_diff,
            text=output.text,
            token_ids_diff=list(output.token_ids_diff),
            is_final=is_final,
            aborted=result._aborted,
            streaming=streaming,
        )
        try:
            verdict = hook(chunk)
        except Exception:
            logger.exception(
                "Post-processor hook failed for request %s; failing the request closed.",
                result.id,
            )
            raise
        prefix = output.text[: output._last_text_len]
        if verdict.action is PostProcAction.EMIT:
            output.text = prefix + verdict.text
        elif verdict.action is PostProcAction.SUPPRESS:
            output.text = prefix
            _withhold_token_channel(output, streaming)
        elif verdict.action is PostProcAction.TERMINATE:
            output.text = prefix + verdict.text
            _withhold_token_channel(output, streaming)
            output.finish_reason = "stop"
            output.stop_reason = verdict.reason
            result._aborted = True
            result._done = True
            # Cancel the engine request to stop wasted generation (on the worker
            # path the proxy does the actual cancel via should_abort). getattr
            # guard is defensive; real results always define abort().
            abort = getattr(result, "abort", None)
            if callable(abort):
                try:
                    abort()
                except Exception:
                    logger.exception(
                        "Failed to abort request %s after terminate verdict.", result.id
                    )
        else:
            # Unreachable for hook-returned verdicts (validated in
            # ``__post_init__``); guards an unhandled future enum member.
            raise ValueError(f"Unhandled post-processor action: {verdict.action!r}")
