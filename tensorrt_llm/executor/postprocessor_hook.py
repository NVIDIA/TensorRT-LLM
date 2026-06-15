# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""User-pluggable post-processing hook for ``trtllm-serve`` (TRTLLM-12622).

This provides a native, per-request, stateful post-processing seam equivalent
to a Triton python-backend post-processor. A user supplies a picklable,
importable callable class via the ``--post_processor`` import path; trtllm
builds one instance per owner (the ``LLM`` for the in-proxy detok path, and
each post-processing worker process when enabled) and invokes it once per
output, per streaming chunk (plus a final call), *after* detokenization and
*before* the per-endpoint response formatter. Ownership is per-instance, not a
process global: the instance is threaded onto each result alongside the
tokenizer, so independent ``LLM`` instances in one process stay isolated.

The hook owns its own per-request state (keyed by ``chunk.request_id``) exactly
like Triton's model-managed ``self.sequences = {}`` pattern; trtllm passes only
the request id, the per-chunk payload, lifecycle flags, and the cancel signal.

This module is intentionally dependency-light (stdlib only) so it can be loaded
in the post-processing worker process and reasoned about in isolation.
"""

import dataclasses
import importlib
import logging
from typing import List, Optional, Protocol, runtime_checkable

__all__ = [
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
    ``module.path.ClassName``, import the module, fetch the class, instantiate
    it with no arguments. The class must be importable and picklable so it can
    cross the post-processing worker process boundary.

    Each owner (the ``LLM`` and each postproc worker) calls this once and holds
    the returned instance for the lifetime of the owner; the instance is never
    pickled across a process boundary (only ``import_path`` is), and per-request
    state lives inside it.

    Args:
        import_path: Dotted path to the hook class, e.g.
            ``'my_pkg.guardrail.MyPostProcessor'``.

    Returns:
        An instance of the hook class.

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


@dataclasses.dataclass
class PostProcVerdict:
    """The hook's decision for one chunk.

    Use the :func:`emit`, :func:`suppress`, and :func:`terminate` helpers rather
    than constructing this directly.
    """

    action: str  # "emit" | "suppress" | "terminate"
    text: str = ""
    reason: Optional[str] = None


def emit(text: str) -> PostProcVerdict:
    """Emit ``text`` for this chunk (use to rewrite/redact, or pass through)."""
    return PostProcVerdict(action="emit", text=text)


def suppress() -> PostProcVerdict:
    """Withhold this chunk entirely (no client-visible output)."""
    return PostProcVerdict(action="suppress")


def terminate(reason: str) -> PostProcVerdict:
    """Stop the stream for this request. ``reason`` is surfaced as stop_reason."""
    return PostProcVerdict(action="terminate", reason=reason)


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

    ``suppress``/``terminate`` blank the text channel; the raw token-id and
    logprob channels must be withheld too, or a suppressed/terminated output
    leaks via them — ``token_ids`` on ``/v1/completions`` with
    ``detokenize=False``, and ``logprobs`` on both chat and completions.

    The two response shapes withhold differently, matching what each formatter
    emits (verified by the channel audit):

    * **streaming** emits per-chunk *diffs* (``token_ids_diff`` /
      ``logprobs_diff``); advancing the diff watermark empties this chunk.
    * **non-streaming** emits the *full* ``token_ids`` / ``logprobs``; these are
      truncated back to the already-*emitted* prefix (the content the hook chose
      to emit on prior chunks), mirroring exactly how the text channel is blanked
      to ``output.text[:_last_text_len]``. Outputs accumulate via ``list.extend``
      in the hook's single-output scope, so the truncation stays consistent
      across chunks: the result holds exactly the content a streaming client
      would have received before this ``suppress``/``terminate`` — withholding
      this chunk, not retroactively the prior ones.
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

    Hook exceptions are isolated per request: they are logged and the chunk is
    passed through unchanged (fail-open), so a buggy hook cannot wedge the
    worker or crash the serving loop. This is consistent across the in-proxy and
    postproc-worker paths.
    """
    # ``is_final`` is request-level (``result._done``), not per-output; under the
    # locked 1:1 single-output scope (TRTLLM-12622) this is the exact cleanup
    # signal for hooks that release per-request state on ``is_final``.
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
                "Post-processor hook raised for request %s; passing the chunk through unchanged.",
                result.id,
            )
            continue
        prefix = output.text[: output._last_text_len]
        if verdict.action == "emit":
            output.text = prefix + verdict.text
        elif verdict.action == "suppress":
            output.text = prefix
            _withhold_token_channel(output, streaming)
        elif verdict.action == "terminate":
            output.text = prefix + verdict.text
            _withhold_token_channel(output, streaming)
            output.finish_reason = "stop"
            output.stop_reason = verdict.reason
            result._aborted = True
            result._done = True
            # Cancel the engine request as well. On the in-proxy path this stops
            # wasted generation; on the worker path the record's abort() only
            # sets the flag and the engine is cancelled by the proxy via
            # should_abort. The getattr guard is defensive (real results always
            # define abort()).
            abort = getattr(result, "abort", None)
            if callable(abort):
                try:
                    abort()
                except Exception:
                    logger.exception(
                        "Failed to abort request %s after terminate verdict.", result.id
                    )
        else:
            raise ValueError(f"Unknown post-processor verdict action: {verdict.action!r}")
