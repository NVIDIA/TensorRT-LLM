# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the trtllm-serve post-processing hook (TRTLLM-12622)."""

import pytest

from tensorrt_llm.executor.postprocessor_hook import (
    PostProcChunk,
    apply_post_processor_hook,
    emit,
    load_post_processor_hook,
    suppress,
    terminate,
)
from tensorrt_llm.executor.result import CompletionOutput


class _FakeResult:
    """Minimal stand-in for a GenerationResult at the detok chokepoint."""

    def __init__(
        self,
        outputs,
        req_id=1,
        done=False,
        aborted=False,
        has_abort=False,
        streaming=True,
        post_processor_hook=None,
    ):
        self.outputs = outputs
        self.id = req_id
        self._done = done
        self._aborted = aborted
        self._streaming = streaming
        # Per-instance hook ownership (TRTLLM-12622): the detok read site reads
        # this attribute rather than a process global.
        self._post_processor_hook = post_processor_hook
        self.abort_called = 0
        if has_abort:
            self.abort = self._abort

    def _abort(self):
        self.abort_called += 1
        self._aborted = True


def _make_output(text, last_text_len=0, index=0, token_ids=None, logprobs=None):
    out = CompletionOutput(index=index, text=text)
    out.token_ids = token_ids if token_ids is not None else [1, 2, 3]
    out.logprobs = logprobs if logprobs is not None else [-0.1, -0.2, -0.3]
    out._last_text_len = last_text_len
    return out


def test_rewrite_streaming_diff_replaces_only_the_diff():
    out = _make_output("hello world", last_text_len=len("hello"))
    result = _FakeResult([out])

    def hook(chunk: PostProcChunk):
        assert chunk.text_diff == " world"
        assert chunk.text == "hello world"
        return emit(chunk.text_diff.upper())

    apply_post_processor_hook(hook, result, streaming=True)

    # The already-emitted prefix is preserved; only the diff is rewritten.
    assert out.text == "hello WORLD"
    assert out.text_diff == " WORLD"


def test_suppress_withholds_the_diff_keeping_prefix():
    out = _make_output("hello world", last_text_len=len("hello"))
    result = _FakeResult([out])

    apply_post_processor_hook(lambda c: suppress(), result, streaming=True)

    assert out.text == "hello"
    assert out.text_diff == ""


def test_terminate_marks_aborted_and_done_with_reason():
    out = _make_output("safe bad", last_text_len=len("safe "))
    result = _FakeResult([out])

    apply_post_processor_hook(lambda c: terminate("policy_violation"), result, streaming=True)

    assert result._aborted is True
    assert result._done is True
    assert out.finish_reason == "stop"
    assert out.stop_reason == "policy_violation"
    # The violating diff is withheld.
    assert out.text == "safe "


def test_suppress_blanks_token_and_logprob_diffs():
    # mid-stream chunk with one new token/logprob beyond what was emitted.
    out = _make_output("hello world", last_text_len=len("hello"))
    out._last_token_ids_len = 2
    out._last_logprobs_len = 2
    assert out.token_ids_diff == [3]
    assert out.logprobs_diff == [-0.3]
    result = _FakeResult([out])

    apply_post_processor_hook(lambda c: suppress(), result, streaming=True)

    assert out.text == "hello"
    # The raw token/logprob channel is withheld too, not just the text.
    assert out.token_ids_diff == []
    assert out.logprobs_diff == []


def test_terminate_calls_abort_when_available_and_blanks_token_channel():
    out = _make_output("safe bad", last_text_len=len("safe "))
    out._last_token_ids_len = 1
    result = _FakeResult([out], has_abort=True)

    apply_post_processor_hook(lambda c: terminate("policy"), result, streaming=True)

    assert result._aborted is True
    assert result._done is True
    assert result.abort_called == 1
    assert out.token_ids_diff == []
    assert out.finish_reason == "stop"
    assert out.stop_reason == "policy"


def test_terminate_without_abort_attr_does_not_crash():
    out = _make_output("safe bad", last_text_len=len("safe "))
    result = _FakeResult([out], has_abort=False)
    apply_post_processor_hook(lambda c: terminate("policy"), result, streaming=True)
    assert result._aborted is True
    assert result._done is True


def test_hook_exception_fails_open_passthrough():
    out = _make_output("hello world", last_text_len=len("hello"))
    result = _FakeResult([out])

    def boom(chunk):
        raise RuntimeError("hook bug")

    # Must not propagate; the chunk passes through unchanged (fail-open).
    apply_post_processor_hook(boom, result, streaming=True)

    assert out.text == "hello world"
    assert out.text_diff == " world"


def test_non_streaming_rewrites_full_text():
    # Non-stream single response: _last_text_len == 0, so diff == full text.
    out = _make_output("the full answer", last_text_len=0)
    result = _FakeResult([out], done=True)

    def hook(chunk: PostProcChunk):
        assert chunk.text_diff == chunk.text == "the full answer"
        return emit("REDACTED")

    apply_post_processor_hook(hook, result, streaming=False)

    assert out.text == "REDACTED"


def test_passthrough_emit_is_idempotent():
    out = _make_output("hello world", last_text_len=len("hello"))
    result = _FakeResult([out])

    apply_post_processor_hook(lambda c: emit(c.text_diff), result, streaming=True)

    assert out.text == "hello world"
    assert out.text_diff == " world"


def test_per_request_state_is_keyed_by_request_id():
    """The hook owns its per-request state; trtllm only passes request_id."""

    class Counter:
        def __init__(self):
            self.state = {}

        def __call__(self, chunk: PostProcChunk):
            n = self.state.get(chunk.request_id, 0) + 1
            self.state[chunk.request_id] = n
            if chunk.is_final:
                self.state.pop(chunk.request_id, None)
            return emit(f"{chunk.text_diff}#{n}")

    hook = Counter()
    r1a = _FakeResult([_make_output("a", 0)], req_id=1)
    r2 = _FakeResult([_make_output("b", 0)], req_id=2)
    r1b = _FakeResult([_make_output("ac", 1)], req_id=1, done=True)

    apply_post_processor_hook(hook, r1a, streaming=True)
    apply_post_processor_hook(hook, r2, streaming=True)
    apply_post_processor_hook(hook, r1b, streaming=True)

    # Request 1 counts 1 then 2, independently of request 2 (which counts 1).
    assert r1a.outputs[0].text == "a#1"
    assert r2.outputs[0].text == "b#1"
    assert r1b.outputs[0].text == "ac#2"
    # State released on the final chunk.
    assert 1 not in hook.state


def test_unknown_verdict_action_raises():
    from tensorrt_llm.executor.postprocessor_hook import PostProcVerdict

    out = _make_output("x", 0)
    result = _FakeResult([out])
    with pytest.raises(ValueError, match="Unknown post-processor verdict"):
        apply_post_processor_hook(lambda c: PostProcVerdict(action="bogus"), result, streaming=True)


def test_loader_resolves_import_path():
    # Any importable, no-arg-constructible class works as a smoke test.
    hook = load_post_processor_hook("collections.OrderedDict")
    assert hook is not None


def test_loader_raises_on_bad_path():
    with pytest.raises(ValueError, match="Failed to load post-processor hook"):
        load_post_processor_hook("no.such.module.Nope")


def test_loader_builds_independent_instances():
    """Each owner builds its own instance (no shared process-global cache).

    This is the core of per-instance ownership (TRTLLM-12622): two owners
    loading the same import path get distinct instances, so their per-request
    state never collides.
    """
    a = load_post_processor_hook("collections.OrderedDict")
    b = load_post_processor_hook("collections.OrderedDict")
    assert a is not b


def test_apply_method_reads_hook_from_instance_attribute():
    """The detok read site applies the hook owned by the result instance."""
    from tensorrt_llm.executor.result import DetokenizedGenerationResultBase

    out = _make_output("hello world", last_text_len=len("hello"))
    result = _FakeResult([out], post_processor_hook=lambda c: emit(c.text_diff.upper()))

    # Call the real read site against the per-instance attribute.
    DetokenizedGenerationResultBase._apply_post_processor_hook(result)

    assert out.text == "hello WORLD"


def test_apply_method_is_noop_when_instance_has_no_hook():
    """With no hook on the instance, the chunk passes through untouched."""
    from tensorrt_llm.executor.result import DetokenizedGenerationResultBase

    out = _make_output("hello world", last_text_len=len("hello"))
    result = _FakeResult([out], post_processor_hook=None)

    DetokenizedGenerationResultBase._apply_post_processor_hook(result)

    assert out.text == "hello world"


def test_harmony_model_rejects_post_processor():
    """A harmony/gpt-oss model + post_processor must fail fast (TRTLLM-12622).

    The harmony output path is rebuilt from raw token ids and would bypass the
    text-based hook, so the server refuses the combination at startup.
    """
    from tensorrt_llm.serve.openai_server import OpenAIServer

    guard = OpenAIServer._ensure_post_processor_supported
    with pytest.raises(ValueError, match="not supported with harmony"):
        guard(use_harmony=True, post_processor="my_pkg.guardrail.Hook")
    # Every other combination is allowed.
    guard(use_harmony=False, post_processor="my_pkg.guardrail.Hook")
    guard(use_harmony=True, post_processor=None)
    guard(use_harmony=False, post_processor=None)
