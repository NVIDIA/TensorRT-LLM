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
"""Unit tests for the raw model output dump (``RawOutputDump``).

The hook is driven through the real ``apply_post_processor_hook`` over real
``CompletionOutput`` objects, so the served text is checked as the applier
rebuilds it from the hook's verdicts.
"""

import asyncio
import glob
import json
import os
import queue
from pathlib import Path
from types import SimpleNamespace

import pytest

import tensorrt_llm.executor.raw_output_hook as raw_output_hook
from tensorrt_llm.bindings.executor import FinishReason
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.executor.executor import GenerationExecutor
from tensorrt_llm.executor.postproc_worker import PostprocParams, PostprocWorker
from tensorrt_llm.executor.postprocessor_hook import (
    apply_post_processor_hook,
    end_post_processor_hook,
    load_post_processor_hook,
)
from tensorrt_llm.executor.raw_output_hook import RAW_OUTPUT_DIR_ENV, RawOutputDump
from tensorrt_llm.executor.result import CompletionOutput, DetokenizedGenerationResultBase
from tensorrt_llm.executor.utils import ErrorResponse, RequestError
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only

HOOK_PATH = "tensorrt_llm.executor.raw_output_hook.RawOutputDump"


class FakeResult:
    """The parts of a generation result that ``apply_post_processor_hook`` reads."""

    def __init__(self, request_id=7, outputs=None, disaggregated_params=None):
        self.id = request_id
        self.outputs = outputs if outputs is not None else [CompletionOutput(index=0)]
        self.disaggregated_params = disaggregated_params
        self._done = False
        self._aborted = False


def advance(output, text, token_ids=()):
    """Add one chunk to ``output``, moving its diff watermarks as the executor does."""
    output._last_token_ids_len = len(output.token_ids)
    output.token_ids.extend(token_ids)
    output._last_text_len = len(output.text)
    output.text += text


def drive(hook, result, deltas, streaming=True):
    """Feed ``deltas`` through the hook one chunk at a time; the last chunk is final.

    Returns the text each output ends up with, as the applier rebuilt it.
    """
    for i, delta in enumerate(deltas):
        for output in result.outputs:
            advance(output, delta)
        result._done = i == len(deltas) - 1
        apply_post_processor_hook(hook, result, streaming=streaming)
    return [output.text for output in result.outputs]


def record_files(directory):
    return sorted(glob.glob(os.path.join(str(directory), "*", "raw-*.jsonl")))


def records_in(directory):
    records = []
    for path in record_files(directory):
        with open(path, encoding="utf-8") as handle:
            records += [json.loads(line) for line in handle if line.strip()]
    return records


@pytest.fixture
def dump_dir(tmp_path, monkeypatch):
    directory = tmp_path / "raw"
    monkeypatch.setenv(RAW_OUTPUT_DIR_ENV, str(directory))
    return directory


class LetterTokenizer:
    """Decodes token id ``i`` to the ``i``-th lowercase letter.

    It has no ``decode_incrementally``, so a result re-decodes all of an
    output's token ids on every response.
    """

    def decode(self, token_ids, **kwargs):
        return "".join(chr(ord("a") + token_id % 26) for token_id in token_ids)


def engine_response(
    client_id, token_ids, sequence_index=0, cum_logprob=0.0, finished=False, is_final=False
):
    """A response from the engine for one sequence of a request, as a result reads it."""
    result = SimpleNamespace(
        is_final=is_final,
        sequence_index=sequence_index,
        output_token_ids=[list(token_ids)],
        cum_log_probs=[cum_logprob],
        finish_reasons=[FinishReason.LENGTH if finished else FinishReason.NOT_FINISHED],
        context_phase_params=None,
        decoding_iter=1,
        avg_decoded_tokens_per_iter=None,
        request_perf_metrics=None,
        log_probs=None,
        generation_logits=None,
        context_logits=None,
        additional_context_outputs=None,
        additional_generation_outputs=None,
        time_breakdown_metrics=None,
        per_pos_drafted=None,
        per_pos_accepted=None,
        spec_dec_totals=None,
    )
    return SimpleNamespace(
        client_id=client_id, request_id=client_id, result=result, has_error=lambda: False
    )


def engine_error(client_id, message):
    """An error the engine reports for a request."""
    return SimpleNamespace(
        client_id=client_id, request_id=client_id, error_msg=message, has_error=lambda: True
    )


class ExecutorErrors:
    """Handles a result's errors with the executor's own background error handler."""

    _handle_background_error = GenerationExecutor._handle_background_error

    def __init__(self):
        # Errors from the executor's background threads; none in these tests.
        self._error_queue = queue.Queue()


def test_the_served_text_is_unchanged(dump_dir):
    hook = RawOutputDump()
    deltas = ["<think>", "weighing ", "it</think>", "the answer"]

    (text,) = drive(hook, FakeResult(), deltas)

    assert text == "".join(deltas)
    (record,) = records_in(dump_dir)
    assert record["text"] == text


def test_without_the_directory_nothing_is_recorded(tmp_path, monkeypatch):
    monkeypatch.delenv(RAW_OUTPUT_DIR_ENV, raising=False)
    warnings = []
    monkeypatch.setattr(raw_output_hook.logger, "warning", lambda *args: warnings.append(args))

    hook = RawOutputDump()
    (text,) = drive(hook, FakeResult(), ["alpha", "beta"])

    assert text == "alphabeta"
    assert not list(tmp_path.rglob("*.jsonl"))
    # Disabled, not failing: a hook that tried to record and failed would count drops.
    assert hook._dropped == 0
    assert hook._pending == {}
    (warning,) = warnings
    assert RAW_OUTPUT_DIR_ENV in warning


def test_streaming_frames_keep_the_chunk_boundaries(dump_dir):
    """A marker split across chunks stays split in ``frames``; ``text`` holds the whole output."""
    hook = RawOutputDump()
    deltas = ['<tool_call>{"na', 'me": "f"}', "</tool_call>"]

    drive(hook, FakeResult(request_id=99), deltas)

    (record,) = records_in(dump_dir)
    assert record["frames"] == deltas
    assert record["text"] == "".join(deltas)
    assert record["frames_consistent"] is True
    assert record["streaming"] is True
    assert record["request_id"] == "99"
    assert record["output_index"] == 0


def test_chunks_that_add_no_text_are_not_frames(dump_dir):
    hook = RawOutputDump()

    drive(hook, FakeResult(), ["one", "", "two", ""])

    (record,) = records_in(dump_dir)
    assert record["frames"] == ["one", "two"]


def test_each_output_of_a_request_keeps_its_own_frames(dump_dir):
    """With n > 1 the outputs share a request id; over two chunks their frames must not mix."""
    hook = RawOutputDump()
    outputs = [CompletionOutput(index=0), CompletionOutput(index=1)]
    result = FakeResult(request_id=5, outputs=outputs)

    for chunk_index, (first, second) in enumerate([("a1", "b1"), ("a2", "b2")]):
        advance(outputs[0], first)
        advance(outputs[1], second)
        result._done = chunk_index == 1
        apply_post_processor_hook(hook, result, streaming=True)

    records = sorted(records_in(dump_dir), key=lambda record: record["output_index"])
    assert [record["output_index"] for record in records] == [0, 1]
    assert all(record["request_id"] == "5" for record in records)
    assert records[0]["frames"] == ["a1", "a2"]
    assert records[1]["frames"] == ["b1", "b2"]


def test_a_record_that_joins_two_sampled_sequences_is_marked(dump_dir):
    """With best_of > n the output index is a rank the result recomputes on every response.

    Two sequences answer separately; the second ranks higher, so it takes index 0
    from the first, and the record of index 0 joins a chunk of each.
    """
    hook = RawOutputDump()
    result = DetokenizedGenerationResultBase(
        7,
        SamplingParams(n=1, best_of=2, temperature=1.0),
        tokenizer=LetterTokenizer(),
        post_processor_hook=hook,
    )

    result._handle_response(engine_response(7, [0, 1, 2], 0, cum_logprob=-3.0, finished=True))
    result._handle_response(
        engine_response(7, [3, 4], 1, cum_logprob=-1.0, finished=True, is_final=True)
    )

    (record,) = records_in(dump_dir)
    assert record["output_index"] == 0
    assert record["text"] == "de", "the text is that of the sequence returned at index 0"
    assert record["frames"] == ["abc", "de"]
    assert record["frames_consistent"] is False
    assert hook._pending == {}


def test_a_non_streaming_request_records_its_one_chunk(dump_dir):
    hook = RawOutputDump()

    drive(hook, FakeResult(), ["the whole answer"], streaming=False)

    (record,) = records_in(dump_dir)
    assert record["streaming"] is False
    assert record["frames"] == ["the whole answer"]
    assert record["text"] == "the whole answer"


def test_the_token_ids_are_recorded_with_the_text(dump_dir):
    """Token ids that add no text (skipped special tokens) are recorded too."""
    hook = RawOutputDump()
    output = CompletionOutput(index=0)
    result = FakeResult(request_id=3, outputs=[output])
    chunks = [("Do", [10, 11]), ("ne.", [12]), ("", [2, 2, 1])]

    for i, (text, token_ids) in enumerate(chunks):
        advance(output, text, token_ids)
        result._done = i == len(chunks) - 1
        apply_post_processor_hook(hook, result, streaming=True)

    (record,) = records_in(dump_dir)
    assert record["text"] == "Done."
    assert record["token_ids"] == [10, 11, 12, 2, 2, 1]
    assert record["num_output_tokens"] == 6


def test_a_non_streaming_request_records_all_its_token_ids(dump_dir):
    hook = RawOutputDump()
    output = CompletionOutput(index=0)
    result = FakeResult(outputs=[output])
    advance(output, "abc", [5, 6, 7])
    result._done = True

    apply_post_processor_hook(hook, result, streaming=False)

    (record,) = records_in(dump_dir)
    assert record["token_ids"] == [5, 6, 7]
    assert record["num_output_tokens"] == 3


def test_one_record_is_written_when_the_output_finishes(dump_dir):
    hook = RawOutputDump()
    result = FakeResult()

    for delta in ["still ", "going"]:
        advance(result.outputs[0], delta)
        apply_post_processor_hook(hook, result, streaming=True)
    assert records_in(dump_dir) == []

    result._done = True
    apply_post_processor_hook(hook, result, streaming=True)
    assert len(records_in(dump_dir)) == 1
    assert hook._pending == {}


def test_a_finished_record_is_filed_under_its_hour_in_a_per_process_file(dump_dir):
    hook = RawOutputDump()

    drive(hook, FakeResult(), ["x"])

    (path,) = record_files(dump_dir)
    (record,) = records_in(dump_dir)
    assert record["event"] == "raw_output"
    assert record["partial"] is False
    assert record["end_reason"] is None
    assert record["aborted"] is False
    assert record["disagg_request_id"] is None
    assert record["started_at"] <= record["finished_at"]
    assert record["finished_at"].endswith("+00:00")
    assert os.path.basename(os.path.dirname(path)) == record["finished_at"][:13]
    assert os.path.basename(path) == f"raw-{os.getpid()}.jsonl"


def test_the_disaggregated_request_id_is_recorded_as_an_exact_string(dump_dir):
    """An id above 2**53 must survive a JSON reader that parses numbers as doubles."""
    hook = RawOutputDump()
    disagg_request_id = 2**60 + 1
    result = FakeResult(
        request_id=12, disaggregated_params=DisaggregatedParams(disagg_request_id=disagg_request_id)
    )

    drive(hook, result, ["x"])

    (record,) = records_in(dump_dir)
    assert record["request_id"] == "12"
    assert record["disagg_request_id"] == str(disagg_request_id)


def test_an_aborted_output_is_recorded_as_aborted(dump_dir):
    hook = RawOutputDump()
    result = FakeResult()
    advance(result.outputs[0], "<tool_call>{")
    result._aborted = True
    result._done = True

    apply_post_processor_hook(hook, result, streaming=True)

    (record,) = records_in(dump_dir)
    assert record["aborted"] is True
    assert record["text"] == "<tool_call>{"


def test_an_unwritable_directory_costs_the_record_not_the_response(tmp_path, monkeypatch):
    blocked = tmp_path / "wall"
    blocked.write_text("not a directory")
    monkeypatch.setenv(RAW_OUTPUT_DIR_ENV, str(blocked / "raw"))
    hook = RawOutputDump()

    (text,) = drive(hook, FakeResult(), ["served ", "anyway"])

    assert text == "served anyway"
    assert hook._dropped == 1
    assert hook._pending == {}


def test_an_unexpected_failure_costs_only_the_record(dump_dir, monkeypatch):
    hook = RawOutputDump()

    def explode(*_args, **_kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(hook, "_write", explode)

    (text,) = drive(hook, FakeResult(), ["served ", "anyway"])

    assert text == "served anyway"
    assert hook._dropped == 1


def test_it_loads_through_the_documented_import_path(dump_dir):
    """``--post_processor_hook`` takes this exact path."""
    hook = load_post_processor_hook(HOOK_PATH)

    assert isinstance(hook, RawOutputDump)
    (text,) = drive(hook, FakeResult(), ["x"])
    assert text == "x"
    assert len(records_in(dump_dir)) == 1


def test_a_request_ended_by_an_error_is_written_as_partial_and_released(dump_dir):
    hook = RawOutputDump()
    output = CompletionOutput(index=0)
    result = FakeResult(request_id=7, outputs=[output])
    for text, token_ids in [("<tool_call>", [40]), ('{"a"', [41, 42])]:
        advance(output, text, token_ids)
        apply_post_processor_hook(hook, result, streaming=True)
    assert records_in(dump_dir) == []

    hook.on_request_end(7, "error")

    assert hook._pending == {}
    (record,) = records_in(dump_dir)
    assert record["partial"] is True
    assert record["end_reason"] == "error"
    assert record["aborted"] is False
    assert record["frames"] == ["<tool_call>", '{"a"']
    assert record["text"] == '<tool_call>{"a"'
    assert record["token_ids"] == [40, 41, 42]


def test_ending_one_request_leaves_the_others_alone(dump_dir):
    hook = RawOutputDump()
    first, second = FakeResult(request_id=7), FakeResult(request_id=8)
    for result in (first, second):
        advance(result.outputs[0], "partial")
        apply_post_processor_hook(hook, result, streaming=True)

    hook.on_request_end(7, "error")
    assert list(hook._pending) == [(8, 0)]

    advance(second.outputs[0], " and done")
    second._done = True
    apply_post_processor_hook(hook, second, streaming=True)
    by_id = {record["request_id"]: record for record in records_in(dump_dir)}
    assert by_id["7"]["partial"] is True
    assert by_id["8"]["partial"] is False
    assert by_id["8"]["text"] == "partial and done"


def test_a_request_that_ended_before_any_output_writes_nothing(dump_dir):
    hook = RawOutputDump()
    apply_post_processor_hook(hook, FakeResult(request_id=7), streaming=True)
    assert (7, 0) in hook._pending

    hook.on_request_end(7, "error")

    assert hook._pending == {}
    assert records_in(dump_dir) == []


def test_a_request_that_generated_only_hidden_tokens_is_written(dump_dir):
    """Token ids without text (special tokens detokenization skips) are output too."""
    hook = RawOutputDump()
    result = FakeResult(request_id=7)
    advance(result.outputs[0], "", [2, 2])
    apply_post_processor_hook(hook, result, streaming=True)

    hook.on_request_end(7, "postprocessing_error")

    (record,) = records_in(dump_dir)
    assert record["frames"] == []
    assert record["token_ids"] == [2, 2]
    assert record["end_reason"] == "postprocessing_error"


def test_ended_requests_do_not_accumulate(dump_dir):
    hook = RawOutputDump()
    for request_id in range(1000):
        result = FakeResult(request_id=request_id)
        advance(result.outputs[0], "x" * 1024)
        apply_post_processor_hook(hook, result, streaming=True)
        hook.on_request_end(request_id, "error")

    assert hook._pending == {}
    assert len(records_in(dump_dir)) == 1000


def test_ending_an_unknown_request_is_harmless(dump_dir):
    hook = RawOutputDump()

    end_post_processor_hook(hook, 12345, "error")

    assert hook._pending == {}
    assert records_in(dump_dir) == []


def test_an_executor_error_without_the_worker_pool_releases_the_request(dump_dir):
    """The executor's error is raised before the hook runs; the request is still ended in it."""
    hook = RawOutputDump()
    executor = ExecutorErrors()
    result = DetokenizedGenerationResultBase(
        7,
        SamplingParams(),
        tokenizer=LetterTokenizer(),
        streaming=True,
        background_error_handler=executor._handle_background_error,
        post_processor_hook=hook,
    )
    result._handle_response(engine_response(7, [7, 4]))
    assert list(hook._pending) == [(7, 0)]

    with pytest.raises(RequestError, match="boom"):
        result._handle_response(engine_error(7, "boom"))

    assert hook._pending == {}
    (record,) = records_in(dump_dir)
    assert record["partial"] is True
    assert record["end_reason"] == "error"
    assert record["text"] == "he"
    assert record["token_ids"] == [7, 4]


def test_outputs_past_the_limit_are_written_as_evicted(dump_dir):
    """An output that never finishes (a stream whose reader went away) cannot pile up."""
    hook = RawOutputDump()
    hook.max_pending = 2
    results = [FakeResult(request_id=request_id) for request_id in range(3)]

    for result in results:
        advance(result.outputs[0], f"r{result.id}", [result.id])
        apply_post_processor_hook(hook, result, streaming=True)

    assert list(hook._pending) == [(1, 0), (2, 0)]
    (record,) = records_in(dump_dir)
    assert record["request_id"] == "0"
    assert record["partial"] is True
    assert record["end_reason"] == "evicted"
    assert record["text"] == "r0"
    assert record["token_ids"] == [0]


def test_the_least_recently_updated_output_is_evicted_first(dump_dir):
    hook = RawOutputDump()
    hook.max_pending = 2
    first, second, third = (FakeResult(request_id=request_id) for request_id in range(3))
    for result in (first, second, first, third):
        advance(result.outputs[0], "x")
        apply_post_processor_hook(hook, result, streaming=True)

    assert list(hook._pending) == [(0, 0), (2, 0)]
    (record,) = records_in(dump_dir)
    assert record["request_id"] == "1"
    assert record["end_reason"] == "evicted"


def test_an_evicted_output_that_produced_nothing_is_not_written(dump_dir):
    hook = RawOutputDump()
    hook.max_pending = 1
    for request_id in range(2):
        apply_post_processor_hook(hook, FakeResult(request_id=request_id), streaming=True)

    assert list(hook._pending) == [(1, 0)]
    assert records_in(dump_dir) == []


def _capture_chunk(
    hook: RawOutputDump,
    result: FakeResult,
    text: str = "",
    token_ids: tuple[int, ...] = (),
    *,
    final: bool = False,
) -> None:
    """Use the existing result fixture and check pass-through and both key bounds."""
    output = result.outputs[0]
    advance(output, text, token_ids)
    result._done = final
    expected_text, expected_ids = output.text, list(output.token_ids)
    apply_post_processor_hook(hook, result, streaming=True)
    assert output.text == expected_text
    assert output.token_ids == expected_ids
    assert len(hook._pending) <= hook.max_pending
    assert len(hook._evicted_keys) <= hook.max_pending
    assert hook._pending.keys().isdisjoint(hook._evicted_keys)
    assert hook._dropped == 0


@pytest.mark.parametrize("prefix, prefix_ids", [("a", (0,)), ("", (2,))])
def test_a_resumed_evicted_output_marks_its_captured_history(
    dump_dir: Path, monkeypatch: pytest.MonkeyPatch, prefix: str, prefix_ids: tuple[int, ...]
) -> None:
    """Hidden-token prefixes need the eviction marker even when frames add up to text."""
    tick = 0

    def utc_now() -> str:
        nonlocal tick
        timestamp = f"2026-11-13T00:00:{tick:02d}+00:00"
        tick += 1
        return timestamp

    monkeypatch.setattr(raw_output_hook, "_utc_now", utc_now)
    hook = RawOutputDump()
    hook.max_pending = 1
    first, second = FakeResult(request_id=7), FakeResult(request_id=8)
    _capture_chunk(hook, first, prefix, prefix_ids)
    _capture_chunk(hook, second, "x", (23,))
    _capture_chunk(hook, first, "b", (1,), final=True)

    evicted, resumed = records_in(dump_dir)
    assert evicted["token_ids"] == list(prefix_ids)
    assert evicted["end_reason"] == "evicted"
    assert resumed["text"] == prefix + "b"
    assert resumed["frames"] == ["b"]
    assert resumed["token_ids"] == [1]
    assert resumed["num_output_tokens"] == 1
    assert resumed["partial"] is True
    assert resumed["end_reason"] == "evicted"
    assert resumed["frames_consistent"] is (prefix == "")
    assert evicted["started_at"] == "2026-11-13T00:00:00+00:00"
    assert evicted["finished_at"] == "2026-11-13T00:00:02+00:00"
    assert resumed["started_at"] == "2026-11-13T00:00:03+00:00"
    assert resumed["finished_at"] == "2026-11-13T00:00:04+00:00"
    assert first.outputs[0].length == 2
    assert hook._evicted_keys == set()


@pytest.mark.parametrize("end_reason", [None, "error"])
def test_an_empty_resumed_final_is_written_but_an_empty_callback_segment_is_not(
    dump_dir: Path, end_reason: str | None
) -> None:
    hook = RawOutputDump()
    hook.max_pending = 1
    first, second = FakeResult(request_id=7), FakeResult(request_id=8)
    _capture_chunk(hook, first, "a", (0,))
    _capture_chunk(hook, second, "x", (23,))
    _capture_chunk(hook, first, final=end_reason is None)
    if end_reason is not None:
        hook.on_request_end(first.id, end_reason)

    records = [record for record in records_in(dump_dir) if record["request_id"] == "7"]
    if end_reason is None:
        assert len(records) == 2
        assert records[-1]["text"] == "a"
        assert records[-1]["frames"] == []
        assert records[-1]["token_ids"] == []
        assert records[-1]["num_output_tokens"] == 0
        assert records[-1]["partial"] is True
        assert records[-1]["end_reason"] == "evicted"
    else:
        assert len(records) == 1, "the actual callback still suppresses its empty segment"
    assert (first.id, 0) not in hook._pending
    assert (first.id, 0) not in hook._evicted_keys


@pytest.mark.parametrize("reason", ["error", "postprocessing_error", ""])
def test_the_actual_end_reason_takes_priority_over_evicted_history(
    dump_dir: Path, reason: str
) -> None:
    hook = RawOutputDump()
    hook.max_pending = 1
    first, second = FakeResult(request_id=7), FakeResult(request_id=8)
    _capture_chunk(hook, first, "a", (0,))
    _capture_chunk(hook, second, "x", (23,))
    _capture_chunk(hook, first, "b", (1,))

    hook.on_request_end(first.id, reason)

    resumed = records_in(dump_dir)[-1]
    assert resumed["request_id"] == "7"
    assert resumed["partial"] is True
    assert resumed["end_reason"] == reason
    assert resumed["text"] == "ab"
    assert resumed["frames"] == ["b"]
    assert resumed["token_ids"] == [1]
    assert hook._pending == {}
    assert hook._evicted_keys == {(second.id, 0)}


def test_marker_overflow_marks_unknown_new_segments_without_tainting_retained_ones(
    dump_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hook = RawOutputDump()
    hook.max_pending = 1
    warnings = []

    def note_warning(message: str) -> None:
        assert not hook._lock.locked(), "logging must not hold the pending lock"
        warnings.append(message)

    monkeypatch.setattr(raw_output_hook.logger, "warning", note_warning)
    first, second, third, fourth = (FakeResult(request_id=i) for i in range(7, 11))
    _capture_chunk(hook, first, "a", (0,))
    _capture_chunk(hook, second, "b", (1,))
    assert hook._history_uncertain is False
    _capture_chunk(hook, third, "x", (23,))
    assert hook._history_uncertain is True
    assert hook._evicted_keys == {(first.id, 0)}

    # The third output's history was known complete before its arrival overflowed the markers.
    _capture_chunk(hook, third, final=True)
    _capture_chunk(hook, second, "b", (1,), final=True)
    _capture_chunk(hook, first, "a", (0,), final=True)
    assert hook._evicted_keys == set()
    _capture_chunk(hook, fourth, "x", (23,), final=True)

    records = records_in(dump_dir)
    assert [record["end_reason"] for record in records] == [
        "evicted",
        "evicted",
        None,
        "history_unavailable",
        "evicted",
        "history_unavailable",
    ]
    assert [record["partial"] for record in records] == [True, True, False, True, True, True]
    assert records[2]["text"] == "x"
    assert records[3]["text"] == "bb"
    assert records[3]["token_ids"] == [1]
    assert records[-1]["text"] == "x", "even fresh output is still recorded after saturation"
    assert hook._pending == {}
    assert len(warnings) == 1
    assert "history_unavailable" in warnings[0]


def test_ending_a_request_clears_only_its_markers_and_does_not_clear_uncertainty(
    dump_dir: Path,
) -> None:
    hook = RawOutputDump()
    hook.max_pending = 2
    first, second, third, fourth = (FakeResult(request_id=i) for i in range(7, 11))
    for result in (first, second, third, fourth):
        _capture_chunk(hook, result, "a", (0,))
    assert hook._evicted_keys == {(first.id, 0), (second.id, 0)}

    hook.on_request_end(first.id, "error")
    assert hook._evicted_keys == {(second.id, 0)}
    assert set(hook._pending) == {(third.id, 0), (fourth.id, 0)}
    fifth, sixth = FakeResult(request_id=11), FakeResult(request_id=12)
    _capture_chunk(hook, fifth, "b", (1,))
    _capture_chunk(hook, sixth, "x", (23,))
    assert hook._history_uncertain is True
    assert hook._evicted_keys == {(second.id, 0), (third.id, 0)}
    hook.on_request_end(second.id, "error")
    hook.on_request_end(third.id, "error")
    assert hook._evicted_keys == set()
    assert hook._history_uncertain is True

    # Clearing all markers cannot recover an eviction identity that did not fit.
    _capture_chunk(hook, fourth, "b", (1,), final=True)
    record = records_in(dump_dir)[-1]
    assert record["request_id"] == str(fourth.id)
    assert record["partial"] is True
    assert record["end_reason"] == "history_unavailable"


def test_repeated_eviction_consumes_and_recreates_markers_within_the_budget(
    dump_dir: Path,
) -> None:
    hook = RawOutputDump()
    hook.max_pending = 1
    first, second = FakeResult(request_id=7), FakeResult(request_id=8)
    _capture_chunk(hook, first, "a", (0,))
    _capture_chunk(hook, second, "x", (23,))
    _capture_chunk(hook, first, "b", (1,))
    assert hook._evicted_keys == {(second.id, 0)}
    _capture_chunk(hook, second, "b", (1,))
    assert hook._evicted_keys == {(first.id, 0)}
    _capture_chunk(hook, first, final=True)
    _capture_chunk(hook, second, final=True)

    assert hook._pending == {}
    assert hook._evicted_keys == set()
    assert hook._history_uncertain is False
    assert all(record["end_reason"] == "evicted" for record in records_in(dump_dir))
    assert records_in(dump_dir)[-1]["text"] == "xb"


def test_zero_pending_capacity_still_records_evicted_and_uncertain_final_segments(
    dump_dir: Path,
) -> None:
    hook = RawOutputDump()
    hook.max_pending = 0
    result = FakeResult(request_id=7)
    _capture_chunk(hook, result, "a", (0,))
    assert hook._pending == {}
    assert hook._evicted_keys == set()
    assert hook._history_uncertain is True
    _capture_chunk(hook, result, "b", (1,), final=True)
    _capture_chunk(hook, FakeResult(request_id=8), "x", (23,), final=True)

    records = records_in(dump_dir)
    assert [record["end_reason"] for record in records] == [
        "evicted",
        "history_unavailable",
        "history_unavailable",
    ]
    assert all(record["partial"] for record in records)
    assert records[1]["text"] == "ab"
    assert records[1]["frames"] == ["b"]
    assert records[1]["num_output_tokens"] == 1
    assert records[2]["text"] == "x"


class _Pipe:
    def __init__(self, items):
        self._items = list(items)

    async def get_async(self):
        return self._items.pop(0)


def test_the_worker_ends_a_request_that_fails_after_output(dump_dir):
    """An executor error reaching the post-processing worker releases the request in the dump."""
    hook = RawOutputDump()
    result = FakeResult(request_id=7)
    advance(result.outputs[0], "cut off")
    apply_post_processor_hook(hook, result, streaming=True)

    worker = object.__new__(PostprocWorker)
    worker._records = {7: result}
    worker._post_processor_hook = hook
    worker._to_stop = asyncio.Event()
    worker._pull_pipe = _Pipe([[PostprocWorker.Input(rsp=ErrorResponse(7, "boom", 7))], None])

    async def drain():
        async for _ in worker._mainloop():
            pass

    asyncio.run(drain())

    assert hook._pending == {}
    (record,) = records_in(dump_dir)
    assert record["partial"] is True
    assert record["end_reason"] == "error"
    assert record["text"] == "cut off"


def test_the_worker_ends_a_request_whose_post_processing_raises_after_output(dump_dir):
    """The worker builds its own record and runs the hook on it, then the formatter raises.

    The record is the worker's real one (``default_record_creator``), handled by
    ``_handle_input`` for each response: the hook records the first two chunks,
    and the per-endpoint formatter raises on the second, after the hook ran.
    """
    hook = RawOutputDump()
    formatted = []

    def post_processor(record, args):
        formatted.append(record.outputs[0].text_diff)
        if len(formatted) == 2:
            raise RuntimeError("formatter bug")
        return []

    def request_input(token_ids):
        return PostprocWorker.Input(
            rsp=engine_response(7, token_ids),
            sampling_params=SamplingParams(),
            postproc_params=PostprocParams(
                post_processor=post_processor, postproc_args=SimpleNamespace()
            ),
            streaming=True,
        )

    worker = object.__new__(PostprocWorker)
    worker._records = {}
    worker._record_creator = PostprocWorker.default_record_creator
    worker._tokenizer = LetterTokenizer()
    worker._post_processor_hook = hook
    worker._to_stop = asyncio.Event()
    worker._pull_pipe = _Pipe([[request_input([7, 4])], [request_input([11, 11])], None])

    async def drain():
        return [batch async for batch in worker._mainloop() if batch]

    batches = asyncio.run(drain())

    assert formatted == ["he", "ll"]
    assert isinstance(batches[0][0], PostprocWorker.Output)
    (error,) = batches[1]
    assert isinstance(error, ErrorResponse)
    assert "formatter bug" in error.error_msg
    assert worker._records == {}
    assert hook._pending == {}
    (record,) = records_in(dump_dir)
    assert record["partial"] is True
    assert record["end_reason"] == "postprocessing_error"
    assert record["frames"] == ["he", "ll"]
    assert record["text"] == "hell"
    assert record["token_ids"] == [7, 4, 11, 11]
