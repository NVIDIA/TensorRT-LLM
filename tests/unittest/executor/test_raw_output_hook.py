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

import pytest

import tensorrt_llm.executor.raw_output_hook as raw_output_hook
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.executor.postproc_worker import PostprocWorker
from tensorrt_llm.executor.postprocessor_hook import (
    apply_post_processor_hook,
    end_post_processor_hook,
    load_post_processor_hook,
)
from tensorrt_llm.executor.raw_output_hook import RAW_OUTPUT_DIR_ENV, RawOutputDump
from tensorrt_llm.executor.result import CompletionOutput
from tensorrt_llm.executor.utils import ErrorResponse

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
