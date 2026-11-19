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
"""CPU tests of the public GSM8K protocol and durable result records.

Cover prompt construction, request fields, scoring, question selection, the pinned split
inventory, reply classification, and ledger resume rules. No GPU, checkpoint, dataset
download or model server is used.
"""

import argparse
import dataclasses
import errno
import json
import os
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from utils.llm_data import llm_models_root

EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "llm-api"
sys.path.insert(0, str(EXAMPLES_DIR))
import eval_gsm8k  # noqa: E402
from gsm8k_eval import protocol, records, runner  # noqa: E402

pytestmark = pytest.mark.cpu_only

# Change only together with a new PROTOCOL_ID and a regenerated manifest.json.
PROTOCOL_DEFINITION_SHA256 = "1b20973ebed2049625b91f48ef2b6ed7d2089a8787d32765737faa921b632e02"


# ------------------------------------------------------------------------------------ scorer


@pytest.mark.parametrize(
    "content, parsed",
    [
        ("The answer is 42.", "42"),
        ("-7 degrees", "-7"),
        ("x = -5", "-5"),
        ("x-5", "5"),
        ("10 000", "0"),
        ("−5", "5"),
        ("(+5)", "5"),
        ("pages 10-15", "15"),
        ("It costs $1,234.50 in total", "1234.5"),
        ("1,000,000 grains", "1000000"),
        ("3.10", "3.1"),
        ("007", "7"),
        ("-0", "0"),
        ("-0.00", "0"),
        ("first 12, then 7, finally 18", "18"),
        ("He has 3 apples and -2 pears", "-2"),
        ("1,2345", "2345"),
        ("so 18.", "18"),
        ("1/2", "2"),
        ("50%", "50"),
        ("#### 1,234", "1234"),
        ("3.14.15", "3.14"),
    ],
)
def test_scorer_takes_the_last_normalized_number(content, parsed):
    score = protocol.score_content(content, parsed)
    assert (score.parsed, score.parse_status, score.correct) == (parsed, "ok", True)


@pytest.mark.parametrize(
    "content, status",
    [
        (None, "empty_content"),
        ("", "empty_content"),
        (" \n", "empty_content"),
        ("no digits here", "no_number"),
        (".5", "no_number"),
        ("version v1.2 and x2", "no_number"),
    ],
)
def test_scorer_reports_parse_failures(content, status):
    score = protocol.score_content(content, "4")
    assert (score.parsed, score.parse_status, score.correct) == (None, status, False)


def test_scorer_compares_exact_normalized_values():
    assert protocol.score_content("The total is 1,000.0", "1000").correct
    assert not protocol.score_content("The total is -1000", "1000").correct
    assert not protocol.score_content("The total is 1000.5", "1000").correct


def test_gold_is_the_number_after_the_last_delimiter():
    assert protocol.extract_gold("3 + 4 = 7\n#### 7") == "7"
    assert protocol.extract_gold("#### 1 #### 1,234") == "1234"
    assert protocol.extract_gold("drop\n#### -3") == "-3"
    with pytest.raises(ValueError, match="delimiter"):
        protocol.extract_gold("no gold here 7")
    with pytest.raises(ValueError, match="grammar"):
        protocol.extract_gold("#### 7 dollars")


# ------------------------------------------------------------------------ prompt and request


def test_prompt_is_exemplars_then_question_in_one_message():
    prompt = protocol.build_prompt([("Q1 {x}", "A1\n#### 1"), (" Q2 ", "A2\n#### 2")], "Q3?")
    assert prompt == (
        "Question: Q1 {x}\nAnswer: A1\n#### 1\n\n"
        "Question:  Q2 \nAnswer: A2\n#### 2\n\n"
        "Question: Q3?\nAnswer:"
    )


@pytest.mark.parametrize("kwargs", [None, {"reasoning_effort": 128}, {"enable_thinking": False}])
def test_request_sends_exactly_the_protocol_fields(kwargs):
    body = runner.PinnedChatCompletion(model="served").request_body(
        "prompt text", protocol.request_fields(kwargs)
    )
    assert body == {
        "model": "served",
        "messages": [{"role": "user", "content": "prompt text"}],
        "temperature": 0,
        "top_k": 1,
        "seed": 42,
        "max_tokens": 65536,
        "n": 1,
        "stream": False,
        **({"chat_template_kwargs": kwargs} if kwargs is not None else {}),
    }
    record = protocol.request_record(body)
    assert record.pop("messages") == [
        {"role": "user", "content_sha256": protocol.sha256_text("prompt text"), "content_chars": 11}
    ]
    assert record == {key: value for key, value in body.items() if key != "messages"}


def test_protocol_definition_is_frozen():
    digest = protocol.sha256_text(protocol.canonical_json(protocol.protocol_definition()))
    assert digest == PROTOCOL_DEFINITION_SHA256, (
        "the public protocol changed: give it a new PROTOCOL_ID, regenerate manifest.json and "
        "update this digest"
    )


# ---------------------------------------------------------------------------------- manifest


def test_committed_manifest_pins_the_first_100_rows_and_ten_exemplars():
    manifest = protocol.load_manifest()
    questions = manifest["questions"]
    assert [q["index"] for q in questions] == list(range(100))
    assert len({q["id"] for q in questions}) == len({q["prompt_sha256"] for q in questions}) == 100
    assert manifest["splits"]["train"]["rows"] == 7473
    assert manifest["splits"]["test"]["rows"] == 1319
    assert manifest["protocol"]["dataset"]["revision"] == protocol.DATASET_REVISION
    indices = [item["index"] for item in manifest["exemplars"]["items"]]
    assert indices == protocol.exemplar_indices(7473)
    assert manifest["exemplars"]["generated_with"]["call"] == (
        "random.Random(42).sample(range(7473), 10)"
    )


def test_staged_gsm8k_matches_committed_pins():
    root = llm_models_root()
    if root is None:
        pytest.skip("LLM models root is unavailable")
    dataset_dir = root / "datasets/openai/gsm8k"
    if not any((dataset_dir / name).exists() for name in protocol.SPLIT_FILES.values()):
        pytest.skip("GSM8K dataset is not staged")
    manifest = protocol.load_manifest()
    problems, _ = protocol.load_problems(dataset_dir, manifest)
    assert [p.prompt_sha256 for p in problems] == [
        q["prompt_sha256"] for q in manifest["questions"]
    ]


def _rows(count, split):
    if split == "train":
        return [
            (f"Train {i}: what is {i} times 2?", f"{i} * 2 = {2 * i}\n#### {2 * i}")
            for i in range(count)
        ]
    return [
        (f"Row {i}: what is {i} plus 3?", f"{i} + 3 = {i + 3}\n#### {i + 3}") for i in range(count)
    ]


def _synthetic_manifest(train, test, digests=None):
    return protocol.build_manifest(
        train,
        test,
        file_sha256=digests or {"train": "0" * 64, "test": "1" * 64},
        python_version="3.12",
    )


def _rows_digest(manifest, train, test):
    """The pinned-inventory digest of a synthetic test split (what TEST_ROWS_SHA256 pins)."""
    exemplars = [train[item["index"]] for item in manifest["exemplars"]["items"]]
    return protocol.sha256_text(protocol.canonical_json(protocol.inventory_rows(exemplars, test)))


@pytest.fixture
def pin_split(monkeypatch):
    """Pin the test-split inventory to the given synthetic splits."""

    def pin(manifest, train, test):
        monkeypatch.setattr(protocol, "TEST_ROWS_SHA256", _rows_digest(manifest, train, test))

    return pin


def test_manifest_must_embed_the_code_protocol(tmp_path):
    manifest = _synthetic_manifest(_rows(15, "train"), _rows(105, "test"))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert protocol.load_manifest(path)["questions"][0]["id"] == "test-0000"
    manifest["protocol"]["request"]["fields"]["max_tokens"] = 256
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="protocol"):
        protocol.load_manifest(path)


def test_manifest_verification_rejects_changed_rows_and_order(pin_split):
    train, test = _rows(15, "train"), _rows(105, "test")
    manifest = _synthetic_manifest(train, test)
    pin_split(manifest, train, test)
    problems = protocol.build_problems(manifest, train, test)
    assert [p.index for p in problems] == list(range(100))
    assert problems[7].gold == "10"
    assert problems[7].prompt.endswith("Question: Row 7: what is 7 plus 3?\nAnswer:")

    changed = list(test)
    changed[5] = ("Row 5: what is 5 plus 4?", test[5][1])
    with pytest.raises(ValueError, match="test row 5"):
        protocol.build_problems(manifest, train, changed)
    swapped = [test[1], test[0]] + test[2:]
    with pytest.raises(ValueError, match="test row 0"):
        protocol.build_problems(manifest, train, swapped)
    with pytest.raises(ValueError, match="rows"):
        protocol.build_problems(manifest, train, test[:-1])
    shot = manifest["exemplars"]["items"][0]["index"]
    edited = list(train)
    edited[shot] = (train[shot][0], train[shot][1] + " ")
    with pytest.raises(ValueError, match=f"train row {shot}"):
        protocol.build_problems(manifest, edited, test)


def _write_split(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"question": [q for q, _ in rows], "answer": [a for _, a in rows]})
    pq.write_table(table, path)


def _stage_dataset(dataset_dir, train, test):
    _write_split(dataset_dir / protocol.SPLIT_FILES["train"], train)
    _write_split(dataset_dir / protocol.SPLIT_FILES["test"], test)
    return {
        split: protocol.sha256_file(dataset_dir / protocol.SPLIT_FILES[split])
        for split in ("train", "test")
    }


def test_staged_parquet_snapshot_is_read_in_split_order(tmp_path, pin_split):
    train, test = _rows(15, "train"), _rows(105, "test")
    digests = _stage_dataset(tmp_path, train, test)
    manifest = _synthetic_manifest(train, test, digests)
    pin_split(manifest, train, test)
    problems, files = protocol.load_problems(tmp_path, manifest)
    assert [p.question for p in problems] == [q for q, _ in test[:100]]
    assert files["test"]["equals_manifest_snapshot"]
    assert files["train"]["sha256"] == digests["train"]
    (tmp_path / protocol.SPLIT_FILES["train"]).unlink()
    with pytest.raises(FileNotFoundError, match="train"):
        protocol.load_problems(tmp_path, manifest)


# ---------------------------------------------------------------------------- count selection


@pytest.mark.parametrize("count", [1, 2, 99, 100, 101, 105])
def test_count_selects_the_first_n_rows_in_split_order(pin_split, count):
    train, test = _rows(15, "train"), _rows(105, "test")
    manifest = _synthetic_manifest(train, test)
    pin_split(manifest, train, test)
    problems = protocol.build_problems(manifest, train, test, count)
    assert [p.id for p in problems] == [protocol.question_id(i) for i in range(count)]
    assert len({p.prompt_sha256 for p in problems}) == count
    assert problems[-1].question == test[count - 1][0]
    first100 = protocol.build_problems(manifest, train, test)
    assert problems[: min(count, 100)] == first100[: min(count, 100)]


@pytest.mark.parametrize("count", [0, -1, 1320, 100000, True, 5.0, "5", None])
def test_count_outside_1_to_1319_is_refused(count):
    with pytest.raises(ValueError, match="1..1319"):
        protocol.check_count(count)
    with pytest.raises(ValueError, match="1..1319"):
        protocol.selection(count)


def test_count_beyond_the_staged_split_is_refused(pin_split):
    train, test = _rows(15, "train"), _rows(105, "test")
    manifest = _synthetic_manifest(train, test)
    pin_split(manifest, train, test)
    with pytest.raises(ValueError, match="106 questions requested; the test split has 105 rows"):
        protocol.build_problems(manifest, train, test, 106)


def test_rows_after_the_first_100_are_authenticated_by_the_split_inventory(pin_split):
    train, test = _rows(15, "train"), _rows(105, "test")
    manifest = _synthetic_manifest(train, test)
    pin_split(manifest, train, test)
    tail_edit = list(test)
    tail_edit[103] = ("Row 103: what is 103 plus 4?", test[103][1])
    with pytest.raises(ValueError, match="pinned row inventory"):
        protocol.build_problems(manifest, train, tail_edit, 100)
    tail_gold = list(test)
    tail_gold[104] = (test[104][0], "no gold")
    with pytest.raises(ValueError, match="pinned inventory"):
        protocol.build_problems(manifest, train, tail_gold, 101)


def test_pinned_split_constants_describe_the_public_test_split():
    manifest = protocol.load_manifest()
    assert protocol.TEST_SPLIT_ROWS == manifest["splits"]["test"]["rows"] == 1319
    assert protocol.selection(1319) == {
        "split": "test",
        "count": 1319,
        "rows": "0..1318",
        "test_rows_sha256": protocol.TEST_ROWS_SHA256,
    }
    assert protocol.question_id(1318) == "test-1318"


# ---------------------------------------------------------------------- replies and records


def _payload(
    content="The answer is 4.", reasoning="analysis 9", finish="stop", prompt=1800, completion=300
):
    return json.dumps(
        {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": reasoning,
                    },
                    "finish_reason": finish,
                }
            ],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
        }
    )


_NO_USAGE = json.dumps({"choices": [{"message": {"content": "4"}, "finish_reason": "stop"}]})


@pytest.mark.parametrize(
    "http_status, text, max_seq_len, status, error",
    [
        pytest.param(200, _payload(), 69632, records.OK, None, id="stop"),
        pytest.param(200, _payload(content=None), 69632, records.OK, None, id="reasoning-only"),
        pytest.param(
            200,
            _payload(finish="length", completion=65536),
            69632,
            records.TRUNCATED,
            None,
            id="whole-allowance",
        ),
        pytest.param(
            200,
            _payload(finish="length", completion=4000),
            69632,
            records.ERROR,
            "below the 65536-token",
            id="early-length",
        ),
        pytest.param(
            200,
            _payload(prompt=5000),
            69632,
            records.ERROR,
            "cannot hold 5000 prompt tokens",
            id="prompt-too-long",
        ),
        pytest.param(200, _payload(prompt=5000), None, records.OK, None, id="max-seq-len-unknown"),
        pytest.param(
            200,
            _payload(content="<|channel|>final<|message|>4"),
            69632,
            records.ERROR,
            "channel delimiters",
            id="channel-leak",
        ),
        pytest.param(
            200,
            _payload(finish="abort"),
            69632,
            records.ERROR,
            "unexpected finish_reason",
            id="abort",
        ),
        pytest.param(200, "not json", 69632, records.ERROR, "malformed reply", id="not-json"),
        pytest.param(
            200,
            json.dumps({"choices": []}),
            69632,
            records.ERROR,
            "malformed reply",
            id="no-choice",
        ),
        pytest.param(200, _NO_USAGE, 69632, records.ERROR, "no prompt token count", id="no-usage"),
        pytest.param(500, "boom", 69632, records.ERROR, "HTTP 500: boom", id="http-500-body"),
    ],
)
def test_reply_classification(http_status, text, max_seq_len, status, error):
    got_status, got_error, fields = records.interpret_reply(
        http_status, text, max_seq_len=max_seq_len
    )
    assert got_status == status
    assert (got_error is None) if error is None else (error in got_error)
    if status == records.OK:
        assert fields["reasoning_chars"] == len("analysis 9")
    if http_status != 200:
        assert fields == {"http_status": http_status}


def _problem(index, gold="4"):
    return protocol.Problem(
        id=protocol.question_id(index),
        index=index,
        question=f"q{index}",
        answer=f"#### {gold}",
        gold=gold,
        prompt=f"prompt {index}",
        prompt_sha256=protocol.sha256_text(f"prompt {index}"),
    )


def _record(problem, status=records.OK, content="The answer is 4.", identity="run", attempt=1):
    fields = {}
    if status in records.COMPLETED:
        finish = "stop" if status == records.OK else "length"
        _, _, fields = records.interpret_reply(
            200, _payload(content=content, finish=finish, completion=65536), max_seq_len=69632
        )
    return records.make_record(
        problem,
        runner.PinnedChatCompletion(model="served").request_body(
            problem.prompt, protocol.request_fields()
        ),
        status,
        None if status in records.COMPLETED else "failed",
        fields,
        identity=identity,
        attempt=attempt,
        started_at="t0",
        finished_at="t1",
        latency_s=1.5,
    )


def _ledger(path, problems, identity="run"):
    return records.Ledger(path, identity, {p.id: (p.index, p.prompt_sha256) for p in problems})


@pytest.mark.parametrize(
    "status,field",
    [
        (None, "missing"),
        (records.ERROR, "errors"),
        (records.TIMEOUT, "timeouts"),
        (records.TRUNCATED, "truncated"),
    ],
)
def test_incomplete_run_fails_even_when_score_meets_threshold(tmp_path, status, field):
    answered, affected = _problem(0), _problem(1)
    problems = [answered, affected]
    ledger = _ledger(tmp_path / "results.jsonl", problems)
    ledger.append(_record(answered))
    if status is not None:
        ledger.append(_record(affected, status=status, content="7"))
    summary = records.summarize(ledger.load(), problems, protocol.PROTOCOL_ID)
    assert summary[field] == [affected.id]
    assert summary["correct"] == 1
    for minimum in (None, 1):
        verdict = records.verdict(summary, min_correct=minimum)
        assert not verdict["passed"]
        assert affected.id in verdict["problems"][0]


def test_completed_run_score_threshold(tmp_path):
    problems = [_problem(0), _problem(1)]
    ledger = _ledger(tmp_path / "results.jsonl", problems)
    ledger.append(_record(problems[0]))
    ledger.append(_record(problems[1], content="7"))
    summary = records.summarize(ledger.load(), problems, protocol.PROTOCOL_ID)
    assert summary["correct"] == 1
    assert records.verdict(summary, min_correct=1)["passed"]
    assert not records.verdict(summary, min_correct=2)["passed"]
    with pytest.raises(ValueError, match="min_correct"):
        records.verdict(summary, min_correct=3)


def test_reasoning_only_reply_is_a_visible_parse_failure():
    record = _record(_problem(0), content=None)
    assert (record["status"], record["parse_status"], record["correct"]) == (
        records.OK,
        "empty_content",
        False,
    )
    assert record["content"] is None and record["reasoning_chars"] > 0


def test_ledger_keeps_failed_attempts_and_never_rewrites_a_completed_reply(tmp_path):
    problems = [_problem(i) for i in range(3)]
    ledger = _ledger(tmp_path / "results.jsonl", problems)
    assert ledger.load().records == []
    ledger.append(_record(problems[0]))
    ledger.append(_record(problems[1], status=records.ERROR))
    ledger.append(_record(problems[1], status=records.TIMEOUT, attempt=2))
    with pytest.raises(records.LedgerError, match="already has a completed reply"):
        ledger.append(_record(problems[0], status=records.ERROR, attempt=2))
    state = _ledger(tmp_path / "results.jsonl", problems).load()
    assert set(state.completed) == {"test-0000"}
    assert state.latest["test-0001"]["status"] == records.TIMEOUT
    assert len(state.records) == 3


def test_ledger_rejects_foreign_changed_and_duplicate_records(tmp_path):
    problems = [_problem(i) for i in range(2)]
    path = tmp_path / "results.jsonl"
    _ledger(path, problems).append(_record(problems[0]))
    with pytest.raises(records.LedgerError, match="run identity"):
        _ledger(path, problems, identity="other").load()
    with pytest.raises(records.LedgerError, match="foreign"):
        _ledger(path, [problems[1]]).load()
    changed = [dataclasses.replace(problems[0], prompt_sha256="0" * 64), problems[1]]
    with pytest.raises(records.LedgerError, match="different index or prompt"):
        _ledger(path, changed).load()
    with open(path, "a") as f:
        f.write(json.dumps(_record(problems[0])) + "\n")
    with pytest.raises(records.LedgerError, match="already has a completed reply"):
        _ledger(path, problems).load()
    unnamed = _record(problems[1])
    del unnamed["request"]["model"]
    (tmp_path / "unnamed.jsonl").write_text(json.dumps(unnamed) + "\n")
    with pytest.raises(records.LedgerError, match="does not name the served model"):
        _ledger(tmp_path / "unnamed.jsonl", problems).load()


def test_ledger_moves_an_interrupted_last_line_aside(tmp_path):
    problems = [_problem(i) for i in range(2)]
    path = tmp_path / "results.jsonl"
    _ledger(path, problems).append(_record(problems[0]))
    with open(path, "a") as f:
        f.write(json.dumps(_record(problems[1]))[:40])
    state = _ledger(path, problems).load()
    assert set(state.completed) == {"test-0000"}
    (aside,) = state.quarantined
    assert Path(aside).read_text() == json.dumps(_record(problems[1]))[:40]
    assert path.read_text().count("\n") == 1 and path.read_text().endswith("\n")
    with open(path, "a") as f:
        f.write("{corrupt\n" + json.dumps(_record(problems[1])) + "\n")
    with pytest.raises(records.LedgerError, match="unreadable"):
        _ledger(path, problems).load()


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        1,
        "text",
        {"choices": [None]},
        {"choices": [{"message": []}]},
        {"choices": [{"message": "text"}]},
        {"choices": [{"message": None}]},
    ],
)
def test_non_object_reply_is_recorded_as_error(payload):
    status, error, fields = records.interpret_reply(200, json.dumps(payload))
    assert status == records.ERROR
    assert error.startswith("malformed reply:")
    assert fields == {"http_status": 200}


@pytest.mark.parametrize("value", ["inf", "-inf", "nan", "0", "-1"])
def test_time_budgets_refuse_invalid_values_before_setup(tmp_path, value):
    with pytest.raises(argparse.ArgumentTypeError, match="finite and positive"):
        eval_gsm8k._positive_float(value)
    for name in ("time_budget_s", "request_timeout_s"):
        budgets = {"time_budget_s": 60, name: float(value)}
        with pytest.raises(runner.EvaluationError, match=name + " must be finite and positive"):
            runner.run(
                dataset_dir=tmp_path / "absent",
                output_dir=tmp_path / "output",
                endpoint="http://unused",
                concurrency=1,
                **budgets,
            )
    assert not (tmp_path / "output").exists()
    assert eval_gsm8k._positive_float("0.5") == 0.5


@pytest.mark.parametrize(
    "bad_reply",
    [
        _payload(content="\ud800"),
        _payload().replace("analysis 9", "\\udfff"),
        "[" * 2000 + "0" + "]" * 2000,
    ],
)
def test_malformed_reply_is_recorded_and_next_question_continues(tmp_path, bad_reply):
    problems = [_problem(0), _problem(1)]
    ledger = _ledger(tmp_path / "results.jsonl", problems)
    replies = iter([bad_reply, _payload()])
    deps = runner.RunDeps(
        post=AsyncMock(side_effect=lambda *args, **kwargs: (200, next(replies))),
        monotonic=lambda: 0,
    )
    session = runner._Session(
        pending=problems,
        ledger=ledger,
        identity="run",
        attempt=1,
        max_seq_len=69632,
        concurrency=1,
        deadline=60,
        request_timeout_s=None,
        request_fields=protocol.request_fields(),
    )
    runner._ask(session, "http://unused", "served", SimpleNamespace(poll=lambda: None), deps)
    state = ledger.load()
    assert [r["status"] for r in state.records] == [records.ERROR, records.OK]
    assert "malformed reply" in state.records[0]["error"]
    assert set(state.completed) == {problems[1].id}


def test_record_construction_contains_unserializable_endpoint_metadata(tmp_path):
    problem = _problem(0)
    record = records.make_record(
        problem,
        runner.PinnedChatCompletion(model="served").request_body(
            problem.prompt, protocol.request_fields()
        ),
        records.OK,
        None,
        {"usage": {"bad": object()}},
        identity="run",
        attempt=1,
        started_at="t0",
        finished_at="t1",
        latency_s=0,
    )
    ledger = _ledger(tmp_path / "results.jsonl", [problem])
    ledger.append(record)
    assert ledger.load().records[0]["status"] == records.ERROR


def test_ledger_finishes_short_writes_and_escapes_unicode(tmp_path, monkeypatch):
    problem = _problem(0)
    ledger = _ledger(tmp_path / "results.jsonl", [problem])
    record = _record(problem)
    record["stop_reason"] = "\ud800"
    write = os.write
    monkeypatch.setattr(os, "write", lambda fd, data: write(fd, data[:17]))
    ledger.append(record)
    assert ledger.load().completed[problem.id] == record
    assert ledger.path.read_bytes().endswith(b"\n")


@pytest.mark.parametrize("failure", ["write", "zero", "fsync", "rollback"])
def test_failed_append_does_not_complete_or_corrupt_previous_record(tmp_path, monkeypatch, failure):
    problems = [_problem(0), _problem(1)]
    ledger = _ledger(tmp_path / "results.jsonl", problems)
    ledger.append(_record(problems[0]))
    original = ledger.path.read_bytes()
    write, fsync = os.write, os.fsync
    calls = 0

    def fail_write(fd, data):
        nonlocal calls
        calls += 1
        if calls == 1:
            return write(fd, data[:17])
        if failure == "zero":
            return 0
        raise OSError(errno.ENOSPC, "disk full")

    def fail_fsync(fd):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError(errno.EIO, "sync failed")
        return fsync(fd)

    with monkeypatch.context() as patch:
        if failure == "fsync":
            patch.setattr(os, "fsync", fail_fsync)
        else:
            patch.setattr(os, "write", fail_write)
        if failure == "rollback":

            def fail_truncate(fd, size):
                raise OSError(errno.EIO, "truncate failed")

            patch.setattr(os, "ftruncate", fail_truncate)
        with pytest.raises(OSError):
            ledger.append(_record(problems[1]))
        assert problems[1].id not in ledger._completed
        if failure == "rollback":
            with pytest.raises(records.LedgerError, match="reload"):
                ledger.append(_record(problems[1]))
        else:
            assert ledger.path.read_bytes() == original
    # Resume repairs a torn tail if rollback failed; no completed result is lost.
    assert set(ledger.load().completed) == {problems[0].id}
    ledger.append(_record(problems[1]))
    assert set(ledger.load().completed) == {p.id for p in problems}


def test_managed_environment_values_are_private_and_rotation_resumes(tmp_path, monkeypatch, capfd):
    config = tmp_path / "config.yaml"
    config.write_text("max_batch_size: 1\nmax_seq_len: 69632\n")
    output = tmp_path / "results"
    problems = [_problem(0), _problem(1)]
    monkeypatch.setattr(protocol, "load_problems", lambda *args: (problems, {}))
    monkeypatch.setenv("EVAL_INHERITED_TOKEN", "fake-inherited-secret")
    monkeypatch.setenv("HF_TOKEN", "fake-parent-secret")
    launches, prompts = [], []
    replies = iter([200, 500, 200])

    @contextmanager
    def launch(argv, env, log_path):
        launches.append(dict(env))
        yield SimpleNamespace(poll=lambda: None)

    async def post(client, url, *, json, timeout):
        prompts.append(json["messages"])
        return next(replies), _payload()

    deps = runner.RunDeps(
        launch=launch,
        wait_for_addr=lambda *args: ("localhost", 1234),
        memory_poller=lambda *args: nullcontext(),
        get=lambda *args, **kwargs: SimpleNamespace(
            status_code=200, json=lambda: {"data": [{"id": "served"}]}
        ),
        post=post,
    )
    monkeypatch.setattr(runner, "RunDeps", lambda: deps)
    monkeypatch.setattr(eval_gsm8k.signal, "signal", lambda *args: None)
    common = [
        "--model",
        str(tmp_path / "model"),
        "--config",
        str(config),
        "--dataset-dir",
        str(tmp_path),
        "--output-dir",
        str(output),
        "--num-questions",
        "2",
    ]
    assert (
        eval_gsm8k.main(
            common + ["--server-env", "HF_TOKEN=fake-old-secret", "--server-env", "MODE=first"]
        )
        == 1
    )
    first = json.loads((output / "run_manifest.json").read_text())
    assert "chat_template_kwargs" not in first["identity"]["request_fields"]
    assert (
        eval_gsm8k.main(
            common
            + [
                "--server-env",
                "HF_TOKEN=fake-old-secret",
                "--server-env",
                "MODE=first",
                "--chat-template-kwargs",
                '{"reasoning_effort":128}',
            ]
        )
        == 2
    )
    # Reordering overrides and rotating their values preserve the resume identity.
    assert (
        eval_gsm8k.main(
            common + ["--server-env", "MODE=second", "--server-env", "HF_TOKEN=fake-new-secret"]
        )
        == 0
    )
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["identity_sha256"] == first["identity_sha256"]
    assert manifest["attempts"][1]["already_completed"] == 1
    assert len(prompts) == 3 and prompts[1] == prompts[2]
    assert [env["HF_TOKEN"] for env in launches] == ["fake-old-secret", "fake-new-secret"]
    assert all(env["EVAL_INHERITED_TOKEN"] == "fake-inherited-secret" for env in launches)
    settings = [manifest["identity"]["server_settings"]] + [
        attempt["server_settings"] for attempt in manifest["attempts"]
    ]
    assert all(item["server_env_names"] == ["HF_TOKEN", "MODE"] for item in settings)
    assert all("server_env" not in item for item in settings)
    captured = capfd.readouterr()
    persisted = "".join(path.read_text() for path in output.iterdir()) + captured.out + captured.err
    for secret in (
        "fake-old-secret",
        "fake-new-secret",
        "fake-inherited-secret",
        "fake-parent-secret",
    ):
        assert secret not in persisted
        assert protocol.sha256_text(secret) not in persisted


@pytest.mark.parametrize("mode", ["endpoint", "unmonitored", "monitored"])
def test_gpu_cost_preserves_unknown_and_labels_monitored_estimate(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(protocol, "load_problems", lambda *args: ([_problem(0)], {}))
    deps = runner.RunDeps(
        get=lambda *args, **kwargs: SimpleNamespace(
            status_code=200, json=lambda: {"data": [{"id": "served"}]}
        ),
        post=AsyncMock(return_value=(200, _payload())),
    )
    managed = (
        None
        if mode == "endpoint"
        else runner.ManagedServer(
            argv=("unused",), env={}, gpu_ids=("0", "1", "0") if mode == "monitored" else ()
        )
    )
    monkeypatch.setattr(runner, "_serve_and_ask", lambda *args: {"server_lifetime_s": 3600})
    output = tmp_path / "results"
    kwargs = dict(
        dataset_dir=tmp_path,
        output_dir=output,
        num_questions=1,
        concurrency=1,
        time_budget_s=60,
        endpoint="http://unused" if managed is None else None,
        managed=managed,
        deps=deps,
        print_report=lambda _: None,
    )
    summary = runner.run(**kwargs)
    assert json.loads((output / "summary.json").read_text()) == summary
    assert summary["timing"]["gpu_hours"] is None
    assert summary["timing"]["server_gpu_hours"] is None
    assert summary["cost"]["gpu_hours_all_attempts"] is None
    assert summary["cost"]["server_gpu_hours_all_attempts"] is None
    expected = 2.0 if mode == "monitored" else None
    assert summary["timing"]["monitored_gpu_hours"] == expected
    assert summary["cost"]["monitored_gpu_hours_all_attempts"] == expected
    if mode == "monitored":
        # A legacy estimate must not be silently treated as zero on resume.
        path = output / "run_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["attempts"][0]["timing"] = {"gpu_hours": 99}
        path.write_text(json.dumps(manifest))
        resumed = runner.run(**kwargs)
        assert resumed["timing"]["monitored_gpu_hours"] == 2.0
        assert resumed["cost"]["monitored_gpu_hours_all_attempts"] is None


@pytest.mark.parametrize("interrupt", [False, True])
def test_managed_run_owns_directory_and_removes_stale_address(tmp_path, monkeypatch, interrupt):
    monkeypatch.setattr(protocol, "load_problems", lambda *args: ([_problem(0)], {}))
    addr = tmp_path / "server-attempt-1.addr"
    addr.write_text("localhost:9999")
    options = dict(
        dataset_dir=tmp_path,
        output_dir=tmp_path,
        num_questions=1,
        concurrency=1,
        time_budget_s=30,
        print_report=lambda text: None,
    )

    @contextmanager
    def launch(argv, env, log_path):
        assert not addr.exists()
        manifest = (tmp_path / "run_manifest.json").read_bytes()
        # An independent process cannot acquire the lock while the server is alive.
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import fcntl, sys
with open(sys.argv[1], 'a') as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(23)
""",
                str(tmp_path / ".lock"),
            ],
            check=False,
        )
        assert child.returncode == 23
        with pytest.raises(runner.EvaluationError, match="already in use"):
            runner.run(**options, endpoint="http://other-server")
        assert (tmp_path / "run_manifest.json").read_bytes() == manifest
        if interrupt:
            raise KeyboardInterrupt
        yield SimpleNamespace(poll=lambda: None)

    deps = runner.RunDeps(
        launch=launch,
        wait_for_addr=lambda *args: ("localhost", 1234),
        memory_poller=lambda *args: nullcontext(),
        get=lambda *args, **kwargs: SimpleNamespace(
            status_code=200, json=lambda: {"data": [{"id": "served"}]}
        ),
        post=AsyncMock(return_value=(200, _payload())),
    )
    managed = runner.ManagedServer(argv=("trtllm-serve", "model"), env={})
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            runner.run(**options, managed=managed, deps=deps)
    else:
        result = runner.run(**options, managed=managed, deps=deps)
        assert result["verdict"]["passed"]
    # Both normal exit and interruption release ownership without deleting the lock inode.
    with runner._output_lock(tmp_path):
        assert (tmp_path / ".lock").is_file()


def test_json_replacement_uses_unique_temporary_files_and_cleans_up(tmp_path, monkeypatch):
    target = tmp_path / "summary.json"
    target.write_text('{"old": true}')
    sources = []
    replace = os.replace

    def record_replace(src, dst):
        sources.append(src)
        assert src.parent == dst.parent
        replace(src, dst)

    monkeypatch.setattr(os, "replace", record_replace)
    runner._write_json(target, {"value": 1})
    runner._write_json(target, {"value": 2})
    assert len(set(sources)) == 2
    assert json.loads(target.read_text()) == {"value": 2}
    with pytest.raises(TypeError):
        runner._write_json(target, object())
    assert json.loads(target.read_text()) == {"value": 2}
    assert list(tmp_path.glob("*.tmp")) == []
