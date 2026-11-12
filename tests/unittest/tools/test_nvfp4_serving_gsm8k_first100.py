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

import dataclasses
import json
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "llm-api"
sys.path.insert(0, str(EXAMPLES_DIR))
from gsm8k_eval import protocol, records  # noqa: E402

# The command line beside the public package (its directory is on sys.path through ``public``).


pytestmark = pytest.mark.cpu_only

# Change only together with a new PROTOCOL_ID and a regenerated manifest.json.
PROTOCOL_DEFINITION_SHA256 = "9a0730c17d2a7c265dea42e15160043251e8bb3fb74060f09b8d1f148622a0be"


# ------------------------------------------------------------------------------------ scorer


@pytest.mark.parametrize(
    "content, parsed",
    [
        ("The answer is 42.", "42"),
        ("-7 degrees", "-7"),
        ("x = -5", "-5"),
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


def test_request_sends_exactly_the_protocol_fields():
    body = protocol.build_request("served", "prompt text")
    assert body == {
        "model": "served",
        "messages": [{"role": "user", "content": "prompt text"}],
        "temperature": 0,
        "top_k": 1,
        "seed": 42,
        "max_tokens": 65536,
        "n": 1,
        "stream": False,
        "chat_template_kwargs": {"reasoning_effort": 128},
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
        protocol.build_request("served", problem.prompt),
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
