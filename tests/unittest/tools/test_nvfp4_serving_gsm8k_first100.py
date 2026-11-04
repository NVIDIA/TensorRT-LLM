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
"""CPU tests of the public GSM8K evaluator and its first-100 CI qualification job.

They cover the public protocol (frozen manifest, prompt, request fields, scorer, count
selection and the pinned test-split inventory), the durable results and their resume rules,
the verdict and the CI regression criterion on synthetic boundary cases, the test-db context and
its TIMEOUT conversion, the explicit trigger (an untriggered run reads no input and starts no
server), existing-endpoint and managed-server runs against fakes, the lifecycle of owned server
processes, the command line and the client's import isolation. No GPU, checkpoint, dataset
download or model server is used.
"""

import ast
import builtins
import contextlib
import dataclasses
import importlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
import yaml
from integration.defs import test_list_parser
from integration.defs.accuracy.gsm8k_first100 import deployment, runner
from integration.defs.accuracy.gsm8k_first100.public import EXAMPLES_DIR, protocol, records, server
from integration.defs.accuracy.gsm8k_first100.public import runner as evaluator

# The command line beside the public package (its directory is on sys.path through ``public``).
eval_gsm8k = importlib.import_module("eval_gsm8k")

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_LISTS = REPO_ROOT / "tests" / "integration" / "test_lists"
CONTEXT = "manual_gb300_nvfp4_serving_gsm8k_first100"
CONTEXT_FILE = TEST_LISTS / "test-db" / f"{CONTEXT}.yml"
DEFS = REPO_ROOT / "tests" / "integration" / "defs"
TEST_FILE = DEFS / "accuracy" / "test_nvfp4_serving_gsm8k.py"
NODE_ID = "accuracy/test_nvfp4_serving_gsm8k.py::test_gsm8k_first100"
# Change only together with a new PROTOCOL_ID and a regenerated manifest.json.
PROTOCOL_DEFINITION_SHA256 = "9a0730c17d2a7c265dea42e15160043251e8bb3fb74060f09b8d1f148622a0be"


def _gpu(index, **overrides):
    gpu = {
        "index": str(index),
        "name": "NVIDIA GB300",
        "uuid": f"GPU-{index}",
        "compute_cap": "10.3",
        "memory.total": "284208",
        "driver_version": "590.00",
    }
    gpu.update(overrides)
    return gpu


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


def _summary(statuses, contents=None):
    problems = [_problem(i) for i in range(len(statuses))]
    state = records.LedgerState()
    for problem, status in zip(problems, statuses):
        if status == "missing":
            continue
        record = _record(problem, status=status, content=(contents or {}).get(problem.index, "4"))
        state.records.append(record)
        state.latest[problem.id] = record
        if status in records.COMPLETED:
            state.completed[problem.id] = record
    return records.summarize(state, problems)


def test_measurement_mode_passes_complete_runs_without_any_threshold():
    summary = _summary([records.OK] * 100, contents={3: "no number", 4: "5"})
    assert (summary["correct"], summary["parse_failures"]) == (98, ["test-0003"])
    result = records.verdict(summary)
    assert result == {
        "mode": "measurement",
        "operational_ok": True,
        "problems": [],
        "threshold": None,
        "passed": True,
    }


@pytest.mark.parametrize("min_correct", [None, 50], ids=["measurement", "regression"])
@pytest.mark.parametrize(
    "status, label",
    [
        ("missing", "without a reply"),
        (records.ERROR, "with an error"),
        (records.TIMEOUT, "timed out"),
        (records.TRUNCATED, "truncated"),
    ],
)
def test_every_incomplete_question_fails_the_run_visibly(status, label, min_correct):
    summary = _summary([records.OK] * 99 + [status])
    result = records.verdict(summary, min_correct)
    assert not result["passed"] and not result["operational_ok"]
    (problem,) = result["problems"]
    assert problem.startswith(f"1 question(s) {label}") and problem.endswith(": test-0099")
    if min_correct is not None:
        # A met score threshold never passes an incomplete run.
        assert result["criterion"]["met"]


# ------------------------------------------------------------------------ regression criterion


def _criterion(tmp_path, min_correct, manifest_sha256="m" * 64, **extra):
    path = tmp_path / "criterion.json"
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "protocol_id": protocol.PROTOCOL_ID,
                "manifest_sha256": manifest_sha256,
                "min_correct": min_correct,
                **extra,
            }
        )
    )
    return path


def test_regression_mode_needs_an_installed_criterion(tmp_path):
    with pytest.raises(deployment.CriterionError, match="none is installed"):
        deployment.load_criterion(tmp_path / "criterion.json", manifest_sha256="m" * 64)
    for min_correct in (-1, 101, 95.0, True):
        with pytest.raises(ValueError, match="min_correct"):
            records.verdict(_summary([records.OK] * 100), min_correct)


@pytest.mark.parametrize(
    "value, match",
    [
        (101, "min_correct"),
        (-1, "min_correct"),
        ("90", "min_correct"),
        (90.0, "min_correct"),
        (True, "min_correct"),
    ],
)
def test_malformed_criterion_is_refused(tmp_path, value, match):
    with pytest.raises(deployment.CriterionError, match=match):
        deployment.load_criterion(_criterion(tmp_path, value), manifest_sha256="m" * 64)


def test_criterion_for_another_protocol_or_manifest_is_refused(tmp_path):
    path = _criterion(tmp_path, 90)
    with pytest.raises(deployment.CriterionError, match="different item manifest"):
        deployment.load_criterion(path, manifest_sha256="x" * 64)
    data = json.loads(path.read_text())
    data["protocol_id"] = "gsm8k-other"
    path.write_text(json.dumps(data))
    with pytest.raises(deployment.CriterionError, match="protocol"):
        deployment.load_criterion(path, manifest_sha256="m" * 64)
    path.write_text("{")
    with pytest.raises(deployment.CriterionError, match="not valid JSON"):
        deployment.load_criterion(path, manifest_sha256="m" * 64)


@pytest.mark.parametrize(
    "min_correct, wrong, passed",
    [(90, 10, True), (90, 11, False), (100, 0, True), (100, 1, False), (0, 100, True)],
)
def test_criterion_boundary(tmp_path, min_correct, wrong, passed):
    criterion = deployment.load_criterion(
        _criterion(tmp_path, min_correct), manifest_sha256="m" * 64
    )
    summary = _summary([records.OK] * 100, contents={i: "7" for i in range(wrong)})
    result = records.verdict(summary, criterion["min_correct"])
    assert result["mode"] == "regression"
    assert result["criterion"] == {
        "min_correct": min_correct,
        "correct": 100 - wrong,
        "met": passed,
    }
    assert result["passed"] is passed


def test_criterion_names_the_identity_it_was_measured_with(tmp_path):
    criterion = deployment.load_criterion(
        _criterion(tmp_path, 90, required_identity={"tensorrt_llm": "1.0", "kv_scales": {"a": 1}}),
        manifest_sha256="m" * 64,
    )
    identity = {"tensorrt_llm": "1.0", "kv_scales": {"a": 2}}
    assert deployment.criterion_mismatches(criterion, identity) == ["kv_scales"]


# ------------------------------------------------------------------------- timeout and test-db


@pytest.mark.parametrize(
    "total_s, minutes",
    [
        (100, 12),
        (2999, 60),
        (3000, 60),
        (3000.5, 61),
        (3001, 61),
        (6000, 120),
        (6001, 121),
        (7200, 144),
    ],
)
def test_timeout_minutes_formula(total_s, minutes):
    assert deployment.timeout_minutes(total_s) == minutes


@pytest.mark.parametrize("total_s", [0, -1.0])
def test_timeout_minutes_needs_a_measured_total(total_s):
    with pytest.raises(ValueError):
        deployment.timeout_minutes(total_s)


def test_manual_context_holds_only_this_node_with_four_gb300_conditions():
    data = yaml.safe_load(CONTEXT_FILE.read_text())
    assert set(data) == {"version", CONTEXT}
    (entry,) = data[CONTEXT]
    assert entry["condition"] == {
        "ranges": {"system_gpu_count": {"gte": 4, "lte": 4}},
        "wildcards": {"gpu": ["*gb300*"], "linux_distribution_name": "ubuntu*", "cpu": "aarch64"},
        "terms": {"stage": "manual", "backend": "pytorch"},
    }
    (line,) = entry["tests"]
    names, markers = test_list_parser.parse_test_list_lines(str(CONTEXT_FILE), [line], None)
    assert names == [NODE_ID]
    minutes = int(line.rsplit("(", 1)[1].rstrip(")"))
    assert markers[NODE_ID] == ("TIMEOUT", None, minutes * 60)
    assert minutes == 45


def test_only_qa_and_manual_lists_select_the_first100_job():
    hits = [
        str(path.relative_to(TEST_LISTS))
        for path in sorted(TEST_LISTS.rglob("*"))
        if path.is_file()
        and path != CONTEXT_FILE
        and path.suffix in (".yml", ".yaml", ".txt")
        and ("test_nvfp4_serving_gsm8k" in path.read_text() or "gsm8k_first100" in path.read_text())
    ]
    assert hits == ["qa/llm_function_core.txt"]
    qa_list = (TEST_LISTS / hits[0]).read_text()
    assert f"{NODE_ID} TIMEOUT (45)" in qa_list


def test_installed_regression_criterion_matches_the_qualified_protocol():
    criterion = deployment.load_criterion(
        deployment.CRITERION_PATH, manifest_sha256=protocol.sha256_file(protocol.MANIFEST_PATH)
    )
    assert criterion["min_correct"] == 95
    # The manifest moved to the public evaluator unchanged: the criterion still binds it.
    assert criterion["manifest_sha256"] == (
        "d36d25d42c8c9d02f9e927e82efa705523f30106fb90523067c2cc22c9b946ad"
    )


# ------------------------------------------------------------------------ explicit trigger


@pytest.mark.parametrize("value", ["true", "yes", " 1", "2"])
def test_ambiguous_trigger_values_are_rejected(value):
    with pytest.raises(ValueError, match=deployment.GUARD_ENV):
        deployment.explicit_trigger({deployment.GUARD_ENV: value})


def _forbidden(name, calls):
    def call(*args, **kwargs):
        calls.append(name)
        raise AssertionError(f"an untriggered run used {name}")

    return call


def _forbidden_deps(calls):
    return runner.RunDeps(
        **{
            field.name: _forbidden(field.name, calls)
            for field in dataclasses.fields(runner.RunDeps)
        }
    )


@pytest.mark.parametrize("value", [None, "", "0"])
def test_untriggered_run_reads_no_input_and_starts_no_server(monkeypatch, value):
    calls = []
    deps = _forbidden_deps(calls)
    environ = {
        deployment.TARGET_ENV: "target",
        deployment.DRAFTER_ENV: "drafter",
        deployment.KV_SCALES_ENV: "kv.json",
    }
    if value is not None:
        environ[deployment.GUARD_ENV] = value
    skipped = None
    with monkeypatch.context() as patch:
        for module, name in (
            (builtins, "open"),
            (os, "open"),
            (os, "stat"),
            (os, "lstat"),
            (os, "listdir"),
            (os, "scandir"),
            (subprocess, "Popen"),
            (subprocess, "run"),
            (socket, "socket"),
        ):
            patch.setattr(module, name, _forbidden(f"{module.__name__}.{name}", calls))
        runner.require_hardware_when_enabled(environ, deps)
        try:
            runner.run_first100(
                environ=environ,
                models_root=_forbidden("models_root", calls),
                repo_root=_forbidden("repo_root", calls),
                timeout_s=7200,
                output_dir="unused",
                deps=deps,
            )
        except pytest.skip.Exception as skip:
            skipped = str(skip)
    assert calls == []
    assert skipped is not None and deployment.GUARD_ENV in skipped


def _guarded_deps(calls):
    deps = _forbidden_deps(calls)
    deps.monotonic = time.monotonic
    return deps


@pytest.mark.parametrize(
    "environ, timeout_s, match",
    [
        ({deployment.GUARD_ENV: "yes"}, 7200, "must be 1"),
        ({deployment.GUARD_ENV: "1"}, None, "finite pytest timeout"),
        ({deployment.GUARD_ENV: "1"}, 300, "finite pytest timeout"),
        ({deployment.GUARD_ENV: "1", deployment.MODE_ENV: "audit"}, 7200, "measurement or"),
        ({deployment.GUARD_ENV: "1", deployment.MODE_ENV: "regression"}, 7200, "refused"),
    ],
)
def test_triggered_run_fails_before_any_input_without_its_prerequisites(
    tmp_path, environ, timeout_s, match
):
    calls = []
    with pytest.raises(pytest.fail.Exception, match=match):
        runner.run_first100(
            environ=environ,
            models_root=_forbidden("models_root", calls),
            repo_root=_forbidden("repo_root", calls),
            timeout_s=timeout_s,
            output_dir=str(tmp_path),
            deps=_guarded_deps(calls),
            criterion_path=tmp_path / "criterion.json",
        )
    assert calls == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "gpus, match",
    [
        ([_gpu(i) for i in range(2)], "2 GPU"),
        ([_gpu(i, name="NVIDIA GB200") for i in range(4)], "not a GB300"),
        ([_gpu(i, compute_cap="10.0") for i in range(4)], "SM103"),
    ],
)
def test_triggered_run_fails_instead_of_skipping_on_other_hardware(gpus, match):
    deps = runner.RunDeps(query_gpus=lambda environ: gpus)
    with pytest.raises(pytest.fail.Exception, match=match):
        runner.require_hardware_when_enabled({deployment.GUARD_ENV: "1"}, deps)
    runner.require_hardware_when_enabled(
        {deployment.GUARD_ENV: "1"},
        runner.RunDeps(query_gpus=lambda e: [_gpu(i) for i in range(4)]),
    )


def test_gpu_query_follows_cuda_visible_devices():
    output = "".join(f"{i}, NVIDIA GB300, GPU-{i}, 10.3, 284208, 590.00\n" for i in range(4))

    def run(argv, **kwargs):
        assert argv[0] == "nvidia-smi"
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    gpus = deployment.query_gpus({"CUDA_VISIBLE_DEVICES": "2,GPU-0"}, run=run)
    assert [gpu["index"] for gpu in gpus] == ["2", "0"]
    assert deployment.query_gpus({}, run=run)[3] == _gpu(3)
    with pytest.raises(RuntimeError, match="names no GPU"):
        deployment.query_gpus({"CUDA_VISIBLE_DEVICES": "7"}, run=run)


def test_ci_entry_enables_regression_and_preserves_input_failures(monkeypatch):
    tree = ast.parse(TEST_FILE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    node.decorator_list = []
    calls = []
    fake_runner = SimpleNamespace(
        require_hardware_when_enabled=lambda env: calls.append(("hardware", env)),
        run_first100=lambda **kw: calls.append(("run", kw)),
    )
    namespace = dict(
        os=os,
        pytest=pytest,
        runner=fake_runner,
        get_device_count=lambda: 4,
        llm_models_root=lambda: "models",
        get_llm_root=lambda: "repo",
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(TEST_FILE), "exec"), namespace)
    monkeypatch.setenv("TRTLLM_RUN_NVFP4_SERVING_GSM8K100", "0")
    monkeypatch.setenv("TRTLLM_NVFP4_SERVING_GSM8K100_MODE", "measurement")
    request = SimpleNamespace(config=SimpleNamespace(getoption=lambda _: "results"))
    namespace[node.name](request, 2700, None)
    assert calls[0][0] == "hardware"
    assert calls[1][0] == "run"
    env = calls[1][1]["environ"]
    assert env["TRTLLM_RUN_NVFP4_SERVING_GSM8K100"] == "1"
    assert env["TRTLLM_NVFP4_SERVING_GSM8K100_MODE"] == "regression"
    assert calls[1][1]["timeout_s"] == 2700
    assert os.environ["TRTLLM_RUN_NVFP4_SERVING_GSM8K100"] == "0"
    namespace["get_device_count"] = lambda: 8
    with pytest.raises(pytest.skip.Exception, match="exactly four"):
        namespace[node.name](request, 2700, None)
    assert len(calls) == 2


# ---------------------------------------------------------------------- staged inputs, recipe


def _checkpoint(directory, *, layers, shards):
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps({"num_hidden_layers": layers}))
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        (directory / name).write_text(f"{name}\n")
    if shards:
        weight_map = {f"w{i}": f"model-{i:05d}.safetensors" for i in range(shards)}
        index = directory / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": weight_map}))
        for name in weight_map.values():
            (directory / name).write_bytes(b"\0" * 16)
    else:
        (directory / "model.safetensors").write_bytes(b"\0" * 8)


def _scales(path, target=2, draft=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    table = {"version": 1}
    for section, layers in (("target", target), ("draft", draft)):
        if layers:
            table[section] = {"layers": layers, "scales": {str(i): 0.5 + i for i in range(layers)}}
    path.write_text(json.dumps(table))
    return path


def _recipe(**overrides):
    recipe = {
        "tensor_parallel_size": 4,
        "max_batch_size": 4,
        "max_seq_len": 69632,
        "max_num_tokens": 8192,
        "reasoning_parser": "beam_harmony_text",
        "kv_cache_config": {"dtype": "fp8", "max_tokens": 139264},
        "speculative_config": {
            "decoding_type": "DFlash",
            "max_draft_len": 7,
            "speculative_model": deployment.DRAFTER_PLACEHOLDER,
        },
    }
    recipe.update(overrides)
    return recipe


def test_staged_inputs_are_resolved_below_the_models_root(tmp_path):
    for name in ("target", "drafter", deployment.DATASET_RELATIVE_DIR):
        (tmp_path / name).mkdir(parents=True)
    _scales(tmp_path / "scales" / "kv.json")
    environ = {
        deployment.TARGET_ENV: "target",
        deployment.DRAFTER_ENV: "drafter",
        deployment.KV_SCALES_ENV: "scales/kv.json",
    }
    inputs = deployment.resolve_inputs(environ, str(tmp_path))
    assert inputs.target == str(tmp_path / "target")
    assert inputs.dataset_dir == str(tmp_path / "datasets" / "openai" / "gsm8k")
    for value, match in (("", "must name"), ("/abs/target", "below"), ("../target", "below")):
        with pytest.raises(ValueError, match=match):
            deployment.resolve_inputs({**environ, deployment.TARGET_ENV: value}, str(tmp_path))
    with pytest.raises(FileNotFoundError, match="missing"):
        deployment.resolve_inputs({**environ, deployment.DRAFTER_ENV: "missing"}, str(tmp_path))


def test_checkpoints_are_digested_by_metadata_and_weight_inventory(tmp_path):
    _checkpoint(tmp_path / "target", layers=2, shards=2)
    _checkpoint(tmp_path / "drafter", layers=1, shards=0)
    target = deployment.describe_checkpoint(str(tmp_path / "target"), deployment.TARGET_FILES)
    assert set(target["files"]) == set(deployment.TARGET_FILES)
    assert (target["weights"]["count"], target["num_hidden_layers"]) == (2, 2)
    drafter = deployment.describe_checkpoint(str(tmp_path / "drafter"), deployment.DRAFTER_FILES)
    assert drafter["weights"]["count"] == 1
    (tmp_path / "target" / "model-00001.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="model-00001"):
        deployment.describe_checkpoint(str(tmp_path / "target"), deployment.TARGET_FILES)
    (tmp_path / "drafter" / "config.json").unlink()
    with pytest.raises(FileNotFoundError, match="config.json"):
        deployment.describe_checkpoint(str(tmp_path / "drafter"), deployment.DRAFTER_FILES)


def test_scale_table_sections_must_match_both_checkpoints(tmp_path):
    path = _scales(tmp_path / "kv.json")
    digests = deployment.describe_kv_scales(str(path), 2, 1)
    assert digests["sha256"] == protocol.sha256_file(path)
    assert digests["target_sha256"] != digests["draft_sha256"]
    with pytest.raises(ValueError, match="'draft' section"):
        deployment.describe_kv_scales(str(_scales(tmp_path / "t.json", draft=0)), 2, 1)
    with pytest.raises(ValueError, match="checkpoint has 52"):
        deployment.describe_kv_scales(str(path), 52, 1)
    for bad in (0, -1.0, float("inf"), True, "1"):
        table = json.loads(path.read_text())
        table["target"]["scales"]["1"] = bad
        path.write_text(json.dumps(table))
        with pytest.raises(ValueError, match="layer 1"):
            deployment.describe_kv_scales(str(path), 2, 1)


def test_recipe_is_materialized_with_only_the_drafter_replaced(tmp_path):
    template = tmp_path / "recipe.yaml"
    template.write_text(yaml.safe_dump(_recipe()))
    text, info = deployment.materialize_recipe(str(template), "/staged/drafter")
    config = yaml.safe_load(text)
    assert config == _recipe(
        speculative_config={
            "decoding_type": "DFlash",
            "max_draft_len": 7,
            "speculative_model": "/staged/drafter",
        }
    )
    assert (info["max_batch_size"], info["max_seq_len"]) == (4, 69632)
    assert info["template_sha256"] == protocol.sha256_file(template)
    assert "speculative_model" not in info["settings"]["speculative_config"]


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"speculative_config": {"speculative_model": "/fixed"}}, "speculative_model"),
        ({"tokenizer": "<TARGET_CHECKPOINT>"}, "unresolved placeholders"),
        ({"max_batch_size": 0}, "max_batch_size"),
        ({"max_batch_size": None}, "max_batch_size"),
        ({"max_seq_len": 65536}, "max_seq_len"),
        ({"reasoning_parser": None}, "reasoning_parser"),
        ({"kv_cache_config": {"dtype": "auto"}}, "FP8 KV cache"),
    ],
)
def test_recipe_outside_the_qualified_deployment_is_refused(tmp_path, overrides, match):
    template = tmp_path / "recipe.yaml"
    template.write_text(yaml.safe_dump(_recipe(**overrides)))
    with pytest.raises(ValueError, match=match):
        deployment.materialize_recipe(str(template), "/staged/drafter")
    template.write_text("a: [unclosed")
    with pytest.raises(ValueError, match="not valid YAML"):
        deployment.materialize_recipe(str(template), "/staged/drafter")


#: The decode CUDA-graph keys of the graph-and-overlap recipes, up to their max_batch_size.
GO_GRAPH_CONFIG = {"batch_sizes": [1, 2, 4, 8, 16, 32, 64], "enable_padding": True}


@pytest.mark.parametrize(
    "go, eager",
    [
        (deployment.RECIPE_RELATIVE_PATH, deployment.EAGER_RECIPE_RELATIVE_PATH),
        (
            "examples/configs/curated/beam-nvfp4-dflash2-fp8kv-disagg-gen-go.yaml",
            "examples/configs/curated/beam-nvfp4-dflash2-fp8kv-disagg-gen.yaml",
        ),
    ],
)
def test_go_recipes_differ_from_the_eager_recipes_only_in_graphs_and_overlap(go, eager):
    go_text, go_info = deployment.materialize_recipe(str(REPO_ROOT / go), "/staged/drafter")
    eager_text, _ = deployment.materialize_recipe(str(REPO_ROOT / eager), "/staged/drafter")
    go_config, eager_config = yaml.safe_load(go_text), yaml.safe_load(eager_text)
    assert go_config.pop("cuda_graph_config") == GO_GRAPH_CONFIG
    assert go_config.pop("disable_overlap_scheduler") is False
    assert eager_config.pop("cuda_graph_config") is None
    assert eager_config.pop("disable_overlap_scheduler") is True
    assert go_config == eager_config
    # Every generation batch the server can schedule pads to a captured key.
    assert max(GO_GRAPH_CONFIG["batch_sizes"]) == go_info["max_batch_size"]


def test_first100_job_serves_the_go_recipe_and_keeps_the_eager_recipe():
    assert deployment.RECIPE_RELATIVE_PATH == (
        "examples/configs/curated/beam-nvfp4-dflash2-fp8kv-aggregated-go.yaml"
    )
    assert deployment.EAGER_RECIPE_RELATIVE_PATH == (
        "examples/configs/curated/beam-nvfp4-dflash2-fp8kv-aggregated.yaml"
    )
    # The CI runner materializes the selected recipe, never the eager one.
    source = (DEFS / "accuracy" / "gsm8k_first100" / "runner.py").read_text()
    assert "deployment.RECIPE_RELATIVE_PATH" in source
    assert "EAGER_RECIPE_RELATIVE_PATH" not in source


def test_server_launch_command_and_environment():
    argv = server.serve_argv(deployment.serve_command("/t", "/run/recipe.yaml"), "/run/server.addr")
    assert argv == [
        "trtllm-serve",
        "/t",
        "--config",
        "/run/recipe.yaml",
        "--chat_template",
        "/t/chat_template.jinja",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--report_addr",
        "/run/server.addr",
    ]
    env = deployment.server_environment(
        {"PATH": "/bin", "TRTLLM_SKIP_KV_CACHE_ESTIMATION": "0", "HF_HUB_OFFLINE": "0"}, "/kv.json"
    )
    assert env["PATH"] == "/bin" and env["HF_HUB_OFFLINE"] == "0"
    assert env["TRTLLM_SKIP_KV_CACHE_ESTIMATION"] == "1"
    assert env["TRTLLM_MODELING_V2_KV_CACHE_SCALES"] == "/kv.json"


def test_gpu_memory_peaks_and_gpu_hours():
    csv_text = (
        "2026/10/28 10:00:00.000, 0, GPU-0, 1000, 284208\n"
        "2026/10/28 10:00:00.000, 1, GPU-1, 900, 284208\n"
        "2026/10/28 10:00:01.000, 0, GPU-0, 150000, 284208\n"
        "2026/10/28 10:00:01.000, 1, GPU-1, [N/A], 284208\n"
    )
    peaks = records.gpu_memory_peaks(csv_text)
    assert peaks["samples"] == 3 and peaks["max_peak_used_mib"] == 150000
    assert peaks["per_gpu"]["1"]["peak_used_mib"] == 900
    assert records.gpu_hours(4, 5400) == 6.0


# ------------------------------------------------------------- complete runs on a fake server


class _FakeServer:
    def __init__(self, exit_code=None):
        self.code = exit_code
        self.terminated = False

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated = True
        if self.code is None:
            self.code = 0


def _reply(content, finish="stop", completion=300, reasoning="step by step"):
    text = _payload(content=content, finish=finish, completion=completion, reasoning=reasoning)
    return SimpleNamespace(status_code=200, text=text)


class _FakeDeployment:
    """Fake process, endpoints, GPUs and clock; ``model`` answers ``answer(gold, row)``."""

    def __init__(
        self,
        answer=None,
        exit_code=None,
        clock_step=0.0,
        base="http://127.0.0.1:8000",
        model="served",
    ):
        self.answer = answer or (lambda gold, row: _reply(f"So the answer is {gold}."))
        self.exit_code = exit_code
        self.clock_step = clock_step
        self.base = base
        self.model = model
        self.clock = 0.0
        self.gets = []
        self.asked = []
        self.bodies = []
        self.launches = []
        self.servers = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.lock = threading.Lock()

    def deps(self):
        return runner.RunDeps(
            launch=self.launch,
            wait_for_addr=lambda path, timeout, process: ("127.0.0.1", 8000),
            get=self.get,
            post=self.post,
            query_gpus=lambda environ: [_gpu(i) for i in range(4)],
            memory_poller=self.memory_poller,
            monotonic=lambda: time.monotonic() + self.clock,
            sleep=lambda seconds: None,
        )

    @contextlib.contextmanager
    def launch(self, argv, env, log_path):
        """Like server.launch: the server is stopped however the context is left."""
        fake = _FakeServer(self.exit_code)
        self.launches.append((argv, dict(env)))
        self.servers.append(fake)
        Path(log_path).write_text("fake server\n")
        try:
            yield fake
        finally:
            fake.terminate()

    def get(self, url, timeout):
        self.gets.append(url)
        if url.endswith("/health"):
            return SimpleNamespace(status_code=200)
        return SimpleNamespace(status_code=200, json=lambda: {"data": [{"id": self.model}]})

    def post(self, url, json, timeout):
        assert url == f"{self.base}/v1/chat/completions"
        assert json["model"] == self.model and json["max_tokens"] == 65536
        self.bodies.append(json)
        question = json["messages"][0]["content"].rsplit("Question: ", 1)[1]
        row = int(question.split(":")[0].split()[1])
        with self.lock:
            self.asked.append(row)
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            time.sleep(0.002)
            return self.answer(str(row + 3), row)
        finally:
            with self.lock:
                self.in_flight -= 1
                self.clock += self.clock_step

    @contextlib.contextmanager
    def memory_poller(self, path, gpu_ids):
        Path(path).write_text(
            "".join(
                f"2026/10/28 10:00:00.000, {i}, GPU-{i}, {1000 + int(i)}, 284208\n" for i in gpu_ids
            )
        )
        yield None


def _stage(tmp_path, pin, **recipe):
    root = tmp_path / "models"
    train, test = _rows(15, "train"), _rows(105, "test")
    digests = _stage_dataset(root / deployment.DATASET_RELATIVE_DIR, train, test)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_synthetic_manifest(train, test, digests)))
    pin(_synthetic_manifest(train, test, digests), train, test)
    _checkpoint(root / "target", layers=2, shards=2)
    _checkpoint(root / "drafter", layers=1, shards=0)
    _scales(root / "scales" / "kv.json")
    repo = tmp_path / "repo"
    (repo / deployment.RECIPE_RELATIVE_PATH).parent.mkdir(parents=True)
    (repo / deployment.RECIPE_RELATIVE_PATH).write_text(yaml.safe_dump(_recipe(**recipe)))
    environ = {
        deployment.GUARD_ENV: "1",
        deployment.TARGET_ENV: "target",
        deployment.DRAFTER_ENV: "drafter",
        deployment.KV_SCALES_ENV: "scales/kv.json",
    }
    return SimpleNamespace(root=root, repo=repo, manifest=manifest, environ=environ)


def _run(stage, fake, out, *, timeout_s=7200.0, criterion=None, mode=None):
    environ = dict(stage.environ)
    if mode:
        environ[deployment.MODE_ENV] = mode
    return runner.run_first100(
        environ=environ,
        models_root=lambda: str(stage.root),
        repo_root=lambda: str(stage.repo),
        timeout_s=timeout_s,
        output_dir=str(out),
        deps=fake.deps(),
        manifest_path=stage.manifest,
        criterion_path=criterion or out.parent / "no-criterion.json",
    )


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_complete_run_records_every_question_and_reports_its_cost(tmp_path, pin_split):
    stage = _stage(tmp_path, pin_split)
    fake = _FakeDeployment()
    summary = _run(stage, fake, tmp_path / "out")
    results = tmp_path / "out" / runner.RESULTS_DIRNAME
    assert (summary["correct"], summary["total"], summary["distinct_completed"]) == (100, 100, 100)
    assert summary["verdict"] == {
        "mode": "measurement",
        "operational_ok": True,
        "problems": [],
        "threshold": None,
        "passed": True,
    }
    assert sorted(fake.asked) == list(range(100)) and 1 < fake.max_in_flight <= 4
    lines = _lines(results / "results.jsonl")
    assert sorted(r["index"] for r in lines) == list(range(100))
    assert {r["run_identity"] for r in lines} == {summary["run_identity"]}
    assert lines[0]["request"]["chat_template_kwargs"] == {"reasoning_effort": 128}
    assert json.loads((results / "summary.json").read_text())["correct"] == 100

    ((argv, env),) = fake.launches
    config = results / "recipe.yaml"
    assert argv == server.serve_argv(
        deployment.serve_command(str(stage.root / "target"), str(config)),
        results / "server-attempt-1.addr",
    )
    assert yaml.safe_load(config.read_text())["speculative_config"]["speculative_model"] == str(
        stage.root / "drafter"
    )
    assert env["TRTLLM_SKIP_KV_CACHE_ESTIMATION"] == "1"
    assert env["TRTLLM_MODELING_V2_KV_CACHE_SCALES"] == str(stage.root / "scales" / "kv.json")
    (served,) = fake.servers
    assert served.terminated

    timing = summary["timing"]
    for key in ("verification_s", "startup_s", "evaluation_s", "teardown_s", "total_s"):
        assert timing[key] >= 0
    assert timing["gpu_memory"]["max_peak_used_mib"] == 1003
    assert timing["gpu_hours"] == records.gpu_hours(4, timing["total_s"])
    run_manifest = json.loads((results / "run_manifest.json").read_text())
    assert [a["attempt"] for a in run_manifest["attempts"]] == [1]
    assert run_manifest["attempts"][0]["staged_inputs"][deployment.TARGET_ENV] == "target"


def _flaky(gold, row):
    if row == 3:
        return SimpleNamespace(status_code=500, text="internal error")
    if row == 4:
        raise requests.exceptions.ReadTimeout("read timed out")
    if row == 5:
        return _reply(None, finish="length", completion=65536)
    if row == 6:
        return _reply(None, reasoning=f"the answer is {gold}")
    if row == 7:
        return _reply("So the answer is 1.")
    return _reply(f"So the answer is {gold}.")


def test_failures_are_visible_and_resume_asks_only_incomplete_questions(tmp_path, pin_split):
    stage = _stage(tmp_path, pin_split)
    with pytest.raises(pytest.fail.Exception, match="with an error: test-0003"):
        _run(stage, _FakeDeployment(answer=_flaky), tmp_path / "out")
    results = tmp_path / "out" / runner.RESULTS_DIRNAME
    first = json.loads((results / "summary.json").read_text())
    assert (first["errors"], first["timeouts"], first["truncated"]) == (
        ["test-0003"],
        ["test-0004"],
        ["test-0005"],
    )
    assert first["parse_failures"] == ["test-0005", "test-0006"]
    assert first["correct"] == 95

    retry = _FakeDeployment()
    with pytest.raises(pytest.fail.Exception, match="truncated at the token allowance: test-0005"):
        _run(stage, retry, tmp_path / "out")
    assert sorted(retry.asked) == [3, 4]
    second = json.loads((results / "summary.json").read_text())
    assert (second["errors"], second["timeouts"], second["correct"]) == ([], [], 97)
    assert len(_lines(results / "results.jsonl")) == 102
    attempts = json.loads((results / "run_manifest.json").read_text())["attempts"]
    assert [(a["attempt"], a["pending"]) for a in attempts] == [(1, 100), (2, 2)]


@pytest.mark.parametrize("wrong, passed", [(2, True), (3, False)])
def test_regression_mode_applies_the_installed_criterion(tmp_path, wrong, passed, pin_split):
    stage = _stage(tmp_path, pin_split)
    criterion = _criterion(tmp_path, 98, manifest_sha256=protocol.sha256_file(stage.manifest))
    fake = _FakeDeployment(
        answer=lambda gold, row: _reply("So the answer is 1." if row < wrong else f"It is {gold}.")
    )
    if passed:
        summary = _run(stage, fake, tmp_path / "out", criterion=criterion, mode="regression")
        assert summary["verdict"]["criterion"] == {"min_correct": 98, "correct": 98, "met": True}
    else:
        with pytest.raises(pytest.fail.Exception, match="below the required 98"):
            _run(stage, fake, tmp_path / "out", criterion=criterion, mode="regression")


def test_startup_failure_fails_without_asking(tmp_path, pin_split):
    stage = _stage(tmp_path, pin_split)
    fake = _FakeDeployment(exit_code=1)
    with pytest.raises(pytest.fail.Exception, match="server startup failed"):
        _run(stage, fake, tmp_path / "out")
    assert fake.asked == []
    summary = json.loads((tmp_path / "out" / runner.RESULTS_DIRNAME / "summary.json").read_text())
    assert len(summary["missing"]) == 100


def test_run_deadline_stops_asking_and_reports_the_rest_missing(tmp_path, pin_split):
    stage = _stage(tmp_path, pin_split, max_batch_size=1)
    fake = _FakeDeployment(clock_step=1000.0)
    with pytest.raises(pytest.fail.Exception, match="99 question"):
        _run(stage, fake, tmp_path / "out", timeout_s=1000.0)
    assert fake.asked == [0]


def test_results_of_another_run_identity_are_never_mixed(tmp_path, pin_split):
    stage = _stage(tmp_path, pin_split)
    _run(stage, _FakeDeployment(), tmp_path / "out")
    recipe = stage.repo / deployment.RECIPE_RELATIVE_PATH
    recipe.write_text(yaml.safe_dump(_recipe(max_num_tokens=4096)))
    fake = _FakeDeployment()
    with pytest.raises(pytest.fail.Exception, match="holds a run with identity"):
        _run(stage, fake, tmp_path / "out")
    assert fake.launches == []


def _snapshot(directory, *names):
    return {name: (directory / name).read_bytes() for name in names}


def test_resume_refuses_a_managed_server_serving_another_model(tmp_path, pin_split):
    stage = _stage(tmp_path, pin_split)
    with pytest.raises(pytest.fail.Exception, match="with an error: test-0003"):
        _run(stage, _FakeDeployment(answer=_flaky), tmp_path / "out")
    results = tmp_path / "out" / runner.RESULTS_DIRNAME
    kept = _snapshot(results, "results.jsonl", "summary.json")
    renamed = _FakeDeployment(model="renamed")
    with pytest.raises(pytest.fail.Exception, match="replies of model 'served', but 'renamed'"):
        _run(stage, renamed, tmp_path / "out")
    (stopped,) = renamed.servers
    assert stopped.terminated and renamed.bodies == []
    assert _snapshot(results, "results.jsonl", "summary.json") == kept
    refused = json.loads((results / "run_manifest.json").read_text())["attempts"][-1]
    assert refused["served_model"] == "renamed" and refused["timing"]["server_exit_code"] == 0


# ------------------------------------------------------ existing endpoint: same selection/scoring


def _endpoint_run(stage, fake, out, *, count=100, served_model=None, concurrency=4):
    deps = fake.deps()
    deps.launch = _forbidden("launch", [])
    deps.memory_poller = _forbidden("memory_poller", [])
    return evaluator.run(
        dataset_dir=stage.root / deployment.DATASET_RELATIVE_DIR,
        output_dir=out,
        num_questions=count,
        concurrency=concurrency,
        time_budget_s=3600.0,
        endpoint=fake.base + "/",
        served_model=served_model,
        manifest_path=stage.manifest,
        deps=deps,
        print_report=lambda text: None,
    )


def _scored(path):
    return sorted(
        (
            r["id"],
            r["prompt_sha256"],
            json.dumps(r["request"], sort_keys=True),
            r["parsed_answer"],
            r["parse_status"],
            r["correct"],
            r["status"],
        )
        for r in _lines(path)
    )


def test_endpoint_and_ci_runs_select_request_and_score_identically(tmp_path, pin_split):
    stage = _stage(tmp_path, pin_split)
    managed = _FakeDeployment(answer=_flaky)
    with pytest.raises(pytest.fail.Exception):
        _run(stage, managed, tmp_path / "ci")
    external = _FakeDeployment(answer=_flaky, base="http://endpoint.example:9000")
    summary = _endpoint_run(stage, external, tmp_path / "endpoint")
    assert external.launches == []
    ci_results = tmp_path / "ci" / runner.RESULTS_DIRNAME / "results.jsonl"
    assert _scored(ci_results) == _scored(tmp_path / "endpoint" / "results.jsonl")
    assert sorted(b["messages"][0]["content"] for b in managed.bodies) == sorted(
        b["messages"][0]["content"] for b in external.bodies
    )
    ci = json.loads((ci_results.parent / "summary.json").read_text())
    for key in ("correct", "parse_failures", "errors", "timeouts", "truncated", "missing"):
        assert summary[key] == ci[key]
    assert summary["transport"] == "endpoint" and ci["transport"] == "managed"


def test_endpoint_mode_resolves_or_refuses_the_served_model(tmp_path, pin_split):
    stage = _stage(tmp_path, pin_split)
    fake = _FakeDeployment(base="http://endpoint.example:9000")
    summary = _endpoint_run(stage, fake, tmp_path / "named", count=3, served_model="served")
    assert summary["verdict"]["passed"] and summary["timing"]["served_model"] == "served"
    assert summary["timing"]["served_model_listed"] is True
    refused = _endpoint_run(stage, fake, tmp_path / "other", count=3, served_model="other-model")
    assert not refused["verdict"]["passed"] and len(refused["missing"]) == 3
    assert refused["verdict"]["problems"][0].startswith("the endpoint is not usable")
    assert len(fake.bodies) == 3


class _UnlistedDeployment(_FakeDeployment):
    """An endpoint without ``/v1/models``, like the router of ``trtllm-serve disaggregated``."""

    def get(self, url, timeout):
        if url.endswith("/health"):
            return SimpleNamespace(status_code=200)
        return SimpleNamespace(status_code=404)


def test_endpoint_without_model_listing_asks_the_named_model(tmp_path, pin_split):
    stage = _stage(tmp_path, pin_split)
    fake = _UnlistedDeployment(base="http://router.example:8000")
    summary = _endpoint_run(stage, fake, tmp_path / "named", count=3, served_model="served")
    assert summary["verdict"]["passed"] and summary["timing"]["served_model"] == "served"
    assert summary["timing"]["served_model_listed"] is False
    refused = _endpoint_run(stage, fake, tmp_path / "unnamed", count=3)
    assert not refused["verdict"]["passed"] and len(refused["missing"]) == 3
    assert "name the model with --served-model-name" in refused["verdict"]["problems"][0]
    assert len(fake.bodies) == 3


def _busy_second(gold, row):
    if row == 1:
        return SimpleNamespace(status_code=503, text="busy")
    return _reply(f"So the answer is {gold}.")


def test_resume_asks_only_the_model_of_the_recorded_replies(tmp_path, pin_split):
    stage = _stage(tmp_path, pin_split)
    out = tmp_path / "out"
    first = _endpoint_run(stage, _FakeDeployment(answer=_flaky, model="model-a"), out, count=8)
    assert (first["served_model"], first["errors"], first["timeouts"]) == (
        "model-a",
        ["test-0003"],
        ["test-0004"],
    )
    kept = _snapshot(out, "results.jsonl", "summary.json")
    changed = _FakeDeployment(model="model-b")
    with pytest.raises(
        evaluator.EvaluationError, match="replies of model 'model-a', but 'model-b'"
    ):
        _endpoint_run(stage, changed, out, count=8)
    assert changed.bodies == [] and _snapshot(out, "results.jsonl", "summary.json") == kept
    refused = json.loads((out / "run_manifest.json").read_text())["attempts"][-1]
    assert refused["served_model"] == "model-b" and "'model-a'" in refused["timing"]["refused"]

    same = _FakeDeployment(model="model-a")
    resumed = _endpoint_run(stage, same, out, count=8)
    assert sorted(same.asked) == [3, 4]
    assert (resumed["served_model"], resumed["distinct_completed"], resumed["attempt"]) == (
        "model-a",
        8,
        3,
    )
    assert {r["request"]["model"] for r in _lines(out / "results.jsonl")} == {"model-a"}


def test_completed_run_is_reported_without_contacting_the_endpoint(tmp_path, pin_split):
    stage = _stage(tmp_path, pin_split)
    out = tmp_path / "out"
    _endpoint_run(stage, _FakeDeployment(model="model-a"), out, count=3)
    kept = (out / "results.jsonl").read_bytes()
    changed = _FakeDeployment(model="model-b")
    summary = _endpoint_run(stage, changed, out, count=3)
    assert changed.gets == [] and changed.bodies == []
    assert (out / "results.jsonl").read_bytes() == kept
    assert (summary["served_model"], summary["correct"], summary["verdict"]["passed"]) == (
        "model-a",
        3,
        True,
    )
    last = json.loads((out / "run_manifest.json").read_text())["attempts"][-1]
    assert last["pending"] == 0 and "served_model" not in last


def test_results_asked_of_more_than_one_model_are_never_resumed(tmp_path, pin_split):
    stage = _stage(tmp_path, pin_split)
    out = tmp_path / "out"
    first = _endpoint_run(stage, _FakeDeployment(answer=_busy_second), out, count=2)
    problems, _ = protocol.load_problems(
        stage.root / deployment.DATASET_RELATIVE_DIR, protocol.load_manifest(stage.manifest), 2
    )
    _, _, fields = records.interpret_reply(200, _payload(content=f"It is {problems[1].gold}."))
    pooled = records.make_record(
        problems[1],
        protocol.build_request("other", problems[1].prompt),
        records.OK,
        None,
        fields,
        identity=first["run_identity"],
        attempt=2,
        started_at="t0",
        finished_at="t1",
        latency_s=1.0,
    )
    with open(out / "results.jsonl", "a") as f:
        f.write(json.dumps(pooled) + "\n")
    for model in ("served", "other"):
        fake = _FakeDeployment(model=model)
        with pytest.raises(evaluator.EvaluationError, match="more than one model"):
            _endpoint_run(stage, fake, out, count=2)
        assert fake.gets == [] and fake.bodies == []


def test_endpoint_without_model_listing_resumes_with_the_named_model(tmp_path, pin_split):
    stage = _stage(tmp_path, pin_split)
    out = tmp_path / "out"
    router = "http://router.example:8000"
    first = _endpoint_run(
        stage,
        _UnlistedDeployment(answer=_busy_second, base=router),
        out,
        count=3,
        served_model="served",
    )
    assert first["errors"] == ["test-0001"]
    fake = _UnlistedDeployment(base=router)
    resumed = _endpoint_run(stage, fake, out, count=3, served_model="served")
    assert fake.asked == [1] and resumed["served_model"] == "served"
    assert resumed["verdict"]["passed"] and resumed["timing"]["served_model_listed"] is False


def test_endpoint_mode_selects_exactly_n_questions(tmp_path, pin_split):
    stage = _stage(tmp_path, pin_split)
    for count in (1, 100, 105):
        fake = _FakeDeployment(base="http://endpoint.example:9000")
        summary = _endpoint_run(stage, fake, tmp_path / f"n{count}", count=count, concurrency=8)
        assert sorted(fake.asked) == list(range(count))
        assert (summary["expected"], summary["distinct_completed"], summary["num_questions"]) == (
            count,
            count,
            count,
        )
        identity = json.loads((tmp_path / f"n{count}" / "run_manifest.json").read_text())[
            "identity"
        ]
        assert identity["selection"]["count"] == count and identity["transport"] == "endpoint"
    with pytest.raises(evaluator.EvaluationError, match="1..1319"):
        _endpoint_run(stage, _FakeDeployment(), tmp_path / "zero", count=0)
    assert not (tmp_path / "zero").exists()


def test_interruption_stops_the_owned_server_and_keeps_completed_replies(tmp_path, pin_split):
    stage = _stage(tmp_path, pin_split)

    def interrupt(gold, row):
        if row == 50:
            raise KeyboardInterrupt
        return _reply(f"So the answer is {gold}.")

    fake = _FakeDeployment(answer=interrupt)
    with pytest.raises(KeyboardInterrupt):
        _run(stage, fake, tmp_path / "out")
    (served,) = fake.servers
    assert served.terminated
    results = tmp_path / "out" / runner.RESULTS_DIRNAME
    kept = {r["id"] for r in _lines(results / "results.jsonl")}
    assert "test-0050" not in kept and kept <= {protocol.question_id(i) for i in range(100)}
    resumed = _FakeDeployment()
    summary = _run(stage, resumed, tmp_path / "out")
    assert summary["distinct_completed"] == 100
    assert sorted(resumed.asked) == sorted(
        i for i in range(100) if protocol.question_id(i) not in kept
    )


# ----------------------------------------------------------- owned process lifecycle (real)

_SERVER_SCRIPT = """
import os, signal, subprocess, sys, time
addr, pids, ignore_term = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
if ignore_term:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, "-c", "import time\\nwhile True: time.sleep(1)"])
orphan = subprocess.Popen([sys.executable, "-c", "import time\\nwhile True: time.sleep(1)"],
                          start_new_session=False)
with open(pids, "w") as f:
    f.write(f"{os.getpid()} {child.pid} {orphan.pid}")
with open(addr, "w") as f:
    f.write("127.0.0.1:1")
while True:
    time.sleep(1)
"""


def _alive(pid):
    try:
        os.kill(pid, 0)
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (ProcessLookupError, FileNotFoundError):
        return False


def _all_gone(pids, timeout_s=10.0):
    """SIGKILL is delivered asynchronously: wait (bounded) until every pid has exited."""
    deadline = time.monotonic() + timeout_s
    while any(_alive(pid) for pid in pids):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)
    return True


def _server_pids(tmp_path, process):
    assert server.wait_for_addr(tmp_path / "addr", 30, process) == ("127.0.0.1", 1)
    for _ in range(100):
        if (tmp_path / "pids").exists() and len((tmp_path / "pids").read_text().split()) == 3:
            break
        time.sleep(0.1)
    return [int(p) for p in (tmp_path / "pids").read_text().split()]


def test_stopping_a_managed_server_kills_only_its_own_processes(tmp_path):
    script = tmp_path / "fake_server.py"
    script.write_text(_SERVER_SCRIPT)
    sentinel = subprocess.Popen([sys.executable, "-c", "import time\ntime.sleep(300)"])
    try:
        argv = [sys.executable, str(script), str(tmp_path / "addr"), str(tmp_path / "pids"), "0"]
        with server.launch(argv, os.environ, tmp_path / "server.log") as process:
            pids = _server_pids(tmp_path, process)
            assert os.getsid(process.pid) == process.pid != os.getsid(os.getpid())
            assert set(pids[1:]) <= server.owned_processes(process.pid)
        assert _all_gone(pids)
        assert sentinel.poll() is None, "a process this run did not start was signalled"
    finally:
        sentinel.kill()
        sentinel.wait()


def test_a_server_ignoring_sigterm_is_killed_after_the_grace_period(tmp_path):
    script = tmp_path / "fake_server.py"
    script.write_text(_SERVER_SCRIPT)
    argv = [sys.executable, str(script), str(tmp_path / "addr"), str(tmp_path / "pids"), "1"]
    with open(tmp_path / "server.log", "w") as log:
        process = subprocess.Popen(
            argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    pids = _server_pids(tmp_path, process)
    began = time.monotonic()
    assert server.stop(process, grace_s=2.0) == -signal.SIGKILL
    assert time.monotonic() - began < 60
    assert _all_gone(pids)


def test_server_exiting_before_its_address_is_a_startup_error(tmp_path):
    with server.launch(
        [sys.executable, "-c", "raise SystemExit(3)"], os.environ, tmp_path / "log"
    ) as p:
        with pytest.raises(RuntimeError, match="exited with code 3"):
            server.wait_for_addr(tmp_path / "addr", 30, p)
    with pytest.raises(TimeoutError, match="did not report its address"):
        server.wait_for_addr(tmp_path / "never", 0.5)


# ------------------------------------------------------------------------- command line


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--num-questions", "0"], "1..1319"),
        (["--num-questions", "1320"], "1..1319"),
        (["--num-questions", "ten"], "not an integer"),
        (["--num-questions", "5.0"], "not an integer"),
        (["--concurrency", "0"], "at least 1"),
        (["--config", "c.yaml"], "managed mode"),
        (["--min-correct", "101"], "--min-correct"),
    ],
)
def test_command_line_refuses_bad_arguments_before_any_request(capfd, argv, message):
    base = ["--endpoint", "http://unused:1", "--dataset-dir", "d", "--output-dir", "o"]
    with pytest.raises(SystemExit) as exit_info:
        eval_gsm8k.main(base + argv)
    assert exit_info.value.code == 2
    assert message in capfd.readouterr().err


def test_command_line_needs_exactly_one_target_and_a_config_for_managed_mode(capfd):
    for argv in (
        ["--dataset-dir", "d", "--output-dir", "o"],
        ["--endpoint", "http://e:1", "--model", "m", "--dataset-dir", "d", "--output-dir", "o"],
        ["--model", "m", "--dataset-dir", "d", "--output-dir", "o"],
    ):
        with pytest.raises(SystemExit) as exit_info:
            eval_gsm8k.main(argv)
        assert exit_info.value.code == 2
    assert "needs --config" in capfd.readouterr().err


def test_managed_command_line_reads_the_config_and_records_its_identity(tmp_path):
    config = tmp_path / "serve.yaml"
    config.write_text(yaml.safe_dump({"max_batch_size": 64, "max_seq_len": 69632}))
    template = tmp_path / "chat.jinja"
    template.write_text("{{ messages }}")
    args = eval_gsm8k.build_parser().parse_args(
        [
            "--model",
            "/m",
            "--config",
            str(config),
            "--chat-template",
            str(template),
            "--server-env",
            "A=1",
            "--monitor-gpus",
            "0,1",
            "--dataset-dir",
            "d",
            "--output-dir",
            "o",
        ]
    )
    managed, batch, seq = eval_gsm8k._managed(args)
    assert (batch, seq) == (64, 69632)
    assert managed.argv == (
        "trtllm-serve",
        "/m",
        "--config",
        str(config),
        "--chat_template",
        str(template),
    )
    assert managed.env["A"] == "1" and managed.gpu_ids == ("0", "1")
    assert managed.settings["config_sha256"] == protocol.sha256_file(config)
    assert managed.settings["chat_template_sha256"] == protocol.sha256_file(template)


def test_endpoint_client_imports_no_tensorrt_llm_torch_pytest_or_yaml(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import json, sys\n"
        f"sys.path.insert(0, {str(EXAMPLES_DIR)!r})\n"
        "import eval_gsm8k\n"
        "from gsm8k_eval import protocol, records, runner, server\n"
        "heavy = ('tensorrt_llm', 'torch', 'pytest', 'yaml', 'triton')\n"
        "print(json.dumps(sorted(m for m in heavy if m in sys.modules)))\n"
    )
    result = subprocess.run(
        [sys.executable, str(probe)], capture_output=True, text=True, timeout=120, check=True
    )
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []
