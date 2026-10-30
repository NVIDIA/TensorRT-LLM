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
"""Durable per-question results, the run summary, the regression criterion and job costs.

Each question attempt becomes one JSON line in ``results.jsonl``, appended and flushed to disk
as soon as its reply arrives. Every line carries the run identity digest; a results file is
resumed only by a run with the same identity, and only questions without a completed reply
are asked again. A completed reply (``ok`` or ``truncated``) is final: it is never asked
again and never replaced. Failed attempts (``error``, ``timeout``) stay in the file, and the
question is asked again on resume. Loading rejects foreign ids, changed prompts, duplicate
completions and unreadable lines; an incomplete last line (an interrupted write) is moved
aside, never accepted.
"""

from __future__ import annotations

import collections
import datetime
import json
import math
import os
import statistics
import threading
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from . import protocol

RECORD_SCHEMA = 1
OK, TRUNCATED, ERROR, TIMEOUT = "ok", "truncated", "error", "timeout"
STATUSES = (OK, TRUNCATED, ERROR, TIMEOUT)
COMPLETED = (OK, TRUNCATED)

MODES = ("measurement", "regression")
#: The regression criterion is bound to the pinned first-100 protocol and manifest.
CRITERION_PATH = Path(__file__).with_name("criterion.json")
CRITERION_SCHEMA = 1


class LedgerError(RuntimeError):
    """A results file that cannot be resumed by this run."""


class CriterionError(RuntimeError):
    """A missing, malformed or inapplicable regression criterion."""


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")


def interpret_reply(
    http_status: int, text: str, *, max_seq_len: int
) -> tuple[str, Optional[str], dict[str, Any]]:
    """Classify one HTTP reply of the chat-completions endpoint.

    Returns ``(status, error, fields)``. ``ok`` is a ``stop``-finished reply; ``truncated`` a
    reply that used the whole 65,536-token allowance (``length``). A ``length`` finish below
    the allowance, a prompt for which the recipe's ``max_seq_len`` cannot hold the full
    allowance, final content that still holds channel delimiters (the reasoning parser did not
    split the reply) and any other finish reason are errors.
    """
    fields: dict[str, Any] = {"http_status": http_status}
    if http_status != 200:
        return ERROR, f"HTTP {http_status}: {text[:2000]}", fields
    try:
        payload = json.loads(text)
        choices = payload["choices"]
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError(f"expected one choice, got {choices!r:.200}")
        message = choices[0]["message"]
        content = message.get("content")
        reasoning = message.get("reasoning_content")
        if reasoning is None:
            reasoning = message.get("reasoning")
        if not isinstance(content, (str, type(None))) or not isinstance(
            reasoning, (str, type(None))
        ):
            raise ValueError("content and reasoning_content must be strings or null")
    except (ValueError, KeyError, TypeError) as error:
        return ERROR, f"malformed reply: {error}", fields
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else None
    finish = choices[0].get("finish_reason")
    fields.update(
        content=content,
        reasoning_chars=len(reasoning or ""),
        reasoning_sha256=protocol.sha256_text(reasoning) if reasoning else None,
        finish_reason=finish,
        stop_reason=choices[0].get("stop_reason"),
        usage=usage,
        prompt_tokens=usage.get("prompt_tokens") if usage else None,
        completion_tokens=usage.get("completion_tokens") if usage else None,
    )
    if content and any(marker in content for marker in protocol.CHANNEL_MARKERS):
        return ERROR, "final content holds channel delimiters (reasoning not split)", fields
    prompt_tokens = fields["prompt_tokens"]
    if not isinstance(prompt_tokens, int):
        return ERROR, "the reply reports no prompt token count", fields
    if prompt_tokens + protocol.MAX_TOKENS > max_seq_len:
        return (
            ERROR,
            f"max_seq_len {max_seq_len} cannot hold {prompt_tokens} prompt tokens plus the "
            f"{protocol.MAX_TOKENS}-token allowance",
            fields,
        )
    if finish == "stop":
        return OK, None, fields
    if finish == "length":
        if fields["completion_tokens"] != protocol.MAX_TOKENS:
            return (
                ERROR,
                f"length finish at {fields['completion_tokens']} completion tokens, below the "
                f"{protocol.MAX_TOKENS}-token allowance",
                fields,
            )
        return TRUNCATED, None, fields
    return ERROR, f"unexpected finish_reason {finish!r}", fields


def make_record(
    problem: protocol.Problem,
    request: Mapping[str, Any],
    status: str,
    error: Optional[str],
    fields: Mapping[str, Any],
    *,
    identity: str,
    attempt: int,
    started_at: str,
    finished_at: str,
    latency_s: float,
) -> dict[str, Any]:
    """One durable line: the attempt, its request fields, raw final content and score."""
    if status in COMPLETED:
        score = protocol.score_content(fields.get("content"), problem.gold)
    else:
        score = protocol.NO_RESPONSE
    record = {
        "schema": RECORD_SCHEMA,
        "run_identity": identity,
        "attempt": attempt,
        "id": problem.id,
        "index": problem.index,
        "prompt_sha256": problem.prompt_sha256,
        "request": protocol.request_record(request),
        "status": status,
        "error": error,
        "http_status": None,
        "finish_reason": None,
        "stop_reason": None,
        "content": None,
        "reasoning_chars": None,
        "reasoning_sha256": None,
        "usage": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "parsed_answer": score.parsed,
        "parse_status": score.parse_status,
        "gold": problem.gold,
        "correct": score.correct,
        "started_at": started_at,
        "finished_at": finished_at,
        "latency_s": round(latency_s, 3),
    }
    record.update({key: value for key, value in fields.items() if key in record})
    return record


@dataclass
class LedgerState:
    """The validated contents of a results file."""

    records: list[dict[str, Any]] = field(default_factory=list)
    completed: dict[str, dict[str, Any]] = field(default_factory=dict)
    latest: dict[str, dict[str, Any]] = field(default_factory=dict)
    quarantined: list[str] = field(default_factory=list)


class Ledger:
    """The append-only results file of one run identity.

    Args:
        path: The ``results.jsonl`` file.
        identity: The run identity digest every line must carry.
        expected: ``id -> (index, prompt_sha256)`` of the pinned questions.
    """

    def __init__(
        self, path: str | os.PathLike, identity: str, expected: Mapping[str, tuple[int, str]]
    ) -> None:
        self.path = Path(path)
        self.identity = identity
        self.expected = dict(expected)
        self._lock = threading.Lock()
        self._completed: set[str] = set()

    def _validate(self, record: Mapping[str, Any], where: str) -> None:
        if record.get("run_identity") != self.identity:
            raise LedgerError(
                f"{where}: written by run identity {record.get('run_identity')!r}, not "
                f"{self.identity!r}; use a fresh results directory for a different run"
            )
        qid = record.get("id")
        if qid not in self.expected:
            raise LedgerError(f"{where}: foreign question id {qid!r}")
        if (record.get("index"), record.get("prompt_sha256")) != self.expected[qid]:
            raise LedgerError(f"{where}: {qid} has a different index or prompt than pinned")
        if record.get("status") not in STATUSES:
            raise LedgerError(f"{where}: unknown status {record.get('status')!r}")

    def load(self) -> LedgerState:
        """Validate the file; move an incomplete last line aside."""
        state = LedgerState()
        if not self.path.exists():
            self._completed = set()
            return state
        data = self.path.read_bytes()
        complete, _, tail = data.rpartition(b"\n")
        if tail:
            aside = self.path.with_name(
                f"{self.path.name}.partial-{datetime.datetime.now():%Y%m%dT%H%M%S%f}"
            )
            aside.write_bytes(tail)
            with open(self.path, "r+b") as f:
                f.truncate(len(data) - len(tail))
                f.flush()
                os.fsync(f.fileno())
            state.quarantined.append(str(aside))
        lines = complete.split(b"\n") if complete else []
        for lineno, raw in enumerate(lines, 1):
            where = f"{self.path.name}:{lineno}"
            try:
                record = json.loads(raw)
            except ValueError as error:
                raise LedgerError(f"{where}: unreadable record ({error})") from error
            if not isinstance(record, dict):
                raise LedgerError(f"{where}: a record is a JSON object")
            self._validate(record, where)
            qid = record["id"]
            if qid in state.completed:
                raise LedgerError(f"{where}: {qid} already has a completed reply")
            if record["status"] in COMPLETED:
                state.completed[qid] = record
            state.latest[qid] = record
            state.records.append(record)
        self._completed = set(state.completed)
        return state

    def append(self, record: Mapping[str, Any]) -> None:
        """Write one record durably; a completed question can never be written again."""
        self._validate(record, "new record")
        line = (json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        with self._lock:
            if record["id"] in self._completed:
                raise LedgerError(f"{record['id']} already has a completed reply")
            if record["status"] in COMPLETED:
                self._completed.add(record["id"])
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(fd, line)
                os.fsync(fd)
            finally:
                os.close(fd)


def summarize(state: LedgerState, problems: Sequence[protocol.Problem]) -> dict[str, Any]:
    """Per-question outcome and counts over exactly the pinned questions."""
    per_question = []
    counts: collections.Counter[str] = collections.Counter()
    for problem in problems:
        record = state.completed.get(problem.id) or state.latest.get(problem.id)
        status = record["status"] if record else "missing"
        counts[status] += 1
        per_question.append(
            {
                "id": problem.id,
                "index": problem.index,
                "status": status,
                "attempt": record["attempt"] if record else None,
                "attempts": sum(1 for r in state.records if r["id"] == problem.id),
                "finish_reason": record["finish_reason"] if record else None,
                "completion_tokens": record["completion_tokens"] if record else None,
                "parse_status": record["parse_status"] if record else "no_response",
                "parsed_answer": record["parsed_answer"] if record else None,
                "gold": problem.gold,
                "correct": bool(record and record["correct"]),
                "latency_s": record["latency_s"] if record else None,
                "error": record["error"] if record else None,
            }
        )
    completed = [q for q in per_question if q["status"] in COMPLETED]
    latencies = [q["latency_s"] for q in completed]
    tokens = [q["completion_tokens"] for q in completed if isinstance(q["completion_tokens"], int)]
    correct = sum(q["correct"] for q in per_question)

    def ids(predicate) -> list[str]:
        return [q["id"] for q in per_question if predicate(q)]

    return {
        "protocol_id": protocol.PROTOCOL_ID,
        "expected": len(problems),
        "distinct_completed": len(completed),
        "status_counts": dict(sorted(counts.items())),
        "correct": correct,
        "total": len(problems),
        "accuracy": correct / len(problems) if problems else None,
        "parse_failures": ids(lambda q: q["status"] in COMPLETED and q["parse_status"] != "ok"),
        "truncated": ids(lambda q: q["status"] == TRUNCATED),
        "errors": ids(lambda q: q["status"] == ERROR),
        "timeouts": ids(lambda q: q["status"] == TIMEOUT),
        "missing": ids(lambda q: q["status"] == "missing"),
        "quarantined_partial_writes": list(state.quarantined),
        "latency_s": _distribution(latencies),
        "completion_tokens": {"sum": sum(tokens), "max": max(tokens)} if tokens else None,
        "per_question": per_question,
    }


def _distribution(values: Sequence[float]) -> Optional[dict[str, float]]:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p90": ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)],
        "max": ordered[-1],
        "sum": round(sum(ordered), 3),
    }


def load_criterion(
    path: str | os.PathLike = CRITERION_PATH, *, manifest_sha256: str
) -> dict[str, Any]:
    """The installed regression criterion; raises CriterionError when none applies."""
    path = Path(path)
    if not path.is_file():
        raise CriterionError(
            f"regression mode needs an installed criterion ({path.name} beside the protocol); "
            "none is installed. Measurement mode (the default) reports the score without a "
            "threshold."
        )
    try:
        criterion = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise CriterionError(f"{path.name} is not valid JSON: {error}") from error
    if not isinstance(criterion, dict) or criterion.get("schema") != CRITERION_SCHEMA:
        raise CriterionError(f"{path.name}: unsupported criterion schema")
    if criterion.get("protocol_id") != protocol.PROTOCOL_ID:
        raise CriterionError(
            f"{path.name} was set for protocol {criterion.get('protocol_id')!r}, not "
            f"{protocol.PROTOCOL_ID!r}"
        )
    if criterion.get("manifest_sha256") != manifest_sha256:
        raise CriterionError(f"{path.name} was set for a different item manifest")
    min_correct = criterion.get("min_correct")
    if type(min_correct) is not int or not 0 <= min_correct <= protocol.NUM_QUESTIONS:
        raise CriterionError(
            f"{path.name}: min_correct must be an integer in [0, {protocol.NUM_QUESTIONS}]"
        )
    required = criterion.get("required_identity", {})
    if not isinstance(required, dict):
        raise CriterionError(f"{path.name}: required_identity must be an object")
    return criterion


def criterion_mismatches(criterion: Mapping[str, Any], identity: Mapping[str, Any]) -> list[str]:
    """Run-identity fields that differ from those the criterion was measured with."""
    required = criterion.get("required_identity", {})
    return [key for key, value in required.items() if identity.get(key) != value]


def verdict(
    summary: Mapping[str, Any], mode: str, criterion: Optional[Mapping[str, Any]] = None
) -> dict[str, Any]:
    """Pass/fail of one run.

    Both modes require exactly the pinned questions to hold completed replies, with no error,
    timeout, truncation or missing question. Measurement mode sets no score threshold.
    Regression mode additionally requires ``correct >= min_correct`` of the installed
    criterion and cannot be evaluated without one.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    problems = []
    for key, label in (
        ("missing", "without a reply"),
        ("errors", "with an error"),
        ("timeouts", "timed out"),
        ("truncated", "truncated at the token allowance"),
    ):
        if summary[key]:
            problems.append(f"{len(summary[key])} question(s) {label}: {', '.join(summary[key])}")
    if summary["distinct_completed"] != summary["expected"] and not problems:
        problems.append(
            f"{summary['distinct_completed']} completed replies for {summary['expected']} "
            "pinned questions"
        )
    result: dict[str, Any] = {"mode": mode, "operational_ok": not problems, "problems": problems}
    if mode == "measurement":
        result["threshold"] = None
        result["passed"] = not problems
        return result
    if criterion is None:
        raise CriterionError("regression mode cannot be evaluated without a criterion")
    met = summary["correct"] >= criterion["min_correct"]
    result["criterion"] = {
        "min_correct": criterion["min_correct"],
        "correct": summary["correct"],
        "met": met,
    }
    if not met:
        problems.append(
            f"{summary['correct']}/{summary['total']} correct is below the installed criterion "
            f"of {criterion['min_correct']}"
        )
    result["passed"] = not problems
    return result


def timeout_minutes(measured_total_s: float) -> int:
    """The test-db TIMEOUT for a measured job: ceil((total + max(600 s, 20% total)) / 60)."""
    total = Fraction(str(measured_total_s))
    if total <= 0:
        raise ValueError(f"the measured total must be positive, got {measured_total_s}")
    return math.ceil((total + max(Fraction(600), total / 5)) / 60)


def gpu_memory_peaks(csv_text: str) -> dict[str, Any]:
    """Peak used memory per GPU from the memory poller's output.

    The input is ``nvidia-smi`` polling output with the fields ``timestamp, index, uuid,
    memory.used, memory.total`` (``--format=csv,noheader,nounits``).
    """
    peaks: dict[str, dict[str, Any]] = {}
    samples = 0
    for line in csv_text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5 or not parts[3].isdigit() or not parts[4].isdigit():
            continue
        samples += 1
        entry = peaks.setdefault(
            parts[1], {"uuid": parts[2], "peak_used_mib": 0, "total_mib": int(parts[4])}
        )
        entry["peak_used_mib"] = max(entry["peak_used_mib"], int(parts[3]))
    return {
        "samples": samples,
        "per_gpu": peaks,
        "max_peak_used_mib": max((p["peak_used_mib"] for p in peaks.values()), default=None),
        "sum_peak_used_mib": sum(p["peak_used_mib"] for p in peaks.values()) if peaks else None,
    }


def gpu_hours(gpu_count: int, seconds: float) -> float:
    return round(gpu_count * seconds / 3600.0, 4)
