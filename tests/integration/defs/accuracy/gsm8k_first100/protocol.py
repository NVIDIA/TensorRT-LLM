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
"""The public first-100 GSM8K protocol: pinned items, prompt, request fields and scorer.

The protocol is self-contained and public. It evaluates the first 100 questions of the
``openai/gsm8k`` ``main`` test split, in original split order, at a pinned dataset revision.
Every prompt is ten fixed training exemplars followed by the question, sent as one user
message; decoding is greedy with a 65,536-token completion allowance. Only the final
``content`` of a response is scored, never ``reasoning_content``.

``manifest.json`` freezes the item identities: the test indices 0..99 with the SHA-256 of
each question, answer and complete prompt, and the ten exemplar indices drawn once by the
recorded sampler. The test reads a staged copy of the dataset, never the network, and checks
every hash and the order before use. The manifest also embeds ``protocol_definition()``;
the two must be equal, so a change to the prompt, request or scorer needs a new protocol id
and a regenerated manifest.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

PROTOCOL_ID = "gsm8k-first100-public-v1"

DATASET_ID = "openai/gsm8k"
DATASET_CONFIG = "main"
DATASET_REVISION = "cc7b047b6e5bb11b4f1af84efc572db110a51b3c"
#: The pinned revision's split files, relative to the staged dataset directory.
SPLIT_FILES = {
    "train": "main/train-00000-of-00001.parquet",
    "test": "main/test-00000-of-00001.parquet",
}

NUM_QUESTIONS = 100
NUM_SHOTS = 10
EXEMPLAR_SEED = 42

EXEMPLAR_TEMPLATE = "Question: {question}\nAnswer: {answer}\n\n"
QUESTION_TEMPLATE = "Question: {question}\nAnswer:"

MAX_TOKENS = 65536
#: Every request field besides ``model`` and ``messages``. No stop strings are sent: the
#: served recipe's own end-of-sequence handling ends a reply.
SAMPLING_FIELDS = {
    "temperature": 0,
    "top_k": 1,
    "seed": 42,
    "max_tokens": MAX_TOKENS,
    "n": 1,
    "stream": False,
    "chat_template_kwargs": {"reasoning_effort": 128},
}

SCORER_ID = "final-content-last-number-v1"
GOLD_DELIMITER = "####"
#: Channel delimiters that must never reach the final content: their presence means the
#: reasoning parser did not split the reply, so the content would include the analysis.
CHANNEL_MARKERS = (
    "<|start|>",
    "<|end|>",
    "<|message|>",
    "<|channel|>",
    "<|constrain|>",
    "<|return|>",
    "<|call|>",
)

MANIFEST_PATH = Path(__file__).with_name("manifest.json")
MANIFEST_SCHEMA = 1

# A number: an optional sign, ASCII digits with optional comma thousands separators (groups of
# three), and an optional fractional part. It may not start right after a word character or a
# period, so "10-15" holds 10 and 15 and "v1.2" holds none.
_NUMBER_BODY = r"([+-]?)([0-9]{1,3}(?:,[0-9]{3})+(?![0-9])|[0-9]+)(\.[0-9]+)?"
_NUMBER = re.compile(r"(?<![\w.])" + _NUMBER_BODY)
_FULL_NUMBER = re.compile(_NUMBER_BODY)

SCORER_RULE = (
    "Take the last number in the final content. A number is an optional sign (+ or -), ASCII "
    "digits with optional comma thousands separators in groups of three, and an optional "
    "fractional part (a period and digits); it may not start right after a letter, digit, "
    "underscore or period. Normalize exactly: remove the commas, drop leading zeros and "
    "trailing fractional zeros, write -0 as 0. Correct means equal to the normalized gold. "
    "Empty content or content without a number is a parse failure, scored incorrect."
)


def protocol_definition() -> dict[str, Any]:
    """The complete protocol as data; ``manifest.json`` embeds an identical copy."""
    return {
        "id": PROTOCOL_ID,
        "dataset": {
            "id": DATASET_ID,
            "config": DATASET_CONFIG,
            "revision": DATASET_REVISION,
            "files": dict(SPLIT_FILES),
        },
        "selection": {
            "split": "test",
            "count": NUM_QUESTIONS,
            "order": "original test-split order: row indices 0..99",
        },
        "exemplars": {
            "split": "train",
            "count": NUM_SHOTS,
            "sampler": f"random.Random({EXEMPLAR_SEED}).sample(range(len(train)), {NUM_SHOTS})",
            "order": "sampler output order",
        },
        "prompt": {
            "messages": "one user message",
            "exemplar": EXEMPLAR_TEMPLATE,
            "question": QUESTION_TEMPLATE,
            "text": "dataset question and answer strings exactly as stored",
        },
        "request": {
            "endpoint": "/v1/chat/completions",
            "fields": copy.deepcopy(SAMPLING_FIELDS),
            "stop": "none sent; the served recipe's end-of-sequence handling",
        },
        "scoring": {
            "id": SCORER_ID,
            "field": "choices[0].message.content",
            "rule": SCORER_RULE,
            "gold": f"the answer text after its last '{GOLD_DELIMITER}', normalized by the rule",
        },
    }


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    """Key-sorted compact JSON, the input of every structured digest."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def question_id(index: int) -> str:
    """The stable public id of test-split row ``index``."""
    return f"test-{index:04d}"


def exemplar_indices(train_rows: int) -> list[int]:
    """The recorded sampler; the manifest freezes its output so the ids never move."""
    return random.Random(EXEMPLAR_SEED).sample(range(train_rows), NUM_SHOTS)


def build_prompt(exemplars: Sequence[tuple[str, str]], question: str) -> str:
    """The single user message: exemplar (question, answer) pairs, then the question."""
    shots = "".join(EXEMPLAR_TEMPLATE.format(question=q, answer=a) for q, a in exemplars)
    return shots + QUESTION_TEMPLATE.format(question=question)


def build_request(model: str, prompt: str) -> dict[str, Any]:
    """The complete JSON body of one chat-completions request."""
    body: dict[str, Any] = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    body.update(copy.deepcopy(SAMPLING_FIELDS))
    return body


def request_record(body: Mapping[str, Any]) -> dict[str, Any]:
    """Every sent field, with message text replaced by its SHA-256 and length."""
    record = {key: copy.deepcopy(value) for key, value in body.items() if key != "messages"}
    record["messages"] = [
        {
            "role": message["role"],
            "content_sha256": sha256_text(message["content"]),
            "content_chars": len(message["content"]),
        }
        for message in body["messages"]
    ]
    return record


def _canonical_number(sign: str, integer: str, fraction: Optional[str]) -> str:
    value = Decimal(integer.replace(",", "") + (fraction or ""))
    if sign == "-":
        value = -value
    if value == 0:
        return "0"
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def normalize_number(text: str) -> str:
    """The canonical form of ``text``, which must be exactly one number of the grammar."""
    match = _FULL_NUMBER.fullmatch(text.strip())
    if match is None:
        raise ValueError(f"not a number of the scorer's grammar: {text!r}")
    return _canonical_number(*match.groups())


def extract_gold(answer: str) -> str:
    """The normalized gold number after the answer's last ``####``."""
    if GOLD_DELIMITER not in answer:
        raise ValueError(f"the answer has no '{GOLD_DELIMITER}' delimiter: {answer[-80:]!r}")
    return normalize_number(answer.rsplit(GOLD_DELIMITER, 1)[1])


@dataclass(frozen=True)
class Score:
    """The scorer's verdict on one final content.

    ``parse_status`` is ``ok`` (a number was parsed), ``empty_content`` (no final content,
    e.g. a reasoning-only reply), ``no_number`` or ``no_response`` (no reply was received).
    """

    parsed: Optional[str]
    parse_status: str
    correct: bool


NO_RESPONSE = Score(parsed=None, parse_status="no_response", correct=False)


def score_content(content: Optional[str], gold: str) -> Score:
    """Score one final ``content`` against the normalized ``gold``."""
    if content is None or not content.strip():
        return Score(parsed=None, parse_status="empty_content", correct=False)
    matches = list(_NUMBER.finditer(content))
    if not matches:
        return Score(parsed=None, parse_status="no_number", correct=False)
    parsed = _canonical_number(*matches[-1].groups())
    return Score(parsed=parsed, parse_status="ok", correct=parsed == gold)


@dataclass(frozen=True)
class Problem:
    """One pinned question with its prompt and gold answer."""

    id: str
    index: int
    question: str
    answer: str
    gold: str
    prompt: str
    prompt_sha256: str


def read_split(path: str | os.PathLike) -> list[tuple[str, str]]:
    """The (question, answer) rows of one split file, in stored order."""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["question", "answer"])
    return list(zip(table.column("question").to_pylist(), table.column("answer").to_pylist()))


def build_manifest(
    train: Sequence[tuple[str, str]],
    test: Sequence[tuple[str, str]],
    *,
    file_sha256: Mapping[str, str],
    python_version: str,
) -> dict[str, Any]:
    """The frozen item manifest of the pinned splits (run once; the result is committed)."""
    indices = exemplar_indices(len(train))
    exemplars = [train[i] for i in indices]
    questions = []
    for index in range(NUM_QUESTIONS):
        question, answer = test[index]
        extract_gold(answer)
        questions.append(
            {
                "id": question_id(index),
                "index": index,
                "question_sha256": sha256_text(question),
                "answer_sha256": sha256_text(answer),
                "prompt_sha256": sha256_text(build_prompt(exemplars, question)),
            }
        )
    rows = {"train": len(train), "test": len(test)}
    return {
        "schema": MANIFEST_SCHEMA,
        "protocol": protocol_definition(),
        "splits": {
            split: {"file": SPLIT_FILES[split], "rows": rows[split], "sha256": file_sha256[split]}
            for split in ("train", "test")
        },
        "hashing": "SHA-256 of the UTF-8 bytes of each string exactly as stored; "
        "prompt_sha256 covers the complete user message",
        "exemplars": {
            "generated_with": {
                "call": f"random.Random({EXEMPLAR_SEED}).sample(range({len(train)}), {NUM_SHOTS})",
                "python": python_version,
            },
            "items": [
                {
                    "index": i,
                    "question_sha256": sha256_text(train[i][0]),
                    "answer_sha256": sha256_text(train[i][1]),
                }
                for i in indices
            ],
        },
        "questions": questions,
    }


def load_manifest(path: str | os.PathLike = MANIFEST_PATH) -> dict[str, Any]:
    """The committed manifest, checked against this module's protocol and item rules."""
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"{path}: manifest schema {manifest.get('schema')!r} is not supported")
    if manifest.get("protocol") != protocol_definition():
        raise ValueError(f"{path}: the manifest's protocol differs from protocol_definition()")
    questions = manifest.get("questions", [])
    if [q.get("index") for q in questions] != list(range(NUM_QUESTIONS)):
        raise ValueError(f"{path}: questions must be test rows 0..{NUM_QUESTIONS - 1} in order")
    if [q.get("id") for q in questions] != [question_id(i) for i in range(NUM_QUESTIONS)]:
        raise ValueError(f"{path}: question ids do not follow the row indices")
    shots = [item.get("index") for item in manifest.get("exemplars", {}).get("items", [])]
    if len(shots) != NUM_SHOTS or len(set(shots)) != NUM_SHOTS:
        raise ValueError(f"{path}: the manifest must list {NUM_SHOTS} distinct exemplar rows")
    return manifest


def _check_row(split: str, index: int, row: tuple[str, str], item: Mapping[str, Any]) -> None:
    question, answer = row
    if (sha256_text(question), sha256_text(answer)) != (
        item["question_sha256"],
        item["answer_sha256"],
    ):
        raise ValueError(f"{split} row {index} does not match the pinned manifest")


def build_problems(
    manifest: Mapping[str, Any],
    train: Sequence[tuple[str, str]],
    test: Sequence[tuple[str, str]],
) -> list[Problem]:
    """The pinned problems, after checking every hash and the row order of both splits."""
    for split, rows in (("train", train), ("test", test)):
        expected = manifest["splits"][split]["rows"]
        if len(rows) != expected:
            raise ValueError(
                f"the {split} split has {len(rows)} rows; the manifest pins {expected}"
            )
    exemplars = []
    for item in manifest["exemplars"]["items"]:
        _check_row("train", item["index"], train[item["index"]], item)
        exemplars.append(train[item["index"]])
    problems = []
    for item in manifest["questions"]:
        index = item["index"]
        _check_row("test", index, test[index], item)
        question, answer = test[index]
        prompt = build_prompt(exemplars, question)
        digest = sha256_text(prompt)
        if digest != item["prompt_sha256"]:
            raise ValueError(f"the prompt of test row {index} does not match the pinned manifest")
        problems.append(
            Problem(
                id=item["id"],
                index=index,
                question=question,
                answer=answer,
                gold=extract_gold(answer),
                prompt=prompt,
                prompt_sha256=digest,
            )
        )
    return problems


def load_problems(
    dataset_dir: str | os.PathLike, manifest: Mapping[str, Any]
) -> tuple[list[Problem], dict[str, Any]]:
    """Read the staged splits under ``dataset_dir`` and return the verified problems.

    Also returns each split file's digest and whether it equals the snapshot digest recorded
    in the manifest; the row hashes, not the file bytes, are what must match.
    """
    rows = {}
    files = {}
    for split, relative in SPLIT_FILES.items():
        path = os.path.join(dataset_dir, relative)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"staged dataset file is missing: {relative}")
        digest = sha256_file(path)
        files[split] = {
            "file": relative,
            "sha256": digest,
            "equals_manifest_snapshot": digest == manifest["splits"][split]["sha256"],
        }
        rows[split] = read_split(path)
    return build_problems(manifest, rows["train"], rows["test"]), files
