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
"""LoRA options of ``trtllm-bench prepare-dataset real-dataset`` (no network, no tokenizer download)."""

import json
import logging
import os
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import tensorrt_llm.bench.dataset.prepare_real_data as prepare_real_data

pytestmark = pytest.mark.cpu_only

_ROWS = [{"text": f"row {i}"} for i in range(3)]


class _CharTokenizer:
    def encode(self, text):
        return [ord(c) for c in text]


_MULTI_TURN_ROWS = [{"turns": [f"question {i}", f"follow-up {i}"]} for i in range(3)]
_MULTIMODAL_ROWS = [{"text": f"describe {i}", "image": f"/images/{i}.jpg"} for i in range(3)]

_INPUT_ARGS = ("--dataset-input-key", "text")
_MULTI_TURN_ARGS = ("--dataset-input-key", "turns")
_MULTIMODAL_ARGS = ("--dataset-image-key", "image", "--dataset-prompt-key", "text")


def _run(monkeypatch, tmp_path, rows=_ROWS, input_args=_INPUT_ARGS, **root_overrides):
    monkeypatch.setattr(prepare_real_data, "load_dataset_from_hf", lambda config: iter(rows))
    output = tmp_path / "dataset.jsonl"
    root_args = SimpleNamespace(
        tokenizer=_CharTokenizer(),
        output=str(output),
        random_seed=420,
        task_id=-1,
        rand_task_id=None,
        lora_dir=None,
    )
    for key, value in root_overrides.items():
        setattr(root_args, key, value)
    result = CliRunner().invoke(
        prepare_real_data.real_dataset,
        [
            "--dataset-name",
            "dummy",
            "--dataset-split",
            "train",
            *input_args,
            "--output-len-dist",
            "8,0",
        ],
        obj=root_args,
    )
    assert result.exit_code == 0, result.output
    return [json.loads(line) for line in output.read_text().splitlines()]


def test_real_dataset_without_lora_options_has_no_lora_request(monkeypatch, tmp_path):
    rows = _run(monkeypatch, tmp_path)
    assert len(rows) == len(_ROWS)
    assert all("lora_request" not in row for row in rows)


def test_real_dataset_fixed_task_id_adds_lora_request(monkeypatch, tmp_path):
    rows = _run(monkeypatch, tmp_path, task_id=1, lora_dir="/loras")
    assert len(rows) == len(_ROWS)
    for row in rows:
        assert row["lora_request"] == {
            "lora_name": "lora_1",
            "lora_int_id": 1,
            "lora_path": os.path.join("/loras", "1"),
        }


def test_real_dataset_random_task_ids_stay_in_range(monkeypatch, tmp_path):
    rows = _run(monkeypatch, tmp_path, rand_task_id=(0, 2), lora_dir="/loras")
    assert len(rows) == len(_ROWS)
    for row in rows:
        lora_request = row["lora_request"]
        assert 0 <= lora_request["lora_int_id"] <= 2
        assert lora_request["lora_name"] == f"lora_{lora_request['lora_int_id']}"


def test_real_dataset_random_task_ids_follow_random_seed(monkeypatch, tmp_path):
    first = _run(monkeypatch, tmp_path, rand_task_id=(0, 1000), lora_dir="/loras")
    second = _run(monkeypatch, tmp_path, rand_task_id=(0, 1000), lora_dir="/loras")
    assert [row["lora_request"] for row in first] == [row["lora_request"] for row in second]


_LORA_OPTIONS = [
    {"task_id": 1},
    {"rand_task_id": (0, 2)},
    {"lora_dir": "/loras"},
]
_IGNORED_MESSAGE = "LoRA options (--task-id/--rand-task-id/--lora-dir) are not supported for {} "


def _ignored_warnings(caplog):
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "LoRA options" in r.getMessage()
    ]


@pytest.mark.parametrize("options", _LORA_OPTIONS)
@pytest.mark.parametrize(
    "rows, input_args, kind",
    [
        (_MULTI_TURN_ROWS, _MULTI_TURN_ARGS, "multi-turn"),
        (_MULTIMODAL_ROWS, _MULTIMODAL_ARGS, "multimodal"),
    ],
)
def test_real_dataset_warns_when_lora_options_are_ignored(
    monkeypatch, tmp_path, caplog, rows, input_args, kind, options
):
    with caplog.at_level(logging.WARNING):
        written = _run(monkeypatch, tmp_path, rows=rows, input_args=input_args, **options)
    assert len(written) == len(rows)
    assert all("lora_request" not in row for row in written)
    messages = _ignored_warnings(caplog)
    assert len(messages) == 1
    assert messages[0].startswith(_IGNORED_MESSAGE.format(kind))
    assert messages[0].endswith("and will be ignored.")


@pytest.mark.parametrize(
    "rows, input_args",
    [
        (_MULTI_TURN_ROWS, _MULTI_TURN_ARGS),
        (_MULTIMODAL_ROWS, _MULTIMODAL_ARGS),
    ],
)
def test_real_dataset_does_not_warn_without_lora_options(
    monkeypatch, tmp_path, caplog, rows, input_args
):
    with caplog.at_level(logging.WARNING):
        written = _run(monkeypatch, tmp_path, rows=rows, input_args=input_args)
    assert len(written) == len(rows)
    assert _ignored_warnings(caplog) == []


def test_real_dataset_text_rows_do_not_get_the_ignored_lora_warning(monkeypatch, tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        rows = _run(monkeypatch, tmp_path, task_id=1, lora_dir="/loras")
    assert len(rows) == len(_ROWS)
    assert all("lora_request" in row for row in rows)
    assert _ignored_warnings(caplog) == []
