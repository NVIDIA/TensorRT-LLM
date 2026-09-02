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
"""Unit tests for the shared ModelExpress transfer-evidence rules (`mx_evidence.py`).

The module lives next to the integration harness and is standard-library only,
so it is loaded here by file path rather than through the `defs` package.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "integration"
    / "defs"
    / "model_express"
    / "mx_evidence.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("mx_evidence_under_test", _MODULE_PATH)
    assert spec is not None and spec.loader is not None, _MODULE_PATH
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def evidence():
    return _load_module()


def _good_log(rank: int, count: int = 12) -> str:
    return (
        f"INFO some upstream chatter\nMatched {count}/{count} params\n"
        f"Rank {rank}: transferred {count} params\n"
    )


def test_module_is_standard_library_only():
    source = _MODULE_PATH.read_text(encoding="utf-8")
    for forbidden in ("import torch", "import pytest", "from defs", "import tensorrt_llm"):
        assert forbidden not in source, f"mx_evidence.py must stay stdlib-only, found {forbidden!r}"


def test_complete_evidence_has_no_problems(evidence):
    logs = {0: _good_log(0), 1: _good_log(1)}
    assert evidence.check_receiver_transfer_logs(logs, tp_size=2) == []


def test_missing_rank_is_reported(evidence):
    problems = evidence.check_receiver_transfer_logs({0: _good_log(0)}, tp_size=2)
    assert problems == ["Expected receiver transfer logs for ranks [0, 1], got [0]"]


def test_failure_marker_in_rank_log_or_stdout_is_reported(evidence):
    logs = {0: _good_log(0) + "MX P2P unavailable (no source); loading from disk\n"}
    problems = evidence.check_receiver_transfer_logs(logs, tp_size=1)
    assert problems == ["MX receiver logs contain failure marker 'mx p2p unavailable'"]

    problems = evidence.check_receiver_transfer_logs(
        {0: _good_log(0)}, tp_size=1, extra_text="... Falling back to DISK ..."
    )
    assert problems == ["MX receiver logs contain failure marker 'falling back to disk'"]


def test_incomplete_match_is_reported(evidence):
    logs = {0: "Matched 10/12 params\nRank 0: transferred 10 params\n"}
    problems = evidence.check_receiver_transfer_logs(logs, tp_size=1)
    assert problems == ["MX receiver rank 0 reported incomplete parameter match 10/12"]


def test_duplicate_summaries_are_reported(evidence):
    logs = {0: _good_log(0) + _good_log(0)}
    problems = evidence.check_receiver_transfer_logs(logs, tp_size=1)
    assert len(problems) == 2
    assert problems[0].startswith("Expected one matched-parameter summary for rank 0")
    assert problems[1].startswith("Expected one transfer summary for rank 0")


def test_transfer_summary_must_match_rank_and_count(evidence):
    wrong_rank = {0: "Matched 12/12 params\nRank 1: transferred 12 params\n"}
    assert evidence.check_receiver_transfer_logs(wrong_rank, tp_size=1) == [
        "MX receiver rank 0 matched 12 params but reported transfer summary [1, 12]"
    ]
    wrong_count = {0: "Matched 12/12 params\nRank 0: transferred 11 params\n"}
    assert evidence.check_receiver_transfer_logs(wrong_count, tp_size=1) == [
        "MX receiver rank 0 matched 12 params but reported transfer summary [0, 11]"
    ]


def test_summaries_are_json_friendly(evidence):
    summary = evidence.summarize_rank_log(1, _good_log(1, count=3))
    assert summary.to_dict() == {
        "rank": 1,
        "matched_summaries": [[3, 3]],
        "transfer_summaries": [[1, 3]],
        "failure_markers": [],
    }


def test_transfer_logs_by_rank_validates_directory(evidence, tmp_path: Path):
    with pytest.raises(ValueError, match="no non-empty receiver transfer logs"):
        evidence.transfer_logs_by_rank(tmp_path)

    (tmp_path / "rank0.log").write_text(_good_log(0), encoding="utf-8")
    (tmp_path / "rank1.log").write_text(_good_log(1), encoding="utf-8")
    assert sorted(evidence.transfer_logs_by_rank(tmp_path)) == [0, 1]
    assert sorted(evidence.summarize_transfer_logs(tmp_path)) == [0, 1]

    (tmp_path / "manifest.final.receiver.rank0.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected ModelExpress receiver transfer log"):
        evidence.transfer_logs_by_rank(tmp_path)


def test_duplicate_rank_files_are_rejected(evidence, tmp_path: Path):
    (tmp_path / "rank0.log").write_text(_good_log(0), encoding="utf-8")
    (tmp_path / "RANK0.log").write_text(_good_log(0), encoding="utf-8")
    with pytest.raises(ValueError, match="multiple receiver transfer logs for rank 0"):
        evidence.transfer_logs_by_rank(tmp_path)
