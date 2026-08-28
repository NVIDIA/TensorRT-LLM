#!/usr/bin/env python3
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
"""Tests for CBTS skip-rate calculation and OpenSearch delivery."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_only


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "jenkins/scripts/cbts/tools/report_cbts_decision.py"


@pytest.fixture()
def report_module():
    """Import report_cbts_decision.py without making its tools directory a package."""
    spec = importlib.util.spec_from_file_location("report_cbts_decision", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def fake_blocks(monkeypatch):
    """Provide a small sharded stage universe to the report's lazy blocks import."""
    stages = {
        "H100-PyTorch-1": SimpleNamespace(yaml_stem="single"),
        "H100-PyTorch-2": SimpleNamespace(yaml_stem="single"),
        "H100-PyTorch-PerfSanity-1": SimpleNamespace(yaml_stem="perf_sanity"),
        "H100-4_GPUs-PyTorch-1": SimpleNamespace(yaml_stem="multi"),
        "H100-4_GPUs-PyTorch-2": SimpleNamespace(yaml_stem="multi"),
        "H100-4_GPUs-PyTorch-ModelExpress-OnDemand-1": SimpleNamespace(yaml_stem="on_demand"),
        "H100-PyTorch-Post-Merge-1": SimpleNamespace(yaml_stem="post_merge"),
    }
    case_counts = {
        "single": 100,
        "perf_sanity": 10,
        "multi": 50,
        "on_demand": 25,
        "post_merge": 20,
    }
    blocks = [
        SimpleNamespace(yaml_stem=stem, tests=[f"case-{index}" for index in range(count)])
        for stem, count in case_counts.items()
    ]

    class FakeYAMLIndex:
        @staticmethod
        def load(_path):
            return SimpleNamespace(blocks=blocks)

    fake_module = types.ModuleType("blocks")
    fake_module.YAMLIndex = FakeYAMLIndex
    fake_module.block_matches_stage = lambda block, stage: block.yaml_stem == stage.yaml_stem
    fake_module.parse_stages_from_groovy = lambda _path, include_post_merge: stages
    monkeypatch.setitem(sys.modules, "blocks", fake_module)
    return stages


def _decision() -> dict:
    return {
        "scope": "testsonly",
        "affected_stages": [
            "H100-PyTorch-1",
            "H100-PyTorch-2",
            "H100-4_GPUs-PyTorch-1",
            "H100-4_GPUs-PyTorch-2",
            "H100-4_GPUs-PyTorch-ModelExpress-OnDemand-1",
            "H100-PyTorch-Post-Merge-1",
        ],
        "affected_stage_test_counts": {
            "H100-PyTorch-1": 20,
            "H100-PyTorch-2": 20,
            "H100-4_GPUs-PyTorch-1": 5,
            "H100-4_GPUs-PyTorch-2": 5,
            "H100-4_GPUs-PyTorch-ModelExpress-OnDemand-1": 1,
            "H100-PyTorch-Post-Merge-1": 2,
        },
        "sanity_required": False,
        "perfsanity_required": True,
    }


@pytest.mark.parametrize(
    ("status", "required", "label_gate_open", "expected"),
    [
        ("pre_merge", False, False, False),
        ("pre_merge", True, False, False),
        ("pre_merge", False, True, False),
        ("pre_merge", True, True, True),
        ("post_merge", False, False, True),
    ],
)
def test_multi_gpu_scheduled_requires_policy_and_label_gate(
    report_module, status, required, label_gate_open, expected
):
    assert report_module._multi_gpu_scheduled(status, required, label_gate_open) is expected


def test_case_counts_use_scheduled_unsharded_pre_merge_universe(report_module, fake_blocks):
    """Multi-GPU/OnDemand/post-merge stages and duplicate shards must not inflate totals."""
    cbts_cases, total_cases = report_module._case_counts(
        _decision(), "pre_merge", str(REPO_ROOT), multi_gpu_scheduled=False
    )

    # One 100-case single-GPU family narrowed to 20, plus a force-kept
    # 10-case PerfSanity family. The two shards are one partitioned case set.
    assert (cbts_cases, total_cases) == (30, 110)


def test_case_counts_include_selected_multi_gpu_when_gate_is_open(report_module, fake_blocks):
    cbts_cases, total_cases = report_module._case_counts(
        _decision(), "pre_merge", str(REPO_ROOT), multi_gpu_scheduled=True
    )

    assert (cbts_cases, total_cases) == (35, 160)


def test_case_counts_include_coverage_multi_gpu_at_full_size(report_module, fake_blocks):
    decision = _decision()
    decision["affected_stages"] = ["H100-PyTorch-1", "H100-PyTorch-2"]
    decision["enable_multi_gpu"] = True

    cbts_cases, total_cases = report_module._case_counts(
        decision, "pre_merge", str(REPO_ROOT), multi_gpu_scheduled=True
    )

    assert (cbts_cases, total_cases) == (80, 160)


def test_build_document_filters_unscheduled_stages_and_persists_valid_rate(report_module):
    decision = _decision()
    decision["affected_stage_split_counts"] = {
        "H100-PyTorch-1": 1,
        "H100-4_GPUs-PyTorch-1": 1,
    }

    document = report_module.build_document(
        decision,
        "pre_merge",
        "",
        "123",
        cbts_cases=25,
        total_cases=100,
        multi_gpu_required=True,
        multi_gpu_label_gate_open=False,
    )

    assert document["d_case_skip_rate"] == 0.75
    assert document["b_case_skip_rate_valid"] is True
    assert document["b_non_cbts_multi_gpu_required"] is True
    assert document["b_multi_gpu_label_gate_open"] is False
    assert document["flat_detail"]["hit_stages"] == [
        "H100-PyTorch-1",
        "H100-PyTorch-2",
    ]
    assert document["flat_detail"]["split_counts"] == {"H100-PyTorch-1": 1}


def test_main_posts_case_skip_rate_to_opensearch(report_module, monkeypatch, tmp_path):
    posted_documents = []

    class FakeOpenSearchDB:
        @staticmethod
        def add_id_of_json(document):
            document["_id"] = "test-id"

        @staticmethod
        def postToOpenSearchDB(document, project):
            assert project == "cbts-test-project"
            posted_documents.append(document)
            return True

    fake_module = types.ModuleType("open_search_db")
    fake_module.CBTS_PROJECT_NAME = "cbts-test-project"
    fake_module.OpenSearchDB = FakeOpenSearchDB
    monkeypatch.setitem(sys.modules, "open_search_db", fake_module)
    monkeypatch.setattr(report_module, "_case_counts", lambda *_args, **_kwargs: (25, 100))

    decision_path = tmp_path / "decision.json"
    decision_path.write_text('{"scope": "testsonly", "affected_stages": []}')

    assert (
        report_module.main(
            [
                "--status",
                "pre_merge",
                "--decision",
                str(decision_path),
                "--repo-root",
                ".",
                "--multi-gpu-required",
                "--multi-gpu-label-gate-open",
            ]
        )
        == 0
    )
    assert posted_documents[0]["d_case_skip_rate"] == 0.75
    assert posted_documents[0]["b_case_skip_rate_valid"] is True
    assert posted_documents[0]["b_non_cbts_multi_gpu_required"] is True
    assert posted_documents[0]["b_multi_gpu_label_gate_open"] is True
