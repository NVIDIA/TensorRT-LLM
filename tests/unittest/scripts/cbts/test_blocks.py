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
"""Tests for CBTS test-db loading and narrowed test-db writing when a file holds several contexts."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

__extra_import_path__ = ["~/jenkins/scripts/cbts"]
from cbts.blocks import YAMLIndex, write_filtered_test_db
from cbts.coverage.tier import write_coverage_test_db

pytestmark = pytest.mark.cpu_only

_CONDITION = {"terms": {"stage": "post_merge", "backend": "pytorch"}}


def _write_test_db(test_db_dir: Path) -> None:
    test_db_dir.mkdir()
    (test_db_dir / "l0_x_multi_nodes_perf_sanity.yml").write_text(
        yaml.safe_dump(
            {
                "version": "0.0.1",
                "l0_x_multi_nodes_perf_sanity_a": [
                    {
                        "condition": _CONDITION,
                        "tests": ["perf/test_a.py::test_one", "perf/test_a.py::test_two"],
                    }
                ],
                "l0_x_multi_nodes_perf_sanity_b": [
                    {"condition": _CONDITION, "tests": ["perf/test_b.py::test_one"]}
                ],
            },
            sort_keys=False,
        )
    )
    (test_db_dir / "l0_x.yml").write_text(
        yaml.safe_dump(
            {"version": "0.0.1", "l0_x": [{"condition": _CONDITION, "tests": ["unit/test_c.py"]}]}
        )
    )


def test_yaml_index_loads_every_context_of_a_file(tmp_path: Path) -> None:
    """Each context of a multi-context file gets its own blocks and waive lookup."""
    _write_test_db(tmp_path / "test-db")
    index = YAMLIndex.load(tmp_path / "test-db")

    by_context = {block.yaml_stem: block.tests for block in index.blocks}
    assert by_context == {
        "l0_x_multi_nodes_perf_sanity_a": ["perf/test_a.py::test_one", "perf/test_a.py::test_two"],
        "l0_x_multi_nodes_perf_sanity_b": ["perf/test_b.py::test_one"],
        "l0_x": ["unit/test_c.py"],
    }
    match = index.find_match_for_waive("perf/test_b.py::test_one")
    assert match is not None and [b.yaml_stem for b in match[1]] == [
        "l0_x_multi_nodes_perf_sanity_b"
    ]


def test_narrowed_test_db_has_one_file_per_context(tmp_path: Path) -> None:
    """Both narrowed test-dbs write each affected context alone to `<context>.yml`."""
    _write_test_db(tmp_path / "test-db")

    filtered = tmp_path / "filtered"
    write_filtered_test_db(
        tmp_path / "test-db",
        filtered,
        {("l0_x_multi_nodes_perf_sanity_b", 0): {"perf/test_b.py": {"perf/test_b.py::test_one"}}},
    )
    assert sorted(p.name for p in filtered.glob("*.yml")) == ["l0_x_multi_nodes_perf_sanity_b.yml"]
    assert yaml.safe_load((filtered / "l0_x_multi_nodes_perf_sanity_b.yml").read_text()) == {
        "version": "0.0.1",
        "l0_x_multi_nodes_perf_sanity_b": [
            {"condition": _CONDITION, "tests": ["perf/test_b.py::test_one"]}
        ],
    }

    coverage = tmp_path / "coverage"
    write_coverage_test_db(
        tmp_path / "test-db",
        coverage,
        {("l0_x_multi_nodes_perf_sanity_a", 0): {"perf/test_a.py::test_two"}},
    )
    assert sorted(p.name for p in coverage.glob("*.yml")) == ["l0_x_multi_nodes_perf_sanity_a.yml"]
    assert yaml.safe_load((coverage / "l0_x_multi_nodes_perf_sanity_a.yml").read_text()) == {
        "version": "0.0.1",
        "l0_x_multi_nodes_perf_sanity_a": [
            {"condition": _CONDITION, "tests": ["perf/test_a.py::test_one"]}
        ],
    }
