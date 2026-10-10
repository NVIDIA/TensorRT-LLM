# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the ``test_case`` block of ``task.yaml``.

The property under test is single authority over the measurement
conditions. A test-case id fixes them, so the block is exclusive with both
other ways of naming a workload and with a benchmark block, and the id and
its config are resolved at this boundary rather than deep inside a run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.perf_optimize import task_schema

SANITY_ID = "tests/integration/defs/perf/test_perf_sanity.py::test_e2e[aggr-cfg-entry]"
PERF_ID = "tests/integration/defs/perf/test_perf.py::test_perf[glm_5_fp8-bench-maxbs:512-gpus:8]"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A checkout stub carrying one aggregated perf-sanity config."""
    repo_dir = tmp_path / "repo"
    (repo_dir / "tests/scripts/perf-sanity/aggregated").mkdir(parents=True)
    (repo_dir / "tests/scripts/perf-sanity/aggregated/cfg.yaml").write_text(
        "metadata: {}\n", encoding="utf-8"
    )
    return repo_dir


def _task(tmp_path: Path, repo: Path, **extra) -> Path:
    spec = {
        "checkpoint_path": str(repo),
        "trtllm_repo_path": str(repo),
        **extra,
    }
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return path


def _load(path: Path) -> dict:
    return task_schema.load_and_validate_task_yaml(path)


def _expect_error(path: Path, match: str) -> None:
    with pytest.raises(task_schema.TaskSchemaError, match=match):
        _load(path)


# --------------------------------------------------------------------------- #
# Acceptance and what gets resolved
# --------------------------------------------------------------------------- #


def test_perf_sanity_case_resolves_its_config(tmp_path, repo):
    data = _load(_task(tmp_path, repo, test_case={"name": SANITY_ID}))
    assert task_schema.has_test_case(data)
    assert task_schema.test_case_name(data) == SANITY_ID
    resolved = data[task_schema.TEST_CASE_FIELD][task_schema.TEST_CASE_RESOLVED_KEY]
    assert resolved["family"] == "perf_sanity"
    assert resolved["runtime"] == "aggregated"
    assert resolved["select_pattern"] == "entry"
    assert task_schema.test_case_config_path(data) == str(
        repo / "tests/scripts/perf-sanity/aggregated/cfg.yaml"
    )


def test_perf_case_needs_no_config_file(tmp_path, repo):
    """The QA family generates its configuration from the id itself."""
    data = _load(_task(tmp_path, repo, test_case={"name": PERF_ID}))
    resolved = data[task_schema.TEST_CASE_FIELD][task_schema.TEST_CASE_RESOLVED_KEY]
    assert resolved == {"family": "perf"}
    assert task_schema.test_case_config_path(data) is None


def test_a_task_without_the_block_is_untouched(tmp_path, repo):
    data = _load(_task(tmp_path, repo))
    assert not task_schema.has_test_case(data)
    assert task_schema.test_case_name(data) is None
    assert task_schema.test_case_config_path(data) is None


# --------------------------------------------------------------------------- #
# The id and its config are resolved at this boundary
# --------------------------------------------------------------------------- #


def test_a_malformed_id_is_rejected_here(tmp_path, repo):
    """Not an hour into an allocation."""
    _expect_error(
        _task(tmp_path, repo, test_case={"name": "not-a-node-id"}),
        "not a pytest node id",
    )


def test_a_bad_grammar_id_is_rejected_here(tmp_path, repo):
    bad = "tests/integration/defs/perf/test_perf_sanity.py::test_e2e[disagg-nope-cfg]"
    _expect_error(
        _task(tmp_path, repo, test_case={"name": bad}),
        "invalid disagg benchmark mode",
    )


def test_a_missing_config_is_rejected_here(tmp_path, repo):
    """The case that would otherwise be sized at a single GPU downstream."""
    missing = "tests/integration/defs/perf/test_perf_sanity.py::test_e2e[aggr-absent]"
    _expect_error(
        _task(tmp_path, repo, test_case={"name": missing}),
        "perf-sanity config not found",
    )


@pytest.mark.parametrize("block", [{}, {"name": ""}, {"name": 7}, "a string", []])
def test_a_block_without_a_usable_name_is_rejected(tmp_path, repo, block):
    _expect_error(_task(tmp_path, repo, test_case=block), task_schema.TEST_CASE_FIELD)


# --------------------------------------------------------------------------- #
# Single authority over the measurement conditions
# --------------------------------------------------------------------------- #


def test_exclusive_with_extra_llm_api_options(tmp_path, repo):
    tuning = repo / "tuning.yaml"
    tuning.write_text("{}\n", encoding="utf-8")
    _expect_error(
        _task(
            tmp_path,
            repo,
            test_case={"name": SANITY_ID},
            extra_llm_api_options=str(tuning),
        ),
        "cannot be combined with",
    )


def test_exclusive_with_the_disagg_block(tmp_path, repo):
    _expect_error(
        _task(
            tmp_path,
            repo,
            test_case={"name": SANITY_ID},
            disagg={"config": str(repo / "harness.yaml")},
        ),
        "two measurement systems in one campaign",
    )


def test_a_benchmark_key_beside_a_test_case_is_rejected(tmp_path, repo):
    """Rejected rather than filled: the id already fixes this."""
    _expect_error(
        _task(
            tmp_path,
            repo,
            test_case={"name": SANITY_ID},
            benchmark={"concurrency": 128},
        ),
        "fixes the measurement conditions",
    )


def test_a_benchmark_block_is_still_fine_without_a_test_case(tmp_path, repo):
    data = _load(_task(tmp_path, repo, benchmark={"concurrency": 128}))
    assert data["benchmark"]["concurrency"] == 128


@pytest.mark.parametrize("case_id", [SANITY_ID, PERF_ID])
def test_a_test_case_spec_carries_no_defaulted_benchmark(tmp_path, repo, case_id):
    """The DEFAULTS must go too, not just a user-set block.

    Rejecting a user-set key is only half the guarantee: the base validation
    merges BENCHMARK_DEFAULTS afterwards, so a spec that correctly omitted the
    block still reached the agents carrying ISL 1024 / OSL 128 / concurrency 64.
    Every role's prompt calls task.yaml the source of truth and tells it to
    resolve `benchmark`, so the agent took the id from its prompt and the
    conditions from the file, then measured a workload nobody asked for and
    reported a plausible number for it. Silent, and wrong.
    """
    data = _load(_task(tmp_path, repo, test_case={"name": case_id}))
    assert "benchmark" not in data, (
        f"a defaulted benchmark block survived beside the test case: {data.get('benchmark')!r}"
    )


def test_a_plain_spec_still_gets_its_benchmark_defaults(tmp_path, repo):
    """The removal is scoped to test-case specs, not a global change."""
    from agent_flow.workflows.perf_analyze.task_schema import BENCHMARK_DEFAULTS

    data = _load(_task(tmp_path, repo))
    assert data["benchmark"] == BENCHMARK_DEFAULTS
