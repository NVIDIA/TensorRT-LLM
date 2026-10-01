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
"""Tests for composing the test-case campaign section.

Two properties. The section reaches exactly the roles that launch or
measure a workload and is composed last, because it supersedes the
single-server lifecycle those roles otherwise follow. And composing it
leaves a campaign that does not ask for it byte-identical, so the existing
modes cannot drift when this one changes.
"""

from __future__ import annotations

from agent_flow.workflows.perf_optimize.prompts import (
    DEFAULT_PROMPTS,
    TEST_CASE_CAMPAIGN,
    build_perf_optimize_prompts,
)
from agent_flow.workflows.perf_optimize.prompts._common import DISAGG_CAMPAIGN

#: Roles that stand up or measure a workload, and so need the override.
MEASURING_ROLES = ("benchmarker", "analyzer", "optimizer", "evaluator", "integrator", "qa")
#: Roles deliberately left alone: neither runs a workload.
UNTOUCHED_ROLES = ("projector", "reporter")


def _flat(text: str) -> str:
    """Whitespace-collapsed view, so re-wrapping prose cannot fail a test."""
    return " ".join(text.split())


def test_section_reaches_every_measuring_role():
    bundle = build_perf_optimize_prompts(include_test_case=True)
    for role in MEASURING_ROLES:
        assert TEST_CASE_CAMPAIGN in getattr(bundle, role), role


def test_projector_and_reporter_are_left_alone():
    bundle = build_perf_optimize_prompts(include_test_case=True)
    for role in UNTOUCHED_ROLES:
        assert TEST_CASE_CAMPAIGN not in getattr(bundle, role), role
        assert getattr(bundle, role) == getattr(DEFAULT_PROMPTS, role), role


def test_section_is_composed_last():
    """It supersedes what precedes it, so nothing may follow it."""
    bundle = build_perf_optimize_prompts(
        include_test_case=True,
        include_slurm_environment=True,
        include_sol=True,
        approaches=["code"],
    )
    for role in MEASURING_ROLES:
        text = getattr(bundle, role)
        assert text.rstrip().endswith(TEST_CASE_CAMPAIGN.rstrip()), role


def test_not_composed_by_default():
    bundle = build_perf_optimize_prompts()
    for role in MEASURING_ROLES:
        assert TEST_CASE_CAMPAIGN not in getattr(bundle, role), role


def test_an_unrelated_campaign_is_unchanged():
    """A mode that does not ask for the section must be byte-identical."""
    assert build_perf_optimize_prompts() == DEFAULT_PROMPTS
    disagg_only = build_perf_optimize_prompts(include_disagg=True)
    for role in MEASURING_ROLES:
        assert TEST_CASE_CAMPAIGN not in getattr(disagg_only, role), role
        assert DISAGG_CAMPAIGN in getattr(disagg_only, role), role


def test_the_section_states_what_it_supersedes():
    """An agent carrying it must not also follow the serve lifecycle."""
    flat = _flat(TEST_CASE_CAMPAIGN)
    assert "supersedes" in flat
    for superseded in ("trtllm-serve", "--extra_llm_api_options", "benchmark_serving.py"):
        assert superseded in flat, superseded


def test_the_section_names_both_families_and_their_levers():
    flat = _flat(TEST_CASE_CAMPAIGN)
    assert "test_perf_sanity.py::test_e2e" in flat
    assert "test_perf.py::test_perf" in flat
    # The editable lever for each family, which is what makes a config fix
    # expressible at all.
    assert "tests/scripts/perf-sanity/" in flat
    assert "pytorch_model_config.py" in flat


def test_the_section_forbids_changing_the_id():
    flat = _flat(TEST_CASE_CAMPAIGN)
    assert "Never change it" in flat
    # The consequence that makes it more than a style preference.
    assert "tests/integration/test_lists/" in flat


def test_the_section_classifies_a_config_edit_as_code():
    flat = _flat(TEST_CASE_CAMPAIGN)
    assert "`code` item, not a `config` item" in flat


def test_the_section_carries_the_rc_gate():
    """A pruned selection runs nothing and still exits zero."""
    flat = _flat(TEST_CASE_CAMPAIGN)
    assert "Never gate on the exit code" in flat
    assert "Successful requests" in flat


def test_the_section_asks_for_the_metric_direction():
    assert "direction" in _flat(TEST_CASE_CAMPAIGN)


# --------------------------------------------------------------------------- #
# The layer-2 override
# --------------------------------------------------------------------------- #

SANITY_ID = "tests/integration/defs/perf/test_perf_sanity.py::test_e2e[aggr-cfg-entry]"


def _workflow_for(task_path):
    """A workflow shell wired only for ``_measurement_directive``."""
    from agent_flow.workflows.perf_optimize import task_schema
    from agent_flow.workflows.perf_optimize.workflow import PerfOptimizeWorkflow

    wf = PerfOptimizeWorkflow.__new__(PerfOptimizeWorkflow)
    wf._task_data = lambda: task_schema.load_and_validate_task_yaml(task_path)
    return wf


def _write_test_case_task(tmp_path):
    import yaml

    repo = tmp_path / "repo"
    (repo / "tests/scripts/perf-sanity/aggregated").mkdir(parents=True)
    (repo / "tests/scripts/perf-sanity/aggregated/cfg.yaml").write_text(
        "metadata: {}\n", encoding="utf-8"
    )
    task = tmp_path / "task.yaml"
    task.write_text(
        yaml.safe_dump(
            {
                "checkpoint_path": str(repo),
                "trtllm_repo_path": str(repo),
                "test_case": {"name": SANITY_ID},
            }
        ),
        encoding="utf-8",
    )
    return task


def test_directive_states_the_test_case_mode(tmp_path):
    """The stage instruction also names trtllm-serve, and it arrives last.

    Without the override the agent gets a contradiction: the system prompt
    says "run the test case", the more specific stage instruction says
    "launch trtllm-serve with --extra_llm_api_options and poll it".
    """
    directive = _workflow_for(_write_test_case_task(tmp_path))._measurement_directive()
    assert "TEST CASE" in directive
    assert SANITY_ID in directive
    assert "replaces all of it" in directive
    # It must name the guidance it overrides, or a reader cannot tell what to skip.
    for superseded in ("trtllm-serve", "--extra_llm_api_options", "benchmark_serving.py"):
        assert superseded in directive, superseded


def test_directive_points_at_the_resolved_config(tmp_path):
    directive = _workflow_for(_write_test_case_task(tmp_path))._measurement_directive()
    assert "cfg.yaml" in directive


def test_directive_is_empty_for_a_plain_campaign(tmp_path):
    import yaml

    repo = tmp_path / "repo"
    repo.mkdir()
    task = tmp_path / "task.yaml"
    task.write_text(
        yaml.safe_dump({"checkpoint_path": str(repo), "trtllm_repo_path": str(repo)}),
        encoding="utf-8",
    )
    assert _workflow_for(task)._measurement_directive() == ""
