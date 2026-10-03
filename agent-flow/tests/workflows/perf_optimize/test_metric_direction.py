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
"""Tests for deciding whether the target metric improves up or down.

The property under test is that the sign of a gain is never guessed. A
metric this workflow does not know, whose name carries no unit suffix,
used to be read as better-when-higher — so a latency-like metric scored a
slowdown as an improvement, and the target gate, the accepted-item ledger
and the final report all agreed with it. Resolution is explicit or it
raises.
"""

from __future__ import annotations

import pytest
import yaml

from agent_flow.workflows.perf_optimize import task_schema
from agent_flow.workflows.perf_optimize.workflow import PerfOptimizeWorkflow

#: Lower-is-better, published by the perf-sanity harness, and outside the
#: set this workflow knows. The metric the old suffix rule got wrong.
UNKNOWN_LOWER = "prev_device_step_time"


def _workflow(tmp_path, *, optimize=None):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    spec = {"checkpoint_path": str(repo), "trtllm_repo_path": str(repo)}
    if optimize:
        spec["optimize"] = optimize
    task = tmp_path / "task.yaml"
    task.write_text(yaml.safe_dump(spec), encoding="utf-8")
    data = task_schema.load_and_validate_task_yaml(task)
    wf = PerfOptimizeWorkflow.__new__(PerfOptimizeWorkflow)
    wf._task_data = lambda: data
    wf._optimize_block = lambda: data["optimize"]
    return wf


# --------------------------------------------------------------------------- #
# The sign itself
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "direction,measured,expected_sign",
    [
        ("higher", 110.0, +1),  # throughput rose
        ("higher", 90.0, -1),
        ("lower", 90.0, +1),  # latency fell
        ("lower", 110.0, -1),
    ],
)
def test_gain_sign_follows_the_direction(direction, measured, expected_sign):
    gain = PerfOptimizeWorkflow._normalized_gain_pct(100.0, measured, "whatever", direction)
    assert gain is not None
    assert (gain > 0) == (expected_sign > 0)
    assert abs(gain) == pytest.approx(10.0)


def test_a_zero_reference_yields_no_gain():
    assert PerfOptimizeWorkflow._normalized_gain_pct(0.0, 5.0, "m", "higher") is None


def test_direction_overrides_the_name(tmp_path):
    """The whole point: the name must not decide when direction is known."""
    # Named like a latency metric, declared higher-is-better.
    gain = PerfOptimizeWorkflow._normalized_gain_pct(100.0, 110.0, "mean_ttft_ms", "higher")
    assert gain == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# Three-tier resolution
# --------------------------------------------------------------------------- #


def test_roadmap_baseline_direction_wins(tmp_path):
    wf = _workflow(tmp_path, optimize={"target_metric": UNKNOWN_LOWER, "direction": "higher"})
    roadmap = {"baseline": {"value": 1.0, "source": "s", "direction": "lower"}}
    assert wf._metric_direction(roadmap) == "lower"


def test_task_direction_is_used_when_the_roadmap_has_none(tmp_path):
    wf = _workflow(tmp_path, optimize={"target_metric": UNKNOWN_LOWER, "direction": "lower"})
    assert wf._metric_direction(None) == "lower"
    assert wf._metric_direction({"baseline": {"value": 1.0, "source": "s"}}) == "lower"


@pytest.mark.parametrize(
    "metric,expected",
    [
        ("output_throughput", "higher"),
        ("total_token_throughput", "higher"),
        ("request_throughput", "higher"),
        ("mean_ttft_ms", "lower"),
        ("p99_itl_ms", "lower"),
    ],
)
def test_a_known_metric_falls_back_to_its_name(tmp_path, metric, expected):
    """Mode A keeps working with no task or roadmap entry."""
    wf = _workflow(tmp_path, optimize={"target_metric": metric})
    assert wf._metric_direction(None) == expected


def test_an_unknown_metric_without_a_direction_raises(tmp_path):
    """The defect this commit fixes, stated as a test.

    Before, this returned "higher" by omission and a rising step time was
    reported as a gain.
    """
    wf = _workflow(tmp_path, optimize={"target_metric": UNKNOWN_LOWER})
    with pytest.raises(RuntimeError, match="better when higher or lower"):
        wf._metric_direction(None)


def test_the_error_says_how_to_fix_it(tmp_path):
    wf = _workflow(tmp_path, optimize={"target_metric": UNKNOWN_LOWER})
    with pytest.raises(RuntimeError) as excinfo:
        wf._metric_direction(None)
    message = str(excinfo.value)
    assert "optimize.direction" in message
    assert UNKNOWN_LOWER in message


def test_a_malformed_roadmap_direction_is_ignored_not_trusted(tmp_path):
    wf = _workflow(tmp_path, optimize={"target_metric": "output_throughput"})
    assert wf._metric_direction({"baseline": {"direction": "sideways"}}) == "higher"


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #


def test_task_rejects_a_bad_direction(tmp_path):
    with pytest.raises(task_schema.TaskSchemaError, match="optimize.direction"):
        _workflow(tmp_path, optimize={"direction": "sideways"})


@pytest.mark.parametrize("direction", task_schema.METRIC_DIRECTIONS)
def test_task_accepts_both_directions(tmp_path, direction):
    wf = _workflow(tmp_path, optimize={"direction": direction})
    assert wf._optimize_block()["direction"] == direction


def test_roadmap_accepts_direction_on_the_baseline_block():
    from agent_flow.workflows.perf_optimize import roadmap_schema

    errors: list[str] = []
    roadmap_schema._validate_metric_ref(
        {"baseline": {"value": 1.0, "source": "s", "direction": "lower"}}, "baseline", errors
    )
    assert errors == []


def test_roadmap_rejects_a_bad_direction():
    from agent_flow.workflows.perf_optimize import roadmap_schema

    errors: list[str] = []
    roadmap_schema._validate_metric_ref(
        {"baseline": {"value": 1.0, "source": "s", "direction": "up"}}, "baseline", errors
    )
    assert any("direction" in err for err in errors)


def test_roadmap_still_accepts_a_block_without_direction():
    """Existing roadmaps must keep validating."""
    from agent_flow.workflows.perf_optimize import roadmap_schema

    errors: list[str] = []
    roadmap_schema._validate_metric_ref(
        {"baseline": {"value": 1.0, "source": "s"}}, "baseline", errors
    )
    assert errors == []


# --------------------------------------------------------------------------- #
# The prompt rule the code mirrors
# --------------------------------------------------------------------------- #


def test_the_measurement_protocol_classifies_by_meaning_not_spelling():
    from agent_flow.workflows.perf_optimize.prompts._common import MEASUREMENT_PROTOCOL

    flat = " ".join(MEASUREMENT_PROTOCOL.split())
    assert "what it measures" in flat
    assert "HIGHER is better" in flat and "LOWER is better" in flat
    # Metrics with no unit suffix must be covered by name, since those are
    # the ones a suffix rule silently gets wrong.
    assert "device step time" in flat
    assert "State the direction you used" in flat
