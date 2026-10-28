# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A check's jobs queue at its class: on the way to the first unit's accuracy goes first."""

from __future__ import annotations

from types import SimpleNamespace

from agent_flow.workflows.trtllm_modeling_v2.infra.compute import executor as priority
from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract as c
from agent_flow.workflows.trtllm_modeling_v2.run.validation.check_runner import CheckRunner


class _Stage(CheckRunner):
    def __init__(self):
        self.task_data = {"features": ["mtp"], "parallel": "tp1"}
        self.seen = []

    def _run_contract_check(self, check, log):
        self.seen.append((check.key, priority.current_job_priority()))
        return SimpleNamespace(passed=True)


def test_the_first_units_inner_checks_go_ahead_of_every_other():
    stage = _Stage()
    for check in (
        c.Check("core", "boot", "tp1", c.TIER_INNER),
        c.Check("core", "serve", "tp1", c.TIER_DELIVERY),
        c.Check("mtp", "boot", "tp1", c.TIER_INNER),
        c.Check("mtp", "acceptance", "tp1", c.TIER_PROOF),
    ):
        stage._prioritized_check(check, None)
    assert stage.seen == [
        ("core:boot@tp1", priority.ACCURACY),
        ("core:serve@tp1", priority.FEATURE),
        ("mtp:boot@tp1", priority.FEATURE),
        ("mtp:acceptance@tp1", priority.FEATURE),
    ]
    assert priority.current_job_priority() == priority.PERF
