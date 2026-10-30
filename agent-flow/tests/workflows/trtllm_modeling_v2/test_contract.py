# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the contract: the checks a run owes, and their evaluation."""

from __future__ import annotations

from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract as c


def test_a_campaign_waits_for_the_cores_required_checks_only():
    """A failing variant or feature does not hold the core's tuning back; delivery owes it."""
    contract = c.Contract(
        checks=(
            c.Check("core", "boot", "tp4", c.TIER_INNER),
            c.Check("core", "accuracy[gsm8k]", "dep4", c.TIER_INNER),
            c.Check("core", "serve", "tep4", c.TIER_DELIVERY),
            c.Check("core", "boot", "tp8", c.TIER_INNER, required=False),
            c.Check("disagg", "disagg", "dep4", c.TIER_PROOF),
            c.Check("kv_nvfp4", "accuracy[gsm8k]", "dep4", c.TIER_PROOF),
        )
    )
    core = ["core:boot@tp4", "core:accuracy[gsm8k]@dep4", "core:serve@tep4"]
    assert [k.key for k in c.core_checks(contract)] == core
    assert [k.key for k in c.core_checks(contract, "core")] == core
    assert [k.key for k in c.core_checks(contract, "disagg")] == ["disagg:disagg@dep4"]
    assert c.core_checks(c.Contract(checks=())) == ()
