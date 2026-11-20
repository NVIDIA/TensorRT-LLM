# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from os.path import relpath
from pathlib import Path

import pytest
from _torch.multimodal.conformance.tiers import FEATURE_TIERS

pytestmark = pytest.mark.cpu_only


def test_onboarding_skill_matches_feature_tiers():
    root = Path(__file__).resolve().parents[5]
    onboarding = root / ".claude/skills/trtllm-model-onboard-multimodal/SKILL.md"
    introduction, begin, table = onboarding.read_text().partition(
        "<!-- BEGIN multimodal-feature-tiers -->"
    )
    table, end, _ = table.partition("<!-- END multimodal-feature-tiers -->")
    expected = ["| Feature | Tier | What the model must provide |", "| --- | --- | --- |"]
    expected.extend(
        f"| `{feature}` | {tier} | {requirement} |"
        for feature, (tier, requirement) in FEATURE_TIERS.items()
    )
    assert begin and end and table.strip().splitlines() == expected, (
        "Update FEATURE_TIERS and the marked table in the multimodal onboarding skill together."
    )

    features = (
        root
        / "agent-flow/agent_flow/workflows/trtllm_modeling_v2/agents/skills"
        / "trtllm-modeling-v2-features"
    )
    reference = features / "references/features.md"
    assert "(references/features.md#6-multimodal-inputs)" in (features / "SKILL.md").read_text()
    section = (
        reference.read_text().partition("\n## 6. Multimodal inputs\n")[2].partition("\n## ")[0]
    )
    assert f"({relpath(onboarding, reference.parent)}#required-runtime-features-p0p1)" in section
    assert "\n## Required runtime features (P0/P1)\n" in introduction
