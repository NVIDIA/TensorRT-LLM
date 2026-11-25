# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the modeling-bringup guidance on ModelExpress (MX) checkpoint loading."""

from __future__ import annotations

from pathlib import Path

import yaml

from agent_flow.workflows.modeling_bringup.prompts import build_modeling_bringup_prompts

ROLES = ("plan_drafter", "plan_reviewer", "coder", "reviewer", "qa")
WORKFLOW_DIR = Path(__file__).resolve().parents[3] / "agent_flow" / "workflows" / "modeling_bringup"
MX_DOC = "docs/source/features/model-express.md"


def test_domain_priming_points_every_role_at_staged_post_load_hooks():
    """Every role learns where post-load work goes, so new models stay MX-ready."""
    bundle = build_modeling_bringup_prompts()
    for role in ROLES:
        prompt = getattr(bundle, role)
        assert MX_DOC in prompt, role
        for hook in (
            "setup_aliases()",
            "transform_weights()",
            "_weights_transformed",
            "cache_derived_state()",
            "post_load_weights()",
        ):
            assert hook in prompt, (role, hook)


def _mx_criterion(text: str, *, commented: bool) -> str:
    """Return the MX completion criterion documented in `text`, parsed as YAML."""
    prefix = "  # " if commented else "  "
    lines = text.splitlines()
    start = next(
        index
        for index, line in enumerate(lines)
        if line.startswith(prefix + '- "If the model fits the ModelExpress')
    )
    end = next(index for index in range(start, len(lines)) if lines[index].endswith('."'))
    block = [line.replace("# ", "", 1) if commented else line for line in lines[start : end + 1]]
    (criterion,) = yaml.safe_load("completion_criteria:\n" + "\n".join(block))[
        "completion_criteria"
    ]
    return criterion


def test_documented_mx_criterion_parses_and_names_the_doc():
    """The task example's MX criterion is valid YAML once uncommented and matches the README."""
    example = _mx_criterion(
        (WORKFLOW_DIR / "task.example.yaml").read_text(encoding="utf-8"), commented=True
    )
    readme = _mx_criterion(
        (WORKFLOW_DIR / "README.md").read_text(encoding="utf-8"), commented=False
    )

    assert example == readme
    assert MX_DOC in example
    assert "passing real donor/receiver test" in example
    assert "pending" in example
