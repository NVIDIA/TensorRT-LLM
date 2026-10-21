# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the worker prompt, its tag sections, and the bundle that scopes them to a run."""

from __future__ import annotations

import dataclasses

from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import (
    DEFAULT_PROMPTS,
    build_trtllm_modeling_v2_prompts,
)
from agent_flow.workflows.trtllm_modeling_v2.agents.prompts.tags import (
    TAG_PROMPTS,
    TAG_SUMMARIES,
    tag_section,
    task_tags,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.environment import resolve_environment
from agent_flow.workflows.trtllm_modeling_v2.run.graph import WORKER_TAGS

ROLES = tuple(f.name for f in dataclasses.fields(DEFAULT_PROMPTS))


def test_every_worker_tag_has_a_section_and_a_summary():
    assert tuple(TAG_PROMPTS) == WORKER_TAGS == tuple(TAG_SUMMARIES)
    for tag, section in TAG_PROMPTS.items():
        assert section.startswith(f"## Tag `{tag}`"), tag


def test_a_tasks_tags_are_its_kind_then_its_own_each_after_what_it_brings():
    assert task_tags("kernel") == ("kernel",)
    assert task_tags("kernel", ["tune", "kernel"]) == ("kernel", "tune")
    # A feature is work on the target: the target's rules come first.
    assert task_tags("feature", ["investigate"]) == ("assemble", "feature", "investigate")
    assert task_tags("assemble", ["feature"]) == ("assemble", "feature")
    # A check runs no worker, and an unknown tag briefs nobody.
    assert task_tags("check") == ()
    assert task_tags("investigate", ["nonsense"]) == ("investigate",)


def test_the_sections_follow_the_tags_in_order():
    text = tag_section(("assemble", "feature", "investigate"))
    assert text.index("## Tag `assemble`") < text.index("## Tag `feature`")
    assert text.index("## Tag `feature`") < text.index("## Tag `investigate`")
    assert tag_section(()) == ""


def test_the_environment_block_reaches_every_role():
    environment = resolve_environment(
        {"environment": {"kind": "local", "python": "/opt/venv/bin/python", "env": {}}},
        devices=[0, 1],
        world_size=2,
    )
    bundle = build_trtllm_modeling_v2_prompts(environment=environment)
    for role in ROLES:
        text = getattr(bundle, role)
        assert "CUDA_VISIBLE_DEVICES=0,1" in text, role
        assert "symptom to report" in text, role


def test_a_task_roles_default_prompt_is_the_worker_plus_its_decision_block():
    for role in ("surveyor", "kernel_writer", "assembler"):
        prompt = getattr(DEFAULT_PROMPTS, role)
        assert "## Ending your turn" in prompt, role
        assert prompt.index("## Ending your turn") < prompt.index("## The decision file"), role
    assert "## The decision file" not in DEFAULT_PROMPTS.tuner
    assert "## Your decision" in DEFAULT_PROMPTS.tuner


def test_only_the_committing_roles_get_the_commit_helper():
    bundle = build_trtllm_modeling_v2_prompts(commit_script="/ws/bin/commit")
    for role in ROLES:
        told = "/ws/bin/commit" in getattr(bundle, role)
        assert told == (role in ("kernel_writer", "assembler")), role
