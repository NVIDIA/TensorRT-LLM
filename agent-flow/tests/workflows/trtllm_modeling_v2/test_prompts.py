# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the worker prompt, its tag sections, and the bundle that scopes them to a run."""

from __future__ import annotations

from agent_flow.workflows.trtllm_modeling_v2.agents.prompts.tags import (
    TAG_PROMPTS,
    TAG_SUMMARIES,
    tag_section,
)
from agent_flow.workflows.trtllm_modeling_v2.run.graph import WORKER_TAGS


def test_every_worker_tag_has_a_section_and_a_summary_and_sections_follow_the_tags():
    assert tuple(TAG_PROMPTS) == WORKER_TAGS == tuple(TAG_SUMMARIES)
    for tag, section in TAG_PROMPTS.items():
        assert section.startswith(f"## Tag `{tag}`"), tag
    text = tag_section(("assemble", "feature", "investigate"))
    assert text.index("## Tag `assemble`") < text.index("## Tag `feature`")
    assert text.index("## Tag `feature`") < text.index("## Tag `investigate`")
    assert tag_section(()) == ""
