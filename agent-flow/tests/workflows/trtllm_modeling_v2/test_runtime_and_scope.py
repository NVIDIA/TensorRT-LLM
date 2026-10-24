# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every role is told the run-scoped declarations: the source embargo and the scope."""

from __future__ import annotations

from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import build_trtllm_modeling_v2_prompts


def test_every_role_gets_the_embargo_and_the_scope():
    """A role not told the rule is the role that breaks it."""
    bundle = build_trtllm_modeling_v2_prompts(
        source_embargo=["tensorrt_llm/_torch/models/modeling_qwen3_5.py"],
        scope="text_only",
    )
    for role in ("surveyor", "kernel_writer", "assembler", "tuner", "planner"):
        prompt = getattr(bundle, role)
        assert "Embargoed sources" in prompt, role
        assert "Scope: text only" in prompt, role
