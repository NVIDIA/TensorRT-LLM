# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every role is told the run-scoped declarations: the source embargo and the scope."""

from __future__ import annotations

from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import build_trtllm_modeling_v2_prompts


def test_every_role_gets_the_embargo_and_the_scope_and_every_worker_the_prior_art():
    """A role not told the rule is the role that breaks it.

    Tests, models, kernel libraries and other engines are sources; only listed paths are not.
    """
    bundle = build_trtllm_modeling_v2_prompts(
        source_embargo=["tensorrt_llm/_torch/models/modeling_qwen3_5.py"],
        scope="text_only",
    )
    for role in ("surveyor", "kernel_writer", "assembler", "tuner", "planner"):
        prompt = getattr(bundle, role)
        assert "Embargoed sources" in prompt, role
        assert "Scope: text only" in prompt, role
        if role == "planner":
            continue
        assert "## Prior art: read widely, then build on this engine" in prompt, role
        assert "Other inference engines** (vLLM, SGLang" in prompt, role
        assert "never a source" not in prompt, role
        assert "Everything not listed stays open, and you are expected to use it" in prompt, role
