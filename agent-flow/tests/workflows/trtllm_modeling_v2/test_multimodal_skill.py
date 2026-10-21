# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The multimodal skill hands a modeling_v2 target's processor over through its routing module.

`@register_input_processor` on a target's class registers a processor the
launcher never asks for: `create_input_processor` looks it up by the model
class the built-in zoo maps the architecture to, and a modeling_v2 target is
never in the zoo. The skill recommended it anyway; a target's onboarding
found it reaching nothing and added a routing-level hook instead.
"""

from __future__ import annotations

from pathlib import Path

import agent_flow.workflows.trtllm_modeling_v2 as pkg

SKILLS = Path(pkg.__file__).parent / "agents" / "skills"


def _read(*parts: str) -> str:
    return " ".join(SKILLS.joinpath(*parts).read_text(encoding="utf-8").split())


def test_the_multimodal_skill_hands_the_processor_over_through_routing():
    skill = _read("trtllm-modeling-v2-multimodal", "SKILL.md")
    path = _read("trtllm-modeling-v2-multimodal", "references", "vision-path.md")
    assert "the processor goes through the routing module, not `@register_input_processor`" in skill
    assert "Through the family's routing module, never `@register_input_processor`." in path
    assert "`input_processor(pretrained_config)`" in path and "RoutedInputProcessor(" in path
    assert "defines no `modeling_v2_input_processor` has no hook yet" in path
    # Applied to the model class, the decorator is no longer offered as the way in.
    assert "applied to the **model class**" not in path
    # With the hand-off, reuse stays on and a chunk may hold part of an image.
    assert "**A context chunk may hold part of an image.**" in path
