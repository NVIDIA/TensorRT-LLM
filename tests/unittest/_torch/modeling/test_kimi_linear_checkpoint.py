# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-name tests for the Kimi Linear model."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("fla")

from tensorrt_llm._torch.models.modeling_kimi_linear import KimiLinearForCausalLM  # noqa: E402


def test_checkpoint_plan_preserves_external_attention_names():
    class _PlanHarness:
        checkpoint_name_plan = KimiLinearForCausalLM.checkpoint_name_plan
        model = SimpleNamespace(layers=[])

        def _trunk_parameters(self):
            return {
                "model.layers.0.linear_attn.q_proj.weight": torch.empty(0),
                "model.layers.1.self_attn.mixer.q_a_proj.weight": torch.empty(0),
                "lm_head.weight": torch.empty(0),
            }

    name_map, expected_keys, expert_jobs = _PlanHarness().checkpoint_name_plan("language_model.")

    assert name_map == {
        "model.layers.0.linear_attn.q_proj.weight": (
            "language_model.model.layers.0.self_attn.q_proj.weight"
        ),
        "model.layers.1.self_attn.mixer.q_a_proj.weight": (
            "language_model.model.layers.1.self_attn.q_a_proj.weight"
        ),
        "lm_head.weight": "language_model.lm_head.weight",
    }
    assert expected_keys == set(name_map.values())
    assert expert_jobs == []
