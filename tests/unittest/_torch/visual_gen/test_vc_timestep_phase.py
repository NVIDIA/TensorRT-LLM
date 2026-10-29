# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The V-Smooth schedule of VC-Attention on the normalized timestep and its CUDA graph key."""

from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.timestep_phase import (
    VC_SCHEDULE_STEPS,
    vc_denoise_step_for_phase,
    vc_denoise_step_for_timestep,
    vc_smooth_phase_for_timestep,
)
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig
from tensorrt_llm._torch.visual_gen.models.modeling import BaseDiffusionModel
from tensorrt_llm.visual_gen.args import AttentionConfig, QuantAttentionConfig


@pytest.mark.parametrize(
    "timestep, step",
    [(None, None), (1.0, 0), (0.999, 0), (0.95, 3), (0.9375, 4), (0.75, 16), (0.0, 63)],
)
def test_timestep_maps_onto_the_virtual_schedule(timestep, step):
    expected = None if step is None else (step, VC_SCHEDULE_STEPS)
    assert vc_denoise_step_for_timestep(timestep) == expected
    if timestep is not None:
        assert vc_denoise_step_for_timestep(torch.tensor([timestep, 0.0])) == expected


@pytest.mark.parametrize(
    "timestep, phase",
    [(None, None), (1.0, 2), (0.98, 1), (0.9375, 2), (0.9, 1), (0.875, 2), (0.75, 0), (0.1, 0)],
)
def test_smooth_phase_follows_the_schedule(timestep, phase):
    assert vc_smooth_phase_for_timestep(timestep) == phase


@pytest.mark.parametrize("phase", [0, 1, 2])
def test_representative_step_lands_in_its_phase(phase):
    step, num_steps = vc_denoise_step_for_phase(phase)
    timestep = 1.0 - step / num_steps
    assert vc_smooth_phase_for_timestep(timestep) == phase


def _model(vc_attention: bool, vc_repair_budget: float = 0.0) -> BaseDiffusionModel:
    with patch("tensorrt_llm.visual_gen.args.get_sm_version", return_value=100):
        attention = AttentionConfig(
            backend="TRTLLM",
            quant_attention_config=QuantAttentionConfig(
                qk_dtype="bf16", vc_attention=vc_attention, vc_repair_budget=vc_repair_budget
            ),
        )
    model = object.__new__(BaseDiffusionModel)
    torch.nn.Module.__init__(model)
    model.model_config = DiffusionModelConfig(attention=attention)
    return model


def test_vc_attention_keys_the_graph_on_the_smooth_phase():
    runner = CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))
    _model(vc_attention=True).register_cuda_graph_extra_key_fns(runner)
    assert "vc_smooth_phase" in runner._extra_key_fns
    x = torch.empty(1, 2)
    regroup_key = runner.get_graph_key(x, timestep=torch.tensor([1.0]))
    keep_key = runner.get_graph_key(x, timestep=torch.tensor([0.98]))
    plain_key = runner.get_graph_key(x, timestep=torch.tensor([0.5]))
    assert ("vc_smooth_phase", 2) in regroup_key
    assert ("vc_smooth_phase", 1) in keep_key
    assert ("vc_smooth_phase", 0) in plain_key
    assert len({regroup_key, keep_key, plain_key}) == 3
    assert runner.get_graph_key(x) == runner._get_tensor_shape_key(x)


@pytest.mark.parametrize(
    "vc_attention, vc_repair_budget", [(False, 0.0), (True, 0.005)], ids=["sage", "repair"]
)
def test_schedule_free_recipes_do_not_register_the_phase(vc_attention, vc_repair_budget):
    runner = CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))
    _model(vc_attention, vc_repair_budget).register_cuda_graph_extra_key_fns(runner)
    assert "vc_smooth_phase" not in runner._extra_key_fns
