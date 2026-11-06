# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The V-Smooth schedule of VC-Attention, the step the pipeline publishes and the CUDA graph key."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.visual_gen.attention.backends.trtllm import (
    vc_denoise_step_for_phase,
    vc_smooth_phase,
)
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig
from tensorrt_llm._torch.visual_gen.models.modeling import BaseDiffusionModel
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline
from tensorrt_llm.visual_gen.args import AttentionConfig, QuantAttentionConfig

pytestmark = pytest.mark.cpu_only


def test_smooth_phase_follows_the_schedule():
    """The first quarter of the steps smooths V and regroups every fourth step; the rest is plain.
    The representative step of each phase under CUDA graph capture lands in that phase."""
    schedule = {
        None: None,
        (0, 50): 2,
        (1, 50): 1,
        (4, 50): 2,
        (12, 50): 2,
        (13, 50): 0,
        (49, 50): 0,
    }
    assert [vc_smooth_phase(step) for step in schedule] == list(schedule.values())
    assert [vc_smooth_phase(vc_denoise_step_for_phase(phase)) for phase in (0, 1, 2)] == [0, 1, 2]


def test_profile_denoise_steps_publishes_and_clears_the_step():
    """Every transformer component sees ``(step, num_steps)`` during a step and ``None`` after
    the loop, whether it ends normally or by an exception."""
    pipeline = object.__new__(BasePipeline)
    pipeline._is_warmup = True
    state = {"metadata_cache": {}}
    pipeline.transformer = SimpleNamespace(
        model_config=SimpleNamespace(attention_metadata_state=state)
    )
    seen = [(i, t, state["denoise_step"]) for i, t in pipeline._profile_denoise_steps([0.9, 0.5])]
    assert seen == [(0, 0.9, (0, 2)), (1, 0.5, (1, 2))]
    assert state["denoise_step"] is None
    with pytest.raises(RuntimeError, match="step failed"):
        for _ in pipeline._profile_denoise_steps([0.9, 0.5]):
            assert state["denoise_step"] == (0, 2)
            raise RuntimeError("step failed")
    assert state["denoise_step"] is None


def _model(vc_attention: bool, vc_repair_budget: float = 0.0) -> tuple[BaseDiffusionModel, dict]:
    model = object.__new__(BaseDiffusionModel)
    torch.nn.Module.__init__(model)
    # The attention config validates the recipe against the GPU, so build the whole
    # model config under the patch for the CPU-only test workers.
    with patch("tensorrt_llm.visual_gen.args.get_sm_version", return_value=100):
        model.model_config = DiffusionModelConfig(
            attention=AttentionConfig(
                backend="TRTLLM",
                quant_attention_config=QuantAttentionConfig(
                    qk_dtype="bf16", vc_attention=vc_attention, vc_repair_budget=vc_repair_budget
                ),
            ),
            attention_metadata_state={"metadata_cache": {}},
        )
    return model, model.model_config.attention_metadata_state


def test_graph_key_follows_the_published_step():
    """The tile-mean recipe keys its graphs on the published step's phase; the SageAttention
    and repair recipes, which do not follow the schedule, register no key."""
    runner = CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))
    model, state = _model(vc_attention=True)
    model.register_cuda_graph_extra_key_fns(runner)
    x = torch.empty(1, 2)
    keys = {}
    for step in ((0, 8), (1, 8), (4, 8), None):
        state["denoise_step"] = step
        keys[step] = runner.get_graph_key(x)
    assert ("vc_smooth_phase", 2) in keys[(0, 8)]
    assert ("vc_smooth_phase", 1) in keys[(1, 8)]
    assert ("vc_smooth_phase", 0) in keys[(4, 8)]
    assert keys[None] == runner._get_tensor_shape_key(x)
    assert len(set(keys.values())) == 4
    for vc_attention, vc_repair_budget in ((False, 0.0), (True, 0.005)):
        runner = CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))
        model, _ = _model(vc_attention, vc_repair_budget)
        model.register_cuda_graph_extra_key_fns(runner)
        assert "vc_smooth_phase" not in runner._extra_key_fns
