# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared VisualGen denoising-loop behavior."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.cosmos3.transformer_cosmos3 import Cosmos3VFMTransformer
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline
from tensorrt_llm._torch.visual_gen.profiler import VisualGenProfiler


class _AdditiveScheduler:
    def __init__(self, timesteps: torch.Tensor) -> None:
        self.timesteps = timesteps

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        return_dict: bool = False,
    ) -> tuple[torch.Tensor]:
        del timestep
        assert return_dict is False
        return (sample + model_output,)


def _pipeline() -> BasePipeline:
    pipeline = object.__new__(BasePipeline)
    pipeline.pipeline_config = SimpleNamespace(visual_gen_mapping=None)
    pipeline.cache_accelerator = None
    pipeline._is_warmup = False
    pipeline._profiler = VisualGenProfiler()
    return pipeline


@pytest.mark.parametrize("split_prompt_inputs", [True, False])
def test_guidance_interval_skips_inactive_cfg_batches(split_prompt_inputs: bool) -> None:
    """The denoiser emits the same guided tensors while omitting inactive CFG branches."""
    pipeline = _pipeline()
    timesteps = torch.tensor([1000.0, 750.0, 500.0, 250.0])
    scheduler = _AdditiveScheduler(timesteps)
    action_scheduler = _AdditiveScheduler(timesteps)
    positive_prompt = torch.tensor([[2.0]])
    negative_prompt = torch.tensor([[-1.0]])
    prompt_embeds = (
        positive_prompt if split_prompt_inputs else torch.cat([negative_prompt, positive_prompt])
    )
    neg_prompt_embeds = negative_prompt if split_prompt_inputs else None
    positive_ids = torch.tensor([[20.0]])
    negative_ids = torch.tensor([[-10.0]])
    call_batches = []
    cfg_batch_transitions = []

    def forward_fn(
        latent_input,
        extra_streams,
        step_index,
        timestep,
        encoder_hidden_states,
        extra_tensors,
    ):
        del step_index, timestep
        call_batches.append(
            (
                latent_input.shape[0],
                extra_streams["action"].shape[0],
                encoder_hidden_states.flatten().tolist(),
                extra_tensors["text_ids"].flatten().tolist(),
            )
        )
        video_prediction = encoder_hidden_states.reshape_as(latent_input)
        action_prediction = extra_tensors["text_ids"].reshape_as(extra_streams["action"])
        return video_prediction, {"action": action_prediction}

    video, extra = pipeline.denoise(
        latents=torch.zeros(1, 1),
        scheduler=scheduler,
        prompt_embeds=prompt_embeds,
        neg_prompt_embeds=neg_prompt_embeds,
        guidance_scale=3.0,
        guidance_interval=(960.0, 1001.0),
        forward_fn=forward_fn,
        extra_cfg_tensors={"text_ids": (positive_ids, negative_ids)},
        extra_streams={"action": (torch.zeros(1, 1), action_scheduler)},
        cfg_batch_transition_fn=cfg_batch_transitions.append,
    )

    assert [call[0] for call in call_batches] == [2, 1, 1, 1]
    assert [call[1] for call in call_batches] == [2, 1, 1, 1]
    assert cfg_batch_transitions == [False]
    assert call_batches[0][2:] == ([-1.0, 2.0], [-10.0, 20.0])
    assert all(call[2:] == ([2.0], [20.0]) for call in call_batches[1:])
    assert torch.equal(video, torch.tensor([[14.0]]))
    assert torch.equal(extra["action"], torch.tensor([[140.0]]))


def test_guidance_without_interval_keeps_every_cfg_batch() -> None:
    pipeline = _pipeline()
    timesteps = torch.tensor([1000.0, 750.0, 500.0, 250.0])
    scheduler = _AdditiveScheduler(timesteps)
    call_batches = []

    def forward_fn(
        latent_input,
        extra_streams,
        step_index,
        timestep,
        encoder_hidden_states,
        extra_tensors,
    ):
        del extra_streams, step_index, timestep, extra_tensors
        call_batches.append(latent_input.shape[0])
        return encoder_hidden_states.reshape_as(latent_input)

    result = pipeline.denoise(
        latents=torch.zeros(1, 1),
        scheduler=scheduler,
        prompt_embeds=torch.tensor([[2.0]]),
        neg_prompt_embeds=torch.tensor([[-1.0]]),
        guidance_scale=3.0,
        forward_fn=forward_fn,
    )

    assert call_batches == [2, 2, 2, 2]
    assert torch.equal(result, torch.tensor([[32.0]]))


def test_cosmos3_cfg_cache_transition_retains_conditional_batch() -> None:
    transformer = object.__new__(Cosmos3VFMTransformer)
    torch.nn.Module.__init__(transformer)
    key = torch.arange(12).reshape(2, 3, 2)
    value = key + 100
    visual_freq = torch.arange(8).reshape(2, 2, 2)
    combined_freq = visual_freq + 200
    transformer.cached_kv = [(key, value)]
    transformer.cached_freqs_gen = (visual_freq, visual_freq + 10)
    transformer.cached_freqs_gen_combined = (combined_freq, combined_freq + 10)

    transformer.retain_cfg_conditional_cache(batch_size=1)

    cached_key, cached_value = transformer.cached_kv[0]
    assert torch.equal(cached_key, key[1:])
    assert torch.equal(cached_value, value[1:])
    assert all(freq.shape[0] == 1 for freq in transformer.cached_freqs_gen)
    assert all(freq.shape[0] == 1 for freq in transformer.cached_freqs_gen_combined)

    key[1:].zero_()
    assert torch.count_nonzero(cached_key) > 0
