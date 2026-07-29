# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``QwenImagePipeline.forward`` denoise-loop orchestration.

Heavy model components are replaced by lightweight test doubles so the tests
exercise pipeline wiring without loading a checkpoint or running a real
transformer/VAE. Covers both the
true-CFG path (dual cond/uncond forward plus the norm-preserving CFG
combination) and the non-CFG path, on CPU.
"""

import torch

from tensorrt_llm._torch.visual_gen.models.qwen_image import QwenImagePipeline


class _RecordingTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 4
        self._device_anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls = []

    def forward(
        self,
        *,
        hidden_states,
        timestep,
        encoder_hidden_states_mask,
        encoder_hidden_states,
        img_shapes,
        return_dict,
    ):
        self.calls.append(
            {
                "hidden_states_shape": tuple(hidden_states.shape),
                "timestep": timestep.clone(),
                "encoder_hidden_states": encoder_hidden_states.clone(),
                "encoder_hidden_states_mask": encoder_hidden_states_mask.clone(),
                "img_shapes": img_shapes,
                "return_dict": return_dict,
            }
        )
        values = [-0.5, 0.25, 1.5, -1.0] if encoder_hidden_states[0, 0, 0] < 0 else [1, 2, 3, 4]
        pattern = torch.tensor(values, dtype=hidden_states.dtype, device=hidden_states.device)
        return (pattern.view(1, 1, -1).expand_as(hidden_states).clone(),)


class _RecordingScheduler:
    def __init__(self):
        self.config = {}
        self.set_timesteps_call = None
        self.begin_index = None
        self.step_calls = []
        self.timesteps = torch.empty(0)

    def set_timesteps(self, *, sigmas, device, mu):
        self.set_timesteps_call = {"sigmas": sigmas, "device": device, "mu": mu}
        self.timesteps = torch.arange(len(sigmas), 0, -1, device=device, dtype=torch.float32)

    def set_begin_index(self, index):
        self.begin_index = index

    def step(self, noise_pred, timestep, latents, *, return_dict):
        self.step_calls.append(
            {
                "noise_pred_shape": tuple(noise_pred.shape),
                "noise_pred": noise_pred.clone(),
                "timestep": timestep.clone(),
                "latents_shape": tuple(latents.shape),
                "return_dict": return_dict,
            }
        )
        return (latents - noise_pred * 0.1,)


def _pipeline_with_test_doubles():
    pipe = QwenImagePipeline.__new__(QwenImagePipeline)
    torch.nn.Module.__init__(pipe)
    pipe.vae_scale_factor = 8
    pipe.transformer = _RecordingTransformer()
    pipe.scheduler = _RecordingScheduler()
    captured = {"encoded_prompts": []}

    def _encode_prompt(prompt, device, max_sequence_length):
        captured["encoded_prompts"].append((list(prompt), device, max_sequence_length))
        sign = -1.0 if prompt and prompt[0] == "bad" else 1.0
        embeds = torch.full((len(prompt), 2, 3), sign, device=device)
        mask = torch.ones((len(prompt), 2), dtype=torch.bool, device=device)
        return embeds, mask

    def _prepare_latents(
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
    ):
        captured["prepared_latents"] = {
            "batch_size": batch_size,
            "num_channels_latents": num_channels_latents,
            "height": height,
            "width": width,
            "dtype": dtype,
            "device": device,
            "generator_device": generator.device,
        }
        return torch.ones((batch_size, 6, num_channels_latents * 4), dtype=dtype, device=device)

    def _decode_latents(latents, height, width):
        captured["decoded"] = {
            "latents_shape": tuple(latents.shape),
            "height": height,
            "width": width,
        }
        return torch.full((latents.shape[0], height, width, 3), 7, dtype=torch.uint8)

    pipe._encode_prompt = _encode_prompt
    pipe._prepare_latents = _prepare_latents
    pipe._decode_latents = _decode_latents
    return pipe, captured


def _expanded_noise(values, *, batch_size, seq_len, dtype=torch.float32):
    pattern = torch.tensor(values, dtype=dtype)
    return pattern.view(1, 1, -1).expand(batch_size, seq_len, -1)


def test_forward_runs_without_true_cfg():
    pipe, captured = _pipeline_with_test_doubles()

    output = pipe.forward(
        prompt=["a cat"],
        negative_prompt=None,
        height=32,
        width=48,
        num_inference_steps=3,
        true_cfg_scale=4.0,
        seed=123,
        max_sequence_length=16,
        sigmas=[1.0, 0.5, 0.25],
    )

    assert output.image.shape == (1, 32, 48, 3)
    assert output.image.dtype == torch.uint8
    assert captured["encoded_prompts"] == [(["a cat"], torch.device("cpu"), 16)]
    assert captured["prepared_latents"]["batch_size"] == 1
    assert captured["decoded"] == {"latents_shape": (1, 6, 4), "height": 32, "width": 48}
    assert len(pipe.transformer.calls) == 3
    assert pipe.transformer.calls[0]["img_shapes"] == [[(1, 2, 3)]]
    assert pipe.transformer.calls[0]["hidden_states_shape"] == (1, 6, 4)
    assert torch.all(pipe.transformer.calls[0]["encoder_hidden_states"] > 0)
    assert len(pipe.scheduler.step_calls) == 3
    assert torch.allclose(
        pipe.scheduler.step_calls[0]["noise_pred"],
        _expanded_noise([1, 2, 3, 4], batch_size=1, seq_len=6),
    )


def test_forward_runs_true_cfg_pipeline():
    pipe, captured = _pipeline_with_test_doubles()

    output = pipe.forward(
        prompt=["a cat", "a dog"],
        negative_prompt="bad",
        height=32,
        width=48,
        num_inference_steps=2,
        true_cfg_scale=3.0,
        seed=123,
        max_sequence_length=16,
        sigmas=[1.0, 0.5],
    )

    assert output.image.shape == (2, 32, 48, 3)
    assert output.image.dtype == torch.uint8
    assert captured["encoded_prompts"] == [
        (["a cat", "a dog"], torch.device("cpu"), 16),
        (["bad", "bad"], torch.device("cpu"), 16),
    ]
    assert captured["prepared_latents"] == {
        "batch_size": 2,
        "num_channels_latents": 1,
        "height": 32,
        "width": 48,
        "dtype": torch.float32,
        "device": torch.device("cpu"),
        "generator_device": torch.device("cpu"),
    }
    assert captured["decoded"] == {"latents_shape": (2, 6, 4), "height": 32, "width": 48}
    assert len(pipe.transformer.calls) == 4
    assert pipe.transformer.calls[0]["img_shapes"] == [[(1, 2, 3)]] * 2
    assert pipe.transformer.calls[0]["hidden_states_shape"] == (2, 6, 4)
    assert torch.all(pipe.transformer.calls[0]["encoder_hidden_states"] > 0)
    assert torch.all(pipe.transformer.calls[1]["encoder_hidden_states"] < 0)
    assert pipe.transformer.calls[0]["return_dict"] is False
    assert pipe.scheduler.set_timesteps_call["sigmas"] == [1.0, 0.5]
    assert pipe.scheduler.set_timesteps_call["device"] == torch.device("cpu")
    assert pipe.scheduler.begin_index == 0
    assert len(pipe.scheduler.step_calls) == 2
    assert pipe.scheduler.step_calls[0]["noise_pred_shape"] == (2, 6, 4)
    cond_noise = _expanded_noise([1, 2, 3, 4], batch_size=2, seq_len=6)
    neg_noise = _expanded_noise([-0.5, 0.25, 1.5, -1.0], batch_size=2, seq_len=6)
    combined_noise = neg_noise + 3.0 * (cond_noise - neg_noise)
    expected_cfg_noise = combined_noise * (
        torch.norm(cond_noise, dim=-1, keepdim=True)
        / torch.norm(combined_noise, dim=-1, keepdim=True)
    )
    assert torch.allclose(pipe.scheduler.step_calls[0]["noise_pred"], expected_cfg_noise)
    assert pipe.scheduler.step_calls[0]["return_dict"] is False
