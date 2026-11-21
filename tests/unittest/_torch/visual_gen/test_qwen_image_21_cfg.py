# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.qwen_image_21 import pipeline_qwen_image_21 as module


class _Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(in_channels=4, causal_condition=True)
        self.transformer_blocks = [None]
        self.calls = []

    def cache_context(self, branch):
        self.branch = branch
        return nullcontext()

    def forward(self, **kwargs):
        self.calls.append((self.branch, kwargs))
        value = 2 if self.branch == "cond" else 1
        return (torch.full_like(kwargs["hidden_states"], value),)


def _pipeline(monkeypatch, rank=None):
    pipe = module.QwenImage21Pipeline.__new__(module.QwenImage21Pipeline)
    torch.nn.Module.__init__(pipe)
    pipe.transformer = _Transformer()
    pipe.vae_scale_factor = 16
    pipe.latent_channels = 4
    pipe.pipeline_config = SimpleNamespace(visual_gen_mapping=None)
    pipe.encoded = []

    def encode(prompt, device, max_length, image=None):
        pipe.encoded.append(prompt)
        length = 2 if len(pipe.encoded) == 1 else 3
        return (
            torch.zeros(len(prompt), length, 4),
            torch.ones(len(prompt), length),
            torch.zeros(len(prompt), length, dtype=torch.bool),
        )

    pipe._encode_prompt = encode
    pipe._prepare_latents = lambda batch, *args: (torch.zeros(batch, 4, 4), None)
    pipe._decode_latents = lambda latents, *args: latents
    pipe._profile_denoise_steps = enumerate
    pipe.predictions = []

    def step(prediction, timestep, latents, **kwargs):
        pipe.predictions.append(prediction.clone())
        return (latents - prediction,)

    pipe.scheduler = SimpleNamespace(
        config={},
        timesteps=torch.tensor([1000.0, 500.0]),
        set_timesteps=lambda **kwargs: None,
        set_begin_index=lambda index: None,
        step=step,
    )
    timer = SimpleNamespace(
        mark_pre_start=lambda: None,
        mark_denoise_start=lambda: None,
        mark_post_start=lambda: None,
        mark_end=lambda: None,
        fill=lambda output: output,
    )
    monkeypatch.setattr(module, "CudaPhaseTimer", lambda: timer)
    if rank is not None:
        pipe._cfg_parallel_state = lambda enabled: (enabled, 2, rank, "cfg_group")

        def gather(outputs, local, group):
            assert group == "cfg_group"
            torch.testing.assert_close(local, torch.full_like(local, 2 if rank == 0 else 1))
            outputs[0].fill_(2)
            outputs[1].fill_(1)

        monkeypatch.setattr(module.dist, "all_gather", gather)
    return pipe


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("rank", [None, 0, 1])
def test_cfg_predictions_and_branch_caches(monkeypatch, cache, rank):
    pipe = _pipeline(monkeypatch, rank)
    pipe.forward("positive", negative_prompt="negative", true_cfg_scale=3, use_kv_cache=cache)
    assert pipe.encoded == [["positive"], ["negative"]]
    for prediction in pipe.predictions:
        torch.testing.assert_close(prediction, torch.full_like(prediction, 4))
    calls = pipe.transformer.calls
    assert len(calls) == (4 if rank is None else 2)
    by_branch = {}
    for branch, kwargs in calls:
        by_branch.setdefault(branch, []).append(kwargs)
        assert kwargs["encoder_hidden_states"].shape[1] == (2 if branch == "cond" else 3)
        assert kwargs["img_mask"].shape[1] == (3 if branch == "cond" else 4)
    for branch_calls in by_branch.values():
        assert [c["kv_cache_mode"] for c in branch_calls] == (
            ["extract", "cached"] if cache else [None, None]
        )
        assert branch_calls[0]["kv_cache"] is branch_calls[1]["kv_cache"]
    if cache and rank is None:
        assert by_branch["cond"][0]["kv_cache"] is not by_branch["uncond"][0]["kv_cache"]


@pytest.mark.parametrize(
    "negative,scale,override,enabled",
    [
        (None, 1, None, False),
        (None, 3, None, False),
        ("bad", 1, None, False),
        ("", 3, None, True),
        ("bad", 3, 1, False),
        ("bad", 1, 3, True),
    ],
)
def test_cfg_activation_matches_diffusers(monkeypatch, negative, scale, override, enabled):
    pipe = _pipeline(monkeypatch)
    pipe.forward(
        "positive",
        negative_prompt=negative,
        negative_prompt_cfg_scale=scale,
        true_cfg_scale=override,
    )
    assert len(pipe.transformer.calls) == (4 if enabled else 2)


def test_infer_expands_and_forwards_negative_prompts():
    pipe = module.QwenImage21Pipeline.__new__(module.QwenImage21Pipeline)
    pipe.forward = lambda **kwargs: kwargs
    params = SimpleNamespace(
        num_images_per_prompt=2,
        negative_prompt=["n0", "n1"],
        extra_params={},
        image_reference=[],
        height=32,
        width=32,
        num_inference_steps=2,
        guidance_scale=3,
        seed=42,
        max_sequence_length=64,
    )
    result = pipe.infer(SimpleNamespace(prompt=["p0", "p1"], params=params))
    assert result["prompt"] == ["p0", "p0", "p1", "p1"]
    assert result["negative_prompt"] == ["n0", "n0", "n1", "n1"]
    assert result["negative_prompt_cfg_scale"] == 3
    params.negative_prompt = ["a", "b", "c"]
    with pytest.raises(ValueError, match="negative_prompt"):
        pipe.infer(SimpleNamespace(prompt=["p0", "p1"], params=params))
