# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``QwenImagePipeline.infer`` request orchestration.

Covers batch fan-out (``num_images_per_prompt``, multiple prompts),
negative-prompt broadcasting/validation, and generation-parameter pass-through.
``forward`` is stubbed to capture the arguments it receives, so these run on
CPU with no model or weights.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.visual_gen.models.qwen_image import QwenImagePipeline


def _pipeline_with_captured_forward():
    pipe = QwenImagePipeline.__new__(QwenImagePipeline)
    captured = {}
    pipe.forward = lambda **kwargs: captured.update(kwargs) or "image"
    return pipe, captured


def _req(
    prompt,
    *,
    num_images_per_prompt=1,
    negative_prompt=None,
    height=1328,
    width=1328,
    num_inference_steps=2,
    guidance_scale=4.0,
    seed=42,
    max_sequence_length=64,
):
    params = SimpleNamespace(
        num_images_per_prompt=num_images_per_prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        max_sequence_length=max_sequence_length,
    )
    return SimpleNamespace(prompt=prompt, params=params)


@pytest.mark.parametrize(
    ("prompt", "num_per", "negative", "exp_prompts", "exp_negs"),
    [
        ("a cat", 1, None, ["a cat"], None),
        ("a cat", 3, None, ["a cat", "a cat", "a cat"], None),
        (["a", "b"], 2, None, ["a", "a", "b", "b"], None),
        (["a", "b"], 2, "bad", ["a", "a", "b", "b"], ["bad", "bad", "bad", "bad"]),
        (["a", "b"], 2, ["bad"], ["a", "a", "b", "b"], ["bad", "bad", "bad", "bad"]),
        (["a", "b"], 2, ["n0", "n1"], ["a", "a", "b", "b"], ["n0", "n0", "n1", "n1"]),
    ],
)
def test_infer_fans_out_prompts_and_negatives(prompt, num_per, negative, exp_prompts, exp_negs):
    pipe, captured = _pipeline_with_captured_forward()
    result = pipe.infer(_req(prompt, num_images_per_prompt=num_per, negative_prompt=negative))
    assert result == "image"
    assert captured["prompt"] == exp_prompts
    assert captured["negative_prompt"] == exp_negs


def test_infer_forwards_generation_params():
    pipe, captured = _pipeline_with_captured_forward()
    pipe.infer(
        _req(
            "a cat",
            height=768,
            width=1024,
            num_inference_steps=7,
            guidance_scale=3.25,
            seed=123,
            max_sequence_length=256,
        )
    )
    assert captured["height"] == 768
    assert captured["width"] == 1024
    assert captured["num_inference_steps"] == 7
    assert captured["true_cfg_scale"] == 3.25
    assert captured["seed"] == 123
    assert captured["max_sequence_length"] == 256


def test_infer_rejects_mismatched_negative_prompt_count():
    pipe, _ = _pipeline_with_captured_forward()
    req = _req(["a", "b", "c"], negative_prompt=["n0", "n1"])
    with pytest.raises(ValueError, match="negative_prompt"):
        pipe.infer(req)
