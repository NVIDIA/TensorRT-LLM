# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request batching through the public API, worker IPC and native image pipelines."""

import pytest
import torch
from test_common.llm_data import get_checkpoint

from defs.examples.visual_gen.visual_gen_test_utils import (
    _assert_lpips_below_threshold,
    _golden_media_path,
    _run_lpips_eval,
)
from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "checkpoint,cuda_graph,num_images,golden,steps,compile_enabled",
    [
        ("FLUX.1-dev", False, 1, "flux1_lpips_golden.png", 4, False),
        ("FLUX.1-dev", True, 1, "flux1_lpips_golden.png", 4, False),
        ("FLUX.1-dev", False, 2, None, 4, False),
        ("FLUX.2-dev", False, 1, "flux2_lpips_golden.png", 4, False),
        ("qwen-image", False, 2, None, 4, False),
        ("qwen-image", True, 2, None, 4, False),
        ("qwen-image", False, 2, None, 50, False),
        ("qwen-image", False, 2, None, 50, True),
        ("qwen-image", True, 1, None, 4, True),
    ],
    ids=[
        "flux1",
        "flux1-cuda-graph",
        "flux1-multiple-outputs",
        "flux2",
        "qwen-image",
        "qwen-image-cuda-graph",
        "qwen-image-50-steps",
        "qwen-image-compile-50-steps",
        "qwen-image-compile-cuda-graph",
    ],
)
def test_dynamic_batching_matches_singletons(
    checkpoint, cuda_graph, num_images, golden, steps, compile_enabled, tmp_path, capfd, monkeypatch
):
    monkeypatch.setenv("TLLM_LOG_LEVEL", "INFO")
    prompts = [
        "a tiny astronaut hatching from an egg on the moon",
        "A red fox resting under an oak tree in autumn, sunlight filtering through golden leaves.",
    ]
    params = [
        VisualGenParams(
            height=256,
            width=256,
            num_inference_steps=steps,
            guidance_scale=3.5,
            seed=seed,
            num_images_per_prompt=num_images,
        )
        for seed in (42, 17)
    ]
    args = VisualGenArgs(
        batching_config={"max_batch_size": 2 * num_images, "max_queue_delay_ms": 100},
        compilation_config={"skip_warmup": True},
        torch_compile_config={"enable": compile_enabled},
        cuda_graph_config={"enable": cuda_graph},
    )
    # Serial calls exercise the singleton path on the same weights/device/RNG mode.
    with VisualGen(model=get_checkpoint(checkpoint), args=args) as generator:
        baselines = [generator.generate(prompt, param) for prompt, param in zip(prompts, params)]
        handles = [
            generator.generate_async(prompt, param) for prompt, param in zip(prompts, params)
        ]
        results = [handle.result(timeout=600) for handle in handles]
        assert [out.request_id for out in results] == [handle.request_id for handle in handles]
        assert results[0].metrics.generation == results[1].metrics.generation
        for i, (baseline, result) in enumerate(zip(baselines, results)):
            assert result.image.shape == (num_images, 256, 256, 3)
            assert result.image.dtype == torch.uint8
            assert result.image.count_nonzero() > 0
            baseline.save([tmp_path / f"singleton_{i}_{j}.png" for j in range(num_images)])
            result.save([tmp_path / f"batched_{i}_{j}.png" for j in range(num_images)])

        # A new singleton shape after a batch must remain usable, including graph replay.
        trailing = generator.generate(prompts[0], params[0])
        assert trailing.image.shape == baselines[0].image.shape

    captured = capfd.readouterr()
    assert "Dynamic batch: requests=" in captured.out + captured.err
    for i, prompt in enumerate(prompts):
        for j in range(num_images):
            score = _run_lpips_eval(
                tmp_path,
                f"batch_{i}_{j}",
                "image",
                prompt,
                tmp_path / f"singleton_{i}_{j}.png",
                tmp_path / f"batched_{i}_{j}.png",
            )
            _assert_lpips_below_threshold(score, 0.05)
    if golden is not None:
        reference = _golden_media_path(tmp_path, golden, "Dynamic batching golden image")
        score = _run_lpips_eval(
            tmp_path,
            "batch_golden",
            "image",
            prompts[0],
            reference,
            tmp_path / "batched_0_0.png",
        )
        _assert_lpips_below_threshold(score, 0.05)
