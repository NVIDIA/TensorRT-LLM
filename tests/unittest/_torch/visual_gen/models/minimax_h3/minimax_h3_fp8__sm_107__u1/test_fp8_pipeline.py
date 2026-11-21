# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM107 end-to-end QA for the native static-FP8 H3 scenario."""

import json
import os
from pathlib import Path

import pytest
import torch
from test_common.llm_data import get_checkpoint


def test_static_fp8_pipeline_generates_finite_video():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("The fused static-FP8 scenario requires an SM107 GPU")
    if torch.cuda.get_device_properties(0).multi_processor_count != 212:
        pytest.skip("The fused static-FP8 scenario requires the qualified 212-SM GPU")
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        pytest.skip("The static-FP8 scenario requires NVIDIA_TF32_OVERRIDE=0 at process start")
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.visual_gen.args import VisualGenArgs

    repo = Path(__file__).resolve().parents[7]
    request = json.loads(
        (repo / "examples/visual_gen/requests/minimax-h3-768x1344x124.json").read_text()
    )
    try:
        checkpoint = get_checkpoint("MiniMax-H3-FP8")
    except (AssertionError, FileNotFoundError) as error:
        pytest.skip(f"The static-FP8 checkpoint is not staged: {error}")
    args = VisualGenArgs(
        model=checkpoint,
        pipeline_config={"visual_gen_scenario": "require"},
        attention_config={
            "backend": "CUDNN",
            "quant_attention_config": {"qk_dtype": "fp8", "v_dtype": "fp8"},
        },
        parallel_config={"ulysses_size": 1, "tp_size": 1, "parallel_vae_size": 1},
        torch_compile_config={"enable": True},
        cuda_graph_config={"enable": False},
        compilation_config={"resolutions": [[768, 1344]], "num_frames": [124]},
        cache_config=None,
    )
    pipeline = PipelineLoader(args).load()
    assert type(pipeline).__name__ == "MiniMaxH3Fp8Sm107U1Pipeline"
    # Reduced denoising steps keep the regression focused on the complete path;
    # full 50-step performance/LPIPS studies use the report's fixed request.
    output = pipeline.forward(
        **{
            k: request[k] for k in ["prompt", "seed", "height", "width", "num_frames", "frame_rate"]
        },
        num_inference_steps=8,
    )
    assert output.video.dtype == torch.uint8
    assert tuple(output.video.shape) == (1, 124, 768, 1344, 3)
    assert int(output.video.max()) > int(output.video.min())
    assert tuple(output.audio.shape) == (1, 2, 165600)
    assert torch.isfinite(output.audio).all()
    assert output.audio_sample_rate == 32000
    # The conversion kernels are optional: without the CUDA toolchain the installed operators run.
    assert "unary_qualified" in pipeline.transformer.fp8_kernel_qualification
    assert pipeline.transformer.fp8_conditioning_cache.key_reads > 0
