# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public VDN example smoke test; requires the released VDN and H3 checkpoints."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vdn_h3_example(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[5]
    models = Path(os.environ["LLM_MODELS_ROOT"])
    vdn = Path(os.environ.get("VDN_H3_CHECKPOINT", str(models / "vdn-minimax-h3")))
    h3 = Path(os.environ.get("MINIMAX_H3_CHECKPOINT", str(models / "MiniMax-H3")))
    assert (vdn / "stage-dmd-step-250/linear_branch/model.safetensors").is_file()
    assert (h3 / "transformer/config.json").is_file()
    output = tmp_path / "vdn.pt"
    subprocess.run(
        [
            sys.executable,
            str(root / "examples/visual_gen/models/vdn_minimax_h3.py"),
            "--model",
            str(vdn),
            "--base_checkpoint",
            str(h3),
            "--output",
            str(output),
        ],
        check=True,
        timeout=1500,
    )
    media = torch.load(output, map_location="cpu", weights_only=True)
    assert media["video"].shape == (124, 768, 1344, 3)
    assert media["video"].float().std() > 1.0
    assert media["audio"].shape[0] == 2
    assert torch.isfinite(media["audio"]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vdn_h3_cached_prompt_reference() -> None:
    """Compare decoded media to an upstream VDN golden at the same seed/sigma grid.

    The fixture uses the independent upstream renderer and prompts/example_0.pt, 1344x768, 124 frames, seed 0,
    and nine sigma points (eight evaluations). Missing CI assets fail loudly.
    """
    from defs.examples.visual_gen.visual_gen_test_utils import (
        _disable_inductor_compile_worker_quiesce,
    )

    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent
    from tensorrt_llm.visual_gen.args import VisualGenArgs

    _disable_inductor_compile_worker_quiesce()
    models = Path(os.environ["LLM_MODELS_ROOT"])
    vdn = Path(os.environ.get("VDN_H3_CHECKPOINT", str(models / "vdn-minimax-h3")))
    base = os.environ.get("MINIMAX_H3_CHECKPOINT", str(models / "MiniMax-H3"))
    assets = Path(os.environ.get("VDN_H3_VALIDATION_DIR", str(vdn / "validation")))
    prompt = torch.load(assets / "prompt.pt", map_location="cpu", weights_only=True)
    expected = torch.load(assets / "reference.pt", map_location="cpu", weights_only=True)
    args = VisualGenArgs(
        model=str(vdn),
        pipeline_config={"base_checkpoint": base},
        attention_config={"backend": "FA4"},
        cuda_graph_config={"enable": False},
        torch_compile_config={"enable": True, "enable_fullgraph": False},
        compilation_config={"skip_warmup": True},
    )
    pipeline = PipelineLoader(args).load(
        skip_warmup=True,
        skip_components=[
            PipelineComponent.TEXT_ENCODER,
            PipelineComponent.TOKENIZER,
            PipelineComponent.PROCESSOR,
        ],
    )
    embeds = prompt["prompt_embeds"]
    if embeds.ndim == 2:
        embeds = embeds.unsqueeze(0)
    pipeline._encode_prompt = lambda prompt_text, keyframes: (
        embeds.cuda().bfloat16(),
        prompt["text_token_tags"],
    )
    with torch.inference_mode():
        result = pipeline.forward(
            prompt="upstream cached example_0",
            seed=0,
            height=768,
            width=1344,
            num_frames=124,
            frame_rate=24.0,
            num_inference_steps=9,
        )
    from test_minimax_h3_e2e import _mean_lpips_distance

    video = result.video.cpu()
    lpips_distance = _mean_lpips_distance(
        video, expected["video"].permute(0, 1, 4, 2, 3).float().div(255)
    )
    audio = result.audio.cpu().float()
    assert video.shape == expected["video"].shape
    assert audio.shape == expected["audio"].shape
    audio_relative_l2 = (audio - expected["audio"].float()).norm() / expected[
        "audio"
    ].float().norm().clamp_min(1e-8)
    print(f"VDN LPIPS={lpips_distance:.6f}, audio relative L2={audio_relative_l2.item():.6f}")
    assert lpips_distance < 0.15
    assert audio_relative_l2 < 0.1
