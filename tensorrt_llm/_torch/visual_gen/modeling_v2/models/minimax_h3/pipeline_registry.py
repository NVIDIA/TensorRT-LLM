# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_llm._torch.visual_gen.pipeline_registry import register_pipeline

from .checkpoint import load_fasth3_manifest
from .fasth3.pipeline import FastH3Pipeline, configure_checkpoint
from .h3_pipeline.pipeline import MiniMaxH3Pipeline


def route(config, checkpoint_dir):
    manifest = load_fasth3_manifest(checkpoint_dir) if checkpoint_dir else {}
    if manifest:
        configure_checkpoint(config, manifest)
        return FastH3Pipeline
    return MiniMaxH3Pipeline


register_pipeline(
    "MiniMaxH3ModularPipeline",
    hf_ids=[
        "MiniMaxAI/MiniMax-H3",
        "FastVideo/FastVideo-FastH3-8-Step-V2",
    ],
    defaults={"workflow": "fl2va"},
    download_patterns=[
        "modular_model_index.json",
        "fastvideo_inference.json",
        "checkpoint_content.json",
        "checkpoint_metadata.json",
        "LICENSE",
        "README.md",
        "transformer/*",
        "transformer_ref/*",
        "text_encoder/*",
        "tokenizer/*",
        "processor/*",
        "vae/*",
        "audio_vae/*",
        "scheduler/*",
        "audio_scheduler/*",
    ],
    doc=(
        "MiniMax-H3 initial BF16 support for text-to-video-with-audio and "
        "first/last-frame FL2VA or image/video/audio-conditioned Ref2VA."
    ),
    router=route,
)(MiniMaxH3Pipeline)
