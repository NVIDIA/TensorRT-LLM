# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture the FastH3 QA reference with a baseline installation in a separate process."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm.visual_gen.args import VisualGenArgs

MODEL = "FastVideo/FastVideo-FastH3-8-Step-V2"


def generate(checkpoint: Path, backend: str) -> dict:
    manifest_bytes = (checkpoint / "fastvideo_inference.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    steps = manifest["dmd_denoising_steps"]
    if not manifest["schema_version"].startswith("fasth3-") or len(steps) != 8:
        raise ValueError(f"Expected the eight-forward {MODEL} checkpoint in {checkpoint}")
    params = dict(
        prompt="An astronaut walks through a rainforest at dusk with soft music and birdsong.",
        height=384,
        width=384,
        num_frames=124,
        frame_rate=24.0,
        num_inference_steps=len(steps) + 1,
        seed=42,
    )
    pipeline = PipelineLoader(
        VisualGenArgs(
            model=str(checkpoint),
            attention_config={"backend": backend},
            torch_compile_config={"enable": False},
            cuda_graph_config={"enable": False},
        )
    ).load(skip_warmup=True)
    assert pipeline.pipeline_config.attention.backend == backend
    assert pipeline.pipeline_config.attention.sparse_attention_config.algorithm == "vsa"
    with torch.inference_mode():
        output = pipeline.forward(**params)
    return {
        "case": {
            "model": MODEL,
            "backend": backend,
            "sm": list(torch.cuda.get_device_capability()),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "request": params,
        },
        "target": f"{type(pipeline).__module__}.{type(pipeline).__name__}",
        "video": output.video.detach().cpu(),
        "audio": output.audio.detach().cpu(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--backend", choices=("TRTLLM", "CUTEDSL"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = generate(args.checkpoint, args.backend)
    torch.save(result, args.output)
