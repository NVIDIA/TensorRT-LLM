# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Qwen-Image text-to-image generation example.

Builds the :class:`QwenImagePipeline` directly (bypassing
``AutoPipeline`` / ``DiffusionModelConfig`` to keep this first example
self-contained) and generates one image from a prompt.

Phase 2 will ship an ``AutoPipeline``-driven entry point with full
``trtllm-serve`` integration and FP8/NVFP4 quantization. Until then,
this script demonstrates the native BF16 path end-to-end.

Usage:

    python examples/visual_gen/visual_gen_qwen_image.py \\
        --checkpoint /path/to/Qwen-Image \\
        --prompt "A cat holding a sign that says hello world" \\
        --output qwen_image.png
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from safetensors.torch import load_file

from tensorrt_llm._torch.visual_gen.models.qwen_image import (
    QwenImagePipeline,
    QwenImageTransformer2DModel,
)


def _minimal_visualgen_config(dtype: torch.dtype) -> SimpleNamespace:
    """SimpleNamespace substitute for ``DiffusionModelConfig``."""
    return SimpleNamespace(
        torch_dtype=dtype,
        pretrained_config=None,
        cuda_graph=SimpleNamespace(enable_cuda_graph=False),
        torch_compile=SimpleNamespace(enable_torch_compile=False),
        enable_parallel_vae=False,
        cache_backend="none",
        mapping=None,
        visual_gen_mapping=None,
        compilation=SimpleNamespace(resolutions=None, num_frames=None),
        parallel_vae_split_dim=0,
        teacache=SimpleNamespace(),
        cache_dit=SimpleNamespace(),
    )


def _build_pipeline(
    checkpoint: Path, dtype: torch.dtype, device: torch.device
) -> QwenImagePipeline:
    pipe = QwenImagePipeline.__new__(QwenImagePipeline)
    torch.nn.Module.__init__(pipe)
    pipe.model_config = _minimal_visualgen_config(dtype)
    pipe.config = None
    pipe.mapping = None
    pipe._cuda_graph_runners = {}
    pipe._parallel_vae_enabled = False
    pipe._warmed_up_shapes = set()
    pipe.cache_accelerator = None
    pipe.transformer = None
    pipe.vae = None
    pipe.text_encoder = None
    pipe.tokenizer = None
    pipe.scheduler = None
    pipe.vae_scale_factor = 8
    pipe.tokenizer_max_length = 1024

    # Transformer.
    tcfg = json.loads((checkpoint / "transformer" / "config.json").read_text())
    pipe.transformer = (
        QwenImageTransformer2DModel.from_config_dict(tcfg).to(dtype).to(device).eval()
    )
    sd: dict[str, torch.Tensor] = {}
    for shard in sorted((checkpoint / "transformer").glob("*.safetensors")):
        sd.update(load_file(str(shard)))
    pipe.transformer.load_weights(sd)
    del sd

    # VAE / text encoder / tokenizer / scheduler.
    pipe.load_standard_components(str(checkpoint), device=device)
    return pipe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to the Qwen-Image diffusers directory")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative_prompt", default="")
    p.add_argument("--height", type=int, default=1328)
    p.add_argument("--width", type=int, default=1328)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--true_cfg_scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--output", default="qwen_image.png")
    args = p.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    device = torch.device("cuda")

    pipe = _build_pipeline(Path(args.checkpoint), dtype, device)

    print(
        f"[generate] prompt={args.prompt!r} "
        f"{args.height}x{args.width}, {args.steps} steps, seed={args.seed}"
    )
    t0 = time.time()
    media = pipe.forward(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or None,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        true_cfg_scale=args.true_cfg_scale,
        seed=args.seed,
    )
    print(f"[generate] done in {time.time() - t0:.1f}s")

    image = media.image[0].cpu().numpy()
    Image.fromarray(image).save(args.output)
    print(f"[generate] wrote {args.output}")


if __name__ == "__main__":
    main()
