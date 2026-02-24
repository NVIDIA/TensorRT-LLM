#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLUX Text-to-Image generation using TensorRT-LLM Visual Generation.

Supports both FLUX.1 and FLUX.2 models. The pipeline type is auto-detected
from the model checkpoint (model_index.json).

Single image mode:
    python visual_gen_flux.py --model_path black-forest-labs/FLUX.1-dev \
        --prompt "A cat sitting on a windowsill" --guidance_scale 3.5

    python visual_gen_flux.py --model_path black-forest-labs/FLUX.2-dev \
        --prompt "A cat sitting on a windowsill" --guidance_scale 4.0

    # With FP8 quantization
    python visual_gen_flux.py --model_path black-forest-labs/FLUX.2-dev \
        --prompt "A cat" --linear_type trtllm-fp8-per-tensor

Batch mode (generates multiple images from a prompts file):
    python visual_gen_flux.py --model_path black-forest-labs/FLUX.1-dev \
        --prompts_file prompts.txt --output_dir results/bf16/ --seed 42

    # With FP8 quantization
    python visual_gen_flux.py --model_path black-forest-labs/FLUX.2-dev \
        --prompts_file prompts.txt --output_dir results/fp8/ \
        --linear_type trtllm-fp8-per-tensor

    # Multi-GPU with CFG + Ulysses parallelism
    python visual_gen_flux.py --model_path black-forest-labs/FLUX.1-dev \
        --prompts_file prompts.txt --output_dir results/ \
        --cfg_size 2 --ulysses_size 2
"""

import argparse
import json
import os
import time

from example_utils import add_common_args, build_diffusion_config, build_generation_params
from output_handler import OutputHandler

from tensorrt_llm import logger
from tensorrt_llm.llmapi.visual_gen import VisualGen

logger.set_level("info")

FLUX_DEFAULTS = {
    "height": 1024,
    "width": 1024,
    "steps": 50,
    "guidance_scale": 3.5,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="TRTLLM VisualGen - FLUX Text-to-Image Inference Example (FLUX.1 / FLUX.2)"
    )
    add_common_args(parser, prompt_required=False)

    # Batch mode (FLUX-specific)
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="File with prompts (one per line) for batch generation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for batch mode (images named 00.png, 01.png, ...)",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=None,
        help="Limit number of prompts from file (batch mode)",
    )

    args = parser.parse_args()

    if args.prompt is None and args.prompts_file is None:
        parser.error("Either --prompt or --prompts_file is required")
    if args.prompt is not None and args.prompts_file is not None:
        parser.error("--prompt and --prompts_file are mutually exclusive")
    if args.prompts_file is not None and args.output_dir is None:
        parser.error("--output_dir is required when using --prompts_file")

    return args


def load_prompts(prompts_file, num_prompts=None):
    """Load prompts from file (one per line, skip empty/comments)."""
    with open(prompts_file) as f:
        prompts = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    if num_prompts is not None:
        prompts = prompts[:num_prompts]
    return prompts


def main():
    args = parse_args()

    n_workers = args.cfg_size * args.ulysses_size
    diffusion_config = build_diffusion_config(args)

    logger.info(
        f"Initializing VisualGen: world_size={n_workers} "
        f"(cfg_size={args.cfg_size}, ulysses_size={args.ulysses_size})"
    )
    visual_gen = VisualGen(
        model_path=args.model_path,
        n_workers=n_workers,
        diffusion_config=diffusion_config,
    )

    try:
        if args.prompts_file:
            _run_batch(args, visual_gen)
        else:
            _run_single(args, visual_gen)
    finally:
        visual_gen.shutdown()


def _run_single(args, visual_gen):
    """Single image mode."""
    params = build_generation_params(args, defaults=FLUX_DEFAULTS)

    logger.info(f"Generating image for prompt: '{args.prompt}'")
    logger.info(f"Resolution: {params.height}x{params.width}, Steps: {params.num_inference_steps}")

    start_time = time.time()

    output = visual_gen.generate(inputs=args.prompt, params=params)

    logger.info(f"Generation completed in {time.time() - start_time:.2f}s")

    OutputHandler.save(output, args.output_path)


def _run_batch(args, visual_gen):
    """Batch mode: generate images for multiple prompts with timing metadata."""
    prompts = load_prompts(args.prompts_file, args.num_prompts)
    os.makedirs(args.output_dir, exist_ok=True)

    base_params = build_generation_params(args, defaults=FLUX_DEFAULTS)
    logger.info(f"Batch mode: {len(prompts)} prompts -> {args.output_dir}")
    logger.info(
        f"Resolution: {base_params.height}x{base_params.width}, "
        f"Steps: {base_params.num_inference_steps}"
    )

    timing_records = []
    total_start = time.time()

    for i, prompt in enumerate(prompts):
        logger.info(f"[{i + 1}/{len(prompts)}] {prompt[:60]}...")
        start_time = time.time()

        params = build_generation_params(args, defaults=FLUX_DEFAULTS, seed=args.seed + i)
        output = visual_gen.generate(inputs=prompt, params=params)

        elapsed = time.time() - start_time
        output_path = os.path.join(args.output_dir, f"{i:02d}.png")
        OutputHandler.save(output, output_path)
        logger.info(f"  Saved {output_path} ({elapsed:.1f}s)")

        timing_records.append(
            {"index": i, "prompt": prompt, "time": round(elapsed, 2), "seed": args.seed + i}
        )

    total_elapsed = time.time() - total_start
    times = [r["time"] for r in timing_records]

    timing_data = {
        "images": timing_records,
        "total_time": round(total_elapsed, 2),
        "avg_time": round(sum(times) / len(times), 2) if times else 0,
        "config": {
            "model_path": args.model_path,
            "linear_type": args.linear_type,
            "attention_backend": args.attention_backend,
            "height": base_params.height,
            "width": base_params.width,
            "steps": base_params.num_inference_steps,
            "guidance_scale": base_params.guidance_scale,
        },
    }
    timing_path = os.path.join(args.output_dir, "timing.json")
    with open(timing_path, "w") as f:
        json.dump(timing_data, f, indent=2)

    logger.info(
        f"Batch complete: {len(prompts)} images in {total_elapsed:.1f}s "
        f"(avg {timing_data['avg_time']:.1f}s/image)"
    )
    logger.info(f"Timing saved to {timing_path}")


if __name__ == "__main__":
    main()
