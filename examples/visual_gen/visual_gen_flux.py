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

from output_handler import OutputHandler

from tensorrt_llm import logger
from tensorrt_llm.llmapi.visual_gen import VisualGen, VisualGenParams

# Set logger level to ensure timing logs are printed
logger.set_level("info")


def parse_args():
    parser = argparse.ArgumentParser(
        description="TRTLLM VisualGen - FLUX Text-to-Image Inference Example (FLUX.1 / FLUX.2)"
    )

    # Model & Input
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Local path or HuggingFace Hub model ID "
        "(e.g., black-forest-labs/FLUX.1-dev, black-forest-labs/FLUX.2-dev)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="HuggingFace Hub revision (branch, tag, or commit SHA)",
    )

    # Single image mode
    parser.add_argument(
        "--prompt", type=str, default=None, help="Text prompt for single image generation"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output.png",
        help="Path to save the output image (single image mode)",
    )

    # Batch mode
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

    # Generation Params
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="Embedded guidance scale (3.5 for FLUX.1-dev, 4.0 for FLUX.2-dev)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # TeaCache Arguments
    parser.add_argument(
        "--enable_teacache", action="store_true", help="Enable TeaCache acceleration"
    )
    parser.add_argument(
        "--teacache_thresh",
        type=float,
        default=0.2,
        help="TeaCache similarity threshold (rel_l1_thresh)",
    )

    # Quantization
    parser.add_argument(
        "--linear_type",
        type=str,
        default="default",
        choices=["default", "trtllm-fp8-per-tensor", "trtllm-fp8-blockwise", "svd-nvfp4"],
        help="Linear layer quantization type",
    )

    # Attention Backend
    parser.add_argument(
        "--attention_backend",
        type=str,
        default="VANILLA",
        choices=["VANILLA", "TRTLLM", "FA4"],
        help="Attention backend (VANILLA: PyTorch SDPA, TRTLLM: optimized kernels, "
        "FA4: Flash Attention 4). "
        "Note: TRTLLM and FA4 automatically fall back to VANILLA for cross-attention.",
    )

    # torch.compile
    parser.add_argument(
        "--disable_torch_compile", action="store_true", help="Disable TorchCompile acceleration"
    )
    parser.add_argument(
        "--torch_compile_mode",
        type=str,
        default="default",
        help="Torch compile mode",
        choices=["default", "max-autotune", "reduce-overhead"],
    )

    # Warmup
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1,
        help="Number of warmup steps (0 to disable)",
    )

    # Parallelism
    parser.add_argument(
        "--cfg_size",
        type=int,
        default=1,
        choices=[1, 2],
        help="CFG parallel size (1 or 2). Set to 2 for CFG Parallelism.",
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="Ulysses (sequence) parallel size within each CFG group.",
    )

    args = parser.parse_args()

    # Validate: either --prompt or --prompts_file is required
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


def build_diffusion_config(args):
    """Build diffusion_config dict from parsed args."""
    # Convert linear_type to quant_config
    quant_config = None
    if args.linear_type == "trtllm-fp8-per-tensor":
        quant_config = {"quant_algo": "FP8", "dynamic": True}
    elif args.linear_type == "trtllm-fp8-blockwise":
        quant_config = {"quant_algo": "FP8_BLOCK_SCALES", "dynamic": True}
    elif args.linear_type == "svd-nvfp4":
        quant_config = {"quant_algo": "NVFP4", "dynamic": True}

    # Note: pipeline type (FLUX.1 vs FLUX.2) is auto-detected from model_index.json
    diffusion_config = {
        "revision": args.revision,
        "attention": {
            "backend": args.attention_backend,
        },
        "teacache": {
            "enable_teacache": args.enable_teacache,
            "teacache_thresh": args.teacache_thresh,
        },
        "parallel": {
            "dit_cfg_size": args.cfg_size,
            "dit_ulysses_size": args.ulysses_size,
        },
        "pipeline": {
            "enable_torch_compile": not args.disable_torch_compile,
            "torch_compile_mode": args.torch_compile_mode,
            "warmup_steps": args.warmup_steps,
        },
    }

    if quant_config is not None:
        diffusion_config["quant_config"] = quant_config

    return diffusion_config


def main():
    args = parse_args()

    # world_size = cfg_size * ulysses_size
    n_workers = args.cfg_size * args.ulysses_size

    diffusion_config = build_diffusion_config(args)

    # Initialize VisualGen
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
            # Batch mode
            prompts = load_prompts(args.prompts_file, args.num_prompts)
            os.makedirs(args.output_dir, exist_ok=True)

            logger.info(f"Batch mode: {len(prompts)} prompts → {args.output_dir}")
            logger.info(f"Resolution: {args.height}x{args.width}, Steps: {args.steps}")

            timing_records = []
            total_start = time.time()

            for i, prompt in enumerate(prompts):
                logger.info(f"[{i + 1}/{len(prompts)}] {prompt[:60]}...")
                start_time = time.time()

                output = visual_gen.generate(
                    inputs=prompt,
                    params=VisualGenParams(
                        height=args.height,
                        width=args.width,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance_scale,
                        seed=args.seed + i,
                    ),
                )

                elapsed = time.time() - start_time
                output_path = os.path.join(args.output_dir, f"{i:02d}.png")
                OutputHandler.save(output, output_path)
                logger.info(f"  Saved {output_path} ({elapsed:.1f}s)")

                timing_records.append(
                    {
                        "index": i,
                        "prompt": prompt,
                        "time": round(elapsed, 2),
                        "seed": args.seed + i,
                    }
                )

            total_elapsed = time.time() - total_start
            times = [r["time"] for r in timing_records]

            # Write timing metadata
            timing_data = {
                "images": timing_records,
                "total_time": round(total_elapsed, 2),
                "avg_time": round(sum(times) / len(times), 2) if times else 0,
                "config": {
                    "model_path": args.model_path,
                    "linear_type": args.linear_type,
                    "attention_backend": args.attention_backend,
                    "height": args.height,
                    "width": args.width,
                    "steps": args.steps,
                    "guidance_scale": args.guidance_scale,
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

        else:
            # Single image mode
            logger.info(f"Generating image for prompt: '{args.prompt}'")
            logger.info(f"Resolution: {args.height}x{args.width}, Steps: {args.steps}")

            start_time = time.time()

            output = visual_gen.generate(
                inputs=args.prompt,
                params=VisualGenParams(
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    seed=args.seed,
                ),
            )

            end_time = time.time()
            logger.info(f"Generation completed in {end_time - start_time:.2f}s")

            OutputHandler.save(output, args.output_path)

    finally:
        visual_gen.shutdown()


if __name__ == "__main__":
    main()
