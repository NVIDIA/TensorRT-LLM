#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image text-to-image generation example using TensorRT-LLM VisualGen.

Phase 1 supports the native BF16 reference path -- FP8 / NVFP4 / CUDA
graph / CFG+Ulysses / TeaCache are scheduled for follow-up PRs (see the
feature matrix row in ``docs/source/models/visual-generation.md``).

Single image:

    python visual_gen_qwen_image.py --model_path Qwen/Qwen-Image \\
        --prompt "A cat holding a sign that says hello world"

Batch (one prompt per line in prompts.txt):

    python visual_gen_qwen_image.py --model_path Qwen/Qwen-Image \\
        --prompts_file prompts.txt --output_dir out/
"""

import argparse
import json
import os
import time

from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams, logger
from tensorrt_llm.serve.media_storage import MediaStorage

logger.set_level("info")


def parse_args():
    parser = argparse.ArgumentParser(
        description="TRTLLM VisualGen - Qwen-Image Text-to-Image Inference Example"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Local path or HuggingFace Hub model ID (e.g. Qwen/Qwen-Image)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="HuggingFace Hub revision (branch, tag, or commit SHA)",
    )
    parser.add_argument(
        "--prompt", type=str, default=None, help="Single prompt (mutually exclusive with --prompts_file)"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Negative prompt for classifier-free guidance (enable with --true_cfg_scale > 1.0)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="qwen_image.png",
        help="Output image path (single-prompt mode)",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Text file with one prompt per line (batch mode)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for --prompts_file mode (required when --prompts_file is set)",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=None,
        help="Limit the number of prompts from --prompts_file",
    )
    parser.add_argument("--height", type=int, default=1328, help="Image height")
    parser.add_argument("--width", type=int, default=1328, help="Image width")
    parser.add_argument("--steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument(
        "--true_cfg_scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale (uses --negative_prompt when > 1.0)",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=1024,
        help="Maximum Qwen2.5-VL prompt token length (max 1024)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--attention_backend",
        type=str,
        default="VANILLA",
        choices=["VANILLA", "TRTLLM"],
        help="Attention backend (Phase 1 defaults to VANILLA / torch SDPA)",
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="Ulysses sequence-parallel size (Phase 2; keep at 1 today)",
    )
    parser.add_argument(
        "--disable_torch_compile",
        action="store_true",
        help="Disable torch.compile (recommended during Phase 1 bring-up)",
    )
    parser.add_argument(
        "--disable_autotune",
        action="store_true",
        help="Disable autotuning during warmup",
    )
    parser.add_argument(
        "--skip_warmup",
        action="store_true",
        help="Skip the warmup forward after pipeline load (faster startup)",
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
    with open(prompts_file) as f:
        prompts = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    if num_prompts is not None:
        prompts = prompts[:num_prompts]
    return prompts


def build_diffusion_args(args) -> VisualGenArgs:
    return VisualGenArgs(
        revision=args.revision,
        attention={"backend": args.attention_backend},
        parallel={"dit_ulysses_size": args.ulysses_size},
        torch_compile={
            "enable_torch_compile": not args.disable_torch_compile,
            "enable_autotune": not args.disable_autotune,
        },
        cuda_graph={"enable_cuda_graph": False},
        skip_warmup=args.skip_warmup,
    )


def main():
    args = parse_args()

    diffusion_args = build_diffusion_args(args)
    logger.info("Initializing VisualGen for Qwen-Image")
    visual_gen = VisualGen(model=args.model_path, args=diffusion_args)

    params = VisualGenParams(
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.true_cfg_scale,
        max_sequence_length=args.max_sequence_length,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
    )

    try:
        if args.prompts_file:
            prompts = load_prompts(args.prompts_file, args.num_prompts)
            os.makedirs(args.output_dir, exist_ok=True)
            logger.info(
                f"Batch mode: {len(prompts)} prompts -> {args.output_dir} "
                f"at {args.height}x{args.width}, {args.steps} steps"
            )

            total_start = time.time()
            records = []
            for i, prompt in enumerate(prompts):
                logger.info(f"[{i + 1}/{len(prompts)}] {prompt[:60]}...")
                t0 = time.time()
                output = visual_gen.generate(
                    inputs=prompt, params=params.model_copy(update={"seed": args.seed + i})
                )
                elapsed = time.time() - t0
                output_path = os.path.join(args.output_dir, f"{i:02d}.png")
                MediaStorage.save_image(output.image, output_path)
                logger.info(f"  Saved {output_path} ({elapsed:.1f}s)")
                records.append(
                    {"index": i, "prompt": prompt, "time": round(elapsed, 2), "seed": args.seed + i}
                )

            total_elapsed = time.time() - total_start
            times = [r["time"] for r in records]
            timing_path = os.path.join(args.output_dir, "timing.json")
            with open(timing_path, "w") as f:
                json.dump(
                    {
                        "images": records,
                        "total_time": round(total_elapsed, 2),
                        "avg_time": round(sum(times) / len(times), 2) if times else 0.0,
                        "config": {
                            "model_path": args.model_path,
                            "height": args.height,
                            "width": args.width,
                            "steps": args.steps,
                            "true_cfg_scale": args.true_cfg_scale,
                        },
                    },
                    f,
                    indent=2,
                )
            logger.info(
                f"Batch complete: {len(prompts)} images in {total_elapsed:.1f}s "
                f"(avg {round(sum(times) / len(times), 2)}s/image); timing -> {timing_path}"
            )
        else:
            logger.info(f"Generating image for prompt: {args.prompt!r}")
            t0 = time.time()
            output = visual_gen.generate(inputs=args.prompt, params=params)
            elapsed = time.time() - t0
            MediaStorage.save_image(output.image, args.output_path)
            logger.info(f"Saved {args.output_path} ({elapsed:.1f}s)")
    finally:
        visual_gen.shutdown()


if __name__ == "__main__":
    main()
