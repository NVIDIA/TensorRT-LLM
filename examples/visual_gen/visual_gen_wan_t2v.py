#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WAN Text-to-Video generation using TensorRT-LLM Visual Generation."""

import argparse
import time

from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams, logger
from tensorrt_llm._torch.visual_gen.example_utils import (
    add_attention_backend_args,
    add_cache_args,
    add_optimization_args,
    add_quant_args,
    build_cache_config,
)
from tensorrt_llm._torch.visual_gen.utils import linear_type_to_quant_config

logger.set_level("info")


def parse_args():
    parser = argparse.ArgumentParser(
        description="TRTLLM VisualGen - Wan Text-to-Video Inference Example (supports Wan 2.1 and Wan 2.2)"
    )

    # Model & Input
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Local path or HuggingFace Hub model ID (e.g., Wan-AI/Wan2.1-T2V-1.3B-Diffusers)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="HuggingFace Hub revision (branch, tag, or commit SHA)",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Negative prompt. Default is model-specific.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output.png",
        help="Path to save the output image/video frame",
    )

    # Generation Params
    parser.add_argument(
        "--height", type=int, default=None, help="Video height (default: auto-detect)"
    )
    parser.add_argument(
        "--width", type=int, default=None, help="Video width (default: auto-detect)"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="Number of frames to generate (default: auto-detect)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of denoising steps (default: auto-detect, 50 for Wan2.1 and Wan 2.2 5B, 40 for Wan2.2 A14B)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="Guidance scale (default: auto-detect, 5.0 for Wan2.1 and Wan 2.2 5B, 4.0 for Wan2.2 A14B)",
    )
    parser.add_argument(
        "--guidance_scale_2",
        type=float,
        default=None,
        help="Second-stage guidance scale for Wan2.2 A14B two-stage denoising (default: 3.0)",
    )
    parser.add_argument(
        "--boundary_ratio",
        type=float,
        default=None,
        help="Custom boundary ratio for two-stage denoising (default: auto-detect)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    add_cache_args(parser)

    add_quant_args(parser)
    add_attention_backend_args(parser)

    # Parallelism
    parser.add_argument(
        "--cfg_size",
        type=int,
        default=1,
        choices=[1, 2],
        help="CFG parallel size (1 or 2). "
        "Distributes positive/negative prompts across GPUs. "
        "Example: cfg_size=2 on 4 GPUs -> 2 GPUs per prompt.",
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="Ulysses (head-sharding) parallel size within each CFG group. "
        "Requirements: num_heads (12) must be divisible by ulysses_size. "
        "Example: ulysses_size=2 on 4 GPUs with cfg_size=2 -> "
        "2 CFG groups x 2 Ulysses ranks = 4 GPUs total. "
        "Cannot be combined with --attn2d_row_size / --attn2d_col_size (not yet implemented).",
    )
    parser.add_argument(
        "--attn2d_row_size",
        type=int,
        default=1,
        help="Attention2D row mesh size (Q all-gather dimension). "
        "Can be set independently of --attn2d_col_size; asymmetric meshes (e.g. 1x4 or 4x1) are valid. "
        "Total context parallelism degree = attn2d_row_size * attn2d_col_size. "
        "Cannot be combined with --ulysses_size (not yet implemented).",
    )
    parser.add_argument(
        "--attn2d_col_size",
        type=int,
        default=1,
        help="Attention2D column mesh size (K/V all-gather dimension). "
        "Can be set independently of --attn2d_row_size; asymmetric meshes (e.g. 1x4 or 4x1) are valid. "
        "Cannot be combined with --ulysses_size (not yet implemented).",
    )
    parser.add_argument("--disable_parallel_vae", action="store_true", help="Disable parallel VAE")

    add_optimization_args(parser)

    return parser.parse_args()


def _wan_needs_fine_grained_sage(model_path: str) -> bool:
    """Hard-coded heuristics for determining if a WAN model needs finer-grained SageAttentionConfig."""
    lower = model_path.lower().replace(".", "_").replace("-", "_")
    return "_1_3b" in lower


def main():
    args = parse_args()

    attn2d_size = args.attn2d_row_size * args.attn2d_col_size
    if attn2d_size > 1 and args.ulysses_size > 1:
        raise ValueError(
            "Combining --ulysses_size with --attn2d_row_size/--attn2d_col_size is not yet implemented."
        )

    if args.ulysses_size > 1:
        parallel_str = f"Ulysses(size={args.ulysses_size})"
    elif attn2d_size > 1:
        parallel_str = (
            f"Attention2D(row={args.attn2d_row_size}, col={args.attn2d_col_size}, "
            f"total={attn2d_size})"
        )
    else:
        parallel_str = "None"

    attention_cfg = {
        "backend": args.attention_backend,
    }
    if args.enable_sage_attention:
        num_elts_per_blk_k = 4 if _wan_needs_fine_grained_sage(args.model_path) else 16
        sage_cfg = {
            "num_elts_per_blk_q": 1,
            "num_elts_per_blk_k": num_elts_per_blk_k,
            "num_elts_per_blk_v": 1,
            "qk_int8": True,
        }
        attention_cfg["sage_attention_config"] = sage_cfg
        logger.info(f"SageAttention: INT8 Q/K, blocks (1, {num_elts_per_blk_k}, 1)")

    cache_kwargs = build_cache_config(args)

    kwargs = dict(
        revision=args.revision,
        attention=attention_cfg,
        **cache_kwargs,
        parallel={
            "dit_cfg_size": args.cfg_size,
            "dit_ulysses_size": args.ulysses_size,
            "dit_attn2d_row_size": args.attn2d_row_size,
            "dit_attn2d_col_size": args.attn2d_col_size,
            "enable_parallel_vae": not args.disable_parallel_vae,
        },
        torch_compile={
            "enable_torch_compile": not args.disable_torch_compile,
            "enable_fullgraph": args.enable_fullgraph,
            "enable_autotune": not args.disable_autotune,
        },
        cuda_graph={"enable_cuda_graph": args.enable_cudagraph},
        pipeline={"enable_layerwise_nvtx_marker": args.enable_layerwise_nvtx_marker},
    )
    quant_config = linear_type_to_quant_config(args.linear_type)
    if quant_config is not None:
        kwargs["quant_config"] = quant_config

    diffusion_args = VisualGenArgs(**kwargs)

    logger.info(
        f"Initializing VisualGen: "
        f"cfg_size={diffusion_args.parallel.dit_cfg_size}, "
        f"parallelism={parallel_str}"
    )
    visual_gen = VisualGen(
        model=args.model_path,
        args=diffusion_args,
    )

    try:
        defaults = visual_gen.default_params
        negative_prompt_log = (
            args.negative_prompt if args.negative_prompt is not None else "[model default]"
        )
        height = args.height if args.height is not None else defaults.height
        width = args.width if args.width is not None else defaults.width
        num_frames = args.num_frames if args.num_frames is not None else defaults.num_frames
        steps = args.steps if args.steps is not None else defaults.num_inference_steps
        frame_rate = defaults.frame_rate

        logger.info(f"Generating video for prompt: '{args.prompt}'")
        logger.info(f"Negative prompt: '{negative_prompt_log}'")
        logger.info(f"Resolution: {height}x{width}, Frames: {num_frames}, Steps: {steps}")

        start_time = time.time()

        extra_params = {}
        if args.guidance_scale_2 is not None:
            extra_params["guidance_scale_2"] = args.guidance_scale_2
        if args.boundary_ratio is not None:
            extra_params["boundary_ratio"] = args.boundary_ratio

        output = visual_gen.generate(
            inputs=args.prompt,
            params=VisualGenParams(
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                num_frames=args.num_frames,
                frame_rate=frame_rate,
                negative_prompt=args.negative_prompt,
                extra_params=extra_params if extra_params else None,
            ),
        )

        logger.info(f"Generation completed in {time.time() - start_time:.2f}s")

        output.save(args.output_path)

    finally:
        visual_gen.shutdown()


if __name__ == "__main__":
    main()
