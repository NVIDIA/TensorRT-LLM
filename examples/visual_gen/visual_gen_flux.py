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

from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams, logger
from tensorrt_llm._torch.visual_gen.config import CacheDiTConfig, TeaCacheConfig

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

    # Diffusion cache acceleration (TeaCache and Cache-DiT; mutually exclusive)
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--enable_teacache", action="store_true", help="Enable TeaCache acceleration"
    )
    cache_group.add_argument(
        "--enable_cache_dit",
        action="store_true",
        help=(
            "Enable Cache-DiT per-block acceleration (requires the cache_dit package; "
            "see https://github.com/vipshop/cache-dit). Incompatible with --enable_teacache."
        ),
    )
    parser.add_argument(
        "--teacache_thresh",
        type=float,
        default=None,
        help="TeaCache similarity threshold (default: 0.6 for FLUX.1, 0.2 for FLUX.2); "
        "ignored when using --enable_cache_dit",
    )
    parser.add_argument(
        "--use_ret_steps",
        action="store_true",
        help="Use ret_steps mode for TeaCache. "
        "Using Retention Steps will result in faster generation speed and better generation quality. "
        "Ignored when using --enable_cache_dit.",
    )

    # Cache-DiT overrides (only apply with --enable_cache_dit; omitted fields use CacheDiTConfig defaults)
    parser.add_argument(
        "--cache_dit_fn_compute_blocks",
        type=int,
        default=None,
        help="DBCache Fn_compute_blocks (default: from CacheDiTConfig).",
    )
    parser.add_argument(
        "--cache_dit_bn_compute_blocks",
        type=int,
        default=None,
        help="DBCache Bn_compute_blocks (default: from CacheDiTConfig).",
    )
    parser.add_argument(
        "--cache_dit_max_warmup_steps",
        type=int,
        default=None,
        help="DBCache max_warmup_steps (default: from CacheDiTConfig).",
    )
    parser.add_argument(
        "--cache_dit_max_cached_steps",
        type=int,
        default=None,
        help="DBCache max_cached_steps (-1 = no cap; default: from CacheDiTConfig).",
    )
    parser.add_argument(
        "--cache_dit_residual_threshold",
        type=float,
        default=None,
        help="DBCache residual_diff_threshold (default: from CacheDiTConfig).",
    )
    parser.add_argument(
        "--cache_dit_enable_taylorseer",
        action="store_true",
        help="Enable TaylorSeer calibrator (default: off).",
    )
    parser.add_argument(
        "--cache_dit_taylorseer_order",
        type=int,
        default=None,
        choices=[1, 2, 3, 4],
        help="TaylorSeer order; implies TaylorSeer on if set. Default order from CacheDiTConfig.",
    )
    parser.add_argument(
        "--cache_dit_scm_mask_policy",
        type=str,
        default=None,
        help="SCM steps_mask policy name (e.g. fast, medium, slow, ultra). Omit to disable SCM.",
    )
    parser.add_argument(
        "--cache_dit_scm_steps_policy",
        type=str,
        default=None,
        choices=["dynamic", "static"],
        help="SCM steps_computation_policy (default: dynamic if not overridden).",
    )

    # Quantization
    parser.add_argument(
        "--linear_type",
        type=str,
        default="default",
        choices=["default", "trtllm-fp8-per-tensor", "trtllm-fp8-blockwise", "trtllm-nvfp4"],
        help=(
            "Dynamic quantization mode for linear layers. "
            "Quantizes weights on-the-fly during loading from an unquantized checkpoint."
        ),
    )

    # Attention Backend
    parser.add_argument(
        "--attention_backend",
        type=str,
        default="VANILLA",
        choices=["VANILLA", "TRTLLM", "FA4"],
        help="Attention backend (VANILLA: PyTorch SDPA, TRTLLM: optimized kernels, "
        "FA4: Flash Attention 4). "
        "Note: TRTLLM falls back to VANILLA for cross-attention.",
    )
    parser.add_argument(
        "--enable_sage_attention",
        action="store_true",
        help="Enable SageAttention (per-block quantized Q/K/V). Requires TRTLLM backend.",
    )

    # Parallelism
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="Ulysses (head-sharding) parallel size within each CFG group. "
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

    # CUDA graph
    parser.add_argument(
        "--enable_cudagraph", action="store_true", help="Enable CudaGraph acceleration"
    )

    # torch.compile
    parser.add_argument(
        "--disable_torch_compile", action="store_true", help="Disable TorchCompile acceleration"
    )
    parser.add_argument(
        "--enable_fullgraph", action="store_true", help="Enable fullgraph for TorchCompile"
    )

    # Autotune
    parser.add_argument(
        "--disable_autotune", action="store_true", help="Disable autotuning during warmup"
    )

    # Debug / profiling
    parser.add_argument(
        "--enable_layerwise_nvtx_marker", action="store_true", help="Enable layerwise NVTX markers"
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


def _linear_type_to_quant_config(linear_type: str):
    """Map --linear_type CLI shortcut to quant_config dict for VisualGenArgs."""
    mapping = {
        "trtllm-fp8-per-tensor": {"quant_algo": "FP8", "dynamic": True},
        "trtllm-fp8-blockwise": {"quant_algo": "FP8_BLOCK_SCALES", "dynamic": True},
        "trtllm-nvfp4": {"quant_algo": "NVFP4", "dynamic": True},
    }
    return mapping.get(linear_type)


def _teacache_config_from_args(args) -> TeaCacheConfig:
    """Build TeaCacheConfig from CLI args; unset options keep Pydantic defaults."""
    kwargs: dict = {"use_ret_steps": args.use_ret_steps}
    if args.teacache_thresh is not None:
        kwargs["teacache_thresh"] = args.teacache_thresh
    return TeaCacheConfig(**kwargs)


def _cache_dit_config_from_args(args) -> CacheDiTConfig:
    """Subset of CacheDiTConfig from CLI; unset options keep Pydantic defaults."""
    overrides: dict = {}
    if args.cache_dit_fn_compute_blocks is not None:
        overrides["Fn_compute_blocks"] = args.cache_dit_fn_compute_blocks
    if args.cache_dit_bn_compute_blocks is not None:
        overrides["Bn_compute_blocks"] = args.cache_dit_bn_compute_blocks
    if args.cache_dit_max_warmup_steps is not None:
        overrides["max_warmup_steps"] = args.cache_dit_max_warmup_steps
    if args.cache_dit_max_cached_steps is not None:
        overrides["max_cached_steps"] = args.cache_dit_max_cached_steps
    if args.cache_dit_residual_threshold is not None:
        overrides["residual_diff_threshold"] = args.cache_dit_residual_threshold
    if args.cache_dit_enable_taylorseer or args.cache_dit_taylorseer_order is not None:
        overrides["enable_taylorseer"] = True
    if args.cache_dit_taylorseer_order is not None:
        overrides["taylorseer_order"] = args.cache_dit_taylorseer_order
    if args.cache_dit_scm_mask_policy is not None:
        overrides["scm_steps_mask_policy"] = args.cache_dit_scm_mask_policy
    if args.cache_dit_scm_steps_policy is not None:
        overrides["scm_steps_policy"] = args.cache_dit_scm_steps_policy
    return CacheDiTConfig(**overrides)


def build_diffusion_args(args) -> VisualGenArgs:
    """Build VisualGenArgs from parsed CLI args."""
    if args.enable_cache_dit:
        cache_kwargs = {"cache": _cache_dit_config_from_args(args)}
    elif args.enable_teacache:
        cache_kwargs = {"cache": _teacache_config_from_args(args)}
    else:
        cache_kwargs = {}

    attention_cfg: dict = {"backend": args.attention_backend}
    if args.enable_sage_attention:
        attention_cfg["sage_attention_config"] = {
            "num_elts_per_blk_q": 1,
            "num_elts_per_blk_k": 16,
            "num_elts_per_blk_v": 1,
            "qk_int8": True,
        }
        logger.info("SageAttention: INT8 Q/K, blocks (1, 16, 1)")

    kwargs = dict(
        revision=args.revision,
        attention=attention_cfg,
        **cache_kwargs,
        parallel={
            "dit_ulysses_size": args.ulysses_size,
            "dit_attn2d_row_size": args.attn2d_row_size,
            "dit_attn2d_col_size": args.attn2d_col_size,
        },
        torch_compile={
            "enable_torch_compile": not args.disable_torch_compile,
            "enable_fullgraph": args.enable_fullgraph,
            "enable_autotune": not args.disable_autotune,
        },
        cuda_graph={"enable_cuda_graph": args.enable_cudagraph},
        pipeline={"enable_layerwise_nvtx_marker": args.enable_layerwise_nvtx_marker},
    )
    quant_config = _linear_type_to_quant_config(args.linear_type)
    if quant_config is not None:
        kwargs["quant_config"] = quant_config
    return VisualGenArgs(**kwargs)


def main():
    args = parse_args()

    attn2d_size = args.attn2d_row_size * args.attn2d_col_size
    if attn2d_size > 1 and args.ulysses_size > 1:
        raise ValueError(
            "Combining --ulysses_size with --attn2d_row_size/--attn2d_col_size is not yet implemented."
        )

    diffusion_args = build_diffusion_args(args)

    if args.ulysses_size > 1:
        parallel_str = f"Ulysses(size={args.ulysses_size})"
    elif attn2d_size > 1:
        parallel_str = (
            f"Attention2D(row={args.attn2d_row_size}, col={args.attn2d_col_size}, "
            f"total={attn2d_size})"
        )
    else:
        parallel_str = "None"
    logger.info(f"Initializing VisualGen: parallelism={parallel_str}")
    visual_gen = VisualGen(
        model=args.model_path,
        args=diffusion_args,
    )

    try:
        if args.prompts_file:
            prompts = load_prompts(args.prompts_file, args.num_prompts)
            os.makedirs(args.output_dir, exist_ok=True)

            logger.info(f"Batch mode: {len(prompts)} prompts -> {args.output_dir}")
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
                output.save(output_path)
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

            logger.info(f"Generation completed in {time.time() - start_time:.2f}s")

            output.save(args.output_path)

    finally:
        visual_gen.shutdown()


if __name__ == "__main__":
    main()
