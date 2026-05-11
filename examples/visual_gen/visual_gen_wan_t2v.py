#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WAN Text-to-Video generation using TensorRT-LLM Visual Generation."""

import argparse
import time

from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams, logger
from tensorrt_llm._torch.visual_gen.config import CacheDiTConfig, TeaCacheConfig

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

    # Diffusion cache acceleration (TeaCache vs Cache-DiT; mutually exclusive)
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
        default=0.2,
        help="TeaCache similarity threshold (rel_l1_thresh); ignored when using --enable_cache_dit",
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

    return parser.parse_args()


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

    if args.enable_cache_dit:
        cache_kwargs = {"cache": _cache_dit_config_from_args(args)}
    elif args.enable_teacache:
        cache_kwargs = {"cache": _teacache_config_from_args(args)}
    else:
        cache_kwargs = {}

    kwargs = dict(
        revision=args.revision,
        attention={"backend": args.attention_backend},
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
    quant_config = _linear_type_to_quant_config(args.linear_type)
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
