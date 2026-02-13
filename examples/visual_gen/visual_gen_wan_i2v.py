#!/usr/bin/env python3
"""WAN Image-to-Video generation using TensorRT-LLM Visual Generation."""

import argparse
import time

from output_handler import OutputHandler

from tensorrt_llm import logger
from tensorrt_llm.llmapi.visual_gen import VisualGen, VisualGenParams

# Set logger level to ensure timing logs are printed
logger.set_level("info")


def parse_args():
    parser = argparse.ArgumentParser(
        description="TRTLLM VisualGen - Wan Image-to-Video Inference Example (supports Wan 2.1 and Wan 2.2)"
    )

    # Model & Input
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to Wan I2V Diffusers model directory (2.1 or 2.2)",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to input image for I2V conditioning",
    )
    parser.add_argument(
        "--last_image_path",
        type=str,
        default=None,
        help="Optional path to last frame image for interpolation (Wan 2.1 only)",
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
    parser.add_argument("--height", type=int, default=720, help="Video height")
    parser.add_argument("--width", type=int, default=1280, help="Video width")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to generate")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of denoising steps (default: auto-detect, 50 for Wan2.1, 40 for Wan2.2)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="Guidance scale (default: auto-detect, 5.0 for Wan2.1, 4.0 for Wan2.2)",
    )
    parser.add_argument(
        "--guidance_scale_2",
        type=float,
        default=None,
        help="Second-stage guidance scale for Wan2.2 two-stage denoising (default: 3.0)",
    )
    parser.add_argument(
        "--boundary_ratio",
        type=float,
        default=None,
        help="Custom boundary ratio for two-stage denoising (default: auto-detect)",
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
        choices=["VANILLA", "TRTLLM"],
        help="Attention backend (VANILLA: PyTorch SDPA, TRTLLM: optimized kernels). "
        "Note: TRTLLM automatically falls back to VANILLA for cross-attention.",
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

    return parser.parse_args()


def main():
    args = parse_args()

    # world_size = cfg_size * ulysses_size
    # Example: cfg_size=2, ulysses_size=4 -> 8 GPUs
    #   GPU 0-3: CFG group 0 (positive prompt), internal Ulysses parallel
    #   GPU 4-7: CFG group 1 (negative prompt), internal Ulysses parallel
    n_workers = args.cfg_size * args.ulysses_size

    # Convert linear_type to quant_config
    quant_config = None
    if args.linear_type == "trtllm-fp8-per-tensor":
        quant_config = {"quant_algo": "FP8", "dynamic": True}
    elif args.linear_type == "trtllm-fp8-blockwise":
        quant_config = {"quant_algo": "FP8_BLOCK_SCALES", "dynamic": True}
    elif args.linear_type == "svd-nvfp4":
        quant_config = {"quant_algo": "NVFP4", "dynamic": True}

    # 1. Setup Configuration
    diffusion_config = {
        "model_type": "wan2",
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
    }

    # Add quant_config if specified
    if quant_config is not None:
        diffusion_config["quant_config"] = quant_config

    # 2. Initialize VisualGen
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
        # 2. Run Inference
        logger.info(f"Generating video for prompt: '{args.prompt}'")
        logger.info(f"Negative prompt: '{args.negative_prompt}'")
        logger.info(f"Input image: {args.image_path}")
        if args.last_image_path:
            logger.info(f"Last frame image: {args.last_image_path}")
        logger.info(
            f"Resolution: {args.height}x{args.width}, Frames: {args.num_frames}, Steps: {args.steps}"
        )

        start_time = time.time()

        # Build parameters with explicit I2V and Wan 2.2 support
        output = visual_gen.generate(
            inputs={
                "prompt": args.prompt,
                "negative_prompt": args.negative_prompt,
            },
            params=VisualGenParams(
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                num_frames=args.num_frames,
                input_reference=args.image_path,
                last_image=args.last_image_path if args.last_image_path else None,
                guidance_scale_2=args.guidance_scale_2,
                boundary_ratio=args.boundary_ratio,
            ),
        )

        end_time = time.time()
        logger.info(f"Generation completed in {end_time - start_time:.2f}s")

        # 3. Save Output
        OutputHandler.save(output, args.output_path, frame_rate=16.0)

    finally:
        # 4. Shutdown
        visual_gen.shutdown()


if __name__ == "__main__":
    main()
