#!/usr/bin/env python3
"""LTX2 Text/Image-to-Video generation using TensorRT-LLM Visual Generation."""

import argparse
import time

from output_handler import OutputHandler

from tensorrt_llm import logger
from tensorrt_llm.llmapi.visual_gen import VisualGen, VisualGenParams

logger.set_level("info")

def parse_args():
    parser = argparse.ArgumentParser(
        description="TRTLLM VisualGen - LTX2 Text-to-Video with Audio Inference Example"
    )

    # Model & Input
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the LTX2 checkpoint (directory containing .safetensors)",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        required=True,
        help="Path to the Gemma3 text encoder model directory",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        help="Negative prompt to guide generation away from undesired content",
    )
    parser.add_argument(
        "--output_path", "--output-path",
        type=str,
        default="output.mp4",
        help="Path to save the output video with audio (supports .mp4, .gif, .png)",
    )

    # Image-to-video conditioning
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image for image-to-video conditioning",
    )
    parser.add_argument(
        "--image_cond_strength", "--image-cond-strength",
        type=float,
        default=1.0,
        help="Conditioning strength for the input image (0.0 to 1.0, default: 1.0)",
    )

    # Generation Params
    parser.add_argument("--height", type=int, default=512, help="Video height (divisible by 32)")
    parser.add_argument("--width", type=int, default=768, help="Video width (divisible by 32)")
    parser.add_argument("--num_frames", "--num-frames", type=int, default=121, help="Number of frames to generate")
    parser.add_argument(
        "--frame_rate", type=float, default=24.0, help="Frames per second for the video"
    )
    parser.add_argument("--steps", "--num-inference-steps", "--num_inference_steps",
                        type=int, default=40, help="Number of denoising steps")
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--guidance_rescale",
        type=float,
        default=0.0,
        help="Guidance rescale factor to fix overexposure",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=1024,
        help="Maximum sequence length for prompt encoding",
    )

    # TeaCache Arguments
    parser.add_argument(
        "--enable_teacache",
        action="store_true",
        help="Enable TeaCache acceleration",
    )
    parser.add_argument(
        "--teacache_thresh",
        type=float,
        default=0.2,
        help="TeaCache similarity threshold",
    )

    # Multi-modal guidance (STG / modality)
    parser.add_argument(
        "--stg_scale", type=float, default=0.0,
        help="Spatiotemporal guidance scale (0=disabled). Reference default: 1.0",
    )
    parser.add_argument(
        "--stg_blocks", type=int, nargs="*", default=None,
        help="Transformer block indices for STG perturbation (e.g., 29). Reference default: [29]",
    )
    parser.add_argument(
        "--modality_scale", type=float, default=1.0,
        help="Cross-modal guidance scale (1=disabled). Reference default: 3.0",
    )
    parser.add_argument(
        "--rescale_scale", type=float, default=0.0,
        help="Variance-preserving rescale factor (0=disabled). Reference default: 0.7",
    )
    parser.add_argument(
        "--guidance_skip_step", type=int, default=0,
        help="Skip guidance every N+1 steps (0=never skip)",
    )
    parser.add_argument(
        "--enhance_prompt", action="store_true",
        help="Use Gemma3 to enhance the text prompt before encoding",
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

    # torch compile
    parser.add_argument(
        "--disable_torch_compile", action="store_true", help="Disable TorchCompile acceleration"
    )
    parser.add_argument(
        "--torch_compile_models",
        type=str,
        nargs="+",
        default=[],
        help="Torch compile models (empty = auto detect transformer components)",
    )
    parser.add_argument(
        "--torch_compile_mode",
        type=str,
        default="default",
        choices=["default", "max-autotune", "reduce-overhead"],
        help="Torch compile mode",
    )
    parser.add_argument(
        "--enable_fullgraph", action="store_true", help="Enable fullgraph for TorchCompile"
    )

    # Warmup
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1,
        help="Warmup steps. Useful for performance benchmarking.",
    )

    # Layerwise nvtx marker
    parser.add_argument(
        "--enable_layerwise_nvtx_marker", action="store_true", help="Enable layerwise nvtx marker"
    )

    # Dynamic quantization
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

    # Output format
    parser.add_argument(
        "--output_type",
        type=str,
        default="np",
        choices=["np", "pil", "latent"],
        help="Output type: 'np' for numpy arrays, 'pil' for PIL images, 'latent' for raw latents",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # world_size = cfg_size * ulysses_size
    n_workers = args.cfg_size * args.ulysses_size

    # Convert linear_type to quant_config
    quant_config = None
    if args.linear_type == "trtllm-fp8-per-tensor":
        quant_config = {"quant_algo": "FP8", "dynamic": True}
    elif args.linear_type == "trtllm-fp8-blockwise":
        quant_config = {"quant_algo": "FP8_BLOCK_SCALES", "dynamic": True}
    elif args.linear_type == "trtllm-nvfp4":
        quant_config = {"quant_algo": "NVFP4", "dynamic": True}

    # Setup Configuration
    diffusion_config = {
        "model_type": "ltx2",
        "text_encoder_path": args.text_encoder_path,
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
            "torch_compile_models": args.torch_compile_models,
            "torch_compile_mode": args.torch_compile_mode,
            "enable_fullgraph": args.enable_fullgraph,
            "warmup_steps": args.warmup_steps,
            "enable_layerwise_nvtx_marker": args.enable_layerwise_nvtx_marker,
        },
    }

    if quant_config is not None:
        diffusion_config["quant_config"] = quant_config

    # Initialize VisualGen
    logger.info(
        f"Initializing VisualGen (LTX2): world_size={n_workers} "
        f"(cfg_size={args.cfg_size}, ulysses_size={args.ulysses_size})"
    )
    visual_gen = VisualGen(
        model_path=args.model_path,
        n_workers=n_workers,
        diffusion_config=diffusion_config,
    )

    try:
        # Run Inference
        logger.info(f"Generating video with audio for prompt: '{args.prompt}'")
        logger.info(
            f"Resolution: {args.height}x{args.width}, "
            f"Frames: {args.num_frames}, "
            f"FPS: {args.frame_rate}, "
            f"Steps: {args.steps}"
        )

        start_time = time.time()

        inputs = {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
        }

        params = VisualGenParams(
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            max_sequence_length=args.max_sequence_length,
            seed=args.seed,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            guidance_rescale=args.guidance_rescale,
            output_type=args.output_type,
            input_reference=args.image,
            image_cond_strength=args.image_cond_strength,
            stg_scale=args.stg_scale,
            stg_blocks=args.stg_blocks,
            modality_scale=args.modality_scale,
            rescale_scale=args.rescale_scale,
            guidance_skip_step=args.guidance_skip_step,
            enhance_prompt=args.enhance_prompt,
        )

        output = visual_gen.generate(inputs=inputs, params=params)

        end_time = time.time()
        logger.info(f"Generation completed in {end_time - start_time:.2f}s")

        # Save Output
        OutputHandler.save(output, args.output_path, frame_rate=args.frame_rate)

    finally:
        # Shutdown
        visual_gen.shutdown()


if __name__ == "__main__":
    main()
