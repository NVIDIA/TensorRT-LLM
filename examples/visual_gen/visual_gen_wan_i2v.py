#!/usr/bin/env python3
"""WAN Image-to-Video generation using TensorRT-LLM Visual Generation."""

import argparse
import time

from example_utils import add_common_args, build_diffusion_config, build_generation_params
from output_handler import OutputHandler

from tensorrt_llm import logger
from tensorrt_llm.llmapi.visual_gen import VisualGen

logger.set_level("info")

WAN_DEFAULTS = {
    "height": 720,
    "width": 1280,
    "num_frames": 81,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="TRTLLM VisualGen - Wan Image-to-Video Inference Example (supports Wan 2.1 and Wan 2.2)"
    )
    add_common_args(parser)

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

    return parser.parse_args()


def main():
    args = parse_args()

    n_workers = args.cfg_size * args.ulysses_size

    diffusion_config = build_diffusion_config(args)
    diffusion_config["model_type"] = "wan2"
    params = build_generation_params(
        args,
        defaults=WAN_DEFAULTS,
        input_reference=args.image_path,
        last_image=args.last_image_path if args.last_image_path else None,
    )

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
        logger.info(f"Generating video for prompt: '{args.prompt}'")
        logger.info(f"Negative prompt: '{args.negative_prompt}'")
        logger.info(f"Input image: {args.image_path}")
        if args.last_image_path:
            logger.info(f"Last frame image: {args.last_image_path}")
        logger.info(
            f"Resolution: {params.height}x{params.width}, "
            f"Frames: {params.num_frames}, Steps: {params.num_inference_steps}"
        )

        start_time = time.time()

        output = visual_gen.generate(
            inputs={"prompt": args.prompt, "negative_prompt": args.negative_prompt},
            params=params,
        )

        logger.info(f"Generation completed in {time.time() - start_time:.2f}s")

        OutputHandler.save(output, args.output_path, frame_rate=16.0)

    finally:
        visual_gen.shutdown()


if __name__ == "__main__":
    main()
