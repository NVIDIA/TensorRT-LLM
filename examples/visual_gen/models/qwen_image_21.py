# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image 2.1 text-to-image and image-conditioned generation.

Usage:
    python qwen_image_21.py
    python qwen_image_21.py --visual_gen_args ../configs/qwen-image-2.1-bf16-1gpu.yaml
    python qwen_image_21.py --image input.png --prompt "restyle this scene as watercolor"

The native TRTLLM runtime owns the scheduler, transformer/attention, and
conditioning path.  The image VAE boundary uses the declared Diffusers
``AutoencoderKLQwenImage21`` fallback until a native VAE replacement is ready.
"""

import argparse
import os
from pathlib import Path

from tensorrt_llm import VisualGen, VisualGenArgs
from tensorrt_llm.visual_gen import MediaRef


def _env_int(name: str, default: int | None = None) -> int | None:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


def _output_paths(output_path: str, num_images: int) -> str | list[str]:
    if num_images == 1:
        return output_path

    path = Path(output_path)
    return [str(path.with_name(f"{path.stem}_{index}{path.suffix}")) for index in range(num_images)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen-Image-2.1",
        help="Hugging Face model id or local checkpoint path.",
    )
    parser.add_argument(
        "--visual_gen_args",
        "--extra_visual_gen_options",
        dest="visual_gen_args",
        help="Optional VisualGenArgs YAML file.",
    )
    parser.add_argument(
        "--prompt",
        default="A serene mountain lake at sunrise, watercolor style, highly detailed",
        help="Text prompt for image generation.",
    )
    parser.add_argument(
        "--image",
        help=(
            "Optional reference image path. When supplied, Qwen-Image 2.1 uses "
            "image-conditioned generation and preserves the reference aspect ratio "
            "unless --height/--width are set."
        ),
    )
    parser.add_argument("--height", type=int, help="Optional output height in pixels.")
    parser.add_argument("--width", type=int, help="Optional output width in pixels.")
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=_env_int("VISUALGEN_NUM_INFERENCE_STEPS", _env_int("NUM_INFERENCE_STEPS")),
        help=(
            "Optional denoising step count. If omitted, the model default is used; "
            "VISUALGEN_NUM_INFERENCE_STEPS or NUM_INFERENCE_STEPS can provide a reproducible default."
        ),
    )
    parser.add_argument("--seed", type=int, help="Optional random seed.")
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate for the prompt.",
    )
    parser.add_argument(
        "--output_resolution",
        type=int,
        default=1024,
        help=(
            "Qwen-Image 2.1 reference-conditioned resolution hint used when "
            "--image is set and --height/--width are omitted."
        ),
    )
    parser.add_argument(
        "--disable_kv_cache",
        action="store_true",
        help="Disable the Qwen-Image 2.1 prefix KV cache for debugging/parity runs.",
    )
    parser.add_argument(
        "--output_path",
        default="qwen_image_21_output.png",
        help="Image output path. Multiple images append an index before the suffix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_images_per_prompt < 1:
        raise ValueError("--num_images_per_prompt must be >= 1")

    extra_args = VisualGenArgs.from_yaml(args.visual_gen_args) if args.visual_gen_args else None
    visual_gen = VisualGen(model=args.model, args=extra_args)
    params = visual_gen.default_params
    params.num_images_per_prompt = args.num_images_per_prompt
    if args.height is not None:
        params.height = args.height
    if args.width is not None:
        params.width = args.width
    if args.num_inference_steps is not None:
        params.num_inference_steps = args.num_inference_steps
    if args.seed is not None:
        params.seed = args.seed
    params.extra_params = {
        "output_resolution": args.output_resolution,
        "use_kv_cache": not args.disable_kv_cache,
    }
    if args.image:
        params.image_reference = [MediaRef(content=args.image, format="path")]

    output = visual_gen.generate(inputs=args.prompt, params=params)
    saved = output.save(_output_paths(args.output_path, args.num_images_per_prompt))
    print(f"Saved image(s) to {saved}")


if __name__ == "__main__":
    main()
