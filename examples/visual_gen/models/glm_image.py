#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM-Image text-to-image generation through TRTLLM VisualGen.

Usage:
    python glm_image.py
    python glm_image.py --visual_gen_args ../configs/glm-image-1gpu.yaml
"""

import argparse
import os
from pathlib import Path


def _output_paths(output_path: str, num_images: int) -> str | list[str]:
    if num_images == 1:
        return output_path
    path = Path(output_path)
    return [str(path.with_name(f"{path.stem}_{idx + 1}{path.suffix}")) for idx in range(num_images)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="zai-org/GLM-Image",
        help="Hugging Face model id or local checkpoint path.",
    )
    parser.add_argument(
        "--revision",
        help="Optional Hugging Face revision. Ignored for local checkpoint paths.",
    )
    parser.add_argument(
        "--hf_cache_dir",
        help=(
            "Optional HF_HOME cache directory to use before importing TRTLLM. "
            "Prefer a persistent scratch path for large GLM-Image checkpoints."
        ),
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Set HF_HUB_OFFLINE=1 so VisualGen only uses local Hub cache files.",
    )
    parser.add_argument(
        "--visual_gen_args",
        "--extra_visual_gen_options",
        dest="visual_gen_args",
        help="Optional VisualGenArgs YAML file.",
    )
    parser.add_argument(
        "--prompt",
        default="A tiny astronaut hatching from an egg on the moon",
        help="Text prompt for image generation.",
    )
    parser.add_argument(
        "--image",
        action="append",
        help="Optional conditioning image path. Repeat for multi-image conditioning.",
    )
    parser.add_argument("--height", type=int, help="Output height. Defaults to model setting.")
    parser.add_argument("--width", type=int, help="Output width. Defaults to model setting.")
    parser.add_argument(
        "--steps",
        type=int,
        help="Number of denoising steps. Defaults to model setting.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        help="Classifier-free guidance scale. Defaults to model setting.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        help="Maximum glyph text sequence length. Defaults to model setting.",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate for the prompt.",
    )
    parser.add_argument(
        "--output_path",
        default="glm_image_output.png",
        help="Image output path. Multiple images append an index before the suffix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hf_cache_dir:
        os.environ.setdefault("HF_HOME", args.hf_cache_dir)
    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"

    from tensorrt_llm import VisualGen, VisualGenArgs

    if args.num_images_per_prompt < 1:
        raise ValueError("--num_images_per_prompt must be >= 1")

    overrides = {}
    if args.revision is not None:
        overrides["revision"] = args.revision
    extra_args = (
        VisualGenArgs.from_yaml(args.visual_gen_args, **overrides)
        if args.visual_gen_args
        else VisualGenArgs(**overrides)
    )
    visual_gen = VisualGen(model=args.model, args=extra_args)
    params = visual_gen.default_params
    params.seed = args.seed
    params.num_images_per_prompt = args.num_images_per_prompt
    if args.image:
        params.image = args.image
    if args.height is not None:
        params.height = args.height
    if args.width is not None:
        params.width = args.width
    if args.steps is not None:
        params.num_inference_steps = args.steps
    if args.guidance_scale is not None:
        params.guidance_scale = args.guidance_scale
    if args.max_sequence_length is not None:
        params.max_sequence_length = args.max_sequence_length

    try:
        output = visual_gen.generate(inputs=args.prompt, params=params)
        saved = output.save(_output_paths(args.output_path, args.num_images_per_prompt))
        print(f"Saved image(s) to {saved}")
    finally:
        visual_gen.shutdown()


if __name__ == "__main__":
    main()
