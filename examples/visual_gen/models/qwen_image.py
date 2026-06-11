# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen-Image text-to-image generation.

Usage:
    python qwen_image.py
    python qwen_image.py --visual_gen_args ../configs/qwen-image-fp4-1gpu.yaml
"""

import argparse
from pathlib import Path

from tensorrt_llm import VisualGen, VisualGenArgs


def _output_paths(output_path: str, num_images: int) -> str | list[str]:
    if num_images == 1:
        return output_path

    path = Path(output_path)
    return [str(path.with_name(f"{path.stem}_{index}{path.suffix}")) for index in range(num_images)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen-Image",
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
        "--negative_prompt",
        help="Optional negative prompt.",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate for the prompt.",
    )
    parser.add_argument(
        "--output_path",
        default="qwen_image_output.png",
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
    if args.negative_prompt is not None:
        params.negative_prompt = args.negative_prompt

    output = visual_gen.generate(inputs=args.prompt, params=params)
    saved = output.save(_output_paths(args.output_path, args.num_images_per_prompt))
    print(f"Saved image(s) to {saved}")


if __name__ == "__main__":
    main()
