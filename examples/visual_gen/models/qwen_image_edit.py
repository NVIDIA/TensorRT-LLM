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

"""Qwen-Image-Edit-2511 image editing.

Usage:
    python qwen_image_edit.py --image input.png --prompt "Make the image look like a watercolor painting"
    python qwen_image_edit.py \
        --visual_gen_args ../configs/qwen-image-edit-2511-fp8-1gpu.yaml \
        --image input.png
"""

import argparse

from tensorrt_llm import VisualGen, VisualGenArgs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen-Image-Edit-2511",
        help="Hugging Face model id or local checkpoint path.",
    )
    parser.add_argument(
        "--visual_gen_args",
        dest="visual_gen_args",
        type=str,
        default=None,
        help="Path to YAML config (same as trtllm-serve --visual_gen_args)",
    )
    parser.add_argument(
        "--image",
        nargs="+",
        required=True,
        help="One or more input image paths or URLs.",
    )
    parser.add_argument(
        "--prompt",
        default="Make the image look like a watercolor painting while preserving the main subject.",
        help="Text edit instruction.",
    )
    parser.add_argument(
        "--output_path",
        default="qwen_image_edit_output.png",
        help="Edited image output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extra_args = VisualGenArgs.from_yaml(args.visual_gen_args) if args.visual_gen_args else None
    visual_gen = VisualGen(model=args.model, args=extra_args)
    params = visual_gen.default_params
    params.image = args.image if len(args.image) > 1 else args.image[0]
    output = visual_gen.generate(inputs=args.prompt, params=params)
    saved = output.save(args.output_path)
    print(f"Saved edited image to {saved}")


if __name__ == "__main__":
    main()
