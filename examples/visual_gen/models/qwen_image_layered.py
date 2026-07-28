# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen-Image-Layered image decomposition.

Usage:
    python qwen_image_layered.py --image input.png
    python qwen_image_layered.py --visual_gen_args ../configs/qwen-image-layered-1gpu.yaml \
        --image input.png
"""

import argparse

from tensorrt_llm import VisualGen, VisualGenArgs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen-Image-Layered",
        help="Hugging Face model id or local checkpoint path.",
    )
    parser.add_argument(
        "--visual_gen_args",
        "--extra_visual_gen_options",
        dest="visual_gen_args",
        help="Optional VisualGenArgs YAML file.",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Input image path.",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Optional text prompt. Empty prompt enables image captioning.",
    )
    parser.add_argument(
        "--output_path",
        default="qwen_image_layered_output.png",
        help="Path to save the layer grid image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extra_args = VisualGenArgs.from_yaml(args.visual_gen_args) if args.visual_gen_args else None
    visual_gen = VisualGen(model=args.model, args=extra_args)

    params = visual_gen.default_params
    params.image = args.image

    output = visual_gen.generate(inputs=args.prompt, params=params)
    saved = output.save(args.output_path)
    print(f"Saved image to {saved}")


if __name__ == "__main__":
    main()
