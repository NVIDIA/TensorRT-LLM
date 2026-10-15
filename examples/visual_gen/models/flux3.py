#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""FLUX.3 image generation and editing from a Hugging Face or local checkpoint.

Usage:
    python flux3.py --model black-forest-labs/flux-3-image-klein-base
    python flux3.py --model black-forest-labs/flux-3-image-klein-base --image reference.png
"""

import argparse

from tensorrt_llm import VisualGen, VisualGenArgs
from tensorrt_llm.visual_gen import MediaRef


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Hugging Face model ID or local checkpoint path"
    )
    parser.add_argument(
        "--variant", default=None, help="Checkpoint variant; omit for the root BF16 package"
    )
    parser.add_argument("--visual_gen_args", default=None)
    parser.add_argument("--prompt", default="A red fox standing in fresh snow at sunrise")
    parser.add_argument(
        "--image", action="append", help="Reference image; repeat for multiple references"
    )
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_path", default="flux3.png")
    args = parser.parse_args()
    config = (
        VisualGenArgs.from_yaml(args.visual_gen_args) if args.visual_gen_args else VisualGenArgs()
    )
    if args.variant is not None:
        config.pipeline_config["variant"] = args.variant
    model = VisualGen(model=args.model, args=config)
    params = model.default_params
    params.height, params.width, params.seed = args.height, args.width, args.seed
    if args.num_inference_steps is not None:
        params.num_inference_steps = args.num_inference_steps
    if args.image:
        params.image_reference = [MediaRef(content=path, format="path") for path in args.image]
    output = model.generate(inputs=args.prompt, params=params)
    saved = output.save(args.output_path)
    print(f"Saved: {saved}")


if __name__ == "__main__":
    main()
