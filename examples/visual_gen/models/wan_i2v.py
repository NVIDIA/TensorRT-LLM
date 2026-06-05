#!/usr/bin/env python3
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
"""Wan Image-to-Video generation.

Usage:
    python wan_i2v.py
    python wan_i2v.py --visual_gen_args ../configs/wan2.2-i2v-fp4-1gpu.yaml
"""

import argparse
import os

from tensorrt_llm import VisualGen, VisualGenArgs

_DEFAULT_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cat_piano.png")


def main():
    parser = argparse.ArgumentParser(description="Wan Image-to-Video example")
    parser.add_argument(
        "--model",
        type=str,
        default="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        help="Model path or HuggingFace Hub ID",
    )
    parser.add_argument(
        "--visual_gen_args",
        "--extra_visual_gen_options",
        dest="visual_gen_args",
        type=str,
        default=None,
        help="Path to YAML config (same as trtllm-serve --visual_gen_args)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=_DEFAULT_IMAGE,
        help="Path to input image for I2V conditioning",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="wan_i2v_output.mp4",
        help="Path to save the output video",
    )
    args = parser.parse_args()

    # Engine config from shared YAML (optional); model-specific defaults apply otherwise.
    extra_args = VisualGenArgs.from_yaml(args.visual_gen_args) if args.visual_gen_args else None
    visual_gen = VisualGen(model=args.model, args=extra_args)

    # --- Model-specific: I2V request construction ---
    # Start from per-model defaults (steps, guidance, seed, etc.) and set the input image.
    params = visual_gen.default_params
    params.image = args.image

    output = visual_gen.generate(
        inputs="A cat presses the piano keys with its paws, soft notes filling the quiet room.",
        params=params,
    )

    output.save(args.output_path)
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
