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
"""LTX-2.3 Text-to-Video generation with audio.

Usage:
    python ltx23.py
    python ltx23.py --visual_gen_args ../configs/ltx23-t2v-bf16-1gpu.yaml
"""

import argparse

from tensorrt_llm import VisualGen, VisualGenArgs


def main():
    parser = argparse.ArgumentParser(description="LTX-2.3 Text-to-Video example")
    parser.add_argument(
        "--model",
        type=str,
        default="Lightricks/LTX-2.3",
        help="Model path or HuggingFace Hub ID",
    )
    parser.add_argument(
        "--visual_gen_args",
        dest="visual_gen_args",
        type=str,
        default=None,
        help="Path to YAML config (same as trtllm-serve --visual_gen_args)",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help=(
            "Gemma3 text encoder path. Overrides pipeline_config.text_encoder_path "
            "from --visual_gen_args when set."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="ltx23_t2v_output.mp4",
        help="Path to save the output video",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cinematic shot of a cat walking through a field of flowers",
        help="Text prompt",
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--frame_rate", type=float, default=24.0)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    args = parser.parse_args()

    # LTX-2.3 requires pipeline_config.text_encoder_path for the Gemma3 text
    # encoder. The YAML path is preferred for production configs; the default
    # below keeps this script runnable as a minimal offline example.
    extra_args = (
        VisualGenArgs.from_yaml(args.visual_gen_args) if args.visual_gen_args else VisualGenArgs()
    )
    text_encoder_path = args.text_encoder_path
    if text_encoder_path is None and not args.visual_gen_args:
        text_encoder_path = "google/gemma-3-12b-it"
    if text_encoder_path is not None:
        extra_args.pipeline_config = {
            **extra_args.pipeline_config,
            "text_encoder_path": text_encoder_path,
        }
    visual_gen = VisualGen(model=args.model, args=extra_args)

    # --- Model-specific: T2V request construction ---
    # Start from LTX-2.3 defaults and override the main request shape explicitly.
    params = visual_gen.default_params
    params.height = args.height
    params.width = args.width
    params.num_frames = args.num_frames
    params.frame_rate = args.frame_rate
    params.num_inference_steps = args.num_inference_steps
    params.guidance_scale = args.guidance_scale

    output = visual_gen.generate(
        inputs=args.prompt,
        params=params,
    )

    output.save(args.output_path)
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
