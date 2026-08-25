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
"""LTX-2.3 HQ text-to-video generation with audio.

Usage:
    python ltx2_3.py
    python ltx2_3.py --model /path/to/LTX-2.3-checkpoint         --visual_gen_args ../configs/ltx2-3-hq-1gpu.yaml
"""

import argparse

from tensorrt_llm import VisualGen, VisualGenArgs


DEFAULT_PROMPT = (
    "A cinematic close-up of a glass greenhouse at sunrise, dew on the leaves, "
    "soft warm light, slow camera push-in, realistic ambience."
)
DEFAULT_NEGATIVE_PROMPT = "worst quality, blurry, jittery, distorted, inconsistent motion"


def main():
    parser = argparse.ArgumentParser(description="LTX-2.3 HQ text-to-video example")
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
        help="Gemma3 text encoder path; overrides pipeline_config.text_encoder_path.",
    )
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Text prompt")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt used for classifier-free guidance",
    )
    parser.add_argument("--height", type=int, default=1088, help="Output video height")
    parser.add_argument("--width", type=int, default=1920, help="Output video width")
    parser.add_argument("--num_frames", type=int, default=121, help="Number of video frames")
    parser.add_argument("--frame_rate", type=float, default=24.0, help="Output frame rate")
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=15,
        help="Number of Res2S denoising steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.0,
        help="Video classifier-free guidance scale",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_path",
        type=str,
        default="ltx2_3_hq_output.mp4",
        help="Path to save the output video",
    )
    args = parser.parse_args()

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
    params = visual_gen.default_params
    params.height = args.height
    params.width = args.width
    params.num_frames = args.num_frames
    params.frame_rate = args.frame_rate
    params.num_inference_steps = args.num_inference_steps
    params.guidance_scale = args.guidance_scale
    params.negative_prompt = args.negative_prompt
    params.seed = args.seed
    params.extra_params = {
        **(params.extra_params or {}),
        "sampler": "res2s",
        "res2s_eta": 0.5,
        "stg_scale": 0.0,
        "stg_blocks": [],
        "modality_scale": 3.0,
        "rescale_scale": 0.45,
        "audio_rescale_scale": 1.0,
    }

    output = visual_gen.generate(inputs=args.prompt, params=params)
    output.save(args.output_path)
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
