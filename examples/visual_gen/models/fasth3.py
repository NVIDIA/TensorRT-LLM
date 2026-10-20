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

"""FastH3 text-to-video generation with stereo audio."""

import argparse

from tensorrt_llm import VisualGen, VisualGenArgs

FASTH3_MODEL_ID = "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=FASTH3_MODEL_ID,
        help="FastH3 checkpoint path or Hugging Face model ID.",
    )
    parser.add_argument(
        "--visual_gen_args",
        default="configs/fasth3-bf16-1gpu.yaml",
        help="Path to YAML config (same as trtllm-serve --visual_gen_args).",
    )
    parser.add_argument(
        "--output_path",
        default="fasth3_t2va_output.mp4",
        help="Path to save the generated video and audio.",
    )
    parser.add_argument("--prompt", default=None, help="Prompt.")
    args = parser.parse_args()

    extra_args = VisualGenArgs.from_yaml(args.visual_gen_args) if args.visual_gen_args else None
    visual_gen = VisualGen(model=args.model, args=extra_args)
    params = visual_gen.default_params
    params.height = 768
    params.width = 1344
    params.num_frames = 124
    params.num_inference_steps = 5
    params.seed = 42

    output = visual_gen.generate(
        inputs=args.prompt
        or (
            "A cinematic wide shot of an astronaut walking through a neon-lit "
            "rainforest at dusk, with soft ambient music and detailed natural sounds."
        ),
        params=params,
    )
    saved = output.save(args.output_path)
    print(f"Saved: {saved}")


if __name__ == "__main__":
    main()
