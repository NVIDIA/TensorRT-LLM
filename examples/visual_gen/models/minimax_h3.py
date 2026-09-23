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

"""MiniMax-H3 text-to-video generation with stereo audio.

The checkpoint license restricts permitted territories. Obtain legal approval
before downloading or running the model.
"""

import argparse

from tensorrt_llm import VisualGen, VisualGenArgs
from tensorrt_llm.visual_gen.params import MediaRef


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Approved checkpoint path or Hugging Face model ID.",
    )
    parser.add_argument(
        "--visual_gen_args",
        help="Path to YAML config (same as trtllm-serve --visual_gen_args).",
    )
    parser.add_argument(
        "--output_path",
        default="minimax_h3_t2va_output.mp4",
        help="Path to save the generated video and audio.",
    )
    parser.add_argument(
        "--image_reference",
        action="append",
        default=[],
        help="Ref2VA image path; repeat for multiple images.",
    )
    parser.add_argument(
        "--video_reference",
        action="append",
        default=[],
        help="Ref2VA video path, including its soundtrack.",
    )
    parser.add_argument(
        "--audio_reference",
        action="append",
        default=[],
        help="Ref2VA audio path; requires an image or video.",
    )
    parser.add_argument(
        "--reference_order",
        nargs="+",
        help="Optional ordered image:N/video:N/audio:N entries, zero-based.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt; references use <Picture 1>, <Video 1>, and <Audio 1> labels.",
    )
    args = parser.parse_args()

    extra_args = VisualGenArgs.from_yaml(args.visual_gen_args) if args.visual_gen_args else None
    visual_gen = VisualGen(model=args.model, args=extra_args)
    params = visual_gen.default_params
    # Match the reference app's 960x544 canvas and seed; inherit 28 steps and
    # 124 frames at 24 FPS (the app's aligned 5-second request).
    params.height = 544
    params.width = 960
    params.seed = 42

    for kind in ("image", "video", "audio"):
        paths = getattr(args, f"{kind}_reference")
        if paths:
            setattr(
                params,
                f"{kind}_reference",
                [MediaRef(content=path, format="path", role="reference") for path in paths],
            )
    if args.reference_order is not None:
        params.extra_params = {"reference_order": args.reference_order}

    output = visual_gen.generate(
        inputs=args.prompt
        or (
            "A woman with long brown hair and light skin smiles at the camera "
            "while standing in a sunlit park, her hair gently blowing in the "
            "breeze as she tilts her head slightly to the side."
        ),
        params=params,
    )
    saved = output.save(args.output_path)
    print(f"Saved: {saved}")


if __name__ == "__main__":
    main()
