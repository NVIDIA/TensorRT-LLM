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
    args = parser.parse_args()

    extra_args = VisualGenArgs.from_yaml(args.visual_gen_args) if args.visual_gen_args else None
    visual_gen = VisualGen(model=args.model, args=extra_args)
    params = visual_gen.default_params
    params.height = 512
    params.width = 512

    output = visual_gen.generate(
        inputs=(
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
