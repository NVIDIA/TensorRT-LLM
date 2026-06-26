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

"""HunyuanVideo 1.5 Text-to-Video generation.

Single-GPU text-to-video. Image-to-video, parallelism, and caching are not yet
supported for this model.

Usage:
    python hunyuan_t2v.py
    python hunyuan_t2v.py --visual_gen_args ../configs/hunyuan-t2v-fp8-1gpu.yaml
"""

import argparse
from pathlib import Path

from tensorrt_llm import VisualGen, VisualGenArgs


def _output_paths(output_path: str, num_videos: int) -> str | list[str]:
    if num_videos == 1:
        return output_path

    path = Path(output_path)
    return [str(path.with_name(f"{path.stem}_{idx + 1}{path.suffix}")) for idx in range(num_videos)]


def main() -> None:
    parser = argparse.ArgumentParser(description="HunyuanVideo 1.5 Text-to-Video example")
    parser.add_argument(
        "--model",
        type=str,
        default="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
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
        "--prompt",
        type=str,
        default="A cute cat playing piano",
        help="Text prompt for video generation",
    )
    parser.add_argument(
        "--num_videos_per_prompt",
        type=int,
        default=1,
        help="Number of videos to generate for the prompt",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="hunyuan_t2v_output.mp4",
        help="Path to save the output video. For multiple videos, an index is appended.",
    )
    args = parser.parse_args()
    if args.num_videos_per_prompt < 1:
        raise ValueError("--num_videos_per_prompt must be >= 1")

    # Engine config from shared YAML (optional); model-specific defaults apply otherwise.
    extra_args = VisualGenArgs.from_yaml(args.visual_gen_args) if args.visual_gen_args else None
    visual_gen = VisualGen(model=args.model, args=extra_args)

    # --- Model-specific: T2V request construction ---
    # Start from per-model defaults (steps, guidance, seed, etc.) and override resolution/frames.
    params = visual_gen.default_params
    params.height = 480
    params.width = 832
    params.num_frames = 121
    params.num_images_per_prompt = args.num_videos_per_prompt

    output = visual_gen.generate(inputs=args.prompt, params=params)

    saved = output.save(_output_paths(args.output_path, args.num_videos_per_prompt))
    print(f"Saved: {saved}")


if __name__ == "__main__":
    main()
