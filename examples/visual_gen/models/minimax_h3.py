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
"""MiniMax-H3 Text-to-Video+Audio generation.

MiniMax-H3 is a guidance-distilled omni-modal system that generates video with
native stereo audio from text (and, in the FL2VA workflow, first/last
keyframes). The released checkpoint generates at 24 fps for 5-15 seconds on a
768-short-edge canvas; ``num_frames`` is snapped to the next ``17 * n + 5``
the video VAE can decode and ``num_inference_steps`` counts sigma grid points
(the terminal zero included), so it drives one model evaluation less.

The checkpoint is guidance-distilled: there is no negative prompt and no
guidance scale, and every denoising step runs exactly one forward pass.

``--fp4`` enables online NVFP4 quantization of the transformer: the bf16
checkpoint weights are quantized to FP4 on the fly at load time, shrinking the
~62 GB transformer to ~15 GB so the full pipeline (transformer + Qwen3-VL
conditioner + both VAEs) fits on a single 96 GB accelerator. The conditioner
is swapped in per request by default in that configuration.

Example:
    python minimax_h3.py --prompt "A red fox trotting through a snowy pine forest, snow crunching underfoot"
    python minimax_h3.py --num_frames 124 --num_inference_steps 50
    python minimax_h3.py --fp4 --num_inference_steps 50
    python minimax_h3.py --visual_gen_args ../configs/minimax-h3-t2va-fp4-1gpu.yaml
"""

import argparse

from tensorrt_llm import VisualGen, VisualGenArgs
from tensorrt_llm.visual_gen.args import CompilationConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="MiniMaxAI/MiniMax-H3",
        help="HuggingFace model ID or local checkpoint directory.",
    )
    parser.add_argument(
        "--prompt",
        default="A red fox trotting through a snowy pine forest, snow crunching underfoot",
    )
    parser.add_argument("--height", type=int, default=None, help="Video height, a multiple of 32.")
    parser.add_argument("--width", type=int, default=None, help="Video width, a multiple of 32.")
    parser.add_argument(
        "--num_frames", type=int, default=124, help="Frames to generate, at 24 fps."
    )
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Sigma grid points.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fp4",
        action="store_true",
        help="Quantize the transformer to NVFP4 on the fly at load time.",
    )
    parser.add_argument(
        "--visual_gen_args",
        dest="visual_gen_args",
        default=None,
        help="Path to a VisualGenArgs YAML config (same as trtllm-serve --visual_gen_args).",
    )
    parser.add_argument("--output_path", default="minimax_h3_output.mp4")
    args = parser.parse_args()

    if args.visual_gen_args:
        visual_gen_args = VisualGenArgs.from_yaml(args.visual_gen_args)
    else:
        quant_config = {"quant_algo": "NVFP4"} if args.fp4 else None
        visual_gen_args = VisualGenArgs(
            compilation_config=CompilationConfig(skip_warmup=True),
            quant_config=quant_config,
        )

    visual_gen = VisualGen(model=args.model, args=visual_gen_args)

    params = visual_gen.default_params
    params.height = args.height
    params.width = args.width
    params.num_frames = args.num_frames
    params.num_inference_steps = args.num_inference_steps
    params.seed = args.seed

    output = visual_gen.generate(inputs=args.prompt, params=params)

    saved = output.save(args.output_path)
    print(f"Saved: {saved}")


if __name__ == "__main__":
    main()
