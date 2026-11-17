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
    # Optional: override the default HuggingFace model with a local checkpoint.
    python ltx23.py --model /path/to/ltx-2.3-22b-dev.safetensors
"""

import argparse
from pathlib import Path

from tensorrt_llm import VisualGen, VisualGenArgs
from tensorrt_llm._torch.visual_gen.models.ltx23.checkpoint import LTX23_MODEL_ID

_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "ltx23-t2v-bf16-1gpu.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="LTX-2.3 Text-to-Video example")
    parser.add_argument(
        "--model",
        type=str,
        default=LTX23_MODEL_ID,
        help="Checkpoint file, directory, or Hub ID. Directories/Hub IDs select the dev checkpoint.",
    )
    parser.add_argument(
        "--visual_gen_args",
        dest="visual_gen_args",
        type=str,
        default=str(_DEFAULT_CONFIG),
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
    args = parser.parse_args()

    extra_args = VisualGenArgs.from_yaml(args.visual_gen_args)
    text_encoder_path = args.text_encoder_path
    if text_encoder_path is not None:
        extra_args.pipeline_config = {
            **extra_args.pipeline_config,
            "text_encoder_path": text_encoder_path,
        }
    extra_args.compilation_config.resolutions = [(512, 768)]
    extra_args.compilation_config.num_frames = [49]
    visual_gen = VisualGen(model=args.model, args=extra_args)

    params = visual_gen.default_params
    params.height = 512
    params.width = 768
    params.num_frames = 49
    params.frame_rate = 24.0
    params.num_inference_steps = 8
    params.guidance_scale = 5.0
    params.seed = 42
    params.negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

    output = visual_gen.generate(
        inputs=(
            "A woman with long brown hair and light skin smiles at the camera while standing in a "
            "sunlit park, her hair gently blowing in the breeze as she tilts her head slightly to the side."
        ),
        params=params,
    )

    metrics = output.metrics
    if metrics is not None:
        print(
            f"latency: generation={metrics.generation:.3f}s "
            f"pre_denoise={metrics.pre_denoise:.3f}s "
            f"denoise={metrics.denoise:.3f}s "
            f"post_denoise={metrics.post_denoise:.3f}s"
        )

    output.save(args.output_path)
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
