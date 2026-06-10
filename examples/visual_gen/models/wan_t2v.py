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
"""Wan Text-to-Video generation.

Usage:
    python wan_t2v.py
    python wan_t2v.py --visual_gen_args ../configs/wan2.2-t2v-fp4-1gpu.yaml
    python wan_t2v.py --enable_offloading \
        --offload_stages text_encoder,denoising_transformer,denoising_transformer_2,vae
"""

import argparse

from tensorrt_llm import VisualGen, VisualGenArgs


def _parse_offload_stages(offload_stages: str | None) -> list[str] | None:
    if offload_stages is None:
        return None
    stages = [stage.strip() for stage in offload_stages.split(",") if stage.strip()]
    if not stages:
        raise ValueError("--offload_stages must contain at least one stage name")
    return stages


def main():
    parser = argparse.ArgumentParser(description="Wan Text-to-Video example")
    parser.add_argument(
        "--model",
        type=str,
        default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
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
        "--enable_offloading",
        action="store_true",
        help="Enable Wan T2V text-encoder/transformer-block CPU offloading.",
    )
    parser.add_argument(
        "--offload_stages",
        type=str,
        default=None,
        help=(
            "Comma-separated offload stage names, e.g. "
            "text_encoder,denoising_transformer,denoising_transformer_2,vae. "
            "Implies --enable_offloading."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="wan_t2v_output.avi",
        help="Path to save the output video",
    )
    args = parser.parse_args()

    # Engine config from shared YAML (optional); model-specific defaults apply otherwise.
    if args.visual_gen_args:
        visual_gen_args = VisualGenArgs.from_yaml(args.visual_gen_args)
    else:
        visual_gen_args = VisualGenArgs()

    offload_stages = _parse_offload_stages(args.offload_stages)
    if args.enable_offloading or offload_stages is not None:
        visual_gen_args.cpu_offload_config.enable = True
        if offload_stages is not None:
            visual_gen_args.cpu_offload_config.stages = offload_stages

    visual_gen = VisualGen(model=args.model, args=visual_gen_args)

    # --- Model-specific: T2V request construction ---
    # Start from per-model defaults (steps, guidance, seed, etc.) and override resolution/frames.
    params = visual_gen.default_params
    params.height = 480
    params.width = 832
    params.num_frames = 165

    output = visual_gen.generate(
        inputs="A cute cat playing piano",
        params=params,
    )

    output.save(args.output_path)
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
