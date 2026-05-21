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
    python wan_t2v.py --extra_visual_gen_options ../configs/wan2.2-t2v-fp4-1gpu.yaml
"""

import argparse

from tensorrt_llm import VisualGen, VisualGenArgs


def main():
    parser = argparse.ArgumentParser(description="Wan Text-to-Video example")
    parser.add_argument(
        "--model",
        type=str,
        default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        help="Model path or HuggingFace Hub ID",
    )
    parser.add_argument(
        "--extra_visual_gen_options",
        type=str,
        default=None,
        help="Path to YAML config (same as trtllm-serve --extra_visual_gen_options)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="wan_t2v_output.avi",
        help="Path to save the output video",
    )
    args = parser.parse_args()

    # Engine config from shared YAML (optional); model-specific defaults apply otherwise.
    extra_args = (
        VisualGenArgs.from_yaml(args.extra_visual_gen_options)
        if args.extra_visual_gen_options
        else None
    )
    visual_gen = VisualGen(model=args.model, args=extra_args)

    # --- Model-specific: T2V request construction ---
    # Query per-model defaults (resolution, steps, guidance, seed, etc.).
    params = visual_gen.default_params

    output = visual_gen.generate(
        inputs="A cat playing piano in a sunny room",
        params=params,
    )

    output.save(args.output_path)
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
