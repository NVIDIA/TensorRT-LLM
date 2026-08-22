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
"""LTX-2 / LTX-2.3 Text-to-Video generation with audio.

Both generations share the request construction and the Gemma3 text encoder, so
--model_type only selects the default checkpoint and output name. The pipeline
itself is chosen from the checkpoint config.

Usage:
    python ltx2.py
    python ltx2.py --visual_gen_args ../configs/ltx2-1gpu.yaml
    python ltx2.py --model_type ltx23 --visual_gen_args ../configs/ltx23-t2v-bf16-1gpu.yaml
    # Force two-stage on a checkpoint lacking the aux files:
    python ltx2.py --spatial_upsampler_path <upsampler.safetensors> \
        --distilled_lora_path <distilled-lora.safetensors>
"""

import argparse

from tensorrt_llm import VisualGen, VisualGenArgs

MODEL_DEFAULTS = {
    "ltx2": {"model": "Lightricks/LTX-2", "output_path": "ltx2_t2v_output.mp4"},
    "ltx23": {"model": "Lightricks/LTX-2.3", "output_path": "ltx23_t2v_output.mp4"},
}


def main():
    parser = argparse.ArgumentParser(description="LTX-2 / LTX-2.3 Text-to-Video example")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=sorted(MODEL_DEFAULTS),
        default="ltx2",
        help="LTX generation to run. Selects the default checkpoint and output name.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model path or HuggingFace Hub ID (defaults per --model_type)",
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
        "--spatial_upsampler_path",
        type=str,
        default=None,
        help=(
            "Spatial upsampler safetensors path. Setting both this and "
            "--distilled_lora_path forces two-stage inference. "
            "Auto-discovered from the checkpoint dir when unset."
        ),
    )
    parser.add_argument(
        "--distilled_lora_path",
        type=str,
        default=None,
        help=(
            "Distilled-LoRA safetensors path. Setting both this and "
            "--spatial_upsampler_path forces two-stage inference. "
            "Auto-discovered from the checkpoint dir when unset."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the output video (defaults per --model_type)",
    )
    args = parser.parse_args()

    defaults = MODEL_DEFAULTS[args.model_type]
    model = args.model or defaults["model"]
    output_path = args.output_path or defaults["output_path"]

    # Both generations require pipeline_config.text_encoder_path for the Gemma3
    # text encoder. The YAML path is preferred for production configs; the default
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
    # Two-stage auto-enables when both aux paths resolve (explicit here, or
    # auto-discovered from the checkpoint dir); pass both to force it on a
    # checkpoint that does not bundle them.
    for key, value in (
        ("spatial_upsampler_path", args.spatial_upsampler_path),
        ("distilled_lora_path", args.distilled_lora_path),
    ):
        if value is not None:
            extra_args.pipeline_config = {**extra_args.pipeline_config, key: value}
    visual_gen = VisualGen(model=model, args=extra_args)

    # --- Model-specific: T2V request construction ---
    # Start from the pipeline defaults and override the main request shape explicitly.
    params = visual_gen.default_params
    params.height = 512
    params.width = 768
    params.num_frames = 121
    params.frame_rate = 24.0
    params.num_inference_steps = 40
    params.guidance_scale = 4.0

    output = visual_gen.generate(
        inputs="A cinematic shot of a cat walking through a field of flowers",
        params=params,
    )

    output.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
