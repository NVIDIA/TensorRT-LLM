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
r"""Cosmos3 Text(+Image)-to-Video generation.

Cosmos3 OmniMoT supports text-only (T2V) and image-conditioned (I2V/TI2V)
generation from the same checkpoint. Pass ``--image_path`` to condition on a
reference frame.

Checkpoints (pass the Hub ID or local path via ``--model``):

- `nvidia/Cosmos3-Nano <https://huggingface.co/nvidia/Cosmos3-Nano>`_
- `nvidia/Cosmos3-Super <https://huggingface.co/nvidia/Cosmos3-Super>`_

Guardrails are enabled by default (required by the
`NVIDIA Open Model License Agreement
<https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license>`_).
Install and authenticate as follows::

    pip install cosmos_guardrail==0.3.0 && pip uninstall opencv-python

Accept the terms for the guardrail checkpoint at
https://huggingface.co/nvidia/Cosmos-1.0-Guardrail and set a valid ``HF_TOKEN``
(the checkpoint is downloaded automatically on first run).

To run without guardrails (you are responsible for safe deployment)::

    export TRTLLM_DISABLE_COSMOS3_GUARDRAILS=1

Deployment configs (``examples/visual_gen/configs/``):

- ``cosmos3-nano-1gpu.yaml`` — 1 GPU, FP8 dynamic quant
- ``cosmos3-super-4gpu.yaml`` — 4 GPU, CFG + Ulysses + parallel VAE

Usage:
    python cosmos3_ti2v.py --model nvidia/Cosmos3-Nano \
        --prompt "The video opens with a view of a well-lit indoor space featuring a " \\
        "wooden display case with compartments filled with various fruits, " \\
        "including bananas, apples, pears, oranges, and carambolas. " \\
        "The bananas are neatly arranged in the middle compartment, while apples " \\
        "are in the left and a mix of pears, oranges, and carambolas are in the " \\
        "right. " \\
        "Two robotic arms with grippers are positioned at the bottom of the frame, " \\
        "with the one on the left remaining stationary, partially obscuring the " \\
        "apples. " \\
        "The robotic arm on the right begins its action, extending towards the " \\
        "right side of the display case. " \\
        "It carefully picks up a pear from the fruit section, placing it into a " \\
        "plastic bag in the shopping cart nearby, which has red handles. " \\
        "After securing the pear, the arm retracts back to its original position. " \\
        "The process repeats as the robotic arm picks up an orange and places it " \\
        "in the bag, followed by a carambola. " \\
        "The final frame captures the robotic arm returning to its initial " \\
        "position, leaving the display case and surrounding area unchanged. " \\
        "The video showcases a seamless and efficient automated fruit-picking " \\
        "process, highlighting the precision and efficiency of modern robotics " \\
        "in a retail setting." \
        --visual_gen_args ../configs/cosmos3-nano-1gpu.yaml

    python cosmos3_ti2v.py --model nvidia/Cosmos3-Nano \
        --prompt "A low-angle tracking shot follows a man riding a vintage black motorcycle " \\
        "across a lush green grassy yard. Sunlight filters through overhead trees, casting " \\
        "dappled shadows across the vibrating chrome exhaust and the rider's leather jacket. " \\
        "He kicks up small blades of grass as he maneuvers the bike. He gradually decelerates, " \\
        "the front fork compressing slightly as he brakes to a smooth halt beside another " \\
        "individual standing in the shade. The camera settles into a medium two-shot, capturing " \\
        "the rider lifting his visor to speak, his face framed by a matte helmet. The video is " \\
        "8 seconds long and is of 24 FPS. This video is of 1280x720 resolution. Audio description: " \\
        "The rhythmic, mechanical chugging of a four-stroke motorcycle engine dominates the " \\
        "foreground, characterized by a throaty, guttural timbre. Periodic high-pitched revs " \\
        "punctuate the steady idle as the throttle is twisted. The sound of tires crunching " \\
        "softly over dry grass and twigs provides a textured background layer. As the vehicle " \\
        "slows, the engine note drops to a low-frequency rumble before clicking into neutral. " \\
        "A muffled, mid-range male voice begins speaking, accompanied by the metallic clink of " \\
        "a helmet visor snapping upward and the faint chirping of distant birds in an open-air " \\
        "environment." \
        --visual_gen_args ../configs/cosmos3-nano-1gpu.yaml \
        --enable_audio

    python cosmos3_ti2v.py --model nvidia/Cosmos3-Nano \
        --prompt "A cute puppy playing with a ball in a park" \
        --visual_gen_args ../configs/cosmos3-nano-1gpu.yaml \
        --output_type image \
        --output_path output.png
"""

import argparse
import json
import os

from tensorrt_llm import VisualGen, VisualGenArgs


def main():
    parser = argparse.ArgumentParser(description="Cosmos3 Text(+Image)-to-Video example")
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/Cosmos3-Nano",
        help="Model path or HuggingFace Hub ID (nvidia/Cosmos3-Nano, nvidia/Cosmos3-Super)",
    )
    parser.add_argument(
        "--visual_gen_args",
        dest="visual_gen_args",
        type=str,
        default=None,
        help="Path to YAML config (same as trtllm-serve --visual_gen_args)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="cosmos3_negative_prompt.json",
        help="Text prompt or path to JSON file for negative prompt",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Optional conditioning image path for I2V/TI2V",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="cosmos3_ti2v_output.mp4",
        help="Path to save the output video",
    )
    parser.add_argument(
        "--enable_duration_template", action="store_true", help="Enable duration template in prompt"
    )
    parser.add_argument(
        "--enable_resolution_template",
        action="store_true",
        help="Enable resolution template in prompt",
    )
    parser.add_argument(
        "--use_system_prompt", action="store_true", help="Use system prompt in prompt"
    )
    parser.add_argument("--enable_audio", action="store_true", help="Enable audio generation")
    parser.add_argument(
        "--output_type", type=str, default="video", help="Output type (video, image)"
    )

    # Guardrails
    parser.add_argument(
        "--disable_guardrails", action="store_true", help="NOT RECOMMENDED: Disable guardrails"
    )
    args = parser.parse_args()

    # Engine config from shared YAML (optional); model-specific defaults apply otherwise.
    extra_args = VisualGenArgs.from_yaml(args.visual_gen_args) if args.visual_gen_args else None
    visual_gen = VisualGen(model=args.model, args=extra_args)

    # --- Model-specific: T2V / TI2V request construction ---
    # Query per-model defaults (resolution, steps, guidance, seed, etc.).
    params = visual_gen.default_params
    if args.image_path is not None:
        params.image = args.image_path

    if args.negative_prompt is not None:
        if os.path.isfile(args.negative_prompt) and args.negative_prompt.endswith(".json"):
            negative_prompt = json.load(open(args.negative_prompt))
        else:
            negative_prompt = args.negative_prompt
    else:
        negative_prompt = None

    params.extra_params["use_duration_template"] = args.enable_duration_template
    params.extra_params["use_resolution_template"] = args.enable_resolution_template
    params.extra_params["use_system_prompt"] = args.use_system_prompt
    params.extra_params["enable_audio"] = args.enable_audio
    params.extra_params["use_guardrails"] = not args.disable_guardrails
    params.extra_params["output_type"] = args.output_type

    if negative_prompt is None:
        params.negative_prompt = None
    elif isinstance(negative_prompt, str):
        params.negative_prompt = negative_prompt
    else:
        params.negative_prompt = json.dumps(negative_prompt)

    output = visual_gen.generate(
        inputs=args.prompt,
        params=params,
    )

    output.save(args.output_path)
    print(f"Saved: {args.output_path}")
    print(output.metrics)


if __name__ == "__main__":
    main()
