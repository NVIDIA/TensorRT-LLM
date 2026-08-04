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
    python cosmos3_ti2v.py --model nvidia/Cosmos3-Nano \\
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
        "in a retail setting." \\
        --visual_gen_args ../configs/cosmos3-nano-1gpu.yaml
"""

import argparse

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
    args = parser.parse_args()

    # Engine config from shared YAML (optional); model-specific defaults apply otherwise.
    extra_args = VisualGenArgs.from_yaml(args.visual_gen_args) if args.visual_gen_args else None
    visual_gen = VisualGen(model=args.model, args=extra_args)

    # --- Model-specific: T2V / TI2V request construction ---
    # Query per-model defaults (resolution, steps, guidance, seed, etc.).
    params = visual_gen.default_params
    if args.image_path is not None:
        params.image = args.image_path

    output = visual_gen.generate(
        inputs=args.prompt,
        params=params,
    )

    output.save(args.output_path)
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
