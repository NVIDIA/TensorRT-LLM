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
reference frame, or use ``prompts/i2v.json`` which includes a ``vision_path``.

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

Example prompts live under ``prompts/`` (mirroring ``cosmos3-internal/inputs/omni``).

Usage::

    # Text-to-video
    python cosmos3_ti2v.py --model nvidia/Cosmos3-Nano \
        --prompt_file prompts/t2v.json \
        --visual_gen_args ../configs/cosmos3-nano-1gpu.yaml

    # Image-to-video (vision_path is read from the prompt file)
    python cosmos3_ti2v.py --model nvidia/Cosmos3-Nano \
        --prompt_file prompts/i2v.json \
        --visual_gen_args ../configs/cosmos3-nano-1gpu.yaml

    # Text-to-video with audio
    python cosmos3_ti2v.py --model nvidia/Cosmos3-Nano \
        --prompt_file prompts/t2av.json \
        --visual_gen_args ../configs/cosmos3-nano-1gpu.yaml

    # Text-to-image
    python cosmos3_ti2v.py --model nvidia/Cosmos3-Nano \
        --prompt_file prompts/t2i.json \
        --visual_gen_args ../configs/cosmos3-nano-1gpu.yaml \
        --output_path output.png

    # Inline prompt (``--prompt`` or a JSON file path)
    python cosmos3_ti2v.py --model nvidia/Cosmos3-Nano \
        --prompt "A cute puppy playing with a ball in a park" \
        --visual_gen_args ../configs/cosmos3-nano-1gpu.yaml
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from tensorrt_llm import VisualGen, VisualGenArgs

_SCRIPT_DIR = Path(__file__).resolve().parent


def _resolve_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_file():
        return str(candidate.resolve())
    relative_to_script = _SCRIPT_DIR / path
    if relative_to_script.is_file():
        return str(relative_to_script.resolve())
    return path


def load_prompt_file(path: str) -> Dict[str, Any]:
    """Load a Cosmos3 omni prompt JSON (``prompt``, optional ``vision_path``, etc.)."""
    resolved = _resolve_path(path)
    with open(resolved, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Prompt file must be a JSON object, got {type(data)!r}.")
    if not data.get("prompt"):
        raise ValueError(f"Prompt file {resolved!r} is missing a non-empty 'prompt' field.")
    return data


def resolve_prompt_and_options(
    *,
    prompt: Optional[str],
    prompt_file: Optional[str],
    image_path: Optional[str],
    enable_audio: bool,
    output_type: str,
) -> tuple[str, Optional[str], bool, str]:
    """Merge CLI args with optional prompt-file defaults."""
    prompt_data: Dict[str, Any] = {}
    if prompt_file is not None:
        prompt_data = load_prompt_file(prompt_file)

    resolved_prompt = prompt
    if resolved_prompt is None:
        resolved_prompt = prompt_data.get("prompt")
    if not resolved_prompt:
        raise ValueError("Provide --prompt or --prompt_file with a 'prompt' field.")

    resolved_image = image_path
    if resolved_image is None:
        resolved_image = prompt_data.get("vision_path") or prompt_data.get("image_path")

    resolved_enable_audio = enable_audio or bool(prompt_data.get("enable_audio", False))

    resolved_output_type = output_type
    model_mode = str(prompt_data.get("model_mode", "")).lower()
    if model_mode == "text2image" and output_type == "video":
        resolved_output_type = "image"

    return resolved_prompt, resolved_image, resolved_enable_audio, resolved_output_type


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
        default=None,
        help="Text prompt for generation (overrides --prompt_file when both are set)",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="prompts/t2v.json",
        help="Path to a JSON prompt file (default: prompts/t2v.json)",
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
        help="Optional conditioning image path or URL for I2V/TI2V",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="cosmos3_ti2v_output.mp4",
        help="Path to save the output video",
    )
    parser.add_argument(
        "--disable_duration_template",
        action="store_true",
        help="Disable duration metadata template (enabled by default, matching cosmos-framework CLI)",
    )
    parser.add_argument(
        "--disable_resolution_template",
        action="store_true",
        help="Disable resolution metadata template (enabled by default, matching cosmos-framework CLI)",
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

    prompt, image_path, enable_audio, output_type = resolve_prompt_and_options(
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        image_path=args.image_path,
        enable_audio=args.enable_audio,
        output_type=args.output_type,
    )

    # Engine config from shared YAML (optional); model-specific defaults apply otherwise.
    extra_args = VisualGenArgs.from_yaml(args.visual_gen_args) if args.visual_gen_args else None
    visual_gen = VisualGen(model=args.model, args=extra_args)

    # --- Model-specific: T2V / TI2V request construction ---
    # Query per-model defaults (resolution, steps, guidance, seed, etc.).
    params = visual_gen.default_params
    if image_path is not None:
        params.image = image_path

    negative_prompt_path = _resolve_path(args.negative_prompt)
    if args.negative_prompt is not None:
        if os.path.isfile(negative_prompt_path) and negative_prompt_path.endswith(".json"):
            with open(negative_prompt_path, encoding="utf-8") as f:
                negative_prompt = json.load(f)
        else:
            negative_prompt = args.negative_prompt
    else:
        negative_prompt = None

    if args.disable_duration_template:
        params.extra_params["use_duration_template"] = False
    if args.disable_resolution_template:
        params.extra_params["use_resolution_template"] = False
    params.extra_params["use_system_prompt"] = args.use_system_prompt
    params.extra_params["enable_audio"] = enable_audio
    params.extra_params["use_guardrails"] = not args.disable_guardrails
    params.extra_params["output_type"] = output_type

    if negative_prompt is None:
        params.negative_prompt = None
    elif isinstance(negative_prompt, str):
        params.negative_prompt = negative_prompt
    else:
        params.negative_prompt = json.dumps(negative_prompt)

    output = visual_gen.generate(
        inputs=prompt,
        params=params,
    )

    output.save(args.output_path)
    print(f"Saved: {args.output_path}")
    print(output.metrics)


if __name__ == "__main__":
    main()
