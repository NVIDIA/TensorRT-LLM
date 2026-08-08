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
"""Cosmos3 Text(+Image/Video)-to-Video(+Audio) generation.

One checkpoint serves T2V, T2I, I2V/TI2V, V2V, Transfer and T2AV;
``prompts/`` holds a prompt file per mode and ``--help`` lists the flags.
See ``README.md`` in this directory for the checkpoints, guardrail setup,
deployment configs, and a worked command line per mode.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from tensorrt_llm import VisualGen, VisualGenArgs
from tensorrt_llm._torch.visual_gen.models.cosmos3.transfer import TRANSFER_HINT_KEYS

_SCRIPT_DIR = Path(__file__).resolve().parent


def _resolve_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_file():
        return str(candidate.resolve())
    relative_to_script = _SCRIPT_DIR / path
    if relative_to_script.is_file():
        return str(relative_to_script.resolve())
    return path


def _load_transfer_controls(extra_params: dict[str, Any]) -> None:
    """Read precomputed transfer controls into ``control`` bytes, client-side.

    A hint may name a control file (``{"edge": "ctrl.mp4"}`` or
    ``{"edge": {"control_path": "ctrl.mp4"}}``); the worker only accepts encoded
    bytes, so the media is read here.
    """
    for key in TRANSFER_HINT_KEYS:
        hint = extra_params.get(key)
        if isinstance(hint, str):
            hint = {"control_path": hint}
        if not isinstance(hint, dict):
            continue
        control_path = hint.pop("control_path", None)
        if control_path is None:
            continue
        if not isinstance(control_path, str) or not control_path.strip():
            raise ValueError(
                f"--extra_params {key}.control_path must be a non-empty file path, "
                f"got {control_path!r}."
            )
        hint["control"] = Path(_resolve_path(control_path)).read_bytes()
        extra_params[key] = hint


def _json_object(text: str) -> dict[str, Any]:
    """Argparse type for a JSON *object*.

    ``json.loads`` alone also accepts arrays, scalars and null, which then
    either fail deep in the merge or, for ``[]``, succeed while doing nothing.
    """
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError(
            f"expected a JSON object, got {type(value).__name__}: {text!r}"
        )
    return value


def load_prompt_file(path: str) -> dict[str, Any]:
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
    prompt: str | None,
    prompt_file: str | None,
    image_path: str | None,
    enable_audio: bool,
    output_type: str,
) -> tuple[str, str | None, bool, str]:
    """Merge CLI args with optional prompt-file defaults."""
    prompt_data: dict[str, Any] = {}
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
    parser = argparse.ArgumentParser(description="Cosmos3 Text(+Image)-to-Video(+Audio) example")
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
        default="cosmos3_output.mp4",
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
        "--use_system_prompt",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Prepend the Cosmos3 system prompt (--no-use_system_prompt to disable). "
            "When omitted, V2V uses it and every other mode takes the checkpoint's "
            "declared default."
        ),
    )
    parser.add_argument("--enable_audio", action="store_true", help="Enable audio generation")
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Reference video for V2V: a local MP4/AVI file (decoded on worker NVDEC)",
    )
    parser.add_argument(
        "--output_type", type=str, default="video", help="Output type (video, image)"
    )
    parser.add_argument(
        "--extra_params",
        type=_json_object,
        default=None,
        help=(
            "Model-specific extra params as a JSON object, merged last (overrides "
            "flag-derived values). Keys are validated against the pipeline's "
            "extra_param_specs. Transfer example: "
            '\'{"edge": true, "blur": true, "control_guidance": 1.5}\' with --video_path, '
            'or \'{"edge": "/path/control.mp4"}\' for a precomputed control (read here and '
            "sent as encoded bytes)."
        ),
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
    if args.use_system_prompt is not None:
        params.extra_params["use_system_prompt"] = args.use_system_prompt
    params.extra_params["enable_audio"] = enable_audio
    params.extra_params["use_guardrails"] = not args.disable_guardrails
    params.extra_params["output_type"] = output_type

    if args.video_path is not None:
        params.extra_params["video"] = Path(args.video_path).read_bytes()
    if args.extra_params:
        # Merged last: explicit JSON wins over flag-derived values.
        params.extra_params.update(args.extra_params)
    # The pipeline fits the output to the reference's aspect when height/width
    # are unset, so there is nothing to do client-side.
    _load_transfer_controls(params.extra_params)

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
