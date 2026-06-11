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

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from tensorrt_llm import VisualGen, VisualGenArgs
from tensorrt_llm._torch.visual_gen.models.cosmos3.action import VIDEO_RES_SIZE_INFO
from tensorrt_llm._torch.visual_gen.models.cosmos3.defaults import (
    COSMOS3_ACTION_PARAMS,
    get_domain_preset,
    resolve_domain_action_config,
)

_SCRIPT_DIR = Path(__file__).resolve().parent

ACTION_MODES = frozenset({"policy", "forward_dynamics", "inverse_dynamics"})


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


def _validate_action_args(
    args: argparse.Namespace, resolved_image_path: Optional[str] = None
) -> None:
    if args.action_mode is None:
        return

    # The first frame may come from --image_path or a prompt file's vision_path.
    has_first_frame = resolved_image_path is not None or args.video_path is not None

    mode = args.action_mode.strip().lower()
    if mode not in ACTION_MODES:
        raise SystemExit(
            f"Invalid --action_mode {args.action_mode!r}; expected one of {sorted(ACTION_MODES)}."
        )
    if args.enable_audio:
        raise SystemExit("Cosmos3 does not support joint action and audio generation.")
    if args.output_type != "video":
        raise SystemExit("Action generation requires --output_type video.")

    if mode == "forward_dynamics":
        if args.action_json is None:
            raise SystemExit(f"{mode} requires --action_json.")
        if not has_first_frame:
            raise SystemExit(
                f"{mode} requires --image_path, a prompt-file vision_path, or --video_path "
                "for the first frame."
            )
    elif mode == "policy":
        if not has_first_frame:
            raise SystemExit(
                f"{mode} requires --image_path, a prompt-file vision_path, or --video_path "
                "for the first frame."
            )
        preset = get_domain_preset(args.domain_name, args.domain_id)
        effective_raw_dim = args.raw_action_dim or (preset or {}).get("raw_action_dim")
        if effective_raw_dim is None:
            raise SystemExit(
                f"{mode} requires --raw_action_dim or a known --domain_name with a preset."
            )
    elif mode == "inverse_dynamics":
        if args.video_path is None:
            raise SystemExit(
                f"{mode} requires --video_path (frame directory, .mp4/.avi, or image)."
            )
        preset = get_domain_preset(args.domain_name, args.domain_id)
        effective_raw_dim = args.raw_action_dim or (preset or {}).get("raw_action_dim")
        if effective_raw_dim is None:
            raise SystemExit(
                f"{mode} requires --raw_action_dim or a known --domain_name with a preset."
            )


def _apply_action_generation_params(params, args: argparse.Namespace) -> None:
    """Set action defaults on the request; domain presets override generic 480p."""
    cfg = resolve_domain_action_config(
        domain_name=args.domain_name,
        domain_id=args.domain_id,
        raw_action_dim=args.raw_action_dim,
        action_chunk_size=args.action_chunk_size,
        action_resolution=args.action_resolution,
    )
    bucket = str(cfg["action_resolution"])
    width, height = VIDEO_RES_SIZE_INFO[bucket]["16,9"]
    params.width = width
    params.height = height
    params.num_frames = cfg["num_frames"]
    params.num_inference_steps = COSMOS3_ACTION_PARAMS["num_inference_steps"]
    params.guidance_scale = COSMOS3_ACTION_PARAMS["guidance_scale"]
    params.frame_rate = cfg["frame_rate"]
    params.extra_params["action_chunk_size"] = cfg["action_chunk_size"]
    if cfg["raw_action_dim"] is not None:
        params.extra_params["raw_action_dim"] = cfg["raw_action_dim"]
    params.extra_params["action_resolution"] = cfg["action_resolution"]


def _default_action_output_path(video_path: str) -> str:
    stem = Path(video_path)
    if stem.suffix:
        return str(stem.with_name(f"{stem.stem}_action.json"))
    return f"{video_path}_action.json"


def _save_action_output(output, path: str) -> None:
    if output.action is None:
        return

    action = output.action
    if action.ndim == 3 and action.shape[0] == 1:
        action_data = action[0].tolist()
        shape = list(action.shape[1:])
    else:
        action_data = action.tolist()
        shape = list(action.shape)

    payload = {
        "action_mode": output.action_mode,
        "domain_id": output.domain_id,
        "raw_action_dim": output.raw_action_dim,
        "shape": shape,
        "dtype": str(action.dtype).replace("torch.", ""),
        "data": action_data,
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


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
        "--use_system_prompt", action="store_true", help="Use system prompt in prompt"
    )
    parser.add_argument("--enable_audio", action="store_true", help="Enable audio generation")
    parser.add_argument(
        "--action_mode",
        type=str,
        default=None,
        choices=sorted(ACTION_MODES),
        help="Action mode: policy, forward_dynamics, or inverse_dynamics",
    )
    parser.add_argument(
        "--domain_name",
        type=str,
        default=None,
        help="Embodiment domain name (e.g. bridge_orig_lerobot, av, droid_lerobot)",
    )
    parser.add_argument(
        "--domain_id",
        type=int,
        default=None,
        help="Embodiment domain id (alternative to --domain_name)",
    )
    parser.add_argument(
        "--raw_action_dim",
        type=int,
        default=None,
        help="Raw action DOF for policy/inverse_dynamics",
    )
    parser.add_argument(
        "--action_chunk_size",
        type=int,
        default=None,
        help=f"Action tokens to generate (default {COSMOS3_ACTION_PARAMS['action_chunk_size']})",
    )
    parser.add_argument(
        "--action_json",
        type=str,
        default=None,
        help="JSON file with action trajectory [T, D] for forward_dynamics",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Frame directory, .mp4/.avi video, or image path for inverse_dynamics",
    )
    parser.add_argument(
        "--action_resolution",
        type=int,
        default=None,
        choices=[256, 480, 704, 720],
        help=("Resolution bucket for action image sizing. Defaults to the domain preset or 480."),
    )
    parser.add_argument(
        "--action_output_path",
        type=str,
        default=None,
        help="Path to save predicted action JSON (default: <output_stem>_action.json)",
    )
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
    _validate_action_args(args, resolved_image_path=image_path)

    # Engine config from shared YAML (optional); model-specific defaults apply otherwise.
    extra_args = VisualGenArgs.from_yaml(args.visual_gen_args) if args.visual_gen_args else None
    visual_gen = VisualGen(model=args.model, args=extra_args)

    # --- Model-specific: T2V / TI2V request construction ---
    # Query per-model defaults (resolution, steps, guidance, seed, etc.).
    params = visual_gen.default_params
    if image_path is not None:
        params.image = image_path

    if args.action_mode is not None:
        _apply_action_generation_params(params, args)

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
    params.extra_params["action_resolution"] = args.action_resolution

    if args.action_mode is not None:
        params.extra_params["action_mode"] = args.action_mode
    if args.domain_name is not None:
        params.extra_params["domain_name"] = args.domain_name
    if args.domain_id is not None:
        params.extra_params["domain_id"] = args.domain_id
    if args.raw_action_dim is not None:
        params.extra_params["raw_action_dim"] = args.raw_action_dim
    if args.action_chunk_size is not None:
        params.extra_params["action_chunk_size"] = args.action_chunk_size
    if args.action_json is not None:
        with open(args.action_json, encoding="utf-8") as f:
            params.extra_params["action"] = json.load(f)
    if args.video_path is not None:
        params.extra_params["video"] = args.video_path

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

    if args.action_mode is not None:
        action_path = args.action_output_path or _default_action_output_path(args.output_path)
        _save_action_output(output, action_path)
        if output.action is not None:
            print(f"Saved action: {action_path}")
            print(
                f"Action shape: {tuple(output.action.shape)}, "
                f"raw_action_dim={output.raw_action_dim}, domain_id={output.domain_id}"
            )
        else:
            print("Warning: action_mode was set but the output carried no action tensor.")

    print(output.metrics)


if __name__ == "__main__":
    main()
