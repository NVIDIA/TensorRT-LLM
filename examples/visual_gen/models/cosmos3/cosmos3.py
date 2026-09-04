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
from typing import Any, Dict, Optional

from tensorrt_llm import VisualGen, VisualGenArgs
from tensorrt_llm._torch.visual_gen.models.cosmos3.transfer import TRANSFER_HINT_KEYS
from tensorrt_llm.visual_gen import MediaRef

_SCRIPT_DIR = Path(__file__).resolve().parent
_ACTION_MODES = ("policy", "forward_dynamics", "inverse_dynamics")
_TENSOR_OUTPUT_SUFFIXES = {".pt", ".safetensors"}

DEFAULT_PROMPT_FILE = "prompts/t2v.json"
DEFAULT_NEGATIVE_PROMPT_FILE = "cosmos3_negative_prompt.json"


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


def _is_prompt_file(value: str) -> bool:
    """Whether a ``--prompt``/``--negative_prompt`` value names an existing file."""
    return bool(value) and os.path.isfile(_resolve_path(value))


def _read_prompt_payload(path: str) -> Any:
    """Read a prompt file, decoding it as JSON when it parses and as text otherwise."""
    resolved = _resolve_path(path)
    if not os.path.isfile(resolved):
        raise ValueError(f"Prompt file {path!r} does not exist (resolved to {resolved!r}).")
    with open(resolved, encoding="utf-8") as f:
        raw = f.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip()


def load_prompt_file(path: str) -> Dict[str, Any]:
    """Load a Cosmos3 prompt file.

    Three shapes are accepted: an omni prompt object (``prompt`` plus optional
    ``vision_path`` / ``model_mode`` / ``enable_audio``), a structured caption
    object such as the ``assets/*_prompt.json`` files shipped with a checkpoint,
    or plain text. The latter two carry no options, so they yield ``prompt`` only.
    """
    data = _read_prompt_payload(path)
    if isinstance(data, str):
        if not data:
            raise ValueError(f"Prompt file {path!r} is empty.")
        return {"prompt": data}
    if not isinstance(data, dict):
        raise ValueError(
            f"Prompt file {path!r} must hold a JSON object or text, got {type(data).__name__}."
        )
    if "prompt" not in data:
        if not data:
            raise ValueError(f"Prompt file {path!r} is an empty JSON object.")
        return {"prompt": json.dumps(data)}
    if not data["prompt"]:
        raise ValueError(f"Prompt file {path!r} is missing a non-empty 'prompt' field.")
    return data


def load_negative_prompt_file(path: str) -> str:
    """Load a negative prompt file (structured JSON object or plain text)."""
    data = _read_prompt_payload(path)
    if isinstance(data, dict):
        return json.dumps(data)
    if isinstance(data, str):
        return data
    raise ValueError(
        f"Negative prompt file {path!r} must hold a JSON object or text, got {type(data).__name__}."
    )


def resolve_negative_prompt(
    *,
    negative_prompt: Optional[str],
    negative_prompt_file: Optional[str],
) -> str:
    """Pick the negative prompt: ``--negative_prompt``, then the file, then the default."""
    if negative_prompt is not None:
        # --negative_prompt takes either literal text or a path to a prompt file.
        if _is_prompt_file(negative_prompt):
            return load_negative_prompt_file(negative_prompt)
        return negative_prompt
    return load_negative_prompt_file(negative_prompt_file or DEFAULT_NEGATIVE_PROMPT_FILE)


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

    inline_prompt: Optional[str] = None
    if prompt is not None:
        # --prompt takes either literal text or a path to a prompt file.
        if _is_prompt_file(prompt):
            prompt_data = {**prompt_data, **load_prompt_file(prompt)}
        else:
            inline_prompt = prompt

    resolved_prompt = inline_prompt
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
    if mode not in _ACTION_MODES:
        raise SystemExit(
            f"Invalid --action_mode {args.action_mode!r}; expected one of {list(_ACTION_MODES)}."
        )
    args.action_mode = mode
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
        if args.raw_action_dim is None and args.domain_name is None and args.domain_id is None:
            raise SystemExit(f"{mode} requires --raw_action_dim, --domain_name, or --domain_id.")
    elif mode == "inverse_dynamics":
        if args.video_path is None:
            raise SystemExit(f"{mode} requires --video_path (an .mp4 or .avi file).")
        if args.raw_action_dim is None and args.domain_name is None and args.domain_id is None:
            raise SystemExit(f"{mode} requires --raw_action_dim, --domain_name, or --domain_id.")


def _resolved_output_path(path: str, action_mode: Optional[str]) -> str:
    if action_mode is None:
        return path
    output_path = Path(path)
    if output_path.suffix.lower() in _TENSOR_OUTPUT_SUFFIXES:
        return str(output_path)
    return str(output_path.with_suffix(".safetensors"))


def _default_action_output_path(output_path: str) -> str:
    stem = Path(output_path)
    return str(stem.with_suffix(".action.json"))


def _save_action_output(output, path: str, args: argparse.Namespace) -> None:
    """Write the trajectory plus the request that produced it.

    The mode and embodiment are this script's own inputs, so they are read
    from *args* rather than echoed back through the output schema.
    """
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
        "action_mode": args.action_mode,
        "domain_name": args.domain_name,
        "domain_id": args.domain_id,
        "raw_action_dim": action.shape[-1],
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
        help="Model path or HuggingFace Hub ID "
        "(nvidia/Cosmos3-Nano, nvidia/Cosmos3-Super, nvidia/Cosmos3-Edge)",
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
        help="Prompt text, or a path to a prompt file (overrides --prompt_file when both are set)",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=DEFAULT_PROMPT_FILE,
        help=f"Path to a prompt file; must exist (default: {DEFAULT_PROMPT_FILE})",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Negative prompt text, or a path to a negative prompt file "
        f"(overrides --negative_prompt_file; default: {DEFAULT_NEGATIVE_PROMPT_FILE})",
    )
    parser.add_argument(
        "--negative_prompt_file",
        type=str,
        default=None,
        help=f"Path to a negative prompt file; must exist "
        f"(default: {DEFAULT_NEGATIVE_PROMPT_FILE})",
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
        "--action_mode",
        type=str,
        default=None,
        choices=list(_ACTION_MODES),
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
        help="Action tokens to generate. Defaults to the domain preset or model default.",
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
        help=(
            "Reference video (MP4/AVI, decoded on worker NVDEC): V2V conditioning, "
            "or the observation clip for action inverse_dynamics"
        ),
    )
    parser.add_argument(
        "--action_resolution",
        type=int,
        default=None,
        choices=[256, 480, 704, 720],
        help=("Resolution bucket for action image sizing. Defaults to the domain preset or 480."),
    )
    parser.add_argument(
        "--action_fps",
        type=float,
        default=None,
        help="Action-token temporal rate for mRoPE (Hz). Defaults to frame_rate.",
    )
    parser.add_argument(
        "--view_point",
        type=str,
        default=None,
        choices=["ego_view", "third_person_view", "wrist_view", "concat_view"],
        help="Camera perspective for the action caption (default: ego_view).",
    )
    parser.add_argument(
        "--action_output_path",
        type=str,
        default=None,
        help="Path to save predicted action JSON (default: <output_stem>.action.json)",
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
    _validate_action_args(args, resolved_image_path=image_path)

    # Engine config from shared YAML (optional); model-specific defaults apply otherwise.
    extra_args = VisualGenArgs.from_yaml(args.visual_gen_args) if args.visual_gen_args else None
    visual_gen = VisualGen(model=args.model, args=extra_args)

    # --- Model-specific: T2V / TI2V request construction ---
    # Query per-model defaults (resolution, steps, guidance, seed, etc.).
    params = visual_gen.default_params
    if image_path is not None:
        params.image_reference = [MediaRef(content=image_path, format="path")]

    negative_prompt = resolve_negative_prompt(
        negative_prompt=args.negative_prompt,
        negative_prompt_file=args.negative_prompt_file,
    )

    if args.disable_duration_template:
        params.extra_params["use_duration_template"] = False
    if args.disable_resolution_template:
        params.extra_params["use_resolution_template"] = False
    if args.use_system_prompt is not None:
        params.extra_params["use_system_prompt"] = args.use_system_prompt
    params.extra_params["enable_audio"] = enable_audio
    params.extra_params["use_guardrails"] = not args.disable_guardrails
    params.extra_params["output_type"] = output_type

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
    if args.action_resolution is not None:
        params.extra_params["action_resolution"] = args.action_resolution
    if args.action_fps is not None:
        params.extra_params["action_fps"] = args.action_fps
    if args.view_point is not None:
        params.extra_params["view_point"] = args.view_point
    if args.action_json is not None:
        with open(args.action_json, encoding="utf-8") as f:
            params.extra_params["action"] = json.load(f)
    if args.video_path is not None:
        params.video_reference = [MediaRef(content=args.video_path, format="path")]
    if args.extra_params:
        # Merged last: explicit JSON wins over flag-derived values.
        params.extra_params.update(args.extra_params)
    # The pipeline fits the output to the reference's aspect when height/width
    # are unset, so there is nothing to do client-side.
    _load_transfer_controls(params.extra_params)

    params.negative_prompt = negative_prompt

    output = visual_gen.generate(
        inputs=prompt,
        params=params,
    )

    output_path = _resolved_output_path(args.output_path, args.action_mode)
    output.save(output_path)
    print(f"Saved: {output_path}")

    if args.action_mode is not None:
        action_path = args.action_output_path or _default_action_output_path(output_path)
        _save_action_output(output, action_path, args)
        if output.action is not None:
            print(f"Saved action: {action_path}")
            print(f"Action shape: {tuple(output.action.shape)}")
        else:
            print("Warning: action_mode was set but the output carried no action tensor.")

    print(output.metrics)


if __name__ == "__main__":
    main()
