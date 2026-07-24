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
"""Per-model default generation parameters for Cosmos3 pipelines.

Shared by the Cosmos3 OmniMoT text-to-video and image-to-video generation paths.

Action generation
-----------------
``COSMOS3_DOMAIN_PRESETS`` lists training-aligned defaults per embodiment. When
``domain_name`` (or a uniquely mapped ``domain_id``) is set, the pipeline fills
omitted ``raw_action_dim``, ``action_chunk_size``, ``num_frames``,
``action_resolution``, and ``frame_rate`` from the preset and logs a warning if
explicit values differ. See Cosmos3 omni ``action_*.json`` inputs for reference
configs (bridge, av, droid, libero, etc.).
"""

from typing import Any, TypedDict

from tensorrt_llm._torch.visual_gen.models.cosmos3.action import (
    COSMOS3_ACTION_RESOLUTIONS,
    EMBODIMENT_TO_DOMAIN_ID,
    normalize_action_resolution,
)
from tensorrt_llm._torch.visual_gen.pipeline import ExtraParamSchema

# ---------------------------------------------------------------------------
# Constant tables
# ---------------------------------------------------------------------------

COSMOS3_720P_PARAMS = {
    "height": 720,
    "width": 1280,
    "num_inference_steps": 35,
    "guidance_scale": 6.0,
    "max_sequence_length": 4096,
    "num_frames": 189,
    "frame_rate": 24.0,
}

# Fields merged by the executor for every request. Modality-specific values
# (height/width/num_frames/steps/guidance) are declared with ``None`` so the
# executor accepts explicit overrides; ``forward()`` resolves unset fields from
# T2V/T2I/action context.
COSMOS3_PIPELINE_DEFAULTS = {
    "height": None,
    "width": None,
    "num_frames": None,
    "num_inference_steps": None,
    "guidance_scale": None,
    "max_sequence_length": COSMOS3_720P_PARAMS["max_sequence_length"],
    "frame_rate": COSMOS3_720P_PARAMS["frame_rate"],
}

# Text-to-image (``output_type="image"``) defaults. Applied when the request
# field is ``None``.
COSMOS3_T2I_PARAMS = {
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50,
    "guidance_scale": 7.0,
    "flow_shift": 3.0,
    "guidance_interval": (400.0, 1000.0),
}

COSMOS3_ACTION_PARAMS = {
    "action_chunk_size": 16,
    "num_frames": 17,
    "num_inference_steps": 30,
    "guidance_scale": 1.0,
    "flow_shift": 5.0,
    "frame_rate": 24.0,
}


class Cosmos3DomainPreset(TypedDict, total=False):
    """Recommended action-generation settings for a trained embodiment."""

    raw_action_dim: int
    action_chunk_size: int
    num_frames: int
    action_resolution: int
    frame_rate: float


# Training-aligned defaults. Values mirror Cosmos3 omni action JSON examples where available.
COSMOS3_DOMAIN_PRESETS: dict[str, Cosmos3DomainPreset] = {
    # WidowX bridge; 7-DOF arm + gripper in 10-D state.
    "bridge_orig_lerobot": {
        "raw_action_dim": 10,
        "action_chunk_size": 16,
        "num_frames": 17,
        "action_resolution": 480,
        "frame_rate": 5.0,
    },
    # Autonomous-vehicle steering/throttle; longer action horizon.
    "av": {
        "raw_action_dim": 9,
        "action_chunk_size": 60,
        "num_frames": 61,
        "action_resolution": 480,
        "frame_rate": 10.0,
    },
    # 6-DoF camera pose + shutter; matches AV-style horizon.
    "camera_pose": {
        "raw_action_dim": 9,
        "action_chunk_size": 60,
        "num_frames": 61,
        "action_resolution": 480,
        "frame_rate": 30.0,
    },
    # Franka single-arm tabletop; same domain_id as robomind-franka.
    "droid_lerobot": {
        "raw_action_dim": 10,
        "action_chunk_size": 16,
        "num_frames": 17,
        "action_resolution": 480,
        "frame_rate": 15.0,
    },
    # LIBERO sim single-arm; lower action resolution bucket.
    "libero": {
        "raw_action_dim": 10,
        "action_chunk_size": 16,
        "num_frames": 17,
        "action_resolution": 256,
        "frame_rate": 10.0,
    },
    # MANO hand pose; high-DOF hand articulation.
    "hand_pose": {
        "raw_action_dim": 57,
        "action_chunk_size": 16,
        "num_frames": 17,
        "action_resolution": 480,
        "frame_rate": 24.0,
    },
    # AgiBot humanoid; shared domain_id with agibot_gear_gripper*.
    "agibotworld": {
        "raw_action_dim": 29,
        "action_chunk_size": 16,
        "num_frames": 17,
        "action_resolution": 480,
        "frame_rate": 10.0,
    },
    # Google Robot (RT-1 / fractal) single-arm.
    "fractal": {
        "raw_action_dim": 10,
        "action_chunk_size": 16,
        "num_frames": 17,
        "action_resolution": 480,
        "frame_rate": 5.0,
    },
    # 2-D planar push task.
    "pusht": {
        "raw_action_dim": 2,
        "action_chunk_size": 16,
        "num_frames": 17,
        "action_resolution": 256,
        "frame_rate": 10.0,
    },
    # UMI handheld gripper setup.
    "umi": {
        "raw_action_dim": 10,
        "action_chunk_size": 16,
        "num_frames": 17,
        "action_resolution": 480,
        "frame_rate": 10.0,
    },
}

# Map alias domain_name keys to a canonical preset entry.
COSMOS3_DOMAIN_PRESET_ALIASES: dict[str, str] = {
    "robomind-franka": "droid_lerobot",
    "robomind-franka-dual": "droid_lerobot",
    "robomind-ur": "droid_lerobot",
    "agibot_gear_gripper": "agibotworld",
    "agibot_gear_gripper_ext": "agibotworld",
    "galbot": "agibotworld",
}


def canonical_domain_preset_key(
    domain_name: str | None = None,
    domain_id: str | int | None = None,
) -> str | None:
    if domain_name is not None and str(domain_name).strip():
        key = str(domain_name).strip().lower()
        key = COSMOS3_DOMAIN_PRESET_ALIASES.get(key, key)
        if key in COSMOS3_DOMAIN_PRESETS:
            return key
        return None

    if domain_id is None:
        return None

    resolved_id = int(domain_id)
    if resolved_id == 0:
        return None

    candidates: list[str] = []
    for name, mapped_id in EMBODIMENT_TO_DOMAIN_ID.items():
        if mapped_id != resolved_id:
            continue
        canon = COSMOS3_DOMAIN_PRESET_ALIASES.get(name, name)
        if canon in COSMOS3_DOMAIN_PRESETS and canon not in candidates:
            candidates.append(canon)

    if len(candidates) == 1:
        return candidates[0]
    return None


def get_domain_preset(
    domain_name: str | None = None,
    domain_id: str | int | None = None,
) -> Cosmos3DomainPreset | None:
    key = canonical_domain_preset_key(domain_name, domain_id)
    if key is None:
        return None
    return COSMOS3_DOMAIN_PRESETS[key]


def resolve_domain_action_config(
    *,
    domain_name: str | None = None,
    domain_id: str | int | None = None,
    raw_action_dim: int | None = None,
    action_chunk_size: int | None = None,
    action_resolution: int | None = None,
    frame_rate: float | None = None,
    action_fps: float | None = None,
    num_frames: int | None = None,
) -> dict[str, Any]:
    """Merge user action params with domain presets and generic fallbacks."""
    preset_key = canonical_domain_preset_key(domain_name, domain_id)
    preset = COSMOS3_DOMAIN_PRESETS.get(preset_key) if preset_key else None
    warnings: list[str] = []

    domain_requested = (domain_name is not None and str(domain_name).strip() != "") or (
        domain_id is not None and str(domain_id).strip() not in {"", "0"}
    )
    if domain_requested and preset is None:
        warnings.append(
            "Cosmos3 action domain preset was not found for "
            f"domain_name={domain_name!r}, domain_id={domain_id!r}; "
            "using generic action defaults for omitted fields."
        )

    def _resolve_field(
        field: str,
        current: Any,
        *,
        fallback: Any = None,
    ) -> Any:
        recommended = preset.get(field) if preset else None
        if current is not None:
            if recommended is not None and current != recommended:
                warnings.append(
                    f"Cosmos3 {field}={current} differs from recommended "
                    f"{recommended} for domain {preset_key!r}."
                )
            return current
        if recommended is not None:
            return recommended
        return fallback

    resolved_raw_action_dim = _resolve_field("raw_action_dim", raw_action_dim)
    resolved_chunk = _resolve_field(
        "action_chunk_size",
        action_chunk_size,
        fallback=COSMOS3_ACTION_PARAMS["action_chunk_size"],
    )
    resolved_resolution = normalize_action_resolution(
        _resolve_field(
            "action_resolution",
            action_resolution,
            fallback=480,
        )
    )
    resolved_frame_rate = _resolve_field(
        "frame_rate",
        frame_rate,
        fallback=COSMOS3_ACTION_PARAMS["frame_rate"],
    )
    resolved_num_frames = _resolve_field("num_frames", num_frames)
    if resolved_num_frames is None:
        resolved_num_frames = int(resolved_chunk) + 1
    resolved_action_fps = (
        float(action_fps) if action_fps is not None else float(resolved_frame_rate)
    )
    if resolved_raw_action_dim is not None and int(resolved_raw_action_dim) <= 0:
        raise ValueError(f"Cosmos3 raw_action_dim must be positive, got {resolved_raw_action_dim}.")
    if int(resolved_chunk) <= 0:
        raise ValueError(f"Cosmos3 action_chunk_size must be positive, got {resolved_chunk}.")
    if float(resolved_frame_rate) <= 0.0:
        raise ValueError(f"Cosmos3 frame_rate must be positive, got {resolved_frame_rate}.")
    if resolved_action_fps <= 0.0:
        raise ValueError(f"Cosmos3 action_fps must be positive, got {resolved_action_fps}.")
    if int(resolved_num_frames) <= 0:
        raise ValueError(f"Cosmos3 num_frames must be positive, got {resolved_num_frames}.")

    return {
        "raw_action_dim": resolved_raw_action_dim,
        "action_chunk_size": int(resolved_chunk),
        "action_resolution": resolved_resolution,
        "frame_rate": float(resolved_frame_rate),
        "action_fps": resolved_action_fps,
        "num_frames": int(resolved_num_frames),
        "preset_key": preset_key,
        "warnings": warnings,
    }


COSMOS3_EXTRA_SPECS: dict[str, ExtraParamSchema] = {
    "use_duration_template": ExtraParamSchema(
        type="bool",
        default=True,
        description="Whether to use the duration template.",
    ),
    "use_resolution_template": ExtraParamSchema(
        type="bool",
        default=True,
        description="Whether to use the resolution template.",
    ),
    "use_system_prompt": ExtraParamSchema(
        type="bool",
        default=False,
        description="Whether to use the system prompt.",
    ),
    "use_guardrails": ExtraParamSchema(
        type="bool",
        default=True,
        description="Whether to use the guardrails.",
    ),
    "enable_audio": ExtraParamSchema(
        type="bool",
        default=False,
        description="Whether to enable audio generation.",
    ),
    "output_type": ExtraParamSchema(
        type="Literal['video', 'image']",
        default="video",
        description="Output modality: 'video' (T2V/I2V) or 'image' (text-to-image).",
    ),
    "action_mode": ExtraParamSchema(
        type="Literal['policy', 'forward_dynamics', 'inverse_dynamics']",
        default=None,
        description="Action generation mode: policy, forward_dynamics, or inverse_dynamics.",
    ),
    "domain_name": ExtraParamSchema(
        type="str",
        default=None,
        description=(
            "Embodiment domain name for action generation (e.g. bridge_orig_lerobot, av). "
            "When set, omitted raw_action_dim/action_chunk_size/action_resolution/frame_rate "
            "are filled from COSMOS3_DOMAIN_PRESETS; mismatches are logged as warnings."
        ),
    ),
    "domain_id": ExtraParamSchema(
        type="int",
        default=None,
        description="Embodiment domain id for action generation.",
    ),
    "raw_action_dim": ExtraParamSchema(
        type="int",
        default=None,
        description=(
            "Raw action DOF for policy/inverse_dynamics (e.g. 10 bridge, 9 av, 29 agibot). "
            "Inferred from domain_name preset when omitted."
        ),
    ),
    "action_chunk_size": ExtraParamSchema(
        type="int",
        default=None,
        description=(
            "Number of action tokens to generate (16 for most robots, 60 for av/camera_pose). "
            "Inferred from domain_name preset when omitted."
        ),
    ),
    "action": ExtraParamSchema(
        type="list",
        default=None,
        description="Action trajectory [T, D] for forward_dynamics mode.",
    ),
    "action_resolution": ExtraParamSchema(
        type="Literal[256, 480, 704, 720]",
        default=None,
        description=(
            "Resolution bucket for action image sizing. Must be one of "
            f"{list(COSMOS3_ACTION_RESOLUTIONS)}. Inferred from domain_name preset when omitted."
        ),
        range=(min(COSMOS3_ACTION_RESOLUTIONS), max(COSMOS3_ACTION_RESOLUTIONS)),
    ),
    "action_fps": ExtraParamSchema(
        type="float",
        default=None,
        description=(
            "Action-token temporal rate for mRoPE (Hz). Defaults to frame_rate when omitted."
        ),
    ),
    "video": ExtraParamSchema(
        type="path_or_list",
        default=None,
        description=(
            "Video for inverse_dynamics: .mp4/.avi file, frame directory, "
            "image path, or list of PIL images / frame paths."
        ),
    ),
}
