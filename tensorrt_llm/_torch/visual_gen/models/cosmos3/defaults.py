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
"""

from typing import Any, Dict, Iterable, TypedDict

from tensorrt_llm._torch.visual_gen.models.cosmos3.action import (
    COSMOS3_ACTION_RESOLUTIONS,
    DEFAULT_ACTION_VIEW_POINT,
    EMBODIMENT_TO_DOMAIN_ID,
    normalize_action_resolution,
    resolve_raw_action_dim,
)
from tensorrt_llm._torch.visual_gen.pipeline import ExtraParamSchema
from tensorrt_llm.inputs.media_io import sniff_media_kind

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

COSMOS3_DEFAULT_CONDITION_VIDEO_LATENT_INDEXES = (0, 1)
COSMOS3_DEFAULT_CONDITION_VIDEO_KEEP = "first"


# ---------------------------------------------------------------------------
# Conditioning-value normalizers / validators. Declared as the ``validator``
# of the matching extra-param specs below, so invalid values 400 at preflight;
# the pipeline reuses them at run time to normalize the same inputs.
# ---------------------------------------------------------------------------


def _normalize_condition_video_latent_indexes(
    indexes: Iterable[int] | None,
) -> tuple[int, ...]:
    if indexes is None:
        return COSMOS3_DEFAULT_CONDITION_VIDEO_LATENT_INDEXES
    values = []
    for index in indexes:
        # Strict: reject non-integers instead of silently truncating (1.9 -> 1)
        # or TypeError-ing on None. Integral floats (JSON emitters) coerce.
        if isinstance(index, bool) or not isinstance(index, (int, float)):
            raise ValueError(
                f"Cosmos3 condition_video_latent_indexes must be integers, got {index!r}."
            )
        if isinstance(index, float):
            if not index.is_integer():
                raise ValueError(
                    f"Cosmos3 condition_video_latent_indexes must be integers, got {index!r}."
                )
            index = int(index)
        values.append(index)
    normalized = tuple(values)

    if not normalized:
        raise ValueError("Cosmos3 condition_video_latent_indexes must not be empty.")
    if any(index < 0 for index in normalized):
        raise ValueError(
            f"Cosmos3 condition_video_latent_indexes must be non-negative, got {normalized}."
        )
    return normalized


def _normalize_condition_video_keep(keep: str | None) -> str:
    normalized = str(keep or COSMOS3_DEFAULT_CONDITION_VIDEO_KEEP).strip().lower()
    if normalized not in {"first", "last"}:
        raise ValueError("Cosmos3 condition_video_keep must be either first or last.")
    return normalized


def _validate_output_type(output_type: str) -> None:
    if output_type not in ("video", "image"):
        raise ValueError(f"Cosmos3 output_type must be 'video' or 'image', got {output_type!r}.")


def _validate_video_reference(video) -> None:
    """Preflight for the ``video`` extra param: encoded MP4/AVI bytes."""
    if not video:
        raise ValueError("Cosmos3 video reference bytes are empty.")
    if sniff_media_kind(video) != "video":
        raise ValueError(
            "Cosmos3 video reference bytes are not a recognized video "
            "container (supported: MP4/AVI)."
        )


# Text-to-image (``output_type="image"``) defaults; resolved in ``infer()``.
COSMOS3_T2I_PARAMS = {
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50,
    "guidance_scale": 7.0,
    "flow_shift": 3.0,
    "guidance_interval": (400.0, 1000.0),
}

# Fields merged by the executor into every request. Mode-dependent values
# remain None until infer() selects the request mode; key membership also
# declares these fields supported during request validation.
COSMOS3_PIPELINE_DEFAULTS = {
    **COSMOS3_720P_PARAMS,
    "height": None,
    "width": None,
    "num_inference_steps": None,
    "guidance_scale": None,
}

COSMOS3_ACTION_PARAMS = {
    "action_chunk_size": 16,
    "num_inference_steps": 30,
    "guidance_scale": 1.0,
    "frame_rate": 24.0,
}


class Cosmos3DomainPreset(TypedDict, total=False):
    """Recommended action sampling settings for a trained embodiment.

    Sampling settings only — the embodiment's action width lives in
    ``action.EMBODIMENT_TO_RAW_ACTION_DIM``, keyed by the unaliased domain name.
    """

    action_chunk_size: int
    action_resolution: int
    frame_rate: float


# Training-aligned defaults, mirroring the Cosmos3 omni ``action_*.json`` inputs
# (bridge, av, droid, libero, ...) where those exist.
COSMOS3_DOMAIN_PRESETS: dict[str, Cosmos3DomainPreset] = {
    # WidowX bridge.
    "bridge_orig_lerobot": {
        "action_chunk_size": 16,
        "action_resolution": 480,
        "frame_rate": 5.0,
    },
    # Autonomous-vehicle steering/throttle; longer action horizon.
    "av": {
        "action_chunk_size": 60,
        "action_resolution": 480,
        "frame_rate": 10.0,
    },
    # 6-DoF camera pose + shutter; matches AV-style horizon.
    "camera_pose": {
        "action_chunk_size": 60,
        "action_resolution": 480,
        "frame_rate": 30.0,
    },
    # Franka single-arm tabletop; same domain_id as robomind-franka.
    "droid_lerobot": {
        "action_chunk_size": 16,
        "action_resolution": 480,
        "frame_rate": 15.0,
    },
    # LIBERO sim single-arm; lower action resolution bucket.
    "libero": {
        "action_chunk_size": 16,
        "action_resolution": 256,
        "frame_rate": 10.0,
    },
    # MANO hand pose.
    "hand_pose": {
        "action_chunk_size": 16,
        "action_resolution": 480,
        "frame_rate": 24.0,
    },
    # AgiBot humanoid; shared domain_id with agibot_gear_gripper*.
    "agibotworld": {
        "action_chunk_size": 16,
        "action_resolution": 480,
        "frame_rate": 10.0,
    },
    # Google Robot (RT-1 / fractal) single-arm.
    "fractal": {
        "action_chunk_size": 16,
        "action_resolution": 480,
        "frame_rate": 5.0,
    },
    # 2-D planar push task.
    "pusht": {
        "action_chunk_size": 16,
        "action_resolution": 256,
        "frame_rate": 10.0,
    },
    # UMI handheld gripper setup.
    "umi": {
        "action_chunk_size": 16,
        "action_resolution": 480,
        "frame_rate": 10.0,
    },
}

# Map alias domain_name keys to a canonical preset entry. These share *sampling*
# settings only; each alias keeps its own action width (e.g. robomind-franka-dual
# is 20-D and galbot is 30-D, unlike the presets they borrow here).
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
) -> dict[str, Any]:
    """Merge user action params with domain presets and generic fallbacks.

    A recognized ``domain_name`` (or a uniquely mapped ``domain_id``) fills
    whichever of ``action_chunk_size``, ``action_resolution`` and ``frame_rate``
    the caller left unset; an explicit value wins but is reported in
    ``warnings`` when it differs from the preset. ``num_frames`` is derived as
    ``action_chunk_size + 1`` and is never a preset field.
    """
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

    # The action width is canonical per embodiment, so it comes from the
    # embodiment table rather than the (alias-shared) sampling preset.
    canonical_raw_action_dim = resolve_raw_action_dim(domain_name=domain_name, domain_id=domain_id)
    if raw_action_dim is not None:
        if canonical_raw_action_dim is not None and int(raw_action_dim) != canonical_raw_action_dim:
            warnings.append(
                f"Cosmos3 raw_action_dim={raw_action_dim} differs from the canonical width "
                f"{canonical_raw_action_dim} for domain_name={domain_name!r}."
            )
        resolved_raw_action_dim = raw_action_dim
    else:
        resolved_raw_action_dim = canonical_raw_action_dim
        if domain_requested and canonical_raw_action_dim is None:
            warnings.append(
                "Cosmos3 has no canonical action width for "
                f"domain_name={domain_name!r}, domain_id={domain_id!r}; "
                "pass raw_action_dim explicitly for policy/inverse_dynamics."
            )
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
    # Always derived: an action clip is the chunk plus its initial frame. Both
    # references fix this, and diffusers rejects a caller-supplied num_frames
    # for action runs outright, so a preset must not pin it independently of
    # an overridden action_chunk_size.
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


COSMOS3_EXTRA_SPECS: Dict[str, ExtraParamSchema] = {
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
        default=None,
        description=(
            "Whether to prepend the system prompt. Unset means the model "
            "decides: V2V uses it, other modes take the checkpoint's "
            "declared default."
        ),
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
        type="str",
        default="video",
        description="Output modality: 'video' (T2V/I2V) or 'image' (text-to-image).",
        validator=_validate_output_type,
    ),
    "condition_video_latent_indexes": ExtraParamSchema(
        type="list",
        default=list(COSMOS3_DEFAULT_CONDITION_VIDEO_LATENT_INDEXES),
        description=(
            "Latent frame indexes OF THE OUTPUT video to pin to the encoded "
            "reference (not source-frame selection). Each latent frame spans 4 "
            "pixel frames, so the worker consumes the first (or last, per "
            "condition_video_keep) max(indexes)*4+1 reference frames."
        ),
        validator=_normalize_condition_video_latent_indexes,
    ),
    "condition_video_keep": ExtraParamSchema(
        type="str",
        default=COSMOS3_DEFAULT_CONDITION_VIDEO_KEEP,
        description="Which side of the input video to use for conditioning: first or last.",
        validator=_normalize_condition_video_keep,
    ),
    "flow_shift": ExtraParamSchema(
        type="float",
        default=None,
        description="Optional scheduler flow shift override. Uses the Cosmos3 mode default when omitted.",
    ),
    "video": ExtraParamSchema(
        type="bytes",
        default=None,
        description=(
            "V2V reference: encoded MP4/AVI bytes (e.g. "
            "Path(video).read_bytes()). Each worker rank demuxes them from "
            "memory and NVDEC-decodes only the conditioning window per "
            "condition_video_latent_indexes / condition_video_keep, resized "
            "to the output resolution, then VAE-encodes it."
        ),
        validator=_validate_video_reference,
    ),
    "action_mode": ExtraParamSchema(
        type="Literal['policy', 'forward_dynamics', 'inverse_dynamics']",
        default=None,
        description=(
            "Action generation mode: policy, forward_dynamics, or inverse_dynamics. "
            "The predicted trajectory is not representable in a video container, so "
            "an action request is served as a tensor payload."
        ),
        requires_tensor_output=True,
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
            "Resolved from the embodiment when omitted; required for domains with no "
            "canonical width (libero)."
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
        # No range: the buckets are not an interval, and validation stops at the
        # literal check anyway.
    ),
    "view_point": ExtraParamSchema(
        type="Literal['ego_view', 'third_person_view', 'wrist_view', 'concat_view']",
        default=DEFAULT_ACTION_VIEW_POINT,
        description=(
            "Camera perspective for action generation. Fills the trained action caption's "
            "cinematography.framing field."
        ),
    ),
    "action_fps": ExtraParamSchema(
        type="float",
        default=None,
        description=(
            "Action-token temporal rate for mRoPE (Hz). Defaults to frame_rate when omitted."
        ),
    ),
}
