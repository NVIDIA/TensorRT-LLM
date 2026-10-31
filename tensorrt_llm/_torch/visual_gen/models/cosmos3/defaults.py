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

from collections.abc import Mapping
from typing import Any, Dict, Iterable, TypedDict

from tensorrt_llm._torch.visual_gen.pipeline import ExtraParamSchema
from tensorrt_llm.inputs.media_io import sniff_media_kind

# ---------------------------------------------------------------------------
# Constant tables
# ---------------------------------------------------------------------------

# The reference's ``W,H`` aspect buckets: every canvas is one of these shapes, and a
# request's bucket string is carried into the caption verbatim, so a resolved frame
# size maps back to the nearest bucket rather than to its exact reduced ratio
# (832x480 reduces to "15,26", which the model never saw).
COSMOS3_ASPECT_RATIO_BUCKETS = ("1,1", "4,3", "3,4", "16,9", "9,16")


# Output sizes per resolution level and aspect bucket, as (width, height).
VIDEO_RES_SIZE_INFO: dict[str, dict[str, tuple[int, int]]] = {
    "256": {
        "1,1": (256, 256),
        "4,3": (320, 256),
        "3,4": (256, 320),
        "16,9": (320, 192),
        "9,16": (192, 320),
    },
    "480": {
        "1,1": (640, 640),
        "4,3": (736, 544),
        "3,4": (544, 736),
        "16,9": (832, 480),
        "9,16": (480, 832),
    },
    "704": {
        "1,1": (960, 960),
        "4,3": (1088, 832),
        "3,4": (832, 1088),
        "16,9": (1280, 704),
        "9,16": (704, 1280),
    },
    "720": {
        "1,1": (960, 960),
        "4,3": (1104, 832),
        "3,4": (832, 1104),
        "16,9": (1280, 720),
        "9,16": (720, 1280),
    },
}


COSMOS3_ACTION_RESOLUTIONS = tuple(int(key) for key in sorted(VIDEO_RES_SIZE_INFO, key=int))


def normalize_action_resolution(resolution: Any) -> int:
    if resolution is None:
        raise ValueError("Cosmos3 action_resolution is required for action generation.")
    try:
        bucket = int(resolution)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Cosmos3 action_resolution must be an int bucket, got {resolution!r}."
        ) from exc
    if bucket not in COSMOS3_ACTION_RESOLUTIONS:
        raise ValueError(
            f"Unknown Cosmos3 action_resolution={bucket}; "
            f"expected one of {COSMOS3_ACTION_RESOLUTIONS}."
        )
    return bucket


def find_closest_target_size(h: int, w: int, resolution: str | int) -> tuple[int, int]:
    """The ``resolution`` bucket whose aspect ratio is closest to ``h/w``, as
    ``(width, height)``, so a source frame lands on a supported size undistorted."""
    key = str(resolution)
    if key not in VIDEO_RES_SIZE_INFO:
        raise ValueError(
            f"Unknown Cosmos3 resolution={resolution!r}; "
            f"expected one of {sorted(VIDEO_RES_SIZE_INFO)}."
        )
    input_ratio = h / w
    best_size = None
    best_diff = float("inf")
    for cand_w, cand_h in VIDEO_RES_SIZE_INFO[key].values():
        diff = abs(input_ratio - cand_h / cand_w)
        if diff < best_diff:
            best_diff = diff
            best_size = (cand_w, cand_h)
    assert best_size is not None
    return best_size


# The default video resolution is the 720p 16:9 bucket, ``(width, height)``.
_DEFAULT_VIDEO_W, _DEFAULT_VIDEO_H = VIDEO_RES_SIZE_INFO["720"]["16,9"]
COSMOS3_720P_PARAMS = {
    "height": _DEFAULT_VIDEO_H,
    "width": _DEFAULT_VIDEO_W,
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


# ---------------------------------------------------------------------------
# Transfer preflight validators.
#
# These mirror ``transfer.resolve_transfer_config``'s parsing rather than adding
# rules of their own: a validator that is stricter than the worker would reject
# requests the pipeline would have served. The worker keeps its own checks --
# offline callers do not go through preflight -- so these exist purely to turn a
# deterministic client mistake into a 400 at enqueue instead of a failure deep
# in the pipeline, after the request has already been accepted with a 202.
# ---------------------------------------------------------------------------
def _transfer_hint_payload(value: Any) -> Mapping:
    """Normalize a control hint to its object form, as the worker does.

    Normalization first, checks after, so a hint carried as bare bytes and the
    same hint carried as ``{"control": <bytes>}`` are held to one standard.
    """
    if value is True:
        payload: Mapping = {}
    elif isinstance(value, bytes):
        payload = {"control": value}
    elif isinstance(value, bool):  # False, having already excluded True
        raise ValueError(
            "control hint must be true, encoded MP4/AVI bytes, or an object; got false. "
            "Omit the key entirely to leave the hint off."
        )
    elif not isinstance(value, Mapping):
        raise TypeError(
            "control hint must be an object, encoded control bytes, or true; "
            f"got {type(value).__name__}."
        )
    else:
        if value.get("control_path") is not None:
            raise ValueError(
                "control hint no longer accepts 'control_path'; pass the encoded control clip "
                "as 'control' bytes (Path(control).read_bytes())."
            )
        payload = value

    control = payload.get("control")
    if control is not None:
        if not isinstance(control, bytes):
            raise TypeError(
                "control hint 'control' must be encoded MP4/AVI bytes, got "
                f"{type(control).__name__}."
            )
        # Same bar the `video` reference is held to: undecodable bytes fail at
        # decode anyway, so name the problem now rather than mid-request.
        if sniff_media_kind(control) != "video":
            raise ValueError(
                "control hint bytes are not a recognized video container (supported: MP4/AVI)."
            )
    return payload


def _validate_edge_hint(value: Any) -> None:
    # Imported at call time: transfer imports this module, so a module-level
    # import would be circular. Specs pickle by name, so this stays picklable.
    from .transfer import EDGE_PRESETS

    payload = _transfer_hint_payload(value)
    # `or "medium"` matches the worker, which treats empty/None as the default.
    preset = str(payload.get("preset_edge_threshold") or "medium").lower()
    if preset not in EDGE_PRESETS:
        raise ValueError(
            f"unsupported preset_edge_threshold {preset!r}; expected one of {sorted(EDGE_PRESETS)}."
        )


def _validate_blur_hint(value: Any) -> None:
    from .transfer import BLUR_PRESETS

    payload = _transfer_hint_payload(value)
    preset = str(payload.get("preset_blur_strength") or "medium").lower()
    if preset not in BLUR_PRESETS:
        raise ValueError(
            f"unsupported preset_blur_strength {preset!r}; expected one of {sorted(BLUR_PRESETS)}."
        )


def _validate_precomputed_control_hint(value: Any) -> None:
    """For hints with no on-the-fly generator: a control clip is mandatory."""
    if _transfer_hint_payload(value).get("control") is None:
        raise ValueError(
            "this control has no on-the-fly generator, so it requires a precomputed clip "
            "as encoded MP4/AVI bytes; only 'edge' and 'blur' accept true."
        )


def _validate_control_guidance_interval(value: Any) -> None:
    from .transfer import _as_interval

    _as_interval(value)  # reused outright, so preflight cannot drift from the worker


def _validate_positive_frames(value: Any) -> None:
    if int(value) <= 0:
        raise ValueError(f"must be a positive frame count, got {value}.")


def _validate_non_negative_frames(value: Any) -> None:
    if int(value) < 0:
        raise ValueError(f"must be a non-negative frame count, got {value}.")


# Text-to-image (``output_type="image"``) defaults; resolved in ``infer()``.
COSMOS3_T2I_PARAMS = {
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50,
    "guidance_scale": 7.0,
    "flow_shift": 3.0,
    "guidance_interval": (400.0, 1000.0),
}

# Edge (Nemotron-dense backbone) is 480p-native. Video values follow the
# model-card I2V command (T2V mirrors it — the model card documents I2V only);
# ``flow_shift`` rides the checkpoint-declared native flow schedule. T2I values
# are the cosmos-framework t2i mode defaults at Edge's native resolution
# (480p at 1:1 aspect), with full-range CFG.
COSMOS3_EDGE_VIDEO_PARAMS = {
    "height": 480,
    "width": 832,
    "num_inference_steps": 50,
    "guidance_scale": 5.0,
    "max_sequence_length": 4096,
    "num_frames": 121,
    "frame_rate": 24.0,
    "flow_shift": 3.0,
}

COSMOS3_EDGE_T2I_PARAMS = {
    "height": 640,
    "width": 640,
    "num_inference_steps": 50,
    "guidance_scale": 4.0,
    "flow_shift": 3.0,
    "guidance_interval": None,
}

# Model-card validated envelope for Edge; advisory only (the reference
# runtime accepts a wider range), surfaced as a log line per request.
COSMOS3_EDGE_ENVELOPE = {
    "num_frames": (50, 150),
    "frame_rate": (12.0, 30.0),
    "max_sequence_length": 4096,
    "resolutions": frozenset(
        {
            (640, 640),
            (544, 736),
            (736, 544),
            (480, 832),
            (832, 480),
            (256, 256),
            (256, 320),
            (320, 256),
            (192, 320),
            (320, 192),
        }
    ),
}

# (family, mode) → generation defaults. Family is the architecture recipe
# name resolved from the transformer config; mode is the request's output
# type — never inferred from the checkpoint name (a task-specialized
# checkpoint can still be asked to run any mode).
COSMOS3_GENERATION_DEFAULTS: Dict = {
    ("qwen3", "video"): COSMOS3_720P_PARAMS,
    ("qwen3", "image"): COSMOS3_T2I_PARAMS,
    ("nemotron_dense", "video"): COSMOS3_EDGE_VIDEO_PARAMS,
    ("nemotron_dense", "image"): COSMOS3_EDGE_T2I_PARAMS,
}

# Action's table carries the sampling recipe only; the canvas, clip length and
# frame rate resolve from the embodiment preset in the pipeline, and anything
# missing here falls back to the family's video table inside
# _resolve_generation_params. Same recipe for both families.


# Families without an entry get no envelope advisory.
COSMOS3_ENVELOPES: Dict = {
    "nemotron_dense": COSMOS3_EDGE_ENVELOPE,
}

COSMOS3_ACTION_PARAMS = {
    "action_chunk_size": 16,
    "num_inference_steps": 30,
    "guidance_scale": 1.0,
    "frame_rate": 24.0,
}

# Released policy checkpoints carry a ``policy`` block in checkpoint.json.
# The block owns checkpoint-specific horizon/rate/domain values; the remaining
# recipe facts are keyed by that domain below.
COSMOS3_POLICY_SAMPLING_PARAMS = {
    "num_inference_steps": 4,
    "guidance_scale": 3.0,
    # Apply CFG only to the first, highest-noise UniPC step.
    "guidance_interval": (960.0, 1001.0),
}


class Cosmos3PolicyDomainPreset(TypedDict, total=False):
    """Policy-only representation facts not carried by checkpoint.json."""

    num_inference_steps: int
    guidance_scale: float
    guidance_interval: tuple[float, float]
    raw_action_dim: int
    action_resolution: int
    use_state: bool


COSMOS3_POLICY_DOMAIN_PRESETS: dict[str, Cosmos3PolicyDomainPreset] = {
    # DROID Policy is joint_pos: seven joints plus one gripper channel. The
    # current 8-D state is a clean leading action row and is not returned as a
    # predicted command.
    "droid_lerobot": {
        **COSMOS3_POLICY_SAMPLING_PARAMS,
        "raw_action_dim": 8,
        "action_resolution": 480,
        "use_state": True,
    },
}

COSMOS3_GENERATION_DEFAULTS[("qwen3", "action")] = COSMOS3_ACTION_PARAMS
COSMOS3_GENERATION_DEFAULTS[("nemotron_dense", "action")] = COSMOS3_ACTION_PARAMS


def resolve_checkpoint_policy_defaults(policy: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize a checkpoint.json ``policy`` block into pipeline defaults."""
    if policy is None:
        return {}
    if not isinstance(policy, Mapping):
        raise ValueError("Cosmos3 checkpoint.json 'policy' must be a JSON object.")

    defaults: dict[str, Any] = {}
    domain_name = policy.get("domain_name")
    if domain_name is not None:
        domain_name = str(domain_name).strip().lower()
        if not domain_name:
            raise ValueError("Cosmos3 checkpoint policy domain_name must not be empty.")
        defaults["domain_name"] = domain_name
        defaults.update(COSMOS3_POLICY_DOMAIN_PRESETS.get(domain_name, {}))

    if policy.get("action_chunk_size") is not None:
        action_chunk_size = int(policy["action_chunk_size"])
        if action_chunk_size <= 0:
            raise ValueError(
                "Cosmos3 checkpoint policy action_chunk_size must be positive, "
                f"got {action_chunk_size}."
            )
        defaults["action_chunk_size"] = action_chunk_size

    if policy.get("conditioning_fps") is not None:
        conditioning_fps = float(policy["conditioning_fps"])
        if conditioning_fps <= 0.0:
            raise ValueError(
                "Cosmos3 checkpoint policy conditioning_fps must be positive, "
                f"got {conditioning_fps}."
            )
        defaults["frame_rate"] = conditioning_fps
        defaults["action_fps"] = conditioning_fps

    if policy.get("raw_action_dim") is not None:
        raw_action_dim = int(policy["raw_action_dim"])
        if raw_action_dim <= 0:
            raise ValueError(
                f"Cosmos3 checkpoint policy raw_action_dim must be positive, got {raw_action_dim}."
            )
        defaults["raw_action_dim"] = raw_action_dim

    if policy.get("action_resolution") is not None:
        defaults["action_resolution"] = normalize_action_resolution(policy["action_resolution"])

    if policy.get("use_state") is not None:
        if not isinstance(policy["use_state"], bool):
            raise ValueError("Cosmos3 checkpoint policy use_state must be a boolean.")
        defaults["use_state"] = policy["use_state"]

    return defaults


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

    from .action import EMBODIMENT_TO_DOMAIN_ID

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
    use_state: bool | None = None,
    checkpoint_policy_defaults: Mapping[str, Any] | None = None,
    embodiments: Mapping[str, Any] | None = None,
    input_action_dim: int | None = None,
) -> dict[str, Any]:
    """The action fields a request runs with, by the request rules
    (``request_resolver.ACTION_RULES``): the caller's value, else the checkpoint policy
    block when its domain is the request's, else the checkpoint's embodiment table
    (``embodiments``) or the code's width table, else the domain preset, else the
    generic action defaults. ``num_frames`` is always ``action_chunk_size + 1``. A
    caller's value that differs from the recommendation is reported in ``warnings``.
    """
    from .request_resolver import RequestContext, embodiment_entry, policy_domain, resolve_action

    policy_defaults = dict(checkpoint_policy_defaults or {})
    domain_name, policy_active, warnings = policy_domain(policy_defaults, domain_name, domain_id)
    ctx = RequestContext(
        family="",
        action_mode="action",
        policy_defaults=policy_defaults,
        policy_active=policy_active,
        domain_name=domain_name,
        domain_id=None if domain_id is None else int(domain_id),
        embodiment=embodiment_entry(embodiments or {}, input_action_dim, domain_name, domain_id),
        checkpoint_has_embodiments=bool(embodiments),
    )
    resolved = resolve_action(
        ctx,
        raw_action_dim=raw_action_dim,
        action_chunk_size=action_chunk_size,
        action_resolution=action_resolution,
        frame_rate=frame_rate,
        action_fps=action_fps,
        use_state=use_state,
    )
    width = resolved["raw_action_dim"]
    return {
        "domain_name": domain_name,
        "checkpoint_policy_domain_matches": policy_active,
        "raw_action_dim": None if width is None else int(width),
        "action_chunk_size": int(resolved["action_chunk_size"]),
        "action_resolution": resolved["action_resolution"],
        "frame_rate": float(resolved["frame_rate"]),
        "action_fps": float(resolved["action_fps"]),
        "use_state": bool(resolved["use_state"]),
        "num_frames": int(resolved["num_frames"]),
        "preset_key": ctx.preset_key,
        "warnings": warnings + resolved.warnings,
        "sources": {name: r.source for name, r in resolved.values.items()},
    }


# Video-to-video and transfer sample on the shifted flow grid with this shift; the
# other modes carry ``flow_shift`` in their tables above or take the checkpoint's.
COSMOS3_V2V_FLOW_SHIFT = 10.0

# Prompt format the checkpoints were trained with. The "give" typo is the training
# text; keep it. Metadata sentences are appended to the caption the same way.
COSMOS3_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant who will generate videos from a give prompt."
)
COSMOS3_T2I_SYSTEM_PROMPT = (
    "You are a helpful assistant who will generate images from a give prompt."
)
COSMOS3_DURATION_TEMPLATE = "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
COSMOS3_DEFAULT_RESOLUTION_TEMPLATE = "This video is of {height}x{width} resolution."
COSMOS3_IMAGE_RESOLUTION_TEMPLATE = "This image is of {height}x{width} resolution."

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
        description=(
            "Action trajectory [T, D] for forward_dynamics, or the current model-space "
            "state [D]/[1, D] for policy mode when use_state is true."
        ),
    ),
    "use_state": ExtraParamSchema(
        type="bool",
        default=None,
        description=(
            "Condition policy generation on the current state supplied in action. "
            "Defaults from checkpoint policy metadata when available."
        ),
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
        default=None,
        description=(
            "Deprecated compatibility parameter; accepted and ignored. Supply any "
            "trained structured action caption directly in the request prompt."
        ),
    ),
    "action_fps": ExtraParamSchema(
        type="float",
        default=None,
        description=(
            "Action-token temporal rate for mRoPE (Hz). Defaults to frame_rate when omitted."
        ),
    ),
    # Transfer
    "edge": ExtraParamSchema(
        type="bool_or_bytes_or_dict",
        default=None,
        description=(
            "Canny-edge control. true auto-computes it from the `video` extra param; "
            "or pass the encoded MP4/AVI bytes of a precomputed control clip, or an "
            'object {"control": <bytes>, "preset_edge_threshold": "medium"}.'
        ),
        validator=_validate_edge_hint,
    ),
    "blur": ExtraParamSchema(
        type="bool_or_bytes_or_dict",
        default=None,
        description=(
            "Low-frequency (color/lighting) control. true auto-computes it from the "
            "`video` extra param; or pass encoded control bytes, or an object "
            '{"control": <bytes>, "preset_blur_strength": "medium"}.'
        ),
        validator=_validate_blur_hint,
    ),
    "depth": ExtraParamSchema(
        type="bool_or_bytes_or_dict",
        default=None,
        description=(
            "Depth control. Requires precomputed encoded MP4/AVI bytes (no auto-computation)."
        ),
        validator=_validate_precomputed_control_hint,
    ),
    "seg": ExtraParamSchema(
        type="bool_or_bytes_or_dict",
        default=None,
        description="Semantic-segmentation control. Requires precomputed encoded MP4/AVI bytes.",
        validator=_validate_precomputed_control_hint,
    ),
    "wsm": ExtraParamSchema(
        type="bool_or_bytes_or_dict",
        default=None,
        description=(
            "World-scenario-model control. Requires precomputed encoded MP4/AVI bytes; "
            "runs 101-frame chunks at 10 fps."
        ),
        validator=_validate_precomputed_control_hint,
    ),
    "control_guidance": ExtraParamSchema(
        type="float",
        default=None,
        description="Transfer control-guidance scale (CFG for the control branch).",
    ),
    "control_guidance_interval": ExtraParamSchema(
        type="list",
        default=None,
        description=(
            "[lo, hi] window where control guidance is active, in raw scheduler "
            "timesteps (typically 0-1000, counting down) rather than as a fraction "
            "of the schedule: [0.0, 0.8] gates on the final step, not the first "
            "80%. To cover most of the schedule use e.g. [200, 1000]."
        ),
        validator=_validate_control_guidance_interval,
    ),
    "num_video_frames_per_chunk": ExtraParamSchema(
        type="int",
        default=None,
        description="Transfer chunk length in frames (default 93; 101 for wsm).",
        validator=_validate_positive_frames,
    ),
    "num_conditional_frames": ExtraParamSchema(
        type="int",
        default=None,
        description="Overlap frames pinned from the previous chunk when stitching.",
        validator=_validate_non_negative_frames,
    ),
    "num_first_chunk_conditional_frames": ExtraParamSchema(
        type="int",
        default=None,
        description="Input-video frames pinned at the start of the first chunk.",
        validator=_validate_non_negative_frames,
    ),
    "max_frames": ExtraParamSchema(
        type="int",
        default=None,
        description="Cap on frames decoded from transfer inputs/controls.",
        validator=_validate_positive_frames,
    ),
    "show_control_condition": ExtraParamSchema(
        type="bool", default=False, description="Concatenate the control video beside the output."
    ),
    "show_input": ExtraParamSchema(
        type="bool", default=False, description="Concatenate the input video beside the output."
    ),
    "share_vision_temporal_positions": ExtraParamSchema(
        type="bool",
        default=None,
        description="Controls share the target frames' temporal mRoPE positions.",
    ),
    "emphasize_control_in_prompt": ExtraParamSchema(
        type="bool",
        default=None,
        description=(
            "Append a one-sentence control-adherence directive naming the active "
            "hints to the user prompt (default true). Set false for clean "
            "baselines / ablations. The system prompt is unchanged."
        ),
    ),
}
