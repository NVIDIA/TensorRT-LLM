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
from typing import Any, Dict, Iterable

from tensorrt_llm._torch.visual_gen.pipeline import ExtraParamSchema
from tensorrt_llm.inputs.media_io import sniff_media_kind

# ---------------------------------------------------------------------------
# Constant tables
# ---------------------------------------------------------------------------

# Cosmos3 output resolution buckets keyed by target level, then aspect ratio;
# each value is (width, height). A source frame maps onto the bucket whose
# aspect ratio is closest (see ``find_closest_target_size`` in ``transfer.py``).
VIDEO_RES_SIZE_INFO = {
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


def _validate_video_reference(video) -> None:
    """Preflight for the ``video`` extra param: encoded MP4/AVI bytes."""
    if not video:
        raise ValueError("Cosmos3 video reference bytes are empty.")
    if sniff_media_kind(video) != "video":
        raise ValueError(
            "Cosmos3 video reference bytes are not a recognized video "
            "container (supported: MP4/AVI)."
        )


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

# Families without an entry get no envelope advisory.
COSMOS3_ENVELOPES: Dict = {
    "nemotron_dense": COSMOS3_EDGE_ENVELOPE,
}

COSMOS3_V2V_DEFAULT_FLOW_SHIFT = 10.0

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
