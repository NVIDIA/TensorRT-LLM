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

from typing import Dict, Iterable

import torch

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
    normalized = tuple(int(index) for index in indexes)

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


def _crop_video_frames(video, extra_params) -> torch.Tensor:
    """Crop the V2V reference to the conditioning window before transport.

    Runs once in the coordinator (spec ``reducer``) before the request is
    deep-copied, pickled over ZMQ, and broadcast per rank: a full 189-frame
    720p reference is ~520 MiB while the default conditioning window is 5
    frames (~14 MiB). Semantics-preserving — the worker's own first/last crop
    is idempotent, so reduced and unreduced tensors generate identically.
    Anything invalid is returned unchanged for the validators to reject.
    """
    if not isinstance(video, torch.Tensor) or video.ndim != 4:
        return video
    try:
        indexes = _normalize_condition_video_latent_indexes(
            extra_params.get("condition_video_latent_indexes")
        )
        keep = _normalize_condition_video_keep(extra_params.get("condition_video_keep"))
    except (TypeError, ValueError):
        return video
    # 4 = Cosmos3 VAE temporal compression; if a future VAE changes it, the
    # worker pads/crops the window itself, so a mismatch degrades gracefully.
    window = max(indexes) * 4 + 1
    if video.shape[0] <= window:
        return video
    sliced = video[-window:] if keep == "last" else video[:window]
    # A slice is a view over the full storage and would pickle all of it;
    # clone so the transport payload owns only the window.
    return sliced.clone()


def _validate_video_reference_tensor(video: torch.Tensor) -> None:
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(
            f"Cosmos3 video reference must be a uint8 [T, H, W, C] RGB tensor, "
            f"got shape {tuple(video.shape)}."
        )
    if video.dtype != torch.uint8:
        raise ValueError(f"Cosmos3 video reference must have dtype uint8, got {video.dtype}.")
    if video.device.type != "cpu":
        raise ValueError(
            f"Cosmos3 video reference must be a CPU tensor, got device '{video.device}' "
            "(it is pickled to the workers; keep decoded references on the host)."
        )


# Fields merged by the executor for every request. Modality-specific values
# (height/width/num_frames/steps/guidance) are declared with ``None`` so the
# executor accepts explicit overrides; ``forward()`` resolves unset fields from
# T2V/T2I context.
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
        type="tensor",
        default=None,
        description=(
            "V2V reference: decoded video frames as a uint8 [T, H, W, C] RGB "
            "torch.Tensor (build one from a file with "
            "tensorrt_llm.inputs.media_io.load_video_frames_tensor). The "
            "coordinator crops it to the conditioning window per "
            "condition_video_latent_indexes / condition_video_keep before "
            "dispatch; the worker VAE-encodes it. Media is always decoded "
            "by the producer."
        ),
        validator=_validate_video_reference_tensor,
        reducer=_crop_video_frames,
    ),
}
