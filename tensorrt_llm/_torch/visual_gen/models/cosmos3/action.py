# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Action-token helpers for Cosmos3 UVA/action generation."""

from __future__ import annotations

import json
import math
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import PIL.Image
import torch
from diffusers.utils.torch_utils import randn_tensor

from tensorrt_llm.logger import logger

ACTION_MODE_POLICY = "policy"
ACTION_MODE_FORWARD_DYNAMICS = "forward_dynamics"
ACTION_MODE_INVERSE_DYNAMICS = "inverse_dynamics"
ACTION_MODES = {
    ACTION_MODE_POLICY,
    ACTION_MODE_FORWARD_DYNAMICS,
    ACTION_MODE_INVERSE_DYNAMICS,
}

EMBODIMENT_TO_DOMAIN_ID: dict[str, int] = {
    "no_action": 0,
    "av": 1,
    "camera_pose": 2,
    "hand_pose": 3,
    "pusht": 4,
    "libero": 5,
    "umi": 6,
    "bridge_orig_lerobot": 7,
    "droid_lerobot": 8,
    "robomind-franka": 8,
    "galbot": 9,
    "robomind-franka-dual": 12,
    "robomind-ur": 13,
    "agibotworld": 15,
    "agibot_gear_gripper": 15,
    "agibot_gear_gripper_ext": 15,
    "fractal": 20,
}

# Canonical unpadded action width per embodiment.  Widths compose the Cosmos3
# unified action representation from shared geometric blocks: a 9-D pose (3-D
# translation + 6-D rotation), a 1-D grasp state, and a 15-D fingertip state.
# One arm is 9 + 1 = 10; a dual-arm setup is 20; the AgiBot humanoid is
# 9 + 2 x (9 + 1) = 29; two-hand egocentric motion is 9 + 2 x (9 + 15) = 57.
#
# This is a property of the embodiment, not a tunable, so it is keyed by the
# real domain name rather than by the sampling presets in ``defaults.py`` (where
# several of these names share one preset).  ``libero`` is absent on purpose:
# its width depends on the dataset's rotation space (7/10/13), so callers must
# pass ``raw_action_dim`` explicitly.
EMBODIMENT_TO_RAW_ACTION_DIM: dict[str, int] = {
    "av": 9,
    "camera_pose": 9,
    "hand_pose": 57,
    "pusht": 2,
    "umi": 10,
    "bridge_orig_lerobot": 10,
    "droid_lerobot": 10,
    "robomind-franka": 10,
    "robomind-franka-dual": 20,
    "robomind-ur": 10,
    "galbot": 30,
    "agibotworld": 29,
    "agibot_gear_gripper": 29,
    "agibot_gear_gripper_ext": 29,
    "fractal": 10,
}


def resolve_raw_action_dim(
    domain_name: Any = None,
    domain_id: Any = None,
) -> int | None:
    """Look up the canonical action width, or None when it cannot be determined.

    Resolves by name first.  A bare ``domain_id`` is only usable when every
    embodiment sharing that id agrees on a width (true for all current ids).
    """
    if domain_name is not None and str(domain_name).strip():
        return EMBODIMENT_TO_RAW_ACTION_DIM.get(str(domain_name).strip().lower())

    if domain_id is None:
        return None

    widths = {
        EMBODIMENT_TO_RAW_ACTION_DIM[name]
        for name, mapped_id in EMBODIMENT_TO_DOMAIN_ID.items()
        if mapped_id == int(domain_id) and name in EMBODIMENT_TO_RAW_ACTION_DIM
    }
    return widths.pop() if len(widths) == 1 else None


# Camera perspective -> framing sentence. The action model was trained on these
# exact sentences, so they are reproduced verbatim rather than paraphrased.
ACTION_VIEWPOINT_TEMPLATES: dict[str, str] = {
    "ego_view": "This video is captured from a first-person perspective looking at the scene.",
    "third_person_view": (
        "This video is captured from a third-person perspective looking towards the agent "
        "from the front."
    ),
    "wrist_view": "This video is captured from a wrist-mounted camera.",
    "concat_view": "This video contains concatenated views from multiple camera perspectives.",
}

DEFAULT_ACTION_VIEW_POINT = "ego_view"

# Canonical ``W,H`` labels; every action canvas is one of these bucket shapes.
ACTION_ASPECT_RATIO_LABELS = ("1,1", "4,3", "3,4", "16,9", "9,16")


def action_aspect_ratio_label(height: int, width: int) -> str:
    """Closest canonical aspect label, e.g. 832x480 -> ``"16,9"``.

    Bucket sizes are only approximately their label (832/480 is 1.733, not
    1.778), so the label is matched by nearest ratio instead of reducing H/W.
    """
    ratio = width / height if height > 0 else 1.0
    return min(
        ACTION_ASPECT_RATIO_LABELS,
        key=lambda label: abs(int(label.split(",")[0]) / int(label.split(",")[1]) - ratio),
    )


def build_action_json_prompt(
    description: str,
    *,
    view_point: str | None,
    num_frames: int,
    frame_rate: float,
    height: int,
    width: int,
) -> str:
    """Build the structured action caption the action model was trained on.

    Replaces the flat duration/resolution templates used by the video paths: the
    JSON already carries duration, fps, resolution and aspect ratio. Key order is
    part of the trained format and is preserved.
    """
    duration_seconds = num_frames / frame_rate if frame_rate > 0 else 0.0
    if not math.isfinite(duration_seconds) or duration_seconds < 0:
        duration_seconds = 0.0
    minutes, seconds = divmod(round(duration_seconds), 60)

    text = description.strip()
    if text and not text.endswith((".", "!", "?")):
        text = f"{text}."

    framing = ACTION_VIEWPOINT_TEMPLATES.get(view_point) if view_point is not None else None
    if view_point is not None and framing is None:
        logger.warning(
            f"Unrecognized Cosmos3 action view_point={view_point!r}; expected one of "
            f"{sorted(ACTION_VIEWPOINT_TEMPLATES)}. Dropping the cinematography.framing field."
        )

    prompt: dict[str, Any] = {}
    if framing:
        prompt["cinematography"] = {"framing": framing}
    prompt["actions"] = [{"time": f"0:00-{minutes}:{seconds:02d}", "description": text}]
    prompt["duration"] = f"{int(duration_seconds)}s"
    prompt["fps"] = float(frame_rate)
    prompt["resolution"] = {"H": int(height), "W": int(width)}
    prompt["aspect_ratio"] = action_aspect_ratio_label(height, width)
    return json.dumps(prompt)


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


def normalize_action_mode(mode: Any) -> str | None:
    if mode is None:
        return None
    normalized = str(mode).strip().lower()
    if not normalized:
        return None
    if normalized not in ACTION_MODES:
        raise ValueError(
            f"Unsupported Cosmos3 action_mode={mode!r}; expected one of {sorted(ACTION_MODES)}."
        )
    return normalized


def resolve_domain_id(
    *,
    domain_id: Any = None,
    domain_name: Any = None,
    require_explicit: bool = False,
) -> int:
    if domain_id is not None:
        resolved = int(domain_id)
        if resolved < 0:
            raise ValueError(f"Cosmos3 domain_id must be non-negative, got {resolved}.")
        # domain_id wins so unlisted embodiments stay reachable, but a caller
        # that passes both and disagrees would otherwise silently get a
        # trajectory in a different robot's dialect.
        if domain_name is not None and str(domain_name).strip():
            key = str(domain_name).strip().lower()
            named_id = EMBODIMENT_TO_DOMAIN_ID.get(key)
            if named_id is not None and named_id != resolved:
                raise ValueError(
                    f"Cosmos3 domain_id={resolved} contradicts domain_name={domain_name!r}, "
                    f"which maps to domain_id={named_id}. Pass only one, or make them agree."
                )
        return resolved

    if domain_name is None or str(domain_name).strip() == "":
        if require_explicit:
            raise ValueError(
                "Cosmos3 action generation requires domain_id or non-empty domain_name."
            )
        return 0

    key = str(domain_name).strip().lower()
    if key not in EMBODIMENT_TO_DOMAIN_ID:
        raise ValueError(
            f"Unknown Cosmos3 action domain_name={domain_name!r}; "
            f"expected one of {sorted(EMBODIMENT_TO_DOMAIN_ID)} or pass domain_id directly."
        )
    return EMBODIMENT_TO_DOMAIN_ID[key]


def action_condition_indexes(mode: str, action_length: int) -> list[int]:
    mode = normalize_action_mode(mode)
    if mode == ACTION_MODE_FORWARD_DYNAMICS:
        return list(range(action_length))
    if mode in {ACTION_MODE_POLICY, ACTION_MODE_INVERSE_DYNAMICS}:
        return []
    raise AssertionError(f"Unexpected action mode: {mode!r}")


def vision_condition_indexes(
    mode: str, video_length: int, temporal_compression_factor: int
) -> list[int]:
    mode = normalize_action_mode(mode)
    latent_frames = (video_length - 1) // temporal_compression_factor + 1
    if mode in {ACTION_MODE_POLICY, ACTION_MODE_FORWARD_DYNAMICS}:
        return [0]
    if mode == ACTION_MODE_INVERSE_DYNAMICS:
        return list(range(latent_frames))
    raise AssertionError(f"Unexpected action mode: {mode!r}")


def action_start_frame_offset(mode: str, action_length: int, video_length: int) -> int:
    del mode
    if action_length == video_length - 1:
        return 1
    if action_length == video_length:
        return 0
    raise ValueError(
        "Cosmos3 action_chunk_size must equal num_frames - 1 or num_frames; "
        f"got action_chunk_size={action_length}, num_frames={video_length}."
    )


def build_action_condition_mask(
    mode: str,
    action_length: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask = torch.zeros(1, action_length, 1, device=device, dtype=dtype)
    for idx in action_condition_indexes(mode, action_length):
        mask[:, idx, :] = 1.0
    return mask


def build_vision_condition_mask(
    mode: str,
    video_length: int,
    temporal_compression_factor: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    latent_frames = (video_length - 1) // temporal_compression_factor + 1
    mask = torch.zeros(1, 1, latent_frames, 1, 1, device=device, dtype=dtype)
    for idx in vision_condition_indexes(mode, video_length, temporal_compression_factor):
        mask[:, :, idx, :, :] = 1.0
    return mask


def pad_action_to_dim(action: torch.Tensor, action_dim: int) -> torch.Tensor:
    if action.shape[-1] > action_dim:
        raise ValueError(
            f"Cosmos3 action dimension {action.shape[-1]} exceeds model action_dim={action_dim}."
        )
    if action.shape[-1] == action_dim:
        return action
    padding = torch.zeros(
        *action.shape[:-1], action_dim - action.shape[-1], dtype=action.dtype, device=action.device
    )
    return torch.cat([action, padding], dim=-1)


def load_action_tensor(action: Any = None) -> torch.Tensor:
    if action is None:
        raise ValueError(
            "Cosmos3 forward_dynamics action mode requires an action tensor of shape [T, D]."
        )
    if isinstance(action, torch.Tensor):
        tensor = action.detach().to(dtype=torch.float32)
    else:
        tensor = torch.as_tensor(np.asarray(action), dtype=torch.float32)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"Cosmos3 action must have shape [T, D], got {tuple(tensor.shape)}.")
    if tensor.shape[0] == 0:
        raise ValueError(
            f"Cosmos3 action trajectory must have at least one timestep, got shape "
            f"{tuple(tensor.shape)}."
        )
    return tensor


def find_closest_target_size(h: int, w: int, resolution: str | int) -> tuple[int, int]:
    key = str(resolution)
    if key not in VIDEO_RES_SIZE_INFO:
        raise ValueError(
            f"Unknown Cosmos3 action resolution={resolution!r}; "
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


ACTION_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})
ACTION_VIDEO_EXTENSIONS = frozenset({".mp4", ".avi"})


def pil_to_rgb(value: Any) -> PIL.Image.Image:
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            image = PIL.Image.open(BytesIO(bytes(value)))
            image.load()
            return image.convert("RGB")
        except OSError as exc:
            raise ValueError(
                f"Cosmos3 action image reference could not be decoded; it may be "
                f"truncated, corrupt, or in an unsupported format: {exc}"
            ) from exc
    if isinstance(value, (str, Path)):
        # load_image, not PIL.Image.open: the bundled action prompts carry
        # https:// frame references, and it also handles file:// and data: URIs.
        from tensorrt_llm.inputs.utils import load_image

        return load_image(str(value), format="pil").convert("RGB")
    if isinstance(value, PIL.Image.Image):
        return value.convert("RGB")
    raise TypeError(
        f"Cosmos3 action preprocessing expected image bytes, PIL image, or image path, "
        f"got {type(value)!r}."
    )


def resolve_action_size(
    height: int | None,
    width: int | None,
    source_h: int,
    source_w: int,
    action_resolution: int,
) -> tuple[int, int]:
    """Fill unset action H/W from the action resolution bucket; honor explicit values.

    The bucket is the canvas whose shape is closest to the source's, so the
    reference only ever needs a modest pad to reach it.
    """
    if height is not None and width is not None:
        return height, width
    target_w, target_h = find_closest_target_size(source_h, source_w, action_resolution)
    return (
        height if height is not None else target_h,
        width if width is not None else target_w,
    )


def action_reference_size(
    *,
    action_mode: str,
    image: Any,
    video: Any,
) -> tuple[int, int]:
    """Source ``(height, width)`` of the reference, for choosing the canvas.

    Video references are encoded bytes, so their size comes from the container
    header rather than a decode; images are measured directly.
    """
    source_is_video = action_mode == ACTION_MODE_INVERSE_DYNAMICS or image is None
    source = video if source_is_video else image
    if source is None:
        raise ValueError(f"Cosmos3 action_mode={action_mode!r} requires an image or video input.")
    if source_is_video and isinstance(source, bytes):
        from tensorrt_llm.media.decoding import video_stream_info

        info = video_stream_info(source)
        if info is None:
            raise ValueError(
                f"Cosmos3 action_mode={action_mode!r} video reference could not be demuxed "
                "(corrupt or not a supported container)."
            )
        return info.height, info.width
    reference = pil_to_rgb(source)
    return reference.height, reference.width


def action_reference_frame_step(source_frame_rate: float | None, target_frame_rate: float) -> int:
    """Source frames to advance per reference frame retained.

    An embodiment's frame rate is a property of what it was trained on, not of
    the clip a caller happens to send: bridge learned one command per frame at
    5 Hz, so 200ms of gripper motion between frames. A 30 fps clip read
    consecutively shows the model a sixth of that motion while the caption and
    the mRoPE positions still claim 5 Hz, so the reference is thinned to match
    -- every sixth frame here.

    A clip slower than the embodiment returns 1: selection can drop frames,
    never invent them, and closing that gap needs interpolation rather than a
    step. The caller is expected to say so rather than let it pass silently.
    """
    if not source_frame_rate or not target_frame_rate:
        return 1
    if source_frame_rate <= 0 or target_frame_rate <= 0:
        return 1
    return max(1, round(source_frame_rate / target_frame_rate))


def resize_and_pad_action_image(
    image: PIL.Image.Image, target_h: int, target_w: int
) -> PIL.Image.Image:
    scale = min(target_w / image.width, target_h / image.height, 1.0)
    resize_w = max(1, int(scale * image.width + 0.5))
    resize_h = max(1, int(scale * image.height + 0.5))
    if (resize_w, resize_h) != image.size:
        image = image.resize((resize_w, resize_h), PIL.Image.Resampling.BICUBIC)

    array = np.asarray(image)
    pad_h = target_h - resize_h
    pad_w = target_w - resize_w
    if pad_h < 0 or pad_w < 0:
        raise ValueError(
            f"Cosmos3 action image resize exceeded target size: resized={(resize_h, resize_w)}, "
            f"target={(target_h, target_w)}."
        )
    if pad_h == 0 and pad_w == 0:
        return image
    pad_mode = "reflect" if pad_h < resize_h and pad_w < resize_w else "edge"
    padded = np.pad(array, ((0, pad_h), (0, pad_w), (0, 0)), mode=pad_mode)
    return PIL.Image.fromarray(padded)


def prepare_action_latents(
    *,
    mode: str,
    action_chunk_size: int,
    raw_action_dim: int | None,
    action_dim: int,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    action_input: Any = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Prepare action latents and conditioning masks for a denoise request.

    Args:
        mode: Cosmos3 action mode. ``forward_dynamics`` conditions all action
            tokens on ``action_input``; ``policy`` and ``inverse_dynamics`` start
            from action noise and predict the raw action dimensions.
        action_chunk_size: Number of action tokens in the generated trajectory.
        raw_action_dim: Number of semantic action dimensions before padding.
            Required for ``policy`` and ``inverse_dynamics``. For
            ``forward_dynamics``, omitted values are inferred from
            ``action_input.shape[-1]`` and explicit values must match it.
        action_dim: Model action embedding width after zero-padding.
        generator: Random generator used for action noise.
        device: Target device for the returned tensors.
        dtype: Target dtype for the returned tensors.
        action_input: Forward-dynamics action trajectory with shape ``[T, D]``.

    Returns:
        Tuple of ``(action_latents, action_velocity_mask, clean_action,
        raw_action_dim)``. The tensors have shape ``[1, action_chunk_size,
        action_dim]`` except the mask, which has shape ``[1, action_chunk_size,
        1]``.
    """
    if mode == ACTION_MODE_FORWARD_DYNAMICS:
        action = load_action_tensor(action_input)
        if action.shape[0] < action_chunk_size:
            pad = action[-1:].repeat(action_chunk_size - action.shape[0], 1)
            action = torch.cat([action, pad], dim=0)
        elif action.shape[0] > action_chunk_size:
            action = action[:action_chunk_size]
        if raw_action_dim is None:
            raw_action_dim = int(action.shape[-1])
        elif int(raw_action_dim) != int(action.shape[-1]):
            raise ValueError(
                "Cosmos3 forward_dynamics raw_action_dim must match action input width; "
                f"got raw_action_dim={raw_action_dim}, action width={action.shape[-1]}."
            )
        clean_action = pad_action_to_dim(action, action_dim)
    else:
        if raw_action_dim is None:
            raise ValueError(
                "Cosmos3 action_mode='policy' and 'inverse_dynamics' require raw_action_dim."
            )
        clean_action = torch.zeros(action_chunk_size, action_dim, dtype=torch.float32)

    raw_action_dim = int(raw_action_dim)
    if raw_action_dim <= 0 or raw_action_dim > action_dim:
        raise ValueError(
            f"Cosmos3 raw_action_dim must be in [1, {action_dim}], got {raw_action_dim}."
        )

    clean_action = clean_action.to(device=device, dtype=dtype).unsqueeze(0)
    condition_mask = build_action_condition_mask(
        mode,
        action_chunk_size,
        device=device,
        dtype=dtype,
    )
    noise = randn_tensor(
        (1, action_chunk_size, action_dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    noise[:, :, raw_action_dim:] = 0
    clean_action[:, :, raw_action_dim:] = 0
    action_latents = condition_mask * clean_action + (1.0 - condition_mask) * noise
    action_velocity_mask = 1.0 - condition_mask
    return action_latents, action_velocity_mask, clean_action, raw_action_dim
