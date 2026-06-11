# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Action-token helpers for Cosmos3 UVA/action generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import PIL.Image
import torch
from diffusers.utils.torch_utils import randn_tensor

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
    if isinstance(value, str):
        return PIL.Image.open(value).convert("RGB")
    if isinstance(value, PIL.Image.Image):
        return value.convert("RGB")
    raise TypeError(
        f"Cosmos3 action preprocessing expected PIL image or image path, got {type(value)!r}."
    )


def decode_action_video_file(path: Path, max_frames: Optional[int] = None) -> List[PIL.Image.Image]:
    import torchvision.io as io

    frames, _, _ = io.read_video(str(path), pts_unit="sec")
    if frames.numel() == 0:
        raise ValueError(f"Cosmos3 action video file contains no frames: {path}")
    if max_frames is not None:
        frames = frames[:max_frames]
    return [PIL.Image.fromarray(frames[i].numpy()) for i in range(frames.shape[0])]


def normalize_action_video_path(path: Path, max_frames: Optional[int] = None) -> List[Any]:
    if not path.exists():
        raise ValueError(f"Cosmos3 action video path does not exist: {path}")
    if path.is_dir():
        frames = sorted(p for p in path.iterdir() if p.suffix.lower() in ACTION_IMAGE_EXTENSIONS)
        if not frames:
            raise ValueError(f"No image frames found in Cosmos3 action video directory: {path}")
        frame_paths = [str(p) for p in frames]
        if max_frames is not None:
            frame_paths = frame_paths[:max_frames]
        return frame_paths

    suffix = path.suffix.lower()
    if suffix in ACTION_IMAGE_EXTENSIONS:
        return [str(path)]
    if suffix in ACTION_VIDEO_EXTENSIONS:
        return decode_action_video_file(path, max_frames=max_frames)
    raise ValueError(
        "Cosmos3 action video path must be a frame directory, an image file "
        f"{sorted(ACTION_IMAGE_EXTENSIONS)}, or a video file "
        f"{sorted(ACTION_VIDEO_EXTENSIONS)}; got {path}"
    )


def normalize_action_video_input(video: Any, max_frames: Optional[int] = None) -> List[Any]:
    """Normalize action video input to a frame list.

    Accepts a list of PIL images / paths, a single image or video file path,
    or a directory of frame images (sorted lexicographically).
    """
    if video is None:
        return []
    if isinstance(video, list):
        if not video:
            raise ValueError("Cosmos3 action video input must contain at least one frame.")
        if max_frames is not None:
            return video[:max_frames]
        return video
    if isinstance(video, (str, Path)):
        return normalize_action_video_path(Path(video), max_frames=max_frames)
    return [video]


def resolve_action_size(
    height: Optional[int],
    width: Optional[int],
    ref_image: PIL.Image.Image,
    action_resolution: int,
) -> tuple[int, int]:
    """Fill unset action H/W from the action resolution bucket; honor explicit values."""
    if height is not None and width is not None:
        return height, width
    target_w, target_h = find_closest_target_size(
        ref_image.height, ref_image.width, action_resolution
    )
    return (
        height if height is not None else target_h,
        width if width is not None else target_w,
    )


def action_reference_image(
    *,
    action_mode: str,
    image: Any,
    video: Any,
) -> PIL.Image.Image:
    """Resolve the reference frame used for action sizing and conditioning."""
    if action_mode == ACTION_MODE_INVERSE_DYNAMICS:
        source = video if video is not None else image
        frames = normalize_action_video_input(source, max_frames=1)
        if not frames:
            raise ValueError("Cosmos3 action_mode='inverse_dynamics' requires a video input.")
        return pil_to_rgb(frames[0])

    source = image if image is not None else video
    if source is None:
        raise ValueError(f"Cosmos3 action_mode={action_mode!r} requires an image or video input.")
    if isinstance(source, PIL.Image.Image):
        return source.convert("RGB")
    if isinstance(source, str):
        path = Path(source)
        if path.is_file() and path.suffix.lower() in ACTION_IMAGE_EXTENSIONS:
            return PIL.Image.open(source).convert("RGB")
        frames = normalize_action_video_input(source, max_frames=1)
        if not frames:
            raise ValueError(
                f"Cosmos3 action_mode={action_mode!r} requires an image or video input."
            )
        return pil_to_rgb(frames[0])
    raise TypeError(
        f"Cosmos3 action reference image must be PIL.Image or path, got {type(source)!r}."
    )


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
    raw_action_dim: Optional[int],
    action_dim: int,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    action_input: Any = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if mode == ACTION_MODE_FORWARD_DYNAMICS:
        action = load_action_tensor(action_input)
        if action.shape[0] < action_chunk_size:
            pad = action[-1:].repeat(action_chunk_size - action.shape[0], 1)
            action = torch.cat([action, pad], dim=0)
        elif action.shape[0] > action_chunk_size:
            action = action[:action_chunk_size]
        if raw_action_dim is None:
            raw_action_dim = int(action.shape[-1])
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
