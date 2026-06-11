# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Action-token helpers for Cosmos3 UVA/action generation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

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
