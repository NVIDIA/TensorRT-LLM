# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos3 transfer inference helpers."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

from .defaults import VIDEO_RES_SIZE_INFO
from .utils import read_video_tensor


def find_closest_target_size(h: int, w: int, resolution: str | int) -> tuple[int, int]:
    """Pick the ``resolution`` bucket whose aspect ratio best matches ``h/w``.

    Returns ``(target_w, target_h)`` so a source frame maps onto a supported
    output size without distorting its aspect ratio.
    """
    key = str(resolution)
    if key not in VIDEO_RES_SIZE_INFO:
        raise ValueError(
            f"Unknown Cosmos3 transfer resolution={resolution!r}; "
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


# ---------------------------------------------------------------------------
# Transfer sampling + per-control-hint defaults, ported from vllm-omni's
# Cosmos3 transfer implementation; the guidance and edge/blur presets are
# empirically tuned per control modality:
# https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transfer.py
# ---------------------------------------------------------------------------

# Supported control hints, in application order.
TRANSFER_HINT_KEYS: tuple[str, ...] = ("edge", "blur", "depth", "seg", "wsm")

# Chunked-sampling defaults for long-video transfer.
TRANSFER_SAMPLE_DEFAULTS: dict[str, Any] = {
    "num_video_frames_per_chunk": 93,  # frames per autoregressive chunk (4k+1 form)
    "num_conditional_frames": 1,  # overlap frames reused from the previous chunk
    "max_frames": 5000,  # hard cap on total input frames
    "show_control_condition": False,  # debug: also emit the control frames
    "show_input": False,  # debug: also emit the input frames
    "num_first_chunk_conditional_frames": 0,  # chunk 0 has no prior chunk to condition on
    "share_vision_temporal_positions": True,  # control tokens reuse the video patches' positions
}

# Per-hint guidance tuning: text guidance_scale, control_guidance, and flow_shift,
# chosen empirically per control modality.
TRANSFER_DEFAULTS: dict[str, dict[str, Any]] = {
    "edge": {"guidance_scale": 3.0, "control_guidance": 1.5, "flow_shift": 10.0},
    "blur": {"guidance_scale": 3.0, "control_guidance": 1.5, "flow_shift": 10.0},
    "depth": {"guidance_scale": 3.0, "control_guidance": 1.5, "flow_shift": 10.0},
    "seg": {
        "guidance_scale": 3.0,
        "control_guidance": 2.0,
        "flow_shift": 10.0,
    },  # leans harder on control
    # Precomputed control; control-only guidance (text off), 10 fps / 101-frame chunks.
    "wsm": {
        "guidance_scale": 1.0,
        "control_guidance": 3.0,
        "flow_shift": 10.0,
        "num_frames": 101,
        "fps": 10,
        "num_video_frames_per_chunk": 101,
    },
}

# Canny (lower, upper) thresholds per edge-strength preset (fed to cv2.Canny);
# higher preset = higher thresholds = sparser edges.
EDGE_PRESETS: dict[str, tuple[int, int]] = {
    "none": (20, 50),
    "very_low": (20, 50),
    "low": (50, 100),
    "medium": (100, 200),
    "high": (200, 300),
    "very_high": (300, 400),
}

# Bilateral-blur strength presets: ``pre_blur_downscale`` (downscale applied
# before the bilateral filter) and ``downup`` (down/up-sample factor after);
# larger = blurrier.
BLUR_PRESETS: dict[str, dict[str, int]] = {
    "none": {"pre_blur_downscale": 1, "downup": 1},
    "very_low": {"pre_blur_downscale": 1, "downup": 4},
    "low": {"pre_blur_downscale": 4, "downup": 4},
    "medium": {"pre_blur_downscale": 2, "downup": 10},
    "high": {"pre_blur_downscale": 1, "downup": 16},
    "very_high": {"pre_blur_downscale": 4, "downup": 16},
}

# cv2.bilateralFilter parameters for the blur control, tuned at a 720p reference.
BILATERAL_REFERENCE_RESOLUTION = 720  # resolution the params below are tuned for
BILATERAL_D = 30  # filter diameter in pixels
BILATERAL_SIGMA_COLOR = 150  # color-space sigma
BILATERAL_SIGMA_SPACE = 100  # coordinate-space sigma
BILATERAL_ITERATIONS = 1  # number of bilateral passes


@dataclass
class Cosmos3TransferHint:
    key: str
    control_path: str | None = None
    control: Any | None = None
    preset_edge_threshold: str = "medium"
    preset_blur_strength: str = "medium"


@dataclass
class Cosmos3TransferConfig:
    hints: dict[str, Cosmos3TransferHint] = field(default_factory=dict)
    guidance_scale: float | None = None
    control_guidance: float = 1.0
    control_guidance_interval: tuple[float, float] | None = None
    flow_shift: float | None = None
    num_video_frames_per_chunk: int = 93
    num_conditional_frames: int = 1
    max_frames: int = 5000
    show_control_condition: bool = False
    show_input: bool = False
    num_first_chunk_conditional_frames: int = 0
    share_vision_temporal_positions: bool = True
    num_frames: int | None = None
    fps: float | None = None
    resolution: Any | None = None

    @property
    def ordered_hints(self) -> list[Cosmos3TransferHint]:
        return [self.hints[key] for key in TRANSFER_HINT_KEYS if key in self.hints]


def _as_interval(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            "Cosmos3 transfer control_guidance_interval must contain exactly two values."
        )
    lo, hi = float(value[0]), float(value[1])
    if lo > hi:
        raise ValueError(
            f"Cosmos3 transfer control_guidance_interval must be ordered as [lo, hi], got {(lo, hi)}."
        )
    return lo, hi


def _extra_or_default(extra_params: dict, key: str, default: Any = None) -> Any:
    value = extra_params.get(key, None)
    return default if value is None else value


def resolve_transfer_config(
    extra_params: dict, req_params: Any, prompt_data: Any = None
) -> Cosmos3TransferConfig | None:
    hints: dict[str, Cosmos3TransferHint] = {}
    for key in TRANSFER_HINT_KEYS:
        raw = extra_params.get(key, None)
        if raw is None:
            continue
        if raw is True:
            raw = {}
        elif isinstance(raw, str | Path):
            raw = {"control_path": str(raw)}
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"Cosmos3 transfer hint '{key}' must be an object, path string, or true; got {type(raw)!r}."
            )
        hints[key] = Cosmos3TransferHint(
            key=key,
            control_path=str(raw["control_path"]) if raw.get("control_path") is not None else None,
            control=raw.get("control"),
            preset_edge_threshold=str(raw.get("preset_edge_threshold") or "medium").lower(),
            preset_blur_strength=str(raw.get("preset_blur_strength") or "medium").lower(),
        )

    if not hints:
        transfer_only = (
            "control_guidance",
            "control_guidance_interval",
            "num_video_frames_per_chunk",
            "num_conditional_frames",
            "num_first_chunk_conditional_frames",
            "max_frames",
            "show_control_condition",
            "show_input",
            "share_vision_temporal_positions",
        )
        if any(extra_params.get(key, None) for key in transfer_only):
            raise ValueError(
                "Cosmos3 transfer options were provided, but no transfer hint was selected."
            )
        return None

    request_guidance_scale = getattr(req_params, "guidance_scale", None)
    if request_guidance_scale is None:
        # vLLM-compatible transfer guidance default: OmniDiffusionRequest
        # materializes omitted guidance_scale to 1.0 before transfer defaults
        # are resolved, so single-hint presets must not silently promote it.
        request_guidance_scale = 1.0

    config = Cosmos3TransferConfig(
        hints=hints,
        guidance_scale=request_guidance_scale,
        control_guidance=_extra_or_default(extra_params, "control_guidance", 1.0),
        control_guidance_interval=_as_interval(
            _extra_or_default(extra_params, "control_guidance_interval", None)
        ),
        flow_shift=_extra_or_default(extra_params, "flow_shift", None),
        num_video_frames_per_chunk=_extra_or_default(
            extra_params,
            "num_video_frames_per_chunk",
            TRANSFER_SAMPLE_DEFAULTS["num_video_frames_per_chunk"],
        ),
        num_conditional_frames=_extra_or_default(
            extra_params,
            "num_conditional_frames",
            TRANSFER_SAMPLE_DEFAULTS["num_conditional_frames"],
        ),
        max_frames=_extra_or_default(
            extra_params, "max_frames", TRANSFER_SAMPLE_DEFAULTS["max_frames"]
        ),
        show_control_condition=_extra_or_default(
            extra_params,
            "show_control_condition",
            TRANSFER_SAMPLE_DEFAULTS["show_control_condition"],
        ),
        show_input=_extra_or_default(
            extra_params, "show_input", TRANSFER_SAMPLE_DEFAULTS["show_input"]
        ),
        num_first_chunk_conditional_frames=_extra_or_default(
            extra_params,
            "num_first_chunk_conditional_frames",
            TRANSFER_SAMPLE_DEFAULTS["num_first_chunk_conditional_frames"],
        ),
        share_vision_temporal_positions=_extra_or_default(
            extra_params,
            "share_vision_temporal_positions",
            TRANSFER_SAMPLE_DEFAULTS["share_vision_temporal_positions"],
        ),
        num_frames=_extra_or_default(
            extra_params, "num_frames", getattr(req_params, "num_frames", None)
        ),
        fps=_extra_or_default(
            extra_params,
            "fps",
            _extra_or_default(
                extra_params,
                "frame_rate",
                _extra_or_default(
                    extra_params, "resolved_frame_rate", getattr(req_params, "frame_rate", None)
                ),
            ),
        ),
        resolution=_extra_or_default(
            extra_params, "resolution", _extra_or_default(extra_params, "image_size", 720)
        ),
    )

    if len(hints) == 1:
        hint_key = next(iter(hints))
        for field_name, default_value in TRANSFER_DEFAULTS[hint_key].items():
            if field_name == "guidance_scale":
                user_set = config.guidance_scale is not None
            elif field_name == "flow_shift":
                user_set = extra_params.get("flow_shift", None) is not None
            elif field_name == "fps":
                user_set = (
                    extra_params.get("fps", None) is not None
                    or extra_params.get("frame_rate", None) is not None
                    or extra_params.get("resolved_frame_rate", None) is not None
                    or getattr(req_params, "frame_rate", None) is not None
                )
            elif field_name == "num_frames":
                user_set = (
                    extra_params.get("num_frames", None) is not None
                    or getattr(req_params, "num_frames", None) is not None
                )
            else:
                user_set = extra_params.get(field_name, None) is not None
            if not user_set:
                setattr(config, field_name, default_value)

    if config.num_video_frames_per_chunk <= 0:
        raise ValueError("Cosmos3 transfer num_video_frames_per_chunk must be positive.")
    if config.num_conditional_frames < 0:
        raise ValueError("Cosmos3 transfer num_conditional_frames must be non-negative.")
    if config.max_frames <= 0:
        raise ValueError("Cosmos3 transfer max_frames must be positive.")
    if config.num_first_chunk_conditional_frames < 0:
        raise ValueError(
            "Cosmos3 transfer num_first_chunk_conditional_frames must be non-negative."
        )
    for hint in hints.values():
        if hint.key == "edge" and hint.preset_edge_threshold not in EDGE_PRESETS:
            raise ValueError(f"Unsupported Cosmos3 edge preset: {hint.preset_edge_threshold!r}.")
        if hint.key == "blur" and hint.preset_blur_strength not in BLUR_PRESETS:
            raise ValueError(f"Unsupported Cosmos3 blur preset: {hint.preset_blur_strength!r}.")
    return config


def _height_width_from_shape(shape) -> tuple[int, int] | None:
    """Infer ``(H, W)`` from a frame tensor/array shape by locating the channel axis.

    Treats a leading or trailing size-3/4 axis as the channel axis and reads the
    two spatial axes around it: handles ``[B?]{CTHW, THWC, CHW, HWC}``.
    """
    dims = list(shape)
    if len(dims) == 5:  # drop the leading batch axis
        dims = dims[1:]
    if len(dims) == 4:
        if dims[0] in (3, 4) and dims[-1] not in (3, 4):  # C T H W
            return int(dims[-2]), int(dims[-1])
        if dims[-1] in (3, 4):  # T H W C
            return int(dims[1]), int(dims[2])
    if len(dims) == 3:
        if dims[0] in (3, 4) and dims[-1] not in (3, 4):  # C H W
            return int(dims[-2]), int(dims[-1])
        if dims[-1] in (3, 4):  # H W C
            return int(dims[0]), int(dims[1])
    return None


def _video_file_height_width(media_path: Path) -> tuple[int, int] | None:
    """Read ``(H, W)`` from a video container header via OpenCV — no frame decode.

    Returns None if OpenCV is unavailable or the file will not open; the actual
    decode elsewhere raises the clear ``pip install`` hint.
    """
    try:
        from tensorrt_llm.inputs.media_io import _get_cv2

        cv2 = _get_cv2()
    except ImportError:
        return None
    capture = cv2.VideoCapture(str(media_path))
    try:
        if not capture.isOpened():
            return None
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    return (height, width) if width > 0 and height > 0 else None


def media_height_width(value: Any) -> tuple[int, int] | None:
    """Return the ``(height, width)`` of a transfer control input, or None if unknown.

    Accepts any form a control can take — a PIL image, a file path (image or
    video, read from the header without a full decode), a tensor/ndarray
    (inferred from shape), or a list (its first element). Callers feed the result
    to ``find_closest_target_size`` to pick the output resolution whose aspect
    ratio best matches the control.
    """
    if isinstance(value, PIL.Image.Image):
        return value.height, value.width
    if isinstance(value, str | Path):
        from tensorrt_llm.inputs.media_io import is_image_file

        media_path = Path(value)
        if is_image_file(media_path):
            with PIL.Image.open(media_path) as image:
                return image.height, image.width
        return _video_file_height_width(media_path)
    if isinstance(value, torch.Tensor):
        return _height_width_from_shape(value.shape)
    if isinstance(value, np.ndarray):
        return _height_width_from_shape(value.shape)
    if isinstance(value, list | tuple) and value:
        return media_height_width(value[0])
    return None


def resize_center_crop_uint8_cthw(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Scale-to-cover then center-crop control frames to ``(height, width)``.

    ``frames``: uint8 ``[3, T, H, W]`` -> uint8 ``[3, T, H, W]``. Aspect ratio is
    preserved (scaled so the target fits, excess cropped) so the control signal
    stays aligned with the generated video.
    """
    if frames.ndim != 4 or frames.shape[0] != 3:
        raise ValueError(
            f"Cosmos3 transfer frames must have shape [3, T, H, W], got {tuple(frames.shape)}."
        )
    orig_h, orig_w = int(frames.shape[2]), int(frames.shape[3])
    scale = max(width / orig_w, height / orig_h)
    resize_h = int(np.ceil(scale * orig_h))
    resize_w = int(np.ceil(scale * orig_w))
    frames_tchw = frames.permute(1, 0, 2, 3).to(dtype=torch.float32)
    resized = F.interpolate(
        frames_tchw, size=(resize_h, resize_w), mode="bilinear", align_corners=False
    )
    top = (resize_h - height) // 2
    left = (resize_w - width) // 2
    cropped = resized[:, :, top : top + height, left : left + width]
    return cropped.round().clamp(0, 255).to(torch.uint8).permute(1, 0, 2, 3).contiguous()


def _pil_to_uint8_rgb(value: Any) -> np.ndarray:
    """Coerce one frame (PIL image, path, tensor, or ndarray) to a uint8 RGB array.

    Returns ``[H, W, 3]`` uint8. Float inputs in ``[-1, 1]`` or ``[0, 1]`` are
    rescaled to ``[0, 255]``. uint8 RGB is the form the cv2 control generators
    (edge/blur) operate on.
    """
    if isinstance(value, PIL.Image.Image):
        return np.asarray(value.convert("RGB"), dtype=np.uint8)
    if isinstance(value, str | Path):
        return np.asarray(PIL.Image.open(value).convert("RGB"), dtype=np.uint8)
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] in (3, 4):
            tensor = tensor[:3].permute(1, 2, 0)
        if tensor.is_floating_point():
            if tensor.numel() and (tensor.min().item() < 0.0 or tensor.max().item() > 1.0):
                tensor = tensor.clamp(-1.0, 1.0).mul(0.5).add(0.5)
            tensor = tensor.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
        return tensor[..., :3].numpy().astype(np.uint8)
    if isinstance(value, np.ndarray):
        array = value
        if array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
            array = np.transpose(array[:3], (1, 2, 0))
        if np.issubdtype(array.dtype, np.floating):
            if array.size and (array.min() < 0.0 or array.max() > 1.0):
                array = np.clip(array, -1.0, 1.0) * 0.5 + 0.5
            array = (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        return array[..., :3].astype(np.uint8)
    raise TypeError(f"Cosmos3 transfer expected an RGB frame, got {type(value)!r}.")


def _path_media_to_uint8_cthw(path: str | Path, max_frames: int | None) -> torch.Tensor:
    """Decode a control-media file (image or video) to uint8 ``[3, T, H, W]`` frames.

    Classified by content (image vs video) via the shared probe, not the suffix;
    a single image becomes a one-frame clip.
    """
    from tensorrt_llm.inputs.media_io import is_image_file

    media_path = Path(path)
    if not media_path.exists():
        raise FileNotFoundError(f"Missing Cosmos3 transfer control_path: {media_path}")
    # Classify by content, not suffix — the same content probe as the V2V path.
    if is_image_file(media_path):
        array = _pil_to_uint8_rgb(media_path)
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(1).contiguous()

    # Same OpenCV decode stack as V2V, consumed as a tensor:
    # [T, H, W, C] -> [C, T, H, W] with the layout `contiguous` as the only copy.
    frames_thwc = read_video_tensor(media_path, max_frames=max_frames)
    return frames_thwc[..., :3].permute(3, 0, 1, 2).contiguous()


def media_to_uint8_cthw(
    value: Any, *, height: int, width: int, max_frames: int | None = None
) -> torch.Tensor:
    """Load any control/input media into uint8 ``[3, T, H, W]`` frames fit to ``(height, width)``.

    Accepts a file path, a video tensor/ndarray, or a list of frames, and returns
    the canonical uint8 form the cv2 control generators (edge/blur) consume — the
    entry point for turning a transfer control (or input) reference into frames.
    """
    if isinstance(value, str | Path):
        frames = _path_media_to_uint8_cthw(value, max_frames=max_frames)
    elif isinstance(value, torch.Tensor):
        frames = normalized_video_to_uint8_cthw(value)
    elif isinstance(value, np.ndarray):
        frames = normalized_video_to_uint8_cthw(torch.from_numpy(value))
    elif isinstance(value, list | tuple):
        if not value:
            raise ValueError("Cosmos3 transfer control frames cannot be empty.")
        selected = value[:max_frames] if max_frames is not None else value
        tensors = [
            torch.from_numpy(_pil_to_uint8_rgb(frame)).permute(2, 0, 1) for frame in selected
        ]
        frames = torch.stack(tensors, dim=1).contiguous()
    else:
        raise TypeError(f"Unsupported Cosmos3 transfer control payload type: {type(value)!r}.")
    if max_frames is not None:
        frames = frames[:, : int(max_frames)]
    return resize_center_crop_uint8_cthw(frames, int(height), int(width))


def normalized_video_to_uint8_cthw(video: torch.Tensor) -> torch.Tensor:
    """Convert an in-memory video tensor to uint8 ``[3, T, H, W]`` control frames.

    Accepts ``[1, 3, T, H, W]`` / ``[3, T, H, W]`` (or channels-last); float values
    in ``[-1, 1]`` / ``[0, 1]`` are rescaled to ``[0, 255]``.
    """
    tensor = video.detach().cpu()
    if tensor.ndim == 5:
        if tensor.shape[0] != 1:
            raise ValueError("Cosmos3 transfer supports only batch size 1 video inputs.")
        tensor = tensor[0]
    if tensor.ndim != 4:
        raise ValueError(
            f"Cosmos3 transfer video must have shape [1, 3, T, H, W] or [3, T, H, W], got {tuple(video.shape)}."
        )
    if tensor.shape[0] != 3:
        if tensor.shape[-1] in (3, 4):
            tensor = tensor[..., :3].permute(3, 0, 1, 2)
        else:
            raise ValueError(
                f"Cosmos3 transfer video must have 3 RGB channels, got {tuple(video.shape)}."
            )
    if tensor.is_floating_point():
        if tensor.numel() and (tensor.min().item() < 0.0 or tensor.max().item() > 1.0):
            tensor = tensor.clamp(-1.0, 1.0).mul(0.5).add(0.5)
        tensor = tensor.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
    else:
        tensor = tensor.clamp(0, 255).to(torch.uint8)
    return tensor.contiguous()


def uint8_cthw_to_normalized_5d(frames: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    """Normalize uint8 control frames into the transfer model's input tensor.

    ``frames``: uint8 ``[3, T, H, W]`` -> ``[1, 3, T, H, W]`` in ``[-1, 1]``
    (``/127.5 - 1``), the batched normalized form the control encoder consumes.
    """
    if frames.ndim != 4 or frames.shape[0] != 3:
        raise ValueError(
            f"Cosmos3 transfer frames must have shape [3, T, H, W], got {tuple(frames.shape)}."
        )
    return frames.to(dtype=dtype).div(127.5).sub(1.0).unsqueeze(0).contiguous()


def make_edge_control(frames: torch.Tensor, preset: str) -> torch.Tensor:
    cv2 = _import_cv2("edge")
    try:
        lower, upper = EDGE_PRESETS[preset]
    except KeyError as exc:
        raise ValueError(f"Unsupported Cosmos3 edge preset: {preset!r}.") from exc
    frames_np = frames.detach().cpu().numpy().astype(np.uint8)
    edge_maps = []
    for idx in range(frames_np.shape[1]):
        frame = np.ascontiguousarray(np.transpose(frames_np[:, idx], (1, 2, 0)))
        edge_maps.append(cv2.Canny(frame, lower, upper))
    edge = np.stack(edge_maps, axis=0)[None]
    return torch.from_numpy(edge).expand(3, -1, -1, -1).contiguous()


def _scale_for_bilateral_resolution(value: float, longest_side: int) -> float:
    if longest_side <= 0:
        return value
    return value * (longest_side / BILATERAL_REFERENCE_RESOLUTION)


def _scaled_bilateral_params(height: int, width: int) -> tuple[int, float, float]:
    longest_side = int(max(height, width))
    diameter = max(1, int(round(_scale_for_bilateral_resolution(float(BILATERAL_D), longest_side))))
    if diameter % 2 == 0:
        diameter += 1
    sigma_color = max(
        1.0, _scale_for_bilateral_resolution(float(BILATERAL_SIGMA_COLOR), longest_side)
    )
    sigma_space = max(
        1.0, _scale_for_bilateral_resolution(float(BILATERAL_SIGMA_SPACE), longest_side)
    )
    return diameter, sigma_color, sigma_space


def _import_cv2(hint_key: str):
    try:
        return importlib.import_module("cv2")
    except ImportError as exc:
        raise ImportError(
            f"Cosmos3 transfer hint '{hint_key}' requires opencv-python for on-the-fly control generation. "
            "Install opencv-python in the environment, or provide a precomputed control_path for this hint."
        ) from exc


def make_blur_control(frames: torch.Tensor, preset: str) -> torch.Tensor:
    cv2 = _import_cv2("blur")
    preset = preset.lower()
    if preset not in BLUR_PRESETS:
        raise ValueError(f"Unsupported Cosmos3 blur preset: {preset!r}.")
    if preset == "none":
        return frames.clone()

    frames_np = frames.detach().cpu().numpy().astype(np.uint8)
    _, t, h, w = frames_np.shape
    blur_params = BLUR_PRESETS[preset]
    pre_blur_factor = max(1, int(blur_params["pre_blur_downscale"]))
    downup_factor = max(1, int(blur_params["downup"]))
    result = np.empty_like(frames_np)
    for idx in range(t):
        frame = np.ascontiguousarray(np.transpose(frames_np[:, idx], (1, 2, 0)))
        if pre_blur_factor > 1:
            small_w = max(1, w // pre_blur_factor)
            small_h = max(1, h // pre_blur_factor)
            frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
        diameter, sigma_color, sigma_space = _scaled_bilateral_params(
            frame.shape[0], frame.shape[1]
        )
        for _ in range(BILATERAL_ITERATIONS):
            frame = cv2.bilateralFilter(frame, diameter, sigma_color, sigma_space)
        if pre_blur_factor > 1:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        if downup_factor > 1:
            small_w = max(1, w // downup_factor)
            small_h = max(1, h // downup_factor)
            frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_CUBIC)
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC)
        result[:, idx] = np.transpose(frame, (2, 0, 1))
    return torch.from_numpy(result).contiguous()


def load_or_compute_control_frames(
    hint: Cosmos3TransferHint,
    *,
    height: int,
    width: int,
    max_frames: int,
    input_frames: torch.Tensor | None,
) -> torch.Tensor:
    if hint.control is not None:
        return media_to_uint8_cthw(hint.control, height=height, width=width, max_frames=max_frames)
    if hint.control_path is not None:
        return media_to_uint8_cthw(
            hint.control_path, height=height, width=width, max_frames=max_frames
        )
    if hint.key == "edge":
        if input_frames is None:
            raise ValueError(
                "Cosmos3 transfer hint 'edge' requires either a video input for on-the-fly control generation "
                "or a precomputed control_path."
            )
        return make_edge_control(input_frames[:, :max_frames], hint.preset_edge_threshold)
    if hint.key == "blur":
        if input_frames is None:
            raise ValueError(
                "Cosmos3 transfer hint 'blur' requires either a video input for on-the-fly control generation "
                "or a precomputed control_path."
            )
        return make_blur_control(input_frames[:, :max_frames], hint.preset_blur_strength)
    raise FileNotFoundError(
        f"Cosmos3 transfer hint '{hint.key}' requires a precomputed control_path; "
        "on-the-fly generation is supported only for edge and blur."
    )


def pad_temporal_frames(frames: torch.Tensor, target_frames: int) -> torch.Tensor:
    if frames.ndim != 4:
        raise ValueError(
            f"Cosmos3 transfer frames must have shape [C, T, H, W], got {tuple(frames.shape)}."
        )
    target_frames = int(target_frames)
    if target_frames <= 0:
        raise ValueError("Cosmos3 transfer target frame count must be positive.")
    if frames.shape[1] >= target_frames:
        return frames
    if frames.shape[1] == 0:
        raise ValueError("Cannot pad an empty Cosmos3 transfer frame tensor.")
    padded = frames
    while padded.shape[1] < target_frames:
        pad_len = min(padded.shape[1] - 1, target_frames - padded.shape[1])
        if pad_len <= 0:
            pad_frame = padded[:, -1:].repeat(1, target_frames - padded.shape[1], 1, 1)
            padded = torch.cat([padded, pad_frame], dim=1)
            break
        padded = torch.cat([padded, padded.flip(dims=[1])[:, :pad_len]], dim=1)
    return padded
