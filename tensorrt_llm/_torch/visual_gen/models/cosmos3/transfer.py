# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos3 transfer inference helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from tensorrt_llm._torch.visual_gen.triton_kernels import (
    bilateral_filter,
    canny_edges,
    resize_area_u8,
    resize_cubic_u8,
    resize_linear_u8,
)
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.media.decoding import decode_video_reference_window

from .defaults import VIDEO_RES_SIZE_INFO


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
    "emphasize_control_in_prompt": True,  # name the active hints in the user prompt
}

# Appended to the user prompt when ``emphasize_control_in_prompt`` is on, naming
# the active hint modalities so the model is told which control it is following.
# The system prompt is left alone, which keeps the text in the training
# distribution.
CONTROL_DIRECTIVE_TEMPLATE = (
    " Follow the {hints} control video precisely: shape, contour, silhouette,"
    " position, and motion of every visible structure must align with the {hints}"
    " signal at every frame."
)

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

# Canny (lower, upper) hysteresis thresholds per edge-strength preset;
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

# Bilateral-filter parameters for the blur control, tuned at a 720p reference.
BILATERAL_REFERENCE_RESOLUTION = 720  # resolution the params below are tuned for
BILATERAL_D = 30  # filter diameter in pixels
BILATERAL_SIGMA_COLOR = 150  # color-space sigma
BILATERAL_SIGMA_SPACE = 100  # coordinate-space sigma
BILATERAL_ITERATIONS = 1  # number of bilateral passes

# Control generation is batched over frames, so unwindowed its scratch scales
# with the whole clip: a 189-frame 704p clip peaked at 7.5 GiB for edge and
# 10.2 GiB for blur before denoising had allocated anything. No kernel reaches
# across the temporal axis, so a bounded window is bitwise identical and makes
# the scratch O(window) rather than O(clip). Measured at 704p: 32 frames holds
# both under 3.4 GiB for +5% (edge) / +0.7% (blur) wall time, and still gives
# the kernels enough parallelism to saturate the GPU.
CONTROL_FRAME_WINDOW = 32


@dataclass
class Cosmos3TransferHint:
    key: str
    control: bytes | None = None
    """Precomputed control clip as encoded MP4/AVI bytes, or None to auto-compute."""
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
    emphasize_control_in_prompt: bool = True

    @property
    def ordered_hints(self) -> list[Cosmos3TransferHint]:
        return [self.hints[key] for key in TRANSFER_HINT_KEYS if key in self.hints]

    def emphasized_prompt(self, prompt: str) -> str:
        """The user prompt with the control-adherence directive appended.

        Returned unchanged when the directive is off or no hint is active.
        """
        if not self.emphasize_control_in_prompt or not self.hints:
            return prompt
        hint_names = ", ".join(hint.key for hint in self.ordered_hints)
        return prompt.rstrip() + CONTROL_DIRECTIVE_TEMPLATE.format(hints=hint_names)


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
        elif isinstance(raw, bytes):
            raw = {"control": raw}
        if isinstance(raw, str | Path):
            raise ValueError(
                f"Cosmos3 transfer hint '{key}' must carry encoded control bytes, not a path "
                f"({raw!r}); read the file client-side (Path(control).read_bytes()), as the "
                "'video' extra param does."
            )
        if not isinstance(raw, Mapping):
            # ValueError, not TypeError: the worker's error classifier maps
            # ValueError to a client error (400) and anything else to an
            # unclassified server fault (500). A malformed hint is the caller's
            # to fix, so it must not be reported as our failure.
            raise ValueError(
                f"Cosmos3 transfer hint '{key}' must be an object, encoded control bytes, or "
                f"true; got {type(raw)!r}."
            )
        if raw.get("control_path") is not None:
            raise ValueError(
                f"Cosmos3 transfer hint '{key}' no longer accepts 'control_path'; pass the "
                "encoded control clip as 'control' bytes (Path(control).read_bytes())."
            )
        control = raw.get("control")
        if control is not None and not isinstance(control, bytes):
            raise ValueError(
                f"Cosmos3 transfer hint '{key}' control must be encoded MP4/AVI bytes, got "
                f"{type(control)!r}."
            )
        hints[key] = Cosmos3TransferHint(
            key=key,
            control=control,
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
            "emphasize_control_in_prompt",
        )
        if any(extra_params.get(key, None) for key in transfer_only):
            raise ValueError(
                "Cosmos3 transfer options were provided, but no transfer hint was selected."
            )
        return None

    # `guidance_scale`, `frame_rate` and `num_frames` are advertised defaults the
    # executor merges into every request, so only `model_fields_set` distinguishes
    # a caller's value from a merged one.
    specified = getattr(req_params, "model_fields_set", frozenset())

    # Stays None unless the caller asked for a value, so the single-hint preset
    # below can apply. Reading `req_params.guidance_scale` unconditionally would
    # capture the executor-merged default and make every request look explicit,
    # which pins transfer to the generic scale and never reaches the tuned
    # per-hint presets. With no preset (multi-hint), `_forward_transfer` falls
    # back to the generic video default -- matching the reference, where the
    # per-task table applies only to a single hint.
    guidance_scale_user_set = (
        "guidance_scale" in specified or extra_params.get("guidance_scale", None) is not None
    )
    request_guidance_scale = (
        getattr(req_params, "guidance_scale", None) if guidance_scale_user_set else None
    )

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
        emphasize_control_in_prompt=_extra_or_default(
            extra_params,
            "emphasize_control_in_prompt",
            TRANSFER_SAMPLE_DEFAULTS["emphasize_control_in_prompt"],
        ),
        # `num_frames` and `frame_rate` are request fields, not extra params, so
        # they are read from `req_params` alone: `extra_params` spellings of them
        # would be a second name for the same knob with different precedence, and
        # `validate_visual_gen_params` rejects them as undeclared keys anyway.
        num_frames=getattr(req_params, "num_frames", None),
        # Only a caller-supplied frame_rate seeds this. Taking the request's
        # value unconditionally would capture the executor-merged default,
        # which then wins in _forward_transfer over a rate the pipeline
        # inferred from the source -- silently pinning every transfer to 24.
        fps=getattr(req_params, "frame_rate", None) if "frame_rate" in specified else None,
    )

    if len(hints) == 1:
        hint_key = next(iter(hints))
        for field_name, default_value in TRANSFER_DEFAULTS[hint_key].items():
            if field_name == "guidance_scale":
                user_set = guidance_scale_user_set
            elif field_name == "flow_shift":
                user_set = extra_params.get("flow_shift", None) is not None
            elif field_name == "fps":
                user_set = "frame_rate" in specified
            elif field_name == "num_frames":
                user_set = "num_frames" in specified
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


def decode_media_to_uint8_cthw(
    data: bytes, *, height: int, width: int, max_frames: int, device: torch.device
) -> torch.Tensor:
    """Decode encoded MP4/AVI ``data`` to uint8 ``[3, T, H, W]`` frames on ``device``.

    The decoder resizes to ``(height, width)`` before retaining each frame, so a
    high-resolution control never materializes at full size.
    """
    if not isinstance(data, bytes):
        raise ValueError(
            f"Cosmos3 transfer media must be encoded MP4/AVI bytes, got {type(data)!r}."
        )
    max_frames = int(max_frames)
    if max_frames < 1:
        raise ValueError(f"Cosmos3 transfer max_frames must be positive, got {max_frames}.")
    frames_thwc = decode_video_reference_window(
        data,
        first_frame=0,
        last_frame=max_frames - 1,
        target_h=int(height),
        target_w=int(width),
        device=device,
    )
    return frames_thwc.permute(3, 0, 1, 2).contiguous()


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


@nvtx_range("make_edge_control", color="blue")
def make_edge_control(frames: torch.Tensor, preset: str) -> torch.Tensor:
    """Canny edge control: uint8 ``[3, T, H, W]`` CUDA -> the same shape.

    The single-channel edge map is broadcast across RGB, since the control
    encoder consumes three channels.
    """
    try:
        lower, upper = EDGE_PRESETS[preset]
    except KeyError as exc:
        raise ValueError(f"Unsupported Cosmos3 edge preset: {preset!r}.") from exc
    edges = torch.empty_like(frames)
    for start in range(0, frames.shape[1], CONTROL_FRAME_WINDOW):
        stop = min(start + CONTROL_FRAME_WINDOW, frames.shape[1])
        # assigning [t, H, W] into [3, t, H, W] broadcasts the map across RGB
        edges[:, start:stop] = canny_edges(frames[:, start:stop], lower, upper)
    return edges


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


@nvtx_range("make_blur_control", color="blue")
def make_blur_control(frames: torch.Tensor, preset: str) -> torch.Tensor:
    """Bilateral-blur control: uint8 ``[3, T, H, W]`` CUDA -> the same shape.

    Edge-preserving blur at ``pre_blur_downscale`` resolution, then a
    ``downup`` round trip that discards high-frequency detail.
    """
    preset = preset.lower()
    if preset not in BLUR_PRESETS:
        raise ValueError(f"Unsupported Cosmos3 blur preset: {preset!r}.")
    if preset == "none":
        return frames.clone()

    _, t, h, w = frames.shape
    blur_params = BLUR_PRESETS[preset]
    pre_blur_factor = max(1, int(blur_params["pre_blur_downscale"]))
    downup_factor = max(1, int(blur_params["downup"]))

    blurred = torch.empty_like(frames)
    for start in range(0, t, CONTROL_FRAME_WINDOW):
        stop = min(start + CONTROL_FRAME_WINDOW, t)
        # The kernels are channels-last, so permute once per window rather than
        # per frame; windowing the whole chain also bounds the tensors handed
        # between its stages, not just each stage's own scratch.
        result = frames[:, start:stop].permute(1, 2, 3, 0).contiguous()
        if pre_blur_factor > 1:
            result = resize_area_u8(result, pre_blur_factor)
        diameter, sigma_color, sigma_space = _scaled_bilateral_params(
            result.shape[1], result.shape[2]
        )
        for _ in range(BILATERAL_ITERATIONS):
            result = bilateral_filter(result, diameter, sigma_color, sigma_space)
        if pre_blur_factor > 1:
            result = resize_linear_u8(result, w, h)
        if downup_factor > 1:
            result = resize_cubic_u8(result, max(1, w // downup_factor), max(1, h // downup_factor))
            result = resize_cubic_u8(result, w, h)
        blurred[:, start:stop] = result.permute(3, 0, 1, 2)
    return blurred


def load_or_compute_control_frames(
    hint: Cosmos3TransferHint,
    *,
    height: int,
    width: int,
    max_frames: int,
    input_frames: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    """Decode a hint's precomputed control, or derive one from the input video."""
    if hint.control is not None:
        return decode_media_to_uint8_cthw(
            hint.control, height=height, width=width, max_frames=max_frames, device=device
        )
    # Generated controls stay on the input frames' device, which is where the
    # decoded ones already are, so hints never mix devices.
    if hint.key == "edge":
        if input_frames is None:
            raise ValueError(
                "Cosmos3 transfer hint 'edge' requires either a video input for on-the-fly "
                "control generation or precomputed control bytes."
            )
        return make_edge_control(input_frames[:, :max_frames], hint.preset_edge_threshold)
    if hint.key == "blur":
        if input_frames is None:
            raise ValueError(
                "Cosmos3 transfer hint 'blur' requires either a video input for on-the-fly "
                "control generation or precomputed control bytes."
            )
        return make_blur_control(input_frames[:, :max_frames], hint.preset_blur_strength)
    raise ValueError(
        f"Cosmos3 transfer hint '{hint.key}' requires precomputed control bytes; "
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
