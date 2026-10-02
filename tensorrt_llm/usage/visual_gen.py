# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded VisualGen summaries; no request identifiers or content are retained."""

import json
import math
import time
from bisect import bisect_left
from functools import wraps

UINT_MAX = 4_294_967_295
MODALITIES = ("image", "video", "video_audio", "layered_image", "mixed", "unknown")
ENDPOINTS = (
    "/v1/images/generations",
    "/v1/images/edits",
    "/v1/videos",
    "/v1/videos/generations",
)
# Only names explicitly reviewed for usage collection belong here.
EXTRA_KEYS = ("stg_scale",)
SHAPE_BOUNDS = {
    "resolution": (512, 1024, 2048, 4096),
    "numFrames": (16, 64, 128, 256),
    "numInferenceSteps": (10, 30, 50, 100),
    "batchSize": (1, 2, 4, 8),
}
# Fixed logarithmic bins retain a bounded session summary, not individual timings.
LATENCY_BOUNDS = tuple(0.001 * 1.1**i for i in range(180))


def bucket(value: object, bounds: tuple, *, allow_zero: bool = False) -> str:
    """Map a positive finite measurement to a fixed inclusive upper-bound bucket."""
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < 0
        or (value == 0 and not allow_zero)
    ):
        return "unknown"
    index = bisect_left(bounds, value)
    return f"le_{bounds[index]}" if index < len(bounds) else f"gt_{bounds[-1]}"


def supplied_extra_keys(params: object) -> tuple[str, ...]:
    """Keep only reviewed key names explicitly supplied before default expansion."""
    try:
        values = getattr(params, "extra_params", None) or {}
        return tuple(key for key in EXTRA_KEYS if key in values)
    except Exception:
        return ()


def request_shape(params: object, batch_size: int, extra_keys: tuple = ()) -> dict:
    """Bucket resolved shapes without inspecting prompts or reference content."""
    width, height = getattr(params, "width", None), getattr(params, "height", None)
    resolution = max(width, height) if type(width) is int and type(height) is int else None
    reference = (
        "video"
        if getattr(params, "video_reference", None)
        else ("image" if getattr(params, "image_reference", None) else "none")
    )
    return {
        "resolution": bucket(resolution, SHAPE_BOUNDS["resolution"]),
        "numFrames": bucket(getattr(params, "num_frames", None), SHAPE_BOUNDS["numFrames"]),
        "numInferenceSteps": bucket(
            getattr(params, "num_inference_steps", None), SHAPE_BOUNDS["numInferenceSteps"]
        ),
        "batchSize": bucket(batch_size, SHAPE_BOUNDS["batchSize"]),
        "inputReferenceKind": reference,
        "extraParamsKeysUsed": [key for key in EXTRA_KEYS if key in extra_keys],
    }


class VisualGenMetrics:
    """Session aggregates protected by the owning telemetry session's lock."""

    def __init__(self) -> None:
        self.started = time.monotonic()
        self.counters: dict[str, dict[str, int]] = {}
        self.latencies = {
            phase: [0] * (len(LATENCY_BOUNDS) + 1)
            for phase in ("generation", "pre_denoise", "denoise", "post_denoise")
        }
        self.peak_queued = 0
        self.peak_active = 0
        self.failed_component = "none"

    def _count(self, group: str, key: str) -> None:
        counts = self.counters.setdefault(group, {})
        counts[key] = min(counts.get(key, 0) + 1, UINT_MAX)

    def endpoint(self, path: str) -> None:
        if path == "/v1/videos/sync":
            path = "/v1/videos/generations"
        if path in ENDPOINTS:
            self._count("endpointRequests", path)

    def queue(self, queued: int, active: int) -> None:
        self.peak_queued = min(max(self.peak_queued, queued), UINT_MAX)
        self.peak_active = min(max(self.peak_active, active), UINT_MAX)

    def error(self, category: str) -> None:
        self._count(
            "errors", category if category in ("client", "capacity", "timeout") else "unclassified"
        )

    def request(self, modality: str) -> None:
        self._count("requestsByModality", modality if modality in MODALITIES else "unknown")

    def complete(self, shape: dict, timings: dict, error: str | None) -> None:
        for name, bounds in SHAPE_BOUNDS.items():
            value = shape.get(name, "unknown")
            if value in {*(f"le_{bound}" for bound in bounds), f"gt_{bounds[-1]}", "unknown"}:
                self._count(name, value)
        reference = shape.get("inputReferenceKind")
        if reference in ("none", "image", "video"):
            self._count("inputReferenceKind", reference)
        keys = shape.get("extraParamsKeysUsed", ())
        for key in EXTRA_KEYS:
            if key in keys:
                self._count("extraParamsKeysUsed", key)
        if error is not None:
            self.error(error)
            return
        for phase, bins in self.latencies.items():
            value = timings.get(phase)
            if type(value) in (float, int) and math.isfinite(value) and value > 0:
                index = bisect_left(LATENCY_BOUNDS, value)
                bins[index] = min(bins[index] + 1, UINT_MAX)

    def snapshot(self) -> dict:
        percentiles = {}
        for phase, bins in self.latencies.items():
            count = sum(bins)
            if not count:
                continue
            values = {"count": min(count, UINT_MAX)}
            for label, quantile in (("p50", 0.5), ("p95", 0.95)):
                target, cumulative = math.ceil(count * quantile), 0
                for index, n in enumerate(bins):
                    cumulative += n
                    if cumulative >= target:
                        values[label] = round(
                            LATENCY_BOUNDS[min(index, len(LATENCY_BOUNDS) - 1)], 3
                        )
                        if index == len(LATENCY_BOUNDS):
                            values[label + "Overflow"] = True
                        break
            percentiles[phase] = values
        return {
            "visualGenMetricsJson": json.dumps(
                {**self.counters, "latencySec": percentiles}, sort_keys=True, separators=(",", ":")
            ),
            "peakNumQueuedRequests": self.peak_queued,
            "peakNumActiveRequests": self.peak_active,
            "sessionDurationSec": min(int(time.monotonic() - self.started), UINT_MAX),
            "failedComponent": self.failed_component,
        }


def record(action: str, *args: object) -> None:
    """Update an existing enabled session without letting telemetry fail inference."""
    try:
        from . import usage_lib

        if not usage_lib.is_usage_stats_enabled():
            return
        session = usage_lib._get_session()
        if session is None:
            return
        with session.lock:
            if not session.disabled and not session.terminal_reported:
                getattr(session.visual_gen_metrics, action)(*args)
    except Exception:
        pass


def track_submission_errors(function):
    """Count synchronous submission failures without retaining exception details."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as error:
            category = (
                "client"
                if isinstance(error, (ValueError, TypeError))
                else ("capacity" if isinstance(error, MemoryError) else "unclassified")
            )
            record("error", category)
            raise

    return wrapped


class VisualGenTelemetryMiddleware:
    """Count allowlisted generation endpoints without retaining HTTP request data."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http" and scope.get("method") == "POST":
            record("endpoint", scope.get("path", ""))
        await self.app(scope, receive, send)
