# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct tests for :mod:`tensorrt_llm.serve.visual_gen_utils`.

These tests bypass the HTTP transport and call ``parse_visual_gen_params``
and the ``_warn_if_set_with_no_semantic`` / ``_merge_extra_params``
helpers directly against constructed Pydantic request objects and a
stub :class:`VisualGen`. They cover the ``extra_params`` merge truth
table plus the field-by-field overlay contract.
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any, Dict, Optional

import pytest
from PIL import Image

from tensorrt_llm.serve.openai_protocol import ImageGenerationRequest, VideoGenerationRequest
from tensorrt_llm.serve.visual_gen_utils import (
    _merge_extra_params,
    _warn_if_set_with_no_semantic,
    parse_visual_gen_params,
)
from tensorrt_llm.visual_gen import VisualGenParams


class _StubExtraParamSpec:
    def __init__(self, default: Any = None) -> None:
        self.default = default


class _StubVisualGen:
    """Minimal :class:`VisualGen` stand-in for direct conversion tests.

    The conversion layer only reads ``default_params``, ``model``, and
    ``extra_param_specs`` — populate those directly.
    """

    def __init__(
        self,
        defaults: Optional[Dict[str, Any]] = None,
        extra_param_specs: Optional[Dict[str, Any]] = None,
        model: str = "stub",
    ) -> None:
        self._defaults = defaults or {}
        self.extra_param_specs = extra_param_specs or {}
        self.model = model

    @property
    def default_params(self) -> VisualGenParams:
        # Always return a fresh instance so the conversion layer can
        # mutate it without leaking across tests.
        return VisualGenParams(**self._defaults)


@pytest.fixture
def image_request_defaults():
    return ImageGenerationRequest(prompt="cat", response_format="b64_json")


@pytest.fixture
def video_request_defaults():
    return VideoGenerationRequest(prompt="storm", response_format="b64_json")


# =============================================================================
# Default overlay — only client-sent fields override pipeline defaults
# =============================================================================


class TestDefaultOverlay:
    def test_all_none_request_keeps_pipeline_defaults(self, image_request_defaults):
        generator = _StubVisualGen(
            defaults={"width": 1024, "height": 1024, "num_inference_steps": 30},
        )
        params = parse_visual_gen_params(image_request_defaults, "id-1", generator)
        assert params.width == 1024
        assert params.height == 1024
        assert params.num_inference_steps == 30

    def test_image_explicit_fields_override_defaults(self):
        generator = _StubVisualGen(
            defaults={"width": 1024, "height": 1024, "num_inference_steps": 30},
        )
        request = ImageGenerationRequest(
            prompt="cat",
            width=512,
            height=512,
            num_inference_steps=10,
            guidance_scale=4.0,
            max_sequence_length=128,
            seed=99,
            n=4,
            negative_prompt="blurry",
        )
        params = parse_visual_gen_params(request, "id-2", generator)
        assert (params.width, params.height) == (512, 512)
        assert params.num_inference_steps == 10
        assert params.guidance_scale == 4.0
        assert params.max_sequence_length == 128
        assert params.seed == 99
        assert params.num_images_per_prompt == 4
        assert params.negative_prompt == "blurry"

    def test_size_string_used_when_width_height_absent(self):
        generator = _StubVisualGen()
        request = ImageGenerationRequest(prompt="cat", size="768x256")
        params = parse_visual_gen_params(request, "id-3", generator)
        assert (params.width, params.height) == (768, 256)

    def test_width_height_pair_wins_over_size(self):
        generator = _StubVisualGen()
        request = ImageGenerationRequest(prompt="cat", size="768x256", width=128, height=64)
        params = parse_visual_gen_params(request, "id-4", generator)
        assert (params.width, params.height) == (128, 64)

    def test_image_seed_propagates(self):
        generator = _StubVisualGen()
        request = ImageGenerationRequest(prompt="cat", seed=12345)
        params = parse_visual_gen_params(request, "id-seed", generator)
        assert params.seed == 12345


# =============================================================================
# Seed range clamp on the serve boundary
# =============================================================================


class TestSeedLowerBoundOnServeBoundary:
    """Negative seeds are rejected at the HTTP request schema; the rest
    of the int64 range is accepted, matching what the underlying
    ``torch.Generator.manual_seed`` supports.
    """

    def test_image_seed_accepts_zero_and_large_values(self):
        from tensorrt_llm.serve.openai_protocol import ImageGenerationRequest

        assert ImageGenerationRequest(prompt="x", seed=0).seed == 0
        large = 2**40
        assert ImageGenerationRequest(prompt="x", seed=large).seed == large

    def test_image_seed_rejects_negative(self):
        from pydantic import ValidationError

        from tensorrt_llm.serve.openai_protocol import ImageGenerationRequest

        with pytest.raises(ValidationError):
            ImageGenerationRequest(prompt="x", seed=-1)

    def test_video_seed_rejects_negative(self):
        from pydantic import ValidationError

        from tensorrt_llm.serve.openai_protocol import VideoGenerationRequest

        with pytest.raises(ValidationError):
            VideoGenerationRequest(prompt="x", seed=-1)


# =============================================================================
# OpenAI-shape "warn-on-set" fields
# =============================================================================


class TestWarnOnSet:
    """The TRT-LLM logger doesn't propagate through Python's root logger,
    so these tests monkeypatch :func:`logger.warning` directly and
    inspect what the helper would have emitted."""

    def _capture_warnings(self, monkeypatch):
        captured: list[str] = []

        def _fake_warning(msg: str, *args: object, **kwargs: object) -> None:
            try:
                rendered = msg % args if args else msg
            except (TypeError, ValueError):
                rendered = str(msg)
            captured.append(rendered)

        from tensorrt_llm.serve import visual_gen_utils as vgu

        monkeypatch.setattr(vgu.logger, "warning", _fake_warning)
        return captured

    def test_quality_hd_does_not_override_steps(self):
        generator = _StubVisualGen(defaults={"num_inference_steps": 25})
        request = ImageGenerationRequest(prompt="cat", quality="hd")
        params = parse_visual_gen_params(request, "id-q", generator)
        # ``quality`` is an OpenAI-shape no-semantic field. The pipeline
        # default for ``num_inference_steps`` must reach the engine
        # unchanged.
        assert params.num_inference_steps == 25

    def test_style_set_logs_warning(self, monkeypatch):
        captured = self._capture_warnings(monkeypatch)
        request = ImageGenerationRequest(prompt="cat", style="vivid")
        _warn_if_set_with_no_semantic(request, "stub")
        assert any("'style'" in m for m in captured)

    def test_user_set_does_not_log_warning(self, monkeypatch):
        captured = self._capture_warnings(monkeypatch)
        request = ImageGenerationRequest(prompt="cat", user="abc")
        _warn_if_set_with_no_semantic(request, "stub")
        assert not any("'user'" in m for m in captured)

    def test_model_mismatch_logs_warning(self, monkeypatch):
        captured = self._capture_warnings(monkeypatch)
        request = ImageGenerationRequest(prompt="cat", model="some-other")
        _warn_if_set_with_no_semantic(request, "flux2")
        assert any("'model'" in m for m in captured)

    def test_model_match_does_not_log_warning(self, monkeypatch):
        captured = self._capture_warnings(monkeypatch)
        request = ImageGenerationRequest(prompt="cat", model="flux2")
        _warn_if_set_with_no_semantic(request, "flux2")
        assert not any("'model'" in m for m in captured)


# =============================================================================
# Video frame-budget derivation
# =============================================================================


class TestVideoFrameBudget:
    def test_num_frames_wins_over_seconds_times_frame_rate(self):
        generator = _StubVisualGen(defaults={"frame_rate": 24.0})
        request = VideoGenerationRequest(prompt="x", num_frames=33, seconds=10.0)
        params = parse_visual_gen_params(request, "id-v1", generator)
        assert params.num_frames == 33

    def test_seconds_and_frame_rate_derive_num_frames(self):
        generator = _StubVisualGen(defaults={"frame_rate": 12.0})
        # fps alias resolves to frame_rate via populate_by_name=True
        request = VideoGenerationRequest(prompt="x", seconds=2.5, fps=24)
        params = parse_visual_gen_params(request, "id-v2", generator)
        assert params.frame_rate == 24.0
        assert params.num_frames == int(2.5 * 24.0)

    def test_seconds_alone_uses_pipeline_frame_rate(self):
        generator = _StubVisualGen(defaults={"frame_rate": 16.0})
        request = VideoGenerationRequest(prompt="x", seconds=4.0)
        params = parse_visual_gen_params(request, "id-v3", generator)
        assert params.frame_rate == 16.0
        assert params.num_frames == int(4.0 * 16.0)

    def test_video_does_not_carry_n(self):
        generator = _StubVisualGen()
        # Video request has no ``n`` field — Pydantic rejects it at
        # schema time, but constructing the request without it must
        # leave ``num_images_per_prompt`` unchanged from the pipeline
        # default.
        request = VideoGenerationRequest(prompt="x")
        params = parse_visual_gen_params(request, "id-v4", generator)
        assert params.num_images_per_prompt == 1


# =============================================================================
# input_reference materialization
# =============================================================================


class TestInputReferenceMaterialization:
    def test_base64_reference_written_to_disk(self, tmp_path):
        generator = _StubVisualGen()
        img = Image.new("RGB", (4, 4), (10, 20, 30))
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        request = VideoGenerationRequest(prompt="x", input_reference=b64)
        params = parse_visual_gen_params(
            request, "vid-1", generator, media_storage_path=str(tmp_path)
        )
        assert params.image is not None
        assert str(params.image).endswith("vid-1_reference.png")
        # The decoded image is identical to what we passed in.
        with open(params.image, "rb") as f:
            decoded = Image.open(f).convert("RGB")
            assert decoded.size == (4, 4)

    def test_missing_media_storage_path_raises(self):
        generator = _StubVisualGen()
        img = Image.new("RGB", (2, 2))
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        request = VideoGenerationRequest(prompt="x", input_reference=b64)
        with pytest.raises(ValueError, match="media_storage_path"):
            parse_visual_gen_params(request, "vid-2", generator, media_storage_path=None)


# =============================================================================
# _merge_extra_params — the merge truth table
# =============================================================================


class TestMergeExtraParams:
    def _make_params(self, defaults: Optional[Dict[str, Any]] = None) -> VisualGenParams:
        return VisualGenParams(extra_params=dict(defaults) if defaults else None)

    def test_omitted_key_keeps_default(self):
        specs = {"stg_scale": _StubExtraParamSpec(default=1.0)}
        params = self._make_params({"stg_scale": 1.0})
        _merge_extra_params(params, request_extras=None, extra_param_specs=specs)
        assert params.extra_params == {"stg_scale": 1.0}

    def test_known_non_null_overrides_default(self):
        specs = {"stg_scale": _StubExtraParamSpec(default=1.0)}
        params = self._make_params({"stg_scale": 1.0})
        _merge_extra_params(params, {"stg_scale": 2.5}, specs)
        assert params.extra_params["stg_scale"] == 2.5

    def test_known_null_keeps_default(self):
        """Schema-aware null sentinel: ``{"stg_scale": null}`` does not
        clear the pre-seeded pipeline default and does not pass through
        to the executor as ``None`` either."""
        specs = {"stg_scale": _StubExtraParamSpec(default=1.0)}
        params = self._make_params({"stg_scale": 1.0})
        _merge_extra_params(params, {"stg_scale": None}, specs)
        assert params.extra_params["stg_scale"] == 1.0

    def test_unknown_key_passes_through_with_value(self):
        """Unknown keys are preserved verbatim so the executor's
        strict-key validator raises ``unknown_extra_param``."""
        specs = {"stg_scale": _StubExtraParamSpec(default=1.0)}
        params = self._make_params({"stg_scale": 1.0})
        _merge_extra_params(params, {"stg_sclae": 9.9}, specs)
        assert params.extra_params == {"stg_scale": 1.0, "stg_sclae": 9.9}

    def test_unknown_key_with_null_passes_through(self):
        """Critical: unknown + null is *not* stripped. A schema-blind
        "drop every null" rule would let typos like ``{"stg_sclae":
        null}`` reach the engine as a silent no-op."""
        specs = {"stg_scale": _StubExtraParamSpec(default=1.0)}
        params = self._make_params({"stg_scale": 1.0})
        _merge_extra_params(params, {"stg_sclae": None}, specs)
        assert params.extra_params["stg_sclae"] is None

    def test_empty_extras_dict_normalizes_to_none(self):
        params = self._make_params()
        _merge_extra_params(params, request_extras=None, extra_param_specs={})
        assert params.extra_params is None
