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
import os
import tempfile
from io import BytesIO
from typing import Any, Dict, Optional

import numpy as np
import pytest
from fastapi import UploadFile
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
        assert str(params.image).endswith("vid-1_reference")
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

    @staticmethod
    def _mp4_bytes(num_frames: int = 2) -> bytes:
        """Encode a 16x16 mp4v-in-mp4 clip and return its bytes.

        ``mp4v`` is a built-in FFmpeg mpeg4 encoder present in the opencv wheel.
        OpenCV writes only to a path, so encode to a tempfile and read it back.
        """
        cv2 = pytest.importorskip("cv2")
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            path = tmp.name
        try:
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (16, 16))
            for _ in range(num_frames):
                writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
            writer.release()
            with open(path, "rb") as f:
                return f.read()
        finally:
            os.remove(path)

    def test_multipart_video_reference_routes_to_extra_params_tensor(self, tmp_path):
        import torch

        generator = _StubVisualGen()
        upload = UploadFile(file=BytesIO(self._mp4_bytes()), filename="clip.mp4")
        request = VideoGenerationRequest(prompt="x", input_reference=upload)
        params = parse_visual_gen_params(
            request, "vid-3", generator, media_storage_path=str(tmp_path)
        )
        # Video content is decoded into a uint8 [T, H, W, C] tensor on the
        # model-specific ``video`` extra param, not params.image. The worker
        # crops the conditioning window + VAE-encodes.
        assert params.image is None
        video = params.extra_params["video"]
        assert isinstance(video, torch.Tensor)
        assert video.dtype == torch.uint8
        assert video.ndim == 4 and video.shape[0] == 2 and video.shape[-1] == 3
        # Video references are decoded in memory — nothing lands in media storage.
        assert list(tmp_path.iterdir()) == []

    def test_video_reference_needs_no_media_storage(self):
        # The decode is in-memory, so V2V works without a storage path at all
        # (only image references persist a file for the worker to read).
        import torch

        generator = _StubVisualGen()
        b64 = base64.b64encode(self._mp4_bytes()).decode()
        request = VideoGenerationRequest(prompt="x", input_reference=b64)
        params = parse_visual_gen_params(request, "vid-9", generator, media_storage_path=None)
        assert isinstance(params.extra_params["video"], torch.Tensor)

    def test_base64_video_reference_routes_to_extra_params_tensor(self, tmp_path):
        # Classification is content-based, so the JSON/base64 path can
        # carry video even though it has no content-type or filename.
        import torch

        generator = _StubVisualGen()
        b64 = base64.b64encode(self._mp4_bytes()).decode()
        request = VideoGenerationRequest(prompt="x", input_reference=b64)
        params = parse_visual_gen_params(
            request, "vid-4", generator, media_storage_path=str(tmp_path)
        )
        assert params.image is None
        assert isinstance(params.extra_params["video"], torch.Tensor)

    def test_video_reference_reduced_before_routes_hold_params(self):
        """``parse_visual_gen_params`` applies the spec reducers itself: the
        sync/async routes hold the returned params for the whole job lifetime,
        and ``generate_async`` reduces non-mutatively — without reduction at
        parse, the serve would retain the full decoded clip per queued
        request."""
        from tensorrt_llm._torch.visual_gen.models.cosmos3.defaults import COSMOS3_EXTRA_SPECS

        generator = _StubVisualGen(extra_param_specs=COSMOS3_EXTRA_SPECS)
        b64 = base64.b64encode(self._mp4_bytes(num_frames=8)).decode()
        request = VideoGenerationRequest(prompt="x", input_reference=b64)
        params = parse_visual_gen_params(request, "vid-10", generator, media_storage_path=None)
        # Cropped to the default conditioning window (5) at parse, not later.
        assert params.extra_params["video"].shape[0] == 5

    def test_budget_error_message_survives_parse(self, monkeypatch):
        # Helper-level: DecodedVideoTooLargeError passes through the generic
        # "undecodable" handler with its message intact. The HTTP 400 itself
        # is asserted in test_trtllm_serve_endpoints.py.
        from tensorrt_llm.inputs import media_io

        monkeypatch.setattr(media_io, "MAX_DECODED_VIDEO_BYTES", 100)
        generator = _StubVisualGen()
        b64 = base64.b64encode(self._mp4_bytes(num_frames=4)).decode()
        request = VideoGenerationRequest(prompt="x", input_reference=b64)
        with pytest.raises(ValueError, match="decoded-size budget"):
            parse_visual_gen_params(request, "vid-11", generator, media_storage_path=None)

    def test_multipart_image_reference_routes_to_image(self, tmp_path):
        # JPEG upload: content sniffing classifies it as an image and routes
        # to params.image. The stored file has no type-suffix (PIL identifies
        # by content, not name).
        generator = _StubVisualGen()
        img = Image.new("RGB", (4, 4), (10, 20, 30))
        buf = BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        upload = UploadFile(file=buf, filename="ref.jpg")
        request = VideoGenerationRequest(prompt="x", input_reference=upload)
        params = parse_visual_gen_params(
            request, "vid-5", generator, media_storage_path=str(tmp_path)
        )
        assert params.extra_params is None
        assert str(params.image).endswith("vid-5_reference")

    def test_undecodable_reference_raises_and_cleans_up(self, tmp_path):
        pytest.importorskip("cv2")
        generator = _StubVisualGen()
        b64 = base64.b64encode(b"neither an image nor a video").decode()
        request = VideoGenerationRequest(prompt="x", input_reference=b64)
        with pytest.raises(ValueError, match="neither a decodable image"):
            parse_visual_gen_params(request, "vid-6", generator, media_storage_path=str(tmp_path))
        # Classification runs on the bytes; rejected content never touches disk.
        assert list(tmp_path.iterdir()) == []

    def test_malformed_base64_reference_raises_and_cleans_up(self, tmp_path):
        generator = _StubVisualGen()
        # "ABC" survives the lenient alphabet filter but has an invalid
        # length, so b64decode raises.
        request = VideoGenerationRequest(prompt="x", input_reference="ABC")
        with pytest.raises(ValueError, match="not valid base64"):
            parse_visual_gen_params(request, "vid-7", generator, media_storage_path=str(tmp_path))
        assert list(tmp_path.iterdir()) == []

    def test_upload_stream_failure_cleans_up_tmp(self, tmp_path):
        generator = _StubVisualGen()

        class _BrokenStream:
            def read(self, *args, **kwargs):
                raise OSError("client went away")

        upload = UploadFile(file=_BrokenStream(), filename="clip.mp4")
        request = VideoGenerationRequest(prompt="x", input_reference=upload)
        # I/O failures keep their server-error semantics (no 400 masking) …
        with pytest.raises(OSError, match="client went away"):
            parse_visual_gen_params(request, "vid-8", generator, media_storage_path=str(tmp_path))
        # … but the partial materialization must not leak.
        assert list(tmp_path.iterdir()) == []


class TestMediaBytesProbes:
    """The in-memory probe/decode primitives the serve boundary runs on."""

    def test_is_decodable_image_bytes(self):
        from tensorrt_llm.inputs.media_io import is_decodable_image_bytes

        buf = BytesIO()
        Image.new("RGB", (4, 4), (1, 2, 3)).save(buf, format="PNG")
        assert is_decodable_image_bytes(buf.getvalue())
        assert not is_decodable_image_bytes(b"definitely not an image")
        # Video bytes are not an image (mp4 has no PIL-openable header).
        assert not is_decodable_image_bytes(TestInputReferenceMaterialization._mp4_bytes())

    def test_decode_video_frames_from_bytes(self):
        pytest.importorskip("cv2")
        from tensorrt_llm.inputs.media_io import decode_video_frames_from_bytes

        frames = decode_video_frames_from_bytes(TestInputReferenceMaterialization._mp4_bytes())
        assert len(frames) == 2
        assert all(isinstance(f, Image.Image) for f in frames)

    def test_decode_video_frames_from_bytes_max_frames(self):
        pytest.importorskip("cv2")
        from tensorrt_llm.inputs.media_io import decode_video_frames_from_bytes

        frames = decode_video_frames_from_bytes(
            TestInputReferenceMaterialization._mp4_bytes(), max_frames=1
        )
        assert len(frames) == 1

    def test_decode_video_frames_from_bytes_rejects_garbage(self):
        pytest.importorskip("cv2")
        from tensorrt_llm.inputs.media_io import decode_video_frames_from_bytes

        with pytest.raises(ValueError):
            decode_video_frames_from_bytes(b"not a video at all")

    def test_truncated_image_bytes_are_not_decodable(self):
        # A truncated PNG still opens (the header parses) but cannot decode
        # its pixels; the probe must reject it so the boundary 400s instead
        # of the worker 500ing at load time.
        rng_pixels = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        buf = BytesIO()
        Image.fromarray(rng_pixels).save(buf, format="PNG")
        whole = buf.getvalue()
        truncated = whole[: len(whole) // 2]
        Image.open(BytesIO(truncated))  # sanity: header-only open succeeds

        from tensorrt_llm.inputs.media_io import is_decodable_image_bytes

        assert is_decodable_image_bytes(whole)
        assert not is_decodable_image_bytes(truncated)

    def test_truncated_image_reference_rejected_at_parse(self):
        # End of the chain: a truncated image upload is rejected as a client
        # error at the boundary (never routed into the worker).
        pytest.importorskip("cv2")
        rng_pixels = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        buf = BytesIO()
        Image.fromarray(rng_pixels).save(buf, format="PNG")
        truncated = buf.getvalue()[: len(buf.getvalue()) // 2]

        generator = _StubVisualGen()
        request = VideoGenerationRequest(
            prompt="x", input_reference=base64.b64encode(truncated).decode()
        )
        with pytest.raises(ValueError, match="neither a decodable"):
            parse_visual_gen_params(request, "vid-12", generator, media_storage_path=None)

    def test_decode_video_tensor_matches_pil_route(self):
        # The streaming decoder (single preallocated buffer — the low-peak
        # path the serve uses) must produce byte-identical output to the
        # PIL-frames route.
        pytest.importorskip("cv2")
        import torch

        from tensorrt_llm.inputs.media_io import (
            decode_video_frames_from_bytes,
            decode_video_tensor_from_bytes,
            frames_to_tensor,
        )

        data = TestInputReferenceMaterialization._mp4_bytes()
        streamed = decode_video_tensor_from_bytes(data)
        via_pil = frames_to_tensor(decode_video_frames_from_bytes(data))
        assert streamed.dtype == torch.uint8 and streamed.ndim == 4
        assert torch.equal(streamed, via_pil)
        assert torch.equal(decode_video_tensor_from_bytes(data, max_frames=1), via_pil[:1])

    def test_decode_video_tensor_rejects_garbage(self):
        pytest.importorskip("cv2")
        from tensorrt_llm.inputs.media_io import decode_video_tensor_from_bytes

        with pytest.raises(ValueError):
            decode_video_tensor_from_bytes(b"not a video at all")

    def test_tempfile_fallback_without_stream_backend(self, monkeypatch):
        # Old OpenCV builds have no stream-buffered backend; the bytes spill
        # to an auto-deleted tempfile and decode through the path route.
        pytest.importorskip("cv2")
        from tensorrt_llm.inputs import media_io

        monkeypatch.setattr(media_io, "_select_cv2_stream_buffered_backend", lambda: None)
        frames = media_io.decode_video_frames_from_bytes(
            TestInputReferenceMaterialization._mp4_bytes()
        )
        assert len(frames) == 2


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


class _FakeCapture:
    """Stands in for ``cv2.VideoCapture`` to exercise declared-count handling."""

    def __init__(self, frames, declared):
        self._frames = list(frames)
        self._pos = 0
        self._declared = declared

    def isOpened(self):
        return True

    def get(self, prop):
        return self._declared

    def read(self):
        if self._pos < len(self._frames):
            frame = self._frames[self._pos]
            self._pos += 1
            return True, frame
        return False, None


class _FakeCv2:
    CAP_PROP_FRAME_COUNT = 7
    COLOR_BGR2RGB = 4

    @staticmethod
    def cvtColor(frame, code):
        return frame


class TestDecodeCaptureGuards:
    """``_decode_capture_to_tensor`` against containers that misreport length.

    The declared frame count is metadata, not evidence — the decoder must
    stream correctly whether it is accurate, unknown, under-, over-, or
    absurdly reported."""

    def _decode(self, num_frames, declared, max_frames=None):
        import torch

        from tensorrt_llm.inputs.media_io import _decode_capture_to_tensor

        frames = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(num_frames)]
        out = _decode_capture_to_tensor(
            _FakeCv2, _FakeCapture(frames, declared), max_frames, "test"
        )
        assert out.dtype == torch.uint8
        return out

    def test_accurate_declaration(self):
        out = self._decode(3, declared=3)
        assert out.shape == (3, 4, 4, 3)
        assert int(out[2, 0, 0, 0]) == 2  # frame order preserved

    def test_unknown_declaration_falls_back(self):
        assert self._decode(3, declared=0).shape == (3, 4, 4, 3)
        assert self._decode(3, declared=-1).shape == (3, 4, 4, 3)

    def test_underreported_declaration_keeps_overflow(self):
        out = self._decode(5, declared=2)
        assert out.shape == (5, 4, 4, 3)
        assert int(out[4, 0, 0, 0]) == 4

    def test_overreported_declaration_trims_storage(self):
        out = self._decode(3, declared=10)
        assert out.shape == (3, 4, 4, 3)
        # The oversized buffer is not retained behind the result.
        assert out.untyped_storage().size() == out.numel()

    def test_absurd_declaration_allocation_is_byte_budgeted(self, monkeypatch):
        # Realistic 720p frames + an absurd declared count: the preallocation
        # request itself must stay within MAX_DECODED_VIDEO_BYTES (a frame
        # cap alone would still be ~18.5 GiB at 720p). Spy on np.empty to
        # assert the requested size, not just the result.
        import math

        from tensorrt_llm.inputs import media_io

        requested = []
        real_empty = media_io.np.empty

        def spy(shape, dtype=None):
            requested.append((tuple(shape), dtype))
            return real_empty(shape, dtype=dtype)

        monkeypatch.setattr(media_io.np, "empty", spy)
        # Lower the budget so the (real) allocation the spy delegates to stays
        # small; the assertion is about the *requested* size honoring it.
        monkeypatch.setattr(media_io, "MAX_DECODED_VIDEO_BYTES", 32 << 20)
        frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(3)]
        out = media_io._decode_capture_to_tensor(
            _FakeCv2, _FakeCapture(frames, declared=10**9), None, "test"
        )
        assert out.shape == (3, 720, 1280, 3)
        (shape, _dtype) = requested[0]
        assert math.prod(shape) <= media_io.MAX_DECODED_VIDEO_BYTES

    def test_decoded_byte_budget_rejects_oversized_streams(self, monkeypatch):
        # Total accumulation is bounded too — a stream that exceeds the budget
        # raises instead of growing without bound (unknown-length containers
        # included, where no buffer is preallocated at all).
        from tensorrt_llm.inputs import media_io

        monkeypatch.setattr(media_io, "MAX_DECODED_VIDEO_BYTES", 100)
        frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(5)]  # 48 B each
        with pytest.raises(ValueError, match="decoded-size budget"):
            media_io._decode_capture_to_tensor(
                _FakeCv2, _FakeCapture(frames, declared=0), None, "test"
            )

    def test_max_frames_bounds_decode(self):
        assert self._decode(5, declared=5, max_frames=2).shape[0] == 2
