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
import pickle
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional
from unittest import mock

import numpy as np
import pytest
import torch
from fastapi import UploadFile
from PIL import Image

from tensorrt_llm.serve import visual_gen_utils
from tensorrt_llm.serve.openai_protocol import ImageGenerationRequest, VideoGenerationRequest
from tensorrt_llm.serve.visual_gen_utils import (
    _merge_extra_params,
    _warn_if_set_with_no_semantic,
    parse_visual_gen_params,
)
from tensorrt_llm.visual_gen import VisualGenParams
from tensorrt_llm.visual_gen.media_refs import prepare_reference_slots


def _parse_and_prepare(request, generator):
    """Run the production reference flow: serve transport, then engine resolve.

    ``parse_visual_gen_params`` only normalizes transport (upload -> bytes,
    strings pass through with their declared format); resolution to bytes
    happens at the engine choke point, so tests that assert on the resolved
    payload drive both stages.
    """
    params = parse_visual_gen_params(request, generator)
    prepare_reference_slots(params)
    return params


pytestmark = pytest.mark.cpu_only


class _StubExtraParamSpec:
    def __init__(self, default: Any = None, type: str = "str") -> None:
        self.default = default
        self.type = type


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
    return VideoGenerationRequest(prompt="storm", response_format="file")


# =============================================================================
# Default overlay — only client-sent fields override pipeline defaults
# =============================================================================


class TestDefaultOverlay:
    def test_all_none_request_keeps_pipeline_defaults(self, image_request_defaults):
        generator = _StubVisualGen(
            defaults={"width": 1024, "height": 1024, "num_inference_steps": 30},
        )
        params = parse_visual_gen_params(image_request_defaults, generator)
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
        params = parse_visual_gen_params(request, generator)
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
        params = parse_visual_gen_params(request, generator)
        assert (params.width, params.height) == (768, 256)

    def test_width_height_pair_wins_over_size(self):
        generator = _StubVisualGen()
        request = ImageGenerationRequest(prompt="cat", size="768x256", width=128, height=64)
        params = parse_visual_gen_params(request, generator)
        assert (params.width, params.height) == (128, 64)

    def test_image_seed_propagates(self):
        generator = _StubVisualGen()
        request = ImageGenerationRequest(prompt="cat", seed=12345)
        params = parse_visual_gen_params(request, generator)
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
        params = parse_visual_gen_params(request, generator)
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
        params = parse_visual_gen_params(request, generator)
        assert params.num_frames == 33

    def test_seconds_and_frame_rate_derive_num_frames(self):
        generator = _StubVisualGen(defaults={"frame_rate": 12.0})
        # fps alias resolves to frame_rate via populate_by_name=True
        request = VideoGenerationRequest(prompt="x", seconds=2.5, fps=24)
        params = parse_visual_gen_params(request, generator)
        assert params.frame_rate == 24.0
        assert params.num_frames == int(2.5 * 24.0)

    def test_seconds_alone_uses_pipeline_frame_rate(self):
        generator = _StubVisualGen(defaults={"frame_rate": 16.0})
        request = VideoGenerationRequest(prompt="x", seconds=4.0)
        params = parse_visual_gen_params(request, generator)
        assert params.frame_rate == 16.0
        assert params.num_frames == int(4.0 * 16.0)

    def test_video_does_not_carry_n(self):
        generator = _StubVisualGen()
        # Video request has no ``n`` field — Pydantic rejects it at
        # schema time, but constructing the request without it must
        # leave ``num_images_per_prompt`` unchanged from the pipeline
        # default.
        request = VideoGenerationRequest(prompt="x")
        params = parse_visual_gen_params(request, generator)
        assert params.num_images_per_prompt == 1


# =============================================================================
# reference resolution
# =============================================================================


class TestInputReferenceResolution:
    def test_base64_image_reference_resolves_to_bytes(self, tmp_path):
        generator = _StubVisualGen()
        img = Image.new("RGB", (4, 4), (10, 20, 30))
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        request = VideoGenerationRequest(
            prompt="x", image_reference={"content": b64, "format": "base64"}
        )
        params = _parse_and_prepare(request, generator)
        assert len(params.image_reference) == 1
        ref_path = params.image_reference[0].content
        assert ref_path == buf.getvalue()
        assert params.image_reference[0].format == "bytes"

    def test_image_reference_role_and_list(self, tmp_path):
        generator = _StubVisualGen()
        buf = BytesIO()
        Image.new("RGB", (4, 4)).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        request = VideoGenerationRequest(
            prompt="x",
            image_reference=[
                {"content": b64, "format": "base64"},
                {"content": b64, "format": "base64", "role": "last_frame"},
            ],
        )
        params = _parse_and_prepare(request, generator)
        assert [r.role for r in params.image_reference] == [None, "last_frame"]
        assert [r.content for r in params.image_reference] == [buf.getvalue()] * 2

    _TEST_DATA = Path(__file__).parent / "test_data"

    @staticmethod
    def _mp4_bytes() -> bytes:
        """9-frame H.264-in-MP4 fixture (provenance: test_data/README.md)."""
        return (
            TestInputReferenceResolution._TEST_DATA / "cosmos3_v2v_ref_9f_bframes.mp4"
        ).read_bytes()

    @staticmethod
    def _avi_bytes() -> bytes:
        """Same 9 frames as H.264-in-AVI (provenance: test_data/README.md)."""
        return (
            TestInputReferenceResolution._TEST_DATA / "cosmos3_v2v_ref_9f_bframes.avi"
        ).read_bytes()

    def test_multipart_avi_video_reference_resolves_to_bytes(self, tmp_path):
        # The AVI container survives the boundary and is persisted as untouched
        # encoded bytes for the worker to demux.
        generator = _StubVisualGen()
        payload = self._avi_bytes()
        upload = UploadFile(file=BytesIO(payload), filename="clip.avi")
        request = VideoGenerationRequest(prompt="x", video_reference=upload)
        params = _parse_and_prepare(request, generator)
        assert params.image_reference is None
        assert params.video_reference[0].content == payload

    def test_multipart_mp4_video_reference_resolves_to_bytes(self, tmp_path):
        generator = _StubVisualGen()
        payload = self._mp4_bytes()
        upload = UploadFile(file=BytesIO(payload), filename="clip.mp4")
        request = VideoGenerationRequest(prompt="x", video_reference=upload)
        params = _parse_and_prepare(request, generator)
        # Encoded payload is persisted byte-identical — the boundary never
        # decodes video; the worker demuxes/NVDEC-decodes the conditioning
        # window from the stored file.
        assert params.image_reference is None
        assert params.video_reference[0].content == payload

    def test_deprecated_input_reference_routes_by_sniff(self, tmp_path):
        # The deprecated single input_reference is sniff-routed to the typed slot.
        generator = _StubVisualGen()
        buf = BytesIO()
        Image.new("RGB", (4, 4)).save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        vid_b64 = base64.b64encode(self._mp4_bytes()).decode()

        p = _parse_and_prepare(
            VideoGenerationRequest(
                prompt="x", input_reference=img_b64, input_reference_format="base64"
            ),
            generator,
        )
        assert len(p.image_reference) == 1 and p.video_reference is None

        p = _parse_and_prepare(
            VideoGenerationRequest(
                prompt="x", input_reference=vid_b64, input_reference_format="base64"
            ),
            generator,
        )
        assert len(p.video_reference) == 1 and p.image_reference is None

    def test_input_reference_ignored_when_typed_reference_set(self, tmp_path):
        # A typed reference takes precedence; the deprecated input_reference is dropped.
        generator = _StubVisualGen()
        buf = BytesIO()
        Image.new("RGB", (4, 4)).save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        vid_b64 = base64.b64encode(self._mp4_bytes()).decode()
        p = _parse_and_prepare(
            VideoGenerationRequest(
                prompt="x",
                image_reference={"content": img_b64, "format": "base64"},
                input_reference=vid_b64,
                input_reference_format="base64",
            ),
            generator,
        )
        assert len(p.image_reference) == 1
        assert p.video_reference is None  # input_reference video dropped

    def test_base64_video_reference_resolves_to_bytes(self, tmp_path):
        # The JSON/base64 path carries video even though it has no content-type
        # or filename; modality is declared by the field name.
        generator = _StubVisualGen()
        payload = self._mp4_bytes()
        b64 = base64.b64encode(payload).decode()
        request = VideoGenerationRequest(
            prompt="x", video_reference={"content": b64, "format": "base64"}
        )
        params = _parse_and_prepare(request, generator)
        assert params.image_reference is None
        assert params.video_reference[0].content == payload

    def test_video_reference_survives_real_specs(self, tmp_path):
        """With the real cosmos3 specs loaded, the encoded payload is persisted
        byte-identical — the boundary never transforms video content; the
        worker decodes the conditioning window."""
        from tensorrt_llm._torch.visual_gen.models.cosmos3.defaults import COSMOS3_EXTRA_SPECS

        generator = _StubVisualGen(extra_param_specs=COSMOS3_EXTRA_SPECS)
        payload = self._mp4_bytes()
        b64 = base64.b64encode(payload).decode()
        request = VideoGenerationRequest(
            prompt="x", video_reference={"content": b64, "format": "base64"}
        )
        params = _parse_and_prepare(request, generator)
        assert params.video_reference[0].content == payload

    def test_multipart_image_reference_resolves_to_bytes(self, tmp_path):
        # JPEG upload routed by field name to image_reference. The stored file
        # has no type-suffix (PIL identifies by content, not name).
        generator = _StubVisualGen()
        img = Image.new("RGB", (4, 4), (10, 20, 30))
        buf = BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        upload = UploadFile(file=buf, filename="ref.jpg")
        request = VideoGenerationRequest(prompt="x", image_reference=upload)
        params = _parse_and_prepare(request, generator)
        assert params.extra_params is None
        assert isinstance(params.image_reference[0].content, bytes)

    def test_wrong_modality_content_raises(self, tmp_path):
        # The field name declares modality; mismatched content is a client error.
        generator = _StubVisualGen()
        buf = BytesIO()
        Image.new("RGB", (2, 2)).save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        vid_b64 = base64.b64encode(self._mp4_bytes()).decode()
        with pytest.raises(ValueError, match="video_reference is not a recognized"):
            _parse_and_prepare(
                VideoGenerationRequest(
                    prompt="x", video_reference={"content": img_b64, "format": "base64"}
                ),
                generator,
            )
        with pytest.raises(ValueError, match="image_reference is not a recognized image"):
            _parse_and_prepare(
                VideoGenerationRequest(
                    prompt="x", image_reference={"content": vid_b64, "format": "base64"}
                ),
                generator,
            )
        assert list(tmp_path.iterdir()) == []  # nothing is ever written to disk

    def test_undecodable_image_reference_raises_and_cleans_up(self, tmp_path):
        generator = _StubVisualGen()
        b64 = base64.b64encode(b"neither an image nor a video").decode()
        request = VideoGenerationRequest(
            prompt="x", image_reference={"content": b64, "format": "base64"}
        )
        with pytest.raises(ValueError, match="not a recognized image"):
            _parse_and_prepare(request, generator)
        # Classification runs on the bytes; rejected content never touches disk.
        assert list(tmp_path.iterdir()) == []  # nothing is ever written to disk

    def test_malformed_base64_reference_raises_and_cleans_up(self, tmp_path):
        generator = _StubVisualGen()
        # "ABC" has an invalid base64 length. The declared format is honored,
        # so this is a decode error rather than a fallback to a filesystem read.
        request = VideoGenerationRequest(
            prompt="x", image_reference={"content": "ABC", "format": "base64"}
        )
        with pytest.raises(ValueError, match="not valid base64"):
            _parse_and_prepare(request, generator)
        assert list(tmp_path.iterdir()) == []  # nothing is ever written to disk

    def test_upload_stream_failure_cleans_up_tmp(self, tmp_path):
        generator = _StubVisualGen()

        class _BrokenStream:
            def read(self, *args, **kwargs):
                raise OSError("client went away")

        upload = UploadFile(file=_BrokenStream(), filename="clip.mp4")
        request = VideoGenerationRequest(prompt="x", video_reference=upload)
        # I/O failures keep their server-error semantics (no 400 masking) …
        with pytest.raises(OSError, match="client went away"):
            _parse_and_prepare(request, generator)
        # … and the payload read fails before any file is written, so nothing leaks.
        assert list(tmp_path.iterdir()) == []  # nothing is ever written to disk

    def test_multi_reference_partial_failure_cleans_up(self, tmp_path):
        # A later item's rejection removes the files earlier items already wrote,
        # so a rejected multi-reference request leaves nothing on disk.
        generator = _StubVisualGen()
        buf = BytesIO()
        Image.new("RGB", (4, 4)).save(buf, format="PNG")
        good = base64.b64encode(buf.getvalue()).decode()
        bad = base64.b64encode(b"neither an image nor a video").decode()
        request = VideoGenerationRequest(
            prompt="x",
            image_reference=[
                {"content": good, "format": "base64"},
                {"content": bad, "format": "base64"},
            ],
        )
        with pytest.raises(ValueError, match="not a recognized image"):
            _parse_and_prepare(request, generator)
        assert list(tmp_path.iterdir()) == []  # nothing is ever written to disk

    def test_file_uri_image_reference_is_read(self, tmp_path):
        # format="path" also accepts a file:// URI, normalized before the read.
        generator = _StubVisualGen()
        src = tmp_path / "ref.png"
        Image.new("RGB", (4, 4), (7, 8, 9)).save(src, format="PNG")
        request = VideoGenerationRequest(
            prompt="x", image_reference={"content": src.as_uri(), "format": "path"}
        )
        params = _parse_and_prepare(request, generator)
        assert params.image_reference[0].content == src.read_bytes()
        assert params.image_reference[0].format == "bytes"

    def test_bare_path_image_reference_is_read(self, tmp_path):
        # A path is read at the coordinator, so the worker needs no shared
        # filesystem to see what the client named.
        generator = _StubVisualGen()
        src = tmp_path / "ref.png"
        Image.new("RGB", (4, 4), (11, 22, 33)).save(src, format="PNG")
        request = VideoGenerationRequest(
            prompt="x", image_reference={"content": str(src), "format": "path"}
        )
        params = _parse_and_prepare(request, generator)
        assert params.image_reference[0].content == src.read_bytes()
        assert params.image_reference[0].format == "bytes"

    def test_http_url_image_reference_is_fetched(self, tmp_path, monkeypatch):
        # An http(s) reference is fetched through the guarded loader.
        generator = _StubVisualGen()
        buf = BytesIO()
        Image.new("RGB", (4, 4)).save(buf, format="PNG")
        png = buf.getvalue()

        class _FakeResp:
            def __init__(self, content):
                self.content = content

        monkeypatch.setattr(
            "tensorrt_llm.visual_gen.media_refs._safe_request_get",
            lambda url, **kwargs: _FakeResp(png),
        )
        request = VideoGenerationRequest(
            prompt="x",
            image_reference={"content": "https://example.com/a.png", "format": "url"},
        )
        params = _parse_and_prepare(request, generator)
        assert params.image_reference[0].content == png

    def test_http_url_fetch_failure_is_client_error(self, tmp_path, monkeypatch):
        # A blocked/failed fetch (e.g. SSRF guard) is a client 400, not a 500,
        # and leaves nothing on disk.
        generator = _StubVisualGen()

        def _blocked(url, **kwargs):
            raise RuntimeError("URL resolves to a non-public address (10.0.0.1)")

        monkeypatch.setattr("tensorrt_llm.visual_gen.media_refs._safe_request_get", _blocked)
        request = VideoGenerationRequest(
            prompt="x", image_reference={"content": "http://10.0.0.1/a.png", "format": "url"}
        )
        with pytest.raises(ValueError, match="reference URL could not be fetched"):
            _parse_and_prepare(request, generator)
        assert list(tmp_path.iterdir()) == []  # nothing is ever written to disk

    def test_missing_file_uri_is_client_error(self, tmp_path):
        # A file:// path that does not exist is a client 400, not a server 500.
        generator = _StubVisualGen()
        missing = (tmp_path / "does_not_exist.png").as_uri()
        request = VideoGenerationRequest(
            prompt="x", image_reference={"content": missing, "format": "path"}
        )
        with pytest.raises(ValueError, match="reference file could not be read"):
            _parse_and_prepare(request, generator)
        assert list(tmp_path.iterdir()) == []  # nothing is ever written to disk

    def test_bare_reference_string_is_rejected(self):
        # The bare-string shorthand is gone: a reference must declare its wire
        # form rather than have it guessed from the shape of the value.
        from pydantic import ValidationError

        buf = BytesIO()
        Image.new("RGB", (4, 4)).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        with pytest.raises(ValidationError):
            VideoGenerationRequest(prompt="x", image_reference=b64)
        with pytest.raises(ValidationError, match="a bare str is no longer accepted"):
            VisualGenParams(image_reference=b64)

    def test_json_reference_cannot_declare_bytes(self):
        # JSON cannot carry raw bytes; the HTTP schema says so instead of
        # letting a str reach the engine claiming to be bytes.
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="multipart/form-data"):
            VideoGenerationRequest(
                prompt="x", image_reference={"content": "abc", "format": "bytes"}
            )


class TestMediaBytesProbes:
    """The in-memory signature probes the serve boundary routes on."""

    def test_sniff_recognizes_audio_containers(self):
        """Audio has to classify as itself, not fall through to ``None``.

        ``None`` means "not media", which is what a caller rejects on, so an
        audio reference that sniffs to ``None`` is indistinguishable from a
        text file. Headers are the real ones ffmpeg emits.
        """
        from tensorrt_llm.inputs.media_io import sniff_media_kind

        assert sniff_media_kind(b"RIFF\x2e\x45\x00\x00WAVEfmt ") == "audio"
        assert sniff_media_kind(b"fLaC\x00\x00\x00\x22") == "audio"
        assert sniff_media_kind(b"OggS\x00\x02\x00\x00\x00\x00\x00\x00") == "audio"
        assert sniff_media_kind(b"ID3\x04\x00\x00\x00\x00\x00\x00") == "audio"
        # bare MPEG/ADTS frame sync — an MP3 or AAC with no leading tag
        assert sniff_media_kind(b"\xff\xf1\x50\x40\x21\x3f\xfc\xde") == "audio"
        assert sniff_media_kind(b"\xff\xfb\x90\x00\x00\x00\x00\x00") == "audio"

    def test_sniff_separates_m4a_from_mp4(self):
        """An ``.m4a`` is an MP4 whose brand says audio-only; without the brand
        check it takes the video default."""
        from tensorrt_llm.inputs.media_io import sniff_media_kind

        assert sniff_media_kind(self._ftyp(b"M4A ", (b"isom", b"mp42"))) == "audio"
        assert sniff_media_kind(self._ftyp(b"M4B ")) == "audio"
        assert sniff_media_kind(self._ftyp(b"isom", (b"iso2", b"avc1"))) == "video"

    def test_sniff_media_kind(self):
        from tensorrt_llm.inputs.media_io import sniff_media_kind

        png = BytesIO()
        Image.new("RGB", (2, 2)).save(png, format="PNG")
        jpg = BytesIO()
        Image.new("RGB", (2, 2)).save(jpg, format="JPEG")
        assert sniff_media_kind(png.getvalue()) == "image"
        assert sniff_media_kind(jpg.getvalue()) == "image"
        assert sniff_media_kind(TestInputReferenceResolution._mp4_bytes()) == "video"
        assert sniff_media_kind(TestInputReferenceResolution._avi_bytes()) == "video"
        assert sniff_media_kind(b"plain text, not media") is None
        assert sniff_media_kind(b"") is None
        # RIFF carries both: the form type at [8:12] is what separates them.
        assert sniff_media_kind(b"RIFF\x00\x00\x00\x00WAVEfmt ") == "audio"
        assert sniff_media_kind(b"RIFF\x00\x00\x00\x00WEBP") is None

    @staticmethod
    def _ftyp(major: bytes, compatible: tuple = (), *, size: int = None) -> bytes:
        """Build an ISO-BMFF `ftyp` box: size|'ftyp'|major|minor|compatible*."""
        body = major + b"\x00\x00\x00\x00" + b"".join(compatible)
        declared = len(body) + 8 if size is None else size
        return declared.to_bytes(4, "big") + b"ftyp" + body

    @staticmethod
    def _ftyp_ext(major: bytes, compatible: tuple = (), *, largesize: int = None) -> bytes:
        """Build a 64-bit `ftyp` box: 1|'ftyp'|largesize|major|minor|compat*.

        ``size == 1`` inserts an 8-byte largesize between the type and the
        brands, so the major brand lives at [16:20], not [8:12].
        """
        body = major + b"\x00\x00\x00\x00" + b"".join(compatible)
        declared = len(body) + 16 if largesize is None else largesize
        return (1).to_bytes(4, "big") + b"ftyp" + declared.to_bytes(8, "big") + body

    def test_sniff_isobmff_still_images_are_not_video(self):
        """`ftyp` marks the ISO-BMFF family, not video: HEIF/AVIF photos share
        it with MP4. Routing them to the video slot would hand a still image to
        NVDEC; they must classify as image instead."""
        from tensorrt_llm.inputs.media_io import sniff_media_kind

        # HEIC as major brand (the iOS camera default).
        assert sniff_media_kind(self._ftyp(b"heic", (b"mif1", b"heic"))) == "image"
        # `mif1` major with `heic` compatible.
        assert sniff_media_kind(self._ftyp(b"mif1", (b"heic",))) == "image"
        # AVIF: the spec requires avif/avis among the COMPATIBLE brands, so a
        # major-brand-only check would miss this one.
        assert sniff_media_kind(self._ftyp(b"iso8", (b"avif", b"mif1"))) == "image"
        assert sniff_media_kind(self._ftyp(b"avis", (b"avif",))) == "image"
        # HEVC image sequence brands.
        for brand in (b"heix", b"heim", b"heis", b"hevc", b"hevx", b"hevm", b"hevs"):
            assert sniff_media_kind(self._ftyp(brand)) == "image", brand
        # Ordinary video ISO-BMFF stays video.
        assert sniff_media_kind(self._ftyp(b"mp42", (b"isom", b"mp42"))) == "video"
        assert sniff_media_kind(self._ftyp(b"isom", (b"iso2", b"avc1"))) == "video"
        assert sniff_media_kind(self._ftyp(b"qt  ")) == "video"

    def test_sniff_ftyp_reads_the_whole_declared_box(self):
        """A brand list longer than a fixed peek window must still be seen —
        an `avif` brand at the tail decides image-vs-video."""
        from tensorrt_llm.inputs.media_io import sniff_media_kind

        # 21 compatible brands = a 100-byte box; `avif` sits past byte 64.
        padded = tuple([b"free"] * 20 + [b"avif"])
        box = self._ftyp(b"isom", padded)
        assert len(box) > 64 and box.index(b"avif") > 64
        assert sniff_media_kind(box) == "image"

    def test_sniff_ftyp_extended_size_shifts_the_brands(self):
        """`size == 1` puts a 64-bit largesize at [8:16]; a parser that reads
        the major brand at [8:12] sees half of that integer instead."""
        from tensorrt_llm.inputs.media_io import sniff_media_kind

        assert sniff_media_kind(self._ftyp_ext(b"heic", (b"mif1",))) == "image"
        assert sniff_media_kind(self._ftyp_ext(b"isom", (b"avif",))) == "image"
        assert sniff_media_kind(self._ftyp_ext(b"mp42", (b"isom",))) == "video"

    def test_sniff_ftyp_bounds_are_checked(self):
        """An `ftyp` box we cannot read in full is unclassifiable: guessing
        "video" off a partial scan is how a HEIC reaches NVDEC."""
        from tensorrt_llm.inputs.media_io import sniff_media_kind

        # Declared size larger than the payload (truncated file).
        assert sniff_media_kind(self._ftyp(b"heic", (b"mif1",), size=4096)) is None
        # Declared size 0 means "box runs to EOF" — readable, so classify.
        assert sniff_media_kind(self._ftyp(b"iso8", (b"avif",), size=0)) == "image"
        # Nonsense small size: no room for even a major brand.
        assert sniff_media_kind(self._ftyp(b"heic", (b"mif1",), size=8)) is None
        assert sniff_media_kind(self._ftyp(b"mp42", (b"avif",), size=8)) is None
        # `minor_version` is mandatory, so a box is at least 16 bytes (24 with
        # the largesize escape); stopping at the major brand is malformed.
        assert sniff_media_kind(self._ftyp(b"heic", (b"mif1",), size=12)) is None
        assert sniff_media_kind(self._ftyp_ext(b"heic", (b"mif1",), largesize=20)) is None
        # Compatible-brand area must hold whole four-byte brands.
        assert sniff_media_kind(self._ftyp(b"mp42", (b"isom",), size=18) + b"\x00\x00") is None
        # Past the scan bound: bounded rather than scanned.
        assert sniff_media_kind(self._ftyp(b"mp42", tuple([b"free"] * 2000))) is None
        # Header shorter than a brand.
        assert sniff_media_kind(b"\x00\x00\x00\x18ftyp") is None

    def test_heif_reference_rejected_with_actionable_message(self):
        """A HEIC upload is a valid file we simply do not support — the 400
        must say so, not claim the file is corrupt."""
        generator = _StubVisualGen()
        heic = self._ftyp(b"heic", (b"mif1", b"heic")) + b"\x00" * 64
        request = VideoGenerationRequest(
            prompt="x",
            image_reference={"content": base64.b64encode(heic).decode(), "format": "base64"},
        )
        with pytest.raises(ValueError, match="HEIF/AVIF"):
            _parse_and_prepare(request, generator)

    def test_truncated_image_reference_is_routed_not_decoded(self, tmp_path):
        """The boundary routes on signature and never decodes.

        A truncated PNG reaches the image slot; the worker's load is what
        rejects it (as a client error). Decoding here would put unbounded,
        client-controlled CPU on an async request path and duplicate work
        the worker repeats anyway.
        """
        rng_pixels = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        buf = BytesIO()
        Image.fromarray(rng_pixels).save(buf, format="PNG")
        whole = buf.getvalue()
        truncated = whole[: len(whole) // 2]
        with pytest.raises(OSError):
            Image.open(BytesIO(truncated)).load()  # sanity: it really is broken

        generator = _StubVisualGen()
        request = VideoGenerationRequest(
            prompt="x",
            image_reference={"content": base64.b64encode(truncated).decode(), "format": "base64"},
        )
        params = _parse_and_prepare(request, generator)
        assert params.image_reference[0].content == truncated


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


# =============================================================================
# Inline binary extra params — base64 in, bytes out
# =============================================================================


class TestInlineMediaDecoding:
    """A pipeline that declares a ``bytes`` extra param must receive bytes.

    JSON has no byte type, so an HTTP client can only inline binary as base64.
    Decoding happens here rather than in the pipeline, which keeps a
    bytes-only contract. Cosmos3 transfer's precomputed controls
    (`depth`/`seg`/`wsm`) are unreachable over serving without it.
    """

    def _generator(self):
        return _StubVisualGen(
            extra_param_specs={
                "video": _StubExtraParamSpec(type="bytes"),
                "edge": _StubExtraParamSpec(type="bool_or_bytes_or_dict"),
                "resolution": _StubExtraParamSpec(type="str"),
            }
        )

    def test_base64_extra_param_reaches_the_pipeline_as_bytes(self):
        request = VideoGenerationRequest(
            prompt="storm",
            extra_params={"video": base64.b64encode(b"\x00mp4").decode()},
        )
        params = parse_visual_gen_params(request, self._generator())
        assert params.extra_params["video"] == b"\x00mp4"

    def test_nested_control_reaches_the_pipeline_as_bytes(self):
        request = VideoGenerationRequest(
            prompt="storm",
            extra_params={"edge": {"control": base64.b64encode(b"\x00ctrl").decode()}},
        )
        params = parse_visual_gen_params(request, self._generator())
        assert params.extra_params["edge"]["control"] == b"\x00ctrl"

    def test_non_media_params_are_untouched(self):
        request = VideoGenerationRequest(
            prompt="storm", extra_params={"resolution": "720", "edge": True}
        )
        params = parse_visual_gen_params(request, self._generator())
        assert params.extra_params["resolution"] == "720"
        assert params.extra_params["edge"] is True

    def test_malformed_base64_is_a_client_error(self):
        request = VideoGenerationRequest(prompt="storm", extra_params={"video": "not!b64!"})
        with pytest.raises(ValueError, match="not valid base64"):
            parse_visual_gen_params(request, self._generator())


class TestPrepareReferenceSlots:
    """The engine choke point: every declared form resolves to raw bytes."""

    def test_every_format_resolves_to_the_same_bytes(self, tmp_path):
        """path / base64 / bytes all name the same payload, so all three must
        land on byte-identical content with format rewritten to "bytes"."""
        from tensorrt_llm.visual_gen import MediaRef, VisualGenParams
        from tensorrt_llm.visual_gen.media_refs import prepare_reference_slots

        buf = BytesIO()
        Image.new("RGB", (4, 4), (7, 8, 9)).save(buf, format="PNG")
        png = buf.getvalue()
        src = tmp_path / "ref.png"
        src.write_bytes(png)

        for ref in (
            MediaRef(content=str(src), format="path"),
            MediaRef(content=base64.b64encode(png).decode(), format="base64"),
            MediaRef(content=png, format="bytes"),
        ):
            params = VisualGenParams(image_reference=ref)
            prepare_reference_slots(params)
            assert params.image_reference[0].content == png
            assert params.image_reference[0].format == "bytes"

    def test_nothing_is_resolves_to_bytes(self, tmp_path, monkeypatch):
        """References never touch the filesystem, so a worker needs no shared
        filesystem to read what the coordinator resolved."""
        from tensorrt_llm.visual_gen import MediaRef, VisualGenParams
        from tensorrt_llm.visual_gen.media_refs import prepare_reference_slots

        monkeypatch.setenv("TRTLLM_MEDIA_STORAGE_PATH", str(tmp_path))
        buf = BytesIO()
        Image.new("RGB", (4, 4)).save(buf, format="PNG")
        params = VisualGenParams(
            image_reference=MediaRef(
                content=base64.b64encode(buf.getvalue()).decode(), format="base64"
            )
        )
        prepare_reference_slots(params)
        assert list(tmp_path.iterdir()) == []

    def test_wrong_modality_is_rejected(self, tmp_path):
        """Content is validated against the slot's modality before dispatch."""
        from tensorrt_llm.visual_gen import MediaRef, VisualGenParams
        from tensorrt_llm.visual_gen.media_refs import prepare_reference_slots

        buf = BytesIO()
        Image.new("RGB", (2, 2)).save(buf, format="PNG")
        params = VisualGenParams(
            image_reference=MediaRef(content=buf.getvalue(), format="bytes"),
            video_reference=MediaRef(content=buf.getvalue(), format="bytes"),
        )
        with pytest.raises(ValueError, match="video_reference is not a recognized"):
            prepare_reference_slots(params)

    def test_missing_path_is_a_client_error(self, tmp_path):
        from tensorrt_llm.visual_gen import MediaRef, VisualGenParams
        from tensorrt_llm.visual_gen.media_refs import prepare_reference_slots

        params = VisualGenParams(
            image_reference=MediaRef(content=str(tmp_path / "nope.png"), format="path")
        )
        with pytest.raises(ValueError, match="could not be read"):
            prepare_reference_slots(params)


class TestResolveReference:
    """``_resolve_reference`` dispatches on the declared format, never on the value."""

    @staticmethod
    def _png() -> bytes:
        buf = BytesIO()
        Image.new("RGB", (4, 4), (1, 2, 3)).save(buf, format="PNG")
        return buf.getvalue()

    def test_every_format_resolves_to_the_same_bytes(self, tmp_path, monkeypatch):
        from tensorrt_llm.visual_gen.media_refs import _resolve_reference

        png = self._png()
        src = tmp_path / "ref.png"
        src.write_bytes(png)
        b64 = base64.b64encode(png).decode()

        class _FakeResp:
            def __init__(self, content):
                self.content = content

        monkeypatch.setattr(
            "tensorrt_llm.visual_gen.media_refs._safe_request_get",
            lambda url, **kwargs: _FakeResp(png),
        )
        assert _resolve_reference(str(src), "path") == png
        assert _resolve_reference(src.as_uri(), "path") == png
        assert _resolve_reference("https://example.com/ref.png", "url") == png
        assert _resolve_reference(b64, "base64") == png
        assert _resolve_reference(f"data:image/png;base64,{b64}", "base64") == png
        assert _resolve_reference(png, "bytes") == png

    def test_missing_path_is_a_read_error(self, tmp_path):
        from tensorrt_llm.visual_gen.media_refs import _resolve_reference

        with pytest.raises(ValueError, match="reference file could not be read"):
            _resolve_reference(str(tmp_path / "absent.png"), "path")

    def test_base64_does_not_fall_back_to_a_disk_read(self, tmp_path):
        from tensorrt_llm.visual_gen.media_refs import _resolve_reference

        src = tmp_path / "ref.png"
        src.write_bytes(self._png())
        with pytest.raises(ValueError, match="not valid base64"):
            _resolve_reference(str(src), "base64")

    def test_url_fetch_failure_is_a_client_error(self, monkeypatch):
        from tensorrt_llm.visual_gen.media_refs import _resolve_reference

        def _blocked(url, **kwargs):
            raise RuntimeError("URL resolves to a non-public address (10.0.0.1)")

        monkeypatch.setattr("tensorrt_llm.visual_gen.media_refs._safe_request_get", _blocked)
        with pytest.raises(ValueError, match="reference URL could not be fetched"):
            _resolve_reference("http://10.0.0.1/a.png", "url")

    def test_non_base64_data_uri_is_rejected(self):
        from tensorrt_llm.visual_gen.media_refs import _resolve_reference

        with pytest.raises(ValueError, match="only base64 data: URIs"):
            _resolve_reference("data:image/png,%89PNG", "base64")
        with pytest.raises(ValueError, match="data: URI is malformed"):
            _resolve_reference("data:image/png;base64", "base64")

    def test_content_type_must_match_the_declared_format(self):
        from tensorrt_llm.visual_gen.media_refs import _resolve_reference

        with pytest.raises(ValueError, match="requires bytes content"):
            _resolve_reference("not bytes", "bytes")
        for content_format in ("path", "url", "base64"):
            with pytest.raises(ValueError, match="requires string content"):
                _resolve_reference(b"raw bytes", content_format)

    def test_unknown_format_is_rejected(self):
        from tensorrt_llm.visual_gen.media_refs import _resolve_reference

        with pytest.raises(ValueError, match="unsupported reference format"):
            _resolve_reference("a.png", "filepath")


# =============================================================================
# reference transport (coordinator -> rank0)
# =============================================================================


class TestReferenceHandleTransport:
    """References cross the coordinator -> rank0 hop as shared-memory handles.

    The payload must survive the round trip byte-identically and must not ride
    the request pickle, which is the whole point of the handle.
    """

    @staticmethod
    def _request(*payloads: bytes):
        from tensorrt_llm._torch.visual_gen import DiffusionRequest
        from tensorrt_llm.visual_gen import MediaRef, VisualGenParams

        params = VisualGenParams(
            image_reference=[MediaRef(content=p, format="bytes") for p in payloads]
        )
        return DiffusionRequest(request_id=1, prompt=["x"], params=params)

    def test_round_trip_is_byte_identical(self):
        payloads = (b"\x89PNG\r\n\x1a\n" + os.urandom(4096), os.urandom(1024))
        req = self._request(*payloads)

        req.refs_to_handles()
        req.refs_to_bytes()

        assert tuple(r.content for r in req.params.image_reference) == payloads
        assert all(r.format == "bytes" for r in req.params.image_reference)

    def test_payload_leaves_the_request_pickle(self):
        """The handle is the transport, so the bytes must not also be pickled —
        otherwise the hop still pays for a full copy of every reference."""
        payload = os.urandom(256 * 1024)
        req = self._request(payload)
        before = len(pickle.dumps(req))

        req.refs_to_handles()
        after = len(pickle.dumps(req))

        assert after < before - len(payload) // 2
        assert payload not in pickle.dumps(req)

    def test_survives_a_real_pickle_round_trip(self):
        """A handle is only useful if it still resolves after being serialized
        and rebuilt, which is what the IPC queue does to it."""
        payload = os.urandom(8192)
        req = self._request(payload)
        req.refs_to_handles()

        received = pickle.loads(pickle.dumps(req))
        received.refs_to_bytes()

        assert received.params.image_reference[0].content == payload

    def test_no_references_costs_nothing(self):
        """T2V/T2I requests carry no handle, so the hop is untouched for them."""
        from tensorrt_llm._torch.visual_gen import DiffusionRequest
        from tensorrt_llm.visual_gen import VisualGenParams

        req = DiffusionRequest(request_id=1, prompt=["x"], params=VisualGenParams())
        req.refs_to_handles()
        assert req.ref_handles is None

    def test_restore_is_idempotent(self):
        """rank0 restores unconditionally; a second call must not re-consume a
        handle that has already been resolved."""
        payload = os.urandom(2048)
        req = self._request(payload)
        req.refs_to_handles()
        req.refs_to_bytes()
        req.refs_to_bytes()
        assert req.params.image_reference[0].content == payload


class TestReferenceBroadcastSplit:
    """Reference payloads leave the object before the rank0 -> N-rank hop.

    ``broadcast_object_list`` serializes whatever it is handed into a tensor,
    so leaving the bytes inside the request would copy every reference byte
    before the collective even starts.
    """

    @staticmethod
    def _request(*payloads: bytes):
        from tensorrt_llm._torch.visual_gen import DiffusionRequest
        from tensorrt_llm.visual_gen import MediaRef, VisualGenParams

        params = VisualGenParams(
            image_reference=[MediaRef(content=p, format="bytes") for p in payloads],
            video_reference=MediaRef(content=b"\x00\x00\x00\x18ftypmp42", format="bytes"),
        )
        return DiffusionRequest(request_id=1, prompt=["x"], params=params)

    def test_detach_empties_the_object_and_records_sizes(self):
        payloads = (os.urandom(4096), os.urandom(64))
        req = self._request(*payloads)

        detached = req.refs_detach()

        assert detached[:2] == list(payloads)
        assert req.ref_sizes == [len(p) for p in detached]
        assert all(r.content == b"" for r in req.params.image_reference)
        assert payloads[0] not in pickle.dumps(req)

    def test_attach_restores_every_slot_in_order(self):
        payloads = (os.urandom(2048), os.urandom(128))
        req = self._request(*payloads)

        detached = req.refs_detach()
        req.refs_attach(detached)

        assert tuple(r.content for r in req.params.image_reference) == payloads
        assert req.params.video_reference[0].content == b"\x00\x00\x00\x18ftypmp42"
        assert req.ref_sizes is None

    def test_sizes_let_a_peer_size_its_buffers(self):
        """Non-source ranks allocate from ``ref_sizes`` alone, so it has to
        survive the object hop and match the payloads exactly."""
        payloads = (os.urandom(1024), os.urandom(7))
        req = self._request(*payloads)
        req.refs_detach()

        peer = pickle.loads(pickle.dumps(req))
        assert peer.ref_sizes == req.ref_sizes
        assert peer.ref_sizes[:2] == [1024, 7]

    @staticmethod
    def _blocks_of(req):
        """The shared-memory files this request's handles point at.

        Named per handle rather than scanned out of /dev/shm, which is global
        to the machine and picks up unrelated processes.
        """
        return [
            Path("/dev/shm") / base64.b64decode(e["handle"]["storage_handle"]).decode().lstrip("/")
            for e in req.ref_handles or []
        ]

    def test_dropped_request_releases_its_shared_memory(self):
        """An unconsumed handle keeps its block mapped until the process exits,
        so a request that fails on its way to rank0 has to consume its own
        handles on the way out."""
        import gc

        req = self._request(os.urandom(1024 * 1024))
        req.refs_to_handles()
        blocks = self._blocks_of(req)
        assert blocks and all(b.exists() for b in blocks)

        # What the sender thread does when the request never reaches rank0.
        req.refs_to_bytes()
        del req
        gc.collect()
        assert not any(b.exists() for b in blocks)

    def test_attach_rejects_a_count_mismatch(self):
        """Peers size their collectives from ``ref_sizes``, so a payload list
        that does not match the reference count is a bug worth raising on
        rather than silently leaving references empty."""
        req = self._request(os.urandom(64), os.urandom(64))
        detached = req.refs_detach()

        with pytest.raises(ValueError, match="expected 3 reference payloads, got 2"):
            req.refs_attach(detached[:2])

    def test_partial_handle_failure_stays_reclaimable(self):
        """If a later reference cannot reach shared memory, the blocks already
        taken must still be reachable, or nothing can free them."""
        import gc

        req = self._request(os.urandom(256 * 1024), os.urandom(256 * 1024))
        real = torch.frombuffer
        calls = {"n": 0}

        def flaky(buffer, **kwargs):
            calls["n"] += 1
            if calls["n"] == 3:  # the third of this request's three references
                raise RuntimeError("shared memory exhausted")
            return real(buffer, **kwargs)

        with mock.patch.object(torch, "frombuffer", flaky):
            with pytest.raises(RuntimeError, match="shared memory exhausted"):
                req.refs_to_handles()

        blocks = self._blocks_of(req)
        assert len(blocks) == 2, "handles taken before the failure must stay reachable"
        assert all(b.exists() for b in blocks)

        req.refs_to_bytes()
        del req
        gc.collect()
        assert not any(b.exists() for b in blocks)

    def test_a_request_that_never_ships_hands_its_blocks_back(self):
        """The blocks staying reachable is only half of it; the call site is
        what has to hand them back when the request never reaches rank0."""
        import gc
        import itertools
        from types import SimpleNamespace

        from tensorrt_llm.visual_gen import VisualGen, VisualGenParams
        from tensorrt_llm.visual_gen.params import MediaRef

        taken = {}
        blocks_of = self._blocks_of

        class _DeadExecutor:
            default_generation_params = {}
            extra_param_specs = {}
            ref_slot_specs = None

            def enqueue_requests(self, requests):
                # Read the handles while they still exist, then fail the way a
                # dead worker queue would.
                taken["blocks"] = blocks_of(requests[0])
                raise RuntimeError("worker queue is gone")

        caller = SimpleNamespace(
            _req_counter=itertools.count(),
            default_params=VisualGenParams(),
            executor=_DeadExecutor(),
        )
        buf = BytesIO()
        # A real PNG: the engine choke point content-checks before the handles
        # are minted, so random bytes never get far enough to take a block.
        Image.new("RGB", (256, 256)).save(buf, format="PNG")
        params = VisualGenParams(image_reference=[MediaRef(content=buf.getvalue(), format="bytes")])

        with pytest.raises(RuntimeError, match="worker queue is gone"):
            VisualGen.generate_async(caller, "x", params)

        assert taken["blocks"], "the request must have taken a block to begin with"
        gc.collect()
        assert not any(b.exists() for b in taken["blocks"])


class TestSafeLocalFileRead:
    """A ``path`` reference bounds what an unlucky path can cost.

    A remote caller naming the path is the case worth defending: reading a
    character device or a FIFO never returns, so an unbounded read is a denial
    of service rather than a bad request.
    """

    @staticmethod
    def _png(tmp_path):
        buffer = BytesIO()
        Image.new("RGB", (8, 8)).save(buffer, format="PNG")
        target = tmp_path / "ref.png"
        target.write_bytes(buffer.getvalue())
        return target

    def test_a_regular_file_reads(self, tmp_path):
        from tensorrt_llm.visual_gen.media_refs import _safe_read_local_file

        target = self._png(tmp_path)

        assert _safe_read_local_file(str(target)) == target.read_bytes()
        assert _safe_read_local_file(target.as_uri()) == target.read_bytes()

    def test_a_symlink_to_a_regular_file_reads(self, tmp_path):
        from tensorrt_llm.visual_gen.media_refs import _safe_read_local_file

        target = self._png(tmp_path)
        link = tmp_path / "link.png"
        link.symlink_to(target)

        assert _safe_read_local_file(str(link)) == target.read_bytes()

    @pytest.mark.parametrize("device", ["/dev/zero", "/dev/null"])
    def test_a_character_device_is_refused(self, device):
        """The unbounded read: `/dev/zero` never reaches EOF."""
        from tensorrt_llm.visual_gen.media_refs import _safe_read_local_file

        with pytest.raises(ValueError, match="not a regular file"):
            _safe_read_local_file(device)

    def test_a_symlink_to_a_device_is_refused(self, tmp_path):
        """``stat`` follows the link, so the check sees what will be read."""
        from tensorrt_llm.visual_gen.media_refs import _safe_read_local_file

        link = tmp_path / "innocent.png"
        link.symlink_to("/dev/zero")

        with pytest.raises(ValueError, match="not a regular file"):
            _safe_read_local_file(str(link))

    def test_a_fifo_is_refused(self, tmp_path):
        """Reading a FIFO blocks rather than returning, so size caps cannot help."""
        import os

        from tensorrt_llm.visual_gen.media_refs import _safe_read_local_file

        fifo = tmp_path / "pipe"
        os.mkfifo(fifo)

        with pytest.raises(ValueError, match="not a regular file"):
            _safe_read_local_file(str(fifo))

    def test_a_directory_is_refused(self, tmp_path):
        from tensorrt_llm.visual_gen.media_refs import _safe_read_local_file

        with pytest.raises(ValueError, match="not a regular file"):
            _safe_read_local_file(str(tmp_path))

    def test_a_missing_file_is_a_client_error(self, tmp_path):
        from tensorrt_llm.visual_gen.media_refs import _safe_read_local_file

        with pytest.raises(ValueError, match="could not be read"):
            _safe_read_local_file(str(tmp_path / "nope.png"))


class TestLocalMediaPathCanBeDisallowed:
    """``format='path'`` reads server-side files, so a deployment can refuse it.

    Allowed by default: a co-located client naming a shared path is a real
    setup, and the code cannot tell that deployment from an untrusted one.
    """

    @staticmethod
    def _request(fmt="path", content="/tmp/ref.png"):
        return VideoGenerationRequest(
            prompt="x", image_reference={"content": content, "format": fmt}
        )

    def test_path_is_accepted_by_default(self, monkeypatch):
        monkeypatch.delenv("TRTLLM_DISALLOW_LOCAL_MEDIA_PATH", raising=False)
        params = parse_visual_gen_params(self._request(), _StubVisualGen())

        assert params.image_reference[0].format == "path"

    def test_path_is_refused_when_disallowed(self, monkeypatch):
        monkeypatch.setenv("TRTLLM_DISALLOW_LOCAL_MEDIA_PATH", "1")

        with pytest.raises(ValueError, match="is disallowed on this server"):
            parse_visual_gen_params(self._request(), _StubVisualGen())

    def test_disallowing_path_leaves_the_other_formats_alone(self, monkeypatch):
        """The gate is about reading server-side files, not about references."""
        monkeypatch.setenv("TRTLLM_DISALLOW_LOCAL_MEDIA_PATH", "1")

        params = parse_visual_gen_params(
            self._request(fmt="base64", content="aGk="), _StubVisualGen()
        )

        assert params.image_reference[0].format == "base64"

    def test_an_unrecognized_value_warns_and_stays_allowed(self, monkeypatch):
        """Silently reading a typo as "1" would break working deployments, and
        silently reading it as "0" would leave one that believes it is locked
        down wide open. Neither is safe to do quietly."""
        monkeypatch.setenv("TRTLLM_DISALLOW_LOCAL_MEDIA_PATH", "true")
        warnings: list[str] = []
        monkeypatch.setattr(visual_gen_utils.logger, "warning", warnings.append)

        params = parse_visual_gen_params(self._request(), _StubVisualGen())

        assert params.image_reference[0].format == "path"
        assert any("TRTLLM_DISALLOW_LOCAL_MEDIA_PATH" in w for w in warnings)

    def test_the_deprecated_field_is_gated_too(self, monkeypatch):
        monkeypatch.setenv("TRTLLM_DISALLOW_LOCAL_MEDIA_PATH", "1")
        request = VideoGenerationRequest(
            prompt="x", input_reference="/tmp/ref.png", input_reference_format="path"
        )

        with pytest.raises(ValueError, match="is disallowed on this server"):
            parse_visual_gen_params(request, _StubVisualGen())
