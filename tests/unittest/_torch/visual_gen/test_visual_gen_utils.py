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
from pathlib import Path
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
from tensorrt_llm.visual_gen.media_refs import cleanup_reference_files

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
# reference materialization
# =============================================================================


class TestInputReferenceMaterialization:
    def test_base64_image_reference_written_to_disk(self, tmp_path):
        generator = _StubVisualGen()
        img = Image.new("RGB", (4, 4), (10, 20, 30))
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        request = VideoGenerationRequest(prompt="x", image_reference=b64)
        params = parse_visual_gen_params(
            request, "vid-1", generator, media_storage_path=str(tmp_path)
        )
        assert len(params.image_reference) == 1
        ref_path = params.image_reference[0].content
        assert str(ref_path).endswith("vid-1_image_ref_0")
        # The decoded image is identical to what we passed in.
        with open(ref_path, "rb") as f:
            decoded = Image.open(f).convert("RGB")
            assert decoded.size == (4, 4)

    def test_image_reference_role_and_list(self, tmp_path):
        generator = _StubVisualGen()
        buf = BytesIO()
        Image.new("RGB", (4, 4)).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        request = VideoGenerationRequest(
            prompt="x", image_reference=[b64, {"content": b64, "role": "last_frame"}]
        )
        params = parse_visual_gen_params(
            request, "vid-r", generator, media_storage_path=str(tmp_path)
        )
        assert [r.role for r in params.image_reference] == [None, "last_frame"]
        paths = [r.content for r in params.image_reference]
        assert len(set(paths)) == 2  # unique file per index

    def test_missing_media_storage_path_raises(self):
        generator = _StubVisualGen()
        img = Image.new("RGB", (2, 2))
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        request = VideoGenerationRequest(prompt="x", image_reference=b64)
        with pytest.raises(ValueError, match="media_storage_path"):
            parse_visual_gen_params(request, "vid-2", generator, media_storage_path=None)

    _TEST_DATA = Path(__file__).parent / "test_data"

    @staticmethod
    def _mp4_bytes() -> bytes:
        """9-frame H.264-in-MP4 fixture (provenance: test_data/README.md)."""
        return (
            TestInputReferenceMaterialization._TEST_DATA / "cosmos3_v2v_ref_9f_bframes.mp4"
        ).read_bytes()

    @staticmethod
    def _avi_bytes() -> bytes:
        """Same 9 frames as H.264-in-AVI (provenance: test_data/README.md)."""
        return (
            TestInputReferenceMaterialization._TEST_DATA / "cosmos3_v2v_ref_9f_bframes.avi"
        ).read_bytes()

    def test_multipart_avi_video_reference_written_to_disk(self, tmp_path):
        # The AVI container survives the boundary and is persisted as untouched
        # encoded bytes for the worker to demux.
        generator = _StubVisualGen()
        payload = self._avi_bytes()
        upload = UploadFile(file=BytesIO(payload), filename="clip.avi")
        request = VideoGenerationRequest(prompt="x", video_reference=upload)
        params = parse_visual_gen_params(
            request, "vid-avi", generator, media_storage_path=str(tmp_path)
        )
        assert params.image_reference is None
        assert Path(params.video_reference[0].content).read_bytes() == payload

    def test_multipart_mp4_video_reference_written_to_disk(self, tmp_path):
        generator = _StubVisualGen()
        payload = self._mp4_bytes()
        upload = UploadFile(file=BytesIO(payload), filename="clip.mp4")
        request = VideoGenerationRequest(prompt="x", video_reference=upload)
        params = parse_visual_gen_params(
            request, "vid-3", generator, media_storage_path=str(tmp_path)
        )
        # Encoded payload is persisted byte-identical — the boundary never
        # decodes video; the worker demuxes/NVDEC-decodes the conditioning
        # window from the stored file.
        assert params.image_reference is None
        vpath = params.video_reference[0].content
        assert str(vpath).endswith("vid-3_video_ref_0")
        assert Path(vpath).read_bytes() == payload

    def test_video_reference_needs_media_storage(self):
        # Video references now persist to disk (the worker reads the path), so
        # a storage path is required just like image references.
        generator = _StubVisualGen()
        b64 = base64.b64encode(self._mp4_bytes()).decode()
        request = VideoGenerationRequest(prompt="x", video_reference=b64)
        with pytest.raises(ValueError, match="media_storage_path"):
            parse_visual_gen_params(request, "vid-9", generator, media_storage_path=None)

    def test_deprecated_input_reference_routes_by_sniff(self, tmp_path):
        # The deprecated single input_reference is sniff-routed to the typed slot.
        generator = _StubVisualGen()
        buf = BytesIO()
        Image.new("RGB", (4, 4)).save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        vid_b64 = base64.b64encode(self._mp4_bytes()).decode()

        p = parse_visual_gen_params(
            VideoGenerationRequest(prompt="x", input_reference=img_b64),
            "vid-i",
            generator,
            media_storage_path=str(tmp_path),
        )
        assert len(p.image_reference) == 1 and p.video_reference is None

        p = parse_visual_gen_params(
            VideoGenerationRequest(prompt="x", input_reference=vid_b64),
            "vid-v",
            generator,
            media_storage_path=str(tmp_path),
        )
        assert len(p.video_reference) == 1 and p.image_reference is None

    def test_input_reference_ignored_when_typed_reference_set(self, tmp_path):
        # A typed reference takes precedence; the deprecated input_reference is dropped.
        generator = _StubVisualGen()
        buf = BytesIO()
        Image.new("RGB", (4, 4)).save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        vid_b64 = base64.b64encode(self._mp4_bytes()).decode()
        p = parse_visual_gen_params(
            VideoGenerationRequest(prompt="x", image_reference=img_b64, input_reference=vid_b64),
            "vid-x",
            generator,
            media_storage_path=str(tmp_path),
        )
        assert len(p.image_reference) == 1
        assert p.video_reference is None  # input_reference video dropped

    def test_base64_video_reference_written_to_disk(self, tmp_path):
        # The JSON/base64 path carries video even though it has no content-type
        # or filename; modality is declared by the field name.
        generator = _StubVisualGen()
        payload = self._mp4_bytes()
        b64 = base64.b64encode(payload).decode()
        request = VideoGenerationRequest(prompt="x", video_reference=b64)
        params = parse_visual_gen_params(
            request, "vid-4", generator, media_storage_path=str(tmp_path)
        )
        assert params.image_reference is None
        assert Path(params.video_reference[0].content).read_bytes() == payload

    def test_video_reference_survives_real_specs(self, tmp_path):
        """With the real cosmos3 specs loaded, the encoded payload is persisted
        byte-identical — the boundary never transforms video content; the
        worker decodes the conditioning window."""
        from tensorrt_llm._torch.visual_gen.models.cosmos3.defaults import COSMOS3_EXTRA_SPECS

        generator = _StubVisualGen(extra_param_specs=COSMOS3_EXTRA_SPECS)
        payload = self._mp4_bytes()
        b64 = base64.b64encode(payload).decode()
        request = VideoGenerationRequest(prompt="x", video_reference=b64)
        params = parse_visual_gen_params(
            request, "vid-10", generator, media_storage_path=str(tmp_path)
        )
        assert Path(params.video_reference[0].content).read_bytes() == payload

    def test_multipart_image_reference_written_to_disk(self, tmp_path):
        # JPEG upload routed by field name to image_reference. The stored file
        # has no type-suffix (PIL identifies by content, not name).
        generator = _StubVisualGen()
        img = Image.new("RGB", (4, 4), (10, 20, 30))
        buf = BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        upload = UploadFile(file=buf, filename="ref.jpg")
        request = VideoGenerationRequest(prompt="x", image_reference=upload)
        params = parse_visual_gen_params(
            request, "vid-5", generator, media_storage_path=str(tmp_path)
        )
        assert params.extra_params is None
        assert str(params.image_reference[0].content).endswith("vid-5_image_ref_0")

    def test_wrong_modality_content_raises(self, tmp_path):
        # The field name declares modality; mismatched content is a client error.
        generator = _StubVisualGen()
        buf = BytesIO()
        Image.new("RGB", (2, 2)).save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        vid_b64 = base64.b64encode(self._mp4_bytes()).decode()
        with pytest.raises(ValueError, match="video_reference is not a recognized"):
            parse_visual_gen_params(
                VideoGenerationRequest(prompt="x", video_reference=img_b64),
                "vid-m1",
                generator,
                media_storage_path=str(tmp_path),
            )
        with pytest.raises(ValueError, match="image_reference is not a recognized image"):
            parse_visual_gen_params(
                VideoGenerationRequest(prompt="x", image_reference=vid_b64),
                "vid-m2",
                generator,
                media_storage_path=str(tmp_path),
            )
        assert list(tmp_path.iterdir()) == []

    def test_undecodable_image_reference_raises_and_cleans_up(self, tmp_path):
        generator = _StubVisualGen()
        b64 = base64.b64encode(b"neither an image nor a video").decode()
        request = VideoGenerationRequest(prompt="x", image_reference=b64)
        with pytest.raises(ValueError, match="not a recognized image"):
            parse_visual_gen_params(request, "vid-6", generator, media_storage_path=str(tmp_path))
        # Classification runs on the bytes; rejected content never touches disk.
        assert list(tmp_path.iterdir()) == []

    def test_malformed_base64_reference_raises_and_cleans_up(self, tmp_path):
        generator = _StubVisualGen()
        # "ABC" survives the lenient alphabet filter but has an invalid
        # length, so b64decode raises.
        request = VideoGenerationRequest(prompt="x", image_reference="ABC")
        with pytest.raises(ValueError, match="not valid base64"):
            parse_visual_gen_params(request, "vid-7", generator, media_storage_path=str(tmp_path))
        assert list(tmp_path.iterdir()) == []

    def test_upload_stream_failure_cleans_up_tmp(self, tmp_path):
        generator = _StubVisualGen()

        class _BrokenStream:
            def read(self, *args, **kwargs):
                raise OSError("client went away")

        upload = UploadFile(file=_BrokenStream(), filename="clip.mp4")
        request = VideoGenerationRequest(prompt="x", video_reference=upload)
        # I/O failures keep their server-error semantics (no 400 masking) …
        with pytest.raises(OSError, match="client went away"):
            parse_visual_gen_params(request, "vid-8", generator, media_storage_path=str(tmp_path))
        # … and the payload read fails before any file is written, so nothing leaks.
        assert list(tmp_path.iterdir()) == []

    def test_multi_reference_partial_failure_cleans_up(self, tmp_path):
        # A later item's rejection removes the files earlier items already wrote,
        # so a rejected multi-reference request leaves nothing on disk.
        generator = _StubVisualGen()
        buf = BytesIO()
        Image.new("RGB", (4, 4)).save(buf, format="PNG")
        good = base64.b64encode(buf.getvalue()).decode()
        bad = base64.b64encode(b"neither an image nor a video").decode()
        request = VideoGenerationRequest(prompt="x", image_reference=[good, bad])
        with pytest.raises(ValueError, match="not a recognized image"):
            parse_visual_gen_params(request, "vid-11", generator, media_storage_path=str(tmp_path))
        assert list(tmp_path.iterdir()) == []

    def test_file_uri_image_reference_read_and_materialized(self, tmp_path):
        # A file:// reference is read from local disk and persisted like any other.
        generator = _StubVisualGen()
        src = tmp_path / "ref.png"
        Image.new("RGB", (4, 4), (7, 8, 9)).save(src, format="PNG")
        store = tmp_path / "store"
        store.mkdir()
        request = VideoGenerationRequest(prompt="x", image_reference=src.as_uri())
        params = parse_visual_gen_params(
            request, "vid-file", generator, media_storage_path=str(store)
        )
        assert Path(params.image_reference[0].content).read_bytes() == src.read_bytes()

    def test_bare_path_image_reference_read_and_materialized(self, tmp_path):
        # A bare local path (no file:// scheme) is read from disk after the
        # base64 decode attempt fails.
        generator = _StubVisualGen()
        src = tmp_path / "ref.png"
        Image.new("RGB", (4, 4), (11, 22, 33)).save(src, format="PNG")
        store = tmp_path / "store"
        store.mkdir()
        request = VideoGenerationRequest(prompt="x", image_reference=str(src))
        params = parse_visual_gen_params(
            request, "vid-bare", generator, media_storage_path=str(store)
        )
        assert Path(params.image_reference[0].content).read_bytes() == src.read_bytes()

    def test_http_url_image_reference_fetched_and_materialized(self, tmp_path, monkeypatch):
        # An http(s) reference is fetched through the guarded loader, then stored.
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
        request = VideoGenerationRequest(prompt="x", image_reference="https://example.com/a.png")
        params = parse_visual_gen_params(
            request, "vid-url", generator, media_storage_path=str(tmp_path)
        )
        assert Path(params.image_reference[0].content).read_bytes() == png

    def test_http_url_fetch_failure_is_client_error(self, tmp_path, monkeypatch):
        # A blocked/failed fetch (e.g. SSRF guard) is a client 400, not a 500,
        # and leaves nothing on disk.
        generator = _StubVisualGen()

        def _blocked(url, **kwargs):
            raise RuntimeError("URL resolves to a non-public address (10.0.0.1)")

        monkeypatch.setattr("tensorrt_llm.visual_gen.media_refs._safe_request_get", _blocked)
        request = VideoGenerationRequest(prompt="x", image_reference="http://10.0.0.1/a.png")
        with pytest.raises(ValueError, match="reference URL could not be fetched"):
            parse_visual_gen_params(
                request, "vid-ssrf", generator, media_storage_path=str(tmp_path)
            )
        assert list(tmp_path.iterdir()) == []

    def test_missing_file_uri_is_client_error(self, tmp_path):
        # A file:// path that does not exist is a client 400, not a server 500.
        generator = _StubVisualGen()
        missing = (tmp_path / "does_not_exist.png").as_uri()
        request = VideoGenerationRequest(prompt="x", image_reference=missing)
        with pytest.raises(ValueError, match="reference file could not be read"):
            parse_visual_gen_params(request, "vid-nf", generator, media_storage_path=str(tmp_path))
        assert list(tmp_path.iterdir()) == []


class TestMediaBytesProbes:
    """The in-memory signature probes the serve boundary routes on."""

    def test_sniff_media_kind(self):
        from tensorrt_llm.inputs.media_io import sniff_media_kind

        png = BytesIO()
        Image.new("RGB", (2, 2)).save(png, format="PNG")
        jpg = BytesIO()
        Image.new("RGB", (2, 2)).save(jpg, format="JPEG")
        assert sniff_media_kind(png.getvalue()) == "image"
        assert sniff_media_kind(jpg.getvalue()) == "image"
        assert sniff_media_kind(TestInputReferenceMaterialization._mp4_bytes()) == "video"
        assert sniff_media_kind(TestInputReferenceMaterialization._avi_bytes()) == "video"
        assert sniff_media_kind(b"plain text, not media") is None
        assert sniff_media_kind(b"") is None
        # RIFF alone is not AVI (e.g. WAV audio is RIFF too).
        assert sniff_media_kind(b"RIFF\x00\x00\x00\x00WAVEfmt ") is None

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
            prompt="x", image_reference=base64.b64encode(heic).decode()
        )
        with pytest.raises(ValueError, match="HEIF/AVIF"):
            parse_visual_gen_params(request, "vid-heic", generator, media_storage_path=None)

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
            prompt="x", image_reference=base64.b64encode(truncated).decode()
        )
        params = parse_visual_gen_params(
            request, "vid-12", generator, media_storage_path=str(tmp_path)
        )
        assert Path(params.image_reference[0].content).read_bytes() == truncated


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
        params = parse_visual_gen_params(request, "id-b64", self._generator())
        assert params.extra_params["video"] == b"\x00mp4"

    def test_nested_control_reaches_the_pipeline_as_bytes(self):
        request = VideoGenerationRequest(
            prompt="storm",
            extra_params={"edge": {"control": base64.b64encode(b"\x00ctrl").decode()}},
        )
        params = parse_visual_gen_params(request, "id-nested", self._generator())
        assert params.extra_params["edge"]["control"] == b"\x00ctrl"

    def test_non_media_params_are_untouched(self):
        request = VideoGenerationRequest(
            prompt="storm", extra_params={"resolution": "720", "edge": True}
        )
        params = parse_visual_gen_params(request, "id-plain", self._generator())
        assert params.extra_params["resolution"] == "720"
        assert params.extra_params["edge"] is True

    def test_malformed_base64_is_a_client_error(self):
        request = VideoGenerationRequest(prompt="storm", extra_params={"video": "not!b64!"})
        with pytest.raises(ValueError, match="not valid base64"):
            parse_visual_gen_params(request, "id-bad", self._generator())


class TestCleanupReferenceFiles:
    """The reference-file reclaim helper keyed on the request id prefix."""

    def test_removes_only_this_request_ref_files(self, tmp_path):
        vid = "video_abc123"
        (tmp_path / f"{vid}_image_ref_0").write_bytes(b"a")
        (tmp_path / f"{vid}_video_ref_1").write_bytes(b"b")
        (tmp_path / f"{vid}_input_ref").write_bytes(b"c")  # deprecated alias
        (tmp_path / f"{vid}_0.mp4").write_bytes(b"out")  # output — keep
        (tmp_path / "video_other_image_ref_0").write_bytes(b"d")  # other id — keep
        cleanup_reference_files(str(tmp_path), vid)
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            f"{vid}_0.mp4",
            "video_other_image_ref_0",
        ]

    def test_none_storage_is_noop(self):
        cleanup_reference_files(None, "video_x")  # no raise

    def test_missing_files_are_ignored(self, tmp_path):
        cleanup_reference_files(str(tmp_path), "video_absent")  # no raise
