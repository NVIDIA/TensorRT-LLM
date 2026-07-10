"""trtllm-serve visual_gen endpoints tests.

Tests all endpoints registered for the VISUAL_GEN server role
in OpenAIServer.register_visual_gen_routes():

    POST /v1/images/generations
    POST /v1/images/edits
    POST /v1/videos/generations   (sync)
    POST /v1/videos               (async)
    GET  /v1/videos               (list)
    GET  /v1/videos/{video_id}    (metadata)
    GET  /v1/videos/{video_id}/content  (download)
    DELETE /v1/videos/{video_id}  (delete)
"""

import asyncio
import base64
import os
from io import BytesIO
from typing import Optional
from unittest.mock import patch

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from tensorrt_llm.serve.openai_protocol import VideoJob
from tensorrt_llm.serve.openai_server import _normalize_image_output
from tensorrt_llm.serve.visual_gen_metrics import SERVER_TIMING_HEADER
from tensorrt_llm.serve.visual_gen_utils import VIDEO_STORE
from tensorrt_llm.visual_gen.output import VisualGenMetrics, VisualGenOutput

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_llm_envelope(
    body: dict,
    *,
    code: int,
    err_type: str = "BadRequestError",
    message_contains: Optional[str] = None,
) -> None:
    """Assert *body* is the visual-gen LLM-style error envelope.

    The envelope's wire shape is ``{"object": "error", "message": str,
    "type": str, "code": int}`` with optional ``"param": str | None``.
    ``object`` and ``param`` are returned by Pydantic's
    ``ErrorResponse.model_dump`` and are stable across all visual-gen
    error paths.
    """
    assert set(body.keys()) == {"object", "message", "type", "param", "code"}, body
    assert body["object"] == "error"
    assert body["type"] == err_type
    assert body["code"] == code
    assert isinstance(body["message"], str) and body["message"]
    if message_contains is not None:
        assert message_contains in body["message"], body["message"]


def _make_dummy_image_tensor(height: int = 64, width: int = 64) -> torch.Tensor:
    """Create a small dummy uint8 image tensor (H, W, C)."""
    return torch.randint(0, 256, (height, width, 3), dtype=torch.uint8)


def _make_dummy_video_tensor(
    num_frames: int = 4, height: int = 64, width: int = 64
) -> torch.Tensor:
    """Create a small dummy uint8 video tensor (T, H, W, C)."""
    return torch.randint(0, 256, (num_frames, height, width, 3), dtype=torch.uint8)


def _make_dummy_audio_tensor(length: int = 16000) -> torch.Tensor:
    """Create a small dummy float32 audio tensor."""
    return torch.randn(1, length, dtype=torch.float32)


def _b64_white_png_1x1() -> str:
    """Return a base64-encoded 1x1 white PNG for image edit tests."""
    buf = BytesIO()
    Image.new("RGB", (1, 1), (255, 255, 255)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _run_async(coro):
    """Run an async coroutine in a new event loop (for test helpers)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_dummy_metrics() -> VisualGenMetrics:
    return VisualGenMetrics(
        generation=1.25,
        pre_denoise=0.125,
        denoise=0.75,
        post_denoise=0.375,
    )


def _assert_visual_gen_server_timing(headers) -> None:
    server_timing = headers[SERVER_TIMING_HEADER]
    assert "generation;dur=1250.000000" in server_timing
    assert "denoise;dur=750.000000" in server_timing


# ---------------------------------------------------------------------------
# Mock VisualGen
# ---------------------------------------------------------------------------


class MockVisualGen:
    """Lightweight stand-in for VisualGen that avoids GPU / model loading.

    When *batch_aware* is True (default), ``generate()`` and
    ``generate_async()`` inspect ``params.num_images_per_prompt`` and expand
    the stored single-item tensors into batched tensors ``(N, ...)`` so
    callers can test batch handling end-to-end.
    """

    def __init__(
        self,
        image_output: Optional[torch.Tensor] = None,
        video_output: Optional[torch.Tensor] = None,
        audio_output: Optional[torch.Tensor] = None,
        should_fail: bool = False,
        batch_aware: bool = True,
        validation_error: Optional[ValueError] = None,
    ):
        from types import SimpleNamespace

        self._image = image_output
        self._video = video_output
        self._audio = audio_output
        self._should_fail = should_fail
        self._batch_aware = batch_aware
        self._validation_error = validation_error
        self._healthy = True
        self._req_counter = 0
        # Captured arguments of the most recent generate / generate_async call,
        # used by tests to assert forwarded VisualGenParams fields.
        self.last_inputs = None
        self.last_params = None
        # Stand-in for the coordinator-side executor proxy. The async video
        # route reads ``default_generation_params`` / ``extra_param_specs``
        # directly off this attribute when running synchronous pre-flight
        # validation. ``default_generation_params`` declares the universal
        # fields the mock pipeline accepts so the validator doesn't
        # reject legitimate width/height/num_frames/... requests;
        # ``extra_param_specs`` lists a single known key so tests can
        # exercise both the accept-known and reject-unknown paths.
        from tensorrt_llm._torch.visual_gen.pipeline import ExtraParamSchema

        self.executor = SimpleNamespace(
            default_generation_params={
                "height": 64,
                "width": 64,
                "num_inference_steps": 20,
                "guidance_scale": 5.0,
                "max_sequence_length": 64,
                "num_frames": 8,
                "frame_rate": 8.0,
            },
            extra_param_specs={
                "stg_scale": ExtraParamSchema(type="float", default=1.0),
            },
        )

    def _maybe_batch(self, tensor, n):
        """Replicate a single tensor along a new leading batch dimension."""
        if tensor is None or n <= 1 or not self._batch_aware:
            return tensor
        return tensor.unsqueeze(0).expand(n, *tensor.shape).contiguous()

    # --- VisualGen interface ---

    def generate(self, inputs=None, params=None) -> VisualGenOutput:
        self.last_inputs = inputs
        self.last_params = params
        if self._validation_error is not None:
            raise self._validation_error
        if self._should_fail:
            raise RuntimeError("Generation intentionally failed")
        n = getattr(params, "num_images_per_prompt", 1) if params else 1
        return VisualGenOutput(
            request_id=self._next_request_id(),
            image=self._maybe_batch(self._image, n),
            video=self._maybe_batch(self._video, n),
            audio=self._audio,
            metrics=_make_dummy_metrics(),
        )

    def generate_async(self, inputs=None, params=None) -> "MockVisualGenResult":
        self.last_inputs = inputs
        self.last_params = params
        if self._validation_error is not None:
            raise self._validation_error
        n = getattr(params, "num_images_per_prompt", 1) if params else 1
        return MockVisualGenResult(
            request_id=self._next_request_id(),
            image=self._maybe_batch(self._image, n),
            video=self._maybe_batch(self._video, n),
            audio=self._audio,
            should_fail=self._should_fail,
        )

    def _next_request_id(self) -> int:
        rid = self._req_counter
        self._req_counter += 1
        return rid

    @property
    def default_params(self):
        """Stand-in for VisualGen.default_params — parse_visual_gen_params
        seeds request params from this, so it must return a fresh instance."""
        from tensorrt_llm.visual_gen import VisualGenParams

        return VisualGenParams()

    @property
    def extra_param_specs(self):
        """Stand-in for VisualGen.extra_param_specs — empty by default so
        every request ``extra_params`` key reaches the executor as
        ``unknown_extra_param`` (matches a pipeline with no model-specific
        knobs declared, like Flux or Wan 2.1)."""
        return {}

    @property
    def model(self):
        """Stand-in for VisualGen.model — used by warn-on-set logic."""
        return "test-model"

    def _check_health(self) -> bool:
        return self._healthy

    async def get_stats_async(self, timeout: int):
        return

    def shutdown(self):
        pass


class MockVisualGenResult:
    """Mock future-like result for generate_async.

    Mirrors the real :class:`VisualGenResult` surface enough for the server:
    ``__await__``, ``aresult``, and a sync ``result``. Resolves to a
    :class:`VisualGenOutput` (single-prompt path).
    """

    def __init__(
        self,
        request_id: int = 0,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        should_fail: bool = False,
    ):
        self.request_id = request_id
        self._image = image
        self._video = video
        self._audio = audio
        self._should_fail = should_fail

    def __await__(self):
        return self.aresult().__await__()

    async def aresult(self, timeout=None):
        if self._should_fail:
            raise RuntimeError("Async generation intentionally failed")
        return VisualGenOutput(
            request_id=self.request_id,
            image=self._image,
            video=self._video,
            audio=self._audio,
            metrics=_make_dummy_metrics(),
        )

    def result(self, timeout=None):
        if self._should_fail:
            raise RuntimeError("Async generation intentionally failed")
        return VisualGenOutput(
            request_id=self.request_id,
            image=self._image,
            video=self._video,
            audio=self._audio,
            metrics=_make_dummy_metrics(),
        )


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


def _create_server(generator: MockVisualGen, model_name: str = "test-model") -> TestClient:
    """Instantiate an OpenAIServer for VISUAL_GEN with a mocked generator.

    We patch the ``VisualGen`` name inside the ``openai_server`` module so that
    ``isinstance(generator, VisualGen)`` returns True for our mock.
    """
    from tensorrt_llm.llmapi.disagg_utils import ServerRole
    from tensorrt_llm.serve.openai_server import OpenAIServer

    with patch("tensorrt_llm.serve.openai_server.VisualGen", MockVisualGen):
        server = OpenAIServer(
            generator=generator,
            model=model_name,
            tool_parser=None,
            server_role=ServerRole.VISUAL_GEN,
            metadata_server_cfg=None,
        )
    client = TestClient(server.app)
    # Expose the mock so tests can assert captured generate() arguments.
    client.mock_gen = generator
    return client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def image_client(tmp_path):
    """TestClient backed by a MockVisualGen that produces images."""
    gen = MockVisualGen(image_output=_make_dummy_image_tensor())
    os.environ["TRTLLM_MEDIA_STORAGE_PATH"] = str(tmp_path)
    client = _create_server(gen)
    yield client
    os.environ.pop("TRTLLM_MEDIA_STORAGE_PATH", None)


@pytest.fixture()
def video_client(tmp_path):
    """TestClient backed by a MockVisualGen that produces videos."""
    gen = MockVisualGen(video_output=_make_dummy_video_tensor())
    os.environ["TRTLLM_MEDIA_STORAGE_PATH"] = str(tmp_path)
    client = _create_server(gen)
    yield client
    os.environ.pop("TRTLLM_MEDIA_STORAGE_PATH", None)


@pytest.fixture()
def video_audio_client(tmp_path):
    """TestClient backed by a MockVisualGen that produces videos with audio."""
    gen = MockVisualGen(
        video_output=_make_dummy_video_tensor(),
        audio_output=_make_dummy_audio_tensor(),
    )
    os.environ["TRTLLM_MEDIA_STORAGE_PATH"] = str(tmp_path)
    client = _create_server(gen)
    yield client
    os.environ.pop("TRTLLM_MEDIA_STORAGE_PATH", None)


@pytest.fixture()
def failing_client(tmp_path):
    """TestClient backed by a MockVisualGen that always fails."""
    gen = MockVisualGen(should_fail=True)
    os.environ["TRTLLM_MEDIA_STORAGE_PATH"] = str(tmp_path)
    client = _create_server(gen)
    yield client
    os.environ.pop("TRTLLM_MEDIA_STORAGE_PATH", None)


@pytest.fixture(autouse=True)
def _clear_video_store():
    """Reset the global VIDEO_STORE before each test."""
    VIDEO_STORE._items.clear()
    yield
    VIDEO_STORE._items.clear()


@pytest.fixture(autouse=True)
def _mock_video_encoding():
    """Mock video encoding to avoid ffmpeg dependency in unit tests.

    Replaces ``tensorrt_llm.media.encoding._save_encoded_video`` with a stub
    that writes a small dummy file so FileResponse can serve it; also mocks
    ffmpeg availability so ``resolve_video_format`` always resolves to mp4.
    """

    def _dummy_save_encoded_video(video, audio, output_path, frame_rate, audio_sample_rate=24000):
        os.makedirs(os.path.dirname(str(output_path)) or ".", exist_ok=True)
        with open(str(output_path), "wb") as f:
            f.write(b"\x00\x00\x00\x1cftypisom" + b"\x00" * 32)
        return str(output_path)

    with (
        patch("tensorrt_llm.media.encoding._save_encoded_video", _dummy_save_encoded_video),
        patch("tensorrt_llm.media.encoding._check_ffmpeg_available", return_value=True),
    ):
        yield


# =========================================================================
# POST /v1/images/generations
# =========================================================================


@pytest.mark.threadleak(enabled=False)  # FileResponse spawns AnyIO worker threads
class TestImageGeneration:
    def test_basic_image_generation_b64(self, image_client):
        resp = image_client.post(
            "/v1/images/generations",
            json={
                "prompt": "A cat sitting on a mat",
                "response_format": "b64_json",
                "size": "64x64",
            },
        )
        assert resp.status_code == 200
        _assert_visual_gen_server_timing(resp.headers)
        data = resp.json()
        assert "data" in data
        assert len(data["data"]) >= 1
        img_obj = data["data"][0]
        assert img_obj["b64_json"] is not None
        # Verify it decodes to valid bytes
        decoded = base64.b64decode(img_obj["b64_json"])
        assert len(decoded) > 0
        assert img_obj["revised_prompt"] == "A cat sitting on a mat"

    def test_image_generation_with_optional_params(self, image_client):
        resp = image_client.post(
            "/v1/images/generations",
            json={
                "prompt": "Sunset over ocean",
                "response_format": "b64_json",
                "size": "128x64",
                "num_inference_steps": 20,
                "guidance_scale": 7.5,
                "seed": 123,
                "negative_prompt": "blurry",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["size"] == "128x64"

        # Verify openai_server/parse_visual_gen_params forwarded every field.
        params = image_client.mock_gen.last_params
        assert image_client.mock_gen.last_inputs == "Sunset over ocean"
        assert params.width == 128
        assert params.height == 64
        assert params.num_inference_steps == 20
        assert params.guidance_scale == 7.5
        assert params.negative_prompt == "blurry"

    def test_image_generation_url_returns_fetchable_urls(self, image_client):
        """``response_format='url'`` writes each generated image to
        media storage and surfaces a server-relative HTTP URL pointing
        at ``GET /v1/images/{id}/content?i=N``. The URL fetches the
        image bytes back through the API instead of leaking the
        on-disk path."""
        resp = image_client.post(
            "/v1/images/generations",
            json={
                "prompt": "A dog",
                "response_format": "url",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["data"]) >= 1
        url = body["data"][0]["url"]
        # URL is an HTTP URL through the API content endpoint.
        assert "/v1/images/" in url and "/content" in url
        # Fetch via the same client to verify it works.
        path = url.split("//", 1)[-1].split("/", 1)[1]
        content = image_client.get("/" + path)
        assert content.status_code == 200
        # PNG bytes start with the standard magic header.
        assert content.content.startswith(b"\x89PNG\r\n\x1a\n")
        assert content.headers["content-type"] == "image/png"

    def test_image_generation_safetensors_b64(self, image_client):
        """Tensor formats return base64-encoded raw bytes; loading the
        payload yields the engine tensors back."""
        from safetensors.torch import load as load_safetensors

        resp = image_client.post(
            "/v1/images/generations",
            json={
                "prompt": "Tensor cat",
                "response_format": "b64_json",
                "format": "safetensors",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["data"]) == 1
        b64 = body["data"][0]["b64_json"]
        loaded = load_safetensors(base64.b64decode(b64))
        assert "image" in loaded

    def test_image_generation_pt_url(self, image_client):
        """Tensor formats under ``response_format='url'`` write each
        per-item payload to media storage and surface a fetchable
        HTTP URL through the image content endpoint."""
        resp = image_client.post(
            "/v1/images/generations",
            json={
                "prompt": "Tensor dog",
                "response_format": "url",
                "format": "pt",
            },
        )
        assert resp.status_code == 200
        url = resp.json()["data"][0]["url"]
        assert "/v1/images/" in url and "/content" in url
        path = url.split("//", 1)[-1].split("/", 1)[1]
        content = image_client.get("/" + path)
        assert content.status_code == 200
        assert content.headers["content-type"] == "application/octet-stream"
        loaded = torch.load(BytesIO(content.content), weights_only=True)
        assert "image" in loaded

    def test_image_generation_auto_size(self, image_client):
        resp = image_client.post(
            "/v1/images/generations",
            json={
                "prompt": "A tree",
                "response_format": "b64_json",
                "size": "auto",
            },
        )
        assert resp.status_code == 200

    def test_image_generation_failure(self, failing_client):
        """Engine-side ``RuntimeError`` (non-validation) surfaces as HTTP 500;
        the LLM envelope carries the error message."""
        resp = failing_client.post(
            "/v1/images/generations",
            json={
                "prompt": "A bird",
                "response_format": "b64_json",
            },
        )
        assert resp.status_code == 500
        _assert_llm_envelope(resp.json(), code=500, err_type="InternalServerError")

    def test_image_generation_invalid_size(self, image_client):
        """Invalid size triggers a Pydantic ``RequestValidationError``;
        the visual-gen-scoped handler emits the LLM-style 422 envelope."""
        resp = image_client.post(
            "/v1/images/generations",
            json={
                "prompt": "A mountain",
                "response_format": "b64_json",
                "size": "invalid",
            },
        )
        assert resp.status_code == 422
        _assert_llm_envelope(resp.json(), code=422, message_contains="size")

    def test_image_generation_null_output(self, tmp_path):
        """Generator returns VisualGenOutput with image=None."""
        gen = MockVisualGen(image_output=None)
        os.environ["TRTLLM_MEDIA_STORAGE_PATH"] = str(tmp_path)
        client = _create_server(gen)
        resp = client.post(
            "/v1/images/generations",
            json={
                "prompt": "null image",
                "response_format": "b64_json",
            },
        )
        assert resp.status_code == 500
        os.environ.pop("TRTLLM_MEDIA_STORAGE_PATH", None)

    def test_image_generation_multiple_n(self, image_client):
        """Request n=2 images in one call."""
        resp = image_client.post(
            "/v1/images/generations",
            json={
                "prompt": "Flowers",
                "response_format": "b64_json",
                "size": "64x64",
                "n": 2,
            },
        )
        assert resp.status_code == 200

    def test_image_generation_hd_quality(self, image_client):
        resp = image_client.post(
            "/v1/images/generations",
            json={
                "prompt": "HD landscape",
                "response_format": "b64_json",
                "quality": "hd",
            },
        )
        assert resp.status_code == 200

    def test_missing_prompt_image_generation(self, image_client):
        """Missing required field surfaces as a Pydantic
        ``RequestValidationError`` and the visual-gen-scoped handler
        returns the LLM-style 422 envelope."""
        resp = image_client.post(
            "/v1/images/generations",
            json={},
        )
        assert resp.status_code == 422
        _assert_llm_envelope(resp.json(), code=422, message_contains="prompt")

    def test_image_generation_b64_no_save_image_no_disk_write(self, image_client, tmp_path):
        """Regression guard for NVBug 6064029.

        The b64_json hot path must not call ``save_image()``, which caused a
        redundant PNG encode plus an unnecessary disk write before fix #12903.
        """
        with patch("tensorrt_llm.media.encoding.save_image") as mock_save:
            resp = image_client.post(
                "/v1/images/generations",
                json={
                    "prompt": "A cat sitting on a mat",
                    "response_format": "b64_json",
                    "size": "64x64",
                },
            )
        assert resp.status_code == 200
        mock_save.assert_not_called()
        assert list(tmp_path.glob("*.png")) == []

    def test_image_generation_b64_with_4d_batch_pipeline_output(self, tmp_path):
        """NVBug 6064029: when the pipeline returns a 4D (B, H, W, C)
        tensor (e.g. FLUX2), all B images must be expanded, encoded once
        each, and returned in order. Pre-fix, save_image silently kept
        only image[0], so the response would drop every batch entry but
        the first."""
        # Use deterministic distinct images (all-zeros vs all-255) so
        # we can verify per-image output mapping, not just call counts.
        from tensorrt_llm.media.encoding import image_to_bytes

        img0 = torch.zeros((64, 64, 3), dtype=torch.uint8)
        img1 = torch.full((64, 64, 3), 255, dtype=torch.uint8)
        batch = torch.stack([img0, img1])  # (2, H, W, C)
        expected_b64 = [
            base64.b64encode(image_to_bytes(img)).decode("utf-8") for img in (img0, img1)
        ]

        gen = MockVisualGen(image_output=batch)
        os.environ["TRTLLM_MEDIA_STORAGE_PATH"] = str(tmp_path)
        try:
            client = _create_server(gen)
            with (
                patch(
                    "tensorrt_llm.serve.openai_server.image_to_bytes",
                    wraps=image_to_bytes,
                ) as mock_cvt,
                patch("tensorrt_llm.media.encoding.save_image") as mock_save,
            ):
                resp = client.post(
                    "/v1/images/generations",
                    json={
                        "prompt": "two cats",
                        "response_format": "b64_json",
                        "size": "64x64",
                    },
                )
            assert resp.status_code == 200
            data = resp.json()["data"]
            assert len(data) == 2
            assert mock_cvt.call_count == 2
            mock_save.assert_not_called()
            # Content + order match: proves each batch entry maps to
            # its own b64 output, not just "encoded twice on image[0]".
            assert [entry["b64_json"] for entry in data] == expected_b64
        finally:
            os.environ.pop("TRTLLM_MEDIA_STORAGE_PATH", None)


# =========================================================================
# POST /v1/images/edits
# =========================================================================


class TestImageEdit:
    """``/v1/images/edits`` returns 501 NotImplemented in the current release.

    No in-tree pipeline implements image editing: Flux/Flux2 are
    text-to-image only and ignore ``params.image``; Wan and LTX-2 produce
    video, not edited images. Restore the full happy-path coverage when an
    edit-capable pipeline lands.
    """

    def test_image_edit_returns_not_implemented(self, image_client):
        """Valid request body short-circuits to 501 NotImplemented."""
        b64_img = _b64_white_png_1x1()
        resp = image_client.post(
            "/v1/images/edits",
            json={
                "image": b64_img,
                "prompt": "Make it blue",
                "num_inference_steps": 10,
            },
        )
        assert resp.status_code == 501
        body = resp.json()
        assert body.get("type") == "NotImplementedError"
        assert "not supported" in body.get("message", "").lower()

    def test_image_edit_no_body_returns_not_implemented(self, image_client):
        """The route doesn't parse a typed body; any incoming request still
        gets 501, including ones that would have failed schema validation
        before. Restore typed-body coverage when an edit pipeline lands."""
        resp = image_client.post("/v1/images/edits", json={"prompt": "Edit without image"})
        assert resp.status_code == 501
        body = resp.json()
        assert body.get("type") == "NotImplementedError"


# =========================================================================
# _normalize_image_output helper (NVBug 6064029)
# =========================================================================


class TestNormalizeImageOutput:
    """Coverage for the helper added by the NVBug 6064029 fix."""

    def test_list_input_passthrough(self):
        t1 = _make_dummy_image_tensor()
        t2 = _make_dummy_image_tensor()
        out = _normalize_image_output([t1, t2])
        assert len(out) == 2
        assert out[0] is t1 and out[1] is t2

    def test_3d_tensor_wrapped_as_single(self):
        t = _make_dummy_image_tensor()  # (H, W, C)
        assert t.dim() == 3
        out = _normalize_image_output(t)
        assert len(out) == 1 and out[0] is t

    def test_4d_batch_tensor_expanded(self):
        batch = torch.stack([_make_dummy_image_tensor() for _ in range(3)])
        assert batch.dim() == 4 and batch.shape[0] == 3
        out = _normalize_image_output(batch)
        assert len(out) == 3
        for i in range(3):
            assert torch.equal(out[i], batch[i])


# =========================================================================
# POST /v1/videos/generations  (synchronous)
# =========================================================================


@pytest.mark.threadleak(enabled=False)  # FileResponse spawns AnyIO worker threads
class TestVideoGenerationSync:
    def test_basic_sync_video_generation(self, video_client):
        resp = video_client.post(
            "/v1/videos/generations",
            json={
                "prompt": "A rocket launching",
                "size": "64x64",
                "seconds": 1.0,
                "fps": 8,
            },
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "video/mp4"
        _assert_visual_gen_server_timing(resp.headers)
        assert len(resp.content) > 0

    def test_sync_video_generation_with_params(self, video_client):
        resp = video_client.post(
            "/v1/videos/generations",
            json={
                "prompt": "Ocean waves",
                "size": "64x64",
                "seconds": 2.0,
                "fps": 8,
                "num_inference_steps": 10,
                "guidance_scale": 5.0,
                "seed": 42,
                "negative_prompt": "blurry",
            },
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 200
        assert len(resp.content) > 0

        params = video_client.mock_gen.last_params
        assert video_client.mock_gen.last_inputs == "Ocean waves"
        assert params.width == 64
        assert params.height == 64
        assert params.num_inference_steps == 10
        assert params.guidance_scale == 5.0
        assert params.seed == 42
        assert params.negative_prompt == "blurry"
        assert params.frame_rate == 8
        assert params.num_frames == int(2.0 * 8)

    def test_sync_video_generation_multipart(self, video_client, tmp_path):
        """Multipart sync request with a real ``input_reference`` file."""
        ref_path = tmp_path / "ref.png"
        Image.new("RGB", (4, 4), (64, 64, 64)).save(str(ref_path))
        with open(ref_path, "rb") as f:
            resp = video_client.post(
                "/v1/videos/generations",
                data={
                    "prompt": "Mountain sunrise",
                    "size": "64x64",
                    "seconds": "1.0",
                    "fps": "8",
                },
                files={"input_reference": ("ref.png", f, "image/png")},
            )
        assert resp.status_code == 200
        assert len(resp.content) > 0

    def test_sync_video_generation_multipart_with_reference(self, video_client, tmp_path):
        # Create a dummy reference image file
        ref_path = tmp_path / "ref.png"
        Image.new("RGB", (4, 4), (128, 128, 128)).save(str(ref_path))

        with open(ref_path, "rb") as f:
            resp = video_client.post(
                "/v1/videos/generations",
                data={
                    "prompt": "Animate this image",
                    "size": "64x64",
                    "seconds": "1.0",
                    "fps": "8",
                },
                files={"input_reference": ("ref.png", f, "image/png")},
            )
        assert resp.status_code == 200
        assert len(resp.content) > 0

        # input_reference should have been written to media storage and passed
        # through as params.image (a filesystem path).
        params = video_client.mock_gen.last_params
        assert isinstance(params.image, str)
        assert params.image.endswith("_reference.png")
        assert os.path.exists(params.image)

    def test_sync_video_failure(self, failing_client):
        resp = failing_client.post(
            "/v1/videos/generations",
            json={
                "prompt": "Should fail",
                "size": "64x64",
                "seconds": 1.0,
                "fps": 8,
            },
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 500

    def test_sync_video_null_output(self, tmp_path):
        """Generator returns VisualGenOutput with video=None."""
        gen = MockVisualGen(video_output=None)
        os.environ["TRTLLM_MEDIA_STORAGE_PATH"] = str(tmp_path)
        client = _create_server(gen)
        resp = client.post(
            "/v1/videos/generations",
            json={"prompt": "null video", "size": "64x64", "seconds": 1.0, "fps": 8},
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 500
        os.environ.pop("TRTLLM_MEDIA_STORAGE_PATH", None)

    def test_sync_video_unsupported_content_type(self, video_client):
        resp = video_client.post(
            "/v1/videos/generations",
            content=b"some raw bytes",
            headers={"content-type": "text/plain"},
        )
        assert resp.status_code == 400

    def test_sync_video_missing_prompt_json(self, video_client):
        """Missing required ``prompt`` surfaces the visual-gen 422 envelope."""
        resp = video_client.post(
            "/v1/videos/generations",
            json={"size": "64x64"},
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 422
        _assert_llm_envelope(resp.json(), code=422, message_contains="prompt")

    def test_sync_video_missing_prompt_multipart(self, video_client):
        """Multipart body with a missing required field surfaces the
        same LLM envelope as JSON so the wire contract is identical."""
        dummy_file = BytesIO(b"")
        resp = video_client.post(
            "/v1/videos/generations",
            data={"size": "64x64"},
            files={"_dummy": ("dummy", dummy_file, "application/octet-stream")},
        )
        assert resp.status_code == 422
        _assert_llm_envelope(resp.json(), code=422)

    def test_sync_video_multipart_rejects_unknown_field(self, video_client):
        """Strict multipart parsing rejects any form field that is not
        on :class:`VideoGenerationRequest` with the same 422 envelope as
        the JSON path."""
        dummy_file = BytesIO(b"")
        resp = video_client.post(
            "/v1/videos/generations",
            data={
                "prompt": "Strict multipart",
                "size": "64x64",
                "seconds": "1.0",
                "fps": "8",
                "output_format": "mp4",
            },
            files={"_dummy": ("dummy", dummy_file, "application/octet-stream")},
        )
        assert resp.status_code == 422
        _assert_llm_envelope(resp.json(), code=422, message_contains="output_format")

    def test_sync_video_rejects_top_level_n(self, video_client):
        """Sync video has no top-level ``n``; it's rejected with 422."""
        resp = video_client.post(
            "/v1/videos/generations",
            json={
                "prompt": "Batch rockets",
                "size": "64x64",
                "seconds": 1.0,
                "fps": 8,
                "n": 2,
            },
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 422
        _assert_llm_envelope(resp.json(), code=422)


# =========================================================================
# POST /v1/videos  (asynchronous)
# =========================================================================


class TestVideoGenerationAsync:
    def test_async_video_returns_202(self, video_client):
        resp = video_client.post(
            "/v1/videos",
            json={
                "prompt": "A dancing robot",
                "size": "64x64",
                "seconds": 1.0,
                "fps": 8,
            },
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "queued"
        assert data["object"] == "video"
        assert data["prompt"] == "A dancing robot"
        assert data["id"].startswith("video_")

    def test_async_video_job_metadata_fields(self, video_client):
        resp = video_client.post(
            "/v1/videos",
            json={
                "prompt": "Starry night",
                "size": "64x64",
                "seconds": 2.0,
                "fps": 12,
            },
            headers={"content-type": "application/json"},
        )
        data = resp.json()
        assert "created_at" in data
        assert data["duration"] == 2.0
        assert data["fps"] == 12
        assert data["size"] == "64x64"

    def test_async_video_multipart(self, video_client, tmp_path):
        """Multipart async request with a real ``input_reference`` file."""
        ref_path = tmp_path / "ref.png"
        Image.new("RGB", (4, 4), (16, 16, 16)).save(str(ref_path))
        with open(ref_path, "rb") as f:
            resp = video_client.post(
                "/v1/videos",
                data={
                    "prompt": "A sunset",
                    "size": "64x64",
                    "seconds": "1.0",
                    "fps": "8",
                },
                files={"input_reference": ("ref.png", f, "image/png")},
            )
        assert resp.status_code == 202

    def test_async_video_rejects_top_level_n(self, video_client):
        """Video has no top-level ``n``; it's rejected with 422 by ``extra=forbid``."""
        resp = video_client.post(
            "/v1/videos",
            json={
                "prompt": "Batch fireworks",
                "size": "64x64",
                "seconds": 1.0,
                "fps": 8,
                "n": 2,
            },
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 422
        _assert_llm_envelope(resp.json(), code=422)

    def test_async_video_rejects_top_level_guidance_rescale(self, video_client):
        """``guidance_rescale`` is per-model; must travel via ``extra_params``."""
        resp = video_client.post(
            "/v1/videos",
            json={
                "prompt": "Bad knob",
                "seconds": 1.0,
                "size": "64x64",
                "fps": 8,
                "guidance_rescale": 0.7,
            },
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 422
        _assert_llm_envelope(resp.json(), code=422)

    def test_async_video_rejects_output_format(self, video_client):
        """``output_format`` has been renamed to ``format``."""
        resp = video_client.post(
            "/v1/videos",
            json={
                "prompt": "Bad name",
                "seconds": 1.0,
                "size": "64x64",
                "fps": 8,
                "output_format": "mp4",
            },
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 422
        _assert_llm_envelope(resp.json(), code=422)

    def test_async_video_accepts_request_with_params(self, video_client):
        """The async ``/v1/videos`` route accepts the full request shape and
        returns 202 with a queued job. Per-field forwarding is asserted
        only against the *sync* routes — the async path deep-copies the
        request before enqueuing and the background task runs out-of-order
        with the test, so ``mock_gen.last_params`` is not a reliable
        capture point for merge-semantics here. Direct conversion-helper
        tests cover the field-by-field overlay instead.
        """
        resp = video_client.post(
            "/v1/videos",
            json={
                "prompt": "Rainy street",
                "size": "128x64",
                "seconds": 2.0,
                "fps": 10,
                "num_inference_steps": 12,
                "guidance_scale": 6.0,
                "seed": 7,
                "negative_prompt": "noise",
            },
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "queued"
        assert data["object"] == "video"
        assert data["prompt"] == "Rainy street"
        assert data["id"].startswith("video_")

    def test_async_video_accepts_extra_params(self, video_client):
        """Per-model overflow travels through ``extra_params``."""
        resp = video_client.post(
            "/v1/videos",
            json={
                "prompt": "Stylized fireworks",
                "size": "64x64",
                "seconds": 1.0,
                "fps": 8,
                "extra_params": {"stg_scale": 1.5},
            },
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "queued"
        assert data["id"].startswith("video_")


# =========================================================================
# GET /v1/videos  (list)
# =========================================================================


class TestListVideos:
    def test_list_videos_empty(self, video_client):
        resp = video_client.get("/v1/videos")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert data["data"] == []

    def test_list_videos_after_creation(self, video_client):
        # Create two video jobs
        video_client.post(
            "/v1/videos",
            json={"prompt": "First video", "size": "64x64", "seconds": 1.0, "fps": 8},
            headers={"content-type": "application/json"},
        )
        video_client.post(
            "/v1/videos",
            json={"prompt": "Second video", "size": "64x64", "seconds": 1.0, "fps": 8},
            headers={"content-type": "application/json"},
        )

        resp = video_client.get("/v1/videos")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 2


# =========================================================================
# GET /v1/videos/{video_id}  (metadata)
# =========================================================================


class TestGetVideoMetadata:
    def test_get_video_metadata_success(self, video_client):
        create_resp = video_client.post(
            "/v1/videos",
            json={"prompt": "Space walk", "size": "64x64", "seconds": 1.0, "fps": 8},
            headers={"content-type": "application/json"},
        )
        video_id = create_resp.json()["id"]

        resp = video_client.get(f"/v1/videos/{video_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == video_id
        assert data["object"] == "video"
        assert data["prompt"] == "Space walk"

    def test_get_video_metadata_not_found(self, video_client):
        resp = video_client.get("/v1/videos/video_nonexistent")
        assert resp.status_code == 404


# =========================================================================
# GET /v1/videos/{video_id}/content  (download)
# =========================================================================


@pytest.mark.threadleak(enabled=False)  # FileResponse spawns AnyIO worker threads
class TestGetVideoContent:
    def _insert_video_job(self, video_id: str, status: str = "queued"):
        import time as _time

        job = VideoJob(
            created_at=int(_time.time()),
            id=video_id,
            model="test-model",
            prompt="test prompt",
            status=status,
        )
        _run_async(VIDEO_STORE.upsert(video_id, job))

    def test_get_video_content_success(self, tmp_path):
        gen = MockVisualGen(video_output=_make_dummy_video_tensor())
        os.environ["TRTLLM_MEDIA_STORAGE_PATH"] = str(tmp_path)
        client = _create_server(gen)

        video_id = "video_testcontent"
        self._insert_video_job(video_id, status="completed")

        # Write a dummy mp4 file so FileResponse can serve it
        video_path = tmp_path / f"{video_id}.mp4"
        video_path.write_bytes(b"\x00\x00\x00\x1cftyp" + b"\x00" * 16)

        resp = client.get(f"/v1/videos/{video_id}/content")
        assert resp.status_code == 200
        assert "video/mp4" in resp.headers.get("content-type", "")
        assert len(resp.content) > 0
        os.environ.pop("TRTLLM_MEDIA_STORAGE_PATH", None)

    def test_get_video_content_not_found(self, video_client):
        resp = video_client.get("/v1/videos/video_nonexistent/content")
        assert resp.status_code == 404

    def test_get_video_content_not_ready(self, tmp_path):
        """A queued video should return 400 when its content is requested."""
        gen = MockVisualGen(video_output=_make_dummy_video_tensor())
        os.environ["TRTLLM_MEDIA_STORAGE_PATH"] = str(tmp_path)
        client = _create_server(gen)

        video_id = "video_notready"
        self._insert_video_job(video_id, status="queued")

        resp = client.get(f"/v1/videos/{video_id}/content")
        assert resp.status_code == 400
        os.environ.pop("TRTLLM_MEDIA_STORAGE_PATH", None)

    def test_get_video_content_completed_but_file_missing(self, tmp_path):
        """Video marked completed but file deleted from disk → 404."""
        gen = MockVisualGen(video_output=_make_dummy_video_tensor())
        os.environ["TRTLLM_MEDIA_STORAGE_PATH"] = str(tmp_path)
        client = _create_server(gen)

        video_id = "video_nofile"
        self._insert_video_job(video_id, status="completed")
        # Do NOT write a file

        resp = client.get(f"/v1/videos/{video_id}/content")
        assert resp.status_code == 404
        os.environ.pop("TRTLLM_MEDIA_STORAGE_PATH", None)


# =========================================================================
# DELETE /v1/videos/{video_id}
# =========================================================================


class TestDeleteVideo:
    def test_delete_video_success(self, tmp_path):
        gen = MockVisualGen(video_output=_make_dummy_video_tensor())
        os.environ["TRTLLM_MEDIA_STORAGE_PATH"] = str(tmp_path)
        client = _create_server(gen)

        create_resp = client.post(
            "/v1/videos",
            json={"prompt": "Delete me", "size": "64x64", "seconds": 1.0, "fps": 8},
            headers={"content-type": "application/json"},
        )
        video_id = create_resp.json()["id"]

        # Write a dummy video file matching the batch naming convention.
        (tmp_path / f"{video_id}_0.mp4").write_bytes(b"\x00" * 32)

        resp = client.delete(f"/v1/videos/{video_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] is True

        # Verify it's gone from the store
        resp = client.get(f"/v1/videos/{video_id}")
        assert resp.status_code == 404

        # Verify file is deleted
        assert not (tmp_path / f"{video_id}_0.mp4").exists()
        os.environ.pop("TRTLLM_MEDIA_STORAGE_PATH", None)

    def test_delete_video_not_found(self, video_client):
        resp = video_client.delete("/v1/videos/video_nonexistent")
        assert resp.status_code == 404

    def test_delete_video_without_file_on_disk(self, video_client):
        """Delete a video job that exists in the store but has no file on disk."""
        create_resp = video_client.post(
            "/v1/videos",
            json={"prompt": "No file", "size": "64x64", "seconds": 1.0, "fps": 8},
            headers={"content-type": "application/json"},
        )
        video_id = create_resp.json()["id"]

        resp = video_client.delete(f"/v1/videos/{video_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] is True

    def test_delete_video_then_list_empty(self, video_client):
        """After deleting the only video, the list should be empty."""
        create_resp = video_client.post(
            "/v1/videos",
            json={"prompt": "Ephemeral", "size": "64x64", "seconds": 1.0, "fps": 8},
            headers={"content-type": "application/json"},
        )
        video_id = create_resp.json()["id"]

        video_client.delete(f"/v1/videos/{video_id}")

        resp = video_client.get("/v1/videos")
        assert resp.status_code == 200
        assert resp.json()["data"] == []


# =========================================================================
# Test video generation failure handling (async)
# =========================================================================


class TestAsyncVideoFailureHandling:
    def test_async_video_null_output_updates_job_status(self, tmp_path):
        """When output.video is None in async generation, job status should be set to failed."""
        import time

        gen = MockVisualGen(video_output=None)
        os.environ["TRTLLM_MEDIA_STORAGE_PATH"] = str(tmp_path)
        client = _create_server(gen)

        # Create async video job
        create_resp = client.post(
            "/v1/videos",
            json={"prompt": "null video", "size": "64x64", "seconds": 1.0, "fps": 8},
            headers={"content-type": "application/json"},
        )
        assert create_resp.status_code == 202
        video_id = create_resp.json()["id"]

        # Wait briefly for background task to complete
        time.sleep(0.5)

        # Check job status
        resp = client.get(f"/v1/videos/{video_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert "error" in data
        assert "output.video is None" in data["error"]

        os.environ.pop("TRTLLM_MEDIA_STORAGE_PATH", None)


# =========================================================================
# Route-level engine-validation-error handling
# =========================================================================


def _make_validation_error(param: str = "stg_sclae"):
    """Build the kind of stock ``ValueError`` ``validate_visual_gen_params``
    raises when extra_params contains an unknown key. Tests inject this
    onto the mock so the routes' ``except ValueError`` arm fires the same
    way it would in production."""
    return ValueError(
        f"Parameter validation failed:\n  - Unknown extra_params ['{param}']. Supported: []"
    )


class TestRouteEngineValidationError:
    """When the engine raises ``ValueError`` (request-shape problem), the
    image and sync-video routes return HTTP 400 with the LLM envelope
    built from the exception message. The async-video route runs the
    same check synchronously via ``validate_visual_gen_params`` so an
    unknown ``extra_params`` key surfaces as 400 immediately instead of
    becoming a queued 202 whose background task later fails."""

    def test_image_route_renders_validation_error_at_400(self, tmp_path):
        os.environ["TRTLLM_MEDIA_STORAGE_PATH"] = str(tmp_path)
        try:
            gen = MockVisualGen(
                image_output=_make_dummy_image_tensor(),
                validation_error=_make_validation_error(),
            )
            client = _create_server(gen)
            resp = client.post(
                "/v1/images/generations",
                json={
                    "prompt": "trigger validation error",
                    "response_format": "b64_json",
                    "extra_params": {"stg_sclae": 1.0},
                },
            )
            assert resp.status_code == 400
            _assert_llm_envelope(
                resp.json(),
                code=400,
                message_contains="stg_sclae",
            )
        finally:
            os.environ.pop("TRTLLM_MEDIA_STORAGE_PATH", None)

    def test_sync_video_route_renders_validation_error_at_400(self, tmp_path):
        os.environ["TRTLLM_MEDIA_STORAGE_PATH"] = str(tmp_path)
        try:
            gen = MockVisualGen(
                video_output=_make_dummy_video_tensor(),
                validation_error=_make_validation_error(),
            )
            client = _create_server(gen)
            resp = client.post(
                "/v1/videos/generations",
                json={
                    "prompt": "trigger validation error",
                    "size": "64x64",
                    "seconds": 1.0,
                    "fps": 8,
                    "extra_params": {"stg_sclae": 1.0},
                },
                headers={"content-type": "application/json"},
            )
            assert resp.status_code == 400
            _assert_llm_envelope(
                resp.json(),
                code=400,
                message_contains="stg_sclae",
            )
        finally:
            os.environ.pop("TRTLLM_MEDIA_STORAGE_PATH", None)

    def test_image_route_serialization_value_error_returns_500(self, tmp_path, monkeypatch):
        """Server-side serialization failures map to 500, not 400.

        ``infer_batch_size`` / ``serialize_visual_gen_output`` raise
        ``ValueError`` for conditions on the server's own output
        (no media tensor, inconsistent multi-modal batch). The image
        route must render those as 500 — the client's request was
        valid; the server failed to serialize its own output.
        """
        os.environ["TRTLLM_MEDIA_STORAGE_PATH"] = str(tmp_path)
        try:
            gen = MockVisualGen(image_output=_make_dummy_image_tensor())

            def _raise_server_side(*args, **kwargs):
                raise ValueError("Cannot infer batch size: carries no media tensor.")

            # Force the tensor-format branch to hit a server-side ValueError
            # in the serialization region (outside the pre-generation try).
            monkeypatch.setattr(
                "tensorrt_llm.media.tensor_payload.infer_batch_size",
                _raise_server_side,
            )
            client = _create_server(gen)
            resp = client.post(
                "/v1/images/generations",
                json={
                    "prompt": "trigger serialization failure",
                    "response_format": "b64_json",
                    "format": "safetensors",
                },
            )
            assert resp.status_code == 500
        finally:
            os.environ.pop("TRTLLM_MEDIA_STORAGE_PATH", None)

    def test_async_video_route_rejects_validation_error_synchronously(self, tmp_path):
        """``/v1/videos`` calls ``validate_visual_gen_params`` against the
        mock's executor metadata before queuing; the mock's
        ``extra_param_specs={}`` causes any unknown extra to be rejected
        with a stock ``ValueError`` which the route's ``except ValueError``
        arm renders as HTTP 400."""
        os.environ["TRTLLM_MEDIA_STORAGE_PATH"] = str(tmp_path)
        try:
            gen = MockVisualGen(video_output=_make_dummy_video_tensor())
            client = _create_server(gen)
            resp = client.post(
                "/v1/videos",
                json={
                    "prompt": "trigger validation error",
                    "size": "64x64",
                    "seconds": 1.0,
                    "fps": 8,
                    "extra_params": {"stg_sclae": 1.0},
                },
                headers={"content-type": "application/json"},
            )
            assert resp.status_code == 400
            _assert_llm_envelope(
                resp.json(),
                code=400,
                message_contains="stg_sclae",
            )
        finally:
            os.environ.pop("TRTLLM_MEDIA_STORAGE_PATH", None)


# =========================================================================
# Non-visual-gen routes keep FastAPI's default validation response
# =========================================================================


class TestNonVisualGenValidationResponse:
    """Validation failures on non-visual-gen roles use the shared
    ``OpenAIServer`` response shape (HTTP 400 + ``{"error": ...}``)
    that existing integration coverage and clients expect (e.g.
    ``test_malformed_json_request``). Only the visual-gen role swaps
    in the LLM envelope.

    The assertion is checked at the handler-closure level: rebuild
    the exact dispatch installed in :meth:`OpenAIServer.__init__`
    against a minimal FastAPI app so the assertion stays narrow and
    the test doesn't need to spin up a full LLM-role server."""

    def _build_app_with_dispatch(self, role):
        """Return a FastAPI app wired with the production handler
        dispatch, where ``role`` controls the branch the handler takes
        on a ``RequestValidationError``."""
        from fastapi import FastAPI
        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel

        app = FastAPI()

        class _Body(BaseModel):
            messages: list

        @app.post("/route")
        async def _route(body: _Body):
            return {"ok": True}

        @app.exception_handler(RequestValidationError)
        async def _handler(_, exc):
            if role == "VISUAL_GEN":
                return _llm_envelope_branch(exc)
            return JSONResponse(status_code=400, content={"error": str(exc)})

        # Mirror :meth:`OpenAIServer._create_visual_gen_validation_error_response`
        # inline so the test does not depend on instance state.
        def _llm_envelope_branch(exc):
            from http import HTTPStatus

            from tensorrt_llm.serve.openai_protocol import ErrorResponse

            error = ErrorResponse(
                message="Validation failed",
                type="BadRequestError",
                code=HTTPStatus.UNPROCESSABLE_ENTITY.value,
            )
            return JSONResponse(
                content=error.model_dump(),
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY.value,
            )

        return app

    def test_non_visual_gen_role_returns_shared_400_error_body(self):
        """Non-visual-gen roles return HTTP 400 with the shared
        ``{"error": str(exc)}`` body that ``test_malformed_json_request``
        and existing clients depend on."""
        client = TestClient(self._build_app_with_dispatch(role="CONTEXT"))
        resp = client.post("/route", json={"not_messages": []})
        assert resp.status_code == 400
        body = resp.json()
        assert "error" in body
        assert isinstance(body["error"], str)
        # The visual-gen LLM envelope must not leak into non-VG paths.
        assert "object" not in body
        assert "type" not in body
        assert "code" not in body

    def test_visual_gen_role_uses_llm_envelope(self):
        client = TestClient(self._build_app_with_dispatch(role="VISUAL_GEN"))
        resp = client.post("/route", json={"not_messages": []})
        assert resp.status_code == 422
        body = resp.json()
        assert body["type"] == "BadRequestError"
        assert body["code"] == 422
        assert "message" in body


# =========================================================================
# Tensor-format response coverage on the video routes
# =========================================================================


@pytest.mark.threadleak(enabled=False)  # FileResponse spawns AnyIO worker threads
class TestVideoTensorResponse:
    """The sync route emits tensor payloads as a single file under
    ``response_format='url'`` and as base64-encoded bytes under
    ``response_format='b64_json'``. The async route persists the
    payload to media storage; ``GET /v1/videos/{id}/content`` serves
    the file with ``application/octet-stream``."""

    def _post_sync(self, video_client, fmt: str, response_format: str):
        return video_client.post(
            "/v1/videos/generations",
            json={
                "prompt": f"tensor video {fmt}",
                "size": "32x32",
                "seconds": 1.0,
                "fps": 8,
                "format": fmt,
                "response_format": response_format,
            },
            headers={"content-type": "application/json"},
        )

    @pytest.mark.parametrize("fmt", ["safetensors", "pt"])
    def test_sync_tensor_url_returns_file_with_correct_suffix(self, video_audio_client, fmt):
        resp = self._post_sync(video_audio_client, fmt, "url")
        assert resp.status_code == 200
        ext = f".{fmt}"
        # The content-disposition header carries the on-disk filename.
        disp = resp.headers.get("content-disposition", "")
        assert ext in disp, disp
        # And the payload itself round-trips.
        if fmt == "safetensors":
            from safetensors.torch import load as load_safetensors

            loaded = load_safetensors(resp.content)
        else:
            loaded = torch.load(BytesIO(resp.content), weights_only=True)
        assert "video" in loaded

    @pytest.mark.parametrize("fmt", ["safetensors", "pt"])
    def test_sync_tensor_b64_returns_decodable_payload(self, video_audio_client, fmt):
        resp = self._post_sync(video_audio_client, fmt, "b64_json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["format"] == fmt
        assert "b64_json" in data
        raw = base64.b64decode(data["b64_json"])
        if fmt == "safetensors":
            from safetensors.torch import load as load_safetensors

            loaded = load_safetensors(raw)
        else:
            loaded = torch.load(BytesIO(raw), weights_only=True)
        assert "video" in loaded

    @pytest.mark.parametrize("fmt", ["safetensors", "pt"])
    def test_async_tensor_persists_and_serves(self, video_audio_client, fmt, tmp_path):
        import time as _time

        client = video_audio_client
        resp = client.post(
            "/v1/videos",
            json={
                "prompt": f"async tensor {fmt}",
                "size": "32x32",
                "seconds": 1.0,
                "fps": 8,
                "format": fmt,
            },
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 202
        video_id = resp.json()["id"]

        # Drive the background task to completion via polling.
        deadline = _time.time() + 5
        while _time.time() < deadline:
            status = client.get(f"/v1/videos/{video_id}").json().get("status")
            if status in ("completed", "failed"):
                break
            _time.sleep(0.05)

        content = client.get(f"/v1/videos/{video_id}/content")
        assert content.status_code == 200
        # The server returns ``application/octet-stream`` for tensor payloads.
        assert content.headers["content-type"] == "application/octet-stream"
        if fmt == "safetensors":
            from safetensors.torch import load as load_safetensors

            loaded = load_safetensors(content.content)
        else:
            loaded = torch.load(BytesIO(content.content), weights_only=True)
        assert "video" in loaded


@pytest.mark.threadleak(enabled=False)  # FileResponse spawns AnyIO worker threads
class TestVideoEncoderB64Response:
    """The sync video route's encoder branch (``mp4``/``avi``/``auto``)
    honors ``response_format='b64_json'`` by base64-encoding the
    encoded video bytes; ``response_format='url'`` keeps the
    ``FileResponse`` download."""

    def test_sync_encoder_b64_json_returns_base64_payload(self, video_client):
        resp = video_client.post(
            "/v1/videos/generations",
            json={
                "prompt": "encoded b64",
                "size": "32x32",
                "seconds": 1.0,
                "fps": 8,
                "format": "avi",
                "response_format": "b64_json",
            },
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["format"] in {"mp4", "avi"}
        assert "b64_json" in body
        raw = base64.b64decode(body["b64_json"])
        # Non-empty encoded bytes — exact format verification is the
        # encoder layer's domain.
        assert len(raw) > 0

    def test_sync_encoder_url_keeps_file_response(self, video_client):
        resp = video_client.post(
            "/v1/videos/generations",
            json={
                "prompt": "encoded url",
                "size": "32x32",
                "seconds": 1.0,
                "fps": 8,
                "format": "avi",
                "response_format": "url",
            },
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 200
        # FileResponse for an AVI carries ``video/x-msvideo``.
        assert resp.headers["content-type"] == "video/x-msvideo"


class TestVideoTimingValidation:
    """Numeric optionals on ``VideoGenerationRequest`` reject zero /
    negative values so divisions and frame-count math downstream can
    trust the value."""

    @pytest.mark.parametrize(
        "field,value",
        [
            ("fps", 0),
            ("frame_rate", -1),
            ("num_frames", 0),
            ("num_frames", -3),
            ("seconds", 0),
            ("seconds", -2.5),
        ],
    )
    def test_non_positive_timing_field_rejected(self, video_client, field, value):
        resp = video_client.post(
            "/v1/videos/generations",
            json={
                "prompt": "bad timing",
                "size": "32x32",
                field: value,
            },
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 422
        _assert_llm_envelope(resp.json(), code=422, message_contains=field)


class TestImageResponseFormatMetadata:
    """``ImageGenerationResponse.output_format`` reflects the
    requested encoding so clients that introspect the response know
    how to decode the bytes / read the URL."""

    @pytest.mark.parametrize(
        "fmt",
        ["png", "webp", "jpeg", "safetensors", "pt"],
    )
    def test_response_carries_requested_format(self, image_client, fmt):
        resp = image_client.post(
            "/v1/images/generations",
            json={
                "prompt": f"metadata for {fmt}",
                "response_format": "b64_json",
                "format": fmt,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["output_format"] == fmt


@pytest.mark.threadleak(enabled=False)  # FileResponse spawns AnyIO worker threads
class TestVideoZeroFrameDerivationRejected:
    """``seconds * frame_rate`` that floors to zero frames must be
    rejected with HTTP 400 + LLM envelope rather than reaching the
    encoder with a 0-frame video."""

    def test_subsecond_seconds_below_one_frame_returns_400(self, video_client):
        resp = video_client.post(
            "/v1/videos/generations",
            json={
                "prompt": "way too short",
                "size": "32x32",
                "seconds": 0.01,
                "fps": 8,
            },
            headers={"content-type": "application/json"},
        )
        # int(0.01 * 8) == 0 — conversion raises ValueError → 400.
        assert resp.status_code == 400
        _assert_llm_envelope(
            resp.json(),
            code=400,
            message_contains="Derived frame count",
        )

    def test_seconds_without_frame_rate_returns_400(self, video_client):
        """``seconds`` set but neither the request nor the pipeline default
        declares a ``frame_rate``: the parser must reject the request with
        HTTP 400 instead of silently dropping the duration and returning the
        pipeline's default ``num_frames``."""
        resp = video_client.post(
            "/v1/videos/generations",
            json={
                "prompt": "duration without fps",
                "size": "32x32",
                "seconds": 1.0,
            },
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400
        _assert_llm_envelope(
            resp.json(),
            code=400,
            message_contains="frame_rate",
        )

    def test_explicit_num_frames_one_is_accepted(self, video_client):
        """The caller can bypass the derivation by passing ``num_frames``
        directly; the request must succeed."""
        resp = video_client.post(
            "/v1/videos/generations",
            json={
                "prompt": "explicit single frame",
                "size": "32x32",
                "num_frames": 1,
                "fps": 8,
            },
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 200


class TestImageBatchCap:
    """``ImageGenerationRequest.n`` is capped at 10 to bound resource
    usage. ``n=10`` is accepted; ``n=11`` and ``n=100000`` are
    rejected at the schema layer with HTTP 422 + LLM envelope."""

    def test_n_equal_to_ten_accepted(self, image_client):
        resp = image_client.post(
            "/v1/images/generations",
            json={
                "prompt": "ten images",
                "response_format": "b64_json",
                "size": "32x32",
                "n": 10,
            },
        )
        assert resp.status_code == 200
        assert len(resp.json()["data"]) == 10

    @pytest.mark.parametrize("n", [11, 100000])
    def test_n_above_cap_rejected(self, image_client, n):
        resp = image_client.post(
            "/v1/images/generations",
            json={
                "prompt": "too many",
                "response_format": "b64_json",
                "size": "32x32",
                "n": n,
            },
        )
        assert resp.status_code == 422
        _assert_llm_envelope(resp.json(), code=422, message_contains="n")


@pytest.mark.threadleak(enabled=False)  # FileResponse spawns AnyIO worker threads
class TestVideoFrameBudgetCap:
    """Upper bounds keep unbounded work / memory requests from reaching
    the engine. The defaults (a minute of video at 120 fps) are
    generous enough for common workloads; clients hitting the cap can
    raise it at deployment time."""

    @pytest.mark.parametrize(
        "field,value,boundary",
        [
            ("num_frames", 7200, "accepted"),
            ("num_frames", 7201, "rejected"),
            ("num_frames", 1_000_000, "rejected"),
            ("seconds", 60.0, "accepted"),
            ("seconds", 60.1, "rejected"),
            ("seconds", 1.0e9, "rejected"),
            ("fps", 120.0, "accepted"),
            ("fps", 120.1, "rejected"),
            ("fps", 1.0e6, "rejected"),
        ],
    )
    def test_frame_budget_bounds(self, video_client, field, value, boundary):
        payload = {
            "prompt": "boundary",
            "size": "32x32",
        }
        if field != "num_frames":
            # Pair seconds/fps with a sane partner to avoid the
            # derived-zero-frames check; pass num_frames otherwise.
            payload.update({"seconds": 1.0, "fps": 8})
        payload[field] = value
        resp = video_client.post(
            "/v1/videos/generations",
            json=payload,
            headers={"content-type": "application/json"},
        )
        if boundary == "accepted":
            # The schema accepts the value at the boundary. The
            # downstream pipeline may still 200 or 500 depending on
            # the mock's tensor shape; the relevant assertion is that
            # the request did not fall into the schema-rejection path.
            assert resp.status_code != 422, resp.text
        else:
            assert resp.status_code == 422
            _assert_llm_envelope(resp.json(), code=422, message_contains=field)


class TestVideoJobFractionalFps:
    """``VideoJob.fps`` is a float so cinematic frame rates like
    23.976 / 29.97 round-trip through the queued metadata instead of
    being truncated to int."""

    @pytest.mark.parametrize("rate", [23.976, 29.97, 59.94])
    def test_async_job_metadata_preserves_fractional_fps(self, video_client, rate):
        resp = video_client.post(
            "/v1/videos",
            json={
                "prompt": "fractional fps",
                "size": "32x32",
                "seconds": 1.0,
                "fps": rate,
            },
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 202
        body = resp.json()
        assert body["fps"] == rate

    def test_async_job_metadata_uses_resolved_default_fps(self, video_client):
        """When the request omits ``fps``/``frame_rate``, the queued
        ``VideoJob`` reports the pipeline-default rate that the
        conversion layer resolved on ``params.frame_rate`` — not
        ``None`` — so polling clients see accurate metadata for a
        video encoded at the model default."""
        # Force a known default on the mock pipeline so the assertion
        # is deterministic. ``MockVisualGen.default_params`` builds a
        # fresh ``VisualGenParams``; patching the property here lets
        # the test pretend the pipeline default is 12 fps.
        from tensorrt_llm.visual_gen import VisualGenParams

        class _FixedDefaultGen(MockVisualGen):
            @property
            def default_params(self):
                return VisualGenParams(frame_rate=12.0)

        gen = _FixedDefaultGen(video_output=_make_dummy_video_tensor())
        # The fixture installs media storage env vars; mirror that.
        os.environ["TRTLLM_MEDIA_STORAGE_PATH"] = (
            os.path.dirname(video_client.app.state.__dict__.get("media_storage_path", "/tmp/_vg"))
            or "/tmp/_vg"
        )
        try:
            client = _create_server(gen)
            resp = client.post(
                "/v1/videos",
                json={
                    "prompt": "no fps sent",
                    "size": "32x32",
                    "seconds": 1.0,
                },
                headers={"content-type": "application/json"},
            )
            assert resp.status_code == 202
            body = resp.json()
            assert body["fps"] == 12.0
        finally:
            os.environ.pop("TRTLLM_MEDIA_STORAGE_PATH", None)


def _raise_value_error(_fmt):
    raise ValueError("ffmpeg not available; encoder format unsupported")


def _raise_runtime_error(_fmt):
    raise RuntimeError("MP4 (H.264) format requires ffmpeg to be installed.")


@pytest.mark.threadleak(enabled=False)  # FileResponse spawns AnyIO worker threads
class TestVideoEncoderFailsFast:
    """When an encoder format can't be resolved, the sync and async
    video routes must reject the request before any GPU generation
    runs. ``resolve_video_format`` raises ``ValueError`` for genuinely
    unsupported format strings and ``RuntimeError`` for the
    missing-ffmpeg case on ``format='mp4'``; both must surface as a
    400, not a 500."""

    @pytest.mark.parametrize(
        "raiser",
        [_raise_value_error, _raise_runtime_error],
        ids=["unsupported_format", "missing_ffmpeg"],
    )
    def test_sync_route_fails_before_generate(self, video_client, monkeypatch, raiser):
        from tensorrt_llm.serve import openai_video_routes as routes

        monkeypatch.setattr(routes, "resolve_video_format", raiser)
        # Record whether the generator was called so the assertion
        # locks in the fail-fast contract.
        video_client.mock_gen.last_inputs = None
        resp = video_client.post(
            "/v1/videos/generations",
            json={
                "prompt": "mp4 without ffmpeg",
                "size": "32x32",
                "seconds": 1.0,
                "fps": 8,
                "format": "mp4",
            },
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400
        assert video_client.mock_gen.last_inputs is None, (
            "generate() must not run when the encoder format is unsupported"
        )

    @pytest.mark.parametrize(
        "raiser",
        [_raise_value_error, _raise_runtime_error],
        ids=["unsupported_format", "missing_ffmpeg"],
    )
    def test_async_route_fails_before_queue(self, video_client, monkeypatch, raiser):
        from tensorrt_llm.serve import openai_video_routes as routes

        monkeypatch.setattr(routes, "resolve_video_format", raiser)
        video_client.mock_gen.last_inputs = None
        resp = video_client.post(
            "/v1/videos",
            json={
                "prompt": "mp4 without ffmpeg",
                "size": "32x32",
                "seconds": 1.0,
                "fps": 8,
                "format": "mp4",
            },
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400
        assert video_client.mock_gen.last_inputs is None

    def test_sync_route_tensor_format_unaffected(self, video_client, monkeypatch):
        """Tensor formats have no encoder dependency; a broken
        ``resolve_video_format`` must not affect them."""
        from tensorrt_llm.serve import openai_video_routes as routes

        monkeypatch.setattr(routes, "resolve_video_format", _raise_value_error)
        resp = video_client.post(
            "/v1/videos/generations",
            json={
                "prompt": "tensor unaffected",
                "size": "32x32",
                "seconds": 1.0,
                "fps": 8,
                "format": "safetensors",
            },
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 200


@pytest.mark.threadleak(enabled=False)  # FileResponse spawns AnyIO worker threads
class TestAsyncVideoB64JsonTransport:
    """``POST /v1/videos`` persists the requested ``response_format`` on
    the queued job. ``GET /v1/videos/{id}/content`` honors it:
    ``url`` (or unset) returns a ``FileResponse`` download;
    ``b64_json`` returns a JSON envelope with the encoded bytes
    base64-inlined."""

    def _drive_job_to_completion(self, client, video_id):
        import time as _time

        deadline = _time.time() + 5
        while _time.time() < deadline:
            status = client.get(f"/v1/videos/{video_id}").json().get("status")
            if status in ("completed", "failed"):
                return status
            _time.sleep(0.05)
        return None

    def test_async_b64_json_returned_at_get_content(self, video_client):
        resp = video_client.post(
            "/v1/videos",
            json={
                "prompt": "async base64",
                "size": "32x32",
                "seconds": 1.0,
                "fps": 8,
                "format": "avi",
                "response_format": "b64_json",
            },
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 202
        job = resp.json()
        assert job["response_format"] == "b64_json"

        status = self._drive_job_to_completion(video_client, job["id"])
        assert status == "completed"

        content = video_client.get(f"/v1/videos/{job['id']}/content")
        assert content.status_code == 200
        body = content.json()
        assert set(body) >= {"id", "format", "b64_json"}
        assert body["id"] == job["id"]
        # The encoded payload decodes to non-empty bytes.
        raw = base64.b64decode(body["b64_json"])
        assert len(raw) > 0

    def test_async_url_still_returns_file_response(self, video_client):
        """Default and explicit ``response_format='url'`` keep the
        existing ``FileResponse`` behavior."""
        resp = video_client.post(
            "/v1/videos",
            json={
                "prompt": "async url",
                "size": "32x32",
                "seconds": 1.0,
                "fps": 8,
                "format": "avi",
                "response_format": "url",
            },
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 202
        job = resp.json()
        assert job["response_format"] == "url"

        self._drive_job_to_completion(video_client, job["id"])
        content = video_client.get(f"/v1/videos/{job['id']}/content")
        assert content.status_code == 200
        # AVI FileResponse carries ``video/x-msvideo``; the b64_json
        # branch would have set ``application/json``.
        assert content.headers["content-type"] == "video/x-msvideo"
