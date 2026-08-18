"""trtllm-serve visual_gen endpoints tests.

Tests all endpoints registered for the VISUAL_GEN server role
in OpenAIServer.register_visual_gen_routes():

    POST /v1/images/generations
    POST /v1/images/edits
    POST /v1/videos/sync          (sync)
    POST /v1/videos               (async)
    GET  /v1/videos               (list)
    GET  /v1/videos/{video_id}    (metadata)
    GET  /v1/videos/{video_id}/content  (download)
    DELETE /v1/videos/{video_id}  (delete)
"""

import asyncio
import base64
import json
import os
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio
import torch
from fastapi.testclient import TestClient
from PIL import Image

from tensorrt_llm.serve.openai_protocol import VideoJob
from tensorrt_llm.serve.openai_server import _normalize_image_output
from tensorrt_llm.serve.visual_gen_metrics import SERVER_TIMING_HEADER
from tensorrt_llm.serve.visual_gen_utils import VIDEO_STORE
from tensorrt_llm.visual_gen.media_refs import (
    cleanup_reference_files,
    prepare_reference_slots,
    resolve_media_storage_path,
)
from tensorrt_llm.visual_gen.output import VisualGenMetrics, VisualGenOutput
from tensorrt_llm.visual_gen.params import validate_visual_gen_params

pytestmark = pytest.mark.cpu_only


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


_V2V_FIXTURE_MP4 = Path(__file__).parent / "test_data" / "cosmos3_v2v_ref_9f_bframes.mp4"


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


def _server_timing_ms(headers, name: str) -> float:
    """Parse the ``dur`` (ms) of one metric out of the Server-Timing header."""
    server_timing = headers[SERVER_TIMING_HEADER]
    for part in server_timing.split(","):
        part = part.strip()
        if part.startswith(f"{name};dur="):
            return float(part[len(f"{name};dur=") :])
    raise AssertionError(f"{name!r} not in Server-Timing: {server_timing!r}")


def _drive_job_to_completion(client, video_id, timeout: float = 5.0):
    """Poll ``GET /v1/videos/{id}`` until the job reaches a terminal state.

    Returns the terminal status (``"completed"``/``"failed"``) or ``None`` on
    timeout. Shared by the async-video tests so the polling deadline lives in
    one place.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/v1/videos/{video_id}").json().get("status")
        if status in ("completed", "failed"):
            return status
        time.sleep(0.05)
    return None


async def _adrive_job_to_completion(client, video_id, timeout: float = 10.0):
    """Async counterpart of :func:`_drive_job_to_completion` for the httpx
    ``AsyncClient`` — awaits polls on the live loop so the background task's
    offloaded encode can progress to a terminal state.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = await client.get(f"/v1/videos/{video_id}")
        status = resp.json().get("status")
        if status in ("completed", "failed"):
            return status
        await asyncio.sleep(0.05)
    return None


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
        generate_error: Optional[BaseException] = None,
        extra_param_specs: Optional[dict] = None,
        model: str = "test-model",
        supports_image_edit: bool = False,
    ):
        from types import SimpleNamespace

        from tensorrt_llm._torch.visual_gen.pipeline import ExtraParamSchema

        self._image = image_output
        self._video = video_output
        self._audio = audio_output
        self._should_fail = should_fail
        self._batch_aware = batch_aware
        self._validation_error = validation_error
        # Raised out of generate(): models an engine-side failure class,
        # where validation_error models a coordinator preflight rejection.
        self._generate_error = generate_error
        self._extra_param_specs = extra_param_specs or {}
        self._model = model
        self._healthy = True
        self._req_counter = 0
        # Captured arguments of the most recent generate / generate_async call,
        # used by tests to assert forwarded VisualGenParams fields.
        self.last_inputs = None
        self.last_params = None
        # Snapshot of materialized reference-file contents at generation time,
        # captured before the route cleans them up. Keyed by stored path.
        self.last_ref_bytes = {}
        # Stand-in for the coordinator-side executor proxy. The async video
        # route reads ``default_generation_params`` / ``extra_param_specs``
        # directly off this attribute when running synchronous pre-flight
        # validation. ``default_generation_params`` declares the universal
        # fields the mock pipeline accepts so the validator doesn't
        # reject legitimate width/height/num_frames/... requests;
        # ``extra_param_specs`` lists a single known key so tests can
        # exercise both the accept-known and reject-unknown paths.
        from tensorrt_llm._torch.visual_gen.pipeline import RefSlotSpec, RoleSpec

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
            extra_param_specs=extra_param_specs
            or {"stg_scale": ExtraParamSchema(type="float", default=1.0)},
            supports_image_edit=supports_image_edit,
            ref_slot_specs={
                "image_reference": RefSlotSpec(
                    modality="image", roles=[RoleSpec(role="first_frame", min=0, max=1)]
                ),
                "video_reference": RefSlotSpec(
                    modality="video", roles=[RoleSpec(role="reference", min=0, max=1)]
                ),
            },
        )

    def _maybe_batch(self, tensor, n):
        """Replicate a single tensor along a new leading batch dimension."""
        if tensor is None or n <= 1 or not self._batch_aware:
            return tensor
        return tensor.unsqueeze(0).expand(n, *tensor.shape).contiguous()

    # --- VisualGen interface ---

    def _snapshot_refs(self, params) -> None:
        # Capture materialized reference bytes before the route cleans them up,
        # so tests can still assert byte-identity after the request finishes.
        self.last_ref_bytes = {}
        for field in ("image_reference", "video_reference", "audio_reference"):
            for ref in getattr(params, field, None) or []:
                path = getattr(ref, "content", None)
                if isinstance(path, str) and os.path.exists(path):
                    with open(path, "rb") as fh:
                        self.last_ref_bytes[path] = fh.read()

    def generate(self, inputs=None, params=None) -> VisualGenOutput:
        return self.generate_async(inputs=inputs, params=params).result()

    def generate_async(self, inputs=None, params=None) -> "MockVisualGenResult":
        self.last_inputs = inputs
        self.last_params = params
        if self._validation_error is not None:
            raise self._validation_error
        # Mirror the real engine entry: validate against the pipeline metadata
        # (unknown extra_params / undeclared fields / ref arity) before doing any
        # work, so the route's synchronous 400 path is exercised end-to-end.
        if params is not None:
            validate_visual_gen_params(
                params,
                declared_defaults=self.executor.default_generation_params,
                extra_param_specs=self.executor.extra_param_specs,
                ref_slot_specs=self.executor.ref_slot_specs,
            )
        # Materialize references at the coordinator, then hand the result a
        # terminal cleanup keyed on the same request id.
        req_id = self._next_request_id()
        media_storage_path = str(resolve_media_storage_path())
        prepare_reference_slots(
            params, request_id=str(req_id), media_storage_path=media_storage_path
        )
        self._snapshot_refs(params)
        n = getattr(params, "num_images_per_prompt", 1) if params else 1
        return MockVisualGenResult(
            request_id=req_id,
            image=self._maybe_batch(self._image, n),
            video=self._maybe_batch(self._video, n),
            audio=self._audio,
            should_fail=self._should_fail,
            generate_error=self._generate_error,
            on_finish=lambda: cleanup_reference_files(media_storage_path, str(req_id)),
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

        return VisualGenParams(**self.executor.default_generation_params)

    @property
    def extra_param_specs(self):
        """Stand-in for VisualGen.extra_param_specs — empty by default so
        every request ``extra_params`` key reaches the executor as
        ``unknown_extra_param`` (matches a pipeline with no model-specific
        knobs declared, like Flux or Wan 2.1)."""
        return self._extra_param_specs

    @property
    def model(self):
        """Stand-in for VisualGen.model — used by warn-on-set logic."""
        return self._model

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
        generate_error: Optional[BaseException] = None,
        on_finish=None,
    ):
        self.request_id = request_id
        self._image = image
        self._video = video
        self._audio = audio
        self._should_fail = should_fail
        # Engine-side failure surfaced through the result (capacity/client),
        # distinct from a coordinator preflight rejection.
        self._generate_error = generate_error
        self._on_finish = on_finish
        self._cleaned = False

    def _run_finish(self):
        # Terminal reference cleanup, run once (idempotent), mirroring the real
        # VisualGenResult so it fires on success and failure alike.
        if self._cleaned or self._on_finish is None:
            return
        self._cleaned = True
        try:
            self._on_finish()
        except Exception:
            pass

    def _resolve(self) -> VisualGenOutput:
        if self._generate_error is not None:
            raise self._generate_error
        if self._should_fail:
            raise RuntimeError("Async generation intentionally failed")
        return VisualGenOutput(
            request_id=self.request_id,
            image=self._image,
            video=self._video,
            audio=self._audio,
            metrics=_make_dummy_metrics(),
        )

    def __await__(self):
        return self.aresult().__await__()

    async def aresult(self, timeout=None):
        try:
            return self._resolve()
        finally:
            self._run_finish()

    def result(self, timeout=None):
        try:
            return self._resolve()
        finally:
            self._run_finish()


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


class _ThreadSettlingTestClient(TestClient):
    """TestClient that outwaits starlette's FileResponse reader thread.

    pytest-threadleak samples running threads at the end of a test's call
    phase. ``FileResponse`` reads the download on an anyio pool thread
    ("AnyIO worker thread") that is told to stop when the request's portal
    closes but exits a few milliseconds later -- inside the sampling window,
    so any file-shipping test in this module can fail as a phantom leak
    (pipeline #52355 did). Joining here, still in the call phase, waits out
    those milliseconds deterministically; stop is already queued, so the
    deadline is a guard, not an expected wait.
    """

    def request(self, *args, **kwargs):
        response = super().request(*args, **kwargs)
        deadline = time.monotonic() + 10.0
        for thread in threading.enumerate():
            if thread.name == "AnyIO worker thread":
                thread.join(max(0.0, deadline - time.monotonic()))
        return response


def _create_server(
    generator: MockVisualGen,
    model_name: str = "test-model",
) -> TestClient:
    """Instantiate an OpenAIServer for VISUAL_GEN with a mocked generator.

    The server detects VisualGen generators via ``_is_visual_gen_instance``
    (a sys.modules probe, so plain LLM serving never imports visual_gen) and
    caches the result in ``__init__``; patching the probe during construction
    makes it recognize our mock.
    """
    from tensorrt_llm.llmapi.disagg_utils import ServerRole
    from tensorrt_llm.serve.openai_server import OpenAIServer

    with patch(
        "tensorrt_llm.serve.openai_server._is_visual_gen_instance",
        return_value=True,
    ):
        server = OpenAIServer(
            generator=generator,
            model=model_name,
            tool_parser=None,
            server_role=ServerRole.VISUAL_GEN,
            metadata_server_cfg=None,
        )
    client = _ThreadSettlingTestClient(server.app)
    # Expose the mock so tests can assert captured generate() arguments.
    client.mock_gen = generator
    return client


def _create_async_client(generator: MockVisualGen, model_name: str = "test-model"):
    """Build the VISUAL_GEN server and return an ``httpx.AsyncClient`` over its
    ASGI app.

    The sync ``TestClient`` drives each request through a portal and does not run
    the event loop between requests, so a detached ``/v1/videos`` background task
    (and its offloaded encode) never progresses. Exercising the app on the
    caller's live loop lets the background task run to completion.
    """
    from tensorrt_llm.llmapi.disagg_utils import ServerRole
    from tensorrt_llm.serve.openai_server import OpenAIServer

    with patch(
        "tensorrt_llm.serve.openai_server._is_visual_gen_instance",
        return_value=True,
    ):
        server = OpenAIServer(
            generator=generator,
            model=model_name,
            tool_parser=None,
            server_role=ServerRole.VISUAL_GEN,
            metadata_server_cfg=None,
        )
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://testserver"
    )


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


@pytest_asyncio.fixture()
async def async_video_client(tmp_path):
    """Async httpx client over the video server — drives the async
    ``/v1/videos`` background task (incl. its offloaded encode) on a live loop.
    """
    gen = MockVisualGen(video_output=_make_dummy_video_tensor())
    os.environ["TRTLLM_MEDIA_STORAGE_PATH"] = str(tmp_path)
    client = _create_async_client(gen)
    try:
        yield client
    finally:
        await client.aclose()
        os.environ.pop("TRTLLM_MEDIA_STORAGE_PATH", None)


@pytest.fixture()
def action_video_client(tmp_path):
    """Video client whose pipeline declares a tensor-only extra param.

    Stands in for Cosmos3 action: the route must learn "this result needs a
    tensor payload" from the spec, never from the parameter's name.
    """
    from tensorrt_llm._torch.visual_gen.pipeline import ExtraParamSchema

    gen = MockVisualGen(video_output=_make_dummy_video_tensor())
    specs = {
        "action_mode": ExtraParamSchema(type="str", default=None, requires_tensor_output=True),
    }
    gen.executor.extra_param_specs = specs
    type(gen).extra_param_specs = property(lambda self: specs)
    os.environ["TRTLLM_MEDIA_STORAGE_PATH"] = str(tmp_path)
    client = _create_server(gen)
    yield client
    os.environ.pop("TRTLLM_MEDIA_STORAGE_PATH", None)
    del type(gen).extra_param_specs


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


@pytest.mark.parametrize(
    "endpoint,payload",
    [
        ("/v1/images/generations", {"prompt": "cat", "response_format": "path"}),
        (
            "/v1/videos/sync",
            {
                "prompt": "cat",
                "size": "64x64",
                "seconds": 1.0,
                "fps": 8,
                "response_format": "path",
            },
        ),
        (
            "/v1/videos",
            {
                "prompt": "cat",
                "size": "64x64",
                "seconds": 1.0,
                "fps": 8,
                "response_format": "path",
            },
        ),
    ],
)
def test_response_format_path_rejected_when_disabled(tmp_path, monkeypatch, endpoint, payload):
    """With ``TRTLLM_DISALLOW_LOCAL_MEDIA_PATH=1``, ``response_format='path'``
    is rejected with 400 on the image and video (sync + async) endpoints."""
    monkeypatch.setenv("TRTLLM_DISALLOW_LOCAL_MEDIA_PATH", "1")
    monkeypatch.setenv("TRTLLM_MEDIA_STORAGE_PATH", str(tmp_path))
    gen = MockVisualGen(
        image_output=_make_dummy_image_tensor(),
        video_output=_make_dummy_video_tensor(),
    )
    client = _create_server(gen)
    resp = client.post(endpoint, json=payload, headers={"content-type": "application/json"})
    assert resp.status_code == 400
    body = resp.json()
    _assert_llm_envelope(body, code=400)
    assert "path" in body["message"] and "disabled" in body["message"]


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

    def test_image_generation_server_timing_has_total(self, image_client):
        """The Server-Timing header carries generation, denoise, and ``total``
        (full server time; real wall-clock, so only checked > 0)."""
        resp = image_client.post(
            "/v1/images/generations",
            json={"prompt": "timing", "response_format": "b64_json", "size": "64x64"},
        )
        assert resp.status_code == 200
        assert _server_timing_ms(resp.headers, "generation") == 1250.0
        assert _server_timing_ms(resp.headers, "denoise") == 750.0
        assert _server_timing_ms(resp.headers, "total") > 0

    def test_image_generation_total_anchored_to_server_arrival(self, image_client, monkeypatch):
        """``total`` measures from the middleware's arrival stamp, not from a
        handler-local clock — this route is handed an already-parsed
        ``ImageGenerationRequest``, so stamping in the handler would silently
        exclude body parsing.

        Backdating only the middleware's clock leaves the handler's end
        reading on the real one, so ``total`` must absorb the full offset.
        """
        import tensorrt_llm.serve.responses_utils as _ru

        real = _ru.get_steady_clock_now_in_seconds
        monkeypatch.setattr(_ru, "get_steady_clock_now_in_seconds", lambda: real() - 5.0)

        resp = image_client.post(
            "/v1/images/generations",
            json={"prompt": "timing", "response_format": "b64_json", "size": "64x64"},
        )
        assert resp.status_code == 200
        assert _server_timing_ms(resp.headers, "total") >= 5000.0

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

    def test_image_generation_path_returns_output_path(self, image_client):
        """``response_format='path'`` writes each image to media storage and
        surfaces its on-disk path per ``data[]`` item; ``n>1`` fans out to one
        object per image (distinct path); ``url``/``b64_json`` stay unset."""
        resp = image_client.post(
            "/v1/images/generations",
            json={
                "prompt": "A dog",
                "response_format": "path",
                "n": 2,
            },
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data) == 2
        assert len({obj["path"] for obj in data}) == 2  # one distinct path per image
        for obj in data:
            assert obj["url"] is None and obj["b64_json"] is None
            assert obj["path"] is not None and os.path.exists(obj["path"])
        with open(data[0]["path"], "rb") as fh:
            assert fh.read().startswith(b"\x89PNG\r\n\x1a\n")

    def test_image_generation_pt_path(self, image_client):
        """Tensor formats under ``response_format='path'`` persist each
        per-item payload and return its on-disk path."""
        resp = image_client.post(
            "/v1/images/generations",
            json={
                "prompt": "Tensor dog",
                "response_format": "path",
                "format": "pt",
            },
        )
        assert resp.status_code == 200
        obj = resp.json()["data"][0]
        assert obj["path"] is not None and obj["url"] is None and obj["b64_json"] is None
        assert os.path.exists(obj["path"])
        with open(obj["path"], "rb") as fh:
            loaded = torch.load(BytesIO(fh.read()), weights_only=True)
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
    """``/v1/images/edits`` support is gated by the loaded visual model."""

    def _client(
        self,
        tmp_path,
        monkeypatch,
        *,
        image_output: Optional[torch.Tensor] = None,
        extra_param_specs: Optional[dict] = None,
        model: str = "Qwen/Qwen-Image-Layered",
        should_fail: bool = False,
        supports_image_edit: bool = True,
    ):
        gen = MockVisualGen(
            image_output=image_output if image_output is not None else _make_dummy_image_tensor(),
            extra_param_specs=extra_param_specs,
            model=model,
            should_fail=should_fail,
            supports_image_edit=supports_image_edit,
        )
        monkeypatch.setenv("TRTLLM_MEDIA_STORAGE_PATH", str(tmp_path))
        return _create_server(gen, model_name=model), gen

    @pytest.mark.parametrize(
        ("model", "supports_image_edit", "expected_status"),
        [
            ("not-a-canonical-edit-model-id", True, 200),
            ("Qwen/Qwen-Image-Layered", False, 501),
        ],
    )
    def test_image_edit_support_uses_loaded_pipeline_capability(
        self, tmp_path, monkeypatch, model, supports_image_edit, expected_status
    ):
        client, gen = self._client(
            tmp_path,
            monkeypatch,
            model=model,
            supports_image_edit=supports_image_edit,
        )

        resp = client.post(
            "/v1/images/edits",
            json={
                "prompt": "Make it red",
                "image": _b64_white_png_1x1(),
                "response_format": "b64_json",
            },
        )

        assert resp.status_code == expected_status
        if expected_status == 501:
            assert gen.last_params is None

    def test_image_edit_accepts_json_base64_image(self, tmp_path, monkeypatch):
        """JSON edit requests materialize inputs and map OpenAI-shaped fields."""
        client, gen = self._client(
            tmp_path,
            monkeypatch,
            image_output=_make_dummy_image_tensor(4, 4),
        )

        resp = client.post(
            "/v1/images/edits",
            content=json.dumps(
                {
                    "prompt": "split layers",
                    "image": _b64_white_png_1x1(),
                    "n": 2,
                    "output_format": "webp",
                    "response_format": "b64_json",
                }
            ),
            headers={"content-type": "Application/JSON"},
        )

        assert resp.status_code == 200
        assert str(gen.last_params.image).startswith(str(tmp_path))
        assert not os.path.exists(gen.last_params.image)
        assert gen.last_params.num_images_per_prompt == 2
        body = resp.json()
        assert body["output_format"] == "webp"
        assert body["size"] == "4x4"
        assert len(body["data"]) == 2

    @pytest.mark.parametrize(
        ("size", "expected_dimensions"),
        [
            ("auto", (None, None)),
            ("32x48", (32, 48)),
        ],
    )
    def test_image_edit_auto_size_allows_reference_size_derivation(
        self, tmp_path, monkeypatch, size, expected_dimensions
    ):
        client, gen = self._client(tmp_path, monkeypatch)

        resp = client.post(
            "/v1/images/edits",
            json={
                "prompt": "use reference dimensions",
                "image": _b64_white_png_1x1(),
                "size": size,
                "response_format": "b64_json",
            },
        )

        assert resp.status_code == 200
        assert (gen.last_params.width, gen.last_params.height) == expected_dimensions

    @pytest.mark.threadleak(enabled=False)  # FileResponse spawns AnyIO worker threads
    def test_image_edit_default_url_returns_fetchable_output(self, tmp_path, monkeypatch):
        """The default edit response writes a fetchable image content URL."""
        client, gen = self._client(tmp_path, monkeypatch)

        resp = client.post(
            "/v1/images/edits",
            json={
                "prompt": "split layers",
                "image": _b64_white_png_1x1(),
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        url = body["data"][0]["url"]
        assert "/v1/images/" in url and "/content" in url
        assert str(gen.last_params.image).startswith(str(tmp_path))
        assert not os.path.exists(gen.last_params.image)

        path = url.split("//", 1)[-1].split("/", 1)[1]
        content = client.get("/" + path)
        assert content.status_code == 200
        assert content.content.startswith(b"\x89PNG\r\n\x1a\n")
        assert content.headers["content-type"] == "image/png"

    def test_image_edit_server_timing_has_total(self, tmp_path, monkeypatch):
        """The edit route reports ``total`` too (real wall-clock, so > 0)."""
        client, _ = self._client(tmp_path, monkeypatch)

        resp = client.post(
            "/v1/images/edits",
            json={
                "prompt": "timing",
                "image": _b64_white_png_1x1(),
                "response_format": "b64_json",
            },
        )

        assert resp.status_code == 200
        assert _server_timing_ms(resp.headers, "generation") == 1250.0
        assert _server_timing_ms(resp.headers, "denoise") == 750.0
        assert _server_timing_ms(resp.headers, "total") > 0

    def test_image_edit_response_format_path_returns_on_disk_path(self, tmp_path, monkeypatch):
        """``response_format='path'`` returns the server-side output path
        instead of a fetchable URL, matching ``/v1/images/generations``."""
        client, _ = self._client(tmp_path, monkeypatch)

        resp = client.post(
            "/v1/images/edits",
            json={
                "prompt": "split layers",
                "image": _b64_white_png_1x1(),
                "response_format": "path",
            },
        )

        assert resp.status_code == 200
        item = resp.json()["data"][0]
        assert item["url"] is None
        assert item["path"].startswith(str(tmp_path))
        assert os.path.exists(item["path"])

    def test_image_edit_response_format_path_rejected_when_disabled(self, tmp_path, monkeypatch):
        """``TRTLLM_DISALLOW_LOCAL_MEDIA_PATH=1`` gates the edit route too,
        so the path transport cannot leak server-side paths on shared
        deployments."""
        client, _ = self._client(tmp_path, monkeypatch)
        monkeypatch.setenv("TRTLLM_DISALLOW_LOCAL_MEDIA_PATH", "1")

        resp = client.post(
            "/v1/images/edits",
            json={
                "prompt": "split layers",
                "image": _b64_white_png_1x1(),
                "response_format": "path",
            },
        )

        assert resp.status_code == 400
        body = resp.json()
        _assert_llm_envelope(body, code=400)
        assert "path" in body["message"] and "disabled" in body["message"]

    def test_image_edit_rejects_json_array_body(self, tmp_path, monkeypatch):
        """Non-object JSON bodies are client errors, not server errors."""
        client, gen = self._client(tmp_path, monkeypatch)

        resp = client.post(
            "/v1/images/edits",
            content=json.dumps(
                [
                    {
                        "prompt": "split layers",
                        "image": _b64_white_png_1x1(),
                    }
                ]
            ),
            headers={"content-type": "application/json"},
        )

        assert resp.status_code == 400
        assert "must be an object" in resp.json()["message"]
        assert gen.last_params is None

    def test_image_edit_rejects_file_extra_params(self, tmp_path, monkeypatch):
        """Multipart extra_params must be a JSON string field."""
        client, gen = self._client(tmp_path, monkeypatch)

        image_bytes = BytesIO(base64.b64decode(_b64_white_png_1x1()))
        resp = client.post(
            "/v1/images/edits",
            data={
                "prompt": "split layers",
                "response_format": "b64_json",
            },
            files={
                "image": ("input.png", image_bytes, "image/png"),
                "extra_params": ("extra.json", b"{}", "application/json"),
            },
        )

        assert resp.status_code == 400
        assert "extra_params" in resp.json()["message"]
        assert gen.last_params is None

    def test_image_edit_rejects_empty_image_list(self, tmp_path, monkeypatch):
        """Empty image lists fail request validation before pipeline dispatch."""
        client, gen = self._client(tmp_path, monkeypatch)

        resp = client.post(
            "/v1/images/edits",
            json={
                "prompt": "split layers",
                "image": [],
                "response_format": "b64_json",
            },
        )

        assert resp.status_code == 422
        _assert_llm_envelope(resp.json(), code=422, message_contains="image")
        assert gen.last_params is None
        assert list(tmp_path.iterdir()) == []

    def test_image_edit_rejects_non_image_base64_input(self, tmp_path, monkeypatch):
        """Decoded image-edit bytes must be a supported image before disk write."""
        client, gen = self._client(tmp_path, monkeypatch)

        resp = client.post(
            "/v1/images/edits",
            json={
                "prompt": "split layers",
                "image": base64.b64encode(b"not an image").decode("utf-8"),
                "response_format": "b64_json",
            },
        )

        assert resp.status_code == 400
        _assert_llm_envelope(
            resp.json(),
            code=400,
            message_contains="image edit input is not a PNG/JPEG image",
        )
        assert gen.last_params is None
        assert list(tmp_path.iterdir()) == []

    def test_image_edit_rejects_non_image_upload_input(self, tmp_path, monkeypatch):
        """Multipart image-edit bytes are sniffed before materialization."""
        client, gen = self._client(tmp_path, monkeypatch)

        resp = client.post(
            "/v1/images/edits",
            data={
                "prompt": "split layers",
                "response_format": "b64_json",
            },
            files={"image": ("input.png", BytesIO(b"not an image"), "image/png")},
        )

        assert resp.status_code == 400
        _assert_llm_envelope(
            resp.json(),
            code=400,
            message_contains="image edit input is not a PNG/JPEG image",
        )
        assert gen.last_params is None
        assert list(tmp_path.iterdir()) == []

    def test_image_edit_rejects_mask_with_clear_error(self, tmp_path, monkeypatch):
        """Mask is OpenAI-shaped but not implemented by TRTLLM image edit yet."""
        client, gen = self._client(tmp_path, monkeypatch)

        resp = client.post(
            "/v1/images/edits",
            json={
                "prompt": "split layers",
                "image": _b64_white_png_1x1(),
                "mask": _b64_white_png_1x1(),
                "response_format": "b64_json",
            },
        )

        assert resp.status_code == 400
        assert "mask input is not supported" in resp.json()["message"]
        assert gen.last_params is None

    def test_image_edit_rejects_too_many_input_images(self, tmp_path, monkeypatch):
        """Input image count is capped before files are materialized."""
        client, gen = self._client(tmp_path, monkeypatch)

        resp = client.post(
            "/v1/images/edits",
            json={
                "prompt": "split layers",
                "image": [_b64_white_png_1x1()] * 17,
                "response_format": "b64_json",
            },
        )

        assert resp.status_code == 400
        assert "at most 16 input images" in resp.json()["message"]
        assert gen.last_params is None
        assert list(tmp_path.iterdir()) == []

    def test_image_edit_allows_max_input_images_without_output_fanout(self, tmp_path, monkeypatch):
        """Multiple edit inputs are joint conditioning, not output fan-out."""
        client, gen = self._client(tmp_path, monkeypatch)

        resp = client.post(
            "/v1/images/edits",
            json={
                "prompt": "split layers",
                "image": [_b64_white_png_1x1()] * 16,
                "response_format": "b64_json",
            },
        )

        assert resp.status_code == 200
        assert len(gen.last_params.image) == 16
        assert len(resp.json()["data"]) == 1
        assert list(tmp_path.iterdir()) == []

    def test_image_edit_rejects_excessive_output_fanout(self, tmp_path, monkeypatch):
        """Layered output fan-out is capped before files are materialized."""
        from tensorrt_llm._torch.visual_gen.pipeline import ExtraParamSchema

        client, gen = self._client(
            tmp_path,
            monkeypatch,
            extra_param_specs={
                "layers": ExtraParamSchema(type="int", default=4, range=(1, 16)),
            },
        )

        resp = client.post(
            "/v1/images/edits",
            json={
                "prompt": "split layers",
                "image": _b64_white_png_1x1(),
                "n": 5,
                "extra_params": {"layers": 16},
                "response_format": "b64_json",
            },
        )

        assert resp.status_code == 400
        assert "at most 64 output images" in resp.json()["message"]
        assert gen.last_params is None
        assert list(tmp_path.iterdir()) == []

    def test_image_edit_rejects_oversized_base64_image_before_decode(self, tmp_path, monkeypatch):
        """Base64 image size is capped before allocating decoded bytes."""
        from tensorrt_llm.serve import visual_gen_utils

        client, gen = self._client(tmp_path, monkeypatch)
        monkeypatch.setattr(visual_gen_utils, "IMAGE_EDIT_MAX_IMAGE_BYTES", 8)
        monkeypatch.setattr(
            visual_gen_utils.base64,
            "b64decode",
            lambda *args, **kwargs: pytest.fail("oversized payload was decoded"),
        )

        resp = client.post(
            "/v1/images/edits",
            json={
                "prompt": "split layers",
                "image": "A" * 13,
                "response_format": "b64_json",
            },
        )

        assert resp.status_code == 400
        assert "per-image byte limit" in resp.json()["message"]
        assert gen.last_params is None
        assert list(tmp_path.iterdir()) == []

    def test_image_edit_cleans_inputs_when_generation_fails(self, tmp_path, monkeypatch):
        """Temporary edit inputs are removed even when generation raises."""
        client, gen = self._client(
            tmp_path,
            monkeypatch,
            should_fail=True,
        )

        resp = client.post(
            "/v1/images/edits",
            json={
                "prompt": "split layers",
                "image": _b64_white_png_1x1(),
                "response_format": "b64_json",
            },
        )

        assert resp.status_code == 500
        assert gen.last_params is not None
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize(
        "image_value",
        [
            "/tmp/server-local-image.png",
            "file:///tmp/server-local-image.png",
            "https://example.com/server-local-image.png",
        ],
    )
    def test_image_edit_rejects_json_path_or_url_image(self, tmp_path, monkeypatch, image_value):
        """Serving image-edit input strings must be base64, not server paths or URLs."""
        client, gen = self._client(tmp_path, monkeypatch)

        resp = client.post(
            "/v1/images/edits",
            json={
                "prompt": "split layers",
                "image": image_value,
                "response_format": "b64_json",
            },
        )

        assert resp.status_code == 400
        assert gen.last_params is None


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
# POST /v1/videos/sync  (synchronous)
# =========================================================================


@pytest.mark.threadleak(enabled=False)  # FileResponse spawns AnyIO worker threads
class TestVideoGenerationSync:
    def test_basic_sync_video_generation(self, video_client):
        resp = video_client.post(
            "/v1/videos/sync",
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

    def test_sync_video_server_timing_has_total(self, video_client):
        """The sync Server-Timing header carries generation, denoise, and the
        new ``total`` (full server time; real wall-clock, so only checked > 0)."""
        resp = video_client.post(
            "/v1/videos/sync",
            json={
                "prompt": "timing",
                "size": "32x32",
                "seconds": 1.0,
                "fps": 8,
                "format": "avi",
            },
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 200
        assert _server_timing_ms(resp.headers, "generation") == 1250.0
        assert _server_timing_ms(resp.headers, "denoise") == 750.0
        assert _server_timing_ms(resp.headers, "total") > 0
        assert len(resp.content) > 0

    def test_deprecated_generations_alias_routes_to_sync(self, video_client):
        """The pre-rename /v1/videos/generations route is kept as a deprecated
        alias of /v1/videos/sync (upstream back-compat) — same handler, so it
        returns the video bytes rather than 404/405."""
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
        assert len(resp.content) > 0

    @pytest.mark.parametrize("removed", ["url", "b64_json"])
    def test_removed_response_format_names_replacement(self, video_client, removed):
        """Legacy video response_format values (url/b64_json) are rejected with
        a 422 whose message names the replacement, not the generic
        "Input should be 'file' or 'path'"."""
        resp = video_client.post(
            "/v1/videos/sync",
            json={
                "prompt": "x",
                "size": "64x64",
                "seconds": 1.0,
                "fps": 8,
                "response_format": removed,
            },
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 422
        body = resp.json()
        _assert_llm_envelope(body, code=422, message_contains=removed)
        assert "removed" in body["message"] and "file" in body["message"], body["message"]

    def test_sync_video_generation_with_params(self, video_client):
        resp = video_client.post(
            "/v1/videos/sync",
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
        """Multipart sync request with a real ``image_reference`` file."""
        ref_path = tmp_path / "ref.png"
        Image.new("RGB", (4, 4), (64, 64, 64)).save(str(ref_path))
        with open(ref_path, "rb") as f:
            resp = video_client.post(
                "/v1/videos/sync",
                data={
                    "prompt": "Mountain sunrise",
                    "size": "64x64",
                    "seconds": "1.0",
                    "fps": "8",
                },
                files={"image_reference": ("ref.png", f, "image/png")},
            )
        assert resp.status_code == 200
        assert len(resp.content) > 0

    def test_sync_video_generation_multipart_with_reference(self, video_client, tmp_path):
        # Create a dummy reference image file
        ref_path = tmp_path / "ref.png"
        Image.new("RGB", (4, 4), (128, 128, 128)).save(str(ref_path))

        with open(ref_path, "rb") as f:
            resp = video_client.post(
                "/v1/videos/sync",
                data={
                    "prompt": "Animate this image",
                    "size": "64x64",
                    "seconds": "1.0",
                    "fps": "8",
                },
                files={"image_reference": ("ref.png", f, "image/png")},
            )
        assert resp.status_code == 200
        assert len(resp.content) > 0

        # image_reference is materialized to media storage and passed through as
        # a MediaRef carrying the filesystem path.
        params = video_client.mock_gen.last_params
        ref_path = params.image_reference[0].content
        assert isinstance(ref_path, str)
        assert ref_path.endswith("_image_ref_0")
        # The materialized reference is input-only and is cleaned up once the
        # request finishes, so it must not linger in media storage.
        assert not os.path.exists(ref_path)

    def test_sync_video_generation_multipart_with_video_reference(self, video_client):
        """A ``video_reference`` upload is persisted byte-identical (V2V) — the
        serve never decodes video; the worker demuxes/NVDEC-decodes the stored
        file. A checked-in H.264/MP4 fixture drives the boundary directly.
        """
        payload = _V2V_FIXTURE_MP4.read_bytes()
        with open(_V2V_FIXTURE_MP4, "rb") as f:
            resp = video_client.post(
                "/v1/videos/sync",
                data={
                    "prompt": "Continue the same scene",
                    "size": "64x64",
                    "seconds": "1.0",
                    "fps": "8",
                },
                files={"video_reference": ("ref.mp4", f, "video/mp4")},
            )
        assert resp.status_code == 200
        assert len(resp.content) > 0

        # Video conditioning arrives as a MediaRef holding a stored path; no
        # image_reference is set, and the encoded bytes were persisted
        # byte-identical (snapshotted at generation time, since the route cleans
        # the reference up once the request finishes).
        params = video_client.mock_gen.last_params
        assert params.image_reference is None
        ref_path = params.video_reference[0].content
        assert video_client.mock_gen.last_ref_bytes[ref_path] == payload
        assert not os.path.exists(ref_path)

    def test_sync_video_generation_undecodable_reference_400(self, video_client):
        """Content matching no image or video container signature is rejected
        at the boundary."""
        resp = video_client.post(
            "/v1/videos/sync",
            data={"prompt": "x"},
            files={"image_reference": ("doc.txt", BytesIO(b"not media"), "text/plain")},
        )
        assert resp.status_code == 400
        assert "not a recognized image" in resp.text

    def test_sync_video_failure(self, failing_client):
        resp = failing_client.post(
            "/v1/videos/sync",
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
            "/v1/videos/sync",
            json={"prompt": "null video", "size": "64x64", "seconds": 1.0, "fps": 8},
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 500
        os.environ.pop("TRTLLM_MEDIA_STORAGE_PATH", None)

    def test_sync_video_capacity_failure_is_503(self, tmp_path, monkeypatch):
        """A valid request the deployment cannot fit is retryable (503).

        The engine signals capacity with ``MemoryError`` — a built-in, so the
        error contract carries no VisualGen-specific exception type — and the
        route must not fold it into the generic 500.
        """
        gen = MockVisualGen(
            generate_error=MemoryError("Out of device memory while preparing generation")
        )
        monkeypatch.setenv("TRTLLM_MEDIA_STORAGE_PATH", str(tmp_path))
        client = _create_server(gen)
        resp = client.post(
            "/v1/videos/sync",
            json={"prompt": "big", "size": "64x64", "seconds": 1.0, "fps": 8},
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 503
        assert "Out of device memory" in resp.text

    def test_sync_video_client_failure_is_400(self, tmp_path, monkeypatch):
        """Engine-side client errors stay 400, distinct from capacity.

        The detail rides in the message — that is what a primitive exception
        type buys instead of a taxonomy — so it must reach the client.
        """
        gen = MockVisualGen(generate_error=ValueError("reference has no decodable frames"))
        monkeypatch.setenv("TRTLLM_MEDIA_STORAGE_PATH", str(tmp_path))
        client = _create_server(gen)
        resp = client.post(
            "/v1/videos/sync",
            json={"prompt": "bad ref", "size": "64x64", "seconds": 1.0, "fps": 8},
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400
        assert "no decodable frames" in resp.text

    def test_sync_video_unsupported_content_type(self, video_client):
        resp = video_client.post(
            "/v1/videos/sync",
            content=b"some raw bytes",
            headers={"content-type": "text/plain"},
        )
        assert resp.status_code == 400

    def test_sync_video_missing_prompt_json(self, video_client):
        """Missing required ``prompt`` surfaces the visual-gen 422 envelope."""
        resp = video_client.post(
            "/v1/videos/sync",
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
            "/v1/videos/sync",
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
            "/v1/videos/sync",
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
            "/v1/videos/sync",
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

    @pytest.mark.threadleak(enabled=False)  # offloaded encode uses a worker thread
    @pytest.mark.asyncio
    async def test_async_video_status_transitions_generating_then_postprocessing(
        self, async_video_client, monkeypatch
    ):
        """queued -> generating -> postprocessing -> completed, in order, so
        clients can detect when generation finishes before postprocessing."""
        seen = []
        original_upsert = VIDEO_STORE.upsert

        async def _spy_upsert(video_id, job):
            seen.append(job.status)
            return await original_upsert(video_id, job)

        monkeypatch.setattr(VIDEO_STORE, "upsert", _spy_upsert)

        resp = await async_video_client.post(
            "/v1/videos",
            json={"prompt": "lifecycle", "size": "32x32", "seconds": 1.0, "fps": 8},
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 202
        video_id = resp.json()["id"]

        status = await _adrive_job_to_completion(async_video_client, video_id)
        assert status == "completed"

        assert "generating" in seen and "postprocessing" in seen
        assert seen.index("generating") < seen.index("postprocessing") < seen.index("completed")

    @pytest.mark.threadleak(enabled=False)  # offloaded encode uses a worker thread
    @pytest.mark.asyncio
    async def test_async_postprocessing_state_observable_during_encode(
        self, async_video_client, monkeypatch
    ):
        """The encode is offloaded to a thread, so the event loop stays
        responsive and a poll observes ``postprocessing`` while the file is
        written — not just ``generating`` then ``completed``."""
        import threading

        release = threading.Event()
        original_save = VisualGenOutput.save

        def _blocking_save(self, *args, **kwargs):
            # Runs in the executor thread; hold until the test sees the state.
            release.wait(timeout=5)
            return original_save(self, *args, **kwargs)

        monkeypatch.setattr(VisualGenOutput, "save", _blocking_save)

        resp = await async_video_client.post(
            "/v1/videos",
            json={
                "prompt": "observe postprocessing",
                "size": "32x32",
                "seconds": 1.0,
                "fps": 8,
                "format": "auto",
            },
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 202
        video_id = resp.json()["id"]

        observed = None
        deadline = time.time() + 5
        while time.time() < deadline:
            poll = await async_video_client.get(f"/v1/videos/{video_id}")
            observed = poll.json().get("status")
            if observed in ("postprocessing", "completed", "failed"):
                break
            await asyncio.sleep(0.02)
        release.set()  # never leave the encoder blocked
        assert observed == "postprocessing", (
            f"GET never observed 'postprocessing' (saw {observed!r})"
        )

    def test_async_video_multipart(self, video_client, tmp_path):
        """Multipart async request with a real ``image_reference`` file."""
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
                files={"image_reference": ("ref.png", f, "image/png")},
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
        # Status-only listing: the internal path fields never appear on the wire.
        for item in data["data"]:
            assert "output_path" not in item and "output_paths" not in item


# =========================================================================
# GET /v1/videos/{video_id}  (metadata)
# =========================================================================


class TestGetVideoMetadata:
    @pytest.mark.threadleak(enabled=False)  # offloaded encode uses a worker thread
    @pytest.mark.asyncio
    async def test_get_video_metadata_success(self, async_video_client):
        create_resp = await async_video_client.post(
            "/v1/videos",
            json={"prompt": "Space walk", "size": "64x64", "seconds": 1.0, "fps": 8},
            headers={"content-type": "application/json"},
        )
        video_id = create_resp.json()["id"]

        # Drive to completion so output_path/output_paths are populated on the
        # job, then confirm the status endpoint returns status only (no leak).
        await _adrive_job_to_completion(async_video_client, video_id)

        resp = await async_video_client.get(f"/v1/videos/{video_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == video_id
        assert data["object"] == "video"
        assert data["prompt"] == "Space walk"
        assert data["status"] == "completed"
        # Status-only: the internal server path(s) are not leaked here.
        assert "output_path" not in data and "output_paths" not in data

    def test_get_video_metadata_not_found(self, video_client):
        resp = video_client.get("/v1/videos/video_nonexistent")
        assert resp.status_code == 404


# =========================================================================
# GET /v1/videos/{video_id}/content  (download)
# =========================================================================


@pytest.mark.threadleak(enabled=False)  # FileResponse spawns AnyIO worker threads
class TestGetVideoContent:
    def _insert_video_job(self, video_id: str, status: str = "queued"):
        job = VideoJob(
            created_at=int(time.time()),
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

    @pytest.mark.parametrize("status", ["generating", "postprocessing"])
    def test_get_video_content_not_ready_in_flight(self, tmp_path, status):
        """A generating/postprocessing job is not downloadable yet → 400."""
        gen = MockVisualGen(video_output=_make_dummy_video_tensor())
        os.environ["TRTLLM_MEDIA_STORAGE_PATH"] = str(tmp_path)
        client = _create_server(gen)

        video_id = f"video_{status}"
        self._insert_video_job(video_id, status=status)

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
                "/v1/videos/sync",
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
    ``response_format='file'`` and as a server-side path JSON under
    ``response_format='path'``. The async route persists the
    payload to media storage; ``GET /v1/videos/{id}/content`` serves
    the file with ``application/octet-stream``."""

    def _post_sync(self, video_client, fmt: str, response_format: str):
        return video_client.post(
            "/v1/videos/sync",
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
    def test_sync_tensor_file_returns_file_with_correct_suffix(self, video_audio_client, fmt):
        resp = self._post_sync(video_audio_client, fmt, "file")
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
    def test_sync_tensor_path_returns_readable_output_path(self, video_audio_client, fmt):
        resp = self._post_sync(video_audio_client, fmt, "path")
        assert resp.status_code == 200
        # path responses carry the Server-Timing metrics too.
        _assert_visual_gen_server_timing(resp.headers)
        data = resp.json()
        assert set(data) >= {"id", "output_path"}
        # Co-located client reads the returned server-side path directly.
        assert os.path.exists(data["output_path"])
        with open(data["output_path"], "rb") as fh:
            raw = fh.read()
        if fmt == "safetensors":
            from safetensors.torch import load as load_safetensors

            loaded = load_safetensors(raw)
        else:
            loaded = torch.load(BytesIO(raw), weights_only=True)
        assert "video" in loaded

    @pytest.mark.parametrize("fmt", ["safetensors", "pt"])
    def test_async_tensor_persists_and_serves(self, video_audio_client, fmt, tmp_path):
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
        _drive_job_to_completion(client, video_id)

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
class TestVideoEncoderResponse:
    """The sync video route's encoder branch (``mp4``/``avi``/``auto``)
    honors ``response_format='path'`` by returning the server-side output
    path(s) as JSON; ``response_format='file'`` keeps the
    ``FileResponse`` download."""

    def test_sync_encoder_path_returns_output_path(self, video_client):
        resp = video_client.post(
            "/v1/videos/sync",
            json={
                "prompt": "encoded path",
                "size": "32x32",
                "seconds": 1.0,
                "fps": 8,
                "format": "avi",
                "response_format": "path",
            },
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 200
        # path responses carry the Server-Timing metrics too.
        _assert_visual_gen_server_timing(resp.headers)
        body = resp.json()
        assert set(body) >= {"id", "output_path"}
        # The returned server-side path points at non-empty encoded bytes.
        assert os.path.exists(body["output_path"])
        assert os.path.getsize(body["output_path"]) > 0

    def test_sync_encoder_file_keeps_file_response(self, video_client):
        resp = video_client.post(
            "/v1/videos/sync",
            json={
                "prompt": "encoded file",
                "size": "32x32",
                "seconds": 1.0,
                "fps": 8,
                "format": "avi",
                "response_format": "file",
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
            "/v1/videos/sync",
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
            "/v1/videos/sync",
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
        video_client.mock_gen.executor.default_generation_params.pop("frame_rate", None)
        resp = video_client.post(
            "/v1/videos/sync",
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
            "/v1/videos/sync",
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
            "/v1/videos/sync",
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
            "/v1/videos/sync",
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
            "/v1/videos/sync",
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
class TestAsyncVideoTransport:
    """``POST /v1/videos`` persists the requested ``response_format`` on
    the queued job. ``GET /v1/videos/{id}/content`` honors it:
    ``file`` (or unset) returns a ``FileResponse`` download;
    ``path`` returns a JSON envelope with the server-side output path(s)."""

    @pytest.mark.asyncio
    async def test_async_path_returned_at_get_content(self, async_video_client):
        resp = await async_video_client.post(
            "/v1/videos",
            json={
                "prompt": "async path",
                "size": "32x32",
                "seconds": 1.0,
                "fps": 8,
                "format": "avi",
                "response_format": "path",
            },
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 202
        job = resp.json()
        assert job["response_format"] == "path"

        status = await _adrive_job_to_completion(async_video_client, job["id"])
        assert status == "completed"

        content = await async_video_client.get(f"/v1/videos/{job['id']}/content")
        assert content.status_code == 200
        body = content.json()
        assert set(body) >= {"id", "output_path"}
        assert body["id"] == job["id"]
        # The returned server-side path points at non-empty bytes.
        assert os.path.exists(body["output_path"])
        assert os.path.getsize(body["output_path"]) > 0

    @pytest.mark.asyncio
    async def test_async_content_server_timing_has_total(self, async_video_client):
        """`/content` carries the same Server-Timing header as the sync route,
        rebuilt from the timings the background task stored on the job."""
        resp = await async_video_client.post(
            "/v1/videos",
            json={
                "prompt": "timing",
                "size": "32x32",
                "seconds": 1.0,
                "fps": 8,
                "format": "avi",
            },
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 202
        video_id = resp.json()["id"]
        assert await _adrive_job_to_completion(async_video_client, video_id) == "completed"

        content = await async_video_client.get(f"/v1/videos/{video_id}/content")
        assert content.status_code == 200
        assert _server_timing_ms(content.headers, "generation") == 1250.0
        assert _server_timing_ms(content.headers, "denoise") == 750.0
        assert _server_timing_ms(content.headers, "total") > 0

    @pytest.mark.asyncio
    async def test_async_total_anchored_to_server_arrival(self, async_video_client, monkeypatch):
        """The async path round-trips the arrival stamp through
        ``VideoJob.request_started`` and closes ``total`` out in a background
        task, so the stamp and the end reading must stay on one clock.
        """
        import tensorrt_llm.serve.responses_utils as _ru

        real = _ru.get_steady_clock_now_in_seconds
        monkeypatch.setattr(_ru, "get_steady_clock_now_in_seconds", lambda: real() - 5.0)

        resp = await async_video_client.post(
            "/v1/videos",
            json={
                "prompt": "timing",
                "size": "32x32",
                "seconds": 1.0,
                "fps": 8,
                "format": "avi",
            },
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 202
        video_id = resp.json()["id"]
        assert await _adrive_job_to_completion(async_video_client, video_id) == "completed"

        content = await async_video_client.get(f"/v1/videos/{video_id}/content")
        assert content.status_code == 200
        assert _server_timing_ms(content.headers, "total") >= 5000.0

    @pytest.mark.asyncio
    async def test_async_file_still_returns_file_response(self, async_video_client):
        """Default and explicit ``response_format='file'`` keep the
        existing ``FileResponse`` behavior."""
        resp = await async_video_client.post(
            "/v1/videos",
            json={
                "prompt": "async file",
                "size": "32x32",
                "seconds": 1.0,
                "fps": 8,
                "format": "avi",
                "response_format": "file",
            },
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 202
        job = resp.json()
        assert job["response_format"] == "file"

        await _adrive_job_to_completion(async_video_client, job["id"])
        content = await async_video_client.get(f"/v1/videos/{job['id']}/content")
        assert content.status_code == 200
        # AVI FileResponse carries ``video/x-msvideo``; the path
        # branch would have set ``application/json``.
        assert content.headers["content-type"] == "video/x-msvideo"


class TestTensorOnlyFormatResolution:
    """A request whose result an encoder cannot carry must not be served as video."""

    @staticmethod
    def _post(client, **body):
        return client.post(
            "/v1/videos/generations",
            json={"prompt": "pick up the block", "size": "64x64", "seconds": 1.0, "fps": 8, **body},
            headers={"content-type": "application/json"},
        )

    def test_auto_resolves_to_tensor_payload(self, action_video_client):
        resp = self._post(action_video_client, extra_params={"action_mode": "policy"})
        assert resp.status_code == 200
        # 'auto' would otherwise have produced an encoded video, silently
        # dropping the modality the request was made for.
        assert resp.headers["content-type"] == "application/octet-stream"
        assert resp.headers["content-disposition"].endswith('.safetensors"')

    def test_explicit_tensor_format_passes_through(self, action_video_client):
        resp = self._post(action_video_client, format="pt", extra_params={"action_mode": "policy"})
        assert resp.status_code == 200
        assert resp.headers["content-disposition"].endswith('.pt"')

    @pytest.mark.parametrize("fmt", ["mp4", "avi"])
    def test_explicit_encoder_format_is_rejected(self, action_video_client, fmt):
        """The caller stated two incompatible things; guessing either way is wrong."""
        resp = self._post(action_video_client, format=fmt, extra_params={"action_mode": "policy"})
        assert resp.status_code == 400
        body = resp.json()
        _assert_llm_envelope(body, code=400, message_contains="action_mode")
        # The message must name the way out, not just the refusal.
        assert "safetensors" in body["message"]

    def test_untriggered_request_keeps_encoder_default(self, action_video_client):
        """No tensor-only param set -> ordinary video request, unchanged."""
        resp = self._post(action_video_client)
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("video/")


class TestTensorOnlyFormatRule:
    """Unit coverage of the resolution rule itself, without a server."""

    @staticmethod
    def _specs(**flags):
        from tensorrt_llm._torch.visual_gen.pipeline import ExtraParamSchema

        return {
            name: ExtraParamSchema(type="str", default=None, requires_tensor_output=flag)
            for name, flag in flags.items()
        }

    def test_no_specs_is_a_noop(self):
        from tensorrt_llm.serve.openai_video_routes import _resolve_tensor_only_format

        assert _resolve_tensor_only_format("auto", {"action_mode": "policy"}, None) == "auto"
        assert _resolve_tensor_only_format("auto", None, self._specs(a=True)) == "auto"

    def test_only_a_declared_param_triggers(self):
        from tensorrt_llm.serve.openai_video_routes import _resolve_tensor_only_format

        specs = self._specs(action_mode=True, stg_scale=False)
        # a non-declaring param must not force a tensor payload
        assert _resolve_tensor_only_format("auto", {"stg_scale": 2.0}, specs) == "auto"
        assert (
            _resolve_tensor_only_format("auto", {"action_mode": "policy"}, specs) == "safetensors"
        )

    def test_null_value_does_not_trigger(self):
        from tensorrt_llm.serve.openai_video_routes import _resolve_tensor_only_format

        specs = self._specs(action_mode=True)
        assert _resolve_tensor_only_format("auto", {"action_mode": None}, specs) == "auto"

    def test_encoder_format_raises_naming_the_parameter(self):
        from tensorrt_llm.serve.openai_video_routes import _resolve_tensor_only_format

        specs = self._specs(action_mode=True)
        with pytest.raises(ValueError, match="action_mode"):
            _resolve_tensor_only_format("mp4", {"action_mode": "policy"}, specs)
