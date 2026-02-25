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

from tensorrt_llm._torch.visual_gen.output import MediaOutput
from tensorrt_llm.serve.media_storage import MediaStorage
from tensorrt_llm.serve.openai_protocol import VideoJob
from tensorrt_llm.serve.visual_gen_utils import VIDEO_STORE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Mock VisualGen
# ---------------------------------------------------------------------------


class MockVisualGen:
    """Lightweight stand-in for VisualGen that avoids GPU / model loading."""

    def __init__(
        self,
        image_output: Optional[torch.Tensor] = None,
        video_output: Optional[torch.Tensor] = None,
        audio_output: Optional[torch.Tensor] = None,
        should_fail: bool = False,
    ):
        self._image = image_output
        self._video = video_output
        self._audio = audio_output
        self._should_fail = should_fail
        self._healthy = True
        self.req_counter = 0

    # --- VisualGen interface ---

    def generate(self, inputs=None, params=None) -> MediaOutput:
        if self._should_fail:
            raise RuntimeError("Generation intentionally failed")
        return MediaOutput(
            image=self._image,
            video=self._video,
            audio=self._audio,
        )

    def generate_async(self, inputs=None, params=None) -> "MockDiffusionGenerationResult":
        return MockDiffusionGenerationResult(
            image=self._image,
            video=self._video,
            audio=self._audio,
            should_fail=self._should_fail,
        )

    def _check_health(self) -> bool:
        return self._healthy

    async def get_stats_async(self, timeout: int):
        return

    def shutdown(self):
        pass


class MockDiffusionGenerationResult:
    """Mock future-like result for generate_async."""

    def __init__(
        self,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        should_fail: bool = False,
    ):
        self._image = image
        self._video = video
        self._audio = audio
        self._should_fail = should_fail

    async def result(self, timeout=None):
        if self._should_fail:
            raise RuntimeError("Async generation intentionally failed")
        return MediaOutput(
            image=self._image,
            video=self._video,
            audio=self._audio,
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
    return TestClient(server.app)


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
    """Mock MP4 encoding to avoid PyAV dependency in unit tests.

    Replaces MediaStorage._save_mp4 with a stub that writes a small
    dummy file so FileResponse can serve it.
    """

    def _dummy_save_mp4(video, audio, output_path, frame_rate):
        os.makedirs(os.path.dirname(str(output_path)) or ".", exist_ok=True)
        with open(str(output_path), "wb") as f:
            f.write(b"\x00\x00\x00\x1cftypisom" + b"\x00" * 32)
        return str(output_path)

    with patch.object(MediaStorage, "_save_mp4", staticmethod(_dummy_save_mp4)):
        yield


# =========================================================================
# POST /v1/images/generations
# =========================================================================


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

    def test_image_generation_url_format_not_supported(self, image_client):
        resp = image_client.post(
            "/v1/images/generations",
            json={
                "prompt": "A dog",
                "response_format": "url",
            },
        )
        assert resp.status_code == 501

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
        resp = failing_client.post(
            "/v1/images/generations",
            json={
                "prompt": "A bird",
                "response_format": "b64_json",
            },
        )
        assert resp.status_code == 400

    def test_image_generation_invalid_size(self, image_client):
        """Invalid size triggers RequestValidationError → custom handler → 400."""
        resp = image_client.post(
            "/v1/images/generations",
            json={
                "prompt": "A mountain",
                "response_format": "b64_json",
                "size": "invalid",
            },
        )
        assert resp.status_code == 400

    def test_image_generation_null_output(self, tmp_path):
        """Generator returns MediaOutput with image=None."""
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
        """Missing required field → RequestValidationError → custom handler → 400."""
        resp = image_client.post(
            "/v1/images/generations",
            json={},
        )
        assert resp.status_code == 400


# =========================================================================
# POST /v1/images/edits
# =========================================================================


class TestImageEdit:
    def test_basic_image_edit(self, image_client):
        b64_img = _b64_white_png_1x1()
        resp = image_client.post(
            "/v1/images/edits",
            json={
                "image": b64_img,
                "prompt": "Make it blue",
                "num_inference_steps": 10,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        assert len(data["data"]) >= 1
        assert data["data"][0]["b64_json"] is not None

    def test_image_edit_with_list_images(self, image_client):
        b64_img = _b64_white_png_1x1()
        resp = image_client.post(
            "/v1/images/edits",
            json={
                "image": [b64_img, b64_img],
                "prompt": "Merge them",
                "num_inference_steps": 10,
            },
        )
        assert resp.status_code == 200

    def test_image_edit_with_mask(self, image_client):
        b64_img = _b64_white_png_1x1()
        b64_mask = _b64_white_png_1x1()
        resp = image_client.post(
            "/v1/images/edits",
            json={
                "image": b64_img,
                "prompt": "Remove object",
                "mask": b64_mask,
                "num_inference_steps": 10,
            },
        )
        assert resp.status_code == 200

    def test_image_edit_with_optional_params(self, image_client):
        b64_img = _b64_white_png_1x1()
        resp = image_client.post(
            "/v1/images/edits",
            json={
                "image": b64_img,
                "prompt": "Enhance colors",
                "size": "128x128",
                "guidance_scale": 8.0,
                "num_inference_steps": 15,
                "seed": 42,
                "negative_prompt": "dark",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["size"] == "128x128"

    def test_image_edit_failure(self, failing_client):
        b64_img = _b64_white_png_1x1()
        resp = failing_client.post(
            "/v1/images/edits",
            json={
                "image": b64_img,
                "prompt": "Edit this",
                "num_inference_steps": 10,
            },
        )
        assert resp.status_code == 500

    def test_missing_image_for_edit(self, image_client):
        """Missing required field → RequestValidationError → custom handler → 400."""
        resp = image_client.post(
            "/v1/images/edits",
            json={
                "prompt": "Edit without image",
            },
        )
        assert resp.status_code == 400


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

    def test_sync_video_generation_multipart(self, video_client):
        # Use files={} with a dummy file to ensure multipart/form-data
        dummy_file = BytesIO(b"")
        resp = video_client.post(
            "/v1/videos/generations",
            data={
                "prompt": "Mountain sunrise",
                "size": "64x64",
                "seconds": "1.0",
                "fps": "8",
            },
            files={"_dummy": ("dummy", dummy_file, "application/octet-stream")},
        )
        # The server will parse fields; _dummy is ignored since it's not "input_reference"
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
        assert resp.status_code == 400

    def test_sync_video_null_output(self, tmp_path):
        """Generator returns MediaOutput with video=None."""
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
        """Missing required prompt → Pydantic ValidationError → 400."""
        resp = video_client.post(
            "/v1/videos/generations",
            json={"size": "64x64"},
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400

    def test_sync_video_missing_prompt_multipart(self, video_client):
        """Missing prompt in multipart form → ValueError → 400."""
        dummy_file = BytesIO(b"")
        resp = video_client.post(
            "/v1/videos/generations",
            data={"size": "64x64"},
            files={"_dummy": ("dummy", dummy_file, "application/octet-stream")},
        )
        assert resp.status_code == 400


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

    def test_async_video_multipart(self, video_client):
        """Multipart encoding requires a file field to trigger the correct content-type."""
        dummy_file = BytesIO(b"")
        resp = video_client.post(
            "/v1/videos",
            data={
                "prompt": "A sunset",
                "size": "64x64",
                "seconds": "1.0",
                "fps": "8",
            },
            files={"_dummy": ("dummy", dummy_file, "application/octet-stream")},
        )
        assert resp.status_code == 202

    def test_async_video_invalid_seconds(self, video_client):
        """Seconds must be between 1.0 and 16.0. Validation error → 400."""
        resp = video_client.post(
            "/v1/videos",
            json={
                "prompt": "Too short",
                "seconds": 0.1,
                "size": "64x64",
                "fps": 8,
            },
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400

    def test_async_video_invalid_fps(self, video_client):
        """Fps must be between 8 and 60. Validation error → 400."""
        resp = video_client.post(
            "/v1/videos",
            json={
                "prompt": "Bad fps",
                "seconds": 1.0,
                "fps": 2,
                "size": "64x64",
            },
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400


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

        # Write a dummy video file
        (tmp_path / f"{video_id}.mp4").write_bytes(b"\x00" * 32)

        resp = client.delete(f"/v1/videos/{video_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] is True

        # Verify it's gone from the store
        resp = client.get(f"/v1/videos/{video_id}")
        assert resp.status_code == 404

        # Verify file is deleted
        assert not (tmp_path / f"{video_id}.mp4").exists()
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
