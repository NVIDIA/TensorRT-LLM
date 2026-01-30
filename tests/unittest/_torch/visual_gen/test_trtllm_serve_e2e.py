"""End-to-end tests for trtllm-serve visual_gen with real models.

Tests text-to-video (t2v) and text+image-to-video (ti2v) generation through
the full ``trtllm-serve`` stack backed by real VisualGen models.

The server is launched as a subprocess (same pattern as
``tests/unittest/llmapi/apps/openai_server.py``), so each test class gets an
isolated ``trtllm-serve`` process.

Usage::

    # Run all real-model tests (requires GPU + models in $HOME/llm-models-ci)
    pytest tests/visual_gen/test_trtllm_serve_e2e.py -v

    # Run only t2v tests
    pytest tests/visual_gen/test_trtllm_serve_e2e.py -v -k TestWanT2V

    # Run only ti2v tests
    pytest tests/visual_gen/test_trtllm_serve_e2e.py -v -k TestWanI2V
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import pytest
import requests
import yaml

from tensorrt_llm._utils import get_free_port

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------


def _llm_models_root() -> str:
    """Return LLM_MODELS_ROOT path if it is set in env, assert when it's set but not a valid path."""
    root = Path("/home/scratch.trt_llm_data_ci/llm-models/")
    if "LLM_MODELS_ROOT" in os.environ:
        root = Path(os.environ["LLM_MODELS_ROOT"])
    if not root.exists():
        root = Path("/scratch.trt_llm_data/llm-models/")
    assert root.exists(), (
        "You shall set LLM_MODELS_ROOT env or be able to access scratch.trt_llm_data to run this test"
    )
    return str(root)


_WAN_T2V_PATH = Path(_llm_models_root()) / "Wan2.1-T2V-1.3B-Diffusers"
_WAN_I2V_PATH = Path(_llm_models_root()) / "Wan2.2-I2V-A14B-Diffusers"

# Reference image used for image-to-video (ti2v) tests
_PROJECT_ROOT = Path(__file__).resolve().parents[4]  # repo root
_REF_IMAGE_PATH = _PROJECT_ROOT / "examples" / "visual_gen" / "cat_piano.png"


# ---------------------------------------------------------------------------
# Remote server helper (follows RemoteOpenAIServer pattern)
# ---------------------------------------------------------------------------


class RemoteVisualGenServer:
    """Launch ``trtllm-serve`` for a visual-gen model as a subprocess.

    Mirrors the interface of ``tests.unittest.llmapi.apps.openai_server.RemoteOpenAIServer``
    adapted for diffusion / visual-gen models.
    """

    MAX_SERVER_START_WAIT_S = 1200  # 20 min – large models need time to load

    def __init__(
        self,
        model: str,
        extra_visual_gen_options: Optional[dict] = None,
        cli_args: Optional[List[str]] = None,
        host: str = "localhost",
        port: Optional[int] = None,
        env: Optional[dict] = None,
    ) -> None:
        self.host = host
        self.port = port if port is not None else get_free_port()
        self._config_file: Optional[str] = None
        self.proc: Optional[subprocess.Popen] = None

        args = ["--host", self.host, "--port", str(self.port)]
        if cli_args:
            args += cli_args

        # Write the visual-gen YAML config to a temp file
        if extra_visual_gen_options:
            fd, self._config_file = tempfile.mkstemp(suffix=".yml", prefix="vg_cfg_")
            with os.fdopen(fd, "w") as f:
                yaml.dump(extra_visual_gen_options, f)
            args += ["--extra_visual_gen_options", self._config_file]

        launch_cmd = ["trtllm-serve", model] + args

        if env is None:
            env = os.environ.copy()

        self.proc = subprocess.Popen(
            launch_cmd,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        self._wait_for_server(timeout=self.MAX_SERVER_START_WAIT_S)

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate()

    def terminate(self):
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=30)
        self.proc = None

        if self._config_file:
            try:
                os.remove(self._config_file)
            except OSError:
                pass
            self._config_file = None

    # -- readiness ---------------------------------------------------------

    def _wait_for_server(self, timeout: float):
        url = self.url_for("health")
        start = time.time()
        while True:
            try:
                if requests.get(url).status_code == 200:
                    return
            except Exception as err:
                result = self.proc.poll()
                if result is not None and result != 0:
                    raise RuntimeError("Visual-gen server exited unexpectedly.") from err
            time.sleep(1)
            if time.time() - start > timeout:
                self.terminate()
                raise RuntimeError(f"Visual-gen server failed to start within {timeout}s.")

    # -- URL helpers -------------------------------------------------------

    @property
    def url_root(self) -> str:
        return f"http://{self.host}:{self.port}"

    def url_for(self, *parts: str) -> str:
        return self.url_root + "/" + "/".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model_available(path: Path) -> bool:
    return path.is_dir()


def _av_available() -> bool:
    """Check if PyAV is installed (required for video encoding in E2E tests)."""
    try:
        import av  # noqa: F401

        return True
    except ImportError:
        return False


def _make_visual_gen_options(**extra) -> dict:
    """Build the YAML dict passed via ``--extra_visual_gen_options``."""
    config = {
        "linear": {"type": "default"},
        "parallel": {"dit_cfg_size": 1, "dit_ulysses_size": 1},
    }
    config.update(extra)
    return config


# =========================================================================
# WAN 2.1 – Text-to-Video (t2v)
# =========================================================================


@pytest.mark.skipif(
    not _model_available(_WAN_T2V_PATH), reason=f"Wan2.1-T2V model not found at {_WAN_T2V_PATH}"
)
@pytest.mark.skipif(
    not _av_available(), reason="PyAV (av) not installed — required for video encoding in E2E tests"
)
class TestWanTextToVideo:
    """Test Wan2.1-T2V-1.3B-Diffusers text-to-video generation via serve API."""

    @pytest.fixture(scope="class")
    def server(self):
        with RemoteVisualGenServer(
            model=str(_WAN_T2V_PATH),
            extra_visual_gen_options=_make_visual_gen_options(),
        ) as srv:
            yield srv

    # ------------------------------------------------------------------

    def test_health(self, server):
        resp = requests.get(server.url_for("health"))
        assert resp.status_code == 200

    def test_t2v_sync(self, server):
        """Synchronous text-to-video via POST /v1/videos/generations."""
        resp = requests.post(
            server.url_for("v1", "videos", "generations"),
            json={
                "prompt": "A cute cat playing piano",
                "size": "480x320",
                "seconds": 1.0,
                "fps": 8,
                "num_inference_steps": 4,
                "seed": 42,
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"] == "video/mp4"
        assert len(resp.content) > 1000, "Video file too small"

    def test_t2v_async_lifecycle(self, server):
        """Async video generation: create job → poll → download → delete."""
        base = server.url_for("v1", "videos")

        # 1. Create job
        create_resp = requests.post(
            base,
            json={
                "prompt": "A rocket launching into a starry sky",
                "size": "480x320",
                "seconds": 1.0,
                "fps": 8,
                "num_inference_steps": 4,
                "seed": 42,
            },
        )
        assert create_resp.status_code == 202, create_resp.text
        job = create_resp.json()
        video_id = job["id"]
        assert video_id.startswith("video_")
        assert job["status"] == "queued"

        # 2. Poll until completed (or timeout)
        deadline = time.time() + 600  # 10 min
        status = "queued"
        while status not in ("completed", "failed") and time.time() < deadline:
            time.sleep(2)
            meta_resp = requests.get(f"{base}/{video_id}")
            assert meta_resp.status_code == 200
            status = meta_resp.json()["status"]

        assert status == "completed", f"Video generation did not complete: {status}"

        # 3. Download video content
        content_resp = requests.get(f"{base}/{video_id}/content")
        assert content_resp.status_code == 200
        assert "video/mp4" in content_resp.headers.get("content-type", "")
        assert len(content_resp.content) > 1000

        # 4. Verify it appears in list
        list_resp = requests.get(base)
        assert list_resp.status_code == 200
        ids = [v["id"] for v in list_resp.json()["data"]]
        assert video_id in ids

        # 5. Delete
        del_resp = requests.delete(f"{base}/{video_id}")
        assert del_resp.status_code == 200
        assert del_resp.json()["deleted"] is True

        # 6. Confirm gone
        gone_resp = requests.get(f"{base}/{video_id}")
        assert gone_resp.status_code == 404


# =========================================================================
# WAN 2.2 – Image-to-Video (ti2v)
# =========================================================================


@pytest.mark.skipif(
    not _model_available(_WAN_I2V_PATH), reason=f"Wan2.2-I2V model not found at {_WAN_I2V_PATH}"
)
@pytest.mark.skipif(
    not _REF_IMAGE_PATH.is_file(), reason=f"Reference image not found at {_REF_IMAGE_PATH}"
)
@pytest.mark.skipif(
    not _av_available(), reason="PyAV (av) not installed — required for video encoding in E2E tests"
)
class TestWanImageToVideo:
    """Test Wan2.2-I2V-A14B-Diffusers image-to-video generation via serve API."""

    @pytest.fixture(scope="class")
    def server(self):
        with RemoteVisualGenServer(
            model=str(_WAN_I2V_PATH),
            extra_visual_gen_options=_make_visual_gen_options(),
        ) as srv:
            yield srv

    # ------------------------------------------------------------------

    def test_health(self, server):
        resp = requests.get(server.url_for("health"))
        assert resp.status_code == 200

    def test_ti2v_sync(self, server):
        """Synchronous image-to-video via multipart POST /v1/videos/generations."""
        with open(_REF_IMAGE_PATH, "rb") as f:
            resp = requests.post(
                server.url_for("v1", "videos", "generations"),
                data={
                    "prompt": "The cat starts playing piano, keys moving",
                    "size": "480x320",
                    "seconds": "1.0",
                    "fps": "8",
                    "num_inference_steps": "4",
                    "seed": "42",
                },
                files={
                    "input_reference": ("cat_piano.png", f, "image/png"),
                },
            )
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"] == "video/mp4"
        assert len(resp.content) > 1000, "Video file too small"

    def test_ti2v_async_lifecycle(self, server):
        """Async i2v: create job with image → poll → download → delete."""
        base = server.url_for("v1", "videos")

        # 1. Create job via multipart
        with open(_REF_IMAGE_PATH, "rb") as f:
            create_resp = requests.post(
                base,
                data={
                    "prompt": "Snow falls on the piano and the cat",
                    "size": "480x320",
                    "seconds": "1.0",
                    "fps": "8",
                    "num_inference_steps": "4",
                    "seed": "42",
                },
                files={
                    "input_reference": ("cat_piano.png", f, "image/png"),
                },
            )
        assert create_resp.status_code == 202, create_resp.text
        job = create_resp.json()
        video_id = job["id"]
        assert job["status"] == "queued"

        # 2. Poll until completed
        deadline = time.time() + 600
        status = "queued"
        while status not in ("completed", "failed") and time.time() < deadline:
            time.sleep(2)
            meta_resp = requests.get(f"{base}/{video_id}")
            assert meta_resp.status_code == 200
            status = meta_resp.json()["status"]

        assert status == "completed", f"Video generation did not complete: {status}"

        # 3. Download
        content_resp = requests.get(f"{base}/{video_id}/content")
        assert content_resp.status_code == 200
        assert "video/mp4" in content_resp.headers.get("content-type", "")
        assert len(content_resp.content) > 1000

        # 4. Delete
        del_resp = requests.delete(f"{base}/{video_id}")
        assert del_resp.status_code == 200
        assert del_resp.json()["deleted"] is True

        # 5. Confirm gone
        gone_resp = requests.get(f"{base}/{video_id}")
        assert gone_resp.status_code == 404
