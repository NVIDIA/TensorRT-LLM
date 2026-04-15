# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for async media loading in tensorrt_llm.inputs.utils.

Covers:
- async_load_image / async_load_audio with data URLs and file paths
- CPU-bound work is offloaded to a thread pool (event loop not blocked)
- aiohttp session is reused across calls (_get_aiohttp_session)
- MultimodalDataTracker.retrieve_all_async gathers all modalities concurrently
"""

import base64
import tempfile
import threading
from io import BytesIO
from unittest.mock import patch

import numpy as np
import pytest
import pytest_asyncio
import soundfile
import torch
from PIL import Image

import tensorrt_llm.inputs.utils as utils_module
from tensorrt_llm.inputs.utils import (
    MultimodalDataTracker,
    _get_aiohttp_session,
    async_load_audio,
    async_load_image,
)

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


def _make_image_data_url() -> str:
    img = Image.new("RGB", (8, 8), color=(100, 150, 200))
    image_buf = BytesIO()
    img.save(image_buf, format="JPEG")
    b64_str = base64.b64encode(image_buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64_str}"


def _make_audio_file(tmp_path: str) -> str:
    """Write a short WAV file and return its path."""
    sample_rate = 16000
    time_axis = np.linspace(0, 0.05, int(sample_rate * 0.05), endpoint=False)
    audio_samples = (np.sin(2 * np.pi * 440 * time_axis) * 0.5).astype(np.float32)
    soundfile.write(tmp_path, audio_samples, sample_rate)
    return tmp_path


# ──────────────────────────────────────────────────────────────
# async_load_image
# ──────────────────────────────────────────────────────────────


class TestAsyncLoadImage:
    @pytest.mark.asyncio
    async def test_load_image_from_data_url_pil(self):
        url = _make_image_data_url()
        image = await async_load_image(url, format="pil")
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"

    @pytest.mark.asyncio
    async def test_load_image_from_data_url_pt(self):
        url = _make_image_data_url()
        image_tensor = await async_load_image(url, format="pt")
        assert isinstance(image_tensor, torch.Tensor)
        assert image_tensor.shape[0] == 3  # C x H x W

    @pytest.mark.asyncio
    async def test_load_image_pil_input_passthrough(self):
        """Already-decoded PIL image should be returned as RGB."""
        grayscale_img = Image.new("L", (4, 4))
        rgb_image = await async_load_image(grayscale_img, format="pil")
        assert isinstance(rgb_image, Image.Image)
        assert rgb_image.mode == "RGB"

    @pytest.mark.asyncio
    async def test_cpu_work_runs_in_executor(self):
        """PIL decoding must not run on the event loop thread."""
        event_loop_thread_id = threading.current_thread().ident
        worker_thread_ids = []

        original_load = utils_module._load_and_convert_image

        def tracking_load(*args, **kwargs):
            worker_thread_ids.append(threading.current_thread().ident)
            return original_load(*args, **kwargs)

        url = _make_image_data_url()
        with patch.object(utils_module, "_load_and_convert_image", tracking_load):
            await async_load_image(url, format="pil")

        assert len(worker_thread_ids) == 1
        assert worker_thread_ids[0] != event_loop_thread_id, (
            "PIL decoding ran on the event loop thread — event loop is being blocked"
        )


# ──────────────────────────────────────────────────────────────
# async_load_audio
# ──────────────────────────────────────────────────────────────


class TestAsyncLoadAudio:
    @pytest.mark.asyncio
    async def test_load_audio_from_file(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = _make_audio_file(f.name)
        audio_array, sample_rate = await async_load_audio(wav_path)
        assert isinstance(audio_array, np.ndarray)
        assert sample_rate == 16000  # matches the sr=16000 used in _make_audio_file

    @pytest.mark.asyncio
    async def test_cpu_work_runs_in_executor(self):
        """soundfile.read must not run on the event loop thread."""
        event_loop_thread_id = threading.current_thread().ident
        worker_thread_ids = []

        original_read = soundfile.read

        def tracking_read(*args, **kwargs):
            worker_thread_ids.append(threading.current_thread().ident)
            return original_read(*args, **kwargs)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = _make_audio_file(f.name)

        with patch("tensorrt_llm.inputs.utils.soundfile.read", tracking_read):
            await async_load_audio(wav_path)

        assert len(worker_thread_ids) == 1
        assert worker_thread_ids[0] != event_loop_thread_id, (
            "soundfile.read ran on the event loop thread — event loop is being blocked"
        )


# ──────────────────────────────────────────────────────────────
# Session reuse
# ──────────────────────────────────────────────────────────────


class TestSessionReuse:
    @pytest_asyncio.fixture(autouse=True)
    async def reset_global_session(self):
        utils_module._global_aiohttp_session = None
        yield
        if utils_module._global_aiohttp_session is not None:
            await utils_module._global_aiohttp_session.close()
        utils_module._global_aiohttp_session = None

    @pytest.mark.asyncio
    async def test_same_session_returned_on_repeated_calls(self):
        first_session = await _get_aiohttp_session()
        second_session = await _get_aiohttp_session()
        assert first_session is second_session, (
            "Expected the same ClientSession object to be returned"
        )

    @pytest.mark.asyncio
    async def test_new_session_created_after_close(self):
        first_session = await _get_aiohttp_session()
        await first_session.close()
        second_session = await _get_aiohttp_session()
        assert first_session is not second_session, (
            "Expected a fresh session after the old one was closed"
        )
        assert not second_session.closed


# ──────────────────────────────────────────────────────────────
# MultimodalDataTracker.retrieve_all_async — concurrency
#
# We bypass add_data (which requires a registry-registered model type)
# and insert coroutines directly into _data / _embeddings to isolate
# the gather logic being tested.
# ──────────────────────────────────────────────────────────────


class TestRetrieveAllAsync:
    def _make_tracker(self) -> MultimodalDataTracker:
        return MultimodalDataTracker(model_type="test_model")

    def _inject(
        self,
        tracker: MultimodalDataTracker,
        modality: str,
        values: list,
        *,
        is_embedding: bool = False,
    ):
        """Directly insert coroutines into the tracker without registry checks."""
        target = tracker._embeddings if is_embedding else tracker._data
        for val in values:

            async def _coro(x=val):
                return x

            target[modality].append(_coro())

    @pytest.mark.asyncio
    async def test_single_modality_returns_correct_data(self):
        image_values = [1, 2, 3]
        tracker = self._make_tracker()
        self._inject(tracker, "image", image_values)

        data, embeddings = await tracker.retrieve_all_async()
        assert data == {"image": image_values}
        assert embeddings is None

    @pytest.mark.asyncio
    async def test_multiple_modalities_returns_correct_data(self):
        image_values = ["img1", "img2"]
        video_values = ["vid1"]
        tracker = self._make_tracker()
        self._inject(tracker, "image", image_values)
        self._inject(tracker, "video", video_values)

        data, _ = await tracker.retrieve_all_async()
        assert data["image"] == image_values
        assert data["video"] == video_values

    @pytest.mark.asyncio
    async def test_embeddings_returned_separately(self):
        image_data = ["img"]
        embedding_data = ["emb"]
        tracker = self._make_tracker()
        self._inject(tracker, "image", image_data, is_embedding=False)
        self._inject(tracker, "image", embedding_data, is_embedding=True)

        data, embeddings = await tracker.retrieve_all_async()
        assert data == {"image": image_data}
        assert embeddings == {"image": embedding_data}

    @pytest.mark.asyncio
    async def test_empty_tracker_returns_none(self):
        tracker = self._make_tracker()
        data, embeddings = await tracker.retrieve_all_async()
        assert data is None
        assert embeddings is None
