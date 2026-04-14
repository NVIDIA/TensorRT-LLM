# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for async media loading in tensorrt_llm.inputs.utils.

Covers:
- async_load_image / async_load_audio with data URLs and file paths
- CPU-bound work is offloaded to a thread pool (event loop not blocked)
- aiohttp session is reused across calls (_get_aiohttp_session)
- MultimodalDataTracker.retrieve_all_async gathers all modalities concurrently
"""

import asyncio
import base64
import tempfile
import threading
import time
from io import BytesIO
from unittest.mock import patch

import numpy as np
import pytest
import soundfile
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
    buf = BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _make_audio_file(tmp_path: str) -> str:
    """Write a short WAV file and return its path."""
    sr = 16000
    t = np.linspace(0, 0.05, int(sr * 0.05), endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    soundfile.write(tmp_path, audio, sr)
    return tmp_path


# ──────────────────────────────────────────────────────────────
# async_load_image
# ──────────────────────────────────────────────────────────────


class TestAsyncLoadImage:
    @pytest.mark.asyncio
    async def test_load_image_from_data_url_pil(self):
        url = _make_image_data_url()
        result = await async_load_image(url, format="pil")
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    @pytest.mark.asyncio
    async def test_load_image_from_data_url_pt(self):
        import torch

        url = _make_image_data_url()
        result = await async_load_image(url, format="pt")
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 3  # C x H x W

    @pytest.mark.asyncio
    async def test_load_image_pil_input_passthrough(self):
        """Already-decoded PIL image should be returned as RGB."""
        img = Image.new("L", (4, 4))  # grayscale
        result = await async_load_image(img, format="pil")
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    @pytest.mark.asyncio
    async def test_cpu_work_runs_in_executor(self):
        """PIL decoding must not run on the event loop thread."""
        event_loop_thread_id = threading.current_thread().ident
        executor_thread_ids = []

        original_load = utils_module._load_and_convert_image

        def tracking_load(*args, **kwargs):
            executor_thread_ids.append(threading.current_thread().ident)
            return original_load(*args, **kwargs)

        url = _make_image_data_url()
        with patch.object(utils_module, "_load_and_convert_image", tracking_load):
            await async_load_image(url, format="pil")

        assert len(executor_thread_ids) == 1
        assert executor_thread_ids[0] != event_loop_thread_id, (
            "PIL decoding ran on the event loop thread — event loop is being blocked"
        )


# ──────────────────────────────────────────────────────────────
# async_load_audio
# ──────────────────────────────────────────────────────────────


class TestAsyncLoadAudio:
    @pytest.mark.asyncio
    async def test_load_audio_from_file(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = _make_audio_file(f.name)
        result = await async_load_audio(path)
        audio_array, sample_rate = result
        assert isinstance(audio_array, np.ndarray)
        assert sample_rate == 16000

    @pytest.mark.asyncio
    async def test_cpu_work_runs_in_executor(self):
        """soundfile.read must not run on the event loop thread."""
        event_loop_thread_id = threading.current_thread().ident
        executor_thread_ids = []

        original_read = soundfile.read

        def tracking_read(*args, **kwargs):
            executor_thread_ids.append(threading.current_thread().ident)
            return original_read(*args, **kwargs)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = _make_audio_file(f.name)

        with patch("tensorrt_llm.inputs.utils.soundfile.read", tracking_read):
            await async_load_audio(path)

        assert len(executor_thread_ids) == 1
        assert executor_thread_ids[0] != event_loop_thread_id, (
            "soundfile.read ran on the event loop thread — event loop is being blocked"
        )


# ──────────────────────────────────────────────────────────────
# Session reuse
# ──────────────────────────────────────────────────────────────


class TestSessionReuse:
    @pytest.mark.asyncio
    async def test_same_session_returned_on_repeated_calls(self):
        utils_module._global_aiohttp_session = None
        try:
            s1 = await _get_aiohttp_session()
            s2 = await _get_aiohttp_session()
            assert s1 is s2, "Expected the same ClientSession object to be returned"
        finally:
            if utils_module._global_aiohttp_session is not None:
                await utils_module._global_aiohttp_session.close()
            utils_module._global_aiohttp_session = None

    @pytest.mark.asyncio
    async def test_new_session_created_after_close(self):
        utils_module._global_aiohttp_session = None
        try:
            s1 = await _get_aiohttp_session()
            await s1.close()
            s2 = await _get_aiohttp_session()
            assert s1 is not s2, "Expected a fresh session after the old one was closed"
            assert not s2.closed
        finally:
            if utils_module._global_aiohttp_session is not None:
                await utils_module._global_aiohttp_session.close()
            utils_module._global_aiohttp_session = None


# ──────────────────────────────────────────────────────────────
# MultimodalDataTracker.retrieve_all_async — concurrency
#
# We bypass add_data (which requires a registry-registered model type)
# and insert coroutines directly into _data / _embeddings to isolate
# the gather logic being tested.
# ──────────────────────────────────────────────────────────────


class TestRetrieveAllAsync:
    def _make_tracker(self) -> MultimodalDataTracker:
        tracker = MultimodalDataTracker(model_type="test_model")
        return tracker

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
        for v in values:

            async def _coro(x=v):
                return x

            target[modality].append(_coro())

    @pytest.mark.asyncio
    async def test_single_modality_returns_correct_data(self):
        tracker = self._make_tracker()
        self._inject(tracker, "image", [1, 2, 3])

        data, embeddings = await tracker.retrieve_all_async()
        assert data == {"image": [1, 2, 3]}
        assert embeddings is None

    @pytest.mark.asyncio
    async def test_multiple_modalities_returns_correct_data(self):
        tracker = self._make_tracker()
        self._inject(tracker, "image", ["img1", "img2"])
        self._inject(tracker, "video", ["vid1"])

        data, _ = await tracker.retrieve_all_async()
        assert data["image"] == ["img1", "img2"]
        assert data["video"] == ["vid1"]

    @pytest.mark.asyncio
    async def test_embeddings_returned_separately(self):
        tracker = self._make_tracker()
        self._inject(tracker, "image", ["img"], is_embedding=False)
        self._inject(tracker, "image", ["emb"], is_embedding=True)

        data, embeddings = await tracker.retrieve_all_async()
        assert data == {"image": ["img"]}
        assert embeddings == {"image": ["emb"]}

    @pytest.mark.asyncio
    async def test_cross_modality_runs_concurrently(self):
        """Image and video coroutines must overlap, not run sequentially.

        Each coroutine sleeps for DELAY seconds. If they ran sequentially
        (old behaviour: per-modality await), total time ≥ 2*DELAY.
        With concurrent gather total time ≈ DELAY.
        """
        DELAY = 0.15
        TOLERANCE = 0.08

        tracker = self._make_tracker()

        async def _slow(v):
            await asyncio.sleep(DELAY)
            return v

        tracker._data["image"].extend([_slow("img1"), _slow("img2")])
        tracker._data["video"].extend([_slow("vid1"), _slow("vid2")])

        start = time.perf_counter()
        data, _ = await tracker.retrieve_all_async()
        elapsed = time.perf_counter() - start

        assert data["image"] == ["img1", "img2"]
        assert data["video"] == ["vid1", "vid2"]
        assert elapsed < DELAY + TOLERANCE, (
            f"retrieve_all_async took {elapsed:.3f}s — "
            f"expected < {DELAY + TOLERANCE:.3f}s (concurrent). "
            f"Likely running modalities sequentially."
        )

    @pytest.mark.asyncio
    async def test_empty_tracker_returns_none(self):
        tracker = self._make_tracker()
        data, embeddings = await tracker.retrieve_all_async()
        assert data is None
        assert embeddings is None
