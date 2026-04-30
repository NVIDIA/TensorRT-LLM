# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for VisualGenOutput / VisualGenMetrics / VisualGenResult."""

import asyncio
from dataclasses import fields, is_dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch

from tensorrt_llm._torch.visual_gen.executor import DiffusionResponse
from tensorrt_llm._torch.visual_gen.output import (
    PipelineOutput,
    split_visual_gen_output,
    to_visual_gen_output,
)
from tensorrt_llm.visual_gen import VisualGenMetrics, VisualGenOutput

# ---------------------------------------------------------------------------
# VisualGenOutput shape and re-exports
# ---------------------------------------------------------------------------


def test_visual_gen_output_is_dataclass():
    """VisualGenOutput must be a flat @dataclass."""
    assert is_dataclass(VisualGenOutput)
    field_names = {f.name for f in fields(VisualGenOutput)}
    assert field_names == {
        "request_id",
        "image",
        "video",
        "audio",
        "frame_rate",
        "audio_sample_rate",
        "error",
        "metrics",
    }


def test_top_level_reexport():
    """``from tensorrt_llm import VisualGenOutput`` succeeds."""
    from tensorrt_llm import VisualGenOutput as ReexportedVgo

    assert ReexportedVgo is VisualGenOutput


def test_subpackage_reexport():
    """``from tensorrt_llm.visual_gen import VisualGenOutput`` succeeds."""
    from tensorrt_llm.visual_gen import VisualGenOutput as SubExport

    assert SubExport is VisualGenOutput


def test_minimal_construction_defaults():
    """Defaults leave non-set fields at their documented None state."""
    out = VisualGenOutput(request_id=7, image=torch.zeros(2, 2, 3, dtype=torch.uint8))
    assert out.request_id == 7
    assert out.image is not None
    assert out.video is None
    assert out.audio is None
    assert out.frame_rate is None
    assert out.audio_sample_rate is None
    assert out.error is None
    assert out.metrics is None


def test_round_trip_video_and_audio_rates():
    """Constructing with video+audio+rates round-trips all four values."""
    video = torch.zeros(4, 16, 16, 3, dtype=torch.uint8)
    audio = torch.zeros(2, 800, dtype=torch.float32)
    out = VisualGenOutput(
        request_id=1,
        video=video,
        frame_rate=24.0,
        audio=audio,
        audio_sample_rate=22050,
    )
    assert out.video is video
    assert out.audio is audio
    assert out.frame_rate == 24.0
    assert out.audio_sample_rate == 22050


def test_media_output_is_unimportable_publicly():
    """``MediaOutput`` is not a public re-export."""
    import tensorrt_llm
    import tensorrt_llm.visual_gen as visual_gen_pkg

    assert not hasattr(tensorrt_llm, "MediaOutput")
    assert not hasattr(visual_gen_pkg, "MediaOutput")


# ---------------------------------------------------------------------------
# VisualGenMetrics shape and error semantics
# ---------------------------------------------------------------------------


def test_visual_gen_metrics_field_set():
    """VisualGenMetrics has exactly four float fields with 0.0 defaults."""
    assert is_dataclass(VisualGenMetrics)
    field_names = {f.name for f in fields(VisualGenMetrics)}
    assert field_names == {"pipeline_ms", "pre_denoise_ms", "denoise_ms", "post_denoise_ms"}
    m = VisualGenMetrics()
    assert m.pipeline_ms == 0.0
    assert m.pre_denoise_ms == 0.0
    assert m.denoise_ms == 0.0
    assert m.post_denoise_ms == 0.0


def test_error_response_yields_metrics_none():
    """An error wire response produces metrics is None on the public output."""
    resp = DiffusionResponse(request_id=42, error_msg="boom", pipeline_ms=0.0)
    out = to_visual_gen_output(resp)
    assert out.error == "boom"
    assert out.image is None
    assert out.video is None
    assert out.audio is None
    assert out.metrics is None


def test_sub_phase_sum_bounded_by_pipeline_ms():
    """Sub-phase sum ``pre + denoise + post`` stays within ``pipeline_ms + slack``.

    Sub-phase events measure GPU-stream time only; small host-side work
    (tokenization, scheduler updates) shows up as
    ``pipeline_ms - (pre + denoise + post)`` and stays non-negative on a real
    run. This pins the relationship at the public-output factory boundary.
    """
    pipeline_out = PipelineOutput(
        image=torch.zeros(1, 4, 4, 3, dtype=torch.uint8),
        pre_denoise_ms=5.0,
        denoise_ms=42.0,
        post_denoise_ms=3.0,
    )
    # Executor measures pipeline_ms with a slightly larger envelope (host
    # wall-clock around pipeline.infer()), so the sub-phase sum stays under
    # pipeline_ms.
    resp = DiffusionResponse(request_id=11, output=pipeline_out, pipeline_ms=55.0)
    out = to_visual_gen_output(resp)
    sub_sum = out.metrics.pre_denoise_ms + out.metrics.denoise_ms + out.metrics.post_denoise_ms
    slack_ms = 1.0
    assert sub_sum <= out.metrics.pipeline_ms + slack_ms, (
        f"sub-phase sum {sub_sum} > pipeline_ms+slack {out.metrics.pipeline_ms + slack_ms}"
    )


# ---------------------------------------------------------------------------
# to_visual_gen_output success path
# ---------------------------------------------------------------------------


def _make_image_pipeline_output() -> PipelineOutput:
    """Helper: a Flux-shaped PipelineOutput with timing populated."""
    image = torch.full((1, 8, 8, 3), 200, dtype=torch.uint8)
    return PipelineOutput(
        image=image,
        pre_denoise_ms=5.0,
        denoise_ms=42.0,
        post_denoise_ms=3.0,
    )


def test_from_response_success_image():
    """Success path populates request_id, media tensor, and four metrics."""
    pipeline_out = _make_image_pipeline_output()
    resp = DiffusionResponse(request_id=11, output=pipeline_out, pipeline_ms=55.5)
    out = to_visual_gen_output(resp)
    assert out.request_id == 11
    assert out.image is pipeline_out.image
    assert out.error is None
    assert out.metrics is not None
    assert out.metrics.pipeline_ms == 55.5
    assert out.metrics.pre_denoise_ms == 5.0
    assert out.metrics.denoise_ms == 42.0
    assert out.metrics.post_denoise_ms == 3.0


def test_from_response_success_video_with_rates():
    """Video pipelines surface frame_rate and audio_sample_rate on the output."""
    video = torch.full((1, 4, 8, 8, 3), 128, dtype=torch.uint8)
    audio = torch.zeros(1, 2, 800, dtype=torch.float32)
    pipeline_out = PipelineOutput(
        video=video,
        audio=audio,
        frame_rate=24.0,
        audio_sample_rate=48000,
        pre_denoise_ms=1.0,
        denoise_ms=10.0,
        post_denoise_ms=2.0,
    )
    resp = DiffusionResponse(request_id=3, output=pipeline_out, pipeline_ms=20.0)
    out = to_visual_gen_output(resp)
    assert out.frame_rate == 24.0
    assert out.audio_sample_rate == 48000
    assert out.video is video
    assert out.audio is audio
    assert out.metrics.pipeline_ms == 20.0


# ---------------------------------------------------------------------------
# split_visual_gen_output batch fan-out
# ---------------------------------------------------------------------------


def test_batch_split_success_image():
    """Batch fan-out slices image along dim 0 into per-item outputs."""
    pipeline_out = PipelineOutput(
        image=torch.stack(
            [
                torch.full((4, 4, 3), 10, dtype=torch.uint8),
                torch.full((4, 4, 3), 20, dtype=torch.uint8),
                torch.full((4, 4, 3), 30, dtype=torch.uint8),
            ]
        ),
        pre_denoise_ms=1.0,
        denoise_ms=2.0,
        post_denoise_ms=3.0,
    )
    resp = DiffusionResponse(request_id=99, output=pipeline_out, pipeline_ms=10.0)
    items = split_visual_gen_output(resp, batch_size=3)
    assert len(items) == 3
    for i, item in enumerate(items):
        assert item.request_id == 99
        assert item.error is None
        assert item.image is not None
        assert item.image.shape == (4, 4, 3)
        # Per-item tensor is unbatched.
        assert int(item.image[0, 0, 0]) == (i + 1) * 10
        # Metrics shared across items (single batched inference).
        assert item.metrics is not None
        assert item.metrics.pipeline_ms == 10.0


def test_batch_split_full_batch_failure():
    """On full-batch failure each item carries the same executor-level error."""
    resp = DiffusionResponse(request_id=7, error_msg="OOM in transformer")
    items = split_visual_gen_output(resp, batch_size=4)
    assert len(items) == 4
    for item in items:
        assert item.image is None
        assert item.video is None
        assert item.audio is None
        assert item.error == "OOM in transformer"
        assert item.metrics is None


# ---------------------------------------------------------------------------
# VisualGenOutput.save routing
# ---------------------------------------------------------------------------


def test_save_image_routes_to_encoding(tmp_path):
    """save() on an image output routes to media.encoding.save_image."""
    out = VisualGenOutput(
        request_id=1,
        image=torch.full((8, 8, 3), 255, dtype=torch.uint8),
    )
    target = tmp_path / "x.png"
    with patch("tensorrt_llm.media.encoding.save_image") as mock_save:
        mock_save.return_value = str(target)
        out.save(target)
        mock_save.assert_called_once()
        # First positional is the tensor; second is the path.
        args, kwargs = mock_save.call_args
        assert args[0] is out.image


def test_save_video_routes_with_rate(tmp_path):
    """save() on a video output uses self.frame_rate by default."""
    out = VisualGenOutput(
        request_id=2,
        video=torch.zeros(4, 8, 8, 3, dtype=torch.uint8),
        frame_rate=16.0,
    )
    target = tmp_path / "x.mp4"
    with patch("tensorrt_llm.media.encoding.save_video") as mock_save:
        mock_save.return_value = str(target)
        out.save(target)
        mock_save.assert_called_once()
        _, kwargs = mock_save.call_args
        assert kwargs["frame_rate"] == 16.0


def test_save_video_kwarg_overrides_rate(tmp_path):
    """Explicit ``frame_rate=`` kwarg overrides the carried rate."""
    out = VisualGenOutput(
        request_id=2,
        video=torch.zeros(4, 8, 8, 3, dtype=torch.uint8),
        frame_rate=16.0,
    )
    target = tmp_path / "x.mp4"
    with patch("tensorrt_llm.media.encoding.save_video") as mock_save:
        mock_save.return_value = str(target)
        out.save(target, frame_rate=30.0)
        _, kwargs = mock_save.call_args
        assert kwargs["frame_rate"] == 30.0


def test_save_errored_output_raises(tmp_path):
    """save() on an errored output raises VisualGenError mentioning the error."""
    from tensorrt_llm.visual_gen import VisualGenError

    out = VisualGenOutput(request_id=3, error="kernel launch failed")
    with pytest.raises(VisualGenError, match="kernel launch failed"):
        out.save(tmp_path / "x.png")


def test_save_video_without_rate_raises(tmp_path):
    """Video output without frame_rate (and no kwarg) raises VisualGenError."""
    from tensorrt_llm.visual_gen import VisualGenError

    out = VisualGenOutput(
        request_id=4,
        video=torch.zeros(4, 8, 8, 3, dtype=torch.uint8),
    )
    with pytest.raises(VisualGenError, match="frame_rate"):
        out.save(tmp_path / "x.mp4")


def test_save_no_media_raises(tmp_path):
    """save() with no media tensor raises VisualGenError."""
    from tensorrt_llm.visual_gen import VisualGenError

    out = VisualGenOutput(request_id=5)
    with pytest.raises(VisualGenError, match="no media"):
        out.save(tmp_path / "x.png")


# ---------------------------------------------------------------------------
# VisualGenResult Future-like awaitable
# ---------------------------------------------------------------------------


class _FakeExecutor:
    """Stub executor that resolves a request to a pre-built DiffusionResponse."""

    def __init__(self, response: DiffusionResponse):
        self._response = response
        # Build a real running event loop so run_coroutine_threadsafe works.
        self._loop = asyncio.new_event_loop()
        self._thread_started = False
        import threading

        def _run():
            self._loop.run_forever()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        self._thread_started = True

    @property
    def _event_loop(self):
        return self._loop

    async def await_responses(self, request_id, timeout=None):
        return self._response

    def stop(self):
        if self._thread_started:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=2.0)


@pytest.fixture()
def fake_image_executor():
    pipe = _make_image_pipeline_output()
    resp = DiffusionResponse(request_id=10, output=pipe, pipeline_ms=12.5)
    fx = _FakeExecutor(resp)
    yield fx
    fx.stop()


def test_await_resolves_to_visual_gen_output(fake_image_executor):
    """``await handle`` resolves to a VisualGenOutput for single-prompt input."""
    from tensorrt_llm.visual_gen.visual_gen import VisualGenResult

    handle = VisualGenResult(request_id=10, executor=fake_image_executor, batch_size=None)
    out = asyncio.run(handle.aresult())
    assert isinstance(out, VisualGenOutput)
    assert handle.done is True


def test_aresult_matches_await():
    """``await handle.aresult()`` and ``await handle`` produce the same value."""
    from tensorrt_llm.visual_gen.visual_gen import VisualGenResult

    pipe = _make_image_pipeline_output()
    resp = DiffusionResponse(request_id=11, output=pipe, pipeline_ms=7.0)
    fx = _FakeExecutor(resp)
    try:

        async def _both():
            h1 = VisualGenResult(request_id=11, executor=fx, batch_size=None)
            r1 = await h1.aresult()
            h2 = VisualGenResult(request_id=11, executor=fx, batch_size=None)
            r2 = await h2
            return r1.request_id == r2.request_id

        assert asyncio.run(_both())
    finally:
        fx.stop()


def test_sync_result_is_blocking_value():
    """Sync ``result()`` blocks and returns the resolved value (not a coroutine)."""
    from tensorrt_llm.visual_gen.visual_gen import VisualGenResult

    pipe = _make_image_pipeline_output()
    resp = DiffusionResponse(request_id=12, output=pipe, pipeline_ms=1.0)
    fx = _FakeExecutor(resp)
    try:
        handle = VisualGenResult(request_id=12, executor=fx, batch_size=None)
        out = handle.result(timeout=5.0)
        assert isinstance(out, VisualGenOutput)
        assert out.request_id == 12
    finally:
        fx.stop()


def test_batch_handle_resolves_to_list():
    """Batch handle resolves to List[VisualGenOutput] of the right length."""
    from tensorrt_llm.visual_gen.visual_gen import VisualGenResult

    pipe = PipelineOutput(
        image=torch.stack(
            [
                torch.zeros(4, 4, 3, dtype=torch.uint8),
                torch.zeros(4, 4, 3, dtype=torch.uint8),
            ]
        ),
    )
    resp = DiffusionResponse(request_id=13, output=pipe, pipeline_ms=3.0)
    fx = _FakeExecutor(resp)
    try:
        handle = VisualGenResult(request_id=13, executor=fx, batch_size=2)
        outs = asyncio.run(handle.aresult())
        assert isinstance(outs, list)
        assert len(outs) == 2
        assert all(isinstance(o, VisualGenOutput) for o in outs)
    finally:
        fx.stop()


def test_batch_error_yields_error_per_item():
    """Batch handle with an errored response returns a list with errors set."""
    from tensorrt_llm.visual_gen.visual_gen import VisualGenResult

    resp = DiffusionResponse(request_id=14, error_msg="model died")
    fx = _FakeExecutor(resp)
    try:
        handle = VisualGenResult(request_id=14, executor=fx, batch_size=3)
        outs = asyncio.run(handle.aresult())
        assert len(outs) == 3
        for o in outs:
            assert o.error == "model died"
            assert o.image is None
    finally:
        fx.stop()


def test_single_error_raises_visual_gen_error():
    """Awaiting a single-prompt handle whose response errored raises VisualGenError."""
    from tensorrt_llm.visual_gen import VisualGenError
    from tensorrt_llm.visual_gen.visual_gen import VisualGenResult

    resp = DiffusionResponse(request_id=15, error_msg="boom")
    fx = _FakeExecutor(resp)
    try:
        handle = VisualGenResult(request_id=15, executor=fx, batch_size=None)
        with pytest.raises(VisualGenError, match="boom"):
            asyncio.run(handle.aresult())
    finally:
        fx.stop()


def test_awaiting_sync_result_raises_typeerror():
    """``await handle.result()`` (treating sync ``result`` as coroutine) fails."""
    from tensorrt_llm.visual_gen.visual_gen import VisualGenResult

    pipe = _make_image_pipeline_output()
    resp = DiffusionResponse(request_id=16, output=pipe, pipeline_ms=0.5)
    fx = _FakeExecutor(resp)
    try:
        handle = VisualGenResult(request_id=16, executor=fx, batch_size=None)

        async def _bad():
            await handle.result()

        with pytest.raises(TypeError):
            asyncio.run(_bad())
    finally:
        fx.stop()


# ---------------------------------------------------------------------------
# NotImplementedError guard for List[VisualGenParams]
# ---------------------------------------------------------------------------


def test_params_list_raises_not_implemented():
    """Passing ``params`` as a list raises NotImplementedError.

    Exercised without a live executor by patching the inputs validation that
    happens before any IPC.
    """
    from tensorrt_llm.visual_gen import VisualGenParams
    from tensorrt_llm.visual_gen.visual_gen import VisualGen

    fake = MagicMock(spec=VisualGen)
    # Bind the real ``generate_async`` method to the fake so we exercise the
    # actual guard logic.
    fake.generate_async = VisualGen.generate_async.__get__(fake, VisualGen)
    with pytest.raises(NotImplementedError, match="Per-item params"):
        fake.generate_async(inputs="a cat", params=[VisualGenParams()])


# ---------------------------------------------------------------------------
# media.encoding free-function imports
# ---------------------------------------------------------------------------


def test_encoding_free_functions_importable():
    """media.encoding exposes the five free functions."""
    from tensorrt_llm.media.encoding import (
        image_to_bytes,
        resolve_video_format,
        save_image,
        save_video,
        video_to_bytes,
    )

    assert callable(image_to_bytes)
    assert callable(resolve_video_format)
    assert callable(save_image)
    assert callable(save_video)
    assert callable(video_to_bytes)


def test_encoding_not_top_level_reexport():
    """Encoding free functions are not re-exported from tensorrt_llm."""
    import tensorrt_llm

    for name in ("save_image", "save_video", "image_to_bytes", "video_to_bytes"):
        assert not hasattr(tensorrt_llm, name), (
            f"tensorrt_llm.{name} should not be re-exported; encoding is internal-by-convention."
        )


# ---------------------------------------------------------------------------
# MediaStorage no longer carries encoding methods
# ---------------------------------------------------------------------------


def test_media_storage_lacks_encoding_methods():
    """MediaStorage instances no longer have save_image/save_video/etc."""
    from tensorrt_llm.serve.media_storage import MediaStorage

    ms = MediaStorage()
    for name in ("save_image", "save_video", "convert_image_to_bytes", "convert_video_to_bytes"):
        assert not hasattr(ms, name), (
            f"MediaStorage.{name} should be removed; use tensorrt_llm.media.encoding instead."
        )


def test_resolve_video_format_moved():
    """resolve_video_format is not importable from media_storage."""
    with pytest.raises(ImportError):
        from tensorrt_llm.serve.media_storage import resolve_video_format  # noqa: F401


# ---------------------------------------------------------------------------
# PipelineOutput shape
# ---------------------------------------------------------------------------


def test_pipeline_output_has_eight_fields():
    """PipelineOutput has the eight expected fields."""
    field_names = {f.name for f in fields(PipelineOutput)}
    assert field_names == {
        "image",
        "video",
        "audio",
        "frame_rate",
        "audio_sample_rate",
        "pre_denoise_ms",
        "denoise_ms",
        "post_denoise_ms",
    }


def test_pipeline_output_default_construction():
    """PipelineOutput() constructs with all-None / 0.0 defaults."""
    p = PipelineOutput()
    assert p.image is None
    assert p.video is None
    assert p.audio is None
    assert p.frame_rate is None
    assert p.audio_sample_rate is None
    assert p.pre_denoise_ms == 0.0
    assert p.denoise_ms == 0.0
    assert p.post_denoise_ms == 0.0


def test_media_output_unimportable():
    """``MediaOutput`` is not importable from any path."""
    with pytest.raises(ImportError):
        from tensorrt_llm._torch.visual_gen.output import MediaOutput  # noqa: F401
