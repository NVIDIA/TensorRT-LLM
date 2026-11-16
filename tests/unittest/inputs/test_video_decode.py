# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for `_load_video_by_cv2` return shapes and the HF passthrough."""

from __future__ import annotations

import asyncio
import base64
import importlib
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from PIL import Image
from transformers.video_utils import make_batched_videos

pytest.importorskip("cv2")
import cv2  # noqa: E402

from tensorrt_llm.inputs import media_io  # noqa: E402
from tensorrt_llm.inputs import utils as input_utils  # noqa: E402
from tensorrt_llm.inputs.content_format import ContentFormat  # noqa: E402
from tensorrt_llm.inputs.media_io import VideoMediaIO, _load_video_by_cv2  # noqa: E402
from tensorrt_llm.inputs.multimodal import default_hasher  # noqa: E402
from tensorrt_llm.inputs.multimodal_data import VideoData  # noqa: E402
from tensorrt_llm.inputs.registry import (  # noqa: E402
    INPUT_PROCESSOR_REGISTRY,
    MULTIMODAL_PLACEHOLDER_REGISTRY,
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
)
from tensorrt_llm.inputs.utils import (  # noqa: E402
    async_build_multimodal_prompt,
    async_load_video,
    default_multimodal_input_loader,
    load_base64_video,
    load_video,
)

pytestmark = pytest.mark.cpu_only


@pytest.fixture(scope="module")
def sample_video_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Encode a tiny mp4 with distinguishable per-frame pixel values."""
    width, height, num_frames = 64, 64, 20
    path = tmp_path_factory.mktemp("video_decode") / "sample video.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
    for i in range(num_frames):
        writer.write(np.full((height, width, 3), (i * 10) % 256, dtype=np.uint8))
    writer.release()
    return str(path)


def test_np_format_returns_stacked_uint8_ndarray(sample_video_path: str) -> None:
    video = _load_video_by_cv2(sample_video_path, num_frames=10, fps=-1, format="np")
    assert isinstance(video.frames, np.ndarray)
    assert video.frames.shape == (10, 64, 64, 3)
    assert video.frames.dtype == np.uint8
    assert video.frames.flags["C_CONTIGUOUS"]


def test_pt_format_returns_list_of_chw_tensors(sample_video_path: str) -> None:
    video = _load_video_by_cv2(sample_video_path, num_frames=10, fps=-1, format="pt")
    assert isinstance(video.frames, list)
    assert len(video.frames) == 10
    for frame in video.frames:
        assert isinstance(frame, torch.Tensor)
        assert frame.shape == (3, 64, 64)
        assert frame.dtype == torch.float32
        assert 0.0 <= frame.min().item() and frame.max().item() <= 1.0


def test_pil_format_returns_list_of_pil_images(sample_video_path: str) -> None:
    video = _load_video_by_cv2(sample_video_path, num_frames=10, fps=-1, format="pil")
    assert isinstance(video.frames, list)
    assert len(video.frames) == 10
    for frame in video.frames:
        assert isinstance(frame, Image.Image)
        assert frame.size == (64, 64)


def _digest(video: VideoData) -> str:
    """Hash the video data and sampling metadata."""
    hasher = default_hasher()
    video.update_hash(hasher)
    return hasher.hexdigest()


def test_loaders_default_to_uint8_frames_and_keep_pt(sample_video_path: str) -> None:
    """Every video loader returns uint8 frames by default; `format="pt"` still gives float frames."""
    defaults = [
        VideoMediaIO(num_frames=10).load_file(sample_video_path),
        load_video(sample_video_path, num_frames=10),
        asyncio.run(async_load_video(sample_video_path, num_frames=10)),
        asyncio.run(async_load_video(sample_video_path, num_frames=10, format="np")),
    ]
    floats = [
        VideoMediaIO(num_frames=10, format="pt").load_file(sample_video_path),
        load_video(sample_video_path, num_frames=10, format="pt"),
        asyncio.run(async_load_video(sample_video_path, num_frames=10, format="pt")),
    ]
    expected = defaults[0]
    for video in defaults:
        assert isinstance(video.frames, np.ndarray)
        assert video.frames.dtype == np.uint8
        assert video.frames.shape == (10, 64, 64, 3)
        np.testing.assert_array_equal(video.frames, expected.frames)
        assert video.metadata == expected.metadata
    for video in floats:
        stacked = torch.stack(video.frames)
        assert stacked.dtype == torch.float32
        assert stacked.shape == (10, 3, 64, 64)
        assert 0.0 <= stacked.min().item() and stacked.max().item() <= 1.0
        # Same pixels: the float frames are the uint8 frames / 255 in CHW layout.
        restored = (stacked.permute(0, 2, 3, 1) * 255).round().to(torch.uint8).numpy()
        np.testing.assert_array_equal(restored, expected.frames)
        assert video.metadata == expected.metadata


def test_video_hash_uses_source_bytes_not_frame_format(sample_video_path: str) -> None:
    """Loaded videos hash their source bytes; a `VideoData` built from frames hashes the frames."""
    uint8_video = load_video(sample_video_path, num_frames=10)
    float_video = load_video(sample_video_path, num_frames=10, format="pt")
    assert uint8_video.raw_bytes_hash == float_video.raw_bytes_hash
    assert _digest(uint8_video) == _digest(float_video)

    built = VideoData(frames=uint8_video.frames.copy(), metadata=uint8_video.metadata)
    same = VideoData(frames=uint8_video.frames.copy(), metadata=uint8_video.metadata)
    changed_frames = uint8_video.frames.copy()
    changed_frames[-1, 0, 0, 0] ^= 1
    changed = VideoData(frames=changed_frames, metadata=uint8_video.metadata)
    assert _digest(built) == _digest(same)
    assert _digest(built) != _digest(changed)
    assert _digest(built) != _digest(uint8_video)


@pytest.mark.parametrize(
    ("image_data_format", "media_io_kwargs", "expected_format"),
    [
        (None, None, "np"),
        (None, {"video": {"format": "pt"}}, "pt"),
        ("pt", {"video": {"format": "np"}}, "pt"),
    ],
    ids=["default", "model_preference", "explicit_format"],
)
def test_default_loader_video_format(
    sample_video_path: str,
    image_data_format: str | None,
    media_io_kwargs: dict | None,
    expected_format: str,
) -> None:
    """`default_multimodal_input_loader` uses an explicit format, else the given preference."""
    model_type = "test_video_format_model"
    MULTIMODAL_PLACEHOLDER_REGISTRY.set_placeholder_metadata(
        model_type,
        MultimodalPlaceholderMetadata(
            placeholder_map={"video": "<video>"},
            placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
            content_format=ContentFormat.PASSTHROUGH,
        ),
    )
    try:
        [inputs] = default_multimodal_input_loader(
            tokenizer=None,
            model_dir="unused",
            model_type=model_type,
            modality="video",
            prompts=["Describe the video."],
            media=[sample_video_path],
            image_data_format=image_data_format,
            num_frames=4,
            media_io_kwargs=media_io_kwargs,
        )
    finally:
        MULTIMODAL_PLACEHOLDER_REGISTRY.remove_placeholder_metadata(model_type)
    [video] = inputs["multi_modal_data"]["video"]
    if expected_format == "np":
        assert isinstance(video.frames, np.ndarray)
        assert video.frames.shape == (4, 64, 64, 3)
    else:
        assert [tuple(frame.shape) for frame in video.frames] == [(3, 64, 64)] * 4


@pytest.mark.parametrize(
    ("module_name", "class_name", "video_format"),
    [
        # Their video paths read CHW tensors or PIL images.
        ("modeling_nemotron_h_multimodal", "NemotronHMultimodalInputProcessor", "pt"),
        ("modeling_kimi_k25", "KimiK25InputProcessor", "pt"),
        ("modeling_vila", "VilaInputProcessor", "pt"),
        # Their HF processors take uint8 frames, so they use the loader default.
        ("modeling_gemma4mm", "Gemma4InputProcessor", None),
        ("modeling_gemma4_unified", "Gemma4UnifiedInputProcessor", None),
        ("modeling_glm5_next_vision", "Glm5NextInputProcessor", None),
        ("modeling_minicpmv4_6", "MiniCPMV4_6InputProcessor", None),
    ],
)
def test_model_video_format_preference(
    module_name: str, class_name: str, video_format: str | None
) -> None:
    """Models whose processors need float video frames ask for `"pt"`."""
    module = importlib.import_module(f"tensorrt_llm._torch.models.{module_name}")
    processor_cls = getattr(module, class_name)
    # The preference does not depend on the checkpoint, so skip `__init__`.
    preferences = processor_cls.__new__(processor_cls).get_preferred_media_io_kwargs()
    assert preferences.get("video", {}).get("format") == video_format


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [
        ("qwen2_5_vl", {"image": {"format": "pt"}, "video": {"format": "pt"}}),
        # Registered with the same processor as "cosmos3".
        (
            "cosmos3_omni",
            {"image": {"format": "np"}, "video": {"format": "np", "num_frames": -1, "fps": -1}},
        ),
        ("gemma4", {}),
        # Its preference reads the checkpoint config, so helpers use the loader defaults.
        ("muse_spark", {}),
        ("not_a_registered_model_type", {}),
    ],
)
def test_preferred_media_io_kwargs_by_model_type(model_type: str, expected: dict) -> None:
    assert INPUT_PROCESSOR_REGISTRY.get_preferred_media_io_kwargs(model_type) == expected


@pytest.mark.parametrize(
    ("model_type", "media_io_kwargs", "video_format"),
    [
        # Declared "pt": their video paths need float frames.
        ("qwen2_5_vl", None, "pt"),
        ("NemotronH_Nano_VL_V2", None, "pt"),
        # Declared "np", and nothing declared (the loader default).
        ("qwen3_vl", None, "np"),
        ("gemma4", None, "np"),
        # The caller's kwargs win, as per-request kwargs do in trtllm-serve.
        ("qwen2_5_vl", {"video": {"format": "np"}}, "np"),
    ],
)
@pytest.mark.parametrize("helper", ["async_build_multimodal_prompt", "default_loader"])
def test_prompt_helpers_follow_model_video_format(
    monkeypatch: pytest.MonkeyPatch,
    sample_video_path: str,
    helper: str,
    model_type: str,
    media_io_kwargs: dict | None,
    video_format: str,
) -> None:
    """Helpers that name the model decode video in the format it declares."""

    # Only the media path is under test: skip loading and rendering the template.
    async def render(**kwargs) -> str:
        return "rendered"

    monkeypatch.setattr(input_utils, "async_apply_chat_template", render)
    monkeypatch.setattr(input_utils, "apply_chat_template", lambda **kwargs: "rendered")
    monkeypatch.setattr(input_utils, "resolve_hf_chat_template", lambda *args: None)
    monkeypatch.setattr(input_utils, "AutoProcessor", Mock())
    if helper == "async_build_multimodal_prompt":
        inputs = asyncio.run(
            async_build_multimodal_prompt(
                model_type=model_type,
                tokenizer=Mock(),
                processor=None,
                prompt="Describe the video.",
                media=[sample_video_path],
                modality="video",
                media_io_kwargs=media_io_kwargs,
            )
        )
    else:
        [inputs] = default_multimodal_input_loader(
            tokenizer=Mock(),
            model_dir="unused",
            model_type=model_type,
            modality="video",
            prompts=["Describe the video."],
            media=[sample_video_path],
            num_frames=4,
            media_io_kwargs=media_io_kwargs,
        )
    [video] = inputs["multi_modal_data"]["video"]
    if video_format == "np":
        assert isinstance(video.frames, np.ndarray)
        assert video.frames.dtype == np.uint8
    else:
        assert all(
            isinstance(frame, torch.Tensor) and frame.dtype == torch.float32
            for frame in video.frames
        )


def test_np_format_hits_hf_video_processor_fast_path(sample_video_path: str) -> None:
    """HF `make_batched_videos` returns a 4D ndarray input without copying."""
    video = _load_video_by_cv2(sample_video_path, num_frames=10, fps=-1, format="np")
    batched = make_batched_videos([video.frames])

    assert len(batched) == 1
    assert np.shares_memory(video.frames, batched[0])


@pytest.mark.parametrize(("format", "peak_bytes_per_value"), [("np", 2), ("pil", 2), ("pt", 9)])
def test_decode_limit_is_checked_before_frame_stack_allocation(
    sample_video_path: str, format: str, peak_bytes_per_value: int
) -> None:
    peak_bytes = 10 * 64 * 64 * 3 * peak_bytes_per_value

    with patch("tensorrt_llm.inputs.media_io.np.empty") as allocate:
        with pytest.raises(ValueError, match=f"require {peak_bytes} bytes"):
            _load_video_by_cv2(
                sample_video_path,
                num_frames=10,
                fps=-1,
                format=format,
                max_decoded_frame_bytes=peak_bytes - 1,
            )

    allocate.assert_not_called()


def test_pt_decode_reserves_peak_then_commits_retained_bytes(sample_video_path: str) -> None:
    class _Reservation:
        committed = None
        released = False

        def check_cancelled(self) -> None:
            pass

        def commit(self, retained_bytes: int) -> None:
            self.committed = retained_bytes

        def release(self) -> None:
            self.released = True

    reservation = _Reservation()
    reserved = []

    def reserve(peak_bytes: int, retained_bytes: int) -> _Reservation:
        reserved.append((peak_bytes, retained_bytes))
        return reservation

    _load_video_by_cv2(
        sample_video_path,
        num_frames=10,
        fps=-1,
        format="pt",
        decoded_memory_reserver=reserve,
    )

    assert reserved == [(10 * 64 * 64 * 3 * 9, 10 * 64 * 64 * 3 * 4)]
    assert reservation.committed == 10 * 64 * 64 * 3 * 4
    assert reservation.released is False


def test_decode_releases_reservation_after_allocation_failure(sample_video_path: str) -> None:
    class _Reservation:
        released = False

        def check_cancelled(self) -> None:
            pass

        def commit(self, retained_bytes: int) -> None:
            raise AssertionError("failed decode must not commit")

        def release(self) -> None:
            self.released = True

    reservation = _Reservation()

    with patch("tensorrt_llm.inputs.media_io.np.empty", side_effect=MemoryError("allocation")):
        with pytest.raises(MemoryError, match="allocation"):
            _load_video_by_cv2(
                sample_video_path,
                num_frames=10,
                fps=-1,
                format="np",
                decoded_memory_reserver=lambda _peak_bytes, _retained_bytes: reservation,
            )

    assert reservation.released is True


def test_truncated_numpy_video_owns_only_the_counted_frames(sample_video_path: str) -> None:
    capture = cv2.VideoCapture(sample_video_path)
    get = capture.get
    committed = []

    class _Reservation:
        def check_cancelled(self) -> None:
            pass

        def commit(self, retained_bytes: int) -> None:
            committed.append(retained_bytes)

        def release(self) -> None:
            pass

    # Container metadata overstates the number of frames that actually decode.
    truncated = Mock(wraps=capture)
    truncated.get.side_effect = lambda key: 50 if key == cv2.CAP_PROP_FRAME_COUNT else get(key)
    with patch.object(cv2, "VideoCapture", return_value=truncated):
        video = _load_video_by_cv2(
            sample_video_path,
            num_frames=10,
            fps=-1,
            format="np",
            decoded_memory_reserver=lambda _peak, _retained: _Reservation(),
        )

    assert 0 < len(video.frames) < 10
    assert video.frames.base is None
    assert committed == [video.frames.nbytes]


@pytest.mark.parametrize("input_kind", ["path", "file_uri", "data_uri"])
@pytest.mark.parametrize("force_tempfile", [False, True], ids=["auto_backend", "tempfile"])
def test_offline_loaders_match_serve_video_hash(
    sample_video_path: str,
    input_kind: str,
    force_tempfile: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The offline loaders must produce the same cache identity as serve's `VideoMediaIO`."""
    path = Path(sample_video_path)
    encoded = path.read_bytes()
    monkeypatch.setattr(media_io, "_MAX_RESPONSE_BYTES", len(encoded))
    if input_kind == "file_uri":
        source = path.as_uri()
    elif input_kind == "data_uri":
        source = "data:video/mp4;base64," + base64.b64encode(encoded).decode("ascii")
    else:
        source = str(path)
    if force_tempfile:
        monkeypatch.setattr(
            "tensorrt_llm.inputs.media_io._select_cv2_stream_buffered_backend", lambda: None
        )

    serve = asyncio.run(VideoMediaIO(num_frames=10).async_load(source))
    offline = load_video(source, num_frames=10)
    offline_async = asyncio.run(async_load_video(source, num_frames=10))

    assert serve.raw_bytes_hash == default_hasher(encoded).hexdigest()
    assert offline.raw_bytes_hash == serve.raw_bytes_hash
    assert offline_async.raw_bytes_hash == serve.raw_bytes_hash
    assert _digest(offline) == _digest(serve) == _digest(offline_async)


@pytest.mark.parametrize("file_uri", [False, True], ids=["path", "file_uri"])
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("read_error", [FileNotFoundError, PermissionError])
def test_offline_local_video_read_errors_are_value_errors(
    tmp_path: Path,
    file_uri: bool,
    use_async: bool,
    read_error: type[OSError],
) -> None:
    """Local read failures retain the offline loaders' ValueError contract."""
    path = tmp_path / "input video.mp4"
    path.write_bytes(b"not decoded")
    source = path.as_uri() if file_uri else str(path)
    failure = read_error("cannot read video")

    with patch.object(Path, "open", side_effect=failure):
        with pytest.raises(ValueError) as error:
            if use_async:
                asyncio.run(async_load_video(source))
            else:
                load_video(source)

    assert error.value.__cause__ is failure


@pytest.mark.parametrize("file_uri", [False, True], ids=["path", "file_uri"])
@pytest.mark.parametrize("loader", ["serve", "sync", "async"])
def test_oversized_local_video_is_rejected_before_read(
    sample_video_path: str,
    file_uri: bool,
    loader: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = Path(sample_video_path)
    source = path.as_uri() if file_uri else str(path)
    monkeypatch.setattr(media_io, "_MAX_RESPONSE_BYTES", path.stat().st_size - 1)

    with patch.object(Path, "open", side_effect=AssertionError("Oversized file was opened")):
        with pytest.raises(ValueError, match="maximum allowed size"):
            if loader == "serve":
                asyncio.run(VideoMediaIO().async_load(source))
            elif loader == "async":
                asyncio.run(async_load_video(source))
            else:
                load_video(source)


def test_local_media_read_is_bounded_if_file_grows(tmp_path: Path) -> None:
    path = tmp_path / "growing.mp4"
    path.write_bytes(b"x")
    source = BytesIO(b"x" * 100)
    with (
        patch.object(Path, "open", return_value=source),
        patch.object(source, "read", wraps=source.read) as read,
    ):
        with pytest.raises(ValueError, match="maximum allowed size"):
            media_io._safe_read_local_file(str(path), max_bytes=4)
        requested_sizes = [call.args[0] for call in read.call_args_list]
        assert requested_sizes and all(size >= 0 for size in requested_sizes)
        assert sum(requested_sizes) <= 5
    assert source.closed


def test_oversized_video_bytes_are_rejected_before_hashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(media_io, "_MAX_RESPONSE_BYTES", 4)
    with patch.object(media_io, "blake3", side_effect=AssertionError("Oversized video was hashed")):
        with pytest.raises(ValueError, match="maximum allowed size"):
            VideoMediaIO().load_bytes(b"video")


@pytest.mark.parametrize("loader", ["serve", "sync", "async", "base64_helper"])
@pytest.mark.parametrize("extra_bytes", [1, 4])
def test_oversized_base64_video_is_rejected(
    loader: str,
    extra_bytes: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(media_io, "_MAX_RESPONSE_BYTES", 4)
    encoded = base64.b64encode(b"x" * (4 + extra_bytes)).decode("ascii")
    source = "data:video/mp4;base64," + encoded
    decoder = media_io.base64.b64decode
    with (
        patch.object(media_io.base64, "b64decode", wraps=decoder) as decode,
        patch.object(media_io, "blake3", side_effect=AssertionError("Oversized video was hashed")),
    ):
        with pytest.raises(ValueError, match="maximum allowed size"):
            if loader == "serve":
                asyncio.run(VideoMediaIO().async_load(source))
            elif loader == "async":
                asyncio.run(async_load_video(source))
            elif loader == "base64_helper":
                load_base64_video(source)
            else:
                load_video(source)
        if extra_bytes == 4:
            decode.assert_not_called()


def test_base64_video_helper_accepts_input_at_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(media_io, "_MAX_RESPONSE_BYTES", 4)
    assert load_base64_video("data:video/mp4;base64,eHh4eA==") == b"xxxx"
