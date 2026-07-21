# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for :mod:`tensorrt_llm.media.tensor_payload`.

Covers the rank-aware batch semantics on
:meth:`VisualGenOutput.save` and :meth:`VisualGenOutput._save_bytes`
across both supported tokens (``"safetensors"`` and ``"pt"``).
"""

from __future__ import annotations

import io

import pytest
import torch

from tensorrt_llm.media.tensor_payload import (
    TENSOR_FORMATS,
    infer_batch_size,
    is_tensor_format,
    serialize_visual_gen_output,
)
from tensorrt_llm.visual_gen.output import VisualGenOutput


def _safetensors_load(data: bytes) -> dict:
    from safetensors.torch import load as load_safetensors

    return load_safetensors(data)


def _pt_load(data: bytes) -> dict:
    return torch.load(io.BytesIO(data), weights_only=True)


def _make_image_output(batch: int = 1, h: int = 8, w: int = 8) -> VisualGenOutput:
    """Image tensor uses canonical ``(B, H, W, C)`` shape."""
    img = torch.arange(batch * h * w * 3, dtype=torch.uint8).reshape(batch, h, w, 3)
    return VisualGenOutput(request_id=1, image=img)


def _make_video_output(batch: int = 1, t: int = 2, h: int = 4, w: int = 4) -> VisualGenOutput:
    """Video uses ``(B, T, H, W, C)``; LTX-2-style audio uses ``(B, channels, T_audio)``."""
    vid = torch.arange(batch * t * h * w * 3, dtype=torch.uint8).reshape(batch, t, h, w, 3)
    audio = torch.arange(batch * 2 * 32, dtype=torch.float32).reshape(batch, 2, 32) / 100.0
    return VisualGenOutput(
        request_id=2,
        video=vid,
        audio=audio,
        frame_rate=24.0,
        audio_sample_rate=16000,
    )


class TestIsTensorFormat:
    def test_accepts_supported_tokens(self):
        for token in TENSOR_FORMATS:
            assert is_tensor_format(token)

    def test_rejects_encoder_tokens(self):
        for token in ("png", "webp", "jpeg", "mp4", "avi", "auto", None, "npz"):
            assert not is_tensor_format(token)


class TestInferBatchSize:
    def test_image_rank_4_is_batched(self):
        output = _make_image_output(batch=3)
        assert infer_batch_size(output) == 3

    def test_image_rank_3_is_unbatched(self):
        output = VisualGenOutput(request_id=1, image=torch.zeros(8, 8, 3, dtype=torch.uint8))
        assert infer_batch_size(output) == 1

    def test_video_rank_5_is_batched(self):
        output = _make_video_output(batch=2)
        assert infer_batch_size(output) == 2

    def test_video_rank_4_is_unbatched(self):
        """An unbatched video has shape ``(T, H, W, C)``; the frame axis
        must not be confused with a batch dimension."""
        video = torch.zeros(8, 4, 4, 3, dtype=torch.uint8)
        output = VisualGenOutput(request_id=1, video=video, frame_rate=12.0)
        assert infer_batch_size(output) == 1

    def test_inconsistent_batches_raise(self):
        # Image rank-4 batch=2 vs video rank-5 batch=3 — must error out.
        output = VisualGenOutput(
            request_id=1,
            image=torch.zeros(2, 4, 4, 3, dtype=torch.uint8),
            video=torch.zeros(3, 2, 4, 4, 3, dtype=torch.uint8),
            frame_rate=24.0,
        )
        with pytest.raises(ValueError, match="Inconsistent batch sizes"):
            infer_batch_size(output)

    def test_no_media_raises(self):
        with pytest.raises(ValueError, match="carries no media"):
            infer_batch_size(VisualGenOutput(request_id=1))


@pytest.mark.parametrize("fmt", ["safetensors", "pt"])
class TestSingleSavePath:
    """A single path writes one logical output. Unbatched tensors and
    ``batch == 1`` tensors save as-is; ``batch > 1`` raises ``ValueError``
    to force the caller to pass a list of paths."""

    def test_unbatched_image_writes_full_tensor(self, fmt, tmp_path):
        img = torch.arange(8 * 8 * 3, dtype=torch.uint8).reshape(8, 8, 3)
        output = VisualGenOutput(request_id=1, image=img)
        target = tmp_path / "img"
        saved = output.save(target, format=fmt)
        loaded = (_safetensors_load if fmt == "safetensors" else _pt_load)(saved.read_bytes())
        assert loaded["image"].shape == (8, 8, 3)
        assert torch.equal(loaded["image"], img)

    def test_batched_image_single_path_raises(self, fmt, tmp_path):
        output = _make_image_output(batch=3)
        target = tmp_path / "img"
        with pytest.raises(ValueError, match="batched tensor of size 3"):
            output.save(target, format=fmt)

    def test_batched_video_single_path_raises(self, fmt, tmp_path):
        output = _make_video_output(batch=2)
        target = tmp_path / "vid"
        with pytest.raises(ValueError, match="batched tensor of size 2"):
            output.save(target, format=fmt)

    def test_path_suffix_is_normalized(self, fmt, tmp_path):
        output = _make_image_output(batch=1)
        saved = output.save(tmp_path / "no_ext", format=fmt)
        assert saved.suffix == f".{fmt}"


@pytest.mark.parametrize("fmt", ["safetensors", "pt"])
class TestListSavePath:
    """A list of paths writes one payload per batch item."""

    def test_batched_image_fans_out(self, fmt, tmp_path):
        output = _make_image_output(batch=3)
        paths = [tmp_path / f"img_{i}" for i in range(3)]
        saved = output.save(paths, format=fmt)
        assert len(saved) == 3
        for i, p in enumerate(saved):
            loaded = (_safetensors_load if fmt == "safetensors" else _pt_load)(p.read_bytes())
            assert loaded["image"].shape == (8, 8, 3)
            assert torch.equal(loaded["image"], output.image[i])

    def test_path_count_mismatch_raises(self, fmt, tmp_path):
        output = _make_image_output(batch=3)
        with pytest.raises(ValueError, match="does not match batch size"):
            output.save([tmp_path / "img_0", tmp_path / "img_1"], format=fmt)

    def test_batched_video_fans_out_with_audio_slice(self, fmt, tmp_path):
        output = _make_video_output(batch=2)
        paths = [tmp_path / f"vid_{i}" for i in range(2)]
        saved = output.save(paths, format=fmt)
        for i, p in enumerate(saved):
            loaded = (_safetensors_load if fmt == "safetensors" else _pt_load)(p.read_bytes())
            assert loaded["video"].shape == (2, 4, 4, 3)
            assert loaded["audio"].shape == (2, 32)
            assert torch.equal(loaded["video"], output.video[i])
            assert torch.equal(loaded["audio"], output.audio[i])


@pytest.mark.parametrize("fmt", ["safetensors", "pt"])
class TestSaveBytes:
    """``_save_bytes`` returns the same payload as :meth:`save` for the
    bytes-based transport."""

    def test_batch_index_slices_image(self, fmt):
        output = _make_image_output(batch=2)
        for i in range(2):
            data = output._save_bytes(fmt, batch_index=i)
            loaded = (_safetensors_load if fmt == "safetensors" else _pt_load)(data)
            assert loaded["image"].shape == (8, 8, 3)
            assert torch.equal(loaded["image"], output.image[i])

    def test_batch_index_none_writes_unbatched_as_is(self, fmt):
        output = VisualGenOutput(
            request_id=1,
            image=torch.zeros(4, 4, 3, dtype=torch.uint8),
        )
        data = output._save_bytes(fmt, batch_index=None)
        loaded = (_safetensors_load if fmt == "safetensors" else _pt_load)(data)
        assert loaded["image"].shape == (4, 4, 3)

    def test_rejects_image_encoder_format(self, fmt):
        # The bytes-based path is tensor-only; image/video encoders use
        # the file-based ``save`` API.
        output = _make_image_output(batch=1)
        with pytest.raises(ValueError, match="tensor formats"):
            output._save_bytes("png")


class TestSerializeDirect:
    """Direct calls to :func:`serialize_visual_gen_output` for low-level
    coverage of the rank-aware behavior."""

    def test_unbatched_image_not_sliced(self):
        output = VisualGenOutput(request_id=1, image=torch.zeros(7, 5, 3, dtype=torch.uint8))
        data = serialize_visual_gen_output(output, "safetensors", batch_index=0)
        loaded = _safetensors_load(data)
        # Height axis must survive; the helper must not confuse it with
        # a batch axis on a rank-3 image.
        assert loaded["image"].shape == (7, 5, 3)

    def test_unbatched_video_not_sliced(self):
        video = torch.zeros(9, 4, 4, 3, dtype=torch.uint8)
        output = VisualGenOutput(request_id=1, video=video, frame_rate=24.0)
        data = serialize_visual_gen_output(output, "safetensors", batch_index=0)
        loaded = _safetensors_load(data)
        # Frame axis must survive on a rank-4 video.
        assert loaded["video"].shape == (9, 4, 4, 3)


@pytest.mark.parametrize("fmt", ["safetensors", "pt"])
class TestSaveMetadataOverrides:
    """``frame_rate`` and ``audio_sample_rate`` kwargs on
    :meth:`VisualGenOutput.save` and :meth:`VisualGenOutput._save_bytes`
    override the corresponding fields on the output. Matches the
    video-encoder path's existing override semantics so a caller who
    fills in missing or stale metadata gets it into the serialized
    payload as well."""

    def _load(self, fmt, data):
        return _safetensors_load(data) if fmt == "safetensors" else _pt_load(data)

    def test_save_overrides_unset_metadata(self, fmt, tmp_path):
        """Output carries no rate fields; ``save`` overrides put the
        right metadata into the payload."""
        video = torch.zeros(1, 2, 4, 4, 3, dtype=torch.uint8)
        audio = torch.zeros(1, 2, 16, dtype=torch.float32)
        output = VisualGenOutput(request_id=1, video=video, audio=audio)
        target = tmp_path / "out"
        saved = output.save(target, format=fmt, frame_rate=24.0, audio_sample_rate=16000)
        loaded = self._load(fmt, saved.read_bytes())
        # Both serializers expose scalar metadata through the
        # canonical ``load`` path: pt as native Python values, safetensors
        # as 0-d tensors (which compare equal to the Python scalar).
        assert loaded["frame_rate"] == 24.0
        assert loaded["audio_sample_rate"] == 16000
        if fmt == "safetensors":
            # The string-keyed file header is preserved for consumers that
            # use ``safe_open(...).metadata()`` instead of ``load()``.
            data_bytes = output._save_bytes(
                fmt, batch_index=0, frame_rate=24.0, audio_sample_rate=16000
            )
            import tempfile

            from safetensors import safe_open

            with tempfile.NamedTemporaryFile(suffix=".safetensors") as tf:
                tf.write(data_bytes)
                tf.flush()
                with safe_open(tf.name, framework="pt") as f:
                    meta = f.metadata() or {}
            assert meta.get("frame_rate") == "24.0"
            assert meta.get("audio_sample_rate") == "16000"

    def test_save_overrides_take_precedence(self, fmt, tmp_path):
        """Override values win even when the output has its own."""
        video = torch.zeros(1, 2, 4, 4, 3, dtype=torch.uint8)
        output = VisualGenOutput(
            request_id=1, video=video, frame_rate=12.0, audio_sample_rate=24000
        )
        target = tmp_path / "out"
        saved = output.save(target, format=fmt, frame_rate=60.0, audio_sample_rate=48000)
        loaded = self._load(fmt, saved.read_bytes())
        assert loaded["frame_rate"] == 60.0
        assert loaded["audio_sample_rate"] == 48000

    def test_save_bytes_overrides(self, fmt):
        """``_save_bytes`` honors the same overrides for the
        ``b64_json`` transport."""
        video = torch.zeros(1, 2, 4, 4, 3, dtype=torch.uint8)
        output = VisualGenOutput(request_id=1, video=video)
        data = output._save_bytes(fmt, batch_index=0, frame_rate=30.0, audio_sample_rate=44100)
        loaded = self._load(fmt, data)
        assert loaded["frame_rate"] == 30.0
        assert loaded["audio_sample_rate"] == 44100


class TestSaveFormatInference:
    """When the caller omits ``format``, ``VisualGenOutput.save`` infers
    the serializer from the path suffix — the same contract the image
    and video encoders honor for ``.png`` / ``.mp4`` / etc."""

    def test_safetensors_suffix_dispatches_to_tensor_path(self, tmp_path):
        output = _make_image_output(batch=1)
        saved = output.save(tmp_path / "out.safetensors")
        assert saved.suffix == ".safetensors"
        loaded = _safetensors_load(saved.read_bytes())
        # Reaching the tensor path means the image lives under the
        # ``image`` key; an encoder fallback would have produced a PNG
        # with no parsable safetensors structure.
        assert loaded["image"].shape == (8, 8, 3)

    def test_pt_suffix_dispatches_to_tensor_path(self, tmp_path):
        output = _make_image_output(batch=1)
        saved = output.save(tmp_path / "out.pt")
        assert saved.suffix == ".pt"
        loaded = _pt_load(saved.read_bytes())
        assert loaded["image"].shape == (8, 8, 3)

    def test_list_path_inference_when_all_tensor(self, tmp_path):
        output = _make_image_output(batch=2)
        paths = [tmp_path / f"img_{i}.safetensors" for i in range(2)]
        saved = output.save(paths)
        assert all(p.suffix == ".safetensors" for p in saved)
        for i, p in enumerate(saved):
            loaded = _safetensors_load(p.read_bytes())
            assert torch.equal(loaded["image"], output.image[i])

    def test_mixed_list_paths_skip_inference(self, tmp_path):
        """A list of paths with mixed suffixes does not match a single
        tensor format; inference returns ``None`` and the dispatch
        falls through to the encoder path (no inferred-format wrong
        file). The encoder behavior on mixed paths is owned by the
        encoder layer and not asserted here."""
        from tensorrt_llm.visual_gen.output import _infer_format_from_path

        assert _infer_format_from_path([tmp_path / "a.safetensors", tmp_path / "b.png"]) is None

    def test_image_encoder_suffix_still_uses_encoder(self, tmp_path):
        """A ``.png`` path with no explicit ``format`` keeps the
        encoder path (regression guard for the inference logic)."""
        output = _make_image_output(batch=1)
        saved = output.save(tmp_path / "out.png")
        assert saved.suffix == ".png"
        # PNG magic bytes confirm the encoder path ran.
        assert saved.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
