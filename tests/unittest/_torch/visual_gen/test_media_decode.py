# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct tests for :mod:`tensorrt_llm._torch.visual_gen.media_decode`.

The decode tests run on the checked-in H.264 fixtures (see
``test_data/README.md`` for provenance). Each fixture frame encodes its own
display index three ways — red-channel ramp, green horizontal bar, blue
vertical bar — so ordering, channel layout, and surface ownership are all
observable from content alone.

GPU (NVDEC) tests skip only on CUDA-less platforms; a missing PyNvVideoCodec
on a supported platform fails them (it is a declared dependency).
"""

import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from tensorrt_llm._torch.visual_gen.media_decode import (
    MediaDecodeError,
    _lanczos_taps,
    decode_video_reference_window,
    max_reference_decode_frames,
    resize_center_crop_uint8,
    synchronize_media_prepare_status,
)

_TEST_DATA = Path(__file__).parent / "test_data"
_MP4 = _TEST_DATA / "cosmos3_v2v_ref_9f_bframes.mp4"
_AVI = _TEST_DATA / "cosmos3_v2v_ref_9f_bframes.avi"


def _frame_indices(frames: torch.Tensor) -> list[int]:
    """Recover each frame's display index from the red-channel ramp."""
    return [round((f[:, :, 0].float().mean().item() - 20) / 25) for f in frames]


class TestResizeCenterCrop:
    """CPU-runnable checks of the shared Lanczos resize/crop front."""

    def _pil_reference(self, frames_u8: torch.Tensor, target_h: int, target_w: int):
        """The reference implementation's geometry: PIL LANCZOS cover-scale
        + center crop (vllm-omni ``_preprocess_condition_image``)."""
        out = []
        for frame in frames_u8.numpy():
            h, w = frame.shape[:2]
            scale = max(target_w / w, target_h / h)
            resize_w = int(math.ceil(scale * w))
            resize_h = int(math.ceil(scale * h))
            img = Image.fromarray(frame).resize((resize_w, resize_h), Image.Resampling.LANCZOS)
            left = (resize_w - target_w) // 2
            top = (resize_h - target_h) // 2
            out.append(np.asarray(img.crop((left, top, left + target_w, top + target_h))))
        return torch.from_numpy(np.stack(out))

    def test_parity_with_pil_lanczos(self):
        rng = np.random.default_rng(7)
        frames = torch.from_numpy(rng.integers(0, 256, (3, 64, 96, 3), dtype=np.uint8))
        for target_h, target_w in ((32, 48), (48, 32), (128, 96)):
            ours = resize_center_crop_uint8(frames, target_h, target_w)
            ref = self._pil_reference(frames, target_h, target_w)
            assert ours.shape == ref.shape == (3, target_h, target_w, 3)
            diff = (ours.int() - ref.int()).abs()
            # PIL quantizes filter coefficients to fixed point; we keep
            # float. Bounded, not bit-exact.
            assert diff.max().item() <= 2, f"{target_h}x{target_w}: max diff {diff.max()}"

    def test_native_resolution_is_identity(self):
        frames = torch.zeros(2, 32, 32, 3, dtype=torch.uint8)
        assert resize_center_crop_uint8(frames, 32, 32) is frames

    def test_taps_are_local_support_and_cached(self):
        # The taps depend only on (in, out) sizes — one build per clip, not
        # one per frame — and their width is the filter's true support
        # (K = ceil(2 * a * scale) + 1), NOT the full input row: this is what
        # makes the resample O(out * K) instead of O(out * in).
        _lanczos_taps.cache_clear()
        weights, taps = _lanczos_taps(1920, 1280, "cpu")
        assert taps.shape == weights.shape
        assert taps.shape[1] <= 2 * math.ceil(3 * (1920 / 1280)) + 1  # K = 10 << 1920
        frames = torch.zeros(4, 40, 40, 3, dtype=torch.uint8)
        resize_center_crop_uint8(frames, 20, 20)
        info = _lanczos_taps.cache_info()
        resize_center_crop_uint8(frames, 20, 20)
        assert _lanczos_taps.cache_info().hits >= info.hits + 2


class TestImportIsolation:
    def test_import_tensorrt_llm_does_not_load_pynvvideocodec(self):
        """PyNvVideoCodec is driver-linked; ``import tensorrt_llm`` (and the
        decode module itself) must not load it — only an actual decode may."""
        code = (
            "import sys; import tensorrt_llm; "
            "import tensorrt_llm._torch.visual_gen.media_decode; "
            "assert 'PyNvVideoCodec' not in sys.modules, "
            "'driver-linked PyNvVideoCodec loaded at import time'"
        )
        subprocess.run([sys.executable, "-c", code], check=True, timeout=600)


def _status_protocol_rank(rank: int, world_size: int, init_file: str, results_dir: str):
    """Spawn target: run the convergence protocol with rank 1 failing."""
    import os

    # Containers often have hostnames that don't resolve to a usable
    # interface; pin gloo to loopback or its rendezvous hangs.
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    os.environ.setdefault("TLLM_DISABLE_MPI", "1")

    import torch.distributed as dist

    from tensorrt_llm._torch.visual_gen.media_decode import (
        MediaDecodeError,
        synchronize_media_prepare_status,
    )

    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        local_error = MediaDecodeError("rank-local decode failure") if rank == 1 else None
        try:
            synchronize_media_prepare_status(local_error)
            outcome = "no-error"
        except MediaDecodeError as exc:
            outcome = f"client:{exc}"
        except Exception as exc:  # pragma: no cover - diagnostic path
            outcome = f"unexpected:{type(exc).__name__}:{exc}"
    finally:
        dist.destroy_process_group()
    Path(results_dir, f"rank{rank}.txt").write_text(outcome)


class TestPrepareStatusProtocol:
    def test_local_failure_is_reraised_without_group(self):
        err = MediaDecodeError("boom")
        with pytest.raises(MediaDecodeError, match="boom"):
            synchronize_media_prepare_status(err)

    def test_success_passes_through_without_group(self):
        synchronize_media_prepare_status(None)

    def test_two_rank_convergence_over_gloo(self, tmp_path):
        """Rank 1 fails decode, rank 0 is healthy: both must exit the
        protocol with the SAME client error — rank 1 re-raising its own,
        rank 0 raising the reconstructed equivalent — instead of rank 0
        proceeding into (and hanging in) model collectives."""
        import torch.multiprocessing as mp

        init_file = tmp_path / "gloo_init"
        ctx = mp.spawn(
            _status_protocol_rank,
            args=(2, str(init_file), str(tmp_path)),
            nprocs=2,
            join=False,
        )
        # ``ProcessContext.join`` returns False whenever ANY child has
        # exited but others still run — it is designed to be polled.
        deadline = time.monotonic() + 240
        converged = False
        while time.monotonic() < deadline:
            if ctx.join(timeout=deadline - time.monotonic()):
                converged = True
                break
        if not converged:
            for proc in ctx.processes:
                if proc.is_alive():
                    proc.terminate()
        assert converged, "status protocol did not converge (hang)"
        healthy = (tmp_path / "rank0.txt").read_text()
        failing = (tmp_path / "rank1.txt").read_text()
        assert failing == "client:rank-local decode failure"
        assert healthy == "client:[rank 1] rank-local decode failure"


class TestDecodeFrameLimit:
    def test_default_and_env_override(self, monkeypatch):
        monkeypatch.delenv("TRTLLM_MAX_REFERENCE_DECODE_FRAMES", raising=False)
        assert max_reference_decode_frames() == 7200
        monkeypatch.setenv("TRTLLM_MAX_REFERENCE_DECODE_FRAMES", "12")
        assert max_reference_decode_frames() == 12
        monkeypatch.setenv("TRTLLM_MAX_REFERENCE_DECODE_FRAMES", "0")
        assert max_reference_decode_frames() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestDecodeVideoReferenceWindow:
    _DEVICE = torch.device("cuda:0")

    def _decode(self, data: bytes, **kwargs):
        defaults = dict(window=5, keep="first", target_h=64, target_w=64, device=self._DEVICE)
        defaults.update(kwargs)
        return decode_video_reference_window(data, **defaults)

    @pytest.mark.parametrize("fixture", [_MP4, _AVI], ids=["mp4", "avi"])
    def test_keep_first_display_order(self, fixture):
        # Display order despite forced B-frames, in both supported containers.
        window = self._decode(fixture.read_bytes())
        assert window.shape == (5, 64, 64, 3) and window.dtype == torch.uint8
        assert window.device.type == "cuda"
        assert _frame_indices(window) == [0, 1, 2, 3, 4]

    @pytest.mark.parametrize("fixture", [_MP4, _AVI], ids=["mp4", "avi"])
    def test_keep_last_ring_reorder(self, fixture):
        window = self._decode(fixture.read_bytes(), keep="last")
        assert _frame_indices(window) == [4, 5, 6, 7, 8]

    def test_rgb_channel_layout(self):
        # Frame i carries a green horizontal bar at rows [7i, 7i+7) and a
        # blue vertical bar at cols [7i, 7i+7): asserts the NVDEC output is
        # RGB (not BGR) and spatially unflipped. 4:2:0 chroma blurs edges,
        # so thresholds are generous.
        window = self._decode(_MP4.read_bytes(), window=9, keep="first")
        for i in (2, 6):
            frame = window[i].float()
            band = slice(7 * i, 7 * i + 7)
            assert frame[band, :, 1].mean() > 150  # green bar rows
            assert frame[:, band, 2].mean() > 150  # blue bar cols
            assert frame[:, :, 1].mean() < frame[band, :, 1].mean() - 60

    def test_surface_ownership_across_full_decode(self):
        # If the ring held DLPack views instead of owned copies, NVDEC's
        # surface recycling would overwrite earlier frames during the later
        # decodes; per-frame content proves each retained frame is intact
        # after the decoder finished the whole stream.
        window = self._decode(_MP4.read_bytes(), window=9, keep="first")
        assert _frame_indices(window) == list(range(9))

    def test_target_resolution_resize(self):
        window = self._decode(_MP4.read_bytes(), window=3, target_h=96, target_w=128)
        assert window.shape == (3, 96, 128, 3)

    def test_window_longer_than_clip_returns_all(self):
        window = self._decode(_MP4.read_bytes(), window=20)
        assert _frame_indices(window) == list(range(9))

    def test_corrupt_bytes_with_valid_magic_is_client_error(self):
        payload = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64
        with pytest.raises(MediaDecodeError):
            self._decode(payload)

    def test_frame_limit_trips_on_emitted_frames(self, monkeypatch):
        monkeypatch.setenv("TRTLLM_MAX_REFERENCE_DECODE_FRAMES", "4")
        with pytest.raises(MediaDecodeError, match="decode limit"):
            self._decode(_MP4.read_bytes(), keep="last")

    def test_frame_limit_disabled(self, monkeypatch):
        monkeypatch.setenv("TRTLLM_MAX_REFERENCE_DECODE_FRAMES", "0")
        window = self._decode(_MP4.read_bytes(), window=20, keep="last")
        assert window.shape[0] == 9

    def test_resize_perf_representative(self):
        # Representative evidence for the local-tap resample: a 1080p frame
        # to 720p-cover must be in the low-millisecond range. The bound is
        # ~100x actual so it never flakes; a dense O(out*in) implementation
        # (~11 GMAC/frame) would still trip it under contention.
        frame = torch.randint(0, 256, (1, 1080, 1920, 3), dtype=torch.uint8, device=self._DEVICE)
        resize_center_crop_uint8(frame, 720, 1280)  # warmup + tap-cache build
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            resize_center_crop_uint8(frame, 720, 1280)
        torch.cuda.synchronize()
        per_frame = (time.perf_counter() - start) / 10
        assert per_frame < 0.25, f"resize took {per_frame * 1e3:.1f} ms/frame"
