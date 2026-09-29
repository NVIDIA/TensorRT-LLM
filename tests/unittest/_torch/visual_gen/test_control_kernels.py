# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU tests for the VisualGen control-generation kernels.

Every kernel is asserted **bitwise** against the torch reference in
``triton_kernels/reference.py``.  Bitwise, not approximate: these produce
control frames that condition a diffusion model, and the fixed-point
arithmetic they reproduce has no tolerance band -- a 1-LSB drift means one of
the two implementations is simply wrong.  The reference shares the kernels'
axis/tap tables, so a failure points at the arithmetic rather than at setup.

The last class covers the Cosmos3 transfer entry points that compose them.
"""

import os

import pytest
import torch

os.environ["TLLM_DISABLE_MPI"] = "1"

from tensorrt_llm._torch.visual_gen.models.cosmos3 import transfer as transfer_module
from tensorrt_llm._torch.visual_gen.models.cosmos3.transfer import (
    BLUR_PRESETS,
    EDGE_PRESETS,
    make_blur_control,
    make_edge_control,
)
from tensorrt_llm._torch.visual_gen.triton_kernels import (
    bilateral_filter,
    canny_edges,
    reference,
    resize_area_u8,
    resize_cubic_u8,
    resize_linear_u8,
)
from tensorrt_llm._torch.visual_gen.triton_kernels import resize as resize_module

pytestmark = [
    pytest.mark.cosmos3,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="control kernels require CUDA"),
]


def _clip(h: int, w: int, t: int = 3, c: int = 3, *, seed: int = 0) -> torch.Tensor:
    """Smoothly-varying uint8 ``[T, H, W, C]``: video-like, so Canny's
    hysteresis and the bilateral colour lookup see realistic neighbourhoods
    instead of white noise."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randint(0, 256, (t, h, w, c), dtype=torch.uint8, device="cuda", generator=g)
    f = x.to(torch.float32)
    return ((f + f.roll(1, 1) + f.roll(1, 2) + f.roll(2, 2)) / 4).to(torch.uint8)


class TestResizeKernels:
    @pytest.mark.parametrize("src", [(90, 160), (181, 321)])
    @pytest.mark.parametrize("scale", ["up", "down", "fractional", "near_identity"])
    def test_linear_matches_reference(self, src, scale):
        h, w = src
        dst = {
            "up": (w * 2, h * 2),
            "down": (w // 2, h // 2),
            "fractional": (int(w * 1.7), int(h * 1.3)),
            "near_identity": (w + 1, h - 1),
        }[scale]
        frames = _clip(h, w)
        got = resize_linear_u8(frames, *dst)
        assert got.shape == (3, dst[1], dst[0], 3) and got.dtype == torch.uint8
        assert torch.equal(got, reference.resize_linear_u8(frames, *dst))

    @pytest.mark.parametrize("src", [(704, 1280), (92, 164)])
    @pytest.mark.parametrize("channels", [1, 2, 3, 4])
    @pytest.mark.parametrize("factor", [2, 4])
    def test_area_matches_reference(self, src, channels, factor):
        # Channel count matters: the C=3 path takes a word-load fast path, and
        # factor 2 with 1/3/4 channels rounds half-up where the rest round
        # half-even.
        h, w = src
        frames = _clip(h, w, c=channels)
        got = resize_area_u8(frames, factor)
        assert got.shape == (3, h // factor, w // factor, channels)
        assert torch.equal(got, reference.resize_area_u8(frames, factor))

    @pytest.mark.parametrize("src", [(90, 160), (704, 1280), (180, 320)])
    @pytest.mark.parametrize("scale", ["up", "down", "fractional", "decimate", "near_identity"])
    def test_cubic_matches_reference(self, src, scale):
        # "decimate" and "near_identity" hit the row-tail path where the
        # vertical pass falls back to integer fixed point.
        h, w = src
        dst = {
            "up": (w * 2, h * 2),
            "down": (w // 2, h // 2),
            "fractional": (int(w * 1.7), int(h * 1.3)),
            "decimate": (max(1, w // 10), max(1, h // 10)),
            "near_identity": (w + 1, h - 1),
        }[scale]
        frames = _clip(h, w)
        got = resize_cubic_u8(frames, *dst)
        assert got.shape == (3, dst[1], dst[0], 3)
        assert torch.equal(got, reference.resize_cubic_u8(frames, *dst))

    @pytest.mark.parametrize("byte_offset", [0, 1, 2, 3])
    def test_area_handles_unaligned_input(self, byte_offset):
        # The C=3 fast path reinterprets the buffer as int32. A contiguous
        # *view* can still start on an odd byte, which faults the device rather
        # than returning wrong data, so the fast path must decline it.
        t, h, w = 2, 64, 96
        n = t * h * w * 3
        g = torch.Generator(device="cuda").manual_seed(0)
        base = torch.randint(0, 256, (n + 4,), dtype=torch.uint8, device="cuda", generator=g)
        frames = base[byte_offset : byte_offset + n].view(t, h, w, 3)
        assert frames.is_contiguous()
        assert torch.equal(resize_area_u8(frames, 2), reference.resize_area_u8(frames, 2))

    def test_axis_table_cache_is_bounded(self):
        # The cache key carries caller-supplied source dimensions, so an
        # unbounded cache would let a long-lived worker retain a GPU table for
        # every resolution it ever served.
        resize_module._cubic_tables_x.cache_clear()
        clip = _clip(64, 96, t=1)
        for dst_w in range(8, 8 + 2 * resize_module._TABLE_CACHE_ENTRIES):
            resize_cubic_u8(clip, dst_w, 16)
        info = resize_module._cubic_tables_x.cache_info()
        assert info.currsize <= resize_module._TABLE_CACHE_ENTRIES

    def test_area_rejects_unsupported_geometry(self):
        frames = _clip(64, 64)
        with pytest.raises(ValueError, match=r"factor=3, expected one of \(2, 4\)"):
            resize_area_u8(frames, 3)
        with pytest.raises(ValueError, match="65x64 not divisible by factor=2"):
            resize_area_u8(_clip(64, 65), 2)


class TestCannyKernel:
    @pytest.mark.parametrize("size", [(96, 128), (704, 1280), (91, 161)])
    @pytest.mark.parametrize("thresholds", sorted(set(EDGE_PRESETS.values())))
    def test_matches_reference(self, size, thresholds):
        h, w = size
        frames = _clip(h, w).permute(3, 0, 1, 2).contiguous()
        low, high = thresholds
        got = canny_edges(frames, low, high)
        assert got.shape == (3, h, w) and got.dtype == torch.uint8
        assert torch.equal(got, reference.canny_edges(frames, low, high))

    def test_output_is_binary(self):
        got = canny_edges(_clip(96, 128).permute(3, 0, 1, 2).contiguous(), 100, 200)
        assert set(got.unique().tolist()) <= {0, 255}

    def test_higher_thresholds_give_sparser_edges(self):
        frames = _clip(96, 128).permute(3, 0, 1, 2).contiguous()
        counts = [
            (canny_edges(frames, lo, hi) > 0).sum().item()
            for lo, hi in sorted(set(EDGE_PRESETS.values()))
        ]
        assert counts == sorted(counts, reverse=True)


class TestBilateralKernel:
    @pytest.mark.parametrize("size", [(48, 64), (128, 96)])
    @pytest.mark.parametrize("params", [(9, 75.0, 75.0), (31, 150.0, 100.0), (13, 60.0, 40.0)])
    def test_matches_reference(self, size, params):
        h, w = size
        frames = _clip(h, w)
        got = bilateral_filter(frames, *params)
        assert got.shape == frames.shape and got.dtype == torch.uint8
        assert torch.equal(got, reference.bilateral_filter(frames, *params))

    def test_preserves_flat_regions(self):
        # Every weight is equal over a constant patch, so the filter is the
        # identity there regardless of sigma.
        flat = torch.full((2, 32, 32, 3), 100, dtype=torch.uint8, device="cuda")
        assert torch.equal(bilateral_filter(flat, 9, 75.0, 75.0), flat)


class TestKernelInputValidation:
    def test_rejects_cpu_tensors(self):
        cpu = torch.zeros(1, 8, 8, 3, dtype=torch.uint8)
        with pytest.raises(ValueError, match="requires a CUDA tensor, got device=cpu"):
            bilateral_filter(cpu, 9, 75.0, 75.0)
        with pytest.raises(ValueError, match="requires a CUDA tensor, got device=cpu"):
            resize_linear_u8(cpu, 4, 4)
        with pytest.raises(ValueError, match="requires a CUDA tensor, got device=cpu"):
            canny_edges(torch.zeros(3, 1, 8, 8, dtype=torch.uint8), 100, 200)

    def test_rejects_non_contiguous(self):
        # The kernels address storage densely and never read strides, so a
        # strided view used to be accepted and silently produce wrong pixels.
        g = torch.Generator(device="cuda").manual_seed(0)
        chw = torch.randint(0, 256, (2, 3, 64, 96), dtype=torch.uint8, device="cuda", generator=g)
        view = chw.permute(0, 2, 3, 1)  # valid [T, H, W, C] shape, not contiguous
        assert not view.is_contiguous()
        for call in (
            lambda: resize_linear_u8(view, 48, 32),
            lambda: resize_cubic_u8(view, 48, 32),
            lambda: resize_area_u8(view, 2),
            lambda: bilateral_filter(view, 9, 75.0, 75.0),
        ):
            with pytest.raises(ValueError, match="requires a contiguous tensor"):
                call()

    def test_canny_rejects_strided_frame_planes(self):
        # canny takes dim 0's stride, but the [T, H, W] block behind it must
        # still be dense -- a channel-permuted view is not.
        g = torch.Generator(device="cuda").manual_seed(0)
        thw = torch.randint(0, 256, (2, 3, 64, 96), dtype=torch.uint8, device="cuda", generator=g)
        view = thw.permute(1, 0, 2, 3)
        assert not view[0].is_contiguous()
        with pytest.raises(ValueError, match="slice along dim 0 must be contiguous"):
            canny_edges(view, 100, 200)

    @pytest.mark.parametrize("window", [(0, 8), (8, 24), (24, 32)])
    def test_canny_reads_a_windowed_clip_in_place(self, window):
        # Slicing [C, T, H, W] along T leaves dim 0 striding over the *whole*
        # clip, so this used to need a .contiguous() copy per window. The result
        # must be bit-identical to materializing it.
        start, stop = window
        frames = _clip(64, 96, t=32).permute(3, 0, 1, 2).contiguous()
        view = frames[:, start:stop]
        assert not view.is_contiguous() and view.stride(0) == 32 * 64 * 96
        assert torch.equal(canny_edges(view, 100, 200), canny_edges(view.contiguous(), 100, 200))

    def test_rejects_non_uint8(self):
        # Silently casting would be a data copy on the inference path; the
        # kernels require the caller to hand over the dtype they expect.
        f32 = torch.zeros(1, 8, 8, 3, dtype=torch.float32, device="cuda")
        with pytest.raises(TypeError, match="requires uint8 frames, got dtype=torch.float32"):
            bilateral_filter(f32, 9, 75.0, 75.0)
        with pytest.raises(TypeError, match="requires uint8 frames, got dtype=torch.float32"):
            resize_cubic_u8(f32, 4, 4)


class TestTransferControlGeneration:
    """The Cosmos3 entry points that compose the kernels above."""

    @pytest.mark.parametrize("preset", sorted(EDGE_PRESETS))
    def test_edge_control_shape_and_broadcast(self, preset):
        frames = _clip(64, 96).permute(3, 0, 1, 2).contiguous()
        edge = make_edge_control(frames, preset)
        assert edge.shape == frames.shape and edge.dtype == torch.uint8
        assert edge.is_cuda and edge.is_contiguous()
        # the single-channel edge map is broadcast across RGB
        assert torch.equal(edge[0], edge[1]) and torch.equal(edge[0], edge[2])

    def test_edge_control_uses_every_channel(self):
        # R and G step in opposite directions, so the luma is flat across the
        # seam: a grayscale-first detector sees nothing, while per-channel
        # selection sees a full-scale edge in both.
        frames = torch.zeros(3, 1, 32, 32, dtype=torch.uint8, device="cuda")
        frames[0, :, :, :16] = 255
        frames[1, :, :, 16:] = 255
        frames[2] = 128
        assert make_edge_control(frames, "medium").any()

    @pytest.mark.parametrize("preset", sorted(BLUR_PRESETS))
    def test_blur_control_shape(self, preset):
        frames = _clip(64, 128).permute(3, 0, 1, 2).contiguous()
        blurred = make_blur_control(frames, preset)
        assert blurred.shape == frames.shape and blurred.dtype == torch.uint8
        assert blurred.is_cuda and blurred.is_contiguous()

    def test_blur_none_preset_is_identity(self):
        frames = _clip(64, 128).permute(3, 0, 1, 2).contiguous()
        assert torch.equal(make_blur_control(frames, "none"), frames)

    def test_blur_reduces_variance(self):
        frames = _clip(64, 128).permute(3, 0, 1, 2).contiguous()
        sharp = frames.to(torch.float32)
        for preset in ("low", "medium", "high"):
            blurred = make_blur_control(frames, preset).to(torch.float32)
            assert blurred.var().item() < sharp.var().item()

    @pytest.mark.parametrize("preset", ["medium", "high"])
    def test_generation_is_window_invariant(self, preset, monkeypatch):
        # Control generation is windowed to bound preprocessing memory. Frames
        # are independent, so the window size is a memory/parallelism knob and
        # must not move a single pixel -- if this fails, some kernel grew a
        # dependency across the temporal axis.
        frames = _clip(64, 128, t=5).permute(3, 0, 1, 2).contiguous()
        monkeypatch.setattr(transfer_module, "CONTROL_FRAME_WINDOW", frames.shape[1])
        edge_unwindowed = make_edge_control(frames, preset)
        blur_unwindowed = make_blur_control(frames, preset)

        for window in (1, 2, 4):
            monkeypatch.setattr(transfer_module, "CONTROL_FRAME_WINDOW", window)
            assert torch.equal(make_edge_control(frames, preset), edge_unwindowed)
            assert torch.equal(make_blur_control(frames, preset), blur_unwindowed)

    @pytest.mark.parametrize("preset", ["nonsense", ""])
    def test_unknown_presets_raise(self, preset):
        frames = _clip(32, 32).permute(3, 0, 1, 2).contiguous()
        with pytest.raises(ValueError, match="Unsupported Cosmos3 edge preset"):
            make_edge_control(frames, preset)
        with pytest.raises(ValueError, match="Unsupported Cosmos3 blur preset"):
            make_blur_control(frames, preset)
