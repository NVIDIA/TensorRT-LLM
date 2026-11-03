# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU tests for MuseSparkOpen vision preprocessing and placeholder expansion.

Two kinds of test live here.

The unit tests pin each preprocessing rule against a golden written out by hand
from the model contract -- canvas fitting, LANCZOS + normalization, patch
vectorization order, still-image temporal duplication, RGBA compositing, video
frame sampling -- so no shared helper can make both sides agree on the wrong
answer.

`TestOracleExpansion` is the gate: it replays a captured reference prompt set
and asserts the expanded prompt ids match token for token. That refset is a
large out-of-tree artifact, so the class is skipped unless
`MUSESPARK_OPEN_VISION_ORACLE` points at a directory holding
`vision-manifest.json`, `media-manifest.json` and `media/`.
"""

import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from tensorrt_llm._torch.models.musespark_open_vision_processing import (
    VISION_TOKEN_IDS,
    ExpandedPrompt,
    MuseSparkOpenPreprocessConfig,
    VisionPreprocessError,
    expand_prompt_placeholders,
    expand_temporal,
    find_resize_target,
    image_placeholder_ids,
    is_embed_mask,
    least_distortion_fit,
    normalize_pixels,
    num_patches,
    num_placeholder_rows,
    patch_dim,
    patchify,
    preprocess_image,
    preprocess_video,
    processor_config_overrides,
    resize_lanczos,
    sample_video_frame_indices,
    to_rgb,
    video_placeholder_ids,
    with_processor_config,
)

pytestmark = pytest.mark.cpu_only

CFG = MuseSparkOpenPreprocessConfig()

FIT_KWARGS = dict(
    patch_size=CFG.patch_size,
    downsample_factor=CFG.downsample_factor,
    min_tokens=CFG.min_num_tokens,
    max_tokens=CFG.max_num_image_tokens,
)


def textured(width, height, mode="RGB"):
    """Deterministic image with both a ramp and a high-frequency component.

    The ramp makes a transposition visible; the high-frequency term makes two
    different resampling filters disagree. Alpha ramps left to right in RGBA.
    """
    xs = np.arange(width)
    ys = np.arange(height)[:, None]
    red = ((xs * 7 + (xs % 2) * 90) % 256).repeat(height).reshape(width, height).T
    green = ((ys * 5 + (ys % 3) * 70) % 256) * np.ones((1, width), dtype=int)
    blue = ((xs + ys) * 3 % 256) + np.zeros((height, width), dtype=int)
    channels = [red, green, blue]
    if mode == "RGBA":
        channels.append((xs * 4 % 256) + np.zeros((height, width), dtype=int))
    array = np.stack(channels, axis=-1).astype(np.uint8)
    return Image.fromarray(array)


class TestLeastDistortionFit:
    """Canvas fitting: multiple of 28, inside the budget, least distortion."""

    # (width, height) -> (canvas_width, canvas_height), worked by hand from
    # ideal = (H/28, W/28) and the floor/ceil combination closest to H/W.
    CASES = [
        ((480, 640), (476, 644)),  # ideal (22.86, 17.14) -> 23x17 = 391
        ((480, 270), (504, 280)),  # ideal (9.64, 17.14)  -> 10x18 = 180
        ((240, 135), (252, 140)),  # ideal (4.82, 8.57)   -> 5x9   = 45
        ((1296, 968), (1316, 980)),  # ideal (34.57, 46.29) -> 35x47 = 1645
        ((640, 360), (644, 364)),  # ideal (12.86, 22.86) -> 13x23 = 299
        ((84, 448), (84, 448)),  # exact (16, 3)        -> 16x3  = 48
        ((448, 448), (448, 448)),  # exact (16, 16)       -> 256
        ((1280, 720), (1288, 728)),  # ideal (25.71, 45.71) -> 26x46 = 1196
    ]

    @pytest.mark.parametrize("size,expected", CASES)
    def test_known_canvases(self, size, expected):
        assert least_distortion_fit(*size, **FIT_KWARGS) == expected

    @pytest.mark.parametrize("width", [1, 7, 29, 133, 640, 1921, 3840])
    @pytest.mark.parametrize("height", [1, 13, 28, 355, 480, 2160])
    def test_canvas_is_a_multiple_of_28_and_within_budget(self, width, height):
        canvas_w, canvas_h = least_distortion_fit(width, height, **FIT_KWARGS)
        assert canvas_w % 28 == 0 and canvas_h % 28 == 0
        assert canvas_w > 0 and canvas_h > 0
        tokens = (canvas_h // 28) * (canvas_w // 28)
        assert CFG.min_num_tokens <= tokens <= CFG.max_num_image_tokens

    def test_oversized_image_is_capped_at_the_token_budget(self):
        canvas_w, canvas_h = least_distortion_fit(3840, 2160, **FIT_KWARGS)
        assert (canvas_h // 28) * (canvas_w // 28) == 4080 <= 4096
        # Aspect ratio survives the shrink to within one cell.
        assert abs(canvas_w / canvas_h - 3840 / 2160) < 28 / canvas_h

    def test_tiny_image_grows_to_the_minimum(self):
        assert least_distortion_fit(4, 4, **FIT_KWARGS) == (28, 28)

    def test_width_first_in_and_out(self):
        """The contract signature is width-first; a transpose must show up."""
        assert least_distortion_fit(84, 448, **FIT_KWARGS) == (84, 448)
        assert least_distortion_fit(448, 84, **FIT_KWARGS) == (448, 84)

    @pytest.mark.parametrize(
        "size,expected",
        [
            ((29, 31), (28, 28)),  # ideal (1.11, 1.04): (1,1) ties (2,2)
            ((64, 64), (84, 84)),  # ideal (2.29, 2.29): (2,2) ties (3,3)
            ((100, 100), (112, 112)),  # ideal (3.57, 3.57): (3,3) ties (4,4)
        ],
    )
    def test_square_image_tie_break(self, size, expected):
        """A near-square image ties two grids that differ ~4x in token count.

        The reference resolves the tie by set iteration order, so these are the
        values that keep our prompt length equal to its. Any "tidier" candidate
        ordering breaks one of these three.
        """
        assert least_distortion_fit(*size, **FIT_KWARGS) == expected

    def test_extreme_aspect_falls_back_to_least_distance(self):
        """No floor/ceil combination fits, so the fallback scan runs."""
        (canvas_h, canvas_w), (nph, npw) = find_resize_target(
            img_h=28 * 200,
            img_w=28,
            token_patch=28,
            min_num_tokens=1,
            max_num_tokens=64,
        )
        assert (nph, npw) == (64, 1)
        assert (canvas_h, canvas_w) == (64 * 28, 28)

    def test_zero_extent_is_rejected(self):
        with pytest.raises(VisionPreprocessError):
            find_resize_target(
                img_h=0, img_w=28, token_patch=28, min_num_tokens=1, max_num_tokens=4096
            )


class TestPixelPipeline:
    """LANCZOS resize, fp32 scaling, RGBA compositing."""

    def test_normalization_is_exact_and_fp32(self):
        pixels = (
            torch.tensor([0, 1, 127, 128, 254, 255], dtype=torch.uint8)
            .repeat(3, 1)
            .reshape(3, 2, 3)
        )
        out = normalize_pixels(pixels.clone(), CFG)
        assert out.dtype == torch.float32
        expected = (pixels.float() / 255.0 - 0.5) / 0.5
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        assert float(out.min()) == -1.0 and float(out.max()) == 1.0

    def test_normalization_rejects_non_uint8_input(self):
        with pytest.raises(VisionPreprocessError):
            normalize_pixels(torch.zeros(3, 4, 4), CFG)

    def test_resize_uses_pil_lanczos(self):
        image = textured(97, 61)
        got = resize_lanczos(image, 56, 84)
        reference = image.resize((84, 56), resample=Image.Resampling.LANCZOS)
        expected = torch.from_numpy(np.asarray(reference).copy()).permute(2, 0, 1)
        assert got.dtype == torch.uint8
        assert tuple(got.shape) == (3, 56, 84)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        # Guard against a silent swap to another filter.
        for other in (Image.Resampling.BICUBIC, Image.Resampling.BILINEAR):
            alternative = torch.from_numpy(
                np.asarray(image.resize((84, 56), resample=other)).copy()
            ).permute(2, 0, 1)
            assert not torch.equal(got, alternative)

    def test_rgba_is_composited_over_white_before_resize(self):
        rgba = textured(64, 32, mode="RGBA")
        got = to_rgb(rgba)
        assert got.mode == "RGB"
        expected = Image.new("RGB", rgba.size, (255, 255, 255))
        expected.paste(rgba, mask=rgba.getchannel("A"))
        assert np.array_equal(np.asarray(got), np.asarray(expected))
        # Fully transparent pixels become white, not the arbitrary stored RGB.
        alpha = np.asarray(rgba)[..., 3]
        assert (alpha == 0).any()
        assert np.all(np.asarray(got)[alpha == 0] == 255)

    def test_grayscale_is_converted(self):
        assert to_rgb(textured(32, 16).convert("L")).mode == "RGB"

    def test_rgb_passes_through_untouched(self):
        rgb = textured(32, 16)
        assert to_rgb(rgb) is rgb


class TestPatchVectorization:
    """[t][c][py][px] layout, clip-major then patch row-major rows."""

    def test_patch_dim_is_1176(self):
        assert patch_dim(CFG) == 3 * 2 * 14 * 14 == 1176

    def test_layout_is_t_c_py_px(self):
        cfg = MuseSparkOpenPreprocessConfig(patch_size=2, patch_temporal=2, downsample_factor=2)
        # Distinct value per (t, c, y, x) so any transposition is visible.
        frames = torch.arange(2 * 3 * 4 * 6, dtype=torch.float32).reshape(2, 3, 4, 6)
        out = patchify(frames, cfg)
        grid_h, grid_w = 4 // 2, 6 // 2
        assert tuple(out.shape) == (grid_h * grid_w, 2 * 3 * 2 * 2)
        for gy in range(grid_h):
            for gx in range(grid_w):
                row = out[gy * grid_w + gx]
                k = 0
                for t in range(2):
                    for c in range(3):
                        for py in range(2):
                            for px in range(2):
                                assert row[k] == frames[t, c, gy * 2 + py, gx * 2 + px]
                                k += 1

    def test_rows_are_clip_major(self):
        cfg = MuseSparkOpenPreprocessConfig(patch_size=2, patch_temporal=2, downsample_factor=2)
        frames = torch.zeros(4, 3, 4, 4)
        frames[0:2] = 1.0  # clip 0
        frames[2:4] = 2.0  # clip 1
        out = patchify(frames, cfg)
        assert tuple(out.shape) == (2 * 2 * 2, 2 * 3 * 2 * 2)
        assert torch.all(out[:4] == 1.0)
        assert torch.all(out[4:] == 2.0)

    def test_odd_frame_count_is_rejected(self):
        with pytest.raises(VisionPreprocessError):
            patchify(torch.zeros(3, 3, 28, 28), CFG)

    def test_unaligned_canvas_is_rejected(self):
        with pytest.raises(VisionPreprocessError):
            patchify(torch.zeros(2, 3, 30, 28), CFG)


class TestTemporalExpansion:
    """A still image is duplicated to patch_temporal frames, never padded."""

    def test_still_image_is_duplicated(self):
        pixels = torch.randn(3, 28, 28)
        frames = expand_temporal(pixels, CFG)
        assert tuple(frames.shape) == (2, 3, 28, 28)
        torch.testing.assert_close(frames[0], frames[1], rtol=0, atol=0)
        torch.testing.assert_close(frames[0], pixels, rtol=0, atol=0)

    def test_duplicated_halves_of_a_patch_vector_are_equal(self):
        """Zero padding instead of duplication would zero the second half."""
        pixel_values, grid_thw = preprocess_image(textured(96, 96), CFG)
        half = pixel_values.shape[1] // 2
        first, second = pixel_values[:, :half], pixel_values[:, half:]
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        assert torch.count_nonzero(second) > 0
        assert grid_thw[0] == 1

    def test_multi_frame_input_passes_through(self):
        frames = torch.randn(4, 3, 28, 28)
        assert expand_temporal(frames, CFG) is frames

    def test_odd_multi_frame_input_is_rejected(self):
        with pytest.raises(VisionPreprocessError):
            expand_temporal(torch.randn(3, 3, 28, 28), CFG)


class TestPreprocessImage:
    @pytest.mark.parametrize(
        "size,grid_thw,rows",
        [
            ((480, 640), (1, 46, 34), 391),
            ((480, 270), (1, 20, 36), 180),
            ((240, 135), (1, 10, 18), 45),
            ((84, 448), (1, 32, 6), 48),
            ((640, 360), (1, 26, 46), 299),
        ],
    )
    def test_grid_and_row_count(self, size, grid_thw, rows):
        pixel_values, got = preprocess_image(textured(*size), CFG)
        assert got == grid_thw
        assert num_placeholder_rows(got, CFG) == rows
        assert tuple(pixel_values.shape) == (num_patches(got, CFG), patch_dim(CFG))
        assert pixel_values.dtype == torch.float32

    def test_rows_are_patches_over_four(self):
        _, grid_thw = preprocess_image(textured(333, 211), CFG)
        assert num_placeholder_rows(grid_thw, CFG) * 4 == num_patches(grid_thw, CFG)

    def test_rgba_image_takes_the_same_grid_as_its_composite(self):
        rgba = textured(640, 360, mode="RGBA")
        _, rgba_grid = preprocess_image(rgba, CFG)
        _, rgb_grid = preprocess_image(to_rgb(rgba), CFG)
        assert rgba_grid == rgb_grid == (1, 26, 46)

    def test_values_stay_in_range(self):
        pixel_values, _ = preprocess_image(textured(60, 90), CFG)
        assert float(pixel_values.min()) >= -1.0
        assert float(pixel_values.max()) <= 1.0

    def test_default_config_matches_an_explicit_one(self):
        explicit, grid_a = preprocess_image(textured(60, 90), CFG)
        implicit, grid_b = preprocess_image(textured(60, 90))
        assert grid_a == grid_b
        torch.testing.assert_close(explicit, implicit, rtol=0, atol=0)


class TestVideoFrameSampling:
    """2.0 fps, <= 96 frames, floored to a multiple of 2, linspace."""

    @pytest.mark.parametrize(
        "num_frames,fps,expected",
        [
            (1870, 24.0, 96),  # 2 fps target 155 -> capped at 96
            (240, 24.0, 20),  # target 20
            (241, 24.0, 20),  # 20.08 -> 20
            (245, 24.0, 20),  # 20.41 -> floored to 20
            (252, 24.0, 20),  # 21.0  -> floored to 20
            (10, 30.0, 2),  # target 0 -> clamped up to the alignment
            (1, 30.0, 1),  # fewer frames than the alignment: keep what we have
            (60, 1.0, 60),  # 2 fps of a 1 fps source -> every frame
            (100000, 30.0, 96),  # long video: hard 96 cap
        ],
    )
    def test_target_frame_count(self, num_frames, fps, expected):
        assert len(sample_video_frame_indices(num_frames, fps, CFG)) == expected

    def test_count_is_even_whenever_it_can_be(self):
        for num_frames in range(2, 60):
            keep = sample_video_frame_indices(num_frames, 24.0, CFG)
            assert len(keep) % 2 == 0, num_frames

    def test_indices_are_linspace_not_a_stride(self):
        keep = sample_video_frame_indices(1870, 24.0, CFG)
        assert keep == [int(i) for i in np.linspace(0, 1869, 96, dtype=int)]
        assert keep[0] == 0
        # linspace keeps the last frame; a stride of 19 would stop at 1843.
        assert keep[-1] == 1869

    def test_presampled_decoder_does_not_sample_twice(self):
        """96 already-decoded frames of an 1870-frame source stay as they are."""
        assert sample_video_frame_indices(96, 24.0, CFG, total_num_frames=1870) == list(range(96))

    def test_empty_video_is_rejected(self):
        with pytest.raises(VisionPreprocessError):
            sample_video_frame_indices(0, 24.0, CFG)


class TestPreprocessVideo:
    @staticmethod
    def frames(count, width, height):
        return [np.asarray(textured(width, height))] * count

    def test_per_frame_budget_and_clip_grid(self):
        pixel_values, grid_thw, timestamps = preprocess_video(self.frames(8, 1280, 720), 24.0, CFG)
        # 8 frames at 24 fps -> 2 fps target 0 -> clamped to 2 frames, 1 clip.
        # 1280x720 under a 144-token budget -> 9x16 cells -> 252x448 canvas.
        assert grid_thw == (1, 18, 32)
        assert num_placeholder_rows(grid_thw, CFG) == 144
        assert tuple(pixel_values.shape) == (num_patches(grid_thw, CFG), patch_dim(CFG))
        assert timestamps == [0.0]

    def test_one_timestamp_per_clip_from_the_first_frame(self):
        _, grid_thw, timestamps = preprocess_video(self.frames(12, 64, 64), 1.0, CFG)
        assert grid_thw[0] == len(timestamps) == 6
        assert timestamps == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]

    def test_video_token_cap_is_not_enforced_per_frame(self):
        """max_num_tokens_per_frame wins outright over max_num_video_tokens."""
        _, grid_thw, _ = preprocess_video(self.frames(4, 1280, 720), 1.0, CFG)
        assert num_placeholder_rows(grid_thw, CFG) == 2 * 144

    def test_single_frame_video_is_padded_to_one_clip(self):
        _, grid_thw, timestamps = preprocess_video(self.frames(1, 64, 64), 1.0, CFG)
        assert grid_thw[0] == 1
        assert timestamps == [0.0]

    def test_alpha_frames_are_rejected(self):
        rgba = [np.asarray(textured(32, 32, mode="RGBA"))] * 2
        with pytest.raises(VisionPreprocessError):
            preprocess_video(rgba, 1.0, CFG)

    def test_pil_frames_are_accepted(self):
        # 64x64 ties (2,2) against (3,3); the reference takes (3,3) -> 84x84.
        _, grid_thw, _ = preprocess_video([textured(64, 64)] * 2, 1.0, CFG)
        assert grid_thw == (1, 6, 6)


class TestProcessorConfigDeclaration:
    """The checkpoint's own `processor_config.json` decides the budgets.

    Exports that dropped `config.json -> vision_config.navit_config` declare
    the preprocessing knobs in a HuggingFace-style `processor_config.json`
    under different names. Before this was read, every one of those values
    silently fell back to this module's defaults -- including the per-frame
    video token budget, where the default and the declaration differ and the
    difference is a 1.78x change in how many decoder rows a video costs.
    """

    DECLARED_PER_FRAME = 256

    @staticmethod
    def _checkpoint(tmp_path, video=None, image=None):
        body = {"processor_class": "MuseSparkProcessor"}
        if video is not None:
            body["video_processor"] = video
        if image is not None:
            body["image_processor"] = image
        (tmp_path / "processor_config.json").write_text(json.dumps(body))
        return tmp_path

    def test_a_checkpoint_without_the_file_changes_nothing(self, tmp_path):
        """The identity case, and the reason captured reference runs stay valid.

        Older exports ship no `processor_config.json`. They must resolve to
        byte-identical preprocessing, so the same object comes back -- not a
        copy that merely happens to compare equal.
        """
        config = MuseSparkOpenPreprocessConfig()
        assert processor_config_overrides(tmp_path) == {}
        assert with_processor_config(config, tmp_path) is config
        assert with_processor_config(config, None) is config
        assert with_processor_config(config, "/nonexistent/checkpoint") is config

    def test_declared_per_frame_budget_is_read_under_its_own_name(self, tmp_path):
        root = self._checkpoint(
            tmp_path,
            video={"max_video_frame_tokens": self.DECLARED_PER_FRAME, "fps": 2.0},
            image={"max_image_tokens": 4096},
        )
        assert processor_config_overrides(root) == {
            "max_num_tokens_per_frame": self.DECLARED_PER_FRAME,
            "sampling_fps": 2.0,
            "max_num_image_tokens": 4096,
        }

    def test_the_declaration_reaches_the_resolved_config(self, tmp_path):
        root = self._checkpoint(tmp_path, video={"max_video_frame_tokens": self.DECLARED_PER_FRAME})
        resolved = MuseSparkOpenPreprocessConfig.from_config(
            with_processor_config(MuseSparkOpenPreprocessConfig(), root)
        )
        assert resolved.max_num_tokens_per_frame == self.DECLARED_PER_FRAME
        assert CFG.max_num_tokens_per_frame == 144, "the default must not move"

    def test_the_declared_budget_changes_the_canvas_it_is_fitted_to(self, tmp_path):
        """The measurable consequence, not just the plumbing.

        Four 448x448 frames at 1 fps make two clips. 448x448 is exactly 16x16
        cells of 28 px, so a 256-cell budget passes the frame through at native
        resolution while the 144 default shrinks it to 12x12.
        """
        frames = [textured(448, 448)] * 4
        _, default_grid, _ = preprocess_video(frames, 1.0, CFG)
        assert default_grid == (2, 24, 24)
        assert num_placeholder_rows(default_grid, CFG) == 2 * 144

        root = self._checkpoint(tmp_path, video={"max_video_frame_tokens": self.DECLARED_PER_FRAME})
        declared = with_processor_config(MuseSparkOpenPreprocessConfig(), root)
        _, grid, _ = preprocess_video(frames, 1.0, declared)
        assert grid == (2, 32, 32)
        assert num_placeholder_rows(grid, declared) == 2 * self.DECLARED_PER_FRAME

    def test_undeclared_keys_keep_their_defaults(self, tmp_path):
        """Reading the file is not adopting all of it.

        `num_frames` and `frame_timestamp_format` are declared by the same
        block and deliberately not mapped; this pins that so a later widening
        of the alias map is a deliberate edit rather than a side effect.
        """
        root = self._checkpoint(
            tmp_path,
            video={
                "max_video_frame_tokens": self.DECLARED_PER_FRAME,
                "num_frames": 600,
                "frame_timestamp_format": "Time: {timestamp:.2f}s",
            },
        )
        resolved = MuseSparkOpenPreprocessConfig.from_config(
            with_processor_config(MuseSparkOpenPreprocessConfig(), root)
        )
        assert resolved.num_frames_per_video == CFG.num_frames_per_video == 96
        assert resolved.video_frame_timestamp_format == CFG.video_frame_timestamp_format

    def test_every_skipped_key_is_named_with_a_reason(self):
        """Silence is not a decision.

        A key the checkpoint declares and this module ignores has to say why,
        and no key may be in both lists -- otherwise "we ignore it" and "we
        read it" can drift apart without anything noticing.
        """
        from tensorrt_llm._torch.models.musespark_open_vision_processing import (
            _PROCESSOR_ALIASES,
            _PROCESSOR_KEYS_NOT_ADOPTED,
        )

        adopted = {name for block in _PROCESSOR_ALIASES.values() for name in block}
        assert not adopted & set(_PROCESSOR_KEYS_NOT_ADOPTED)
        assert set(_PROCESSOR_KEYS_NOT_ADOPTED) == {"num_frames", "frame_timestamp_format"}
        assert all(_PROCESSOR_KEYS_NOT_ADOPTED.values())

    def test_a_malformed_file_is_not_fatal(self, tmp_path):
        """A checkpoint must not become unloadable because this file is bad."""
        (tmp_path / "processor_config.json").write_text("{ not json")
        assert processor_config_overrides(tmp_path) == {}
        (tmp_path / "processor_config.json").write_text('["not a mapping"]')
        assert processor_config_overrides(tmp_path) == {}

    def test_the_original_config_object_is_not_mutated(self, tmp_path):
        """The tower and the weight mapper share the config this came from."""
        root = self._checkpoint(tmp_path, video={"max_video_frame_tokens": self.DECLARED_PER_FRAME})
        config = MuseSparkOpenPreprocessConfig()
        enriched = with_processor_config(config, root)
        assert enriched is not config
        assert config.max_num_tokens_per_frame == 144


class TestPlaceholderIds:
    def test_image_expansion_shape(self):
        ids = image_placeholder_ids((1, 46, 34), CFG)
        assert len(ids) == 391 + 2
        assert ids[0] == VISION_TOKEN_IDS["image_start"] == 200080
        assert ids[-1] == VISION_TOKEN_IDS["image_end"] == 200081
        assert set(ids[1:-1]) == {VISION_TOKEN_IDS["patch"]}
        assert VISION_TOKEN_IDS["patch"] == 200092

    def test_image_expansion_has_no_separators_and_no_image_token(self):
        ids = image_placeholder_ids((1, 46, 34), CFG)
        assert VISION_TOKEN_IDS["image"] == 200090
        assert VISION_TOKEN_IDS["image"] not in ids  # never emitted
        assert VISION_TOKEN_IDS["vid_frame_separator"] not in ids
        # A row separator would break the run into 23 pieces.
        changes = [k for k in range(1, len(ids)) if ids[k] != ids[k - 1]]
        assert changes == [1, len(ids) - 1]

    def test_rows_match_the_expansion_length(self):
        for grid_thw in [(1, 46, 34), (1, 10, 18), (1, 32, 6), (3, 18, 32)]:
            rows = num_placeholder_rows(grid_thw, CFG)
            assert len(image_placeholder_ids(grid_thw, CFG)) == rows + 2

    def test_odd_grid_is_rejected(self):
        with pytest.raises(VisionPreprocessError):
            num_placeholder_rows((1, 45, 34), CFG)

    def test_video_expansion_layout(self):
        def encode(text):
            assert text.startswith("Time: ") and text.endswith("s")
            return [7, 8, 9]

        grid_thw = (3, 18, 32)
        ids = video_placeholder_ids(grid_thw, [0.0, 1.6, 3.2], CFG, encode)
        clip = [7, 8, 9] + [VISION_TOKEN_IDS["video"]] * 144
        separator = [VISION_TOKEN_IDS["vid_frame_separator"]]
        assert ids == (
            [VISION_TOKEN_IDS["vid_start"]]
            + clip
            + separator
            + clip
            + separator
            + clip
            + [VISION_TOKEN_IDS["vid_end"]]
        )
        assert ids.count(VISION_TOKEN_IDS["video"]) == num_placeholder_rows(grid_thw, CFG)

    def test_timestamp_format_is_one_decimal_no_newline(self):
        seen = []
        video_placeholder_ids(
            (2, 2, 2), [0.0, 1.6333333], CFG, lambda text: (seen.append(text) or [1])
        )
        assert seen == ["Time: 0.0s", "Time: 1.6s"]

    def test_timestamp_count_must_match_clips(self):
        with pytest.raises(VisionPreprocessError):
            video_placeholder_ids((3, 18, 32), [0.0], CFG, lambda text: [1])

    def test_missing_tokenizer_is_rejected(self):
        with pytest.raises(VisionPreprocessError):
            video_placeholder_ids((1, 2, 2), [0.0], CFG, None)

    def test_hf_style_tokenizer_is_called_without_special_tokens(self):
        class Tokenizer:
            def __init__(self):
                self.calls = []

            def encode(self, text, add_special_tokens=True):
                self.calls.append((text, add_special_tokens))
                return [11, 12]

        tokenizer = Tokenizer()
        video_placeholder_ids((1, 2, 2), [0.0], CFG, tokenizer)
        assert tokenizer.calls == [("Time: 0.0s", False)]

    def test_embed_mask_covers_only_filler_positions(self):
        ids = image_placeholder_ids((1, 4, 4), CFG)
        assert is_embed_mask(ids) == [False] + [True] * 4 + [False]
        assert sum(is_embed_mask(ids)) == num_placeholder_rows((1, 4, 4), CFG)


class TestExpandPromptPlaceholders:
    IMG_MARKER = (101, 102)
    VID_MARKER = (201, 202)

    def expand(self, prompt, expansions):
        return expand_prompt_placeholders(
            prompt, expansions, image_marker_ids=self.IMG_MARKER, video_marker_ids=self.VID_MARKER
        )

    def test_multi_image_prompt_expands_in_order(self):
        prompt = [1, 101, 102, 2, 101, 102, 3]
        first = image_placeholder_ids((1, 2, 2), CFG)
        second = image_placeholder_ids((1, 4, 4), CFG)
        out = self.expand(prompt, [("image", first), ("image", second)])
        assert isinstance(out, ExpandedPrompt)
        assert out.token_ids == [1] + first + [2] + second + [3]
        assert out.spans == [
            ("image", 1, 1 + len(first)),
            ("image", 2 + len(first), 2 + len(first) + len(second)),
        ]

    def test_embed_mask_lines_up_with_the_ids(self):
        ids = image_placeholder_ids((1, 4, 4), CFG)
        out = self.expand([1, 101, 102, 2], [("image", ids)])
        assert len(out.is_embed) == len(out.token_ids)
        assert sum(out.is_embed) == num_placeholder_rows((1, 4, 4), CFG)
        assert not out.is_embed[0] and not out.is_embed[-1]

    def test_too_few_media_is_an_error(self):
        with pytest.raises(VisionPreprocessError):
            self.expand([101, 102, 101, 102], [("image", image_placeholder_ids((1, 2, 2), CFG))])

    def test_too_many_media_is_an_error(self):
        with pytest.raises(VisionPreprocessError):
            self.expand(
                [101, 102],
                [
                    ("image", image_placeholder_ids((1, 2, 2), CFG)),
                    ("image", image_placeholder_ids((1, 2, 2), CFG)),
                ],
            )


# --------------------------------------------------------------------------
# Reference-refset replay
# --------------------------------------------------------------------------

ORACLE_ENV = "MUSESPARK_OPEN_VISION_ORACLE"
TOKENIZER_ENV = "MUSESPARK_OPEN_TOKENIZER"

_IMAGE_MARKER = (-1,)
_VIDEO_MARKER = (-2,)


def _oracle_dir():
    root = os.environ.get(ORACLE_ENV)
    if not root:
        return None
    path = Path(root)
    return path if (path / "vision-manifest.json").is_file() else None


def _split_expansions(token_ids):
    """Cut every vision expansion out of a reference prompt.

    Returns `(skeleton, segments)`: the skeleton carries a one-token marker
    where each expansion was, and the segments are the expansions themselves in
    prompt order. Replaying then reconstructs the *whole* prompt, so the
    assertion covers the surrounding text as well as the expansion.
    """
    starts = {
        VISION_TOKEN_IDS["image_start"]: ("image", VISION_TOKEN_IDS["image_end"], _IMAGE_MARKER),
        VISION_TOKEN_IDS["vid_start"]: ("video", VISION_TOKEN_IDS["vid_end"], _VIDEO_MARKER),
    }
    skeleton, segments = [], []
    index = 0
    while index < len(token_ids):
        token = token_ids[index]
        if token in starts:
            _, end_id, marker = starts[token]
            stop = index
            while token_ids[stop] != end_id:
                stop += 1
            segments.append((starts[token][0], list(token_ids[index : stop + 1])))
            skeleton.extend(marker)
            index = stop + 1
        else:
            skeleton.append(token)
            index += 1
    return skeleton, segments


def _mp4_video_track(path):
    """Frame count, fps and frame size straight out of the MP4 container.

    The test node has no video decoder, and the placeholder expansion needs
    only this metadata: the frame sampler is driven by frame count and rate,
    and the canvas by the frame size.
    """
    buffer = Path(path).read_bytes()

    def walk(start, end):
        pos = start
        while pos + 8 <= end:
            size = struct.unpack_from(">I", buffer, pos)[0]
            kind = buffer[pos + 4 : pos + 8]
            header = 8
            if size == 1:
                size = struct.unpack_from(">Q", buffer, pos + 8)[0]
                header = 16
            elif size == 0:
                size = end - pos
            yield kind, pos + header, pos + size
            pos += size

    def find(start, end, parts):
        for kind, box_start, box_end in walk(start, end):
            if kind == parts[0]:
                if len(parts) == 1:
                    return box_start, box_end
                found = find(box_start, box_end, parts[1:])
                if found:
                    return found
        return None

    moov = find(0, len(buffer), [b"moov"])
    for kind, track_start, track_end in walk(*moov):
        if kind != b"trak":
            continue
        handler = find(track_start, track_end, [b"mdia", b"hdlr"])
        if buffer[handler[0] + 8 : handler[0] + 12] != b"vide":
            continue
        mdhd = find(track_start, track_end, [b"mdia", b"mdhd"])
        wide = buffer[mdhd[0]] == 1
        offset = mdhd[0] + (20 if wide else 12)
        timescale = struct.unpack_from(">I", buffer, offset)[0]
        duration = struct.unpack_from(">Q" if wide else ">I", buffer, offset + 4)[0]
        stts = find(track_start, track_end, [b"mdia", b"minf", b"stbl", b"stts"])
        entries = struct.unpack_from(">I", buffer, stts[0] + 4)[0]
        frames = sum(
            struct.unpack_from(">II", buffer, stts[0] + 8 + 8 * i)[0] for i in range(entries)
        )
        stsd = find(track_start, track_end, [b"mdia", b"minf", b"stbl", b"stsd"])
        width, height = struct.unpack_from(">HH", buffer, stsd[0] + 40)
        return frames, frames * timescale / duration, (width, height)
    raise AssertionError(f"no video track in {path}")


@pytest.mark.skipif(
    _oracle_dir() is None, reason=f"set {ORACLE_ENV} to the captured refset directory"
)
class TestOracleExpansion:
    """Replay the captured prompt set: expansion must match token for token."""

    IMAGE_TAGS = [
        "mc_single_1",
        "mc_single_2",
        "mc_single_3",
        "mc_multi_2",
        "mc_multi_3",
        "mc_multi_4",
        "syn_rgba",
        "syn_tall",
    ]

    @pytest.fixture(scope="class")
    def oracle(self):
        root = _oracle_dir()
        with open(root / "vision-manifest.json") as handle:
            manifest = json.load(handle)
        return root, manifest

    @pytest.fixture(scope="class")
    def tokenizer(self, oracle):
        path = os.environ.get(TOKENIZER_ENV) or oracle[1].get("checkpoint_dir")
        if not path or not Path(path, "tokenizer.json").is_file():
            return None
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    @staticmethod
    def _record(manifest, tag):
        for record in manifest["records"]:
            if record["tag"] == tag:
                return record
        raise AssertionError(f"no record {tag} in the manifest")

    @pytest.mark.parametrize("tag", IMAGE_TAGS)
    def test_image_prompt_expansion(self, oracle, tag):
        root, manifest = oracle
        record = self._record(manifest, tag)
        expected = record["prompt_token_ids"]
        skeleton, segments = _split_expansions(expected)
        assert [modality for modality, _ in segments] == ["image"] * len(record["images"])

        expansions = []
        for relative in record["images"]:
            with Image.open(root / relative) as image:
                image.load()
                _, grid_thw = preprocess_image(image)
            expansions.append(("image", image_placeholder_ids(grid_thw)))

        got = expand_prompt_placeholders(
            skeleton, expansions, image_marker_ids=_IMAGE_MARKER, video_marker_ids=_VIDEO_MARKER
        )
        assert got.token_ids == expected
        assert sum(got.is_embed) == sum(len(ids) - 2 for _, ids in expansions)

    def test_video_prompt_expansion(self, oracle, tokenizer):
        root, manifest = oracle
        if tokenizer is None:
            pytest.skip(f"set {TOKENIZER_ENV} to the checkpoint tokenizer dir")
        record = self._record(manifest, "vmme_short_1")
        expected = record["prompt_token_ids"]
        skeleton, segments = _split_expansions(expected)
        assert [modality for modality, _ in segments] == ["video"]

        # The reference decodes the clip itself; replay the sampling against the
        # source frame count and rate instead of pulling in a decoder. Only the
        # frame geometry reaches the expansion, never the pixels.
        count, fps, size = _mp4_video_track(root / record["video"])
        assert (count, fps, size) == (1870, 24.0, (1280, 720))
        _, grid_thw, timestamps = preprocess_video([Image.new("RGB", size)] * count, fps)
        assert grid_thw == (48, 18, 32)
        assert timestamps[:3] == [0.0, 39 / 24, 78 / 24]

        ids = video_placeholder_ids(grid_thw, timestamps, None, tokenizer)
        got = expand_prompt_placeholders(
            skeleton,
            [("video", ids)],
            image_marker_ids=_IMAGE_MARKER,
            video_marker_ids=_VIDEO_MARKER,
        )
        assert got.token_ids == expected
        assert num_placeholder_rows(grid_thw) == record["tensor_shapes"]["tower.out"][0][0]

    def test_placeholder_rows_match_the_reference_encoder_rows(self, oracle):
        """num_placeholder_rows must equal the tower's output first dim.

        The reference caches encoder output per unique image, so an image that a
        previous prompt already ran through the tower contributes no rows to the
        later capture; dedupe by content digest before comparing.
        """
        root, manifest = oracle
        with open(root / "media-manifest.json") as handle:
            digests = {entry["file"]: entry["sha256"] for entry in json.load(handle)["media"]}
        seen = set()
        checked = 0
        for record in manifest["records"]:
            if not record["images"]:
                continue
            expected_rows = 0
            for relative in record["images"]:
                with Image.open(root / relative) as image:
                    image.load()
                    _, grid_thw = preprocess_image(image)
                if digests[relative] not in seen:
                    expected_rows += num_placeholder_rows(grid_thw)
                    seen.add(digests[relative])
            captured = record["tensor_shapes"]["tower.out"][0][0]
            assert captured == expected_rows, record["tag"]
            checked += 1
        assert checked == len(self.IMAGE_TAGS)
