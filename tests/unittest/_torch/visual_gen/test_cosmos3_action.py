# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Cosmos3 action sizing helpers (no checkpoint / GPU required).

Run:
    pytest tests/unittest/_torch/visual_gen/test_cosmos3_action.py -v
"""

import numpy as np
import PIL.Image
import pytest

from tensorrt_llm._torch.visual_gen.models.cosmos3.action import (
    VIDEO_RES_SIZE_INFO,
    action_reference_image,
    find_closest_target_size,
    normalize_action_video_input,
    resolve_action_size,
)
from tensorrt_llm._torch.visual_gen.models.cosmos3.defaults import COSMOS3_EXTRA_SPECS

pytestmark = pytest.mark.cosmos3


class TestFindClosestTargetSize:
    @pytest.mark.parametrize(
        "input_h,input_w,action_resolution,expected",
        [
            (480, 832, 480, (832, 480)),
            (832, 480, 480, (480, 832)),
            (512, 512, 480, (640, 640)),
            (704, 1280, 704, (1280, 704)),
            (256, 256, 256, (256, 256)),
            (720, 1280, 720, (1280, 720)),
        ],
    )
    def test_picks_closest_aspect_bucket(self, input_h, input_w, action_resolution, expected):
        assert find_closest_target_size(input_h, input_w, action_resolution) == expected

    def test_accepts_string_and_int_resolution_keys(self):
        ref_h, ref_w = 480, 832
        assert find_closest_target_size(ref_h, ref_w, 480) == find_closest_target_size(
            ref_h, ref_w, "480"
        )

    def test_unknown_resolution_raises(self):
        with pytest.raises(ValueError, match="Unknown Cosmos3 action resolution"):
            find_closest_target_size(480, 832, 1080)

    @pytest.mark.parametrize("action_resolution", sorted(VIDEO_RES_SIZE_INFO))
    def test_all_buckets_have_aspect_entries(self, action_resolution):
        assert VIDEO_RES_SIZE_INFO[action_resolution]


class TestResolveActionSize:
    @staticmethod
    def _ref_image(width: int, height: int) -> PIL.Image.Image:
        return PIL.Image.new("RGB", (width, height))

    def test_explicit_height_and_width_are_unchanged(self):
        ref = self._ref_image(832, 480)
        assert resolve_action_size(400, 600, ref, 480) == (400, 600)

    def test_unset_height_and_width_use_action_resolution_bucket(self):
        ref = self._ref_image(832, 480)
        assert resolve_action_size(None, None, ref, 480) == (480, 832)

    def test_partial_height_fills_width_from_bucket(self):
        ref = self._ref_image(832, 480)
        assert resolve_action_size(400, None, ref, 480) == (400, 832)

    def test_partial_width_fills_height_from_bucket(self):
        ref = self._ref_image(832, 480)
        assert resolve_action_size(None, 600, ref, 480) == (480, 600)


class TestActionResolutionExtraParam:
    def test_extra_param_spec_uses_action_resolution_key(self):
        spec = COSMOS3_EXTRA_SPECS["action_resolution"]
        assert spec.type == "int"
        assert spec.default == 480


class TestDomainActionPresets:
    def test_bridge_preset_fills_missing_fields(self):
        from tensorrt_llm._torch.visual_gen.models.cosmos3.defaults import (
            resolve_domain_action_config,
        )

        cfg = resolve_domain_action_config(domain_name="bridge_orig_lerobot")
        assert cfg["raw_action_dim"] == 10
        assert cfg["action_chunk_size"] == 16
        assert cfg["num_frames"] == 17
        assert cfg["action_resolution"] == 480
        assert cfg["frame_rate"] == 5.0
        assert cfg["warnings"] == []

    def test_av_preset_uses_longer_chunk(self):
        from tensorrt_llm._torch.visual_gen.models.cosmos3.defaults import (
            resolve_domain_action_config,
        )

        cfg = resolve_domain_action_config(domain_name="av")
        assert cfg["action_chunk_size"] == 60
        assert cfg["num_frames"] == 61
        assert cfg["raw_action_dim"] == 9

    def test_mismatch_emits_warning(self):
        from tensorrt_llm._torch.visual_gen.models.cosmos3.defaults import (
            resolve_domain_action_config,
        )

        cfg = resolve_domain_action_config(
            domain_name="bridge_orig_lerobot",
            raw_action_dim=9,
        )
        assert cfg["raw_action_dim"] == 9
        assert len(cfg["warnings"]) == 1
        assert "raw_action_dim=9" in cfg["warnings"][0]

    def test_action_fps_defaults_to_frame_rate(self):
        from tensorrt_llm._torch.visual_gen.models.cosmos3.defaults import (
            resolve_domain_action_config,
        )

        cfg = resolve_domain_action_config(domain_name="av")
        assert cfg["frame_rate"] == 10.0
        assert cfg["action_fps"] == 10.0

    def test_explicit_action_fps_overrides_default(self):
        from tensorrt_llm._torch.visual_gen.models.cosmos3.defaults import (
            resolve_domain_action_config,
        )

        cfg = resolve_domain_action_config(domain_name="av", action_fps=5.0, frame_rate=24.0)
        assert cfg["frame_rate"] == 24.0
        assert cfg["action_fps"] == 5.0

    def test_alias_maps_to_canonical_preset(self):
        from tensorrt_llm._torch.visual_gen.models.cosmos3.defaults import get_domain_preset

        preset = get_domain_preset("robomind-franka")
        assert preset is not None
        assert preset["raw_action_dim"] == 10

    def test_unknown_resolution_raises(self):
        from tensorrt_llm._torch.visual_gen.models.cosmos3.action import normalize_action_resolution

        with pytest.raises(ValueError, match="Unknown Cosmos3 action_resolution"):
            normalize_action_resolution(1080)


class TestActionReferenceImage:
    def test_forward_dynamics_accepts_mp4_on_image_path(self, tmp_path, monkeypatch):
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"fake")
        expected = PIL.Image.new("RGB", (4, 2), "red")

        def _fake_read_video(path, pts_unit="sec"):
            import torch

            tensor = torch.from_numpy(np.array(expected)).unsqueeze(0)
            return tensor, None, {}

        monkeypatch.setattr("torchvision.io.read_video", _fake_read_video)
        ref = action_reference_image(
            action_mode="forward_dynamics",
            image=str(video_path),
            video=None,
        )
        assert ref.size == expected.size
        assert ref.getpixel((0, 0)) == (255, 0, 0)

    def test_policy_prefers_image_path_over_video(self, tmp_path):
        image_path = tmp_path / "frame.png"
        PIL.Image.new("RGB", (3, 3), "blue").save(image_path)
        ref = action_reference_image(
            action_mode="policy",
            image=str(image_path),
            video=str(tmp_path / "unused.mp4"),
        )
        assert ref.getpixel((0, 0)) == (0, 0, 255)


class TestNormalizeActionVideoInput:
    def test_image_path_returns_singleton_list(self, tmp_path):
        image_path = tmp_path / "frame.png"
        PIL.Image.new("RGB", (8, 4), "red").save(image_path)
        assert normalize_action_video_input(str(image_path)) == [str(image_path)]

    def test_frame_directory_returns_sorted_paths(self, tmp_path):
        (tmp_path / "b.png").write_bytes(b"")
        (tmp_path / "a.png").write_bytes(b"")
        (tmp_path / "skip.txt").write_text("x")
        assert normalize_action_video_input(str(tmp_path)) == [
            str(tmp_path / "a.png"),
            str(tmp_path / "b.png"),
        ]

    def test_unsupported_file_extension_raises(self, tmp_path):
        bad_path = tmp_path / "clip.mov"
        bad_path.write_bytes(b"fake")
        with pytest.raises(ValueError, match="must be a frame directory"):
            normalize_action_video_input(str(bad_path))

    def test_decode_mp4_returns_pil_frames(self, tmp_path, monkeypatch):
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"fake")
        expected = [
            PIL.Image.new("RGB", (2, 2), "red"),
            PIL.Image.new("RGB", (2, 2), "blue"),
        ]

        def _fake_read_video(path, pts_unit="sec"):
            assert path == str(video_path)
            assert pts_unit == "sec"
            import torch

            tensor = torch.stack(
                [torch.from_numpy(np.array(image)) for image in expected],
                dim=0,
            )
            return tensor, None, {}

        monkeypatch.setattr(
            "torchvision.io.read_video",
            _fake_read_video,
        )
        frames = normalize_action_video_input(str(video_path))
        assert len(frames) == 2
        assert all(isinstance(frame, PIL.Image.Image) for frame in frames)
        assert frames[0].getpixel((0, 0)) == (255, 0, 0)

    def test_decode_respects_max_frames(self, tmp_path, monkeypatch):
        video_path = tmp_path / "clip.avi"
        video_path.write_bytes(b"fake")
        images = [PIL.Image.new("RGB", (1, 1), color) for color in ("red", "green", "blue")]

        def _fake_read_video(path, pts_unit="sec"):
            import torch

            tensor = torch.stack(
                [torch.from_numpy(np.array(image)) for image in images],
                dim=0,
            )
            return tensor, None, {}

        monkeypatch.setattr("torchvision.io.read_video", _fake_read_video)
        frames = normalize_action_video_input(
            str(video_path),
            max_frames=2,
        )
        assert len(frames) == 2
