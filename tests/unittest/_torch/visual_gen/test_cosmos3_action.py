# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Cosmos3 action sizing helpers (no checkpoint / GPU required).

Run:
    pytest tests/unittest/_torch/visual_gen/test_cosmos3_action.py -v
"""

import json

import numpy as np
import PIL.Image
import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.cosmos3.action import (
    ACTION_VIEWPOINT_TEMPLATES,
    DEFAULT_ACTION_VIEW_POINT,
    EMBODIMENT_TO_DOMAIN_ID,
    EMBODIMENT_TO_RAW_ACTION_DIM,
    VIDEO_RES_SIZE_INFO,
    action_aspect_ratio_label,
    action_reference_image,
    build_action_json_prompt,
    find_closest_target_size,
    normalize_action_resolution,
    normalize_action_video_input,
    prepare_action_latents,
    resolve_action_size,
    resolve_raw_action_dim,
)
from tensorrt_llm._torch.visual_gen.models.cosmos3.defaults import (
    COSMOS3_DOMAIN_PRESET_ALIASES,
    COSMOS3_DOMAIN_PRESETS,
    COSMOS3_EXTRA_SPECS,
    get_domain_preset,
    resolve_domain_action_config,
)
from tensorrt_llm._torch.visual_gen.models.cosmos3.transformer_cosmos3 import (
    compute_mrope_position_ids_action,
    compute_mrope_position_ids_vision,
)

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
        assert spec.type == "Literal[256, 480, 704, 720]"
        assert spec.default is None


class TestDomainActionPresets:
    def test_bridge_preset_fills_missing_fields(self):
        cfg = resolve_domain_action_config(domain_name="bridge_orig_lerobot")
        assert cfg["raw_action_dim"] == 10
        assert cfg["action_chunk_size"] == 16
        assert cfg["num_frames"] == 17
        assert cfg["action_resolution"] == 480
        assert cfg["frame_rate"] == 5.0
        assert cfg["warnings"] == []

    def test_av_preset_uses_longer_chunk(self):
        cfg = resolve_domain_action_config(domain_name="av")
        assert cfg["action_chunk_size"] == 60
        assert cfg["num_frames"] == 61
        assert cfg["raw_action_dim"] == 9

    def test_mismatch_emits_warning(self):
        cfg = resolve_domain_action_config(
            domain_name="bridge_orig_lerobot",
            raw_action_dim=9,
        )
        assert cfg["raw_action_dim"] == 9
        assert len(cfg["warnings"]) == 1
        assert "raw_action_dim=9" in cfg["warnings"][0]

    def test_action_fps_defaults_to_frame_rate(self):
        cfg = resolve_domain_action_config(domain_name="av")
        assert cfg["frame_rate"] == 10.0
        assert cfg["action_fps"] == 10.0

    def test_explicit_action_fps_overrides_default(self):
        cfg = resolve_domain_action_config(domain_name="av", action_fps=5.0, frame_rate=24.0)
        assert cfg["frame_rate"] == 24.0
        assert cfg["action_fps"] == 5.0

    def test_alias_maps_to_canonical_preset(self):
        preset = get_domain_preset("robomind-franka")
        assert preset is not None
        assert preset == get_domain_preset("droid_lerobot")

    def test_presets_carry_sampling_settings_only(self):
        """Width is per-embodiment; aliases share presets, so it must not live there."""
        for preset in COSMOS3_DOMAIN_PRESETS.values():
            assert "raw_action_dim" not in preset

    @pytest.mark.parametrize(
        "domain_name,expected",
        [
            ("bridge_orig_lerobot", 10),
            ("droid_lerobot", 10),
            ("robomind-franka", 10),
            ("robomind-ur", 10),
            ("robomind-franka-dual", 20),  # dual arm, not the droid preset's 10
            ("galbot", 30),  # humanoid stack, not agibotworld's 29
            ("agibotworld", 29),
            ("agibot_gear_gripper", 29),
            ("agibot_gear_gripper_ext", 29),
            ("av", 9),
            ("camera_pose", 9),
            ("hand_pose", 57),
            ("pusht", 2),
            ("umi", 10),
            ("fractal", 10),
        ],
    )
    def test_canonical_action_width_per_embodiment(self, domain_name, expected):
        assert resolve_raw_action_dim(domain_name=domain_name) == expected
        assert resolve_domain_action_config(domain_name=domain_name)["raw_action_dim"] == expected

    def test_aliased_domains_keep_their_own_width(self):
        """Sharing a sampling preset must not import that preset's action width."""
        for alias, canonical in COSMOS3_DOMAIN_PRESET_ALIASES.items():
            alias_dim = resolve_raw_action_dim(domain_name=alias)
            if alias_dim is None:
                continue
            assert alias_dim == EMBODIMENT_TO_RAW_ACTION_DIM[alias], (
                f"{alias} must keep its own width, not {canonical}'s"
            )

    def test_libero_has_no_canonical_width(self):
        """LIBERO's width depends on the dataset's rotation space (7/10/13)."""
        assert "libero" not in EMBODIMENT_TO_RAW_ACTION_DIM
        cfg = resolve_domain_action_config(domain_name="libero")
        assert cfg["raw_action_dim"] is None
        assert cfg["action_resolution"] == 256  # sampling preset still applies
        assert any("canonical action width" in w for w in cfg["warnings"])

    def test_explicit_raw_action_dim_overrides_with_warning(self):
        cfg = resolve_domain_action_config(domain_name="libero", raw_action_dim=7)
        assert cfg["raw_action_dim"] == 7
        assert cfg["warnings"] == []

    def test_domain_id_resolves_width_when_unambiguous(self):
        assert resolve_raw_action_dim(domain_id=12) == 20  # robomind-franka-dual
        assert resolve_raw_action_dim(domain_id=8) == 10  # droid / robomind-franka agree
        assert resolve_raw_action_dim(domain_id=15) == 29  # all agibot variants agree
        assert resolve_raw_action_dim(domain_id=5) is None  # libero
        assert resolve_raw_action_dim(domain_id=0) is None  # no_action

    def test_every_width_entry_has_a_domain_id(self):
        assert set(EMBODIMENT_TO_RAW_ACTION_DIM) <= set(EMBODIMENT_TO_DOMAIN_ID)

    def test_unknown_domain_warns_and_uses_generic_defaults(self):
        cfg = resolve_domain_action_config(domain_name="typo_domain")
        assert cfg["action_chunk_size"] == 16
        assert cfg["action_resolution"] == 480
        assert cfg["warnings"]
        assert "preset was not found" in cfg["warnings"][0]

    def test_non_positive_action_timing_raises(self):
        with pytest.raises(ValueError, match="action_fps must be positive"):
            resolve_domain_action_config(domain_name="av", action_fps=0.0)

    def test_unknown_resolution_raises(self):
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

    def test_policy_accepts_path_image(self, tmp_path):
        image_path = tmp_path / "frame.png"
        PIL.Image.new("RGB", (3, 3), "green").save(image_path)
        ref = action_reference_image(
            action_mode="policy",
            image=image_path,
            video=None,
        )
        assert ref.getpixel((0, 0)) == (0, 128, 0)


class TestNormalizeActionVideoInput:
    def test_none_returns_empty_list(self):
        assert normalize_action_video_input(None) == []

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="at least one frame"):
            normalize_action_video_input([])

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


class TestActionJsonPrompt:
    """The trained action caption: structured JSON, not the flat video templates."""

    BRIDGE = dict(num_frames=17, frame_rate=5.0, height=480, width=832)

    def test_matches_trained_shape(self):
        payload = json.loads(
            build_action_json_prompt(
                "Pick up the pear and place it in the bag",
                view_point="ego_view",
                **self.BRIDGE,
            )
        )
        assert payload == {
            "cinematography": {
                "framing": (
                    "This video is captured from a first-person perspective looking at the scene."
                )
            },
            "actions": [
                {
                    "time": "0:00-0:03",
                    "description": "Pick up the pear and place it in the bag.",
                }
            ],
            "duration": "3s",
            "fps": 5.0,
            "resolution": {"H": 480, "W": 832},
            "aspect_ratio": "16,9",
        }

    def test_key_order_is_preserved(self):
        """Field order is part of the trained caption format."""
        text = build_action_json_prompt("Do a thing", view_point="ego_view", **self.BRIDGE)
        assert list(json.loads(text).keys()) == [
            "cinematography",
            "actions",
            "duration",
            "fps",
            "resolution",
            "aspect_ratio",
        ]

    @pytest.mark.parametrize("view_point", sorted(ACTION_VIEWPOINT_TEMPLATES))
    def test_every_viewpoint_emits_its_trained_sentence(self, view_point):
        payload = json.loads(
            build_action_json_prompt("Do a thing", view_point=view_point, **self.BRIDGE)
        )
        assert payload["cinematography"]["framing"] == ACTION_VIEWPOINT_TEMPLATES[view_point]

    def test_default_view_point_is_known(self):
        assert DEFAULT_ACTION_VIEW_POINT in ACTION_VIEWPOINT_TEMPLATES

    @pytest.mark.parametrize("view_point", [None, "sideways_view"])
    def test_unknown_or_missing_view_point_drops_framing(self, view_point):
        payload = json.loads(
            build_action_json_prompt("Do a thing", view_point=view_point, **self.BRIDGE)
        )
        assert "cinematography" not in payload
        assert list(payload.keys())[0] == "actions"

    @pytest.mark.parametrize(
        "description,expected",
        [
            ("Pick up the pear", "Pick up the pear."),
            ("Pick up the pear.", "Pick up the pear."),
            ("Is it a pear?", "Is it a pear?"),
            ("Grab it!", "Grab it!"),
            ("  padded  ", "padded."),
            ("", ""),
        ],
    )
    def test_description_is_terminated_once(self, description, expected):
        payload = json.loads(build_action_json_prompt(description, view_point=None, **self.BRIDGE))
        assert payload["actions"][0]["description"] == expected

    @pytest.mark.parametrize(
        "num_frames,frame_rate,duration,time_range",
        [
            (17, 5.0, "3s", "0:00-0:03"),  # bridge: 3.4s truncates, rounds to 3
            (17, 24.0, "0s", "0:00-0:01"),  # 0.708s truncates to 0, rounds to 1
            (61, 10.0, "6s", "0:00-0:06"),  # av preset
            (241, 2.0, "120s", "0:00-2:00"),  # crosses the minute boundary
        ],
    )
    def test_duration_truncates_while_time_range_rounds(
        self, num_frames, frame_rate, duration, time_range
    ):
        payload = json.loads(
            build_action_json_prompt(
                "Do a thing",
                view_point=None,
                num_frames=num_frames,
                frame_rate=frame_rate,
                height=480,
                width=832,
            )
        )
        assert payload["duration"] == duration
        assert payload["actions"][0]["time"] == time_range
        assert payload["fps"] == float(frame_rate)

    @pytest.mark.parametrize("action_resolution", sorted(VIDEO_RES_SIZE_INFO))
    def test_aspect_label_matches_the_bucket_it_came_from(self, action_resolution):
        """Every canvas is a bucket entry, so its label must round-trip."""
        for label, (width, height) in VIDEO_RES_SIZE_INFO[action_resolution].items():
            assert action_aspect_ratio_label(height, width) == label

    def test_aspect_label_is_not_a_reduced_fraction(self):
        """832x480 reduces to 26,15 but the trained label is 16,9."""
        assert action_aspect_ratio_label(480, 832) == "16,9"

    def test_resolution_is_reported_as_the_padded_canvas(self):
        payload = json.loads(build_action_json_prompt("Do a thing", view_point=None, **self.BRIDGE))
        assert payload["resolution"] == {"H": 480, "W": 832}

    def test_zero_frame_rate_does_not_raise(self):
        payload = json.loads(
            build_action_json_prompt(
                "Do a thing",
                view_point=None,
                num_frames=17,
                frame_rate=0.0,
                height=480,
                width=832,
            )
        )
        assert payload["duration"] == "0s"
        assert payload["actions"][0]["time"] == "0:00-0:00"

    def test_view_point_spec_defaults_to_ego_view(self):
        spec = COSMOS3_EXTRA_SPECS["view_point"]
        assert spec.default == DEFAULT_ACTION_VIEW_POINT
        for view_point in ACTION_VIEWPOINT_TEMPLATES:
            assert repr(view_point) in spec.type


def _reference_scaled_positions(
    *,
    grid_t: int,
    temporal_offset: float,
    fps: float,
    base_fps: float,
    temporal_compression_factor: int,
    base_temporal_compression_factor: int,
    start_frame_offset: int,
) -> list[float]:
    """Transcription of cosmos-framework ``get_3d_mrope_ids_vae_tokens``.

    Reference: ``cosmos_framework/data/generator/sequence_packing/mrope.py``
    (mirrored by diffusers ``pipeline_cosmos3_omni.get_3d_mrope_ids_vae_tokens``).
    """
    tps = fps / temporal_compression_factor
    base_tps = base_fps / base_temporal_compression_factor
    return [(i + start_frame_offset) / tps * base_tps + temporal_offset for i in range(grid_t)]


class TestActionMropePositionIds:
    """Action tokens run at frame rate but must share the vision latent timeline."""

    VISION_TCF = 4

    @pytest.mark.parametrize(
        "grid_t,temporal_offset,action_fps,base_fps,start_frame_offset",
        [
            (4, 0.0, 24.0, 24.0, 1),
            (4, 15032.0, 24.0, 24.0, 1),
            (16, 0.0, 5.0, 24.0, 1),
            (60, 0.0, 10.0, 24.0, 1),
            (4, 0.0, 24.0, 24.0, 0),
        ],
    )
    def test_matches_reference_formula(
        self, grid_t, temporal_offset, action_fps, base_fps, start_frame_offset
    ):
        ids, _ = compute_mrope_position_ids_action(
            grid_t,
            temporal_offset=temporal_offset,
            action_fps=action_fps,
            base_fps=base_fps,
            base_temporal_compression_factor=self.VISION_TCF,
            enable_fps_modulation=True,
            start_frame_offset=start_frame_offset,
        )
        expected = _reference_scaled_positions(
            grid_t=grid_t,
            temporal_offset=temporal_offset,
            fps=action_fps,
            base_fps=base_fps,
            temporal_compression_factor=1,
            base_temporal_compression_factor=self.VISION_TCF,
            start_frame_offset=start_frame_offset,
        )
        torch.testing.assert_close(
            ids[0], torch.tensor(expected, dtype=ids.dtype), rtol=0, atol=1e-5
        )

    def test_action_step_advances_one_source_frame(self):
        """Consecutive action tokens are 1/vision_tcf of a latent frame apart."""
        ids, _ = compute_mrope_position_ids_action(
            8,
            temporal_offset=0.0,
            action_fps=24.0,
            base_fps=24.0,
            base_temporal_compression_factor=self.VISION_TCF,
            enable_fps_modulation=True,
            start_frame_offset=1,
        )
        deltas = ids[0, 1:] - ids[0, :-1]
        torch.testing.assert_close(
            deltas, torch.full_like(deltas, 1.0 / self.VISION_TCF), rtol=0, atol=1e-5
        )

    @pytest.mark.parametrize(
        "action_chunk_size,num_frames,fps",
        [
            (16, 17, 24.0),  # generic COSMOS3_ACTION_PARAMS default
            (16, 17, 5.0),  # bridge_orig_lerobot preset
            (60, 61, 10.0),  # av preset
        ],
    )
    def test_last_action_token_lands_on_last_vision_latent_frame(
        self, action_chunk_size, num_frames, fps
    ):
        """The 4x-scaling regression: action must not outrun the video it conditions.

        Vision and action are packed into one temporal axis, so the paired
        (num_frames, action_chunk_size) config must place the final action token
        exactly on the final vision latent frame.
        """
        latent_t = (num_frames - 1) // self.VISION_TCF + 1
        vision_ids, _ = compute_mrope_position_ids_vision(
            latent_t,
            1,
            1,
            temporal_offset=0.0,
            fps=fps,
            base_fps=24.0,
            temporal_compression_factor=self.VISION_TCF,
            enable_fps_modulation=True,
        )
        action_ids, _ = compute_mrope_position_ids_action(
            action_chunk_size,
            temporal_offset=0.0,
            action_fps=fps,
            base_fps=24.0,
            base_temporal_compression_factor=self.VISION_TCF,
            enable_fps_modulation=True,
            start_frame_offset=1,
        )
        assert action_ids[0, -1].item() == pytest.approx(vision_ids[0, -1].item(), abs=1e-5)
        assert action_ids[0, 0].item() > vision_ids[0, 0].item()

    def test_spatial_rows_are_zero(self):
        ids, _ = compute_mrope_position_ids_action(
            5,
            temporal_offset=0.0,
            action_fps=24.0,
            base_fps=24.0,
            base_temporal_compression_factor=self.VISION_TCF,
            enable_fps_modulation=True,
        )
        assert ids.shape == (3, 5)
        assert torch.all(ids[1] == 0)
        assert torch.all(ids[2] == 0)

    def test_fps_modulation_disabled_gives_integer_frame_indices(self):
        ids, _ = compute_mrope_position_ids_action(
            4,
            temporal_offset=7.0,
            action_fps=24.0,
            base_fps=24.0,
            base_temporal_compression_factor=self.VISION_TCF,
            enable_fps_modulation=False,
            start_frame_offset=1,
        )
        assert ids[0].tolist() == [8, 9, 10, 11]

    def test_lower_action_fps_stretches_positions(self):
        kwargs = dict(
            temporal_offset=0.0,
            base_fps=24.0,
            base_temporal_compression_factor=self.VISION_TCF,
            enable_fps_modulation=True,
            start_frame_offset=1,
        )
        fast, _ = compute_mrope_position_ids_action(4, action_fps=24.0, **kwargs)
        slow, _ = compute_mrope_position_ids_action(4, action_fps=12.0, **kwargs)
        torch.testing.assert_close(slow[0], fast[0] * 2.0, rtol=0, atol=1e-5)


class TestVisionMropeBaseCompressionDefault:
    """``base_temporal_compression_factor=None`` must not disturb vision or audio."""

    def test_vision_positions_unchanged_by_default(self):
        kwargs = dict(
            temporal_offset=3.0,
            fps=30.0,
            base_fps=24.0,
            temporal_compression_factor=4,
            enable_fps_modulation=True,
        )
        implicit, next_implicit = compute_mrope_position_ids_vision(5, 2, 2, **kwargs)
        explicit, next_explicit = compute_mrope_position_ids_vision(
            5, 2, 2, base_temporal_compression_factor=4, **kwargs
        )
        torch.testing.assert_close(implicit, explicit, rtol=0, atol=0)
        assert next_implicit == next_explicit

    def test_audio_style_call_unchanged(self):
        """Audio packs with tcf=1 and no base override (sound base tcf is also 1)."""
        ids, _ = compute_mrope_position_ids_vision(
            3,
            1,
            1,
            temporal_offset=0.0,
            fps=25.0,
            base_fps=24.0,
            temporal_compression_factor=1,
            enable_fps_modulation=True,
        )
        expected = _reference_scaled_positions(
            grid_t=3,
            temporal_offset=0.0,
            fps=25.0,
            base_fps=24.0,
            temporal_compression_factor=1,
            base_temporal_compression_factor=1,
            start_frame_offset=0,
        )
        torch.testing.assert_close(
            ids[0], torch.tensor(expected, dtype=ids.dtype), rtol=0, atol=1e-5
        )


class TestPrepareActionLatents:
    def test_forward_dynamics_raw_dim_mismatch_raises(self):
        with pytest.raises(ValueError, match="raw_action_dim must match"):
            prepare_action_latents(
                mode="forward_dynamics",
                action_chunk_size=2,
                raw_action_dim=3,
                action_dim=8,
                generator=torch.Generator(device="cpu").manual_seed(0),
                device=torch.device("cpu"),
                dtype=torch.float32,
                action_input=[[0.0, 1.0], [2.0, 3.0]],
            )
