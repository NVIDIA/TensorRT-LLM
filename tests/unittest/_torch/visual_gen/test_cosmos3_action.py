# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Cosmos3 action sizing helpers (no checkpoint / GPU required).

Run:
    pytest tests/unittest/_torch/visual_gen/test_cosmos3_action.py -v
"""

import numpy as np
import PIL.Image
import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.cosmos3.action import (
    ACTION_ASPECT_RATIO_LABELS,
    EMBODIMENT_TO_DOMAIN_ID,
    EMBODIMENT_TO_RAW_ACTION_DIM,
    VIDEO_RES_SIZE_INFO,
    action_reference_frame_step,
    action_reference_size,
    crop_action_latent,
    find_closest_target_size,
    normalize_action_resolution,
    prepare_action_latents,
    resize_and_pad_action_image,
    resolve_action_content_size,
    resolve_action_size,
    resolve_domain_id,
    resolve_raw_action_dim,
)
from tensorrt_llm._torch.visual_gen.models.cosmos3.defaults import (
    COSMOS3_DOMAIN_PRESET_ALIASES,
    COSMOS3_DOMAIN_PRESETS,
    COSMOS3_EXTRA_SPECS,
    COSMOS3_POLICY_SAMPLING_PARAMS,
    get_domain_preset,
    resolve_checkpoint_policy_defaults,
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
        # find_closest_target_size picks from whatever entries exist, so a bucket
        # that lost "9,16" would silently land portrait sources on another canvas.
        assert set(VIDEO_RES_SIZE_INFO[action_resolution]) == set(ACTION_ASPECT_RATIO_LABELS)


class TestResolveActionSize:
    SOURCE_H, SOURCE_W = 480, 832

    def test_explicit_height_and_width_are_unchanged(self):
        assert resolve_action_size(400, 600, self.SOURCE_H, self.SOURCE_W, 480) == (400, 600)

    def test_unset_height_and_width_use_action_resolution_bucket(self):
        assert resolve_action_size(None, None, self.SOURCE_H, self.SOURCE_W, 480) == (480, 832)

    def test_partial_height_fills_width_from_bucket(self):
        assert resolve_action_size(400, None, self.SOURCE_H, self.SOURCE_W, 480) == (400, 832)

    def test_partial_width_fills_height_from_bucket(self):
        assert resolve_action_size(None, 600, self.SOURCE_H, self.SOURCE_W, 480) == (480, 600)


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


class TestCheckpointPolicyDefaults:
    METADATA = {
        "action_chunk_size": 32,
        "conditioning_fps": 15.0,
        "domain_name": "droid_lerobot",
    }

    def test_edge_policy_metadata_resolves_reference_recipe(self):
        policy = resolve_checkpoint_policy_defaults(self.METADATA)
        assert policy == {
            **COSMOS3_POLICY_SAMPLING_PARAMS,
            "domain_name": "droid_lerobot",
            "raw_action_dim": 8,
            "action_resolution": 480,
            "use_state": True,
            "action_chunk_size": 32,
            "frame_rate": 15.0,
            "action_fps": 15.0,
        }

        cfg = resolve_domain_action_config(checkpoint_policy_defaults=policy)
        assert cfg["domain_name"] == "droid_lerobot"
        assert cfg["raw_action_dim"] == 8
        assert cfg["action_chunk_size"] == 32
        assert cfg["num_frames"] == 33
        assert cfg["frame_rate"] == 15.0
        assert cfg["action_fps"] == 15.0
        assert cfg["use_state"] is True
        assert cfg["warnings"] == []

    def test_explicit_non_policy_domain_does_not_inherit_droid_width(self):
        policy = resolve_checkpoint_policy_defaults(self.METADATA)
        cfg = resolve_domain_action_config(
            domain_name="bridge_orig_lerobot",
            checkpoint_policy_defaults=policy,
        )
        assert cfg["raw_action_dim"] == 10
        assert cfg["action_chunk_size"] == 16
        assert cfg["use_state"] is False
        assert any("overrides checkpoint policy" in warning for warning in cfg["warnings"])

    @pytest.mark.parametrize(
        "metadata,error",
        [
            ({"action_chunk_size": 0}, "action_chunk_size"),
            ({"conditioning_fps": 0}, "conditioning_fps"),
            ({"domain_name": ""}, "domain_name"),
            ({"use_state": "yes"}, "use_state"),
        ],
    )
    def test_invalid_checkpoint_policy_metadata_raises(self, metadata, error):
        with pytest.raises(ValueError, match=error):
            resolve_checkpoint_policy_defaults(metadata)


class TestActionReferenceSize:
    """Canvas selection needs the source size, not a decoded frame."""

    def test_policy_measures_image(self, tmp_path):
        image_path = tmp_path / "frame.png"
        PIL.Image.new("RGB", (640, 480), "blue").save(image_path)
        assert action_reference_size(action_mode="policy", image=str(image_path), video=None) == (
            480,
            640,
        )

    def test_policy_prefers_image_over_video(self, tmp_path):
        image_path = tmp_path / "frame.png"
        PIL.Image.new("RGB", (320, 240), "blue").save(image_path)
        # Bytes would raise if consulted: the image must win.
        assert action_reference_size(
            action_mode="policy", image=str(image_path), video=b"not-a-video"
        ) == (240, 320)

    def test_accepts_pil_image_directly(self):
        assert action_reference_size(
            action_mode="forward_dynamics",
            image=PIL.Image.new("RGB", (256, 128)),
            video=None,
        ) == (128, 256)

    def test_missing_source_raises(self):
        with pytest.raises(ValueError, match="requires an image or video"):
            action_reference_size(action_mode="policy", image=None, video=None)

    def test_inverse_dynamics_ignores_image(self, tmp_path):
        image_path = tmp_path / "frame.png"
        PIL.Image.new("RGB", (640, 480), "blue").save(image_path)
        # inverse_dynamics conditions on the clip, so an image is not a source.
        with pytest.raises(ValueError, match="requires an image or video"):
            action_reference_size(action_mode="inverse_dynamics", image=str(image_path), video=None)

    def test_https_reference_goes_through_the_repo_loader(self, monkeypatch):
        """Bundled action prompts point at https:// frames, so a bare
        PIL.Image.open(path) would fail on every one of them."""
        import tensorrt_llm.inputs.utils as inputs_utils

        requested = []

        def fake_load_image(source, format="pt", device="cpu"):
            requested.append((source, format))
            return PIL.Image.new("RGB", (640, 480), "blue")

        monkeypatch.setattr(inputs_utils, "load_image", fake_load_image)
        assert action_reference_size(
            action_mode="policy",
            image="https://example.invalid/frame.png",
            video=None,
        ) == (480, 640)
        assert requested == [("https://example.invalid/frame.png", "pil")]

    def test_video_bytes_probe_the_container_header(self, monkeypatch):
        """Bytes are measured from the header, never by decoding a frame."""
        import tensorrt_llm.media.decoding as decoding

        monkeypatch.setattr(
            decoding,
            "video_stream_info",
            lambda data: decoding.VideoStreamInfo(480, 640, 30.0),
        )
        assert action_reference_size(
            action_mode="inverse_dynamics", image=None, video=b"\x00mp4"
        ) == (480, 640)

    def test_unreadable_video_bytes_are_rejected(self, monkeypatch):
        """An unreadable container fails here, not silently at decode."""
        import tensorrt_llm.media.decoding as decoding

        monkeypatch.setattr(decoding, "video_stream_info", lambda data: None)
        with pytest.raises(ValueError, match="could not be demuxed"):
            action_reference_size(action_mode="inverse_dynamics", image=None, video=b"\x00bad")


class TestActionReferenceFrameStep:
    """The reference is thinned to the embodiment's rate, never invented."""

    @pytest.mark.parametrize(
        "source_frame_rate, target_frame_rate, expected",
        [
            (30.0, 5.0, 6),  # bridge: every sixth frame of a 30 fps clip
            (5.0, 5.0, 1),  # already at the trained rate
            (24.0, 5.0, 5),  # 4.8 rounds to 5
            (10.0, 30.0, 1),  # slower than trained: cannot be thinned
            (None, 5.0, 1),  # header unreadable
            (0.0, 5.0, 1),  # header reported nothing usable
        ],
    )
    def test_step_from_rates(self, source_frame_rate, target_frame_rate, expected):
        assert action_reference_frame_step(source_frame_rate, target_frame_rate) == expected


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
    """The CPU half of the action contract: which tokens start clean, what the
    mask says, and how a caller's trajectory is fitted to the chunk."""

    ACTION_DIM = 8
    CHUNK = 4

    def _prepare(self, mode, **kwargs):
        kwargs.setdefault("action_chunk_size", self.CHUNK)
        kwargs.setdefault("action_dim", self.ACTION_DIM)
        return prepare_action_latents(
            mode=mode,
            generator=torch.Generator(device="cpu").manual_seed(0),
            device=torch.device("cpu"),
            dtype=torch.float32,
            **kwargs,
        )

    def test_forward_dynamics_pads_a_short_trajectory_by_holding_the_last_step(self):
        latents, _, clean, raw_dim = self._prepare(
            "forward_dynamics", raw_action_dim=None, action_input=[[1.0, 2.0], [3.0, 4.0]]
        )
        assert raw_dim == 2
        assert latents.shape == (1, self.CHUNK, self.ACTION_DIM)
        # Steps 2 and 3 repeat the supplied final step rather than going to zero.
        for step in range(2, self.CHUNK):
            torch.testing.assert_close(clean[0, step, :2], torch.tensor([3.0, 4.0]))

    def test_forward_dynamics_truncates_a_long_trajectory(self):
        _, _, clean, _ = self._prepare(
            "forward_dynamics",
            raw_action_dim=None,
            action_input=[[float(i), float(i)] for i in range(self.CHUNK + 3)],
        )
        torch.testing.assert_close(clean[0, -1, :2], torch.tensor([3.0, 3.0]))

    def test_forward_dynamics_conditions_every_step(self):
        """All action tokens are given, so none carries velocity."""
        latents, mask, clean, _ = self._prepare(
            "forward_dynamics", raw_action_dim=None, action_input=[[1.0, 2.0]] * self.CHUNK
        )
        assert torch.all(mask == 0.0)
        torch.testing.assert_close(latents, clean)

    @pytest.mark.parametrize("mode", ["policy", "inverse_dynamics"])
    def test_predicted_modes_start_from_noise_everywhere(self, mode):
        latents, mask, clean, _ = self._prepare(mode, raw_action_dim=2)
        assert torch.all(mask == 1.0)
        assert torch.all(clean == 0.0)
        assert torch.any(latents[..., :2] != 0.0)

    @pytest.mark.parametrize("mode", ["policy", "forward_dynamics", "inverse_dynamics"])
    def test_columns_above_raw_action_dim_are_zero(self, mode):
        """The head is action_dim wide but only raw_action_dim is meaningful;
        padding must not carry noise into the model or out of it."""
        kwargs = (
            {"raw_action_dim": None, "action_input": [[1.0, 2.0]] * self.CHUNK}
            if mode == "forward_dynamics"
            else {"raw_action_dim": 2}
        )
        latents, _, clean, raw_dim = self._prepare(mode, **kwargs)
        assert raw_dim == 2
        assert torch.all(latents[..., raw_dim:] == 0.0)
        assert torch.all(clean[..., raw_dim:] == 0.0)

    def test_empty_trajectory_raises(self):
        """Without this the empty slice reaches the mask broadcast and dies with
        an opaque torch shape error instead of a client-facing one."""
        with pytest.raises(ValueError, match="at least one timestep"):
            self._prepare("forward_dynamics", raw_action_dim=None, action_input=torch.zeros(0, 2))

    @pytest.mark.parametrize("raw_action_dim", [0, -1, ACTION_DIM + 1])
    def test_out_of_range_raw_action_dim_raises(self, raw_action_dim):
        with pytest.raises(ValueError, match=r"raw_action_dim must be in \[1, \d+\]"):
            self._prepare("policy", raw_action_dim=raw_action_dim)

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

    def test_policy_state_is_a_clean_leading_row(self):
        state = [0.1, 0.2]
        latents, velocity_mask, clean, raw_dim = self._prepare(
            "policy",
            raw_action_dim=2,
            action_input=state,
            use_state=True,
        )
        assert raw_dim == 2
        assert latents.shape == (1, self.CHUNK + 1, self.ACTION_DIM)
        assert velocity_mask.shape == (1, self.CHUNK + 1, 1)
        assert torch.all(velocity_mask[:, 0] == 0.0)
        assert torch.all(velocity_mask[:, 1:] == 1.0)
        torch.testing.assert_close(clean[0, 0, :2], torch.tensor(state))
        torch.testing.assert_close(latents[:, 0], clean[:, 0])
        assert torch.any(latents[:, 1:, :2] != 0.0)

    def test_policy_state_width_must_match_raw_action_dim(self):
        with pytest.raises(ValueError, match="current state width"):
            self._prepare(
                "policy",
                raw_action_dim=3,
                action_input=[0.1, 0.2],
                use_state=True,
            )

    def test_use_state_rejects_non_policy_modes(self):
        with pytest.raises(ValueError, match="only for policy"):
            self._prepare(
                "inverse_dynamics",
                raw_action_dim=2,
                action_input=[0.1, 0.2],
                use_state=True,
            )


class TestResizeAndPadActionImage:
    """Action pads to the canvas where V2V crops to it: a gripper works at the
    frame edge, so cover-scale would cut away what the policy acts on."""

    def test_contain_scale_then_pad_to_canvas(self):
        image = PIL.Image.new("RGB", (800, 400), "blue")
        out = resize_and_pad_action_image(image, target_h=480, target_w=832)
        assert (out.height, out.width) == (480, 832)

    def test_small_source_is_never_enlarged(self):
        """min(..., 1.0): a small clip keeps its own pixels and a wider border
        rather than being upscaled into blur."""
        image = PIL.Image.new("RGB", (100, 50), "blue")
        out = resize_and_pad_action_image(image, target_h=480, target_w=832)
        assert (out.height, out.width) == (480, 832)
        assert np.asarray(out)[:50, :100].any()

    def test_exact_size_is_returned_unchanged(self):
        image = PIL.Image.new("RGB", (832, 480), "blue")
        assert resize_and_pad_action_image(image, 480, 832).size == (832, 480)

    def test_aspect_ratio_is_preserved(self):
        """Contain-scale keeps the source's own aspect; the leftover strip is
        padding, not stretch."""
        image = PIL.Image.new("RGB", (400, 400), "blue")
        out = np.asarray(resize_and_pad_action_image(image, target_h=480, target_w=832))
        assert out.shape[:2] == (480, 832)
        # A square source contained in a 16:9 canvas fills the height, so the
        # scaled content is square and the pad lands on the right.
        assert out[:480, :480].any()


class TestActionContentLatent:
    def test_droid_content_keeps_source_size_before_padding(self):
        assert resolve_action_content_size(540, 640, 544, 736) == (540, 640)

    def test_large_source_is_scaled_to_fit_canvas(self):
        assert resolve_action_content_size(1080, 1920, 544, 736) == (414, 736)

    def test_droid_padding_is_removed_at_vae_resolution(self):
        latent = torch.zeros(1, 2, 3, 34, 46)
        cropped = crop_action_latent(latent, 540, 640, spatial_compression_factor=16)

        assert cropped.shape == (1, 2, 3, 33, 40)
        assert cropped.data_ptr() == latent.data_ptr()

    def test_bucket_sized_content_is_a_noop(self):
        latent = torch.zeros(1, 2, 3, 34, 46)
        cropped = crop_action_latent(latent, 544, 736, spatial_compression_factor=16)

        assert cropped is latent

    def test_content_cannot_exceed_encoded_canvas(self):
        latent = torch.zeros(1, 2, 3, 34, 46)
        with pytest.raises(ValueError, match="content exceeds encoded latent size"):
            crop_action_latent(latent, 560, 752, spatial_compression_factor=16)


class TestResolveDomainId:
    """domain_id wins, but a caller that contradicts itself is a real mistake:
    the wrong embodiment yields a fluent trajectory in another robot's dialect."""

    def test_agreeing_pair_is_accepted(self):
        assert resolve_domain_id(domain_id=7, domain_name="bridge_orig_lerobot") == 7

    def test_contradicting_pair_raises(self):
        with pytest.raises(ValueError, match="contradicts domain_name"):
            resolve_domain_id(domain_id=20, domain_name="bridge_orig_lerobot")

    def test_unlisted_name_leaves_domain_id_authoritative(self):
        assert resolve_domain_id(domain_id=31, domain_name="some-new-robot") == 31

    def test_name_alone_still_resolves(self):
        assert resolve_domain_id(domain_name="fractal") == 20

    def test_negative_domain_id_raises(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            resolve_domain_id(domain_id=-1)
