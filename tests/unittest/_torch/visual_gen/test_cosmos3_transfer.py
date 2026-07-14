# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for Cosmos3 Transfer (control-video conditioning).

Ported from vllm-omni ``tests/diffusion/models/cosmos3/test_cosmos3_pipeline.py``
(post-PR #4379) and adapted to TRT-LLM APIs, plus TRT-LLM-specific coverage for
the tensor-direct media decode and the chunk arithmetic. The diffuse-transfer
CFG tests assert the exact combination arithmetic (254/152/104/508) via
deterministic stubs, so any drift in the nested control/text CFG math fails
loudly.
"""

import os
from types import SimpleNamespace

os.environ["TLLM_DISABLE_MPI"] = "1"
os.environ["TRTLLM_DISABLE_COSMOS3_GUARDRAILS"] = "1"

import numpy as np
import PIL.Image
import pytest
import torch
import torch.nn as nn

from tensorrt_llm._torch.visual_gen.models.cosmos3 import pipeline_cosmos3 as pipeline_module
from tensorrt_llm._torch.visual_gen.models.cosmos3 import transfer as transfer_module
from tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniMoTPipeline
from tensorrt_llm._torch.visual_gen.models.cosmos3.transfer import (
    _path_media_to_uint8_cthw,
    find_closest_target_size,
    load_or_compute_control_frames,
    make_blur_control,
    make_edge_control,
    media_hw,
    pad_temporal_frames,
    resolve_transfer_config,
)
from tensorrt_llm._torch.visual_gen.models.cosmos3.transformer_cosmos3 import TransformerOutput
from tensorrt_llm._torch.visual_gen.models.cosmos3.utils import read_video_tensor

pytestmark = pytest.mark.cosmos3


def _ids(value: int) -> torch.Tensor:
    return torch.tensor([[value]], dtype=torch.long)


def _mask() -> torch.Tensor:
    return torch.ones(1, 1, dtype=torch.long)


def _req(**overrides):
    values = {"num_frames": None, "guidance_scale": None, "frame_rate": None}
    values.update(overrides)
    return SimpleNamespace(**values)


class StubScheduler:
    def __init__(self, timesteps=None):
        self.timesteps = torch.tensor(timesteps or [9, 3], dtype=torch.int64)
        self.config = SimpleNamespace(num_train_timesteps=1000, flow_shift=1.0, use_karras_sigmas=True)
        self.set_timesteps_calls = []

    def set_timesteps(self, num_steps, device=None):
        self.set_timesteps_calls.append((num_steps, device))
        self.timesteps = torch.arange(num_steps, 0, -1, dtype=torch.int64)

    def step(self, noise_pred, timestep, latents, return_dict=False):
        # No **kwargs on purpose: new pipeline-side arguments must fail loudly
        # here, not get silently swallowed. `timestep` is accepted (the
        # pipeline passes it) but unused.
        assert return_dict is False
        return (latents + noise_pred,)


class StubTransformer(nn.Module):
    """Deterministic transformer: returns full(token + 100·has_control).

    Also locks the calling convention: ``timestep`` must be the normalized
    value and ``raw_timestep`` the raw scheduler value (the regression we
    fixed after the VSA rebase).
    """

    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.cached_kv = None
        self.cached_freqs_gen = None
        self.calls = []
        self.reset_calls = 0

    def reset_cache(self):
        self.reset_calls += 1
        self.cached_kv = None
        self.cached_freqs_gen = None

    def forward(self, *, hidden_states, timestep, raw_timestep, text_ids, text_mask, **kwargs):
        del text_mask
        token = int(text_ids.reshape(-1)[0].item()) if text_ids.numel() else 0
        control_latents = kwargs.get("control_latents")
        torch.testing.assert_close(timestep, raw_timestep / self.calls_num_train_timesteps)
        self.calls.append({"token": token, "has_control": control_latents is not None})
        if self.cached_kv is None:
            marker = torch.tensor([token], dtype=torch.float32)
            self.cached_kv = [(marker, marker + 100)]
            self.cached_freqs_gen = (marker + 200, marker + 300)
        control_bonus = 100 if control_latents is not None else 0
        video = torch.full_like(hidden_states, float(token + control_bonus))
        return TransformerOutput(video=video, image=video)

    calls_num_train_timesteps = 1000


def _make_pipeline():
    pipeline = Cosmos3OmniMoTPipeline.__new__(Cosmos3OmniMoTPipeline)
    nn.Module.__init__(pipeline)
    pipeline.transformer = StubTransformer()
    pipeline.scheduler = StubScheduler()
    pipeline.pipeline_config = SimpleNamespace(torch_dtype=torch.float32)
    pipeline.vae_scale_factor_temporal = 4
    pipeline._guidance_scale = None
    pipeline._num_timesteps = None
    # Stub VAE encode: temporal subsample stands in for compression.
    pipeline._encode_video_tensor = lambda video: video[:, :, :: pipeline.vae_scale_factor_temporal]
    return pipeline


# =============================================================================
# Config resolution (transfer.py)
# =============================================================================


class TestTransferConfig:
    def test_resolve_defaults_for_edge(self):
        cfg = resolve_transfer_config({"edge": True}, _req())
        assert cfg is not None
        assert list(cfg.hints) == ["edge"]
        # DELIBERATE deviation from the reference resolve-level test (which
        # asserts the per-hint preset 3.0): TRT-LLM cannot distinguish an
        # omitted request guidance_scale from a merged pipeline default, and
        # mirrors vllm-omni's *serving* behavior where an omitted value
        # materializes to 1.0 before resolve, leaving the preset unapplied.
        assert cfg.guidance_scale == 1.0
        assert resolve_transfer_config({"edge": True}, _req(guidance_scale=6.0)).guidance_scale == 6.0
        assert cfg.control_guidance == 1.5
        assert cfg.flow_shift == 10.0
        assert cfg.num_video_frames_per_chunk == 93
        assert cfg.share_vision_temporal_positions is True

    def test_no_hints_resolves_none(self):
        assert resolve_transfer_config({}, _req()) is None
        assert resolve_transfer_config({"guidance_scale": 3.0}, _req()) is None

    def test_wsm_fps_preset_default_and_override(self):
        assert resolve_transfer_config({"wsm": True}, _req()).fps == 10
        assert resolve_transfer_config({"wsm": True, "fps": 24.0}, _req()).fps == 24.0

    def test_precomputed_control_tensor_passthrough(self):
        precomputed = torch.zeros(3, 2, 8, 8, dtype=torch.uint8)
        cfg = resolve_transfer_config({"edge": {"control": precomputed}}, _req())
        loaded = load_or_compute_control_frames(
            cfg.hints["edge"], height=8, width=8, max_frames=2, input_frames=None
        )
        assert tuple(loaded.shape) == (3, 2, 8, 8)


# =============================================================================
# Media helpers (transfer.py + utils.py)
# =============================================================================


class TestTransferMediaHelpers:
    def test_pad_temporal_frames_reflects(self):
        # Reference parity: [0, 3, 6] padded to 5 reflects the tail -> [0, 3, 6, 6, 3].
        frames = torch.arange(3 * 3, dtype=torch.uint8).reshape(1, 3, 1, 3)
        assert pad_temporal_frames(frames, 5)[0, :, 0, 0].tolist() == [0, 3, 6, 6, 3]

    def test_missing_cv2_raises_clear_error(self, monkeypatch):
        real_import_module = transfer_module.importlib.import_module

        def raise_missing_cv2(name, *args, **kwargs):
            if name == "cv2":
                raise ImportError("missing cv2")
            return real_import_module(name, *args, **kwargs)

        monkeypatch.setattr(transfer_module.importlib, "import_module", raise_missing_cv2)
        cfg = resolve_transfer_config({"edge": True}, _req())
        with pytest.raises(ImportError, match="opencv-python"):
            load_or_compute_control_frames(
                cfg.hints["edge"],
                height=8,
                width=8,
                max_frames=1,
                input_frames=torch.zeros(3, 1, 8, 8, dtype=torch.uint8),
            )

    def test_edge_uses_rgb_canny(self, monkeypatch):
        class FakeCv2:
            def __init__(self):
                self.canny_inputs = []

            def Canny(self, image, lower, upper):
                assert (lower, upper) == (100, 200)
                self.canny_inputs.append(image.copy())
                return np.zeros(image.shape[:2], dtype=np.uint8)

        fake_cv2 = FakeCv2()
        monkeypatch.setattr(transfer_module, "_import_cv2", lambda _key: fake_cv2)

        frames = torch.zeros(3, 1, 4, 5, dtype=torch.uint8)
        frames[0] = 255
        edge = make_edge_control(frames, "medium")

        assert tuple(edge.shape) == (3, 1, 4, 5)
        assert len(fake_cv2.canny_inputs) == 1
        assert fake_cv2.canny_inputs[0].shape == (4, 5, 3)

    def test_blur_uses_scaled_bilateral(self, monkeypatch):
        class FakeCv2:
            INTER_AREA = 1
            INTER_LINEAR = 2
            INTER_CUBIC = 3

            def __init__(self):
                self.bilateral_calls = []

            def resize(self, image, size, interpolation):
                del interpolation
                width, height = size
                return np.zeros((height, width, image.shape[2]), dtype=image.dtype)

            def bilateralFilter(self, image, diameter, sigma_color, sigma_space):
                self.bilateral_calls.append((image.shape, diameter, sigma_color, sigma_space))
                return image

            def GaussianBlur(self, *args, **kwargs):
                raise AssertionError("blur must use bilateralFilter, not GaussianBlur")

        fake_cv2 = FakeCv2()
        monkeypatch.setattr(transfer_module, "_import_cv2", lambda _key: fake_cv2)

        frames = torch.zeros(3, 1, 72, 72, dtype=torch.uint8)
        blurred = make_blur_control(frames, "high")

        assert tuple(blurred.shape) == (3, 1, 72, 72)
        assert fake_cv2.bilateral_calls == [((72, 72, 3), 3, 15.0, 10.0)]

    @staticmethod
    def _write_mp4(path, num_frames=5, size=16):
        cv2 = pytest.importorskip("cv2")
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (size, size))
        try:
            for _ in range(num_frames):
                writer.write(np.zeros((size, size, 3), dtype=np.uint8))
        finally:
            writer.release()

    def test_video_decode_paths(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        self._write_mp4(clip)

        frames = read_video_tensor(clip)
        assert tuple(frames.shape) == (5, 16, 16, 3) and frames.dtype == torch.uint8
        assert read_video_tensor(clip, max_frames=3).shape[0] == 3

        cthw = _path_media_to_uint8_cthw(clip, max_frames=None)
        assert tuple(cthw.shape) == (3, 5, 16, 16)
        assert media_hw(str(clip)) == (16, 16)

        image_path = tmp_path / "frame.png"
        PIL.Image.new("RGB", (8, 4), "red").save(image_path)
        image_cthw = _path_media_to_uint8_cthw(image_path, max_frames=None)
        assert tuple(image_cthw.shape) == (3, 1, 4, 8)

        with pytest.raises(FileNotFoundError):
            _path_media_to_uint8_cthw(tmp_path / "missing.mp4", max_frames=None)


# =============================================================================
# diffuse_transfer — nested control/text CFG arithmetic (ported verbatim)
# =============================================================================


class TestDiffuseTransferCFG:
    def _run(self, pipeline, *, timesteps, guidance_scale, control_guidance, **overrides):
        latents = torch.zeros(1, 2, 1, 1, 1)
        velocity_mask = torch.ones(1, 1, 1, 1, 1)
        kwargs = dict(
            latents=latents,
            timesteps=torch.tensor(timesteps),
            cond_ids=_ids(2),
            cond_mask=_mask(),
            uncond_ids=_ids(1),
            uncond_mask=_mask(),
            guidance_scale=guidance_scale,
            control_guidance=control_guidance,
            control_guidance_interval=None,
            control_latents=[torch.zeros_like(latents)],
            shared_kwargs={"video_shape": (1, 1, 1), "fps": 24.0, "noisy_frame_mask": velocity_mask},
            velocity_mask=velocity_mask,
            condition_latents=torch.zeros_like(latents),
        )
        kwargs.update(overrides)
        return pipeline.diffuse_transfer(**kwargs), latents

    def test_applies_control_and_text_cfg(self):
        pipeline = _make_pipeline()
        result, latents = self._run(pipeline, timesteps=[7], guidance_scale=3.0, control_guidance=1.5)
        # cond_full=102, no_control=2, uncond=101:
        # control_cond = 2 + 1.5*(102-2) = 152; 101 + 3*(152-101) = 254
        assert [(c["token"], c["has_control"]) for c in pipeline.transformer.calls] == [
            (2, True),
            (2, False),
            (1, True),
        ]
        torch.testing.assert_close(result, torch.full_like(latents, 254.0))

    def test_skips_idle_cfg_branches(self):
        control_only = _make_pipeline()
        result, latents = self._run(control_only, timesteps=[7], guidance_scale=1.0, control_guidance=1.5)
        assert [(c["token"], c["has_control"]) for c in control_only.transformer.calls] == [
            (2, True),
            (2, False),
        ]
        torch.testing.assert_close(result, torch.full_like(latents, 152.0))

        text_only = _make_pipeline()
        result, latents = self._run(text_only, timesteps=[7], guidance_scale=3.0, control_guidance=1.0)
        assert [(c["token"], c["has_control"]) for c in text_only.transformer.calls] == [
            (2, True),
            (1, True),
        ]
        torch.testing.assert_close(result, torch.full_like(latents, 104.0))

    def test_interval_switches_branch_counts(self):
        pipeline = _make_pipeline()
        result, latents = self._run(
            pipeline,
            timesteps=[900, 500, 100],
            guidance_scale=3.0,
            control_guidance=1.5,
            control_guidance_interval=(400.0, 1000.0),
            guidance_interval=(800.0, 1000.0),
        )
        # t=900: 3 branches -> +254; t=500: control only -> +152; t=100: single -> +102
        assert [(c["token"], c["has_control"]) for c in pipeline.transformer.calls] == [
            (2, True),
            (2, False),
            (1, True),
            (2, True),
            (2, False),
            (2, True),
        ]
        torch.testing.assert_close(result, torch.full_like(latents, 508.0))


# =============================================================================
# _forward_transfer — chunk arithmetic and multichunk stitching
# =============================================================================


class TestForwardTransferChunks:
    def test_get_transfer_num_chunks_arithmetic(self):
        chunks = Cosmos3OmniMoTPipeline._get_transfer_num_chunks
        assert chunks(93, 93, 1) == (1, 93)
        assert chunks(189, 93, 1) == (3, 92)
        assert chunks(5, 93, 1) == (1, 93)
        with pytest.raises(ValueError, match="num_conditional_frames"):
            chunks(189, 93, 93)

    def test_multichunk_overlap_path(self, monkeypatch):
        pipeline = _make_pipeline()
        captured = {"targets": [], "conditional_frames": [], "decode_calls": [], "flow_shifts": []}

        pipeline._transfer_bucket_size = lambda cfg, source_hw: (16, 16)
        tokenized = iter([( _ids(2), _mask()), (_ids(1), _mask())])
        pipeline._tokenize_prompt = lambda *args, **kwargs: next(tokenized)
        pipeline._set_flow_shift = lambda target, **kwargs: captured["flow_shifts"].append(target)

        original_prepare = pipeline._prepare_transfer_latents

        def recording_prepare(target_norm, current_conditional_frames, generator):
            captured["targets"].append(target_norm.detach().clone())
            captured["conditional_frames"].append(current_conditional_frames)
            return original_prepare(target_norm, current_conditional_frames, generator)

        pipeline._prepare_transfer_latents = recording_prepare

        decoded_chunks = [
            torch.tensor([-0.6, -0.5, -0.4, -0.3, -0.2], dtype=torch.float32),
            torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32),
        ]

        def fake_decode(latents):
            values = decoded_chunks[len(captured["decode_calls"])]
            captured["decode_calls"].append(latents.detach().clone())
            return values.view(1, 1, 5, 1, 1).expand(1, 3, 5, 16, 16).clone()

        pipeline._decode_latents_raw = fake_decode
        # Identity postprocess so assertions run on the raw float assembly.
        monkeypatch.setattr(pipeline_module, "postprocess_video_tensor", lambda video: video)

        # Input video: frame0 black (-1 normalized), frame1+ white (+1).
        video = [PIL.Image.new("RGB", (16, 16), "black")] + [
            PIL.Image.new("RGB", (16, 16), "white") for _ in range(7)
        ]
        control = torch.zeros(3, 8, 16, 16, dtype=torch.uint8)
        cfg = resolve_transfer_config(
            {
                "edge": {"control": control},
                "guidance_scale": 1.0,
                "control_guidance": 1.0,
                "max_frames": 8,
                "num_video_frames_per_chunk": 5,
                "num_conditional_frames": 1,
                "num_first_chunk_conditional_frames": 2,
            },
            _req(num_frames=8, guidance_scale=1.0),
        )

        output = pipeline._forward_transfer(
            prompt="transfer",
            negative_prompt="",
            height=16,
            width=16,
            max_frames=8,
            num_inference_steps=1,
            max_sequence_length=8,
            use_system_prompt=False,
            use_duration_template=False,
            use_resolution_template=False,
            seed=123,
            transfer_config=cfg,
            video=video,
        )

        assert captured["conditional_frames"] == [2, 1]
        assert len(captured["decode_calls"]) == 2
        assert output.video.shape == (1, 3, 8, 16, 16)
        torch.testing.assert_close(
            output.video[0, 0, :, 0, 0],
            torch.tensor([-0.6, -0.5, -0.4, -0.3, -0.2, 0.2, 0.3, 0.4]),
        )
        # First chunk target: frame0 = normalized black input, frame1 = white,
        # remainder filled by repeating the last conditional frame.
        torch.testing.assert_close(captured["targets"][0][:, :, 0], torch.full((1, 3, 16, 16), -1.0))
        torch.testing.assert_close(captured["targets"][0][:, :, 1], torch.full((1, 3, 16, 16), 1.0))
        torch.testing.assert_close(captured["targets"][0][:, :, 2:], torch.full((1, 3, 3, 16, 16), 1.0))


class TestFindClosestTargetSize:
    """Maps a source frame onto the aspect-ratio-closest output bucket for a
    resolution level, returning ``(width, height)``."""

    @pytest.mark.parametrize(
        "h, w, resolution, expected",
        [
            # Exact aspect ratios at the 720 level resolve to their own bucket.
            (720, 1280, 720, (1280, 720)),  # 16:9 landscape
            (1280, 720, 720, (720, 1280)),  # 9:16 portrait
            (512, 512, 720, (960, 960)),  # 1:1 square
            (768, 1024, 720, (1104, 832)),  # 4:3 landscape
            (1024, 768, 720, (832, 1104)),  # 3:4 portrait
            # Other levels select from that level's own table.
            (720, 1280, 480, (832, 480)),
            (256, 256, 256, (256, 256)),
        ],
    )
    def test_maps_to_matching_bucket(self, h, w, resolution, expected):
        assert find_closest_target_size(h, w, resolution) == expected

    def test_returns_width_height_order(self):
        # A landscape source (w > h) must yield a landscape bucket (target_w >
        # target_h); guards against an (h, w) transposition of the return value.
        target_w, target_h = find_closest_target_size(720, 1280, 720)
        assert (target_w, target_h) == (1280, 720)
        assert target_w > target_h

    def test_picks_nearest_when_no_exact_match(self):
        # A 2:1 ultra-wide source has no exact bucket; the closest ratio at the
        # 720 level is 16:9 (1280x720).
        assert find_closest_target_size(500, 1000, 720) == (1280, 720)

    def test_resolution_accepts_int_or_str(self):
        assert find_closest_target_size(720, 1280, 720) == find_closest_target_size(720, 1280, "720")

    def test_unknown_resolution_raises(self):
        with pytest.raises(ValueError, match="Unknown Cosmos3 transfer resolution"):
            find_closest_target_size(720, 1280, 1080)
