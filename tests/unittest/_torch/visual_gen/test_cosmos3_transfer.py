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
import pickle
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

os.environ["TLLM_DISABLE_MPI"] = "1"
os.environ["TRTLLM_DISABLE_COSMOS3_GUARDRAILS"] = "1"

import pytest
import torch
import torch.nn as nn

from tensorrt_llm._torch.visual_gen.models.cosmos3 import pipeline_cosmos3 as pipeline_module
from tensorrt_llm._torch.visual_gen.models.cosmos3 import transfer as transfer_module
from tensorrt_llm._torch.visual_gen.models.cosmos3.defaults import (
    COSMOS3_720P_PARAMS,
    COSMOS3_EXTRA_SPECS,
    COSMOS3_GENERATION_DEFAULTS,
    VIDEO_RES_SIZE_INFO,
)
from tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniMoTPipeline
from tensorrt_llm._torch.visual_gen.models.cosmos3.sampling import DISTILLED_GUIDANCE_SCALE
from tensorrt_llm._torch.visual_gen.models.cosmos3.transfer import (
    BILATERAL_D,
    BILATERAL_SIGMA_COLOR,
    BILATERAL_SIGMA_SPACE,
    _scaled_bilateral_params,
    decode_media_to_uint8_cthw,
    find_closest_target_size,
    load_or_compute_control_frames,
    pad_temporal_frames,
    resolve_transfer_config,
    uint8_cthw_to_normalized_5d,
)
from tensorrt_llm._torch.visual_gen.models.cosmos3.transformer_cosmos3 import (
    QWEN3_RECIPE,
    TransformerOutput,
)
from tensorrt_llm._torch.visual_gen.offloading import PipelineOffloader
from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer
from tensorrt_llm.media.decoding import VideoStreamInfo
from tensorrt_llm.visual_gen.params import VisualGenParams

pytestmark = pytest.mark.cosmos3

# What a plain Cosmos3 video request advertises, and so what the executor
# merges into every request before infer() sees it.
_ADVERTISED_VIDEO_DEFAULTS = COSMOS3_GENERATION_DEFAULTS[("qwen3", "video")]


def _ids(value: int) -> torch.Tensor:
    return torch.tensor([[value]], dtype=torch.long)


def _mask() -> torch.Tensor:
    return torch.ones(1, 1, dtype=torch.long)


def _fake_decode_window(num_frames: int):
    """Stand in for the NVDEC window decode, returning ``[T, H, W, 3]`` uint8."""

    def _decode(data, *, first_frame, last_frame, target_h, target_w, device):
        return torch.zeros(num_frames, target_h, target_w, 3, dtype=torch.uint8)

    return _decode


def _stub_info(height, width, frame_rate=None):
    """Stand in for the container-header probe."""
    return lambda _data: VideoStreamInfo(height, width, frame_rate)


def _req(**overrides):
    """Params as a caller supplied them: what is passed is marked caller intent.

    A real ``VisualGenParams``, not a stand-in, because telling a caller's value
    from an executor-merged default is exactly what ``model_fields_set``
    carries — and a ``SimpleNamespace`` cannot express the difference.
    """
    return VisualGenParams(**overrides)


def _merged_req(**merged):
    """Params as the executor hands them to ``infer()``.

    The values are present but marked as pipeline defaults rather than caller
    intent, mirroring ``DiffusionExecutor._merge_defaults``.
    """
    params = VisualGenParams(**merged)
    for field_name in merged:
        params.model_fields_set.discard(field_name)
    return params


class StubScheduler:
    def __init__(self, timesteps=None):
        self.timesteps = torch.tensor(timesteps or [9, 3], dtype=torch.int64)
        self.config = SimpleNamespace(
            num_train_timesteps=1000, flow_shift=1.0, use_karras_sigmas=True
        )
        self.set_timesteps_calls = []
        self.sigmas_calls = []
        self.step_generators = []

    def set_timesteps(self, num_steps=None, device=None, sigmas=None):
        if sigmas is not None:
            # Distilled: the policy installs a fixed sigma list, not a count.
            self.sigmas_calls.append(list(sigmas))
            self.timesteps = torch.arange(len(sigmas), 0, -1, dtype=torch.int64)
            return
        self.set_timesteps_calls.append((num_steps, device))
        self.timesteps = torch.arange(num_steps, 0, -1, dtype=torch.int64)

    def step(self, noise_pred, timestep, latents, return_dict=False, generator=None):
        # `generator` is accepted but nothing else is: a new pipeline-side
        # argument must fail loudly here, not get silently swallowed.
        assert return_dict is False
        if generator is not None:
            self.step_generators.append(generator)
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
        offload_context = kwargs.get("offload_context")
        torch.testing.assert_close(timestep, raw_timestep / self.calls_num_train_timesteps)
        self.calls.append(
            {
                "token": token,
                "has_control": control_latents is not None,
                "offload_context": offload_context,
            }
        )
        if self.cached_kv is None:
            marker = torch.tensor([token], dtype=torch.float32)
            self.cached_kv = [(marker, marker + 100)]
            self.cached_freqs_gen = (marker + 200, marker + 300)
        control_bonus = 100 if control_latents is not None else 0
        video = torch.full_like(hidden_states, float(token + control_bonus))
        return TransformerOutput(video=video, image=video)

    calls_num_train_timesteps = 1000


class StubSamplingPolicy:
    """Base-checkpoint stand-in for Cosmos3SamplingPolicy.

    Transfer programs the scheduler through the policy, so the stub records
    what it was asked for; ``is_distilled`` flips the distilled contract on.
    """

    def __init__(self, is_distilled=False, fixed_sigmas=(1.0, 0.75, 0.5, 0.25)):
        self.is_distilled = is_distilled
        self.fixed_sigmas = fixed_sigmas
        self.set_timesteps_calls = []
        self.step_kwargs_calls = 0
        self.flow_shift_calls = []

    def set_flow_shift(self, scheduler, target_shift, use_karras_sigmas=None):
        """Record the programmed shift and return the scheduler to install.

        Returns a distinct instance, like the real policy: the base scheduler
        is kept pristine so shifts never accumulate, which means the caller has
        to assign the result. Implemented here so the tests drive the
        pipeline's real ``_scheduler_for`` instead of a stand-in -- stubbing
        the pipeline method would re-create the hole that let a call to a
        deleted helper ship green.
        """
        del scheduler
        self.flow_shift_calls.append(target_shift)
        return StubScheduler()

    def set_timesteps(self, scheduler, num_inference_steps, device=None):
        self.set_timesteps_calls.append(num_inference_steps)
        if self.is_distilled:
            scheduler.set_timesteps(sigmas=list(self.fixed_sigmas), device=device)
        else:
            scheduler.set_timesteps(num_inference_steps, device=device)

    def scheduler_step_kwargs(self, generator):
        self.step_kwargs_calls += 1
        return {"generator": generator} if self.is_distilled else {}

    def generation_default_overrides(self):
        # Mirrors the real policy rather than returning {}, so a distilled stub
        # still overrides the table the way the checkpoint would.
        if not self.is_distilled:
            return {}
        return {
            "num_inference_steps": len(self.fixed_sigmas),
            "guidance_scale": DISTILLED_GUIDANCE_SCALE,
        }

    def num_steps(self, default):
        return len(self.fixed_sigmas) if self.is_distilled else default


def _started_timer() -> CudaPhaseTimer:
    """A timer in the state `forward()` hands to `_forward_transfer`."""
    timer = CudaPhaseTimer()
    timer.mark_pre_start()
    return timer


def _make_pipeline(sampling=None):
    pipeline = Cosmos3OmniMoTPipeline.__new__(Cosmos3OmniMoTPipeline)
    nn.Module.__init__(pipeline)
    # __new__ skips BasePipeline.__init__, which is where ``_device`` is set.
    pipeline._device = torch.device("cpu")
    pipeline.transformer = StubTransformer()
    # __new__ skips __init__, which is where the real pipeline resolves this
    # from the transformer config; the mode-defaults tables are keyed on it.
    pipeline.family = QWEN3_RECIPE.name
    pipeline.scheduler = StubScheduler()
    pipeline.sampling = sampling or StubSamplingPolicy()
    pipeline.safety_checker = None
    pipeline.pipeline_config = SimpleNamespace(
        torch_dtype=torch.float32, cpu_offload_config=None
    )
    pipeline.offloader = PipelineOffloader(pipeline)
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
        # The per-hint preset applies when the caller omitted guidance_scale,
        # matching both references. The executor merges a pipeline default into
        # every request, so `model_fields_set` -- not "is the value None" -- is
        # what separates caller intent from a merged default.
        assert cfg.guidance_scale == 3.0
        assert (
            resolve_transfer_config({"edge": True}, _req(guidance_scale=6.0)).guidance_scale == 6.0
        )
        # An executor-merged default must not read as caller intent, or the
        # preset becomes unreachable for every request that goes through infer().
        merged = resolve_transfer_config({"edge": True}, _merged_req(guidance_scale=7.0))
        assert merged.guidance_scale == 3.0
        assert cfg.control_guidance == 1.5
        assert cfg.flow_shift == 10.0
        assert cfg.num_video_frames_per_chunk == 93
        assert cfg.share_vision_temporal_positions is True

    def test_control_directive_is_appended_by_default(self):
        """Reference parity: cosmos-framework names the active control modality
        in the user prompt unless the caller opts out. The system prompt is
        untouched, which keeps the text in the training distribution."""
        cfg = resolve_transfer_config({"edge": True}, _req())
        assert cfg.emphasize_control_in_prompt is True
        emphasized = cfg.emphasized_prompt("a robot dancing")
        assert emphasized.startswith("a robot dancing")
        assert "Follow the edge control video precisely" in emphasized

    def test_control_directive_names_every_active_hint(self):
        cfg = resolve_transfer_config({"edge": True, "seg": b"clip"}, _req())
        # Hint order is TRANSFER_HINT_KEYS order, not caller order, so the
        # directive text is stable across equivalent requests.
        assert "Follow the edge, seg control video precisely" in cfg.emphasized_prompt("x")

    def test_control_directive_can_be_disabled(self):
        cfg = resolve_transfer_config({"edge": True, "emphasize_control_in_prompt": False}, _req())
        assert cfg.emphasize_control_in_prompt is False
        assert cfg.emphasized_prompt("a robot dancing") == "a robot dancing"

    def test_no_hints_resolves_none(self):
        assert resolve_transfer_config({}, _req()) is None
        assert resolve_transfer_config({"guidance_scale": 3.0}, _req()) is None

    def test_wsm_fps_preset_default_and_override(self):
        """The override is a request field, not an extra param.

        `frame_rate` reaches the pipeline as a declared generation field, so
        that is the only spelling a real request can use -- an `extra_params`
        copy of it would be rejected as an undeclared key by
        `validate_visual_gen_params` before `infer()` ever runs.
        """
        assert resolve_transfer_config({"wsm": True}, _req()).fps == 10
        assert resolve_transfer_config({"wsm": True}, _req(frame_rate=24.0)).fps == 24.0
        # The rate the executor merges into every request is not caller intent,
        # so it must not defeat the preset.
        assert resolve_transfer_config({"wsm": True}, _merged_req(frame_rate=24.0)).fps == 10

    def test_wsm_clip_presets_survive_merged_request_defaults(self):
        """wsm wants 101 frames at 10 fps, but `num_frames` and `frame_rate`
        are advertised defaults the executor merges into every request before
        `infer()` sees it. A request carrying only those merged values must
        still get the preset; values the caller actually chose must win."""
        merged = _merged_req(
            num_frames=_ADVERTISED_VIDEO_DEFAULTS["num_frames"],
            frame_rate=_ADVERTISED_VIDEO_DEFAULTS["frame_rate"],
        )
        cfg = resolve_transfer_config({"wsm": True}, merged)
        assert (cfg.num_frames, cfg.fps) == (101, 10)

        chosen = _req(num_frames=200, frame_rate=30.0)
        cfg = resolve_transfer_config({"wsm": True}, chosen)
        assert (cfg.num_frames, cfg.fps) == (200, 30.0)

    def test_advertised_defaults_are_left_intact(self):
        """The preset is recovered inside transfer, not by nulling the model's
        published defaults — clients read those to learn the output shape."""
        # Positive, not merely equal to the table they are read from: the
        # advertised entry *is* that table, so an identity check would hold
        # even if both were zeroed to make the preset win.
        assert _ADVERTISED_VIDEO_DEFAULTS["num_frames"] > 0
        assert _ADVERTISED_VIDEO_DEFAULTS["frame_rate"] > 0

    def test_precomputed_control_bytes_reach_the_decoder(self, monkeypatch):
        monkeypatch.setattr(
            transfer_module, "decode_video_reference_window", _fake_decode_window(2)
        )
        cfg = resolve_transfer_config({"edge": {"control": b"\x00control"}}, _req())
        loaded = load_or_compute_control_frames(
            cfg.hints["edge"],
            height=8,
            width=8,
            max_frames=2,
            input_frames=None,
            device=torch.device("cpu"),
        )
        assert tuple(loaded.shape) == (3, 2, 8, 8) and loaded.dtype == torch.uint8


# =============================================================================
# Media helpers (transfer.py + utils.py)
# =============================================================================


class TestTransferMediaHelpers:
    def test_pad_temporal_frames_reflects(self):
        # Reference parity: [0, 3, 6] padded to 5 reflects the tail -> [0, 3, 6, 6, 3].
        frames = torch.arange(3 * 3, dtype=torch.uint8).reshape(1, 3, 1, 3)
        assert pad_temporal_frames(frames, 5)[0, :, 0, 0].tolist() == [0, 3, 6, 6, 3]

    def test_malformed_hints_are_client_errors(self):
        # The worker classifier maps ValueError to a client error (400) and
        # anything else to an unclassified server fault (500). A caller's
        # malformed hint must not be reported as our failure.
        for payload in (123, ["edge.mp4"], {"control": 123}):
            with pytest.raises(ValueError):
                resolve_transfer_config({"edge": payload}, _req())
        with pytest.raises(ValueError):
            decode_media_to_uint8_cthw(
                "not-bytes", height=8, width=8, max_frames=1, device=torch.device("cpu")
            )

    def test_bilateral_params_scale_with_resolution(self):
        # Tuned at a 720p reference: a 72px longest side is 1/10 of it, so the
        # diameter and both sigmas scale down by the same factor.
        assert _scaled_bilateral_params(72, 72) == (3, 15.0, 10.0)
        assert _scaled_bilateral_params(720, 720) == (
            BILATERAL_D + 1,  # 30 is even; diameters are forced odd
            float(BILATERAL_SIGMA_COLOR),
            float(BILATERAL_SIGMA_SPACE),
        )
        # The longest side drives the scale, and the sigmas have a floor of 1.
        assert _scaled_bilateral_params(4, 1280) == _scaled_bilateral_params(1280, 4)
        assert _scaled_bilateral_params(1, 1) == (1, 1.0, 1.0)

    def test_generated_control_hints_require_input_frames(self):
        for key in ("edge", "blur"):
            cfg = resolve_transfer_config({key: True}, _req())
            with pytest.raises(ValueError, match="requires either a video input"):
                load_or_compute_control_frames(
                    cfg.hints[key],
                    height=8,
                    width=8,
                    max_frames=1,
                    input_frames=None,
                    device=torch.device("cpu"),
                )


class TestSourceDerivedDefaults:
    """Unset output size and frame rate follow the reference, worker-side.

    Previously only the offline example fitted the aspect, so a served portrait
    or square reference was center-cropped into the default landscape bucket.
    The probe reads the container header (no GPU, no frame decoded).
    """

    REFERENCE = Path(__file__).parent / "test_data" / "cosmos3_v2v_ref_9f_bframes.mp4"

    def _infer_req(self, _params=None, **extra):
        # Executor-merged shape: num_frames/frame_rate carry pipeline defaults,
        # height/width are declared None, and nothing reads as caller intent.
        params = (
            _params
            if _params is not None
            else _merged_req(
                num_frames=COSMOS3_720P_PARAMS["num_frames"],
                frame_rate=COSMOS3_720P_PARAMS["frame_rate"],
                max_sequence_length=COSMOS3_720P_PARAMS["max_sequence_length"],
                seed=0,
            )
        )
        params.extra_params = dict(extra)
        return SimpleNamespace(params=params, prompt="a prompt")

    def _captured(self, req):
        pipeline = _make_pipeline()  # rank is 0 while dist is uninitialized
        captured = {}
        pipeline.forward = lambda **kwargs: captured.update(kwargs)
        pipeline.infer(req)
        return captured

    def _size(self, req):
        captured = self._captured(req)
        return captured["width"], captured["height"]

    def test_square_reference_picks_the_square_bucket(self):
        # The checked-in reference is 64x64, so a real header probe end to end.
        req = self._infer_req(video=self.REFERENCE.read_bytes())
        assert self._size(req) == VIDEO_RES_SIZE_INFO["720"]["1,1"]

    # source_hw is (height, width); the bucket table is keyed (width, height).
    @pytest.mark.parametrize(
        "source_hw, bucket",
        [
            ((320, 192), "9,16"),  # portrait
            ((192, 320), "16,9"),  # landscape
            ((1104, 832), "3,4"),  # tall, not 9:16
            ((832, 1104), "4,3"),  # wide, not 16:9
            ((600, 600), "1,1"),  # square
        ],
    )
    def test_aspect_selects_the_matching_bucket(self, source_hw, bucket, monkeypatch):
        monkeypatch.setattr(pipeline_module, "video_stream_info", _stub_info(*source_hw))
        req = self._infer_req(video=b"stand-in for encoded bytes")
        assert self._size(req) == VIDEO_RES_SIZE_INFO["720"][bucket]

    def test_explicit_dimensions_win(self, monkeypatch):
        monkeypatch.setattr(pipeline_module, "video_stream_info", _stub_info(320, 192))
        req = self._infer_req(video=b"stand-in")
        req.params.height, req.params.width = 704, 1280  # assignment marks them
        assert self._size(req) == (1280, 704)

    def test_a_half_specified_size_is_left_alone(self, monkeypatch):
        # Overriding the unset half of a stated intent would be worse than
        # leaving the request on the mode defaults.
        monkeypatch.setattr(pipeline_module, "video_stream_info", _stub_info(320, 192))
        req = self._infer_req(video=b"stand-in")
        req.params.width = 1280
        assert self._size(req) == (1280, COSMOS3_720P_PARAMS["height"])

    def test_no_reference_keeps_the_mode_defaults(self):
        req = self._infer_req()
        assert self._size(req) == (COSMOS3_720P_PARAMS["width"], COSMOS3_720P_PARAMS["height"])

    def test_unreadable_reference_falls_back_to_defaults(self):
        # A convenience probe must not fail the request; the real decode still
        # reports the problem properly.
        req = self._infer_req(video=b"not a video container")
        assert self._size(req) == (COSMOS3_720P_PARAMS["width"], COSMOS3_720P_PARAMS["height"])

    def test_transfer_fits_to_a_control_when_there_is_no_video(self, monkeypatch):
        # Transfer can run on precomputed controls alone.
        monkeypatch.setattr(pipeline_module, "video_stream_info", _stub_info(320, 192))
        req = self._infer_req(edge={"control": b"stand-in"})
        assert self._size(req) == VIDEO_RES_SIZE_INFO["720"]["9,16"]

    # --- frame rate -------------------------------------------------------

    def test_source_frame_rate_is_adopted_when_unset(self, monkeypatch):
        # Emitting an 8 fps source at the merged 24 fps default replays it at
        # 3x speed and misreports its duration to the text conditioning.
        monkeypatch.setattr(pipeline_module, "video_stream_info", _stub_info(192, 320, 8.0))
        req = self._infer_req(video=b"stand-in")
        assert self._captured(req)["frame_rate"] == 8.0

    def test_explicit_frame_rate_wins(self, monkeypatch):
        # The whole point of the caller-intent bit: an explicit 24.0 is the
        # same *value* the executor would have merged, but not the same intent.
        monkeypatch.setattr(pipeline_module, "video_stream_info", _stub_info(192, 320, 8.0))
        for chosen in (24.0, 30.0):
            req = self._infer_req(_params=_req(seed=0, frame_rate=chosen), video=b"stand-in")
            assert self._captured(req)["frame_rate"] == chosen

    def test_default_stands_without_a_usable_source_rate(self, monkeypatch):
        monkeypatch.setattr(pipeline_module, "video_stream_info", _stub_info(192, 320, None))
        req = self._infer_req(video=b"stand-in")
        assert self._captured(req)["frame_rate"] == COSMOS3_720P_PARAMS["frame_rate"]

    def test_default_stands_without_a_reference(self):
        assert (
            self._captured(self._infer_req())["frame_rate"] == (COSMOS3_720P_PARAMS["frame_rate"])
        )


class TestTransferPreflightValidation:
    """Deterministic client mistakes must 400 at enqueue, not 202 then fail.

    These validators run in the coordinator (``visual_gen/params.py``), so they
    must stay in step with ``resolve_transfer_config``: anything they reject has
    to be something the worker would have rejected too, only later.
    """

    # Minimal ISO-BMFF header, enough for the container sniff.
    MP4 = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2"

    def _check(self, key, value):
        COSMOS3_EXTRA_SPECS[key].validator(value)

    def test_validators_are_picklable(self):
        # Specs are pickled to the coordinator in the READY handshake, so a
        # closure or lambda here would break serving but pass every other test.
        for key in ("edge", "blur", "depth", "seg", "wsm", "control_guidance_interval"):
            validator = COSMOS3_EXTRA_SPECS[key].validator
            assert pickle.loads(pickle.dumps(validator)) is validator

    @pytest.mark.parametrize("key", ["edge", "blur"])
    def test_generated_hints_accept_auto_compute_and_controls(self, key):
        self._check(key, True)
        self._check(key, self.MP4)
        self._check(key, {"control": self.MP4})
        self._check(key, {})

    @pytest.mark.parametrize("key", ["depth", "seg", "wsm"])
    def test_precomputed_hints_reject_auto_compute(self, key):
        # These have no generator; `true` used to 202 and then fail in the worker.
        for value in (True, {}):
            with pytest.raises(ValueError, match="no on-the-fly generator"):
                self._check(key, value)
        self._check(key, self.MP4)  # a real control clip is still fine

    def test_presets_are_checked_against_the_tables(self):
        self._check("edge", {"preset_edge_threshold": "very_high"})
        self._check("blur", {"preset_blur_strength": "very_high"})
        # Empty/None means "default", exactly as resolve_transfer_config reads it.
        self._check("edge", {"preset_edge_threshold": ""})
        with pytest.raises(ValueError, match="unsupported preset_edge_threshold"):
            self._check("edge", {"preset_edge_threshold": "sharpish"})
        with pytest.raises(ValueError, match="unsupported preset_blur_strength"):
            self._check("blur", {"preset_blur_strength": "soupy"})

    def test_rejects_paths_and_undecodable_bytes(self):
        with pytest.raises(ValueError, match="control_path"):
            self._check("edge", {"control_path": "/tmp/control.mp4"})
        with pytest.raises(ValueError, match="not a recognized video container"):
            self._check("edge", b"not a video")
        # Bare bytes and the object form are held to the same bar.
        with pytest.raises(ValueError, match="not a recognized video container"):
            self._check("edge", {"control": b"not a video"})
        with pytest.raises(ValueError, match="Omit the key"):
            self._check("edge", False)

    def test_interval_must_be_an_ordered_pair(self):
        self._check("control_guidance_interval", [0.1, 0.9])
        with pytest.raises(ValueError, match="exactly two values"):
            self._check("control_guidance_interval", [0.5])
        with pytest.raises(ValueError, match="ordered as"):
            self._check("control_guidance_interval", [0.9, 0.1])

    def test_frame_counts_are_bounded(self):
        self._check("num_video_frames_per_chunk", 93)
        self._check("num_conditional_frames", 0)
        for key in ("num_video_frames_per_chunk", "max_frames"):
            with pytest.raises(ValueError, match="positive frame count"):
                self._check(key, 0)
        for key in ("num_conditional_frames", "num_first_chunk_conditional_frames"):
            with pytest.raises(ValueError, match="non-negative frame count"):
                self._check(key, -1)


class TestTransferFrameConversions:
    """Unit tests for the transfer control-frame conversion helpers (CPU-only)."""

    def test_uint8_cthw_to_normalized_5d_maps_0_255_to_pm1(self):
        black = torch.zeros(3, 2, 4, 5, dtype=torch.uint8)
        out = uint8_cthw_to_normalized_5d(black, dtype=torch.float32)
        assert out.shape == (1, 3, 2, 4, 5) and out.dtype == torch.float32
        assert torch.allclose(out, torch.full_like(out, -1.0))  # 0 -> -1
        white = torch.full((3, 1, 2, 2), 255, dtype=torch.uint8)
        assert torch.allclose(
            uint8_cthw_to_normalized_5d(white, dtype=torch.float32),
            torch.ones(1, 3, 1, 2, 2),  # 255 / 127.5 - 1 == 1.0
        )

    def test_uint8_cthw_to_normalized_5d_rejects_bad_shape(self):
        with pytest.raises(ValueError, match="3, T, H, W"):
            uint8_cthw_to_normalized_5d(
                torch.zeros(2, 4, 4, dtype=torch.uint8), dtype=torch.float32
            )


class TestTransferControlPayloads:
    """Controls cross the extra-param boundary as encoded bytes, like ``video``."""

    def test_hint_accepts_bytes_bare_and_in_an_object(self):
        assert resolve_transfer_config({"edge": b"MP4"}, _req()).hints["edge"].control == b"MP4"
        cfg = resolve_transfer_config({"depth": {"control": b"MP4"}}, _req())
        assert cfg.hints["depth"].control == b"MP4"

    def test_hint_rejects_a_bare_path(self):
        with pytest.raises(ValueError, match="not a path"):
            resolve_transfer_config({"edge": "/tmp/control.mp4"}, _req())

    def test_hint_rejects_control_path(self):
        with pytest.raises(ValueError, match="control_path"):
            resolve_transfer_config({"edge": {"control_path": "/tmp/control.mp4"}}, _req())

    def test_hint_rejects_non_bytes_control(self):
        with pytest.raises(ValueError, match="encoded MP4/AVI bytes"):
            resolve_transfer_config({"edge": {"control": torch.zeros(3, 1, 4, 4)}}, _req())

    def test_decode_asks_for_the_leading_window_and_returns_cthw(self, monkeypatch):
        seen = {}

        def fake_decode(data, *, first_frame, last_frame, target_h, target_w, device):
            seen.update(data=data, first=first_frame, last=last_frame, h=target_h, w=target_w)
            return torch.zeros(3, target_h, target_w, 3, dtype=torch.uint8)

        monkeypatch.setattr(transfer_module, "decode_video_reference_window", fake_decode)
        out = decode_media_to_uint8_cthw(
            b"clip", height=8, width=6, max_frames=4, device=torch.device("cpu")
        )
        assert tuple(out.shape) == (3, 3, 8, 6) and out.dtype == torch.uint8
        assert seen == {"data": b"clip", "first": 0, "last": 3, "h": 8, "w": 6}

    def test_decode_rejects_unencoded_payloads(self):
        with pytest.raises(ValueError, match="encoded MP4/AVI bytes"):
            decode_media_to_uint8_cthw(
                torch.zeros(3, 1, 4, 4),
                height=4,
                width=4,
                max_frames=1,
                device=torch.device("cpu"),
            )

    def test_decode_rejects_a_nonpositive_window(self):
        with pytest.raises(ValueError, match="max_frames must be positive, got 0"):
            decode_media_to_uint8_cthw(
                b"clip", height=4, width=4, max_frames=0, device=torch.device("cpu")
            )


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
            shared_kwargs={
                "video_shape": (1, 1, 1),
                "fps": 24.0,
                "noisy_frame_mask": velocity_mask,
            },
            velocity_mask=velocity_mask,
            condition_latents=torch.zeros_like(latents),
            generator=torch.Generator().manual_seed(0),
        )
        kwargs.update(overrides)
        return pipeline.diffuse_transfer(**kwargs), latents

    def test_applies_control_and_text_cfg(self):
        pipeline = _make_pipeline()
        result, latents = self._run(
            pipeline, timesteps=[7], guidance_scale=3.0, control_guidance=1.5
        )
        # cond_full=102, no_control=2, uncond=101:
        # control_cond = 2 + 1.5*(102-2) = 152; 101 + 3*(152-101) = 254
        assert [(c["token"], c["has_control"]) for c in pipeline.transformer.calls] == [
            (2, True),
            (2, False),
            (1, True),
        ]
        torch.testing.assert_close(result, torch.full_like(latents, 254.0))

    def test_passes_offload_context_to_each_transformer_branch(self):
        pipeline = _make_pipeline()

        def offload_context(_component_name):
            return nullcontext()

        pipeline.offloader.context_if_requested = offload_context
        self._run(pipeline, timesteps=[7], guidance_scale=3.0, control_guidance=1.5)

        assert len(pipeline.transformer.calls) == 3
        assert all(
            call["offload_context"] is offload_context for call in pipeline.transformer.calls
        )

    def test_skips_idle_cfg_branches(self):
        control_only = _make_pipeline()
        result, latents = self._run(
            control_only, timesteps=[7], guidance_scale=1.0, control_guidance=1.5
        )
        assert [(c["token"], c["has_control"]) for c in control_only.transformer.calls] == [
            (2, True),
            (2, False),
        ]
        torch.testing.assert_close(result, torch.full_like(latents, 152.0))

        text_only = _make_pipeline()
        result, latents = self._run(
            text_only, timesteps=[7], guidance_scale=3.0, control_guidance=1.0
        )
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

    def test_decode_window_is_bounded_by_the_output_length(self, monkeypatch):
        """The decoder reserves its retention ring from the requested window, so
        asking for ``max_frames`` (5000 by default) would reserve ~14 GB at 720p
        before a frame lands. Only what gets generated is decoded."""
        pipeline = _make_pipeline()
        windows = []

        class StopAfterDecode(Exception):
            pass

        def recording_decode(data, *, first_frame, last_frame, target_h, target_w, device):
            windows.append((first_frame, last_frame))
            raise StopAfterDecode

        monkeypatch.setattr(transfer_module, "decode_video_reference_window", recording_decode)
        cfg = resolve_transfer_config({"edge": {"control": b"clip"}}, _req())
        assert cfg.max_frames == 5000  # the default ceiling stays in place

        with pytest.raises(StopAfterDecode):
            pipeline._forward_transfer(
                prompt="transfer",
                negative_prompt="",
                height=16,
                width=16,
                max_frames=cfg.max_frames,
                num_inference_steps=1,
                max_sequence_length=8,
                use_system_prompt=False,
                use_duration_template=False,
                use_resolution_template=False,
                seed=1,
                frame_rate=24.0,
                num_frames=189,
                use_guardrails=False,
                timer=_started_timer(),
                transfer_config=cfg,
                video=b"input-clip",
            )

        assert windows == [(0, 188)]

    def test_multichunk_overlap_path(self, monkeypatch):
        pipeline = _make_pipeline()
        captured = {"targets": [], "conditional_frames": [], "decode_calls": []}

        tokenized = iter([(_ids(2), _mask()), (_ids(1), _mask())])
        pipeline._tokenize_prompt = lambda *args, **kwargs: next(tokenized)

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

        # Input video: frame0 black (-1 normalized), frame1+ white (+1); the
        # control is an all-black clip. Both arrive as encoded bytes and are
        # decoded on the worker, so the decode is what the stub stands in for.
        def fake_media_decode(data, *, first_frame, last_frame, target_h, target_w, device):
            frames = torch.zeros(8, target_h, target_w, 3, dtype=torch.uint8)
            if data == b"input-clip":
                frames[1:] = 255
            return frames

        monkeypatch.setattr(transfer_module, "decode_video_reference_window", fake_media_decode)
        cfg = resolve_transfer_config(
            {
                "edge": {"control": b"control-clip"},
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
            frame_rate=24.0,
            num_frames=8,
            use_guardrails=False,
            timer=_started_timer(),
            transfer_config=cfg,
            video=b"input-clip",
        )

        assert captured["conditional_frames"] == [2, 1]
        assert len(captured["decode_calls"]) == 2
        # (B, T, H, W, C) uint8 -- what postprocess_video_tensor declares.
        assert output.video.shape == (1, 8, 16, 16, 3)
        # The stitched sequence, run through the same conversion the pipeline
        # applies, so this still asserts the chunk arithmetic rather than a
        # hand-computed uint8 table.
        expected = torch.tensor([-0.6, -0.5, -0.4, -0.3, -0.2, 0.2, 0.3, 0.4])
        torch.testing.assert_close(
            output.video[0, :, 0, 0, 0],
            pipeline_module.postprocess_video_tensor(
                expected.view(1, 1, 8, 1, 1).expand(1, 3, 8, 16, 16)
            )[0, :, 0, 0, 0],
        )
        # First chunk target: frame0 = normalized black input, frame1 = white,
        # remainder filled by repeating the last conditional frame.
        torch.testing.assert_close(
            captured["targets"][0][:, :, 0], torch.full((1, 3, 16, 16), -1.0)
        )
        torch.testing.assert_close(captured["targets"][0][:, :, 1], torch.full((1, 3, 16, 16), 1.0))
        torch.testing.assert_close(
            captured["targets"][0][:, :, 2:], torch.full((1, 3, 3, 16, 16), 1.0)
        )


class TestControlLengthMismatch:
    """Hint lengths decide the output length and how much control is invented."""

    @staticmethod
    def _frames(count: int) -> torch.Tensor:
        return torch.zeros(3, count, 16, 16, dtype=torch.uint8)

    def _warning(self, monkeypatch, per_hint: dict, total_frames: int) -> str:
        """Warnings emitted for one (hints, total_frames) pair.

        ``tensorrt_llm.logger`` does not propagate to the root logger, so this
        captures through the module's logger rather than pytest's caplog.
        """
        pipeline = _make_pipeline()
        seen: list[str] = []
        monkeypatch.setattr(
            pipeline_module.logger,
            "warning",
            lambda *args, **kwargs: seen.append(" ".join(str(a) for a in args)),
        )
        pipeline._warn_on_control_length_mismatch(per_hint, total_frames)
        return " ".join(seen)

    def test_warns_when_a_hint_is_mostly_padding(self, monkeypatch):
        text = self._warning(monkeypatch, {"edge": self._frames(200), "seg": self._frames(50)}, 200)
        assert "seg: 50" in text
        assert "200 frames" in text
        # The hint that set the length was not padded, so it is not named.
        assert "edge:" not in text

    def test_silent_for_a_short_tail_difference(self, monkeypatch):
        """A few frames of ping-pong at the tail is the normal case the
        reference pads for; warning on it would train people to ignore this."""
        assert (
            self._warning(monkeypatch, {"edge": self._frames(200), "seg": self._frames(195)}, 200)
            == ""
        )

    def test_silent_when_nothing_is_padded(self, monkeypatch):
        """A request that pins `num_frames` down to the shortest clip truncates
        the long hint rather than padding the short one, so no control is
        invented and there is nothing to warn about."""
        assert (
            self._warning(monkeypatch, {"edge": self._frames(200), "seg": self._frames(50)}, 50)
            == ""
        )

    def test_output_length_follows_the_longest_hint(self, monkeypatch):
        """`edge` sorts before `seg` in TRANSFER_HINT_KEYS, so taking the first
        hint's length would truncate the longer seg control to 4 frames and cut
        the output in half -- on nothing the request expressed."""
        pipeline = _make_pipeline()
        tokenized = iter([(_ids(2), _mask()), (_ids(1), _mask())])
        pipeline._tokenize_prompt = lambda *a, **k: next(tokenized)
        pipeline._decode_latents_raw = lambda latents: torch.zeros(1, 3, 5, 16, 16)

        lengths = {b"edge-clip": 4, b"seg-clip": 8}

        def fake_media_decode(data, *, first_frame, last_frame, target_h, target_w, device):
            count = min(lengths[data], last_frame + 1)
            return torch.zeros(count, target_h, target_w, 3, dtype=torch.uint8)

        monkeypatch.setattr(transfer_module, "decode_video_reference_window", fake_media_decode)
        cfg = resolve_transfer_config(
            {
                "edge": {"control": b"edge-clip"},
                "seg": {"control": b"seg-clip"},
                "max_frames": 8,
                "num_video_frames_per_chunk": 5,
                "num_conditional_frames": 1,
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
            seed=1,
            frame_rate=24.0,
            num_frames=8,
            use_guardrails=False,
            timer=_started_timer(),
            transfer_config=cfg,
            video=None,
        )
        # (B, T, H, W, C): frames are dim 1.
        assert output.video.shape[1] == 8


class TestTransferFrameRateResolution:
    """What `_forward_transfer` actually emits at, not what `infer()` hands it.

    `resolve_transfer_config` runs before the pipeline probes the source, and
    `_forward_transfer` prefers ``transfer_config.fps`` over its ``frame_rate``
    argument. A test that stubs ``forward()`` and asserts what ``infer()``
    passes cannot see that, and will pass while the value is discarded one
    layer down -- which is exactly what happened.
    """

    def _emitted_fps(self, monkeypatch, *, hints, caller, infer_fps):
        pipeline = _make_pipeline()
        monkeypatch.setattr(
            transfer_module, "decode_video_reference_window", _fake_decode_window(5)
        )
        tokenized = iter([(_ids(2), _mask()), (_ids(1), _mask())])
        pipeline._tokenize_prompt = lambda *a, **k: next(tokenized)
        pipeline._decode_latents_raw = lambda latents: torch.zeros(1, 3, 5, 16, 16)

        params = _merged_req(frame_rate=COSMOS3_720P_PARAMS["frame_rate"], num_frames=5)
        for key, value in caller.items():
            setattr(params, key, value)  # assignment marks it as caller intent
        cfg = resolve_transfer_config({**hints, "num_video_frames_per_chunk": 5}, params)
        return pipeline._forward_transfer(
            prompt="transfer",
            negative_prompt="",
            height=16,
            width=16,
            max_frames=cfg.max_frames,
            num_inference_steps=35,
            max_sequence_length=8,
            use_system_prompt=False,
            use_duration_template=False,
            use_resolution_template=False,
            seed=1,
            frame_rate=infer_fps,  # what infer() resolved and passed down
            num_frames=5,
            use_guardrails=False,
            timer=_started_timer(),
            transfer_config=cfg,
            video=None,
        ).frame_rate

    def test_source_rate_reaches_the_output(self, monkeypatch):
        # The regression: config.fps used to capture the executor-merged 24 and
        # shadow the rate the pipeline inferred from the source.
        fps = self._emitted_fps(
            monkeypatch, hints={"edge": {"control": b"clip"}}, caller={}, infer_fps=8.0
        )
        assert fps == 8.0

    def test_explicit_request_rate_wins_over_the_source(self, monkeypatch):
        fps = self._emitted_fps(
            monkeypatch,
            hints={"edge": {"control": b"clip"}},
            caller={"frame_rate": 30.0},
            infer_fps=30.0,
        )
        assert fps == 30.0

    def test_a_pinned_frame_count_keeps_the_default_rate(self, monkeypatch):
        # `seconds` is converted to num_frames at the default rate before the
        # worker sees the media, so adopting the source rate here would change
        # the duration the caller asked for.
        fps = self._emitted_fps(
            monkeypatch,
            hints={"edge": {"control": b"clip"}},
            caller={"num_frames": 5},
            infer_fps=COSMOS3_720P_PARAMS["frame_rate"],
        )
        assert fps == COSMOS3_720P_PARAMS["frame_rate"]

    def test_wsm_preset_outranks_the_source(self, monkeypatch):
        fps = self._emitted_fps(
            monkeypatch, hints={"wsm": {"control": b"clip"}}, caller={}, infer_fps=8.0
        )
        assert fps == 10


class TestTransferSamplingAndSafety:
    """The transfer branch must not skip what every other mode goes through."""

    def _run(self, pipeline, monkeypatch, *, use_guardrails=False, cfg_extra=None, tokenize=None):
        monkeypatch.setattr(
            transfer_module, "decode_video_reference_window", _fake_decode_window(5)
        )
        tokenized = iter([(_ids(2), _mask()), (_ids(1), _mask())])
        pipeline._tokenize_prompt = tokenize or (lambda *a, **k: next(tokenized))
        pipeline._decode_latents_raw = lambda latents: torch.zeros(1, 3, 5, 16, 16)
        cfg = resolve_transfer_config(
            {"edge": {"control": b"clip"}, "num_video_frames_per_chunk": 5, **(cfg_extra or {})},
            _req(num_frames=5, guidance_scale=1.0),
        )
        return pipeline._forward_transfer(
            prompt="transfer",
            negative_prompt="",
            height=16,
            width=16,
            max_frames=cfg.max_frames,
            num_inference_steps=35,
            max_sequence_length=8,
            use_system_prompt=False,
            use_duration_template=False,
            use_resolution_template=False,
            seed=1,
            frame_rate=24.0,
            num_frames=5,
            use_guardrails=use_guardrails,
            timer=_started_timer(),
            transfer_config=cfg,
            video=None,
        )

    def test_control_directive_reaches_the_tokenizer(self, monkeypatch):
        """Resolving the directive is not enough -- it has to reach the text
        the model actually conditions on, and only the positive prompt."""
        pipeline = _make_pipeline()
        seen = []
        tokenized = iter([(_ids(2), _mask()), (_ids(1), _mask())])

        def capture(prompt, *args, **kwargs):
            seen.append(prompt)
            return next(tokenized)

        self._run(pipeline, monkeypatch, tokenize=capture)
        assert "Follow the edge control video precisely" in seen[0]
        assert "Follow the edge" not in seen[1], "directive leaked into the negative prompt"

    def test_transfer_installs_a_shifted_scheduler(self, monkeypatch):
        """Transfer owns its scheduler setup, so it must actually perform it.

        ``forward()`` deliberately skips the scheduler rebuild when a transfer
        config is present, which makes this the only place the shift is applied
        -- and the only thing standing between a request and whatever schedule
        the previous request left on the worker. Asserted through the real
        ``_scheduler_for``: a rename of that helper breaks here rather than
        being papered over by a stub.
        """
        sampling = StubSamplingPolicy()
        pipeline = _make_pipeline(sampling=sampling)
        pipeline.scheduler = StubScheduler()
        baseline = pipeline.scheduler
        self._run(pipeline, monkeypatch)
        assert sampling.flow_shift_calls == [
            transfer_module.TRANSFER_DEFAULTS["edge"]["flow_shift"]
        ], "transfer did not program the hint's flow shift"
        # Installed, not merely built: a bare call that drops the returned
        # scheduler leaves the previous request's schedule in place.
        assert pipeline.scheduler is not baseline, (
            "transfer built a scheduler but never installed it"
        )

    def test_schedule_is_programmed_through_the_sampling_policy(self, monkeypatch):
        """A distilled checkpoint runs a fixed sigma list, not a step count."""
        sampling = StubSamplingPolicy(is_distilled=True)
        pipeline = _make_pipeline(sampling=sampling)
        self._run(pipeline, monkeypatch)
        assert sampling.set_timesteps_calls, "transfer bypassed the sampling policy"
        # Distilled: the policy substitutes its own step count for the request's.
        assert pipeline.scheduler.sigmas_calls == [list(sampling.fixed_sigmas)]

    def test_distilled_step_draws_noise_from_the_seeded_generator(self, monkeypatch):
        """Unseeded SDE noise diverges the replicated latents across ranks."""
        sampling = StubSamplingPolicy(is_distilled=True)
        pipeline = _make_pipeline(sampling=sampling)
        self._run(pipeline, monkeypatch)
        assert pipeline.scheduler.step_generators, "scheduler.step() got no generator"
        assert all(g is not None for g in pipeline.scheduler.step_generators)

    def test_base_checkpoint_passes_no_generator(self, monkeypatch):
        pipeline = _make_pipeline(sampling=StubSamplingPolicy(is_distilled=False))
        self._run(pipeline, monkeypatch)
        assert pipeline.scheduler.step_generators == []

    def test_output_is_screened_when_guardrails_are_on(self, monkeypatch):
        pipeline = _make_pipeline()
        seen = {}

        def offload_context(component_name):
            seen["offload_component"] = component_name
            return nullcontext()

        pipeline.offloader.context_if_requested = offload_context

        class Checker:
            pass

        pipeline.safety_checker = Checker()
        monkeypatch.setattr(
            pipeline_module,
            "check_video_safety",
            lambda video, checker: seen.setdefault("called", True) and video,
        )
        self._run(pipeline, monkeypatch, use_guardrails=True)
        assert seen.get("called"), "transfer returned generated video unscreened"
        assert (
            seen["offload_component"]
            == pipeline_module.COSMOS3_VIDEO_GUARDRAIL_OFFLOAD_COMPONENT
        )

    def test_phase_timings_are_populated(self, monkeypatch):
        output = self._run(_make_pipeline(), monkeypatch)
        assert output.denoise > 0.0, "transfer reported a zero denoise phase"


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
        assert find_closest_target_size(720, 1280, 720) == find_closest_target_size(
            720, 1280, "720"
        )

    def test_unknown_resolution_raises(self):
        with pytest.raises(ValueError, match="Unknown Cosmos3 transfer resolution"):
            find_closest_target_size(720, 1280, 1080)
