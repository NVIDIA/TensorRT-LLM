# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for Cosmos3OmniMoTPipeline.

Loads Cosmos3-Nano when available, runs end-to-end generation, and asserts
valid uint8 video/image outputs and float32 audio when enabled. No diffusers
reference comparison.

Run all pipeline smoke tests:
    pytest tests/unittest/_torch/visual_gen/test_cosmos3_pipeline.py -v -s -m cosmos3

Run T2I only:
    pytest tests/unittest/_torch/visual_gen/test_cosmos3_pipeline.py -v -s -m cosmos3_t2i

Run audio only:
    pytest tests/unittest/_torch/visual_gen/test_cosmos3_pipeline.py -v -s -m cosmos3_audio

Run prompt metadata unit tests (no GPU):
    pytest tests/unittest/_torch/visual_gen/test_cosmos3_pipeline.py -v -k FormatPromptWithMetadata

Override checkpoint:
    DIFFUSION_MODEL_PATH_COSMOS3=/path/to/Cosmos3-Nano \\
        pytest tests/unittest/_torch/visual_gen/test_cosmos3_pipeline.py -v -s
"""

import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ["TLLM_DISABLE_MPI"] = "1"
os.environ["TRTLLM_DISABLE_COSMOS3_GUARDRAILS"] = "1"

import PIL.Image
import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_cosmos3 import (
    COSMOS3_DEFAULT_RESOLUTION_TEMPLATE,
    COSMOS3_DEFAULT_SYSTEM_PROMPT,
    COSMOS3_DURATION_TEMPLATE,
    COSMOS3_IMAGE_RESOLUTION_TEMPLATE,
    Cosmos3OmniMoTPipeline,
    _condition_pixel_frame_count,
    _normalize_condition_frame_indexes_vision,
    _normalize_condition_video_keep,
)
from tensorrt_llm._torch.visual_gen.models.cosmos3.defaults import (
    COSMOS3_ACTION_PARAMS,
    COSMOS3_DEFAULT_CONDITION_FRAME_INDEXES_VISION,
    COSMOS3_DEFAULT_CONDITION_VIDEO_KEEP,
    COSMOS3_EXTRA_SPECS,
    COSMOS3_T2I_PARAMS,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm.visual_gen.args import TorchCompileConfig, VisualGenArgs

pytestmark = pytest.mark.cosmos3


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


@pytest.fixture(autouse=True)
def _cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _llm_models_root() -> str:
    root = Path("/home/scratch.trt_llm_data_ci/llm-models/")
    if "LLM_MODELS_ROOT" in os.environ:
        root = Path(os.environ["LLM_MODELS_ROOT"])
    if not root.exists():
        root = Path("/scratch.trt_llm_data/llm-models/")
    assert root.exists(), (
        "Set LLM_MODELS_ROOT or ensure /home/scratch.trt_llm_data_ci/llm-models/ is accessible."
    )
    return str(root)


def _checkpoint(env_var: str, default_name: str) -> str:
    return os.environ.get(env_var) or os.path.join(_llm_models_root(), default_name)


COSMOS3_NANO_PATH = _checkpoint("DIFFUSION_MODEL_PATH_COSMOS3", "Cosmos3-Nano")

PROMPT = "A serene mountain lake at sunrise with mist rising from the water."
NUM_STEPS = 4
SEED = 42
HEIGHT = 720
WIDTH = 1280
NUM_FRAMES = 9
GUIDANCE_SCALE = 6.0
FRAME_RATE = 24.0

# T2I smoke resolution — smaller than the 1024 default to keep CI memory down;
# ``output_type="image"`` still exercises flow_shift and guidance_interval.
T2I_HEIGHT = 512
T2I_WIDTH = 512
T2I_GUIDANCE_SCALE = COSMOS3_T2I_PARAMS["guidance_scale"]

COSMOS3_FP8_QUANT_CONFIG = {
    "quant_algo": "FP8",
    "dynamic": True,
    "ignore": ["language_model.*", "vae2llm", "llm2vae", "time_embedder.*"],
}


def _require_checkpoint() -> str:
    if not COSMOS3_NANO_PATH or not os.path.exists(COSMOS3_NANO_PATH):
        pytest.skip(f"Checkpoint not found: {COSMOS3_NANO_PATH}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return COSMOS3_NANO_PATH


def _load_pipeline(checkpoint_path: str, **visual_gen_kwargs):
    args = VisualGenArgs(
        model=checkpoint_path,
        torch_compile_config=TorchCompileConfig(enable=False),
        **visual_gen_kwargs,
    )
    return PipelineLoader(args).load(skip_warmup=True)


def _run_forward(
    pipeline,
    *,
    image=None,
    num_frames=NUM_FRAMES,
    height=HEIGHT,
    width=WIDTH,
    guidance_scale=GUIDANCE_SCALE,
    **extra,
):
    return pipeline.forward(
        prompt=PROMPT,
        image=image,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=NUM_STEPS,
        guidance_scale=guidance_scale,
        seed=SEED,
        frame_rate=FRAME_RATE,
        use_guardrails=False,
        **extra,
    )


def _assert_valid_video(
    video: torch.Tensor,
    *,
    num_frames: int,
    height: int = HEIGHT,
    width: int = WIDTH,
):
    """PipelineOutput.video is (B, T, H, W, C) uint8 per output.py."""
    assert video is not None
    assert video.dtype == torch.uint8
    assert video.dim() == 5, f"Expected (B,T,H,W,C), got {video.shape}"
    batch, t, h, w, c = video.shape
    assert batch == 1
    assert t == num_frames
    assert h == height and w == width
    assert c == 3
    vf = video.float()
    assert not torch.isnan(vf).any()
    assert not torch.isinf(vf).any()
    assert vf.min() >= 0 and vf.max() <= 255


def _assert_valid_image(
    image: torch.Tensor,
    *,
    height: int = T2I_HEIGHT,
    width: int = T2I_WIDTH,
):
    """PipelineOutput.image is (B, H, W, C) uint8 per output.py."""
    assert image is not None
    assert image.dtype == torch.uint8
    assert image.dim() == 4, f"Expected (B,H,W,C), got {image.shape}"
    batch, h, w, c = image.shape
    assert batch == 1
    assert h == height and w == width
    assert c == 3
    img = image.float()
    assert not torch.isnan(img).any()
    assert not torch.isinf(img).any()
    assert img.min() >= 0 and img.max() <= 255


def _assert_valid_audio(
    audio: torch.Tensor,
    audio_sample_rate: int,
):
    """PipelineOutput.audio is (B, C, T) float32."""
    assert audio is not None
    assert audio_sample_rate is not None and audio_sample_rate > 0
    assert audio.dtype == torch.float32
    assert audio.dim() == 3, f"Expected (B,C,T), got {audio.shape}"
    batch, channels, samples = audio.shape
    assert batch == 1
    assert channels >= 1
    assert samples > 0
    af = audio.float()
    assert not torch.isnan(af).any()
    assert not torch.isinf(af).any()


def _require_audio_pipeline(pipeline) -> None:
    if not getattr(pipeline, "audio_gen", False):
        pytest.skip("Checkpoint does not enable audio generation")
    if not hasattr(pipeline, "audio_tokenizer"):
        pytest.skip("Audio tokenizer was not loaded for this pipeline")


def _require_action_pipeline(pipeline) -> None:
    if not getattr(pipeline, "action_gen", False):
        pytest.skip("Checkpoint does not enable action generation")


def _assert_valid_action(action: torch.Tensor, *, raw_action_dim: int, chunk_size: int):
    assert action is not None
    assert action.dtype == torch.float32
    assert action.dim() == 3, f"Expected (B,T,D), got {action.shape}"
    batch, t, d = action.shape
    assert batch == 1
    assert t == chunk_size
    assert d == raw_action_dim
    af = action.float()
    assert not torch.isnan(af).any()
    assert not torch.isinf(af).any()


def _scheduler_use_karras_sigmas(scheduler) -> bool | None:
    value = getattr(scheduler.config, "use_karras_sigmas", None)
    return None if value is None else bool(value)


def _assert_scheduler_config(
    pipeline,
    *,
    flow_shift: float,
    use_karras_sigmas: bool | None,
):
    assert float(getattr(pipeline.scheduler.config, "flow_shift")) == pytest.approx(
        float(flow_shift)
    )
    assert float(pipeline._current_flow_shift) == pytest.approx(float(flow_shift))
    assert pipeline._current_scheduler_use_karras_sigmas == use_karras_sigmas
    assert _scheduler_use_karras_sigmas(pipeline.scheduler) == use_karras_sigmas


def _assert_default_video_scheduler_config(pipeline):
    _assert_scheduler_config(
        pipeline,
        flow_shift=pipeline._engine_init_flow_shift,
        use_karras_sigmas=pipeline._base_scheduler_use_karras_sigmas,
    )


def _make_test_image() -> PIL.Image.Image:
    image_path = os.environ.get("COSMOS3_TEST_IMAGE")
    if image_path and os.path.exists(image_path):
        return PIL.Image.open(image_path).convert("RGB")
    return PIL.Image.new("RGB", (WIDTH, HEIGHT), color=(64, 128, 192))


def _make_test_video(
    num_frames: int = NUM_FRAMES,
    *,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> list[PIL.Image.Image]:
    image = _make_test_image().resize((width, height))
    return [image.copy() for _ in range(num_frames)]


@pytest.fixture
def cosmos3_format_pipeline():
    """Minimal pipeline for prompt formatting helpers (no checkpoint)."""
    return Cosmos3OmniMoTPipeline.__new__(Cosmos3OmniMoTPipeline)


def _format_prompt_with_metadata(
    pipeline,
    prompt: str,
    *,
    height: int = HEIGHT,
    width: int = WIDTH,
    num_frames: int = 189,
    frame_rate: float = FRAME_RATE,
    duration_template=COSMOS3_DURATION_TEMPLATE,
    resolution_template=COSMOS3_DEFAULT_RESOLUTION_TEMPLATE,
    force_duration_template: bool = False,
) -> str:
    return pipeline._format_prompt_with_metadata(
        prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        duration_template=duration_template,
        resolution_template=resolution_template,
        force_duration_template=force_duration_template,
    )


class TestFormatPromptWithMetadataPlainText:
    def test_appends_duration_and_resolution(self, cosmos3_format_pipeline):
        result = _format_prompt_with_metadata(cosmos3_format_pipeline, "A cat on a beach")
        assert result.startswith("A cat on a beach.")
        assert "7.9 seconds long" in result
        assert "720x1280" in result

    def test_matches_apply_metadata_templates(self, cosmos3_format_pipeline):
        prompt = "Mountain lake at sunrise"
        via_format = _format_prompt_with_metadata(cosmos3_format_pipeline, prompt)
        via_apply = cosmos3_format_pipeline._apply_metadata_templates(
            prompt,
            height=HEIGHT,
            width=WIDTH,
            num_frames=189,
            frame_rate=FRAME_RATE,
            duration_template=COSMOS3_DURATION_TEMPLATE,
            resolution_template=COSMOS3_DEFAULT_RESOLUTION_TEMPLATE,
        )
        assert via_format == via_apply

    def test_templates_disabled_returns_prompt_only(self, cosmos3_format_pipeline):
        result = _format_prompt_with_metadata(
            cosmos3_format_pipeline,
            "Plain prompt",
            duration_template=None,
            resolution_template=None,
        )
        assert result == "Plain prompt."

    def test_empty_prompt_with_templates(self, cosmos3_format_pipeline):
        result = _format_prompt_with_metadata(cosmos3_format_pipeline, "")
        assert "7.9 seconds long" in result
        assert "720x1280" in result

    def test_invalid_json_prefix_falls_back_to_append(self, cosmos3_format_pipeline):
        result = _format_prompt_with_metadata(cosmos3_format_pipeline, "{not valid json")
        assert result.startswith("{not valid json.")
        assert "720x1280" in result

    def test_json_array_falls_back_to_append(self, cosmos3_format_pipeline):
        result = _format_prompt_with_metadata(cosmos3_format_pipeline, '["a", "b"]')
        assert result.startswith('["a", "b"].')
        assert "720x1280" in result


class _CapturingTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def __init__(self):
        self.conversations = []

    def apply_chat_template(
        self,
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    ):
        assert tokenize is True
        assert add_generation_prompt is True
        assert return_dict is False
        self.conversations.append(conversations)
        return [1, 2, 3]

    def convert_tokens_to_ids(self, token):
        assert token == "<|vision_start|>"
        return 98


class TestTokenizePrompt:
    def test_system_prompt_included_when_enabled(self, cosmos3_format_pipeline):
        tokenizer = _CapturingTokenizer()
        cosmos3_format_pipeline.tokenizer = tokenizer
        cosmos3_format_pipeline.transformer = SimpleNamespace(device=torch.device("cpu"))

        input_ids, attention_mask = cosmos3_format_pipeline._tokenize_prompt(
            "Describe motion.",
            max_sequence_length=8,
            use_system_prompt=True,
            system_prompt="System text.",
        )

        assert tokenizer.conversations == [
            [
                {"role": "system", "content": "System text."},
                {"role": "user", "content": "Describe motion."},
            ]
        ]
        assert input_ids.tolist() == [[1, 2, 3, 99, 98, 0, 0, 0]]
        assert attention_mask.tolist() == [[1, 1, 1, 1, 1, 0, 0, 0]]

    def test_system_prompt_omitted_when_disabled(self, cosmos3_format_pipeline):
        tokenizer = _CapturingTokenizer()
        cosmos3_format_pipeline.tokenizer = tokenizer
        cosmos3_format_pipeline.transformer = SimpleNamespace(device=torch.device("cpu"))

        cosmos3_format_pipeline._tokenize_prompt(
            "Describe motion.",
            max_sequence_length=8,
            use_system_prompt=False,
            system_prompt="System text.",
        )

        assert tokenizer.conversations == [
            [{"role": "user", "content": "Describe motion."}]
        ]


class TestFormatPromptWithMetadataJson:
    def test_injects_metadata_fields(self, cosmos3_format_pipeline):
        prompt = json.dumps({"prompt": "A foundry pour", "subjects": []})
        result = _format_prompt_with_metadata(cosmos3_format_pipeline, prompt)
        data = json.loads(result)
        assert data["prompt"] == "A foundry pour"
        assert data["subjects"] == []
        assert data["duration"] == "7.9s"
        assert data["fps"] == 24
        assert data["resolution"] == {"W": 1280, "H": 720}
        assert data["aspect_ratio"] == "9,16"

    def test_overwrites_existing_metadata_fields(self, cosmos3_format_pipeline):
        prompt = json.dumps(
            {
                "prompt": "test",
                "duration": "5s",
                "fps": 30,
                "resolution": {"W": 640, "H": 480},
                "aspect_ratio": "3,4",
            }
        )
        data = json.loads(_format_prompt_with_metadata(cosmos3_format_pipeline, prompt))
        assert data["duration"] == "7.9s"
        assert data["fps"] == 24
        assert data["resolution"] == {"W": 1280, "H": 720}
        assert data["aspect_ratio"] == "9,16"

    def test_single_frame_skips_duration_by_default(self, cosmos3_format_pipeline):
        prompt = json.dumps({"prompt": "still life"})
        data = json.loads(
            _format_prompt_with_metadata(
                cosmos3_format_pipeline,
                prompt,
                num_frames=1,
                resolution_template=COSMOS3_IMAGE_RESOLUTION_TEMPLATE,
            )
        )
        assert "duration" not in data
        assert data["resolution"] == {"W": 1280, "H": 720}

    def test_single_frame_duration_when_forced(self, cosmos3_format_pipeline):
        prompt = json.dumps({"prompt": "still life"})
        data = json.loads(
            _format_prompt_with_metadata(
                cosmos3_format_pipeline,
                prompt,
                num_frames=1,
                force_duration_template=True,
            )
        )
        assert data["duration"] == "0.0s"

    def test_non_integer_fps_preserved(self, cosmos3_format_pipeline):
        prompt = json.dumps({"prompt": "test"})
        data = json.loads(
            _format_prompt_with_metadata(cosmos3_format_pipeline, prompt, frame_rate=23.976)
        )
        assert data["fps"] == 23.976

    def test_resolution_only_when_duration_template_disabled(self, cosmos3_format_pipeline):
        prompt = json.dumps({"prompt": "test"})
        data = json.loads(
            _format_prompt_with_metadata(
                cosmos3_format_pipeline,
                prompt,
                duration_template=None,
                resolution_template=COSMOS3_DEFAULT_RESOLUTION_TEMPLATE,
            )
        )
        assert "duration" not in data
        assert "fps" not in data
        assert data["resolution"] == {"W": 1280, "H": 720}


@pytest.fixture(scope="class")
def cosmos3_pipeline():
    checkpoint = _require_checkpoint()
    pipeline = _load_pipeline(checkpoint)
    yield pipeline
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.integration
@pytest.mark.cosmos3_t2v
@pytest.mark.high_cuda_memory
class TestCosmos3T2V:
    def test_t2v_smoke(self, cosmos3_pipeline):
        result = _run_forward(cosmos3_pipeline, image=None, num_frames=NUM_FRAMES)
        _assert_valid_video(result.video, num_frames=NUM_FRAMES)
        assert result.frame_rate == FRAME_RATE
        _assert_default_video_scheduler_config(cosmos3_pipeline)


@pytest.mark.integration
@pytest.mark.cosmos3_i2v
@pytest.mark.high_cuda_memory
class TestCosmos3I2V:
    def test_i2v_smoke(self, cosmos3_pipeline):
        image = _make_test_image()
        result = _run_forward(cosmos3_pipeline, image=image, num_frames=NUM_FRAMES)
        _assert_valid_video(result.video, num_frames=NUM_FRAMES)
        assert result.frame_rate == FRAME_RATE
        _assert_default_video_scheduler_config(cosmos3_pipeline)


class TestCosmos3V2VExtraParams:
    def test_condition_defaults_are_declared(self):
        assert COSMOS3_EXTRA_SPECS["condition_frame_indexes_vision"].default == list(
            COSMOS3_DEFAULT_CONDITION_FRAME_INDEXES_VISION
        )
        assert (
            COSMOS3_EXTRA_SPECS["condition_video_keep"].default
            == COSMOS3_DEFAULT_CONDITION_VIDEO_KEEP
        )

    def test_flow_shift_default_is_request_optional(self):
        spec = COSMOS3_EXTRA_SPECS["flow_shift"]
        assert spec.type == "float"
        assert spec.default is None

    def test_video_spec_declares_path_or_list_input(self):
        spec = COSMOS3_EXTRA_SPECS["video"]
        assert spec.type == "path_or_list"
        assert spec.default is None


class TestCosmos3V2VConditioningParams:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, (0, 1)),
            (0, (0,)),
            ([0, 2], (0, 2)),
            ((1, 3), (1, 3)),
            ("0, 2", (0, 2)),
        ],
    )
    def test_normalize_condition_frame_indexes_vision(self, value, expected):
        assert _normalize_condition_frame_indexes_vision(value) == expected

    @pytest.mark.parametrize("value", [[], "", [-1], "0, -1", [0, -2]])
    def test_invalid_condition_frame_indexes_vision_raise(self, value):
        with pytest.raises(ValueError):
            _normalize_condition_frame_indexes_vision(value)

    @pytest.mark.parametrize(
        "indexes,expected",
        [
            ((0,), 1),
            ((0, 1), 5),
            ((2,), 9),
        ],
    )
    def test_condition_pixel_frame_count(self, indexes, expected):
        assert _condition_pixel_frame_count(indexes, temporal_compression=4) == expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, "first"),
            ("first", "first"),
            ("FIRST", "first"),
            (" last ", "last"),
        ],
    )
    def test_normalize_condition_video_keep(self, value, expected):
        assert _normalize_condition_video_keep(value) == expected

    def test_invalid_condition_video_keep_raises(self):
        with pytest.raises(ValueError, match="first or last"):
            _normalize_condition_video_keep("middle")


@pytest.mark.integration
@pytest.mark.cosmos3_v2v
@pytest.mark.high_cuda_memory
class TestCosmos3V2V:
    def test_v2v_smoke(self, cosmos3_pipeline):
        video = _make_test_video(NUM_FRAMES)
        result = _run_forward(
            cosmos3_pipeline,
            image=None,
            video=video,
            num_frames=NUM_FRAMES,
            condition_frame_indexes_vision=[0, 1],
            condition_video_keep="first",
        )
        _assert_valid_video(result.video, num_frames=NUM_FRAMES)
        assert result.frame_rate == FRAME_RATE
        _assert_scheduler_config(
            cosmos3_pipeline,
            flow_shift=10.0,
            use_karras_sigmas=False,
        )

    def test_v2v_flow_shift_override_request_path(self):
        pipeline = Cosmos3OmniMoTPipeline.__new__(Cosmos3OmniMoTPipeline)
        pipeline.transformer = SimpleNamespace(device=torch.device("cpu"))
        pipeline.action_gen = False
        pipeline.audio_gen = False
        calls = []
        token_calls = []

        class StopAfterTokenize(Exception):
            pass

        def fake_set_flow_shift(target, *, use_karras_sigmas=None):
            calls.append((target, use_karras_sigmas))

        def fake_tokenize_prompt(text, max_sequence_length, use_system_prompt, system_prompt=None):
            token_calls.append((text, max_sequence_length, use_system_prompt, system_prompt))
            raise StopAfterTokenize

        pipeline._set_flow_shift = fake_set_flow_shift
        pipeline._tokenize_prompt = fake_tokenize_prompt

        with pytest.raises(StopAfterTokenize):
            pipeline.forward(
                prompt="continue",
                video=_make_test_video(5, width=16, height=16),
                height=16,
                width=16,
                num_frames=5,
                num_inference_steps=1,
                guidance_scale=1.0,
                seed=1,
                max_sequence_length=8,
                frame_rate=8.0,
                use_duration_template=False,
                use_resolution_template=False,
                use_system_prompt=None,
                use_guardrails=False,
                flow_shift=7.0,
            )

        assert calls == [(7.0, False)]
        assert token_calls[0][2] is True
        assert token_calls[0][3] == COSMOS3_DEFAULT_SYSTEM_PROMPT

    def test_image_and_video_rejected(self, cosmos3_pipeline):
        with pytest.raises(ValueError, match="not both image and video"):
            _run_forward(
                cosmos3_pipeline,
                image=_make_test_image(),
                video=_make_test_video(5),
            )

    def test_t2i_and_video_rejected(self, cosmos3_pipeline):
        with pytest.raises(ValueError, match="supported only for video outputs"):
            _run_forward(
                cosmos3_pipeline,
                image=None,
                video=_make_test_video(5),
                output_type="image",
                height=T2I_HEIGHT,
                width=T2I_WIDTH,
            )


@pytest.mark.integration
@pytest.mark.cosmos3_t2i
@pytest.mark.high_cuda_memory
class TestCosmos3T2I:
    def test_t2i_smoke(self, cosmos3_pipeline):
        result = _run_forward(
            cosmos3_pipeline,
            image=None,
            output_type="image",
            height=T2I_HEIGHT,
            width=T2I_WIDTH,
            guidance_scale=T2I_GUIDANCE_SCALE,
        )
        assert result.video is None
        _assert_valid_image(result.image, height=T2I_HEIGHT, width=T2I_WIDTH)
        _assert_scheduler_config(
            cosmos3_pipeline,
            flow_shift=COSMOS3_T2I_PARAMS["flow_shift"],
            use_karras_sigmas=cosmos3_pipeline._base_scheduler_use_karras_sigmas,
        )


@pytest.mark.integration
@pytest.mark.cosmos3_audio
@pytest.mark.high_cuda_memory
class TestCosmos3Audio:
    def test_audio_smoke(self, cosmos3_pipeline):
        _require_audio_pipeline(cosmos3_pipeline)
        result = _run_forward(cosmos3_pipeline, enable_audio=True)
        _assert_valid_video(result.video, num_frames=NUM_FRAMES)
        assert result.frame_rate == FRAME_RATE
        _assert_valid_audio(result.audio, result.audio_sample_rate)


@pytest.mark.integration
@pytest.mark.cosmos3_action
@pytest.mark.high_cuda_memory
class TestCosmos3Action:
    ACTION_HEIGHT = 480
    ACTION_WIDTH = 832
    ACTION_FRAMES = COSMOS3_ACTION_PARAMS["num_frames"]
    ACTION_CHUNK = COSMOS3_ACTION_PARAMS["action_chunk_size"]
    RAW_ACTION_DIM = 10

    def test_policy_smoke(self, cosmos3_pipeline):
        _require_action_pipeline(cosmos3_pipeline)
        image = _make_test_image().resize((self.ACTION_WIDTH, self.ACTION_HEIGHT))
        result = _run_forward(
            cosmos3_pipeline,
            image=image,
            height=self.ACTION_HEIGHT,
            width=self.ACTION_WIDTH,
            num_frames=self.ACTION_FRAMES,
            guidance_scale=COSMOS3_ACTION_PARAMS["guidance_scale"],
            action_mode="policy",
            domain_name="bridge_orig_lerobot",
            raw_action_dim=self.RAW_ACTION_DIM,
            action_chunk_size=self.ACTION_CHUNK,
        )
        _assert_valid_video(
            result.video,
            num_frames=self.ACTION_FRAMES,
            height=self.ACTION_HEIGHT,
            width=self.ACTION_WIDTH,
        )
        _assert_valid_action(
            result.action,
            raw_action_dim=self.RAW_ACTION_DIM,
            chunk_size=self.ACTION_CHUNK,
        )
        assert result.action_mode == "policy"
        assert result.domain_id == 7
        _assert_scheduler_config(
            cosmos3_pipeline,
            flow_shift=COSMOS3_ACTION_PARAMS["flow_shift"],
            use_karras_sigmas=cosmos3_pipeline._base_scheduler_use_karras_sigmas,
        )

    def test_forward_dynamics_smoke(self, cosmos3_pipeline):
        _require_action_pipeline(cosmos3_pipeline)
        image = _make_test_image().resize((self.ACTION_WIDTH, self.ACTION_HEIGHT))
        action_traj = [[0.1] * self.RAW_ACTION_DIM for _ in range(self.ACTION_CHUNK)]
        result = _run_forward(
            cosmos3_pipeline,
            image=image,
            height=self.ACTION_HEIGHT,
            width=self.ACTION_WIDTH,
            num_frames=self.ACTION_FRAMES,
            guidance_scale=COSMOS3_ACTION_PARAMS["guidance_scale"],
            action_mode="forward_dynamics",
            domain_name="bridge_orig_lerobot",
            action=action_traj,
            action_chunk_size=self.ACTION_CHUNK,
        )
        _assert_valid_video(
            result.video,
            num_frames=self.ACTION_FRAMES,
            height=self.ACTION_HEIGHT,
            width=self.ACTION_WIDTH,
        )
        _assert_valid_action(
            result.action,
            raw_action_dim=self.RAW_ACTION_DIM,
            chunk_size=self.ACTION_CHUNK,
        )
        _assert_scheduler_config(
            cosmos3_pipeline,
            flow_shift=COSMOS3_ACTION_PARAMS["flow_shift"],
            use_karras_sigmas=cosmos3_pipeline._base_scheduler_use_karras_sigmas,
        )

    def test_inverse_dynamics_smoke(self, cosmos3_pipeline):
        _require_action_pipeline(cosmos3_pipeline)
        image = _make_test_image().resize((self.ACTION_WIDTH, self.ACTION_HEIGHT))
        video = [image.copy() for _ in range(NUM_FRAMES)]
        result = _run_forward(
            cosmos3_pipeline,
            image=None,
            height=self.ACTION_HEIGHT,
            width=self.ACTION_WIDTH,
            num_frames=NUM_FRAMES,
            guidance_scale=COSMOS3_ACTION_PARAMS["guidance_scale"],
            action_mode="inverse_dynamics",
            domain_name="bridge_orig_lerobot",
            raw_action_dim=self.RAW_ACTION_DIM,
            action_chunk_size=NUM_FRAMES,
            video=video,
        )
        _assert_valid_video(
            result.video,
            num_frames=NUM_FRAMES,
            height=self.ACTION_HEIGHT,
            width=self.ACTION_WIDTH,
        )
        _assert_valid_action(
            result.action,
            raw_action_dim=self.RAW_ACTION_DIM,
            chunk_size=NUM_FRAMES,
        )
        _assert_scheduler_config(
            cosmos3_pipeline,
            flow_shift=COSMOS3_ACTION_PARAMS["flow_shift"],
            use_karras_sigmas=cosmos3_pipeline._base_scheduler_use_karras_sigmas,
        )

    def test_action_and_audio_rejected(self, cosmos3_pipeline):
        _require_action_pipeline(cosmos3_pipeline)
        with pytest.raises(ValueError, match="joint action and audio"):
            _run_forward(
                cosmos3_pipeline,
                image=_make_test_image(),
                action_mode="policy",
                domain_name="bridge_orig_lerobot",
                raw_action_dim=self.RAW_ACTION_DIM,
                enable_audio=True,
            )

    def test_action_and_t2i_rejected(self, cosmos3_pipeline):
        _require_action_pipeline(cosmos3_pipeline)
        with pytest.raises(ValueError, match="output_type='image'"):
            _run_forward(
                cosmos3_pipeline,
                output_type="image",
                action_mode="policy",
                domain_name="bridge_orig_lerobot",
                raw_action_dim=self.RAW_ACTION_DIM,
            )


@pytest.mark.integration
@pytest.mark.cosmos3_t2v
@pytest.mark.high_cuda_memory
class TestCosmos3PromptTemplates:
    @pytest.mark.parametrize(
        "use_duration_template,use_resolution_template,use_system_prompt",
        [
            (True, True, True),
            (False, False, False),
            (False, False, True),
            (False, False, None),
        ],
        ids=["all-on", "all-off", "system-prompt-only", "system-prompt-default"],
    )
    def test_template_variants(
        self,
        cosmos3_pipeline,
        use_duration_template,
        use_resolution_template,
        use_system_prompt,
    ):
        result = _run_forward(
            cosmos3_pipeline,
            use_duration_template=use_duration_template,
            use_resolution_template=use_resolution_template,
            use_system_prompt=use_system_prompt,
        )
        _assert_valid_video(result.video, num_frames=NUM_FRAMES)


@pytest.mark.integration
@pytest.mark.cosmos3_t2v
@pytest.mark.high_cuda_memory
class TestCosmos3NegativePrompt:
    def test_default_negative_prompt(self, cosmos3_pipeline):
        result = _run_forward(cosmos3_pipeline, negative_prompt=None)
        _assert_valid_video(result.video, num_frames=NUM_FRAMES)


@pytest.mark.integration
@pytest.mark.cosmos3_t2v
class TestCosmos3BatchRejected:
    def test_batch_prompt_raises(self, cosmos3_pipeline):
        with pytest.raises(ValueError, match="Batch generation is not supported"):
            cosmos3_pipeline.forward(
                prompt=["first prompt", "second prompt"],
                height=HEIGHT,
                width=WIDTH,
                num_frames=NUM_FRAMES,
                num_inference_steps=NUM_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                seed=SEED,
                frame_rate=FRAME_RATE,
                use_guardrails=False,
            )


@pytest.mark.integration
@pytest.mark.cosmos3_t2v
@pytest.mark.high_cuda_memory
class TestCosmos3FP8Load:
    def test_fp8_load_and_t2v(self):
        checkpoint = _require_checkpoint()
        pipeline = _load_pipeline(checkpoint, quant_config=COSMOS3_FP8_QUANT_CONFIG)
        try:
            assert pipeline.transformer.model_config.quant_config.quant_algo is not None
            result = _run_forward(pipeline, image=None, num_frames=NUM_FRAMES)
            _assert_valid_video(result.video, num_frames=NUM_FRAMES)
            assert result.frame_rate == FRAME_RATE
        finally:
            del pipeline
            gc.collect()
            torch.cuda.empty_cache()
