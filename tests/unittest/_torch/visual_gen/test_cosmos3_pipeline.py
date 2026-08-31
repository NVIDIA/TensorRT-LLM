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
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace

os.environ["TLLM_DISABLE_MPI"] = "1"

import PIL.Image
import pytest
import torch

import tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_cosmos3 as pipe_mod
from tensorrt_llm._torch.visual_gen.models.cosmos3.defaults import (
    COSMOS3_ACTION_PARAMS,
    COSMOS3_DEFAULT_CONDITION_VIDEO_KEEP,
    COSMOS3_DEFAULT_CONDITION_VIDEO_LATENT_INDEXES,
    COSMOS3_EXTRA_SPECS,
    COSMOS3_T2I_PARAMS,
    _normalize_condition_video_keep,
)
from tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_cosmos3 import (
    COSMOS3_DEFAULT_RESOLUTION_TEMPLATE,
    COSMOS3_DEFAULT_SYSTEM_PROMPT,
    COSMOS3_DURATION_TEMPLATE,
    COSMOS3_IMAGE_RESOLUTION_TEMPLATE,
    Cosmos3OmniMoTPipeline,
    _condition_pixel_frame_count,
    _load_reference_image,
    _normalize_condition_video_latent_indexes,
)
from tensorrt_llm._torch.visual_gen.models.cosmos3.sampling import Cosmos3SamplingPolicy
from tensorrt_llm._torch.visual_gen.models.cosmos3.transformer_cosmos3 import QWEN3_RECIPE
from tensorrt_llm._torch.visual_gen.models.wan.vae_loader import TRTLLM_USE_DIFFUSER_VAE_ENV
from tensorrt_llm._torch.visual_gen.models.wan.wan_vae import WanVAE
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm.visual_gen.args import TorchCompileConfig, VisualGenArgs

pytestmark = [pytest.mark.cosmos3, pytest.mark.usefixtures("disable_cosmos3_guardrails")]


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
    frame_rate=FRAME_RATE,
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
        frame_rate=frame_rate,
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


def _base_use_karras_sigmas(pipeline) -> bool:
    """Karras setting the checkpoint shipped with — what ``None`` restores."""
    return bool(getattr(pipeline.sampling.unipc_base_config, "use_karras_sigmas", False))


def _assert_scheduler_config(
    pipeline,
    *,
    flow_shift: float,
    use_karras_sigmas: bool | None,
):
    assert float(getattr(pipeline.scheduler.config, "flow_shift")) == pytest.approx(
        float(flow_shift)
    )
    assert _scheduler_use_karras_sigmas(pipeline.scheduler) == use_karras_sigmas


def _assert_default_video_scheduler_config(pipeline):
    _assert_scheduler_config(
        pipeline,
        flow_shift=pipeline.sampling.checkpoint_flow_shift,
        use_karras_sigmas=_base_use_karras_sigmas(pipeline),
    )


def _make_test_image() -> PIL.Image.Image:
    image_path = os.environ.get("COSMOS3_TEST_IMAGE")
    if image_path and os.path.exists(image_path):
        return PIL.Image.open(image_path).convert("RGB")
    return PIL.Image.new("RGB", (WIDTH, HEIGHT), color=(64, 128, 192))


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

        assert tokenizer.conversations == [[{"role": "user", "content": "Describe motion."}]]


class TestFormatPromptWithMetadataJson:
    def test_injects_metadata_fields(self, cosmos3_format_pipeline):
        prompt = json.dumps({"prompt": "A foundry pour", "subjects": []})
        result = _format_prompt_with_metadata(cosmos3_format_pipeline, prompt)
        data = json.loads(result)
        assert data["prompt"] == "A foundry pour"
        assert data["subjects"] == []
        # Reference semantics: integer-truncated seconds, float fps, H before W,
        # and the aspect-ratio *bucket* rather than the exact reduced ratio.
        assert data["duration"] == "7s"
        assert data["fps"] == 24.0
        assert data["resolution"] == {"H": 720, "W": 1280}
        assert data["aspect_ratio"] == "16,9"
        assert '"resolution": {"H": 720, "W": 1280}' in result

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
        assert data["duration"] == "7s"
        assert data["fps"] == 24.0
        assert data["resolution"] == {"H": 720, "W": 1280}
        assert data["aspect_ratio"] == "16,9"

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
        assert data["duration"] == "0s"

    def test_still_drops_stale_duration_and_fps(self, cosmos3_format_pipeline):
        """A caller's JSON may already declare a duration; a still must not keep it."""
        prompt = json.dumps({"prompt": "still life", "duration": "7s", "fps": 24.0})
        data = json.loads(
            _format_prompt_with_metadata(
                cosmos3_format_pipeline,
                prompt,
                num_frames=1,
                resolution_template=COSMOS3_IMAGE_RESOLUTION_TEMPLATE,
            )
        )
        assert "duration" not in data
        assert "fps" not in data

    def test_non_ascii_is_escaped(self, cosmos3_format_pipeline):
        """The reference serializes with the json default (``ensure_ascii=True``)."""
        result = _format_prompt_with_metadata(
            cosmos3_format_pipeline, json.dumps({"prompt": "moiré — artifacts"})
        )
        assert "\\u00e9" in result and "\\u2014" in result
        assert "é" not in result and "—" not in result

    @pytest.mark.parametrize(
        "height,width,bucket",
        [
            (480, 832, "16,9"),
            (832, 480, "9,16"),
            (640, 640, "1,1"),
            (544, 736, "4,3"),
            (736, 544, "3,4"),
            (720, 1280, "16,9"),
            (1024, 1024, "1,1"),
        ],
    )
    def test_aspect_ratio_maps_to_reference_bucket(
        self, cosmos3_format_pipeline, height, width, bucket
    ):
        data = json.loads(
            _format_prompt_with_metadata(
                cosmos3_format_pipeline,
                json.dumps({"prompt": "test"}),
                height=height,
                width=width,
            )
        )
        assert data["aspect_ratio"] == bucket

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


class TestNegativePromptMetadata:
    """The negative prompt takes the sentence-append path even when it is JSON.

    cosmos-framework applies its plain-text formatter to the negative prompt
    unconditionally and reserves JSON field injection for the positive prompt, so
    a JSON negative prompt must keep its serialized form and gain the sentences
    after it -- not grow ``duration``/``fps``/``resolution`` keys inside it.
    """

    NEGATIVE = json.dumps({"subjects": [{"description": "Blurry, poorly defined subjects."}]})

    def _negative(self, pipeline, **kwargs):
        """Format a negative prompt the way ``forward`` does."""
        return pipeline._apply_metadata_templates(
            self.NEGATIVE,
            height=HEIGHT,
            width=WIDTH,
            num_frames=189,
            frame_rate=FRAME_RATE,
            duration_template=COSMOS3_DURATION_TEMPLATE,
            resolution_template=COSMOS3_DEFAULT_RESOLUTION_TEMPLATE,
            **kwargs,
        )

    def test_json_negative_keeps_object_and_appends_sentences(self, cosmos3_format_pipeline):
        result = self._negative(cosmos3_format_pipeline)
        assert result.startswith(self.NEGATIVE.rstrip("."))
        assert result.endswith("This video is of 720x1280 resolution.")
        assert "7.9 seconds long" in result

    def test_json_negative_gains_no_injected_fields(self, cosmos3_format_pipeline):
        result = self._negative(cosmos3_format_pipeline)
        # The metadata must live outside the object, so the result stops being
        # parseable JSON and the object itself is untouched.
        with pytest.raises(json.JSONDecodeError):
            json.loads(result)
        for field in ("duration", "fps", "resolution", "aspect_ratio"):
            assert f'"{field}"' not in result

    def test_matches_reference_sentence_append(self, cosmos3_format_pipeline):
        """Byte-for-byte against cosmos-framework's ``_format_prompt_with_template``."""
        expected = (
            self.NEGATIVE.strip().rstrip(".")
            + ". "
            + COSMOS3_DURATION_TEMPLATE.format(duration=189 / FRAME_RATE, fps=FRAME_RATE)
        )
        expected = (
            expected.strip().rstrip(".")
            + ". "
            + COSMOS3_DEFAULT_RESOLUTION_TEMPLATE.format(height=HEIGHT, width=WIDTH)
        )
        assert self._negative(cosmos3_format_pipeline) == expected.lstrip(".").strip()

    def test_positive_json_still_injects_fields(self, cosmos3_format_pipeline):
        """The positive branch keeps field injection -- the two paths differ by design."""
        data = json.loads(_format_prompt_with_metadata(cosmos3_format_pipeline, self.NEGATIVE))
        assert data["resolution"] == {"W": 1280, "H": 720}
        assert self._negative(cosmos3_format_pipeline) != _format_prompt_with_metadata(
            cosmos3_format_pipeline, self.NEGATIVE
        )


class TestDefaultNegativePrompt:
    """Video modes inherit the reference's default negative prompt; image modes do not.

    cosmos-framework wires ``negative_prompt_file: neg_prompts.json`` into
    ``defaults/{text2video,image2video,video2video,audio_image2video}`` and leaves it
    unset for ``text2image``/``image2image``.
    """

    def test_video_default_serializes_like_the_reference(self):
        from tensorrt_llm._torch.visual_gen.models.cosmos3.negative_prompt import (
            COSMOS3_VIDEO_NEGATIVE_PROMPT,
        )
        from tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_cosmos3 import (
            default_video_negative_prompt,
        )

        # The reference loads it as json.dumps(json.loads(...)); ensure_ascii and key
        # order both matter, so round-tripping must be a no-op.
        text = default_video_negative_prompt()
        assert text == json.dumps(COSMOS3_VIDEO_NEGATIVE_PROMPT)
        assert json.dumps(json.loads(text)) == text
        assert "\\u2014" in text, "non-ASCII must be escaped, as the reference emits it"

    def test_video_default_is_a_json_object_with_expected_shape(self):
        from tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_cosmos3 import (
            default_video_negative_prompt,
        )

        data = json.loads(default_video_negative_prompt())
        assert isinstance(data, dict)
        for field in ("subjects", "background_setting", "cinematography"):
            assert field in data

    def test_image_default_is_empty(self):
        from tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_cosmos3 import (
            COSMOS3_DEFAULT_NEGATIVE_PROMPT,
        )

        assert COSMOS3_DEFAULT_NEGATIVE_PROMPT == ""

    def test_default_is_cached(self):
        from tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_cosmos3 import (
            default_video_negative_prompt,
        )

        assert default_video_negative_prompt() is default_video_negative_prompt()

    @pytest.mark.parametrize(
        "output_type,expects_video_default",
        [("video", True), ("image", False)],
    )
    def test_resolution_is_keyed_on_output_kind(self, output_type, expects_video_default):
        from tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_cosmos3 import (
            default_negative_prompt,
            default_video_negative_prompt,
        )

        resolved = default_negative_prompt(output_type)
        if expects_video_default:
            assert resolved == default_video_negative_prompt()
        else:
            assert resolved == ""


@pytest.fixture(scope="class")
def cosmos3_pipeline() -> Generator[Cosmos3OmniMoTPipeline, None, None]:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.delenv(TRTLLM_USE_DIFFUSER_VAE_ENV, raising=False)
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
    def test_native_wan_vae_is_default(self, cosmos3_pipeline: Cosmos3OmniMoTPipeline) -> None:
        assert isinstance(cosmos3_pipeline.vae, WanVAE)

        config = cosmos3_pipeline.vae.config
        assert len(config.latents_mean) == config.z_dim
        assert len(config.latents_std) == config.z_dim
        assert config.scale_factor_spatial == cosmos3_pipeline.vae_scale_factor_spatial
        assert config.scale_factor_temporal == cosmos3_pipeline.vae_scale_factor_temporal

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
        assert COSMOS3_EXTRA_SPECS["condition_video_latent_indexes"].default == list(
            COSMOS3_DEFAULT_CONDITION_VIDEO_LATENT_INDEXES
        )
        assert (
            COSMOS3_EXTRA_SPECS["condition_video_keep"].default
            == COSMOS3_DEFAULT_CONDITION_VIDEO_KEEP
        )

    def test_flow_shift_default_is_request_optional(self):
        spec = COSMOS3_EXTRA_SPECS["flow_shift"]
        assert spec.type == "float"
        assert spec.default is None


class TestCosmos3V2VConditioningParams:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, (0, 1)),
            ([0, 2], (0, 2)),
            ((1, 3), (1, 3)),
        ],
    )
    def test_normalize_condition_video_latent_indexes(self, value, expected):
        assert _normalize_condition_video_latent_indexes(value) == expected

    @pytest.mark.parametrize("value", [[], [-1], [0, -2]])
    def test_invalid_condition_video_latent_indexes_raise(self, value):
        with pytest.raises(ValueError):
            _normalize_condition_video_latent_indexes(value)

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


class TestReferenceImageLoad:
    """The worker's image load is the acceptance check for an I2V reference.

    The serve boundary only routes on the container signature, so unreadable
    content has to surface here as a client error (``ValueError`` → 400) and
    not as a server fault.
    """

    def test_truncated_image_is_a_client_error(self, tmp_path):
        # Incompressible content, so half the file is genuinely half the image.
        noise = PIL.Image.frombytes("RGB", (64, 64), os.urandom(64 * 64 * 3))
        whole = tmp_path / "whole.png"
        noise.save(whole, format="PNG")
        data = whole.read_bytes()
        path = tmp_path / "truncated.png"
        path.write_bytes(data[: len(data) // 2])

        with pytest.raises(ValueError, match="could not be decoded"):
            _load_reference_image(str(path))

    def test_unidentifiable_content_is_a_client_error(self, tmp_path):
        path = tmp_path / "notreally.png"
        path.write_bytes(b"not an image at all")
        with pytest.raises(ValueError, match="could not be decoded"):
            _load_reference_image(str(path))

    def test_missing_file_is_a_client_error(self, tmp_path):
        with pytest.raises(ValueError, match="could not be decoded"):
            _load_reference_image(str(tmp_path / "nope.png"))

    def test_valid_image_loads(self, tmp_path):
        path = tmp_path / "ok.png"
        PIL.Image.new("RGB", (8, 8), (1, 2, 3)).save(path, format="PNG")
        assert _load_reference_image(str(path)).size == (8, 8)


_V2V_FIXTURE_MP4 = Path(__file__).parent / "test_data" / "cosmos3_v2v_ref_9f_bframes.mp4"


@pytest.mark.integration
@pytest.mark.cosmos3_v2v
@pytest.mark.high_cuda_memory
class TestCosmos3V2V:
    def test_v2v_smoke(self, cosmos3_pipeline):
        """The production V2V path end to end: encoded MP4 bytes (the only
        ``video`` form) — each rank demuxes from memory, NVDEC-decodes the
        conditioning window, resizes to the output resolution, VAE-encodes,
        and generates with the V2V scheduler policy."""
        result = _run_forward(
            cosmos3_pipeline,
            image=None,
            video=_V2V_FIXTURE_MP4.read_bytes(),
            num_frames=NUM_FRAMES,
            condition_video_latent_indexes=[0, 1],
        )
        _assert_valid_video(result.video, num_frames=NUM_FRAMES)
        assert result.frame_rate == FRAME_RATE
        _assert_scheduler_config(
            cosmos3_pipeline,
            flow_shift=10.0,
            use_karras_sigmas=False,
        )

    def test_v2v_keep_last_smoke(self, cosmos3_pipeline):
        """condition_video_keep="last" pins the tail of the input, not the head.

        Drives the real bytes path end to end: ``keep`` is applied inside the
        worker's NVDEC decode (ring buffer over the demuxed stream), exactly
        as a request flows. The fixture's red channel encodes the frame index
        (R = 20 + 25*i over 9 frames); with keep="last" the conditioning
        window is frames 4-8, so output frame 0 must be a pinned VAE
        round-trip of fixture frame 4 (R=120) — not fixture frame 0 (R=20).
        """
        result = _run_forward(
            cosmos3_pipeline,
            image=None,
            video=_V2V_FIXTURE_MP4.read_bytes(),
            num_frames=NUM_FRAMES,
            condition_video_latent_indexes=[0, 1],
            condition_video_keep="last",
        )
        _assert_valid_video(result.video, num_frames=NUM_FRAMES)
        red_mean = result.video[0, 0, :, :, 0].float().mean().item()
        assert red_mean > 70, (
            f"keep='last' must condition on the tail frames (R=120..220); "
            f"output frame-0 red mean {red_mean:.1f} matches the head (R=20) instead"
        )
        _assert_scheduler_config(
            cosmos3_pipeline,
            flow_shift=10.0,
            use_karras_sigmas=False,
        )

    def test_v2v_flow_shift_override_request_path(self):
        pipeline = Cosmos3OmniMoTPipeline.__new__(Cosmos3OmniMoTPipeline)
        pipeline.transformer = SimpleNamespace(device=torch.device("cpu"))
        pipeline.audio_gen = False
        # Bypassing __init__ means the generation-default family has to be set
        # here; forward() reads its per-mode table through it.
        pipeline.family = QWEN3_RECIPE.name
        calls = []
        token_calls = []

        class StopAfterTokenize(Exception):
            pass

        class FakeSampling:
            """Minimal Cosmos3SamplingPolicy stand-in recording flow-shift calls.

            forward() consults the policy for request validation and the
            distilled guard before it ever reaches the flow-shift block, so a
            stub carrying only ``set_flow_shift`` never gets there.
            """

            is_distilled = False
            checkpoint_flow_shift = 1.0

            def validate_request(self, num_inference_steps, guidance_scale):
                return None

            def generation_default_overrides(self):
                return {}

            def set_flow_shift(self, scheduler, target, *, use_karras_sigmas=None):
                calls.append((target, use_karras_sigmas))
                return scheduler

        def fake_tokenize_prompt(text, max_sequence_length, use_system_prompt, system_prompt=None):
            token_calls.append((text, max_sequence_length, use_system_prompt, system_prompt))
            raise StopAfterTokenize

        pipeline.scheduler = SimpleNamespace(config=SimpleNamespace(flow_shift=1.0))
        pipeline.sampling = FakeSampling()
        pipeline._tokenize_prompt = fake_tokenize_prompt

        with pytest.raises(StopAfterTokenize):
            pipeline.forward(
                prompt="continue",
                video=_V2V_FIXTURE_MP4.read_bytes(),
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

    def test_v2v_rebuilds_the_audio_scheduler_too(self):
        """Video and audio denoise in lockstep in one loop, so a V2V request
        must rebuild both. Rebuilding only the video scheduler leaves audio on
        the checkpoint's flow shift / Karras sigmas and the streams step on
        different schedules."""
        pipeline = Cosmos3OmniMoTPipeline.__new__(Cosmos3OmniMoTPipeline)
        pipeline.transformer = SimpleNamespace(device=torch.device("cpu"))
        pipeline.audio_gen = True
        # Bypassing __init__ means the generation-default family has to be set
        # here; forward() reads its per-mode table through it.
        pipeline.family = QWEN3_RECIPE.name
        rebuilt = []

        class StopAfterTokenize(Exception):
            pass

        class FakeSampling:
            is_distilled = False
            checkpoint_flow_shift = 1.0

            def validate_request(self, num_inference_steps, guidance_scale):
                return None

            def generation_default_overrides(self):
                return {}

            def set_flow_shift(self, scheduler, target, *, use_karras_sigmas=None):
                rebuilt.append((scheduler.name, target, use_karras_sigmas))
                return scheduler

        def fake_tokenize_prompt(text, max_sequence_length, use_system_prompt, system_prompt=None):
            raise StopAfterTokenize

        pipeline.scheduler = SimpleNamespace(name="video", config=SimpleNamespace(flow_shift=1.0))
        pipeline.audio_scheduler = SimpleNamespace(
            name="audio", config=SimpleNamespace(flow_shift=1.0)
        )
        pipeline.sampling = FakeSampling()
        pipeline._tokenize_prompt = fake_tokenize_prompt

        with pytest.raises(StopAfterTokenize):
            pipeline.forward(
                prompt="continue",
                video=_V2V_FIXTURE_MP4.read_bytes(),
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
                enable_audio=True,
            )

        assert rebuilt == [("video", 10.0, False), ("audio", 10.0, False)]

    def test_action_video_is_not_classified_as_v2v(self):
        """An action reference arrives as the same `video` bytes V2V uses, but
        it is an observation, not a clip to continue. Treating it as V2V forces
        the system prompt, so the same frame would tokenize differently
        depending on whether it was passed as an image or a one-frame clip."""
        pipeline = Cosmos3OmniMoTPipeline.__new__(Cosmos3OmniMoTPipeline)
        pipeline.transformer = SimpleNamespace(
            device=torch.device("cpu"), num_embodiment_domains=32
        )
        pipeline.audio_gen = False
        pipeline.action_gen = True
        pipeline.default_use_system_prompt = False
        pipeline.family = QWEN3_RECIPE.name
        token_calls = []

        class StopAfterTokenize(Exception):
            pass

        class FakeSampling:
            is_distilled = False
            checkpoint_flow_shift = 1.0

            def validate_request(self, num_inference_steps, guidance_scale):
                return None

            def generation_default_overrides(self):
                return {}

            def set_flow_shift(self, scheduler, target, *, use_karras_sigmas=None):
                return scheduler

        def fake_tokenize_prompt(text, max_sequence_length, use_system_prompt, system_prompt=None):
            token_calls.append(use_system_prompt)
            raise StopAfterTokenize

        pipeline.scheduler = SimpleNamespace(config=SimpleNamespace(flow_shift=1.0))
        pipeline.sampling = FakeSampling()
        pipeline._tokenize_prompt = fake_tokenize_prompt

        with pytest.raises(StopAfterTokenize):
            pipeline.forward(
                prompt="pick up the block",
                video=_V2V_FIXTURE_MP4.read_bytes(),
                num_frames=NUM_FRAMES,
                num_inference_steps=1,
                guidance_scale=1.0,
                seed=1,
                max_sequence_length=8,
                use_system_prompt=None,
                use_guardrails=False,
                action_mode="inverse_dynamics",
                domain_name="bridge_orig_lerobot",
                raw_action_dim=10,
                action_chunk_size=NUM_FRAMES - 1,
            )

        assert token_calls[0] is False

    def test_action_restores_the_checkpoint_flow_shift(self):
        """set_flow_shift is a no-op when both knobs are None, and the pipeline
        instance outlives the request. An action request that followed a V2V one
        would otherwise keep V2V's uniform sigmas instead of the checkpoint's."""
        pipeline = Cosmos3OmniMoTPipeline.__new__(Cosmos3OmniMoTPipeline)
        pipeline.transformer = SimpleNamespace(
            device=torch.device("cpu"), num_embodiment_domains=32
        )
        pipeline.audio_gen = False
        pipeline.action_gen = True
        pipeline.default_use_system_prompt = False
        pipeline.family = QWEN3_RECIPE.name
        calls = []

        class StopAfterTokenize(Exception):
            pass

        class FakeSampling:
            is_distilled = False
            checkpoint_flow_shift = 7.0

            def validate_request(self, num_inference_steps, guidance_scale):
                return None

            def generation_default_overrides(self):
                return {}

            def set_flow_shift(self, scheduler, target, *, use_karras_sigmas=None):
                calls.append((scheduler.name, target, use_karras_sigmas))
                return scheduler

        def fake_tokenize_prompt(text, max_sequence_length, use_system_prompt, system_prompt=None):
            raise StopAfterTokenize

        pipeline.scheduler = SimpleNamespace(name="video", config=SimpleNamespace(flow_shift=1.0))
        pipeline.action_scheduler = SimpleNamespace(
            name="action", config=SimpleNamespace(flow_shift=1.0)
        )
        pipeline.sampling = FakeSampling()
        pipeline._tokenize_prompt = fake_tokenize_prompt

        with pytest.raises(StopAfterTokenize):
            pipeline.forward(
                prompt="pick up the block",
                video=_V2V_FIXTURE_MP4.read_bytes(),
                num_frames=NUM_FRAMES,
                num_inference_steps=1,
                guidance_scale=1.0,
                seed=1,
                max_sequence_length=8,
                use_guardrails=False,
                action_mode="inverse_dynamics",
                domain_name="bridge_orig_lerobot",
                raw_action_dim=10,
                action_chunk_size=NUM_FRAMES - 1,
            )

        assert calls == [("video", 7.0, None), ("action", 7.0, None)]

    def test_scheduler_for_serves_every_stream(self):
        """Video, audio and action denoise in lockstep, so one resolved
        (shift, karras) configuration must yield a scheduler per stream --
        separate instances (schedulers mutate state on every .step()) built
        from that stream's own base."""
        pipeline = Cosmos3OmniMoTPipeline.__new__(Cosmos3OmniMoTPipeline)
        pipeline.family = QWEN3_RECIPE.name
        rebuilt = []

        class FakeSampling:
            checkpoint_flow_shift = 1.0

            def set_flow_shift(self, scheduler, target, *, use_karras_sigmas=None):
                rebuilt.append((scheduler.name, target, use_karras_sigmas))
                return scheduler

        pipeline.sampling = FakeSampling()
        pipeline.scheduler = SimpleNamespace(name="video")
        pipeline.audio_scheduler = SimpleNamespace(name="audio")
        pipeline.action_scheduler = SimpleNamespace(name="action")

        for stream in ("video", "audio", "action"):
            pipeline._scheduler_for(10.0, False, stream=stream)

        assert rebuilt == [
            ("video", 10.0, False),
            ("audio", 10.0, False),
            ("action", 10.0, False),
        ]

    def test_image_and_video_rejected(self, cosmos3_pipeline):
        with pytest.raises(ValueError, match="not both image and video"):
            _run_forward(
                cosmos3_pipeline,
                image=_make_test_image(),
                video=_V2V_FIXTURE_MP4.read_bytes(),
            )

    def test_t2i_and_video_rejected(self, cosmos3_pipeline):
        with pytest.raises(ValueError, match="supported only for video outputs"):
            _run_forward(
                cosmos3_pipeline,
                image=None,
                video=_V2V_FIXTURE_MP4.read_bytes(),
                output_type="image",
                height=T2I_HEIGHT,
                width=T2I_WIDTH,
            )


class TestCosmos3TransferRouting:
    def test_transfer_rejects_an_image_reference(self):
        """`_forward_transfer` takes no image, so a request carrying both used
        to have its image silently dropped. The sibling guards already reject
        transfer with image output and with audio; this one completes them."""
        from tensorrt_llm._torch.visual_gen.models.cosmos3.transfer import resolve_transfer_config

        pipeline = Cosmos3OmniMoTPipeline.__new__(Cosmos3OmniMoTPipeline)
        pipeline.transformer = SimpleNamespace(device=torch.device("cpu"))
        pipeline.action_gen = False
        # __new__ skips __init__, where the real pipeline resolves this.
        pipeline.family = QWEN3_RECIPE.name
        pipeline.audio_gen = False

        class FakeSampling:
            is_distilled = False
            checkpoint_flow_shift = 1.0

            def validate_request(self, num_inference_steps, guidance_scale):
                return None

            def generation_default_overrides(self):
                return {}

        pipeline.sampling = FakeSampling()
        pipeline._forward_transfer = lambda **kwargs: None
        # A precomputed control and no video: an existing guard already rejects
        # image+video, so this is the shape where the image used to reach
        # `_forward_transfer` and be discarded.
        cfg = resolve_transfer_config(
            {"edge": _V2V_FIXTURE_MP4.read_bytes()},
            SimpleNamespace(num_frames=93, guidance_scale=None),
            None,
        )

        with pytest.raises(ValueError, match="cannot be combined with an image reference"):
            pipeline.forward(
                prompt="bounce",
                image="frame.png",
                transfer_config=cfg,
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
            )

    def test_transfer_use_system_prompt_defaults_off(self):
        """Reference parity: transfer defaults ``use_system_prompt=False`` even
        when a video input is present — V2V's default-True rule must not leak
        into the transfer branch (vllm-omni ``_forward_transfer`` defaults False).
        An explicit request value is still honored."""
        from tensorrt_llm._torch.visual_gen.models.cosmos3.transfer import resolve_transfer_config

        pipeline = Cosmos3OmniMoTPipeline.__new__(Cosmos3OmniMoTPipeline)
        pipeline.transformer = SimpleNamespace(device=torch.device("cpu"))
        pipeline.action_gen = False
        # __new__ skips __init__, where the real pipeline resolves this.
        pipeline.family = QWEN3_RECIPE.name
        pipeline.audio_gen = False

        class FakeSampling:
            is_distilled = False
            checkpoint_flow_shift = 1.0

            def validate_request(self, num_inference_steps, guidance_scale):
                return None

            def generation_default_overrides(self):
                return {}

        pipeline.sampling = FakeSampling()
        captured = {}

        def fake_forward_transfer(**kwargs):
            captured.update(kwargs)
            return None

        pipeline._forward_transfer = fake_forward_transfer
        cfg = resolve_transfer_config(
            {"edge": True}, SimpleNamespace(num_frames=93, guidance_scale=None), None
        )

        for explicit, expected in ((None, False), (True, True)):
            captured.clear()
            pipeline.forward(
                prompt="bounce",
                video=_V2V_FIXTURE_MP4.read_bytes(),
                transfer_config=cfg,
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
                use_system_prompt=explicit,
                use_guardrails=False,
            )
            assert captured["use_system_prompt"] is expected


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
            use_karras_sigmas=_base_use_karras_sigmas(cosmos3_pipeline),
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

    def test_v2v_audio_smoke(self, cosmos3_pipeline):
        """Audio + V2V combined — allowed in both implementations (no guard),
        previously untested in either."""
        _require_audio_pipeline(cosmos3_pipeline)
        result = _run_forward(
            cosmos3_pipeline,
            enable_audio=True,
            video=_V2V_FIXTURE_MP4.read_bytes(),
            condition_video_latent_indexes=[0, 1],
        )
        _assert_valid_video(result.video, num_frames=NUM_FRAMES)
        _assert_valid_audio(result.audio, result.audio_sample_rate)
        _assert_scheduler_config(
            cosmos3_pipeline,
            flow_shift=10.0,
            use_karras_sigmas=False,
        )


@pytest.mark.integration
@pytest.mark.cosmos3_action
@pytest.mark.high_cuda_memory
class TestCosmos3Action:
    ACTION_HEIGHT = 480
    ACTION_WIDTH = 832
    ACTION_CHUNK = COSMOS3_ACTION_PARAMS["action_chunk_size"]
    # Derived, not configured: both references fix the clip at chunk + 1.
    ACTION_FRAMES = ACTION_CHUNK + 1
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

    def test_inverse_dynamics_smoke(self, cosmos3_pipeline):
        _require_action_pipeline(cosmos3_pipeline)
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
            # The clip is chunk + 1 frames, and the fixture holds NUM_FRAMES.
            action_chunk_size=NUM_FRAMES - 1,
            video=_V2V_FIXTURE_MP4.read_bytes(),
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
            chunk_size=NUM_FRAMES - 1,
        )

    def test_inverse_dynamics_rejects_short_video(self, cosmos3_pipeline):
        _require_action_pipeline(cosmos3_pipeline)
        with pytest.raises(ValueError, match=r"requires \d+ frames at"):
            _run_forward(
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
                video=_V2V_FIXTURE_MP4.read_bytes(),
            )

    def test_inverse_dynamics_thins_to_the_requested_rate(self, cosmos3_pipeline):
        """A 24 fps reference asked for at 5 fps keeps every 5th frame, so the
        window widens accordingly and the 9-frame fixture comes up short."""
        _require_action_pipeline(cosmos3_pipeline)
        with pytest.raises(ValueError, match=r"thinned by 5 \(41 source frames needed\)"):
            _run_forward(
                cosmos3_pipeline,
                image=None,
                height=self.ACTION_HEIGHT,
                width=self.ACTION_WIDTH,
                num_frames=NUM_FRAMES,
                guidance_scale=COSMOS3_ACTION_PARAMS["guidance_scale"],
                frame_rate=5.0,
                action_mode="inverse_dynamics",
                domain_name="bridge_orig_lerobot",
                raw_action_dim=self.RAW_ACTION_DIM,
                action_chunk_size=NUM_FRAMES - 1,
                video=_V2V_FIXTURE_MP4.read_bytes(),
            )

    def test_out_of_range_domain_id_rejected_before_decode(self, cosmos3_pipeline):
        _require_action_pipeline(cosmos3_pipeline)
        with pytest.raises(ValueError, match=r"domain_id must be in \[0, \d+\)"):
            _run_forward(
                cosmos3_pipeline,
                image=_make_test_image(),
                height=self.ACTION_HEIGHT,
                width=self.ACTION_WIDTH,
                num_frames=self.ACTION_FRAMES,
                guidance_scale=COSMOS3_ACTION_PARAMS["guidance_scale"],
                action_mode="policy",
                domain_id=10_000,
                raw_action_dim=self.RAW_ACTION_DIM,
                action_chunk_size=self.ACTION_CHUNK,
            )

    def test_first_frame_failure_is_synchronized(self, cosmos3_pipeline, monkeypatch):
        """Every rank decodes its own reference. A failure on one rank has to
        reach the others before the transformer's collectives, or the job hangs."""
        _require_action_pipeline(cosmos3_pipeline)
        from tensorrt_llm._torch.visual_gen.models.cosmos3 import pipeline_cosmos3

        seen = []
        real = pipeline_cosmos3.synchronize_media_prepare_status

        def spy(error):
            seen.append(error)
            return real(error)

        monkeypatch.setattr(pipeline_cosmos3, "synchronize_media_prepare_status", spy)

        with pytest.raises(Exception):
            _run_forward(
                cosmos3_pipeline,
                image="/nonexistent/action_reference_frame.png",
                height=self.ACTION_HEIGHT,
                width=self.ACTION_WIDTH,
                num_frames=self.ACTION_FRAMES,
                guidance_scale=COSMOS3_ACTION_PARAMS["guidance_scale"],
                action_mode="policy",
                domain_name="bridge_orig_lerobot",
                raw_action_dim=self.RAW_ACTION_DIM,
                action_chunk_size=self.ACTION_CHUNK,
            )

        assert seen and isinstance(seen[0], Exception)

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


class TestCosmos3TextGuardrailBlocked:
    """A blocked prompt must return an empty output, not raise.

    The block exits before the text encoder, transformer and VAE run, so the
    path is reachable on a bare instance carrying only the config-derived
    attributes ``forward()`` reads on the way there.
    """

    @staticmethod
    def _blocked_pipeline():
        pipeline = Cosmos3OmniMoTPipeline.__new__(Cosmos3OmniMoTPipeline)
        pipeline.transformer = SimpleNamespace(device=torch.device("cpu"))
        pipeline.audio_gen = False
        pipeline.audio_scheduler = None
        pipeline.family = QWEN3_RECIPE.name
        # All-None policy is the documented pre-load placeholder.
        pipeline.sampling = Cosmos3SamplingPolicy()
        pipeline.default_use_system_prompt = False
        # ``rank`` and ``device`` are read-only properties: rank resolves to 0
        # with no distributed init, device comes off the transformer stub.
        # The guardrail model is not under test, so a checker that reports
        # "unsafe" for any input drives the path without an unsafe prompt.
        pipeline.safety_checker = SimpleNamespace(check_text_safety=lambda _prompt: False)
        pipeline._scheduler_for = lambda *args, **kwargs: None
        return pipeline

    # Without CUDA the phase timer is a no-op and the test passes vacuously.
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CudaPhaseTimer needs CUDA")
    def test_blocked_prompt_returns_empty_output(self, monkeypatch):
        # The module-wide disable fixture forces use_guardrails False; without
        # restoring it here the guardrail never runs and nothing is exercised.
        monkeypatch.setattr(pipe_mod, "TRTLLM_DISABLE_COSMOS3_GUARDRAILS", False)

        result = self._blocked_pipeline().forward(
            prompt="a calm sunny meadow", seed=SEED, use_guardrails=True
        )

        assert result.video is None
        assert result.image is None
        assert (result.pre_denoise, result.denoise, result.post_denoise) == (0.0, 0.0, 0.0)


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


class TestErrorClassificationIsOptIn:
    """Failure classification is per-pipeline, not inferred from the exception
    class alone. A ``ValueError`` is as likely to be an internal invariant as a
    rejected input, so a pipeline that hasn't opted in must leave every failure
    unclassified -- otherwise adding V2V would silently turn other models'
    internal errors into client errors on the public API.
    """

    def test_base_pipeline_classifies_nothing(self):
        from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline

        pipeline = object.__new__(BasePipeline)
        for exc in (ValueError("x"), MemoryError("x"), RuntimeError("x")):
            assert pipeline.classify_request_failure(exc) is None

    def test_cosmos3_opts_in(self):
        pipeline = object.__new__(Cosmos3OmniMoTPipeline)
        assert pipeline.classify_request_failure(ValueError("bad reference")) == "client"
        assert pipeline.classify_request_failure(MemoryError("no room")) == "capacity"
        assert (
            pipeline.classify_request_failure(torch.cuda.OutOfMemoryError("no room")) == "capacity"
        )
        # Unclassified stays unclassified: an internal fault is not the
        # caller's fault.
        assert pipeline.classify_request_failure(RuntimeError("internal")) is None
