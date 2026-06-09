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

Override checkpoint:
    DIFFUSION_MODEL_PATH_COSMOS3=/path/to/Cosmos3-Nano \\
        pytest tests/unittest/_torch/visual_gen/test_cosmos3_pipeline.py -v -s
"""

import gc
import os
from pathlib import Path

os.environ["TLLM_DISABLE_MPI"] = "1"
os.environ["TRTLLM_DISABLE_COSMOS3_GUARDRAILS"] = "1"

import PIL.Image
import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.cosmos3.defaults import COSMOS3_T2I_PARAMS
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


def _run_forward(pipeline, *, image=None, num_frames=NUM_FRAMES, **extra):
    return pipeline.forward(
        prompt=PROMPT,
        image=image,
        height=HEIGHT,
        width=WIDTH,
        num_frames=num_frames,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
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


def _make_test_image() -> PIL.Image.Image:
    image_path = os.environ.get("COSMOS3_TEST_IMAGE")
    if image_path and os.path.exists(image_path):
        return PIL.Image.open(image_path).convert("RGB")
    return PIL.Image.new("RGB", (WIDTH, HEIGHT), color=(64, 128, 192))


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


@pytest.mark.integration
@pytest.mark.cosmos3_i2v
@pytest.mark.high_cuda_memory
class TestCosmos3I2V:
    def test_i2v_smoke(self, cosmos3_pipeline):
        image = _make_test_image()
        result = _run_forward(cosmos3_pipeline, image=image, num_frames=NUM_FRAMES)
        _assert_valid_video(result.video, num_frames=NUM_FRAMES)
        assert result.frame_rate == FRAME_RATE


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
@pytest.mark.cosmos3_t2v
@pytest.mark.high_cuda_memory
class TestCosmos3PromptTemplates:
    @pytest.mark.parametrize(
        "use_duration_template,use_resolution_template,use_system_prompt",
        [
            (True, True, True),
            (False, False, False),
            (False, False, True),
        ],
        ids=["all-on", "all-off", "system-prompt-only"],
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
            assert pipeline.pipeline_config.quant_config.quant_algo is not None
            result = _run_forward(pipeline, image=None, num_frames=NUM_FRAMES)
            _assert_valid_video(result.video, num_frames=NUM_FRAMES)
            assert result.frame_rate == FRAME_RATE
        finally:
            del pipeline
            gc.collect()
            torch.cuda.empty_cache()
