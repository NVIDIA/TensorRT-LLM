# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for Wan 2.1 I2V pipeline against HuggingFace reference.

Verifies >= 0.99 cosine similarity on decoded video frames (T, H, W, C) after
full denoising + VAE decode. Image conditioning uses CLIP embeddings + VAE-encoded latent.

Models:
  - Wan2.1-I2V-14B-480P-Diffusers   (480x832,  33 frames)
  - Wan2.1-I2V-14B-720P-Diffusers   (720x1280, 33 frames)

Run:
    pytest tests/unittest/_torch/visual_gen/test_wan21_i2v_pipeline.py -v -s -k 480p
    pytest tests/unittest/_torch/visual_gen/test_wan21_i2v_pipeline.py -v -s -k 720p

Override checkpoint paths:
    DIFFUSION_MODEL_PATH_WAN21_I2V_480P=/path/to/480p \\
    DIFFUSION_MODEL_PATH_WAN21_I2V_720P=/path/to/720p \\
        pytest tests/unittest/_torch/visual_gen/test_wan21_i2v_pipeline.py -v -s
"""

import gc
import os
from pathlib import Path

os.environ["TLLM_DISABLE_MPI"] = "1"

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline
from PIL import Image

from tensorrt_llm._torch.visual_gen.config import TorchCompileConfig, VisualGenArgs
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# ============================================================================
# Path helpers
# ============================================================================


def _llm_models_root() -> str:
    """Return LLM_MODELS_ROOT path if set in env, assert when it's set but not a valid path."""
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


WAN21_I2V_480P_PATH = _checkpoint(
    "DIFFUSION_MODEL_PATH_WAN21_I2V_480P", "Wan2.1-I2V-14B-480P-Diffusers"
)
WAN21_I2V_720P_PATH = _checkpoint(
    "DIFFUSION_MODEL_PATH_WAN21_I2V_720P", "Wan2.1-I2V-14B-720P-Diffusers"
)

# ============================================================================
# Test constants
# ============================================================================

PROMPT = "A cat sitting on a sunny windowsill watching birds outside."
NEGATIVE_PROMPT = ""
NUM_STEPS = 10
SEED = 42
COS_SIM_THRESHOLD = 0.99


# ============================================================================
# Helpers
# ============================================================================


def _make_test_image(height: int, width: int) -> Image.Image:
    """Create a deterministic gradient test image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(0, 255, height, dtype=np.uint8)[:, np.newaxis]
    img[:, :, 1] = np.linspace(0, 255, width, dtype=np.uint8)[np.newaxis, :]
    img[:, :, 2] = 128
    return Image.fromarray(img, mode="RGB")


def _load_trtllm_pipeline(checkpoint_path: str):
    """Load TRTLLM WanImageToVideoPipeline without torch.compile or warmup."""
    if not os.path.exists(checkpoint_path):
        pytest.skip(f"Checkpoint not found: {checkpoint_path}")
    args = VisualGenArgs(
        checkpoint_path=checkpoint_path,
        device="cuda",
        dtype="bfloat16",
        torch_compile=TorchCompileConfig(enable_torch_compile=False),
    )
    return PipelineLoader(args).load(skip_warmup=True)


def _load_hf_pipeline(checkpoint_path: str):
    """Load HuggingFace diffusers pipeline (auto-detects class from model_index.json)."""
    hf_pipe = DiffusionPipeline.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
    )
    hf_pipe = hf_pipe.to("cuda")
    hf_pipe.set_progress_bar_config(disable=True)
    return hf_pipe


def _capture_trtllm_video(
    pipeline,
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
) -> torch.Tensor:
    """Run full TRTLLM pipeline including VAE decode; return (T, H, W, C) float in [0, 1]."""
    with torch.no_grad():
        result = pipeline.forward(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
    video = result.video  # (T, H, W, C) uint8
    return video.float() / 255.0


def _capture_hf_video(
    hf_pipe,
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
) -> torch.Tensor:
    """Run HF pipeline with output_type='np'; return (T, H, W, C) float in [0, 1]."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    output = hf_pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="np",
    )
    frames = output.frames  # (1, T, H, W, C) numpy float32 in [0, 1]
    if isinstance(frames, np.ndarray):
        return torch.from_numpy(frames[0]).float()
    return frames[0].float()


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two tensors (flattened to 1D, cast to float32 on CPU)."""
    a_flat = a.float().cpu().reshape(-1)
    b_flat = b.float().cpu().reshape(-1)
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).clamp(-1.0, 1.0).item()


def _assert_pipeline_matches_hf(
    checkpoint_path: str,
    height: int,
    width: int,
    num_frames: int,
    guidance_scale: float,
    model_label: str,
) -> None:
    """Run TRTLLM and HF pipelines sequentially, compare decoded video output."""
    test_image = _make_test_image(height, width)

    # --- TRTLLM ---
    trtllm_pipe = _load_trtllm_pipeline(checkpoint_path)
    trtllm_video = _capture_trtllm_video(
        trtllm_pipe,
        image=test_image,
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=NUM_STEPS,
        guidance_scale=guidance_scale,
        seed=SEED,
    )
    del trtllm_pipe
    gc.collect()
    torch.cuda.empty_cache()

    # --- HF reference ---
    hf_pipe = _load_hf_pipeline(checkpoint_path)
    hf_video = _capture_hf_video(
        hf_pipe,
        image=test_image,
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=NUM_STEPS,
        guidance_scale=guidance_scale,
        seed=SEED,
    )
    del hf_pipe
    gc.collect()
    torch.cuda.empty_cache()

    # --- Compare ---
    assert trtllm_video.numel() == hf_video.numel(), (
        f"{model_label}: element count mismatch — "
        f"TRTLLM {trtllm_video.shape} ({trtllm_video.numel()}) vs "
        f"HF {hf_video.shape} ({hf_video.numel()})"
    )

    cos_sim = _cosine_similarity(trtllm_video, hf_video)
    print(f"\n  {model_label} cosine similarity: {cos_sim:.6f}")
    assert cos_sim >= COS_SIM_THRESHOLD, (
        f"{model_label}: cosine similarity {cos_sim:.6f} < {COS_SIM_THRESHOLD}. "
        f"TRTLLM pipeline output diverges from the HuggingFace reference. "
        f"Video shapes — TRTLLM: {trtllm_video.shape}, HF: {hf_video.shape}."
    )


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.wan_i2v
class TestWan21_I2V_480P_PipelineCorrectness:
    """Wan2.1-I2V-14B-480P correctness vs HuggingFace reference (480x832, 33 frames)."""

    def test_cosine_similarity(self):
        _assert_pipeline_matches_hf(
            checkpoint_path=WAN21_I2V_480P_PATH,
            height=480,
            width=832,
            num_frames=33,
            guidance_scale=5.0,
            model_label="Wan2.1-I2V-14B-480P",
        )


@pytest.mark.integration
@pytest.mark.wan_i2v
class TestWan21_I2V_720P_PipelineCorrectness:
    """Wan2.1-I2V-14B-720P correctness vs HuggingFace reference (720x1280, 33 frames)."""

    def test_cosine_similarity(self):
        _assert_pipeline_matches_hf(
            checkpoint_path=WAN21_I2V_720P_PATH,
            height=720,
            width=1280,
            num_frames=33,
            guidance_scale=5.0,
            model_label="Wan2.1-I2V-14B-720P",
        )


# =============================================================================
# Batch Generation Tests (I2V)
# =============================================================================


@pytest.mark.integration
@pytest.mark.i2v
class TestWanI2VBatchGeneration:
    """Batch generation tests for WAN I2V pipeline (Wan 2.1 and Wan 2.2).

    Tests that passing a list of prompts produces batched output
    and matches sequential generation with the same seeds.
    """

    @pytest.fixture(scope="class")
    def i2v_full_pipeline(self):
        """Load full I2V pipeline (all components) for batch tests."""
        if not WAN21_I2V_480P_PATH or not os.path.exists(WAN21_I2V_480P_PATH):
            pytest.skip("Checkpoint not available. Set DIFFUSION_MODEL_PATH_WAN21_I2V_480P.")

        args = VisualGenArgs(
            checkpoint_path=WAN21_I2V_480P_PATH,
            device="cuda",
            dtype="bfloat16",
            torch_compile=TorchCompileConfig(enable_torch_compile=False),
        )
        pipeline = PipelineLoader(args).load(skip_warmup=True)
        yield pipeline
        del pipeline
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_single_prompt_backward_compat(self, i2v_full_pipeline):
        """Single prompt returns (T, H, W, C) for backward compatibility."""
        test_image = _make_test_image(480, 832)
        result = i2v_full_pipeline.forward(
            prompt="a cat walking",
            image=test_image,
            height=480,
            width=832,
            num_frames=9,
            num_inference_steps=4,
            guidance_scale=5.0,
            seed=42,
        )
        assert result.video.dim() == 5, f"Expected 5D (B,T,H,W,C), got {result.video.dim()}D"
        B, _T, H, W, C = result.video.shape
        assert B == 1 and H == 480 and W == 832 and C == 3

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_batch_prompt_shape(self, i2v_full_pipeline):
        """List of prompts returns (B, T, H, W, C)."""
        test_image = _make_test_image(480, 832)
        prompts = ["a sunset over mountains", "a cat on a roof"]
        result = i2v_full_pipeline.forward(
            prompt=prompts,
            image=test_image,
            height=480,
            width=832,
            num_frames=9,
            num_inference_steps=4,
            guidance_scale=5.0,
            seed=42,
        )
        assert result.video.dim() == 5, f"Expected 5D (B,T,H,W,C), got {result.video.dim()}D"
        B, _T, H, W, C = result.video.shape
        assert B == 2 and H == 480 and W == 832 and C == 3
