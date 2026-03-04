# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for Wan 2.1 I2V pipeline against HuggingFace reference.

Tests verify that the TRTLLM WanImageToVideoPipeline (I2V) produces denoised latents
with >= 0.97 cosine similarity to the HuggingFace diffusers WanImageToVideoPipeline
baseline. Latents are captured before VAE decoding for maximum numerical precision.
The threshold is 0.97 (not 0.99) because bfloat16 error accumulates across 10
denoising steps in the full pipeline, and the 14B I2V model (40 layers, hidden
size 5120) accumulates more error per step than the smaller 1.3B T2V model. The
single-step transformer test (test_wan_transformer.py) enforces the tighter 0.99
threshold.

Wan 2.1 I2V uses CLIP image embeddings for image conditioning alongside the
concatenated VAE-encoded image condition latent.

Models tested:
  - Wan2.1-I2V-14B-480P-Diffusers   (480x832,   33 frames)
  - Wan2.1-I2V-14B-720P-Diffusers   (720x1280,  33 frames)

Run all:
    pytest tests/unittest/_torch/visual_gen/test_wan_21_i2v_pipeline.py -v -s

Run one model:
    pytest tests/unittest/_torch/visual_gen/test_wan_21_i2v_pipeline.py -v -s -k 480p
    pytest tests/unittest/_torch/visual_gen/test_wan_21_i2v_pipeline.py -v -s -k 720p

Override checkpoint paths:
    DIFFUSION_MODEL_PATH_WAN21_I2V_480P=/path/to/480p \\
    DIFFUSION_MODEL_PATH_WAN21_I2V_720P=/path/to/720p \\
        pytest tests/unittest/_torch/visual_gen/test_wan_21_i2v_pipeline.py -v -s
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

import gc

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline
from PIL import Image

from tensorrt_llm._torch.visual_gen.config import DiffusionArgs, TorchCompileConfig
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# ============================================================================
# Path helpers
# ============================================================================


def _require_checkpoint(env_var: str) -> str:
    """Return checkpoint path from env var, or empty string if not set.

    Tests skip automatically when the path is empty or does not exist.
    Set the env var before running:
        export DIFFUSION_MODEL_PATH_WAN21_I2V_480P=/path/to/Wan2.1-I2V-14B-480P-Diffusers
        export DIFFUSION_MODEL_PATH_WAN21_I2V_720P=/path/to/Wan2.1-I2V-14B-720P-Diffusers
    """
    path = os.environ.get(env_var, "")
    return path


WAN21_I2V_480P_PATH = _require_checkpoint("DIFFUSION_MODEL_PATH_WAN21_I2V_480P")
WAN21_I2V_720P_PATH = _require_checkpoint("DIFFUSION_MODEL_PATH_WAN21_I2V_720P")

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
    if not checkpoint_path:
        pytest.skip(
            "Checkpoint path not set. Provide it via "
            "DIFFUSION_MODEL_PATH_WAN21_I2V_480P or DIFFUSION_MODEL_PATH_WAN21_I2V_720P."
        )
    if not os.path.exists(checkpoint_path):
        pytest.skip(f"Checkpoint not found: {checkpoint_path}")
    args = DiffusionArgs(
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
