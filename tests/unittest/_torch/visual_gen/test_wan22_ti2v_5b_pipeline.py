# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for Wan 2.2 TI2V-5B pipeline against HuggingFace reference.

Tests verify that the TRTLLM WanPipeline produces decoded video with high cosine
similarity to the HuggingFace diffusers baseline for both T2V and I2V.
The threshold is 0.99 for T2V and 0.98 (not 0.99) for I2V because I2V image
conditioning (VAE-encoded image + mask concatenated to the latent) accumulates
additional bfloat16 error compared to T2V.

Model tested:
  - Wan2.2-TI2V-5B-Diffusers   (704x1280, 9 frames)

Run:
    pytest tests/unittest/_torch/visual_gen/test_wan22_ti2v_5b_pipeline.py -v -s

Override checkpoint path:
    DIFFUSION_MODEL_PATH_WAN22_TI2V_5B=/path/to/wan22_ti2v_5b \\
        pytest tests/unittest/_torch/visual_gen/test_wan22_ti2v_5b_pipeline.py -v -s
"""

import importlib
import os

os.environ["TLLM_DISABLE_MPI"] = "1"

import gc
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from diffusers import WanImageToVideoPipeline as HFWanImageToVideoPipeline
from diffusers import WanPipeline as HFWanPipeline
from PIL import Image

from tensorrt_llm._torch.visual_gen.config import (
    AttentionConfig,
    CacheDiTConfig,
    TorchCompileConfig,
    VisualGenArgs,
)
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


WAN22_TI2V_5B_PATH = _checkpoint(
    "DIFFUSION_MODEL_PATH_WAN22_TI2V_5B",
    "Wan2.2-TI2V-5B-Diffusers",
)

# ============================================================================
# Test constants
# ============================================================================


PROMPT = "A cat sitting on a sunny windowsill watching birds outside."
NEGATIVE_PROMPT = ""
NUM_STEPS = 4
SEED = 42
T2V_COS_SIM_THRESHOLD = 0.99
I2V_COS_SIM_THRESHOLD = 0.98


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
    """Load TRTLLM WanPipeline without torch.compile or warmup."""
    if not os.path.exists(checkpoint_path):
        pytest.skip(f"Checkpoint not found: {checkpoint_path}")
    args = VisualGenArgs(
        checkpoint_path=checkpoint_path,
        device="cuda",
        dtype="bfloat16",
        torch_compile=TorchCompileConfig(enable_torch_compile=False),
    )
    return PipelineLoader(args).load(skip_warmup=True)


def _load_hf_pipeline(checkpoint_path: str, mode: str):
    """Load HuggingFace diffusers pipeline for Wan2.2 TI2V-5B."""
    pipeline_cls = HFWanPipeline if mode == "t2v" else HFWanImageToVideoPipeline
    hf_pipe = pipeline_cls.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
    )
    hf_pipe = hf_pipe.to("cuda")
    hf_pipe.set_progress_bar_config(disable=True)
    return hf_pipe


def _capture_trtllm_video(
    pipeline,
    mode: str,
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
    call_kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    if mode == "i2v":
        call_kwargs["image"] = _make_test_image(height, width)

    with torch.no_grad():
        result = pipeline.forward(**call_kwargs)
    video = result.video.float() / 255.0
    if video.dim() == 5:
        video = video[0]
    return video


def _capture_hf_video(
    hf_pipe,
    mode: str,
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
    call_kwargs = dict(
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
    if mode == "i2v":
        call_kwargs["image"] = _make_test_image(height, width)

    output = hf_pipe(**call_kwargs)
    frames = output.frames
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
    mode: str,
    height: int,
    width: int,
    num_frames: int,
    guidance_scale: float,
    threshold: float,
    model_label: str,
) -> None:
    """Run TRTLLM and HF pipelines sequentially, compare decoded video output."""
    # --- TRTLLM ---
    trtllm_pipe = _load_trtllm_pipeline(checkpoint_path)

    # Confirm this is the single-stage Wan 2.2 TI2V-5B path.
    assert trtllm_pipe.is_wan22_5b is True
    assert trtllm_pipe.transformer_2 is None
    assert trtllm_pipe.expand_timesteps is True

    trtllm_video = _capture_trtllm_video(
        trtllm_pipe,
        mode=mode,
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
    hf_pipe = _load_hf_pipeline(checkpoint_path, mode)
    hf_video = _capture_hf_video(
        hf_pipe,
        mode=mode,
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
    assert cos_sim >= threshold, (
        f"{model_label}: cosine similarity {cos_sim:.6f} < {threshold}. "
        f"TRTLLM pipeline output diverges from the HuggingFace reference. "
        f"Video shapes — TRTLLM: {trtllm_video.shape}, HF: {hf_video.shape}."
    )


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.wan_t2v
class TestWan22TI2V5B_T2V_PipelineCorrectness:
    """Wan2.2-TI2V-5B T2V correctness vs HuggingFace reference."""

    def test_cosine_similarity(self):
        _assert_pipeline_matches_hf(
            checkpoint_path=WAN22_TI2V_5B_PATH,
            mode="t2v",
            height=704,
            width=1280,
            num_frames=9,
            guidance_scale=5.0,
            threshold=T2V_COS_SIM_THRESHOLD,
            model_label="Wan2.2-TI2V-5B T2V",
        )


@pytest.mark.integration
@pytest.mark.wan_i2v
class TestWan22TI2V5B_I2V_PipelineCorrectness:
    """Wan2.2-TI2V-5B I2V correctness vs HuggingFace reference."""

    def test_cosine_similarity(self):
        _assert_pipeline_matches_hf(
            checkpoint_path=WAN22_TI2V_5B_PATH,
            mode="i2v",
            height=704,
            width=1280,
            num_frames=9,
            guidance_scale=5.0,
            threshold=I2V_COS_SIM_THRESHOLD,
            model_label="Wan2.2-TI2V-5B I2V",
        )


# =============================================================================
# Batch Generation Tests
# =============================================================================


class TestWan22TI2V5BBatchGeneration:
    """Batch generation tests for Wan 2.2 TI2V-5B (single-stage, T2V and I2V modes)."""

    @pytest.fixture(scope="class")
    def wan22_ti2v_5b_full_pipeline(self):
        """Load full Wan 2.2 TI2V-5B pipeline (all components) for batch tests."""
        if not WAN22_TI2V_5B_PATH or not os.path.exists(WAN22_TI2V_5B_PATH):
            pytest.skip("Checkpoint not available. Set DIFFUSION_MODEL_PATH_WAN22_TI2V_5B.")

        args = VisualGenArgs(
            checkpoint_path=WAN22_TI2V_5B_PATH,
            device="cuda",
            dtype="bfloat16",
            torch_compile=TorchCompileConfig(enable_torch_compile=False),
        )
        pipeline = PipelineLoader(args).load(skip_warmup=True)
        yield pipeline
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_t2v_single_prompt_backward_compat(self, wan22_ti2v_5b_full_pipeline):
        """T2V single prompt returns (B, T, H, W, C) for backward compatibility."""
        result = wan22_ti2v_5b_full_pipeline.forward(
            prompt="a cat walking",
            height=704,
            width=1280,
            num_frames=9,
            num_inference_steps=4,
            guidance_scale=5.0,
            seed=42,
        )
        assert result.video.dim() == 5, f"Expected 5D (B,T,H,W,C), got {result.video.dim()}D"
        B, _T, H, W, C = result.video.shape
        assert B == 1 and H == 704 and W == 1280 and C == 3

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_t2v_batch_prompt_shape(self, wan22_ti2v_5b_full_pipeline):
        """T2V list of prompts returns (B, T, H, W, C)."""
        prompts = ["a sunset over mountains", "a cat on a roof"]
        result = wan22_ti2v_5b_full_pipeline.forward(
            prompt=prompts,
            height=704,
            width=1280,
            num_frames=9,
            num_inference_steps=4,
            guidance_scale=5.0,
            seed=42,
        )
        assert result.video.dim() == 5, f"Expected 5D (B,T,H,W,C), got {result.video.dim()}D"
        B, _T, H, W, C = result.video.shape
        assert B == 2 and H == 704 and W == 1280 and C == 3

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_i2v_single_prompt_backward_compat(self, wan22_ti2v_5b_full_pipeline):
        """I2V single prompt returns (B, T, H, W, C) for backward compatibility."""
        test_image = _make_test_image(704, 1280)
        result = wan22_ti2v_5b_full_pipeline.forward(
            prompt="a cat walking",
            image=test_image,
            height=704,
            width=1280,
            num_frames=9,
            num_inference_steps=4,
            guidance_scale=5.0,
            seed=42,
        )
        assert result.video.dim() == 5, f"Expected 5D (B,T,H,W,C), got {result.video.dim()}D"
        B, _T, H, W, C = result.video.shape
        assert B == 1 and H == 704 and W == 1280 and C == 3

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_i2v_batch_prompt_shape(self, wan22_ti2v_5b_full_pipeline):
        """I2V list of prompts with a single broadcast image returns (B, T, H, W, C)."""
        test_image = _make_test_image(704, 1280)
        prompts = ["a sunset over mountains", "a cat on a roof"]
        result = wan22_ti2v_5b_full_pipeline.forward(
            prompt=prompts,
            image=test_image,
            height=704,
            width=1280,
            num_frames=9,
            num_inference_steps=4,
            guidance_scale=5.0,
            seed=42,
        )
        assert result.video.dim() == 5, f"Expected 5D (B,T,H,W,C), got {result.video.dim()}D"
        B, _T, H, W, C = result.video.shape
        assert B == 2 and H == 704 and W == 1280 and C == 3


# =============================================================================
# Combined Optimization Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.wan_t2v
@pytest.mark.wan_i2v
@pytest.mark.skipif(importlib.util.find_spec("cache_dit") is None, reason="cache_dit not installed")
class TestWan22TI2V5BCombinedOptimizations:
    """FP8 + CacheDiT + TRTLLM attention combined on Wan 2.2 TI2V-5B (704x1280)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fp8_cache_dit_trtllm(self):
        if not os.path.exists(WAN22_TI2V_5B_PATH):
            pytest.skip(f"Checkpoint not found: {WAN22_TI2V_5B_PATH}")
        args = VisualGenArgs(
            checkpoint_path=WAN22_TI2V_5B_PATH,
            device="cuda",
            dtype="bfloat16",
            torch_compile=TorchCompileConfig(enable_torch_compile=False),
            quant_config={"quant_algo": "FP8", "dynamic": True},
            attention=AttentionConfig(backend="TRTLLM"),
            cache=CacheDiTConfig(),
        )
        pipeline = PipelineLoader(args).load(skip_warmup=True)
        try:
            with torch.no_grad():
                t2v_result = pipeline.forward(
                    prompt="a cat sitting on a windowsill",
                    negative_prompt="",
                    height=704,
                    width=1280,
                    num_frames=9,
                    num_inference_steps=10,
                    guidance_scale=5.0,
                    seed=42,
                )
            assert t2v_result.video.dim() == 5
            B, _T, H, W, C = t2v_result.video.shape
            assert B == 1 and H == 704 and W == 1280 and C == 3

            assert pipeline.cache_accelerator is not None
            assert pipeline.cache_accelerator.is_enabled()

            with torch.no_grad():
                i2v_result = pipeline.forward(
                    image=_make_test_image(704, 1280),
                    prompt="a cat sitting on a windowsill",
                    negative_prompt="",
                    height=704,
                    width=1280,
                    num_frames=9,
                    num_inference_steps=10,
                    guidance_scale=5.0,
                    seed=42,
                )
            assert i2v_result.video.dim() == 5
            B, _T, H, W, C = i2v_result.video.shape
            assert B == 1 and H == 704 and W == 1280 and C == 3

            assert pipeline.cache_accelerator is not None
            assert pipeline.cache_accelerator.is_enabled()
        finally:
            del pipeline
            gc.collect()
            torch.cuda.empty_cache()
