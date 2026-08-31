# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for VSA T2V pipeline with Video Sparse Attention (VSA).

Verifies >= 0.95 cosine similarity between the CuTe-DSL VSA kernel and the
SDPA-fallback VSA path (same gated coarse+fine formulation, different fine kernel).

Models:
  - FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers (720x1280, 9 frames)

Run:
    pytest tests/unittest/_torch/visual_gen/test_wan_vsa_pipeline.py -v -s

Override checkpoint path:
    DIFFUSION_MODEL_PATH_WAN21_VSA=/path/to/vsa \\
        pytest tests/unittest/_torch/visual_gen/test_wan_vsa_pipeline.py -v -s
"""

import gc
import os
from pathlib import Path

os.environ["TLLM_DISABLE_MPI"] = "1"

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.attention_backend.cute_dsl import _cute_dsl_import_error
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm.visual_gen.args import (
    AttentionConfig,
    TorchCompileConfig,
    VideoSparseAttentionConfig,
    VisualGenArgs,
)

_cute_dsl_available = _cute_dsl_import_error is None


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


WAN21_VSA_PATH = _checkpoint("DIFFUSION_MODEL_PATH_WAN21_VSA", "Wan2.1-VSA-T2V-14B-720P-Diffusers")

# ============================================================================
# Test constants
# ============================================================================

PROMPT = "A cat sitting on a sunny windowsill watching birds outside."
NEGATIVE_PROMPT = ""
NUM_STEPS = 4
SEED = 42
COS_SIM_THRESHOLD = 0.95


# ============================================================================
# Helpers
# ============================================================================


def _load_vsa_pipeline(checkpoint_path: str, vsa_sparsity: float = 0.0):
    """Load TRTLLM WanPipeline with CUTEDSL + VSA backend."""
    if not os.path.exists(checkpoint_path):
        pytest.skip(f"Checkpoint not found: {checkpoint_path}")
    if not _cute_dsl_available:
        pytest.skip(f"CUTEDSL not available (requires Blackwell GPU): {_cute_dsl_import_error}")
    args = VisualGenArgs(
        model=checkpoint_path,
        attention_config=AttentionConfig(
            backend="CUTEDSL",
            sparse_attention_config=VideoSparseAttentionConfig(vsa_sparsity=vsa_sparsity),
        ),
        torch_compile_config=TorchCompileConfig(enable=False),
    )
    return PipelineLoader(args).load(skip_warmup=True)


def _capture_trtllm_video(
    pipeline,
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
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
    video = result.video  # (B, T, H, W, C) uint8
    return video.float() / 255.0


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two tensors (flattened to 1D, cast to float32 on CPU)."""
    a_flat = a.float().cpu().reshape(-1)
    b_flat = b.float().cpu().reshape(-1)
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).clamp(-1.0, 1.0).item()


def _assert_vsa_matches_dense(
    checkpoint_path: str,
    height: int,
    width: int,
    num_frames: int,
    guidance_scale: float,
    model_label: str,
    vsa_sparsity: float = 0.0,
) -> None:
    """Compare CuTe-DSL VSA against SDPA-fallback VSA (same gated formulation, different fine kernel)."""
    from unittest.mock import patch

    from tensorrt_llm._torch.visual_gen.attention_backend.cute_dsl import vsa as _vsa_module

    common_kwargs = dict(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=NUM_STEPS,
        guidance_scale=guidance_scale,
        seed=SEED,
    )

    # --- CuTe-DSL path ---
    vsa_pipe = _load_vsa_pipeline(checkpoint_path, vsa_sparsity=vsa_sparsity)
    vsa_video = _capture_trtllm_video(vsa_pipe, **common_kwargs)
    del vsa_pipe
    gc.collect()
    torch.cuda.empty_cache()

    # --- SDPA fallback reference (same VSA formulation, fine attn via SDPA) ---
    sdpa_pipe = _load_vsa_pipeline(checkpoint_path, vsa_sparsity=vsa_sparsity)
    with patch.object(_vsa_module, "is_cute_supported", return_value=False):
        sdpa_video = _capture_trtllm_video(sdpa_pipe, **common_kwargs)
    del sdpa_pipe
    gc.collect()
    torch.cuda.empty_cache()

    # --- Compare ---
    assert vsa_video.numel() == sdpa_video.numel(), (
        f"{model_label}: element count mismatch — "
        f"CuTe {vsa_video.shape} ({vsa_video.numel()}) vs "
        f"SDPA {sdpa_video.shape} ({sdpa_video.numel()})"
    )

    cos_sim = _cosine_similarity(vsa_video, sdpa_video)
    print(f"\n  {model_label} cosine similarity: {cos_sim:.6f}")
    assert cos_sim >= COS_SIM_THRESHOLD, (
        f"{model_label}: cosine similarity {cos_sim:.6f} < {COS_SIM_THRESHOLD}. "
        f"CuTe-DSL VSA diverges from SDPA-fallback VSA. "
        f"Video shapes — CuTe: {vsa_video.shape}, SDPA: {sdpa_video.shape}."
    )


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.wan_t2v
class TestWanVsa14B_PipelineCorrectness:
    """Wan2.1-VSA-T2V-14B: CuTe-DSL vs SDPA-fallback correctness (720x1280, 9 frames).

    Verifies CuTe-DSL kernel at sparsity=0.0 matches SDPA fallback with >= 0.95 cosine sim.
    """

    def test_cosine_similarity(self):
        _assert_vsa_matches_dense(
            checkpoint_path=WAN21_VSA_PATH,
            height=720,
            width=1280,
            num_frames=9,
            guidance_scale=5.0,
            model_label="Wan2.1-VSA-T2V-14B+VSA",
            vsa_sparsity=0.0,
        )


@pytest.mark.integration
@pytest.mark.wan_t2v
class TestWanVsaSparse:
    """VSA at sparsity=0.9: config propagates, output is correctly shaped and finite."""

    def test_sparse_vsa(self):
        pipeline = _load_vsa_pipeline(WAN21_VSA_PATH, vsa_sparsity=0.9)
        try:
            attn_cfg = pipeline.pipeline_config.primary_model_config.attention
            assert attn_cfg.backend == "CUTEDSL"
            assert attn_cfg.sparse_attention_config.vsa_sparsity == 0.9

            with torch.no_grad():
                result = pipeline.forward(
                    prompt=PROMPT,
                    negative_prompt=NEGATIVE_PROMPT,
                    height=720,
                    width=1280,
                    num_frames=9,
                    num_inference_steps=NUM_STEPS,
                    guidance_scale=5.0,
                    seed=SEED,
                )
            assert result.video.dim() == 5
            B, _T, H, W, C = result.video.shape
            assert B == 1 and H == 720 and W == 1280 and C == 3
            assert torch.isfinite(result.video.float()).all()
            assert pipeline.transformer.blocks[0].attn1.attn_backend == "CUTEDSL"
        finally:
            del pipeline
            gc.collect()
            torch.cuda.empty_cache()
