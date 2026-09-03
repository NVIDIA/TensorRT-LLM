# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Wan 2.2 T2V CPU offloading using the full pipeline.

Wan 2.2 two-stage T2V is tested with cpu_offload_config.enable=True:
  - Wan2.2-T2V-A14B-Diffusers   480x832

Offloaded output is checked against the non-offload TRT-LLM baseline. HuggingFace parity
with offloading is in test_wan22_t2v_pipeline.py::test_cosine_similarity_with_offload.
FP8 dynamic quantization is checked against the TRT-LLM baseline only.
CUDA graphs with offloading are not supported (test_wan22_raises_if_cuda_graph_and_offload_enabled).
CUDA-graph correctness is in test_wan22_t2v_pipeline.py.

Each test loads all pipeline components and calls pipeline.forward().

Run all:
    pytest tests/unittest/_torch/visual_gen/test_wan22_t2v_offload.py -v -s

Override checkpoint path:
    DIFFUSION_MODEL_PATH_WAN22_T2V=/path/to/wan22 \\
        pytest tests/unittest/_torch/visual_gen/test_wan22_t2v_offload.py -v -s
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

import gc
from pathlib import Path
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm.visual_gen.args import (
    CpuOffloadConfig,
    CudaGraphConfig,
    TorchCompileConfig,
    VisualGenArgs,
)


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


@pytest.fixture(autouse=True)
def _cleanup_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    yield
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================================
# Path helpers
# ============================================================================


def _llm_models_root() -> Path:
    if "LLM_MODELS_ROOT" in os.environ:
        root = Path(os.environ["LLM_MODELS_ROOT"])
    else:
        root = Path("/home/scratch.trt_llm_data_ci/llm-models/")
    if not root.exists():
        root = Path("/scratch.trt_llm_data/llm-models/")
    assert root.exists(), (
        "Set LLM_MODELS_ROOT or ensure /home/scratch.trt_llm_data_ci/llm-models/ is accessible."
    )
    return root


def _checkpoint(env_var: str, default_name: str) -> str:
    return os.environ.get(env_var) or str(_llm_models_root() / default_name)


WAN22_A14B_PATH = _checkpoint("DIFFUSION_MODEL_PATH_WAN22_T2V", "Wan2.2-T2V-A14B-Diffusers")

INFER_PROMPT = "A cat sitting on a sunny windowsill watching birds outside."
INFER_NEGATIVE_PROMPT = ""
INFER_NUM_FRAMES = 9
INFER_NUM_STEPS = 4
INFER_SEED = 42
INFER_GUIDANCE_SCALE = 4.0
COS_SIM_THRESHOLD = 0.99


# ============================================================================
# Pipeline fixture factory
# ============================================================================


def _make_pipeline(
    checkpoint_path: str,
    *,
    enable_offload: bool = False,
    quant_config: Optional[dict] = None,
):
    if not os.path.exists(checkpoint_path):
        pytest.skip(f"Checkpoint not found: {checkpoint_path}")
    args = VisualGenArgs(
        model=checkpoint_path,
        torch_compile_config=TorchCompileConfig(enable=False),
        cpu_offload_config=CpuOffloadConfig(enable=enable_offload),
        quant_config=quant_config,
    )
    pipeline = PipelineLoader(args).load(skip_warmup=True)
    assert pipeline.transformer_2 is not None, (
        "Expected two-stage Wan 2.2 pipeline (transformer_2 should not be None)"
    )
    return pipeline


def _require_fp8_quant_ops() -> None:
    try:
        _ = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor
        _ = torch.ops.tensorrt_llm.quantize_e4m3_activation
    except (AttributeError, RuntimeError) as e:
        pytest.skip(f"FP8 quantization ops not available: {e}")


@pytest.fixture
def wan22_offload_pipeline():
    pipeline = _make_pipeline(WAN22_A14B_PATH, enable_offload=True)
    yield pipeline
    del pipeline
    torch.cuda.empty_cache()


# ============================================================================
# Shared assertion helpers
# ============================================================================


def _assert_offload_forward(pipeline, *, model: str = "") -> None:
    assert pipeline.offloader.stages(), (
        f"{model}: cpu_offload_config.enable=True but offloader.stages() is empty"
    )
    assert pipeline.offloader.offload_pipeline is not None, (
        f"{model}: offload pipeline was not initialized after load"
    )

    with torch.no_grad():
        pipeline.forward(
            prompt=INFER_PROMPT,
            negative_prompt=INFER_NEGATIVE_PROMPT,
            height=480,
            width=832,
            num_frames=INFER_NUM_FRAMES,
            num_inference_steps=INFER_NUM_STEPS,
            guidance_scale=INFER_GUIDANCE_SCALE,
            seed=INFER_SEED,
        )

    print(f"\n  ===== Offload — Wan 2.2 {model} 480x832 =====")
    print(f"  stages: {pipeline.offloader.stages()}")
    print("  =================================================")


def _assert_offload_matches_baseline(
    checkpoint_path: str,
    *,
    model: str,
    quant_config: Optional[dict] = None,
) -> None:
    def _capture_video(pipe) -> torch.Tensor:
        with torch.no_grad():
            result = pipe.forward(
                prompt=INFER_PROMPT,
                negative_prompt=INFER_NEGATIVE_PROMPT,
                height=480,
                width=832,
                num_frames=INFER_NUM_FRAMES,
                num_inference_steps=INFER_NUM_STEPS,
                guidance_scale=INFER_GUIDANCE_SCALE,
                seed=INFER_SEED,
            )
        return result.video.float() / 255.0

    offload_pipe = _make_pipeline(checkpoint_path, enable_offload=True, quant_config=quant_config)
    offload_video = _capture_video(offload_pipe)
    del offload_pipe
    gc.collect()
    torch.cuda.empty_cache()

    baseline_pipe = _make_pipeline(checkpoint_path, enable_offload=False, quant_config=quant_config)
    baseline_video = _capture_video(baseline_pipe)
    del baseline_pipe
    gc.collect()
    torch.cuda.empty_cache()

    assert offload_video.shape == baseline_video.shape, (
        f"{model}: shape mismatch offload {offload_video.shape} vs baseline {baseline_video.shape}"
    )
    a_flat = offload_video.float().cpu().reshape(-1)
    b_flat = baseline_video.float().cpu().reshape(-1)
    cos_sim = F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).clamp(-1.0, 1.0).item()
    print(
        f"\n  ===== Offload accuracy — {model} =====\n"
        f"  cosine_similarity={cos_sim:.6f}  (threshold={COS_SIM_THRESHOLD})\n"
        f"  ========================================================"
    )
    assert cos_sim >= COS_SIM_THRESHOLD, (
        f"{model}: offload vs baseline cosine similarity {cos_sim:.6f} < {COS_SIM_THRESHOLD}"
    )


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.wan_t2v
class TestWan22_A14B_Offload:
    """Wan2.2-T2V-A14B  480x832  two-stage with CPU offloading."""

    def test_wan22_offload(self, wan22_offload_pipeline):
        _assert_offload_forward(wan22_offload_pipeline, model="T2V-A14B")

    def test_wan22_offload_matches_baseline(self):
        _assert_offload_matches_baseline(WAN22_A14B_PATH, model="Wan2.2-T2V-A14B")

    def test_wan22_fp8_offload_matches_baseline(self):
        _require_fp8_quant_ops()
        _assert_offload_matches_baseline(
            WAN22_A14B_PATH,
            model="Wan2.2-T2V-A14B FP8",
            quant_config={"quant_algo": "FP8", "dynamic": True},
        )


@pytest.mark.integration
@pytest.mark.wan_t2v
class TestWanT2VOffloadCudaGraphRaisesError:
    """CUDA graphs plus offloading must raise NotImplementedError on pipeline load."""

    def test_wan22_raises_if_cuda_graph_and_offload_enabled(self):
        if not os.path.exists(WAN22_A14B_PATH):
            pytest.skip(f"Checkpoint not found: {WAN22_A14B_PATH}")
        args = VisualGenArgs(
            model=WAN22_A14B_PATH,
            torch_compile_config=TorchCompileConfig(enable=False),
            cuda_graph_config=CudaGraphConfig(enable=True),
            cpu_offload_config=CpuOffloadConfig(enable=True),
        )
        with pytest.raises(
            NotImplementedError,
            match="CUDA graphs are not supported with visual generation offloading",
        ):
            PipelineLoader(args).load(skip_warmup=True)
