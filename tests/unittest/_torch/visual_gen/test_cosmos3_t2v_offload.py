# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline-level integration tests for Cosmos3 T2V CPU offloading.

Loads the full model and calls pipeline.forward() with cpu_offload_config.enable=True;
offloaded output is checked against the non-offload TRT-LLM baseline. CUDA graphs
with offloading must raise. These skip unless the checkpoint is available.

Fast, GPU-free offload unit tests (including the Cosmos3-specific offload wiring)
live in test_offloading.py.

Run:
    pytest tests/unittest/_torch/visual_gen/test_cosmos3_t2v_offload.py -v -s

Override checkpoint path:
    DIFFUSION_MODEL_PATH_COSMOS3=/path/to/cosmos3 \\
        pytest tests/unittest/_torch/visual_gen/test_cosmos3_t2v_offload.py -v -s
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"
# Cosmos3 guardrails require the cosmos_guardrail package (NVIDIA Open Model
# License), which is not installed in CI. Disable guardrails by default so the
# test runs without it: the guardrail offload stages below stay valid for
# validation and are filtered out at runtime when the safety checker is absent.
# Set this to "0" locally (with cosmos_guardrail installed) to also exercise
# guardrail offloading.
os.environ.setdefault("TRTLLM_DISABLE_COSMOS3_GUARDRAILS", "1")

import gc
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_cosmos3 import (
    COSMOS3_GENERATOR_OFFLOAD_COMPONENT,
    COSMOS3_REASONER_OFFLOAD_COMPONENT,
    COSMOS3_TEXT_GUARDRAIL_OFFLOAD_COMPONENT,
    COSMOS3_VIDEO_GUARDRAIL_OFFLOAD_COMPONENT,
)


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)
    os.environ.pop("TRTLLM_DISABLE_COSMOS3_GUARDRAILS", None)


@pytest.fixture(autouse=True)
def _cleanup_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    yield
    gc.collect()
    torch.cuda.empty_cache()


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


COSMOS3_PATH = _checkpoint("DIFFUSION_MODEL_PATH_COSMOS3", "Cosmos3-Nano")

# Mirrors a validated manual offload run; reduced steps keep the test fast.
INFER_PROMPT = "A cute cat playing piano."
# None -> Cosmos3 uses its safe default negative prompt and skips the guardrail
# blocklist check on it (an empty string is flagged as "Input is empty").
INFER_NEGATIVE_PROMPT = None
INFER_HEIGHT = 720
INFER_WIDTH = 1080
INFER_NUM_FRAMES = 33
INFER_NUM_STEPS = 4
INFER_GUIDANCE_SCALE = 6.0
INFER_SEED = 42
# Exercise both transformer towers, the VAE, and both guardrails.
INFER_OFFLOAD_STAGES = [
    COSMOS3_REASONER_OFFLOAD_COMPONENT,
    COSMOS3_GENERATOR_OFFLOAD_COMPONENT,
    "vae",
    COSMOS3_TEXT_GUARDRAIL_OFFLOAD_COMPONENT,
    COSMOS3_VIDEO_GUARDRAIL_OFFLOAD_COMPONENT,
]
COS_SIM_THRESHOLD = 0.99


def _make_pipeline(checkpoint_path: str, *, enable_offload: bool = False):
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.visual_gen.args import CpuOffloadConfig as ArgsOffloadConfig
    from tensorrt_llm.visual_gen.args import TorchCompileConfig, VisualGenArgs

    if not os.path.exists(checkpoint_path):
        pytest.skip(f"Checkpoint not found: {checkpoint_path}")
    args = VisualGenArgs(
        model=checkpoint_path,
        torch_compile_config=TorchCompileConfig(enable=False),
        cpu_offload_config=ArgsOffloadConfig(
            enable=enable_offload,
            stages=INFER_OFFLOAD_STAGES if enable_offload else None,
        ),
    )
    pipeline = PipelineLoader(args).load(skip_warmup=True)
    assert pipeline.transformer is not None, "Expected a loaded Cosmos3 transformer"
    return pipeline


def _capture_video(pipe) -> torch.Tensor:
    with torch.no_grad():
        result = pipe.forward(
            prompt=INFER_PROMPT,
            negative_prompt=INFER_NEGATIVE_PROMPT,
            height=INFER_HEIGHT,
            width=INFER_WIDTH,
            num_frames=INFER_NUM_FRAMES,
            num_inference_steps=INFER_NUM_STEPS,
            guidance_scale=INFER_GUIDANCE_SCALE,
            seed=INFER_SEED,
        )
    assert result.video is not None, "Pipeline returned no video (guardrail blocked?)"
    return result.video.float() / 255.0


@pytest.fixture
def cosmos3_offload_pipeline():
    pipeline = _make_pipeline(COSMOS3_PATH, enable_offload=True)
    yield pipeline
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.integration
class TestCosmos3Offload:
    """Cosmos3 T2V with CPU offloading of both towers, VAE, and guardrails."""

    def test_cosmos3_offload_forward(self, cosmos3_offload_pipeline):
        pipeline = cosmos3_offload_pipeline
        assert pipeline.offloader.stages(), (
            "cpu_offload_config.enable=True but offloader.stages() is empty"
        )
        assert pipeline.offloader.offload_pipeline is not None, (
            "offload pipeline was not initialized after load"
        )
        _capture_video(pipeline)
        print(f"\n  ===== Offload — Cosmos3 {INFER_HEIGHT}x{INFER_WIDTH} =====")
        print(f"  stages: {pipeline.offloader.stages()}")

    def test_cosmos3_offload_matches_baseline(self):
        offload_pipe = _make_pipeline(COSMOS3_PATH, enable_offload=True)
        offload_video = _capture_video(offload_pipe)
        del offload_pipe
        gc.collect()
        torch.cuda.empty_cache()

        baseline_pipe = _make_pipeline(COSMOS3_PATH, enable_offload=False)
        baseline_video = _capture_video(baseline_pipe)
        del baseline_pipe
        gc.collect()
        torch.cuda.empty_cache()

        assert offload_video.shape == baseline_video.shape, (
            f"shape mismatch offload {offload_video.shape} vs baseline {baseline_video.shape}"
        )
        a_flat = offload_video.float().cpu().reshape(-1)
        b_flat = baseline_video.float().cpu().reshape(-1)
        cos_sim = (
            F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).clamp(-1.0, 1.0).item()
        )
        print(
            f"\n  ===== Offload accuracy — Cosmos3 =====\n"
            f"  cosine_similarity={cos_sim:.6f}  (threshold={COS_SIM_THRESHOLD})"
        )
        assert cos_sim >= COS_SIM_THRESHOLD, (
            f"offload vs baseline cosine similarity {cos_sim:.6f} < {COS_SIM_THRESHOLD}"
        )


@pytest.mark.integration
class TestCosmos3OffloadCudaGraphRaisesError:
    """CUDA graphs plus offloading must raise NotImplementedError on pipeline load."""

    def test_cosmos3_raises_if_cuda_graph_and_offload_enabled(self):
        from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
        from tensorrt_llm.visual_gen.args import CpuOffloadConfig as ArgsOffloadConfig
        from tensorrt_llm.visual_gen.args import CudaGraphConfig, TorchCompileConfig, VisualGenArgs

        if not os.path.exists(COSMOS3_PATH):
            pytest.skip(f"Checkpoint not found: {COSMOS3_PATH}")
        args = VisualGenArgs(
            model=COSMOS3_PATH,
            torch_compile_config=TorchCompileConfig(enable=False),
            cuda_graph_config=CudaGraphConfig(enable=True),
            cpu_offload_config=ArgsOffloadConfig(enable=True, stages=INFER_OFFLOAD_STAGES),
        )
        with pytest.raises(
            NotImplementedError,
            match="CUDA graphs are not supported with visual generation offloading",
        ):
            PipelineLoader(args).load(skip_warmup=True)
