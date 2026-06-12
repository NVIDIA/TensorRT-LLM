# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multi-GPU end-to-end test for the Wan T2V VSA pipeline.

Run with:
    pytest tests/unittest/_torch/visual_gen/multi_gpu/test_wan_vsa_ulysses.py -v -s

Override checkpoint path:
    DIFFUSION_MODEL_PATH_WAN21_VSA=/path/to/vsa \\
        pytest tests/unittest/_torch/visual_gen/multi_gpu/test_wan_vsa_ulysses.py -v -s
"""

import gc
import os
from pathlib import Path
from typing import Callable

os.environ["TLLM_DISABLE_MPI"] = "1"

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

try:
    from tensorrt_llm._torch.visual_gen.attention_backend.cute_dsl import _cute_dsl_import_error
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm._utils import get_free_port
    from tensorrt_llm.visual_gen.args import (
        AttentionConfig,
        ParallelConfig,
        TorchCompileConfig,
        VideoSparseAttentionConfig,
        VisualGenArgs,
    )

    MODULES_AVAILABLE = True
    _cute_dsl_available = _cute_dsl_import_error is None
except ImportError:
    MODULES_AVAILABLE = False
    _cute_dsl_available = False


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# =============================================================================
# Path helpers (mirrors test_wan_vsa_pipeline.py)
# =============================================================================


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


WAN21_VSA_PATH = _checkpoint("DIFFUSION_MODEL_PATH_WAN21_VSA", "Wan2.1-VSA-T2V-14B-720P-Diffusers")


# =============================================================================
# Inference constants
# =============================================================================

PROMPT = "A cat sitting on a sunny windowsill watching birds outside."
NEGATIVE_PROMPT = ""

HEIGHT = 720
WIDTH = 1280
NUM_FRAMES = 9
NUM_STEPS = 4
GUIDANCE_SCALE = 5.0
SEED = 42

COS_SIM_THRESHOLD = 0.97


# =============================================================================
# Distributed harness (mirrors test_wan_pipeline_parallel.py)
# =============================================================================


def init_distributed_worker(rank: int, world_size: int, backend: str = "nccl", port: int = 29500):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def _distributed_worker(rank, world_size, backend, test_fn, port, kwargs):
    try:
        init_distributed_worker(rank, world_size, backend, port)
        test_fn(rank, world_size, **kwargs)
    except Exception as e:
        print(f"Rank {rank} failed with error: {e}")
        raise
    finally:
        cleanup_distributed()


def run_test_in_distributed(world_size: int, test_fn: Callable, **kwargs):
    if not MODULES_AVAILABLE:
        pytest.skip("Required modules not available")
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Test requires {world_size} GPUs, only {torch.cuda.device_count()} available")
    port = get_free_port()
    mp.spawn(
        _distributed_worker,
        args=(world_size, "nccl", test_fn, port, kwargs),
        nprocs=world_size,
        join=True,
    )


# =============================================================================
# Inference helpers
# =============================================================================


VSA_SPARSITY = 0.9


def _build_vsa_parallel_args(checkpoint_path: str) -> "VisualGenArgs":
    """cfg=2, ulysses=4, CUTEDSL backend with vsa_sparsity=0.9."""
    return VisualGenArgs(
        model=checkpoint_path,
        torch_compile_config=TorchCompileConfig(enable=False),
        attention_config=AttentionConfig(
            backend="CUTEDSL",
            sparse_attention_config=VideoSparseAttentionConfig(vsa_sparsity=VSA_SPARSITY),
        ),
        parallel_config=ParallelConfig(cfg_size=2, ulysses_size=4),
    )


def _build_vsa_single_args(checkpoint_path: str) -> "VisualGenArgs":
    """Single-GPU CUTEDSL reference at the same vsa_sparsity=0.9."""
    return VisualGenArgs(
        model=checkpoint_path,
        torch_compile_config=TorchCompileConfig(enable=False),
        attention_config=AttentionConfig(
            backend="CUTEDSL",
            sparse_attention_config=VideoSparseAttentionConfig(vsa_sparsity=VSA_SPARSITY),
        ),
    )


def _capture_trtllm_video(pipeline) -> "torch.Tensor | None":
    """Run full TRTLLM pipeline; return (T, H, W, C) float in [0, 1] or None."""
    with torch.no_grad():
        result = pipeline.forward(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            num_inference_steps=NUM_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            seed=SEED,
        )
    if result is None or result.video is None:
        return None
    return result.video.float() / 255.0


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.float().cpu().reshape(-1)
    b_flat = b.float().cpu().reshape(-1)
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).clamp(-1.0, 1.0).item()


def _free(*objs) -> None:
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


# =============================================================================
# Worker logic (module-level for mp.spawn pickling)
# =============================================================================


def _logic_vsa_cfg2_ulysses4(rank: int, world_size: int, *, checkpoint_path: str) -> None:
    """End-to-end pipeline: 8-GPU (cfg=2, ulysses=4) VSA vs 1-GPU VSA reference.

    Both runs use vsa_sparsity=0.9. All 8 ranks run the parallel denoising loop.
    Rank 0 then frees the distributed model and loads a single-GPU VSA reference
    at the same sparsity to compare.
    """
    assert world_size == 8, f"This test is hardcoded to world_size=8, got {world_size}"

    vsa_pipe = PipelineLoader(_build_vsa_parallel_args(checkpoint_path)).load(skip_warmup=True)
    vsa_video = _capture_trtllm_video(vsa_pipe)

    if rank == 0:
        assert vsa_video is not None, "Rank 0 produced no video from the VSA pipeline."

    _free(vsa_pipe)
    if rank != 0:
        vsa_video = None
    dist.barrier()

    if rank != 0:
        return

    ref_pipe = PipelineLoader(_build_vsa_single_args(checkpoint_path)).load(skip_warmup=True)
    ref_video = _capture_trtllm_video(ref_pipe)
    _free(ref_pipe)

    assert vsa_video.numel() == ref_video.numel(), (
        f"Element count mismatch — 8-GPU {tuple(vsa_video.shape)} "
        f"({vsa_video.numel()}) vs 1-GPU {tuple(ref_video.shape)} ({ref_video.numel()})"
    )

    cos_sim = _cosine_similarity(vsa_video, ref_video)
    print(
        f"\n  Wan2.1-VSA-T2V-14B (cfg=2, ulysses=4, sparsity={VSA_SPARSITY}) "
        f"cosine similarity vs 1-GPU VSA: {cos_sim:.6f}"
    )
    assert cos_sim >= COS_SIM_THRESHOLD, (
        f"8-GPU VSA pipeline diverges from 1-GPU VSA reference: "
        f"cosine similarity {cos_sim:.6f} < {COS_SIM_THRESHOLD}. "
        f"Shapes — 8-GPU: {tuple(vsa_video.shape)}, 1-GPU: {tuple(ref_video.shape)}."
    )


# =============================================================================
# Test class
# =============================================================================


@pytest.mark.integration
@pytest.mark.wan_t2v
class TestWanVsaUlysses:
    """Multi-GPU correctness for the Wan T2V VSA pipeline (cfg=2, ulysses=4, 8 GPUs)."""

    def test_cfg2_ulysses4(self):
        """world=8, cfg=2, ulysses=4, VSA sparsity=0.9 vs 1-GPU VSA reference."""
        if not MODULES_AVAILABLE:
            pytest.skip("Required modules not available")
        if not _cute_dsl_available:
            pytest.skip(f"CUTEDSL not available (requires Blackwell GPU): {_cute_dsl_import_error}")
        if not os.path.exists(WAN21_VSA_PATH):
            pytest.skip(
                f"Checkpoint not found: {WAN21_VSA_PATH}. Set DIFFUSION_MODEL_PATH_WAN21_VSA."
            )
        run_test_in_distributed(
            world_size=8,
            test_fn=_logic_vsa_cfg2_ulysses4,
            checkpoint_path=WAN21_VSA_PATH,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
