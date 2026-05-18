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
"""Multi-GPU end-to-end test for the Wan T2V pipeline.

Run with:
    pytest tests/unittest/_torch/visual_gen/multi_gpu/test_wan_pipeline_parallel.py -v -s

Override checkpoint path:
    DIFFUSION_MODEL_PATH_WAN21_1_3B=/path/to/1.3b \\
        pytest tests/unittest/_torch/visual_gen/multi_gpu/test_wan_pipeline_parallel.py -v -s
"""

import gc
import os
from pathlib import Path
from typing import Callable

os.environ["TLLM_DISABLE_MPI"] = "1"

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

try:
    from diffusers import DiffusionPipeline

    from tensorrt_llm._torch.visual_gen.config import (
        ParallelConfig,
        TorchCompileConfig,
        VisualGenArgs,
    )
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm._utils import get_free_port

    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# =============================================================================
# Path helpers (mirror tests/unittest/_torch/visual_gen/test_wan21_t2v_pipeline.py)
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


WAN21_1_3B_PATH = _checkpoint("DIFFUSION_MODEL_PATH_WAN21_1_3B", "Wan2.1-T2V-1.3B-Diffusers")


# =============================================================================
# Inference constants
# =============================================================================

PROMPT = "A cat sitting on a sunny windowsill watching birds outside."
NEGATIVE_PROMPT = ""

HEIGHT = 480
WIDTH = 832
NUM_FRAMES = 9
NUM_STEPS = 4
GUIDANCE_SCALE = 5.0
SEED = 42

# Relaxed vs the 0.99 single-GPU threshold: cross-rank CFG all-gather, Ulysses
# all-to-all, and parallel-VAE halo exchange add bf16 reduction-order noise.
COS_SIM_THRESHOLD = 0.97


# =============================================================================
# Distributed harness (mirrors tests/.../multi_gpu/test_wan_attn2d.py)
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


def _build_parallel_args(checkpoint_path: str) -> "VisualGenArgs":
    """Build VisualGenArgs with cfg=2, ulysses=2, parallel_vae=2."""
    return VisualGenArgs(
        checkpoint_path=checkpoint_path,
        device="cuda",
        dtype="bfloat16",
        torch_compile=TorchCompileConfig(enable_torch_compile=False),
        parallel=ParallelConfig(
            dit_cfg_size=2,
            dit_ulysses_size=2,
            parallel_vae_size=2,
            parallel_vae_split_dim="width",
        ),
    )


def _capture_trtllm_video(
    pipeline,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
):
    """Run full TRTLLM pipeline (text -> denoise -> parallel VAE decode).

    Returns (T, H, W, C) float in [0, 1] on the calling rank if it owns the
    VAE decode, else ``None``. With parallel-VAE enabled, every rank in
    ``vae_ranks`` ends up with the gathered full-resolution video, so on the
    full-world config used here every rank gets a video.
    """
    with torch.no_grad():
        result = pipeline.forward(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
    if result is None or result.video is None:
        return None
    video = result.video  # (T, H, W, C) uint8
    return video.float() / 255.0


def _capture_hf_video(
    hf_pipe,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
) -> torch.Tensor:
    """Single-GPU HF reference. Returns (T, H, W, C) float in [0, 1]."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    output = hf_pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="np",
    )
    frames = output.frames  # (1, T, H, W, C) np.float32 in [0, 1]
    if isinstance(frames, np.ndarray):
        return torch.from_numpy(frames[0]).float()
    return frames[0].float()


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


def _logic_wan_cfg_ulysses_pvae(rank: int, world_size: int, *, checkpoint_path: str) -> None:
    """End-to-end pipeline run with cfg=2, ulysses=2, parallel_vae=2.

    All ranks participate in the TRTLLM forward (CFG parallel, Ulysses, parallel
    VAE). Only rank 0 holds onto the video and runs the HF reference for the
    cosine-similarity assertion; other ranks free state and exit.
    """
    assert world_size == 4, f"This test is hardcoded to world_size=4, got {world_size}"

    trtllm_pipe = PipelineLoader(_build_parallel_args(checkpoint_path)).load(skip_warmup=True)
    trtllm_video = _capture_trtllm_video(
        trtllm_pipe,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        seed=SEED,
    )

    # With parallel_vae_size == world_size, rank 0 is in vae_ranks and must
    # have a decoded video. Surface a clearer error than a downstream None deref.
    if rank == 0:
        assert trtllm_video is not None, (
            "Rank 0 unexpectedly produced no video — parallel VAE decode ownership is broken."
        )

    # Free TRTLLM state on every rank before rank 0 spins up HF on cuda:0.
    _free(trtllm_pipe)
    if rank != 0:
        trtllm_video = None  # not used; release the reference
    dist.barrier()

    if rank != 0:
        return

    hf_pipe = DiffusionPipeline.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16)
    hf_pipe = hf_pipe.to("cuda")
    hf_pipe.set_progress_bar_config(disable=True)
    hf_video = _capture_hf_video(
        hf_pipe,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        seed=SEED,
    )
    _free(hf_pipe)

    assert trtllm_video.numel() == hf_video.numel(), (
        f"Element count mismatch — TRTLLM {tuple(trtllm_video.shape)} "
        f"({trtllm_video.numel()}) vs HF {tuple(hf_video.shape)} ({hf_video.numel()})"
    )

    cos_sim = _cosine_similarity(trtllm_video, hf_video)
    print(
        f"\n  Wan2.1-T2V-1.3B (cfg=2, ulysses=2, parallel_vae=2) cosine similarity: {cos_sim:.6f}"
    )
    assert cos_sim >= COS_SIM_THRESHOLD, (
        f"Multi-GPU TRTLLM pipeline diverges from HF reference: "
        f"cosine similarity {cos_sim:.6f} < {COS_SIM_THRESHOLD}. "
        f"Shapes — TRTLLM: {tuple(trtllm_video.shape)}, HF: {tuple(hf_video.shape)}."
    )


# =============================================================================
# Test class
# =============================================================================


@pytest.mark.integration
@pytest.mark.wan_t2v
class TestWanPipelineParallel:
    """Multi-GPU correctness for the Wan T2V pipeline with combined parallelism."""

    def test_cfg2_ulysses2_pvae2(self):
        """world=4, cfg=2, ulysses=2, parallel_vae=2 vs HF reference."""
        if not MODULES_AVAILABLE:
            pytest.skip("Required modules not available")
        if not os.path.exists(WAN21_1_3B_PATH):
            pytest.skip(
                f"Checkpoint not found: {WAN21_1_3B_PATH}. Set DIFFUSION_MODEL_PATH_WAN21_1_3B."
            )
        run_test_in_distributed(
            world_size=4,
            test_fn=_logic_wan_cfg_ulysses_pvae,
            checkpoint_path=WAN21_1_3B_PATH,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
