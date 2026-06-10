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
"""Multi-GPU integration tests for VisualGen LPIPS quality checks."""

import os
from typing import Callable

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from defs.examples.test_visual_gen import (
    WAN22_LPIPS_FRAME_RATE,
    WAN22_LPIPS_GUIDANCE_SCALE,
    WAN22_LPIPS_HEIGHT,
    WAN22_LPIPS_NEGATIVE_PROMPT,
    WAN22_LPIPS_NUM_FRAMES,
    WAN22_LPIPS_NUM_INFERENCE_STEPS,
    WAN22_LPIPS_PROMPT,
    WAN22_LPIPS_SEED,
    WAN22_LPIPS_WIDTH,
    _assert_lpips_below_threshold,
    _golden_media_path,
    _lpips_model_path,
    _run_lpips_eval,
    _run_wan_lpips_pipeline,
    _save_lpips_video_mp4,
)

try:
    from tensorrt_llm._utils import get_free_port
    from tensorrt_llm.visual_gen.args import ParallelConfig

    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False

# Keep it as 0.25 as the worst case scenario at NVL72 scale
WAN_MULTI_GPU_LPIPS_THRESHOLD = 0.25
WAN22_LPIPS_MULTI_GPU_VARIANTS = [
    ("ulysses4", {"ulysses_size": 4}),
    ("cfg2_ulysses2", {"cfg_size": 2, "ulysses_size": 2}),
    ("ulysses2_ring2", {"ulysses_size": 2, "ring_size": 2}),
    ("attn2d_2x2", {"attn2d_size": (2, 2)}),
    ("cfg2_ulysses2_ring2", {"cfg_size": 2, "ulysses_size": 2, "ring_size": 2}),
    ("attn2d_2x2_ulysses2", {"attn2d_size": (2, 2), "ulysses_size": 2}),
]

# Tensor Parallel variants (tp_size is counted in n_workers, the launcher world size).
WAN22_LPIPS_TP_VARIANTS = [
    ("tp2", {"tp_size": 2}),
    ("cfg2_tp2", {"cfg_size": 2, "tp_size": 2}),
    ("tp2_ulysses2", {"tp_size": 2, "ulysses_size": 2}),
]


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# =============================================================================
# Distributed helpers (mirrors unittest multi_gpu harness)
# =============================================================================


def init_distributed_worker(rank: int, world_size: int, backend: str = "nccl", port: int = 29500):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["TLLM_DISABLE_MPI"] = "1"
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


def run_test_in_distributed(world_size: int, test_fn: Callable, use_cuda: bool = True, **kwargs):
    if not MODULES_AVAILABLE:
        pytest.skip("Required modules not available")
    if use_cuda and torch.cuda.device_count() < world_size:
        pytest.skip(f"Test requires {world_size} GPUs, only {torch.cuda.device_count()} available")
    backend = "nccl" if use_cuda else "gloo"
    port = get_free_port()
    mp.spawn(
        _distributed_worker,
        args=(world_size, backend, test_fn, port, kwargs),
        nprocs=world_size,
        join=True,
    )


def _skip_if_insufficient_gpus_for_parallel(parallel):
    # Use n_workers (the actual VisualGen launcher world size); total_parallel_size omits
    # tp_size and would under-count TP configs.
    parallel_cfg = ParallelConfig(**parallel)
    required = parallel_cfg.n_workers
    available = torch.cuda.device_count()
    if available < required:
        pytest.skip(
            f"Insufficient GPUs for parallel={parallel}: requires {required}, available {available}"
        )


def _wan22_lpips_distributed_worker(rank: int, world_size: int, **kwargs) -> None:
    parallel = kwargs["parallel"]
    ParallelConfig(**parallel).validate_world_size(world_size)

    generated_video = _run_wan_lpips_pipeline(
        kwargs["model_path"],
        kwargs["prompt"],
        kwargs["negative_prompt"],
        kwargs["height"],
        kwargs["width"],
        kwargs["num_frames"],
        kwargs["num_inference_steps"],
        kwargs["guidance_scale"],
        kwargs["seed"],
        attention_backend="FA4",
        parallel=parallel,
    )

    if rank == 0:
        assert generated_video is not None, (
            "Rank 0 produced no video — distributed Wan LPIPS decode ownership is broken."
        )
        _save_lpips_video_mp4(
            generated_video,
            kwargs["generated_path"],
            frame_rate=kwargs["frame_rate"],
        )

    if dist.is_initialized():
        dist.barrier()


def _run_wan22_t2v_lpips_case(tmp_path, variant_name, parallel):
    """Shared body: run the real Wan 2.2 T2V pipeline under `parallel` and compare the
    generated video against the golden via LPIPS. World size is n_workers (the launcher
    world size), so it covers TP, CFG, Ulysses, Ring and Attention2D uniformly.
    """
    _skip_if_insufficient_gpus_for_parallel(parallel)
    parallel_cfg = ParallelConfig(**parallel)
    generated_path = tmp_path / f"wan22_t2v_generated_{variant_name}.mp4"
    golden_path = _golden_media_path(
        tmp_path, "wan22_t2v_lpips_golden_video.mp4", "Wan 2.2 LPIPS golden video"
    )

    run_test_in_distributed(
        world_size=parallel_cfg.n_workers,
        test_fn=_wan22_lpips_distributed_worker,
        model_path=_lpips_model_path("Wan2.2-T2V-A14B-Diffusers"),
        generated_path=str(generated_path),
        prompt=WAN22_LPIPS_PROMPT,
        negative_prompt=WAN22_LPIPS_NEGATIVE_PROMPT,
        height=WAN22_LPIPS_HEIGHT,
        width=WAN22_LPIPS_WIDTH,
        num_frames=WAN22_LPIPS_NUM_FRAMES,
        num_inference_steps=WAN22_LPIPS_NUM_INFERENCE_STEPS,
        guidance_scale=WAN22_LPIPS_GUIDANCE_SCALE,
        seed=WAN22_LPIPS_SEED,
        frame_rate=WAN22_LPIPS_FRAME_RATE,
        parallel=parallel,
    )

    assert generated_path.is_file(), f"Distributed run did not produce {generated_path}"
    score = _run_lpips_eval(
        tmp_path,
        f"wan22_t2v_{variant_name}",
        "video",
        WAN22_LPIPS_PROMPT,
        golden_path,
        generated_path,
    )
    _assert_lpips_below_threshold(score, WAN_MULTI_GPU_LPIPS_THRESHOLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "variant_name,parallel",
    WAN22_LPIPS_MULTI_GPU_VARIANTS,
    ids=[name for name, _ in WAN22_LPIPS_MULTI_GPU_VARIANTS],
)
def test_wan22_t2v_lpips_against_golden_multi_gpu(tmp_path, variant_name, parallel):
    _run_wan22_t2v_lpips_case(tmp_path, variant_name, parallel)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "variant_name,parallel",
    WAN22_LPIPS_TP_VARIANTS,
    ids=[name for name, _ in WAN22_LPIPS_TP_VARIANTS],
)
def test_wan22_t2v_lpips_against_golden_tp(tmp_path, variant_name, parallel):
    """End-to-end Tensor Parallel quality check: exercises the real VisualGenArgs ->
    PipelineLoader -> checkpoint load -> generate path with tp_size > 1 (alone and
    combined with CFG / Ulysses), which the synthetic module-parity unit tests
    (test_*_tp.py) do not cover.
    """
    _run_wan22_t2v_lpips_case(tmp_path, variant_name, parallel)
