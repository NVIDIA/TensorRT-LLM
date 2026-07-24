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
"""Multi-GPU integration tests for VisualGen models."""

import glob
import os
import sys
from typing import Callable

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from defs.examples.visual_gen.test_visual_gen import (
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


def _parallel_config(**kwargs):
    # Imported lazily so that mp.spawn child processes resolve tensorrt_llm only after
    # _distributed_worker has prepended the installed-wheel location to sys.path. A
    # module-level tensorrt_llm import would run during the child's module re-import
    # (before sys.path is fixed) and resolve to the bindings-less source tree.
    from tensorrt_llm.visual_gen.args import ParallelConfig

    return ParallelConfig(**kwargs)


# Keep it as 0.25 as the worst case scenario at NVL72 scale
WAN_MULTI_GPU_LPIPS_THRESHOLD = 0.25
WAN22_MULTI_GPU_LPIPS_ATTENTION_BACKEND = "FA4"
WAN22_MULTI_GPU_LPIPS_GOLDEN_VIDEO = "wan22_t2v_fa4_fully_eager_lpips_golden_video.mp4"
WAN22_LPIPS_MULTI_GPU_VARIANTS = [
    ("ulysses4", {"ulysses_size": 4}),
    ("cfg2_ulysses2", {"cfg_size": 2, "ulysses_size": 2}),
    ("attn2d_2x2", {"attn2d_size": (2, 2)}),
    ("cfg2_ulysses2_attn2d_2x1", {"cfg_size": 2, "ulysses_size": 2, "attn2d_size": (2, 1)}),
    ("attn2d_2x2_ulysses2", {"attn2d_size": (2, 2), "ulysses_size": 2}),
]

WAN22_LPIPS_TP_VARIANTS = [
    ("tp2", {"tp_size": 2}),
    ("tp3", {"tp_size": 3}),
    ("cfg2_tp2", {"cfg_size": 2, "tp_size": 2}),
    ("tp2_ulysses2", {"tp_size": 2, "ulysses_size": 2}),
]

QWEN_IMAGE_ATTENTION_PARALLEL_VARIANTS = [
    ("tp2", {"tp_size": 2}, "VANILLA", "tp"),
    ("ring2_ulysses2", {"ring_size": 2, "ulysses_size": 2}, "FA4", "ring"),
    (
        "attn2d_2x1_ulysses2",
        {"attn2d_size": (2, 1), "ulysses_size": 2},
        "FA4",
        "attn2d",
    ),
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


def _validated_tllm_site(site_dir):
    """Return the realpath of ``site_dir`` after verifying it holds the installed wheel.

    The spawn workers rely on this directory to import tensorrt_llm with compiled
    bindings; accepting an arbitrary path would let the import silently fall through
    to the bindings-less source tree, so reject anything that does not contain the
    package plus its compiled bindings extension.
    """
    resolved = os.path.realpath(site_dir) if site_dir else ""
    package_init = os.path.join(resolved, "tensorrt_llm", "__init__.py")
    bindings = glob.glob(os.path.join(resolved, "tensorrt_llm", "bindings*.so")) + glob.glob(
        os.path.join(resolved, "tensorrt_llm", "bindings", "*.so")
    )
    if not (resolved and os.path.isfile(package_init) and bindings):
        raise RuntimeError(
            f"tllm_site={site_dir!r} does not contain an installed tensorrt_llm package "
            "with compiled bindings; spawn workers would import the bindings-less "
            "source tree instead of the wheel."
        )
    return resolved


def _distributed_worker(rank, world_size, backend, test_fn, port, kwargs, tllm_site):
    # mp.spawn starts a fresh interpreter whose sys.path (set up by the integration
    # `defs` harness) puts the source checkout ahead of the installed wheel, so a bare
    # `import tensorrt_llm` would resolve to the bindings-less source tree and crash the
    # worker. Prepend the parent's installed-package location so the child imports
    # tensorrt_llm (with compiled bindings) from the wheel before any such import.
    tllm_site = _validated_tllm_site(tllm_site)
    sys.path[:] = [path for path in sys.path if os.path.realpath(path) != tllm_site]
    sys.path.insert(0, tllm_site)
    try:
        init_distributed_worker(rank, world_size, backend, port)
        test_fn(rank, world_size, **kwargs)
    except Exception as e:
        print(f"Rank {rank} failed with error: {e}")
        raise
    finally:
        cleanup_distributed()


def run_test_in_distributed(world_size: int, test_fn: Callable, use_cuda: bool = True, **kwargs):
    try:
        import tensorrt_llm.bindings as tllm_bindings
        from tensorrt_llm._utils import get_free_port
    except ImportError:
        pytest.skip("Required modules not available")
    if use_cuda and torch.cuda.device_count() < world_size:
        pytest.skip(f"Test requires {world_size} GPUs, only {torch.cuda.device_count()} available")
    backend = "nccl" if use_cuda else "gloo"
    port = get_free_port()
    # Directory containing the installed tensorrt_llm package (i.e. site-packages),
    # passed to spawn workers so they prepend it to sys.path and import the wheel with
    # compiled bindings instead of the source-tree package. Validated here as well so a
    # bad environment fails before any worker is spawned.
    tllm_site = _validated_tllm_site(
        os.path.dirname(os.path.dirname(os.path.abspath(tllm_bindings.__file__)))
    )
    mp.spawn(
        _distributed_worker,
        args=(world_size, backend, test_fn, port, kwargs, tllm_site),
        nprocs=world_size,
        join=True,
    )


def _skip_if_insufficient_gpus_for_parallel(parallel):
    parallel_cfg = _parallel_config(**parallel)
    required = parallel_cfg.n_workers
    available = torch.cuda.device_count()
    if available < required:
        pytest.skip(
            f"Insufficient GPUs for parallel={parallel}: requires {required}, available {available}"
        )


def _qwen_image_attention_distributed_worker(rank: int, world_size: int, **kwargs) -> None:
    from tensorrt_llm._torch.visual_gen.attention_backend.parallel import (
        Attention2DAttention,
        RingAttention,
        UlyssesAttention,
    )
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
    from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
    from tensorrt_llm._torch.visual_gen.models.qwen_image import QwenJointAttention
    from tensorrt_llm.visual_gen.args import AttentionConfig

    parallel_cfg = _parallel_config(**kwargs["parallel"])
    parallel_cfg.validate_world_size(world_size)
    attn2d_row_size, attn2d_col_size = parallel_cfg.attn2d_size
    visual_gen_mapping = VisualGenMapping(
        world_size=world_size,
        rank=rank,
        tp_size=parallel_cfg.tp_size,
        ring_size=parallel_cfg.ring_size,
        ulysses_size=parallel_cfg.ulysses_size,
        attn2d_row_size=attn2d_row_size,
        attn2d_col_size=attn2d_col_size,
    )
    config = DiffusionModelConfig(
        mapping=visual_gen_mapping.to_llm_mapping(),
        visual_gen_mapping=visual_gen_mapping,
        attention=AttentionConfig(backend=kwargs["backend"]),
        parallel=parallel_cfg,
    )
    attention = QwenJointAttention(
        dim=256,
        num_attention_heads=4,
        attention_head_dim=64,
        config=config,
    ).cuda()

    torch.manual_seed(11)
    with torch.no_grad():
        for name, parameter in attention.named_parameters():
            if name.endswith("bias"):
                parameter.zero_()
            elif "norm" in name and name.endswith("weight"):
                parameter.fill_(1)
            else:
                parameter.normal_(mean=0.0, std=0.02)

    topology = kwargs["topology"]
    if topology == "tp":
        assert not isinstance(attention.attn, UlyssesAttention)
    else:
        assert isinstance(attention.attn, UlyssesAttention)
        if topology == "ring":
            assert isinstance(attention.attn.inner_backend, RingAttention)
        else:
            assert isinstance(attention.attn.inner_backend, Attention2DAttention)

    hidden_states = torch.randn(1, 8, 256, device="cuda", dtype=torch.bfloat16)
    encoder_hidden_states = torch.randn(1, 4, 256, device="cuda", dtype=torch.bfloat16)
    image_output, text_output = attention(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
    )

    assert image_output.shape == hidden_states.shape
    assert text_output.shape == encoder_hidden_states.shape
    assert torch.isfinite(image_output).all()
    assert torch.isfinite(text_output).all()
    dist.barrier()


@pytest.mark.parametrize(
    "variant_name,parallel,backend,topology",
    QWEN_IMAGE_ATTENTION_PARALLEL_VARIANTS,
    ids=[name for name, *_ in QWEN_IMAGE_ATTENTION_PARALLEL_VARIANTS],
)
def test_qwen_image_attention_parallel_topologies(variant_name, parallel, backend, topology):
    _skip_if_insufficient_gpus_for_parallel(parallel)
    parallel_cfg = _parallel_config(**parallel)
    run_test_in_distributed(
        world_size=parallel_cfg.n_workers,
        test_fn=_qwen_image_attention_distributed_worker,
        parallel=parallel,
        backend=backend,
        topology=topology,
    )


def _wan22_lpips_distributed_worker(rank: int, world_size: int, **kwargs) -> None:
    parallel = kwargs["parallel"]
    _parallel_config(**parallel).validate_world_size(world_size)

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
        attention_backend=WAN22_MULTI_GPU_LPIPS_ATTENTION_BACKEND,
        parallel=parallel,
        fully_eager=True,
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
    _skip_if_insufficient_gpus_for_parallel(parallel)
    parallel_cfg = _parallel_config(**parallel)
    generated_path = tmp_path / f"wan22_t2v_generated_{variant_name}.mp4"
    golden_path = _golden_media_path(
        tmp_path,
        WAN22_MULTI_GPU_LPIPS_GOLDEN_VIDEO,
        "Wan 2.2 FA4 fully-eager LPIPS golden video",
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


@pytest.mark.parametrize(
    "variant_name,parallel",
    WAN22_LPIPS_MULTI_GPU_VARIANTS,
    ids=[name for name, _ in WAN22_LPIPS_MULTI_GPU_VARIANTS],
)
def test_wan22_t2v_lpips_against_golden_multi_gpu(
    _visual_gen_deps, tmp_path, variant_name, parallel
):
    _run_wan22_t2v_lpips_case(tmp_path, variant_name, parallel)


@pytest.mark.parametrize(
    "variant_name,parallel",
    WAN22_LPIPS_TP_VARIANTS,
    ids=[name for name, _ in WAN22_LPIPS_TP_VARIANTS],
)
def test_wan22_t2v_lpips_against_golden_tp(_visual_gen_deps, tmp_path, variant_name, parallel):
    _run_wan22_t2v_lpips_case(tmp_path, variant_name, parallel)
