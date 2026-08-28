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

import glob
import os
import sys
from typing import Callable

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from defs.examples.visual_gen.visual_gen_test_utils import (
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
    _skip_if_missing,
)


def _parallel_config(**kwargs):
    # Imported lazily so that mp.spawn child processes resolve tensorrt_llm only after
    # _distributed_worker has prepended the installed-wheel location to sys.path. A
    # module-level tensorrt_llm import would run during the child's module re-import
    # (before sys.path is fixed) and resolve to the bindings-less source tree.
    from tensorrt_llm.visual_gen.args import ParallelConfig

    return ParallelConfig(**kwargs)


# Primary gate: within-build parallelism check. The invariant these tests
# protect is "parallelism does not change the output", so every variant is
# scored against a fully-eager single-GPU reference generated in this test
# session at the current build (wan22_within_build_reference fixture). Both
# sides of that comparison shift together under benign whole-build numerics
# changes -- e.g. PR #17693 moved GELU onto the cuBLASLt fp32 accumulator and
# shifted the entire bf16 rounding trajectory, stepping the frozen-golden
# score of [attn2d_2x2] 0.2243 -> 0.2795 (nvbug 6655990) with no quality
# change -- so this gate fails only when parallelism itself changes the
# output relative to the same build's single-GPU run.
#
# Calibration (4xB200, nvbug 6655990 repro workspace; rc24 image +- the
# #17693 mlp.py hunks, torch 2.12.0a0+...nv26.05) showed the within-build
# scores are bimodal, so each variant carries its class threshold:
#   * EXACT class -- CFG splitting, Ulysses head repartition, and the attn2d
#     head-dim split reproduce the single-GPU output bit-exactly (LPIPS
#     0.000000 measured for ulysses4, cfg2_ulysses2 and
#     cfg2_ulysses2_attn2d_2x1; GEMM rows and attention heads are
#     reduction-order invariant under these decompositions). 0.05 keeps
#     margin for codec jitter while any real defect lands at 0.2+.
#   * REDUCTION-REORDERING class -- TP-split GEMMs (allreduce) and the
#     attn2d sequence-KV split change floating-point reduction order; the
#     one-ULP seed amplifies over the denoising steps to a saturation band
#     (measured: tp2-family 0.2098, attn2d_2x2-family 0.2597, tp3 0.2618;
#     stable composition -- cfg/ulysses/attn2d-head add exactly 0 on top).
#     0.32 gives ~0.06 headroom over the measured worst case; genuinely
#     broken output (wrong seed control run) measures far above it.
WAN_MULTI_GPU_EXACT_WITHIN_BUILD_LPIPS_THRESHOLD = 0.05
WAN_MULTI_GPU_REORDERED_WITHIN_BUILD_LPIPS_THRESHOLD = 0.32
# Backstop gate: loose catastrophic-quality bound vs the frozen golden. The
# frozen golden drifts away from the current trajectory whenever bf16
# numerics legitimately change (even the single-GPU fully-eager run -- the
# golden's own configuration -- measures 0.2223 against it on the current
# trajectory), so this bound intentionally has large headroom and exists
# only to catch outputs that are far from everything.
WAN_MULTI_GPU_GOLDEN_BACKSTOP_LPIPS_THRESHOLD = 0.32
WAN22_MULTI_GPU_LPIPS_ATTENTION_BACKEND = "FA4"
WAN22_MULTI_GPU_LPIPS_GOLDEN_VIDEO = "wan22_t2v_fa4_fully_eager_lpips_golden_video.mp4"
# (variant name, parallel kwargs, within-build LPIPS bound) -- pick the bound
# per the class rationale above when adding a variant.
WAN22_LPIPS_MULTI_GPU_VARIANTS = [
    ("ulysses4", {"ulysses_size": 4}, WAN_MULTI_GPU_EXACT_WITHIN_BUILD_LPIPS_THRESHOLD),
    (
        "cfg2_ulysses2",
        {"cfg_size": 2, "ulysses_size": 2},
        WAN_MULTI_GPU_EXACT_WITHIN_BUILD_LPIPS_THRESHOLD,
    ),
    ("attn2d_2x2", {"attn2d_size": (2, 2)}, WAN_MULTI_GPU_REORDERED_WITHIN_BUILD_LPIPS_THRESHOLD),
    (
        "cfg2_ulysses2_attn2d_2x1",
        {"cfg_size": 2, "ulysses_size": 2, "attn2d_size": (2, 1)},
        WAN_MULTI_GPU_EXACT_WITHIN_BUILD_LPIPS_THRESHOLD,
    ),
    (
        "attn2d_2x2_ulysses2",
        {"attn2d_size": (2, 2), "ulysses_size": 2},
        WAN_MULTI_GPU_REORDERED_WITHIN_BUILD_LPIPS_THRESHOLD,
    ),
]

WAN22_LPIPS_TP_VARIANTS = [
    ("tp2", {"tp_size": 2}, WAN_MULTI_GPU_REORDERED_WITHIN_BUILD_LPIPS_THRESHOLD),
    ("tp3", {"tp_size": 3}, WAN_MULTI_GPU_REORDERED_WITHIN_BUILD_LPIPS_THRESHOLD),
    (
        "cfg2_tp2",
        {"cfg_size": 2, "tp_size": 2},
        WAN_MULTI_GPU_REORDERED_WITHIN_BUILD_LPIPS_THRESHOLD,
    ),
    (
        "tp2_ulysses2",
        {"tp_size": 2, "ulysses_size": 2},
        WAN_MULTI_GPU_REORDERED_WITHIN_BUILD_LPIPS_THRESHOLD,
    ),
    (
        "tp2_attn2d_2x1",
        {"tp_size": 2, "attn2d_size": (2, 1)},
        WAN_MULTI_GPU_REORDERED_WITHIN_BUILD_LPIPS_THRESHOLD,
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


def _wan22_reference_media_worker(rank, kwargs, tllm_site):
    # mp.spawn target: generate the fully-eager single-GPU reference in a child
    # process so the pytest parent stays CUDA-free for the distributed spawns.
    # Applies the same installed-wheel sys.path fix as _distributed_worker; no
    # torch.distributed init — this is the plain single-GPU pipeline path.
    tllm_site = _validated_tllm_site(tllm_site)
    sys.path[:] = [path for path in sys.path if os.path.realpath(path) != tllm_site]
    sys.path.insert(0, tllm_site)
    torch.cuda.set_device(0)
    video = _run_wan_lpips_pipeline(
        kwargs["model_path"],
        WAN22_LPIPS_PROMPT,
        WAN22_LPIPS_NEGATIVE_PROMPT,
        WAN22_LPIPS_HEIGHT,
        WAN22_LPIPS_WIDTH,
        WAN22_LPIPS_NUM_FRAMES,
        WAN22_LPIPS_NUM_INFERENCE_STEPS,
        WAN22_LPIPS_GUIDANCE_SCALE,
        WAN22_LPIPS_SEED,
        attention_backend=WAN22_MULTI_GPU_LPIPS_ATTENTION_BACKEND,
        parallel=None,
        fully_eager=True,
    )
    assert video is not None, "Single-GPU within-build reference run produced no video"
    _save_lpips_video_mp4(video, kwargs["reference_path"], frame_rate=WAN22_LPIPS_FRAME_RATE)


@pytest.fixture(scope="session")
def wan22_within_build_reference(tmp_path_factory):
    """Fully-eager single-GPU WAN2.2 reference video cut at the current build.

    Session-scoped: one extra generation amortized over every multi-GPU/TP
    variant in the session. Scoring each variant against this reference
    (instead of only the frozen golden) isolates the parallelism invariant
    from whole-build numerics drift: both sides of the comparison shift
    together when bf16 numerics legitimately change (nvbug 6655990).
    """
    try:
        import tensorrt_llm.bindings as tllm_bindings
    except ImportError:
        pytest.skip("Required modules not available")
    if torch.cuda.device_count() < 1:
        pytest.skip("Within-build reference generation requires a GPU")
    model_path = _lpips_model_path("Wan2.2-T2V-A14B-Diffusers")
    _skip_if_missing(model_path, "Wan checkpoint", is_dir=True)
    tllm_site = _validated_tllm_site(
        os.path.dirname(os.path.dirname(os.path.abspath(tllm_bindings.__file__)))
    )
    reference_path = (
        tmp_path_factory.mktemp("wan22_within_build_ref")
        / "wan22_t2v_fa4_fully_eager_within_build_reference.mp4"
    )
    mp.spawn(
        _wan22_reference_media_worker,
        args=({"model_path": model_path, "reference_path": str(reference_path)}, tllm_site),
        nprocs=1,
        join=True,
    )
    assert reference_path.is_file(), f"Reference generation did not produce {reference_path}"
    return reference_path


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


def _run_wan22_t2v_lpips_case(
    tmp_path, variant_name, parallel, within_build_reference, within_build_threshold
):
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
    # Compute both scores before asserting so a failure log always carries the
    # full picture (which gate tripped, and where the other one stood).
    within_build_score = _run_lpips_eval(
        tmp_path,
        f"wan22_t2v_{variant_name}_within_build",
        "video",
        WAN22_LPIPS_PROMPT,
        within_build_reference,
        generated_path,
    )
    golden_score = _run_lpips_eval(
        tmp_path,
        f"wan22_t2v_{variant_name}",
        "video",
        WAN22_LPIPS_PROMPT,
        golden_path,
        generated_path,
    )
    _assert_lpips_below_threshold(
        within_build_score,
        within_build_threshold,
        label=f"{variant_name} vs within-build single-GPU reference (parallelism gate)",
    )
    _assert_lpips_below_threshold(
        golden_score,
        WAN_MULTI_GPU_GOLDEN_BACKSTOP_LPIPS_THRESHOLD,
        label=f"{variant_name} vs frozen golden (catastrophic backstop)",
    )


@pytest.mark.parametrize(
    "variant_name,parallel,within_build_threshold",
    WAN22_LPIPS_MULTI_GPU_VARIANTS,
    ids=[variant[0] for variant in WAN22_LPIPS_MULTI_GPU_VARIANTS],
)
def test_wan22_t2v_lpips_against_golden_multi_gpu(
    _visual_gen_deps,
    tmp_path,
    variant_name,
    parallel,
    within_build_threshold,
    wan22_within_build_reference,
):
    # Test name kept (test-db lists and waives.txt reference it); the primary
    # gate is now the within-build parallelism check, with the frozen golden
    # retained as a loose catastrophic backstop.
    _run_wan22_t2v_lpips_case(
        tmp_path, variant_name, parallel, wan22_within_build_reference, within_build_threshold
    )


@pytest.mark.parametrize(
    "variant_name,parallel,within_build_threshold",
    WAN22_LPIPS_TP_VARIANTS,
    ids=[variant[0] for variant in WAN22_LPIPS_TP_VARIANTS],
)
def test_wan22_t2v_lpips_against_golden_tp(
    _visual_gen_deps,
    tmp_path,
    variant_name,
    parallel,
    within_build_threshold,
    wan22_within_build_reference,
):
    _run_wan22_t2v_lpips_case(
        tmp_path, variant_name, parallel, wan22_within_build_reference, within_build_threshold
    )
