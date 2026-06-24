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
import shutil
import subprocess
import sys
from pathlib import Path
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
    ("tp2_attn2d_2x1", {"tp_size": 2, "attn2d_size": (2, 1)}),
]

WAN22_LPIPS_MULTINODE_WORLD_SIZE = 16
WAN22_LPIPS_MULTINODE_NODES = 2
WAN22_LPIPS_MULTINODE_GPUS_PER_NODE = 8
WAN22_LPIPS_MULTINODE_VARIANTS = [
    (
        "cfg2_attn2d_2x2_ulysses2",
        {"cfg_size": 2, "attn2d_size": (2, 2), "ulysses_size": 2},
    ),
]
_MULTINODE_SLURM_CHILD_ENV = "TRTLLM_VISUAL_GEN_MULTINODE_SLURM_CHILD"


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


def _slurm_rank_env():
    if "SLURM_PROCID" not in os.environ or "SLURM_NTASKS" not in os.environ:
        return None
    return int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])


def _default_master_port():
    job_id = int(os.environ.get("SLURM_JOB_ID", "0") or 0)
    return str(20000 + job_id % 20000)


def _slurm_node_count():
    for var in ("SLURM_JOB_NUM_NODES", "SLURM_NNODES"):
        if var in os.environ:
            return int(os.environ[var])
    return None


def _trtllm_launch_wrapper_world_size():
    try:
        return int(os.environ.get("tllm_mpi_size", "1") or 1)
    except ValueError:
        return 1


def _multinode_subprocess_timeout():
    return int(os.environ.get("TRTLLM_VISUAL_GEN_MULTINODE_TIMEOUT", "3600"))


def _resolve_slurm_master_addr():
    if os.environ.get("MASTER_ADDR"):
        return os.environ["MASTER_ADDR"]

    nodelist = os.environ.get("SLURM_JOB_NODELIST")
    if not nodelist:
        pytest.skip("SLURM_JOB_NODELIST is required to resolve MASTER_ADDR")
    if shutil.which("scontrol") is None:
        pytest.skip("scontrol is required to resolve MASTER_ADDR from SLURM_JOB_NODELIST")

    result = subprocess.run(
        ["scontrol", "show", "hostnames", nodelist],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"Failed to resolve SLURM master host:\n{result.stdout}")

    master_addr = next((line.strip() for line in result.stdout.splitlines() if line.strip()), "")
    if not master_addr:
        pytest.fail(f"scontrol returned no hostnames for SLURM_JOB_NODELIST={nodelist!r}")
    os.environ["MASTER_ADDR"] = master_addr
    return master_addr


def _ensure_slurm_external_launch_env():
    os.environ["MASTER_ADDR"] = _resolve_slurm_master_addr()
    os.environ.setdefault("MASTER_PORT", _default_master_port())

    # Force VisualGen to exercise its SLURM detection branch even when the
    # surrounding CI wrapper leaves torchrun-like variables behind.
    for var in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        os.environ.pop(var, None)


def _run_wan22_multinode_slurm_rank(tmp_path, variant_name, parallel):
    if not MODULES_AVAILABLE:
        pytest.skip("Required modules not available")

    rank_env = _slurm_rank_env()
    if rank_env is None:
        pytest.skip("This VisualGen multi-node case must run under SLURM rank env")
    rank, world_size = rank_env
    if world_size != WAN22_LPIPS_MULTINODE_WORLD_SIZE:
        pytest.skip(f"Requires {WAN22_LPIPS_MULTINODE_WORLD_SIZE} SLURM tasks, got {world_size}")
    node_count = _slurm_node_count()
    if node_count is not None and node_count < WAN22_LPIPS_MULTINODE_NODES:
        pytest.skip(
            f"Requires at least {WAN22_LPIPS_MULTINODE_NODES} SLURM nodes, got {node_count}"
        )

    _ensure_slurm_external_launch_env()
    ParallelConfig(**parallel).validate_world_size(world_size)
    model_path = _lpips_model_path("Wan2.2-T2V-A14B-Diffusers")
    _skip_if_missing(model_path, "Wan 2.2 checkpoint", is_dir=True)

    from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams
    from tensorrt_llm.visual_gen.args import AttentionConfig, CompilationConfig, TorchCompileConfig

    visual_gen_args = VisualGenArgs(
        model=model_path,
        compilation_config=CompilationConfig(skip_warmup=True),
        torch_compile_config=TorchCompileConfig(enable=False),
        attention_config=AttentionConfig(backend="FA4"),
        parallel_config=parallel,
    )

    visual_gen = None
    try:
        try:
            visual_gen = VisualGen(model=model_path, args=visual_gen_args)
        except SystemExit as exc:
            assert rank != 0, "Only non-zero SLURM ranks should exit through worker mode"
            assert exc.code in (0, None)
            return

        assert rank == 0
        params = VisualGenParams(
            height=WAN22_LPIPS_HEIGHT,
            width=WAN22_LPIPS_WIDTH,
            num_frames=WAN22_LPIPS_NUM_FRAMES,
            num_inference_steps=WAN22_LPIPS_NUM_INFERENCE_STEPS,
            guidance_scale=WAN22_LPIPS_GUIDANCE_SCALE,
            seed=WAN22_LPIPS_SEED,
            frame_rate=WAN22_LPIPS_FRAME_RATE,
            negative_prompt=WAN22_LPIPS_NEGATIVE_PROMPT,
        )
        output = visual_gen.generate(inputs=WAN22_LPIPS_PROMPT, params=params)
        assert output.error is None, f"unexpected error on Wan 2.2 multi-node run: {output.error}"
        assert output.video is not None

        generated_path = tmp_path / f"wan22_t2v_generated_{variant_name}_slurm.mp4"
        output.save(generated_path, frame_rate=WAN22_LPIPS_FRAME_RATE)
        assert generated_path.is_file(), (
            f"VisualGen multi-node run did not produce {generated_path}"
        )

        golden_path = _golden_media_path(
            tmp_path, "wan22_t2v_lpips_golden_video.mp4", "Wan 2.2 LPIPS golden video"
        )
        score = _run_lpips_eval(
            tmp_path,
            f"wan22_t2v_{variant_name}_slurm",
            "video",
            WAN22_LPIPS_PROMPT,
            golden_path,
            generated_path,
        )
        _assert_lpips_below_threshold(score, WAN_MULTI_GPU_LPIPS_THRESHOLD)
    finally:
        if visual_gen is not None:
            visual_gen.shutdown()


def _run_wan22_multinode_slurm_parent(variant_name):
    if (
        os.environ.get("TLLM_SPAWN_PROXY_PROCESS") == "1"
        and _trtllm_launch_wrapper_world_size() >= WAN22_LPIPS_MULTINODE_WORLD_SIZE
    ):
        pytest.fail(
            "VisualGen SLURM external-launch coverage cannot run under "
            "trtllm-llmapi-launch because that wrapper removes SLURM_* env "
            "before user code. Run this nodeid with direct srun so "
            "_detect_external_launch() sees the real SLURM rank environment."
        )

    if os.environ.get("SLURM_JOB_ID") is None:
        pytest.skip("A SLURM allocation is required for the VisualGen multi-node LPIPS case")
    node_count = _slurm_node_count()
    if node_count is not None and node_count < WAN22_LPIPS_MULTINODE_NODES:
        pytest.skip(
            f"Requires at least {WAN22_LPIPS_MULTINODE_NODES} SLURM nodes, got {node_count}"
        )
    if shutil.which("srun") is None:
        pytest.skip("srun is required for the VisualGen multi-node LPIPS case")

    # Pre-check the checkpoint here so a missing model skips the parent honestly,
    # instead of letting every srun rank skip and the parent report a false pass
    # (an all-skipped pytest run still exits 0).
    _skip_if_missing(
        _lpips_model_path("Wan2.2-T2V-A14B-Diffusers"),
        "Wan 2.2 checkpoint",
        is_dir=True,
    )

    env = os.environ.copy()
    env[_MULTINODE_SLURM_CHILD_ENV] = "1"
    env["MASTER_ADDR"] = _resolve_slurm_master_addr()
    env.setdefault("MASTER_PORT", _default_master_port())
    env["PYTHONUNBUFFERED"] = "1"
    for var in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        env.pop(var, None)

    test_file = str(Path(__file__).resolve())
    nodeid = f"{test_file}::test_wan22_t2v_lpips_against_golden_multinode_slurm[{variant_name}]"
    cmd = [
        "srun",
        "-l",
        "--overlap",
        f"--nodes={WAN22_LPIPS_MULTINODE_NODES}",
        f"--ntasks={WAN22_LPIPS_MULTINODE_WORLD_SIZE}",
        f"--ntasks-per-node={WAN22_LPIPS_MULTINODE_GPUS_PER_NODE}",
        "--export=ALL",
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-s",
        nodeid,
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parents[5],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=_multinode_subprocess_timeout(),
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.output or ""
        pytest.fail(
            "VisualGen multi-node SLURM subprocess timed out after "
            f"{exc.timeout} seconds:\n{output}"
        )

    if result.returncode != 0:
        pytest.fail(
            "VisualGen multi-node SLURM subprocess failed with "
            f"exit code {result.returncode}:\n{result.stdout}"
        )

    # Guard against a false pass: an all-skipped/empty pytest run also exits 0.
    # The parent has already validated every precondition, so once srun launches
    # all ranks must actually pass; any skip or empty collection is a real defect.
    output = result.stdout or ""
    if "passed" not in output or "skipped" in output or "no tests ran" in output:
        pytest.fail(
            "VisualGen multi-node SLURM run exited 0 but did not actually execute "
            "(expected all ranks to pass with no skips):\n"
            f"{output}"
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


@pytest.mark.parametrize(
    "variant_name,parallel",
    WAN22_LPIPS_MULTINODE_VARIANTS,
    ids=[name for name, _ in WAN22_LPIPS_MULTINODE_VARIANTS],
)
def test_wan22_t2v_lpips_against_golden_multinode_slurm(tmp_path, variant_name, parallel):
    if _slurm_rank_env() is not None:
        _run_wan22_multinode_slurm_rank(tmp_path, variant_name, parallel)
        return

    if os.environ.get(_MULTINODE_SLURM_CHILD_ENV):
        pytest.skip("VisualGen SLURM child was not launched with SLURM rank env")

    _run_wan22_multinode_slurm_parent(variant_name)
