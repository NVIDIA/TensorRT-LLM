# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from subprocess import check_call
from typing import Iterable

import pytest
import torch
from utils.llm_data import llm_models_root
from utils.util import skip_pre_blackwell

from tensorrt_llm.tools.layer_wise_benchmarks.calibrator import Calibrator, Mode

_LAYER_WISE_BENCHMARKS_NVBUG = pytest.mark.skip(reason="https://nvbugs/6337228")


@pytest.fixture(scope="module", autouse=True)
def require_nsys_cuda_tracing():
    """Skip all tests in this module when nsys cannot record CUDA kernel activity.

    Every test here captures an nsys trace and feeds it to parse.py or
    parse_e2e.py, which require CUDA kernel events. On nodes where the host
    driver cannot natively serve the container's CUDA stack (CUDA forward
    compatibility mode), the workload runs normally but nsys silently records
    no CUDA activity: whole-process traces contain NVTX ranges only, and with
    `-c cudaProfilerApi` no report is produced at all because the profiler
    start callback is never delivered (https://nvbugs/6162541). Detect this
    once with a small canary run and skip with a clear reason instead of
    failing inside the parsers.
    """
    if torch.cuda.device_count() < 1:
        # No GPU visible: let the per-test GPU-count checks report their own
        # skip reason.
        return

    def environment_summary(profile_stdout: str) -> str:
        # Enough context to act on the skip from the CI log alone: which
        # libcuda the canary actually loaded (host driver vs. the container's
        # forward-compat one), the host driver version, and the nsys version.
        parts = []
        if m := re.search(r"LIBCUDA: (\S+)", profile_stdout):
            parts.append(f"canary loaded {m.group(1)}")
        for cmd, label in (
            (["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], "host driver"),
            (["nsys", "--version"], "nsys"),
        ):
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if lines := out.stdout.strip().splitlines():
                    parts.append(f"{label} {lines[-1]}")
            except (OSError, subprocess.SubprocessError):
                pass
        return "; ".join(parts)

    canary_code = (
        "import torch\n"
        "a = torch.ones(1024, device='cuda')\n"
        "torch.cuda.synchronize()\n"
        "print((a + a).sum().item())\n"
        "for line in open('/proc/self/maps'):\n"
        "    if 'libcuda.so' in line:\n"
        "        print('LIBCUDA:', line.split()[-1])\n"
        "        break\n"
    )
    profile_stdout = ""
    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = Path(tmpdir) / "canary.nsys-rep"
        sqlite_path = Path(tmpdir) / "canary.sqlite"
        try:
            profile_result = subprocess.run(
                [
                    "nsys",
                    "profile",
                    "-t",
                    "cuda",
                    "-s",
                    "none",
                    "--cpuctxsw",
                    "none",
                    "-o",
                    str(report_path),
                    "--force-overwrite",
                    "true",
                    sys.executable,
                    "-c",
                    canary_code,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=600,
            )
            profile_stdout = profile_result.stdout
            subprocess.run(
                [
                    "nsys",
                    "export",
                    "--type",
                    "sqlite",
                    "-o",
                    str(sqlite_path),
                    "--force-overwrite=true",
                    str(report_path),
                ],
                check=True,
                capture_output=True,
                timeout=600,
            )
        except (OSError, subprocess.SubprocessError) as e:
            pytest.skip(
                f"nsys cannot profile CUDA on this node: {e}"
                f" ({environment_summary(profile_stdout)})"
            )
        conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
        try:
            num_kernel_tables = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'"
                " AND name = 'CUPTI_ACTIVITY_KIND_KERNEL'"
            ).fetchone()[0]
        finally:
            conn.close()
    if num_kernel_tables == 0:
        pytest.skip(
            "nsys records no CUDA kernel activity on this node, e.g. because"
            " the host driver cannot natively serve the container's CUDA stack"
            " (CUDA forward compatibility mode); the layer-wise benchmark"
            " parsers require CUDA kernel events (https://nvbugs/6162541)"
            f" ({environment_summary(profile_stdout)})"
        )


# The pinned DeepSeek FP4 checkpoint requires SM100+.
@skip_pre_blackwell
@pytest.mark.parametrize(
    "world_size",
    [
        pytest.param(1, marks=_LAYER_WISE_BENCHMARKS_NVBUG),
        4,
    ],
)
def test_deepseek_r1_ctx_dep(llm_root, world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size:d} GPUs to run this test")
    model_root = llm_models_root(check=True)
    profile_dir = f"profiles/test_deepseek_r1_ctx_dep_{world_size}"
    check_call(
        [
            "./mpi_launch.sh",
            "./run.sh",
            "config_ctx.yaml",
            "--model",
            model_root / "DeepSeek-R1" / "DeepSeek-R1-0528-FP4-v2",
        ],
        cwd=llm_root / "examples" / "layer_wise_benchmarks",
        env={
            **os.environ,
            "NP": f"{world_size:d}",
            "PROFILE_DIR": profile_dir,
        },
    )
    check_call(
        ["python3", "parse.py", "--profile-dir", profile_dir, f"--world-size={world_size}"],
        cwd=llm_root / "examples" / "layer_wise_benchmarks",
    )


# The pinned DeepSeek FP4 checkpoint requires SM100+.
@skip_pre_blackwell
@pytest.mark.parametrize(
    "world_size",
    [
        pytest.param(1, marks=_LAYER_WISE_BENCHMARKS_NVBUG),
        4,
    ],
)
def test_deepseek_r1_ctx_tep(llm_root, world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size:d} GPUs to run this test")
    model_root = llm_models_root(check=True)
    profile_dir = f"profiles/test_deepseek_r1_ctx_tep_{world_size}"
    check_call(
        [
            "./mpi_launch.sh",
            "./run.sh",
            "config_ctx.yaml",
            "--model",
            model_root / "DeepSeek-R1" / "DeepSeek-R1-0528-FP4-v2",
            "--no-enable-attention-dp",
            "--moe-backend=TRTLLM",
        ],
        cwd=llm_root / "examples" / "layer_wise_benchmarks",
        env={
            **os.environ,
            "NP": f"{world_size:d}",
            "PROFILE_DIR": profile_dir,
        },
    )
    check_call(
        ["python3", "parse.py", "--profile-dir", profile_dir, f"--world-size={world_size}"],
        cwd=llm_root / "examples" / "layer_wise_benchmarks",
    )


# The pinned config (DeepSeek-V3.2 with the DEEPGEMM MoE backend) targets SM100+.
@skip_pre_blackwell
@pytest.mark.parametrize(
    "world_size",
    [
        pytest.param(1, marks=_LAYER_WISE_BENCHMARKS_NVBUG),
        4,
    ],
)
def test_deepseek_v32_ctx_dep(llm_root, world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size:d} GPUs to run this test")
    model_root = llm_models_root(check=True)
    profile_dir = f"profiles/test_deepseek_v32_ctx_dep_{world_size}"
    check_call(
        [
            "./mpi_launch.sh",
            "./run.sh",
            "config_ctx.yaml",
            "--model",
            model_root / "DeepSeek-V3.2-Exp-hf",
            "--tokens-per-block=64",
            "--moe-backend=DEEPGEMM",
        ],
        cwd=llm_root / "examples" / "layer_wise_benchmarks",
        env={
            **os.environ,
            "NP": f"{world_size:d}",
            "PROFILE_DIR": profile_dir,
        },
    )
    check_call(
        ["python3", "parse.py", "--profile-dir", profile_dir, f"--world-size={world_size}"],
        cwd=llm_root / "examples" / "layer_wise_benchmarks",
    )


# The pinned DeepSeek FP4 checkpoint requires SM100+.
@pytest.mark.skip(
    reason="--scaled-from makes the CTX prefill pack come out all-NaN, independently of "
    "the MoE backend: on 4x B200 every combination of gen backend (CUTEDSL, CUTLASS) and "
    "prefill backend (CUTLASS, DEEPGEMM) fails the NaN check, while the same command "
    "without --scaled-from passes. Re-enable once weak scaling yields finite activations."
)
@skip_pre_blackwell
@pytest.mark.parametrize("world_size", [4])
def test_deepseek_r1_gen_scaled_from_16_dep(llm_root, world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size:d} GPUs to run this test")
    model_root = llm_models_root(check=True)
    profile_dir = f"profiles/test_deepseek_r1_gen_scaled_from_16_dep_{world_size}"
    check_call(
        [
            "./mpi_launch.sh",
            "./run.sh",
            "config_gen.yaml",
            "--model",
            model_root / "DeepSeek-R1" / "DeepSeek-R1-0528-FP4-v2",
            "--layer-indices=5,6",
            "--scaled-from=16",
            "--moe-backend=CUTEDSL",
        ],
        cwd=llm_root / "examples" / "layer_wise_benchmarks",
        env={
            **os.environ,
            "NP": f"{world_size:d}",
            "PROFILE_DIR": profile_dir,
        },
    )
    check_call(
        ["python3", "parse.py", "--profile-dir", profile_dir, f"--world-size={world_size}"],
        cwd=llm_root / "examples" / "layer_wise_benchmarks",
    )


@pytest.mark.parametrize("world_size", [1, 4])
def test_nemotron_gen_dep(llm_root, world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size:d} GPUs to run this test")
    model_root = llm_models_root(check=True)
    profile_dir = f"profiles/test_nemotron_gen_dep_{world_size}"
    check_call(
        [
            "./mpi_launch.sh",
            "./run.sh",
            "config_gen.yaml",
            "--model",
            model_root / "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            "--layer-indices=4,5,6",
            "--mamba-ssm-cache-dtype=float16",
        ],
        cwd=llm_root / "examples" / "layer_wise_benchmarks",
        env={
            **os.environ,
            "NP": f"{world_size:d}",
            "PROFILE_DIR": profile_dir,
        },
    )
    check_call(
        ["python3", "parse.py", "--profile-dir", profile_dir, f"--world-size={world_size}"],
        cwd=llm_root / "examples" / "layer_wise_benchmarks",
    )


@pytest.mark.parametrize("world_size", [1, 4])
def test_qwen3_next_gen_tep(llm_root, world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size:d} GPUs to run this test")
    model_root = llm_models_root(check=True)
    profile_dir = f"profiles/test_qwen3_next_gen_tep_{world_size}"
    check_call(
        [
            "./mpi_launch.sh",
            "./run.sh",
            "config_gen.yaml",
            "--model",
            model_root / "Qwen3" / "Qwen3-Next-80B-A3B-Instruct",
            "--layer-indices=6,7",
            "--no-enable-attention-dp",
            "--mamba-ssm-cache-dtype=float16",
            "--moe-backend=TRTLLM",
        ],
        cwd=llm_root / "examples" / "layer_wise_benchmarks",
        env={
            **os.environ,
            "NP": f"{world_size:d}",
            "PROFILE_DIR": profile_dir,
        },
    )
    check_call(
        ["python3", "parse.py", "--profile-dir", profile_dir, f"--world-size={world_size}"],
        cwd=llm_root / "examples" / "layer_wise_benchmarks",
    )


# Kimi K3's MXFP4 routed experts and KDA kernels require SM100+.
@skip_pre_blackwell
@pytest.mark.parametrize("world_size", [1, 4])
def test_kimi_k3_gen_dep(llm_root, world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size:d} GPUs to run this test")
    model_root = llm_models_root(check=True)
    profile_dir = f"profiles/test_kimi_k3_gen_dep_{world_size}"
    if world_size == 1:
        # EP1 puts all 896 experts on one GPU, and GEN builds a second (prefill)
        # model, so halve the slice: layer 6 is KDA and 7 is MLA, still covering
        # both attention paths. Balanced routing needs the top-k computed outside
        # the MoE kernel, which only happens once there is expert parallelism.
        layer_args = ["--layer-indices=6,7", "--balance-method=NotModified"]
    else:
        # 0-based: three KDA layers then one full-attention (MLA) layer.
        layer_args = ["--layer-indices=4,5,6,7"]
    check_call(
        [
            "./mpi_launch.sh",
            "./run.sh",
            "config_gen.yaml",
            "--model",
            model_root / "Kimi-K3",
            *layer_args,
            "--tokens-per-block=64",
            # SiTU routed experts support no other backend, and GEN also builds
            # a prefill runner.
            "--moe-backend=TRTLLM",
            "--moe-backend-for-prefill=TRTLLM",
            # 1 golden + 3 draft tokens per generation step.
            "--batch-size=32",
            "--seq-len-q=4",
            "--spec-max-draft-len=3",
        ],
        cwd=llm_root / "examples" / "layer_wise_benchmarks",
        env={
            **os.environ,
            "NP": f"{world_size:d}",
            "PROFILE_DIR": profile_dir,
        },
    )
    check_call(
        ["python3", "parse.py", "--profile-dir", profile_dir, f"--world-size={world_size}"],
        cwd=llm_root / "examples" / "layer_wise_benchmarks",
    )


# The pinned DeepSeek-V3-Lite NVFP4 checkpoint requires SM100+; on older
# architectures the benchmark crashes the test process (seen on A10, where
# this module runs as part of the unittest/tools directory).
@skip_pre_blackwell
@pytest.mark.parametrize("world_size", [1, 4])
def test_performance_alignment(llm_root, world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size:d} GPUs to run this test")
    model_root = llm_models_root(check=True)
    profile_dir = f"profiles/test_performance_alignment_{world_size}"
    check_call(
        [
            "./sample_performance_alignment.sh",
        ],
        cwd=llm_root / "examples" / "layer_wise_benchmarks",
        env={
            **os.environ,
            "MODEL": model_root / "DeepSeek-V3-Lite" / "nvfp4_moe_only",
            "NP": f"{world_size:d}",
            "PROFILE_DIR": profile_dir,
        },
    )


# ---------------------------------------------------------------------------
# Replay-window and replay-shape checks. No GPU: these read the replay database
# and nothing else.
#
# Built by hand rather than through Calibrator.init(), which cannot be used here --
# _init_replay_mode() decodes every record and moves the slots to CUDA.
# ---------------------------------------------------------------------------


def _replay_calibrator(
    iterations: Iterable[int],
    tokens: int = 32,
    top_k: int = 6,
    layers: int = 4,
) -> Calibrator:
    calibrator = Calibrator()
    calibrator.mode = Mode.REPLAY
    calibrator._replay_db = {
        i: {
            "metadata": [
                {"layer_idx": k, "token_selected_slots_shape": [tokens, top_k]}
                for k in range(layers)
            ]
        }
        for i in iterations
    }
    return calibrator


def test_missing_replay_iterations_none_when_the_window_fits() -> None:
    calibrator = _replay_calibrator(range(100, 126))
    assert calibrator.get_missing_replay_iterations(105, 125) == []
    assert calibrator.get_missing_replay_iterations(100, 125) == []


def test_missing_replay_iterations_past_the_end() -> None:
    calibrator = _replay_calibrator(range(100, 126))
    assert calibrator.get_missing_replay_iterations(124, 128) == [126, 127, 128]
    assert calibrator.get_missing_replay_iterations(0, 3) == [0, 1, 2, 3]


def test_missing_replay_iterations_sees_a_hole() -> None:
    """Report the case a first/last comparison cannot answer.

    get_replay_iteration_range() raises on a non-contiguous calibration, so a bounds
    check has nothing to compare against and the KeyError comes back at pre_step().
    A window that stays inside one contiguous run is still legal and must pass.
    """
    calibrator = _replay_calibrator(list(range(100, 111)) + list(range(113, 126)))
    assert calibrator.get_missing_replay_iterations(113, 125) == []
    assert calibrator.get_missing_replay_iterations(105, 125) == [111, 112]


def test_missing_replay_iterations_requires_replay_mode() -> None:
    with pytest.raises(ValueError, match="only valid in REPLAY mode"):
        Calibrator().get_missing_replay_iterations(0, 1)


def test_replay_token_count_when_every_layer_agrees() -> None:
    assert _replay_calibrator(range(100, 103), tokens=64).get_replay_token_count() == 64


def test_replay_token_count_is_none_when_layers_disagree() -> None:
    """Return None, rather than an error, when the recorded layers disagree.

    A calibration whose layers disagree cannot be replayed under one CUDA graph
    either; None leaves that complaint to the caller instead of raising a second
    error about the first problem.
    """
    calibrator = _replay_calibrator(range(100, 103), tokens=64)
    calibrator._replay_db[101]["metadata"][0]["token_selected_slots_shape"] = [32, 6]
    assert calibrator.get_replay_token_count() is None


def test_replay_token_count_compares_the_whole_shape() -> None:
    """Reject a range that agrees on tokens and differs in top_k.

    One CUDA graph holds one shape, and [64, 6] is not [64, 8]; comparing only the
    token dimension would call this range replayable.
    """
    calibrator = _replay_calibrator(range(100, 103), tokens=64, top_k=6)
    calibrator._replay_db[101]["metadata"][0]["token_selected_slots_shape"] = [64, 8]
    assert calibrator.get_replay_token_count() is None


def test_replay_token_count_is_scoped_to_the_window() -> None:
    """Ignore records outside the window being replayed.

    Unscoped, a single stray iteration at another shape makes the whole file look
    inconsistent and takes a perfectly replayable window down with it.
    """
    calibrator = _replay_calibrator(range(105, 126), tokens=64)
    calibrator._replay_db[99] = {
        "metadata": [{"layer_idx": k, "token_selected_slots_shape": [32, 6]} for k in range(4)]
    }
    assert calibrator.get_replay_token_count() is None
    assert calibrator.get_replay_token_count(105, 125) == 64
    assert calibrator.get_replay_token_count(99, 125) is None


def test_replay_token_count_is_none_for_an_empty_window() -> None:
    calibrator = _replay_calibrator(range(100, 126), tokens=64)
    assert calibrator.get_replay_token_count(200, 300) is None


def test_replay_token_count_requires_replay_mode() -> None:
    with pytest.raises(ValueError, match="only valid in REPLAY mode"):
        Calibrator().get_replay_token_count()
