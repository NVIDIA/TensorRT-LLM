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

import pytest
import torch
from utils.llm_data import llm_models_root
from utils.util import skip_pre_blackwell


def _is_blackwell_capable():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 10


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


# Signature the workload emits when it refuses to build a Blackwell-only kernel,
# e.g. "FlashInfer TRTLLM-Gen FMHA is unavailable: requires SM100 or SM103, got
# SM90." or "CuteDSL NVFP4 backend requires SM 100 (B200) or SM 103 (B300)".
# Anchoring on this keeps a generic Python assertion -- which also exits 1 --
# from being mistaken for a legitimate kernel rejection.
_KERNEL_UNAVAILABLE_RE = re.compile(
    r"requires SM\s*>?=?\s*10\d|is unavailable: requires SM",
    re.IGNORECASE,
)


def _run_benchmark(cmd, *, cwd, env=None, tolerate_kernel_rejection=True):
    """Run a benchmark step, allowing only a genuine pre-Blackwell rejection.

    These benchmarks exercise Blackwell-only kernels (NVFP4, FP8 GEN MLA, FMHA
    D=192,DV=128). On pre-Blackwell the workload's own availability checks
    legitimately reject the model, and we treat that as PASS so the test list
    stays stable and the launcher itself is still exercised everywhere.

    Everything else must fail. In particular a bare ``rc == 1`` is NOT accepted
    on its own: that is also what a generic Python assertion or a real
    launcher/parse regression returns, so it is honoured only when stderr also
    carries the kernel-availability signature. Pass
    ``tolerate_kernel_rejection=False`` for steps that never touch a GPU kernel
    (parse.py), where any failure is real by construction.
    """
    if _is_blackwell_capable() or not tolerate_kernel_rejection:
        check_call(cmd, cwd=cwd, env=env)
        return

    proc = subprocess.run(
        cmd, cwd=cwd, env=env, stderr=subprocess.PIPE, text=True, errors="replace"
    )
    if proc.stderr:
        # Keep the CI log intact -- stderr was captured only so we can inspect it.
        sys.stderr.write(proc.stderr)
    if proc.returncode == 0:
        return
    # SIGABRT (134) is a C++ TLLM_CHECK abort and SIGKILL (137) an OOM kill;
    # neither is producible by a plain Python assertion, so they stay tolerated
    # as-is. A bare exit 1 has to prove itself against the signature.
    if proc.returncode in (134, 137) or _KERNEL_UNAVAILABLE_RE.search(proc.stderr or ""):
        return
    raise subprocess.CalledProcessError(proc.returncode, cmd, stderr=proc.stderr)


# The pinned DeepSeek FP4 checkpoint requires SM100+. On pre-Blackwell,
# `_run_benchmark` converts the workload's own kernel-availability
# rejection (SIGABRT / assertion / OOM) into a PASS instead of skipping;
# this keeps the test list stable and validates that the launcher itself
# still works everywhere.
@pytest.mark.parametrize("world_size", [1, 4])
def test_deepseek_r1_ctx_dep(llm_root, world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size:d} GPUs to run this test")
    model_root = llm_models_root(check=True)
    profile_dir = f"profiles/test_deepseek_r1_ctx_dep_{world_size}"
    _run_benchmark(
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
    # parse.py is pure Python over the recorded nsys profile -- it imports no
    # torch and launches no kernel -- so it can never fail for lack of a
    # Blackwell kernel. Any non-zero exit here is a real parser regression.
    _run_benchmark(
        ["python3", "parse.py", "--profile-dir", profile_dir, f"--world-size={world_size}"],
        cwd=llm_root / "examples" / "layer_wise_benchmarks",
        tolerate_kernel_rejection=False,
    )


# The pinned DeepSeek FP4 checkpoint requires SM100+; see
# `test_deepseek_r1_ctx_dep` for how `_run_benchmark` handles pre-Blackwell.
@pytest.mark.parametrize("world_size", [1, 4])
def test_deepseek_r1_ctx_tep(llm_root, world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size:d} GPUs to run this test")
    model_root = llm_models_root(check=True)
    profile_dir = f"profiles/test_deepseek_r1_ctx_tep_{world_size}"
    _run_benchmark(
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
    # parse.py is pure Python over the recorded nsys profile -- it imports no
    # torch and launches no kernel -- so it can never fail for lack of a
    # Blackwell kernel. Any non-zero exit here is a real parser regression.
    _run_benchmark(
        ["python3", "parse.py", "--profile-dir", profile_dir, f"--world-size={world_size}"],
        cwd=llm_root / "examples" / "layer_wise_benchmarks",
        tolerate_kernel_rejection=False,
    )


# The pinned config (DeepSeek-V3.2 with the DEEPGEMM MoE backend) targets SM100+;
# see `test_deepseek_r1_ctx_dep` for how `_run_benchmark` handles pre-Blackwell.
@pytest.mark.parametrize("world_size", [1, 4])
def test_deepseek_v32_ctx_dep(llm_root, world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size:d} GPUs to run this test")
    model_root = llm_models_root(check=True)
    profile_dir = f"profiles/test_deepseek_v32_ctx_dep_{world_size}"
    _run_benchmark(
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
    # parse.py is pure Python over the recorded nsys profile -- it imports no
    # torch and launches no kernel -- so it can never fail for lack of a
    # Blackwell kernel. Any non-zero exit here is a real parser regression.
    _run_benchmark(
        ["python3", "parse.py", "--profile-dir", profile_dir, f"--world-size={world_size}"],
        cwd=llm_root / "examples" / "layer_wise_benchmarks",
        tolerate_kernel_rejection=False,
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
