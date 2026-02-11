import os
from subprocess import check_call

import pytest
import torch
from utils.llm_data import llm_models_root


@pytest.mark.parametrize("world_size", [1, 4])
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


@pytest.mark.parametrize("world_size", [1, 4])
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


@pytest.mark.parametrize("world_size", [1, 4])
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
            "--moe-backend=WIDEEP",
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
