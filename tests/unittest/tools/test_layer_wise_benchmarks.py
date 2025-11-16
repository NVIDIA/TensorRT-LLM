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
    check_call(
        [
            "./mpi_launch.sh",
            "./run_single.sh",
            "config_ctx.yaml",
            "--model",
            model_root / "DeepSeek-R1" / "DeepSeek-R1-0528-FP4-v2",
        ],
        cwd=llm_root / "examples" / "layer_wise_benchmarks",
        env={
            **os.environ,
            "NP": f"{world_size:d}",
        },
    )


@pytest.mark.parametrize("world_size", [1, 4])
def test_deepseek_r1_ctx_tep(llm_root, world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size:d} GPUs to run this test")
    model_root = llm_models_root(check=True)
    check_call(
        [
            "./mpi_launch.sh",
            "./run_single.sh",
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
            "TRTLLM_ENABLE_PDL": "1",
        },
    )


@pytest.mark.parametrize("world_size", [1, 4])
def test_deepseek_v32_ctx_dep(llm_root, world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size:d} GPUs to run this test")
    model_root = llm_models_root(check=True)
    check_call(
        [
            "./mpi_launch.sh",
            "./run_single.sh",
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
        },
    )


@pytest.mark.parametrize("world_size", [4])
def test_deepseek_r1_gen_scaled_from_16_dep(llm_root, world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size:d} GPUs to run this test")
    model_root = llm_models_root(check=True)
    check_call(
        [
            "./mpi_launch.sh",
            "./run_single.sh",
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
        },
    )


@pytest.mark.parametrize("world_size", [1, 4])
def test_qwen3_next_gen_tep(llm_root, world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size:d} GPUs to run this test")
    model_root = llm_models_root(check=True)
    check_call(
        [
            "./mpi_launch.sh",
            "./run_single.sh",
            "config_gen.yaml",
            "--model",
            model_root / "Qwen3" / "Qwen3-Next-80B-A3B-Instruct",
            "--layer-indices=6,7",
            "--no-enable-attention-dp",
            "--moe-backend=TRTLLM",
            "--balance-method=NotModified",
        ],
        cwd=llm_root / "examples" / "layer_wise_benchmarks",
        env={
            **os.environ,
            "NP": f"{world_size:d}",
            "TRTLLM_ENABLE_PDL": "1",
        },
    )
