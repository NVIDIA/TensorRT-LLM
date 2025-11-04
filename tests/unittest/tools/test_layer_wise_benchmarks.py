import os
from subprocess import check_call

import pytest
import torch
from utils.cpp_paths import llm_root  # noqa: F401
from utils.llm_data import llm_models_root


@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="needs 4 GPUs to run this test")
def test_deepseek_r1_ctx_tep(llm_root):
    model_root = llm_models_root(check=True)
    check_call([
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
                   "NP": "4",
                   "TRTLLM_ENABLE_PDL": "1",
               })


@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="needs 4 GPUs to run this test")
def test_deepseek_v32_ctx_dep(llm_root):
    model_root = llm_models_root(check=True)
    check_call([
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
                   "NP": "4",
               })


@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="needs 4 GPUs to run this test")
def test_deepseek_r1_gen_scaled_from_16_dep(llm_root):
    model_root = llm_models_root(check=True)
    check_call([
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
                   "NP": "4",
               })
