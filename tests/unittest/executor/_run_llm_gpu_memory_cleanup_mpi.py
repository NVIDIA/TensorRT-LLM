# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import traceback

import torch

from tensorrt_llm import LLM
from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams

# isort: off
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.llm_data import llm_models_root
from utils.util import get_current_process_gpu_memory
# isort: on

TINYLLAMA_REL_PATH = "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
MEMORY_TOLERANCE_BYTES = 1_500_000_000


def _process_gpu_memory_info_available() -> bool:
    tensor = torch.zeros(4096, dtype=torch.int32, device="cuda")
    torch.cuda.synchronize()
    usage = get_current_process_gpu_memory()
    del tensor
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return usage != 0


def _tinyllama_model_path() -> str:
    root = llm_models_root()
    if root is None:
        raise RuntimeError("LLM_MODELS_ROOT is not set or not accessible")
    model_path = root / TINYLLAMA_REL_PATH
    if not model_path.is_dir():
        raise RuntimeError(f"TinyLlama model not found at {model_path}")
    return str(model_path)


def _llm_kwargs(model: str) -> dict:
    return dict(
        model=model,
        tensor_parallel_size=2,
        kv_cache_config=KvCacheConfig(enable_block_reuse=False, max_tokens=512),
        cuda_graph_config=None,
        max_seq_len=512,
    )


def _run_llm_cycle(model: str, sampling_params: SamplingParams) -> None:
    with LLM(**_llm_kwargs(model)) as llm:
        llm.generate(["hi"], sampling_params)


def main() -> None:
    model = _tinyllama_model_path()
    sampling_params = SamplingParams(max_tokens=4)

    baseline = None
    if mpi_rank() == 0 and _process_gpu_memory_info_available():
        baseline = get_current_process_gpu_memory(include_subprocess=True)

    _run_llm_cycle(model, sampling_params)

    if mpi_rank() == 0 and baseline is not None:
        current = get_current_process_gpu_memory(include_subprocess=True)
        allowed = baseline + MEMORY_TOLERANCE_BYTES
        if current > allowed:
            raise RuntimeError(
                f"GPU memory {current} bytes exceeds baseline {baseline} + "
                f"tolerance {MEMORY_TOLERANCE_BYTES} bytes (allowed: {allowed} bytes)"
            )

    _run_llm_cycle(model, sampling_params)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
